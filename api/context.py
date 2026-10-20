"""
Shared app state, the login dependency and the background job runner.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from db import AuthSession, Project, User
from workflow import ProjectWorkflow

from .security import token_hash

logger = logging.getLogger(__name__)

SESSION_COOKIE = "synapse_session"


@dataclass
class AppContext:
    engine: Engine
    workflow: ProjectWorkflow
    uploads_dir: Path
    exports_dir: Path
    discovery_dir: Path
    jobs: "JobRunner" = field(init=False)
    run_inline: bool = False

    def __post_init__(self) -> None:
        self.jobs = JobRunner(self)


def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx


def get_db(ctx: AppContext = Depends(get_ctx)):
    with Session(ctx.engine) as session:
        yield session


def _aware(dt: datetime) -> datetime:
    # SQLite returns naive datetimes; everything we store is UTC.
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        sess = db.get(AuthSession, token_hash(token))
        if sess and _aware(sess.expires_at) > datetime.now(timezone.utc):
            user = db.get(User, sess.user_id)
            if user:
                return user
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not logged in")


# ─── Background jobs ──────────────────────────────────────────────────────────

class JobRunner:
    """
    Runs workflow steps (which call the LLM and can take minutes) off the
    request thread, one at a time (MVP: one RFP is processed at a time, and
    the checkpoint DB is a single SQLite file). The project row is marked
    running with a progress message, then refreshed from the workflow status
    when the step finishes or fails.
    """

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="synapse-job")

    def submit(self, project_id: str, progress: str, step: Callable[[], Dict[str, Any]]) -> None:
        with Session(self.ctx.engine) as db:
            project = db.get(Project, project_id)
            project.running, project.progress, project.error = True, progress, None
            db.commit()
        if self.ctx.run_inline:
            self._run(project_id, step)
        else:
            self._pool.submit(self._run, project_id, step)

    def _run(self, project_id: str, step: Callable[[], Dict[str, Any]]) -> None:
        try:
            result = step()
        except Exception as exc:  # unexpected; workflow errors are returned in status
            logger.exception("Job for project %s crashed", project_id)
            result = {"stage": "failed", "error": str(exc), "state": {}}
        sync_project(self.ctx, project_id, result)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def sync_project(ctx: AppContext, project_id: str, status_: Optional[Dict[str, Any]] = None) -> None:
    """Copy the workflow's status into the project row's summary columns."""
    if status_ is None:
        try:
            status_ = ctx.workflow.status(project_id)
        except KeyError:
            status_ = {"stage": "failed", "error": "The project never started.", "state": {}}
    state = status_.get("state", {})
    with Session(ctx.engine) as db:
        project = db.get(Project, project_id)
        if project is None:
            return
        project.running = False
        project.progress = None
        project.stage = status_["stage"]
        project.error = status_.get("error")
        project.grand_total = (state.get("pricing_summary") or {}).get("grand_total", project.grand_total)
        project.fulfillment_score = state.get("fulfillment_score", project.fulfillment_score)
        project.proposal_path = state.get("report_pdf_path", project.proposal_path)
        db.commit()
