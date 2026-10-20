"""
FastAPI app factory.

    python -m api            # http://127.0.0.1:8000  (API docs at /docs)

create_app() takes overrides so tests can use temporary databases, a fake
workflow and inline (synchronous) jobs.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from sqlalchemy import select
from sqlalchemy.orm import Session

from db import Project, get_engine
from settings import DB_PATH, GEMINI_API_KEY, OPENROUTER_API_KEY, VAR_DIR
from workflow import ProjectWorkflow

from . import routes_auth, routes_catalogue, routes_projects, routes_rfps
from .context import AppContext, sync_project

logger = logging.getLogger(__name__)


def _recover_interrupted_jobs(ctx: AppContext) -> None:
    """Projects left 'running' by a crash or restart: refresh from the workflow."""
    with Session(ctx.engine) as db:
        stale = list(db.scalars(select(Project.id).where(Project.running.is_(True))))
    for project_id in stale:
        sync_project(ctx, project_id)


def _warm_up_models() -> None:
    """Load the search models in the background so the first project is faster."""
    try:
        from bu_rag import get_bu_rag_tool
        get_bu_rag_tool()
    except Exception:  # e.g. models not downloaded and offline
        logger.exception("Model warm-up failed; models will load on first use")


def create_app(
    *,
    db_path: Optional[str] = None,
    workflow: Optional[ProjectWorkflow] = None,
    var_dir: Optional[Path] = None,
    run_inline: bool = False,
    warm_up: bool = True,
) -> FastAPI:
    var = Path(var_dir or VAR_DIR)
    ctx = AppContext(
        engine=get_engine(db_path or DB_PATH),
        workflow=workflow or ProjectWorkflow(),
        uploads_dir=var / "uploads",
        exports_dir=var / "exports",
        run_inline=run_inline,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _recover_interrupted_jobs(ctx)
        if warm_up:
            threading.Thread(target=_warm_up_models, daemon=True).start()
        yield
        ctx.jobs.shutdown()

    app = FastAPI(title="SYNAPSE API", version="0.1.0", lifespan=lifespan)
    app.state.ctx = ctx

    @app.get("/api/health", tags=["health"])
    def health() -> Dict[str, Any]:
        """Lets the UI warn when API keys are missing (never returns the keys)."""
        return {
            "status": "ok",
            "llm_key_configured": bool(os.environ.get("OPENROUTER_API_KEY") or OPENROUTER_API_KEY),
            "gemini_key_configured": bool(os.environ.get("GEMINI_API_KEY") or GEMINI_API_KEY),
        }

    for module in (routes_auth, routes_rfps, routes_projects, routes_catalogue):
        app.include_router(module.router)
    return app
