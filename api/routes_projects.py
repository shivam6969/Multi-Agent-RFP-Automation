"""Projects: one review workflow per RFP (start, status, review, pricing, files)."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from db import Project, Rfp, User
from pricing_agent import reprice_with_margin
from quote_export import export_quote_csv
from settings import PRICING_RULES
from workflow import STAGE_MATCHING, STAGE_PRICING, STAGE_REQUIREMENTS

from .context import AppContext, current_user, get_ctx, get_db
from .schemas import (
    OutcomeIn,
    PricingPreviewIn,
    PricingPreviewOut,
    ProjectCreateIn,
    ProjectOut,
    ProjectSummaryOut,
    ReviewIn,
    RfpOut,
)

router = APIRouter(prefix="/api/projects", tags=["projects"])

# What the background job is doing, by the stage it starts from.
PROGRESS_AFTER = {
    "start":            "Reading the RFP and extracting requirements",
    STAGE_REQUIREMENTS: "Matching requirements to catalogue products",
    STAGE_MATCHING:     "Calculating prices",
    STAGE_PRICING:      "Writing the proposal",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_project(db: Session, project_id: str) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return project


def _summary(project: Project, rfp: Rfp) -> Dict[str, Any]:
    return {
        "id": project.id,
        "rfp": RfpOut.model_validate(rfp),
        "stage": "running" if project.running else project.stage,
        "running": project.running,
        "progress": project.progress,
        "error": project.error,
        "grand_total": project.grand_total,
        "fulfillment_score": project.fulfillment_score,
        "outcome": project.outcome,
        "proposal_available": bool(project.proposal_path and Path(project.proposal_path).is_file()),
        "created_at": project.created_at,
        "updated_at": project.updated_at,
    }


def _workflow_state(ctx: AppContext, project_id: str) -> Dict[str, Any]:
    try:
        return ctx.workflow.status(project_id)
    except KeyError:   # queued: the first step has not saved anything yet
        return {"state": {}, "warnings": [], "failed_step": None}


def _detail(ctx: AppContext, project: Project, rfp: Rfp) -> Dict[str, Any]:
    wf = _workflow_state(ctx, project.id)
    state = wf["state"]
    return {
        **_summary(project, rfp),
        "failed_step":     None if project.running else wf.get("failed_step"),
        "warnings":        wf.get("warnings", []),
        "rfp_summary":     state.get("rfp_answer"),
        "requirements":    state.get("rfp_requirements", []),
        "matched_items":   state.get("matched_items", []),
        "match_summary":   state.get("match_summary", {}),
        "line_items":      state.get("line_items", []),
        "pricing_summary": state.get("pricing_summary", {}),
        "unpriced":        [q for q in state.get("quoted_items", []) if q.get("catalog_cost_inr") is None],
        "review_log":      state.get("review_log", []),
    }


def _require_idle(project: Project) -> None:
    if project.running:
        raise HTTPException(status.HTTP_409_CONFLICT, "This project is still working; wait for it to finish")


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.post("", response_model=ProjectOut, status_code=status.HTTP_202_ACCEPTED)
def create_project(
    body: ProjectCreateIn,
    ctx: AppContext = Depends(get_ctx),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> Dict[str, Any]:
    rfp = db.get(Rfp, body.rfp_id)
    if rfp is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "RFP not found")
    project = Project(id=uuid.uuid4().hex[:12], rfp_id=rfp.id, owner_id=user.id, stage="queued")
    db.add(project)
    db.commit()

    pid, pdf = project.id, rfp.pdf_path
    ctx.jobs.submit(pid, PROGRESS_AFTER["start"], lambda: ctx.workflow.start(pid, pdf))
    db.refresh(project)
    return _detail(ctx, project, rfp)


@router.get("", response_model=List[ProjectSummaryOut])
def list_projects(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
) -> List[Dict[str, Any]]:
    rows = db.execute(
        select(Project, Rfp).join(Rfp, Rfp.id == Project.rfp_id)
        .order_by(Project.updated_at.desc()).limit(min(limit, 200)).offset(offset)
    ).all()
    return [_summary(p, r) for p, r in rows]


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(
    project_id: str,
    ctx: AppContext = Depends(get_ctx),
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
) -> Dict[str, Any]:
    project = _get_project(db, project_id)
    return _detail(ctx, project, db.get(Rfp, project.rfp_id))


@router.post("/{project_id}/review", response_model=ProjectOut, status_code=status.HTTP_202_ACCEPTED)
def submit_review(
    project_id: str,
    body: ReviewIn,
    ctx: AppContext = Depends(get_ctx),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> Dict[str, Any]:
    project = _get_project(db, project_id)
    _require_idle(project)
    decision = {"reviewer": user.name, **body.decision}
    try:
        checked = ctx.workflow.validate_decision(project_id, decision)
    except KeyError:
        raise HTTPException(status.HTTP_409_CONFLICT, "This project has not started yet")
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))

    stage = project.stage
    ctx.jobs.submit(project_id, PROGRESS_AFTER[stage], lambda: ctx.workflow.resume(project_id, checked))
    db.refresh(project)
    return _detail(ctx, project, db.get(Rfp, project.rfp_id))


@router.post("/{project_id}/retry", response_model=ProjectOut, status_code=status.HTTP_202_ACCEPTED)
def retry_project(
    project_id: str,
    ctx: AppContext = Depends(get_ctx),
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
) -> Dict[str, Any]:
    project = _get_project(db, project_id)
    _require_idle(project)
    rfp = db.get(Rfp, project.rfp_id)
    if project.stage == "failed" and not ctx.workflow.exists(project_id):
        # Failed before the first checkpoint was written: start again.
        pid, pdf = project.id, rfp.pdf_path
        ctx.jobs.submit(pid, PROGRESS_AFTER["start"], lambda: ctx.workflow.start(pid, pdf))
    else:
        failed_step = _workflow_state(ctx, project_id).get("failed_step")
        if failed_step is None:
            raise HTTPException(status.HTTP_409_CONFLICT, "Nothing to retry")
        ctx.jobs.submit(project_id, f"Retrying: {failed_step}", lambda: ctx.workflow.retry(project_id))
    db.refresh(project)
    return _detail(ctx, project, rfp)


@router.post("/{project_id}/pricing/preview", response_model=PricingPreviewOut)
def preview_pricing(
    project_id: str,
    body: PricingPreviewIn,
    ctx: AppContext = Depends(get_ctx),
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
) -> Dict[str, Any]:
    """Instant, deterministic re-price for the margin slider. Saves nothing."""
    _get_project(db, project_id)
    state = _workflow_state(ctx, project_id)["state"]
    line_items = state.get("line_items") or []
    if not line_items:
        raise HTTPException(status.HTTP_409_CONFLICT, "This project has not been priced yet")
    current = state.get("pricing_summary") or {}
    try:
        items, summary = reprice_with_margin(
            line_items,
            body.margin_pct if body.margin_pct is not None else current.get("margin_pct", PRICING_RULES.base_margin_pct),
            PRICING_RULES,
            advance_payment=(body.advance_payment if body.advance_payment is not None
                             else bool(current.get("advance_payment", False))),
            freight=body.freight_inr if body.freight_inr is not None else current.get("freight"),
            quantities=body.quantities,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    return {"line_items": items, "pricing_summary": summary}


@router.get("/{project_id}/quote.csv")
def download_quote(
    project_id: str,
    ctx: AppContext = Depends(get_ctx),
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
):
    _get_project(db, project_id)
    payload = _workflow_state(ctx, project_id)["state"].get("quote_payload")
    if not payload or not payload.get("line_items"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No quote yet")
    path = export_quote_csv(payload, str(ctx.exports_dir / project_id / "quote.csv"))
    return FileResponse(path, media_type="text/csv", filename=f"quote_{project_id}.csv")


@router.get("/{project_id}/proposal.pdf")
def download_proposal(
    project_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
):
    project = _get_project(db, project_id)
    if not (project.proposal_path and Path(project.proposal_path).is_file()):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No proposal PDF yet")
    return FileResponse(project.proposal_path, media_type="application/pdf",
                        filename=f"proposal_{project_id}.pdf", content_disposition_type="inline")


@router.patch("/{project_id}/outcome", response_model=ProjectSummaryOut)
def set_outcome(
    project_id: str,
    body: OutcomeIn,
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
) -> Dict[str, Any]:
    project = _get_project(db, project_id)
    project.outcome = body.outcome
    db.commit()
    db.refresh(project)
    return _summary(project, db.get(Rfp, project.rfp_id))
