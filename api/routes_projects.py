"""Projects: one review workflow per RFP (start, status, review, pricing, files)."""

from __future__ import annotations

import copy
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from db import Project, Rfp, User
from llm_json import LLMOutputError
from pricing_agent import reprice_with_margin
from quote_export import export_quote_csv
from settings import PRICING_RULES
from proposal_writer import amount_warnings, regenerate_section
from report_agent import render_state_pdf
from workflow import STAGE_MATCHING, STAGE_PRICING, STAGE_PROPOSAL, STAGE_REQUIREMENTS, apply_proposal_review, ProposalDecision

from .context import AppContext, current_user, get_ctx, get_db
from .schemas import (
    OutcomeIn,
    PricingPreviewIn,
    PricingPreviewOut,
    ProjectCreateIn,
    ProjectOut,
    ProjectSummaryOut,
    ProposalEditIn,
    RegenerateIn,
    RegenerateOut,
    ReviewIn,
    RfpOut,
)

router = APIRouter(prefix="/api/projects", tags=["projects"])
rules_router = APIRouter(prefix="/api/pricing", tags=["projects"])


@rules_router.get("/rules")
def pricing_rules(_: User = Depends(current_user)) -> Dict[str, Any]:
    """The commercial rules the UI shows (slider limits, GST, discount tiers)."""
    r = PRICING_RULES
    return {
        "base_margin_pct": r.base_margin_pct,
        "min_margin_pct": r.min_margin_pct,
        "max_margin_pct": r.max_margin_pct,
        "gst_rate_pct": r.gst_rate_pct,
        "currency": r.currency,
        "volume_discount_tiers": r.volume_discount_tiers,
        "payment_terms_discount_pct": r.payment_terms_discount_pct,
        "freight_flat_inr": r.freight_flat_inr,
        "default_qty": r.default_qty,
    }

# What the background job is doing, by the stage it starts from.
PROGRESS_AFTER = {
    "start":            "Reading the RFP and extracting requirements",
    STAGE_REQUIREMENTS: "Matching requirements to catalogue products",
    STAGE_MATCHING:     "Calculating prices",
    STAGE_PRICING:      "Drafting the proposal",
    STAGE_PROPOSAL:     "Rendering the proposal PDF",
}


def rfp_meta(rfp: Rfp) -> Dict[str, Any]:
    """What the proposal needs to know about the RFP (from its record)."""
    return {"title": rfp.title, "buyer": rfp.buyer, "reference": (rfp.meta or {}).get("reference"),
            "deadline": rfp.deadline, "location": rfp.location}


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
        "proposal_sections": state.get("proposal_sections", []),
        "proposal_meta":   {k: v for k, v in (state.get("proposal_meta") or {}).items() if k != "profile"},
        "proposal_warnings": amount_warnings(state.get("proposal_sections", [])),
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

    pid, pdf, meta = project.id, rfp.pdf_path, rfp_meta(rfp)
    ctx.jobs.submit(pid, PROGRESS_AFTER["start"], lambda: ctx.workflow.start(pid, pdf, rfp_meta=meta))
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
        pid, pdf, meta = project.id, rfp.pdf_path, rfp_meta(rfp)
        ctx.jobs.submit(pid, PROGRESS_AFTER["start"], lambda: ctx.workflow.start(pid, pdf, rfp_meta=meta))
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


# ─── Proposal editor ──────────────────────────────────────────────────────────

def _proposal_state(ctx: AppContext, db: Session, project_id: str) -> Dict[str, Any]:
    """Working state of a project paused at the proposal review."""
    project = _get_project(db, project_id)
    _require_idle(project)
    wf = _workflow_state(ctx, project_id)
    if project.stage != STAGE_PROPOSAL or not wf["state"].get("proposal_sections"):
        raise HTTPException(status.HTTP_409_CONFLICT, "This project is not at the proposal review")
    return wf["state"]


def _with_edits(state: Dict[str, Any], sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not sections:
        return state
    try:
        return apply_proposal_review(copy.deepcopy(state), ProposalDecision.model_validate({"sections": sections}))
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))


@router.post("/{project_id}/proposal/preview")
def preview_proposal(
    project_id: str,
    body: ProposalEditIn,
    ctx: AppContext = Depends(get_ctx),
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
):
    """Render the PDF with the editor's current (unsaved) text. Saves nothing."""
    state = _with_edits(_proposal_state(ctx, db, project_id), [s.model_dump() for s in body.sections])
    with tempfile.TemporaryDirectory() as tmp:
        path = render_state_pdf(state, str(Path(tmp) / "preview.pdf"))
        data = Path(path).read_bytes()
    return Response(data, media_type="application/pdf", headers={"Cache-Control": "no-store"})


@router.post("/{project_id}/proposal/sections/{key}/regenerate", response_model=RegenerateOut)
def regenerate(
    project_id: str,
    key: str,
    body: RegenerateIn,
    ctx: AppContext = Depends(get_ctx),
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
) -> Dict[str, Any]:
    """AI rewrite of one section (not saved until the reviewer saves or approves)."""
    state = _with_edits(_proposal_state(ctx, db, project_id), [s.model_dump() for s in body.sections])
    try:
        text = regenerate_section(state, key, body.instruction)
    except LLMOutputError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"The AI did not return usable text: {exc}")
    except ValueError as exc:   # not an AI section, or no API key
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    return {"key": key, "body": text,
            "warning": amount_warnings([{"key": key, "body": text}]).get(key)}


@router.put("/{project_id}/proposal/draft", response_model=ProjectOut)
def save_draft(
    project_id: str,
    body: ProposalEditIn,
    ctx: AppContext = Depends(get_ctx),
    db: Session = Depends(get_db),
    _: User = Depends(current_user),
) -> Dict[str, Any]:
    """Save edited sections and stay at the proposal review."""
    _proposal_state(ctx, db, project_id)
    try:
        ctx.workflow.save_proposal_draft(project_id, [s.model_dump() for s in body.sections])
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    project = _get_project(db, project_id)
    return _detail(ctx, project, db.get(Rfp, project.rfp_id))


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
