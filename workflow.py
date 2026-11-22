"""
workflow.py
-----------
The per-RFP project workflow, with human review checkpoints.

    extract ─► ⏸ review_requirements ─► match ─► ⏸ review_matches
            ─► price ─► ⏸ review_pricing ─► draft ─► ⏸ review_proposal
            ─► render ─► END

Each project is one LangGraph thread (thread_id = project_id). At each ⏸
the graph calls interrupt() and stops; the state is saved by the SQLite
checkpointer (settings.CHECKPOINT_DB_PATH), so a review can be finished
later, even after the app restarts. ProjectWorkflow.resume() validates the
reviewer's decision and continues the graph.

The question-answering CLI graph in master_agent.py is unchanged.

Decisions (what a reviewer can send at each stage)
--------------------------------------------------
requirements_review  RequirementsDecision: the full edited requirement list
                     (or nothing, to approve as extracted)
matching_review      MatchesDecision: accept / reject / change product /
                     set quantity per item; confirm / not_met for compliance;
                     accept_all for the remaining suggestions
pricing_review       PricingDecision: margin, advance payment, freight,
                     quantities per item (re-priced deterministically)
proposal_review      ProposalDecision: the edited section texts (or nothing,
                     to keep the AI draft); then the PDF is rendered

Note: when resumed, LangGraph re-runs the interrupted node from the top, so
review nodes must not have side effects before interrupt().
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import sqlite3
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, model_validator

from catalogue import get_catalogue
from matching_agent import apply_product, fulfillment_report, match_summary, matching_agent_node
from pricing_agent import pricing_agent_node
from company_profile import CompanyProfile, get_profile
from proposal_writer import draft_proposal
from report_agent import render_state_pdf, sections_as_text
from rfp_agent import CRITICALITY_WEIGHTS, rfp_agent_node
from settings import CHECKPOINT_DB_PATH, PROPOSALS_DIR
from state import AgentState

logger = logging.getLogger(__name__)

STAGE_REQUIREMENTS = "requirements_review"
STAGE_MATCHING = "matching_review"
STAGE_PRICING = "pricing_review"
STAGE_PROPOSAL = "proposal_review"

# Used as the RFP agent's question, so rfp_answer is a useful overview for
# the proposal writer.
DEFAULT_RFP_QUERY = (
    "Summarise this RFP: the buyer, scope of supply, quantities, delivery terms, "
    "eligibility criteria and submission requirements."
)


# ─── Reviewer decisions ───────────────────────────────────────────────────────

Criticality = Literal["mandatory", "high", "medium", "low"]


class _Decision(BaseModel):
    reviewer: Optional[str] = None
    note: str = ""


class RequirementEdit(BaseModel):
    id: Optional[str] = None            # keep an existing id; omit for a new requirement
    requirement: str = Field(min_length=3)
    criticality: Criticality = "medium"
    quantity: Optional[int] = Field(default=None, gt=0)


class RequirementsDecision(_Decision):
    requirements: Optional[List[RequirementEdit]] = None   # None = approve as extracted


class MatchDecisionItem(BaseModel):
    id: str
    action: Literal["accept", "reject", "change", "confirm", "not_met"]
    product_id: Optional[str] = None    # required for "change"
    quantity: Optional[int] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _change_needs_product(self) -> "MatchDecisionItem":
        if self.action == "change" and not self.product_id:
            raise ValueError("action 'change' needs a product_id")
        return self


class MatchesDecision(_Decision):
    decisions: List[MatchDecisionItem] = []
    accept_all: bool = False            # approve every remaining product suggestion


class PricingDecision(_Decision):
    margin_pct: Optional[float] = Field(default=None, ge=0)
    advance_payment: Optional[bool] = None
    freight_inr: Optional[float] = Field(default=None, ge=0)
    quantities: Dict[str, int] = {}     # matched item id → quantity

    @model_validator(mode="after")
    def _positive_quantities(self) -> "PricingDecision":
        bad = [k for k, v in self.quantities.items() if v <= 0]
        if bad:
            raise ValueError(f"quantities must be positive: {bad}")
        return self


class SectionEdit(BaseModel):
    key: str
    title: Optional[str] = Field(default=None, min_length=1, max_length=120)
    body: str = Field(max_length=20_000)


class ProposalDecision(_Decision):
    sections: Optional[List[SectionEdit]] = None   # None = keep the draft as written


DECISION_SCHEMAS = {
    STAGE_REQUIREMENTS: RequirementsDecision,
    STAGE_MATCHING: MatchesDecision,
    STAGE_PRICING: PricingDecision,
    STAGE_PROPOSAL: ProposalDecision,
}


# ─── Applying decisions (pure; also used to validate before resuming) ────────

def _log(state: AgentState, stage: str, decision: _Decision, changes: Dict[str, Any]) -> None:
    entry = {
        "stage":    stage,
        "at":       datetime.now(timezone.utc).isoformat(),
        "reviewer": decision.reviewer,
        "note":     decision.note,
        "changes":  changes,
    }
    state["review_log"] = list(state.get("review_log", [])) + [entry]


_CARRIED_REQUIREMENT_FIELDS = ("unit", "origin", "source", "attributes", "quantity_breakdown")


def apply_requirements_review(state: AgentState, decision: RequirementsDecision) -> AgentState:
    if decision.requirements is None:
        _log(state, STAGE_REQUIREMENTS, decision, {"edited": False})
        return state

    old_by_id = {r["id"]: r for r in state.get("rfp_requirements", [])}
    old_ids = set(old_by_id)
    numbers = [int(m.group(1)) for i in old_ids if (m := re.fullmatch(r"R(\d+)", i))]
    next_number = max(numbers, default=0) + 1

    used: set = set()
    edited: List[Dict[str, Any]] = []
    for edit in decision.requirements:
        rid = edit.id if edit.id and edit.id not in used else None
        if rid is None:
            rid = f"R{next_number}"
            next_number += 1
        used.add(rid)
        # Where the requirement came from (page / table row), its unit and
        # other table columns are kept for an existing id; the reviewer edits
        # only text, criticality and quantity.
        kept = {k: v for k, v in old_by_id.get(rid, {}).items() if k in _CARRIED_REQUIREMENT_FIELDS}
        edited.append({
            "id":          rid,
            "requirement": edit.requirement.strip(),
            "criticality": edit.criticality,
            "weight":      CRITICALITY_WEIGHTS[edit.criticality],
            "quantity":    edit.quantity,
            **kept,
            **({} if rid in old_ids else {"origin": "reviewer"}),
        })

    state["rfp_requirements"] = edited
    _log(state, STAGE_REQUIREMENTS, decision, {
        "edited":  True,
        "before":  len(old_ids),
        "after":   len(edited),
        "added":   len(used - old_ids),
        "removed": len(old_ids - used),
    })
    return state


def apply_matches_review(state: AgentState, decision: MatchesDecision) -> AgentState:
    items = [dict(m) for m in state.get("matched_items", [])]
    by_id = {m["id"]: m for m in items}
    changes: Counter = Counter()
    decided: set = set()

    for d in decision.decisions:
        item = by_id.get(d.id)
        if item is None:
            raise ValueError(f"Unknown item id: {d.id}")
        kind = item.get("type")

        if d.action in ("confirm", "not_met"):
            if kind != "compliance":
                raise ValueError(f"{d.id} is not a compliance item; use accept, reject or change")
            item["status"] = "met" if d.action == "confirm" else "not_met"
        else:
            if kind != "product":
                raise ValueError(f"{d.id} is a compliance item; use confirm or not_met")
            if d.action == "accept":
                if not item.get("product_id"):
                    raise ValueError(f"{d.id} has no product to accept; use change")
                item.update(status="approved", matched=True)
            elif d.action == "reject":
                item.update(status="rejected", matched=False)
            else:  # change
                product = get_catalogue().get(d.product_id)
                if product is None:
                    raise ValueError(f"Unknown product_id for {d.id}: {d.product_id}")
                replacement = apply_product(item, product, score=None, confidence="manual",
                                            rationale="Chosen by reviewer.")
                item.clear()
                item.update(replacement, status="approved")

        if d.quantity is not None:
            if kind != "product":
                raise ValueError(f"{d.id}: quantities only apply to product items")
            item.update(qty=d.quantity, qty_reviewed=True)
            if item.get("stock_on_hand") is not None:
                item["stock_sufficient"] = item["stock_on_hand"] >= d.quantity

        decided.add(d.id)
        changes[d.action] += 1

    if decision.accept_all:
        for item in items:
            if (item["id"] not in decided and item.get("type") == "product"
                    and item.get("status") in ("matched", "needs_review")):
                item.update(status="approved", matched=True)
                changes["accepted_by_accept_all"] += 1

    summary = match_summary(items)
    state["matched_items"] = items
    state["match_summary"] = summary
    state["fulfillment_score"] = summary["fulfillment_score"]
    state["fulfillment_report"] = fulfillment_report(items, summary)
    _log(state, STAGE_MATCHING, decision, dict(changes))
    return state


def apply_pricing_review(state: AgentState, decision: PricingDecision) -> AgentState:
    changes: Dict[str, Any] = {}
    for field, key in (("margin_pct", "margin_pct"), ("advance_payment", "advance_payment"),
                       ("freight_inr", "freight_inr")):
        value = getattr(decision, field)
        if value is not None:
            state[key] = value
            changes[key] = value

    if decision.quantities:
        items = [dict(m) for m in state.get("matched_items", [])]
        by_id = {m["id"]: m for m in items}
        for item_id, qty in decision.quantities.items():
            item = by_id.get(item_id)
            if item is None or item.get("type") != "product":
                raise ValueError(f"Unknown product item id: {item_id}")
            item.update(qty=qty, qty_reviewed=True)
        state["matched_items"] = items
        changes["quantities"] = dict(decision.quantities)

    # Deterministic re-price (no retrieval, no LLM) with the new inputs.
    state = pricing_agent_node(state)
    _log(state, STAGE_PRICING, decision, changes)
    return state


def apply_proposal_review(state: AgentState, decision: ProposalDecision) -> AgentState:
    sections = [dict(s) for s in state.get("proposal_sections", [])]
    by_key = {s["key"]: s for s in sections}
    edited = []
    for edit in decision.sections or []:
        section = by_key.get(edit.key)
        if section is None:
            raise ValueError(f"Unknown proposal section: {edit.key}")
        if edit.body.strip() != section["body"].strip() or (edit.title and edit.title != section["title"]):
            section["body"] = edit.body.strip()
            section["title"] = edit.title or section["title"]
            section["edited"] = True
            edited.append(edit.key)
    state["proposal_sections"] = sections
    _log(state, STAGE_PROPOSAL, decision, {"edited_sections": edited})
    return state


APPLY = {
    STAGE_REQUIREMENTS: apply_requirements_review,
    STAGE_MATCHING: apply_matches_review,
    STAGE_PRICING: apply_pricing_review,
    STAGE_PROPOSAL: apply_proposal_review,
}


# ─── Graph ────────────────────────────────────────────────────────────────────

def _review_node(stage: str) -> Callable[[AgentState], AgentState]:
    schema, apply = DECISION_SCHEMAS[stage], APPLY[stage]

    def node(state: AgentState) -> AgentState:
        # Nothing before interrupt(): this line re-runs when the graph resumes.
        decision = interrupt({"stage": stage})
        return apply(state, schema.model_validate(decision))

    node.__name__ = f"review_{stage}"
    return node


def _extract(state: AgentState) -> AgentState:
    if not state.get("user_query"):
        state["user_query"] = DEFAULT_RFP_QUERY
    return rfp_agent_node(state)


def project_reference(project_id: str) -> str:
    """Human reference for a project (same as the dashboard shows)."""
    return f"SYN-{project_id[:6].upper()}"


def _make_draft(profile_provider: Callable[[], CompanyProfile]) -> Callable[[AgentState], AgentState]:
    def draft(state: AgentState) -> AgentState:
        sections, meta = draft_proposal(state, profile_provider(),
                                        project_ref=project_reference(state.get("project_id", "")))
        state["proposal_sections"] = sections
        state["proposal_meta"] = meta
        return state
    return draft


def _render(state: AgentState) -> AgentState:
    out_dir = Path(PROPOSALS_DIR) / state.get("project_id", "default")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"proposal_{project_reference(state.get('project_id', ''))}.pdf"
    state["report_pdf_path"] = render_state_pdf(state, str(path))
    state["report_text"] = sections_as_text(state.get("proposal_sections", []))
    return state


def build_workflow(
    checkpointer: Any,
    nodes: Optional[Dict[str, Callable]] = None,
    profile_provider: Callable[[], CompanyProfile] = get_profile,
):
    """
    Compile the review workflow. *nodes* can replace the agent steps
    ("extract", "match", "price", "draft", "render"), e.g. with fakes in tests.
    *profile_provider* returns the company profile used to draft proposals.
    """
    steps = {"extract": _extract, "match": matching_agent_node, "price": pricing_agent_node,
             "draft": _make_draft(profile_provider), "render": _render}
    steps.update(nodes or {})

    graph = StateGraph(AgentState)
    order = [
        ("extract", steps["extract"]),
        (STAGE_REQUIREMENTS, _review_node(STAGE_REQUIREMENTS)),
        ("match", steps["match"]),
        (STAGE_MATCHING, _review_node(STAGE_MATCHING)),
        ("price", steps["price"]),
        (STAGE_PRICING, _review_node(STAGE_PRICING)),
        ("draft", steps["draft"]),
        (STAGE_PROPOSAL, _review_node(STAGE_PROPOSAL)),
        ("render", steps["render"]),
    ]
    for name, fn in order:
        graph.add_node(name, fn)
    graph.add_edge(START, order[0][0])
    for (a, _), (b, _) in zip(order, order[1:]):
        graph.add_edge(a, b)
    graph.add_edge(order[-1][0], END)
    return graph.compile(checkpointer=checkpointer)


# ─── Runner ───────────────────────────────────────────────────────────────────

class ProjectWorkflow:
    """
    Start, inspect and resume review workflows. One instance per process;
    calls for the same project are serialised.

    status() returns:
      {"project_id", "stage", "error", "failed_step", "warnings", "state"}
    where stage is one of requirements_review | matching_review |
    pricing_review | failed | done.
    """

    def __init__(self, checkpoint_path: str = CHECKPOINT_DB_PATH,
                 nodes: Optional[Dict[str, Callable]] = None,
                 profile_provider: Callable[[], CompanyProfile] = get_profile) -> None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(checkpoint_path, check_same_thread=False)
        self.graph = build_workflow(SqliteSaver(self._conn), nodes, profile_provider)
        self._lock = threading.Lock()

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _config(project_id: str) -> Dict[str, Any]:
        return {"configurable": {"thread_id": project_id}}

    def exists(self, project_id: str) -> bool:
        return bool(self.graph.get_state(self._config(project_id)).values)

    def start(self, project_id: str, rfp_pdf_path: str, user_query: Optional[str] = None,
              rfp_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run extraction for a new project and stop at the first review."""
        if self.exists(project_id):
            raise ValueError(f"Project {project_id} already exists")
        initial: AgentState = {"project_id": project_id, "rfp_pdf_path": rfp_pdf_path,
                               "rfp_meta": dict(rfp_meta or {})}
        if user_query:
            initial["user_query"] = user_query
        return self._run(project_id, initial)

    def save_proposal_draft(self, project_id: str, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Save edited proposal sections while staying paused at proposal_review.

        A plain update_state() would drop the pending interrupt, so the update
        is recorded as if the "draft" step produced it, and the graph is run
        forward to pause at the review again (the review node has no side
        effects before interrupt()). Verified against LangGraph 1.2.
        """
        current = self.status(project_id)
        if current["stage"] != STAGE_PROPOSAL:
            raise ValueError(f"Project {project_id} is not at the proposal review (stage: {current['stage']})")
        edits = ProposalDecision.model_validate({"sections": sections})
        updated = apply_proposal_review(copy.deepcopy(current["state"]), edits)
        with self._lock:
            config = self._config(project_id)
            self.graph.update_state(config, {"proposal_sections": updated["proposal_sections"]}, as_node="draft")
            self.graph.invoke(None, config)
        return self.status(project_id)

    def validate_decision(self, project_id: str, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check a reviewer decision against the current stage without moving
        the graph. Returns the normalised decision; raises ValueError (incl.
        pydantic ValidationError) or KeyError for an unknown project.
        """
        current = self.status(project_id)
        stage = current["stage"]
        if stage not in DECISION_SCHEMAS:
            raise ValueError(f"Project {project_id} is not waiting for a review (stage: {stage})")
        parsed = DECISION_SCHEMAS[stage].model_validate(decision)
        # Dry run on a copy, so bad ids are rejected before the graph moves on.
        APPLY[stage](copy.deepcopy(current["state"]), parsed)
        return parsed.model_dump()

    def resume(self, project_id: str, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a reviewer decision for the current stage and continue."""
        checked = self.validate_decision(project_id, decision)
        return self._run(project_id, Command(resume=checked))

    def retry(self, project_id: str) -> Dict[str, Any]:
        """Re-run the step that failed (e.g. after adding a missing API key)."""
        if self.status(project_id)["stage"] != "failed":
            raise ValueError(f"Project {project_id} has no failed step to retry")
        return self._run(project_id, None)

    def status(self, project_id: str) -> Dict[str, Any]:
        snap = self.graph.get_state(self._config(project_id))
        if not snap.values and not snap.next:
            raise KeyError(f"Unknown project: {project_id}")

        stage, error, failed_step = "done", None, None
        for task in snap.tasks:
            if task.interrupts:
                stage = task.interrupts[0].value["stage"]
                break
            if task.error:
                stage, error, failed_step = "failed", str(task.error), task.name
                break
        else:
            if snap.next:  # a step is scheduled but has neither paused nor failed
                stage = "running"

        values = dict(snap.values)
        if stage == "done" and values.get("error"):
            error = values["error"]   # e.g. proposal text could not be generated
        return {
            "project_id":  project_id,
            "stage":       stage,
            "error":       error,
            "failed_step": failed_step,
            "warnings":    values.get("warnings", []),
            "state":       values,
        }

    def _run(self, project_id: str, graph_input: Any) -> Dict[str, Any]:
        with self._lock:
            try:
                self.graph.invoke(graph_input, self._config(project_id))
            except Exception:  # recorded on the checkpoint; status() reports it
                logger.exception("Workflow step failed for project %s", project_id)
        return self.status(project_id)


# ─── CLI (try the review flow without the UI) ─────────────────────────────────

def _print_status(status: Dict[str, Any]) -> None:
    s = status["state"]
    print(f"\nProject {status['project_id']}  →  stage: {status['stage']}")
    if status["error"]:
        print(f"  error: {status['error']}" + (f" (step: {status['failed_step']})" if status["failed_step"] else ""))
    for w in status["warnings"]:
        print(f"  warning: {w}")
    if status["stage"] == STAGE_REQUIREMENTS:
        for r in s.get("rfp_requirements", []):
            qty = f"  qty {r['quantity']}" if r.get("quantity") else ""
            print(f"  {r['id']:>4} [{r['criticality']:<9}] {r['requirement']}{qty}")
    elif status["stage"] == STAGE_MATCHING:
        print(s.get("fulfillment_report", ""))
    elif status["stage"] == STAGE_PRICING:
        print(s.get("pricing_report", ""))
    elif status["stage"] == STAGE_PROPOSAL:
        for section in s.get("proposal_sections", []):
            print(f"\n== {section['title']} ==\n{section['body']}")
    elif status["stage"] == "done":
        print(f"  proposal: {s.get('report_pdf_path', 'not generated')}")
        print(f"  grand total: {(s.get('pricing_summary') or {}).get('grand_total')}")


def main() -> int:
    from log_setup import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser(description="Run the RFP review workflow from the terminal.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_start = sub.add_parser("start", help="start a project for an RFP PDF")
    p_start.add_argument("--rfp", required=True)
    p_start.add_argument("--id", help="project id (default: random)")
    p_status = sub.add_parser("status", help="show where a project is")
    p_status.add_argument("id")
    p_approve = sub.add_parser("approve", help="approve the current review (optionally with edits)")
    p_approve.add_argument("id")
    p_approve.add_argument("--decision", help="JSON file with the reviewer's decision")
    p_retry = sub.add_parser("retry", help="re-run a failed step")
    p_retry.add_argument("id")
    args = parser.parse_args()

    wf = ProjectWorkflow()
    try:
        if args.command == "start":
            status = wf.start(args.id or uuid.uuid4().hex[:8], args.rfp)
        elif args.command == "status":
            status = wf.status(args.id)
        elif args.command == "retry":
            status = wf.retry(args.id)
        else:
            if args.decision:
                decision = json.loads(Path(args.decision).read_text(encoding="utf-8"))
            else:  # approve as-is; at matching that means accepting the suggestions
                decision = {"accept_all": True} if wf.status(args.id)["stage"] == STAGE_MATCHING else {}
            status = wf.resume(args.id, decision)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    _print_status(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
