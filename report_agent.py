"""
agents/report_agent.py
----------------------
Report Generation Agent — LangGraph node (used by master_agent's "report"
route; the review workflow calls the two halves separately so a human can
edit the sections in between).

  1. proposal_writer.draft_proposal()   LLM writes the sections (Gemini when
                                         GEMINI_API_KEY is set, else OpenRouter)
  2. proposal_pdf.render_proposal()      deterministic PDF: every amount comes
                                         from the pricing agent

Writes state["proposal_sections"], state["proposal_meta"],
state["report_pdf_path"] and state["report_text"] (all sections as text).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from company_profile import get_profile
from llm_json import LLMOutputError
from proposal_pdf import not_quoted_items, render_proposal
from proposal_writer import draft_proposal
from state import AgentState

logger = logging.getLogger(__name__)


def sections_as_text(sections) -> str:
    return "\n\n".join(f"{s['title']}\n{s['body']}" for s in sections)


def render_state_pdf(state: AgentState, path: str) -> str:
    """Render the proposal PDF for the sections and pricing in *state*."""
    return render_proposal(
        path,
        sections=state.get("proposal_sections", []),
        proposal_meta=state.get("proposal_meta", {}),
        line_items=state.get("line_items", []),
        pricing_summary=state.get("pricing_summary", {}),
        not_quoted=not_quoted_items(state),
    )


def report_agent_node(
    state: AgentState,
    output_dir: str = ".",
    api_key: Optional[str] = None,   # kept for callers; keys come from settings / .env
) -> AgentState:
    """Draft and render the proposal in one go (no human review)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    try:
        sections, meta = draft_proposal(state, get_profile(), project_ref=f"PRP-{ts}")
    except (LLMOutputError, ValueError) as exc:   # ValueError: no API key configured
        state["error"] = f"Proposal text could not be generated: {exc}"
        logger.error(state["error"])
        return state

    state["proposal_sections"] = sections
    state["proposal_meta"] = meta
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"vendor_proposal_{ts}.pdf")
    try:
        render_state_pdf(state, path)
    except Exception as exc:
        state["error"] = f"PDF rendering failed: {exc}"
        logger.exception("PDF rendering failed")
        return state

    logger.info("[ReportAgent] PDF ready: %s", path)
    state["report_pdf_path"] = path
    state["report_text"] = sections_as_text(sections)
    return state
