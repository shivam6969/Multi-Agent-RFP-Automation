"""
agents/rfp_agent.py
-------------------
RFP Agent — LangGraph node

Responsibility
--------------
  1. Answer the user's query from the RFP document (state["rfp_answer"]).
  2. Extract a structured list of RFP requirements for downstream agents
     (state["rfp_requirements"]): [{id, requirement, criticality, weight,
     quantity}]. The WHOLE document is read, in windows that are processed
     in parallel, and every LLM reply is schema-validated (llm_json.py).

Both outputs are produced in a single run to avoid redundant PDF retrieval.
"""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, field_validator

from settings import LLM_MAX_CONCURRENCY, OPENROUTER_API_KEY, RFP_EXTRACT_WINDOW_CHARS, RFP_TOP_K
from rfp_rag import RfpRagConfig, get_rfp_rag_tool
from llm_json import LLMOutputError, chat_json
from state import AgentState

logger = logging.getLogger(__name__)

_W = 72  # print banner width


def _wrap(text: str, indent: int = 2) -> List[str]:
    """Wrap text to fit inside the banner, yielding padded lines."""
    width = _W - indent - 2  # 2 for ║ chars
    lines = []
    for raw_line in (text or "").splitlines() or [""]:
        if not raw_line:
            lines.append("")
            continue
        for i in range(0, max(len(raw_line), 1), width):
            lines.append(raw_line[i : i + width])
    return lines


# Box-drawing report helpers: append lines to *out*; the node logs it once.
def _banner(out: List[str], title: str) -> None:
    out += [f"╔{'═' * (_W - 2)}╗", f"║  {title:<{_W - 4}}║", f"╠{'═' * (_W - 2)}╣"]


def _section(out: List[str], label: str) -> None:
    out += [f"║  {label:<{_W - 4}}║", f"╠{'─' * (_W - 2)}╣"]


def _line(out: List[str], text: str, indent: int = 2) -> None:
    pad = " " * indent
    out.append(f"║{pad}{text:<{_W - indent - 2}}║")


def _footer(out: List[str]) -> None:
    out.append(f"╚{'═' * (_W - 2)}╝")


# ─── Requirement extraction ───────────────────────────────────────────────────

_REQ_EXTRACTION_PROMPT = """
You are an expert procurement analyst.

Extract every requirement stated in this part of an RFP as a JSON array of
objects. Include product line items (one entry per product line as written),
technical specifications, eligibility criteria, delivery/logistics terms and
submission obligations.

If the RFP gives a general scope summary (e.g. "supply switches, plates and
accessories") AND a detailed list of the same products, extract only the
detailed list items, not the summary line.

Each object must have:
  - "requirement"  : short plain-English statement of the requirement
  - "criticality"  : one of
        "mandatory" — the RFP says must / shall / mandatory, or failing it
                      disqualifies the bid (e.g. eligibility criteria)
        "high"      — core product lines and key commercial terms
        "medium"    — supporting specifications and documents
        "low"       — nice-to-have or "where applicable" items
  - "quantity"     : integer number of units ONLY if the RFP states an explicit
                     order quantity for this specific item, otherwise null.
                     Do not use monthly supply estimates, packing formats or
                     numbers that are part of a product name (e.g. "RJ-45").

If this part contains no requirements, return [].
Return ONLY valid JSON. No markdown, no explanation.

RFP text (part {part} of {parts}):
{context}
""".strip()

CRITICALITY_WEIGHTS = {"mandatory": 4.0, "high": 3.0, "medium": 2.0, "low": 1.0}


class _Requirement(BaseModel):
    requirement: str
    criticality: str = "medium"
    quantity: Optional[int] = None

    @field_validator("criticality", mode="before")
    @classmethod
    def _known_criticality(cls, value: Any) -> str:
        value = str(value or "").strip().lower()
        return value if value in CRITICALITY_WEIGHTS else "medium"

    @field_validator("quantity", mode="before")
    @classmethod
    def _positive_int_or_none(cls, value: Any) -> Optional[int]:
        try:
            number = int(str(value).replace(",", "").strip())
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None


def _windows(chunks: List[str], max_chars: int) -> List[str]:
    """Group consecutive chunks into windows of at most ~max_chars."""
    windows: List[str] = []
    current: List[str] = []
    size = 0
    for chunk in chunks:
        if current and size + len(chunk) > max_chars:
            windows.append("\n\n".join(current))
            current, size = [], 0
        current.append(chunk)
        size += len(chunk)
    if current:
        windows.append("\n\n".join(current))
    return windows


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _extract_requirements(
    rfp_tool,
    api_key: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Read the WHOLE RFP (in windows, in parallel) and return
    ([{id, requirement, criticality, weight, quantity}], warnings).
    """
    chunks = getattr(rfp_tool, "chunks", None) or [
        rfp_tool.build_context("requirements eligibility deliverables mandatory technical", top_k=5)
    ]
    windows = _windows(chunks, RFP_EXTRACT_WINDOW_CHARS)

    def extract(numbered: Tuple[int, str]) -> Optional[List[_Requirement]]:
        part, text = numbered
        try:
            return chat_json(
                system="You extract structured data from RFP documents. Return JSON only.",
                user=_REQ_EXTRACTION_PROMPT.format(context=text, part=part, parts=len(windows)),
                schema=List[_Requirement],
                temperature=0.0,
                max_tokens=3000,
                api_key=api_key,
            )
        except LLMOutputError:
            return None

    with ThreadPoolExecutor(max_workers=max(1, LLM_MAX_CONCURRENCY)) as pool:
        results = list(pool.map(extract, enumerate(windows, 1)))

    warnings = [
        f"Requirement extraction failed for part {i} of {len(windows)} of the RFP; "
        "some requirements may be missing."
        for i, r in enumerate(results, 1) if r is None
    ]

    requirements: List[Dict[str, Any]] = []
    seen: Dict[str, Dict[str, Any]] = {}
    for found in results:
        for req in found or []:
            text = req.requirement.strip()
            key = _normalise(text)
            if len(key) < 6:
                continue
            if key in seen:  # same requirement repeated across windows
                if seen[key]["quantity"] is None:
                    seen[key]["quantity"] = req.quantity
                continue
            item = {
                "id":          f"R{len(requirements) + 1}",
                "requirement": text,
                "criticality": req.criticality,
                "weight":      CRITICALITY_WEIGHTS[req.criticality],
                "quantity":    req.quantity,
            }
            seen[key] = item
            requirements.append(item)
    return requirements, warnings


# ─── LangGraph node ───────────────────────────────────────────────────────────

def rfp_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node.
    Reads user_query (+ optional rfp_pdf_path), writes rfp_answer and
    rfp_requirements.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
    query = state.get("user_query", "")

    pdf_path = state.get("rfp_pdf_path") or RfpRagConfig().pdf_path
    rfp_tool = get_rfp_rag_tool(RfpRagConfig(pdf_path=pdf_path))

    logger.info("[RfpAgent] Answering user query from RFP …")
    # Short RFPs: answer from the whole document (the 3 nearest chunks missed
    # the buyer's name on the sample RFP). Long ones: the nearest chunks.
    chunks = getattr(rfp_tool, "chunks", [])
    whole_doc = 0 < sum(len(c) for c in chunks) <= RFP_EXTRACT_WINDOW_CHARS
    rfp_answer = rfp_tool.run_query(
        query, top_k=len(chunks) if whole_doc else RFP_TOP_K, api_key=api_key
    )
    state["rfp_answer"] = rfp_answer

    logger.info("[RfpAgent] Extracting structured requirements …")
    requirements, warnings = _extract_requirements(rfp_tool, api_key)
    state["rfp_requirements"] = requirements
    if warnings:
        state["warnings"] = list(state.get("warnings", [])) + warnings

    # ── Report for the log ─────────────────────────────────────────────────────
    crit_icon = {"mandatory": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}

    out: List[str] = [""]
    _banner(out, "RFP AGENT — OUTPUT")
    _line(out, f"Query: {query[:_W - 10]}")
    _section(out, "ANSWER")
    for line in _wrap(rfp_answer):
        _line(out, line)
    out.append(f"╠{'═' * (_W - 2)}╣")
    _section(out, f"EXTRACTED REQUIREMENTS  ({len(requirements)} total)")
    for i, req in enumerate(requirements, 1):
        crit = req.get("criticality", "medium")
        icon = crit_icon.get(crit, "•")
        wt   = req.get("weight", 1.0)
        text = req.get("requirement", "")
        _line(out, f"{i:>2}. {icon} [{crit.upper():<9}] wt={wt}  {text[:_W - 30]}", indent=2)
        if len(text) > _W - 30:
            _line(out, f"    ↳ …{text[_W - 30:]}", indent=2)
    _footer(out)
    logger.info("\n".join(out))

    return state