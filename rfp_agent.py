"""
agents/rfp_agent.py
-------------------
RFP Agent — LangGraph node

Responsibility
--------------
  1. Answer the user's query from the RFP document (state["rfp_answer"]).
  2. Extract a structured list of RFP requirements for downstream agents
     (state["rfp_requirements"]).

Both outputs are produced in a single run to avoid redundant PDF retrieval.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from settings import GROQ_API_KEY
from rfp_rag import RfpRagConfig, get_rfp_rag_tool
from llm import chat_completion
from state import AgentState

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


def _print_banner(title: str) -> None:
    print(f"\n╔{'═' * (_W - 2)}╗")
    print(f"║  {title:<{_W - 4}}║")
    print(f"╠{'═' * (_W - 2)}╣")


def _print_section(label: str) -> None:
    print(f"║  {label:<{_W - 4}}║")
    print(f"╠{'─' * (_W - 2)}╣")


def _print_line(text: str, indent: int = 2) -> None:
    pad = " " * indent
    print(f"║{pad}{text:<{_W - indent - 2}}║")


def _print_footer() -> None:
    print(f"╚{'═' * (_W - 2)}╝\n")


# ─── Requirement extraction ───────────────────────────────────────────────────

_REQ_EXTRACTION_PROMPT = """
You are an expert procurement analyst.

Extract RFP requirements from the context as a JSON array of objects.
Each object must have:
  - "requirement"  : short plain-English statement of the requirement
  - "criticality"  : "mandatory" | "high" | "medium" | "low"
  - "weight"       : numeric — mandatory=4, high=3, medium=2, low=1

Return ONLY valid JSON. No markdown, no explanation.

Context:
{context}
""".strip()


def _extract_requirements(
    rfp_tool,
    api_key: Optional[str],
) -> List[Dict[str, Any]]:
    context = rfp_tool.build_context(
        "requirements eligibility deliverables mandatory technical", top_k=5
    )
    raw = chat_completion(
        system="You extract structured data from RFP documents. Return JSON only.",
        user=_REQ_EXTRACTION_PROMPT.format(context=context),
        temperature=0.0,
        max_tokens=1500,
        api_key=api_key,
    )

    fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1)
    else:
        bracket_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if bracket_match:
            raw = bracket_match.group(0)

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            valid = [
                item for item in data
                if isinstance(item, dict)
                and isinstance(item.get("requirement"), str)
                and len(item["requirement"].strip()) > 5
            ]
            return valid if valid else data
    except (json.JSONDecodeError, TypeError):
        pass

    lines = [l.strip("- •\t").strip() for l in raw.splitlines() if l.strip()]
    lines = [
        l for l in lines
        if len(l) > 10
        and not l.startswith(("{", "}", "[", "]", '"weight"', '"criticality"'))
    ]
    return [{"requirement": l, "criticality": "medium", "weight": 2.0} for l in lines]


# ─── LangGraph node ───────────────────────────────────────────────────────────

def rfp_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node.
    Reads user_query, writes rfp_answer and rfp_requirements.
    """
    api_key = os.environ.get("GROQ_API_KEY") or GROQ_API_KEY
    query = state.get("user_query", "")

    rfp_tool = get_rfp_rag_tool(RfpRagConfig())

    print("[RfpAgent] Answering user query from RFP …")
    rfp_answer = rfp_tool.run_query(query, api_key=api_key)
    state["rfp_answer"] = rfp_answer

    print("[RfpAgent] Extracting structured requirements …")
    requirements = _extract_requirements(rfp_tool, api_key)
    state["rfp_requirements"] = requirements

    # ── Rich print output ──────────────────────────────────────────────────────
    crit_icon = {"mandatory": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}

    _print_banner("RFP AGENT — OUTPUT")
    _print_line(f"Query: {query[:_W - 10]}")
    _print_section(f"ANSWER")
    for line in _wrap(rfp_answer):
        _print_line(line)
    print(f"╠{'═' * (_W - 2)}╣")
    _print_section(f"EXTRACTED REQUIREMENTS  ({len(requirements)} total)")
    for i, req in enumerate(requirements, 1):
        crit = req.get("criticality", "medium")
        icon = crit_icon.get(crit, "•")
        wt   = req.get("weight", 1.0)
        text = req.get("requirement", "")
        _print_line(f"{i:>2}. {icon} [{crit.upper():<9}] wt={wt}  {text[:_W - 30]}", indent=2)
        if len(text) > _W - 30:
            _print_line(f"    ↳ …{text[_W - 30:]}", indent=2)
    _print_footer()

    return state