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
from typing import Any, Dict, List, Optional

from settings import GROQ_API_KEY, RFP_PDF_PATH
from rfp_rag import RfpRagConfig, get_rfp_rag_tool
from llm import chat_completion
from state import AgentState


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
    """
    Pull the broadest possible context from the RFP, then ask the LLM to
    return a structured JSON list of requirements.
    """
    # Use a broad query to get maximum coverage
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

    # Strip markdown fences properly (lstrip/rstrip work on char sets, not substrings)
    import re as _re
    fence_match = _re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, _re.DOTALL)
    if fence_match:
        raw = fence_match.group(1)
    else:
        # Try to find a raw JSON array
        bracket_match = _re.search(r"\[.*\]", raw, _re.DOTALL)
        if bracket_match:
            raw = bracket_match.group(0)

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            # Filter out entries without a proper requirement string
            valid = [
                item for item in data
                if isinstance(item, dict)
                and isinstance(item.get("requirement"), str)
                and len(item["requirement"].strip()) > 5
            ]
            return valid if valid else data
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: return as plain strings with weight=1
    # Filter out garbage lines (too short, just punctuation/braces)
    lines = [l.strip("- •\t").strip() for l in raw.splitlines() if l.strip()]
    lines = [l for l in lines if len(l) > 10 and not l.startswith(("{", "}", "[", "]", '"weight"', '"criticality"'))]
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

    # 1. Direct answer to the user query
    print("[RfpAgent] Answering user query from RFP …")
    state["rfp_answer"] = rfp_tool.run_query(query, api_key=api_key)
    print("[RfpAgent] ✓ RFP answer ready.")

    # 2. Structured requirements (always extracted for downstream agents)
    print("[RfpAgent] Extracting structured requirements …")
    state["rfp_requirements"] = _extract_requirements(rfp_tool, api_key)
    print(f"[RfpAgent] ✓ Extracted {len(state['rfp_requirements'])} requirements.")

    return state
