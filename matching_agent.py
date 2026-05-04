"""
agents/matching_agent.py
------------------------
Matching Agent — LangGraph node

Responsibility
--------------
Given a list of RFP requirements (from state["rfp_requirements"]), the
Matching Agent checks each one against the business-unit catalog (via
BuRagTool) and produces:

  - state["matched_items"]      — per-requirement match detail
  - state["fulfillment_score"]  — weighted percentage 0-100
  - state["fulfillment_report"] — human-readable summary

The agent is a plain function that receives and returns AgentState,
making it trivially usable as a LangGraph node.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from settings import GROQ_API_KEY
from bu_agent import bu_query
from llm import chat_completion
from state import AgentState


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _assess_one(
    requirement: str,
    api_key: Optional[str],
) -> Tuple[bool, str]:
    """
    Ask the BU RAG whether it can fulfil *requirement*.
    Returns (is_match, evidence_text).
    """
    prompt = (
        "Based only on the business catalog context, "
        "can the business fulfil this RFP requirement?\n"
        "Reply with 'YES' or 'NO' on the first line, "
        "then one short evidence sentence.\n\n"
        f"Requirement: {requirement}"
    )
    answer = bu_query(prompt, temperature=0.0, api_key=api_key)
    first = answer.strip().splitlines()[0].strip().lower()
    return first.startswith("yes"), answer.strip()


def _parse_requirements(raw: Any) -> List[Dict[str, Any]]:
    """
    Normalise whatever the RFP agent produced into
    [{"requirement": str, "weight": float}, …]
    """
    if isinstance(raw, list):
        out = []
        for item in raw:
            if isinstance(item, dict) and "requirement" in item:
                out.append(
                    {
                        "requirement": str(item["requirement"]).strip(),
                        "weight": float(item.get("weight", 1.0)),
                    }
                )
        if out:
            return out

    # Fallback: treat as plain text lines
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return _parse_requirements(parsed)
        except (json.JSONDecodeError, TypeError):
            pass
        lines = [l.strip("- •\t").strip() for l in raw.splitlines() if l.strip()]
        return [{"requirement": l, "weight": 1.0} for l in lines]

    return []


# ─── LangGraph node ───────────────────────────────────────────────────────────

def matching_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node.  Reads rfp_requirements, writes matched_items,
    fulfillment_score, and fulfillment_report back into state.
    """
    api_key = os.environ.get("GROQ_API_KEY") or GROQ_API_KEY

    requirements = _parse_requirements(state.get("rfp_requirements", []))
    if not requirements:
        state["fulfillment_report"] = "No RFP requirements were available to match."
        state["fulfillment_score"] = 0.0
        state["matched_items"] = []
        return state

    matched_items: List[Dict[str, Any]] = []
    total_weight = 0.0
    matched_weight = 0.0
    total_reqs = len(requirements)

    print(f"[MatchingAgent] Checking {total_reqs} requirements against catalog …")
    for i, req_obj in enumerate(requirements, 1):
        req = req_obj["requirement"]
        weight = req_obj["weight"]
        total_weight += weight

        print(f"[MatchingAgent] {i}/{total_reqs} Checking: {req[:60]}…")
        is_match, evidence = _assess_one(req, api_key)
        verdict = "✅ MATCH" if is_match else "❌ GAP"
        print(f"[MatchingAgent]   → {verdict}")
        evidence_preview = evidence.splitlines()[1].strip() if len(evidence.splitlines()) > 1 else ""
        if evidence_preview:
            print(f"[MatchingAgent]     Evidence: {evidence_preview[:120]}")

        matched_items.append(
            {
                "requirement": req,
                "weight": weight,
                "matched": is_match,
                "evidence": evidence,
            }
        )
        if is_match:
            matched_weight += weight

    score = round((matched_weight / total_weight) * 100, 1) if total_weight else 0.0
    n_matched = sum(1 for m in matched_items if m["matched"])

    # Build human-readable report
    strengths = [m for m in matched_items if m["matched"]]
    gaps = [m for m in matched_items if not m["matched"]]

    strength_lines = [f"  • {m['requirement']}" for m in strengths] or ["  (none)"]
    gap_lines = [f"  • {m['requirement']}" for m in gaps] or ["  (none)"]

    report_lines = [
        f"Fulfillment score: {score}% ({n_matched}/{len(matched_items)} requirements met)",
        "",
        "✅ Strengths:",
        *strength_lines,
        "",
        "❌ Gaps:",
        *gap_lines,
    ]

    state["matched_items"] = matched_items
    state["fulfillment_score"] = score
    state["fulfillment_report"] = "\n".join(report_lines)

    print(f"\n[MatchingAgent] ══ FULFILLMENT REPORT ══════════════════════════")
    print(state["fulfillment_report"])
    print(f"[MatchingAgent] ════════════════════════════════════════════════\n")

    return state