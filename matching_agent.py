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
  - state["fulfillment_score"]  — weighted percentage 0–100
  - state["fulfillment_report"] — human-readable summary

The agent is a plain function that receives and returns AgentState,
making it trivially usable as a LangGraph node.

SKU classification
------------------
Before any RAG queries are run, a single batched LLM call classifies every
requirement as "product_sku" (priceable physical item) or "compliance"
(vendor qualification, delivery term, document obligation, etc.).
This removes the need for keyword heuristics and costs just one API call
for the entire requirement list.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from settings import GROQ_API_KEY
from bu_agent import bu_query
from llm import chat_completion
from state import AgentState

_W = 72  # banner width


# ─── Print helpers ────────────────────────────────────────────────────────────

def _wrap(text: str, width: int = _W - 6) -> List[str]:
    lines = []
    for raw in (text or "").splitlines() or [""]:
        if not raw:
            lines.append("")
            continue
        for i in range(0, max(len(raw), 1), width):
            lines.append(raw[i : i + width])
    return lines


# ─── LLM-based SKU classifier (single batched call) ──────────────────────────

_CLASSIFY_PROMPT = """
You are a procurement assistant. Classify each requirement below as either
"product_sku" or "compliance".

"product_sku"  — A physical product, component, material, or supply item that
                 a vendor would quote with a unit price, SAP code, and HSN code.
                 Examples: switches, plates, sockets, fan regulators, USB chargers.

"compliance"   — A vendor qualification, eligibility criterion, logistical
                 condition, documentation obligation, or process requirement.
                 Examples: GST registration, delivery timelines, company profile,
                 past experience, packaging rules, supply capacity demonstration.

Return ONLY a JSON array. Each element:
  {{"requirement": "<original text>", "type": "product_sku" | "compliance"}}

No markdown. No explanation. Pure JSON only.

Requirements:
{requirements_json}
""".strip()


def _classify_requirements_llm(
    requirements: List[Dict[str, Any]],
    api_key: Optional[str],
) -> Dict[str, str]:
    """
    Single LLM call that returns {requirement_text -> "product_sku"|"compliance"}
    for every item in the list. Fallback: all treated as "product_sku" (safe).
    """
    reqs_json = json.dumps([{"requirement": r["requirement"]} for r in requirements], indent=2)
    raw = chat_completion(
        system="You classify procurement requirements. Return JSON only.",
        user=_CLASSIFY_PROMPT.format(requirements_json=reqs_json),
        temperature=0.0,
        max_tokens=1500,
        api_key=api_key,
    )

    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1)
    else:
        bracket = re.search(r"\[.*\]", raw, re.DOTALL)
        if bracket:
            raw = bracket.group(0)

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return {
                item["requirement"].strip(): item.get("type", "product_sku")
                for item in data
                if isinstance(item, dict) and "requirement" in item
            }
    except Exception:
        pass

    print("[MatchingAgent] ⚠ LLM classification parse failed — treating all as product_sku.")
    return {r["requirement"]: "product_sku" for r in requirements}


# ─── RAG-based fulfillment check ─────────────────────────────────────────────

def _assess_one(
    requirement: str,
    api_key: Optional[str],
) -> Tuple[bool, str]:
    """
    Ask the BU RAG whether the catalog can fulfil *requirement*.
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


# ─── Input normaliser ─────────────────────────────────────────────────────────

def _parse_requirements(raw: Any) -> List[Dict[str, Any]]:
    """Normalise RFP-agent output into [{requirement, weight}, …]."""
    if isinstance(raw, list):
        out = []
        for item in raw:
            if isinstance(item, dict) and "requirement" in item:
                out.append({
                    "requirement": str(item["requirement"]).strip(),
                    "weight": float(item.get("weight", 1.0)),
                })
        if out:
            return out

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
    LangGraph node.
    Reads rfp_requirements → writes matched_items, fulfillment_score,
    fulfillment_report.
    """
    api_key = os.environ.get("GROQ_API_KEY") or GROQ_API_KEY

    requirements = _parse_requirements(state.get("rfp_requirements", []))
    if not requirements:
        state["fulfillment_report"] = "No RFP requirements were available to match."
        state["fulfillment_score"] = 0.0
        state["matched_items"] = []
        return state

    # ── Step 1: classify all requirements in one LLM call ────────────────────
    print(f"[MatchingAgent] Classifying {len(requirements)} requirements (1 LLM call) …")
    type_map = _classify_requirements_llm(requirements, api_key)

    # ── Step 2: RAG fulfillment check for each requirement ────────────────────
    matched_items: List[Dict[str, Any]] = []
    total_weight   = 0.0
    matched_weight = 0.0
    total_reqs     = len(requirements)

    print(f"[MatchingAgent] Running RAG fulfillment check on {total_reqs} requirements …")
    for i, req_obj in enumerate(requirements, 1):
        req    = req_obj["requirement"]
        weight = req_obj["weight"]
        total_weight += weight
        req_type = type_map.get(req.strip(), "product_sku")

        print(f"[MatchingAgent] {i:>2}/{total_reqs}  {req[:65]}")
        is_match, evidence = _assess_one(req, api_key)
        icon = "✅ MATCH" if is_match else "❌ GAP  "
        sku_tag = "🏷 SKU" if req_type == "product_sku" else "📋 Compliance"
        print(f"[MatchingAgent]       → {icon}  {sku_tag}")

        matched_items.append({
            "requirement":    req,
            "weight":         weight,
            "matched":        is_match,
            "is_product_sku": req_type == "product_sku",
            "evidence":       evidence,
        })
        if is_match:
            matched_weight += weight

    # ── Step 3: build fulfillment report ─────────────────────────────────────
    score    = round((matched_weight / total_weight) * 100, 1) if total_weight else 0.0
    n_matched = sum(1 for m in matched_items if m["matched"])

    sku_matched  = [m for m in matched_items if m["matched"]     and m["is_product_sku"]]
    sku_gaps     = [m for m in matched_items if not m["matched"] and m["is_product_sku"]]
    comp_matched = [m for m in matched_items if m["matched"]     and not m["is_product_sku"]]
    comp_gaps    = [m for m in matched_items if not m["matched"] and not m["is_product_sku"]]

    def _bullet_lines(items: List[Dict]) -> List[str]:
        return [f"  • {m['requirement']}" for m in items] if items else ["  (none)"]

    report_lines = [
        f"Fulfillment score: {score}%  ({n_matched}/{len(matched_items)} requirements met)",
        "",
        "─── Product SKUs ─────────────────────────────────────────────────",
        "✅ Can supply:",
        *_bullet_lines(sku_matched),
        "",
        "❌ Cannot supply / gaps:",
        *_bullet_lines(sku_gaps),
        "",
        "─── Compliance & Eligibility ─────────────────────────────────────",
        "✅ Met:",
        *_bullet_lines(comp_matched),
        "",
        "❌ Not met / gaps:",
        *_bullet_lines(comp_gaps),
    ]

    state["matched_items"]      = matched_items
    state["fulfillment_score"]  = score
    state["fulfillment_report"] = "\n".join(report_lines)

    # ── Rich print output ─────────────────────────────────────────────────────
    print(f"\n╔{'═' * (_W - 2)}╗")
    print(f"║  {'MATCHING AGENT — OUTPUT':<{_W - 4}}║")
    print(f"╠{'═' * (_W - 2)}╣")
    print(f"║  Fulfillment score: {score}%  ({n_matched}/{len(matched_items)} met){'':<{_W - 48}}║")
    print(f"╠{'─' * (_W - 2)}╣")

    sections = [
        ("🏷  PRODUCT SKUs — CAN SUPPLY",    sku_matched,  True),
        ("🏷  PRODUCT SKUs — GAPS",           sku_gaps,     False),
        ("📋  COMPLIANCE — MET",              comp_matched, True),
        ("📋  COMPLIANCE — GAPS",             comp_gaps,    False),
    ]
    for heading, items, _ in sections:
        print(f"║  {heading:<{_W - 4}}║")
        if not items:
            print(f"║    {'(none)':<{_W - 6}}║")
        for m in items:
            text = m["requirement"]
            for chunk in _wrap(text):
                print(f"║    • {chunk:<{_W - 8}}║")
        print(f"╠{'─' * (_W - 2)}╣")

    print(f"╚{'═' * (_W - 2)}╝\n")

    return state