"""
agents/matching_agent.py
------------------------
Matching Agent — LangGraph node

Responsibility
--------------
Given the RFP requirements (state["rfp_requirements"]), find the catalogue
product that fulfils each one:

  1. Classify  — one batched LLM call tags each requirement "product_sku"
                 (a physical item we would quote) or "compliance"
                 (eligibility, documents, delivery terms, …).
  2. Atomise   — one batched LLM call splits compound product requirements
                 ("16A and 25A switches") into single-product items.
  3. Match     — per item, IN PARALLEL: catalogue search with relevance
                 scores, then one LLM call picks a candidate by id
                 (see product_matcher.py). Unrelated items become gaps
                 without an LLM call.

Outputs
  - state["matched_items"]      one dict per item (see _product_item())
  - state["fulfillment_score"]  weighted % of product items with a product
  - state["fulfillment_report"] human-readable summary
  - state["match_summary"]      counts for the UI

Compliance requirements cannot be checked against a product catalogue, so
they are listed with status "needs_review" for a human to confirm, and are
not part of the automatic score.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from catalogue import ProductRecord
from llm_json import LLMOutputError, chat_json
from product_matcher import atomise, confidence_band, find_candidates, pick_product
from settings import LLM_MAX_CONCURRENCY, OPENROUTER_API_KEY
from state import AgentState

logger = logging.getLogger(__name__)

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


# ─── Step 1: classify (single batched call) ───────────────────────────────────

_CLASSIFY_PROMPT = """
You are a procurement assistant. Classify each requirement below as either
"product_sku" or "compliance".

"product_sku"  — A physical product, component, material, or supply item that
                 a vendor would quote with a unit price, SAP code, and HSN code.
                 Examples: switches, plates, sockets, fan regulators, USB chargers.

"compliance"   — A vendor qualification, eligibility criterion, logistical
                 condition, documentation obligation, or process requirement.
                 Examples: GST registration, delivery timelines, company profile,
                 past experience, packaging rules, supply capacity demonstration,
                 "submit a price list".

Return ONLY a JSON array with one entry per input, in the same order:
  [{{"index": <input index>, "type": "product_sku" | "compliance"}}, ...]

Requirements:
{requirements_json}
""".strip()


class _Classified(BaseModel):
    index: int
    type: Literal["product_sku", "compliance"]


def _classify_requirements_llm(
    requirements: List[Dict[str, Any]],
    api_key: Optional[str],
) -> List[str]:
    """
    One LLM call → "product_sku" | "compliance" per requirement, in order.
    Fallback: every requirement is treated as a product (it will simply be
    reported as a gap if the catalogue has nothing similar).
    """
    payload = json.dumps(
        [{"index": i, "requirement": r["requirement"]} for i, r in enumerate(requirements)],
        indent=1,
    )
    try:
        result = chat_json(
            system="You classify procurement requirements. Return JSON only.",
            user=_CLASSIFY_PROMPT.format(requirements_json=payload),
            schema=List[_Classified],
            temperature=0.0,
            max_tokens=1500,
            api_key=api_key,
        )
        by_index = {c.index: c.type for c in result}
    except LLMOutputError:
        logger.warning("Classification failed; treating all requirements as products.")
        by_index = {}
    return [by_index.get(i, "product_sku") for i in range(len(requirements))]


# ─── Input normaliser ─────────────────────────────────────────────────────────

def _parse_requirements(raw: Any) -> List[Dict[str, Any]]:
    """Normalise RFP-agent output into [{id, requirement, weight, criticality, qty?}]."""
    if isinstance(raw, str):
        try:
            return _parse_requirements(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            lines = [l.strip("- •\t").strip() for l in raw.splitlines() if l.strip()]
            raw = [{"requirement": l} for l in lines]

    out: List[Dict[str, Any]] = []
    for item in raw if isinstance(raw, list) else []:
        if not (isinstance(item, dict) and str(item.get("requirement", "")).strip()):
            continue
        parsed = {
            "id": str(item.get("id") or f"R{len(out) + 1}"),
            "requirement": str(item["requirement"]).strip(),
            "weight": float(item.get("weight", 1.0)),
            "criticality": item.get("criticality", "medium"),
        }
        qty = item.get("quantity", item.get("qty"))
        if qty is not None:
            parsed["qty"] = qty
        out.append(parsed)
    return out


# ─── Step 3: match one product item ───────────────────────────────────────────

def product_brief(product: ProductRecord, score: Optional[float] = None) -> Dict[str, Any]:
    """Short product description used for the chosen product and alternatives."""
    return {
        "product_id":      product.id,
        "product_name":    product.item_name,
        "product_range":   product.range_name,
        "product_variant": product.variant,
        "sap_code":        product.sap_code,
        "unit_price":      product.unit_price,   # catalogue list price, for display
        "match_score":     round(score * 100) if score is not None else None,
    }


def apply_product(
    item: Dict[str, Any],
    product: ProductRecord,
    *,
    score: Optional[float],
    confidence: str,
    rationale: str,
) -> Dict[str, Any]:
    """
    Return a copy of *item* with *product* attached: catalogue fields, demo
    stock and match metadata. Used by matching and by the human review step
    ("Change Product"), so both produce identical item shapes.
    """
    qty = item.get("qty")
    try:
        qty = int(qty) if qty is not None else None
    except (TypeError, ValueError):
        qty = None

    return {
        **item,
        **product_brief(product, score),
        "matched":           True,
        "confidence":        confidence,
        "evidence":          rationale,
        "rationale":         rationale,
        "hsn_code":          product.hsn_code,
        "packing":           product.packing,
        # Demo stock (synthetic): see catalogue.demo_stock_for()
        "stock_on_hand":     product.on_hand,
        "monthly_available": product.monthly_capacity,
        "lead_time_days":    product.lead_time_days,
        "stock_is_demo":     product.stock_is_demo,
        "stock_sufficient":  (product.on_hand >= qty) if (qty and product.on_hand is not None) else None,
    }


def _product_item(item: Dict[str, Any], api_key: Optional[str]) -> Dict[str, Any]:
    """Search + pick for one single-product item. Safe to run in parallel."""
    text = item["requirement"]
    query = text[7:].strip() if text.lower().startswith("supply ") else text

    candidates = find_candidates(query)
    chosen, reason = pick_product(query, candidates, api_key)

    base: Dict[str, Any] = {**item, "type": "product", "is_product_sku": True}
    if chosen is None:
        return {
            **base,
            "matched":      False,
            "status":       "gap",
            "match_score":  0,
            "confidence":   None,
            "evidence":     reason,
            "rationale":    reason,
            "alternatives": [product_brief(c.product, c.score) for c in candidates[:3]],
        }

    confidence = confidence_band(chosen.score)
    result = apply_product(base, chosen.product, score=chosen.score,
                           confidence=confidence, rationale=reason)
    result["status"] = "matched" if confidence == "high" else "needs_review"
    result["alternatives"] = [
        product_brief(c.product, c.score) for c in candidates if c is not chosen
    ][:3]
    return result


def _compliance_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **item,
        "type":           "compliance",
        "is_product_sku": False,
        "matched":        False,
        "status":         "needs_review",
        "evidence":       "Compliance requirement: confirm against company documents.",
        "rationale":      "Not a catalogue product; needs a human to confirm.",
    }


# ─── LangGraph node ───────────────────────────────────────────────────────────

def matching_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node.
    Reads rfp_requirements → writes matched_items, fulfillment_score,
    fulfillment_report, match_summary.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY") or OPENROUTER_API_KEY

    requirements = _parse_requirements(state.get("rfp_requirements", []))
    if not requirements:
        state["fulfillment_report"] = "No RFP requirements were available to match."
        state["fulfillment_score"] = 0.0
        state["matched_items"] = []
        state["match_summary"] = match_summary([])
        return state

    # ── Step 1: classify ──────────────────────────────────────────────────────
    print(f"[MatchingAgent] Classifying {len(requirements)} requirements (1 LLM call) …")
    types = _classify_requirements_llm(requirements, api_key)
    product_reqs = [r for r, t in zip(requirements, types) if t == "product_sku"]

    # ── Step 2: atomise product requirements ──────────────────────────────────
    print(f"[MatchingAgent] Splitting {len(product_reqs)} product requirements (1 LLM call) …")
    split = atomise([r["requirement"] for r in product_reqs], api_key)
    product_items: Dict[str, List[Dict[str, Any]]] = {}
    for req, parts in zip(product_reqs, split):
        if len(parts) == 1:
            product_items[req["id"]] = [{**req, "requirement": parts[0]}]
            continue
        product_items[req["id"]] = [
            {
                "id":          f"{req['id']}.{k}",
                "parent_id":   req["id"],
                "requirement": part,
                "weight":      round(req["weight"] / len(parts), 4),
                "criticality": req["criticality"],
                # A quantity stated for the whole group is ambiguous per item.
            }
            for k, part in enumerate(parts, 1)
        ]

    # ── Step 3: match every product item in parallel ─────────────────────────
    flat = [item for items in product_items.values() for item in items]
    print(f"[MatchingAgent] Matching {len(flat)} product items "
          f"({LLM_MAX_CONCURRENCY} in parallel) …")
    with ThreadPoolExecutor(max_workers=max(1, LLM_MAX_CONCURRENCY)) as pool:
        matched_flat = list(pool.map(lambda it: _product_item(it, api_key), flat))
    matched_by_id = {m["id"]: m for m in matched_flat}

    # Keep the RFP's order: each requirement, then its split items.
    matched_items: List[Dict[str, Any]] = []
    for req, kind in zip(requirements, types):
        if kind == "compliance":
            matched_items.append(_compliance_item(req))
        else:
            matched_items.extend(matched_by_id[i["id"]] for i in product_items[req["id"]])

    summary = match_summary(matched_items)
    state["matched_items"]      = matched_items
    state["fulfillment_score"]  = summary["fulfillment_score"]
    state["fulfillment_report"] = fulfillment_report(matched_items, summary)
    state["match_summary"]      = summary

    print(f"\n╔{'═' * (_W - 2)}╗")
    print(f"║  {'MATCHING AGENT — OUTPUT':<{_W - 4}}║")
    print(f"╚{'═' * (_W - 2)}╝")
    print(state["fulfillment_report"])
    return state


# ─── Summary and report ───────────────────────────────────────────────────────
#
# Product item status:    matched (high confidence) | needs_review | gap
#                         | approved / rejected (set by a human reviewer)
# Compliance item status: needs_review | met / not_met (set by a reviewer)

def match_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    products = [m for m in items if m.get("type") == "product"]
    compliance = [m for m in items if m.get("type") == "compliance"]
    total_w = sum(m["weight"] for m in products)
    found_w = sum(m["weight"] for m in products if m.get("matched"))

    def count(group: List[Dict[str, Any]], *statuses: str) -> int:
        return sum(1 for m in group if m.get("status") in statuses)

    return {
        "fulfillment_score":  round(found_w / total_w * 100, 1) if total_w else 0.0,
        "product_items":      len(products),
        "matched":            count(products, "matched", "approved"),
        "approved":           count(products, "approved"),
        "needs_review":       count(products, "needs_review"),
        "gaps":               count(products, "gap"),
        "rejected":           count(products, "rejected"),
        "compliance_pending": count(compliance, "needs_review"),
        "compliance_met":     count(compliance, "met"),
        "compliance_not_met": count(compliance, "not_met"),
    }


def fulfillment_report(items: List[Dict[str, Any]], summary: Dict[str, Any]) -> str:
    def product_line(m: Dict[str, Any]) -> str:
        if not m.get("product_id"):
            return f"  • {m['requirement']}  →  no catalogue product ({m.get('rationale', '')})"
        score = f"{m['match_score']}% " if m.get("match_score") is not None else ""
        return (f"  • {m['requirement']}  →  {m['product_name']} ({m['product_range']}), "
                f"SAP {m['sap_code']}, {score}[{m.get('confidence')}]")

    products = [m for m in items if m.get("type") == "product"]
    groups = [
        ("✅ Matched", [m for m in products if m.get("status") in ("matched", "approved")]),
        ("🔍 Needs review", [m for m in products if m.get("status") == "needs_review"]),
        ("❌ Gaps (not in catalogue)", [m for m in products if m.get("status") == "gap"]),
        ("⛔ Rejected by reviewer", [m for m in products if m.get("status") == "rejected"]),
    ]
    lines = [
        f"Fulfillment score: {summary['fulfillment_score']}%  "
        f"({summary['matched'] + summary['needs_review']}/{summary['product_items']} "
        f"product items have a catalogue product)",
        "",
        "─── Product SKUs ─────────────────────────────────────────────────",
    ]
    for title, members in groups:
        if members or title != "⛔ Rejected by reviewer":
            lines.append(f"{title}:")
            lines.extend([product_line(m) for m in members] or ["  (none)"])
            lines.append("")

    marks = {"met": "✅", "not_met": "❌", "needs_review": "☐"}
    lines.append("─── Compliance & Eligibility ─────────────────────────────────────")
    compliance = [m for m in items if m.get("type") == "compliance"]
    lines.extend(
        [f"  {marks.get(m.get('status'), '☐')} {m['requirement']}" for m in compliance]
        or ["  (none)"]
    )
    if summary.get("compliance_pending"):
        lines.append("  ☐ = not yet confirmed by a reviewer")
    lines.append("")
    lines.append("Stock figures are demo data.")
    return "\n".join(lines)
