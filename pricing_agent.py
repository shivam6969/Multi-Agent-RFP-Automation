"""
agents/pricing_agent.py
-----------------------
Pricing Agent — LangGraph node

Responsibility
--------------
For every requirement that was successfully *matched* by the Matching
Agent, the Pricing Agent:

  1. Queries the BU RAG to get unit price, packing, and monthly availability.
  2. Applies the margin, volume-discount, and GST rules from config/settings.py.
  3. Produces a line-item quote and a human-readable pricing report.

Outputs written to state
  - state["quoted_items"]    — list of per-product pricing dicts
  - state["pricing_report"]  — formatted quote table

The agent is a plain function → trivially usable as a LangGraph node.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from settings import GROQ_API_KEY, PRICING_RULES, PricingRules
from bu_agent import bu_query
from llm import chat_completion
from state import AgentState


# ─── Price extraction helpers ─────────────────────────────────────────────────

def _extract_price(text: str) -> Optional[float]:
    """
    Pull the first number that looks like a rupee price from *text*.
    Handles formats: "60/-", "Rs.60", "INR 1,200", "₹ 450.00"
    """
    patterns = [
        r"(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d+)?)",   # Rs./INR/₹ prefix
        r"([\d,]+(?:\.\d+)?)\s*/\-",                 # 60/- suffix
        r"price[:\s]+([\d,]+(?:\.\d+)?)",             # "Price: 60"
    ]
    lowered = text.lower()
    for pat in patterns:
        m = re.search(pat, lowered)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                continue
    # Last resort: first standalone number ≥ 1
    m = re.search(r"\b(\d{2,}(?:\.\d+)?)\b", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _extract_qty(text: str) -> int:
    """
    Try to find a quantity mentioned in the RFP or user query.
    Falls back to 1.
    """
    m = re.search(r"\b(\d+)\s*(?:units?|pcs?|pieces?|nos?\.?|qty)\b", text, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d{2,5})\b", text)
    if m:
        v = int(m.group(1))
        if 1 <= v <= 99999:
            return v
    return 1


def _volume_discount(qty: int, rules: PricingRules) -> float:
    """Return the applicable discount % for a given quantity."""
    for tier in rules.volume_discount_tiers:
        if tier["min_qty"] <= qty <= tier["max_qty"]:
            return tier["discount_pct"]
    return 0.0


def _clamp_margin(margin_pct: float, rules: PricingRules) -> float:
    """Keep requested margin inside configured bounds."""
    return max(rules.min_margin_pct, min(rules.max_margin_pct, margin_pct))


def _build_line_item(
    requirement: str,
    qty: int,
    unit_price: float,
    margin_pct: float,
    rules: PricingRules,
    raw_price_info: str,
) -> Dict[str, Any]:
    """Create one deterministic pricing line from resolved unit price."""
    rounded = rules.rounding_digits
    discount_pct = _volume_discount(qty, rules)
    selling_price = round(unit_price * (1 + margin_pct / 100), rounded)
    discounted_price = round(selling_price * (1 - discount_pct / 100), rounded)
    gst_amount = round(discounted_price * rules.gst_rate_pct / 100, rounded)
    final_unit_price = round(discounted_price + gst_amount, rounded)
    line_total = round(final_unit_price * qty, rounded)

    return {
        "requirement": requirement,
        "qty": qty,
        "currency": rules.currency,
        "base_unit_price": round(unit_price, rounded),
        "margin_pct": round(margin_pct, rounded),
        "volume_discount_pct": round(discount_pct, rounded),
        "net_unit_price": discounted_price,
        "gst_pct": rules.gst_rate_pct,
        "gst_amount_per_unit": gst_amount,
        "final_unit_price": final_unit_price,
        "line_total": line_total,
        "raw_price_info": raw_price_info,
    }


def _build_pricing_summary(
    line_items: List[Dict[str, Any]],
    rules: PricingRules,
    margin_pct: float,
) -> Dict[str, Any]:
    """Build aggregate totals used by frontend and quotation generator."""
    rounded = rules.rounding_digits
    subtotal_ex_gst = round(sum(i["net_unit_price"] * i["qty"] for i in line_items), rounded)
    total_gst = round(sum(i["gst_amount_per_unit"] * i["qty"] for i in line_items), rounded)
    grand_total = round(sum(i["line_total"] for i in line_items), rounded)
    base_subtotal = round(sum(i["base_unit_price"] * i["qty"] for i in line_items), rounded)
    margin_value = round(subtotal_ex_gst - base_subtotal, rounded)

    return {
        "currency": rules.currency,
        "margin_pct": round(margin_pct, rounded),
        "subtotal_base": base_subtotal,
        "subtotal_ex_gst": subtotal_ex_gst,
        "total_gst": total_gst,
        "grand_total": grand_total,
        "margin_value": margin_value,
        "items_count": len(line_items),
    }


def _build_quote_payload(
    state: AgentState,
    line_items: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Payload contract for downstream quotation/pdf generation."""
    return {
        "quote_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "pricing_agent_v1",
            "user_query": state.get("user_query", ""),
        },
        "pricing_summary": summary,
        "line_items": line_items,
    }


# ─── LLM-based requirement filter ────────────────────────────────────────────

_PRICEABLE_FILTER_PROMPT = """
You are a procurement assistant. Given a list of RFP requirements, classify
each one as either "priceable" or "non-priceable".

"priceable"     — The requirement refers to a physical product, component,
                  material, or supply item that would appear as a line item
                  in a vendor quote (e.g. switches, plates, sockets, cables).

"non-priceable" — The requirement is a vendor qualification, compliance
                  criterion, logistical condition, process obligation, or
                  document submission (e.g. GST registration, delivery
                  timelines, company profile, past experience, packaging rules).

Return ONLY a JSON array. Each element must be:
  {{"requirement": "<original text>", "priceable": true | false}}

No markdown, no explanation. Pure JSON only.

Requirements:
{requirements_json}
""".strip()


def _filter_priceable_requirements(
    matched_items: List[Dict[str, Any]],
    api_key: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Use the LLM to split matched requirements into priceable vs non-priceable.
    Returns (priceable_items, skipped_items).
    """
    import json as _json

    reqs = [{"requirement": m["requirement"]} for m in matched_items]
    raw = chat_completion(
        system="You classify procurement requirements. Return JSON only.",
        user=_PRICEABLE_FILTER_PROMPT.format(
            requirements_json=_json.dumps(reqs, indent=2)
        ),
        temperature=0.0,
        max_tokens=1000,
        api_key=api_key,
    )

    # Parse LLM response
    import re as _re
    fence = _re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, _re.DOTALL)
    if fence:
        raw = fence.group(1)
    else:
        bracket = _re.search(r"\[.*\]", raw, _re.DOTALL)
        if bracket:
            raw = bracket.group(0)

    try:
        classifications = _json.loads(raw)
        priceable_set = {
            item["requirement"].strip().lower()
            for item in classifications
            if isinstance(item, dict) and item.get("priceable") is True
        }
    except Exception:
        # Fallback: treat all as priceable if LLM parse fails
        print("[PricingAgent] ⚠ LLM filter parse failed — pricing all matched items.")
        return matched_items, []

    priceable = [m for m in matched_items if m["requirement"].strip().lower() in priceable_set]
    skipped   = [m for m in matched_items if m["requirement"].strip().lower() not in priceable_set]
    return priceable, skipped


# ─── Per-item pricing ─────────────────────────────────────────────────────────

def _price_one_item(
    requirement: str,
    rules: PricingRules,
    user_query: str,
    api_key: Optional[str],
) -> Dict[str, Any]:
    """
    Query BU RAG for price data on *requirement*, apply pricing rules,
    return a pricing dict.
    """
    price_query = (
        f"What is the unit price, packing, SAP code, and monthly availability "
        f"for products matching this requirement: {requirement}"
    )
    price_answer = bu_query(price_query, temperature=0.0, api_key=api_key)

    unit_price = _extract_price(price_answer)
    qty = _extract_qty(user_query + " " + requirement)

    if unit_price is None:
        return {
            "requirement": requirement,
            "raw_price_info": price_answer,
            "unit_price_inr": None,
            "qty": qty,
            "note": "Price not found in catalog",
        }

    margin_pct = _clamp_margin(rules.base_margin_pct, rules)
    line_item = _build_line_item(
        requirement=requirement,
        qty=qty,
        unit_price=unit_price,
        margin_pct=margin_pct,
        rules=rules,
        raw_price_info=price_answer,
    )
    return {
        "requirement": requirement,
        "raw_price_info": price_answer,
        "catalog_cost_inr": line_item["base_unit_price"],
        "selling_price_inr": round(
            line_item["base_unit_price"] * (1 + line_item["margin_pct"] / 100),
            rules.rounding_digits,
        ),
        "volume_discount_pct": line_item["volume_discount_pct"],
        "net_price_inr": line_item["net_unit_price"],
        "gst_pct": line_item["gst_pct"],
        "gst_amount_inr": line_item["gst_amount_per_unit"],
        "price_with_gst_inr": line_item["final_unit_price"],
        "qty": line_item["qty"],
        "line_total_inr": line_item["line_total"],
        "line_item": line_item,
    }


def generate_initial_pricing(
    matched_items: List[Dict[str, Any]],
    rules: PricingRules,
    user_query: str,
    api_key: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate quoted_items and frontend line_items from matched requirements."""
    quoted_items: List[Dict[str, Any]] = []
    line_items: List[Dict[str, Any]] = []
    for item in matched_items:
        priced = _price_one_item(
            requirement=item["requirement"],
            rules=rules,
            user_query=user_query,
            api_key=api_key,
        )
        quoted_items.append(priced)
        if priced.get("line_item"):
            line_items.append(priced["line_item"])
    return quoted_items, line_items


def reprice_with_margin(
    line_items: List[Dict[str, Any]],
    new_margin_pct: float,
    rules: PricingRules,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Recompute pricing using a user-selected margin.
    This is used by frontend interactions and avoids rerunning retrieval.
    """
    margin_pct = _clamp_margin(new_margin_pct, rules)
    updated_items: List[Dict[str, Any]] = []
    for item in line_items:
        repriced = _build_line_item(
            requirement=item["requirement"],
            qty=int(item["qty"]),
            unit_price=float(item["base_unit_price"]),
            margin_pct=margin_pct,
            rules=rules,
            raw_price_info=item.get("raw_price_info", ""),
        )
        updated_items.append(repriced)
    summary = _build_pricing_summary(updated_items, rules, margin_pct)
    return updated_items, summary


# ─── Report formatter ─────────────────────────────────────────────────────────

def _format_report(quoted_items: List[Dict[str, Any]], rules: PricingRules) -> str:
    sep = "-" * 70
    lines = [
        "╔══ PRICING REPORT ══════════════════════════════════════════════════╗",
        f"  Margin: {rules.base_margin_pct}% | GST: {rules.gst_rate_pct}%",
        sep,
    ]

    grand_total = 0.0
    for item in quoted_items:
        req = item["requirement"][:55]
        # Success path sets catalog_cost_inr; failure path does not (legacy check
        # used unit_price_inr, which is never set on success — so totals were always 0).
        if item.get("catalog_cost_inr") is None:
            lines.append(f"  {req}")
            lines.append(f"    ⚠ {item.get('note', 'Price unavailable')}")
        else:
            lines.append(f"  {req}")
            lines.append(
                f"    Catalog cost : ₹ {item['catalog_cost_inr']:>8.2f}"
            )
            lines.append(
                f"    Selling price: ₹ {item['selling_price_inr']:>8.2f}  "
                f"(margin {rules.base_margin_pct}%)"
            )
            if item["volume_discount_pct"]:
                lines.append(
                    f"    Vol. discount: {item['volume_discount_pct']}% "
                    f"→ ₹ {item['net_price_inr']:>8.2f}"
                )
            lines.append(
                f"    GST ({item['gst_pct']}%)   : ₹ {item['gst_amount_inr']:>8.2f}"
            )
            lines.append(
                f"    Price/unit   : ₹ {item['price_with_gst_inr']:>8.2f}"
            )
            lines.append(
                f"    Qty          : {item['qty']:>4d} units"
            )
            lines.append(
                f"    Line total   : ₹ {item['line_total_inr']:>10.2f}"
            )
            grand_total += item["line_total_inr"]
        lines.append(sep)

    lines.append(f"  GRAND TOTAL (incl. GST): ₹ {grand_total:,.2f}")
    lines.append(
        "╚════════════════════════════════════════════════════════════════════╝"
    )
    return "\n".join(lines)


# ─── LangGraph node ───────────────────────────────────────────────────────────

def pricing_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node.
    Reads matched_items and user_query, writes quoted_items and pricing_report.
    """
    api_key = os.environ.get("GROQ_API_KEY") or GROQ_API_KEY

    matched = [
        m for m in state.get("matched_items", []) if m.get("matched")
    ]
    if not matched:
        state["pricing_report"] = (
            "No matched products were found — pricing cannot be generated."
        )
        state["quoted_items"] = []
        state["line_items"] = []
        state["pricing_summary"] = {
            "currency": PRICING_RULES.currency,
            "margin_pct": PRICING_RULES.base_margin_pct,
            "subtotal_base": 0.0,
            "subtotal_ex_gst": 0.0,
            "total_gst": 0.0,
            "grand_total": 0.0,
            "margin_value": 0.0,
            "items_count": 0,
        }
        state["quote_payload"] = _build_quote_payload(
            state=state,
            line_items=[],
            summary=state["pricing_summary"],
        )
        return state

    user_query = state.get("user_query", "")

    # ── LLM-based filter: keep only product/supply requirements ──────────────
    print(f"[PricingAgent] Running LLM filter on {len(matched)} matched requirements …")
    priceable, skipped = _filter_priceable_requirements(matched, api_key)

    print(f"\n[PricingAgent] ══ REQUIREMENT CLASSIFICATION ══════════════════════")
    print(f"[PricingAgent]   Total matched  : {len(matched)}")
    print(f"[PricingAgent]   ✅ Priceable   : {len(priceable)}")
    for item in priceable:
        print(f"[PricingAgent]     • {item['requirement'][:80]}")
    print(f"[PricingAgent]   ⏭  Non-priceable (skipped): {len(skipped)}")
    for item in skipped:
        print(f"[PricingAgent]     • {item['requirement'][:80]}")
    print(f"[PricingAgent] ════════════════════════════════════════════════════\n")

    if not priceable:
        state["pricing_report"] = (
            "No priceable product requirements found among matched items "
            "(all matched items were vendor qualification or compliance criteria)."
        )
        state["quoted_items"] = []
        state["line_items"] = []
        state["pricing_summary"] = {
            "currency": PRICING_RULES.currency,
            "margin_pct": PRICING_RULES.base_margin_pct,
            "subtotal_base": 0.0,
            "subtotal_ex_gst": 0.0,
            "total_gst": 0.0,
            "grand_total": 0.0,
            "margin_value": 0.0,
            "items_count": 0,
        }
        state["quote_payload"] = _build_quote_payload(
            state=state, line_items=[], summary=state["pricing_summary"]
        )
        return state

    total_items = len(priceable)
    print(f"[PricingAgent] Pricing {total_items} priceable items …")
    quoted_items, line_items = generate_initial_pricing(
        matched_items=priceable,
        rules=PRICING_RULES,
        user_query=user_query,
        api_key=api_key,
    )

    print(f"\n[PricingAgent] ══ PRICING RESULTS ═════════════════════════════════")
    for i, priced in enumerate(quoted_items, 1):
        req_short = priced['requirement'][:55]
        if priced.get('catalog_cost_inr') is not None:
            print(f"[PricingAgent]   {i}. {req_short}")
            print(f"[PricingAgent]      Catalog cost : ₹{priced['catalog_cost_inr']:.2f}")
            print(f"[PricingAgent]      Selling price: ₹{priced['selling_price_inr']:.2f}  (margin {PRICING_RULES.base_margin_pct}%)")
            if priced['volume_discount_pct']:
                print(f"[PricingAgent]      Vol. discount: {priced['volume_discount_pct']}% → ₹{priced['net_price_inr']:.2f}")
            print(f"[PricingAgent]      GST ({priced['gst_pct']}%)     : ₹{priced['gst_amount_inr']:.2f}")
            print(f"[PricingAgent]      Price/unit   : ₹{priced['price_with_gst_inr']:.2f}")
            print(f"[PricingAgent]      Qty × Line total: {priced['qty']} × ₹{priced['price_with_gst_inr']:.2f} = ₹{priced['line_total_inr']:.2f}")
        else:
            print(f"[PricingAgent]   {i}. {req_short} — ⚠ price not found")
    print(f"[PricingAgent] ════════════════════════════════════════════════════\n")

    summary = _build_pricing_summary(
        line_items=line_items,
        rules=PRICING_RULES,
        margin_pct=PRICING_RULES.base_margin_pct,
    )
    state["quoted_items"] = quoted_items
    state["line_items"] = line_items
    state["pricing_summary"] = summary
    state["quote_payload"] = _build_quote_payload(
        state=state,
        line_items=line_items,
        summary=summary,
    )
    state["pricing_report"] = _format_report(quoted_items, PRICING_RULES)

    return state