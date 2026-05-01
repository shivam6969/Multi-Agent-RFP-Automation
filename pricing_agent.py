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

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from settings import GROQ_API_KEY, PRICING_RULES, PricingRules
from bu_rag import BuRagTool, get_bu_rag_tool
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


# ─── Per-item pricing ─────────────────────────────────────────────────────────

def _price_one_item(
    requirement: str,
    bu_tool: BuRagTool,
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
    price_answer = bu_tool.run_query(
        price_query, temperature=0.0, api_key=api_key
    )

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

    # Apply margin (cost → selling price)
    selling_price = round(unit_price * (1 + rules.base_margin_pct / 100), 2)

    # Apply volume discount
    disc_pct = _volume_discount(qty, rules)
    discounted_price = round(selling_price * (1 - disc_pct / 100), 2)

    # GST
    gst_amount = round(discounted_price * rules.gst_rate_pct / 100, 2)
    price_with_gst = round(discounted_price + gst_amount, 2)

    # Line total
    line_total = round(price_with_gst * qty, 2)

    return {
        "requirement": requirement,
        "raw_price_info": price_answer,
        "catalog_cost_inr": unit_price,
        "selling_price_inr": selling_price,
        "volume_discount_pct": disc_pct,
        "net_price_inr": discounted_price,
        "gst_pct": rules.gst_rate_pct,
        "gst_amount_inr": gst_amount,
        "price_with_gst_inr": price_with_gst,
        "qty": qty,
        "line_total_inr": line_total,
    }


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
        if item.get("unit_price_inr") is None:
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
        return state

    bu_tool = get_bu_rag_tool()
    user_query = state.get("user_query", "")

    quoted_items: List[Dict[str, Any]] = []
    total_items = len(matched)
    print(f"[PricingAgent] Pricing {total_items} matched items …")
    for i, item in enumerate(matched, 1):
        print(f"[PricingAgent] {i}/{total_items} Pricing: {item['requirement'][:60]}…")
        priced = _price_one_item(
            requirement=item["requirement"],
            bu_tool=bu_tool,
            rules=PRICING_RULES,
            user_query=user_query,
            api_key=api_key,
        )
        quoted_items.append(priced)
        price_str = f"₹{priced['catalog_cost_inr']}" if priced.get('catalog_cost_inr') else 'N/A'
        print(f"[PricingAgent]   → Price: {price_str}")

    state["quoted_items"] = quoted_items
    state["pricing_report"] = _format_report(quoted_items, PRICING_RULES)

    return state
