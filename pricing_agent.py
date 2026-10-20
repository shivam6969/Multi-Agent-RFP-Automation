"""
agents/pricing_agent.py
-----------------------
Pricing Agent — LangGraph node

Responsibility
--------------
Prices the products the Matching Agent chose. Pricing makes NO LLM calls
and no retrieval:

  1. For every matched product item (matched_items[*].product_id), read the
     unit price, SAP code, HSN and packing from the catalogue database.
  2. Quantity comes from the RFP, else an explicit "N units" phrase, else
     PricingRules.default_qty (flagged qty_source="default" for review).
  3. Report — formatted pricing table + compliance checklist.

All money math is deterministic (PricingRules): margin → volume discount →
GST per line; then the optional advance-payment discount and freight on
the totals.

Outputs written to state
  - state["quoted_items"]    — list of per-product pricing dicts
                               (includes sap_code, packing, monthly_available)
  - state["line_items"]      — frontend-ready rows for repricing/export
  - state["pricing_summary"] — aggregate totals
  - state["quote_payload"]   — integration contract for PDF generation
  - state["pricing_report"]  — human-readable formatted report
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from settings import PRICING_RULES, PricingRules
from catalogue import ProductRecord, get_catalogue
from state import AgentState


# ─── Quantity ─────────────────────────────────────────────────────────────────

_EXPLICIT_QTY_PATTERNS = [
    # "500 units", "1,200 pcs", "50 nos" — not "20/120 units" (a packing format)
    re.compile(r"(?<![/\d,])(\d{1,3}(?:,\d{3})+|\d+)\s*(?:units?|pcs?|pieces?|nos?\.?|numbers)\b", re.I),
    # "qty: 500", "quantity - 500"
    re.compile(r"\b(?:qty|quantity)\.?\s*[:=\-]?\s*(\d{1,3}(?:,\d{3})+|\d+)\b", re.I),
]


def _explicit_qty(text: str) -> Optional[int]:
    for pattern in _EXPLICIT_QTY_PATTERNS:
        m = pattern.search(text or "")
        if m:
            value = int(m.group(1).replace(",", ""))
            if value > 0:
                return value
    return None


def _resolve_qty(item: Dict[str, Any], user_query: str, rules: PricingRules) -> Tuple[int, str]:
    """
    Quantity for a line item and where it came from:
      "reviewer"    — entered by a human during review
      "rfp"         — quantity field extracted from the RFP
      "requirement" — explicit "N units" phrase in the requirement text
      "query"       — explicit "N units" phrase in the user's question
      "default"     — nothing stated; rules.default_qty, needs human review
    Numbers inside product names ("RJ-45", "16A") are never used.
    """
    for key in ("qty", "quantity"):
        value = item.get(key)
        try:
            if value is not None and int(value) > 0:
                return int(value), ("reviewer" if item.get("qty_reviewed") else "rfp")
        except (TypeError, ValueError):
            pass
    from_req = _explicit_qty(item.get("requirement", ""))
    if from_req:
        return from_req, "requirement"
    from_query = _explicit_qty(user_query)
    if from_query:
        return from_query, "query"
    return rules.default_qty, "default"


# ─── Deterministic line pricing ───────────────────────────────────────────────

def _volume_discount(qty: int, rules: PricingRules) -> float:
    for tier in rules.volume_discount_tiers:
        if tier["min_qty"] <= qty <= tier["max_qty"]:
            return tier["discount_pct"]
    return 0.0


def _clamp_margin(margin_pct: float, rules: PricingRules) -> float:
    return max(rules.min_margin_pct, min(rules.max_margin_pct, margin_pct))


def _build_line_item(
    requirement: str,
    qty: int,
    unit_price: float,
    margin_pct: float,
    rules: PricingRules,
    raw_price_info: str,
) -> Dict[str, Any]:
    r = rules.rounding_digits
    discount_pct    = _volume_discount(qty, rules)
    selling_price   = round(unit_price * (1 + margin_pct / 100), r)
    discounted      = round(selling_price * (1 - discount_pct / 100), r)
    gst_amount      = round(discounted * rules.gst_rate_pct / 100, r)
    final_unit      = round(discounted + gst_amount, r)
    line_total      = round(final_unit * qty, r)
    return {
        "requirement":         requirement,
        "qty":                 qty,
        "currency":            rules.currency,
        "base_unit_price":     round(unit_price, r),
        "margin_pct":          round(margin_pct, r),
        "volume_discount_pct": round(discount_pct, r),
        "net_unit_price":      discounted,
        "gst_pct":             rules.gst_rate_pct,
        "gst_amount_per_unit": gst_amount,
        "final_unit_price":    final_unit,
        "line_total":          line_total,
        "raw_price_info":      raw_price_info,
    }


def _product_fields(product: ProductRecord) -> Dict[str, Any]:
    """Catalogue fields carried on every line item (and kept on repricing)."""
    return {
        "product_id":        product.id,
        "product_name":      product.item_name,
        "product_range":     product.range_name,
        "product_variant":   product.variant,
        "sap_code":          product.sap_code,
        "hsn_code":          product.hsn_code,
        "packing":           product.packing,
        # Demo stock: synthetic figures, see catalogue.demo_stock_for()
        "monthly_available": product.monthly_capacity,
        "stock_on_hand":     product.on_hand,
        "lead_time_days":    product.lead_time_days,
        "stock_is_demo":     product.stock_is_demo,
    }


def _price_one_item(
    item: Dict[str, Any],
    rules: PricingRules,
    margin_pct: float,
    user_query: str,
) -> Dict[str, Any]:
    """
    Price the catalogue product the Matching Agent chose for *item*
    (item["product_id"]). No retrieval and no LLM call: the unit price,
    SAP code and packing are read from the catalogue database.
    """
    requirement = item["requirement"]
    qty, qty_source = _resolve_qty(item, user_query, rules)
    product_id = item.get("product_id")
    product = get_catalogue().get(product_id) if product_id else None

    if product is None:
        note = ("Selected product is no longer in the catalogue" if product_id
                else "No catalogue product selected")
        return {
            "requirement":       requirement,
            "requirement_id":    item.get("id"),
            "raw_price_info":    item.get("rationale", ""),
            "catalog_cost_inr":  None,
            "qty":               qty,
            "qty_source":        qty_source,
            "note":              note,
            "sap_code":          "N/A",
            "packing":           "N/A",
            "monthly_available": "N/A",
        }

    raw_price_info = (
        f"Catalogue: {product.display_name()} | SAP {product.sap_code} | "
        f"Rs. {product.unit_price:g} per piece ({product.price_raw})"
    )
    li = _build_line_item(requirement, qty, product.unit_price, margin_pct, rules, raw_price_info)
    li.update(_product_fields(product))
    li["qty_source"] = qty_source
    li["requirement_id"] = item.get("id")
    li["match_status"] = item.get("status")
    li["match_reason"] = item.get("rationale", "")

    return {
        "requirement":        requirement,
        "requirement_id":     item.get("id"),
        "raw_price_info":     raw_price_info,
        "catalog_cost_inr":   li["base_unit_price"],
        "selling_price_inr":  round(li["base_unit_price"] * (1 + li["margin_pct"] / 100), rules.rounding_digits),
        "volume_discount_pct":li["volume_discount_pct"],
        "net_price_inr":      li["net_unit_price"],
        "gst_pct":            li["gst_pct"],
        "gst_amount_inr":     li["gst_amount_per_unit"],
        "price_with_gst_inr": li["final_unit_price"],
        "qty":                li["qty"],
        "qty_source":         qty_source,
        "line_total_inr":     li["line_total"],
        "line_item":          li,
        **_product_fields(product),
    }


# ─── Aggregate helpers ────────────────────────────────────────────────────────

def _build_pricing_summary(
    line_items: List[Dict[str, Any]],
    rules: PricingRules,
    margin_pct: float,
    *,
    advance_payment: bool = False,
    freight: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Totals for a quote. Line items already include margin, volume discount
    and GST. On top of that:
      - advance payment: rules.payment_terms_discount_pct off the goods value
      - freight: flat amount (rules.freight_flat_inr unless overridden)
    GST applies to (freight - payment discount) at the same rate as the goods.
    With no discount and no freight the totals equal the sum of the lines.
    """
    r = rules.rounding_digits
    gst_rate        = rules.gst_rate_pct / 100
    subtotal_ex_gst = round(sum(i["net_unit_price"] * i["qty"] for i in line_items), r)
    lines_gst       = round(sum(i["gst_amount_per_unit"] * i["qty"] for i in line_items), r)
    lines_total     = round(sum(i["line_total"] for i in line_items), r)
    base_subtotal   = round(sum(i["base_unit_price"] * i["qty"] for i in line_items), r)
    margin_value    = round(subtotal_ex_gst - base_subtotal, r)

    discount_pct    = rules.payment_terms_discount_pct if advance_payment else 0.0
    discount_value  = round(subtotal_ex_gst * discount_pct / 100, r)
    discount_gst    = round(discount_value * gst_rate, r)
    freight_value   = round(rules.freight_flat_inr if freight is None else max(0.0, float(freight)), r)
    freight_gst     = round(freight_value * gst_rate, r)
    adjustment      = round(freight_value - discount_value, r)
    adjustment_gst  = round(freight_gst - discount_gst, r)

    return {
        "currency":               rules.currency,
        "margin_pct":             round(margin_pct, r),
        "subtotal_base":          base_subtotal,
        "subtotal_ex_gst":        subtotal_ex_gst,
        "advance_payment":        advance_payment,
        "payment_discount_pct":   discount_pct,
        "payment_discount_value": discount_value,
        "payment_discount_incl_gst": round(discount_value + discount_gst, r),
        "freight":                freight_value,
        "freight_incl_gst":       round(freight_value + freight_gst, r),
        "taxable_value":          round(subtotal_ex_gst + adjustment, r),
        "total_gst":              round(lines_gst + adjustment_gst, r),
        "grand_total":            round(lines_total + adjustment + adjustment_gst, r),
        "margin_value":           margin_value,
        "items_count":            len(line_items),
        "qty_defaulted_count":    sum(1 for i in line_items if i.get("qty_source") == "default"),
    }


def _build_quote_payload(
    state: AgentState,
    line_items: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "quote_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source":       "pricing_agent_v2",
            "user_query":   state.get("user_query", ""),
        },
        "pricing_summary": summary,
        "line_items":      line_items,
    }


# ─── Public repricing helper (no re-retrieval) ────────────────────────────────

def reprice_with_margin(
    line_items: List[Dict[str, Any]],
    new_margin_pct: float,
    rules: PricingRules,
    *,
    advance_payment: bool = False,
    freight: Optional[float] = None,
    quantities: Optional[Dict[str, int]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Recompute every line and the totals for a new margin (and optional
    payment/freight terms and per-line quantities, keyed by requirement_id)
    without any retrieval or LLM call. Catalogue fields on each line
    (product_id, sap_code, packing, stock, qty_source) are kept.
    """
    margin_pct    = _clamp_margin(new_margin_pct, rules)
    quantities    = quantities or {}
    updated_items = []
    for item in line_items:
        override = quantities.get(item.get("requirement_id", ""))
        if override is not None:
            if int(override) <= 0:
                raise ValueError(f"Quantity must be positive for {item.get('requirement_id')}")
            item = {**item, "qty": int(override), "qty_source": "reviewer"}
        updated_items.append({**item, **_build_line_item(
            requirement    = item["requirement"],
            qty            = int(item["qty"]),
            unit_price     = float(item["base_unit_price"]),
            margin_pct     = margin_pct,
            rules          = rules,
            raw_price_info = item.get("raw_price_info", ""),
        )})
    summary = _build_pricing_summary(
        updated_items, rules, margin_pct,
        advance_payment=advance_payment, freight=freight,
    )
    return updated_items, summary


def generate_initial_pricing(
    matched_items: List[Dict[str, Any]],
    rules: PricingRules,
    user_query: str,
    api_key: Optional[str] = None,  # unused since pricing makes no LLM calls; kept for callers
    margin_pct: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    margin = _clamp_margin(rules.base_margin_pct if margin_pct is None else margin_pct, rules)
    quoted, line_items = [], []
    for item in matched_items:
        priced = _price_one_item(item, rules, margin, user_query)
        quoted.append(priced)
        if priced.get("line_item"):
            line_items.append(priced["line_item"])
    return quoted, line_items


# ─── Report formatter ─────────────────────────────────────────────────────────

def _format_report(
    quoted_items: List[Dict[str, Any]],
    rules: PricingRules,
    compliance_items: Optional[List[Dict[str, Any]]] = None,
    summary: Optional[Dict[str, Any]] = None,
) -> str:
    summary = summary or {}
    margin = summary.get("margin_pct", rules.base_margin_pct)
    sep  = "─" * 70
    lines = [
        "╔══ PRICING REPORT ══════════════════════════════════════════════════╗",
        f"  Margin: {margin}%  │  GST: {rules.gst_rate_pct}%  │  Currency: {rules.currency}",
        "  Only priceable product SKUs are listed below.",
        "  Stock figures are demo data.",
        sep,
    ]
    grand_total = 0.0
    for item in quoted_items:
        req = item["requirement"][:60]
        lines.append(f"  {req}")
        if item.get("catalog_cost_inr") is None:
            lines.append(f"    ⚠  {item.get('note', 'Price unavailable')}")
        else:
            qty_note = "  ⚠ default, not stated in RFP" if item.get("qty_source") == "default" else ""
            lines.append(f"    Product        : {item.get('product_name', '')} ({item.get('product_range', '')})")
            lines.append(f"    SAP Code       : {item.get('sap_code', 'Not available')}")
            lines.append(f"    Packing        : {item.get('packing', 'Not available')}")
            lines.append(f"    Monthly Avail. : {item.get('monthly_available', 'Not available')} (demo)")
            lines.append(f"    Catalog cost   : ₹ {item['catalog_cost_inr']:>9.2f}")
            lines.append(f"    Selling price  : ₹ {item['selling_price_inr']:>9.2f}  (margin {margin}%)")
            if item["volume_discount_pct"]:
                lines.append(f"    Vol. discount  :  {item['volume_discount_pct']}% → ₹ {item['net_price_inr']:>9.2f}")
            lines.append(f"    GST ({item['gst_pct']}%)      : ₹ {item['gst_amount_inr']:>9.2f}")
            lines.append(f"    Price / unit   : ₹ {item['price_with_gst_inr']:>9.2f}")
            lines.append(f"    Qty            :  {item['qty']:>4d} units{qty_note}")
            lines.append(f"    Line total     : ₹ {item['line_total_inr']:>11.2f}")
            grand_total += item["line_total_inr"]
        lines.append(sep)

    if summary.get("payment_discount_value"):
        lines.append(
            f"  Advance-payment discount ({summary['payment_discount_pct']}%): "
            f"- ₹ {summary['payment_discount_value']:,.2f} (+ GST adjustment)"
        )
    if summary.get("freight"):
        lines.append(f"  Freight: + ₹ {summary['freight']:,.2f} (+ GST)")
    grand_total = summary.get("grand_total", grand_total)
    lines.append(f"  GRAND TOTAL (incl. GST): ₹ {grand_total:,.2f}")
    lines.append("╚════════════════════════════════════════════════════════════════════╝")

    if compliance_items:
        lines.append("")
        lines.append("─── Compliance & Eligibility (not priced, confirm manually) ────────")
        for c in compliance_items:
            lines.append(f"  ☐  {c['requirement']}")

    return "\n".join(lines)


# ─── LangGraph node ───────────────────────────────────────────────────────────

def pricing_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node.
    Reads matched_items + user_query (+ optional margin_pct, advance_payment,
    freight_inr) → writes quoted_items, line_items, pricing_summary,
    quote_payload, pricing_report. Makes no LLM calls.
    """
    user_query = state.get("user_query", "")
    margin_pct = _clamp_margin(state.get("margin_pct", PRICING_RULES.base_margin_pct), PRICING_RULES)
    terms      = {
        "advance_payment": bool(state.get("advance_payment", False)),
        "freight":         state.get("freight_inr"),
    }

    items = state.get("matched_items", [])
    to_price = [m for m in items if m.get("matched") and m.get("type", "product") == "product"]
    compliance_items = [m for m in items if m.get("type") == "compliance"]

    if not to_price:
        empty = _build_pricing_summary([], PRICING_RULES, margin_pct, **terms)
        state["pricing_report"]  = "No matched products were found — pricing cannot be generated."
        state["quoted_items"]    = []
        state["line_items"]      = []
        state["pricing_summary"] = empty
        state["quote_payload"]   = _build_quote_payload(state, [], empty)
        return state

    print(f"[PricingAgent] Pricing {len(to_price)} matched products from the catalogue …")
    quoted_items, line_items = generate_initial_pricing(
        to_price, PRICING_RULES, user_query, margin_pct=margin_pct
    )
    summary = _build_pricing_summary(line_items, PRICING_RULES, margin_pct, **terms)

    state["quoted_items"]    = quoted_items
    state["line_items"]      = line_items
    state["pricing_summary"] = summary
    state["quote_payload"]   = _build_quote_payload(state, line_items, summary)
    state["pricing_report"]  = _format_report(quoted_items, PRICING_RULES, compliance_items, summary)

    print(state["pricing_report"])
    return state
