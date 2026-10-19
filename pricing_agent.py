"""
agents/pricing_agent.py
-----------------------
Pricing Agent — LangGraph node

Responsibility
--------------
For every requirement that was successfully matched by the Matching Agent,
the Pricing Agent:

  1. LLM filter  — one batched call to keep only priceable product SKUs,
                   dropping compliance/eligibility items completely.
  2. Atomiser    — one batched call to split compound requirements like
                   "16A and 25A switches" into individual atomic SKUs.
  3. Pricing     — for each atomic SKU, searches the structured catalogue
                   (embedding search + rerank) and one small LLM call picks
                   a product BY ID from that candidate list. The unit price,
                   SAP code and packing come from the catalogue DB, never
                   from LLM text. Quantity comes from the RFP or an explicit
                   "N units" phrase, else a default flagged for review.
  4. Report      — formatted pricing table + compliance checklist.

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

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from settings import OPENROUTER_API_KEY, PRICING_RULES, PricingRules
from bu_rag import get_bu_rag_tool
from catalogue import ProductRecord
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


def _pl(text: str, indent: int = 2) -> None:
    pad = " " * indent
    print(f"║{pad}{text:<{_W - indent - 2}}║")


def _sep(char: str = "─") -> None:
    print(f"╠{char * (_W - 2)}╣")


# ─── Step 1 — LLM-based priceable filter (single batched call) ───────────────

_FILTER_PROMPT = """
You are a procurement assistant. Classify each requirement as "priceable"
or "non-priceable".

"priceable"     — A physical product/SKU a vendor quotes with a unit price,
                  (switches, plates, sockets, cables, etc.)

"non-priceable" — Vendor qualification, compliance criterion, logistical
                  condition, document obligation (GST registration, delivery
                  timeline, company profile, past experience, packaging rules).
                  Note: ANY requirement that uses informational/documentation
                  phrases like "Submit", "Provide", "Comply", "Certify", or
                  "Confirm" MUST be marked as "non-priceable" (e.g. "Submit
                  price list", "Provide indicator variants").

Return ONLY a JSON array:
  [{{"requirement": "<text>", "priceable": true|false}}, ...]

No markdown. No explanation. Pure JSON.

Requirements:
{requirements_json}
""".strip()


def _filter_priceable(
    matched_items: List[Dict[str, Any]],
    api_key: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Single LLM call → (priceable_items, skipped_compliance_items).
    Fallback on parse failure: treat everything as priceable.
    """
    reqs = [{"requirement": m["requirement"]} for m in matched_items]
    raw = chat_completion(
        system="You classify procurement requirements. Return JSON only.",
        user=_FILTER_PROMPT.format(requirements_json=json.dumps(reqs, indent=2)),
        temperature=0.0,
        max_tokens=1500,
        api_key=api_key,
    )

    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    raw = fence.group(1) if fence else (re.search(r"\[.*\]", raw, re.DOTALL) or type("", (), {"group": lambda *a: raw})()).group(0)

    try:
        classifications = json.loads(raw)
        priceable_set = {
            item["requirement"].strip().lower()
            for item in classifications
            if isinstance(item, dict) and item.get("priceable") is True
        }
        priceable = [m for m in matched_items if m["requirement"].strip().lower() in priceable_set]
        skipped   = [m for m in matched_items if m["requirement"].strip().lower() not in priceable_set]
        return priceable, skipped
    except Exception:
        print("[PricingAgent] ⚠ Filter parse failed — pricing all matched items.")
        return matched_items, []


# ─── Step 2 — LLM-based atomiser (single batched call) ───────────────────────

_ATOMISE_PROMPT = """
You are a procurement data analyst. Split compound requirements that mention
multiple distinct products into individual atomic items — one product per entry.

Rules:
- "X and Y" / "X or Y" where X and Y are different specs → two entries
- Enumerated variants (e.g. "1-way, 2-way, bell push") → one entry each
- Size/rating RANGES (e.g. "1M to 18M", "6-16A") → EXPAND into individual standard specs (e.g., 1M, 2M, 3M, 4M, 6M, 8M, 12M, 18M outer plates, or 6A, 16A combined sockets) — create a separate entry for each size/rating.
- Already-atomic requirements → keep as-is (one entry)
- Never merge, summarise, or invent new requirements

Return ONLY a JSON array of strings — the atomised requirement texts.
No markdown. No explanation.

Input requirements:
{requirements_json}
""".strip()


def _atomise(
    items: List[Dict[str, Any]],
    api_key: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Single LLM call that expands compound requirements.
    Each atomised string inherits all fields from its closest parent item.
    Falls back to the original list on parse failure.
    """
    reqs = [m["requirement"] for m in items]
    raw = chat_completion(
        system="You split compound procurement requirements into atomic items. Return JSON only.",
        user=_ATOMISE_PROMPT.format(requirements_json=json.dumps(reqs, indent=2)),
        temperature=0.0,
        max_tokens=2000,
        api_key=api_key,
    )

    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    raw = fence.group(1) if fence else (re.search(r"\[.*\]", raw, re.DOTALL) or type("", (), {"group": lambda *a: raw})()).group(0)

    try:
        atomised: List[str] = json.loads(raw)
        if not isinstance(atomised, list):
            raise ValueError
        atomised = [str(r).strip() for r in atomised if str(r).strip()]
    except Exception:
        print("[PricingAgent] ⚠ Atomise parse failed — using original list.")
        return items

    # Inherit parent metadata for each atomised string
    def _best_parent(text: str) -> Dict[str, Any]:
        text_l = text.lower()
        best, best_score = items[0], 0
        for orig in items:
            score = sum(w in orig["requirement"].lower() for w in text_l.split())
            if score > best_score:
                best, best_score = orig, score
        return best

    return [{**_best_parent(r), "requirement": r} for r in atomised]


# ─── Step 3 — Per-item pricing from the structured catalogue ────────────────

_PICK_PROMPT = """
You are matching one RFP line item to a product in our catalogue.

RFP item: "{requirement}"

Candidate catalogue products:
{candidates}

Pick the single candidate that best matches the RFP item (rating, way/pole,
module size, product type). If none of them is the same kind of product,
answer null.

Return ONLY a JSON object: {{"product_id": "<id from the list>" | null, "reason": "<short>"}}
No markdown. No explanation outside the JSON.
""".strip()


def _candidate_products(query: str, top_k: int = 8) -> List[ProductRecord]:
    """Catalogue products most relevant to *query* (embedding search + rerank)."""
    return get_bu_rag_tool().search_products(query, top_k=top_k)


def _pick_product(
    requirement: str,
    candidates: List[ProductRecord],
    api_key: Optional[str],
) -> Tuple[Optional[ProductRecord], str]:
    """
    One small LLM call that chooses among *candidates* by id.
    The answer is only accepted if the id is one of the candidates, so the
    LLM can never invent a product or a price.
    """
    if not candidates:
        return None, "No catalogue candidates found."

    listing = "\n".join(
        f'- id={p.id} | {p.item_name} | range: {p.range_name} | variant: {p.variant or "-"}'
        for p in candidates
    )
    raw = chat_completion(
        system="You select catalogue products by id. Return JSON only.",
        user=_PICK_PROMPT.format(requirement=requirement, candidates=listing),
        temperature=0.0,
        max_tokens=150,
        api_key=api_key,
    )

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        parsed = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    by_id = {p.id: p for p in candidates}
    chosen = by_id.get(str(parsed.get("product_id") or "").strip())
    reason = str(parsed.get("reason") or "").strip()
    if chosen is None:
        return None, reason or "Model did not pick a listed product."
    return chosen, reason


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
                return int(value), "rfp"
        except (TypeError, ValueError):
            pass
    from_req = _explicit_qty(item.get("requirement", ""))
    if from_req:
        return from_req, "requirement"
    from_query = _explicit_qty(user_query)
    if from_query:
        return from_query, "query"
    return rules.default_qty, "default"


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
    api_key: Optional[str],
) -> Dict[str, Any]:
    """
    Search the catalogue, let the LLM pick one listed product, then price it
    from the catalogue's stored unit price. No price is ever read from text.
    """
    requirement = item["requirement"]
    clean_req = requirement
    if clean_req.lower().startswith("supply "):
        clean_req = clean_req[7:].strip()

    qty, qty_source = _resolve_qty(item, user_query, rules)
    product, reason = _pick_product(clean_req, _candidate_products(clean_req), api_key)

    if product is None:
        return {
            "requirement":       requirement,
            "raw_price_info":    reason,
            "catalog_cost_inr":  None,
            "qty":               qty,
            "qty_source":        qty_source,
            "note":              "No matching catalogue product",
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
    li["match_reason"] = reason

    return {
        "requirement":        requirement,
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
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Recompute every line and the totals for a new margin (and optional
    payment/freight terms) without any retrieval or LLM call. Catalogue
    fields on each line (product_id, sap_code, packing, stock, qty_source)
    are kept.
    """
    margin_pct    = _clamp_margin(new_margin_pct, rules)
    updated_items = []
    for item in line_items:
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
    api_key: Optional[str],
    margin_pct: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    margin = _clamp_margin(rules.base_margin_pct if margin_pct is None else margin_pct, rules)
    quoted, line_items = [], []
    for item in matched_items:
        priced = _price_one_item(item, rules, margin, user_query, api_key)
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
        f"  Only priceable product SKUs are listed below.",
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
        lines.append("─── Compliance & Eligibility Criteria (not priced) ─────────────────")
        for c in compliance_items:
            lines.append(f"  ✅  {c['requirement']}")

    return "\n".join(lines)


# ─── LangGraph node ───────────────────────────────────────────────────────────

def pricing_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node.
    Reads matched_items + user_query (+ optional margin_pct, advance_payment,
    freight_inr) → writes quoted_items, line_items, pricing_summary,
    quote_payload, pricing_report.
    """
    api_key    = os.environ.get("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
    user_query = state.get("user_query", "")
    margin_pct = _clamp_margin(state.get("margin_pct", PRICING_RULES.base_margin_pct), PRICING_RULES)
    terms      = {
        "advance_payment": bool(state.get("advance_payment", False)),
        "freight":         state.get("freight_inr"),
    }

    all_matched = [m for m in state.get("matched_items", []) if m.get("matched")]

    _empty_summary = _build_pricing_summary([], PRICING_RULES, margin_pct)

    if not all_matched:
        state["pricing_report"]  = "No matched products were found — pricing cannot be generated."
        state["quoted_items"]    = []
        state["line_items"]      = []
        state["pricing_summary"] = _empty_summary
        state["quote_payload"]   = _build_quote_payload(state, [], _empty_summary)
        return state

    # ── Step 1: filter — keep only priceable SKUs ─────────────────────────────
    print(f"\n[PricingAgent] Step 1 — LLM filter on {len(all_matched)} matched items …")
    priceable, compliance_items = _filter_priceable(all_matched, api_key)

    print(f"\n╔{'═' * (_W - 2)}╗")
    print(f"║  {'PRICING AGENT — CLASSIFICATION':<{_W - 4}}║")
    _sep()
    _pl(f"Total matched    : {len(all_matched)}")
    _pl(f"✅ Priceable SKUs : {len(priceable)}")
    for m in priceable:
        _pl(f"   • {m['requirement'][:_W - 10]}", indent=4)
    _pl(f"⏭  Compliance / skipped : {len(compliance_items)}")
    for m in compliance_items:
        _pl(f"   • {m['requirement'][:_W - 10]}", indent=4)
    print(f"╚{'═' * (_W - 2)}╝\n")

    if not priceable:
        msg = "No priceable product requirements among matched items."
        state["pricing_report"]  = msg
        state["quoted_items"]    = []
        state["line_items"]      = []
        state["pricing_summary"] = _empty_summary
        state["quote_payload"]   = _build_quote_payload(state, [], _empty_summary)
        return state

    # ── Step 2: atomise — split compound requirements ─────────────────────────
    print(f"[PricingAgent] Step 2 — Atomising {len(priceable)} priceable requirements …")
    atomic = _atomise(priceable, api_key)

    print(f"\n╔{'═' * (_W - 2)}╗")
    print(f"║  {'PRICING AGENT — ATOMISED SKUs':<{_W - 4}}║")
    _sep()
    _pl(f"Before atomisation : {len(priceable)}  →  After : {len(atomic)}")
    _sep("─")
    for i, item in enumerate(atomic, 1):
        _pl(f"{i:>2}. {item['requirement'][:_W - 10]}", indent=2)
    print(f"╚{'═' * (_W - 2)}╝\n")

    # ── Step 3: price each atomic SKU ────────────────────────────────────────
    print(f"[PricingAgent] Step 3 — Pricing {len(atomic)} atomic SKUs …")
    quoted_items, line_items = generate_initial_pricing(
        atomic, PRICING_RULES, user_query, api_key, margin_pct=margin_pct
    )

    # ── Rich pricing print ────────────────────────────────────────────────────
    print(f"\n╔{'═' * (_W - 2)}╗")
    print(f"║  {'PRICING AGENT — RESULTS':<{_W - 4}}║")
    _sep()
    grand = 0.0
    for i, p in enumerate(quoted_items, 1):
        _pl(f"{i:>2}. {p['requirement'][:_W - 10]}", indent=2)
        if p.get("catalog_cost_inr") is not None:
            _pl(f"    SAP Code       : {p.get('sap_code', 'N/A')}", indent=4)
            _pl(f"    Packing        : {p.get('packing', 'N/A')}", indent=4)
            _pl(f"    Monthly Avail. : {p.get('monthly_available', 'N/A')}", indent=4)
            _pl(f"    Catalog cost   : ₹ {p['catalog_cost_inr']:.2f}", indent=4)
            _pl(f"    Selling price  : ₹ {p['selling_price_inr']:.2f}  (margin {margin_pct}%)", indent=4)
            if p["volume_discount_pct"]:
                _pl(f"    Vol. discount  : {p['volume_discount_pct']}% → ₹ {p['net_price_inr']:.2f}", indent=4)
            _pl(f"    GST ({p['gst_pct']}%)      : ₹ {p['gst_amount_inr']:.2f}", indent=4)
            _pl(f"    Price / unit   : ₹ {p['price_with_gst_inr']:.2f}", indent=4)
            _pl(f"    Qty × Total    : {p['qty']} × ₹{p['price_with_gst_inr']:.2f} = ₹{p['line_total_inr']:.2f}", indent=4)
            grand += p["line_total_inr"]
        else:
            _pl(f"    ⚠  {p.get('note', 'Price not found')}", indent=4)
        _sep("─")
    _pl(f"GRAND TOTAL (incl. GST): ₹ {grand:,.2f}")
    print(f"╚{'═' * (_W - 2)}╝\n")

    summary = _build_pricing_summary(line_items, PRICING_RULES, margin_pct, **terms)

    state["quoted_items"]    = quoted_items
    state["line_items"]      = line_items
    state["pricing_summary"] = summary
    state["quote_payload"]   = _build_quote_payload(state, line_items, summary)
    state["pricing_report"]  = _format_report(quoted_items, PRICING_RULES, compliance_items, summary)

    return state