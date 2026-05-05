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
  3. Pricing     — for each atomic SKU, queries BU RAG once. A second
                   small LLM call on the SAME RAG response extracts
                   SAP code, packing, and monthly availability — zero
                   extra RAG calls for enrichment.
  4. Report      — formatted pricing table + compliance checklist.

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

from settings import GROQ_API_KEY, PRICING_RULES, PricingRules
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


# ─── Step 3 — Per-item pricing + catalog detail extraction ───────────────────

_DETAIL_EXTRACT_PROMPT = """
From the catalog text below, extract the following fields for the product
matching: "{requirement}"

Return ONLY a JSON object with these exact keys:
  - "sap_code"          : A single primary SAP code string (do NOT use "or", return only one code), or "N/A"
  - "packing"           : Very short packing format string (e.g. "20/box"), or "N/A"
  - "monthly_available" : monthly availability/stock (e.g. "500 units"), or "N/A"

No markdown. No explanation. Pure JSON.

Catalog text:
{catalog_text}
""".strip()


def _extract_catalog_details(
    requirement: str,
    catalog_text: str,
    api_key: Optional[str],
) -> Dict[str, str]:
    """
    Extract SAP code, packing, and monthly availability from the RAG response
    that was already fetched for pricing. Zero extra RAG calls.
    """
    raw = chat_completion(
        system="You extract structured catalog fields. Return JSON only.",
        user=_DETAIL_EXTRACT_PROMPT.format(
            requirement=requirement,
            catalog_text=catalog_text[:3000],  # keep within token budget
        ),
        temperature=0.0,
        max_tokens=120,
        api_key=api_key,
    )

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    raw = fence.group(1) if fence else (re.search(r"\{.*?\}", raw, re.DOTALL) or type("", (), {"group": lambda *a: "{}"})()).group(0)

    defaults = {"sap_code": "N/A", "packing": "N/A", "monthly_available": "N/A"}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {k: str(parsed.get(k) or "N/A") for k in defaults}
    except Exception:
        pass
    return defaults


def _extract_price(text: str) -> Optional[float]:
    patterns = [
        r"(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d+)?)",
        r"([\d,]+(?:\.\d+)?)\s*/\-",
        r"price[:\s]+([\d,]+(?:\.\d+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, text.lower())
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                continue
    m = re.search(r"\b(\d{2,}(?:\.\d+)?)\b", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _extract_qty(text: str) -> int:
    m = re.search(r"\b(\d+)\s*(?:units?|pcs?|pieces?|nos?\.?|qty)\b", text, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d{2,5})\b", text)
    if m:
        v = int(m.group(1))
        if 1 <= v <= 99999:
            return v
    return 100


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


def _price_one_item(
    requirement: str,
    rules: PricingRules,
    user_query: str,
    api_key: Optional[str],
) -> Dict[str, Any]:
    """
    Single BU RAG query per item.
    Extracts price via regex AND catalog details (SAP/packing/availability)
    via a small LLM call on the same RAG response — no second RAG call.
    """
    clean_req = requirement
    if clean_req.lower().startswith("supply "):
        clean_req = clean_req[7:].strip()
        
    price_query = (
        f"Identify the EXACT product matching: '{clean_req}'. "
        f"Carefully distinguish between fan regulators and light dimmers. "
        f"MUST return the exact unit price, SAP code, packing format, and monthly availability."
    )
    price_answer = bu_query(price_query, temperature=0.0, api_key=api_key)

    unit_price = _extract_price(price_answer)
    qty        = _extract_qty(user_query + " " + requirement)

    # Extract catalog details from the same RAG response
    details = _extract_catalog_details(requirement, price_answer, api_key)

    if unit_price is None:
        return {
            "requirement":       requirement,
            "raw_price_info":    price_answer,
            "catalog_cost_inr":  None,
            "qty":               qty,
            "note":              "Price not found in catalog",
            **details,
        }

    margin_pct = _clamp_margin(rules.base_margin_pct, rules)
    li = _build_line_item(requirement, qty, unit_price, margin_pct, rules, price_answer)
    
    # Inject catalog details so they appear in line_items for the report agent
    li["sap_code"] = details.get("sap_code", "Not available")
    li["packing"] = details.get("packing", "Not available")
    li["monthly_available"] = details.get("monthly_available", "Not available")

    return {
        "requirement":        requirement,
        "raw_price_info":     price_answer,
        "catalog_cost_inr":   li["base_unit_price"],
        "selling_price_inr":  round(li["base_unit_price"] * (1 + li["margin_pct"] / 100), rules.rounding_digits),
        "volume_discount_pct":li["volume_discount_pct"],
        "net_price_inr":      li["net_unit_price"],
        "gst_pct":            li["gst_pct"],
        "gst_amount_inr":     li["gst_amount_per_unit"],
        "price_with_gst_inr": li["final_unit_price"],
        "qty":                li["qty"],
        "line_total_inr":     li["line_total"],
        "line_item":          li,
        **details,   # sap_code, packing, monthly_available
    }


# ─── Aggregate helpers ────────────────────────────────────────────────────────

def _build_pricing_summary(
    line_items: List[Dict[str, Any]],
    rules: PricingRules,
    margin_pct: float,
) -> Dict[str, Any]:
    r = rules.rounding_digits
    subtotal_ex_gst = round(sum(i["net_unit_price"] * i["qty"] for i in line_items), r)
    total_gst       = round(sum(i["gst_amount_per_unit"] * i["qty"] for i in line_items), r)
    grand_total     = round(sum(i["line_total"] for i in line_items), r)
    base_subtotal   = round(sum(i["base_unit_price"] * i["qty"] for i in line_items), r)
    margin_value    = round(subtotal_ex_gst - base_subtotal, r)
    return {
        "currency":        rules.currency,
        "margin_pct":      round(margin_pct, r),
        "subtotal_base":   base_subtotal,
        "subtotal_ex_gst": subtotal_ex_gst,
        "total_gst":       total_gst,
        "grand_total":     grand_total,
        "margin_value":    margin_value,
        "items_count":     len(line_items),
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
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    margin_pct    = _clamp_margin(new_margin_pct, rules)
    updated_items = []
    for item in line_items:
        updated_items.append(_build_line_item(
            requirement    = item["requirement"],
            qty            = int(item["qty"]),
            unit_price     = float(item["base_unit_price"]),
            margin_pct     = margin_pct,
            rules          = rules,
            raw_price_info = item.get("raw_price_info", ""),
        ))
    summary = _build_pricing_summary(updated_items, rules, margin_pct)
    return updated_items, summary


def generate_initial_pricing(
    matched_items: List[Dict[str, Any]],
    rules: PricingRules,
    user_query: str,
    api_key: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    quoted, line_items = [], []
    for item in matched_items:
        priced = _price_one_item(item["requirement"], rules, user_query, api_key)
        quoted.append(priced)
        if priced.get("line_item"):
            line_items.append(priced["line_item"])
    return quoted, line_items


# ─── Report formatter ─────────────────────────────────────────────────────────

def _format_report(
    quoted_items: List[Dict[str, Any]],
    rules: PricingRules,
    compliance_items: Optional[List[Dict[str, Any]]] = None,
) -> str:
    sep  = "─" * 70
    lines = [
        "╔══ PRICING REPORT ══════════════════════════════════════════════════╗",
        f"  Margin: {rules.base_margin_pct}%  │  GST: {rules.gst_rate_pct}%  │  Currency: {rules.currency}",
        f"  Only priceable product SKUs are listed below.",
        sep,
    ]
    grand_total = 0.0
    for item in quoted_items:
        req = item["requirement"][:60]
        lines.append(f"  {req}")
        if item.get("catalog_cost_inr") is None:
            lines.append(f"    ⚠  {item.get('note', 'Price unavailable')}")
        else:
            lines.append(f"    SAP Code       : {item.get('sap_code', 'Not available')}")
            lines.append(f"    Packing        : {item.get('packing', 'Not available')}")
            lines.append(f"    Monthly Avail. : {item.get('monthly_available', 'Not available')}")
            lines.append(f"    Catalog cost   : ₹ {item['catalog_cost_inr']:>9.2f}")
            lines.append(f"    Selling price  : ₹ {item['selling_price_inr']:>9.2f}  (margin {rules.base_margin_pct}%)")
            if item["volume_discount_pct"]:
                lines.append(f"    Vol. discount  :  {item['volume_discount_pct']}% → ₹ {item['net_price_inr']:>9.2f}")
            lines.append(f"    GST ({item['gst_pct']}%)      : ₹ {item['gst_amount_inr']:>9.2f}")
            lines.append(f"    Price / unit   : ₹ {item['price_with_gst_inr']:>9.2f}")
            lines.append(f"    Qty            :  {item['qty']:>4d} units")
            lines.append(f"    Line total     : ₹ {item['line_total_inr']:>11.2f}")
            grand_total += item["line_total_inr"]
        lines.append(sep)

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
    Reads matched_items + user_query → writes quoted_items, line_items,
    pricing_summary, quote_payload, pricing_report.
    """
    api_key    = os.environ.get("GROQ_API_KEY") or GROQ_API_KEY
    user_query = state.get("user_query", "")

    all_matched = [m for m in state.get("matched_items", []) if m.get("matched")]

    _empty_summary = {
        "currency":        PRICING_RULES.currency,
        "margin_pct":      PRICING_RULES.base_margin_pct,
        "subtotal_base":   0.0,
        "subtotal_ex_gst": 0.0,
        "total_gst":       0.0,
        "grand_total":     0.0,
        "margin_value":    0.0,
        "items_count":     0,
    }

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
    quoted_items, line_items = generate_initial_pricing(atomic, PRICING_RULES, user_query, api_key)

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
            _pl(f"    Selling price  : ₹ {p['selling_price_inr']:.2f}  (margin {PRICING_RULES.base_margin_pct}%)", indent=4)
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

    summary = _build_pricing_summary(line_items, PRICING_RULES, PRICING_RULES.base_margin_pct)

    state["quoted_items"]    = quoted_items
    state["line_items"]      = line_items
    state["pricing_summary"] = summary
    state["quote_payload"]   = _build_quote_payload(state, line_items, summary)
    state["pricing_report"]  = _format_report(quoted_items, PRICING_RULES, compliance_items)

    return state