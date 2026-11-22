"""
evaluation/metrics.py
---------------------
Pure scoring functions for each agent's output against the sample RFPs'
ground truth (tests/sample_rfps.py). No LLM, no I/O: run_eval.py collects
the outputs, these functions turn them into numbers.

Matching rules are deliberately simple and explainable:
  - a requirement "covers" a truth item when every word of the item's
    description appears in it (plural/singular and authorised/authorized
    treated alike)
  - a truth term ("warranty 2 year") is found when all its words appear in
    one extracted requirement
  - a matched product is correct when it is one of the catalogue products the
    truth names ("Range [Variant] / Item name")
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

_SPELLING = {"authorized": "authorised", "authorize": "authorise", "colour": "color", "nos": "no"}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def words(text: str) -> Set[str]:
    out = set()
    for w in norm(text).split():
        w = _SPELLING.get(w, w)
        if w.endswith(("ches", "shes", "xes", "sses")):
            w = w[:-2]                       # switches, boxes
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]                       # sockets, plates
        out.add(w)
    return out


def covers(requirement: str, phrase: str) -> bool:
    need = words(phrase)
    return bool(need) and need <= words(requirement)


def ratio(n: int, d: int) -> Optional[float]:
    return round(n / d, 4) if d else None


# ─── Catalogue ground truth ───────────────────────────────────────────────────

def resolve_catalogue(spec: Optional[str], products: Sequence[Any]) -> Set[str]:
    """Product ids for a truth spec "Range [Variant] / Item name" ({} if None or unknown)."""
    if not spec:
        return set()
    range_part, _, item_part = spec.partition(" / ")
    want_item, want_range = norm(item_part), words(range_part)
    hits = [p for p in products
            if norm(p.item_name) == want_item and want_range <= words(f"{p.range_name} {p.variant or ''}")]
    if not hits:
        return set()
    # "Oro / …" must not also accept "Oro Metallica / …": keep the ranges that
    # add the fewest words beyond the spec.
    extra = lambda p: len(words(p.range_name) - want_range)
    best = min(extra(p) for p in hits)
    return {p.id for p in hits if extra(p) == best}


# ─── Agent scores ─────────────────────────────────────────────────────────────

def score_extraction(requirements: List[Dict[str, Any]], truth) -> Dict[str, Any]:
    """
    Items: found / exact quantity / unit, and whether they came from the table.
    Terms: non-item requirements found. Unmatched: requirements matching no
    item and no term (a precision proxy; listed for a person to check).
    Also returns item_by_req: requirement id -> truth item index (for matching).
    """
    item_by_req: Dict[str, int] = {}
    found = qty_ok = unit_ok = from_table = 0
    misses: List[str] = []
    for i, item in enumerate(truth.items):
        hit = next((r for r in requirements if r["id"] not in item_by_req and covers(r["requirement"], item.description)),
                   None)
        if hit is None:
            misses.append(item.description)
            continue
        item_by_req[hit["id"]] = i
        found += 1
        qty_ok += hit.get("quantity") == item.quantity
        unit_ok += norm(hit.get("unit") or "") == norm(item.unit)
        from_table += hit.get("origin") == "table"

    term_hits = {t: any(covers(r["requirement"], t) for r in requirements) for t in truth.terms}
    unmatched = [r["requirement"] for r in requirements
                 if r["id"] not in item_by_req and not any(covers(r["requirement"], t) for t in truth.terms)]
    n_items = len(truth.items)
    return {
        "requirements": len(requirements),
        "items_total": n_items,
        "item_recall": ratio(found, n_items),
        "qty_accuracy": ratio(qty_ok, n_items),
        "unit_accuracy": ratio(unit_ok, n_items),
        "items_from_table": ratio(from_table, n_items),
        "items_missed": misses,
        "terms_total": len(truth.terms),
        "term_recall": ratio(sum(term_hits.values()), len(truth.terms)),
        "terms_missed": [t for t, ok in term_hits.items() if not ok],
        "unmatched_requirements": unmatched,
        "precision_proxy": ratio(len(requirements) - len(unmatched), len(requirements)),
        "item_by_req": item_by_req,
    }


def score_summary(answer: str, truth) -> Dict[str, Any]:
    hits = {f: f.lower() in (answer or "").lower() for f in truth.summary_facts}
    return {"facts_total": len(hits), "fact_recall": ratio(sum(hits.values()), len(hits)),
            "facts_missed": [f for f, ok in hits.items() if not ok]}


def score_matching(matched_items: List[Dict[str, Any]], truth, item_by_req: Dict[str, int],
                   products: Sequence[Any]) -> Dict[str, Any]:
    """
    Per truth item that extraction found: classified as a product, the chosen
    product correct (top-1), correct within the alternatives (top-3), gaps
    detected for items the catalogue does not have, and accuracy per
    confidence band. "Auto-accept errors" are wrong products marked high
    confidence (status matched), which "Accept all" would approve.
    """
    expected = [resolve_catalogue(i.catalogue, products) for i in truth.items]
    unresolved = [truth.items[i].catalogue for i, ids in enumerate(expected) if truth.items[i].catalogue and not ids]

    rows = []
    for item in matched_items:
        root = str(item.get("id", "")).split(".")[0]
        if root in item_by_req:
            rows.append((item_by_req[root], item))

    classified_product = sum(1 for _, m in rows if m.get("type") == "product")
    in_cat = [(i, m) for i, m in rows if truth.items[i].catalogue and expected[i]]
    gaps_truth = [(i, m) for i, m in rows if truth.items[i].catalogue is None]

    def correct(i: int, m: Dict[str, Any]) -> bool:
        return m.get("product_id") in expected[i]

    top1 = sum(correct(i, m) for i, m in in_cat)
    top3 = sum(correct(i, m) or any(a.get("product_id") in expected[i] for a in m.get("alternatives") or [])
               for i, m in in_cat)
    bands: Dict[str, Dict[str, Any]] = {}
    for band in ("high", "medium", "low"):
        group = [(i, m) for i, m in in_cat if m.get("confidence") == band]
        bands[band] = {"items": len(group), "accuracy": ratio(sum(correct(i, m) for i, m in group), len(group))}
    wrong = [{"requirement": m["requirement"], "chosen": m.get("product_name"), "range": m.get("product_range"),
              "confidence": m.get("confidence"), "expected": truth.items[i].catalogue}
             for i, m in in_cat if not correct(i, m)]
    gap_found = sum(1 for _, m in gaps_truth if m.get("status") == "gap")
    false_gaps = sum(1 for i, m in in_cat if m.get("status") == "gap")
    return {
        "items_scored": len(rows),
        "classified_as_product": ratio(classified_product, len(rows)),
        "top1_accuracy": ratio(top1, len(in_cat)),
        "top3_accuracy": ratio(top3, len(in_cat)),
        "accuracy_by_confidence": bands,
        "auto_accept_errors": sum(1 for i, m in in_cat if not correct(i, m) and m.get("status") == "matched"),
        "gaps_expected": len(gaps_truth),
        "gaps_detected": gap_found,
        "false_gaps": false_gaps,
        "wrong_products": wrong,
        "unresolved_truth": unresolved,
    }


def score_pricing(line_items: List[Dict[str, Any]], summary: Dict[str, Any], rules, products_by_id: Dict[str, Any],
                  truth, item_by_req: Dict[str, int]) -> Dict[str, Any]:
    """
    Recompute every line independently from the catalogue price and the
    pricing rules, check the totals add up, and check quantities against the
    RFP truth.
    """
    r = rules.rounding_digits
    exact = 0
    problems: List[str] = []
    for li in line_items:
        product = products_by_id.get(li.get("product_id"))
        if product is None:
            problems.append(f"{li.get('requirement')}: product not in catalogue")
            continue
        qty = int(li["qty"])
        tier = next((t["discount_pct"] for t in rules.volume_discount_tiers if t["min_qty"] <= qty <= t["max_qty"]), 0.0)
        sell = round(product.unit_price * (1 + li["margin_pct"] / 100), r)
        net = round(sell * (1 - tier / 100), r)
        gst = round(net * rules.gst_rate_pct / 100, r)
        total = round(round(net + gst, r) * qty, r)
        if abs(total - li["line_total"]) < 0.011 and abs(li["base_unit_price"] - product.unit_price) < 0.011:
            exact += 1
        else:
            problems.append(f"{li.get('requirement')}: expected {total}, got {li['line_total']}")
    lines_sum = round(sum(li["line_total"] for li in line_items), r)
    totals_ok = abs(lines_sum + summary.get("freight_incl_gst", 0) - summary.get("payment_discount_incl_gst", 0)
                    - summary.get("grand_total", 0)) < 0.05

    # Every priced line that comes from a truth item, split parts included (a
    # split part that lost the RFP quantity counts as wrong).
    qty_right = qty_scored = 0
    for li in line_items:
        root = str(li.get("requirement_id", "")).split(".")[0]
        if root in item_by_req:
            qty_scored += 1
            qty_right += li["qty"] == truth.items[item_by_req[root]].quantity
    sources: Dict[str, int] = {}
    for li in line_items:
        sources[li.get("qty_source", "?")] = sources.get(li.get("qty_source", "?"), 0) + 1
    return {
        "lines": len(line_items),
        "arithmetic_exact": ratio(exact, len(line_items)),
        "totals_consistent": totals_ok,
        "qty_matches_rfp": ratio(qty_right, qty_scored),
        "qty_sources": sources,
        "grand_total_inr": summary.get("grand_total"),
        "problems": problems[:10],
    }


_CLAIM_RE = re.compile(r"\bwe (hereby )?(confirm|certify|are an? authori[sz]ed|hold a valid|have a valid|comply|meet)\b", re.I)


def score_proposal(sections: List[Dict[str, Any]], matched_items: List[Dict[str, Any]],
                   amount_warnings: Dict[str, str]) -> Dict[str, Any]:
    """Completeness, money amounts in AI text, and compliance claims nobody confirmed."""
    confirmed = sum(1 for m in matched_items if m.get("type") == "compliance" and m.get("status") == "met")
    compliance_text = " ".join(s.get("body", "") for s in sections if s.get("key") == "compliance")
    sentences = [x.strip() for x in re.split(r"[.!?\n]+", compliance_text) if x.strip()]
    claims = [x for x in sentences if _CLAIM_RE.search(x)]
    return {
        "sections": len(sections),
        "empty_sections": [s["key"] for s in sections if not s.get("body", "").strip()],
        "words": sum(len(s.get("body", "").split()) for s in sections),
        "amount_mentions": len(amount_warnings),
        "compliance_confirmed_by_reviewer": confirmed,
        "unconfirmed_claims": len(claims) if confirmed == 0 else 0,
        "claim_phrases": claims if confirmed == 0 else [],
    }


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None
