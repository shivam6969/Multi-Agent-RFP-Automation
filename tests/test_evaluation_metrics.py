"""Scoring functions of the evaluation harness (offline, no LLM)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from catalogue import get_catalogue
from evaluation import metrics as M
from settings import PRICING_RULES
from tests.sample_rfps import SampleTruth, TruthItem

TRUTH = SampleTruth(
    name="t", title="T", buyer="B",
    items=[
        TruthItem("16A 2Way switch", 300, catalogue="Coral Ebony / 16A 2Way Switch"),
        TruthItem("TV socket", 120, catalogue="Coral Ebony / TV Socket"),
        TruthItem("Smart door lock", 10, catalogue=None),
    ],
    terms=["GST registration", "warranty 2 year"],
    summary_facts=["ABC Ltd", "25,000"],
)


def req(rid, text, qty=None, unit=None, origin="table"):
    return {"id": rid, "requirement": text, "quantity": qty, "unit": unit, "origin": origin}


class ExtractionScoreTests(unittest.TestCase):
    def test_items_terms_and_unmatched(self) -> None:
        s = M.score_extraction([
            req("R1", "Switches - 16A 2Way switches", 300, "Nos"),
            req("R2", "TV socket", 100, "Nos"),                     # wrong quantity
            req("R3", "Valid GST registration", origin="text"),
            req("R4", "140124, Some Board, Nangal", origin="text"),  # matches nothing
        ], TRUTH)
        self.assertEqual(s["item_recall"], round(2 / 3, 4))
        self.assertEqual(s["qty_accuracy"], round(1 / 3, 4))
        self.assertEqual(s["items_missed"], ["Smart door lock"])
        self.assertEqual(s["term_recall"], 0.5)
        self.assertEqual(s["terms_missed"], ["warranty 2 year"])
        self.assertEqual(s["unmatched_requirements"], ["140124, Some Board, Nangal"])
        self.assertEqual(s["item_by_req"], {"R1": 0, "R2": 1})

    def test_spelling_and_plurals_are_equivalent(self) -> None:
        self.assertTrue(M.covers("Authorized distributors of the brand", "authorised distributor"))
        self.assertTrue(M.covers("2-year warranty from delivery", "warranty 2 year"))


class MatchingScoreTests(unittest.TestCase):
    def test_top1_top3_gaps_and_auto_accept_errors(self) -> None:
        products = get_catalogue().all_products()
        switch = next(iter(M.resolve_catalogue("Coral Ebony / 16A 2Way Switch", products)))
        tv = next(iter(M.resolve_catalogue("Coral Ebony / TV Socket", products)))
        other = next(p.id for p in products if p.id not in (switch, tv))
        items = [
            {"id": "R1", "requirement": "16A 2Way switch", "type": "product", "product_id": switch,
             "confidence": "high", "status": "matched", "alternatives": []},
            {"id": "R2.1", "requirement": "TV socket", "type": "product", "product_id": other,
             "confidence": "high", "status": "matched", "alternatives": [{"product_id": tv}]},
            {"id": "R3", "requirement": "Smart door lock", "type": "product", "status": "gap"},
        ]
        s = M.score_matching(items, TRUTH, {"R1": 0, "R2": 1, "R3": 2}, products)
        self.assertEqual((s["top1_accuracy"], s["top3_accuracy"]), (0.5, 1.0))
        self.assertEqual(s["accuracy_by_confidence"]["high"], {"items": 2, "accuracy": 0.5})
        self.assertEqual(s["auto_accept_errors"], 1)
        self.assertEqual((s["gaps_expected"], s["gaps_detected"]), (1, 1))

    def test_resolve_prefers_the_exact_range(self) -> None:
        products = get_catalogue().all_products()
        ids = M.resolve_catalogue("Oro / RJ-45 Jack with cat 6", products)
        self.assertEqual(len(ids), 1)
        self.assertEqual(get_catalogue().get(next(iter(ids))).range_name, "Oro")


class PricingScoreTests(unittest.TestCase):
    def test_independent_recomputation(self) -> None:
        p = SimpleNamespace(id="p1", unit_price=100.0)
        good = {"product_id": "p1", "qty": 250, "margin_pct": 20.0, "base_unit_price": 100.0,
                # 100 * 1.2 = 120; tier 200-499 = 5% -> 114; GST 18% -> 134.52; x250
                "line_total": 33630.0, "requirement": "x", "requirement_id": "R1", "qty_source": "rfp"}
        bad = dict(good, line_total=99.0, requirement_id="R2")
        s = M.score_pricing([good, bad], {"grand_total": 33729.0}, PRICING_RULES, {"p1": p}, TRUTH,
                            {"R1": 0, "R2": 1})
        self.assertEqual(s["arithmetic_exact"], 0.5)
        self.assertTrue(s["totals_consistent"])
        self.assertEqual(s["qty_sources"], {"rfp": 2})


class ProposalScoreTests(unittest.TestCase):
    def test_unconfirmed_compliance_claims(self) -> None:
        sections = [{"key": "compliance", "body": "We confirm that we are an authorised distributor."},
                    {"key": "terms", "body": "Valid 90 days."}]
        items = [{"type": "compliance", "status": "needs_review"}]
        s = M.score_proposal(sections, items, {})
        self.assertEqual(s["unconfirmed_claims"], 1)
        items[0]["status"] = "met"
        self.assertEqual(M.score_proposal(sections, items, {})["unconfirmed_claims"], 0)


if __name__ == "__main__":
    unittest.main()
