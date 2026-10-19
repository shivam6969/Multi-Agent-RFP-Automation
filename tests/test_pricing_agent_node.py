from __future__ import annotations

import unittest
from unittest.mock import patch

from pricing_agent import pricing_agent_node
from tests.fake_catalogue_search import product

SWITCH = product("Coral Ebony", "16A 2Way Switch")
TV = product("Coral Ebony", "TV Socket")

MATCHED_ITEMS = [
    {"id": "R1", "requirement": "Supply 16A 2Way switches", "type": "product", "matched": True,
     "status": "matched", "product_id": SWITCH.id, "qty": 300, "rationale": "16A 2-way"},
    {"id": "R2", "requirement": "Supply TV sockets", "type": "product", "matched": True,
     "status": "needs_review", "product_id": TV.id},
    {"id": "R3", "requirement": "Supply garden hoses", "type": "product", "matched": False,
     "status": "gap"},
    {"id": "R4", "requirement": "Provide GST registration", "type": "compliance",
     "matched": False, "status": "needs_review"},
    {"id": "R5", "requirement": "Old-style item without a product", "matched": True},
]


@patch("llm_json.chat_completion", side_effect=AssertionError("pricing must not call an LLM"))
@patch("llm.chat_completion", side_effect=AssertionError("pricing must not call an LLM"))
class PricingAgentNodeTests(unittest.TestCase):
    def test_prices_come_from_the_catalogue(self, *_mocks) -> None:
        out = pricing_agent_node({"user_query": "Prepare a quote", "margin_pct": 20.0,
                                  "matched_items": MATCHED_ITEMS})

        lines = {li["requirement_id"]: li for li in out["line_items"]}
        self.assertEqual(set(lines), {"R1", "R2"})

        switch = lines["R1"]
        self.assertEqual(switch["product_id"], SWITCH.id)
        self.assertEqual(switch["sap_code"], SWITCH.sap_code)
        self.assertEqual(switch["base_unit_price"], 191.0)
        self.assertEqual((switch["qty"], switch["qty_source"]), (300, "rfp"))
        self.assertEqual(switch["final_unit_price"], 256.93)
        self.assertEqual(switch["match_status"], "matched")

        tv = lines["R2"]
        self.assertEqual(tv["base_unit_price"], 115.0)
        self.assertEqual(tv["qty_source"], "default")

        # The gap and the compliance item are not priced; the item without a
        # product is listed as unpriced.
        unpriced = [q for q in out["quoted_items"] if q.get("catalog_cost_inr") is None]
        self.assertEqual([q["requirement_id"] for q in unpriced], ["R5"])
        self.assertEqual(unpriced[0]["note"], "No catalogue product selected")

        summary = out["pricing_summary"]
        self.assertEqual(summary["items_count"], 2)
        self.assertEqual(summary["qty_defaulted_count"], 1)
        self.assertEqual(summary["grand_total"], round(switch["line_total"] + tv["line_total"], 2))
        self.assertIn("Provide GST registration", out["pricing_report"])

    def test_payment_terms_from_state(self, *_mocks) -> None:
        out = pricing_agent_node({"advance_payment": True, "freight_inr": 1000.0,
                                  "matched_items": MATCHED_ITEMS[:1]})
        summary = out["pricing_summary"]
        self.assertTrue(summary["advance_payment"])
        self.assertGreater(summary["payment_discount_value"], 0)
        self.assertEqual(summary["freight"], 1000.0)
        self.assertIn("Advance-payment discount", out["pricing_report"])

    def test_product_removed_from_catalogue(self, *_mocks) -> None:
        item = {**MATCHED_ITEMS[0], "product_id": "no-longer-exists"}
        out = pricing_agent_node({"matched_items": [item]})
        self.assertEqual(out["line_items"], [])
        self.assertEqual(out["quoted_items"][0]["note"],
                         "Selected product is no longer in the catalogue")

    def test_no_matched_items(self, *_mocks) -> None:
        out = pricing_agent_node({"matched_items": []})
        self.assertEqual(out["line_items"], [])
        self.assertEqual(out["pricing_summary"]["grand_total"], 0.0)


if __name__ == "__main__":
    unittest.main()
