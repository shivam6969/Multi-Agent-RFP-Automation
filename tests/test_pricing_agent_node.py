from __future__ import annotations

import json
import re
import unittest
from unittest.mock import patch

from catalogue import get_catalogue
from pricing_agent import pricing_agent_node


def _product(range_name: str, item_name: str):
    for p in get_catalogue().all_products():
        if (p.range_name, p.item_name) == (range_name, item_name):
            return p
    raise KeyError(item_name)


SWITCH = _product("Coral Ebony", "16A 2Way Switch")
SOCKET = _product("Coral Ebony", "TV Socket")


def _fake_chat(*, system: str, user: str, **_kwargs) -> str:
    """Answer each of the pricing agent's LLM calls by which step is asking."""
    if "classify procurement requirements" in system:
        return json.dumps([
            {"requirement": "Supply 16A 2Way switches, 300 units", "priceable": True},
            {"requirement": "Supply TV sockets", "priceable": True},
            {"requirement": "Supply garden hoses", "priceable": True},
            {"requirement": "Provide GST registration", "priceable": False},
        ])
    if "split compound procurement requirements" in system:
        return json.dumps([
            "Supply 16A 2Way switches, 300 units",
            "Supply TV sockets",
            "Supply garden hoses",
        ])
    if "select catalogue products by id" in system:
        item = re.search(r'RFP item: "(.*)"', user).group(1)
        if "switch" in item:
            return json.dumps({"product_id": SWITCH.id, "reason": "16A 2-way"})
        if "TV" in item:
            # An id that was NOT in the candidate list must be rejected.
            return json.dumps({"product_id": "invented-id", "reason": "made up"})
        return json.dumps({"product_id": None, "reason": "not a catalogue product"})
    raise AssertionError(f"Unexpected LLM call: {system}")


def _fake_candidates(query: str, top_k: int = 8):
    if "switch" in query:
        return [SWITCH, SOCKET]
    if "TV" in query:
        return [SOCKET]
    return []


class PricingAgentNodeTests(unittest.TestCase):
    @patch("pricing_agent._candidate_products", side_effect=_fake_candidates)
    @patch("pricing_agent.chat_completion", side_effect=_fake_chat)
    def test_prices_come_from_catalogue_and_picks_are_validated(self, _chat, _cands):
        state = {
            "user_query": "Prepare a quote",
            "margin_pct": 20.0,
            "matched_items": [
                {"requirement": "Supply 16A 2Way switches, 300 units", "matched": True, "weight": 3},
                {"requirement": "Supply TV sockets", "matched": True, "weight": 2},
                {"requirement": "Supply garden hoses", "matched": True, "weight": 1},
                {"requirement": "Provide GST registration", "matched": True, "weight": 4},
                {"requirement": "Unmatched thing", "matched": False, "weight": 1},
            ],
        }

        out = pricing_agent_node(state)

        # Only the switch is priced: the TV pick was invented, hoses have no product.
        self.assertEqual(len(out["quoted_items"]), 3)
        self.assertEqual(len(out["line_items"]), 1)
        line = out["line_items"][0]
        self.assertEqual(line["product_id"], SWITCH.id)
        self.assertEqual(line["sap_code"], SWITCH.sap_code)
        self.assertEqual(line["base_unit_price"], 191.0)
        self.assertEqual(line["qty"], 300)
        self.assertEqual(line["qty_source"], "requirement")
        self.assertEqual(line["final_unit_price"], 256.93)
        self.assertTrue(line["stock_is_demo"])

        unpriced = [q for q in out["quoted_items"] if q.get("catalog_cost_inr") is None]
        self.assertEqual(len(unpriced), 2)
        self.assertTrue(all(q["note"] == "No matching catalogue product" for q in unpriced))

        summary = out["pricing_summary"]
        self.assertEqual(summary["items_count"], 1)
        self.assertEqual(summary["grand_total"], round(256.93 * 300, 2))
        self.assertIn("GRAND TOTAL", out["pricing_report"])
        self.assertIn("Provide GST registration", out["pricing_report"])

    @patch("pricing_agent._candidate_products", side_effect=_fake_candidates)
    @patch("pricing_agent.chat_completion", side_effect=_fake_chat)
    def test_payment_terms_from_state(self, _chat, _cands):
        state = {
            "advance_payment": True,
            "freight_inr": 1000.0,
            "matched_items": [
                {"requirement": "Supply 16A 2Way switches, 300 units", "matched": True, "weight": 3},
            ],
        }
        out = pricing_agent_node(state)
        summary = out["pricing_summary"]
        self.assertTrue(summary["advance_payment"])
        self.assertGreater(summary["payment_discount_value"], 0)
        self.assertEqual(summary["freight"], 1000.0)
        self.assertIn("Advance-payment discount", out["pricing_report"])

    def test_no_matched_items(self):
        out = pricing_agent_node({"matched_items": []})
        self.assertEqual(out["line_items"], [])
        self.assertEqual(out["pricing_summary"]["grand_total"], 0.0)


if __name__ == "__main__":
    unittest.main()
