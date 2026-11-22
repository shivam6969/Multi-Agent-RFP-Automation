from __future__ import annotations

import json
import re
import unittest
from unittest.mock import patch

from matching_agent import matching_agent_node
from tests.fake_catalogue_search import FakeSearchTool, first_named, product

SWITCH_16A = product("Coral Ebony", "16A 2Way Switch")
SWITCH_25A = first_named("25A 1Way Switch")
FAN_REG = product("Coral Ebony", "1M Fan Regulator")
FAN_REG_ALT = product("Oro Metallica", "1M Fan Regulator")

SEARCH = FakeSearchTool({
    "16A": [(SWITCH_16A, 0.99)],
    "25A": [(SWITCH_25A, 0.98)],
    "regulator": [(FAN_REG, 0.72), (FAN_REG_ALT, 0.70)],
    # "garden hose" → no candidates at all
})

REQUIREMENTS = [
    {"id": "R1", "requirement": "Supply 16A and 25A switches", "criticality": "mandatory",
     "weight": 4, "quantity": 500},
    {"id": "R2", "requirement": "Supply fan regulators, 300 units", "criticality": "high",
     "weight": 3, "quantity": 300},
    {"id": "R3", "requirement": "Provide GST registration", "criticality": "mandatory", "weight": 4},
    {"id": "R4", "requirement": "Supply garden hoses", "criticality": "low", "weight": 1},
]


def fake_llm(*, system: str, user: str, **_):
    if "classify procurement requirements" in system:
        return json.dumps([
            {"index": 0, "type": "product_sku"},
            {"index": 1, "type": "product_sku"},
            {"index": 2, "type": "compliance"},
            {"index": 3, "type": "product_sku"},
        ])
    if "split compound procurement requirements" in system:
        return json.dumps([
            {"index": 0, "items": ["16A switch", "25A switch"]},
            {"index": 1, "items": ["Supply fan regulators, 300 units"]},
            {"index": 2, "items": ["Supply garden hoses"]},
        ])
    if "select catalogue products by id" in system:
        ids = re.findall(r"id=(\w+)", user)
        return json.dumps({"product_id": ids[0], "reason": "closest rating and type"})
    raise AssertionError(f"Unexpected LLM call: {system}")


class CarriedFieldTests(unittest.TestCase):
    def test_source_and_unit_reach_matched_items(self) -> None:
        reqs = [{"id": "R1", "requirement": "Supply 16A and 25A switches", "criticality": "high", "weight": 3,
                 "quantity": 40, "unit": "Nos", "origin": "table",
                 "source": {"page": 2, "table": "T1", "row": "7"}}]

        def llm(*, system, user, **_):
            if "classify" in system:
                return json.dumps([{"index": 0, "type": "product_sku"}])
            if "split" in system:
                return json.dumps([{"index": 0, "items": ["16A switch", "25A switch"]}])
            return fake_llm(system=system, user=user)

        with patch("product_matcher.get_bu_rag_tool", return_value=SEARCH), \
             patch("llm_json.chat_completion", side_effect=llm):
            out = matching_agent_node({"rfp_requirements": reqs})
        for item in out["matched_items"]:
            self.assertEqual(item["source"]["row"], "7")
            self.assertEqual(item["origin"], "table")


class MatchingAgentTests(unittest.TestCase):
    def run_node(self):
        with patch("product_matcher.get_bu_rag_tool", return_value=SEARCH), \
             patch("llm_json.chat_completion", side_effect=fake_llm) as mock_chat:
            out = matching_agent_node({"rfp_requirements": REQUIREMENTS})
        return out, mock_chat

    def test_items_keep_rfp_order_and_split_compounds(self) -> None:
        out, _ = self.run_node()
        ids = [m["id"] for m in out["matched_items"]]
        self.assertEqual(ids, ["R1.1", "R1.2", "R2", "R3", "R4"])
        r1 = out["matched_items"][0]
        self.assertEqual(r1["parent_id"], "R1")
        self.assertEqual(r1["weight"], 2.0)          # parent weight split evenly
        self.assertNotIn("qty", r1)                   # group quantity is ambiguous

    def test_high_confidence_product(self) -> None:
        out, _ = self.run_node()
        m = out["matched_items"][0]
        self.assertEqual(m["status"], "matched")
        self.assertTrue(m["matched"])
        self.assertEqual(m["product_id"], SWITCH_16A.id)
        self.assertEqual(m["sap_code"], SWITCH_16A.sap_code)
        self.assertEqual(m["unit_price"], 191.0)
        self.assertEqual(m["match_score"], 99)
        self.assertEqual(m["confidence"], "high")

    def test_medium_confidence_needs_review_with_alternatives_and_stock(self) -> None:
        out, _ = self.run_node()
        m = next(x for x in out["matched_items"] if x["id"] == "R2")
        self.assertEqual(m["status"], "needs_review")
        self.assertEqual(m["confidence"], "medium")
        self.assertEqual([a["product_id"] for a in m["alternatives"]], [FAN_REG_ALT.id])
        self.assertEqual(m["qty"], 300)
        self.assertEqual(m["stock_sufficient"], FAN_REG.on_hand >= 300)
        self.assertTrue(m["stock_is_demo"])

    def test_compliance_and_gaps(self) -> None:
        out, mock_chat = self.run_node()
        by_id = {m["id"]: m for m in out["matched_items"]}
        self.assertEqual(by_id["R3"]["type"], "compliance")
        self.assertEqual(by_id["R3"]["status"], "needs_review")
        self.assertEqual(by_id["R4"]["status"], "gap")
        self.assertFalse(by_id["R4"]["matched"])
        # The hose had no candidates, so no pick call was made for it:
        # classify + atomise + 3 picks (R1.1, R1.2, R2).
        self.assertEqual(mock_chat.call_count, 5)

    def test_score_and_summary(self) -> None:
        out, _ = self.run_node()
        # Product weights: R1.1 2, R1.2 2, R2 3 matched; R4 1 gap → 7 / 8.
        self.assertEqual(out["fulfillment_score"], 87.5)
        self.assertEqual(out["match_summary"], {
            "fulfillment_score": 87.5, "product_items": 4, "matched": 2, "approved": 0,
            "needs_review": 1, "gaps": 1, "rejected": 0,
            "compliance_pending": 1, "compliance_met": 0, "compliance_not_met": 0,
        })
        self.assertIn("Fulfillment score: 87.5%", out["fulfillment_report"])
        self.assertIn("Provide GST registration", out["fulfillment_report"])

    def test_same_product_twice_is_flagged(self) -> None:
        reqs = [
            {"id": "R1", "requirement": "Supply 16A switches", "weight": 3},
            {"id": "R2", "requirement": "Supply 16A 2Way switches", "weight": 3},
        ]

        def llm(*, system, user, **_):
            if "classify procurement requirements" in system:
                return json.dumps([{"index": 0, "type": "product_sku"}, {"index": 1, "type": "product_sku"}])
            if "split compound procurement requirements" in system:
                return json.dumps([{"index": 0, "items": ["Supply 16A switches"]},
                                   {"index": 1, "items": ["Supply 16A 2Way switches"]}])
            return fake_llm(system=system, user=user)

        with patch("product_matcher.get_bu_rag_tool", return_value=SEARCH), \
             patch("llm_json.chat_completion", side_effect=llm):
            out = matching_agent_node({"rfp_requirements": reqs})
        first, second = out["matched_items"]
        # Equal scores: the earlier item keeps the product.
        self.assertEqual(first["status"], "matched")
        self.assertNotIn("duplicate_of", first)
        self.assertEqual(second["status"], "needs_review")
        self.assertEqual(second["duplicate_of"], "R1")
        self.assertIn("possible duplicate", second["rationale"])
        self.assertIn("possible duplicate of R1", out["fulfillment_report"])

        # A stronger later match keeps it; the vaguer earlier line is flagged.
        scored = FakeSearchTool({"2Way": [(SWITCH_16A, 0.99)], "16A": [(SWITCH_16A, 0.93)]})
        with patch("product_matcher.get_bu_rag_tool", return_value=scored), \
             patch("llm_json.chat_completion", side_effect=llm):
            out = matching_agent_node({"rfp_requirements": reqs})
        first, second = out["matched_items"]
        self.assertEqual(first["duplicate_of"], "R2")
        self.assertNotIn("duplicate_of", second)

    def test_no_requirements(self) -> None:
        out = matching_agent_node({"rfp_requirements": []})
        self.assertEqual(out["matched_items"], [])
        self.assertEqual(out["fulfillment_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
