from __future__ import annotations

import unittest
from unittest.mock import patch

from pricing_agent import pricing_agent_node


def _fake_chat(*, system: str, user: str, **_kwargs) -> str:
    """Answer each of the pricing agent's LLM calls by which step is asking."""
    if "classify procurement requirements" in system:
        return (
            '[{"requirement": "Requirement A", "priceable": true},'
            ' {"requirement": "Requirement B", "priceable": true}]'
        )
    if "split compound procurement requirements" in system:
        return '["Requirement A", "Requirement B"]'
    if "extract structured catalog fields" in system:
        return '{"sap_code": "AHLX001", "packing": "20/120", "monthly_available": "N/A"}'
    raise AssertionError(f"Unexpected LLM call: {system}")


class PricingAgentNodeTests(unittest.TestCase):
    @patch("pricing_agent.chat_completion", side_effect=_fake_chat)
    @patch("pricing_agent.bu_query")
    def test_pricing_agent_node_builds_report_and_summary(self, mock_bu_query, _mock_chat):
        mock_bu_query.return_value = "INR 100"
        state = {
            "user_query": "Need 2 units of each",
            "matched_items": [
                {"requirement": "Requirement A", "matched": True, "weight": 1},
                {"requirement": "Requirement B", "matched": True, "weight": 1},
            ],
        }

        out = pricing_agent_node(state)

        self.assertIn("pricing_report", out)
        self.assertIn("GRAND TOTAL", out["pricing_report"])
        self.assertIn("pricing_summary", out)
        self.assertEqual(out["pricing_summary"]["items_count"], 2)
        self.assertEqual(len(out["quoted_items"]), 2)
        self.assertEqual(out["line_items"][0]["sap_code"], "AHLX001")


if __name__ == "__main__":
    unittest.main()
