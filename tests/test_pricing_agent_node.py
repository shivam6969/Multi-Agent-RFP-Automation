from __future__ import annotations

import unittest
from unittest.mock import patch

from pricing_agent import pricing_agent_node


class PricingAgentNodeTests(unittest.TestCase):
    @patch("pricing_agent.bu_query")
    def test_pricing_agent_node_builds_report_and_summary(self, mock_bu_query):
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


if __name__ == "__main__":
    unittest.main()
