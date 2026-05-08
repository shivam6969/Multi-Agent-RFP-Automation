from __future__ import annotations

import unittest
from unittest.mock import patch

from matching_agent import matching_agent_node


class MatchingAgentTests(unittest.TestCase):
    @patch("matching_agent.bu_query")
    def test_matching_agent_scores_requirements(self, mock_bu_query):
        mock_bu_query.side_effect = [
            "YES\nWe have the product",
            "NO\nNot in catalog",
        ]
        state = {
            "rfp_requirements": [
                {"requirement": "Requirement A", "weight": 2},
                {"requirement": "Requirement B", "weight": 1},
            ]
        }
        out = matching_agent_node(state)

        self.assertEqual(len(out["matched_items"]), 2)
        self.assertEqual(out["matched_items"][0]["matched"], True)
        self.assertEqual(out["matched_items"][1]["matched"], False)
        self.assertAlmostEqual(out["fulfillment_score"], 66.7, places=1)
        self.assertIn("Fulfillment score", out["fulfillment_report"])


if __name__ == "__main__":
    unittest.main()
