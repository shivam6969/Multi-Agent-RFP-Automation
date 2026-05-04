from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from rfp_agent import rfp_agent_node


class _FakeRfpTool:
    def run_query(self, query: str, api_key=None):
        return f"Answer: {query}"

    def build_context(self, query: str, top_k: int = 5):
        return "Requirement: Must be ISO certified."


class RfpAgentTests(unittest.TestCase):
    @patch("rfp_agent.chat_completion")
    @patch("rfp_agent.get_rfp_rag_tool")
    def test_rfp_agent_node_sets_answer_and_requirements(self, mock_get_tool, mock_chat):
        mock_get_tool.return_value = _FakeRfpTool()
        mock_chat.return_value = json.dumps(
            [
                {
                    "requirement": "Must be ISO certified",
                    "criticality": "mandatory",
                    "weight": 4,
                }
            ]
        )

        state = {"user_query": "What are the eligibility criteria?"}
        out = rfp_agent_node(state)

        self.assertIn("rfp_answer", out)
        self.assertEqual(out["rfp_answer"], "Answer: What are the eligibility criteria?")
        self.assertIn("rfp_requirements", out)
        self.assertEqual(len(out["rfp_requirements"]), 1)
        self.assertEqual(out["rfp_requirements"][0]["requirement"], "Must be ISO certified")


if __name__ == "__main__":
    unittest.main()
