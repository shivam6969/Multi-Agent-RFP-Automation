from __future__ import annotations

import unittest
from unittest.mock import patch

from bu_agent import bu_agent_node, bu_query


class _FakeBuTool:
    def run_query(self, query: str, temperature=0.4, max_tokens=400, api_key=None):
        return f"Catalog answer for: {query}"


class BuAgentTests(unittest.TestCase):
    @patch("bu_agent.get_bu_rag_tool")
    def test_bu_query_calls_tool(self, mock_get_tool):
        mock_get_tool.return_value = _FakeBuTool()
        answer = bu_query("List LED switches")
        self.assertEqual(answer, "Catalog answer for: List LED switches")

    @patch("bu_agent.get_bu_rag_tool")
    def test_bu_agent_node_sets_answer(self, mock_get_tool):
        mock_get_tool.return_value = _FakeBuTool()
        state = {"user_query": "What is available?"}
        out = bu_agent_node(state)
        self.assertEqual(out["bu_answer"], "Catalog answer for: What is available?")


if __name__ == "__main__":
    unittest.main()
