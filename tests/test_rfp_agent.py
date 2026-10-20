from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from rfp_agent import _windows, rfp_agent_node


class _FakeRfpTool:
    chunks = [
        "1. Scope: supply 16A 2Way switches, quantity 500 units.",
        "2. Eligibility: vendor must be ISO certified.",
        "3. Delivery: within 14 days. Vendor must be ISO certified.",
    ]

    def __init__(self) -> None:
        self.top_k = None

    def run_query(self, query: str, top_k=3, api_key=None):
        self.top_k = top_k
        return f"Answer: {query}"


WINDOW_REPLIES = {
    "16A 2Way": [{"requirement": "Supply 16A 2Way switches", "criticality": "MANDATORY", "quantity": "500"}],
    "Eligibility": [{"requirement": "Must be ISO certified", "criticality": "mandatory", "quantity": None}],
    "Delivery": [
        {"requirement": "Deliver within 14 days", "criticality": "urgent!"},   # unknown → medium
        {"requirement": "Must be ISO-certified.", "criticality": "high"},       # duplicate
        {"requirement": "ok", "criticality": "low"},                            # too short
    ],
}


def fake_llm(*, system: str, user: str, **_):
    for marker, reply in WINDOW_REPLIES.items():
        if marker in user:
            return json.dumps(reply)
    raise AssertionError("unexpected window")


class RfpAgentTests(unittest.TestCase):
    def run_node(self, llm=fake_llm, window=60):
        self.tool = _FakeRfpTool()
        with patch("rfp_agent.get_rfp_rag_tool", return_value=self.tool), \
             patch("rfp_agent.RFP_EXTRACT_WINDOW_CHARS", window), \
             patch("llm_json.chat_completion", side_effect=llm):
            return rfp_agent_node({"user_query": "What are the eligibility criteria?"})

    def test_short_rfp_is_answered_from_the_whole_document(self) -> None:
        self.run_node(window=10_000)
        self.assertEqual(self.tool.top_k, len(_FakeRfpTool.chunks))

    def test_long_rfp_is_answered_from_the_nearest_chunks(self) -> None:
        self.run_node(window=60)
        self.assertEqual(self.tool.top_k, 3)

    def test_reads_every_window_and_cleans_results(self) -> None:
        out = self.run_node()
        self.assertEqual(out["rfp_answer"], "Answer: What are the eligibility criteria?")
        self.assertEqual(out["rfp_requirements"], [
            {"id": "R1", "requirement": "Supply 16A 2Way switches", "criticality": "mandatory",
             "weight": 4.0, "quantity": 500},
            {"id": "R2", "requirement": "Must be ISO certified", "criticality": "mandatory",
             "weight": 4.0, "quantity": None},
            {"id": "R3", "requirement": "Deliver within 14 days", "criticality": "medium",
             "weight": 2.0, "quantity": None},
        ])
        self.assertNotIn("warnings", out)

    def test_failed_window_becomes_a_warning(self) -> None:
        def flaky(*, system, user, **kw):
            return "not json" if "Delivery" in user else fake_llm(system=system, user=user)

        out = self.run_node(flaky)
        self.assertEqual(len(out["rfp_requirements"]), 2)
        self.assertEqual(len(out["warnings"]), 1)
        self.assertIn("part 3 of 3", out["warnings"][0])


class WindowTests(unittest.TestCase):
    def test_groups_chunks_up_to_the_limit(self) -> None:
        self.assertEqual(_windows(["aaaa", "bbbb", "cc"], 9), ["aaaa\n\nbbbb", "cc"])
        self.assertEqual(_windows(["x" * 20], 5), ["x" * 20])  # never splits a chunk
        self.assertEqual(_windows([], 5), [])


if __name__ == "__main__":
    unittest.main()
