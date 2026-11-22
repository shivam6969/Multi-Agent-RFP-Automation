from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import shutil
import tempfile
from pathlib import Path

import rfp_rag
from rfp_agent import _is_bare_address, _is_scope_summary, _map_columns, _needs_column_mapping, _windows, rfp_agent_node
from rfp_tables import TableData, analyse_table, boq_requirements
from tests.fakes import FakeEmbedder, FakeReranker
from tests.sample_rfps import build_all


class _FakeRfpTool:
    chunks = [
        "1. Scope: supply 16A 2Way switches, quantity 500 units.",
        "2. Eligibility: vendor must be ISO certified.",
        "3. Delivery: within 14 days. Vendor must be ISO certified.",
    ]

    def __init__(self) -> None:
        self.whole_document = None

    def run_query(self, query: str, top_k=3, api_key=None, whole_document=False):
        self.whole_document = whole_document
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
        self.assertTrue(self.tool.whole_document)

    def test_long_rfp_is_answered_from_the_nearest_chunks(self) -> None:
        self.run_node(window=60)
        self.assertFalse(self.tool.whole_document)

    def test_reads_every_window_and_cleans_results(self) -> None:
        out = self.run_node()
        self.assertEqual(out["rfp_answer"], "Answer: What are the eligibility criteria?")
        self.assertEqual(out["rfp_requirements"], [
            {"id": "R1", "requirement": "Supply 16A 2Way switches", "criticality": "mandatory",
             "weight": 4.0, "quantity": 500, "origin": "text"},
            {"id": "R2", "requirement": "Must be ISO certified", "criticality": "mandatory",
             "weight": 4.0, "quantity": None, "origin": "text"},
            {"id": "R3", "requirement": "Deliver within 14 days", "criticality": "medium",
             "weight": 2.0, "quantity": None, "origin": "text"},
        ])
        self.assertNotIn("warnings", out)

    def test_failed_window_becomes_a_warning(self) -> None:
        def flaky(*, system, user, **kw):
            return "not json" if "Delivery" in user else fake_llm(system=system, user=user)

        out = self.run_node(flaky)
        self.assertEqual(len(out["rfp_requirements"]), 2)
        self.assertEqual(len(out["warnings"]), 1)
        self.assertIn("part 3 of 3", out["warnings"][0])


class TableExtractionTests(unittest.TestCase):
    """The real parser + index on a sample RFP; only the models and the LLM are faked."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="rfp-agent-"))
        cls.samples = build_all(cls.tmp)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_sample(self, name: str, llm):
        path, truth = self.samples[name]
        prompts = []

        def recording(*, system, user, **kw):
            prompts.append(user)
            return llm(system=system, user=user, **kw)

        with patch("rfp_rag.get_embedder", return_value=FakeEmbedder()), \
             patch("rfp_rag.get_reranker", return_value=FakeReranker()), \
             patch("embedding_cache.EMBEDDING_CACHE_DIR", str(self.tmp / "cache")), \
             patch("rfp_rag.chat_completion", return_value="Answer"), \
             patch("llm_json.chat_completion", side_effect=recording):
            rfp_rag.clear_rfp_tools()
            try:
                out = rfp_agent_node({"user_query": "Summarise", "rfp_pdf_path": str(path)})
            finally:
                rfp_rag.clear_rfp_tools()
        return out, truth, prompts

    def test_item_rows_become_requirements_without_the_llm(self) -> None:
        def llm(*, system, user, **_):
            self.assertNotIn("Electronic key card", user)        # item table is only a placeholder
            self.assertIn("[Table T2", user)
            self.assertIn("| Parameter | Requirement |", user)  # spec table as Markdown
            return json.dumps([
                {"requirement": "Supply USB charger, 2.1A output", "criticality": "high"},  # re-extraction
                {"requirement": "Valid GST registration", "criticality": "mandatory"},
                {"requirement": "Warranty of 2 years from delivery", "criticality": "medium"},
            ])

        out, truth, prompts = self.run_sample("hotel", llm)
        self.assertEqual(len(prompts), 1)
        reqs = out["rfp_requirements"]
        table_reqs = [r for r in reqs if r["origin"] == "table"]
        self.assertEqual([r["quantity"] for r in table_reqs], [i.quantity for i in truth.items])
        self.assertEqual(reqs[0]["id"], "R1")
        self.assertEqual(reqs[0]["source"]["table"], "T2")
        text_reqs = [r["requirement"] for r in reqs if r["origin"] == "text"]
        self.assertEqual(text_reqs, ["Valid GST registration", "Warranty of 2 years from delivery"])
        self.assertEqual([t["kind"] for t in out["rfp_tables"]], ["kv", "boq", "spec", "other"])
        self.assertTrue(any("'Box'" in w for w in out["warnings"]))

    def test_school_total_mismatch_is_a_warning(self) -> None:
        out, _truth, _ = self.run_sample("school", lambda **_: "[]")
        self.assertTrue(any("add up to 180" in w for w in out["warnings"]))
        self.assertEqual(out["rfp_requirements"][0]["quantity"], 1060)

    def test_unrecognised_columns_are_mapped_by_one_llm_call(self) -> None:
        table = TableData(
            id="T9", header=["Sr", "Stores", "A/U Code", "Reqd"],
            rows=[["1", "Switch 6A one way", "Nos", "40"], ["2", "Socket 6A", "Nos", "25"]],
            pages=[1], row_pages=[1, 1], roles=["other"] * 4, kind="other")   # as if the rules had failed
        self.assertTrue(_needs_column_mapping(table))
        reply = json.dumps({"is_item_list": True, "description": 1, "quantity": 3, "unit": 2})
        with patch("llm_json.chat_completion", return_value=reply):
            self.assertTrue(_map_columns(table, None))
        self.assertEqual(table.kind, "boq")
        self.assertEqual([r.quantity for r in boq_requirements(table)], [40, 25])

    def test_bad_column_mapping_is_ignored(self) -> None:
        table = TableData(id="T9", header=["A", "B", "C"], rows=[["x", "1", "2"], ["y", "3", "4"]],
                          roles=["other"] * 3, kind="other")
        reply = json.dumps({"is_item_list": True, "description": 7, "quantity": 1})
        with patch("llm_json.chat_completion", return_value=reply):
            self.assertFalse(_map_columns(table, None))
        self.assertEqual(table.kind, "other")


class ScopeSummaryTests(unittest.TestCase):
    TABLE_WORDS = {"6a", "1", "way", "switch", "3", "pin", "socket", "fan", "regulator", "2m", "cover", "plate"}

    def test_scope_lines_seen_in_live_runs_are_dropped(self) -> None:
        for text in ("Supply modular switches, sockets, support modules and cover plates",
                     "Supply electrical wiring accessories"):
            self.assertTrue(_is_scope_summary(text, None, self.TABLE_WORDS), text)

    def test_real_requirements_are_kept(self) -> None:
        for text, qty in (("Supply ISI marked products only", None),
                          ("Supply 16A 2Way switches", None),
                          ("Supply fan regulators", 300),
                          ("Quote for every item listed in the schedule", None),
                          ("Supply garden hoses", None),            # nothing to do with the table
                          ("Provide GST registration certificate", None)):
            self.assertFalse(_is_scope_summary(text, qty, self.TABLE_WORDS), text)


class AddressLineTests(unittest.TestCase):
    def test_addresses_and_contacts_are_not_requirements(self) -> None:
        for text in ("140124,Bhakra Beas Management Board, Irrigation Wing, NANGAL TOWNSHIP",
                     "Office of Dy CMM General Stores, Lucknow 226005",
                     "Lavdeep Kumar, lavdeep.kumar@bbmb.example",
                     "Contact: 9876543210"):
            self.assertTrue(_is_bare_address(text), text)

    def test_obligations_and_specs_are_kept(self) -> None:
        for text in ("Deliver to Bhakra Beas Management Board, Nangal 140124 within 15 days",
                     "Switch mechanism with minimum 1,00,000 operations",
                     "Rated Wattage (in Watts) should be 40.0",
                     "Supply of LED Batten Light (100 pieces)"):
            self.assertFalse(_is_bare_address(text), text)


class WindowTests(unittest.TestCase):
    def test_groups_chunks_up_to_the_limit(self) -> None:
        self.assertEqual(_windows(["aaaa", "bbbb", "cc"], 9), ["aaaa\n\nbbbb", "cc"])
        self.assertEqual(_windows(["x" * 20], 5), ["x" * 20])  # never splits a chunk
        self.assertEqual(_windows([], 5), [])


if __name__ == "__main__":
    unittest.main()
