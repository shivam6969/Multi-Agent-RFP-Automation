from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

from company_profile import CompanyProfile, get_profile, save_profile
from db import get_engine
from llm_json import LLMOutputError
from pricing_agent import reprice_with_margin
from proposal_pdf import inr, not_quoted_items, render_proposal
from proposal_writer import AI_SECTIONS, SECTIONS, amount_warnings, draft_proposal, regenerate_section
from report_agent import report_agent_node
from settings import PRICING_RULES
from tests.fake_catalogue_search import product

SWITCH = product("Coral Ebony", "16A 2Way Switch")
LINES, SUMMARY = reprice_with_margin(
    [{"requirement": "16A 2Way switches", "requirement_id": "R1", "qty": 300, "base_unit_price": SWITCH.unit_price,
      "product_name": SWITCH.item_name, "product_range": SWITCH.range_name, "sap_code": SWITCH.sap_code}],
    20.0, PRICING_RULES, advance_payment=True, freight=500,
)

STATE = {
    "rfp_meta": {"title": "Modular switches for Hospital Wing", "buyer": "Medicare Plus Hospitals Pvt. Ltd.",
                 "reference": "SMP/2026/1234", "deadline": "2026-11-20", "location": "Gurugram"},
    "rfp_answer": "Buyer: Medicare Plus Hospitals. Scope: modular switches.",
    "matched_items": [
        {"id": "R1", "requirement": "16A 2Way switches", "type": "product", "matched": True, "status": "approved"},
        {"id": "R2", "requirement": "18W LED panel light", "type": "product", "matched": False, "status": "gap"},
        {"id": "R3", "requirement": "Provide GST registration", "type": "compliance", "status": "met"},
        {"id": "R4", "requirement": "ISO 9001 certificate", "type": "compliance", "status": "needs_review"},
    ],
    "line_items": LINES,
    "pricing_summary": SUMMARY,
    "quoted_items": [],
}


def draft_reply(**override) -> str:
    sections = {k: f"Text for {k}.\n- a bullet point" for k in AI_SECTIONS}
    sections.update(override)
    return json.dumps({"sections": sections})


class WriterTests(unittest.TestCase):
    def test_draft_has_every_section_in_order(self) -> None:
        with patch("llm_json.chat_completion", return_value=draft_reply()) as llm:
            sections, meta = draft_proposal(STATE, CompanyProfile(name="Sharma Electricals"), "SYN-ABC123")
        self.assertEqual([s["key"] for s in sections], [k for k, *_ in SECTIONS])
        by_key = {s["key"]: s for s in sections}
        self.assertFalse(by_key["company"]["ai_generated"])
        self.assertIn("GSTIN", by_key["company"]["body"])
        self.assertIn("advance-payment discount", by_key["pricing"]["body"])
        self.assertEqual((meta["reference"], meta["buyer"]), ("SYN-ABC123", "Medicare Plus Hospitals Pvt. Ltd."))
        self.assertEqual(meta["profile"]["name"], "Sharma Electricals")

        prompt = llm.call_args.kwargs["user"]
        self.assertIn("NEVER write prices", prompt)
        self.assertIn("- 18W LED panel light", prompt)                 # not quoted
        confirmed = prompt.split("CONFIRMED as met by our reviewer:")[1].split("Other compliance")[0]
        self.assertIn("Provide GST registration", confirmed)
        self.assertNotIn("ISO 9001", confirmed)                           # only reviewer-confirmed claims

    def test_missing_sections_retry_then_fail(self) -> None:
        bad = json.dumps({"sections": {"executive_summary": "only one"}})
        with patch("llm_json.chat_completion", return_value=bad) as llm:
            with self.assertRaises(LLMOutputError):
                draft_proposal(STATE, CompanyProfile(), "SYN-X")
        self.assertEqual(llm.call_count, 2)

    def test_gemini_is_used_when_configured(self) -> None:
        with patch("proposal_writer.gemini_key", return_value="k"), \
             patch("llm_json.gemini_completion", return_value=draft_reply()) as gem, \
             patch("llm_json.chat_completion") as openrouter:
            draft_proposal(STATE, CompanyProfile(), "SYN-X")
        gem.assert_called_once()
        openrouter.assert_not_called()

    def test_regenerate_one_section(self) -> None:
        state = {**STATE, "proposal_sections": [{"key": "delivery", "title": "Delivery", "body": "Old text."}],
                 "proposal_meta": {"profile": CompanyProfile().model_dump()}}
        with patch("llm_json.chat_completion", return_value=json.dumps({"text": "New delivery text for the buyer."})) as llm:
            text = regenerate_section(state, "delivery", "mention site delivery")
        self.assertEqual(text, "New delivery text for the buyer.")
        self.assertIn("Old text.", llm.call_args.kwargs["user"])
        self.assertIn("mention site delivery", llm.call_args.kwargs["user"])
        with self.assertRaises(ValueError):
            regenerate_section(state, "pricing")          # deterministic section

    def test_amount_warnings(self) -> None:
        flagged = amount_warnings([
            {"key": "a", "body": "Total of ₹ 1,20,000 payable"}, {"key": "b", "body": "Rs.60 per piece"},
            {"key": "c", "body": "A 12 lakh order"}, {"key": "d", "body": "INR 5000"},
            {"key": "ok", "body": "16A switches to IS 3854, delivered in 7 days, 2 year warranty, 100% inspection"},
        ])
        self.assertEqual(set(flagged), {"a", "b", "c", "d"})


class PdfTests(unittest.TestCase):
    def test_inr_formatting(self) -> None:
        self.assertEqual(inr(2367729), "23,67,729.00")
        self.assertEqual(inr(1000), "1,000.00")
        self.assertEqual(inr(-283.2), "-283.20")

    def test_render_uses_pricing_figures_and_escapes_text(self) -> None:
        sections = [{"key": "executive_summary", "title": "Executive Summary",
                     "body": "Tom & Jerry <script> wrote this.\n- first point"},
                    {"key": "pricing", "title": "Pricing Schedule", "body": "Prices per unit."}]
        meta = {"reference": "SYN-ABC123", "generated": "2026-10-21", "valid_until": "2027-01-19",
                "buyer": "Medicare Plus Hospitals Pvt. Ltd.", "rfp_title": "Modular switches",
                "profile": CompanyProfile(name="Sharma Electricals").model_dump()}
        with tempfile.TemporaryDirectory() as tmp:
            path = render_proposal(str(Path(tmp) / "p.pdf"), sections=sections, proposal_meta=meta,
                                   line_items=LINES, pricing_summary=SUMMARY, not_quoted=["18W LED panel light"])
            with pymupdf.open(path) as doc:
                text = "".join(p.get_text() for p in doc)
        for expected in ("Tom & Jerry <script> wrote this.", "Sharma Electricals", "Medicare Plus Hospitals",
                         inr(SUMMARY["grand_total"]), inr(SUMMARY["payment_discount_value"]).lstrip("-"),
                         inr(LINES[0]["final_unit_price"]), "Not quoted: 18W LED panel light", SWITCH.sap_code):
            self.assertIn(expected, text)

    def test_not_quoted_items(self) -> None:
        state = {"matched_items": STATE["matched_items"],
                 "quoted_items": [{"requirement": "Odd item", "catalog_cost_inr": None}]}
        self.assertEqual(not_quoted_items(state), ["18W LED panel light", "Odd item"])


class ReportAgentNodeTests(unittest.TestCase):
    def test_node_drafts_and_renders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch("llm_json.chat_completion", return_value=draft_reply()):
            out = report_agent_node(dict(STATE), output_dir=tmp)
            self.assertTrue(Path(out["report_pdf_path"]).is_file())
        self.assertIn("Executive Summary", out["report_text"])
        self.assertEqual(len(out["proposal_sections"]), len(SECTIONS))

    def test_node_reports_llm_failure(self) -> None:
        out = report_agent_node(dict(STATE), output_dir=tempfile.gettempdir())   # no key in tests
        self.assertIn("Proposal text could not be generated", out["error"])
        self.assertNotIn("report_pdf_path", out)


class CompanyProfileTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = get_engine(str(Path(tmp) / "p.db"))
            self.assertTrue(get_profile(engine).is_demo)
            save_profile(CompanyProfile(name="Sharma Electricals", is_demo=False), engine)
            self.assertEqual(get_profile(engine).name, "Sharma Electricals")
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
