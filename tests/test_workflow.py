from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pymupdf

from matching_agent import apply_product, fulfillment_report, match_summary
from tests.fake_catalogue_search import product
from workflow import ProjectWorkflow

SWITCH = product("Coral Ebony", "16A 2Way Switch")
SWITCH_ORO = product("Oro Metallica", "16A 2Way Switch")
TV = product("Coral Ebony", "TV Socket")

EXTRACTED = [
    {"id": "R1", "requirement": "Supply 16A 2Way switches", "criticality": "mandatory",
     "weight": 4.0, "quantity": 300},
    {"id": "R2", "requirement": "Supply TV sockets", "criticality": "medium", "weight": 2.0,
     "quantity": None},
    {"id": "R3", "requirement": "Provide GST registration", "criticality": "mandatory",
     "weight": 4.0, "quantity": None},
]


class FakeSteps:
    """Deterministic stand-ins for the LLM-backed steps; pricing stays real."""

    def __init__(self) -> None:
        self.calls = {"extract": 0, "match": 0, "draft": 0}
        self.fail_extract_once = False

    def extract(self, state):
        self.calls["extract"] += 1
        if self.fail_extract_once:
            self.fail_extract_once = False
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        state["rfp_answer"] = "ABC Infrastructure wants modular switches."
        state["rfp_requirements"] = [dict(r) for r in EXTRACTED]
        return state

    def match(self, state):
        self.calls["match"] += 1
        items = []
        for req in state["rfp_requirements"]:
            base = {"id": req["id"], "requirement": req["requirement"], "weight": req["weight"],
                    "criticality": req["criticality"]}
            if req.get("quantity"):
                base["qty"] = req["quantity"]
            text = req["requirement"].lower()
            if "gst" in text:
                items.append({**base, "type": "compliance", "is_product_sku": False,
                              "matched": False, "status": "needs_review"})
            elif "switch" in text:
                items.append({**apply_product({**base, "type": "product"}, SWITCH, score=0.99,
                                              confidence="high", rationale="16A 2-way"),
                              "status": "matched"})
            elif "tv" in text:
                items.append({**apply_product({**base, "type": "product"}, TV, score=0.7,
                                              confidence="medium", rationale="TV socket"),
                              "status": "needs_review"})
            else:
                items.append({**base, "type": "product", "matched": False, "status": "gap",
                              "rationale": "No similar product in the catalogue."})
        summary = match_summary(items)
        state.update(matched_items=items, match_summary=summary,
                     fulfillment_score=summary["fulfillment_score"],
                     fulfillment_report=fulfillment_report(items, summary))
        return state

    def draft(self, state):
        """Stands in for the LLM writer; the PDF render step stays real."""
        self.calls["draft"] += 1
        state["proposal_sections"] = [
            {"key": "executive_summary", "title": "Executive Summary", "ai_generated": True, "edited": False,
             "body": "We are pleased to submit this proposal."},
            {"key": "pricing", "title": "Pricing Schedule", "ai_generated": False, "edited": False,
             "body": "Prices are per unit in Indian Rupees."},
        ]
        state["proposal_meta"] = {"reference": "SYN-TEST", "generated": "2026-10-21", "valid_until": "2027-01-19",
                                  "buyer": "ABC Infrastructure Pvt. Ltd.", "rfp_title": "Modular switches",
                                  "profile": {"name": "Demo Electricals Pvt. Ltd."}}
        return state

    def nodes(self):
        return {"extract": self.extract, "match": self.match, "draft": self.draft}


class WorkflowTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / "checkpoints.db")
        self.steps = FakeSteps()
        self.wf = self.open()

    def open(self) -> ProjectWorkflow:
        wf = ProjectWorkflow(self.path, nodes=self.steps.nodes())
        self.addCleanup(wf.close)
        return wf

    def to_pricing(self, project="p1", matching_decision=None):
        self.wf.start(project, "sample.pdf")
        self.wf.resume(project, {})
        return self.wf.resume(project, matching_decision or {"accept_all": True})


class HappyPathTests(WorkflowTestCase):
    def test_full_flow_pauses_at_each_review(self) -> None:
        status = self.wf.start("p1", "sample.pdf")
        self.assertEqual(status["stage"], "requirements_review")
        self.assertEqual(len(status["state"]["rfp_requirements"]), 3)

        status = self.wf.resume("p1", {"reviewer": "khushi"})
        self.assertEqual(status["stage"], "matching_review")
        self.assertEqual(status["state"]["match_summary"]["needs_review"], 1)

        status = self.wf.resume("p1", {"accept_all": True})
        self.assertEqual(status["stage"], "pricing_review")
        self.assertEqual(len(status["state"]["line_items"]), 2)

        status = self.wf.resume("p1", {"margin_pct": 25})
        self.assertEqual(status["stage"], "proposal_review")
        self.assertEqual(len(status["state"]["proposal_sections"]), 2)

        status = self.wf.resume("p1", {"sections": [
            {"key": "executive_summary", "body": "Edited by the reviewer & approved."}]})
        self.assertEqual(status["stage"], "done")
        self.assertIsNone(status["error"])
        state = status["state"]
        self.assertEqual(state["pricing_summary"]["margin_pct"], 25.0)
        summary = next(s for s in state["proposal_sections"] if s["key"] == "executive_summary")
        self.assertTrue(summary["edited"])

        pdf = Path(state["report_pdf_path"])
        self.assertTrue(pdf.is_file())
        with pymupdf.open(pdf) as doc:
            text = "".join(p.get_text() for p in doc)
        self.assertIn("Edited by the reviewer & approved.", text)
        self.assertIn("ABC Infrastructure Pvt. Ltd.", text)

        self.assertEqual([e["stage"] for e in state["review_log"]],
                         ["requirements_review", "matching_review", "pricing_review", "proposal_review"])
        self.assertEqual(state["review_log"][0]["reviewer"], "khushi")
        # Earlier steps never re-ran on resume.
        self.assertEqual(self.steps.calls, {"extract": 1, "match": 1, "draft": 1})

    def test_save_proposal_draft_keeps_the_pause(self) -> None:
        self.to_pricing()
        self.wf.resume("p1", {})
        status = self.wf.save_proposal_draft("p1", [{"key": "pricing", "body": "Saved draft text."}])
        self.assertEqual(status["stage"], "proposal_review")
        pricing = next(s for s in status["state"]["proposal_sections"] if s["key"] == "pricing")
        self.assertEqual(pricing["body"], "Saved draft text.")
        # Approving afterwards renders the saved text without re-drafting.
        done = self.wf.resume("p1", {})
        self.assertEqual(done["stage"], "done")
        with pymupdf.open(done["state"]["report_pdf_path"]) as doc:
            self.assertIn("Saved draft text.", "".join(p.get_text() for p in doc))
        self.assertEqual(self.steps.calls["draft"], 1)
        with self.assertRaises(ValueError):
            self.wf.save_proposal_draft("p1", [])   # no longer at the proposal review

    def test_unknown_section_is_rejected(self) -> None:
        self.to_pricing()
        self.wf.resume("p1", {})
        with self.assertRaises(ValueError):
            self.wf.resume("p1", {"sections": [{"key": "nope", "body": "x"}]})
        self.assertEqual(self.wf.status("p1")["stage"], "proposal_review")

    def test_survives_a_restart(self) -> None:
        self.wf.start("p1", "sample.pdf")
        self.wf.close()

        reopened = self.open()
        self.assertEqual(reopened.status("p1")["stage"], "requirements_review")
        self.assertEqual(reopened.resume("p1", {})["stage"], "matching_review")
        self.assertEqual(self.steps.calls["extract"], 1)

    def test_projects_are_independent(self) -> None:
        self.wf.start("a", "a.pdf")
        self.wf.start("b", "b.pdf")
        self.wf.resume("a", {})
        self.assertEqual(self.wf.status("a")["stage"], "matching_review")
        self.assertEqual(self.wf.status("b")["stage"], "requirements_review")
        self.assertEqual(self.wf.status("b")["state"]["rfp_pdf_path"], "b.pdf")


class RequirementsReviewTests(WorkflowTestCase):
    def test_edits_reach_matching(self) -> None:
        self.wf.start("p1", "sample.pdf")
        status = self.wf.resume("p1", {"requirements": [
            {"id": "R1", "requirement": "Supply 16A 2Way switches", "criticality": "high",
             "quantity": 450},
            {"requirement": "Supply garden hoses", "criticality": "low"},   # new
            {"id": "R3", "requirement": "Provide GST registration", "criticality": "mandatory"},
        ]})
        reqs = status["state"]["rfp_requirements"]
        self.assertEqual([r["id"] for r in reqs], ["R1", "R4", "R3"])   # R2 removed, new id R4
        self.assertEqual(reqs[0]["weight"], 3.0)
        by_id = {m["id"]: m for m in status["state"]["matched_items"]}
        self.assertEqual(by_id["R1"]["qty"], 450)
        self.assertEqual(by_id["R4"]["status"], "gap")
        self.assertNotIn("R2", by_id)
        self.assertEqual(status["state"]["review_log"][0]["changes"],
                         {"edited": True, "before": 3, "after": 3, "added": 1, "removed": 1})

    def test_invalid_edit_is_rejected_and_nothing_moves(self) -> None:
        self.wf.start("p1", "sample.pdf")
        with self.assertRaises(ValueError):
            self.wf.resume("p1", {"requirements": [{"requirement": "x"}]})  # too short
        self.assertEqual(self.wf.status("p1")["stage"], "requirements_review")


class MatchingReviewTests(WorkflowTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.wf.start("p1", "sample.pdf")
        self.wf.resume("p1", {})

    def test_reject_change_quantity_and_compliance(self) -> None:
        status = self.wf.resume("p1", {"decisions": [
            {"id": "R1", "action": "change", "product_id": SWITCH_ORO.id, "quantity": 120},
            {"id": "R2", "action": "reject"},
            {"id": "R3", "action": "confirm"},
        ]})
        self.assertEqual(status["stage"], "pricing_review")
        by_id = {m["id"]: m for m in status["state"]["matched_items"]}
        self.assertEqual(by_id["R1"]["product_id"], SWITCH_ORO.id)
        self.assertEqual(by_id["R1"]["confidence"], "manual")
        self.assertEqual(by_id["R1"]["status"], "approved")
        self.assertEqual(by_id["R2"]["status"], "rejected")
        self.assertEqual(by_id["R3"]["status"], "met")

        lines = status["state"]["line_items"]
        self.assertEqual(len(lines), 1)                         # rejected TV not priced
        self.assertEqual(lines[0]["base_unit_price"], SWITCH_ORO.unit_price)
        self.assertEqual((lines[0]["qty"], lines[0]["qty_source"]), (120, "reviewer"))
        summary = status["state"]["match_summary"]
        self.assertEqual((summary["rejected"], summary["compliance_met"]), (1, 1))

    def test_bad_decisions_are_rejected_before_the_graph_moves(self) -> None:
        bad = [
            {"decisions": [{"id": "R9", "action": "accept"}]},                      # unknown id
            {"decisions": [{"id": "R3", "action": "accept"}]},                      # compliance
            {"decisions": [{"id": "R1", "action": "confirm"}]},                     # product
            {"decisions": [{"id": "R1", "action": "change"}]},                      # no product_id
            {"decisions": [{"id": "R1", "action": "change", "product_id": "nope"}]},
            {"decisions": [{"id": "R1", "action": "accept", "quantity": 0}]},
        ]
        for decision in bad:
            with self.subTest(decision=decision), self.assertRaises(ValueError):
                self.wf.resume("p1", decision)
        self.assertEqual(self.wf.status("p1")["stage"], "matching_review")
        self.assertEqual(self.steps.calls["match"], 1)


class PricingReviewTests(WorkflowTestCase):
    def test_margin_terms_and_quantities_reprice(self) -> None:
        before = self.to_pricing()["state"]["pricing_summary"]["grand_total"]
        status = self.wf.resume("p1", {"margin_pct": 30, "advance_payment": True,
                                       "freight_inr": 500, "quantities": {"R2": 40}})
        state = status["state"]
        self.assertEqual(status["stage"], "proposal_review")
        self.assertEqual(state["pricing_summary"]["margin_pct"], 30.0)
        self.assertTrue(state["pricing_summary"]["advance_payment"])
        self.assertEqual(state["pricing_summary"]["freight"], 500.0)
        tv = next(li for li in state["line_items"] if li["requirement_id"] == "R2")
        self.assertEqual((tv["qty"], tv["qty_source"]), (40, "reviewer"))
        self.assertNotEqual(state["pricing_summary"]["grand_total"], before)

    def test_unknown_item_quantity(self) -> None:
        self.to_pricing()
        with self.assertRaises(ValueError):
            self.wf.resume("p1", {"quantities": {"R3": 10}})   # compliance item
        self.assertEqual(self.wf.status("p1")["stage"], "pricing_review")


class ErrorHandlingTests(WorkflowTestCase):
    def test_failed_step_can_be_retried(self) -> None:
        self.steps.fail_extract_once = True
        status = self.wf.start("p1", "sample.pdf")
        self.assertEqual(status["stage"], "failed")
        self.assertEqual(status["failed_step"], "extract")
        self.assertIn("OPENROUTER_API_KEY", status["error"])

        self.assertEqual(self.wf.retry("p1")["stage"], "requirements_review")

    def test_guards(self) -> None:
        with self.assertRaises(KeyError):
            self.wf.status("nope")
        self.wf.start("p1", "sample.pdf")
        with self.assertRaises(ValueError):
            self.wf.start("p1", "sample.pdf")         # duplicate id
        with self.assertRaises(ValueError):
            self.wf.retry("p1")                       # nothing failed
        self.wf.resume("p1", {})
        self.wf.resume("p1", {"accept_all": True})
        self.wf.resume("p1", {})
        self.assertEqual(self.wf.resume("p1", {})["stage"], "done")
        with self.assertRaises(ValueError):
            self.wf.resume("p1", {})                  # done: nothing to review


if __name__ == "__main__":
    unittest.main()
