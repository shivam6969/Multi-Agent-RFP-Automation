from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from api.main import create_app
from api.security import hash_password, verify_password
from db import Project, get_engine
from tests.test_workflow import SWITCH_ORO, FakeSteps
from workflow import ProjectWorkflow

RFP_TEXT = (
    "Request for Proposal - Supply of Modular Switches\n"
    "1. Introduction\nABC Infrastructure Pvt. Ltd. invites proposals for modular switches, "
    "sockets and plates for residential projects in Gurugram.\n"
)


def make_pdf(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    if text:
        page.insert_textbox(pymupdf.Rect(40, 40, 560, 800), text, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


class SecurityTests(unittest.TestCase):
    def test_password_hash_round_trip(self) -> None:
        stored = hash_password("correct horse")
        self.assertTrue(stored.startswith("scrypt$"))
        self.assertNotIn("correct horse", stored)
        self.assertTrue(verify_password("correct horse", stored))
        self.assertFalse(verify_password("wrong horse", stored))
        self.assertFalse(verify_password("x", "garbage"))


class ApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.steps = FakeSteps()
        self.workflow = ProjectWorkflow(str(self.tmp / "checkpoints.db"), nodes=self.steps.nodes())
        self.addCleanup(self.workflow.close)
        self.db_path = str(self.tmp / "app.db")
        app = create_app(db_path=self.db_path, workflow=self.workflow, var_dir=self.tmp,
                         run_inline=True, warm_up=False)
        self.client = TestClient(app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.addCleanup(get_engine(self.db_path).dispose)

    def login(self, email="khushi@example.com") -> None:
        self.client.post("/api/auth/register",
                         json={"email": email, "name": "Khushi", "password": "s3cret-pass"})
        r = self.client.post("/api/auth/login", json={"email": email, "password": "s3cret-pass"})
        self.assertEqual(r.status_code, 200)

    def upload(self, text: str = RFP_TEXT, name: str = "rfp.pdf", data: bytes = None):
        return self.client.post("/api/rfps/upload",
                                files={"file": (name, data or make_pdf(text), "application/pdf")})


class AuthTests(ApiTestCase):
    def test_register_login_me_logout(self) -> None:
        r = self.client.post("/api/auth/register",
                             json={"email": "Khushi@Example.com", "name": "Khushi", "password": "s3cret-pass"})
        self.assertEqual(r.status_code, 201)
        self.assertEqual(r.json()["email"], "khushi@example.com")

        dup = self.client.post("/api/auth/register",
                               json={"email": "khushi@example.com", "name": "K", "password": "another-pass"})
        self.assertEqual(dup.status_code, 409)

        bad = self.client.post("/api/auth/login", json={"email": "khushi@example.com", "password": "nope-nope"})
        self.assertEqual(bad.status_code, 401)

        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)
        ok = self.client.post("/api/auth/login",
                              json={"email": "khushi@example.com", "password": "s3cret-pass", "remember": True})
        self.assertEqual(ok.status_code, 200)
        self.assertIn("synapse_session", ok.cookies)
        self.assertEqual(self.client.get("/api/auth/me").json()["name"], "Khushi")

        self.assertEqual(self.client.post("/api/auth/logout").status_code, 204)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)

    def test_validation_and_protection(self) -> None:
        short = self.client.post("/api/auth/register",
                                 json={"email": "a@b.co", "name": "A", "password": "short"})
        self.assertEqual(short.status_code, 422)
        for path in ("/api/rfps", "/api/projects", "/api/catalogue/products", "/api/catalogue/stock"):
            self.assertEqual(self.client.get(path).status_code, 401, path)

    def test_health_never_reveals_keys(self) -> None:
        body = self.client.get("/api/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(set(body), {"status", "llm_key_configured", "gemini_key_configured"})


class RfpTests(ApiTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.login()

    def test_upload_list_get_and_document(self) -> None:
        pdf = make_pdf(RFP_TEXT)   # PDFs embed a timestamp: reuse the exact bytes
        r = self.upload(data=pdf)
        self.assertEqual(r.status_code, 201)
        rfp = r.json()
        self.assertEqual(rfp["title"], "Request for Proposal - Supply of Modular Switches")
        self.assertEqual((rfp["source"], rfp["pages"]), ("upload", 1))

        again = self.upload(name="copy.pdf", data=pdf)
        self.assertEqual(again.status_code, 200)
        self.assertEqual(again.json()["id"], rfp["id"])

        self.assertEqual([x["id"] for x in self.client.get("/api/rfps").json()], [rfp["id"]])
        self.assertEqual(len(self.client.get("/api/rfps?q=modular").json()), 1)
        self.assertEqual(len(self.client.get("/api/rfps?q=transformer").json()), 0)
        doc = self.client.get(f"/api/rfps/{rfp['id']}/document")
        self.assertEqual(doc.headers["content-type"], "application/pdf")
        self.assertEqual(self.client.get("/api/rfps/nope").status_code, 404)

    def test_rejects_non_pdf_and_scans(self) -> None:
        not_pdf = self.client.post("/api/rfps/upload", files={"file": ("x.pdf", b"hello", "application/pdf")})
        self.assertEqual(not_pdf.status_code, 422)
        scan = self.upload(text="")
        self.assertEqual(scan.status_code, 422)
        self.assertIn("OCR", scan.json()["detail"])


class DiscoveryApiTests(ApiTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.login()

    def test_sources(self) -> None:
        sources = {s["name"]: s for s in self.client.get("/api/discovery/sources").json()}
        self.assertTrue(sources["sample"]["enabled"])
        self.assertFalse(sources["gem"]["enabled"])
        self.assertIn("Switched off", sources["gem"]["reason"])

    def test_find_rfps_registers_them_once(self) -> None:
        body = {"keywords": ["switches", "sockets"], "count": 3, "location": "Pune",
                "date_from": "2026-11-01", "date_to": "2026-11-30"}
        found = self.client.post("/api/discovery/search", json=body)
        self.assertEqual(found.status_code, 200)
        rfps = found.json()
        self.assertEqual(len(rfps), 3)
        for r in rfps:
            self.assertEqual((r["source"], r["location"]), ("sample", "Pune"))
            self.assertTrue(r["buyer"] and r["deadline"] and r["estimated_value"])
            self.assertTrue(r["meta"]["synthetic"])
            self.assertTrue(r["meta"]["reference"].startswith("SMP/"))
        again = self.client.post("/api/discovery/search", json=body).json()
        self.assertEqual([r["id"] for r in again], [r["id"] for r in rfps])
        self.assertEqual(len(self.client.get("/api/rfps").json()), 3)
        # A discovered RFP can start a project like any other.
        self.assertEqual(self.client.post("/api/projects", json={"rfp_id": rfps[0]["id"]}).status_code, 202)

    def test_gem_refused_and_bad_input(self) -> None:
        self.assertEqual(self.client.post("/api/discovery/search", json={"source": "gem"}).status_code, 409)
        self.assertEqual(self.client.post("/api/discovery/search", json={"count": 0}).status_code, 422)

    def test_upload_reads_metadata(self) -> None:
        reply = json.dumps({"title": "Supply of Modular Switches", "buyer": "ABC Infrastructure Pvt. Ltd.",
                            "deadline": "2026-12-15", "location": "Gurugram", "category": "Modular switches",
                            "estimated_value": None})
        with patch("llm_json.chat_completion", return_value=reply):
            rfp = self.upload().json()
        self.assertEqual((rfp["buyer"], rfp["deadline"], rfp["location"]),
                         ("ABC Infrastructure Pvt. Ltd.", "2026-12-15", "Gurugram"))


class SettingsApiTests(ApiTestCase):
    def test_company_profile_round_trip(self) -> None:
        self.assertEqual(self.client.get("/api/settings/company").status_code, 401)
        self.login()
        default = self.client.get("/api/settings/company").json()
        self.assertTrue(default["is_demo"])
        body = {**default, "name": "Sharma Electricals", "gstin": "06AAAAA0000A1Z5"}
        body.pop("is_demo")
        saved = self.client.put("/api/settings/company", json=body).json()
        self.assertEqual((saved["name"], saved["is_demo"]), ("Sharma Electricals", False))
        self.assertEqual(self.client.get("/api/settings/company").json()["name"], "Sharma Electricals")
        self.assertEqual(self.client.put("/api/settings/company", json={**body, "name": ""}).status_code, 422)


class StockApiTests(ApiTestCase):
    def test_stock_overview_lists_every_product_with_stock(self) -> None:
        self.login()
        body = self.client.get("/api/catalogue/stock").json()
        self.assertEqual(set(body), {"products", "low_stock_units", "is_demo"})
        self.assertGreater(body["low_stock_units"], 0)
        self.assertTrue(body["is_demo"])
        products = body["products"]
        self.assertGreater(len(products), 100)
        self.assertEqual(len({p["product_id"] for p in products}), len(products))
        for p in products:
            self.assertIsInstance(p["stock_on_hand"], int)
            self.assertIsInstance(p["monthly_available"], int)
            self.assertGreater(p["unit_price"], 0)
        # The demo data includes some low-stock products, so the filter has something to show.
        self.assertTrue(any(p["stock_on_hand"] < body["low_stock_units"] for p in products))

    def test_stock_keywords_for_discovery(self) -> None:
        self.assertEqual(self.client.get("/api/discovery/stock-keywords").status_code, 401)
        self.login()
        found = self.client.get("/api/discovery/stock-keywords", params={"limit": 4}).json()
        self.assertEqual(len(found), 4)
        self.assertEqual(set(found[0]), {"keyword", "products", "units_on_hand", "examples"})
        units = [k["units_on_hand"] for k in found]
        self.assertEqual(units, sorted(units, reverse=True))


class TitleGuessTests(unittest.TestCase):
    def test_skips_generic_headings(self) -> None:
        from api.routes_rfps import _guess_title
        self.assertEqual(_guess_title("Request for Proposal (RFP)\nSupply of Modular Switches\n", "fb"),
                         "Supply of Modular Switches")
        self.assertEqual(_guess_title("Request for Proposal - Supply of Modular Switches\n", "fb"),
                         "Request for Proposal - Supply of Modular Switches")
        self.assertEqual(_guess_title("RFP\n", "fallback"), "fallback")


class UiServingTests(ApiTestCase):
    def setUp(self) -> None:
        super().setUp()
        dist = self.tmp / "dist"
        (dist / "assets").mkdir(parents=True)
        (dist / "index.html").write_text("<!doctype html><div id=root></div>", encoding="utf-8")
        (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
        app = create_app(db_path=self.db_path, workflow=self.workflow, var_dir=self.tmp,
                         run_inline=True, warm_up=False, frontend_dist=str(dist))
        self.ui = TestClient(app)

    def test_spa_routes_and_assets(self) -> None:
        self.assertIn("id=root", self.ui.get("/").text)
        self.assertIn("id=root", self.ui.get("/projects/abc123").text)   # React route after refresh
        self.assertEqual(self.ui.get("/assets/app.js").text, "console.log(1)")
        api_404 = self.ui.get("/api/does-not-exist")
        self.assertEqual(api_404.status_code, 404)
        self.assertEqual(api_404.json()["detail"], "Not found")

    def test_cannot_read_files_outside_the_ui_folder(self) -> None:
        (self.tmp / "secret.txt").write_text("TOP SECRET", encoding="utf-8")   # sibling of dist/
        for path in ("/%2e%2e/secret.txt", "/..%2Fsecret.txt", "/assets/..%2F..%2Fsecret.txt"):
            with self.subTest(path=path):
                r = self.ui.get(path)
                self.assertNotIn("TOP SECRET", r.text)

    def test_pricing_rules_endpoint(self) -> None:
        self.assertEqual(self.client.get("/api/pricing/rules").status_code, 401)
        self.login()
        rules = self.client.get("/api/pricing/rules").json()
        self.assertEqual((rules["min_margin_pct"], rules["max_margin_pct"], rules["gst_rate_pct"]), (0.0, 60.0, 18.0))
        self.assertEqual(len(rules["volume_discount_tiers"]), 4)


class ProjectFlowTests(ApiTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.login()
        self.rfp_id = self.upload().json()["id"]

    def start(self) -> dict:
        r = self.client.post("/api/projects", json={"rfp_id": self.rfp_id})
        self.assertEqual(r.status_code, 202)
        return r.json()

    def review(self, pid: str, decision: dict):
        return self.client.post(f"/api/projects/{pid}/review", json={"decision": decision})

    def test_full_flow_over_http(self) -> None:
        project = self.start()
        pid = project["id"]
        self.assertEqual(project["stage"], "requirements_review")
        self.assertEqual(len(project["requirements"]), 3)
        self.assertEqual(project["rfp"]["id"], self.rfp_id)

        # Requirements → matching
        r = self.review(pid, {})
        self.assertEqual(r.json()["stage"], "matching_review")
        self.assertEqual(r.json()["match_summary"]["needs_review"], 1)

        # Change Product: search the catalogue, then swap
        found = self.client.get("/api/catalogue/products", params={"q": "16A 2Way oro metallica"}).json()
        self.assertIn(SWITCH_ORO.id, [p["product_id"] for p in found])
        r = self.review(pid, {"decisions": [{"id": "R1", "action": "change", "product_id": SWITCH_ORO.id}],
                              "accept_all": True})
        project = r.json()
        self.assertEqual(project["stage"], "pricing_review")
        self.assertEqual(project["review_log"][-1]["reviewer"], "Khushi")
        saved_total = project["pricing_summary"]["grand_total"]

        # Margin slider preview: instant, nothing saved
        preview = self.client.post(f"/api/projects/{pid}/pricing/preview",
                                   json={"margin_pct": 30, "quantities": {"R2": 50}}).json()
        self.assertEqual(preview["pricing_summary"]["margin_pct"], 30.0)
        self.assertNotEqual(preview["pricing_summary"]["grand_total"], saved_total)
        tv = next(li for li in preview["line_items"] if li["requirement_id"] == "R2")
        self.assertEqual((tv["qty"], tv["qty_source"]), (50, "reviewer"))
        unchanged = self.client.get(f"/api/projects/{pid}").json()
        self.assertEqual(unchanged["pricing_summary"]["grand_total"], saved_total)

        csv = self.client.get(f"/api/projects/{pid}/quote.csv")
        self.assertEqual(csv.status_code, 200)
        self.assertIn("requirement,qty", csv.text)

        # Pricing → proposal review
        r = self.review(pid, {"margin_pct": 30, "quantities": {"R2": 50}})
        project = r.json()
        self.assertEqual(project["stage"], "proposal_review")
        self.assertEqual(project["pricing_summary"]["margin_pct"], 30.0)
        self.assertEqual([s["key"] for s in project["proposal_sections"]], ["executive_summary", "pricing"])
        self.assertNotIn("profile", project["proposal_meta"])

        # Live preview of unsaved edits (a real PDF, nothing saved)
        edited = [{"key": "executive_summary", "body": "Edited summary for ₹ 5,000 & more."}]
        preview = self.client.post(f"/api/projects/{pid}/proposal/preview", json={"sections": edited})
        self.assertEqual(preview.headers["content-type"], "application/pdf")
        with pymupdf.open(stream=preview.content, filetype="pdf") as doc:
            text = "".join(p.get_text() for p in doc)
        self.assertIn("Edited summary", text)
        self.assertIn("Proposal value", text)

        # Regenerate one section with an instruction
        with patch("llm_json.chat_completion",
                   return_value=json.dumps({"text": "A shorter executive summary for the buyer."})) as llm:
            regen = self.client.post(f"/api/projects/{pid}/proposal/sections/executive_summary/regenerate",
                                     json={"instruction": "make it shorter", "sections": edited}).json()
        self.assertEqual(regen["body"], "A shorter executive summary for the buyer.")
        self.assertIn("make it shorter", llm.call_args.kwargs["user"])
        self.assertEqual(self.client.post(f"/api/projects/{pid}/proposal/sections/pricing/regenerate",
                                          json={}).status_code, 422)   # not an AI section

        # Save a draft: stays at the review, warnings flag the ₹ amount
        saved = self.client.put(f"/api/projects/{pid}/proposal/draft", json={"sections": edited}).json()
        self.assertEqual(saved["stage"], "proposal_review")
        self.assertIn("executive_summary", saved["proposal_warnings"])

        # Approve → rendered PDF
        project = self.review(pid, {}).json()
        self.assertEqual(project["stage"], "done")
        self.assertTrue(project["proposal_available"])
        pdf = self.client.get(f"/api/projects/{pid}/proposal.pdf")
        self.assertEqual(pdf.headers["content-type"], "application/pdf")
        with pymupdf.open(stream=pdf.content, filetype="pdf") as doc:
            self.assertIn("Edited summary", "".join(p.get_text() for p in doc))

        # History
        outcome = self.client.patch(f"/api/projects/{pid}/outcome", json={"outcome": "won"}).json()
        self.assertEqual(outcome["outcome"], "won")
        listed = self.client.get("/api/projects").json()
        self.assertEqual([(p["id"], p["stage"], p["outcome"]) for p in listed], [(pid, "done", "won")])
        self.assertEqual(listed[0]["grand_total"], project["pricing_summary"]["grand_total"])
        self.assertIsNotNone(listed[0]["completed_at"])

        # Dashboard counts it as generated and won this month
        dash = self.client.get("/api/dashboard").json()
        self.assertEqual(dash["kpis"]["proposals_generated"]["value"], 1)
        self.assertEqual(dash["kpis"]["proposals_generated"]["this_month"], 1)
        self.assertEqual(dash["kpis"]["win_rate"]["value"], 100.0)
        self.assertEqual(dash["activity"][-1], {"month": dash["today"][:7], "generated": 1, "won": 1})
        self.assertEqual(dash["pending_total"], 0)

    def test_bad_review_is_rejected_and_nothing_moves(self) -> None:
        pid = self.start()["id"]
        self.review(pid, {})
        r = self.review(pid, {"decisions": [{"id": "R9", "action": "accept"}]})
        self.assertEqual(r.status_code, 422)
        self.assertIn("Unknown item id", r.json()["detail"])
        self.assertEqual(self.client.get(f"/api/projects/{pid}").json()["stage"], "matching_review")
        bad_qty = self.client.post(f"/api/projects/{pid}/pricing/preview", json={"margin_pct": 10})
        self.assertEqual(bad_qty.status_code, 409)   # not priced yet

    def test_failure_then_retry(self) -> None:
        self.steps.fail_extract_once = True
        project = self.start()
        self.assertEqual(project["stage"], "failed")
        self.assertEqual(project["failed_step"], "extract")
        self.assertIn("OPENROUTER_API_KEY", project["error"])

        r = self.client.post(f"/api/projects/{project['id']}/retry")
        self.assertEqual(r.json()["stage"], "requirements_review")
        self.assertIsNone(r.json()["error"])

    def test_busy_project_refuses_reviews(self) -> None:
        pid = self.start()["id"]
        with Session(get_engine(self.db_path)) as db:
            db.get(Project, pid).running = True
            db.commit()
        self.assertEqual(self.client.get(f"/api/projects/{pid}").json()["stage"], "running")
        self.assertEqual(self.review(pid, {}).status_code, 409)

    def test_unknown_ids(self) -> None:
        self.assertEqual(self.client.post("/api/projects", json={"rfp_id": "nope"}).status_code, 404)
        self.assertEqual(self.client.get("/api/projects/nope").status_code, 404)
        self.assertEqual(self.client.get("/api/projects/nope/proposal.pdf").status_code, 404)


if __name__ == "__main__":
    unittest.main()
