from __future__ import annotations

import glob
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pymupdf

from catalogue import get_catalogue
from discovery.base import DiscoveryQuery, SourceUnavailable
from discovery.discovery_agent import discovery_node, list_sources, run_discovery
from discovery.gem import GemSource, looks_like_gem, parse_gem_text
from discovery.metadata import extract_metadata
from discovery.samples import OFF_CATALOGUE, SampleSource, generate_rfp
from rfp_rag import RfpRagTool
from settings import RFP_PDF_PATH

LEGACY = Path(__file__).resolve().parent.parent / "legacy" / "gem_scraper"


def legacy_tender_texts():
    for f in sorted(glob.glob(str(LEGACY / "**" / "session_*.json"), recursive=True)):
        for r in json.loads(Path(f).read_text(encoding="utf-8")).get("rfps", []):
            text = r["analysis"].get("full_text", "")
            if text:
                yield r["bid_title"], text


class QueryTests(unittest.TestCase):
    def test_keywords_are_cleaned(self) -> None:
        q = DiscoveryQuery(keywords=" switches, sockets ,Switches,, ")
        self.assertEqual(q.keywords, ["switches", "sockets"])

    def test_count_is_bounded(self) -> None:
        with self.assertRaises(ValueError):
            DiscoveryQuery(count=0)
        with self.assertRaises(ValueError):
            DiscoveryQuery(count=50)


class SampleGeneratorTests(unittest.TestCase):
    products = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.products = get_catalogue().all_products()

    def rfp(self, i=0, **query):
        return generate_rfp(DiscoveryQuery(**query), i, self.products)

    def test_deterministic(self) -> None:
        a = self.rfp(keywords=["switches"], location="Pune")
        b = self.rfp(keywords=["switches"], location="Pune")
        self.assertEqual(a, b)
        self.assertNotEqual(a, self.rfp(1, keywords=["switches"], location="Pune"))

    def test_respects_the_search(self) -> None:
        q = dict(keywords=["sockets"], location="Kochi", date_from=date(2026, 11, 1), date_to=date(2026, 11, 30),
                 min_value=500_000, max_value=900_000)
        for i in range(5):
            r = self.rfp(i, **q)
            self.assertEqual(r["location"], "Kochi")
            self.assertTrue(date(2026, 11, 1) <= r["published"] <= date(2026, 11, 30))
            self.assertGreater(r["deadline"], r["published"])
            self.assertGreaterEqual((r["deadline"] - date.today()).days, 5)   # still open for bids
            self.assertTrue(500_000 <= r["value"] <= 900_000)
            catalogue_lines = [l for l in r["lines"] if l["catalogue_price"]]
            self.assertTrue(catalogue_lines)
            self.assertTrue(all("socket" in l["description"].lower() for l in catalogue_lines))

    def test_distinct_buyers_and_no_repeated_items(self) -> None:
        rfps = [self.rfp(i, keywords=["modular switches"]) for i in range(6)]
        self.assertEqual(len({r["buyer"] for r in rfps}), 6)
        for r in rfps:
            names = [l["description"].split(",")[0].split(" (")[0].lower() for l in r["lines"]]
            self.assertEqual(len(names), len(set(names)))
            self.assertTrue(all(l["qty"] > 0 for l in r["lines"]))

    def test_keywords_outside_the_catalogue_become_gap_items(self) -> None:
        r = self.rfp(keywords=["switches", "cable"])
        gaps = [l["description"] for l in r["lines"] if l["catalogue_price"] is None]
        self.assertTrue(gaps)
        self.assertTrue(set(gaps) <= set(OFF_CATALOGUE["cable"]))


class SamplePdfTests(unittest.TestCase):
    def test_pdf_reads_like_an_rfp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            found = SampleSource().search(DiscoveryQuery(keywords=["switches"], count=2, location="Jaipur"), tmp)
            again = SampleSource().search(DiscoveryQuery(keywords=["switches"], count=2, location="Jaipur"), tmp + "/b")
            self.assertEqual(len(found), 2)
            c = found[0]
            self.assertEqual((c.source, c.location), ("sample", "Jaipur"))
            self.assertTrue(c.meta["synthetic"])
            self.assertEqual(Path(c.pdf_path).read_bytes(), Path(again[0].pdf_path).read_bytes())

            with pymupdf.open(c.pdf_path) as doc:
                text = "".join(p.get_text() for p in doc)
            self.assertIn("fictitious", text)
            self.assertIn(c.buyer, text)
            for heading in ("1. Introduction", "2. Scope of Supply", "4. Eligibility Criteria", "7. Submission"):
                self.assertIn(heading, text)
            chunks = RfpRagTool._split_large_chunks(RfpRagTool._chunk_by_sections(text), 800)
            self.assertGreaterEqual(len(chunks), 6)


class GemTests(unittest.TestCase):
    def test_parser_on_archived_real_tenders(self) -> None:
        tenders = list(legacy_tender_texts())
        self.assertGreaterEqual(len(tenders), 30)
        for bid_title, text in tenders:
            with self.subTest(bid=bid_title):
                self.assertTrue(looks_like_gem(text))
                meta = parse_gem_text(text)
                for key in ("reference", "buyer", "deadline", "published", "category"):
                    self.assertIn(key, meta)
                date.fromisoformat(meta["deadline"])

    def test_sample_rfp_is_not_gem(self) -> None:
        with pymupdf.open(RFP_PDF_PATH) as doc:
            self.assertFalse(looks_like_gem("".join(p.get_text() for p in doc)))

    def test_gem_is_off_by_default(self) -> None:
        self.assertIn("Switched off", GemSource().available())
        with self.assertRaises(SourceUnavailable):
            run_discovery(DiscoveryQuery(keywords=["switch"]), "gem", out_dir=tempfile.gettempdir())
        by_name = {s["name"]: s for s in list_sources()}
        self.assertTrue(by_name["sample"]["enabled"])
        self.assertFalse(by_name["gem"]["enabled"])


class MetadataTests(unittest.TestCase):
    def test_llm_answer_is_normalised(self) -> None:
        reply = json.dumps({"title": "Supply of switches", "buyer": "ABC Infrastructure Pvt. Ltd.",
                            "deadline": "not stated", "location": "null", "category": "Modular switches",
                            "estimated_value": "12,50,000"})
        with patch("llm_json.chat_completion", return_value=reply):
            meta = extract_metadata("Request for Proposal ... ABC Infrastructure Pvt. Ltd. ...")
        self.assertEqual(meta, {"title": "Supply of switches", "buyer": "ABC Infrastructure Pvt. Ltd.",
                                "category": "Modular switches", "estimated_value": 1250000.0})

    def test_gem_documents_skip_the_llm(self) -> None:
        _, text = next(legacy_tender_texts())
        with patch("llm_json.chat_completion") as mock_chat:
            meta = extract_metadata(text)
        mock_chat.assert_not_called()
        self.assertIn("buyer", meta)

    def test_no_key_means_no_metadata_not_an_error(self) -> None:
        self.assertEqual(extract_metadata("Some RFP text without a key configured"), {})


class DiscoveryNodeTests(unittest.TestCase):
    def test_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch("discovery.discovery_agent.DISCOVERY_DIR", tmp):
            out = discovery_node({"discovery_query": {"keywords": ["plates"], "count": 2}})
            self.assertEqual(len(out["discovered_rfps"]), 2)
            self.assertTrue(Path(out["discovered_rfps"][0]["pdf_path"]).is_file())
            failed = discovery_node({"discovery_source": "gem", "discovery_query": {}})
            self.assertEqual(failed["discovered_rfps"], [])
            self.assertIn("Switched off", failed["error"])


if __name__ == "__main__":
    unittest.main()
