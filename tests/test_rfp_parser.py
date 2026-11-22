"""Parser, chunking and hybrid search on the table-heavy sample RFPs (offline)."""

from __future__ import annotations

import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import rfp_rag
from rfp_agent import _table_requirements
from rfp_parser import parse_pdf
from rfp_rag import BM25, build_chunks, document_segments, reciprocal_rank_fusion
from settings import RFP_PDF_PATH
from tests.fakes import FakeEmbedder, FakeReranker
from tests.sample_rfps import build_all


class SampleRfpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="rfp-samples-"))
        cls.samples = build_all(cls.tmp)
        cls.docs = {name: parse_pdf(str(path)) for name, (path, _truth) in cls.samples.items()}

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def table_reqs(self, name: str):
        out = []
        for t in self.docs[name].tables:
            if t.kind == "boq":
                out += _table_requirements(t)[0]
        return out

    def test_every_item_is_read_with_its_exact_quantity(self) -> None:
        for name, (_path, truth) in self.samples.items():
            with self.subTest(sample=name):
                reqs = self.table_reqs(name)
                self.assertEqual(len(reqs), len(truth.items))
                for item, req in zip(truth.items, reqs):
                    self.assertEqual(req["quantity"], item.quantity, item.description)
                    self.assertEqual(req["unit"], item.unit, item.description)
                    words = re.findall(r"[a-z0-9]+", item.description.lower())
                    self.assertTrue(all(w in req["requirement"].lower() for w in words),
                                    (item.description, req["requirement"]))

    def test_hotel_tables(self) -> None:
        tables = self.docs["hotel"].tables
        self.assertEqual([t.kind for t in tables], ["kv", "boq", "spec", "other"])
        boq = tables[1]
        self.assertEqual(boq.pages, [1, 2])                       # joined across the page break
        self.assertEqual(boq.header[1], "Description of Item")    # repeated header not a data row
        self.assertNotIn("Description of Item", [r[1] for r in boq.rows])
        self.assertEqual(boq.roles[-2:], ["price", "price"])      # rate / amount never read as qty
        self.assertIn("Tender Reference", tables[0].header)       # "Bid Details" heading not a header

    def test_hotel_sources_and_categories(self) -> None:
        reqs = self.table_reqs("hotel")
        usb = next(r for r in reqs if "USB" in r["requirement"])
        self.assertEqual(usb["source"], {"page": 2, "table": "T2", "row": "13",
                                         "section": "2. Schedule of Requirements"})
        self.assertTrue(reqs[0]["requirement"].startswith("Switches (Oro Metallica range"))

    def test_school_merged_header_and_category_cells(self) -> None:
        t = self.docs["school"].tables[0]
        self.assertEqual(t.header[4:], ["Quantity Campus A", "Quantity Campus B", "Quantity Campus C",
                                        "Quantity Total"])
        self.assertEqual([r[1] for r in t.rows[:4]], ["Switches (Coral)"] * 4)   # merged cell filled down
        first = self.table_reqs("school")[0]
        self.assertEqual(first["quantity_breakdown"],
                         {"Quantity Campus A": 420, "Quantity Campus B": 380, "Quantity Campus C": 260})

    def test_office_table_without_gridlines(self) -> None:
        boq, schedule = self.docs["office"].tables
        self.assertEqual((boq.kind, boq.detection, len(boq.rows)), ("boq", "text", 8))
        self.assertEqual(boq.rows[0][1], "Modular switch 6A/10A 1 way, white, Oro range")  # wrapped line joined
        self.assertEqual(schedule.section, "3. Delivery Schedule")   # next heading not swallowed

    def test_running_headers_footers_and_page_numbers_are_removed(self) -> None:
        doc = self.docs["hotel"]
        text = doc.to_text()
        self.assertGreater(doc.removed_lines, 0)
        self.assertNotRegex(text, r"(?m)^Page \d+$")
        self.assertEqual(text.count("SAMPLE RFP for testing"), 0)

    def test_headings_give_sections(self) -> None:
        sections = {b.section for b in self.docs["hotel"].blocks}
        for heading in ("1. Introduction", "3. Technical Specifications", "4. Eligibility Criteria"):
            self.assertIn(heading, sections)

    def test_table_cells_are_not_repeated_as_text(self) -> None:
        text_blocks = " ".join(b.text for b in self.docs["hotel"].blocks if b.kind != "table")
        self.assertNotIn("Electronic key card switch", text_blocks)

    def test_no_text_is_lost_in_chunks(self) -> None:
        for name, doc in self.docs.items():
            with self.subTest(sample=name):
                chunks = " ".join(c.text for c in build_chunks(doc, max_length=300))
                for b in doc.blocks:
                    if b.kind != "table":
                        for line in b.text.splitlines():
                            self.assertIn(line[:40], chunks)

    def test_chunks_have_labels_and_row_records(self) -> None:
        chunks = build_chunks(self.docs["hotel"])
        rows = [c for c in chunks if c.kind == "row" and c.table_id == "T2"]
        usb = next(c for c in rows if "USB" in c.text)
        self.assertIn("Qty: 160", usb.text)
        self.assertIn("[Support modules]", usb.text)
        self.assertEqual(usb.page, 2)
        text = next(c for c in chunks if c.kind == "text" and "Liquidated damages" in c.text)
        self.assertTrue(text.text.startswith("[Section: 5. Delivery and Commercial Terms | p.3]"))
        kv = [c for c in chunks if c.kind == "row" and c.table_id == "T1"]
        self.assertTrue(any("Earnest Money Deposit (EMD): Rs. 25,000" in c.text for c in kv))

    def test_segments_replace_item_tables_with_a_placeholder(self) -> None:
        segs = document_segments(self.docs["hotel"], {"T2"}, max_chars=12000)
        joined = "\n".join(s.text for s in segs)
        self.assertNotIn("Electronic key card", joined)
        self.assertIn("[Table T2", joined)
        self.assertIn("| Parameter | Requirement |", joined)   # other tables reach the LLM as Markdown
        self.assertEqual([s.table_id for s in segs if s.table_id], ["T2"])

    def test_decimals_in_text_are_not_split_or_dropped(self) -> None:
        doc = self.docs["hotel"]
        self.assertIn("2.5 sq mm and 1.5 sq mm", re.sub(r"\s+", " ", doc.to_text()))
        self.assertIn("2.1A output", " ".join(c.text for c in build_chunks(doc)))

    def test_vanilla_sample_rfp_still_parses(self) -> None:
        doc = parse_pdf(RFP_PDF_PATH)
        headings = [b.text for b in doc.blocks if b.kind == "heading"]
        self.assertIn("6. Eligibility Criteria", headings)
        self.assertNotIn("Dimmers", " ".join(headings))   # bullets are not headings
        self.assertIn("RJ-45 data jacks", doc.to_text())


class HybridSearchTests(unittest.TestCase):
    def test_bm25_prefers_exact_tokens(self) -> None:
        bm = BM25(["switches as per IS 3854", "sockets as per IS 1293", "delivery in 14 days"])
        scores = bm.scores("IS 1293")
        self.assertEqual(int(scores.argmax()), 1)
        self.assertEqual(float(bm.scores("unrelated words").max()), 0.0)

    def test_rrf(self) -> None:
        self.assertEqual(reciprocal_rank_fusion([[1, 2, 3], [3, 1]])[:2], [1, 3])

    def test_tool_retrieves_the_item_row(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="rfp-rag-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        path, _truth = build_all(tmp)["hotel"]
        with patch("rfp_rag.get_embedder", return_value=FakeEmbedder()), \
             patch("rfp_rag.get_reranker", return_value=FakeReranker()), \
             patch("embedding_cache.EMBEDDING_CACHE_DIR", str(tmp / "cache")):
            rfp_rag.clear_rfp_tools()
            self.addCleanup(rfp_rag.clear_rfp_tools)
            tool = rfp_rag.get_rfp_rag_tool(rfp_rag.RfpRagConfig(pdf_path=str(path)))
            results = tool.retrieve_chunks("How many TV socket are required?", top_k=3)
        # the row record, or the table slice holding that row (bag-of-words fakes can't tell them apart)
        self.assertTrue(any("TV socket" in c.text and "120" in c.text and c.table_id == "T2"
                            for c in results), [c.text[:80] for c in results])
        self.assertTrue(any(c.kind == "row" for c in results))
        self.assertEqual([t.id for t in tool.tables], ["T1", "T2", "T3", "T4"])
        self.assertIn("| Parameter | Requirement |", tool.full_text)


if __name__ == "__main__":
    unittest.main()
