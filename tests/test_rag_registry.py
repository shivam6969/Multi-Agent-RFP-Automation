from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pymupdf

import bu_rag
import rfp_rag
from embedding_cache import cache_key, encode_cached
from tests.fakes import FakeEmbedder, FakeReranker

SWITCH_RFP = (
    "1. Introduction\nGreen Valley Hotels invites proposals for the supply of modular "
    "switches and sockets for 120 guest rooms in the new Jaipur property.\n"
    "2. Scope\nThe vendor shall supply 16A 2Way switches, 6A sockets and fan "
    "regulators, delivered within 10 days of purchase order.\n"
)
PLATES_RFP = (
    "1. Introduction\nSunrise Public School requests quotations for combined plates "
    "and outer plates for classroom renovation across three campuses.\n"
    "2. Delivery\nAll plates must be delivered to the Gurugram campus before the "
    "start of the academic term, packed in standard cartons.\n"
)


def _write_pdf(path: Path, text: str) -> str:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(40, 40, 560, 800), text, fontsize=10)
    doc.save(str(path))
    doc.close()
    return str(path)


class EmbeddingCacheTests(unittest.TestCase):
    def test_second_call_reads_from_disk(self) -> None:
        embedder = FakeEmbedder()
        with tempfile.TemporaryDirectory() as tmp:
            first = encode_cached(embedder, ["a b", "c d"], cache_dir=tmp)
            second = encode_cached(embedder, ["a b", "c d"], cache_dir=tmp)
        self.assertEqual(embedder.encoded_texts, 2)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(second.dtype, np.float32)

    def test_key_depends_on_model_options_and_texts(self) -> None:
        base = cache_key("m1", ["x"])
        self.assertNotEqual(base, cache_key("m2", ["x"]))
        self.assertNotEqual(base, cache_key("m1", ["y"]))
        self.assertNotEqual(base, cache_key("m1", ["x"], normalize_embeddings=True))


class RfpRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.embedder = FakeEmbedder()
        patchers = [
            patch("rfp_rag.get_embedder", return_value=self.embedder),
            patch("rfp_rag.get_reranker", return_value=FakeReranker()),
            patch("embedding_cache.EMBEDDING_CACHE_DIR", str(self.dir / "cache")),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        rfp_rag.clear_rfp_tools()
        self.addCleanup(rfp_rag.clear_rfp_tools)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def cfg(self, path: str) -> rfp_rag.RfpRagConfig:
        return rfp_rag.RfpRagConfig(pdf_path=path)

    def test_one_tool_per_pdf(self) -> None:
        hotel = _write_pdf(self.dir / "hotel.pdf", SWITCH_RFP)
        school = _write_pdf(self.dir / "school.pdf", PLATES_RFP)

        hotel_tool = rfp_rag.get_rfp_rag_tool(self.cfg(hotel))
        school_tool = rfp_rag.get_rfp_rag_tool(self.cfg(school))

        self.assertIsNot(hotel_tool, school_tool)
        self.assertIs(rfp_rag.get_rfp_rag_tool(self.cfg(hotel)), hotel_tool)
        self.assertIn("Green Valley", " ".join(hotel_tool.retrieve("who is the buyer", top_k=5)))
        self.assertIn("Sunrise", " ".join(school_tool.retrieve("who is the buyer", top_k=5)))

    def test_same_content_at_another_path_reuses_the_tool(self) -> None:
        a = _write_pdf(self.dir / "a.pdf", SWITCH_RFP)
        b = self.dir / "copy.pdf"
        b.write_bytes(Path(a).read_bytes())
        self.assertIs(rfp_rag.get_rfp_rag_tool(self.cfg(a)), rfp_rag.get_rfp_rag_tool(self.cfg(str(b))))

    def test_rebuilding_uses_the_embedding_cache(self) -> None:
        hotel = _write_pdf(self.dir / "hotel.pdf", SWITCH_RFP)
        rfp_rag.get_rfp_rag_tool(self.cfg(hotel))
        encoded_once = self.embedder.encoded_texts
        self.assertGreater(encoded_once, 0)

        rfp_rag.clear_rfp_tools()  # simulate a restart
        rfp_rag.get_rfp_rag_tool(self.cfg(hotel))
        self.assertEqual(self.embedder.encoded_texts, encoded_once)

    def test_changed_file_builds_a_new_tool(self) -> None:
        path = self.dir / "rfp.pdf"
        first = rfp_rag.get_rfp_rag_tool(self.cfg(_write_pdf(path, SWITCH_RFP)))
        second = rfp_rag.get_rfp_rag_tool(self.cfg(_write_pdf(path, PLATES_RFP)))
        self.assertIsNot(first, second)

    def test_missing_pdf(self) -> None:
        with self.assertRaises(FileNotFoundError):
            rfp_rag.get_rfp_rag_tool(self.cfg(str(self.dir / "nope.pdf")))

    def test_pdf_without_text(self) -> None:
        blank = _write_pdf(self.dir / "scan.pdf", "")
        with self.assertRaisesRegex(ValueError, "No readable text"):
            rfp_rag.get_rfp_rag_tool(self.cfg(blank))

    def test_registry_is_bounded(self) -> None:
        with patch("rfp_rag.RFP_TOOL_CACHE_SIZE", 2):
            for i in range(3):
                pdf = _write_pdf(self.dir / f"r{i}.pdf", SWITCH_RFP + f"\n3. Lot\nLot number {i} " * 3)
                rfp_rag.get_rfp_rag_tool(self.cfg(pdf))
            self.assertEqual(len(rfp_rag._TOOLS), 2)


class BuRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embedder = FakeEmbedder()
        cache_dir = tempfile.TemporaryDirectory()
        self.addCleanup(cache_dir.cleanup)
        patchers = [
            patch("bu_rag.get_embedder", return_value=self.embedder),
            patch("bu_rag.get_reranker", return_value=FakeReranker()),
            patch("embedding_cache.EMBEDDING_CACHE_DIR", cache_dir.name),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        bu_rag.clear_bu_tools()
        self.addCleanup(bu_rag.clear_bu_tools)

    def test_tool_is_shared_and_cached(self) -> None:
        tool = bu_rag.get_bu_rag_tool()
        self.assertIs(bu_rag.get_bu_rag_tool(), tool)
        encoded_once = self.embedder.encoded_texts

        bu_rag.clear_bu_tools()  # simulate a restart
        bu_rag.get_bu_rag_tool()
        self.assertEqual(self.embedder.encoded_texts, encoded_once)

    def test_search_products_returns_catalogue_records(self) -> None:
        results = bu_rag.get_bu_rag_tool().search_products("16A 2Way Switch", top_k=3)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r.unit_price > 0 and r.id for r in results))


if __name__ == "__main__":
    unittest.main()
