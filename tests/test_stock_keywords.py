from __future__ import annotations

import unittest

from catalogue import ProductRecord, get_catalogue
from discovery.gem import _card_department, _card_field, is_relevant, parse_gem_text
from discovery.keywords import extract_keywords, product_type


def product(name: str, on_hand: int, pid: str) -> ProductRecord:
    return ProductRecord(id=pid, sap_code="", brand="HAVELLS", range_name="Coral", variant=None,
                         variant_type=None, item_name=name, hsn_code="", packing="", unit_price=10.0,
                         price_raw="10/-", on_hand=on_hand)


class ProductTypeTests(unittest.TestCase):
    def test_strips_ratings_sizes_and_ranges(self) -> None:
        self.assertEqual(product_type("6/16A 3 Pin Combined Shuttered Socket"), "shuttered socket")
        self.assertEqual(product_type("16A 1 way switch with ind."), "switch")
        self.assertEqual(product_type("16 A SP C MINI MCB Coral"), "mini mcb")
        self.assertEqual(product_type("High Speed Fan Regulator"), "fan regulator")
        self.assertEqual(product_type("USB Charger (1M)"), "usb charger")
        self.assertEqual(product_type("4M Box (5X3)"), "box")
        self.assertIsNone(product_type("16A 1M"))


class ExtractKeywordsTests(unittest.TestCase):
    def test_ranked_by_stock_and_mapped_to_tender_wording(self) -> None:
        products = [
            product("6A Switch", 50, "a"), product("16A 2 Way Switch", 60, "b"),
            product("6A Socket", 500, "c"), product("16A Shuttered Socket", 400, "d"),
            product("Bell Push", 999, "e"),   # only one product: not a keyword
        ]
        keywords = extract_keywords(products, limit=5)
        self.assertEqual([k.keyword for k in keywords], ["modular socket", "modular switch"])
        self.assertEqual(keywords[0].units_on_hand, 900)
        self.assertEqual(keywords[0].products, 2)

    def test_real_catalogue_gives_core_product_types(self) -> None:
        words = [k.keyword for k in extract_keywords(get_catalogue().all_products(), limit=8)]
        for expected in ("modular switch", "modular socket", "mcb", "fan regulator"):
            self.assertIn(expected, words)
        self.assertEqual(len(words), len(set(words)))


class GemCardParsingTests(unittest.TestCase):
    CARD = ("BID NO: GEM/2026/B/1234567\nItems: Modular Switch (Q3)\nQuantity: 250\n"
            "Department Name And Address:\nMinistry Of Railways\nNorthern Railway\n"
            "Start Date: 01-11-2026 10:00 AM\nEnd Date: 21-11-2026 6:00 PM")

    def test_relevance_needs_the_product_in_the_item_category(self) -> None:
        self.assertTrue(is_relevant("modular switch", "Modular Switches and Sockets (Q2)"))
        self.assertTrue(is_relevant("mcb", "Miniature Circuit Breakers as per IS/IEC 60898-1"))
        self.assertFalse(is_relevant("modular switch", "PRESSURE REDUCING MODULAR VALVE"))
        self.assertFalse(is_relevant("mcb", "LED Luminaire For Road And Street Lights"))

    def test_current_bilingual_bid_document_layout(self) -> None:
        # Current GeM PDFs: "<Hindi>/<English label>", value on the next line(s).
        text = ("बिड विवरण/Bid Details\nबिड बंद होने की तारीख/समय /Bid End Date/Time\n16-11-2026 09:00:00\n"
                "संगठन का नाम/Organisation Name\nN/A\nविभाग का नाम/Department Name\nDepartment Of Heavy Industry\n"
                "कुल मात्रा/Total Quantity\n1,250\nवस्तु श्रेणी /Item Category\nModular Switch 6A ,\n"
                "Modular Socket 16A\nजेम/GeMARPTS\nबिड संख्या/Bid Number: GEM/2026/B/8196069\nदिनांक/Dated: 24-10-2026\n")
        meta = parse_gem_text(text)
        self.assertEqual(meta["reference"], "GEM/2026/B/8196069")
        self.assertEqual(meta["deadline"], "2026-11-16")
        self.assertEqual(meta["published"], "2026-10-24")
        self.assertEqual(meta["buyer"], "Department Of Heavy Industry")   # "N/A" organisation skipped
        self.assertEqual(meta["category"], "Modular Switch 6A , Modular Socket 16A")
        self.assertEqual(meta["total_quantity"], 1250)

    def test_reads_card_text(self) -> None:
        self.assertEqual(_card_field(self.CARD, "Items:"), "Modular Switch (Q3)")
        self.assertEqual(_card_department(self.CARD), "Ministry Of Railways Northern Railway")
        self.assertEqual(_card_field(self.CARD, "End Date:"), "21-11-2026 6:00 PM")


if __name__ == "__main__":
    unittest.main()
