from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from catalogue import (
    Catalogue,
    demo_stock_for,
    ensure_loaded,
    expand_abbreviations,
    flatten_catalogue,
    get_catalogue,
    parse_price,
)
from db import get_engine
from settings import BU_JSON_PATH


class AbbreviationTests(unittest.TestCase):
    def test_expansions(self) -> None:
        self.assertEqual(expand_abbreviations("16Ax 1way Switch with Ind.."),
                         "16Ax 1way Switch with Indicator")
        self.assertEqual(expand_abbreviations("32A DP Switch with ind. (Mega)"),
                         "32A Double Pole Switch with Indicator (Mega)")
        self.assertEqual(expand_abbreviations("4 Step Energy Saving Fan Reg."),
                         "4 Step Energy Saving Fan Regulator")

    def test_full_words_are_left_alone(self) -> None:
        for name in ("Indicator Lamp", "Fan Regulator", "10A 1Way Switch"):
            self.assertEqual(expand_abbreviations(name), name)

    def test_search_text_carries_both_names(self) -> None:
        p = next(p for p in get_catalogue().all_products() if "Ind." in p.item_name)
        text = p.search_text()
        self.assertIn(p.item_name, text)
        self.assertIn("Also described as:", text)
        self.assertIn("Indicator", text)


class ParsePriceTests(unittest.TestCase):
    def test_catalogue_formats(self) -> None:
        self.assertEqual(parse_price("54/-"), 54.0)
        self.assertEqual(parse_price("1,250/-"), 1250.0)
        self.assertEqual(parse_price("Rs. 60"), 60.0)
        self.assertEqual(parse_price("₹ 99.50"), 99.5)

    def test_rejects_non_prices(self) -> None:
        with self.assertRaises(ValueError):
            parse_price("on request")


class CatalogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalogue = get_catalogue()

    def test_loads_every_product_with_numeric_price(self) -> None:
        products = self.catalogue.all_products()
        self.assertEqual(len(products), 353)
        self.assertTrue(all(p.unit_price > 0 for p in products))

    def test_known_prices(self) -> None:
        by_name = {(p.range_name, p.item_name): p for p in self.catalogue.all_products()}
        self.assertEqual(by_name[("Coral Ebony", "10A 1Way Switch")].unit_price, 60.0)
        self.assertEqual(by_name[("Coral Ebony", "16A 2Way Switch")].unit_price, 191.0)
        self.assertEqual(by_name[("Coral Ebony", "32A DP With Ind. Switch")].unit_price, 404.0)

    def test_duplicate_sap_codes_are_kept_as_separate_products(self) -> None:
        rj45 = self.catalogue.find_by_sap("AHLKMCG451")
        self.assertEqual(len(rj45), 2)
        self.assertEqual({p.unit_price for p in rj45}, {676.0, 765.0})

    def test_get_by_id_round_trips(self) -> None:
        first = self.catalogue.all_products()[0]
        self.assertEqual(self.catalogue.get(first.id), first)
        self.assertIsNone(self.catalogue.get("does-not-exist"))

    def test_every_product_has_demo_stock(self) -> None:
        products = self.catalogue.all_products()
        self.assertTrue(all(p.stock_is_demo for p in products))
        self.assertTrue(all(p.monthly_capacity and p.monthly_capacity > 0 for p in products))
        # Some products are deliberately short so the UI can show it.
        self.assertTrue(any(p.monthly_capacity <= 300 for p in products))


class DemoStockTests(unittest.TestCase):
    def test_is_deterministic(self) -> None:
        self.assertEqual(demo_stock_for("abc123"), demo_stock_for("abc123"))
        self.assertNotEqual(demo_stock_for("abc123", seed=1), demo_stock_for("abc123", seed=2))

    def test_on_hand_never_exceeds_monthly_capacity(self) -> None:
        for i in range(200):
            s = demo_stock_for(f"p{i}")
            self.assertLessEqual(s["on_hand"], s["monthly_capacity"])


class LoaderTests(unittest.TestCase):
    def test_reload_only_when_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = json.loads(Path(BU_JSON_PATH).read_text(encoding="utf-8-sig"))
            json_path = Path(tmp) / "cat.json"
            json_path.write_text(json.dumps(data), encoding="utf-8")
            engine = get_engine(str(Path(tmp) / "cat.db"))

            first = ensure_loaded(engine, str(json_path))
            self.assertTrue(first.loaded)
            self.assertEqual(
                first.duplicate_sap_codes, ["AHLKMCG451", "AHLPIOGV12", "AHLPIOGV18"]
            )
            self.assertFalse(ensure_loaded(engine, str(json_path)).loaded)

            # Change one price: the DB must pick it up on the next load.
            data["product_data"][0]["colors"][0]["items"][0]["unit_price"] = "999/-"
            json_path.write_text(json.dumps(data), encoding="utf-8")
            self.assertTrue(ensure_loaded(engine, str(json_path)).loaded)
            prices = {p.unit_price for p in Catalogue(engine, str(json_path)).all_products()}
            self.assertIn(999.0, prices)
            engine.dispose()

    def test_flatten_handles_items_colors_and_categories(self) -> None:
        data = {
            "brand": "B",
            "product_data": [
                {"range": "R1", "items": [_item("A")]},
                {"range": "R2", "colors": [{"color": "Red", "items": [_item("B")]}]},
                {"range": "R3", "categories": [{"category": "Sw", "items": [_item("C")]}]},
            ],
        }
        rows = flatten_catalogue(data)
        self.assertEqual([(r["range_name"], r["variant"]) for r in rows],
                         [("R1", None), ("R2", "Red"), ("R3", "Sw")])


def _item(name: str) -> dict:
    return {"item_name": name, "sap_code": "S", "hsn_code": "8536",
            "packing": "1/10", "unit_price": "10/-"}


if __name__ == "__main__":
    unittest.main()
