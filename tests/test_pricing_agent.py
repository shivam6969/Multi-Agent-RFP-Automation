from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from dataclasses import replace

from catalogue import get_catalogue
from pricing_agent import _resolve_qty, reprice_with_margin
from quote_export import export_quote_csv
from settings import PRICING_RULES


class PricingAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_line_items = [
            {
                "requirement": "Outdoor LED Streetlight 120W",
                "qty": 10,
                "currency": "INR",
                "base_unit_price": 1000.0,
                "margin_pct": 20.0,
                "volume_discount_pct": 0.0,
                "net_unit_price": 1200.0,
                "gst_pct": 18.0,
                "gst_amount_per_unit": 216.0,
                "final_unit_price": 1416.0,
                "line_total": 14160.0,
                "raw_price_info": "INR 1000",
            }
        ]

    def test_reprice_margin_changes_totals(self) -> None:
        items_10, summary_10 = reprice_with_margin(
            self.sample_line_items, new_margin_pct=10.0, rules=PRICING_RULES
        )
        items_20, summary_20 = reprice_with_margin(
            self.sample_line_items, new_margin_pct=20.0, rules=PRICING_RULES
        )

        self.assertEqual(round(items_10[0]["final_unit_price"], 2), 1298.0)
        self.assertEqual(round(items_20[0]["final_unit_price"], 2), 1416.0)
        self.assertLess(summary_10["grand_total"], summary_20["grand_total"])

    def test_qty_math_and_rounding(self) -> None:
        updated, summary = reprice_with_margin(
            self.sample_line_items, new_margin_pct=12.5, rules=PRICING_RULES
        )
        item = updated[0]

        self.assertAlmostEqual(item["line_total"], item["final_unit_price"] * item["qty"], places=2)
        self.assertEqual(round(item["final_unit_price"], 2), item["final_unit_price"])
        self.assertEqual(round(summary["grand_total"], 2), summary["grand_total"])

    def test_export_quote_csv_writes_expected_rows(self) -> None:
        updated, summary = reprice_with_margin(
            self.sample_line_items, new_margin_pct=15.0, rules=PRICING_RULES
        )
        payload = {"line_items": updated, "pricing_summary": summary}

        with tempfile.TemporaryDirectory() as tmp:
            output = export_quote_csv(payload, str(Path(tmp) / "quote.csv"))
            self.assertTrue(Path(output).exists())

            with open(output, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["requirement"], "Outdoor LED Streetlight 120W")
        self.assertEqual(rows[0]["qty"], "10")

    def test_reprice_keeps_catalogue_fields(self) -> None:
        item = {**self.sample_line_items[0], "product_id": "p1", "sap_code": "AHLX001",
                "packing": "20/120", "qty_source": "default"}
        updated, _ = reprice_with_margin([item], new_margin_pct=30.0, rules=PRICING_RULES)
        self.assertEqual(updated[0]["sap_code"], "AHLX001")
        self.assertEqual(updated[0]["packing"], "20/120")
        self.assertEqual(updated[0]["qty_source"], "default")
        self.assertEqual(updated[0]["margin_pct"], 30.0)

    def test_margin_is_clamped_to_rules(self) -> None:
        _, summary = reprice_with_margin(self.sample_line_items, 95.0, PRICING_RULES)
        self.assertEqual(summary["margin_pct"], PRICING_RULES.max_margin_pct)


class GoldenPricingTests(unittest.TestCase):
    """
    Corrected version of Table 5 in the Project-III report, using real
    catalogue prices (Coral Ebony) and the volume tiers in settings.py.
    """

    CASES = [
        # (range, item_name, qty, expected volume discount %, expected final unit price)
        ("Coral Ebony", "10A 1Way Switch",         499, 5.0,  80.71),
        ("Coral Ebony", "16A 2Way Switch",         300, 5.0, 256.93),
        ("Coral Ebony", "32A DP With Ind. Switch", 100, 0.0, 572.06),
        ("Coral Ebony", "16A 2Way Switch",         199, 0.0, 270.46),  # tier boundary
        ("Coral Ebony", "16A 2Way Switch",         200, 5.0, 256.93),  # tier boundary
        ("Coral Ebony", "10A 1Way Switch",        1000, 12.0, 74.76),
    ]

    def test_final_unit_prices(self) -> None:
        by_name = {(p.range_name, p.item_name): p for p in get_catalogue().all_products()}
        for range_name, name, qty, disc, final in self.CASES:
            product = by_name[(range_name, name)]
            line = {"requirement": name, "qty": qty, "base_unit_price": product.unit_price}
            [priced], _ = reprice_with_margin([line], 20.0, PRICING_RULES)
            with self.subTest(item=name, qty=qty):
                self.assertEqual(priced["volume_discount_pct"], disc)
                self.assertEqual(priced["final_unit_price"], final)
                self.assertAlmostEqual(priced["line_total"], round(final * qty, 2), places=2)


class PaymentTermsTests(unittest.TestCase):
    LINE = {"requirement": "Switch", "qty": 100, "base_unit_price": 100.0}

    def test_no_terms_means_total_equals_sum_of_lines(self) -> None:
        items, summary = reprice_with_margin([self.LINE], 20.0, PRICING_RULES)
        self.assertEqual(summary["grand_total"], items[0]["line_total"])
        self.assertEqual(summary["payment_discount_value"], 0.0)
        self.assertEqual(summary["freight"], 0.0)

    def test_advance_payment_discount(self) -> None:
        # 100 x Rs 120 (20% margin) = 12,000 goods; 2% = 240 off; GST 18% on 11,760.
        _, summary = reprice_with_margin([self.LINE], 20.0, PRICING_RULES, advance_payment=True)
        self.assertEqual(summary["payment_discount_value"], 240.0)
        self.assertEqual(summary["taxable_value"], 11760.0)
        self.assertEqual(summary["total_gst"], 2116.8)
        self.assertEqual(summary["grand_total"], 13876.8)
        self.assertEqual(summary["payment_discount_incl_gst"], 283.2)

    def test_freight_is_added_with_gst(self) -> None:
        _, summary = reprice_with_margin([self.LINE], 20.0, PRICING_RULES, freight=500.0)
        self.assertEqual(summary["freight"], 500.0)
        self.assertEqual(summary["freight_incl_gst"], 590.0)
        self.assertEqual(summary["grand_total"], 14160.0 + 590.0)

    def test_rules_default_freight_is_used(self) -> None:
        rules = replace(PRICING_RULES, freight_flat_inr=250.0)
        _, summary = reprice_with_margin([self.LINE], 20.0, rules)
        self.assertEqual(summary["freight"], 250.0)


class QuantityTests(unittest.TestCase):
    def resolve(self, item, query=""):
        return _resolve_qty(item, query, PRICING_RULES)

    def test_rfp_quantity_wins(self) -> None:
        self.assertEqual(self.resolve({"requirement": "Switch", "qty": 350}), (350, "rfp"))
        self.assertEqual(self.resolve({"requirement": "Switch", "quantity": "40"}), (40, "rfp"))

    def test_explicit_units_in_text(self) -> None:
        self.assertEqual(self.resolve({"requirement": "Supply 1,200 pcs of 6A sockets"}),
                         (1200, "requirement"))
        self.assertEqual(self.resolve({"requirement": "Switch"}, "Need 2 units of each"),
                         (2, "query"))

    def test_numbers_in_product_names_are_not_quantities(self) -> None:
        for text in ("Supply RJ-45 data jacks", "16A and 25A switches",
                     "Outer plates (1M to 18M sizes)", "Packing 20/120 units per box"):
            with self.subTest(text=text):
                self.assertEqual(self.resolve({"requirement": text}),
                                 (PRICING_RULES.default_qty, "default"))


if __name__ == "__main__":
    unittest.main()
