from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from pricing_agent import reprice_with_margin
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


if __name__ == "__main__":
    unittest.main()
