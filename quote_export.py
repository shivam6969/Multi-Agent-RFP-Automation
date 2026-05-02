"""
quote_export.py
---------------
Utility helpers for exporting pricing outputs to files.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict


def export_quote_csv(quote_payload: Dict[str, Any], output_path: str) -> str:
    """
    Export quote payload line items to CSV and return absolute file path.
    """
    line_items = quote_payload.get("line_items", [])
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "requirement",
        "qty",
        "currency",
        "base_unit_price",
        "margin_pct",
        "volume_discount_pct",
        "net_unit_price",
        "gst_pct",
        "gst_amount_per_unit",
        "final_unit_price",
        "line_total",
    ]

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in line_items:
            writer.writerow({name: item.get(name, "") for name in fieldnames})

    return str(out)
