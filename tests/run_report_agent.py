from __future__ import annotations

import os
import sys

from report_agent import report_agent_node


def _prompt(label: str, fallback: str) -> str:
    val = input(f"{label} (leave blank for sample): ").strip()
    return val if val else fallback


def main() -> int:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        api_key = input("Gemini API key: ").strip()

    if not api_key:
        print("GEMINI_API_KEY is required to generate the report.")
        return 1

    state = {
        "user_query": _prompt("User query", "Generate a proposal"),
        "rfp_answer": _prompt("RFP answer", "RFP analysis summary"),
        "rfp_requirements": [
            {"requirement": _prompt("Requirement", "Provide ISO certified products"), "weight": 1}
        ],
        "bu_answer": _prompt("BU answer", "Catalog summary"),
        "fulfillment_report": _prompt("Fulfillment report", "All requirements met"),
        "pricing_report": _prompt("Pricing report", "Pricing report summary"),
        "pricing_summary": {
            "currency": "INR",
            "items_count": 1,
            "margin_pct": 20.0,
            "subtotal_ex_gst": 100.0,
            "total_gst": 18.0,
            "grand_total": 118.0,
        },
        "line_items": [],
    }

    out = report_agent_node(state, output_dir=".", api_key=api_key)

    if out.get("error"):
        print(f"Error: {out['error']}")
        return 1

    print("\nREPORT OUTPUT")
    print("-" * 60)
    print(out.get("report_pdf_path", "No PDF path"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
