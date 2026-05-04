from __future__ import annotations

import unittest
from unittest.mock import patch

from report_agent import report_agent_node


class ReportAgentTests(unittest.TestCase):
    @patch("report_agent._build_pdf")
    @patch("report_agent._call_gemini")
    def test_report_agent_node_writes_report_fields(self, mock_call_gemini, mock_build_pdf):
        mock_call_gemini.return_value = (
            "SECTION: Executive Summary\n"
            "Summary text\n"
            "END_SECTION\n"
        )
        mock_build_pdf.side_effect = lambda output_path, sections, state: output_path

        state = {
            "user_query": "Generate the proposal",
            "rfp_answer": "RFP answer",
            "rfp_requirements": [{"requirement": "Req 1", "weight": 1}],
            "bu_answer": "BU answer",
            "fulfillment_report": "Fulfillment report",
            "pricing_report": "Pricing report",
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

        out = report_agent_node(state, output_dir=".", api_key="test-key")

        self.assertIn("report_pdf_path", out)
        self.assertIn("report_text", out)
        self.assertEqual(out["report_text"], mock_call_gemini.return_value)


if __name__ == "__main__":
    unittest.main()
