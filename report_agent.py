"""
agents/report_agent.py
----------------------
Report Generation Agent — LangGraph node

Responsibility
--------------
Final stage of the pipeline. Synthesises ALL available state data into a
polished, submission-ready vendor proposal PDF.

Key behaviours
--------------
1. Checks whether the RFP specifies a response format — if so, follows it.
2. Pulls in: RFP analysis, BU catalog answers, fulfillment matching, pricing.
3. Calls Google Gemini Flash (free, generous context window) to generate the
   proposal text — deliberately avoids Groq to preserve token budget.
4. Renders the proposal to a styled PDF via reportlab.
5. Writes the output path to state["report_pdf_path"].

Why Gemini Flash?
-----------------
By the time this node runs the pipeline has already consumed significant
Groq tokens on retrieval + matching + pricing.  Gemini 1.5 Flash has a
1 M-token context window, strong instruction-following, and a free tier —
making it ideal for this document-generation step.

Usage
-----
Set env var GEMINI_API_KEY  (or add it to settings.py as GEMINI_API_KEY).
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ── reportlab imports ──────────────────────────────────────────────────────────
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── project imports ────────────────────────────────────────────────────────────
from state import AgentState

# ---------------------------------------------------------------------------
# Optional settings import — gracefully degrade if Gemini config not present
# ---------------------------------------------------------------------------
try:
    from settings import GEMINI_API_KEY, GEMINI_MODEL  # type: ignore
except ImportError:
    GEMINI_API_KEY = ""
    GEMINI_MODEL = "gemini-2.5-flash"


# ─── Gemini client helper ─────────────────────────────────────────────────────

def _call_gemini(prompt: str, api_key: str, model: str = GEMINI_MODEL) -> str:
    """
    Call the Gemini REST API and return the generated text.
    Uses google-genai SDK if installed, falls back to raw requests.
    """
    try:
        from google import genai  # type: ignore

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": 0.3, "max_output_tokens": 8192},
        )
        return response.text or ""
    except ImportError:
        pass

    # Fallback: raw HTTP (no SDK dependency)
    import urllib.request

    url = (
        f"https://generativelanguage.googleapis.com/v1/models/"
        f"{model}:generateContent?key={api_key}"
    )
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 8192},
        }
    ).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["candidates"][0]["content"]["parts"][0]["text"]


# ─── RFP format detection ─────────────────────────────────────────────────────

def _detect_rfp_format(rfp_answer: str, rfp_requirements: List[Dict]) -> Optional[str]:
    """
    Look for explicit submission format instructions in the RFP answer /
    requirements.  Returns a plain-text description of the format or None.
    """
    format_keywords = [
        "submission format", "response format", "vendor shall submit",
        "proposal must include", "format:", "section order",
    ]
    combined = rfp_answer + " ".join(
        r.get("requirement", "") for r in rfp_requirements
    )
    lower = combined.lower()
    if any(kw in lower for kw in format_keywords):
        # Extract the relevant sentences
        sentences = re.split(r"(?<=[.!?])\s+", combined)
        relevant = [s for s in sentences if any(kw in s.lower() for kw in format_keywords)]
        if relevant:
            return " ".join(relevant[:6])
    return None


# ─── Prompt builder ───────────────────────────────────────────────────────────

_BASE_PROMPT = """
You are a senior procurement consultant drafting a formal vendor proposal response
to an RFP on behalf of the supplier.

Your goal: produce a COMPLETE, SUBMISSION-READY vendor proposal document.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### RFP Analysis
{rfp_answer}

### Product Catalog Highlights
{bu_answer}

### Fulfillment Report
{fulfillment_report}

### Pricing Summary
{pricing_summary_text}

### Detailed Pricing
{pricing_report}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{format_instruction}

Write the proposal as structured sections with clear headings.
Use professional, concise language but BE COMPREHENSIVE and DETAILED. Expand significantly on each section. The final document should be approximately 3 to 5 pages long. Be specific with numbers, product names, and capabilities.

Include ALL of:
  1. Executive Summary (At least 3-4 paragraphs highlighting key strengths and the value proposition)
  2. Company / Vendor Overview (Provide a detailed, professional background of Havells India Ltd. as an authorized distributor, highlighting history and scale)
  3. Understanding of Requirements (Thoroughly outline the scope and expectations derived from the RFP)
  4. Product Offering & Compliance (Detail the products proposed, highlighting exact specifications, SAP codes, and how they meet or exceed standards)
  5. Fulfillment Capacity (Provide assurance on stock levels and monthly availability)
  6. Pricing Schedule (Leave a placeholder here, as structured table format using the data will be injected automatically)
  7. Delivery & Logistics Commitment (Commit to firm delivery timelines, packaging norms, and shipping details)
  8. Eligibility & Compliance Declarations (List necessary certifications, ISO standards, and legal compliance points)
  9. Closing Statement & Call to Action (Professional sign-off)

Format each section with:
  SECTION: <Title>
  <content>
  END_SECTION

Do NOT add markdown ```code blocks```.  Use plain text only.
""".strip()


def _build_prompt(state: AgentState, rfp_format: Optional[str]) -> str:
    rfp_answer = state.get("rfp_answer", "Not available.")
    bu_answer = state.get("bu_answer", "Not available.")
    fulfillment_report = state.get("fulfillment_report", "Not available.")
    pricing_report = state.get("pricing_report", "Not available.")

    summary = state.get("pricing_summary") or {}
    pricing_summary_text = (
        f"Currency: {summary.get('currency', 'INR')}\n"
        f"Items quoted: {summary.get('items_count', 0)}\n"
        f"Margin applied: {summary.get('margin_pct', 0)}%\n"
        f"Subtotal (ex-GST): {summary.get('subtotal_ex_gst', 0):,.2f}\n"
        f"Total GST: {summary.get('total_gst', 0):,.2f}\n"
        f"Grand Total: {summary.get('grand_total', 0):,.2f}"
        if summary
        else "Pricing data not available."
    )

    format_instruction = (
        f"IMPORTANT — The RFP specifies the following response format. "
        f"You MUST follow this structure:\n{rfp_format}\n"
        if rfp_format
        else (
            "No specific submission format was detected in the RFP. "
            "Use the standard section structure defined below."
        )
    )

    return _BASE_PROMPT.format(
        rfp_answer=rfp_answer,
        bu_answer=bu_answer,
        fulfillment_report=fulfillment_report,
        pricing_report=pricing_report,
        pricing_summary_text=pricing_summary_text,
        format_instruction=format_instruction,
    )


# ─── Text parser ──────────────────────────────────────────────────────────────

def _parse_sections(text: str) -> List[Dict[str, str]]:
    """
    Parse SECTION: / END_SECTION markers from LLM output.
    Falls back to heading-based splitting if markers are absent.
    """
    sections: List[Dict[str, str]] = []

    # Try structured markers first
    pattern = re.compile(
        r"SECTION:\s*(.+?)\n(.*?)END_SECTION", re.DOTALL | re.IGNORECASE
    )
    matches = list(pattern.finditer(text))
    if matches:
        for m in matches:
            sections.append({"title": m.group(1).strip(), "body": m.group(2).strip()})
        return sections

    # Fallback: split on numbered headings or ALL-CAPS lines
    lines = text.splitlines()
    current_title = "Overview"
    current_body: List[str] = []

    heading_re = re.compile(r"^\s*(?:\d+\.?\s+)?([A-Z][A-Z\s&/]{3,})\s*$")

    for line in lines:
        if heading_re.match(line) and len(line.strip()) < 80:
            if current_body:
                sections.append(
                    {"title": current_title, "body": "\n".join(current_body).strip()}
                )
            current_title = line.strip().title()
            current_body = []
        else:
            current_body.append(line)

    if current_body:
        sections.append({"title": current_title, "body": "\n".join(current_body).strip()})

    return sections if sections else [{"title": "Proposal", "body": text}]


# ─── PDF builder ──────────────────────────────────────────────────────────────

# Brand colour palette
_NAVY = colors.HexColor("#1A3557")
_GOLD = colors.HexColor("#C8942A")
_LIGHT_GREY = colors.HexColor("#F4F6F8")
_MID_GREY = colors.HexColor("#8A9BB0")


def _make_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DocTitle",
            parent=base["Title"],
            fontSize=22,
            textColor=_NAVY,
            spaceAfter=4 * mm,
            fontName="Helvetica-Bold",
        ),
        "subtitle": ParagraphStyle(
            "DocSubtitle",
            parent=base["Normal"],
            fontSize=11,
            textColor=_MID_GREY,
            spaceAfter=6 * mm,
            fontName="Helvetica",
        ),
        "section_heading": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading1"],
            fontSize=13,
            textColor=_NAVY,
            spaceBefore=6 * mm,
            spaceAfter=2 * mm,
            fontName="Helvetica-Bold",
            borderPad=2,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#2C2C2C"),
            spaceAfter=3 * mm,
            fontName="Helvetica",
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            leftIndent=12,
            bulletIndent=4,
            textColor=colors.HexColor("#2C2C2C"),
            fontName="Helvetica",
        ),
        "footer": ParagraphStyle(
            "Footer",
            parent=base["Normal"],
            fontSize=8,
            textColor=_MID_GREY,
            alignment=1,
        ),
    }


def _header_footer(canvas, doc):
    """Draw a thin header bar and page footer on every page."""
    canvas.saveState()
    w, h = A4

    # Top bar
    canvas.setFillColor(_NAVY)
    canvas.rect(0, h - 14 * mm, w, 14 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(15 * mm, h - 9 * mm, "VENDOR PROPOSAL — CONFIDENTIAL")
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(
        w - 15 * mm, h - 9 * mm,
        datetime.now(timezone.utc).strftime("%d %b %Y"),
    )

    # Gold accent line under header
    canvas.setStrokeColor(_GOLD)
    canvas.setLineWidth(1.5)
    canvas.line(0, h - 14.5 * mm, w, h - 14.5 * mm)

    # Footer
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(_MID_GREY)
    canvas.drawCentredString(w / 2, 8 * mm, f"Page {doc.page}")
    canvas.setStrokeColor(_MID_GREY)
    canvas.setLineWidth(0.5)
    canvas.line(15 * mm, 13 * mm, w - 15 * mm, 13 * mm)

    canvas.restoreState()


def _cover_page(story: list, styles: Dict, state: AgentState) -> None:
    """Build a professional cover page."""
    story.append(Spacer(1, 30 * mm))

    story.append(
        Paragraph("VENDOR PROPOSAL", styles["title"])
    )
    story.append(
        Paragraph(
            "In Response to Request for Proposal:<br/>"
            "Supply of Modular Electrical Switches, Plates &amp; Accessories",
            styles["subtitle"],
        )
    )

    story.append(HRFlowable(width="100%", thickness=2, color=_GOLD, spaceAfter=6 * mm))

    meta = [
        ["Prepared for", "ABC Infrastructure Pvt. Ltd."],
        ["Prepared by", "Havells India Ltd. — Authorized Distributor"],
        ["Date", datetime.now(timezone.utc).strftime("%d %B %Y")],
        ["Fulfillment Score", f"{state.get('fulfillment_score', 'N/A')}%"],
        ["Grand Total (INR)", f"₹ {(state.get('pricing_summary') or {}).get('grand_total', 0):,.2f}"],
    ]

    tbl = Table(meta, colWidths=[55 * mm, 110 * mm])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), _LIGHT_GREY),
                ("TEXTCOLOR", (0, 0), (0, -1), _NAVY),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, _LIGHT_GREY]),
                ("GRID", (0, 0), (-1, -1), 0.5, _MID_GREY),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(tbl)
    story.append(PageBreak())


def _pricing_table(story: list, styles: Dict, state: AgentState) -> None:
    """Insert a structured pricing table from line_items if available."""
    line_items: List[Dict] = state.get("line_items") or []
    if not line_items:
        return

    story.append(Paragraph("Pricing Schedule", styles["section_heading"]))
    story.append(HRFlowable(width="100%", thickness=1, color=_GOLD, spaceAfter=3 * mm))

    headers = [
        "Requirement", "SAP", "Pack", "Qty", "Base (₹)",
        "Mrg%", "Disc%", "Net (₹)", "GST", "Total (₹)",
    ]
    
    has_discount = any(item.get("volume_discount_pct", 0) > 0 for item in line_items)
    if not has_discount:
        headers = [
            "Requirement", "SAP", "Pack", "Qty", "Base (₹)",
            "Mrg%", "Net (₹)", "GST", "Total (₹)",
        ]
        
    rows = [headers]
    for item in line_items:
        req = item.get("requirement", "")
        # Hard truncate single-word strings since textwrap eats whole words
        if len(req) > 36: req = req[:34] + "…"
            
        sap = item.get("sap_code", "N/A")
        if len(sap) > 13: sap = sap[:11] + "…"
            
        pack = item.get("packing", "N/A")
        if len(pack) > 9: pack = pack[:7] + "…"
        
        row = [
            req,
            sap,
            pack,
            str(item.get("qty", "")),
            f"{item.get('base_unit_price', 0):,.2f}",
            f"{item.get('margin_pct', 0)}%",
        ]
        
        if has_discount:
            row.append(f"{item.get('volume_discount_pct', 0)}%")
            
        row.extend([
            f"{item.get('net_unit_price', 0):,.2f}",
            f"{item.get('gst_pct', 0)}%",
            f"{item.get('line_total', 0):,.2f}",
        ])
        
        rows.append(row)

    # Totals row
    summary = state.get("pricing_summary") or {}
    total_row = [
        "GRAND TOTAL", "", "", "", "", "", "", "",
        f"{summary.get('grand_total', 0):,.2f}",
    ]
    if has_discount:
        total_row.insert(6, "") # Add empty cell for the discount column
        
    rows.append(total_row)

    if has_discount:
        col_w = [40 * mm, 23 * mm, 13 * mm, 9 * mm, 16 * mm, 11 * mm, 11 * mm, 16 * mm, 10 * mm, 22 * mm]
    else:
        col_w = [48 * mm, 23 * mm, 13 * mm, 9 * mm, 17 * mm, 11 * mm, 18 * mm, 10 * mm, 22 * mm]
        
    tbl = Table(rows, colWidths=col_w, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                # Header
                ("BACKGROUND", (0, 0), (-1, 0), _NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                # Body
                ("FONTNAME", (0, 1), (-1, -2), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -2), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, _LIGHT_GREY]),
                # Totals
                ("BACKGROUND", (0, -1), (-1, -1), _GOLD),
                ("TEXTCOLOR", (0, -1), (-1, -1), colors.white),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, -1), (-1, -1), 9),
                # Grid
                ("GRID", (0, 0), (-1, -1), 0.4, _MID_GREY),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(tbl)
    story.append(Spacer(1, 4 * mm))


def _render_section(story: list, styles: Dict, title: str, body: str) -> None:
    """Render one proposal section into the story."""
    story.append(Paragraph(title, styles["section_heading"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_GOLD, spaceAfter=2 * mm))

    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 2 * mm))
            continue
        if stripped.startswith(("•", "-", "*", "●")):
            story.append(
                Paragraph(f"&bull;&nbsp;&nbsp;{stripped.lstrip('•-*● ').strip()}",
                          styles["bullet"])
            )
        else:
            story.append(Paragraph(stripped, styles["body"]))


def _build_pdf(
    output_path: str,
    sections: List[Dict[str, str]],
    state: AgentState,
) -> str:
    """Render the full proposal to a PDF file and return the path."""
    styles = _make_styles()

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
    )

    story: list = []
    _cover_page(story, styles, state)

    # Pricing section gets a dedicated styled table; skip the text version
    pricing_section_titles = {"pricing schedule", "pricing", "price", "quotation"}

    for sec in sections:
        title = sec["title"]
        body = sec["body"]

        if title.lower() in pricing_section_titles:
            _pricing_table(story, styles, state)
            continue

        _render_section(story, styles, title, body)

    # If no pricing section was produced by the LLM, append it at the end
    lm_titles = {s["title"].lower() for s in sections}
    if not lm_titles.intersection(pricing_section_titles):
        _pricing_table(story, styles, state)

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return output_path


# ─── LangGraph node ───────────────────────────────────────────────────────────

def report_agent_node(
    state: AgentState,
    output_dir: str = ".",
    api_key: Optional[str] = None,
) -> AgentState:
    """
    LangGraph node.

    Reads  : rfp_answer, rfp_requirements, bu_answer, fulfillment_report,
             pricing_report, pricing_summary, line_items, user_query
    Writes : report_pdf_path, report_text
    """
    # Resolve Gemini API key (env > arg > settings.py)
    gemini_key = (
        api_key
        or os.environ.get("GEMINI_API_KEY")
        or GEMINI_API_KEY
    )
    if not gemini_key or gemini_key in ("", "YOUR_GEMINI_API_KEY_HERE"):
        state["error"] = (
            "GEMINI_API_KEY is not set. "
            "Set the env var or add it to settings.py as GEMINI_API_KEY."
        )
        print("[ReportAgent] ✗ GEMINI_API_KEY missing — aborting.")
        return state

    # 1. Detect RFP response format
    rfp_format = _detect_rfp_format(
        state.get("rfp_answer", ""),
        state.get("rfp_requirements", []),
    )
    if rfp_format:
        print(f"[ReportAgent] RFP format detected: {rfp_format[:80]}…")
    else:
        print("[ReportAgent] No specific RFP format detected — using standard structure.")

    # 2. Build and send prompt to Gemini
    prompt = _build_prompt(state, rfp_format)
    print("[ReportAgent] Calling Gemini Flash to generate proposal text …")
    try:
        proposal_text = _call_gemini(prompt, gemini_key)
    except Exception as exc:
        state["error"] = f"Gemini API call failed: {exc}"
        print(f"[ReportAgent] ✗ Gemini error: {exc}")
        return state
    print("[ReportAgent] ✓ Proposal text generated.")

    # 3. Parse into sections
    sections = _parse_sections(proposal_text)
    print(f"[ReportAgent] Parsed {len(sections)} proposal sections.")

    # 4. Render PDF
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(output_dir, f"vendor_proposal_{ts}.pdf")

    print(f"[ReportAgent] Rendering PDF → {pdf_path} …")
    try:
        _build_pdf(pdf_path, sections, state)
    except Exception as exc:
        state["error"] = f"PDF rendering failed: {exc}"
        print(f"[ReportAgent] ✗ PDF error: {exc}")
        return state

    print(f"[ReportAgent] ✓ PDF ready: {pdf_path}")
    state["report_pdf_path"] = pdf_path
    state["report_text"] = proposal_text
    return state
