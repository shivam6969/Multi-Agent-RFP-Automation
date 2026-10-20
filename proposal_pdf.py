"""
proposal_pdf.py
---------------
Renders the proposal PDF from the reviewed sections and the approved
pricing. No LLM here, and no arithmetic: every amount is copied from the
pricing agent's line items and pricing_summary.

Layout follows the dashboard design: a header with the vendor's name and the
reference / date / validity, a title, a facts table (prepared for / by,
proposal value), then the sections in order. The "pricing" section gets the
pricing table, the totals and a list of items not quoted. It ends with a
signature block.

Text from the LLM or the reviewer is XML-escaped before it reaches
reportlab (an "&" or "<" used to break rendering). Helvetica has no rupee
glyph, so amounts are written "Rs. 1,23,456.00".
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

BRAND = colors.HexColor("#6a4fe4")
INK = colors.HexColor("#1e293b")
MUTED = colors.HexColor("#64748b")
LINE = colors.HexColor("#e2e8f0")
TABLE_HEAD = colors.HexColor("#3b3031")
SOFT = colors.HexColor("#f8fafc")


def inr(amount: Optional[float]) -> str:
    """Indian digit grouping: 1234567.5 → "12,34,567.50" (formatting only)."""
    if amount is None:
        return "-"
    negative = amount < 0
    whole, frac = f"{abs(amount):.2f}".split(".")
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    text = ",".join(groups + [tail]) if groups else tail
    return f"{'-' if negative else ''}{text}.{frac}"


def _fmt_date(iso: Optional[str]) -> str:
    try:
        return date.fromisoformat(str(iso)[:10]).strftime("%d %b %Y")
    except (TypeError, ValueError):
        return "-"


def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=base["Normal"], fontName="Helvetica", fontSize=9.5, leading=14,
                          textColor=INK, spaceAfter=2.5 * mm)
    return {
        "body": body,
        "small": ParagraphStyle("small", parent=body, fontSize=8, leading=11, textColor=MUTED, spaceAfter=0),
        "bullet": ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=1.2 * mm),
        "title": ParagraphStyle("title", parent=body, fontName="Helvetica-Bold", fontSize=17, leading=21,
                                spaceAfter=1.5 * mm),
        "h2": ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=10, leading=13,
                             textColor=BRAND, spaceBefore=5 * mm, spaceAfter=1.5 * mm),
        "cell": ParagraphStyle("cell", parent=body, fontSize=8.5, leading=11, spaceAfter=0),
        "cell_bold": ParagraphStyle("cell_bold", parent=body, fontName="Helvetica-Bold", fontSize=8.5,
                                    leading=11, spaceAfter=0),
    }


def _paragraphs(text: str, st: Dict[str, ParagraphStyle]) -> List[Any]:
    out: List[Any] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line[:2] in ("- ", "* ", "• "):
            out.append(Paragraph(escape(line[2:].strip()), st["bullet"], bulletText="•"))
        else:
            out.append(Paragraph(escape(line), st["body"]))
    return out


def _pricing_block(st, line_items: List[Dict[str, Any]], summary: Dict[str, Any],
                   not_quoted: List[str]) -> List[Any]:
    def head(text: str, align: int = 2) -> Paragraph:   # white, wrapping, right-aligned by default
        return Paragraph(text, ParagraphStyle("th", parent=st["cell_bold"], textColor=colors.white,
                                              fontSize=8, leading=10, alignment=align))

    rows = [[head("Item", align=0), head("Qty"), head("Rate excl. GST"), head("GST"),
             head("Rate incl. GST"), head("Amount")]]
    for li in line_items:
        item = Paragraph(
            f"{escape(li.get('product_name') or li['requirement'])}"
            f"<br/><font size=7 color='#64748b'>{escape(li.get('sap_code') or '')}"
            f"{' · ' + escape(li['product_range']) if li.get('product_range') else ''}</font>",
            st["cell"])
        rows.append([item, f"{li['qty']:,}", inr(li["net_unit_price"]), f"{li['gst_pct']:g}%",
                     inr(li["final_unit_price"]), inr(li["line_total"])])
    table = Table(rows, colWidths=[64 * mm, 14 * mm, 27 * mm, 12 * mm, 27 * mm, 30 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEAD),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SOFT]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    totals = [["Subtotal (excl. GST)", inr(summary.get("subtotal_ex_gst"))]]
    if summary.get("payment_discount_value"):
        totals.append([f"Less: advance-payment discount ({summary.get('payment_discount_pct', 0):g}%)",
                       inr(-summary["payment_discount_value"])])
    if summary.get("freight"):
        totals.append(["Freight", inr(summary["freight"])])
    totals.append(["GST", inr(summary.get("total_gst"))])
    totals.append(["Total amount (incl. GST), Rs.", inr(summary.get("grand_total"))])
    t2 = Table(totals, colWidths=[60 * mm, 34 * mm], hAlign="RIGHT")
    t2.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("TEXTCOLOR", (0, 0), (-1, -2), MUTED),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("LINEABOVE", (0, -1), (-1, -1), 0.8, INK),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
    ]))

    block: List[Any] = [Spacer(1, 1 * mm), table, Spacer(1, 2 * mm), t2]
    if not_quoted:
        block += [Spacer(1, 3 * mm),
                  Paragraph("<b>Not quoted:</b> " + escape("; ".join(not_quoted)) +
                            ". These items are outside our current range and can be sourced on request.",
                            st["small"])]
    return block


def render_proposal(
    path: str,
    *,
    sections: List[Dict[str, Any]],
    proposal_meta: Dict[str, Any],
    line_items: List[Dict[str, Any]],
    pricing_summary: Dict[str, Any],
    not_quoted: Optional[List[str]] = None,
) -> str:
    st = _styles()
    profile = proposal_meta.get("profile") or {}
    company = profile.get("name") or "Vendor"
    ref = proposal_meta.get("reference") or ""
    rfp_title = proposal_meta.get("rfp_title") or "the RFP"

    def on_page(canvas, doc):
        canvas.saveState()
        w, h = A4
        canvas.setFillColor(BRAND)
        canvas.rect(0, h - 4 * mm, w, 4 * mm, fill=1, stroke=0)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        contact = " · ".join(x for x in (company, profile.get("email"), profile.get("phone")) if x)
        canvas.drawString(18 * mm, 10 * mm, contact)
        canvas.drawRightString(w - 18 * mm, 10 * mm, f"{ref} · Page {doc.page}")
        canvas.setStrokeColor(LINE)
        canvas.line(18 * mm, 13 * mm, w - 18 * mm, 13 * mm)
        canvas.restoreState()

    initials = "".join(w[0] for w in company.split()[:2]).upper()
    header = Table(
        [[Paragraph(f"<font color='white'><b>{escape(initials)}</b></font>", st["cell_bold"]),
          Paragraph(f"<b>{escape(company)}</b><br/><font size=7.5 color='#64748b'>{escape(profile.get('tagline') or '')}</font>", st["cell"]),
          Paragraph(f"<font size=8 color='#64748b'>Ref: {escape(ref)}<br/>Date: {_fmt_date(proposal_meta.get('generated'))}"
                    f"<br/>Valid until: {_fmt_date(proposal_meta.get('valid_until'))}</font>", st["cell"])]],
        colWidths=[10 * mm, 110 * mm, 54 * mm],
    )
    header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), BRAND),
        ("ALIGN", (0, 0), (0, 0), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (2, 0), (2, 0), "RIGHT"),
    ]))

    def cell(text: str, bold: bool = False) -> Paragraph:
        return Paragraph(escape(text), st["cell_bold" if bold else "cell"])

    facts = Table([
        ["Prepared for", cell(proposal_meta.get("buyer") or "-"), "Prepared by", cell(company)],
        ["RFP reference", cell(proposal_meta.get("rfp_reference") or "-"), "Proposal value (incl. GST)",
         cell(f"Rs. {inr(pricing_summary.get('grand_total'))}", bold=True)],
    ], colWidths=[26 * mm, 60 * mm, 38 * mm, 50 * mm])
    facts.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
        ("TEXTCOLOR", (2, 0), (2, -1), MUTED),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, -1), SOFT),
        ("BOX", (0, 0), (-1, -1), 0.4, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    story: List[Any] = [
        header,
        HRFlowable(width="100%", thickness=0.6, color=LINE, spaceBefore=3 * mm, spaceAfter=5 * mm),
        Paragraph(escape(f"Proposal for {rfp_title}"), st["title"]),
        Paragraph(escape(f"In response to {proposal_meta.get('rfp_reference') or 'the RFP'}"
                         f" · {proposal_meta.get('buyer') or ''}").rstrip(" ·"), st["small"]),
        Spacer(1, 4 * mm),
        facts,
    ]

    for section in sections:
        story.append(Paragraph(escape(section["title"]).upper(), st["h2"]))
        story += _paragraphs(section.get("body", ""), st)
        if section["key"] == "pricing":
            story += _pricing_block(st, line_items, pricing_summary, not_quoted or [])

    story.append(KeepTogether([
        Spacer(1, 10 * mm),
        Paragraph(escape(f"For {company}"), st["body"]),
        Spacer(1, 12 * mm),
        Paragraph(f"<b>{escape(profile.get('signatory_name') or '')}</b>", st["body"]),
        Paragraph(escape(profile.get("signatory_title") or ""), st["small"]),
    ]))

    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=14 * mm, bottomMargin=20 * mm,
                            title=f"Proposal for {rfp_title}", author=company)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return path


def not_quoted_items(state: Dict[str, Any]) -> List[str]:
    """Requested products with no quoted price (gaps, rejected, unpriced)."""
    items = [m["requirement"] for m in state.get("matched_items", [])
             if m.get("type") == "product" and not m.get("matched")]
    items += [q["requirement"] for q in state.get("quoted_items", []) if q.get("catalog_cost_inr") is None]
    return list(dict.fromkeys(items))
