"""
tests/sample_rfps.py
--------------------
Table-heavy sample RFPs for the Havells catalogue, with their ground truth.

Each sample exercises table layouts that real tenders use and that plain text
extraction gets wrong:

  hotel   bid-details key/value table; item list (BOQ) over two pages with
          category rows, a Make column and empty Rate / Amount columns for
          the vendor; a quantity in boxes (not pieces); an item the catalogue
          does not have; a Parameter / Requirement spec table; an eligibility
          table; decimals in the body text ("2.5 sq mm", "0.5%")
  school  quantities split over three campuses under a merged "Quantity"
          header (two header rows), a category column merged over several
          rows, colour variants, and one row whose Total disagrees with its
          campus quantities
  office  an item list drawn WITHOUT gridlines, descriptions on two lines,
          quantities with the unit in the cell ("120 Nos"), and a ruled
          delivery-schedule table

Buyers, references and contacts are fictitious; every page is stamped as a
sample. Output is deterministic (same bytes every run).

    python -m tests.sample_rfps            # writes data/samples/rfp_*.pdf + *_truth.json
    python -m tests.sample_rfps --out DIR

The unit tests build the same PDFs in a temporary folder, so the files in
data/samples/ are for trying the dashboard (upload them on New Session).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

STAMP = "SAMPLE RFP for testing Synapse. Buyer, references and contacts are fictitious."


@dataclass
class TruthItem:
    description: str            # text as printed in the item table (one line)
    quantity: Optional[int]
    unit: str = "Nos"
    catalogue: Optional[str] = None   # expected catalogue item (range / item), None = not in catalogue
    note: str = ""


@dataclass
class SampleTruth:
    name: str
    title: str
    buyer: str
    items: List[TruthItem]
    tables: Dict[str, int] = field(default_factory=dict)   # expected table kinds -> count
    facts: Dict[str, str] = field(default_factory=dict)    # question -> text that must be in the answer context


# ─── Shared styling ───────────────────────────────────────────────────────────

_ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=_ss["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.8, leading=11)
H1 = ParagraphStyle("h1", parent=_ss["Title"], fontName="Helvetica-Bold", fontSize=15, leading=19, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=_ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5,
                    spaceBefore=10, spaceAfter=4)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=12, bulletIndent=2)

GRID = [
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.8),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
]


def _p(text: str, style: ParagraphStyle = CELL) -> Paragraph:
    return Paragraph(text, style)


def _bullets(items: List[str]) -> List[Paragraph]:
    return [Paragraph(i, BULLET, bulletText="•") for i in items]


def _on_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#9a3412"))
    canvas.drawCentredString(A4[0] / 2, 10 * mm, STAMP)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm - 9, f"Page {doc.page}")
    canvas.drawString(18 * mm, A4[1] - 10 * mm, doc.title)
    canvas.restoreState()


def _build(path: Path, title: str, story: List) -> None:
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=18 * mm, bottomMargin=20 * mm, title=title,
                            author="Synapse test samples", invariant=True)
    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)


def _kv_table(pairs: List[Tuple[str, str]]) -> Table:
    t = Table([[_p(f"<b>{k}</b>"), _p(v)] for k, v in pairs], colWidths=[55 * mm, 119 * mm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


# ─── 1. Hotel: multi-page BOQ with categories, spec + eligibility tables ──────

HOTEL_ITEMS: List[Tuple[str, List[TruthItem]]] = [
    ("A. Switches (Oro Metallica range, Magnesium Grey)", [
        TruthItem("6A/10A 1 Way switch", 640, catalogue="Oro Metallica / 6A/10A 1Way Switch"),
        TruthItem("6A/10A 2 Way switch", 180, catalogue="Oro Metallica / 6A/10A 2Way Switch"),
        TruthItem("6A/10A 1 Way switch with indicator", 220, catalogue="Oro Metallica / 6A/10A 1Way With Ind. Switch"),
        TruthItem("16A 1 Way switch with indicator", 260, catalogue="Oro Metallica / 16A 1Way Switch With Ind."),
        TruthItem("6A/10A Bell push switch", 120, catalogue="Oro Metallica / 6A/10A Bell Push Switch"),
        TruthItem("32A DP switch with indicator (Mega)", 40, catalogue="Oro Metallica / 32A DP Switch with ind Mega"),
    ]),
    ("B. Sockets", [
        TruthItem("6A 3 Pin socket", 300, catalogue="Oro Metallica / 6A 3 Pin Socket"),
        TruthItem("6-16A 3 Pin combined socket", 260, catalogue="Oro Metallica / 6-16A 3 Pin Combined Socket"),
        TruthItem("Universal socket", 150, catalogue="Oro Metallica / Universal Socket"),
        TruthItem("25A socket for geysers", 120, catalogue="Oro Metallica / 25A Socket"),
    ]),
    ("C. Support modules", [
        TruthItem("High speed fan regulator", 140, catalogue="Oro Metallica / High Speed Fan Regulator"),
        TruthItem("Dimmer 400 W", 60, catalogue="Oro Metallica / Dimmer 400 W"),
        # a decimal inside an item name: the old section regex split here and dropped the row
        TruthItem("USB charger, 2.1A output", 160, catalogue="Oro Metallica / USB Charger"),
        TruthItem("RJ-45 jack with Cat-6", 140, catalogue="Oro Metallica / RJ-45 Jack With Cat-6"),
        TruthItem("TV socket", 120, catalogue="Oro Metallica / TV Socket"),
        TruthItem("Telephone socket", 120, catalogue="Oro Metallica / Telephone Socket"),
        TruthItem("4M LED footlight", 80, catalogue="Oro Metallica / 4M LED Footlight"),
    ]),
    ("D. Cover plates (Magnesium Grey)", [
        TruthItem("2M cover plate", 260, catalogue="Oro Metallica Magnesium Grey / 2M Cover Plate"),
        TruthItem("4M cover plate", 220, catalogue="Oro Metallica Magnesium Grey / 4M Cover Plate"),
        TruthItem("6M cover plate", 140, catalogue="Oro Metallica Magnesium Grey / 6M Cover Plate"),
        TruthItem("8M (H) cover plate", 90, catalogue="Oro Metallica Magnesium Grey / 8M (H) Cover Plate"),
        TruthItem("12M cover plate", 40, catalogue="Oro Metallica Magnesium Grey / 12 M Cover Plate"),
    ]),
    ("E. Hospitality and accessories", [
        TruthItem("Electronic key card switch", 120, catalogue="Oro / Electronic Key Card"),
        TruthItem("Do not disturb & clean my room indicator", 120, catalogue="Oro / Do not Disturb & Clean My room"),
        TruthItem("Ceiling rose", 30, unit="Box", catalogue="Accessories / Ceiling Rose",
                  note="quantity in boxes, not pieces"),
        TruthItem("Smart door lock with RFID card reader", 120, catalogue=None, note="not in catalogue"),
    ]),
]


def build_hotel(path: Path) -> SampleTruth:
    title = "Supply of Modular Switches and Accessories for Hotel Renovation"
    buyer = "Sunrise Grand Hotels Pvt. Ltd."
    story: List = [
        Paragraph("Request for Proposal (RFP)", BODY),
        Paragraph(title, H1),
        Paragraph(f"Issued by: {buyer}, Udaipur (Rajasthan)", BODY),
        Spacer(1, 3 * mm),
        Paragraph("Bid Details", H2),
        _kv_table([
            ("Tender Reference", "SGH/ELEC/2026/117"),
            ("Date of Issue", "02-11-2026"),
            ("Last Date for Submission", "28-11-2026, 15:00 hrs"),
            ("Bid Validity", "90 days from the last date of submission"),
            ("Earnest Money Deposit (EMD)", "Rs. 25,000 (refundable, by demand draft)"),
            ("Contact", "procurement@sunrisegrand.example"),
        ]),
        Paragraph("1. Introduction", H2),
        Paragraph(
            f"{buyer} is renovating 120 guest rooms and public areas of its Udaipur property and invites "
            "sealed bids for the supply of modular switches, sockets, support modules and cover plates. "
            "Wiring of 2.5 sq mm and 1.5 sq mm copper conductors is in the electrical contractor's scope "
            "and is not part of this tender.", BODY),
        Paragraph("2. Schedule of Requirements", H2),
        Paragraph("The vendor shall quote for every item below. Rates are to be filled in by the vendor.", BODY),
        Spacer(1, 2 * mm),
    ]

    rows = [[_p(f"<b>{h}</b>") for h in
             ("S.No", "Description of Item", "Make", "Unit", "Qty", "Unit Rate (Rs.)", "Amount (Rs.)")]]
    style = list(GRID) + [("ALIGN", (4, 1), (4, -1), "RIGHT")]
    truth: List[TruthItem] = []
    n = 0
    for category, items in HOTEL_ITEMS:
        rows.append(["", _p(f"<b>{category}</b>"), "", "", "", "", ""])
        style.append(("BACKGROUND", (0, len(rows) - 1), (-1, len(rows) - 1), colors.HexColor("#f8fafc")))
        for item in items:
            n += 1
            make = "Equivalent ISI marked" if item.catalogue is None else "Havells / equivalent"
            rows.append([str(n), _p(item.description), _p(make), item.unit, f"{item.quantity:,}", "", ""])
            truth.append(item)
    rows.append(["", _p("<b>Total</b>"), "", "", "", "", ""])
    t = Table(rows, colWidths=[11 * mm, 62 * mm, 30 * mm, 13 * mm, 15 * mm, 22 * mm, 21 * mm], repeatRows=1)
    t.setStyle(TableStyle(style))
    story += [t]

    story += [
        Paragraph("3. Technical Specifications", H2),
        Paragraph("All items shall meet the following specifications:", BODY),
        Spacer(1, 2 * mm),
    ]
    spec = Table([[_p("<b>Parameter</b>"), _p("<b>Requirement</b>")]] + [
        [_p(a), _p(b)] for a, b in [
            ("Standards", "IS 3854:1997 for switches; IS 1293:2019 for sockets"),
            ("Rated voltage", "240 V AC, 50 Hz"),
            ("Switch mechanism", "Silver alloy contacts, minimum 1,00,000 operations"),
            ("Plate material", "Polycarbonate, flame retardant (UL 94 V-0)"),
            ("Warranty", "2 years from the date of delivery"),
        ]
    ], colWidths=[50 * mm, 124 * mm])
    spec.setStyle(TableStyle(GRID))
    story += [spec]

    story += [Paragraph("4. Eligibility Criteria", H2)]
    elig = Table([["S.No", "Criterion", "Documentary Evidence"]] + [
        [str(i), _p(a), _p(b)] for i, (a, b) in enumerate([
            ("Authorised distributor or manufacturer of the quoted brand", "Authorisation letter from the OEM"),
            ("Average annual turnover of at least Rs. 1.5 crore in the last three financial years",
             "CA certified turnover statement"),
            ("Valid GST registration", "GST registration certificate"),
            ("At least two similar supply orders in the last three years", "Copies of purchase orders"),
        ], 1)
    ], colWidths=[11 * mm, 95 * mm, 68 * mm])
    elig.setStyle(TableStyle(GRID))
    story += [elig]

    story += [
        Paragraph("5. Delivery and Commercial Terms", H2),
        *_bullets([
            "Delivery within 21 days of the purchase order, to the hotel stores at Udaipur.",
            "Liquidated damages of 0.5% of the order value per week of delay, up to 5%.",
            "Prices shall be firm, inclusive of packing and freight; GST to be shown separately.",
            "Payment within 30 days of receipt and acceptance of material.",
        ]),
    ]
    _build(path, title, story)
    return SampleTruth(
        name="hotel", title=title, buyer=buyer, items=truth,
        tables={"kv": 1, "boq": 1, "spec": 1, "other_or_spec": 1},
        facts={
            "What is the EMD amount?": "25,000",
            "What is the warranty period?": "2 years",
            "How many USB chargers are required?": "160",
        },
    )


# ─── 2. School: quantities per campus under a merged header ───────────────────

SCHOOL_ITEMS: List[Tuple[str, List[Tuple[TruthItem, Tuple[int, int, int], Optional[int]]]]] = [
    ("Switches (Coral)", [
        (TruthItem("6Ax/10Ax 1 way switch", None, catalogue="Coral / 6Ax/10Ax 1way Switch"), (420, 380, 260), None),
        (TruthItem("6Ax/10Ax 2 way switch", None, catalogue="Coral / 6Ax/10Ax 2way Switch"), (60, 40, 30), None),
        (TruthItem("16Ax 1 way switch with indicator", None, catalogue="Coral / 16Ax 1way Switch with Ind.."),
         (90, 70, 50), None),
        (TruthItem("6Ax/10Ax bell push switch", None, catalogue="Coral / 6Ax/10Ax Bell Push Switch"),
         (24, 20, 12), None),
    ]),
    ("Sockets (Coral)", [
        (TruthItem("6A 3 pin shuttered socket", None, catalogue="Coral / 6A 3 Pin Shuttered Socket"),
         (300, 260, 180), None),
        (TruthItem("6-16A 3 pin shuttered socket", None, catalogue="Coral / 6-16A 3 Pin Shuttered Socket"),
         (80, 60, 40), 200, ),   # printed total 200 != 180: the total column wins, with a warning
    ]),
    ("Outer plates (Coral Magna)", [
        (TruthItem("2M outer plate, Salmon Red", None, catalogue="Coral Magna Salmon Red / 2M Outer Plate"),
         (150, 0, 0), None),
        (TruthItem("4M outer plate, Midnight Blue", None, catalogue="Coral Magna Midnight Blue / 4M Outer Plate"),
         (0, 140, 0), None),
        (TruthItem("6M outer plate, Ocean Green", None, catalogue="Coral Magna Ocean Green / 6M Outer Plate"),
         (0, 0, 90), None),
    ]),
    ("Boxes and protection", [
        (TruthItem("3M flush metal GI box (4X3)", None, catalogue="Boxes / 3M Box (4X3)"), (200, 180, 120), None),
        (TruthItem("6M flush metal GI box (8X3)", None, catalogue="Boxes / 6M Box (8X3)"), (90, 80, 60), None),
        (TruthItem("16 A SP C mini MCB", None, catalogue="Coral / 16 A SP C MINI MCB Coral"), (40, 30, 30), None),
    ]),
]


def build_school(path: Path) -> SampleTruth:
    title = "Supply of Electrical Wiring Accessories for Three School Campuses"
    buyer = "Greenfield Public School Society"
    story: List = [
        Paragraph("Request for Proposal (RFP)", BODY),
        Paragraph(title, H1),
        Paragraph(f"Issued by: {buyer}, Pune. RFP No. GPSS/ADM/2026/044 dated 05-11-2026.", BODY),
        Paragraph("1. Background", H2),
        Paragraph(
            "The Society is upgrading classrooms, laboratories and hostels at its Aundh (Campus A), Baner "
            "(Campus B) and Wakad (Campus C) campuses. Material is to be delivered campus-wise as per the "
            "quantities below.", BODY),
        Paragraph("2. Bill of Quantities", H2),
        Spacer(1, 2 * mm),
    ]

    header1 = ["S.No", "Category", "Item description", "Unit", "Quantity", "", "", ""]
    header2 = ["", "", "", "", "Campus A", "Campus B", "Campus C", "Total"]
    rows = [header1, header2]
    style = list(GRID) + [
        ("SPAN", (4, 0), (7, 0)),
        ("SPAN", (0, 0), (0, 1)), ("SPAN", (1, 0), (1, 1)), ("SPAN", (2, 0), (2, 1)), ("SPAN", (3, 0), (3, 1)),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#e2e8f0")),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("ALIGN", (4, 0), (-1, -1), "CENTER"),
    ]
    truth: List[TruthItem] = []
    n = 0
    for category, items in SCHOOL_ITEMS:
        first = len(rows)
        for item, (a, b, c), printed_total in items:
            n += 1
            total = printed_total if printed_total is not None else a + b + c
            rows.append([str(n), _p(category) if len(rows) == first else "", _p(item.description), "Nos",
                         f"{a:,}" if a else "-", f"{b:,}" if b else "-", f"{c:,}" if c else "-", f"{total:,}"])
            truth.append(TruthItem(item.description, total, "Nos", item.catalogue,
                                   note="total column disagrees with campus quantities" if printed_total else ""))
        style.append(("SPAN", (1, first), (1, len(rows) - 1)))
    t = Table(rows, colWidths=[10 * mm, 27 * mm, 55 * mm, 11 * mm, 18 * mm, 18 * mm, 18 * mm, 17 * mm],
              repeatRows=2)
    t.setStyle(TableStyle(style))
    story += [t]

    story += [
        Paragraph("3. Terms and Conditions", H2),
        *_bullets([
            "All items must be ISI marked and of one brand and range per campus.",
            "Earnest money deposit of Rs. 15,000 is to be submitted with the bid.",
            "Delivery to each campus within 15 days of the purchase order; campus-wise delivery challans required.",
            "Vendors must provide a warranty of at least 12 months against manufacturing defects.",
            "Last date for submission of bids: 25-11-2026.",
        ]),
    ]
    _build(path, title, story)
    return SampleTruth(
        name="school", title=title, buyer=buyer, items=truth, tables={"boq": 1},
        facts={"How many 1 way switches are needed in total?": "1,060"},
    )


# ─── 3. Office: item list without gridlines, units inside quantity cells ──────

OFFICE_ITEMS: List[Tuple[str, str, str, TruthItem]] = [
    # (first line, second line or "", remarks, truth)
    ("Modular switch 6A/10A 1 way,", "white, Oro range", "Oro",
     TruthItem("Modular switch 6A/10A 1 way, white, Oro range", 480, catalogue="Oro / 6A/10A 1 way Switch")),
    ("Modular switch 16A 1 way", "with indicator, Oro range", "Oro",
     TruthItem("Modular switch 16A 1 way with indicator, Oro range", 160, catalogue="Oro / 16A 1 way switch with ind.")),
    ("6/16A 3 pin combined shuttered socket", "", "Oro",
     TruthItem("6/16A 3 pin combined shuttered socket", 220, catalogue="Oro / 6/16A 3 Pin Combined Shuttered Socket")),
    ("RJ-45 jack with Cat 6 for", "workstations and meeting rooms", "Oro",
     TruthItem("RJ-45 jack with Cat 6 for workstations and meeting rooms", 300, catalogue="Oro / RJ-45 Jack with cat 6")),
    ("USB charger (1M)", "", "Oro",
     TruthItem("USB charger (1M)", 120, catalogue="Oro / USB Charger (1M)")),
    ("Step fan regulator (1M)", "", "Oro",
     TruthItem("Step fan regulator (1M)", 60, catalogue="Oro / Step Fan Regulator (1M)")),
    ("4 way extension board", "", "Accessory",
     TruthItem("4 way extension board", 40, unit="Sets", catalogue="Accessories / 4 Way Extension Board")),
    ("8 module surface mounting plastic box", "(5X5) for Oro", "Box",
     TruthItem("8 module surface mounting plastic box (5X5) for Oro", 75, catalogue="Boxes / 8 Module (Oro) (5X5)")),
]


def build_office(path: Path) -> SampleTruth:
    title = "Supply of Wiring Devices for Office Fit-out, Tower B"
    buyer = "Northwind Office Parks LLP"
    story: List = [
        Paragraph("Request for Proposal (RFP)", BODY),
        Paragraph(title, H1),
        Paragraph(f"Issued by: {buyer}, Gurugram. Enquiry No. NOP/FO/2026/219.", BODY),
        Paragraph("1. Scope", H2),
        Paragraph(
            "Supply of modular wiring devices for floors 3 to 7 of Tower B. Installation is not in scope. "
            "Quote for the items listed in the schedule below; partial quotes will not be considered.", BODY),
        Paragraph("2. Schedule of Items", H2),
        Spacer(1, 2 * mm),
    ]
    rows = [["Item No.", "Particulars", "Quantity", "Remarks"]]
    truth: List[TruthItem] = []
    for n, (line1, line2, remark, item) in enumerate(OFFICE_ITEMS, 1):
        unit = item.unit
        rows.append([str(n), line1, f"{item.quantity} {unit}", remark])
        if line2:
            rows.append(["", line2, "", ""])
        truth.append(item)
    t = Table(rows, colWidths=[18 * mm, 95 * mm, 28 * mm, 30 * mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
    ]))
    story += [t]

    story += [
        Paragraph("3. Delivery Schedule", H2),
        Spacer(1, 2 * mm),
    ]
    sched = Table([["Lot", "Floors", "Delivery within"]] + [
        ["Lot 1", "Floors 3 and 4", "10 days of purchase order"],
        ["Lot 2", "Floors 5 to 7", "25 days of purchase order"],
    ], colWidths=[25 * mm, 60 * mm, 89 * mm])
    sched.setStyle(TableStyle(GRID))
    story += [sched]
    story += [
        Paragraph("4. Conditions", H2),
        *_bullets([
            "The vendor must be an authorised dealer of the quoted brand.",
            "Prices must be valid for 60 days and include delivery to site.",
            "Bids must be submitted by email by 30-11-2026.",
        ]),
    ]
    _build(path, title, story)
    return SampleTruth(
        name="office", title=title, buyer=buyer, items=truth, tables={"boq": 1, "other": 1},
        facts={"When is lot 2 delivered?": "25 days"},
    )


BUILDERS: Dict[str, Tuple[str, Callable[[Path], SampleTruth]]] = {
    "hotel": ("rfp_hotel_boq.pdf", build_hotel),
    "school": ("rfp_school_campuses.pdf", build_school),
    "office": ("rfp_office_borderless.pdf", build_office),
}


def build_all(out_dir: Path) -> Dict[str, Tuple[Path, SampleTruth]]:
    """Write every sample PDF into *out_dir*; return {name: (pdf path, truth)}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    built = {}
    for name, (filename, builder) in BUILDERS.items():
        path = out_dir / filename
        built[name] = (path, builder(path))
    return built


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the table-heavy sample RFPs and their ground truth.")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data" / "samples"))
    args = parser.parse_args()
    for name, (path, truth) in build_all(Path(args.out)).items():
        truth_path = path.with_name(path.stem + "_truth.json")
        truth_path.write_text(json.dumps(asdict(truth), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{name:<7} {path}  ({len(truth.items)} items)  truth: {truth_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
