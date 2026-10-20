"""
Sample tender generator: the default discovery source.

Builds realistic but clearly fictitious RFP PDFs from the product catalogue,
shaped by the user's search (keywords, location, dates, budget). Each RFP:

  - names a fictional buyer and is stamped "SAMPLE RFP: fictitious buyer"
    on every page, so it can never be mistaken for a real tender;
  - lists 6-12 product lines from the catalogue with explicit quantities
    (in procurement wording, without SAP codes, as a buyer would write);
  - usually adds 1-2 items the catalogue does not carry, so matching shows
    realistic gaps;
  - adds eligibility, delivery and submission clauses (compliance items);
  - uses the same numbered-section layout as data/samples/sample_rfp.pdf, so
    the RFP agent reads it the same way.

Generation is deterministic: the same search gives the same RFPs (and byte-
identical PDFs, so re-running a search does not create duplicates).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from catalogue import ProductRecord, get_catalogue

from .base import DiscoveryQuery, ProgressFn, RfpCandidate

# ─── Fictional buyers and places ──────────────────────────────────────────────
# Invented names (no real organisations). Each has a project type used in the title.

BUYERS: List[Tuple[str, str]] = [
    ("Green Valley Hotels Pvt. Ltd.", "hotel renovation"),
    ("Sunrise Public School Trust", "new classroom block"),
    ("Northstar Realty Developers", "residential tower"),
    ("Aravali Tech Park Ltd.", "office campus fit-out"),
    ("Lakeview Residency Co-operative Housing Society", "apartment rewiring"),
    ("Medicare Plus Hospitals Pvt. Ltd.", "hospital wing expansion"),
    ("Brightway Infra Projects", "commercial complex"),
    ("Silverline Mall Management Pvt. Ltd.", "mall refurbishment"),
    ("Kaveri Institute of Engineering", "hostel blocks"),
    ("Harmony Heights Builders", "group housing project"),
    ("Orchid Business Centre LLP", "co-working floors"),
    ("Riverside Hospitality Group", "resort cottages"),
]

CITIES = ["Gurugram", "Noida", "Jaipur", "Lucknow", "Chandigarh", "Pune", "Bengaluru",
          "Hyderabad", "Ahmedabad", "Indore", "Kochi", "Bhubaneswar"]

# ─── Product themes (keyword → catalogue products) ────────────────────────────

Theme = Tuple[str, Tuple[str, ...], Callable[[ProductRecord], bool]]

THEMES: List[Theme] = [
    ("modular switches", ("switch", "modular", "bell", "indicator"),
     lambda p: (p.variant or "").lower() == "switches"),
    ("cover plates", ("plate", "cover", "frame"),
     lambda p: "plate" in p.item_name.lower()),
    ("sockets", ("socket", "plug", "outlet"),
     lambda p: "socket" in p.item_name.lower()),
    ("fan regulators and support modules", ("regulator", "dimmer", "usb", "charger", "module", "tv", "telephone", "data"),
     lambda p: (p.variant or "").lower() == "support modules"),
    ("miniature circuit breakers", ("mcb", "breaker", "circuit"),
     lambda p: "mcb" in p.item_name.lower()),
    ("mounting boxes", ("box", "boxes", "mounting"),
     lambda p: p.range_name.lower().startswith("boxes")),
]
GENERAL_WORDS = ("electrical", "wiring", "accessories", "accessory", "fittings", "fixtures")

# Items the catalogue does NOT carry: realistic gaps for the matching step.
OFF_CATALOGUE: Dict[str, List[str]] = {
    "cable": ["4 core 16 sq mm armoured LT power cable", "1.5 sq mm FR PVC insulated copper wire (90 m coil)"],
    "wire": ["2.5 sq mm FRLS copper house wire (90 m coil)"],
    "light": ["18W LED panel light (2x2 ft)", "100W LED flood light"],
    "led": ["36W LED batten", "12W LED downlight"],
    "transformer": ["250 kVA oil-cooled distribution transformer"],
    "conduit": ["25 mm PVC conduit pipe (3 m length)"],
    "panel": ["8-way TPN distribution board"],
    "fan": ["1200 mm ceiling fan, 5-star rated"],
}
DEFAULT_GAPS = ["25 mm PVC conduit pipe (3 m length)", "8-way TPN distribution board",
                "18W LED panel light (2x2 ft)", "1200 mm ceiling fan, 5-star rated"]

ELIGIBILITY = [
    "Be an authorised distributor or manufacturer of the offered brand",
    "Have at least 3 years of experience in supplying electrical products",
    "Provide GST registration and PAN details",
    "Provide ISI / BIS certification for switches and sockets",
    "Demonstrate supply of similar orders in the last two years",
    "Submit a valid trade licence",
]
DELIVERY = [
    "Delivery within {days} days of the purchase order",
    "Materials to be delivered to site in original sealed cartons",
    "Damage-free delivery is mandatory; damaged items must be replaced at no cost",
    "Partial deliveries are acceptable against a delivery schedule",
]
SUBMISSION = [
    "Company profile",
    "Price list with SAP / product codes",
    "Technical compliance sheet",
    "Delivery schedule",
    "Warranty terms",
]

QUANTITY_STEPS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 750, 1000, 1200, 1500, 2000]


# ─── Generation ───────────────────────────────────────────────────────────────

def _seed(query: DiscoveryQuery, index: int) -> random.Random:
    key = json.dumps({"q": query.model_dump(mode="json"), "i": index}, sort_keys=True)
    return random.Random(hashlib.sha256(key.encode()).hexdigest())


def _themes_for(keywords: Sequence[str]) -> List[Theme]:
    words = " ".join(keywords).lower()
    picked = [t for t in THEMES if any(w in words for w in t[1])]
    if not picked or any(w in words for w in GENERAL_WORDS):
        picked = picked or THEMES[:4]
    return picked


def _gaps_for(keywords: Sequence[str], rng: random.Random) -> List[str]:
    words = " ".join(keywords).lower()
    pool = [item for key, items in OFF_CATALOGUE.items() if key in words for item in items]
    if pool:
        return rng.sample(pool, k=min(len(pool), rng.choice([1, 2])))
    return rng.sample(DEFAULT_GAPS, k=1) if rng.random() < 0.7 else []


def _describe(p: ProductRecord, rng: random.Random) -> str:
    """Buyer-style wording: product name, sometimes the range, never the SAP code."""
    name = re.sub(r"\s+", " ", p.item_name).strip()
    if rng.random() < 0.6:
        return f"{name}, {p.range_name} range or equivalent"
    return f"{name} (modular, ISI marked)"


_SMALL_WORDS = {"and", "or", "of", "for", "the", "a", "an", "in"}


def _title_case(text: str) -> str:
    words = text.split()
    return " ".join(w if (i and w.lower() in _SMALL_WORDS) else w[:1].upper() + w[1:] for i, w in enumerate(words))


def _title(themes: List[Theme], project: str) -> str:
    names = [t[0] for t in themes][:2]
    what = " and ".join(names) if names else "modular wiring accessories"
    return _title_case(f"Supply of {what} for {project}")


def _round_value(value: float) -> float:
    return float(max(50_000, round(value / 10_000) * 10_000))


def generate_rfp(query: DiscoveryQuery, index: int, products: List[ProductRecord]) -> Dict:
    """Build one sample RFP's content (no PDF yet)."""
    rng = _seed(query, index)
    # Distinct buyers within one search: a per-search shuffle, indexed.
    buyers = list(BUYERS)
    _seed(query, -1).shuffle(buyers)
    buyer, project = buyers[index % len(buyers)]
    location = query.location or rng.choice(CITIES)
    themes = _themes_for(query.keywords)

    pool = list([p for p in products if any(t[2](p) for t in themes)] or products)
    rng.shuffle(pool)
    target = rng.randint(6, 10)
    lines, seen_names = [], set()
    for p in pool:
        key = re.sub(r"[^a-z0-9]", "", p.item_name.lower())
        if key in seen_names:      # a buyer lists each item once, not once per range
            continue
        seen_names.add(key)
        lines.append({"description": _describe(p, rng), "qty": rng.choice(QUANTITY_STEPS), "unit": "Nos",
                      "catalogue_price": p.unit_price})
        if len(lines) >= target:
            break
    for gap in _gaps_for(query.keywords, rng):
        lines.append({"description": gap, "qty": rng.choice(QUANTITY_STEPS[:8]), "unit": "Nos",
                      "catalogue_price": None})
    rng.shuffle(lines)

    start = query.date_from or (date.today() - timedelta(days=20))
    end = query.date_to or (date.today() + timedelta(days=10))
    if end < start:
        start, end = end, start
    published = start + timedelta(days=rng.randint(0, max(0, (end - start).days)))
    # Tenders you can still bid on: at least 5 days from today.
    deadline = max(published + timedelta(days=rng.randint(10, 30)), date.today() + timedelta(days=rng.randint(5, 20)))

    # Synthetic buyer budget printed in the RFP (demo data, not a quote):
    # catalogue value of the known lines with a typical 25-50% headroom.
    known = sum(l["catalogue_price"] * l["qty"] for l in lines if l["catalogue_price"])
    value = _round_value(known * rng.uniform(1.25, 1.5) + 40_000 * (len(lines) - len([l for l in lines if l["catalogue_price"]])))
    if query.min_value is not None or query.max_value is not None:
        lo = query.min_value or 0
        hi = query.max_value or max(lo * 2, value)
        value = _round_value(min(max(value, lo), hi))

    return {
        "reference": f"SMP/{published.year}/{rng.randint(1000, 9999)}",
        "title": _title(themes, project),
        "buyer": buyer,
        "project": project,
        "location": location,
        "category": query.category or (themes[0][0].title() if themes else "Electrical"),
        "published": published,
        "deadline": deadline,
        "value": value,
        "lines": lines,
        "eligibility": rng.sample(ELIGIBILITY, k=4),
        "delivery_days": rng.choice([7, 10, 14, 21]),
        "submission": rng.sample(SUBMISSION, k=4),
    }


# ─── PDF rendering ────────────────────────────────────────────────────────────

_STAMP = "SAMPLE RFP generated by Synapse for demonstration. Buyer, contacts and figures are fictitious."


def _on_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#9a3412"))
    canvas.drawCentredString(A4[0] / 2, 12 * mm, _STAMP)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawRightString(A4[0] - 18 * mm, 12 * mm - 9, f"Page {doc.page}")
    canvas.restoreState()


def render_pdf(rfp: Dict, path: Path) -> None:
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.5, leading=13)
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=15, leading=19, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=10, spaceAfter=4)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2)

    def bullets(items):
        return [Paragraph(item, bullet, bulletText="•") for item in items]

    story = [
        Paragraph("Request for Proposal (RFP)", body),
        Paragraph(rfp["title"], h1),
        Paragraph(f"Issued by: {rfp['buyer']}, {rfp['location']}", body),
        Paragraph(f"RFP Reference: {rfp['reference']} &nbsp;&nbsp; Date of issue: {rfp['published']:%d-%m-%Y}", body),
        Paragraph(f"<font color='#9a3412'>{_STAMP}</font>", body),
        Spacer(1, 4 * mm),
        Paragraph("1. Introduction", h2),
        Paragraph(
            f"{rfp['buyer']} invites proposals from qualified vendors for the supply of the items listed below "
            f"for its {rfp['project']} in {rfp['location']}. Products from established brands with ISI marking "
            "are preferred.", body),
        Paragraph("2. Scope of Supply", h2),
        Paragraph("The vendor shall supply the following items in the quantities stated:", body),
        Spacer(1, 2 * mm),
    ]

    table = [["S.No", "Item description", "Quantity", "Unit"]]
    for n, line in enumerate(rfp["lines"], 1):
        table.append([str(n), Paragraph(line["description"], body), f"{line['qty']:,}", line["unit"]])
    t = Table(table, colWidths=[12 * mm, 112 * mm, 22 * mm, 16 * mm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ede9fe")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (2, 1), (2, -1), "RIGHT"),
    ]))
    story += [t]

    story += [
        Paragraph("3. Technical Specifications", h2),
        *bullets([
            "All items must comply with relevant BIS standards (IS 3854 for switches, IS 1293 for sockets).",
            "HSN codes must be stated for every item (8536 for switches and sockets, 8538 for plates).",
            "Standard packing formats are to be stated in the price list.",
        ]),
        Paragraph("4. Eligibility Criteria", h2),
        Paragraph("Vendors must:", body),
        *bullets(rfp["eligibility"]),
        Paragraph("5. Delivery and Logistics", h2),
        *bullets([d.format(days=rfp["delivery_days"]) for d in DELIVERY[:3]]),
        Paragraph("6. Commercial Terms", h2),
        *bullets([
            "Unit price (INR) for each item, with GST shown separately.",
            f"Estimated budget for this procurement: INR {rfp['value']:,.0f} (indicative).",
            "Prices shall remain valid for 90 days from the submission deadline.",
        ]),
        Paragraph("7. Submission and Important Dates", h2),
        Paragraph("Vendors must submit:", body),
        *bullets(rfp["submission"]),
        Spacer(1, 2 * mm),
        Paragraph(f"Last date for submission: {rfp['deadline']:%d-%m-%Y}", body),
        Paragraph(f"Procurement Department, {rfp['buyer']} (procurement@example.com)", body),
    ]

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=20 * mm, title=rfp["title"],
                            author="Synapse sample generator", invariant=True)
    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)


# ─── Source ───────────────────────────────────────────────────────────────────

class SampleSource:
    name = "sample"
    label = "Sample tenders (demo)"

    def available(self) -> Optional[str]:
        return None

    def search(self, query: DiscoveryQuery, out_dir: str,
               progress: Optional[ProgressFn] = None) -> List[RfpCandidate]:
        products = get_catalogue().all_products()
        folder = Path(out_dir) / "samples"
        folder.mkdir(parents=True, exist_ok=True)

        found: List[RfpCandidate] = []
        for i in range(query.count):
            rfp = generate_rfp(query, i, products)
            path = folder / f"{rfp['reference'].replace('/', '_')}_{i}.pdf"
            render_pdf(rfp, path)
            found.append(RfpCandidate(
                title=rfp["title"],
                pdf_path=str(path),
                source=self.name,
                buyer=rfp["buyer"],
                deadline=rfp["deadline"].isoformat(),
                published=rfp["published"].isoformat(),
                location=rfp["location"],
                category=rfp["category"],
                estimated_value=rfp["value"],
                reference=rfp["reference"],
                meta={"synthetic": True, "line_items": len(rfp["lines"]),
                      "off_catalogue_items": sum(1 for l in rfp["lines"] if l["catalogue_price"] is None)},
            ))
            if progress:
                progress(i + 1, query.count, f"Prepared {rfp['reference']}")
        return found
