"""
rfp_tables.py
-------------
Understanding tables found in an RFP (pure functions, no PDF access).

rfp_parser.py finds the tables; this module works out what they mean:

  column_roles()          what each column holds (description, quantity, unit,
                          per-location quantity, make, rate, …) from its header
                          and, where the header says nothing, from its values
  classify_table()        boq (item list with quantities) | spec | kv | other
  boq_requirements()      item-list rows -> requirement dicts, by rules (no LLM):
                          quantity read from the cell, unit, category rows used
                          as context, total rows skipped, page/row source kept
  table_markdown()        the table as GitHub Markdown (for the LLM and the index)
  row_record()            one row as "Header: value; …" (for row-level retrieval)
  parse_quantity()        "1,200 Nos" -> (1200, "Nos")

Roles
-----
sno, description, quantity (the order quantity; a "Total" column when the
quantity is split by location), quantity_part (one location's quantity),
unit, category, specification, make, remarks, price (rate / amount columns the
vendor fills in; never read as quantities), other.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ─── Table record (filled by rfp_parser) ──────────────────────────────────────


@dataclass
class TableData:
    """One logical table (a table split over pages is joined into one)."""
    id: str                                   # "T1", "T2", … in document order
    header: List[str]
    rows: List[List[str]]
    pages: List[int] = field(default_factory=list)      # 1-based
    row_pages: List[int] = field(default_factory=list)  # page of each row
    section: str = ""                         # heading path where the table sits
    caption: str = ""                         # short line just above the table
    detection: str = "lines"                  # "lines" (ruled) | "text" (no gridlines)
    roles: List[str] = field(default_factory=list)
    kind: str = "other"                       # boq | spec | kv | other

    @property
    def page_label(self) -> str:
        if not self.pages:
            return ""
        first, last = min(self.pages), max(self.pages)
        return f"p.{first}" if first == last else f"pp.{first}-{last}"

    def summary(self) -> Dict[str, Any]:
        """Compact description stored in the agent state (no cell data)."""
        return {
            "id": self.id,
            "kind": self.kind,
            "pages": list(self.pages),
            "section": self.section,
            "caption": self.caption,
            "columns": list(self.header),
            "roles": list(self.roles),
            "rows": len(self.rows),
            "detection": self.detection,
        }


# ─── Text helpers ─────────────────────────────────────────────────────────────

def clean_cell(value: Any) -> str:
    """Cell text on one line with single spaces ('' for None)."""
    if value is None:
        return ""
    text = re.sub(r"[​‌‍﻿]", "", str(value))   # zero-width chars (Google Docs exports)
    return re.sub(r"\s+", " ", text).strip()


def _norm_header(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


_QTY_RE = re.compile(
    r"^\s*(?:approx(?:imately|\.)?\s*|about\s*|~\s*)?"
    r"(\d{1,3}(?:,\d{2,3})+|\d+)(\.\d+)?"          # 1,200 · 1,20,000 · 500 · 12.5
    r"\s*([A-Za-z][A-Za-z. /]{0,15})?\s*$"
)


def parse_quantity(text: str) -> Tuple[Optional[int], str]:
    """
    Read an order quantity from a cell: "1,200" -> (1200, ""), "120 Nos" ->
    (120, "Nos"), "2 Sets" -> (2, "Sets"). Fractions ("12.5 m") and anything
    that is not a plain number give (None, unit) so a person fills them in.
    """
    m = _QTY_RE.match(clean_cell(text))
    if not m:
        return None, ""
    whole, fraction, unit = m.group(1), m.group(2), (m.group(3) or "").strip(" .")
    if fraction and float(fraction) != 0:
        return None, unit
    value = int(whole.replace(",", ""))
    return (value if value > 0 else None), unit


def _is_number(text: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?", clean_cell(text)))


def _is_quantity_cell(text: str) -> bool:
    return parse_quantity(text)[0] is not None


# ─── Column roles ─────────────────────────────────────────────────────────────

# Checked in this order against the normalised header ("Item Description" ->
# "item description"). Quantity is checked before price ("Total Qty" is a
# quantity) and sno before description ("Item No." is a serial number).
_ROLE_RULES: List[Tuple[str, re.Pattern]] = [
    ("sno", re.compile(
        r"^(s ?no|sl ?no|sr ?no|serial( no| number)?|item ?no|line ?no|s n|no|sl|sr|#|"
        r"s no s|item code|lot no|sn)$")),
    ("quantity", re.compile(
        r"\b(qty|qnty|quantity|quantities|reqd|required quantity|no of (units|items|nos|pieces|pcs))\b"
        r"|^(nos|number of units|total)$")),
    ("price", re.compile(
        r"\b(rate|price|amount|cost|value|basic|gst|tax|inr|rs)\b")),
    ("unit", re.compile(
        r"^(unit|units|uom|u o m|unit of (measure|measurement)|a u|au|acc unit|accounting unit|unit type)$")),
    ("category", re.compile(r"^(category|group|type|section|head|sub ?head|item category|product category)$")),
    ("make", re.compile(r"\b(make|brand|manufacturer|oem)\b")),
    ("remarks", re.compile(r"\b(remark|remarks|note|notes|comment|comments)\b")),
    ("specification", re.compile(
        r"\b(spec|specs|specification|specifications|technical|rating|size|parameter|parameters|"
        r"feature|features|requirement|requirements|value|values|details)\b")),
    ("description", re.compile(
        r"\b(description|particulars|item|items|nomenclature|name of (the )?(item|product|goods|material|stores)|"
        r"material|product|goods|article|stores|work|equipment)\b")),
]

# Headers that name a place: numeric columns under them are per-location quantities.
_LOCATION_RE = re.compile(
    r"\b(block|building|bldg|site|location|consignee|floor|wing|campus|phase|tower|zone|ward|hostel|office)\b"
)

_TOTAL_RE = re.compile(r"\btotal\b")


def _header_role(header: str) -> Optional[str]:
    norm = _norm_header(header)
    if not norm:
        return None
    for role, pattern in _ROLE_RULES:
        if pattern.search(norm):
            return role
    # Narrow columns wrap short headers mid-word ("S.No" -> "S.N o", "U o M").
    compact = norm.replace(" ", "")
    for role, pattern in _ROLE_RULES:
        if role in ("sno", "unit", "quantity") and pattern.search(compact):
            return role
    return None


def _column_values(rows: Sequence[Sequence[str]], col: int) -> List[str]:
    return [clean_cell(r[col]) for r in rows if col < len(r) and clean_cell(r[col])]


def column_roles(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    """
    One role per column (see module docstring). Headers decide first; values
    decide for columns whose header is unknown (e.g. "Block A" holding numbers
    becomes quantity_part, an unnamed long-text column becomes description).
    """
    n = len(header)
    roles: List[Optional[str]] = [_header_role(h) for h in header]

    def numeric_share(col: int) -> float:
        values = _column_values(rows, col)
        return sum(_is_quantity_cell(v) for v in values) / len(values) if values else 0.0

    # "Specification" can also be the item-description column of a BOQ
    # ("Item specification", "Technical specification of item"). If there is
    # no description column, the longest-text spec column is the description.
    if "description" not in roles:
        spec_cols = [c for c, r in enumerate(roles) if r == "specification"]
        if spec_cols and any(r == "quantity" for r in roles):
            best = max(spec_cols, key=lambda c: sum(len(v) for v in _column_values(rows, c)))
            roles[best] = "description"

    # Several quantity columns: a "total" one is the quantity, the others are
    # parts; without a total, they are all parts (summed per row).
    qty_cols = [c for c, r in enumerate(roles) if r == "quantity"]
    location_cols = [c for c in range(n) if roles[c] in (None, "quantity")
                     and _LOCATION_RE.search(_norm_header(header[c])) and numeric_share(c) >= 0.6]
    # One numeric "Block"/"Site" column alone is more likely an identifier.
    if len(location_cols) < 2 and not qty_cols:
        location_cols = []
    if len(qty_cols) > 1 or location_cols:
        totals = [c for c in qty_cols if _TOTAL_RE.search(_norm_header(header[c]))]
        for c in set(qty_cols) | set(location_cols):
            roles[c] = "quantity" if (totals and c == totals[0]) else "quantity_part"

    # Unknown numeric columns next to known quantity parts are parts too
    # (headers like "A", "B", "Stage 1").
    if "quantity_part" in roles:
        for c in range(n):
            if roles[c] is None and numeric_share(c) >= 0.8:
                roles[c] = "quantity_part"

    # No description column yet: the unknown column with the most text.
    if "description" not in roles:
        candidates = [c for c in range(n) if roles[c] is None and numeric_share(c) < 0.5]
        if candidates:
            best = max(candidates, key=lambda c: sum(len(v) for v in _column_values(rows, c)))
            if _column_values(rows, best):
                roles[best] = "description"

    return [r or "other" for r in roles]


def classify_table(header: Sequence[str], rows: Sequence[Sequence[str]], roles: Sequence[str]) -> str:
    """boq | spec | kv | other."""
    has_qty = "quantity" in roles or "quantity_part" in roles
    if "description" in roles and has_qty:
        qty_cols = [c for c, r in enumerate(roles) if r in ("quantity", "quantity_part")]
        filled = [r for r in rows if any(c < len(r) and _is_quantity_cell(r[c]) for c in qty_cols)]
        if filled:
            return "boq"
    if len(header) == 2:
        named = [_header_role(h) for h in header]
        # label | value: the "header" row is itself a pair ("Bid End Date | 05-12-2026")
        if named[0] is None or not any(named) or re.search(r"\d", header[1] or ""):
            return "kv"
        if not has_qty:
            return "spec"        # Parameter | Requirement
    if len(header) == 3 and roles[0] == "sno" and "quantity" not in roles and "specification" in roles:
        return "spec"
    return "other"


# ─── Rendering ────────────────────────────────────────────────────────────────

def _md_cell(text: str) -> str:
    return clean_cell(text).replace("|", "/")


def table_markdown(table: TableData, rows: Optional[Sequence[Sequence[str]]] = None) -> str:
    """GitHub Markdown; key-value tables render as "label: value" lines."""
    rows = table.rows if rows is None else rows
    if table.kind == "kv":
        pairs = [table.header] + list(rows) if table.header else list(rows)
        return "\n".join(
            f"{clean_cell(r[0])}: {' '.join(clean_cell(c) for c in r[1:])}".strip()
            for r in pairs if any(clean_cell(c) for c in r)
        )
    header = [_md_cell(h) or f"Column {i + 1}" for i, h in enumerate(table.header)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [_md_cell(row[i]) if i < len(row) else "" for i in range(len(header))]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_title(table: TableData) -> str:
    where = ", ".join(x for x in (table.section, table.page_label) if x)
    caption = f" {table.caption}" if table.caption else ""
    return f"[Table {table.id}{f' ({where})' if where else ''}]{caption}"


def row_record(table: TableData, row: Sequence[str], context: str = "") -> str:
    """One row as a self-describing record for retrieval."""
    parts = []
    for i, header in enumerate(table.header):
        value = clean_cell(row[i]) if i < len(row) else ""
        if value:
            label = clean_cell(header) or f"Column {i + 1}"
            parts.append(f"{label}: {value}")
    where = ", ".join(x for x in (table.section, table.page_label) if x)
    prefix = f"Table {table.id}" + (f" ({where})" if where else "")
    ctx = f" [{context}]" if context else ""
    return f"{prefix}{ctx}: " + "; ".join(parts)


# ─── Item-list (BOQ) rows -> requirements ─────────────────────────────────────

_TOTAL_ROW_RE = re.compile(r"^(grand\s+)?(sub[\s-]*)?total\b", re.I)
_CATEGORY_PREFIX_RE = re.compile(r"^\s*(?:[A-Z]|[IVX]{1,4}|\d{1,2})[.)]\s+")
_PIECE_UNITS = {"", "no", "nos", "no s", "number", "numbers", "pc", "pcs", "piece", "pieces", "each",
                "ea", "unit", "units", "set", "sets", "item", "items", "qty"}


def is_piece_unit(unit: str) -> bool:
    """True when a quantity in this unit is a count of pieces (Nos, Pcs, Each…)."""
    return _norm_header(unit) in _PIECE_UNITS


@dataclass
class TableRequirement:
    requirement: str
    quantity: Optional[int]
    unit: str
    page: int
    row_label: str                    # S.No as printed, else the row number
    category: str = ""
    attributes: Dict[str, str] = field(default_factory=dict)
    quantity_breakdown: Dict[str, int] = field(default_factory=dict)
    note: str = ""                    # e.g. location quantities do not add up


def _first(row: Sequence[str], roles: Sequence[str], role: str) -> str:
    for i, r in enumerate(roles):
        if r == role and i < len(row) and clean_cell(row[i]):
            return clean_cell(row[i])
    return ""


def _adds_words(category: str, description: str) -> bool:
    """
    True if the category says something the description does not: "Switches
    (Oro Metallica range, Grey)" adds the range and colour to "1 way switch",
    while "Switches" adds nothing to "16A switch" (plural / singular match).
    """
    desc = re.findall(r"[a-z0-9]{3,}", description.lower())

    def covered(word: str) -> bool:
        return any(d.startswith(word[:max(4, len(word) - 2)]) or word.startswith(d) and len(d) >= 4
                   for d in desc)

    return any(not covered(w) for w in re.findall(r"[a-z0-9]{3,}", category.lower())
               if w not in {"and", "the", "for", "with", "range", "items", "item"})


def is_category_row(row: Sequence[str], roles: Sequence[str]) -> bool:
    """A description with nothing else filled in ("A. Switches"): a group label."""
    if not _first(row, roles, "description"):
        return False
    return not any(
        clean_cell(row[i]) for i, r in enumerate(roles)
        if i < len(row) and r not in ("description", "sno", "category", "price")
    )


def row_categories(table: TableData) -> List[str]:
    """For each row of an item list, the category label in force ('' if none)."""
    out, category = [], ""
    for row in table.rows:
        if table.kind == "boq" and is_category_row(row, table.roles):
            category = _CATEGORY_PREFIX_RE.sub("", _first(row, table.roles, "description")).strip(" :")
            out.append("")
            continue
        category = _first(row, table.roles, "category") or category
        out.append(category if table.kind == "boq" else "")
    return out


def boq_requirements(table: TableData) -> List[TableRequirement]:
    """
    Requirements from an item-list table, one per item row:
      - rows with a description but no quantity and nothing else filled in
        ("A. Switches") are category rows: context for the rows below
      - "Total" / "Sub-total" rows are skipped
      - quantity comes from the quantity column, else the sum of per-location
        columns; a unit written in the quantity cell ("120 Nos") is used
      - specification text is appended to the requirement (it matters for
        matching); make / remarks / unknown columns are kept as attributes
    """
    roles = table.roles
    headers = [clean_cell(h) for h in table.header]
    out: List[TableRequirement] = []
    category = ""
    for index, row in enumerate(table.rows):
        description = _first(row, roles, "description")
        if not description or _TOTAL_ROW_RE.match(description):
            continue
        qty_text = _first(row, roles, "quantity")
        parts = {headers[i] or f"Column {i + 1}": parse_quantity(row[i])[0]
                 for i, r in enumerate(roles) if r == "quantity_part" and i < len(row)}
        parts = {k: v for k, v in parts.items() if v}
        row_category = _first(row, roles, "category")
        if is_category_row(row, roles):
            category = _CATEGORY_PREFIX_RE.sub("", description).strip(" :")
            continue
        if row_category:
            category = row_category

        quantity, unit_in_cell = parse_quantity(qty_text) if qty_text else (None, "")
        note = ""
        if parts:
            total = sum(parts.values())
            if quantity is None:
                quantity = total
            elif quantity != total:
                note = (f"location quantities add up to {total:,} but the total column says {quantity:,}; "
                        "the total column is used")
        unit = _first(row, roles, "unit") or unit_in_cell

        spec = _first(row, roles, "specification")
        text = description
        if category and _adds_words(category, description):
            text = f"{category} - {description}"
        if spec and spec.lower() not in text.lower():
            text = f"{text}, {spec}"

        attributes = {
            (headers[i] or f"Column {i + 1}"): clean_cell(row[i])
            for i, r in enumerate(roles)
            if r in ("make", "remarks", "other") and i < len(row) and clean_cell(row[i])
        }
        sno = _first(row, roles, "sno")
        out.append(TableRequirement(
            requirement=text,
            quantity=quantity,
            unit=unit,
            page=table.row_pages[index] if index < len(table.row_pages) else (table.pages or [0])[0],
            row_label=sno or str(index + 1),
            category=category,
            attributes=attributes,
            quantity_breakdown=parts,
            note=note,
        ))
    return out


def analyse_table(table: TableData) -> TableData:
    """Fill in roles and kind (in place) and return the table."""
    table.roles = column_roles(table.header, table.rows)
    table.kind = classify_table(table.header, table.rows, table.roles)
    return table
