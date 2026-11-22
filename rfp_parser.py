"""
rfp_parser.py
-------------
Reads an RFP PDF into an ordered list of blocks: headings, text and tables
(PyMuPDF only; no models, no network).

  parse_pdf(path) -> ParsedDocument

What it does that plain page.get_text() does not:
  - Tables: page.find_tables() on ruled tables; for tables drawn without
    gridlines, a text-alignment search in the space outside ruled tables,
    accepted only when the result reads as an item list (description +
    quantity columns), so ordinary paragraphs are never turned into tables.
  - Table cells are never repeated as body text.
  - A table that continues on the next page is joined into one table (a
    repeated header row is dropped); vertically merged cells are filled down.
  - Lines repeated in the top/bottom margin of most pages (running headers,
    footers, stamps) and "Page N" lines are removed.
  - Headings are found from numbering ("2.", "2.1", "A.") and font style
    (larger or bold), so every block knows its section ("2. Scope of Supply").

Text is never dropped for being short: everything readable reaches a block.
"""

from __future__ import annotations

import logging
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pymupdf

from rfp_tables import TableData, analyse_table, clean_cell, column_roles, table_markdown, table_title

logger = logging.getLogger(__name__)

Rect = Tuple[float, float, float, float]

_MARGIN_SHARE = 0.10          # top/bottom share of the page treated as header/footer band
_PAGE_NO_RE = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.I)
_NUMBERED_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+(\S.*)$")
_LETTERED_RE = re.compile(r"^([A-H]|[IVX]{1,4})[.)]\s+(\S.*)$")
_QTY_HINT_RE = re.compile(r"\b(qty|quantity|nos|uom|unit)\b", re.I)
_QTY_ONLY_RE = re.compile(r"^\d[\d,]*(\.\d+)?(\s*[A-Za-z.]{1,8})?$")


# ─── Output ───────────────────────────────────────────────────────────────────

@dataclass
class Block:
    kind: str                          # "heading" | "text" | "table"
    page: int                          # 1-based
    section: str                       # heading path, e.g. "2. Scope > 2.1 Items"
    text: str = ""
    level: int = 0                     # headings only
    table: Optional[TableData] = None  # tables only


@dataclass
class ParsedDocument:
    path: str
    page_count: int
    blocks: List[Block] = field(default_factory=list)
    removed_lines: int = 0             # running headers / footers / page numbers

    @property
    def tables(self) -> List[TableData]:
        return [b.table for b in self.blocks if b.kind == "table" and b.table is not None]

    def has_text(self) -> bool:
        return any((b.text.strip() if b.kind != "table" else b.table and b.table.rows) for b in self.blocks)

    def to_text(self) -> str:
        """The whole document in reading order, tables as Markdown."""
        parts: List[str] = []
        for b in self.blocks:
            if b.kind == "table" and b.table is not None:
                parts.append(f"{table_title(b.table)}\n{table_markdown(b.table)}")
            else:
                parts.append(b.text)
        return "\n\n".join(p for p in parts if p.strip())


# ─── Lines with style ─────────────────────────────────────────────────────────

@dataclass
class _Line:
    text: str
    bbox: Rect
    size: float
    bold: bool
    block_no: int


def _page_lines(page: pymupdf.Page) -> List[_Line]:
    lines: List[_Line] = []
    data = page.get_text("dict", sort=True)
    for block_no, block in enumerate(data.get("blocks", [])):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            text = clean_cell("".join(s["text"] for s in line["spans"]))
            if not text:
                continue
            size = max(s.get("size", 0.0) for s in spans)
            bold = all((s.get("flags", 0) & 16) or "bold" in s.get("font", "").lower() for s in spans)
            lines.append(_Line(text, tuple(line["bbox"]), size, bold, block_no))
    return lines


def _inside(bbox: Rect, rects: Iterable[Rect], pad: float = 1.5) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return any(r[0] - pad <= cx <= r[2] + pad and r[1] - pad <= cy <= r[3] + pad for r in rects)


def _margin_key(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


def _repeated_margin_lines(pages: Sequence[Tuple[float, List[_Line]]]) -> set:
    """Margin texts (digits normalised) repeated on at least half of the pages (min 2)."""
    if len(pages) < 2:
        return set()
    counts: Counter = Counter()
    for height, lines in pages:
        seen = {_margin_key(l.text) for l in lines
                if l.bbox[3] <= height * _MARGIN_SHARE or l.bbox[1] >= height * (1 - _MARGIN_SHARE)}
        counts.update(seen)
    need = max(2, (len(pages) + 1) // 2)
    return {key for key, n in counts.items() if n >= need}


# ─── Tables on one page ───────────────────────────────────────────────────────

@dataclass
class _RawTable:
    bbox: Rect
    header: List[str]
    rows: List[List[str]]
    detection: str


def _fill_merged_cells(raw: List[List[Optional[str]]], cells: List[List[Optional[Rect]]]) -> List[List[str]]:
    """
    PyMuPDF returns None for cells covered by a merged cell. Covered from the
    left (horizontal span) -> ''; covered from above (vertical span, e.g. a
    category spanning several rows) -> the value above.
    """
    if not raw:
        return []
    ncols = max(len(r) for r in raw)
    centres: List[Optional[float]] = []
    for c in range(ncols):
        xs = [((row[c][0] + row[c][2]) / 2) for row in cells if c < len(row) and row[c] is not None]
        centres.append(statistics.median(xs) if xs else None)

    filled: List[List[str]] = []
    for r, row in enumerate(raw):
        out: List[str] = []
        for c in range(ncols):
            value = row[c] if c < len(row) else None
            if value is not None:
                out.append(clean_cell(value))
                continue
            row_cells = cells[r] if r < len(cells) else []
            x = centres[c]
            covered_from_left = x is not None and any(
                k < c and k < len(row_cells) and row_cells[k] is not None
                and row_cells[k][0] <= x <= row_cells[k][2]
                for k in range(c)
            )
            out.append("" if covered_from_left or r == 0 else filled[r - 1][c])
        filled.append(out)
    return filled


def _combine_two_row_header(header: List[str], rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    """
    "Quantity" spanning "Block A | Block B" gives a header row with blanks and
    a sub-header row with no numbers: merge them into one header row.
    """
    if not rows or "" not in header:
        return header, rows
    sub = rows[0]
    if any(re.fullmatch(r"[\d,.\s]+", c or "") for c in sub if c) or not any(sub):
        return header, rows
    combined, top = [], ""
    for h, s in zip(header, sub):
        top = h or top
        combined.append(clean_cell(f"{top} {s}") if s and s != top else (h or s or top))
    return combined, rows[1:]


def _ruled_tables(page: pymupdf.Page) -> List[_RawTable]:
    out: List[_RawTable] = []
    for tab in page.find_tables().tables:
        try:
            raw = tab.extract()
            cells = [list(row.cells) for row in tab.rows]
        except Exception:  # malformed table on this page: keep the text instead
            logger.exception("Could not read a table on page %d", page.number + 1)
            continue
        grid = _fill_merged_cells(raw, cells)
        names = [clean_cell(h) for h in tab.header.names]
        # PyMuPDF sometimes takes the line above a table (a heading such as
        # "Bid Details") as an outside header; only a real multi-cell header
        # row outside the grid is used.
        if tab.header.external and sum(bool(n) for n in names) >= max(2, len(names) // 2):
            header, body = names, grid
        else:
            if not grid:
                continue
            header, body = grid[0], grid[1:]
            header, body = _combine_two_row_header(header, body)
        body = [r for r in body if any(r)]
        if len(header) < 2 or (not body and len(header) < 3):
            continue
        out.append(_RawTable(tuple(tab.bbox), header, body, "lines"))
    return out


def _free_bands(page: pymupdf.Page, taken: Sequence[Rect]) -> List[pymupdf.Rect]:
    """Vertical strips of the page not covered by ruled tables."""
    height, width = page.rect.height, page.rect.width
    top, bottom = height * _MARGIN_SHARE * 0.5, height * (1 - _MARGIN_SHARE * 0.5)
    bands, cursor = [], top
    for r in sorted(taken, key=lambda r: r[1]):
        if r[1] - cursor > 40:
            bands.append(pymupdf.Rect(0, cursor, width, r[1]))
        cursor = max(cursor, r[3])
    if bottom - cursor > 40:
        bands.append(pymupdf.Rect(0, cursor, width, bottom))
    return bands


def _tidy_rows(header: List[str], rows: List[List[str]]) -> List[List[str]]:
    """Fold wrapped description lines (lowercase / "(" start, or after a connector) into the row above."""
    roles = column_roles(header, rows)
    if "description" not in roles:
        return rows
    d = roles.index("description")
    out: List[List[str]] = []
    for r in rows:
        only_description = r[d] and not any(v for i, v in enumerate(r) if i != d)
        continues = (r[d][:1].islower() or r[d][:1] in "(&/-"
                     or (out and out[-1][d].rstrip().endswith((",", "-", "/", "&", " and", " with", " for"))))
        if out and only_description and continues:
            out[-1][d] = clean_cell(f"{out[-1][d]} {r[d]}")
        else:
            out.append(list(r))
    return out


_DESC_HINT_RE = re.compile(r"\b(description|particulars|item|items|nomenclature|material|product|goods|stores)\b", re.I)


def _visual_lines(words: Sequence[tuple]) -> List[List[tuple]]:
    """Words (x0, y0, x1, y1, text, …) grouped into lines by vertical centre, top to bottom."""
    lines: List[List[tuple]] = []
    for w in sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        cy = (w[1] + w[3]) / 2
        if lines and abs(cy - (lines[-1][0][1] + lines[-1][0][3]) / 2) <= 3:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(l, key=lambda w: w[0]) for l in lines]


def _header_cells(line: List[tuple], gap: float = 9.0) -> List[List[tuple]]:
    """Split a header line into cells where the space between words is wide."""
    cells: List[List[tuple]] = [[line[0]]]
    for prev, w in zip(line, line[1:]):
        if w[0] - prev[2] > gap:
            cells.append([w])
        else:
            cells[-1].append(w)
    return cells


def _text_tables(page: pymupdf.Page, taken: Sequence[Rect]) -> List[_RawTable]:
    """
    Item lists drawn without gridlines. Anchored on a header line that names a
    description AND a quantity column: column edges come from where the header
    cells sit, every word below is put in the column under which it lies, and
    the table ends at a large vertical gap. Accepted only if the result reads
    as an item list, so ordinary paragraphs never become tables.
    """
    out: List[_RawTable] = []
    for band in _free_bands(page, taken):
        if not _QTY_HINT_RE.search(page.get_text(clip=band)):
            continue   # cheap pre-check
        lines = _visual_lines(page.get_text("words", clip=band))
        i = 0
        while i < len(lines):
            text = " ".join(w[4] for w in lines[i])
            cells = _header_cells(lines[i]) if (_QTY_HINT_RE.search(text) and _DESC_HINT_RE.search(text)) else []
            header = [clean_cell(" ".join(w[4] for w in c)) for c in cells]
            if len(cells) < 3 or not {"description", "quantity"} <= set(column_roles(header, [])):
                i += 1
                continue
            # column edges: midway between neighbouring header cells
            spans = [(min(w[0] for w in c), max(w[2] for w in c)) for c in cells]
            edges = [band.x0] + [(spans[k][1] + spans[k + 1][0]) / 2 for k in range(len(spans) - 1)] + [band.x1]
            rows: List[List[str]] = []
            last_y, gaps, j = lines[i][0][3], [], i + 1
            while j < len(lines):
                y0 = min(w[1] for w in lines[j])
                gap = y0 - last_y
                typical = statistics.median(gaps) if gaps else max(gap, 4.0)
                if gap > max(10.0, 3 * typical):
                    break
                row = [""] * len(cells)
                for w in lines[j]:
                    cx = (w[0] + w[2]) / 2
                    col = next((k for k in range(len(cells)) if edges[k] <= cx < edges[k + 1]), len(cells) - 1)
                    row[col] = clean_cell(f"{row[col]} {w[4]}")
                line_text = " ".join(w[4] for w in lines[j])
                heading = _NUMBERED_RE.match(line_text)
                if heading and heading.group(2)[:1].isupper() and not any(
                        _QTY_ONLY_RE.match(row[k]) for k in range(1, len(row))):
                    break   # the next section's heading ("3. Delivery Schedule")
                gaps.append(gap)
                rows.append(row)
                last_y = max(w[3] for w in lines[j])
                j += 1
            rows = _tidy_rows(header, [r for r in rows if any(r)])
            probe = analyse_table(TableData(id="?", header=header, rows=rows))
            if probe.kind == "boq" and len(rows) >= 2:
                bbox = (min(s[0] for s in spans), min(w[1] for w in lines[i]), max(s[1] for s in spans), last_y)
                out.append(_RawTable(bbox, header, rows, "text"))
                i = j
            else:
                i += 1
    return out


# ─── Headings ─────────────────────────────────────────────────────────────────

_BULLET_RE = re.compile(r"^[•●○▪■➢✓✔‣⁃∙*\-–·o]\s")


def _heading_level(line: _Line, body_size: float, styled_numbering: bool,
                   body_bold: bool = False) -> Optional[int]:
    text = line.text.replace("​", "").strip()
    words = text.split()
    if (len(text) > 100 or len(words) > 14 or text.endswith((",", ";"))
            or _BULLET_RE.match(text) or text[:1] in "•●○▪■➢"):
        return None
    bold = line.bold and not body_bold      # bold means nothing if the body is bold too
    styled = line.size >= body_size + 1.0 or bold
    numbered = _NUMBERED_RE.match(text)
    if numbered and not re.match(r"^\d", numbered.group(2)):
        if styled or (not styled_numbering and len(words) <= 9 and numbered.group(2)[:1].isupper()
                      and not text.endswith(".")):
            return numbered.group(1).count(".") + 1 if "." in numbered.group(1).rstrip(".") else 1
        return None
    lettered = _LETTERED_RE.match(text)
    if lettered and len(words) <= 7 and not text.endswith("."):
        return 3 if (styled or not styled_numbering) else None
    if text.endswith(":"):
        return None
    if line.size >= body_size + 1.5 or (bold and len(words) <= 10 and not text.endswith(".")):
        return 1
    return None


def _body_style(lines: Sequence[_Line]) -> Tuple[float, bool]:
    """(most common font size, whether most body text is bold), weighted by characters."""
    weights: Counter = Counter()
    for l in lines:
        weights[round(l.size, 1)] += len(l.text)
    if not weights:
        return 10.0, False
    size = weights.most_common(1)[0][0]
    at_size = [l for l in lines if round(l.size, 1) == size]
    bold_chars = sum(len(l.text) for l in at_size if l.bold)
    return size, bold_chars > 0.5 * sum(len(l.text) for l in at_size)


# ─── Assembly ─────────────────────────────────────────────────────────────────

def _same_header(a: Sequence[str], b: Sequence[str]) -> bool:
    norm = lambda h: [re.sub(r"[^a-z0-9]", "", x.lower()) for x in h]
    return norm(a) == norm(b)


def _caption(blocks: List[Block]) -> str:
    """A short line right before a table ("Table 2: …", "… as below:")."""
    if blocks and blocks[-1].kind == "text":
        last = blocks[-1].text.strip().splitlines()[-1].strip()
        if len(last) <= 140:
            return last
    return ""


def parse_pdf(path: str) -> ParsedDocument:
    """Parse *path* into ordered heading / text / table blocks."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"RFP PDF not found: {path}")

    with pymupdf.open(path) as pdf:
        pages: List[Tuple[int, float, List[_Line], List[_RawTable]]] = []
        for page in pdf:
            ruled = _ruled_tables(page)
            text_tables = _text_tables(page, [t.bbox for t in ruled])
            tables = sorted(ruled + text_tables, key=lambda t: t.bbox[1])
            lines = [l for l in _page_lines(page) if not _inside(l.bbox, [t.bbox for t in tables])]
            pages.append((page.number + 1, page.rect.height, lines, tables))
        page_count = len(pdf)

    repeated = _repeated_margin_lines([(h, lines) for _, h, lines, _ in pages])
    removed = 0
    cleaned: List[Tuple[int, float, List[_Line], List[_RawTable]]] = []
    for number, height, lines, tables in pages:
        kept = []
        for l in lines:
            in_margin = l.bbox[3] <= height * _MARGIN_SHARE or l.bbox[1] >= height * (1 - _MARGIN_SHARE)
            if in_margin and (_margin_key(l.text) in repeated or _PAGE_NO_RE.match(l.text)):
                removed += 1
                continue
            kept.append(l)
        cleaned.append((number, height, kept, tables))

    all_lines = [l for _, _, lines, _ in cleaned for l in lines]
    body, body_bold = _body_style(all_lines)
    styled_numbering = any(
        _NUMBERED_RE.match(l.text) and ((l.bold and not body_bold) or l.size >= body + 1.0) for l in all_lines
    )

    doc = ParsedDocument(path=str(path), page_count=page_count, removed_lines=removed)
    blocks = doc.blocks
    stack: List[Tuple[int, str]] = []        # (level, heading text)
    table_count = 0

    def section() -> str:
        return " > ".join(t for _, t in stack)

    for number, _height, lines, tables in cleaned:
        items: List[Tuple[float, str, object]] = [(l.bbox[1], "line", l) for l in lines]
        items += [(t.bbox[1], "table", t) for t in tables]
        items.sort(key=lambda x: (x[0], 0 if x[1] == "table" else 1))
        first_on_page = True
        current_block_no: Optional[int] = None
        for _y, kind, item in items:
            if kind == "table":
                raw: _RawTable = item  # type: ignore[assignment]
                prev = blocks[-1] if blocks else None
                # Continuation: the previous block is a table ending on the
                # previous page, nothing precedes this one on its page, and the
                # column count matches.
                continues = (
                    first_on_page and prev is not None and prev.kind == "table" and prev.table is not None
                    and max(prev.table.pages) == number - 1
                    and len(prev.table.header) == len(raw.header)
                )
                if continues:
                    table = prev.table
                    new_rows = raw.rows if _same_header(table.header, raw.header) else [raw.header] + raw.rows
                    table.rows.extend(new_rows)
                    table.row_pages.extend([number] * len(new_rows))
                    table.pages.append(number)
                    analyse_table(table)
                else:
                    table_count += 1
                    table = TableData(
                        id=f"T{table_count}", header=raw.header, rows=list(raw.rows),
                        pages=[number], row_pages=[number] * len(raw.rows),
                        section=section(), caption=_caption(blocks), detection=raw.detection,
                    )
                    analyse_table(table)
                    blocks.append(Block("table", number, section(), table=table))
                current_block_no = None
                first_on_page = False
                continue

            line: _Line = item  # type: ignore[assignment]
            first_on_page = False
            level = _heading_level(line, body, styled_numbering, body_bold)
            if level is not None:
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, line.text))
                blocks.append(Block("heading", number, section(), text=line.text, level=level))
                current_block_no = None
                continue
            if (blocks and blocks[-1].kind == "text" and blocks[-1].page == number
                    and current_block_no == line.block_no):
                blocks[-1].text += "\n" + line.text
            else:
                blocks.append(Block("text", number, section(), text=line.text))
            current_block_no = line.block_no

    logger.info("[rfp_parser] %s: %d pages, %d blocks, %d tables (%s), %d margin lines removed",
                Path(path).name, page_count, len(blocks), len(doc.tables),
                ", ".join(f"{t.id}:{t.kind}/{len(t.rows)} rows" for t in doc.tables) or "none", removed)
    return doc
