"""
agents/rfp_agent.py
-------------------
RFP Agent — LangGraph node

Responsibility
--------------
  1. Answer the user's query from the RFP document (state["rfp_answer"]).
  2. Extract a structured list of RFP requirements for downstream agents
     (state["rfp_requirements"]): [{id, requirement, criticality, weight,
     quantity, unit?, origin, source}].

How requirements are found
--------------------------
  - Item-list tables (BOQ / schedule of requirements; see rfp_tables.py) are
    read by RULES: one requirement per item row, the quantity taken from the
    quantity cell (or the sum of per-location columns), the unit, and the
    page / table / row it came from. No LLM guesses which number belongs to
    which item. If a table's headers are unusual (no recognisable description
    or quantity column) one small LLM call maps its columns first.
  - Everything else (narrative sections, and the other tables given to the
    LLM as Markdown) is read by the LLM in windows processed in parallel;
    every reply is schema-validated (llm_json.py). Item tables appear there
    only as a one-line placeholder, so their rows are not extracted twice.

Both outputs are produced in a single run to avoid redundant PDF retrieval.
state["rfp_tables"] gets a compact summary of the tables that were found.
"""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, field_validator

from settings import LLM_MAX_CONCURRENCY, OPENROUTER_API_KEY, RFP_EXTRACT_WINDOW_CHARS, RFP_TOP_K
from rfp_rag import RfpRagConfig, Segment, get_rfp_rag_tool
from rfp_tables import (
    TableData,
    boq_requirements,
    classify_table,
    clean_cell,
    is_piece_unit,
    parse_quantity,
    table_markdown,
)
from llm_json import LLMOutputError, chat_json
from state import AgentState

logger = logging.getLogger(__name__)

_W = 72  # print banner width


def _wrap(text: str, indent: int = 2) -> List[str]:
    """Wrap text to fit inside the banner, yielding padded lines."""
    width = _W - indent - 2  # 2 for ║ chars
    lines = []
    for raw_line in (text or "").splitlines() or [""]:
        if not raw_line:
            lines.append("")
            continue
        for i in range(0, max(len(raw_line), 1), width):
            lines.append(raw_line[i : i + width])
    return lines


# Box-drawing report helpers: append lines to *out*; the node logs it once.
def _banner(out: List[str], title: str) -> None:
    out += [f"╔{'═' * (_W - 2)}╗", f"║  {title:<{_W - 4}}║", f"╠{'═' * (_W - 2)}╣"]


def _section(out: List[str], label: str) -> None:
    out += [f"║  {label:<{_W - 4}}║", f"╠{'─' * (_W - 2)}╣"]


def _line(out: List[str], text: str, indent: int = 2) -> None:
    pad = " " * indent
    out.append(f"║{pad}{text:<{_W - indent - 2}}║")


def _footer(out: List[str]) -> None:
    out.append(f"╚{'═' * (_W - 2)}╝")


# ─── Requirement extraction (LLM, for text and non-item tables) ───────────────

_REQ_EXTRACTION_PROMPT = """
You are an expert procurement analyst.

Extract every requirement stated in this part of an RFP as a JSON array of
objects. Include product line items (one entry per product line as written),
technical specifications, eligibility criteria, delivery/logistics terms and
submission obligations.

If the RFP gives a general scope summary (e.g. "supply switches, plates and
accessories") AND a detailed list of the same products, extract only the
detailed list items, not the summary line.

Tables are given in Markdown (or as "label: value" lines); read each row
together with its column headers. A line starting "[Table T…] … Item list"
stands for the detailed product list, which was read separately: do NOT
extract its items, and treat it as the detailed list for the rule above (so a
scope summary of the same products is not extracted either).
A consignee table (consignee / address / quantity / delivery days) says where
goods are delivered: extract at most one delivery requirement per row (e.g.
"Deliver to <consignee>, <place> within 15 days") with "quantity": null. An
address, buyer name or contact person is never a product.

Each object must have:
  - "requirement"  : short plain-English statement of the requirement
  - "criticality"  : one of
        "mandatory" — the RFP says must / shall / mandatory, or failing it
                      disqualifies the bid (e.g. eligibility criteria)
        "high"      — core product lines and key commercial terms
        "medium"    — supporting specifications and documents
        "low"       — nice-to-have or "where applicable" items
  - "quantity"     : integer number of units ONLY if the RFP states an explicit
                     order quantity for this specific item, otherwise null.
                     Do not use monthly supply estimates, packing formats or
                     numbers that are part of a product name (e.g. "RJ-45").

If this part contains no requirements, return [].
Return ONLY valid JSON. No markdown, no explanation.

RFP text (part {part} of {parts}):
{context}
""".strip()

CRITICALITY_WEIGHTS = {"mandatory": 4.0, "high": 3.0, "medium": 2.0, "low": 1.0}


class _Requirement(BaseModel):
    requirement: str
    criticality: str = "medium"
    quantity: Optional[int] = None

    @field_validator("criticality", mode="before")
    @classmethod
    def _known_criticality(cls, value: Any) -> str:
        value = str(value or "").strip().lower()
        return value if value in CRITICALITY_WEIGHTS else "medium"

    @field_validator("quantity", mode="before")
    @classmethod
    def _positive_int_or_none(cls, value: Any) -> Optional[int]:
        try:
            number = int(str(value).replace(",", "").strip())
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None


def _group_segments(segments: List[Segment], max_chars: int) -> List[List[Segment]]:
    """Consecutive segments in groups of at most ~max_chars (a segment is never split)."""
    groups: List[List[Segment]] = []
    current: List[Segment] = []
    size = 0
    for seg in segments:
        if current and size + len(seg.text) > max_chars:
            groups.append(current)
            current, size = [], 0
        current.append(seg)
        size += len(seg.text)
    if current:
        groups.append(current)
    return groups


def _windows(chunks: List[str], max_chars: int) -> List[str]:
    """Group consecutive chunks into windows of at most ~max_chars."""
    return ["\n\n".join(s.text for s in group)
            for group in _group_segments([Segment(c, (0, 0)) for c in chunks], max_chars)]


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


_FILLER = {"supply", "supplying", "provide", "providing", "of", "the", "a", "an", "and", "for", "to",
           "nos", "no", "pcs", "item", "items", "qty", "quantity", "units", "unit"}


def _tokens(text: str) -> set:
    return {w.rstrip("s") if len(w) > 3 else w for w in _normalise(text).split()} - _FILLER


def _adds_nothing(text: str, table_token_sets: List[set]) -> bool:
    """An LLM item whose words all appear in one table item (i.e. a re-extraction)."""
    words = _tokens(text)
    return len(words) >= 2 and any(words <= t for t in table_token_sets)


_SUPPLY_START = re.compile(r"^(supply|supplying|provision|provide|providing|procure|procurement)\b", re.I)
_GENERIC_GOODS = {"accessorie", "accessory", "item", "material", "product", "good", "equipment", "store",
                  "fitting", "device", "electrical", "wiring"}
_COMPLIANCE_WORDS = {"only", "must", "shall", "isi", "bis", "marked", "certified", "standard", "brand", "make",
                     "warranty", "within", "deliver", "delivery", "sample", "test", "approved", "genuine"}


_PIN_OR_CONTACT = re.compile(
    r"(^\s*\d{6}\s*[,\-])|(\b\d{6}\s*$)|[\w.+-]+@[\w-]+\.[\w.]+|(\+91[\s-]?)?\b[6-9]\d{9}\b", re.I)
_OBLIGATION = re.compile(
    r"\b(deliver|delivery|supply|submit|provide|install|must|shall|should|required|within|days?)\b", re.I)


def _is_bare_address(text: str) -> bool:
    """
    A consignee address or contact line ("140124, Bhakra Beas Management
    Board, Irrigation Wing, NANGAL TOWNSHIP", an e-mail or phone number) with
    no obligation in it. "Deliver to …, Nangal within 15 days" is kept.
    """
    return bool(_PIN_OR_CONTACT.search(text)) and not _OBLIGATION.search(text)


def _is_scope_summary(text: str, quantity: Optional[int], table_words: set) -> bool:
    """
    A general scope line ("Supply modular switches, sockets and plates",
    "Supply electrical wiring accessories") when the RFP's item table already
    lists the products. Llama 4 Scout extracts these despite the prompt; as a
    requirement they get matched to a random product or listed twice.
    Kept: anything with a quantity, a number (a rating such as 16A) or a
    compliance word ("Supply ISI marked products only").
    """
    if quantity is not None or re.search(r"\d", text) or not _SUPPLY_START.match(text.strip()):
        return False
    words = {w.rstrip("s") if len(w) > 3 else w for w in _normalise(text).split()}
    if words & _COMPLIANCE_WORDS or len(words) > 14:
        return False
    content = _tokens(text) - {"supply", "supplying", "provision", "procure", "procurement"}
    return bool(content) and bool(content & (table_words | _GENERIC_GOODS))


# ─── Tables whose headers the rules do not recognise ──────────────────────────

_COLUMNS_PROMPT = """
This table comes from an RFP. Columns are numbered from 0.

Columns: {columns}
First rows:
{rows}

Is it a list of items to supply with a quantity per item? If so, say which
column holds the item description, which the quantity to supply and which the
unit (or null). A column of serial numbers, rates or amounts is not the
quantity.

Return ONLY a JSON object:
{{"is_item_list": true | false, "description": <index or null>, "quantity": <index or null>, "unit": <index or null>}}
""".strip()


class _ColumnMap(BaseModel):
    is_item_list: bool = False
    description: Optional[int] = None
    quantity: Optional[int] = None
    unit: Optional[int] = None


_MAX_COLUMN_MAPPINGS = 4   # LLM calls per RFP at most


def _needs_column_mapping(table: TableData) -> bool:
    """A table the rules did not recognise that has a numeric (non-serial, non-price) column."""
    if table.kind != "other" or len(table.header) < 3 or len(table.rows) < 2:
        return False
    for c, role in enumerate(table.roles):
        if role in ("sno", "price", "quantity", "quantity_part", "location", "delivery"):
            continue
        values = [clean_cell(r[c]) for r in table.rows if c < len(r) and clean_cell(r[c])]
        if values and sum(parse_quantity(v)[0] is not None for v in values) / len(values) >= 0.6:
            return True
    return False


def _map_columns(table: TableData, api_key: Optional[str]) -> bool:
    """Ask the LLM which columns hold description / quantity / unit; True if now an item list."""
    columns = ", ".join(f"{i}: {h or '(blank)'}" for i, h in enumerate(table.header))
    try:
        mapped = chat_json(
            system="You identify the columns of procurement tables. Return JSON only.",
            user=_COLUMNS_PROMPT.format(columns=columns, rows=table_markdown(table, rows=table.rows[:4])),
            schema=_ColumnMap,
            temperature=0.0,
            max_tokens=120,
            api_key=api_key,
        )
    except LLMOutputError:
        return False
    n = len(table.header)

    def valid(i: Optional[int]) -> bool:
        return i is not None and 0 <= i < n

    if not (mapped.is_item_list and valid(mapped.description) and valid(mapped.quantity)
            and mapped.description != mapped.quantity):
        return False
    roles = list(table.roles)
    roles[mapped.description] = "description"
    roles[mapped.quantity] = "quantity"
    if valid(mapped.unit) and mapped.unit not in (mapped.description, mapped.quantity):
        roles[mapped.unit] = "unit"
    kind = classify_table(table.header, table.rows, roles)
    if kind != "boq":
        return False
    table.roles, table.kind = roles, kind
    return True


# ─── Item-list rows -> requirement dicts (rules, no LLM) ──────────────────────

def _table_requirements(table: TableData) -> Tuple[List[Dict[str, Any]], List[str]]:
    """(requirements without ids, warnings) for one item-list table."""
    reqs: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for r in boq_requirements(table):
        item: Dict[str, Any] = {
            "requirement": r.requirement,
            "criticality": "high",
            "weight":      CRITICALITY_WEIGHTS["high"],
            "quantity":    r.quantity,
            "unit":        r.unit or None,
            "origin":      "table",
            "source": {
                "page":    r.page,
                "table":   table.id,
                "row":     r.row_label,
                "section": table.section,
            },
        }
        if r.attributes:
            item["attributes"] = r.attributes
        if r.quantity_breakdown:
            item["quantity_breakdown"] = r.quantity_breakdown
        where = f"table {table.id} row {r.row_label}, p.{r.page}"
        if r.note:
            warnings.append(f"\"{r.requirement[:60]}\" ({where}): {r.note}.")
        if r.quantity is not None and r.unit and not is_piece_unit(r.unit):
            warnings.append(f"\"{r.requirement[:60]}\" ({where}) is quantified in '{r.unit}', "
                            "not pieces; check the quantity before pricing.")
        reqs.append(item)
    if reqs and all(x["quantity"] is None for x in reqs):
        warnings.append(f"Table {table.id} ({table.page_label}) lists items but no quantity could be read; "
                        "enter quantities during review.")
    return reqs, warnings


def _extract_requirements(
    rfp_tool,
    api_key: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Read the WHOLE RFP and return
    ([{id, requirement, criticality, weight, quantity, unit?, origin, source}], warnings).
    """
    tables: List[TableData] = list(getattr(rfp_tool, "tables", None) or [])

    for table in [t for t in tables if _needs_column_mapping(t)][:_MAX_COLUMN_MAPPINGS]:
        if _map_columns(table, api_key):
            logger.info("[RfpAgent] Table %s read as an item list after column mapping.", table.id)

    item_tables = {t.id: t for t in tables if t.kind == "boq"}
    from_tables: Dict[str, List[Dict[str, Any]]] = {}
    table_warnings: List[str] = []
    for table_id, table in item_tables.items():
        from_tables[table_id], found_warnings = _table_requirements(table)
        table_warnings.extend(found_warnings)

    if hasattr(rfp_tool, "segments"):
        segments = rfp_tool.segments(set(item_tables), RFP_EXTRACT_WINDOW_CHARS)
    else:  # a tool without a parsed document: its chunks are the text
        chunks = getattr(rfp_tool, "chunks", None) or [
            rfp_tool.build_context("requirements eligibility deliverables mandatory technical", top_k=5)
        ]
        segments = [Segment(c, (0, 0)) for c in chunks]
    groups = _group_segments(segments, RFP_EXTRACT_WINDOW_CHARS)
    # A window holding only item-table placeholders needs no LLM call.
    llm_groups = [(i, g) for i, g in enumerate(groups) if any(s.table_id is None for s in g)]

    def extract(numbered: Tuple[int, Tuple[int, List[Segment]]]) -> Optional[List[_Requirement]]:
        part, (_index, group) = numbered
        try:
            return chat_json(
                system="You extract structured data from RFP documents. Return JSON only.",
                user=_REQ_EXTRACTION_PROMPT.format(
                    context="\n\n".join(s.text for s in group), part=part, parts=len(llm_groups)),
                schema=List[_Requirement],
                temperature=0.0,
                max_tokens=3000,
                api_key=api_key,
            )
        except LLMOutputError:
            return None

    with ThreadPoolExecutor(max_workers=max(1, LLM_MAX_CONCURRENCY)) as pool:
        results = list(pool.map(extract, enumerate(llm_groups, 1)))
    llm_by_group = {index: found for (index, _group), found in zip(llm_groups, results)}

    warnings = [
        f"Requirement extraction failed for part {i} of {len(llm_groups)} of the RFP; "
        "some requirements may be missing."
        for i, r in enumerate(results, 1) if r is None
    ] + table_warnings

    table_token_sets = [_tokens(r["requirement"]) for reqs in from_tables.values() for r in reqs]
    table_words = set().union(*table_token_sets) if table_token_sets else set()
    requirements: List[Dict[str, Any]] = []
    seen: Dict[str, Dict[str, Any]] = {}

    def add(item: Dict[str, Any]) -> None:
        key = _normalise(item["requirement"])
        if len(key) < 6:
            return
        if key in seen:  # same requirement repeated (across windows or table rows)
            if seen[key]["quantity"] is None:
                seen[key]["quantity"] = item.get("quantity")
            return
        item = {"id": f"R{len(requirements) + 1}", **item}
        seen[key] = item
        requirements.append(item)

    for index, group in enumerate(groups):
        # The RFP's item list first, then what the LLM found in this window.
        for seg in group:
            for item in from_tables.get(seg.table_id or "", []):
                add(dict(item))
        pages = [p for s in group for p in s.pages if p]
        for req in llm_by_group.get(index) or []:
            text = req.requirement.strip()
            if table_token_sets and _adds_nothing(text, table_token_sets):
                continue   # the LLM repeated an item the table already gave
            if _is_bare_address(text):
                logger.info("[RfpAgent] Dropped an address / contact line: %r", text)
                continue
            if table_token_sets and _is_scope_summary(text, req.quantity, table_words):
                logger.info("[RfpAgent] Dropped scope summary (the item table lists the products): %r", text)
                continue
            item: Dict[str, Any] = {
                "requirement": text,
                "criticality": req.criticality,
                "weight":      CRITICALITY_WEIGHTS[req.criticality],
                "quantity":    req.quantity,
                "origin":      "text",
            }
            if pages:
                item["source"] = {"pages": [min(pages), max(pages)]}
            add(item)
    return requirements, warnings


# ─── LangGraph node ───────────────────────────────────────────────────────────

def rfp_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node.
    Reads user_query (+ optional rfp_pdf_path), writes rfp_answer,
    rfp_requirements and rfp_tables.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
    query = state.get("user_query", "")

    pdf_path = state.get("rfp_pdf_path") or RfpRagConfig().pdf_path
    rfp_tool = get_rfp_rag_tool(RfpRagConfig(pdf_path=pdf_path))

    logger.info("[RfpAgent] Answering user query from RFP …")
    # Short RFPs: answer from the whole document, tables as Markdown (the
    # nearest chunks alone missed the buyer's name on the sample RFP).
    # Long ones: hybrid search over text, table and row chunks.
    full_text = getattr(rfp_tool, "full_text", None) or "\n\n".join(getattr(rfp_tool, "chunks", []))
    whole_doc = 0 < len(full_text) <= RFP_EXTRACT_WINDOW_CHARS
    rfp_answer = rfp_tool.run_query(query, top_k=RFP_TOP_K, api_key=api_key, whole_document=whole_doc)
    state["rfp_answer"] = rfp_answer

    logger.info("[RfpAgent] Extracting structured requirements …")
    requirements, warnings = _extract_requirements(rfp_tool, api_key)
    state["rfp_requirements"] = requirements
    state["rfp_tables"] = [t.summary() for t in (getattr(rfp_tool, "tables", None) or [])]
    if warnings:
        state["warnings"] = list(state.get("warnings", [])) + warnings

    # ── Report for the log ─────────────────────────────────────────────────────
    crit_icon = {"mandatory": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}

    out: List[str] = [""]
    _banner(out, "RFP AGENT — OUTPUT")
    _line(out, f"Query: {query[:_W - 10]}")
    _section(out, "ANSWER")
    for line in _wrap(rfp_answer):
        _line(out, line)
    out.append(f"╠{'═' * (_W - 2)}╣")
    tables = state["rfp_tables"]
    if tables:
        _section(out, f"TABLES  ({len(tables)} found)")
        for t in tables:
            _line(out, f"{t['id']:<4} {t['kind']:<6} {t['rows']:>3} rows  p.{t['pages'][0]}  "
                       f"{', '.join(c for c in t['columns'] if c)}"[:_W - 4])
        out.append(f"╠{'═' * (_W - 2)}╣")
    _section(out, f"EXTRACTED REQUIREMENTS  ({len(requirements)} total)")
    for i, req in enumerate(requirements, 1):
        crit = req.get("criticality", "medium")
        icon = crit_icon.get(crit, "•")
        wt   = req.get("weight", 1.0)
        text = req.get("requirement", "")
        _line(out, f"{i:>2}. {icon} [{crit.upper():<9}] wt={wt}  {text[:_W - 30]}", indent=2)
        if len(text) > _W - 30:
            _line(out, f"    ↳ …{text[_W - 30:]}", indent=2)
        src = req.get("source") or {}
        if req.get("quantity") or src.get("table"):
            qty = f"qty {req['quantity']:,} {req.get('unit') or ''}".strip() if req.get("quantity") else "qty ?"
            where = f"  ← table {src['table']} row {src.get('row')}, p.{src.get('page')}" if src.get("table") else ""
            _line(out, f"    {qty}{where}"[:_W - 6], indent=2)
    _footer(out)
    logger.info("\n".join(out))

    return state
