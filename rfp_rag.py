"""
rfp_rag.py
----------
Retrieval-Augmented Generation tool for RFP (Request for Proposal) PDFs.

Pipeline
--------
  rfp_parser.parse_pdf()   headings, text and tables in reading order
  build_chunks()           three kinds of chunk, each labelled with its
                           section and page so the LLM (and a person) can
                           see where it came from:
                             text  - one section's text (split on line
                                     boundaries if long; nothing is dropped)
                             table - a table as Markdown (long tables in
                                     slices, header repeated in each)
                             row   - one table row as "Header: value; …",
                                     so "how many fan regulators?" finds
                                     the exact row
  retrieve()               hybrid search: BM25 keywords (exact tokens such
                           as "IS 3854", "16A", "EMD") + embeddings, fused
                           with reciprocal rank fusion, then reranked by the
                           cross-encoder already used for the catalogue
  run_query()              answer a question from the retrieved chunks, or
                           from the whole document when it is short

Requirement extraction (rfp_agent.py) uses document_segments(): the document
in reading order, with item-list tables replaced by a short placeholder
because their rows are turned into requirements by rules, not by the LLM.

This file is intentionally free of agent logic: it is a pure data tool.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import faiss
import numpy as np

from settings import (
    RFP_CANDIDATE_K,
    RFP_EMBED_MODEL,
    RFP_MAX_CHUNK_LENGTH,
    RFP_PDF_PATH,
    RFP_QUERY_INSTRUCTION,
    RFP_RERANK_CANDIDATES,
    RFP_RERANK_MAX_CHARS,
    RFP_RERANK_MODEL,
    RFP_TOOL_CACHE_SIZE,
    RFP_TOP_K,
    RFP_USE_RERANKER,
)
from embedding_cache import encode_cached
from llm import chat_completion, current_model
from models import get_embedder, get_reranker
from rfp_parser import Block, ParsedDocument, parse_pdf
from rfp_tables import TableData, row_categories, row_record, table_markdown, table_title

logger = logging.getLogger(__name__)


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class RfpRagConfig:
    pdf_path: str = RFP_PDF_PATH
    embed_model_name: str = RFP_EMBED_MODEL
    llm_model_name: str = field(default_factory=current_model)   # LLM_MODEL, read at use
    max_chunk_length: int = RFP_MAX_CHUNK_LENGTH
    rerank_model_name: str = RFP_RERANK_MODEL
    use_reranker: bool = RFP_USE_RERANKER


# ─── Chunks ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Chunk:
    text: str                      # what is embedded, searched and shown to the LLM
    kind: str                      # "text" | "table" | "row"
    page: int                      # first page
    section: str = ""
    table_id: Optional[str] = None


def _page_label(pages: Sequence[int]) -> str:
    pages = sorted(set(p for p in pages if p))
    if not pages:
        return ""
    return f"p.{pages[0]}" if pages[0] == pages[-1] else f"pp.{pages[0]}-{pages[-1]}"


def _split_text(text: str, max_length: int) -> List[str]:
    """Split on line boundaries, then sentences, then hard cuts. Never drops text."""
    if len(text) <= max_length:
        return [text]
    pieces: List[str] = []
    for line in text.splitlines():
        while len(line) > max_length:
            cut = line.rfind(". ", 0, max_length)
            cut = cut + 1 if cut > max_length // 3 else line.rfind(" ", 0, max_length)
            cut = cut if cut > 0 else max_length
            pieces.append(line[:cut].strip())
            line = line[cut:].strip()
        if line:
            pieces.append(line)
    out: List[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > max_length:
            out.append(current)
            current = piece
        else:
            current = f"{current}\n{piece}" if current else piece
    if current:
        out.append(current)
    return out


def table_slices(table: TableData, max_length: int) -> List[Tuple[str, Tuple[int, int]]]:
    """Markdown slices of a table (header repeated), each about max_length or less."""
    if not table.rows:
        return [(table_markdown(table), (0, 0))]
    slices, start, size = [], 0, 0
    header_len = len(table_markdown(table, rows=[]))
    for i, row in enumerate(table.rows):
        row_len = sum(len(c) + 3 for c in row) + 2
        if i > start and header_len + size + row_len > max_length:
            slices.append((table_markdown(table, rows=table.rows[start:i]), (start + 1, i)))
            start, size = i, 0
        size += row_len
    slices.append((table_markdown(table, rows=table.rows[start:]), (start + 1, len(table.rows))))
    return slices


def _table_chunks(table: TableData, max_length: int) -> List[Chunk]:
    out: List[Chunk] = []
    title = table_title(table)
    slices = table_slices(table, max_length)
    for text, (first, last) in slices:
        span = f" rows {first}-{last} of {len(table.rows)}" if len(slices) > 1 else ""
        out.append(Chunk(f"{title}{span}\n{text}", "table", table.pages[0], table.section, table.id))
    if table.kind == "kv":
        for row in [table.header] + table.rows:
            pair = f"{row[0]}: {' '.join(row[1:])}".strip(": ")
            if pair:
                out.append(Chunk(f"Table {table.id} ({table.section or table.page_label}): {pair}",
                                 "row", table.pages[0], table.section, table.id))
        return out
    row_pages = table.row_pages if len(table.row_pages) == len(table.rows) else [table.pages[0]] * len(table.rows)
    for row, page, category in zip(table.rows, row_pages, row_categories(table)):
        if any(row):
            out.append(Chunk(row_record(table, row, category), "row", page, table.section, table.id))
    return out


def build_chunks(doc: ParsedDocument, max_length: int = RFP_MAX_CHUNK_LENGTH) -> List[Chunk]:
    """Text chunks per section, plus table and row chunks, in document order."""
    chunks: List[Chunk] = []
    lines: List[str] = []
    pages: List[int] = []
    section = ""

    def flush() -> None:
        if lines:
            label = " | ".join(x for x in (section and f"Section: {section}", _page_label(pages)) if x)
            for part in _split_text("\n".join(lines), max_length):
                chunks.append(Chunk(f"[{label}]\n{part}" if label else part, "text", pages[0], section))
        lines.clear()
        pages.clear()

    for b in doc.blocks:
        if b.kind == "table":
            flush()
            if b.table is not None:
                chunks.extend(_table_chunks(b.table, max_length))
            continue
        if b.kind == "heading" or b.section != section:
            flush()
            section = b.section
        lines.append(b.text)
        pages.append(b.page)
    flush()
    return chunks


# ─── Segments for requirement extraction ──────────────────────────────────────

@dataclass(frozen=True)
class Segment:
    text: str
    pages: Tuple[int, int]
    table_id: Optional[str] = None     # set on the placeholder of an item-list table


def document_segments(doc: ParsedDocument, item_table_ids: Set[str], max_chars: int) -> List[Segment]:
    """
    The document in reading order, one segment per section (text) or table.
    Item-list tables in *item_table_ids* become a one-line placeholder; other
    tables are given as Markdown (sliced to fit *max_chars*).
    """
    segments: List[Segment] = []
    current: List[str] = []
    pages: List[int] = []

    def flush() -> None:
        if current:
            segments.append(Segment("\n".join(current), (min(pages), max(pages))))
        current.clear()
        pages.clear()

    for b in doc.blocks:
        if b.kind == "table" and b.table is not None:
            flush()
            t = b.table
            span = (min(t.pages), max(t.pages))
            if t.id in item_table_ids:
                # Name a few items so the LLM knows a DETAILED product list exists
                # (its prompt then skips general scope lines such as "supply
                # switches, sockets and plates"), without giving it rows to copy.
                desc = t.roles.index("description") if "description" in t.roles else None
                names = [r[desc] for r in t.rows if desc is not None and desc < len(r) and r[desc]][:4]
                examples = f" e.g. {'; '.join(names)}; …" if names else ""
                segments.append(Segment(
                    f"{table_title(t)} Item list: the detailed list of products to supply, "
                    f"{len(t.rows)} rows{examples} Its items were read separately: do not extract "
                    "them again, and do not extract a general scope line that summarises them.",
                    span, t.id))
            else:
                for text, _rows in table_slices(t, max(400, max_chars // 2)):
                    segments.append(Segment(f"{table_title(t)}\n{text}", span))
            continue
        if b.kind == "heading":
            flush()
        current.append(b.text)
        pages.append(b.page)
    flush()

    # A section longer than a window is split on line boundaries.
    out: List[Segment] = []
    for s in segments:
        if s.table_id or len(s.text) <= max_chars:
            out.append(s)
        else:
            out.extend(Segment(part, s.pages) for part in _split_text(s.text, max_chars))
    return out


# ─── Keyword search ───────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {"the", "a", "an", "of", "and", "or", "to", "in", "for", "on", "with", "by", "be", "shall",
              "will", "are", "as", "at", "from", "this", "that", "all", "any", "what", "which", "how"}


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25:
    """Okapi BM25 over a fixed list of documents."""

    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_tokens = [Counter(tokenize(d)) for d in docs]
        self.lengths = np.array([sum(c.values()) for c in self.doc_tokens], dtype="float32")
        self.avg_len = float(self.lengths.mean()) if len(docs) else 0.0
        df: Counter = Counter()
        for counts in self.doc_tokens:
            df.update(counts.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.doc_tokens), dtype="float32")
        terms = [t for t in set(tokenize(query)) if t in self.idf]
        if not terms or not self.avg_len:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avg_len)
        for i, counts in enumerate(self.doc_tokens):
            s = 0.0
            for t in terms:
                f = counts.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm[i])
            out[i] = s
        return out


def reciprocal_rank_fusion(rankings: Iterable[Sequence[int]], k: int = 60) -> List[int]:
    """Fuse several best-first index lists into one (higher = better)."""
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


# ─── Tool ─────────────────────────────────────────────────────────────────────

class RfpRagTool:
    """
    Loads, parses and indexes an RFP PDF on first use.
    Thread-safe for reads; not designed for concurrent writes.
    """

    def __init__(self, config: RfpRagConfig) -> None:
        self.config = config

        logger.info(f"[RfpRagTool] Loading embed model: {config.embed_model_name}")
        self.embed_model = get_embedder(config.embed_model_name)
        self._reranker = None
        self._reranker_lock = threading.Lock()

        self.document: ParsedDocument = parse_pdf(config.pdf_path)
        self.chunk_records: List[Chunk] = build_chunks(self.document, config.max_chunk_length)
        self.chunks: List[str] = [c.text for c in self.chunk_records]
        if not self.chunks:
            raise ValueError(
                f"No readable text found in {config.pdf_path}. "
                "If it is a scanned document, it needs OCR before it can be analysed."
            )
        self.full_text: str = self.document.to_text()

        kinds = Counter(c.kind for c in self.chunk_records)
        logger.info(f"[RfpRagTool] Indexing {len(self.chunks)} chunks "
                    f"({kinds.get('text', 0)} text, {kinds.get('table', 0)} table, {kinds.get('row', 0)} row) …")
        embeddings = encode_cached(self.embed_model, self.chunks, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.bm25 = BM25(self.chunks)
        logger.info("[RfpRagTool] Ready.")

    # ── Document views ────────────────────────────────────────────────────────

    @property
    def tables(self) -> List[TableData]:
        return self.document.tables

    def segments(self, item_table_ids: Set[str], max_chars: int) -> List[Segment]:
        return document_segments(self.document, item_table_ids, max_chars)

    # ── Search ────────────────────────────────────────────────────────────────

    def _get_reranker(self):
        with self._reranker_lock:
            if self._reranker is None:
                self._reranker = get_reranker(self.config.rerank_model_name)
            return self._reranker

    def retrieve_chunks(self, query: str, top_k: int = RFP_TOP_K) -> List[Chunk]:
        """The top-k chunks for *query* (hybrid search + rerank), best first."""
        n = len(self.chunk_records)
        if n <= top_k:
            return list(self.chunk_records)
        k = min(RFP_CANDIDATE_K, n)

        q = self.embed_model.encode([RFP_QUERY_INSTRUCTION + query], normalize_embeddings=True)
        _, dense = self.index.search(np.asarray(q, dtype="float32"), k)
        dense_ranking = [int(i) for i in dense[0] if i >= 0]

        keyword = self.bm25.scores(query)
        keyword_ranking = [int(i) for i in np.argsort(-keyword)[:k] if keyword[i] > 0]

        fused = reciprocal_rank_fusion([dense_ranking, keyword_ranking])[:k]
        if self.config.use_reranker and len(fused) > top_k:
            head, tail = fused[:RFP_RERANK_CANDIDATES], fused[RFP_RERANK_CANDIDATES:]
            scores = self._get_reranker().predict(
                [(query, self.chunks[i][:RFP_RERANK_MAX_CHARS]) for i in head])
            fused = [i for i, _ in sorted(zip(head, scores), key=lambda x: x[1], reverse=True)] + tail
        return [self.chunk_records[i] for i in fused[:top_k]]

    def retrieve(self, query: str, top_k: int = RFP_TOP_K) -> List[str]:
        """Return the top-k most relevant RFP chunks for *query*."""
        return [c.text for c in self.retrieve_chunks(query, top_k)]

    def build_context(self, query: str, top_k: int = RFP_TOP_K) -> str:
        return "\n\n".join(self.retrieve(query, top_k=top_k))

    def run_query(
        self,
        query: str,
        top_k: int = RFP_TOP_K,
        temperature: float = 0.1,
        max_tokens: int = 600,
        api_key: Optional[str] = None,
        whole_document: bool = False,
    ) -> str:
        """Answer *query* from the retrieved chunks (or the whole document)."""
        context = self.full_text if whole_document else self.build_context(query, top_k=top_k)
        return chat_completion(
            system="You answer strictly from provided RFP context.",
            user=(
                "You are an assistant that answers questions from RFP documents.\n"
                "Extract relevant information from the context. Tables are given in Markdown "
                "(or as 'Header: value' records); read each row together with its column headers.\n"
                "Be precise and structured. Mention the page (p.N) when the context shows it.\n"
                'If not found, say "Not mentioned in document".\n\n'
                f"Context:\n{context}\n\n"
                f"Question:\n{query}\n\n"
                "Answer:"
            ),
            model=self.config.llm_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )


# ─── Per-RFP registry ─────────────────────────────────────────────────────────
#
# One tool per distinct PDF *content* (not path), so re-uploading the same file
# reuses the index and editing a file builds a new one. The few most recently
# used tools stay in memory; their embeddings also live in the disk cache.

_TOOLS: "OrderedDict[Tuple, RfpRagTool]" = OrderedDict()
_TOOLS_LOCK = threading.Lock()


def _file_sha256(path: str) -> str:
    if not Path(path).is_file():
        raise FileNotFoundError(f"RFP PDF not found: {path}")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tool_key(cfg: RfpRagConfig) -> Tuple:
    return (
        _file_sha256(cfg.pdf_path),
        cfg.embed_model_name,
        cfg.llm_model_name,
        cfg.max_chunk_length,
        cfg.rerank_model_name,
        cfg.use_reranker,
    )


def get_rfp_rag_tool(config: Optional[RfpRagConfig] = None) -> RfpRagTool:
    """Return the RFP tool for this config's PDF, building it on first use."""
    cfg = config or RfpRagConfig()
    key = _tool_key(cfg)
    with _TOOLS_LOCK:
        tool = _TOOLS.get(key)
        if tool is None:
            tool = RfpRagTool(cfg)
            _TOOLS[key] = tool
            while len(_TOOLS) > RFP_TOOL_CACHE_SIZE:
                _TOOLS.popitem(last=False)
        _TOOLS.move_to_end(key)
        return tool


def clear_rfp_tools() -> None:
    """Drop all in-memory RFP tools (used by tests)."""
    with _TOOLS_LOCK:
        _TOOLS.clear()
