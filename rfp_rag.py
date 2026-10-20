"""
tools/rfp_rag.py
----------------
Retrieval-Augmented Generation tool for RFP (Request for Proposal) PDFs.

Responsibilities
----------------
- Extract text from a PDF and chunk it by section numbers.
- Build a FAISS index over chunk embeddings.
- Expose  retrieve()  and  run_query()  as the public API.

This file is intentionally free of agent logic — it is a pure data tool.
"""

from __future__ import annotations

import logging
import hashlib
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import pymupdf

from settings import (
    LLM_MODEL,
    RFP_EMBED_MODEL,
    RFP_MAX_CHUNK_LENGTH,
    RFP_PDF_PATH,
    RFP_TOOL_CACHE_SIZE,
    RFP_TOP_K,
)
from embedding_cache import encode_cached
from llm import chat_completion
from models import get_embedder

logger = logging.getLogger(__name__)


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class RfpRagConfig:
    pdf_path: str = RFP_PDF_PATH
    embed_model_name: str = RFP_EMBED_MODEL
    llm_model_name: str = LLM_MODEL
    max_chunk_length: int = RFP_MAX_CHUNK_LENGTH


# ─── Tool ─────────────────────────────────────────────────────────────────────

class RfpRagTool:
    """
    Loads and indexes an RFP PDF on first use.
    Thread-safe for reads; not designed for concurrent writes.
    """

    def __init__(self, config: RfpRagConfig) -> None:
        self.config = config

        logger.info(f"[RfpRagTool] Loading embed model: {config.embed_model_name}")
        self.embed_model = get_embedder(config.embed_model_name)

        raw_text = self._extract_text(config.pdf_path)
        sections = self._chunk_by_sections(raw_text)
        self.chunks: List[str] = self._split_large_chunks(
            sections, max_length=config.max_chunk_length
        )
        if not self.chunks:
            raise ValueError(
                f"No readable text found in {config.pdf_path}. "
                "If it is a scanned document, it needs OCR before it can be analysed."
            )

        logger.info(f"[RfpRagTool] Indexing {len(self.chunks)} chunks …")
        embeddings = encode_cached(self.embed_model, self.chunks)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        logger.info("[RfpRagTool] Ready.")

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_text(pdf_path: str) -> str:
        if not Path(pdf_path).is_file():
            raise FileNotFoundError(f"RFP PDF not found: {pdf_path}")
        with pymupdf.open(pdf_path) as doc:
            return "".join(page.get_text() for page in doc)

    @staticmethod
    def _chunk_by_sections(text: str) -> List[str]:
        pattern = r"(?=\n?\d+\.\s)|(?=\n?\d+\.\d+)"
        raw = re.split(pattern, text)
        return [c.strip() for c in raw if len(c.strip()) > 80]

    @staticmethod
    def _split_large_chunks(chunks: List[str], max_length: int) -> List[str]:
        final: List[str] = []
        for chunk in chunks:
            if len(chunk) <= max_length:
                final.append(chunk)
                continue
            sentences = chunk.split(". ")
            temp = ""
            for sent in sentences:
                if len(temp) + len(sent) < max_length:
                    temp += sent + ". "
                else:
                    if temp:
                        final.append(temp.strip())
                    temp = sent + ". "
            if temp:
                final.append(temp.strip())
        return final

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = RFP_TOP_K) -> List[str]:
        """Return the top-k most relevant RFP chunks for *query*."""
        q_emb = self.embed_model.encode([query]).astype("float32")
        _, indices = self.index.search(q_emb, min(top_k, len(self.chunks)))
        return [self.chunks[i] for i in indices[0] if i >= 0]

    def build_context(self, query: str, top_k: int = RFP_TOP_K) -> str:
        return "\n\n".join(self.retrieve(query, top_k=top_k))

    def run_query(
        self,
        query: str,
        top_k: int = RFP_TOP_K,
        temperature: float = 0.1,
        max_tokens: int = 400,
        api_key: Optional[str] = None,
    ) -> str:
        """Retrieve relevant chunks then generate an answer with the LLM."""
        context = self.build_context(query, top_k=top_k)
        return chat_completion(
            system="You answer strictly from provided RFP context.",
            user=(
                "You are an assistant that answers questions from RFP documents.\n"
                "Extract relevant information from the context.\n"
                "Be precise and structured.\n"
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
