"""
tools/bu_rag.py
---------------
Retrieval-Augmented Generation tool for the Business-Unit product catalog.

Data sources (three complementary indexes)
-------------------------------------------
1. Small chunks   one color/category group per chunk  (fine-grained)
2. Large chunks   one full product range per chunk    (broad context)
3. Product index  one catalogue product per entry, built from the catalogue
                  DB (catalogue.py) with real name, price, SAP, HSN, packing

retrieve() merges all three result sets and re-ranks them with a CrossEncoder
before handing them to the LLM. search_products() searches only the product
index and returns structured ProductRecords, so pricing can pick a product by
id instead of reading prices out of text.

This file is intentionally free of agent logic — it is a pure data tool.
"""

from __future__ import annotations

import os
import threading
from dataclasses import astuple, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np

from settings import (
    BU_EMBED_MODEL,
    BU_JSON_PATH,
    BU_LARGE_CHUNKS_PATH,
    BU_RERANK_MODEL,
    BU_RERANK_TOP_K,
    BU_SMALL_CHUNKS_PATH,
    BU_TOP_K_JSON,
    BU_TOP_K_LARGE,
    BU_TOP_K_SMALL,
    LLM_MODEL,
)
from catalogue import ProductRecord, get_catalogue
from embedding_cache import encode_cached
from llm import chat_completion
from models import get_embedder, get_reranker


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class BuRagConfig:
    small_chunks_path: str = BU_SMALL_CHUNKS_PATH
    large_chunks_path: str = BU_LARGE_CHUNKS_PATH
    json_path: str = BU_JSON_PATH
    embed_model_name: str = BU_EMBED_MODEL
    rerank_model_name: str = BU_RERANK_MODEL
    llm_model_name: str = LLM_MODEL


# ─── Tool ─────────────────────────────────────────────────────────────────────

class BuRagTool:
    """
    Loads product catalog data and exposes retrieve() / run_query().
    """

    def __init__(self, config: BuRagConfig) -> None:
        self.config = config

        print(f"[BuRagTool] Loading embed model: {config.embed_model_name}")
        self.embed_model = get_embedder(config.embed_model_name)

        print(f"[BuRagTool] Loading rerank model: {config.rerank_model_name}")
        self.reranker = get_reranker(config.rerank_model_name)

        # Text chunk indexes
        self.small_chunks = self._load_chunks(config.small_chunks_path)
        self.large_chunks = self._load_chunks(config.large_chunks_path)
        self.small_index, _ = self._create_faiss_index(self.small_chunks)
        self.large_index, _ = self._create_faiss_index(self.large_chunks)

        # Structured product index, built from the catalogue DB (one row per product)
        self.products: List[ProductRecord] = get_catalogue(
            json_path=config.json_path
        ).all_products()
        self.json_texts = [p.search_text() for p in self.products]
        self.json_embeddings = self._embed_docs(self.json_texts)

        print(
            f"[BuRagTool] Ready — "
            f"{len(self.small_chunks)} small / "
            f"{len(self.large_chunks)} large / "
            f"{len(self.json_texts)} JSON products"
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _load_chunks(path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        sep = "=" * 80
        return [c.strip() for c in raw.split(sep) if c.strip()]

    def _embed_docs(self, docs: Sequence[str]) -> np.ndarray:
        prefixed = ["passage: " + d for d in docs]
        return encode_cached(self.embed_model, prefixed, normalize_embeddings=True)

    def _embed_query(self, query: str) -> np.ndarray:
        return self.embed_model.encode(
            ["query: " + query], normalize_embeddings=True
        )

    def _create_faiss_index(
        self, chunks: Sequence[str]
    ) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
        embeddings = self._embed_docs(chunks)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index, embeddings

    def _search_faiss(
        self,
        index: faiss.IndexFlatIP,
        chunks: Sequence[str],
        query: str,
        top_k: int,
    ) -> List[str]:
        q_emb = self._embed_query(query)
        _, indices = index.search(q_emb, top_k)
        return [chunks[i] for i in indices[0]]

    def _top_json_indices(self, query: str, top_k: int) -> List[int]:
        q_emb = self._embed_query(query)
        scores = np.dot(self.json_embeddings, q_emb.T).squeeze()
        return [int(i) for i in np.argsort(scores)[-top_k:][::-1]]

    def _search_json(self, query: str, top_k: int) -> List[str]:
        return [self.json_texts[i] for i in self._top_json_indices(query, top_k)]

    def _rerank(
        self, query: str, candidates: Sequence[str], top_k: int
    ) -> List[str]:
        pairs = [(query, c) for c in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [r[0] for r in ranked[:top_k]]

    # ── Public API ────────────────────────────────────────────────────────────

    def search_products(
        self,
        query: str,
        top_k: int = 8,
        candidate_k: int = 25,
    ) -> List[ProductRecord]:
        """
        Return the catalogue products most relevant to *query*.

        Embedding search over the product index, then cross-encoder rerank.
        Unlike retrieve(), this returns structured records (id, SAP, price),
        so callers can pick a product without parsing text.
        """
        indices = self._top_json_indices(query, candidate_k)
        pairs = [(query, self.json_texts[i]) for i in indices]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)
        return [self.products[i] for i, _ in ranked[:top_k]]


    def retrieve(
        self,
        query: str,
        top_k_small: int = BU_TOP_K_SMALL,
        top_k_large: int = BU_TOP_K_LARGE,
        top_k_json: int = BU_TOP_K_JSON,
        rerank_top_k: int = BU_RERANK_TOP_K,
    ) -> List[str]:
        """Hybrid search across all three sources, then cross-encoder rerank."""
        small = self._search_faiss(
            self.small_index, self.small_chunks, query, top_k_small
        )
        large = self._search_faiss(
            self.large_index, self.large_chunks, query, top_k_large
        )
        json_res = self._search_json(query, top_k_json)
        return self._rerank(query, small + large + json_res, top_k=rerank_top_k)

    def run_query(
        self,
        query: str,
        temperature: float = 0.4,
        max_tokens: int = 400,
        api_key: Optional[str] = None,
    ) -> str:
        """End-to-end: retrieve → build context → LLM answer."""
        chunks = self.retrieve(query)
        context = "\n\n".join(chunks)
        return chat_completion(
            system="You answer strictly from provided product catalog context.",
            user=(
                "You are a product assistant.\n"
                "Answer using ONLY the given context.\n"
                "List clearly if multiple items are present.\n"
                'If truly not present, say "Not available".\n\n'
                f"Context:\n{context}\n\n"
                f"Question:\n{query}\n\n"
                "Answer:"
            ),
            model=self.config.llm_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )


# ─── Module-level singleton ───────────────────────────────────────────────────

# ─── Per-catalogue registry ───────────────────────────────────────────────────
#
# One tool per catalogue config. The key includes each data file's size and
# modification time, so editing the catalogue rebuilds the tool (and the
# embedding cache recomputes only what changed).

_TOOLS: Dict[Tuple, BuRagTool] = {}
_TOOLS_LOCK = threading.Lock()


def _tool_key(cfg: BuRagConfig) -> Tuple:
    stamps = tuple(
        (os.path.getsize(p), os.path.getmtime(p))
        for p in (cfg.small_chunks_path, cfg.large_chunks_path, cfg.json_path)
    )
    return astuple(cfg) + stamps


def get_bu_rag_tool(config: Optional[BuRagConfig] = None) -> BuRagTool:
    """Return the BU tool for this catalogue config, building it on first use."""
    cfg = config or BuRagConfig()
    key = _tool_key(cfg)
    with _TOOLS_LOCK:
        tool = _TOOLS.get(key)
        if tool is None:
            tool = BuRagTool(cfg)
            _TOOLS[key] = tool
        return tool


def clear_bu_tools() -> None:
    """Drop all in-memory BU tools (used by tests)."""
    with _TOOLS_LOCK:
        _TOOLS.clear()
