"""
tools/bu_rag.py
---------------
Retrieval-Augmented Generation tool for the Business-Unit product catalog.

Data sources (three complementary indexes)
-------------------------------------------
1. Small chunks   one color/category group per chunk  (fine-grained)
2. Large chunks   one full product range per chunk    (broad context)
3. JSON index     structured product rows as text     (exact field lookup)

The three result sets are merged and re-ranked with a CrossEncoder before
being handed to the LLM.

This file is intentionally free of agent logic — it is a pure data tool.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

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
from llm import chat_completion


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
        self.embed_model = SentenceTransformer(config.embed_model_name)

        print(f"[BuRagTool] Loading rerank model: {config.rerank_model_name}")
        self.reranker = CrossEncoder(config.rerank_model_name)

        # Text chunk indexes
        self.small_chunks = self._load_chunks(config.small_chunks_path)
        self.large_chunks = self._load_chunks(config.large_chunks_path)
        self.small_index, _ = self._create_faiss_index(self.small_chunks)
        self.large_index, _ = self._create_faiss_index(self.large_chunks)

        # JSON structured index
        with open(config.json_path, "r", encoding="utf-8-sig") as f:
            json_data = json.load(f)
        products = self._flatten_json_products(json_data)
        self.json_texts = [self._product_to_text(p) for p in products]
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
        return self.embed_model.encode(prefixed, normalize_embeddings=True)

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

    def _search_json(self, query: str, top_k: int) -> List[str]:
        q_emb = self._embed_query(query)
        scores = np.dot(self.json_embeddings, q_emb.T).squeeze()
        top_idx = np.argsort(scores)[-top_k:][::-1]
        return [self.json_texts[i] for i in top_idx]

    def _rerank(
        self, query: str, candidates: Sequence[str], top_k: int
    ) -> List[str]:
        pairs = [(query, c) for c in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [r[0] for r in ranked[:top_k]]

    @staticmethod
    def _flatten_json_products(data: Dict) -> List[Dict]:
        brand = data.get("brand", "N/A")
        products: List[Dict] = []
        for range_entry in data.get("product_data", []):
            range_name = range_entry.get("range", "N/A")
            base = {"brand": brand, "range": range_name}
            if "items" in range_entry:
                for item in range_entry["items"]:
                    products.append({**base, "variant": None, **item})
            elif "colors" in range_entry:
                for cb in range_entry["colors"]:
                    for item in cb.get("items", []):
                        products.append({**base, "variant": cb["color"], **item})
            elif "categories" in range_entry:
                for cb in range_entry["categories"]:
                    for item in cb.get("items", []):
                        products.append(
                            {**base, "variant": cb["category"], **item}
                        )
        return products

    @staticmethod
    def _product_to_text(p: Dict) -> str:
        return (
            f"Product: {p.get('product', 'N/A')}\n"
            f"Brand: {p.get('brand', 'N/A')}\n"
            f"Range: {p.get('range', 'N/A')}\n"
            f"Variant: {p.get('variant', 'N/A')}\n"
            f"Price: {p.get('price', 'N/A')}\n"
            f"SAP Code: {p.get('sap_code', 'N/A')}\n"
            f"HSN Code: {p.get('hsn_code', 'N/A')}\n"
            f"Packing: {p.get('packing', 'N/A')}\n"
            f"Monthly Available: {p.get('monthly_avalaible', 'N/A')}\n"
        )

    # ── Public API ────────────────────────────────────────────────────────────

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

_INSTANCE: Optional[BuRagTool] = None


def get_bu_rag_tool(config: Optional[BuRagConfig] = None) -> BuRagTool:
    """Return (and lazily create) the module-level singleton."""
    global _INSTANCE
    cfg = config or BuRagConfig()
    if _INSTANCE is None:
        _INSTANCE = BuRagTool(cfg)
    return _INSTANCE
