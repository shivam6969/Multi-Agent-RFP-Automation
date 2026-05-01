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

import re
from dataclasses import dataclass
from typing import List, Optional

import faiss
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer

from settings import (
    LLM_MODEL,
    RFP_EMBED_MODEL,
    RFP_MAX_CHUNK_LENGTH,
    RFP_PDF_PATH,
    RFP_TOP_K,
)
from llm import chat_completion


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

        print(f"[RfpRagTool] Loading embed model: {config.embed_model_name}")
        self.embed_model = SentenceTransformer(config.embed_model_name)

        raw_text = self._extract_text(config.pdf_path)
        sections = self._chunk_by_sections(raw_text)
        self.chunks: List[str] = self._split_large_chunks(
            sections, max_length=config.max_chunk_length
        )

        print(f"[RfpRagTool] Indexing {len(self.chunks)} chunks …")
        embeddings = self.embed_model.encode(
            self.chunks, show_progress_bar=False
        ).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        print("[RfpRagTool] Ready.")

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_text(pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
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
        _, indices = self.index.search(q_emb, top_k)
        return [self.chunks[i] for i in indices[0]]

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


# ─── Module-level singleton ───────────────────────────────────────────────────

_INSTANCE: Optional[RfpRagTool] = None


def get_rfp_rag_tool(config: Optional[RfpRagConfig] = None) -> RfpRagTool:
    """Return (and lazily create) the module-level singleton."""
    global _INSTANCE
    cfg = config or RfpRagConfig()
    if _INSTANCE is None:
        _INSTANCE = RfpRagTool(cfg)
    return _INSTANCE
