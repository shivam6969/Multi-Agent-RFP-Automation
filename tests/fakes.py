"""Lightweight stand-ins for the embedding and reranker models."""

from __future__ import annotations

import hashlib
from typing import List, Sequence, Tuple

import numpy as np


class FakeEmbedder:
    """Deterministic bag-of-words vectors; counts how many texts it encodes."""

    dim = 64

    def __init__(self, name: str = "fake-embedder") -> None:
        self.name = name
        self.encoded_texts = 0

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for word in text.lower().split():
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def encode(self, texts: Sequence[str], **_kwargs) -> np.ndarray:
        self.encoded_texts += len(texts)
        return np.stack([self._vector(t) for t in texts])


class FakeReranker:
    name = "fake-reranker"

    def predict(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        return [float(len(set(q.lower().split()) & set(d.lower().split()))) for q, d in pairs]
