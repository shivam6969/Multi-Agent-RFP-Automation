"""
models.py
---------
One shared, lazily loaded instance of each embedding / reranker model.

Loading a model takes seconds and hundreds of MB, so every RAG tool (one per
RFP, one per catalogue) shares the same instances. Each wrapper holds a lock
around inference so several threads (e.g. parallel matching) can use one
model safely.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Any, List, Sequence, Tuple

import numpy as np

_load_lock = threading.Lock()


class Embedder:
    """Thread-safe wrapper around a SentenceTransformer."""

    def __init__(self, name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = name
        self._model = SentenceTransformer(name)
        self._lock = threading.Lock()

    def encode(self, texts: Sequence[str], **kwargs: Any) -> np.ndarray:
        kwargs.setdefault("show_progress_bar", False)
        with self._lock:
            return self._model.encode(list(texts), **kwargs)


class Reranker:
    """Thread-safe wrapper around a CrossEncoder."""

    def __init__(self, name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.name = name
        self._model = CrossEncoder(name)
        self._lock = threading.Lock()

    def predict(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        with self._lock:
            return list(self._model.predict(list(pairs), show_progress_bar=False))


@lru_cache(maxsize=None)
def _embedder(name: str) -> Embedder:
    return Embedder(name)


@lru_cache(maxsize=None)
def _reranker(name: str) -> Reranker:
    return Reranker(name)


def get_embedder(name: str) -> Embedder:
    with _load_lock:  # never load the same model twice from parallel threads
        return _embedder(name)


def get_reranker(name: str) -> Reranker:
    with _load_lock:
        return _reranker(name)
