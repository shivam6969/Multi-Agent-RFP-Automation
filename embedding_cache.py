"""
embedding_cache.py
------------------
Disk cache for document embeddings (var/cache/embeddings/*.npy).

The cache key is a hash of the model name, the encode options and the exact
texts, so a cache entry can never be stale: change the catalogue, the RFP or
the model and a new entry is computed. Old entries are simply unused.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from settings import EMBEDDING_CACHE_DIR

logger = logging.getLogger(__name__)


def cache_key(model_name: str, texts: Sequence[str], **encode_kwargs: Any) -> str:
    payload = json.dumps(
        {"model": model_name, "options": encode_kwargs, "texts": list(texts)},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def encode_cached(
    embedder: Any,
    texts: Sequence[str],
    cache_dir: Optional[str] = None,
    **encode_kwargs: Any,
) -> np.ndarray:
    """
    Return float32 embeddings for *texts*, computing them with *embedder*
    (anything with .name and .encode()) only if not already on disk.
    """
    directory = Path(cache_dir or EMBEDDING_CACHE_DIR)
    path = directory / f"{cache_key(embedder.name, texts, **encode_kwargs)}.npy"

    if path.is_file():
        try:
            return np.load(path)
        except (OSError, ValueError):
            logger.warning("Ignoring unreadable embedding cache file %s", path)

    vectors = np.asarray(embedder.encode(texts, **encode_kwargs), dtype="float32")

    directory.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp, vectors)
    os.replace(tmp, path)  # atomic: readers never see a half-written file
    return vectors
