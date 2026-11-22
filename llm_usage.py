"""
llm_usage.py
------------
A record of every LLM call (provider, model, primary/fallback, tokens,
latency, cost), used by the evaluation harness and for logging fallbacks.

llm.py calls record() once per attempt. UsageTracker collects the records
made while it is open:

    with UsageTracker() as usage:
        run_something()
    summarize(usage.records)   # per-model calls, tokens, cost, latency

Cost: OpenRouter returns the real charge on each response (usage.cost);
for other providers it is estimated from settings.MODEL_PRICES_USD_PER_MTOK
and marked cost_source="price_table".
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from settings import MODEL_PRICES_USD_PER_MTOK

_MAX_RECORDS = 20_000


@dataclass
class CallRecord:
    at: float
    provider: str                 # "openrouter" | "gemini" | host of an OpenAI-compatible API
    model: str
    role: str                     # "primary" | "fallback"
    ok: bool
    latency_s: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    cost_source: Optional[str] = None   # "provider" | "price_table" | None
    error: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


_lock = threading.Lock()
_records: List[CallRecord] = []
_dropped = 0   # records discarded from the front when the list is full


def estimate_cost(model: str, prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> Optional[float]:
    price = MODEL_PRICES_USD_PER_MTOK.get(model)
    if price is None or prompt_tokens is None:
        return None
    return (prompt_tokens * price[0] + (completion_tokens or 0) * price[1]) / 1_000_000


def record(
    *,
    provider: str,
    model: str,
    role: str,
    ok: bool,
    latency_s: float,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    cost_usd: Optional[float] = None,
    error: str = "",
) -> CallRecord:
    global _dropped
    source = "provider" if cost_usd is not None else None
    if cost_usd is None and ok:
        cost_usd = estimate_cost(model, prompt_tokens, completion_tokens)
        source = "price_table" if cost_usd is not None else None
    rec = CallRecord(time.time(), provider, model, role, ok, round(latency_s, 3), prompt_tokens,
                     completion_tokens, cost_usd, source, error[:300])
    with _lock:
        _records.append(rec)
        if len(_records) > _MAX_RECORDS:
            extra = len(_records) - _MAX_RECORDS
            del _records[:extra]
            _dropped += extra
    return rec


class UsageTracker:
    """Collects the calls recorded between __enter__ and __exit__ (all threads)."""

    def __enter__(self) -> "UsageTracker":
        with _lock:
            self._start = _dropped + len(_records)
        self._end: Optional[int] = None
        self.started = time.perf_counter()
        self.elapsed_s = 0.0
        return self

    def __exit__(self, *exc: Any) -> None:
        with _lock:
            self._end = _dropped + len(_records)
        self.elapsed_s = time.perf_counter() - self.started

    @property
    def records(self) -> List[CallRecord]:
        with _lock:
            end = self._end if self._end is not None else _dropped + len(_records)
            lo, hi = max(0, self._start - _dropped), max(0, end - _dropped)
            return list(_records[lo:hi])


def summarize(records: List[CallRecord]) -> Dict[str, Any]:
    """Totals overall and per model."""
    def block(rs: List[CallRecord]) -> Dict[str, Any]:
        ok = [r for r in rs if r.ok]
        lat = [r.latency_s for r in ok]
        costs = [r.cost_usd for r in ok if r.cost_usd is not None]
        return {
            "calls": len(ok),
            "failed_attempts": len(rs) - len(ok),
            "fallback_calls": sum(1 for r in ok if r.role == "fallback"),
            "prompt_tokens": sum(r.prompt_tokens or 0 for r in ok),
            "completion_tokens": sum(r.completion_tokens or 0 for r in ok),
            "cost_usd": round(sum(costs), 6) if costs else None,
            "cost_estimated": any(r.cost_source == "price_table" for r in ok),
            "latency_mean_s": round(statistics.mean(lat), 2) if lat else None,
            "latency_p95_s": round(sorted(lat)[max(0, int(len(lat) * 0.95) - 1)], 2) if lat else None,
        }

    models = sorted({r.model for r in records})
    return {
        "total": block(records),
        "by_model": {m: block([r for r in records if r.model == m]) for m in models},
    }
