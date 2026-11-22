"""
utils/llm.py
------------
Thin wrappers around the LLM providers so every agent uses the same
initialisation, retry and fallback logic:

  chat_completion()    OpenRouter (OpenAI-compatible), used by all agents
  gemini_completion()  Google Gemini, used to write long proposal text

Fallback chain (keys from .env; see settings.py and .env.example)
--------------
  chat:    OPENROUTER_API_KEY / LLM_MODEL
           -> LLM_FALLBACK_API_KEY / LLM_FALLBACK_MODEL at LLM_FALLBACK_BASE_URL
  gemini:  GEMINI_API_KEY / GEMINI_MODEL
           -> GEMINI_FALLBACK_API_KEY / GEMINI_FALLBACK_MODEL
           -> the chat chain (if GEMINI_FALLBACK_TO_OPENROUTER, default on)

On each endpoint a transient failure (timeout, connection error, 429, 5xx,
empty reply) is retried LLM_RETRIES_PER_ENDPOINT times with a short backoff;
any other failure (bad key, unknown model, 400) moves on at once. When every
endpoint fails, LLMUnavailableError lists what each one returned. Every
attempt is recorded in llm_usage (tokens, latency, cost) and a fallback is
logged as a warning.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, List, Optional, Tuple
from urllib.parse import urlparse

from openai import OpenAI

import llm_usage
import settings
from settings import GEMINI_API_KEY, OPENROUTER_API_KEY

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_PLACEHOLDER_KEYS = {"", "YOUR_OPENROUTER_API_KEY_HERE"}
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class LLMUnavailableError(RuntimeError):
    """Every configured endpoint failed for this request."""


class _EmptyReply(RuntimeError):
    pass


@dataclass(frozen=True)
class Endpoint:
    provider: str              # "openrouter" | "gemini" | host of an OpenAI-compatible API
    role: str                  # "primary" | "fallback"
    model: str
    key: str
    base_url: Optional[str] = None

    @property
    def label(self) -> str:
        return f"{self.role} {self.provider}/{self.model}"


# ─── Configuration (read at call time, so .env changes and tests apply) ──────

def _env(name: str) -> str:
    return os.environ.get(name) or getattr(settings, name, "") or ""


def current_model() -> str:
    """The chat model in use (LLM_MODEL; the evaluation harness overrides it)."""
    return _env("LLM_MODEL")


def current_gemini_model() -> str:
    return _env("GEMINI_MODEL")


def gemini_key() -> str:
    return os.environ.get("GEMINI_API_KEY") or GEMINI_API_KEY


def _provider_name(base_url: str) -> str:
    host = urlparse(base_url).hostname or base_url
    return "openrouter" if host.endswith("openrouter.ai") else host


def chat_endpoints(api_key: Optional[str] = None, model: Optional[str] = None) -> List[Endpoint]:
    """The chat chain for this call: primary, then the fallback if configured."""
    model = model or current_model()
    out: List[Endpoint] = []
    primary = api_key or os.environ.get("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
    if primary not in _PLACEHOLDER_KEYS:
        out.append(Endpoint("openrouter", "primary", model, primary, OPENROUTER_BASE_URL))
    backup = _env("LLM_FALLBACK_API_KEY")
    if backup not in _PLACEHOLDER_KEYS:
        base = _env("LLM_FALLBACK_BASE_URL") or OPENROUTER_BASE_URL
        out.append(Endpoint(_provider_name(base), "fallback", _env("LLM_FALLBACK_MODEL") or model, backup, base))
    return out


def gemini_endpoints(api_key: Optional[str] = None, model: Optional[str] = None) -> List[Endpoint]:
    model = model or current_gemini_model()
    out: List[Endpoint] = []
    primary = api_key or gemini_key()
    if primary:
        out.append(Endpoint("gemini", "primary", model, primary))
    backup = _env("GEMINI_FALLBACK_API_KEY")
    if backup:
        out.append(Endpoint("gemini", "fallback", _env("GEMINI_FALLBACK_MODEL") or model, backup))
    if str(_env("GEMINI_FALLBACK_TO_OPENROUTER")).lower() not in ("0", "false", "no"):
        out += [Endpoint(e.provider, "fallback", e.model, e.key, e.base_url) for e in chat_endpoints()]
    return out


# ─── Clients and single calls ─────────────────────────────────────────────────

@lru_cache(maxsize=16)
def _openai_client(key: str, base_url: str) -> OpenAI:
    # Retries are handled here (per endpoint, then fallback), not by the SDK.
    return OpenAI(api_key=key, base_url=base_url, timeout=settings.LLM_TIMEOUT_SECONDS, max_retries=0)


def get_openrouter_client(api_key: Optional[str] = None) -> OpenAI:
    """Kept for callers that want a raw client; prefer chat_completion()."""
    key = api_key or os.environ.get("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
    if key in _PLACEHOLDER_KEYS:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to a .env file in the project root (see .env.example) "
            "or set it as an environment variable."
        )
    return _openai_client(key, OPENROUTER_BASE_URL)


Usage = Tuple[Optional[int], Optional[int], Optional[float]]   # prompt, completion, cost (USD)


def _call_openai(ep: Endpoint, system: Optional[str], user: str, temperature: float,
                 max_tokens: int) -> Tuple[str, Usage]:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    resp = _openai_client(ep.key, ep.base_url or OPENROUTER_BASE_URL).chat.completions.create(
        model=ep.model, messages=messages, temperature=temperature, max_tokens=max_tokens,
    )
    text = (resp.choices[0].message.content or "") if resp.choices else ""
    u = resp.usage
    cost = None
    if u is not None:
        cost = getattr(u, "cost", None)
        if cost is None and getattr(u, "model_extra", None):
            cost = u.model_extra.get("cost")
    usage = (getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None),
             float(cost) if cost is not None else None)
    return text, usage


def _call_gemini(ep: Endpoint, prompt: str, temperature: float, max_tokens: int) -> Tuple[str, Usage]:
    from google import genai

    client = genai.Client(api_key=ep.key, http_options={"timeout": int(settings.LLM_TIMEOUT_SECONDS * 1000)})
    response = client.models.generate_content(
        model=ep.model,
        contents=prompt,
        config={"temperature": temperature, "max_output_tokens": max_tokens},
    )
    meta = getattr(response, "usage_metadata", None)
    out_tokens = None
    if meta is not None and meta.candidates_token_count is not None:
        out_tokens = meta.candidates_token_count + (getattr(meta, "thoughts_token_count", None) or 0)
    return response.text or "", (getattr(meta, "prompt_token_count", None), out_tokens, None)


# ─── The chain ────────────────────────────────────────────────────────────────

def _status(exc: BaseException) -> Optional[int]:
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (_EmptyReply, TimeoutError, ConnectionError)):
        return True
    name = type(exc).__name__
    if name in ("APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError",
                "ServerError", "ReadTimeout", "ConnectTimeout", "RemoteProtocolError"):
        return True
    return _status(exc) in _RETRYABLE_STATUS


def run_chain(endpoints: List[Endpoint], call: Callable[[Endpoint], Tuple[str, Usage]],
              retries: Optional[int] = None, sleep: Callable[[float], None] = time.sleep) -> str:
    """Try each endpoint in turn; return the first non-empty reply."""
    retries = settings.LLM_RETRIES_PER_ENDPOINT if retries is None else retries
    errors: List[str] = []
    for ep in endpoints:
        for attempt in range(retries + 1):
            started = time.perf_counter()
            try:
                text, (p_tok, c_tok, cost) = call(ep)
                if not text.strip():
                    raise _EmptyReply("empty reply")
            except Exception as exc:  # any provider error: retry or move on
                llm_usage.record(provider=ep.provider, model=ep.model, role=ep.role, ok=False,
                                 latency_s=time.perf_counter() - started, error=f"{type(exc).__name__}: {exc}")
                errors.append(f"{ep.label}: {type(exc).__name__}: {str(exc)[:200]}")
                if attempt < retries and _is_transient(exc):
                    sleep(min(2.0 ** attempt, 8.0))
                    continue
                logger.warning("LLM endpoint failed (%s): %s: %s", ep.label, type(exc).__name__, str(exc)[:200])
                break
            llm_usage.record(provider=ep.provider, model=ep.model, role=ep.role, ok=True,
                             latency_s=time.perf_counter() - started, prompt_tokens=p_tok,
                             completion_tokens=c_tok, cost_usd=cost)
            if ep.role == "fallback":
                logger.warning("LLM reply came from the fallback (%s)", ep.label)
            return text
    raise LLMUnavailableError("Every LLM endpoint failed:\n  " + "\n  ".join(errors))


# ─── Public API ───────────────────────────────────────────────────────────────

def chat_completion(
    *,
    system: str,
    user: str,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 600,
    api_key: Optional[str] = None,
) -> str:
    """Single-turn chat completion (OpenRouter, then the fallback). Returns the text."""
    endpoints = chat_endpoints(api_key, model)
    if not endpoints:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to a .env file in the project root (see .env.example) "
            "or set it as an environment variable."
        )
    return run_chain(endpoints, lambda ep: _call_openai(ep, system, user, temperature, max_tokens))


def gemini_completion(
    prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 8192,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Single-turn Gemini call with fallbacks. Raises ValueError if nothing is configured."""
    endpoints = gemini_endpoints(api_key, model)
    if not endpoints:
        raise ValueError(
            "GEMINI_API_KEY is not set. Add it to a .env file in the project root "
            "(see .env.example) or set it as an environment variable."
        )

    def call(ep: Endpoint) -> Tuple[str, Usage]:
        if ep.provider == "gemini":
            return _call_gemini(ep, prompt, temperature, max_tokens)
        return _call_openai(ep, None, prompt, temperature, min(max_tokens, 4000))

    return run_chain(endpoints, call)
