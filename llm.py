"""
utils/llm.py
------------
Thin wrapper around the OpenRouter API (OpenAI-compatible) so every agent
uses the same initialisation logic and we only import openai in one place.
"""

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI

from settings import OPENROUTER_API_KEY, LLM_MODEL

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_openrouter_client(api_key: Optional[str] = None) -> OpenAI:
    key = api_key or os.environ.get("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
    if not key or key == "YOUR_OPENROUTER_API_KEY_HERE":
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Either set the env var or edit settings.py."
        )
    return OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)


def chat_completion(
    *,
    system: str,
    user: str,
    model: str = LLM_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 600,
    api_key: Optional[str] = None,
) -> str:
    """Single-turn chat completion via OpenRouter. Returns the assistant text.

    Retries once automatically if the model returns an empty response,
    which can happen transiently with OpenRouter.
    """
    client = get_openrouter_client(api_key)

    def _call() -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    result = _call()
    if not result.strip():
        # Retry once on empty response (transient OpenRouter behaviour)
        result = _call()
    return result
