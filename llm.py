"""
utils/llm.py
------------
Thin wrapper around the Groq client so every agent uses the same
initialisation logic and we only import groq in one place.
"""

from __future__ import annotations

import os
from typing import List, Optional

from groq import Groq

from settings import GROQ_API_KEY, LLM_MODEL


def get_groq_client(api_key: Optional[str] = None) -> Groq:
    key = api_key or os.environ.get("GROQ_API_KEY") or GROQ_API_KEY
    if not key or key == "YOUR_GROQ_API_KEY_HERE":
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Either set the env var or edit config/settings.py."
        )
    return Groq(api_key=key)


def chat_completion(
    *,
    system: str,
    user: str,
    model: str = LLM_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 600,
    api_key: Optional[str] = None,
) -> str:
    """Single-turn chat completion. Returns the assistant text."""
    client = get_groq_client(api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_completion_tokens=max_tokens,
        stream=False,
    )
    return resp.choices[0].message.content or ""
