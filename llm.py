"""
utils/llm.py
------------
Thin wrappers around the two LLM providers so every agent uses the same
initialisation logic:

  chat_completion()    OpenRouter (OpenAI-compatible), used by all agents
  gemini_completion()  Google Gemini, used to write long proposal text
"""

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI

from settings import GEMINI_API_KEY, GEMINI_MODEL, OPENROUTER_API_KEY, LLM_MODEL


def gemini_key() -> str:
    return os.environ.get("GEMINI_API_KEY") or GEMINI_API_KEY


def gemini_completion(
    prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 8192,
    model: str = GEMINI_MODEL,
    api_key: Optional[str] = None,
) -> str:
    """Single-turn Gemini call. Raises ValueError if no key is configured."""
    key = api_key or gemini_key()
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Add it to a .env file in the project root "
            "(see .env.example) or set it as an environment variable."
        )
    from google import genai

    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"temperature": temperature, "max_output_tokens": max_tokens},
    )
    return response.text or ""

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_openrouter_client(api_key: Optional[str] = None) -> OpenAI:
    key = api_key or os.environ.get("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
    if not key or key == "YOUR_OPENROUTER_API_KEY_HERE":
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to a .env file in the project root (see .env.example) "
            "or set it as an environment variable."
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
