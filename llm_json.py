"""
llm_json.py
-----------
Structured (JSON) LLM calls with schema validation.

chat_json() asks the model for JSON, pulls the JSON out of the reply (bare,
inside ```json fences, or surrounded by chatter), validates it against a
pydantic schema and, if anything is wrong, retries once with the validation
error appended to the prompt. If it still fails it raises LLMOutputError, so
callers decide on a safe fallback instead of silently using bad data.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Type, TypeVar

from pydantic import TypeAdapter, ValidationError

from llm import chat_completion

logger = logging.getLogger(__name__)

T = TypeVar("T")

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# Appended to every system prompt. Without it, Llama 4 Scout often "thinks
# out loud" in Markdown for ambiguous inputs and runs out of tokens before the
# JSON (measured on the sample RFP: 28 of 40 product picks). With it: 6/6
# valid on those same items, ~3x faster. OpenRouter's JSON mode is not
# available for this model (404), so the instruction does the work.
JSON_ONLY = "Reply with exactly one JSON value and nothing else: no explanation, no steps, no markdown."


class LLMOutputError(ValueError):
    """The model did not return valid JSON in the requested shape."""


def extract_json(raw: str) -> Any:
    """Parse the JSON value in an LLM reply; raise ValueError if there is none."""
    text = (raw or "").strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    starts = [i for i in (text.find("["), text.find("{")) if i >= 0]
    if not starts:
        raise ValueError("no JSON found in reply")
    start = min(starts)
    end = text.rfind("]" if text[start] == "[" else "}")
    if end <= start:
        raise ValueError("unterminated JSON in reply")
    return json.loads(text[start:end + 1])


def chat_json(
    *,
    system: str,
    user: str,
    schema: Type[T] | Any,
    retries: int = 1,
    **llm_kwargs: Any,
) -> T:
    """
    Call the LLM and return its reply validated as *schema* (a pydantic model
    or any type pydantic understands, e.g. List[MyModel]).
    """
    adapter = TypeAdapter(schema)
    system = f"{system.rstrip()} {JSON_ONLY}"
    prompt = user
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        raw = chat_completion(system=system, user=prompt, **llm_kwargs)
        try:
            return adapter.validate_python(extract_json(raw))
        except (ValueError, ValidationError) as exc:  # includes JSONDecodeError
            last_error = exc
            logger.warning("LLM JSON attempt %d failed: %s | reply starts: %r",
                           attempt + 1, str(exc)[:200], (raw or "")[:120])
            prompt = (
                f"{user}\n\nYour previous reply could not be used: {str(exc)[:300]}\n"
                "Reply again with ONLY valid JSON in exactly the requested format."
            )

    raise LLMOutputError(f"Invalid JSON from model after {retries + 1} attempts: {last_error}")
