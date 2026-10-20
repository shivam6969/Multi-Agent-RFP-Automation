"""
RFP metadata (buyer, deadline, location, category, budget) for uploaded PDFs.

GeM bid / reverse-auction documents have fixed labels, so they are parsed
with rules (gem.parse_gem_text). Anything else gets one schema-validated LLM
call over the first pages. Failures return empty metadata: the upload still
works, the UI just shows "—".
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any, Dict, Optional

from pydantic import BaseModel, field_validator

from llm_json import LLMOutputError, chat_json

from .gem import looks_like_gem, parse_gem_text

logger = logging.getLogger(__name__)

_PROMPT = """
Read the start of this RFP / tender document and return its key facts as a
JSON object with exactly these keys (use null when the document does not say):

  "title"           : short title of what is being procured (not "Request for Proposal")
  "buyer"           : the issuing organisation's name
  "deadline"        : last date for submission, as YYYY-MM-DD
  "location"        : city / place of delivery or of the buyer
  "category"        : 2-4 word category, e.g. "Modular switches"
  "estimated_value" : estimated budget in INR as a plain number, if stated

Document:
{text}
""".strip()


class RfpMetadata(BaseModel):
    title: Optional[str] = None
    buyer: Optional[str] = None
    deadline: Optional[str] = None
    location: Optional[str] = None
    category: Optional[str] = None
    estimated_value: Optional[float] = None

    @field_validator("title", "buyer", "location", "category", mode="before")
    @classmethod
    def _text(cls, value: Any) -> Optional[str]:
        value = re.sub(r"\s+", " ", str(value or "")).strip()
        if not value or value.lower() in {"null", "none", "n/a", "not mentioned", "not stated"}:
            return None
        return value[:200]

    @field_validator("deadline", mode="before")
    @classmethod
    def _iso_date(cls, value: Any) -> Optional[str]:
        try:
            return date.fromisoformat(str(value).strip()[:10]).isoformat()
        except (TypeError, ValueError):
            return None

    @field_validator("estimated_value", mode="before")
    @classmethod
    def _amount(cls, value: Any) -> Optional[float]:
        try:
            amount = float(str(value).replace(",", "").replace("₹", "").strip())
        except (TypeError, ValueError):
            return None
        return amount if amount > 0 else None


def extract_metadata(text: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Best-effort metadata for an RFP's text. Never raises."""
    if looks_like_gem(text):
        return parse_gem_text(text)
    try:
        meta = chat_json(
            system="You extract key facts from procurement documents. Return JSON only.",
            user=_PROMPT.format(text=text[:6000]),
            schema=RfpMetadata,
            temperature=0.0,
            max_tokens=300,
            api_key=api_key,
        )
    except (LLMOutputError, ValueError) as exc:   # ValueError: no API key configured
        logger.warning("RFP metadata extraction skipped: %s", exc)
        return {}
    return meta.model_dump(exclude_none=True)
