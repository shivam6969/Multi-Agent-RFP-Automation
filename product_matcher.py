"""
product_matcher.py
------------------
Matches one RFP line item to one catalogue product.

  find_candidates()  catalogue search (embedding + rerank) with relevance scores;
                     candidates below MATCH_MIN_CANDIDATE_SCORE are dropped, so
                     clearly unrelated items become gaps without any LLM call.
  pick_product()     one LLM call chooses a candidate BY ID (or none). An id
                     that is not in the candidate list is rejected.
  atomise()          one batched LLM call splits compound requirements
                     ("16A and 25A switches") into single-product items.
  confidence_band()  maps a relevance score to high / medium / low.

The LLM never supplies prices, SAP codes or product names: those always come
from the catalogue record of the chosen id.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from pydantic import BaseModel

from bu_rag import get_bu_rag_tool
from catalogue import ProductRecord
from llm_json import LLMOutputError, chat_json
from settings import (
    MATCH_HIGH_CONFIDENCE,
    MATCH_MEDIUM_CONFIDENCE,
    MATCH_MIN_CANDIDATE_SCORE,
    MATCH_TOP_K,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    product: ProductRecord
    score: float  # reranker relevance, 0-1


def find_candidates(query: str, top_k: int = MATCH_TOP_K) -> List[Candidate]:
    pairs = get_bu_rag_tool().search_products_scored(query, top_k=top_k)
    return [Candidate(p, s) for p, s in pairs if s >= MATCH_MIN_CANDIDATE_SCORE]


def confidence_band(score: float) -> str:
    if score >= MATCH_HIGH_CONFIDENCE:
        return "high"
    if score >= MATCH_MEDIUM_CONFIDENCE:
        return "medium"
    return "low"


# ─── Pick one candidate ───────────────────────────────────────────────────────

_PICK_PROMPT = """
You are matching one RFP line item to a product in our catalogue.

RFP item: "{requirement}"

Candidate catalogue products:
{candidates}

Pick the single candidate that best matches the RFP item (rating, way/pole,
module size, product type, colour/range if stated). If none of them is the
same kind of product, answer null. If several candidates are equally good,
pick the first of them (the list is ordered best first).

Return ONLY a JSON object: {{"product_id": "<id from the list>" | null, "reason": "<one short sentence>"}}
""".strip()


class _Pick(BaseModel):
    product_id: Optional[str] = None
    reason: str = ""


def pick_product(
    requirement: str,
    candidates: List[Candidate],
    api_key: Optional[str] = None,
) -> Tuple[Optional[Candidate], str]:
    """Return (chosen candidate or None, short reason)."""
    if not candidates:
        return None, "No similar product in the catalogue."

    listing = "\n".join(
        f'- id={c.product.id} | {c.product.item_name} | range: {c.product.range_name}'
        f' | variant: {c.product.variant or "-"}'
        for c in candidates
    )
    try:
        pick = chat_json(
            system="You select catalogue products by id. Return JSON only.",
            user=_PICK_PROMPT.format(requirement=requirement, candidates=listing),
            schema=_Pick,
            temperature=0.0,
            max_tokens=150,
            api_key=api_key,
        )
    except LLMOutputError:
        return None, "The model did not return a valid product choice."

    by_id = {c.product.id: c for c in candidates}
    chosen = by_id.get((pick.product_id or "").strip())
    if pick.product_id and chosen is None:
        logger.warning("Model picked an id that was not offered: %r", pick.product_id)
        return None, "The model chose a product that was not in the candidate list."
    if chosen is None:
        return None, pick.reason or "No candidate is the same kind of product."
    return chosen, pick.reason


# ─── Split compound requirements ──────────────────────────────────────────────

_ATOMISE_PROMPT = """
You are a procurement data analyst. Split compound requirements that mention
multiple distinct products into individual items — one product per item.

Rules:
- "X and Y" / "X or Y" where X and Y are different specs → two items
- Enumerated variants (e.g. "1-way, 2-way, bell push") → one item each
- Listed colours or finishes (e.g. "combined plates (White, Grey, Silver)") →
  one item per colour, keeping the product name ("White combined plate", …)
- A list of module SIZES given as a range (e.g. "outer plates 1M to 18M") →
  one item per standard size (1M, 2M, 3M, 4M, 6M, 8M, 12M, 18M)
- A rating range or slash rating that is part of ONE product's name stays one
  item: "Combined sockets (6-16A)" is a single 6-16A combined socket, and
  "6A/10A Bell push Switch" is a single 6A/10A switch, not two
- Already-atomic requirements → one item, unchanged
- Never merge, summarise or invent requirements

Return ONLY a JSON array with one entry per input, in the same order:
  [{{"index": <input index>, "items": ["<item text>", ...]}}, ...]

Inputs:
{inputs}
""".strip()


class _AtomGroup(BaseModel):
    index: int
    items: List[str]


def atomise(requirements: List[str], api_key: Optional[str] = None) -> List[List[str]]:
    """
    For each requirement, return the list of single-product items it contains.
    On any failure, each requirement is kept as one item.
    """
    if not requirements:
        return []
    inputs = json.dumps([{"index": i, "requirement": r} for i, r in enumerate(requirements)], indent=1)
    try:
        groups = chat_json(
            system="You split compound procurement requirements into atomic items. Return JSON only.",
            user=_ATOMISE_PROMPT.format(inputs=inputs),
            schema=List[_AtomGroup],
            temperature=0.0,
            max_tokens=3000,
            api_key=api_key,
        )
    except LLMOutputError:
        logger.warning("Atomise failed; keeping requirements as they are.")
        return [[r] for r in requirements]

    by_index = {g.index: [s.strip() for s in g.items if s.strip()] for g in groups}
    return [by_index.get(i) or [r] for i, r in enumerate(requirements)]
