"""
utils/state.py
--------------
The single shared state TypedDict that flows through every node in the
LangGraph graph.  All agents read from and write to this object.

Design rule: every field is Optional so that early nodes don't need to
pre-populate data that is produced later in the pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────────
    user_query: str                        # raw question / instruction from user

    # ── RFP RAG outputs ───────────────────────────────────────────────────────
    rfp_answer: str                        # direct answer from RFP RAG
    rfp_requirements: List[Dict[str, Any]] # structured list of {requirement, weight}

    # ── Business-Unit RAG outputs ─────────────────────────────────────────────
    bu_answer: str                         # direct answer from BU RAG

    # ── Matching Agent outputs ────────────────────────────────────────────────
    matched_items: List[Dict[str, Any]]    # [{product, evidence, matched: bool}]
    fulfillment_score: float               # 0–100
    fulfillment_report: str                # human-readable match summary

    # ── Pricing Agent outputs ─────────────────────────────────────────────────
    pricing_report: str                    # full pricing breakdown
    quoted_items: List[Dict[str, Any]]     # [{product, unit_price, qty, net, gst, total}]

    # ── Master Agent outputs ──────────────────────────────────────────────────
    final_response: str                    # synthesised answer returned to user

    # ── Control / routing ─────────────────────────────────────────────────────
    route: str                             # "rfp" | "bu" | "match" | "price" | "full"
    error: Optional[str]                   # populated if any node fails
