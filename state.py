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
    rfp_pdf_path: str                      # RFP to analyse; default settings.RFP_PDF_PATH
    project_id: str                        # review workflow: one LangGraph thread per project

    # ── Discovery Agent (discovery/discovery_agent.py) ────────────────────────
    discovery_source: str                  # "sample" (default) | "gem" (off unless enabled)
    discovery_query: Dict[str, Any]        # DiscoveryQuery: keywords, count, location, dates, budget
    discovered_rfps: List[Dict[str, Any]]  # RfpCandidate dicts (PDF saved locally)

    # ── Human review (workflow.py) ────────────────────────────────────────────
    review_log: List[Dict[str, Any]]       # [{stage, at, reviewer, note, changes}]

    # ── RFP RAG outputs ───────────────────────────────────────────────────────
    rfp_answer: str                        # direct answer from RFP RAG
    rfp_requirements: List[Dict[str, Any]] # [{id, requirement, criticality, weight, quantity?,
                                           #   unit?, origin: "table"|"text",
                                           #   source?: {page, table, row, section} | {pages},
                                           #   attributes?, quantity_breakdown?}]
    rfp_tables: List[Dict[str, Any]]       # tables found in the RFP: [{id, kind (boq|spec|kv|other),
                                           #   pages, section, caption, columns, roles, rows, detection}]

    # ── Business-Unit RAG outputs ─────────────────────────────────────────────
    bu_answer: str                         # direct answer from BU RAG

    # ── Matching Agent outputs ────────────────────────────────────────────────
    matched_items: List[Dict[str, Any]]    # [{id, requirement, type, status, matched, product_id,
                                           #   sap_code, match_score, confidence, alternatives, …}]
    fulfillment_score: float               # 0–100, weighted share of product items with a product
    fulfillment_report: str                # human-readable match summary
    match_summary: Dict[str, Any]          # counts: matched / needs_review / gaps / compliance_pending

    # ── Pricing inputs (set by the user; all optional) ────────────────────────
    margin_pct: float                      # chosen margin; default PricingRules.base_margin_pct
    advance_payment: bool                  # apply PricingRules.payment_terms_discount_pct
    freight_inr: float                     # flat freight; default PricingRules.freight_flat_inr

    # ── Pricing Agent outputs ─────────────────────────────────────────────────
    pricing_report: str                    # full pricing breakdown
    quoted_items: List[Dict[str, Any]]     # [{product, unit_price, qty, net, gst, total}]
    line_items: List[Dict[str, Any]]       # frontend-ready rows used for repricing/export
    pricing_summary: Dict[str, Any]        # totals + applied margin metadata
    quote_payload: Dict[str, Any]          # contract passed to quotation/pdf generator

    # ── Proposal (report agent / proposal review) ────────────────────────────
    rfp_meta: Dict[str, Any]               # title, buyer, reference, deadline, location (from the RFP record)
    proposal_sections: List[Dict[str, Any]]  # [{key, title, body, ai_generated, edited}]
    proposal_meta: Dict[str, Any]          # reference, generated, valid_until, buyer, profile snapshot

    # ── Report Agent outputs ──────────────────────────────────────────────────
    report_pdf_path: str                   # absolute path to the generated PDF
    report_text: str                       # raw LLM-generated proposal text

    # ── Master Agent outputs ──────────────────────────────────────────────────
    final_response: str                    # synthesised answer returned to user

    # ── Control / routing ─────────────────────────────────────────────────────
    route: str                             # "rfp" | "bu" | "match" | "price" | "full" | "report"
    error: Optional[str]                   # populated if any node fails
    warnings: List[str]                    # non-fatal problems to show the user

    