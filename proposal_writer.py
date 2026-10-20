"""
proposal_writer.py
------------------
Drafts the proposal text, section by section (the AI part of the report).

  draft_proposal(state, profile)            all sections + proposal_meta
  regenerate_section(state, key, hint)      one section, e.g. "make it shorter"
  amount_warnings(sections)                 sections that mention money

Rules the model is given, and that the code enforces where it can:
  - It writes words only. Every price, total and quantity in the PDF comes
    from the approved pricing (proposal_pdf.py), never from model text.
    amount_warnings() flags any section that still mentions a ₹ amount.
  - It may only claim compliance items the reviewer confirmed as met.
  - "Company Information" and the pricing intro are built from the company
    profile and pricing rules without the LLM.

Provider: Gemini when GEMINI_API_KEY is set (long output), else OpenRouter.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, field_validator

from company_profile import CompanyProfile
from llm import gemini_key
from llm_json import chat_json
from settings import PRICING_RULES, PROPOSAL_VALIDITY_DAYS
from state import AgentState

# (key, title, written by the LLM?, what the section should cover)
SECTIONS: List[Tuple[str, str, bool, str]] = [
    ("company", "Company Information", False, ""),
    ("executive_summary", "Executive Summary", True,
     "2-3 short paragraphs: why we are a good fit for this buyer and project, how much of the "
     "requested scope we cover, and our delivery commitment."),
    ("requirements", "Understanding of Requirements", True,
     "Restate the buyer's scope and key conditions in our own words, as short bullets."),
    ("products", "Product Offering", True,
     "Describe the offered products grouped by type and range (no SAP codes, no prices). "
     "Say plainly which requested items are not quoted and that we can source them on request."),
    ("pricing", "Pricing Schedule", False, ""),
    ("delivery", "Delivery & Logistics", True,
     "Delivery timeline, packing, transport and damage replacement, matching the RFP's terms."),
    ("warranty", "Warranty & After-Sales Support", True,
     "Manufacturer warranty, replacement of defective items, and a named support contact."),
    ("compliance", "Eligibility & Compliance", True,
     "Address each eligibility/compliance item. Claim only the items listed as CONFIRMED; for "
     "the others say the supporting documents will be submitted separately."),
    ("terms", "Terms & Conditions", True,
     "Proposal validity, payment terms, GST as applicable, delivery terms and the order of "
     "precedence of documents, as short bullets."),
]
SECTION_TITLES = {key: title for key, title, _, _ in SECTIONS}
AI_SECTIONS = [key for key, _, ai, _ in SECTIONS if ai]

# Money in model text: "₹ 1,250", "Rs. 60", "INR 5,00,000", "12 lakh", "1.2 crore".
_AMOUNT_RE = re.compile(r"(₹|\bRs\.?|\bINR)\s*\d[\d,]*(\.\d+)?|\b\d+(\.\d+)?\s*(lakh|lac|crore|cr)\b", re.I)

_RULES = """
Rules:
- Write in a formal, confident business tone, in English. Plain text only.
- Use "- " at the start of a line for bullet points. No markdown headings, no bold.
- NEVER write prices, totals, discounts, margins or any money amount. The pricing
  table is inserted automatically from approved figures.
- Do not invent certifications, awards, client names or numbers not given below.
- Refer to us as "{company}" and to the buyer as "{buyer}".
""".strip()


# ─── Context given to the model ───────────────────────────────────────────────

def _context(state: AgentState, profile: CompanyProfile) -> str:
    meta = state.get("rfp_meta") or {}
    items = state.get("matched_items", [])
    lines = state.get("line_items", [])
    quoted = [f"- {li['product_name']} ({li.get('product_range', '')}), quantity {li['qty']}" for li in lines]
    not_quoted = [f"- {m['requirement']}" for m in items
                  if m.get("type") == "product" and not m.get("matched")]
    compliance = {"met": [], "other": []}
    for m in items:
        if m.get("type") == "compliance":
            compliance["met" if m.get("status") == "met" else "other"].append(f"- {m['requirement']}")
    summary = state.get("pricing_summary") or {}
    terms = []
    if summary.get("advance_payment"):
        terms.append("- An advance-payment discount is offered (shown in the pricing table).")
    terms.append("- Freight is included in the price." if summary.get("freight") else
                 "- Freight is payable by the buyer unless agreed otherwise.")

    return "\n".join([
        f"RFP: {meta.get('title') or 'Untitled RFP'}",
        f"Buyer: {meta.get('buyer') or 'the buyer'}" + (f", {meta['location']}" if meta.get("location") else ""),
        f"RFP reference: {meta.get('reference') or 'not stated'}; submission deadline: {meta.get('deadline') or 'not stated'}",
        "",
        "RFP summary (from the document):",
        (state.get("rfp_answer") or "Not available.")[:2500],
        "",
        f"Products we are quoting ({len(quoted)}):", *(quoted or ["- none"]),
        "",
        "Requested items we are NOT quoting:", *(not_quoted or ["- none"]),
        "",
        "Compliance items CONFIRMED as met by our reviewer:", *(compliance["met"] or ["- none"]),
        "Other compliance items (not yet confirmed):", *(compliance["other"] or ["- none"]),
        "",
        "Commercial terms:", *terms,
        f"- Proposal validity: {PROPOSAL_VALIDITY_DAYS} days. GST at {PRICING_RULES.gst_rate_pct:g}% is charged extra as shown.",
        "",
        f"About us ({profile.name}): {profile.about}",
        f"Brands supplied: {profile.brands}. Years in business: {profile.years_in_business or 'n/a'}.",
        f"Support contact: {profile.signatory_name}, {profile.signatory_title}, {profile.email}, {profile.phone}",
    ])


# ─── Deterministic sections ───────────────────────────────────────────────────

def company_section(profile: CompanyProfile) -> str:
    lines = [profile.tagline, "", profile.about, ""]
    if profile.brands:
        lines.append(f"- Brands supplied: {profile.brands}")
    if profile.years_in_business:
        lines.append(f"- Years in business: {profile.years_in_business}")
    lines += [f"- Address: {profile.address}",
              f"- GSTIN: {profile.gstin or 'to be provided with the bid documents'}",
              f"- Contact: {profile.email} · {profile.phone}"]
    if profile.website:
        lines.append(f"- Website: {profile.website}")
    return "\n".join(lines).strip()


def pricing_intro(state: AgentState) -> str:
    summary = state.get("pricing_summary") or {}
    text = (f"Prices below are per unit in Indian Rupees. GST at {PRICING_RULES.gst_rate_pct:g}% is shown "
            "separately. Quantities are as stated in the RFP or as agreed during our review. Prices are valid "
            f"for {PROPOSAL_VALIDITY_DAYS} days from the date of this proposal.")
    if summary.get("advance_payment"):
        text += " The advance-payment discount shown applies if payment is made with the order."
    return text


# ─── LLM drafting ─────────────────────────────────────────────────────────────

class _Draft(BaseModel):
    sections: Dict[str, str]

    @field_validator("sections")
    @classmethod
    def _all_sections_written(cls, value: Dict[str, str]) -> Dict[str, str]:
        missing = [k for k in AI_SECTIONS if not str(value.get(k, "")).strip()]
        if missing:
            raise ValueError(f"missing sections: {missing}")
        return {k: str(value[k]).strip() for k in AI_SECTIONS}


class _OneSection(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        if len(value.strip()) < 20:
            raise ValueError("section text is empty or too short")
        return value.strip()


def _provider() -> Dict[str, Any]:
    if gemini_key():
        return {"provider": "gemini", "max_tokens": 8192, "temperature": 0.3}
    return {"provider": "openrouter", "max_tokens": 3500, "temperature": 0.3}


def _names(state: AgentState, profile: CompanyProfile) -> Dict[str, str]:
    return {"company": profile.name, "buyer": (state.get("rfp_meta") or {}).get("buyer") or "the buyer"}


def draft_proposal(state: AgentState, profile: CompanyProfile, project_ref: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Write every section. Returns (sections, proposal_meta)."""
    guide = "\n".join(f'  "{k}": {SECTION_TITLES[k]}: {g}' for k, _, ai, g in SECTIONS if ai)
    prompt = (
        "You are writing a vendor's proposal in response to an RFP.\n\n"
        f"{_context(state, profile)}\n\n{_RULES.format(**_names(state, profile))}\n\n"
        "Return a JSON object: {\"sections\": {<key>: <text>, ...}} with these keys:\n"
        f"{guide}"
    )
    draft = chat_json(system="You write vendor proposals.", user=prompt, schema=_Draft, **_provider())

    sections = []
    for key, title, ai, _ in SECTIONS:
        body = (draft.sections[key] if ai
                else company_section(profile) if key == "company"
                else pricing_intro(state))
        sections.append({"key": key, "title": title, "body": body, "ai_generated": ai, "edited": False})

    meta = state.get("rfp_meta") or {}
    today = date.today()
    proposal_meta = {
        "reference": project_ref,
        "generated": today.isoformat(),
        "valid_until": (today + timedelta(days=PROPOSAL_VALIDITY_DAYS)).isoformat(),
        "buyer": meta.get("buyer"),
        "rfp_title": meta.get("title"),
        "rfp_reference": meta.get("reference"),
        "profile": profile.model_dump(),
    }
    return sections, proposal_meta


def regenerate_section(state: AgentState, key: str, instruction: str = "",
                       profile: Optional[CompanyProfile] = None) -> str:
    """Rewrite one AI section (optionally following the reviewer's instruction)."""
    if key not in AI_SECTIONS:
        raise ValueError(f"'{key}' is not written by the AI and cannot be regenerated")
    profile = profile or CompanyProfile.model_validate((state.get("proposal_meta") or {}).get("profile") or {})
    current = next((s["body"] for s in state.get("proposal_sections", []) if s["key"] == key), "")
    guidance = next(g for k, _, _, g in SECTIONS if k == key)
    prompt = (
        "You are revising one section of a vendor's proposal in response to an RFP.\n\n"
        f"{_context(state, profile)}\n\n{_RULES.format(**_names(state, profile))}\n\n"
        f"Section: {SECTION_TITLES[key]}. It should cover: {guidance}\n"
        f"Current text:\n{current or '(empty)'}\n\n"
        f"Reviewer's instruction: {instruction.strip() or 'Improve clarity and accuracy.'}\n\n"
        'Return a JSON object: {"text": "<the new section text>"}'
    )
    return chat_json(system="You write vendor proposals.", user=prompt, schema=_OneSection, **_provider()).text


def amount_warnings(sections: List[Dict[str, Any]]) -> Dict[str, str]:
    """{section key: message} for sections whose text mentions a money amount."""
    out = {}
    for s in sections:
        m = _AMOUNT_RE.search(s.get("body", ""))
        if m:
            out[s["key"]] = (f'Mentions "{m.group(0)}". Prices come from the pricing table; '
                             "remove amounts from the text or check they match.")
    return out
