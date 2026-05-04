"""
config/settings.py
------------------
Central configuration for all agents and tools.
Edit this file to change paths, models, or API keys.
"""

from dataclasses import dataclass, field


# ─── LLM / API ────────────────────────────────────────────────────────────────

GROQ_API_KEY: str = "YOUR_GROQ_API_KEY_HERE"   # or set env var GROQ_API_KEY
LLM_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"

GEMINI_API_KEY: str = ""  # or set env var GEMINI_API_KEY
GEMINI_MODEL: str = "gemini-3.1-flash-lite-preview"


# ─── Embedding / Reranker models ───────────────────────────────────────────────

BU_EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"
BU_RERANK_MODEL: str = "BAAI/bge-reranker-base"
RFP_EMBED_MODEL: str = "BAAI/bge-small-en"


# ─── Business-Unit RAG data files ─────────────────────────────────────────────

BU_SMALL_CHUNKS_PATH: str = "havells_small_chunks (1).txt"
BU_LARGE_CHUNKS_PATH: str = "havells_large_chunks.txt"
BU_JSON_PATH: str = "havelsdata.json"


# ─── RFP RAG data files ────────────────────────────────────────────────────────

RFP_PDF_PATH: str = "Untitled document.pdf"
RFP_MAX_CHUNK_LENGTH: int = 800


# ─── Retrieval knobs ──────────────────────────────────────────────────────────

BU_TOP_K_SMALL: int = 5
BU_TOP_K_LARGE: int = 3
BU_TOP_K_JSON: int = 5
BU_RERANK_TOP_K: int = 5

RFP_TOP_K: int = 3


# ─── Cache ────────────────────────────────────────────────────────────────────

FULFILLMENT_CACHE_PATH: str = ".rfp_fulfillment_cache.json"


# ─── Pricing defaults ─────────────────────────────────────────────────────────

@dataclass
class PricingRules:
    """Adjust margin and discount bands here without touching agent code."""
    base_margin_pct: float = 20.0          # default gross margin %
    min_margin_pct: float = 0.0            # user cannot set below this
    max_margin_pct: float = 60.0           # user cannot set above this
    volume_discount_tiers: list = field(default_factory=lambda: [
        {"min_qty": 1,    "max_qty": 99,    "discount_pct": 0.0},
        {"min_qty": 100,  "max_qty": 499,   "discount_pct": 5.0},
        {"min_qty": 500,  "max_qty": 999,   "discount_pct": 8.0},
        {"min_qty": 1000, "max_qty": 99999, "discount_pct": 12.0},
    ])
    gst_rate_pct: float = 18.0             # GST applied on top of net price
    currency: str = "INR"
    rounding_digits: int = 2
    freight_flat_inr: float = 0.0          # 0 = buyer pays, >0 = included
    payment_terms_discount_pct: float = 2.0  # extra discount for advance payment


PRICING_RULES = PricingRules()
