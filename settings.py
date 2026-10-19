"""
config/settings.py
------------------
Central configuration for all agents and tools.
Edit this file to change paths or models.

API keys are never stored here. They are read from environment variables,
or from a local `.env` file in the project root (copy `.env.example`).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv not installed yet; plain env vars still work
    load_dotenv = None


# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CATALOGUE_DIR: Path = DATA_DIR / "catalogue"
SAMPLES_DIR: Path = DATA_DIR / "samples"

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


# ─── LLM / API ────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
LLM_MODEL: str = os.environ.get("LLM_MODEL", "meta-llama/llama-4-scout")

GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")


# ─── Embedding / Reranker models ───────────────────────────────────────────────

BU_EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"
BU_RERANK_MODEL: str = "BAAI/bge-reranker-base"
RFP_EMBED_MODEL: str = "BAAI/bge-small-en"


# ─── Business-Unit RAG data files ─────────────────────────────────────────────

BU_SMALL_CHUNKS_PATH: str = str(CATALOGUE_DIR / "havells_small_chunks.txt")
BU_LARGE_CHUNKS_PATH: str = str(CATALOGUE_DIR / "havells_large_chunks.txt")
BU_JSON_PATH: str = str(CATALOGUE_DIR / "havells.json")


# ─── RFP RAG data files ────────────────────────────────────────────────────────

RFP_PDF_PATH: str = os.environ.get("RFP_PDF_PATH", str(SAMPLES_DIR / "sample_rfp.pdf"))
RFP_MAX_CHUNK_LENGTH: int = 800


# ─── Retrieval knobs ──────────────────────────────────────────────────────────

BU_TOP_K_SMALL: int = 5
BU_TOP_K_LARGE: int = 3
BU_TOP_K_JSON: int = 5
BU_RERANK_TOP_K: int = 5

RFP_TOP_K: int = 3


# ─── Cache ────────────────────────────────────────────────────────────────────

FULFILLMENT_CACHE_PATH: str = str(PROJECT_ROOT / ".rfp_fulfillment_cache.json")


# ─── Pricing defaults ─────────────────────────────────────────────────────────

@dataclass
class PricingRules:
    """Adjust margin and discount bands here without touching agent code."""
    base_margin_pct: float = 20.0          # default gross margin %
    min_margin_pct: float = 0.0            # user cannot set below this
    max_margin_pct: float = 60.0           # user cannot set above this
    volume_discount_tiers: list = field(default_factory=lambda: [
        {"min_qty": 1,    "max_qty": 199,    "discount_pct": 0.0},
        {"min_qty": 200,  "max_qty": 499,   "discount_pct": 5.0},
        {"min_qty": 500,  "max_qty": 999,   "discount_pct": 8.0},
        {"min_qty": 1000, "max_qty": 99999, "discount_pct": 12.0},
    ])
    gst_rate_pct: float = 18.0             # GST applied on top of net price
    currency: str = "INR"
    rounding_digits: int = 2
    freight_flat_inr: float = 0.0          # 0 = buyer pays, >0 = included
    payment_terms_discount_pct: float = 2.0  # extra discount for advance payment


PRICING_RULES = PricingRules()
