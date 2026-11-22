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
VAR_DIR: Path = PROJECT_ROOT / "var"   # gitignored runtime data

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


# ─── LLM / API ────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
LLM_MODEL: str = os.environ.get("LLM_MODEL", "meta-llama/llama-4-scout")

GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")

# ─── LLM fallback (all optional; see llm.py) ──────────────────────────────────
# When a call fails (outage, rate limit, bad key, timeout) llm.py moves down a
# chain: primary -> backup. The backup can be any OpenAI-compatible API
# (OpenRouter with another key, Groq, Together, OpenAI …).
LLM_FALLBACK_API_KEY: str = os.environ.get("LLM_FALLBACK_API_KEY", "")
LLM_FALLBACK_MODEL: str = os.environ.get("LLM_FALLBACK_MODEL", "")          # "" = same model as primary
LLM_FALLBACK_BASE_URL: str = os.environ.get("LLM_FALLBACK_BASE_URL", "https://openrouter.ai/api/v1")
# Proposal text: Gemini -> backup Gemini key/model -> the OpenRouter chain above.
GEMINI_FALLBACK_API_KEY: str = os.environ.get("GEMINI_FALLBACK_API_KEY", "")
GEMINI_FALLBACK_MODEL: str = os.environ.get("GEMINI_FALLBACK_MODEL", "")    # "" = same model as primary
GEMINI_FALLBACK_TO_OPENROUTER: bool = os.environ.get("GEMINI_FALLBACK_TO_OPENROUTER", "true").lower() not in ("0", "false", "no")
# Per request; a transient error (timeout, 429, 5xx, empty reply) is retried
# this many times on the same endpoint before moving to the next one.
LLM_TIMEOUT_SECONDS: float = float(os.environ.get("LLM_TIMEOUT_SECONDS", "90"))
LLM_RETRIES_PER_ENDPOINT: int = int(os.environ.get("LLM_RETRIES_PER_ENDPOINT", "1"))

# USD per 1M tokens (input, output), for cost estimates when the provider does
# not report the cost itself (OpenRouter does; Gemini does not). Checked
# Nov 2026 from public price pages; override with MODEL_PRICES_JSON in .env,
# e.g. {"my/model": [0.2, 0.6]}.
MODEL_PRICES_USD_PER_MTOK: dict = {
    "meta-llama/llama-4-scout": (0.10, 0.30),
    "gemini-3.1-flash-lite-preview": (0.25, 1.50),
}
if os.environ.get("MODEL_PRICES_JSON"):
    import json as _json
    MODEL_PRICES_USD_PER_MTOK.update({k: tuple(v) for k, v in _json.loads(os.environ["MODEL_PRICES_JSON"]).items()})


# ─── Embedding / Reranker models ───────────────────────────────────────────────

BU_EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"
BU_RERANK_MODEL: str = "BAAI/bge-reranker-base"
# Same model as the catalogue index, so one instance is shared (models.py).
RFP_EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"
RFP_RERANK_MODEL: str = BU_RERANK_MODEL
# BGE's instruction for short search queries against longer passages.
RFP_QUERY_INSTRUCTION: str = "Represent this sentence for searching relevant passages: "


# ─── Business-Unit RAG data files ─────────────────────────────────────────────

BU_SMALL_CHUNKS_PATH: str = str(CATALOGUE_DIR / "havells_small_chunks.txt")
BU_LARGE_CHUNKS_PATH: str = str(CATALOGUE_DIR / "havells_large_chunks.txt")
BU_JSON_PATH: str = str(CATALOGUE_DIR / "havells.json")


# ─── RFP RAG data files ────────────────────────────────────────────────────────

RFP_PDF_PATH: str = os.environ.get("RFP_PDF_PATH", str(SAMPLES_DIR / "sample_rfp.pdf"))
# Upper bound for one text chunk / one table slice in the RFP index (characters).
RFP_MAX_CHUNK_LENGTH: int = 1000


# ─── Retrieval knobs ──────────────────────────────────────────────────────────

BU_TOP_K_SMALL: int = 5
BU_TOP_K_LARGE: int = 3
BU_TOP_K_JSON: int = 5
BU_RERANK_TOP_K: int = 5

RFP_TOP_K: int = 6
# Hybrid RFP search: this many candidates from each of keyword (BM25) and
# embedding search are fused (reciprocal rank fusion), then reranked.
RFP_CANDIDATE_K: int = 20
# The cross-encoder dominates query time on CPU, so only the best fused
# candidates are reranked, each cut to its first characters. Measured on the
# sample RFPs (evaluation/): 20 x full text = 10.5 s/query; 10 x 500 chars =
# 1.6 s/query, with the same hits (46/46 item rows, 5/5 facts).
RFP_RERANK_CANDIDATES: int = 10
RFP_RERANK_MAX_CHARS: int = 500
RFP_USE_RERANKER: bool = os.environ.get("RFP_USE_RERANKER", "true").lower() not in ("0", "false", "no")

# Requirement extraction reads the whole RFP in windows of about this many
# characters (each window is one LLM call; windows run in parallel).
RFP_EXTRACT_WINDOW_CHARS: int = 12000


# ─── Matching ─────────────────────────────────────────────────────────────────

# Reranker relevance (0-1) of a catalogue candidate. Calibrated on the Havells
# catalogue: real products score 0.55-0.99, unrelated items (cables, hoses,
# certificates) score 0.00.
MATCH_TOP_K: int = 5                    # candidates shown to the LLM / as alternatives
MATCH_MIN_CANDIDATE_SCORE: float = 0.05 # below this a candidate is ignored → gap
MATCH_HIGH_CONFIDENCE: float = 0.90     # at/above: "high", status "matched"
MATCH_MEDIUM_CONFIDENCE: float = 0.50   # at/above: "medium"; below: "low" (both need review)

# Parallel LLM calls (matching, extraction). Keep low for free-tier rate limits.
LLM_MAX_CONCURRENCY: int = int(os.environ.get("LLM_MAX_CONCURRENCY", "4"))


# ─── Cache ────────────────────────────────────────────────────────────────────

# Document embeddings, keyed by model + exact texts (see embedding_cache.py).
CACHE_DIR: str = os.environ.get("SYNAPSE_CACHE_DIR", str(VAR_DIR / "cache"))
EMBEDDING_CACHE_DIR: str = str(Path(CACHE_DIR) / "embeddings")

# How many RFP search indexes to keep in memory at once (one per PDF).
RFP_TOOL_CACHE_SIZE: int = 8

FULFILLMENT_CACHE_PATH: str = str(PROJECT_ROOT / ".rfp_fulfillment_cache.json")


# ─── Database ─────────────────────────────────────────────────────────────────

# SQLite file holding the catalogue, demo stock and (later) users and projects.
# Rebuilt automatically from BU_JSON_PATH whenever that file changes.
DB_PATH: str = os.environ.get("SYNAPSE_DB_PATH", str(VAR_DIR / "synapse.db"))

# LangGraph checkpoints for the review workflow (one thread per project).
# Kept in its own file so graph writes never block catalogue/app queries.
CHECKPOINT_DB_PATH: str = os.environ.get(
    "SYNAPSE_CHECKPOINT_DB_PATH", str(VAR_DIR / "checkpoints.db")
)

# How long a proposal's prices are valid (shown on the cover and in terms).
PROPOSAL_VALIDITY_DAYS: int = 90

# Generated proposal PDFs, one folder per project.
PROPOSALS_DIR: str = os.environ.get("SYNAPSE_PROPOSALS_DIR", str(VAR_DIR / "proposals"))

# ─── Discovery ────────────────────────────────────────────────────────────────

# Generated sample RFPs and downloaded tender PDFs.
DISCOVERY_DIR: str = os.environ.get("SYNAPSE_DISCOVERY_DIR", str(VAR_DIR / "discovery"))

# The live GeM (bidplus.gem.gov.in) scraper is OFF unless explicitly enabled:
# the team believes the portal does not allow automated scraping. Enabling it
# also needs `pip install selenium` and Chrome.
DISCOVERY_GEM_ENABLED: bool = os.environ.get("DISCOVERY_GEM_ENABLED", "").lower() in ("1", "true", "yes")

# Stock figures are synthetic (no real company data). The seed makes every
# laptop generate the same numbers; the UI must label them as demo data.
DEMO_STOCK_SEED: int = 2026

# The Stock page marks a product "low" when fewer units than this are on hand.
STOCK_LOW_UNITS: int = 100


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
    freight_flat_inr: float = 0.0          # 0 = buyer pays, >0 = added to the quote (GST applies)
    payment_terms_discount_pct: float = 2.0  # extra discount on goods value if the buyer pays in advance
    default_qty: int = 100                 # used when the RFP states no quantity; flagged for review


PRICING_RULES = PricingRules()
