# Multi-Agent RFP Automation

An AI-powered pipeline that automates end-to-end analysis of a **Request for Proposal (RFP)** document against a **business-unit product catalog**. The system classifies each query, routes it through the relevant agents, and returns a synthesised procurement response — including gap analysis, pricing, and proposal-PDF generation.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Agent Pipeline](#agent-pipeline)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Running Tests](#running-tests)
- [Proposal Generation (Report Route)](#proposal-generation-report-route)
- [Exporting Quotes](#exporting-quotes)
- [Pricing Rules](#pricing-rules)
- [Troubleshooting](#troubleshooting)

---

## Overview

Given an RFP PDF and a product catalog, the system can answer questions such as:

- *"What are the eligibility criteria in the RFP?"*
- *"Which of our products can fulfil the RFP requirements?"*
- *"Generate a priced quotation for all matched requirements."*

The system is built on **LangGraph** for multi-agent orchestration, **OpenRouter** as the LLM backend (Meta Llama 4 Scout), **Gemini** for proposal text, and **FAISS + sentence-transformers** for retrieval-augmented generation (RAG).

---

## Architecture

```
User Query
    │
    ▼
┌───────────┐
│  Router   │  Classifies the query into one of six routes
└─────┬─────┘
      │
      ├──── "rfp"    ──► RFP Agent ─────────────────────────────────────────────► Synthesise ──► Response
      │
      ├──── "bu"     ──► BU Agent ──────────────────────────────────────────────► Synthesise ──► Response
      │
      ├──── "match"  ──► RFP Agent ──► Matching Agent ──────────────────────────► Synthesise ──► Response
      │
      ├──── "price"  ──► RFP Agent ──► Matching Agent ──► Pricing Agent ───────► Synthesise ──► Response
      │
      ├──── "full"   ──► RFP Agent ──► Matching Agent ──► Pricing Agent ───────► Synthesise ──► Response
      │
      └──── "report" ──► RFP Agent ──► BU Agent ──► Matching Agent ──► Pricing Agent ──► Report Agent ──► Synthesise ──► Response
```

All agents share a single `AgentState` TypedDict that flows through every node in the graph.

---

## Agent Pipeline

### Router
Classifies the incoming user query into one of six route labels using the LLM at `temperature=0.0` so routing is deterministic.

| Route | Meaning |
|-------|---------|
| `rfp` | Question about the RFP document itself (scope, clauses, deadlines) |
| `bu` | Question about the product catalog only (specs, prices, availability) |
| `match` | Gap/fulfillment check — can our products meet the RFP requirements? |
| `price` | Pricing or quotation request for matched RFP requirements |
| `full` | Full pipeline: RFP analysis + matching + pricing |
| `report` | Full proposal generation route that also produces a submission-ready vendor PDF |

### RFP Agent (`rfp_agent.py`)
- Retrieves relevant chunks from the RFP PDF via FAISS and answers the user query.
- Independently extracts a structured list of requirements `[{requirement, criticality, weight}]` for downstream agents.
- Uses **PyMuPDF** to parse the PDF and **BAAI/bge-small-en** for embeddings.

### BU Agent (`bu_agent.py` + `bu_rag.py`)
- `bu_agent.py` is the LangGraph node wrapper.
- It answers catalog-only questions using a three-index hybrid search implemented in `bu_rag.py`:
  1. **Small chunks** — fine-grained color/category groups.
  2. **Large chunks** — full product range context.
  3. **JSON index** — structured product rows (SAP code, HSN code, price, packing, availability).
- Results from all three indexes are merged and re-ranked with a **CrossEncoder** (`BAAI/bge-reranker-base`) before being passed to the LLM.

### Matching Agent (`matching_agent.py`)
- Iterates over each extracted RFP requirement and queries the BU RAG to determine whether the catalog can fulfil it.
- Produces a **weighted fulfillment score** (0–100) and a human-readable gap report listing matched items and gaps.

### Pricing Agent (`pricing_agent.py`)
- For each matched requirement, retrieves unit price, SAP code, packing, and availability from the catalog.
- Applies the configured pricing rules: base margin, volume-discount tiers, and GST.
- Outputs `quoted_items`, `line_items` (frontend-ready), `pricing_summary` (aggregate totals), and `quote_payload` (integration contract for PDF/quotation generation).
- Supports **repricing without re-retrieval** via `reprice_with_margin()`.

### Report Agent (`report_agent.py`)
- Used in the `report` route after pricing to generate a complete proposal document.
- Consumes all prior outputs (RFP analysis, BU answer, fulfillment, pricing) and drafts proposal text with **Google Gemini** (`gemini-2.5-flash` by default).
- Renders a styled vendor proposal PDF via **reportlab** and writes the generated file path to `report_pdf_path`.

### Synthesise Node (`master_agent.py`)
- Combines available outputs (RFP analysis, catalog answer, fulfillment report, pricing report) into a single structured response using the LLM.
- If only one section is present, it is returned directly to avoid a redundant LLM call.

---

## Project Structure

```
.
├── master_agent.py          # LangGraph orchestrator and CLI entrypoint
├── bu_agent.py              # BU node wrapper used by the graph
├── rfp_agent.py             # RFP RAG node — reads and analyses the RFP PDF
├── matching_agent.py        # Matching node — requirement vs. catalog gap analysis
├── pricing_agent.py         # Pricing node — quotes, margins, GST, volume discounts
├── report_agent.py          # Proposal generator node — Gemini + PDF rendering
├── bu_rag.py                # BU product catalog RAG tool (hybrid search + rerank)
├── rfp_rag.py               # RFP PDF RAG tool (FAISS + section chunking)
├── llm.py                   # Thin OpenRouter client wrapper used by all agents
├── state.py                 # Shared AgentState TypedDict
├── settings.py              # Central configuration (models, paths, pricing rules)
├── quote_export.py          # CSV export helper for quote payloads
├── requirements.txt         # Python dependencies (pinned)
├── .env.example             # Template for your local .env (API keys)
├── data/
│   ├── catalogue/
│   │   ├── havells.json              # Structured product catalog (JSON)
│   │   ├── havells_small_chunks.txt  # Fine-grained catalog text chunks
│   │   └── havells_large_chunks.txt  # Broad catalog text chunks
│   └── samples/
│       └── sample_rfp.pdf   # Sample RFP (ABC Infrastructure, modular switches)
├── docs/
│   ├── design/              # UI design screens for the v2 dashboard
│   └── pricing-agent.md     # Detailed pricing agent guide
├── legacy/
│   └── gem_scraper/         # Original standalone GeM scraper (reference only)
└── tests/                   # Unit tests (offline, LLM calls mocked)
```

---

## Prerequisites

- Python 3.12 (the pinned versions are tested on 3.12)
- An [OpenRouter API key](https://openrouter.ai/keys)
- A [Gemini API key](https://aistudio.google.com/apikey) for proposal PDF generation

---

## Installation

```bash
# Clone the repository
git clone https://github.com/shivam6969/Multi-Agent-RFP-Automation.git
cd Multi-Agent-RFP-Automation

# (Recommended) create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Note:** The first run downloads the embedding and reranker models (~200 MB total). Subsequent runs use the cached versions.

---

## Configuration

### Setting API keys

Keys are never stored in code. Copy the template and fill in your keys:

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

```
OPENROUTER_API_KEY=your-openrouter-key
GEMINI_API_KEY=your-gemini-key
```

`.env` is gitignored. Environment variables with the same names also work and take precedence.

### Other settings

Paths, models and pricing rules live in **`settings.py`**. Paths are resolved from the project folder, so commands work from any directory.

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | `meta-llama/llama-4-scout` | OpenRouter model (override in `.env`) |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite-preview` | Gemini model for proposal text (override in `.env`) |
| `RFP_PDF_PATH` | `data/samples/sample_rfp.pdf` | RFP PDF used by the CLI (override in `.env`) |
| `BU_SMALL_CHUNKS_PATH` | `data/catalogue/havells_small_chunks.txt` | Fine-grained catalog chunks |
| `BU_LARGE_CHUNKS_PATH` | `data/catalogue/havells_large_chunks.txt` | Broad catalog chunks |
| `BU_JSON_PATH` | `data/catalogue/havells.json` | Structured catalog JSON |

---

## Usage

Run the CLI from the repository root with your query as an argument:

```bash
python master_agent.py "<your query>"
```

### Example queries

```bash
# Analyze the RFP document
python master_agent.py "What are the eligibility criteria in the RFP?"

# Query the product catalog
python master_agent.py "Which 16A switches do we have and at what price?"

# Check which requirements can be fulfilled
python master_agent.py "Which RFP requirements can our catalog fulfill?"

# Generate a full quote
python master_agent.py "Give me pricing and a quote for all matched RFP requirements"

# Full end-to-end analysis
python master_agent.py "Full RFP analysis with gap report and pricing"

# Generate full proposal PDF (report route)
python master_agent.py "Generate the full vendor proposal PDF"

# Use a different RFP PDF (default: data/samples/sample_rfp.pdf)
python master_agent.py --rfp path/to/other_rfp.pdf "What are the delivery terms?"
```

The first run on a machine downloads the embedding models (~200 MB). After that,
document embeddings are cached in `var/cache/`, so later runs only pay for loading
the models (~15 s). Scanned PDFs without a text layer are rejected with a clear
message; they need OCR first.

### Programmatic usage

```python
from master_agent import run

response = run("Show me pricing for all matched products")
print(response)
```

---

## Running Tests

Unit tests run offline: every LLM call is mocked, and the test package blanks the API keys so an unmocked call fails instead of spending credit.

```bash
python -m unittest discover -s tests -t . -v
```

Expected output: **12 tests, all OK**.

---

## Proposal Generation (Report Route)

When the router classifies a query as `report`, the pipeline runs:

`RFP Agent → BU Agent → Matching Agent → Pricing Agent → Report Agent → Synthesise`

The report agent:
- Calls Gemini to generate structured proposal text.
- Builds a styled vendor proposal PDF with reportlab.
- Stores the generated file path in state (`report_pdf_path`) and includes it in the final response.

---

## Exporting Quotes

The `quote_export.py` module can write the quote payload to a CSV file:

```python
from quote_export import export_quote_csv

# quote_payload is produced by the pricing agent and stored in state["quote_payload"]
csv_path = export_quote_csv(quote_payload, "output/quote.csv")
print(f"Quote written to {csv_path}")
```

The CSV contains one row per line item with columns: `requirement`, `qty`, `currency`, `base_unit_price`, `margin_pct`, `volume_discount_pct`, `net_unit_price`, `gst_pct`, `gst_amount_per_unit`, `final_unit_price`, `line_total`.

---

## Pricing Rules

Pricing behaviour is controlled by the `PricingRules` dataclass in `settings.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_margin_pct` | `20.0` | Default gross margin applied to catalog cost |
| `min_margin_pct` | `0.0` | Minimum allowed margin |
| `max_margin_pct` | `60.0` | Maximum allowed margin |
| `gst_rate_pct` | `18.0` | GST percentage (fixed, not adjustable by frontend) |
| `currency` | `"INR"` | Quote currency |
| `rounding_digits` | `2` | Decimal places for all monetary values |
| `freight_flat_inr` | `0.0` | Flat freight added to the quote, GST applies (0 = buyer pays) |
| `payment_terms_discount_pct` | `2.0` | Discount on goods value when the buyer pays in advance (`advance_payment`) |
| `default_qty` | `100` | Quantity used when the RFP states none; flagged `qty_source: "default"` for review |

Unit prices come from the catalogue database (`catalogue.py`, loaded from `data/catalogue/havells.json` into `var/synapse.db`), never from LLM text. Stock figures are synthetic demo data.

**Volume discount tiers** (qty → discount %), applied after margin and before GST:

| Quantity range | Discount |
|---------------|----------|
| 1 – 199 | 0% |
| 200 – 499 | 5% |
| 500 – 999 | 8% |
| 1 000 + | 12% |

To recalculate pricing with a different margin (e.g. from a frontend slider) without re-running retrieval:

```python
from pricing_agent import reprice_with_margin
from settings import PRICING_RULES

updated_items, summary = reprice_with_margin(line_items, new_margin_pct=15.0, rules=PRICING_RULES)
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in the active virtual environment |
| `ValueError: OPENROUTER_API_KEY is not set` | Add it to `.env` in the project root (see `.env.example`) |
| `GEMINI_API_KEY is not set` | Add it to `.env` before running proposal/report generation queries |
| `FileNotFoundError` for PDF or catalog files | Check the files exist under `data/` and that any `RFP_PDF_PATH` override in `.env` is correct |
| Slow first run | Normal — embedding and reranker models are downloaded and cached on first use |
| OpenRouter API errors | Verify the key at [openrouter.ai/keys](https://openrouter.ai/keys) and that you have remaining credit |
