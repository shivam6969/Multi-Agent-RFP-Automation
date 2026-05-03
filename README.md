# Multi-Agent RFP Automation

An AI-powered pipeline that automates the end-to-end analysis of a **Request for Proposal (RFP)** document against a **business-unit product catalog**. The system classifies each user query, routes it through the relevant agents, and returns a synthesised procurement report — including gap analysis and a detailed pricing quote.

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
- [Exporting Quotes](#exporting-quotes)
- [Pricing Rules](#pricing-rules)
- [Troubleshooting](#troubleshooting)

---

## Overview

Given an RFP PDF and a product catalog, the system can answer questions such as:

- *"What are the eligibility criteria in the RFP?"*
- *"Which of our products can fulfil the RFP requirements?"*
- *"Generate a priced quotation for all matched requirements."*

The system is built on **LangGraph** for multi-agent orchestration, **Groq** as the LLM backend (Meta Llama 4 Scout), and **FAISS + sentence-transformers** for retrieval-augmented generation (RAG).

---

## Architecture

```
User Query
    │
    ▼
┌───────────┐
│  Router   │  Classifies the query into one of five routes
└─────┬─────┘
      │
      ├──── "rfp"   ──► RFP Agent ──────────────────────────────────► Synthesise ──► Response
      │
      ├──── "bu"    ──► BU Agent ───────────────────────────────────► Synthesise ──► Response
      │
      ├──── "match" ──► RFP Agent ──► Matching Agent ───────────────► Synthesise ──► Response
      │
      ├──── "price" ──► RFP Agent ──► Matching Agent ──► Pricing ──► Synthesise ──► Response
      │
      └──── "full"  ──► RFP Agent ──► Matching Agent ──► Pricing ──► Synthesise ──► Response
```

All agents share a single `AgentState` TypedDict that flows through every node in the graph.

---

## Agent Pipeline

### Router
Classifies the incoming user query into one of five route labels using the LLM at `temperature=0.0` so routing is deterministic.

| Route | Meaning |
|-------|---------|
| `rfp` | Question about the RFP document itself (scope, clauses, deadlines) |
| `bu` | Question about the product catalog only (specs, prices, availability) |
| `match` | Gap/fulfillment check — can our products meet the RFP requirements? |
| `price` | Pricing or quotation request for matched RFP requirements |
| `full` | Full pipeline: RFP analysis + matching + pricing |

### RFP Agent (`rfp_agent.py`)
- Retrieves relevant chunks from the RFP PDF via FAISS and answers the user query.
- Independently extracts a structured list of requirements `[{requirement, criticality, weight}]` for downstream agents.
- Uses **PyMuPDF** to parse the PDF and **BAAI/bge-small-en** for embeddings.

### BU Agent (`bu_rag.py`)
- Answers catalog-only questions using a three-index hybrid search:
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

### Synthesise Node (`master_agent.py`)
- Combines available outputs (RFP analysis, catalog answer, fulfillment report, pricing report) into a single structured response using the LLM.
- If only one section is present, it is returned directly to avoid a redundant LLM call.

---

## Project Structure

```
.
├── master_agent.py          # LangGraph orchestrator and CLI entrypoint
├── rfp_agent.py             # RFP RAG node — reads and analyses the RFP PDF
├── matching_agent.py        # Matching node — requirement vs. catalog gap analysis
├── pricing_agent.py         # Pricing node — quotes, margins, GST, volume discounts
├── bu_rag.py                # BU product catalog RAG tool (hybrid search + rerank)
├── rfp_rag.py               # RFP PDF RAG tool (FAISS + section chunking)
├── llm.py                   # Thin Groq client wrapper used by all agents
├── state.py                 # Shared AgentState TypedDict
├── settings.py              # Central configuration (models, paths, pricing rules)
├── quote_export.py          # CSV export helper for quote payloads
├── requirements.txt         # Python dependencies
├── havelsdata.json          # Structured product catalog (JSON)
├── havells_small_chunks (1).txt  # Fine-grained catalog text chunks
├── havells_large_chunks.txt      # Broad catalog text chunks
├── Untitled document.pdf    # RFP document (replace with your own)
├── docs/
│   └── pricing-agent.md     # Detailed pricing agent guide
└── tests/
    └── test_pricing_agent.py  # Unit tests for repricing and CSV export
```

---

## Prerequisites

- Python 3.9 or later
- A [Groq API key](https://console.groq.com/) (free tier available)

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

All configuration lives in **`settings.py`**. The most important settings are:

| Setting | Default | Description |
|---------|---------|-------------|
| `GROQ_API_KEY` | `"YOUR_GROQ_API_KEY_HERE"` | Groq API key — prefer the `GROQ_API_KEY` env var |
| `LLM_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq-hosted LLM |
| `RFP_PDF_PATH` | `"Untitled document.pdf"` | Path to your RFP PDF |
| `BU_SMALL_CHUNKS_PATH` | `"havells_small_chunks (1).txt"` | Fine-grained catalog chunks |
| `BU_LARGE_CHUNKS_PATH` | `"havells_large_chunks.txt"` | Broad catalog chunks |
| `BU_JSON_PATH` | `"havelsdata.json"` | Structured catalog JSON |

### Setting the API key

```bash
# macOS / Linux
export GROQ_API_KEY="your-key-here"

# Windows PowerShell
$env:GROQ_API_KEY = "your-key-here"

# Windows CMD
set GROQ_API_KEY=your-key-here
```

Alternatively, paste your key directly into `settings.py` (not recommended for shared repositories).

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
python master_agent.py "What LED streetlight products do we have and at what price?"

# Check which requirements can be fulfilled
python master_agent.py "Which RFP requirements can our catalog fulfill?"

# Generate a full quote
python master_agent.py "Give me pricing and a quote for all matched RFP requirements"

# Full end-to-end analysis
python master_agent.py "Full RFP analysis with gap report and pricing"
```

### Programmatic usage

```python
from master_agent import run

response = run("Show me pricing for all matched products")
print(response)
```

---

## Running Tests

Unit tests cover repricing logic, rounding correctness, and CSV export — no API key or data files needed.

```bash
python -m unittest tests/test_pricing_agent.py -v
```

Expected output: **3 tests, all OK**.

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
| `freight_flat_inr` | `0.0` | Flat freight charge (0 = buyer pays) |
| `payment_terms_discount_pct` | `2.0` | Extra discount for advance payment |

**Volume discount tiers** (qty → discount %):

| Quantity range | Discount |
|---------------|----------|
| 1 – 99 | 0% |
| 100 – 499 | 5% |
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
| `ValueError: GROQ_API_KEY is not set` | Export `GROQ_API_KEY` in the same terminal session |
| `FileNotFoundError` for PDF or catalog files | Check `RFP_PDF_PATH` and BU paths in `settings.py` match the actual file names |
| Slow first run | Normal — embedding and reranker models are downloaded and cached on first use |
| Groq API errors | Verify the key is valid at [console.groq.com](https://console.groq.com/) and that you have remaining quota |
