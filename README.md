# SYNAPSE: Multi-Agent RFP Automation

A local web dashboard that takes a tender (**Request for Proposal**, RFP) from
discovery to a priced proposal PDF. AI agents read the RFP, match every
requirement to the electrical product catalogue (Havells) and draft the
proposal text; a person reviews and approves each step; prices are calculated
by fixed rules, never by the AI.

**Discover / upload RFP → Requirements ⏸ → Product matching ⏸ → Pricing ⏸ → Proposal ⏸ → PDF**
(⏸ = you review, edit and approve before the next agent runs)

BML Munjal University, Project-III/IV: Shivam, Lakshay, Vidipt, Khushi.

---

## Quick start

You need **Python 3.12**, **Node.js 18+** and an **OpenRouter API key**
(a **Gemini API key** is optional; it writes the proposal text).

```bash
start.bat            # Windows: double-click it, or run it in a terminal
./start.sh           # macOS / Linux
```

The first run creates `venv/`, installs the Python packages (several minutes:
PyTorch is large), builds the dashboard and creates `.env` from `.env.example`.
Put your keys in `.env`:

```
OPENROUTER_API_KEY=...
GEMINI_API_KEY=...
```

then run it again. The browser opens http://127.0.0.1:8000; create an account
and start a session. Later runs start in under a minute (the AI models load
once per start). Stop it with **Ctrl+C** in that window.

Options: `python start.py --port 8001 --no-browser --skip-build`.
Your data stays on this laptop in `var/` (database, uploads, proposals).
A guided demo script is in [docs/DEMO.md](docs/DEMO.md); the full explanation of the
project (for the team, reports and slides) is in [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md).

**What is demo data:** stock levels are synthetic (labelled "demo"), "Sample
tenders" are generated with fictitious buyers and stamped as samples, and the
company in **Settings** is a placeholder until you enter your own. Live GeM
scraping is off by default.

The rest of this README covers the agents, the command-line tools and the API.

---

## Table of Contents

- [Quick start](#quick-start)
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

### RFP Agent (`rfp_agent.py` + `rfp_parser.py`, `rfp_tables.py`, `rfp_rag.py`)
- **Reads tables.** `rfp_parser.py` uses PyMuPDF to split the PDF into headings, text and tables (ruled tables via `find_tables()`, item lists without gridlines, tables continued over pages, merged cells, two-row headers), and removes running headers and footers.
- **Item lists are read by rules.** `rfp_tables.py` recognises the columns (description, quantity, unit, per-location quantities, make, rate…). Each item row becomes a requirement with the exact quantity from its cell and its source ("p.2 · Table T2, row 13"); the LLM never guesses which number belongs to which item.
- **The rest by the LLM.** Narrative sections and the other tables (as Markdown) are read in parallel windows and return `[{requirement, criticality, quantity}]`, schema-validated.
- **Answers questions** with hybrid search over section text, table slices and single table rows: BM25 keywords + **BAAI/bge-small-en-v1.5** embeddings, fused and reranked with the cross-encoder.
- Try it on the table-heavy samples in `data/samples/rfp_*.pdf` (made by `python -m tests.sample_rfps`); `python -m tests.run_rfp_table_eval` scores them without LLM calls.

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
├── start.py / start.bat / start.sh   # one-command start (setup, build, server, browser)
├── log_setup.py             # logging for command-line entry points (UTF-8 safe)
├── workflow.py              # review workflow: agents with pause points (LangGraph + SQLite)
├── api/                     # FastAPI backend for the dashboard (python -m api)
├── frontend/                # React dashboard (built into frontend/dist)
├── discovery/               # Discovery Agent: sample tenders, uploads, GeM (off)
├── db.py / catalogue.py     # SQLite database; catalogue + demo stock loader
├── proposal_writer.py       # AI proposal sections (no amounts)
├── proposal_pdf.py          # deterministic proposal PDF
├── company_profile.py       # "Prepared by" details (Settings page)
├── master_agent.py          # LangGraph Q&A orchestrator and CLI entrypoint
├── bu_agent.py              # BU node wrapper used by the graph
├── rfp_agent.py             # RFP RAG node — reads and analyses the RFP PDF
├── rfp_parser.py            # PDF -> headings, text and tables (PyMuPDF)
├── rfp_tables.py            # table columns/kinds; item rows -> requirements
├── matching_agent.py        # Matching node — requirement vs. catalog gap analysis
├── pricing_agent.py         # Pricing node — quotes, margins, GST, volume discounts
├── report_agent.py          # Proposal generator node — Gemini + PDF rendering
├── bu_rag.py                # BU product catalog RAG tool (hybrid search + rerank)
├── rfp_rag.py               # RFP RAG tool (text/table/row chunks, hybrid search + rerank)
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
│       ├── sample_rfp.pdf   # Sample RFP (ABC Infrastructure, modular switches)
│       └── rfp_*.pdf        # Table-heavy sample RFPs + *_truth.json (tests/sample_rfps.py)
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

`start.bat` / `start.sh` (see [Quick start](#quick-start)) do all of this. By hand:

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

### Backup API (optional)

If a call fails (outage, rate limit, bad key, timeout), the agents retry and then
switch to a backup automatically. Add any OpenAI-compatible API as the backup:

```
LLM_FALLBACK_API_KEY=...
LLM_FALLBACK_MODEL=...                              # empty = same model
LLM_FALLBACK_BASE_URL=https://openrouter.ai/api/v1  # or Groq / OpenAI / Together …
GEMINI_FALLBACK_API_KEY=...                         # proposal text; then falls back to OpenRouter
```

A fallback is logged as a warning; see `.env.example` for all options.

### Evaluating the agents

```bash
python -m evaluation.run_eval --offline-only        # parsing + retrieval, no API calls
python -m evaluation.run_eval                       # full workflow on the 3 sample RFPs
python -m evaluation.run_eval --llm-models meta-llama/llama-4-scout,openai/gpt-4o-mini
```

It scores each agent against the samples' ground truth (items, quantities,
catalogue products, terms) and reports time, LLM calls, tokens and cost per
stage and per model in `var/eval/<timestamp>/report.md`.

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

## Review Workflow (human-in-the-loop)

`workflow.py` runs one RFP as a *project* that pauses for a human at three points:

```
extract requirements → ⏸ review requirements → match products → ⏸ review matches
  → price → ⏸ review pricing → proposal PDF
```

Progress is saved after every step in `var/checkpoints.db`, so a review can be
finished later, even after a restart. The dashboard will drive this; until then
you can try it from the terminal:

```bash
python workflow.py start --rfp data/samples/sample_rfp.pdf --id demo
python workflow.py status demo
python workflow.py approve demo                         # approve as-is
python workflow.py approve demo --decision edits.json   # approve with changes
python workflow.py retry demo                           # re-run a failed step
```

Example `edits.json` at the matching review:

```json
{"decisions": [
  {"id": "R4", "action": "change", "product_id": "<id from alternatives>"},
  {"id": "R7", "action": "reject"},
  {"id": "R12", "action": "confirm"}
], "accept_all": true}
```

At the pricing review: `{"margin_pct": 18, "advance_payment": true, "freight_inr": 1500, "quantities": {"R1": 400}}`.

---

## Dashboard (web UI)

The dashboard is a React app in `frontend/` that talks to the backend API.

**One-time setup** (needs [Node.js](https://nodejs.org) 18+):

```bash
cd frontend
npm install
npm run build        # creates frontend/dist
cd ..
```

**Run it:**

```bash
python -m api        # then open http://127.0.0.1:8000
```

Create an account on the login page ("Create an account"), then on **New Session**
either **Find RFPs** (keywords, location, dates, budget) or **Upload a PDF** (e.g.
`data/samples/sample_rfp.pdf`). Pick one RFP and follow the steps: Requirements →
Product Matching → Pricing → Proposal. While the AI is working the page shows
progress; you can leave and come back.

**Home** shows where things stand, worked out from your projects and RFPs (no AI):
proposals generated, reviews waiting, RFPs received and win rate; reminders
(deadlines today or passed, paused reviews, failed steps, proposals still to
submit; also behind the bell icon); a deadline calendar; a six-month
"Generated vs Won" chart; and the open RFPs, soonest deadline first.
**Completed / History** lists finished proposals; set each one's status
(Submitted / Won / Lost) as you hear back, and the win rate and chart follow.

**Where RFPs come from (Discovery Agent, `discovery/`):**

| Source | Default | What it does |
|--------|---------|--------------|
| Sample tenders | on | Generates realistic sample RFPs from the catalogue for your search, with fictitious buyers. Every page is stamped as a sample. Same search → same RFPs. Unknown keywords (e.g. "cable") become items the catalogue can't supply, so matching shows gaps. |
| Upload | on | Your own PDF. The AI reads the buyer, deadline, location, category and budget (GeM documents are read by rules, no AI). |
| GeM portal (live) | **off** | Port of the team's Selenium scraper. Scraping the portal may not be permitted, so it only runs with `DISCOVERY_GEM_ENABLED=true` in `.env` plus `pip install selenium` and Chrome. Not tested against the live site. |

**Working on the UI:** run `python -m api` in one terminal and `npm run dev` in
`frontend/` in another, then open http://localhost:5173 (changes reload instantly;
`/api` calls are forwarded to the backend).

---

## Backend API (for the dashboard)

```bash
python -m api          # http://127.0.0.1:8000  — interactive docs at /docs
```

It runs only on your own laptop (bound to 127.0.0.1). Data lives in `var/`:
`synapse.db` (users, RFPs, projects, catalogue), `checkpoints.db` (review
workflow progress), `uploads/`, `proposals/`, `exports/`.

| Area | Endpoints |
|------|-----------|
| Accounts | `POST /api/auth/register`, `POST /api/auth/login` (`remember`), `POST /api/auth/logout`, `GET /api/auth/me` |
| RFPs | `POST /api/rfps/upload` (PDF), `GET /api/rfps?q=`, `GET /api/rfps/{id}`, `GET /api/rfps/{id}/document` |
| Projects | `POST /api/projects` `{rfp_id}`, `GET /api/projects`, `GET /api/projects/{id}`, `POST /api/projects/{id}/review` `{decision}`, `POST /api/projects/{id}/retry` |
| Pricing | `POST /api/projects/{id}/pricing/preview` (margin slider, saves nothing), `GET /api/projects/{id}/quote.csv` |
| Proposal / history | `GET /api/projects/{id}/proposal.pdf`, `PATCH /api/projects/{id}/outcome` `{submitted|won|lost}` |
| Catalogue | `GET /api/catalogue/products?q=` ("Change Product" search) |
| Dashboard | `GET /api/dashboard` (KPIs, reminders, deadlines, monthly activity, pending RFPs) |
| Health | `GET /api/health` (whether API keys are configured; never the keys) |

Starting a project or sending a review returns immediately (`stage: "running"`
with a `progress` message); the work continues in the background and the UI
polls `GET /api/projects/{id}`. Review `decision` bodies are the same as in the
workflow section above.

---

## Running Tests

Unit tests run offline: every LLM call is mocked, and the test package blanks the API keys so an unmocked call fails instead of spending credit.

```bash
python -m unittest discover -s tests -t . -v
```

Expected output: **181 tests, all OK**.

---

## Proposal Generation (Report Route)

When the router classifies a query as `report`, the pipeline runs:

`RFP Agent → BU Agent → Matching Agent → Pricing Agent → Report Agent → Synthesise`

The report agent works in two halves:
- **Drafting** (`proposal_writer.py`): the LLM writes each section (Gemini when
  `GEMINI_API_KEY` is set, otherwise OpenRouter). It is told never to write prices
  and to claim only the compliance items a reviewer confirmed; any section that
  still mentions a ₹ amount is flagged.
- **Rendering** (`proposal_pdf.py`): a deterministic PDF. The cover facts, pricing
  table and totals are copied from the approved pricing; no amounts come from the LLM.

"Prepared by" comes from the company profile (dashboard → **Settings**). The
default is a fictitious demo company, so replace it with your details.

In the dashboard, the proposal is a review step of its own: edit any section,
regenerate one with an instruction ("make it shorter"), watch the PDF preview
update, save a draft, then **Complete Generation**.

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
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in the active virtual environment (or use `start.bat`, which does it) |
| Dashboard says "No OpenRouter API key" | Put `OPENROUTER_API_KEY=...` in `.env` in the project folder and restart |
| "address already in use" on start | SYNAPSE is already running in another window, or use `python start.py --port 8001` |
| Dashboard looks out of date after `git pull` | Run `start.bat` again (it rebuilds when the UI code changed), or `npm run build` in `frontend/` |
| `UnicodeEncodeError` in old versions | Fixed: agents log instead of printing; entry points use UTF-8 output |
| `ValueError: OPENROUTER_API_KEY is not set` | Add it to `.env` in the project root (see `.env.example`) |
| `GEMINI_API_KEY is not set` | Add it to `.env` before running proposal/report generation queries |
| `FileNotFoundError` for PDF or catalog files | Check the files exist under `data/` and that any `RFP_PDF_PATH` override in `.env` is correct |
| Slow first run | Normal — embedding and reranker models are downloaded and cached on first use |
| OpenRouter API errors | Verify the key at [openrouter.ai/keys](https://openrouter.ai/keys) and that you have remaining credit |
