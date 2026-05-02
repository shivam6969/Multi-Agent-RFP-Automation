# Pricing Agent V1 Guide

This document describes the pricing pipeline introduced for margin-adjustable quoting.

## Goal

- Build deterministic pricing output from matched requirements.
- Allow frontend users to adjust **margin only** (GST stays fixed).
- Export quote line items as CSV.
- Pass a stable payload to quotation/PDF generation.

## Scope

Primary files:

- `pricing_agent.py`
- `state.py`
- `settings.py`
- `master_agent.py`
- `quote_export.py`

## State Contract

The pricing node writes these fields:

- `line_items`: frontend-ready rows used for repricing and export.
- `pricing_summary`: aggregate totals and metadata.
- `quote_payload`: integration contract for quotation/PDF step.
- `quoted_items` and `pricing_report`: retained for backward compatibility.

## Core Functions

In `pricing_agent.py`:

- `generate_initial_pricing(...)`
  - Builds initial line items from matched requirements + catalog prices.
- `reprice_with_margin(...)`
  - Recomputes totals from existing line items with a new user margin.
  - Does not rerun retrieval.

In `quote_export.py`:

- `export_quote_csv(quote_payload, output_path)`
  - Exports line items for download.

## Pricing Rules

Configured in `settings.py` (`PricingRules`):

- `base_margin_pct` (default starting margin)
- `min_margin_pct`, `max_margin_pct` (allowed frontend margin range)
- `gst_rate_pct` (fixed)
- `currency`
- `rounding_digits`
- volume discount tiers

## Frontend Integration

Suggested flow:

1. Call pipeline and render `pricing_summary` + `line_items`.
2. User changes margin in UI.
3. Call backend repricing (`reprice_with_margin`) with current line items.
4. Refresh summary/table.
5. Call CSV export for download.

## Quotation/PDF Handoff

Use `quote_payload` as the single contract object.

It contains:

- `quote_meta` (timestamp, source, user query)
- `pricing_summary`
- `line_items`

Quotation/PDF generator should consume this payload directly.

## Testing (step by step)

Use these checks in order: **unit tests first** (fast, no API key), then **full pipeline** (needs Groq and catalog/RFP data).

### Prerequisites

1. Open a terminal in the repo root (the folder that contains `master_agent.py`).
2. Install dependencies (once per machine or venv):

   ```bash
   python -m pip install -r requirements.txt
   ```

3. For **full pipeline** runs only, set your Groq API key.

   - **Windows PowerShell:**

     ```powershell
     $env:GROQ_API_KEY = "your-key-here"
     ```

   - **Windows CMD:**

     ```cmd
     set GROQ_API_KEY=your-key-here
     ```

   - **macOS / Linux:**

     ```bash
     export GROQ_API_KEY=your-key-here
     ```

   You can also rely on `GROQ_API_KEY` in the environment if your IDE already sets it.

### 1) Unit tests (recommended first)

Proves **margin repricing**, **rounding**, and **CSV export** without loading BU/RFP indexes or calling Groq.

From repo root:

```bash
python -m unittest tests/test_pricing_agent.py -v
```

Expected: **3 tests**, all **OK**.

### 2) Quick repricing sanity check (optional)

One-off Python check that `reprice_with_margin` runs (same logic the frontend would call later):

```bash
python -c "from pricing_agent import reprice_with_margin; from settings import PRICING_RULES; items=[{'requirement':'Test','qty':2,'currency':'INR','base_unit_price':100,'margin_pct':10,'volume_discount_pct':0,'net_unit_price':110,'gst_pct':18,'gst_amount_per_unit':0,'final_unit_price':0,'line_total':0,'raw_price_info':''}]; lines, s = reprice_with_margin(items, 15, PRICING_RULES); print('margin', s['margin_pct'], 'grand_total', s['grand_total']); print(lines[0])"
```

### 3) Full LangGraph run (pricing path)

This exercises **router → RFP → matching → pricing → synthesis**. First run may be **slow** (embedding/rerank models may download; BU RAG loads indexes).

From repo root, with `GROQ_API_KEY` set:

```bash
python master_agent.py "Give me pricing and a quote for all matched RFP requirements"
```

**What you see:** the CLI prints **`final_response`** (the synthesised report). The **Pricing** section in that text comes from `pricing_report`.

**Structured outputs** (`line_items`, `pricing_summary`, `quote_payload`) are written on the graph state inside `pricing_agent_node`; they are what a future API or frontend will return as JSON. The stock CLI does not dump that JSON unless you add a small debug print or a `--json` flag in `master_agent.py`.

Requirements for this to behave well:

- `Untitled document.pdf` (or whatever `RFP_PDF_PATH` points to in `settings.py`) must exist for RFP steps.
- BU data files (`havelsdata.json`, chunk `.txt` files) must exist as configured in `settings.py`.

### 4) CSV export only

You can test file export without the full graph by building a minimal payload:

```bash
python -c "from quote_export import export_quote_csv; p = export_quote_csv({'line_items': [{'requirement':'Sample','qty':1,'currency':'INR','base_unit_price':100,'margin_pct':20,'volume_discount_pct':0,'net_unit_price':120,'gst_pct':18,'gst_amount_per_unit':21.6,'final_unit_price':141.6,'line_total':141.6,'raw_price_info':''}]}, 'quote_out.csv'); print('Wrote', p)"
```

Open `quote_out.csv` in Excel or any spreadsheet app and confirm columns and values.

### Groq API key (for CLI / full pipeline)

This project talks to **Groq’s hosted LLM API**. You do **not** “create an API” in code for that—you **create an API key** on Groq’s site and put it in the environment as `GROQ_API_KEY`.

1. Open [Groq Console](https://console.groq.com/) and sign in (or create an account).
2. Go to **API Keys** and create a new key.
3. Copy the key. In PowerShell for one session:

   ```powershell
   $env:GROQ_API_KEY = "paste-your-key-here"
   ```

4. Run the pipeline:

   ```powershell
   python master_agent.py "Give me pricing and a quote for all matched RFP requirements"
   ```

Optional: put the key in `settings.py` only for local testing (not recommended for shared repos)—the code already falls back to `GROQ_API_KEY` from `settings.py` if the env var is missing.

### Troubleshooting

- **`ModuleNotFoundError`:** run `python -m pip install -r requirements.txt` in the same Python you use to run tests.
- **Groq / API errors on full run:** confirm `GROQ_API_KEY` is set in the **same** terminal session.
- **Long first run:** model download and FAISS index build are normal; later runs are faster.
