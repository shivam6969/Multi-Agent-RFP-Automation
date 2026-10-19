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

Optional inputs read from state: `margin_pct`, `advance_payment`, `freight_inr`.

## Where prices and quantities come from (v2)

- **Product:** chosen by the Matching Agent (`product_matcher.py`): catalogue
  candidates with relevance scores, then one LLM call picks a candidate
  **by id**; an id not in the list is rejected. Each matched item carries
  `product_id`.
- **Price:** the Pricing Agent makes **no LLM calls**. For every matched
  product item it reads the unit price, SAP code, HSN and packing from the
  catalogue database (`catalogue.py`) by `product_id`. No price is ever
  parsed out of LLM text.
- **Quantity** (`qty_source` on each line):
  `rfp` (quantity field extracted from the RFP) → `requirement` / `query`
  (an explicit "N units / pcs / nos" phrase) → `default`
  (`PricingRules.default_qty`, to be reviewed by a human). Numbers inside
  product names such as "RJ-45" or "16A" are never treated as quantities.
- **Stock:** `monthly_available`, `stock_on_hand`, `lead_time_days` are
  synthetic demo figures (`stock_is_demo: true`).

## Core Functions

In `pricing_agent.py`:

- `generate_initial_pricing(...)`
  - Builds initial line items from matched requirements + catalogue prices.
- `reprice_with_margin(line_items, margin, rules, *, advance_payment=False, freight=None)`
  - Recomputes lines and totals for a new margin / payment terms.
  - Does not rerun retrieval or any LLM call; keeps catalogue fields on each line.

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
- volume discount tiers (per line, after margin, before GST)
- `payment_terms_discount_pct`: off the goods value when `advance_payment` is set
- `freight_flat_inr`: flat freight added to the quote
- `default_qty`: quantity used when the RFP states none

Order of operations: per line, margin → volume discount → GST; then on the
totals, minus the advance-payment discount, plus freight. GST is applied to
(freight − discount) at the same rate. With neither, `grand_total` equals the
sum of `line_total`.

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

Use these checks in order: **unit tests first** (fast, no API key), then **full pipeline** (needs an OpenRouter key and the catalog/RFP data).

### Prerequisites

1. Open a terminal in the repo root (the folder that contains `master_agent.py`).
2. Install dependencies (once per machine or venv):

   ```bash
   python -m pip install -r requirements.txt
   ```

3. For **full pipeline** runs only, copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` (see "API keys" below).

### 1) Unit tests (recommended first)

Proves **margin repricing**, **rounding**, and **CSV export** without loading BU/RFP indexes or calling an LLM.

From repo root:

```bash
python -m unittest tests.test_pricing_agent -v
```

Expected: **3 tests**, all **OK**. Run the whole offline suite with `python -m unittest discover -s tests -t . -v`.

### 2) Quick repricing sanity check (optional)

One-off Python check that `reprice_with_margin` runs (same logic the frontend would call later):

```bash
python -c "from pricing_agent import reprice_with_margin; from settings import PRICING_RULES; items=[{'requirement':'Test','qty':2,'currency':'INR','base_unit_price':100,'margin_pct':10,'volume_discount_pct':0,'net_unit_price':110,'gst_pct':18,'gst_amount_per_unit':0,'final_unit_price':0,'line_total':0,'raw_price_info':''}]; lines, s = reprice_with_margin(items, 15, PRICING_RULES); print('margin', s['margin_pct'], 'grand_total', s['grand_total']); print(lines[0])"
```

### 3) Full LangGraph run (pricing path)

This exercises **router → RFP → matching → pricing → synthesis**. First run may be **slow** (embedding/rerank models may download; BU RAG loads indexes).

From repo root, with `OPENROUTER_API_KEY` set in `.env`:

```bash
python master_agent.py "Give me pricing and a quote for all matched RFP requirements"
```

**What you see:** the CLI prints **`final_response`** (the synthesised report). The **Pricing** section in that text comes from `pricing_report`.

**Structured outputs** (`line_items`, `pricing_summary`, `quote_payload`) are written on the graph state inside `pricing_agent_node`; they are what a future API or frontend will return as JSON. The stock CLI does not dump that JSON unless you add a small debug print or a `--json` flag in `master_agent.py`.

Requirements for this to behave well:

- `data/samples/sample_rfp.pdf` (or whatever `RFP_PDF_PATH` points to) must exist for RFP steps.
- BU data files (`data/catalogue/havells.json` and the chunk `.txt` files) must exist.

### 4) CSV export only

You can test file export without the full graph by building a minimal payload:

```bash
python -c "from quote_export import export_quote_csv; p = export_quote_csv({'line_items': [{'requirement':'Sample','qty':1,'currency':'INR','base_unit_price':100,'margin_pct':20,'volume_discount_pct':0,'net_unit_price':120,'gst_pct':18,'gst_amount_per_unit':21.6,'final_unit_price':141.6,'line_total':141.6,'raw_price_info':''}]}, 'quote_out.csv'); print('Wrote', p)"
```

Open `quote_out.csv` in Excel or any spreadsheet app and confirm columns and values.

### API keys (for CLI / full pipeline)

This project calls LLMs through **OpenRouter** (all agents) and **Gemini** (proposal text only).

1. Create a key at [openrouter.ai/keys](https://openrouter.ai/keys) (and a Gemini key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) if you need the PDF).
2. Copy `.env.example` to `.env` in the repo root and paste the keys:

   ```
   OPENROUTER_API_KEY=paste-your-key-here
   GEMINI_API_KEY=paste-your-key-here
   ```

3. Run the pipeline:

   ```powershell
   python master_agent.py "Give me pricing and a quote for all matched RFP requirements"
   ```

Never put keys in `settings.py` or any other committed file; `.env` is gitignored.

### Troubleshooting

- **`ModuleNotFoundError`:** run `python -m pip install -r requirements.txt` in the same Python you use to run tests.
- **`OPENROUTER_API_KEY is not set`:** check `.env` exists in the repo root and contains the key.
- **Long first run:** model download and FAISS index build are normal; later runs are faster.
