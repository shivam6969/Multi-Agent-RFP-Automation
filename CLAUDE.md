# SYNAPSE: RFP Response Automation

Multi-agent system that takes an RFP PDF plus a product catalogue and produces requirement extraction, catalogue matching, deterministic pricing and a proposal PDF. Team: Shivam, Lakshay, Vidipt, Khushi (BML Munjal University, Project-III/IV).

## Goal of the v2 work
Turn the CLI prototype into a working product:
- New UI (dashboard) to upload/discover RFPs, review each agent's output, edit and approve, adjust margin, download the proposal.
- Human-in-the-loop checkpoints between agents (requirements, matching, pricing).
- Discovery Agent (GeM tender scraper) integrated as a real agent/service.
- Pricing stays deterministic.

## Stack
Python, LangGraph (StateGraph), OpenRouter (`meta-llama/llama-4-scout`) via the `openai` client, Gemini (`google-genai`) for proposal text, sentence-transformers (BGE) + FAISS, cross-encoder rerank, PyMuPDF, reportlab. Discovery: Selenium + pdfplumber.

## Pipeline (master_agent.py)
Router (temp 0) -> one of six routes: rfp | bu | match | price | full | report.
- rfp: rfp_agent -> synthesise
- bu: bu_agent -> synthesise
- match: rfp -> matching -> synthesise
- price / full: rfp -> matching -> pricing -> synthesise
- report: rfp -> bu -> matching -> pricing -> report_agent -> synthesise
All nodes read/write one `AgentState` TypedDict (state.py). Nodes mutate and return the same dict.

## File map
- `master_agent.py` graph + router + synthesis + CLI
- `rfp_agent.py` / `rfp_rag.py` answers query from RFP PDF, extracts `[{requirement, criticality, weight}]`
- `bu_agent.py` / `bu_rag.py` three-index catalogue retrieval (small chunks, large chunks, JSON) + cross-encoder
- `matching_agent.py` one batched LLM call to tag SKU vs compliance, then one RAG+LLM call per requirement; weighted score
- `pricing_agent.py` filter/atomise requirements, pull price/qty, apply margin, volume tier, GST; `reprice_with_margin()` reprices without re-retrieval
- `report_agent.py` Gemini drafts text, reportlab renders PDF
- `quote_export.py` CSV export; `settings.py` all config incl. `PricingRules`; `llm.py` OpenRouter wrapper
- `tests/` unit tests (offline; `tests/__init__.py` blanks API keys) and `run_*.py` manual runners
- `data/catalogue/` Havells catalogue (`havells.json`, small/large chunk files); `data/samples/sample_rfp.pdf` the one real sample RFP
- `var/` (gitignored) runtime output: database, uploads, index cache, generated proposals
- `legacy/gem_scraper/` teammate's original standalone scraper, kept for reference only
- `.env.example` template; real keys go in `.env` (gitignored), loaded by `settings.py` via python-dotenv

## Known issues found in review (fix early)
1. ~~README out of date (Groq, wrong volume tiers).~~ Fixed in M0. Volume tiers in code: 1-199 0%, 200-499 5%, 500-999 8%, 1000+ 12%.
2. `get_rfp_rag_tool` and `get_bu_rag_tool` are module-level singletons, so only one RFP and one catalogue per process. Needs per-RFP instances. (RFP path now defaults to `data/samples/sample_rfp.pdf`, overridable via `RFP_PDF_PATH` in `.env`.)
3. ~~No `.env` loading.~~ Fixed in M0: `settings.py` loads `.env`; no keys in code.
4. ~~`requirements.txt` unpinned.~~ Fixed in M0 (pinned, Python 3.12).
5. ~~`.gitignore` missing `.venv/`.~~ Fixed in M0.
6. Embeddings and FAISS indexes rebuild on every start (8-15 s). Cache them.
7. Matching runs one LLM call per requirement sequentially. Parallelise.
8. LLM JSON is parsed with regex fallbacks; add schema validation.
9. Graph runs straight through with no pause points. Use LangGraph interrupts plus a checkpointer for human review.
10. `_open_pdf` in master_agent.py uses `os.startfile` (Windows only); remove from library code.
11. Agents print banner output; move to logging so an API/UI can capture progress.

## Discovery agent (scraper) notes
The original code is in `legacy/gem_scraper/` (reference only). The new Discovery Agent will be a fresh `discovery/` package with swappable sources: generated sample RFPs + PDF upload (default), GeM scraper adapter off by default (the team believes GeM does not allow scraping).
- `z.py`: minimal Selenium script, `ELECTRICAL_KEYWORDS` is empty and the download path is hardcoded to a personal Windows path.
- `scraper/rfp_agent.py` (904 lines): Streamlit app with keyword list, headless Chrome scraping of `https://bidplus.gem.gov.in/all-bids`, pdfplumber text extraction, regex analysis, stock comparison, session JSON, Move Forward / Hold cards. **Its filename clashes with the repo's `rfp_agent.py`**; rename on import.
- `scraper/rfp_extractor.py`, `app.py` (Flask), `streamlit_app.py`: earlier variants; consolidate.
- `sessions/*.json` hold real scraped tenders (`bid_title`, `pdf_url`, `filename`, `analysis`); useful seed data. `extracted_data/rfps_catalog.json` is dummy sample data.
- Uses pdfplumber; the main repo uses PyMuPDF. Pick one.
- Do not commit its `.venv`, `__pycache__`.

## Rules
- Never put API keys in code or commits.
- LLMs never do price arithmetic; keep all money math in `pricing_agent.py` / `PricingRules`.
- Keep the `AgentState` contract (state.py) backward compatible; extend it, do not rename fields.
- Plan first for anything touching more than two files; commit in small steps on a feature branch.

## Commands
```
python -m venv venv && venv\Scripts\activate   # Windows
pip install -r requirements.txt
copy .env.example .env                           # then add your keys
python master_agent.py "What are the eligibility criteria in the RFP?"
python -m unittest discover -s tests -t . -v     # offline, 12 tests
```
