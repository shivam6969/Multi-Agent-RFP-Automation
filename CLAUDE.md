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
- `tests/` unit tests and `run_*.py` manual runners

## Known issues found in review (fix early)
1. README is out of date: says Groq; code uses OpenRouter. README volume tiers (1-99 / 100-499) differ from `settings.py` (1-199 0%, 200-499 5%, 500-999 8%, 1000+ 12%). Treat code as truth.
2. RFP PDF path is a fixed setting (`Untitled document.pdf`, not in the repo). `get_rfp_rag_tool` and `get_bu_rag_tool` are module-level singletons, so only one RFP and one catalogue per process. Needs per-RFP instances for a multi-RFP product.
3. API keys come from env vars or by editing `settings.py`; `.env` is gitignored but nothing loads it. Add python-dotenv and a config layer.
4. `requirements.txt` is unpinned. Pin versions.
5. `.gitignore` ignores `venv/` but not `.venv/`.
6. Embeddings and FAISS indexes rebuild on every start (8-15 s). Cache them.
7. Matching runs one LLM call per requirement sequentially. Parallelise.
8. LLM JSON is parsed with regex fallbacks; add schema validation.
9. Graph runs straight through with no pause points. Use LangGraph interrupts plus a checkpointer for human review.
10. `_open_pdf` in master_agent.py uses `os.startfile` (Windows only); remove from library code.
11. Agents print banner output; move to logging so an API/UI can capture progress.

## Discovery agent (scraper) notes
Lives outside this repo in the teammate's folder; bring it in as `discovery/`.
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
python master_agent.py "What are the eligibility criteria in the RFP?"
python -m unittest tests/test_pricing_agent.py -v
```
