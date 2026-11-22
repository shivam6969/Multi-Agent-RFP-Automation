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
- `rfp_agent.py` / `rfp_rag.py` answers query from RFP PDF, extracts `[{id, requirement, criticality, weight, quantity, unit?, origin: table|text|reviewer, source?}]`; state `rfp_tables` = table summaries
- `rfp_parser.py` PDF → ordered heading/text/table `Block`s (PyMuPDF only). Ruled tables via `find_tables()`; lists without gridlines via a header line naming description + quantity (column edges from header cells, words assigned by x, stop at a big gap or a numbered heading). Joins tables across pages (repeated header dropped), fills vertically merged cells (`None` covered from above), combines two-row headers, ignores a single-cell "outside header" (PyMuPDF takes the heading above a table as one), strips repeated margin lines / "Page N", headings from numbering + size (bold ignored when the body is bold, bullets never headings). Never drops short text (the old regex split on "2.1A" and dropped pieces ≤80 chars).
- `rfp_tables.py` pure: `column_roles()` (sno, description, quantity, quantity_part, unit, category, specification, make, remarks, price, other), `classify_table()` boq|spec|kv|other, `boq_requirements()` (category rows as context, totals skipped, per-location sums, total-vs-parts note), `table_markdown()`, `row_record()`, `parse_quantity()`.
- RFP extraction: boq rows → requirements by rules (no LLM; ≤4 `_map_columns` LLM calls for unrecognised numeric tables); everything else → LLM windows via `rfp_rag.document_segments()` with item tables as a placeholder; LLM items whose words ⊆ a table item are dropped, and so are scope-summary lines (`_is_scope_summary`: "Supply …", no qty/digits/compliance words; Llama 4 Scout writes them despite the prompt, seen in live runs). Warnings for non-piece units (Box, Mtr) and total mismatches.
- RFP retrieval: chunks = section text (labelled `[Section | p.N]`) + table slices + one row chunk per table row; hybrid BM25 + bge-small-en-v1.5 (shared with BU; BGE query instruction) fused by RRF, reranked by bge-reranker-base (lazy-loaded; patch `rfp_rag.get_reranker` in tests). Short RFPs are answered from `full_text` (tables as Markdown).
- Review/matching keep `unit`, `origin`, `source`, `attributes`, `quantity_breakdown` by id (`workflow._CARRIED_REQUIREMENT_FIELDS`, `matching_agent._CARRIED_FIELDS`); UI shows "p.2 · Table T2, row 13" (`sourceLabel()` in `frontend/src/lib/format.ts`).
- `tests/sample_rfps.py` three table-heavy fictitious RFPs + ground truth (`python -m tests.sample_rfps` → `data/samples/rfp_*.pdf`); `tests/run_rfp_table_eval.py` scores parsing (vs the old chunker) and retrieval, no LLM.
- `bu_agent.py` / `bu_rag.py` three-index catalogue retrieval (small chunks, large chunks, JSON) + cross-encoder
- `matching_agent.py` one batched LLM call to tag SKU vs compliance, then one RAG+LLM call per requirement; weighted score
- `pricing_agent.py` filter/atomise requirements, pull price/qty, apply margin, volume tier, GST; `reprice_with_margin()` reprices without re-retrieval
- `report_agent.py` thin node: `proposal_writer.draft_proposal()` then `proposal_pdf.render_proposal()` (used by master_agent's "report" route; the workflow calls the halves separately)
- `proposal_writer.py` LLM drafts sections (`SECTIONS`: company + pricing intro are deterministic, the rest AI) via `chat_json(provider="gemini")` when GEMINI_API_KEY is set, else OpenRouter; told never to write amounts and to claim only reviewer-confirmed compliance; `regenerate_section()`; `amount_warnings()` flags ₹/Rs/INR/lakh/crore in text
- `proposal_pdf.py` deterministic reportlab render: header, facts table, sections, pricing table + totals copied from `pricing_summary` (no arithmetic), not-quoted list, signature. Escapes text (XML). Helvetica lacks ₹ → "Rs." + Indian grouping (`inr()`).
- `company_profile.py` vendor details ("Prepared by"), stored in `app_settings`; Settings page; projects snapshot it in `proposal_meta.profile` at draft time
- Workflow tail: price → ⏸ pricing_review → draft → ⏸ proposal_review → render. `ProjectWorkflow.save_proposal_draft()` saves edits while paused: `update_state(as_node="draft")` + `invoke(None)` to re-pause (a plain `update_state` drops the pending interrupt). Never resume with an empty dict (LangGraph treats it as no resume); decisions are always `model_dump()`ed.
- `quote_export.py` CSV export; `settings.py` all config incl. `PricingRules`; `llm.py` OpenRouter wrapper
- `db.py` SQLAlchemy models + engine (SQLite at `var/synapse.db`, override `SYNAPSE_DB_PATH`); `catalogue.py` loads the catalogue JSON into it (auto-reload on file change), parses prices, generates seeded demo stock, lookups by id/SAP. SAP codes are NOT unique (3 duplicates); use `product_id`.
- `models.py` shared, thread-safe embedder/reranker instances (load once); `embedding_cache.py` disk cache for document embeddings (override dir with `SYNAPSE_CACHE_DIR`)
- `llm_json.py` `chat_json()` = LLM call + JSON extraction + pydantic validation + one retry; raises `LLMOutputError`. In tests, patch `llm_json.chat_completion`. It appends `JSON_ONLY` to every system prompt: without it Llama 4 Scout "thinks out loud" in Markdown on ambiguous picks and runs out of tokens (28/40 picks failed in the first live run; 0 after). OpenRouter JSON mode (`response_format`) returns 404 for this model; don't use it.
- `catalogue.ProductRecord.search_text()` adds an "Also described as" line with abbreviations expanded (Ind.→Indicator, Reg.→Regulator, SP/DP, MCB); the reranker scored "Switches with Indicator" 0.03 against "with Ind." and 0.99 expanded.
- Matching flags items that resolve to the same product: the highest-scoring keeps it, others get `duplicate_of` and status needs_review.
- `product_matcher.py` catalogue candidates with reranker scores (calibrated: real products 0.55-0.99, unrelated 0.00; thresholds `MATCH_*` in settings), LLM pick by id (validated), atomise compound requirements.
- Matching: classify → atomise → per-item match in parallel. Each matched item has id (R1, R1.2…), type product|compliance, status matched|needs_review|gap, product_id, match_score, confidence, alternatives, demo stock. Compliance items are not auto-assessed (status needs_review). `fulfillment_score` covers product items only.
- RFP extraction reads the whole PDF in `RFP_EXTRACT_WINDOW_CHARS` windows (parallel), dedupes, assigns ids R1…; per-window failures go to state `warnings`.
- Statuses after review: product items approved | rejected (plus matched | needs_review | gap from matching); compliance items met | not_met | needs_review. `matching_agent.apply_product()` attaches a product (used by matching and by the reviewer's "change"). Reviewer quantities set `qty_reviewed` → `qty_source: "reviewer"`.
- Pricing makes NO LLM calls: it prices `matched_items[*].product_id` from the DB. Quantity has a `qty_source` (rfp | requirement | query | default).
- `api/` FastAPI backend (`python -m api`, 127.0.0.1:8000, docs at /docs). `main.create_app()` takes overrides (db_path, workflow, var_dir, run_inline, warm_up) for tests. `context.py`: AppContext, cookie-session login dependency, `JobRunner` (one background worker; marks project running + progress, then `sync_project()` copies workflow status into the `projects` row). Routes: auth, rfps (upload dedupes by sha256, rejects non-PDF / text-less scans), projects (start, status+screen data, review → `validate_decision` synchronously then resume in background, retry, pricing preview via `reprice_with_margin`, quote.csv, proposal.pdf, outcome), catalogue search. Passwords: stdlib scrypt; only SHA-256 of session tokens stored.
- `frontend/` React 18 + TypeScript + Vite 6 + Tailwind 3 + React Query 5 + lucide icons (no component library; small primitives in `src/components/ui.tsx`). `npm run build` → `frontend/dist`, which FastAPI serves at / with SPA fallback (`api/main._serve_frontend`). Dev: `npm run dev` on :5173 proxies /api to :8000. Screens: Login/register, Home (dashboard + bell reminders), New Session (upload), RFPs/Documents (selection), Projects (stage-filtered lists), History (finished proposals only, outcome filters), ProjectPage → RequirementsReview / MatchingReview / PricingReview / Success, plus working/failed states. Pages poll `GET /api/projects/{id}` while `running`. Pricing preview sends only quantities the reviewer changed (sent ones become qty_source "reviewer"). "Accept all" skips items flagged `duplicate_of`. Types in `src/api/types.ts` mirror the API by hand.
- `db.py` tables also: users, sessions, rfps, projects (summary columns; full state is in the LangGraph checkpoint, thread_id = project id). SQLite returns naive datetimes; treat them as UTC. `projects.completed_at` is set by `sync_project` when a project first reaches done, `outcome_at` by the outcome endpoint. New nullable columns reach existing databases via `db._add_missing_columns()` (create_all never alters tables); non-nullable columns need a real migration.
- `api/routes_dashboard.py` `GET /api/dashboard`: `build_dashboard(projects, rfps, today)` is a pure function over the DB rows (no LLM, no checkpoints): KPIs with this/last-month counts, reminders derived from deadlines + project stage (newest project per RFP counts), all deadlines for the calendar, six-month generated-vs-won (bucketed by `completed_at`), pending RFPs. Team-wide, not per user.
- `start.py` (+ `start.bat`, `start.sh`, which only create `venv/`): one-command start. Pip installs when the `requirements.txt` hash changes (stamp in the venv), creates `.env` from the example, rebuilds `frontend/dist` when the UI sources' hash changes (`dist/.source-hash`), runs `python -m api`, waits for `/api/health`, opens the browser. `docs/DEMO.md` is the presenter walkthrough.
- `tests/` unit tests (offline; `tests/__init__.py` blanks API keys) and `run_*.py` manual runners
- `data/catalogue/` Havells catalogue (`havells.json`, small/large chunk files); `data/samples/sample_rfp.pdf` the one real sample RFP (bullets, no tables); `data/samples/rfp_*.pdf` generated table samples
- `var/` (gitignored) runtime output: database, uploads, index cache, generated proposals
- `legacy/gem_scraper/` teammate's original standalone scraper, kept for reference only
- `.env.example` template; real keys go in `.env` (gitignored), loaded by `settings.py` via python-dotenv

## Known issues found in review (fix early)
1. ~~README out of date (Groq, wrong volume tiers).~~ Fixed in M0. Volume tiers in code: 1-199 0%, 200-499 5%, 500-999 8%, 1000+ 12%.
2. ~~Singleton RAG tools.~~ Fixed in M2: `get_rfp_rag_tool(config)` keeps one tool per PDF content (sha256, LRU of 8); `get_bu_rag_tool(config)` one per catalogue. Pick the RFP with state `rfp_pdf_path` (CLI: `--rfp`).
3. ~~No `.env` loading.~~ Fixed in M0: `settings.py` loads `.env`; no keys in code.
4. ~~`requirements.txt` unpinned.~~ Fixed in M0 (pinned, Python 3.12).
5. ~~`.gitignore` missing `.venv/`.~~ Fixed in M0.
6. ~~Embeddings rebuilt on every start.~~ Fixed in M2: `embedding_cache.py` stores document embeddings in `var/cache/embeddings/` keyed by model + exact texts; models are loaded once and shared (`models.py`). Remaining start-up (~15 s) is loading PyTorch and the models, paid once per process.
7. ~~Matching sequential.~~ Fixed: per-item matching runs in a thread pool (`LLM_MAX_CONCURRENCY`, default 4).
8. ~~Regex JSON parsing.~~ Fixed for extraction, classification, atomising and product picks: `llm_json.chat_json()` validates against pydantic schemas and retries once. (Router and synthesis return free text and don't need it.)
9. ~~No pause points.~~ Fixed: `workflow.py` (separate graph from master_agent's Q&A graph) pauses with `interrupt()` at requirements_review, matching_review, pricing_review; SqliteSaver at `var/checkpoints.db` (`SYNAPSE_CHECKPOINT_DB_PATH`). `ProjectWorkflow.start/resume/retry/status`; decisions are pydantic models (`DECISION_SCHEMAS`) and are dry-run applied before resuming so bad input never advances the graph. Review nodes must have no side effects before `interrupt()` (they re-run on resume).
10. ~~`os.startfile`.~~ Fixed: `_open_pdf` (CLI only) uses `webbrowser` + `Path.as_uri()`.
11. ~~Agents print.~~ Fixed: library modules only `logger.info` (report boxes built as one string, one record); entry points call `log_setup.setup_logging()`, which also switches stdout/stderr to UTF-8 (piped output on Windows was cp1252 and crashed on ₹ / box characters). Don't add `print` to library code.

## Discovery agent
`discovery/` package (done in step 5). Sources behind one interface (`base.DiscoverySource`, `DiscoveryQuery`, `RfpCandidate`):
- `samples.SampleSource` (default): generates sample RFP PDFs from the catalogue for the search (keywords → product themes; unknown keywords like "cable" become off-catalogue gap lines). Fictional buyers, stamped "SAMPLE RFP … fictitious" on every page, explicit quantities, deadlines ≥5 days ahead, synthetic budget. Deterministic (same search → byte-identical PDFs, `invariant=True`), so repeated searches dedupe by sha256.
- Upload: `metadata.extract_metadata()` = rule parser for GeM docs, else one `chat_json` call (title, buyer, deadline, location, category, budget); never raises.
- `gem.GemSource`: ported Selenium scraper, refuses unless `DISCOVERY_GEM_ENABLED`; selenium is not a dependency; live path never run. `parse_gem_text()` tested on the 32 archived tenders.
- `discovery_agent.discovery_node` (LangGraph node; state `discovery_source`, `discovery_query`, `discovered_rfps`), `run_discovery()`, `list_sources()`. API: `GET /api/discovery/sources`, `POST /api/discovery/search` → registers RFPs via `routes_rfps.register_rfp()`.
Legacy notes (original code in `legacy/gem_scraper/`, reference only):
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
start.bat                                        # everything: setup, build, server, browser
python -m venv venv && venv\Scripts\activate   # Windows
pip install -r requirements.txt
copy .env.example .env                           # then add your keys
python master_agent.py "What are the eligibility criteria in the RFP?"
python -m unittest discover -s tests -t . -v     # offline, no API calls
python workflow.py start --rfp data/samples/sample_rfp.pdf --id demo   # live review flow (needs .env)
```
