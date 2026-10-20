# SYNAPSE demo walkthrough

A 10–15 minute live demo of the full flow: find a tender, review each agent's
work, price it, generate the proposal, record the outcome. Each AI step takes
about a minute or two and shows progress, so talk over it (suggested lines below).

## Before the demo (the day before)

1. `git pull`, then run `start.bat` once so packages and the dashboard are up to date.
2. Check `.env` has `OPENROUTER_API_KEY` and `GEMINI_API_KEY`, and that the OpenRouter account has credit.
3. In the dashboard: create the presenter's account, then **Settings** → enter your company's
   details (name, address, GSTIN, signatory). This is the "Prepared by" on the proposal.
4. **Do one full rehearsal run** and keep it: if the internet or the API fails during the
   demo, you can open this finished project instead (see "If something goes wrong").
5. On the day: start with `start.bat` **10 minutes early** (the AI models load at start-up)
   and leave the window open.

## The demo

| # | Do | Say / point out |
|---|----|-----------------|
| 1 | Open **Home** | The dashboard is computed from real project data: proposals generated, reviews waiting, deadlines, win rate. The bell shows reminders such as "deadline today". |
| 2 | **New Session → Find RFPs**. Keywords: `modular switches`, `sockets`, `cable`. Location: `Gurugram`. **Find RFPs** | The Discovery Agent has swappable sources. Live GeM scraping is built but off, because the portal's terms may not allow scraping; for the demo it generates realistic sample tenders (stamped as samples). "cable" is deliberately something our catalogue doesn't sell. |
| 3 | Pick one RFP (show **Closing soon** badge, budget, buyer) → **Continue with selected RFP** | One RFP at a time, the way a bid team works. Buyer, deadline and budget were read from the document. |
| 4 | Wait on **Reading the RFP** | The RFP Agent reads the whole PDF in sections in parallel and extracts every requirement with quantity and criticality (mandatory / high / medium / low). |
| 5 | **Requirements review**: edit one requirement's text or quantity, **Remove** one, **Continue** | First human checkpoint: nothing moves on without approval. Every change goes into a review log with the reviewer's name. |
| 6 | **Product matching**: show a **matched** card (score, confidence, alternatives, demo stock), a **gap** (the cable), then **Change product** on one item using the catalogue search | Matching = semantic search (BGE embeddings + FAISS) and a reranker score, then the LLM picks a product **by id from the shortlist only**, so it cannot invent products. Compliance items (GST, ISI certificates) are listed separately for a person to confirm. Gaps are honest: no product is forced. |
| 7 | **Accept all** → **Continue to Pricing** | Duplicates (two requirements that resolve to the same product) are flagged rather than double-quoted. |
| 8 | **Pricing**: move the margin slider; change one quantity; tick **advance payment**; add **freight** | All money maths is plain code with fixed rules: margin, volume discount tiers (5/8/12%), 18% GST, payment-terms discount, freight. The AI never calculates a price. The preview is instant; nothing is saved until you continue. Download **quote.csv** if asked. |
| 9 | **Generate Proposal** → wait → editor | Gemini drafts the text sections. It's told never to write amounts and to claim only compliance items a reviewer marked "Met". |
| 10 | Edit one section; **Regenerate** another with an instruction (e.g. "make it shorter"); watch the PDF preview update | Every number in the PDF is copied from the approved pricing. If anyone types a ₹ amount into the text, the editor warns. |
| 11 | **Complete Generation** → open the PDF | Company profile, buyer, line-item table with GST, totals, items not quoted, signature block. |
| 12 | **Completed / History** → set status to **Submitted**, then **Won** → back to **Home** | The win rate, chart and reminders update straight away. |

## If something goes wrong

- **An AI step fails** (network, credit): the project shows the error and a **Retry**
  button; nothing already approved is lost. Or open the rehearsal project from
  **Completed / History**.
- **The server window was closed**: run `start.bat` again; projects continue where they paused.
- **"No OpenRouter API key" banner**: the key is missing from `.env`; fix it and restart.

## Likely questions (honest answers)

- *Is the stock data real?* No, it is synthetic and labelled "demo" everywhere. The design
  takes real stock from an ERP export in the same table.
- *How accurate is matching?* On the sample RFP the agents extracted 34 requirements and
  matched all 28 product items with no gaps; on a generated tender the planted
  off-catalogue item was correctly reported as a gap. This is a handful of test runs,
  not a measured evaluation.
- *Why not let the LLM price?* Prices must be exact and auditable; LLMs make arithmetic
  errors. The LLM only chooses products from a shortlist; the code does the money.
- *Does it scrape GeM?* The adapter exists (ported from our scraper) but is off by default
  and was not run against the live portal.
- *Where is the data?* Only on this laptop (`var/`): SQLite database, uploaded PDFs,
  generated proposals. API keys stay in `.env`, which is never committed.
