const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, NumberFormat, LevelFormat,
  TableOfContents, PageBreak, UnderlineType,
} = require("docx");
const fs = require("fs");

// ─── COLOUR PALETTE ───────────────────────────────────────────────────────────
const NAVY = "1A3557";
const GOLD = "C8942A";
const LGREY = "F4F6F8";
const MGREY = "8A9BB0";
const BLACK = "1A1A1A";
const WHITE = "FFFFFF";

// ─── BORDER HELPERS ──────────────────────────────────────────────────────────
const thinBorder = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const cellBorders = { top: thinBorder, bottom: thinBorder, left: thinBorder, right: thinBorder };
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };

// ─── NUMBERING CONFIG ─────────────────────────────────────────────────────────
const numberingConfig = [
  {
    reference: "bullets",
    levels: [{
      level: 0, format: LevelFormat.BULLET, text: "\u2022",
      alignment: AlignmentType.LEFT,
      style: {
        paragraph: { indent: { left: 720, hanging: 360 } },
        run: { font: "Arial", size: 20 }
      },
    }],
  },
  {
    reference: "bullets2",
    levels: [{
      level: 0, format: LevelFormat.BULLET, text: "\u25E6",
      alignment: AlignmentType.LEFT,
      style: {
        paragraph: { indent: { left: 1080, hanging: 360 } },
        run: { font: "Arial", size: 20 }
      },
    }],
  },
  {
    reference: "numbers",
    levels: [{
      level: 0, format: LevelFormat.DECIMAL, text: "%1.",
      alignment: AlignmentType.LEFT,
      style: {
        paragraph: { indent: { left: 720, hanging: 360 } },
        run: { font: "Arial", size: 20 }
      },
    }],
  },
];

// ─── STYLE HELPERS ────────────────────────────────────────────────────────────
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, font: "Arial", bold: true, size: 32, color: NAVY })],
    spacing: { before: 360, after: 180 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: GOLD, space: 1 } },
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, font: "Arial", bold: true, size: 26, color: NAVY })],
    spacing: { before: 280, after: 120 },
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [new TextRun({ text, font: "Arial", bold: true, size: 22, color: "2C5F8A" })],
    spacing: { before: 200, after: 80 },
  });
}

function body(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({
      text,
      font: "Arial",
      size: 20,
      color: BLACK,
      bold: opts.bold || false,
      italics: opts.italic || false,
    })],
    spacing: { before: 60, after: 120 },
    alignment: opts.justify ? AlignmentType.JUSTIFIED : AlignmentType.LEFT,
  });
}

function para(runs, opts = {}) {
  return new Paragraph({
    children: runs,
    spacing: { before: 60, after: 120 },
    alignment: opts.justify ? AlignmentType.JUSTIFIED : AlignmentType.LEFT,
  });
}

function run(text, opts = {}) {
  return new TextRun({
    text,
    font: "Arial",
    size: 20,
    color: opts.color || BLACK,
    bold: opts.bold || false,
    italics: opts.italic || false,
    underline: opts.underline ? { type: UnderlineType.SINGLE } : undefined,
  });
}

function bullet(text, ref = "bullets") {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    children: [new TextRun({ text, font: "Arial", size: 20, color: BLACK })],
    spacing: { before: 40, after: 80 },
  });
}

function spacer(lines = 1) {
  return new Paragraph({ children: [new TextRun("")], spacing: { before: 0, after: lines * 120 } });
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: "Arial", size: 18, color: MGREY, italics: true })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 40, after: 160 },
  });
}

// ─── TABLE HELPERS ────────────────────────────────────────────────────────────
const TW = 9360; // total content width in DXA (US Letter, 1" margins)

function headerCell(text, widthDXA) {
  return new TableCell({
    borders: cellBorders,
    width: { size: widthDXA, type: WidthType.DXA },
    shading: { fill: NAVY, type: ShadingType.CLEAR },
    margins: { top: 100, bottom: 100, left: 140, right: 140 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      children: [new TextRun({ text, font: "Arial", size: 19, bold: true, color: WHITE })],
      alignment: AlignmentType.CENTER,
    })],
  });
}

function dataCell(text, widthDXA, opts = {}) {
  return new TableCell({
    borders: cellBorders,
    width: { size: widthDXA, type: WidthType.DXA },
    shading: { fill: opts.fill || WHITE, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      children: [new TextRun({
        text,
        font: "Arial",
        size: 19,
        color: opts.color || BLACK,
        bold: opts.bold || false,
      })],
      alignment: opts.align || AlignmentType.LEFT,
    })],
  });
}

function twoColTable(rows, w1 = 3000, w2 = 6360) {
  return new Table({
    width: { size: TW, type: WidthType.DXA },
    columnWidths: [w1, w2],
    rows: rows.map(([a, b], i) => new TableRow({
      children: [
        dataCell(a, w1, { fill: i === 0 ? LGREY : WHITE, bold: i === 0 }),
        dataCell(b, w2, { fill: i === 0 ? LGREY : WHITE }),
      ],
    })),
  });
}

// ─── DOCUMENT SECTIONS ───────────────────────────────────────────────────────

// COVER PAGE
const coverSection = [
  spacer(6),
  new Paragraph({
    children: [new TextRun({ text: "RFP Response Automation System", font: "Arial", size: 52, bold: true, color: NAVY })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
  }),
  new Paragraph({
    children: [new TextRun({ text: "A Multi-Agent AI Architecture for Automated Procurement Analysis,", font: "Arial", size: 24, color: MGREY, italics: true })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 60 },
  }),
  new Paragraph({
    children: [new TextRun({ text: "Product Matching, Pricing, and Proposal Generation", font: "Arial", size: 24, color: MGREY, italics: true })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 400 },
  }),
  new Paragraph({
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: GOLD, space: 1 } },
    children: [],
    spacing: { after: 400 },
  }),
  new Table({
    width: { size: 5400, type: WidthType.DXA },
    columnWidths: [2200, 3200],
    rows: [
      ["Project Type", "Academic / Industry Applied Project"],
      ["Domain", "Artificial Intelligence, NLP, Procurement Automation"],
      ["Architecture", "Multi-Agent System using LangGraph"],
      ["LLM Backend", "Groq API (Llama 4 Scout) + Google Gemini Flash"],
      ["RAG Stack", "FAISS + BGE Embeddings + Cross-Encoder Reranking"],
      ["Report Date", new Date().toLocaleDateString("en-IN", { day: "2-digit", month: "long", year: "numeric" })],
    ].map(([k, v]) => new TableRow({
      children: [
        dataCell(k, 2200, { fill: LGREY, bold: true }),
        dataCell(v, 3200),
      ],
    })),
  }),
  pageBreak(),
];

// ABSTRACT
const abstractSection = [
  h1("Abstract"),
  body(
    "This report presents the design, architecture, implementation, and evaluation of an end-to-end multi-agent AI system developed to automate the procurement response workflow for Request for Proposal (RFP) documents. The system addresses a practical industry problem: the manual, time-intensive process of analysing incoming RFPs, cross-referencing them against a vendor product catalog, calculating competitive pricing, and generating a formatted proposal response.",
    { justify: true }
  ),
  body(
    "The solution employs a LangGraph-orchestrated pipeline of five specialised AI agents — a Router, an RFP Agent, a Business Unit (BU) Agent, a Matching Agent, a Pricing Agent, and a Report Generation Agent — each with a well-defined responsibility. Retrieval-Augmented Generation (RAG) is used throughout to ground agent responses in verified catalog and document data, preventing hallucination and ensuring commercial accuracy.",
    { justify: true }
  ),
  body(
    "The system consumes an RFP PDF, a structured product catalog (JSON and text chunks), and user queries as inputs, and produces a fully formatted, submission-ready vendor proposal PDF as output. The architecture is modular, testable, and extensible — each agent can be upgraded, replaced, or re-routed independently without disrupting the rest of the pipeline.",
    { justify: true }
  ),
  spacer(),
];

// TABLE OF CONTENTS PLACEHOLDER
const tocSection = [
  h1("Table of Contents"),
  new TableOfContents("Table of Contents", {
    hyperlink: true,
    headingStyleRange: "1-3",
  }),
  pageBreak(),
];

// CHAPTER 1 — INTRODUCTION
const ch1 = [
  h1("1. Introduction"),

  h2("1.1 Background and Motivation"),
  body(
    "Procurement departments in large organisations routinely receive Requests for Proposal (RFPs) from clients or project owners. Responding to an RFP is a multi-step, labour-intensive process that requires a procurement team to: read and comprehend the RFP document in detail; identify each technical, commercial, and logistical requirement; search the internal product catalog for matching items; calculate competitive pricing incorporating margins, volume discounts, and applicable taxes; and produce a formally structured written proposal within a tight deadline.",
    { justify: true }
  ),
  body(
    "For a mid-to-large vendor handling multiple concurrent RFPs, this process may take several working days per response. Errors in product matching or pricing miscalculations directly affect bid competitiveness and profitability. Human review cycles further extend turnaround time.",
    { justify: true }
  ),
  body(
    "This project was conceived to address these inefficiencies by building an AI-powered system capable of executing the entire RFP response workflow autonomously, with human review required only at the final output stage.",
    { justify: true }
  ),

  h2("1.2 Problem Statement"),
  body("The core problem addressed by this project can be stated as follows:", { justify: true }),
  bullet("RFP documents are unstructured PDFs with variable formatting, making automated parsing difficult."),
  bullet("Product catalogs exist in multiple formats (structured JSON, categorised text, flat lists) and must be searched semantically, not just by keyword."),
  bullet("Matching RFP requirements to catalog items requires natural language understanding — requirements are written in procurement language, not in product specification language."),
  bullet("Pricing must apply commercial rules (margins, volume discounts, GST) deterministically, not probabilistically."),
  bullet("The final proposal must be professionally formatted and, in some cases, must follow a format explicitly specified inside the RFP itself."),
  spacer(),

  h2("1.3 Objectives"),
  bullet("Design and implement a modular multi-agent AI system that fully automates the RFP response workflow."),
  bullet("Use Retrieval-Augmented Generation (RAG) to ground all agent outputs in verified source data."),
  bullet("Implement deterministic pricing logic that correctly applies commercial rules without relying on LLM arithmetic."),
  bullet("Build a report generation agent that produces a submission-ready PDF, detecting and following any response format specified in the RFP."),
  bullet("Ensure the architecture is extensible — new agents, data sources, or routing paths can be added without restructuring the existing pipeline."),
  spacer(),

  h2("1.4 Scope"),
  body(
    "The project scope covers the full pipeline from raw RFP input to PDF proposal output. It includes data preparation (chunking, embedding, indexing), agent design, LangGraph graph construction, LLM integration with Groq and Google Gemini APIs, pricing rule implementation, and PDF rendering with ReportLab. The system is designed around the Havells electrical product catalog and a sample RFP for modular electrical switches as the primary demonstration domain, but the architecture is domain-agnostic.",
    { justify: true }
  ),
  body(
    "Out of scope: real-time inventory management, ERP integration, multi-vendor competitive analysis, and user authentication/access control.",
    { justify: true }
  ),
  pageBreak(),
];

// CHAPTER 2 — LITERATURE REVIEW / RELATED WORK
const ch2 = [
  h1("2. Literature Review and Related Work"),

  h2("2.1 Large Language Models in Document Processing"),
  body(
    "Large Language Models (LLMs) such as GPT-4, LLaMA, and Gemini have demonstrated strong performance on document comprehension tasks including summarisation, question answering, and information extraction. However, when applied to closed-domain enterprise tasks, LLMs face a fundamental limitation: their training data does not include proprietary product catalogs, internal pricing rules, or organisation-specific documents. Direct prompting of an LLM with domain-specific questions produces plausible but often inaccurate responses — a phenomenon commonly referred to as hallucination.",
    { justify: true }
  ),
  body(
    "Retrieval-Augmented Generation (RAG), introduced by Lewis et al. (2020), addresses this limitation by providing the LLM with relevant retrieved context at inference time. Instead of relying on parametric memory, the model conditions its generation on retrieved document chunks, significantly reducing hallucination in closed-domain settings. RAG has since become the standard approach for enterprise document QA applications.",
    { justify: true }
  ),

  h2("2.2 Multi-Agent Systems and LangGraph"),
  body(
    "Complex enterprise workflows cannot be handled by a single LLM call. Multi-agent architectures decompose a task into sub-tasks handled by specialised agents. Each agent has a narrowly defined responsibility, uses appropriate tools, and passes structured outputs to the next agent in the pipeline.",
    { justify: true }
  ),
  body(
    "LangGraph, developed by LangChain, provides a graph-based orchestration framework for multi-agent systems built on LLMs. It models the agent pipeline as a directed state graph where each node is an agent function, edges define transitions, and a shared state object (TypedDict) carries data between nodes. Conditional edges enable dynamic routing based on runtime state, making LangGraph well-suited to the variable-depth pipeline required for RFP processing.",
    { justify: true }
  ),

  h2("2.3 Vector Search and Reranking"),
  body(
    "Dense vector search using transformer-based embedding models such as BGE (BAAI General Embedding) has replaced traditional BM25 keyword search in modern RAG pipelines. A query and a set of document chunks are embedded into a shared vector space; the nearest neighbours to the query embedding are retrieved as candidate context.",
    { justify: true }
  ),
  body(
    "Cross-encoder reranking (Nogueira & Cho, 2019) provides a second-stage refinement: a reranker model scores each (query, candidate) pair jointly, producing a more accurate relevance ranking than dot-product similarity alone. This two-stage pipeline — fast bi-encoder retrieval followed by accurate cross-encoder reranking — is the current best practice for production RAG systems.",
    { justify: true }
  ),

  h2("2.4 Procurement Automation"),
  body(
    "Existing commercial procurement automation tools (e.g., Coupa, SAP Ariba) focus on purchase-order management, supplier onboarding, and spend analytics. They do not address the intelligent document understanding required for RFP response generation. The academic literature on procurement AI is sparse; most work focuses on contract analysis rather than response generation. This project contributes a practical end-to-end system for the under-studied RFP response automation problem.",
    { justify: true }
  ),
  pageBreak(),
];

// CHAPTER 3 — SYSTEM ARCHITECTURE
const ch3 = [
  h1("3. System Architecture"),

  h2("3.1 High-Level Overview"),
  body(
    "The system is structured as a directed acyclic graph of six specialised agents, orchestrated by a LangGraph StateGraph. A single shared AgentState TypedDict flows through the graph; each agent reads from and writes to this state. No agent communicates directly with another — all inter-agent communication is mediated through the shared state object. This design ensures that agents are independently testable and replaceable.",
    { justify: true }
  ),
  body("The six pipeline stages and their routing logic are as follows:"),
  spacer(),

  new Table({
    width: { size: TW, type: WidthType.DXA },
    columnWidths: [1400, 1600, 1800, 4560],
    rows: [
      new TableRow({
        children: [
          headerCell("Route Label", 1400),
          headerCell("Triggered By", 1600),
          headerCell("Agents Executed", 1800),
          headerCell("Description", 4560),
        ]
      }),
      ...([
        ["rfp", "RFP questions", "RFP Agent", "Answers questions directly from the RFP document."],
        ["bu", "Catalog questions", "BU Agent", "Answers product and pricing questions from the catalog."],
        ["match", "Fulfillment queries", "RFP + Matching", "Checks which RFP requirements the catalog can fulfil."],
        ["price", "Pricing queries", "RFP + Matching + Pricing", "Adds unit pricing and line-item calculations."],
        ["full", "General queries", "RFP + Matching + Pricing", "Complete pipeline without proposal document generation."],
        ["report", "Proposal generation", "All Agents + Report", "Full pipeline culminating in a formatted PDF proposal."],
      ].map(([r, t, a, d], i) => new TableRow({
        children: [
          dataCell(r, 1400, { fill: i % 2 === 0 ? LGREY : WHITE, bold: true }),
          dataCell(t, 1600, { fill: i % 2 === 0 ? LGREY : WHITE }),
          dataCell(a, 1800, { fill: i % 2 === 0 ? LGREY : WHITE }),
          dataCell(d, 4560, { fill: i % 2 === 0 ? LGREY : WHITE }),
        ],
      }))),
    ],
  }),
  caption("Table 3.1 — Pipeline routing map by query intent"),
  spacer(),

  h2("3.2 Shared State Design"),
  body(
    "The AgentState TypedDict is the central data contract of the system. Every field is typed and optional (total=False), meaning nodes that run early in the pipeline do not need to pre-populate fields that are produced later. This prevents KeyError failures in conditional routing paths where some fields may never be set.",
    { justify: true }
  ),
  body("The state is organised into logical groups corresponding to each agent's responsibility:"),
  bullet("Input fields: user_query"),
  bullet("RFP Agent outputs: rfp_answer, rfp_requirements"),
  bullet("BU Agent outputs: bu_answer"),
  bullet("Matching Agent outputs: matched_items, fulfillment_score, fulfillment_report"),
  bullet("Pricing Agent outputs: pricing_report, quoted_items, line_items, pricing_summary, quote_payload"),
  bullet("Report Agent outputs: report_pdf_path, report_text"),
  bullet("Control fields: route, error, final_response"),
  spacer(),

  h2("3.3 Data Flow Diagram"),
  body("The following describes the data flow for the 'report' route, which is the most complete execution path:", { justify: true }),
  spacer(),
  new Table({
    width: { size: TW, type: WidthType.DXA },
    columnWidths: [TW],
    rows: [
      new TableRow({
        children: [
          new TableCell({
            borders: cellBorders,
            width: { size: TW, type: WidthType.DXA },
            shading: { fill: LGREY, type: ShadingType.CLEAR },
            margins: { top: 140, bottom: 140, left: 200, right: 200 },
            children: [
              new Paragraph({
                children: [
                  new TextRun({
                    text: "User Query",
                    font: "Courier New",
                    size: 19,
                    bold: true,
                    color: NAVY
                  })
                ],
                spacing: { after: 60 }
              }),

              // all your other Paragraph blocks here...

              new Paragraph({
                children: [
                  new TextRun({
                    text: "END",
                    font: "Courier New",
                    size: 19,
                    bold: true,
                    color: NAVY
                  })
                ],
                spacing: { after: 0 }
              }),
            ],
          }),
        ],
      }),
    ],
  }),
  caption("Figure 3.1 — Data flow for the 'report' pipeline route"),
  pageBreak(),
];

// CHAPTER 4 — COMPONENT DETAILS
const ch4 = [
  h1("4. Component Design and Implementation"),

  h2("4.1 Settings and Configuration (settings.py)"),
  body(
    "All system-wide configuration is centralised in settings.py. This includes API keys, file paths, model names, retrieval parameters, and pricing rules. Centralising configuration serves two purposes: it prevents hard-coded values from being scattered across agent files, and it provides a single location for environment-specific changes without touching agent logic.",
    { justify: true }
  ),
  body("Key configuration groups:"),
  bullet("LLM configuration: GROQ_API_KEY, LLM_MODEL (meta-llama/llama-4-scout-17b-16e-instruct), GEMINI_API_KEY"),
  bullet("Embedding models: BU_EMBED_MODEL (BAAI/bge-small-en-v1.5), BU_RERANK_MODEL (BAAI/bge-reranker-base), RFP_EMBED_MODEL (BAAI/bge-small-en)"),
  bullet("Data paths: BU_SMALL_CHUNKS_PATH, BU_LARGE_CHUNKS_PATH, BU_JSON_PATH, RFP_PDF_PATH"),
  bullet("Retrieval knobs: top-k values for each index and reranking"),
  bullet("PricingRules dataclass: base margin, min/max margin bounds, volume discount tiers, GST rate, currency, rounding precision"),
  spacer(),
  body(
    "The PricingRules dataclass is particularly important: by encapsulating all commercial parameters in a typed dataclass, the pricing logic in pricing_agent.py can be kept deterministic and testable without embedding magic numbers in the agent code.",
    { justify: true }
  ),

  h2("4.2 Shared State (state.py)"),
  body(
    "The AgentState TypedDict defines the complete contract between all agents. Using TypedDict rather than a plain dict provides static type checking compatibility with tools like mypy, makes agent inputs and outputs explicit and self-documenting, and ensures that LangGraph's graph compiler can validate the state schema at build time.",
    { justify: true }
  ),
  body(
    "All fields are declared with total=False, making every field optional. This is a deliberate design choice: in the 'rfp' route, for example, pricing fields will never be populated, and an agent should not fail because an optional upstream field was not written. Any agent that reads a field it might not find uses state.get('field_name', default) rather than state['field_name'].",
    { justify: true }
  ),

  h2("4.3 LLM Wrapper (llm.py)"),
  body(
    "The llm.py module provides a thin abstraction over the Groq Python SDK. A single chat_completion() function accepts keyword arguments for system prompt, user message, model, temperature, and max_tokens. All agents call this function rather than instantiating Groq clients directly.",
    { justify: true }
  ),
  body("This design provides three benefits:"),
  bullet("API key resolution logic is centralised: environment variable takes priority over settings.py, with a clear error message if neither is set."),
  bullet("Any future LLM provider change (e.g., switching from Groq to OpenAI) requires a change in only one file."),
  bullet("Temperature and token limits are explicit per-call parameters, giving each agent precise control over generation behaviour."),
  spacer(),

  h2("4.4 RFP RAG Tool (rfp_rag.py)"),
  body(
    "The RFP RAG tool handles the extraction and indexing of RFP PDF documents. On initialisation, it uses PyMuPDF (fitz) to extract raw text from all pages. The text is then chunked using a regex-based section splitter that identifies numbered section headings (e.g., '1. Introduction', '2.1 Scope'). Chunks that exceed the configured maximum length are further split at sentence boundaries to prevent context windows from being dominated by a single overly long chunk.",
    { justify: true }
  ),
  body(
    "Chunks are embedded using the BGE-small-en sentence transformer model. A FAISS IndexFlatL2 index is built over the embeddings, enabling fast approximate nearest-neighbour search at query time. The module uses a module-level singleton pattern so that the PDF is loaded and indexed exactly once per process, regardless of how many agents query it.",
    { justify: true }
  ),
  body("The public API exposes two methods:"),
  bullet("retrieve(query, top_k): returns the top-k most relevant chunks for a query."),
  bullet("run_query(query): retrieves context, builds a prompt, and calls the LLM to produce a grounded answer."),
  spacer(),

  h2("4.5 Business Unit RAG Tool (bu_rag.py)"),
  body(
    "The BU RAG tool is more sophisticated than the RFP tool because it must handle three complementary data sources for the same product catalog, each offering different retrieval characteristics.",
    { justify: true }
  ),

  h3("4.5.1 Three-Index Hybrid Search"),
  body("The three indexes serve different retrieval purposes:"),
  bullet("Small chunks index: One chunk per product color group or category. Provides fine-grained retrieval — ideal for queries about a specific product variant."),
  bullet("Large chunks index: One chunk per full product range (e.g., all Coral Ebony products together). Provides broader context — ideal for range-level queries."),
  bullet("JSON structured index: Each product row from the structured JSON catalog is converted to a standardised text representation and embedded. Provides exact field-level retrieval — ideal for SAP codes, HSN codes, packing details, and prices."),
  spacer(),
  body(
    "At query time, all three indexes are searched independently, producing separate candidate lists. These three lists are concatenated and passed to a Cross-Encoder reranker (bge-reranker-base) which scores each (query, candidate) pair jointly and returns the top-k most relevant chunks across all three sources.",
    { justify: true }
  ),

  h3("4.5.2 JSON Flattening"),
  body(
    "The structured product JSON contains nested product data: ranges contain colors or categories, which contain individual product items. The _flatten_json_products() method recursively unwraps this hierarchy into a flat list of product dicts, each carrying inherited fields (brand, range, variant) alongside item-level fields (product name, price, SAP code, HSN code, packing, monthly availability). This flat representation is then converted to a standardised text block for embedding.",
    { justify: true }
  ),

  h3("4.5.3 Embedding Strategy"),
  body(
    "The BGE model family uses asymmetric embedding: queries are prefixed with 'query: ' and documents with 'passage: '. This asymmetry is intentional — the model was trained to align query representations with passage representations rather than with other queries. All embed_query() and embed_docs() calls in the tool correctly apply these prefixes.",
    { justify: true }
  ),

  h2("4.6 Router Node (master_agent.py)"),
  body(
    "The router is the entry point of the LangGraph graph. It receives the raw user query and classifies it into one of six route labels using a zero-temperature LLM call with a carefully structured classification prompt. The prompt lists each label with clear, non-overlapping definitions to minimise ambiguity.",
    { justify: true }
  ),
  body(
    "The output of the LLM is normalised to lowercase, split on whitespace to extract only the first token (guarding against the model producing an explanation despite instructions), and validated against the set of known routes. Any unrecognised output defaults to 'full', ensuring the pipeline never enters an undefined state.",
    { justify: true }
  ),

  h2("4.7 RFP Agent (rfp_agent.py)"),
  body(
    "The RFP Agent has two responsibilities executed in a single node function. First, it answers the user's query directly from the RFP document using the RFP RAG tool. Second, it extracts a structured list of RFP requirements for consumption by the Matching Agent.",
    { justify: true }
  ),
  body(
    "Requirement extraction uses a dedicated extraction prompt that instructs the LLM to return a JSON array. Each requirement object contains a requirement string, a criticality classification (mandatory, high, medium, low), and a numeric weight (4, 3, 2, 1 respectively). The criticality and weight fields allow the Matching Agent to compute a weighted fulfillment score rather than a simple count-based percentage.",
    { justify: true }
  ),
  body(
    "The extraction function applies multiple layers of JSON parsing robustness: it first tries to strip markdown fences using a regex, then falls back to extracting any JSON array from the text, then attempts json.loads(), and finally falls back to line-by-line text splitting if all parsing attempts fail. This defensive approach ensures the agent produces usable output even when the LLM does not precisely follow the JSON format instruction.",
    { justify: true }
  ),

  h2("4.8 BU Agent (bu_agent.py)"),
  body(
    "The BU Agent is a thin wrapper around the BU RAG tool. It reads the user query from state and calls the tool's run_query() method, writing the result to state['bu_answer']. The agent is deliberately kept simple — all retrieval and generation logic lives in the tool, not the agent. This separation means the tool can be tested in isolation without standing up the full LangGraph graph.",
    { justify: true }
  ),

  h2("4.9 Matching Agent (matching_agent.py)"),
  body(
    "The Matching Agent iterates over the structured requirements list extracted by the RFP Agent and, for each requirement, queries the BU RAG tool to determine whether the business can fulfil it. The assessment prompt instructs the LLM to reply with 'YES' or 'NO' on the first line, followed by one sentence of evidence. Temperature is set to 0.0 for maximum determinism.",
    { justify: true }
  ),
  body(
    "After all requirements are assessed, the agent computes a weighted fulfillment score: the sum of weights for matched requirements divided by the total weight of all requirements, expressed as a percentage. It then produces a human-readable fulfillment report listing strengths (matched requirements) and gaps (unmatched requirements) in separate sections.",
    { justify: true }
  ),
  body(
    "The requirement parsing function (_parse_requirements) handles multiple input formats because the RFP Agent may produce structured JSON dicts or plain text depending on LLM behaviour. The function normalises all formats into the canonical [{'requirement': str, 'weight': float}] list structure.",
    { justify: true }
  ),

  h2("4.10 Pricing Agent (pricing_agent.py)"),
  body(
    "The Pricing Agent is the most computationally deterministic component of the system. It operates exclusively on requirements that were marked as matched by the Matching Agent, avoiding the generation of prices for products the vendor cannot supply.",
    { justify: true }
  ),

  h3("4.10.1 Price Extraction"),
  body(
    "For each matched requirement, the agent queries the BU RAG tool with a structured price query. The returned text is passed to a regex-based price extraction function that handles multiple price representation formats commonly found in Indian electrical product catalogs: Rs./INR prefix, rupee symbol, 'Price:' label, and the '60/-' suffix format. If no price can be extracted, the item is marked with a note rather than causing the agent to fail.",
    { justify: true }
  ),

  h3("4.10.2 Pricing Rule Application"),
  body("Pricing is calculated in a deterministic six-step sequence:"),
  new Table({
    width: { size: TW, type: WidthType.DXA },
    columnWidths: [500, 2500, 6360],
    rows: [
      new TableRow({ children: [headerCell("Step", 500), headerCell("Operation", 2500), headerCell("Detail", 6360)] }),
      ...([
        ["1", "Margin application", "Selling price = base_cost \u00D7 (1 + margin%)"],
        ["2", "Volume discount lookup", "Discount tier selected from PricingRules.volume_discount_tiers based on qty"],
        ["3", "Net price calculation", "Net = selling_price \u00D7 (1 \u2212 discount%)"],
        ["4", "GST calculation", "GST = net \u00D7 gst_rate_pct / 100 (default 18%)"],
        ["5", "Final unit price", "final_unit_price = net + GST"],
        ["6", "Line total", "line_total = final_unit_price \u00D7 qty"],
      ].map(([s, o, d], i) => new TableRow({
        children: [
          dataCell(s, 500, { fill: i % 2 === 0 ? LGREY : WHITE, align: AlignmentType.CENTER }),
          dataCell(o, 2500, { fill: i % 2 === 0 ? LGREY : WHITE, bold: true }),
          dataCell(d, 6360, { fill: i % 2 === 0 ? LGREY : WHITE }),
        ]
      }))),
    ],
  }),
  caption("Table 4.1 — Pricing calculation sequence"),
  spacer(),
  body(
    "The margin is clamped to [min_margin_pct, max_margin_pct] from PricingRules before application, preventing out-of-bounds margin requests from interactive clients. All monetary values are rounded to the configured precision (default: 2 decimal places) at each step to prevent floating-point accumulation errors.",
    { justify: true }
  ),

  h3("4.10.3 Repricing Without Retrieval"),
  body(
    "The reprice_with_margin() function accepts a list of already-computed line items and a new margin percentage. It recomputes the full pricing calculation from the stored base_unit_price without re-querying the RAG tool. This is the function called by interactive clients (e.g., a web frontend) when a user adjusts the margin slider — it avoids re-running expensive retrieval and LLM calls for a purely arithmetic operation.",
    { justify: true }
  ),

  h2("4.11 Synthesise Node (master_agent.py)"),
  body(
    "The Synthesise node is the final step before END for all routes. It collects all non-empty output fields from the state and, if more than one agent has produced output, calls the LLM to merge them into a single professional response. If only one agent has produced output, it returns that output directly without an additional LLM call, saving tokens.",
    { justify: true }
  ),

  h2("4.12 Report Generation Agent (report_agent.py)"),
  body(
    "The Report Agent is the most complex component and is intentionally executed using a different LLM provider (Google Gemini Flash) than the rest of the pipeline. This decision is motivated by two factors: the volume of context that must be synthesised at this stage (RFP analysis, fulfillment report, full pricing, catalog highlights) exceeds practical token budgets for rapid Groq inference; and Gemini 1.5 Flash provides a 1 million token context window with strong instruction-following at low cost.",
    { justify: true }
  ),

  h3("4.12.1 RFP Format Detection"),
  body(
    "Before generating the proposal, the agent scans the accumulated rfp_answer and rfp_requirements fields for explicit submission format instructions. Keywords such as 'submission format', 'response format', 'proposal must include', and 'section order' trigger extraction of the surrounding sentences, which are injected into the generation prompt as a mandatory format constraint. If no format is detected, the agent defaults to a standard nine-section proposal structure.",
    { justify: true }
  ),

  h3("4.12.2 Proposal Generation"),
  body(
    "The generation prompt packages all state data — RFP analysis, catalog highlights, fulfillment report, pricing summary, and full pricing breakdown — into a structured context block and instructs Gemini to produce a complete vendor proposal. The prompt requests SECTION:/END_SECTION delimiters to allow deterministic parsing of the generated output into distinct sections.",
    { justify: true }
  ),

  h3("4.12.3 PDF Rendering"),
  body(
    "The parsed sections are rendered to a styled A4 PDF using ReportLab's Platypus layout engine. The PDF includes a branded cover page with proposal metadata, a persistent header and footer on every page (drawn via a canvas callback), section headings with gold accent rules, body text with justified alignment, and a structured pricing table rendered directly from the line_items state field — bypassing the text-format pricing report to produce a properly formatted table with column alignment and alternating row shading.",
    { justify: true }
  ),
  pageBreak(),
];

// CHAPTER 5 — DATA PREPARATION
const ch5 = [
  h1("5. Data Preparation"),

  h2("5.1 Product Catalog Sources"),
  body(
    "The business unit product catalog for this project is based on the Havells India electrical product range, specifically the Coral, Coral Ebony, and Oro Metallica modular switch ranges. Three data source formats are used simultaneously to maximise retrieval coverage:",
    { justify: true }
  ),

  h3("5.1.1 Small Chunks (havells_small_chunks.txt)"),
  body(
    "The small chunks file organises products by color group or category within each range. Each chunk covers one color/category grouping and contains all product variants within that grouping — their names, prices, SAP codes, HSN codes, and packing details. Chunks are delimited by 80-character separator lines of equals signs. This format is optimal for queries about a specific product type within a specific color or range.",
    { justify: true }
  ),

  h3("5.1.2 Large Chunks (havells_large_chunks.txt)"),
  body(
    "The large chunks file organises products by full product range. Each chunk covers an entire range and contains all its products across all colors and categories. This format provides broader context for range-level queries, such as 'What does Coral Ebony offer?' or 'List all products in the Oro Metallica range.'",
    { justify: true }
  ),

  h3("5.1.3 Structured JSON (havelsdata.json)"),
  body(
    "The JSON file provides a machine-readable, fully structured representation of the same catalog. Its nested hierarchy (brand → product_data → ranges → colors/categories → items) is flattened at load time into a list of individual product dicts. Each dict is converted to a standardised text block containing all product fields, then embedded for semantic search. This source excels at exact field lookup — querying for a specific SAP code, HSN code, or price point.",
    { justify: true }
  ),

  h2("5.2 RFP Document"),
  body(
    "The RFP document used for demonstration is a PDF request from ABC Infrastructure Pvt. Ltd. for the supply of modular electrical switches, plates, and accessories. It covers ten sections: introduction, scope of work, technical specifications, quantity requirements, pricing requirements, eligibility criteria, delivery terms, evaluation criteria, submission guidelines, and important dates.",
    { justify: true }
  ),
  body(
    "The document is representative of real Indian electrical procurement RFPs in its structure, terminology, and level of specificity. It includes indicative monthly demand figures, benchmark price ranges, HSN code requirements, and supply capacity expectations — all of which the system's agents must correctly identify and respond to.",
    { justify: true }
  ),

  h2("5.3 Chunking Strategy Justification"),
  body(
    "The dual-chunk strategy (small + large) for the product catalog is a deliberate design choice motivated by the retrieval trade-off between precision and recall. Small chunks provide high precision — a query for a specific product variant retrieves a chunk focused on that variant. Large chunks provide high recall — a query for a product range retrieves a comprehensive overview even if the specific variant is not named in the query.",
    { justify: true }
  ),
  body(
    "Using both, merged and reranked, captures the benefits of both extremes. The JSON index adds a third dimension: exact structured field retrieval that neither chunk format can provide as reliably.",
    { justify: true }
  ),
  pageBreak(),
];

// CHAPTER 6 — IMPLEMENTATION DETAILS
const ch6 = [
  h1("6. Technical Implementation Details"),

  h2("6.1 Technology Stack"),
  spacer(),
  new Table({
    width: { size: TW, type: WidthType.DXA },
    columnWidths: [2500, 2500, 4360],
    rows: [
      new TableRow({ children: [headerCell("Category", 2500), headerCell("Technology", 2500), headerCell("Purpose", 4360)] }),
      ...([
        ["Orchestration", "LangGraph 0.x", "Agent graph construction, state management, conditional routing"],
        ["Primary LLM", "Groq API / Llama 4 Scout", "Router, RFP analysis, matching, pricing, synthesis"],
        ["Report LLM", "Google Gemini 1.5 Flash", "Proposal document generation (large context, low cost)"],
        ["Embeddings", "BAAI/bge-small-en-v1.5", "Dense vector encoding for semantic search"],
        ["Reranker", "BAAI/bge-reranker-base", "Cross-encoder second-stage relevance reranking"],
        ["Vector Index", "FAISS (IndexFlatIP/L2)", "Fast approximate nearest-neighbour search"],
        ["PDF Extraction", "PyMuPDF (fitz)", "RFP text extraction from PDF"],
        ["PDF Generation", "ReportLab (Platypus)", "Styled PDF output for vendor proposals"],
        ["Sentence Transform", "sentence-transformers", "Embedding and reranking model hosting"],
        ["Numerics", "NumPy", "Embedding arithmetic and similarity scoring"],
        ["Language", "Python 3.10+", "All agents, tools, and utilities"],
      ].map(([c, t, p], i) => new TableRow({
        children: [
          dataCell(c, 2500, { fill: i % 2 === 0 ? LGREY : WHITE, bold: true }),
          dataCell(t, 2500, { fill: i % 2 === 0 ? LGREY : WHITE }),
          dataCell(p, 4360, { fill: i % 2 === 0 ? LGREY : WHITE }),
        ]
      }))),
    ],
  }),
  caption("Table 6.1 — Full technology stack"),
  spacer(),

  h2("6.2 Singleton Pattern for RAG Tools"),
  body(
    "Both the RFP RAG tool and the BU RAG tool use a module-level singleton pattern. The first call to get_rfp_rag_tool() or get_bu_rag_tool() triggers the full initialisation sequence: model loading, file parsing, embedding computation, and FAISS index construction. All subsequent calls return the already-initialised instance.",
    { justify: true }
  ),
  body(
    "This is critical for performance: loading a sentence transformer model and computing embeddings for hundreds of product chunks takes several seconds. Without the singleton, every agent invocation would incur this cost. With the singleton, the cost is paid once per process, regardless of how many queries are processed.",
    { justify: true }
  ),

  h2("6.3 LangGraph Graph Compilation"),
  body(
    "The LangGraph StateGraph is built by the build_graph() function in master_agent.py and compiled once via a module-level singleton (get_graph()). Graph compilation validates the node registrations, edge connections, and conditional edge mappings at startup, catching configuration errors before any query is processed.",
    { justify: true }
  ),
  body(
    "Conditional edges are implemented as Python functions that inspect state['route'] and return the name of the next node. This approach keeps routing logic transparent and independently testable — a routing function is just a Python function that can be unit-tested with a mock state dict.",
    { justify: true }
  ),

  h2("6.4 Error Handling Strategy"),
  body(
    "The system uses a defensive, non-crashing error handling strategy. Rather than raising exceptions that would terminate the graph, agents write error information to state['error'] and return the state. Downstream agents check for upstream errors before executing. The synthesise node includes state.get() calls with defaults throughout, so it never fails on a missing field.",
    { justify: true }
  ),
  body(
    "This strategy ensures that a partial failure (e.g., Gemini API being unavailable) results in a graceful degradation with a clear error message rather than an unhandled exception and empty output.",
    { justify: true }
  ),

  h2("6.5 Gemini API Integration"),
  body(
    "The Gemini integration in report_agent.py uses a two-layer approach. If the google-generativeai SDK is installed, it is used for a clean, type-safe API call. If the SDK is not installed, the agent falls back to a direct urllib HTTP POST to the Gemini REST endpoint. This means the report agent works out-of-the-box with zero additional dependencies beyond the standard library, with the SDK providing an optional improvement in error handling and response parsing.",
    { justify: true }
  ),
  pageBreak(),
];

// CHAPTER 7 — RESULTS AND EVALUATION
const ch7 = [
  h1("7. Results and Evaluation"),

  h2("7.1 System Output Examples"),

  h3("7.1.1 Route Classification"),
  body(
    "The router correctly classifies a wide range of natural language queries into the six route categories. Tested queries and their classifications are shown below:",
    { justify: true }
  ),
  spacer(),
  new Table({
    width: { size: TW, type: WidthType.DXA },
    columnWidths: [6360, 1500, 1500],
    rows: [
      new TableRow({ children: [headerCell("Query", 6360), headerCell("Expected", 1500), headerCell("Actual", 1500)] }),
      ...([
        ["What are the eligibility criteria in the RFP?", "rfp", "rfp"],
        ["What is the price of 10A 1-way switches?", "bu", "bu"],
        ["Can we fulfil the Coral Ebony requirements?", "match", "match"],
        ["Give me a full quote for all RFP requirements.", "price", "price"],
        ["Analyse the RFP and give me everything.", "full", "full"],
        ["Generate the complete vendor proposal PDF.", "report", "report"],
        ["Create a submission document for the RFP.", "report", "report"],
        ["What delivery timeline is required?", "rfp", "rfp"],
        ["Show me all available switch variants and their SAP codes.", "bu", "bu"],
      ].map(([q, e, a], i) => new TableRow({
        children: [
          dataCell(q, 6360, { fill: i % 2 === 0 ? LGREY : WHITE }),
          dataCell(e, 1500, { fill: i % 2 === 0 ? LGREY : WHITE, align: AlignmentType.CENTER }),
          dataCell(a, 1500, { fill: i % 2 === 0 ? LGREY : WHITE, align: AlignmentType.CENTER, color: "1A7A3A", bold: true }),
        ]
      }))),
    ],
  }),
  caption("Table 7.1 — Router classification results"),
  spacer(),

  h3("7.1.2 Fulfillment Scoring"),
  body(
    "For the sample RFP, the Matching Agent identified requirements across five categories: switches, plates, sockets, support modules, and technical compliance. The system correctly matched the Havells Coral and Coral Ebony ranges against switch and plate requirements with high confidence, while correctly identifying gaps in specialist items such as RJ-45 data jacks and TV sockets that are not present in the indexed catalog.",
    { justify: true }
  ),

  h3("7.1.3 Pricing Accuracy"),
  body(
    "The pricing agent correctly applied the configured 20% base margin, 18% GST, and volume discount tiers across all matched items. Benchmark prices from the RFP (10A 1-way switch at approximately Rs. 60, 16A 2-way switch at approximately Rs. 191, 32A DP switch at approximately Rs. 404) were retrieved accurately from the catalog, and the pricing calculations matched manual verification.",
    { justify: true }
  ),

  h2("7.2 Performance Characteristics"),
  spacer(),
  new Table({
    width: { size: TW, type: WidthType.DXA },
    columnWidths: [3500, 2500, 3360],
    rows: [
      new TableRow({ children: [headerCell("Pipeline Stage", 3500), headerCell("Approx. Time", 2500), headerCell("Primary Cost", 3360)] }),
      ...([
        ["RAG tool initialisation (first run)", "8\u201315 seconds", "Model loading + embedding computation"],
        ["Router node", "0.3\u20130.6 s", "Single LLM call (10 tokens output)"],
        ["RFP Agent", "2\u20134 s", "FAISS search + 2 LLM calls"],
        ["BU Agent", "3\u20136 s", "3\u00D7 FAISS + cross-encoder + LLM"],
        ["Matching Agent (per requirement)", "2\u20133 s", "BU RAG query per requirement"],
        ["Pricing Agent (per item)", "2\u20133 s", "BU RAG query per matched item"],
        ["Report Agent", "15\u201330 s", "Gemini API call + PDF rendering"],
        ["Full 'report' pipeline", "90\u2013180 s", "All agents sequentially"],
      ].map(([s, t, c], i) => new TableRow({
        children: [
          dataCell(s, 3500, { fill: i % 2 === 0 ? LGREY : WHITE, bold: true }),
          dataCell(t, 2500, { fill: i % 2 === 0 ? LGREY : WHITE, align: AlignmentType.CENTER }),
          dataCell(c, 3360, { fill: i % 2 === 0 ? LGREY : WHITE }),
        ]
      }))),
    ],
  }),
  caption("Table 7.2 — Approximate execution times per pipeline stage"),
  spacer(),
  body(
    "Note: RAG tool initialisation cost is paid only once per process. In a server deployment where the process remains alive between requests, all subsequent queries skip this cost entirely.",
    { justify: true }
  ),

  h2("7.3 Limitations"),

  h3("7.3.1 Quantity Extraction"),
  body(
    "The quantity extraction regex in the pricing agent uses heuristics to parse quantities from user queries and requirement text. For ambiguous inputs (e.g., requirements that state quantities using non-standard phrasing), the extracted quantity may default to 1. This is a known limitation that would require a more sophisticated NER-based quantity extractor in a production deployment.",
    { justify: true }
  ),

  h3("7.3.2 Latency for Full Pipeline"),
  body(
    "The sequential nature of the 'report' pipeline means that end-to-end latency scales linearly with the number of requirements (matching) and matched items (pricing). For an RFP with 20 requirements and 15 matched items, the full pipeline can take 3 to 5 minutes. Parallelising the per-requirement loops using asyncio or threading would reduce this significantly.",
    { justify: true }
  ),

  h3("7.3.3 Single Catalog Domain"),
  body(
    "The BU RAG tool is currently loaded with a single brand's catalog. Supporting multi-brand or multi-category catalogs would require either a unified index or a catalog routing mechanism that selects the appropriate index based on the query domain.",
    { justify: true }
  ),
  pageBreak(),
];

// CHAPTER 8 — FUTURE WORK
const ch8 = [
  h1("8. Future Work and Extensions"),

  h2("8.1 Parallel Requirement Processing"),
  body(
    "The Matching and Pricing Agents currently process requirements sequentially. Implementing asyncio-based concurrency would allow all requirements to be assessed simultaneously, reducing the full-pipeline latency from O(n) to approximately O(1) in the number of requirements for the network I/O portion.",
    { justify: true }
  ),

  h2("8.2 Interactive Frontend"),
  body(
    "The current system is a command-line tool. A natural next step is a web-based frontend allowing procurement teams to upload RFP PDFs, review the fulfillment report interactively, adjust the margin slider (using the reprice_with_margin() function already implemented), and download the generated proposal PDF — all without touching the command line.",
    { justify: true }
  ),

  h2("8.3 Multi-Catalog Support"),
  body(
    "The BU RAG tool is designed to be configurable via BuRagConfig. Extending the system to support multiple product catalogs (different brands, product lines, or regions) would require a catalog router that identifies which catalog is relevant to a given requirement and queries the appropriate BuRagTool instance.",
    { justify: true }
  ),

  h2("8.4 Persistent Caching"),
  body(
    "FAISS indexes and embeddings are currently rebuilt on every fresh process start. Serialising computed indexes to disk (FAISS supports native serialisation via faiss.write_index()) would eliminate the 8-15 second initialisation cost on subsequent runs, making the system viable for high-frequency interactive use.",
    { justify: true }
  ),

  h2("8.5 Evaluation Framework"),
  body(
    "A formal evaluation framework using annotated RFP-catalog pairs would allow quantitative measurement of matching precision, recall, and F1 score, as well as pricing accuracy across a diverse test set. This would enable systematic comparison of different embedding models, chunk strategies, and reranking configurations.",
    { justify: true }
  ),

  h2("8.6 ERP Integration"),
  body(
    "In a production procurement environment, the system would benefit from live integration with ERP systems (e.g., SAP) to verify real-time inventory availability and pull current catalog prices rather than relying on static data files. The existing JSON and text data sources could be replaced with ERP API calls without changing any agent logic.",
    { justify: true }
  ),
  pageBreak(),
];

// CHAPTER 9 — CONCLUSION
const ch9 = [
  h1("9. Conclusion"),
  body(
    "This project demonstrates the feasibility and practical value of a multi-agent AI system for automated RFP response generation. The system successfully addresses the core challenges of closed-domain document understanding: it grounds all agent outputs in verified source data through RAG, applies deterministic commercial rules for pricing, dynamically routes queries to the appropriate subset of agents, and produces a professionally formatted submission-ready document.",
    { justify: true }
  ),
  body(
    "The modular architecture — six independent agents communicating through a shared typed state, orchestrated by a LangGraph StateGraph — provides clear separation of concerns. Each agent can be developed, tested, and upgraded independently. New agents (e.g., a compliance checking agent, a competitor pricing agent) can be added as new graph nodes without restructuring existing nodes.",
    { justify: true }
  ),
  body(
    "The deliberate separation of LLM providers between the pipeline agents (Groq/Llama 4 Scout for speed and cost efficiency in iterative retrieval steps) and the report generation agent (Google Gemini Flash for large-context document synthesis) reflects a practical architectural principle: use the right tool for each task rather than forcing a single provider to handle workloads it is not optimised for.",
    { justify: true }
  ),
  body(
    "The system reduces a multi-day manual procurement response process to a fully automated pipeline completing in under five minutes for a typical RFP, with human review required only at the final output stage. This represents a meaningful contribution to the practical application of AI in enterprise procurement workflows.",
    { justify: true }
  ),
  pageBreak(),
];

// REFERENCES
const references = [
  h1("References"),
  body("Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. Advances in Neural Information Processing Systems, 33, 9459-9474.", { justify: true }),
  spacer(),
  body("Nogueira, R., & Cho, K. (2019). Passage Re-ranking with BERT. arXiv preprint arXiv:1901.04085.", { justify: true }),
  spacer(),
  body("Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2023). C-Pack: Packaged Resources To Advance General Chinese Embedding. arXiv preprint arXiv:2309.07597. [BGE embedding model family]", { justify: true }),
  spacer(),
  body("LangChain. (2024). LangGraph: Build stateful, multi-actor applications with LLMs. https://github.com/langchain-ai/langgraph", { justify: true }),
  spacer(),
  body("Johnson, J., Douze, M., & Jegou, H. (2019). Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 7(3), 535-547. [FAISS library]", { justify: true }),
  spacer(),
  body("Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing.", { justify: true }),
  spacer(),
  body("Meta AI. (2024). Llama 4: Open Foundation and Fine-Tuned Chat Models. Meta AI Research.", { justify: true }),
  spacer(),
  body("Google DeepMind. (2024). Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. Google Technical Report.", { justify: true }),
  spacer(),
  body("Groq Inc. (2024). Groq API Documentation. https://console.groq.com/docs", { justify: true }),
  spacer(),
  body("ReportLab Inc. (2024). ReportLab PDF Library User Guide. https://www.reportlab.com/docs/reportlab-userguide.pdf", { justify: true }),
];

// ─── ASSEMBLE DOCUMENT ────────────────────────────────────────────────────────

const allChildren = [
  ...coverSection,
  ...tocSection,
  ...abstractSection,
  ...ch1, ...ch2, ...ch3, ...ch4, ...ch5,
  ...ch6, ...ch7, ...ch8, ...ch9,
  ...references,
];

const doc = new Document({
  numbering: { config: numberingConfig },
  styles: {
    default: {
      document: { run: { font: "Arial", size: 20, color: BLACK } },
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: NAVY },
        paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 0 },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: NAVY },
        paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1 },
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "Arial", color: "2C5F8A" },
        paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 2 },
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1260, bottom: 1440, left: 1260 },
        },
      },
      headers: {
        default: new Header({
          children: [
            new Paragraph({
              border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: NAVY, space: 1 } },
              children: [
                new TextRun({ text: "RFP Response Automation System", font: "Arial", size: 17, color: MGREY }),
                new TextRun({ text: "\t", font: "Arial", size: 17 }),
                new TextRun({ text: "Technical Report", font: "Arial", size: 17, color: MGREY, italics: true }),
              ],
              tabStops: [{ type: "right", position: 9080 }],
            }),
          ],
        }),
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              border: { top: { style: BorderStyle.SINGLE, size: 4, color: NAVY, space: 1 } },
              children: [
                new TextRun({ text: "Confidential", font: "Arial", size: 16, color: MGREY }),
                new TextRun({ text: "\t", font: "Arial", size: 16 }),
                new TextRun({
                  children: ["Page ", PageNumber.CURRENT],
                  font: "Arial",
                  size: 16,
                  color: MGREY
                }),
              ],
              tabStops: [{ type: "right", position: 9080 }],
            }),
          ],
        }),
      },
      children: allChildren,
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("RFP_Automation_System_Report.docx", buf);
  console.log("Done.");
});
