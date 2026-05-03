"""
master_agent.py
---------------
Master Agent — LangGraph orchestrator and CLI entrypoint.

Builds a StateGraph that routes the user query through the correct
pipeline of agent nodes:

    "rfp"    → rfp_agent                               → synthesise → END
    "bu"     → bu_agent                                → synthesise → END
    "match"  → rfp_agent → matching_agent              → synthesise → END
    "price"  → rfp_agent → matching_agent → pricing    → synthesise → END
    "full"   → rfp_agent → matching_agent → pricing    → synthesise → END
    "report" → rfp_agent → bu_agent → matching_agent
                        → pricing  → report_agent      → synthesise → END

Run from the command line:
    python master_agent.py "What products can fulfil the RFP requirements?"
    python master_agent.py "Generate the full vendor proposal PDF"
"""

from __future__ import annotations

import os
import sys
from langgraph.graph import END, StateGraph

from state import AgentState
from settings import GROQ_API_KEY
from llm import chat_completion
from bu_agent import bu_agent_node
from rfp_agent import rfp_agent_node
from matching_agent import matching_agent_node
from pricing_agent import pricing_agent_node
from report_agent import report_agent_node          # ← new


# ─── Router node ──────────────────────────────────────────────────────────────

_ROUTER_PROMPT = """\
You are a query classifier for an RFP analysis system.

Classify the user query into EXACTLY ONE of these categories and reply with
ONLY the label — no explanation, no punctuation:

  rfp    — The question is about the RFP document itself (scope, clauses,
           eligibility, deadlines, deliverables).
  bu     — The question is about the business-unit product catalog only
           (product specs, prices, availability) with no RFP context.
  match  — The question asks whether our products can meet RFP requirements
           (fulfillment check, gap analysis).
  price  — The question asks for pricing, quotation, or cost estimation
           for RFP requirements.
  full   — The question requires the complete pipeline: RFP analysis,
           matching, AND pricing together.
  report — The user explicitly asks for a proposal document, vendor response,
           submission PDF, pitch document, or full proposal generation.

User query: {query}
""".strip()


def router_node(state: AgentState) -> AgentState:
    """Classify the user query and write state['route']."""
    query = state.get("user_query", "")
    api_key = os.environ.get("GROQ_API_KEY") or GROQ_API_KEY

    raw = chat_completion(
        system="You are a precise query classifier. Reply with one word only.",
        user=_ROUTER_PROMPT.format(query=query),
        temperature=0.0,
        max_tokens=10,
        api_key=api_key,
    )
    route = raw.strip().lower().split()[0] if raw.strip() else "full"
    if route not in ("rfp", "bu", "match", "price", "full", "report"):
        route = "full"

    state["route"] = route
    print(f"[MasterAgent] Routed to: {route}")
    return state


# ─── Synthesis node ───────────────────────────────────────────────────────────

_SYNTH_PROMPT = """\
You are a senior procurement analyst.  Combine the analysis sections below
into a single, well-structured final response for the user.

Keep it professional, concise, and actionable.
Use headings and bullet points where appropriate.

{sections}

User's original question: {query}
""".strip()


def synthesise_node(state: AgentState) -> AgentState:
    """Combine all available outputs into state['final_response']."""
    sections: list[str] = []

    if state.get("rfp_answer"):
        sections.append(f"## RFP Analysis\n{state['rfp_answer']}")

    if state.get("bu_answer"):
        sections.append(f"## Product Catalog\n{state['bu_answer']}")

    if state.get("fulfillment_report"):
        sections.append(f"## Fulfillment Report\n{state['fulfillment_report']}")

    if state.get("pricing_report"):
        sections.append(f"## Pricing\n{state['pricing_report']}")
    elif state.get("pricing_summary"):
        summary = state["pricing_summary"]
        sections.append(
            "## Pricing Summary\n"
            f"- Margin: {summary.get('margin_pct', 0)}%\n"
            f"- Subtotal (ex GST): {summary.get('currency', 'INR')} {summary.get('subtotal_ex_gst', 0):,.2f}\n"
            f"- Grand Total: {summary.get('currency', 'INR')} {summary.get('grand_total', 0):,.2f}"
        )

    if state.get("report_pdf_path"):
        sections.append(
            f"## Proposal Document\n"
            f"Full vendor proposal PDF generated successfully.\n"
            f"Path: {state['report_pdf_path']}"
        )

    # If only one section, skip the LLM synthesis — just return it directly
    if len(sections) <= 1:
        state["final_response"] = sections[0] if sections else "No results."
        return state

    api_key = os.environ.get("GROQ_API_KEY") or GROQ_API_KEY
    query = state.get("user_query", "")

    state["final_response"] = chat_completion(
        system="You synthesise multi-agent analysis into a professional report.",
        user=_SYNTH_PROMPT.format(
            sections="\n\n---\n\n".join(sections),
            query=query,
        ),
        temperature=0.3,
        max_tokens=1200,
        api_key=api_key,
    )
    return state


# ─── Routing logic ────────────────────────────────────────────────────────────

def _route_after_router(state: AgentState) -> str:
    """Conditional edge: decide which branch to take after router_node."""
    route = state.get("route", "full")
    if route == "bu":
        return "bu_agent"
    # "report" needs everything — starts with RFP like all other full pipelines
    return "rfp_agent"


def _route_after_rfp(state: AgentState) -> str:
    """Conditional edge: after rfp_agent, decide next step."""
    route = state.get("route", "full")
    if route == "rfp":
        return "synthesise"
    if route == "report":
        return "bu_agent"        # report needs BU answer too
    # match, price, full → go to matching
    return "matching_agent"


def _route_after_bu_in_report(state: AgentState) -> str:
    """After bu_agent in the report pipeline, always go to matching."""
    return "matching_agent"


def _route_after_matching(state: AgentState) -> str:
    """Conditional edge: after matching_agent, decide next step."""
    route = state.get("route", "full")
    if route == "match":
        return "synthesise"
    # price, full, report → go to pricing
    return "pricing_agent"


def _route_after_pricing(state: AgentState) -> str:
    """Conditional edge: after pricing_agent, decide next step."""
    route = state.get("route", "full")
    if route == "report":
        return "report_agent"
    # price, full → synthesise
    return "synthesise"


# ─── Graph builder ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construct and compile the LangGraph StateGraph."""
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("router", router_node)
    graph.add_node("rfp_agent", rfp_agent_node)
    graph.add_node("bu_agent", bu_agent_node)
    graph.add_node("matching_agent", matching_agent_node)
    graph.add_node("pricing_agent", pricing_agent_node)
    graph.add_node("report_agent", report_agent_node)   # ← new
    graph.add_node("synthesise", synthesise_node)

    # Entry point
    graph.set_entry_point("router")

    # Conditional edge after router
    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "bu_agent": "bu_agent",
            "rfp_agent": "rfp_agent",
        },
    )

    # BU agent — two possible next steps:
    #   - standalone "bu" route → synthesise
    #   - "report" route (arrived here from rfp_agent) → matching_agent
    graph.add_conditional_edges(
        "bu_agent",
        lambda s: (
            "matching_agent" if s.get("route") == "report" else "synthesise"
        ),
        {
            "matching_agent": "matching_agent",
            "synthesise": "synthesise",
        },
    )

    # Conditional edge after RFP agent
    graph.add_conditional_edges(
        "rfp_agent",
        _route_after_rfp,
        {
            "synthesise": "synthesise",
            "bu_agent": "bu_agent",          # report route detour
            "matching_agent": "matching_agent",
        },
    )

    # Conditional edge after matching agent
    graph.add_conditional_edges(
        "matching_agent",
        _route_after_matching,
        {
            "synthesise": "synthesise",
            "pricing_agent": "pricing_agent",
        },
    )

    # Conditional edge after pricing agent  ← was a plain edge before
    graph.add_conditional_edges(
        "pricing_agent",
        _route_after_pricing,
        {
            "report_agent": "report_agent",
            "synthesise": "synthesise",
        },
    )

    # Report agent → synthesise → END
    graph.add_edge("report_agent", "synthesise")

    # Synthesise → END
    graph.add_edge("synthesise", END)

    return graph


# ─── Public API ───────────────────────────────────────────────────────────────

_COMPILED = None


def get_graph():
    """Return the compiled graph (lazy singleton)."""
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_graph().compile()
    return _COMPILED


def run(query: str) -> str:
    """
    Run the full pipeline for a user query.
    Returns the final synthesised response.
    """
    graph = get_graph()
    result = graph.invoke({"user_query": query})
    return result.get("final_response", result.get("error", "No output produced."))


# ─── CLI entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python master_agent.py \"<your query>\"")
        print()
        print("Examples:")
        print('  python master_agent.py "What are the RFP eligibility criteria?"')
        print('  python master_agent.py "Show me pricing for all matched products"')
        print('  python master_agent.py "Generate the full vendor proposal PDF"')
        sys.exit(1)

    user_query = " ".join(sys.argv[1:])
    print(f"\n{'='*70}")
    print(f"  Query: {user_query}")
    print(f"{'='*70}\n")

    response = run(user_query)

    print(f"\n{'='*70}")
    print("  FINAL RESPONSE")
    print(f"{'='*70}\n")
    print(response)