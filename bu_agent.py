"""
agents/bu_agent.py
-----------------
BU Agent — LangGraph node

Responsibility
--------------
Answer the user's query from the business-unit catalog and write it to
state["bu_answer"].
"""

from __future__ import annotations

import os
from typing import Optional

from settings import GROQ_API_KEY
from bu_rag import BuRagConfig, get_bu_rag_tool
from state import AgentState


def get_bu_tool(config: Optional[BuRagConfig] = None):
    """Return the shared BU RAG tool instance."""
    return get_bu_rag_tool(config)


def bu_query(
    query: str,
    *,
    api_key: Optional[str] = None,
    config: Optional[BuRagConfig] = None,
    temperature: float = 0.4,
    max_tokens: int = 400,
) -> str:
    """Run a BU catalog query via the shared tool."""
    bu_tool = get_bu_tool(config)
    return bu_tool.run_query(
        query,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )


def bu_agent_node(state: AgentState, config: Optional[BuRagConfig] = None) -> AgentState:
    """
    LangGraph node.
    Reads user_query, writes bu_answer.
    """
    api_key = os.environ.get("GROQ_API_KEY") or GROQ_API_KEY
    query = state.get("user_query", "")

    print("[BuAgent] Answering user query from BU catalog …")
    state["bu_answer"] = bu_query(query, api_key=api_key, config=config)
    print(f"\n[BuAgent] ══ BU CATALOG ANSWER ══════════════════════════════════")
    print(state["bu_answer"])
    print(f"[BuAgent] ════════════════════════════════════════════════════\n")

    return state