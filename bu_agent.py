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

_W = 72


def _wrap(text: str) -> list:
    width = _W - 4
    lines = []
    for raw in (text or "").splitlines() or [""]:
        if not raw:
            lines.append("")
            continue
        for i in range(0, max(len(raw), 1), width):
            lines.append(raw[i : i + width])
    return lines


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
    answer = bu_query(query, api_key=api_key, config=config)
    state["bu_answer"] = answer

    # ── Rich print output ──────────────────────────────────────────────────────
    print(f"\n╔{'═' * (_W - 2)}╗")
    print(f"║  {'BU AGENT — OUTPUT':<{_W - 4}}║")
    print(f"╠{'═' * (_W - 2)}╣")
    print(f"║  {'Query: ' + query[:_W - 11]:<{_W - 4}}║")
    print(f"╠{'─' * (_W - 2)}╣")
    print(f"║  {'ANSWER':<{_W - 4}}║")
    print(f"╠{'─' * (_W - 2)}╣")
    for line in _wrap(answer):
        print(f"║  {line:<{_W - 4}}║")
    print(f"╚{'═' * (_W - 2)}╝\n")

    return state