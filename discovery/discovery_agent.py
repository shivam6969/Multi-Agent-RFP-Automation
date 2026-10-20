"""
Discovery Agent: LangGraph node.

Reads  state["discovery_query"]  (DiscoveryQuery fields as a dict)
       state["discovery_source"] ("sample" by default; "gem" if enabled)
Writes state["discovered_rfps"]  (list of RfpCandidate dicts, PDFs saved)
       state["error"]            if the source is unavailable

The user then picks one RFP, which starts the review workflow (workflow.py).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from langgraph.graph import END, START, StateGraph

from settings import DISCOVERY_DIR
from state import AgentState

from .base import DiscoveryQuery, DiscoverySource, ProgressFn, RfpCandidate, SourceUnavailable
from .gem import GemSource
from .samples import SampleSource

logger = logging.getLogger(__name__)

SOURCES: Dict[str, DiscoverySource] = {s.name: s for s in (SampleSource(), GemSource())}
DEFAULT_SOURCE = "sample"


def list_sources() -> List[Dict[str, object]]:
    """For the UI: every source, whether it can run, and why not."""
    out = []
    for source in SOURCES.values():
        reason = source.available()
        out.append({"name": source.name, "label": source.label, "enabled": reason is None, "reason": reason})
    return out


def run_discovery(query: DiscoveryQuery, source_name: str = DEFAULT_SOURCE,
                  out_dir: str = DISCOVERY_DIR, progress: Optional[ProgressFn] = None) -> List[RfpCandidate]:
    source = SOURCES.get(source_name)
    if source is None:
        raise SourceUnavailable(f"Unknown discovery source: {source_name}")
    reason = source.available()
    if reason:
        raise SourceUnavailable(reason)
    return source.search(query, out_dir, progress)


def discovery_node(state: AgentState) -> AgentState:
    query = DiscoveryQuery.model_validate(state.get("discovery_query") or {})
    source = state.get("discovery_source") or DEFAULT_SOURCE
    try:
        found = run_discovery(query, source)
    except SourceUnavailable as exc:
        state["error"] = str(exc)
        state["discovered_rfps"] = []
        return state
    logger.info("Discovery (%s) found %d RFPs", source, len(found))
    state["discovered_rfps"] = [c.to_dict() for c in found]
    return state


def build_discovery_graph():
    graph = StateGraph(AgentState)
    graph.add_node("discovery", discovery_node)
    graph.add_edge(START, "discovery")
    graph.add_edge("discovery", END)
    return graph.compile()
