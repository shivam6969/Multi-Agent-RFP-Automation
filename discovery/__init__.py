"""
Discovery Agent: finds RFPs to respond to, from swappable sources.

  sample  generated, clearly-labelled sample tenders built from the catalogue
          (default; deterministic for a given search)
  upload  a PDF the user provides (metadata read by the LLM, see metadata.py)
  gem     live GeM portal scraper, OFF unless DISCOVERY_GEM_ENABLED is set

See discovery_agent.py for the LangGraph node.
"""
