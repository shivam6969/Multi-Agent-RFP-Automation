"""
Test package.

Unit tests must never call a real LLM. Blanking the keys here (before
settings.py loads .env, which does not override existing env vars) makes any
unmocked LLM call fail fast instead of spending API credit.

The manual `run_*.py` scripts are run directly and are not affected.
"""

import os

os.environ["OPENROUTER_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
