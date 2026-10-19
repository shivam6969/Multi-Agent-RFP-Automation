"""
Test package.

Unit tests must never call a real LLM. Blanking the keys here (before
settings.py loads .env, which does not override existing env vars) makes any
unmocked LLM call fail fast instead of spending API credit.

Tests also use a throwaway SQLite database instead of var/synapse.db.

The manual `run_*.py` scripts are run directly and are not affected.
"""

import os
import tempfile

os.environ["OPENROUTER_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
os.environ["SYNAPSE_DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="synapse-test-"), "test.db")
