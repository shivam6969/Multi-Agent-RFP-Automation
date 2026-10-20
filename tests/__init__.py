"""
Test package.

Unit tests must never call a real LLM. Blanking the keys here (before
settings.py loads .env, which does not override existing env vars) makes any
unmocked LLM call fail fast instead of spending API credit.

Tests also use a throwaway SQLite database and embedding cache instead of
var/synapse.db and var/cache/.

The manual `run_*.py` scripts are run directly and are not affected.
"""

import os
import tempfile

os.environ["OPENROUTER_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
_TMP = tempfile.mkdtemp(prefix="synapse-test-")
os.environ["SYNAPSE_DB_PATH"] = os.path.join(_TMP, "test.db")
os.environ["SYNAPSE_CACHE_DIR"] = os.path.join(_TMP, "cache")
os.environ["SYNAPSE_PROPOSALS_DIR"] = os.path.join(_TMP, "proposals")
