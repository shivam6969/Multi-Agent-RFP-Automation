"""
log_setup.py
------------
One place to turn on logging for command-line entry points (master_agent,
workflow, `python -m api`, tests/run_*.py). Library modules only log; they
never print, so the API can run them without touching the console.

Output is switched to UTF-8 with replacement: on Windows, piped or redirected
output defaults to cp1252, which cannot encode ₹, box-drawing characters or
emoji and used to crash the agents mid-run.
"""

from __future__ import annotations

import logging
import sys

PLAIN = "%(message)s"
DETAILED = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO, fmt: str = PLAIN) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):   # not a text stream we can reconfigure
            pass
    logging.basicConfig(level=level, format=fmt)
    # Third-party request logs drown out the agents' progress at INFO.
    for noisy in ("httpx", "httpcore", "openai", "google_genai", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
