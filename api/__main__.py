"""Start the API on this laptop: python -m api  (then open http://127.0.0.1:8000/docs)."""

import logging
import os

import uvicorn

from .main import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

if __name__ == "__main__":
    # Bound to localhost only: the dashboard is meant to run on your own laptop.
    uvicorn.run(create_app(), host="127.0.0.1", port=int(os.environ.get("SYNAPSE_PORT", "8000")))
