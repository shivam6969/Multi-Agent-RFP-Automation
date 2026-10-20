#!/usr/bin/env sh
# Start SYNAPSE on this laptop (macOS / Linux):  ./start.sh
# First run creates the Python environment in venv/ and installs packages.
set -e
cd "$(dirname "$0")"

if [ ! -x venv/bin/python ]; then
    echo "[start] Creating the Python environment in venv/ (first run only) ..."
    PY=$(command -v python3.12 || command -v python3 || true)
    if [ -z "$PY" ]; then
        echo "[start] PROBLEM: Python 3.12 was not found. Install it from https://www.python.org/downloads/"
        exit 1
    fi
    "$PY" -m venv venv
fi

exec venv/bin/python start.py "$@"
