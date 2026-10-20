"""
start.py
--------
Start SYNAPSE on this laptop with one command:

    python start.py            (or double-click start.bat on Windows, ./start.sh elsewhere)

It checks the setup, builds the dashboard if its code changed, starts the
server and opens the browser. Press Ctrl+C in this window to stop.

Options: --port 8000, --no-browser, --skip-build
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
DIST = FRONTEND / "dist"
STAMP = DIST / ".source-hash"


def say(msg: str) -> None:
    print(f"[start] {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"\n[start] PROBLEM: {msg}\n", flush=True)
    raise SystemExit(1)


def install_requirements() -> None:
    """Inside a virtual environment: pip install when requirements.txt changed."""
    if sys.prefix == sys.base_prefix:
        return   # not a venv: don't touch the system Python, just check below
    req = ROOT / "requirements.txt"
    stamp = Path(sys.prefix) / ".synapse-requirements"
    current = hashlib.sha256(req.read_bytes()).hexdigest()
    if stamp.is_file() and stamp.read_text().strip() == current:
        return
    say("Installing Python packages (first run takes several minutes: PyTorch is large) ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(req)],
                   check=True)
    stamp.write_text(current)


def check_python() -> None:
    if sys.version_info < (3, 12):
        fail(f"Python 3.12 or newer is needed (this is {sys.version.split()[0]}).")
    install_requirements()
    try:
        import fastapi  # noqa: F401
        import langgraph  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError as exc:
        fail(f"A Python package is missing ({exc.name}). Run:  pip install -r requirements.txt\n"
             "        (start.bat / start.sh do this for you inside the venv)")


def check_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        shutil.copy(ROOT / ".env.example", env)
        say("Created .env from .env.example. Open it and paste your OPENROUTER_API_KEY and GEMINI_API_KEY,")
        say("then restart. (The dashboard still opens; the AI steps need the keys.)")
        return
    text = env.read_text(encoding="utf-8", errors="replace")
    for key in ("OPENROUTER_API_KEY", "GEMINI_API_KEY"):
        line = next((l for l in text.splitlines() if l.strip().startswith(f"{key}=")), f"{key}=")
        if not line.split("=", 1)[1].strip():
            say(f"Note: {key} is empty in .env; " +
                ("the AI steps will fail." if key.startswith("OPENROUTER") else
                 "proposal text will be written with OpenRouter instead of Gemini."))


def frontend_hash() -> str:
    """Hash of everything the build depends on, so we rebuild only after changes."""
    h = hashlib.sha256()
    files = [FRONTEND / n for n in ("package.json", "package-lock.json", "index.html", "vite.config.ts",
                                    "tailwind.config.js", "postcss.config.js", "tsconfig.json")]
    files += sorted((FRONTEND / "src").rglob("*"))
    for f in files:
        if f.is_file():
            h.update(f.relative_to(FRONTEND).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def build_frontend() -> None:
    current = frontend_hash()
    if (DIST / "index.html").is_file() and STAMP.is_file() and STAMP.read_text().strip() == current:
        say("Dashboard is up to date.")
        return
    npm = shutil.which("npm")
    if npm is None:
        if (DIST / "index.html").is_file():
            say("Node.js not found; using the existing dashboard build (it may be out of date).")
            return
        fail("The dashboard needs building, which needs Node.js 18+ (https://nodejs.org). Install it and run again.")
    if not (FRONTEND / "node_modules").is_dir():
        say("Installing dashboard packages (first run only, about a minute) ...")
        subprocess.run([npm, "ci"], cwd=FRONTEND, check=True)
    say("Building the dashboard ...")
    subprocess.run([npm, "run", "build"], cwd=FRONTEND, check=True)
    STAMP.write_text(current)


def wait_until_up(url: str, server: subprocess.Popen, timeout: float = 180) -> None:
    start = time.time()
    while time.time() - start < timeout:
        if server.poll() is not None:
            fail("The server stopped while starting; see the messages above.\n"
                 "        If it says the address is in use, SYNAPSE may already be running\n"
                 "        (close the other window) or try:  python start.py --port 8001")
        try:
            urllib.request.urlopen(f"{url}/api/health", timeout=2)
            return
        except OSError:
            time.sleep(1)
    fail(f"The server did not answer within {int(timeout)} s.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the SYNAPSE dashboard on this laptop.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("SYNAPSE_PORT", "8000")))
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    parser.add_argument("--skip-build", action="store_true", help="don't rebuild the dashboard")
    args = parser.parse_args()

    os.chdir(ROOT)
    check_python()
    check_env()
    if not args.skip_build:
        build_frontend()

    url = f"http://127.0.0.1:{args.port}"
    say(f"Starting the server on {url} (loading the AI models takes ~15-30 s) ...")
    server = subprocess.Popen([sys.executable, "-m", "api"], cwd=ROOT,
                              env={**os.environ, "SYNAPSE_PORT": str(args.port)})
    try:
        wait_until_up(url, server)
        say(f"Ready: {url}   (press Ctrl+C here to stop)")
        if not args.no_browser:
            webbrowser.open(url)
        return server.wait()
    except KeyboardInterrupt:
        say("Stopping ...")
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        return 0
    except BaseException:
        server.terminate()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
