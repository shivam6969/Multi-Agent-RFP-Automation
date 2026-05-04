from __future__ import annotations

import sys

from pricing_agent import pricing_agent_node


def _parse_requirements(raw: str) -> list:
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    return [
        {"requirement": p, "weight": 1.0, "matched": True}
        for p in parts
    ]


def main() -> int:
    query = input("User query (for qty context): ").strip()
    if not query:
        query = "qty 1"

    raw = " ".join(sys.argv[1:]).strip()
    if not raw:
        raw = input("Matched requirements (separate with ';'): ").strip()

    if not raw:
        print("No requirements provided.")
        return 1

    state = {
        "user_query": query,
        "matched_items": _parse_requirements(raw),
    }
    out = pricing_agent_node(state)

    print("\nPRICING REPORT")
    print("-" * 60)
    print(out.get("pricing_report", "No report."))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
