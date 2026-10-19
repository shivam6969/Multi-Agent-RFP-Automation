from __future__ import annotations

import sys

from matching_agent import matching_agent_node
from pricing_agent import pricing_agent_node


def _parse_requirements(raw: str) -> list:
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    return [{"requirement": p, "weight": 1.0} for p in parts]


def main() -> int:
    query = input("User query (for qty context, e.g. '200 units of each'): ").strip()

    raw = " ".join(sys.argv[1:]).strip()
    if not raw:
        raw = input("Requirements (separate with ';'): ").strip()

    if not raw:
        print("No requirements provided.")
        return 1

    # Pricing prices the products chosen by matching, so run matching first.
    state = matching_agent_node({"user_query": query, "rfp_requirements": _parse_requirements(raw)})
    out = pricing_agent_node(state)

    print("\nPRICING REPORT")
    print("-" * 60)
    print(out.get("pricing_report", "No report."))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
