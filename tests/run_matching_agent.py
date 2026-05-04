from __future__ import annotations

import sys

from matching_agent import matching_agent_node


def _parse_requirements(raw: str) -> list:
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    return [{"requirement": p, "weight": 1.0} for p in parts]


def main() -> int:
    raw = " ".join(sys.argv[1:]).strip()
    if not raw:
        raw = input("Requirements (separate with ';'): ").strip()

    if not raw:
        print("No requirements provided.")
        return 1

    state = {"rfp_requirements": _parse_requirements(raw)}
    out = matching_agent_node(state)

    print("\nFULFILLMENT REPORT")
    print("-" * 60)
    print(out.get("fulfillment_report", "No report."))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
