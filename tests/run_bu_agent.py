from __future__ import annotations

import sys

from bu_agent import bu_agent_node


def main() -> int:
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = input("Catalog question: ").strip()

    if not query:
        print("No query provided.")
        return 1

    state = {"user_query": query}
    out = bu_agent_node(state)

    print("\nBU ANSWER")
    print("-" * 60)
    print(out.get("bu_answer", "No answer."))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
