from __future__ import annotations

import sys

from rfp_agent import rfp_agent_node


def main() -> int:
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = input("RFP question: ").strip()

    if not query:
        print("No query provided.")
        return 1

    state = {"user_query": query}
    out = rfp_agent_node(state)

    print("\nRFP ANSWER")
    print("-" * 60)
    print(out.get("rfp_answer", "No answer."))

    reqs = out.get("rfp_requirements", [])
    if reqs:
        print("\nEXTRACTED REQUIREMENTS (first 10)")
        print("-" * 60)
        for item in reqs[:10]:
            print(f"- {item.get('requirement', '').strip()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
