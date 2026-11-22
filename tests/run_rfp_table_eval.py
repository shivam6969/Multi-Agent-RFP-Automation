"""
tests/run_rfp_table_eval.py
---------------------------
Manual check of RFP table understanding on the sample RFPs (tests/sample_rfps.py).
No LLM calls. Uses the real embedding + reranker models for the retrieval part
(downloaded on first use), so it is a runner, not a unit test.

    python -m tests.run_rfp_table_eval              # parsing + retrieval
    python -m tests.run_rfp_table_eval --no-retrieval

Reports per sample:
  old chunker   items whose description AND quantity survive in one chunk of
                the previous pipeline (flat text, regex section split, pieces
                of 80 characters or fewer dropped). An upper bound: it does
                not check that the LLM would pair the right number with the
                right item (the table arrived as one cell per line).
  table reader  items read from tables by rules with the exact quantity
  retrieval     for "quantity of <item>", is the item's row (with its
                quantity) in the top-k hybrid results; and do the fact
                questions retrieve the expected text
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path
from typing import List

import pymupdf

from log_setup import setup_logging
from rfp_agent import _table_requirements
from rfp_parser import parse_pdf
from tests.sample_rfps import build_all


def _old_chunks(pdf_path: str) -> List[str]:
    """The chunking used before rfp_parser (kept here only for comparison)."""
    with pymupdf.open(pdf_path) as doc:
        text = "".join(page.get_text() for page in doc)
    raw = re.split(r"(?=\n?\d+\.\s)|(?=\n?\d+\.\d+)", text)
    sections = [c.strip() for c in raw if len(c.strip()) > 80]
    out: List[str] = []
    for chunk in sections:
        if len(chunk) <= 800:
            out.append(chunk)
            continue
        temp = ""
        for sent in chunk.split(". "):
            if len(temp) + len(sent) < 800:
                temp += sent + ". "
            else:
                if temp:
                    out.append(temp.strip())
                temp = sent + ". "
        if temp:
            out.append(temp.strip())
    return out


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="rfp-eval-"))
    totals = {"items": 0, "old": 0, "rules": 0, "retrieved": 0, "facts": 0, "facts_ok": 0}
    for name, (path, truth) in build_all(tmp).items():
        old = [_flat(c) for c in _old_chunks(str(path))]
        doc = parse_pdf(str(path))
        found = []
        for t in doc.tables:
            if t.kind == "boq":
                found += _table_requirements(t)[0]

        old_ok = sum(
            1 for item in truth.items
            if any(_flat(item.description.split(",")[0]) in c and f"{item.quantity:,}".lower() in c
                   or (_flat(item.description.split(",")[0]) in c and str(item.quantity) in c) for c in old)
        )
        rules_ok = sum(1 for item, req in zip(truth.items, found) if req["quantity"] == item.quantity)
        print(f"\n=== {name}: {path.name}  ({doc.page_count} pages, {len(truth.items)} items)")
        print(f"  tables found : {', '.join(f'{t.id} {t.kind}/{len(t.rows)} rows ({t.detection})' for t in doc.tables)}")
        print(f"  old chunker  : {old_ok}/{len(truth.items)} items with description and quantity in one chunk")
        print(f"  table reader : {rules_ok}/{len(truth.items)} items with the exact quantity "
              f"({len(found)} rows read)")
        totals["items"] += len(truth.items)
        totals["old"] += old_ok
        totals["rules"] += rules_ok

        if args.no_retrieval:
            continue
        from rfp_rag import RfpRagConfig, get_rfp_rag_tool

        tool = get_rfp_rag_tool(RfpRagConfig(pdf_path=str(path)))
        hits = 0
        for item in truth.items:
            results = tool.retrieve(f"How many {item.description} are required?", top_k=args.top_k)
            key = _flat(item.description)[:25]
            qty = f"{item.quantity:,}"
            if any(key in _flat(r) and qty in r for r in results):
                hits += 1
            else:
                print(f"    miss: {item.description}  (qty {qty})")
        print(f"  retrieval    : {hits}/{len(truth.items)} item rows in the top {args.top_k}")
        totals["retrieved"] += hits
        for question, expected in truth.facts.items():
            context = tool.build_context(question, top_k=args.top_k)
            ok = expected.lower() in context.lower()
            totals["facts"] += 1
            totals["facts_ok"] += ok
            print(f"  fact         : {'ok  ' if ok else 'MISS'} {question}  (expects '{expected}')")

    print(f"\nTOTAL: old chunker {totals['old']}/{totals['items']}, table reader {totals['rules']}/{totals['items']}"
          + ("" if args.no_retrieval else
             f", retrieval {totals['retrieved']}/{totals['items']}, facts {totals['facts_ok']}/{totals['facts']}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
