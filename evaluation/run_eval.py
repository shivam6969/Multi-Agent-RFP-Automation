"""
evaluation/run_eval.py
----------------------
End-to-end evaluation of the Synapse agents on the sample RFPs that have
ground truth (tests/sample_rfps.py: hotel, school, office; 46 item rows).

    python -m evaluation.run_eval                          # offline + live, default models
    python -m evaluation.run_eval --offline-only           # no API calls
    python -m evaluation.run_eval --llm-models meta-llama/llama-4-scout,openai/gpt-4o-mini
    python -m evaluation.run_eval --samples hotel --no-proposal

What it measures
  offline (no LLM)  parsing + table reader vs the old chunker; retrieval hit
                    rates; local model load / embedding / rerank times
  live, per model   the real review workflow (temporary checkpoint DB, nothing
                    touches the dashboard): extraction, matching, pricing,
                    proposal; accuracy per agent from evaluation/metrics.py;
                    wall time, LLM calls, tokens, cost and fallbacks per stage

The reviewer is simulated as "approve as extracted, accept all suggestions",
so the numbers describe the agents' own output. Output goes to
var/eval/<timestamp>/results.json and report.md.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import patch

from log_setup import setup_logging

import llm_usage
from evaluation import metrics as M
from settings import (
    BU_RERANK_MODEL,
    GEMINI_MODEL,
    LLM_MODEL,
    PRICING_RULES,
    RFP_EMBED_MODEL,
    VAR_DIR,
)
from tests.sample_rfps import build_all

TOP_K = 6


def _secs(start: float) -> float:
    return round(time.perf_counter() - start, 3)


# ─── Offline: parsing, retrieval, local models ────────────────────────────────

def offline_sample(name: str, path: Path, truth, cache_dir: Path) -> Dict[str, Any]:
    import embedding_cache
    import rfp_rag
    from rfp_agent import _table_requirements
    from rfp_parser import parse_pdf
    from tests.run_rfp_table_eval import _flat, _old_chunks

    t = time.perf_counter()
    doc = parse_pdf(str(path))
    parse_s = _secs(t)
    reqs = []
    for table in doc.tables:
        if table.kind == "boq":
            reqs += _table_requirements(table)[0]
    rules_ok = sum(1 for item, r in zip(truth.items, reqs) if r["quantity"] == item.quantity)

    old = [_flat(c) for c in _old_chunks(str(path))]
    old_ok = sum(1 for item in truth.items
                 if any(_flat(item.description.split(",")[0]) in c and str(item.quantity) in c.replace(",", "")
                        for c in old))

    # Cold index build (empty embedding cache) and retrieval.
    with patch("embedding_cache.EMBEDDING_CACHE_DIR", str(cache_dir / name)):
        rfp_rag.clear_rfp_tools()
        t = time.perf_counter()
        tool = rfp_rag.get_rfp_rag_tool(rfp_rag.RfpRagConfig(pdf_path=str(path)))
        index_s = _secs(t)
    latencies, row_hits = [], 0
    for item in truth.items:
        t = time.perf_counter()
        results = tool.retrieve(f"How many {item.description} are required?", top_k=TOP_K)
        latencies.append(time.perf_counter() - t)
        key, qty = M.norm(item.description)[:25], f"{item.quantity:,}"
        row_hits += any(key in M.norm(r) and qty in r for r in results)
    fact_hits = sum(expected.lower() in tool.build_context(q, top_k=TOP_K).lower()
                    for q, expected in truth.facts.items())
    return {
        "pages": doc.page_count,
        "tables": [f"{t.id}:{t.kind}" for t in doc.tables],
        "parse_s": parse_s,
        "table_reader_qty_accuracy": M.ratio(rules_ok, len(truth.items)),
        "old_chunker_upper_bound": M.ratio(old_ok, len(truth.items)),
        "chunks": len(tool.chunks),
        "index_build_cold_s": index_s,
        "retrieval_row_hit_at_6": M.ratio(row_hits, len(truth.items)),
        "retrieval_fact_hit_at_6": M.ratio(fact_hits, len(truth.facts)),
        "retrieval_latency_mean_ms": round(statistics.mean(latencies) * 1000, 1),
    }


def local_models() -> Dict[str, Any]:
    from models import get_embedder, get_reranker

    out = {}
    t = time.perf_counter()
    emb = get_embedder(RFP_EMBED_MODEL)
    out["embedder_load_s"] = _secs(t)
    texts = [f"Sample chunk {i}: 6A/10A 1 way switch, quantity {i * 10} Nos" for i in range(64)]
    t = time.perf_counter()
    emb.encode(texts, normalize_embeddings=True)
    out["embed_ms_per_chunk"] = round((time.perf_counter() - t) * 1000 / len(texts), 2)
    t = time.perf_counter()
    rr = get_reranker(BU_RERANK_MODEL)
    out["reranker_load_s"] = _secs(t)
    t = time.perf_counter()
    rr.predict([("16A switch with indicator", f"Product {i}: 16A 1Way Switch With Ind.") for i in range(20)])
    out["rerank_ms_per_20_pairs"] = round((time.perf_counter() - t) * 1000, 1)
    out["models"] = {"embedder": RFP_EMBED_MODEL, "reranker": BU_RERANK_MODEL}
    return out


# ─── Live: the real workflow ──────────────────────────────────────────────────

def live_sample(name: str, path: Path, truth, wf, with_proposal: bool) -> Dict[str, Any]:
    from catalogue import get_catalogue
    from proposal_writer import amount_warnings

    products = get_catalogue().all_products()
    by_id = {p.id: p for p in products}
    pid = f"eval-{name}-{uuid.uuid4().hex[:6]}"
    out: Dict[str, Any] = {"stages": {}}

    def step(stage: str, fn: Callable[[], Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        with llm_usage.UsageTracker() as usage:
            status = fn()
        out["stages"][stage] = {
            "seconds": round(usage.elapsed_s, 2),
            "llm": llm_usage.summarize(usage.records),
            "error": status.get("error"),
        }
        if status["stage"] == "failed":
            out["failed_at"] = stage
            return None
        return status

    status = step("extract", lambda: wf.start(pid, str(path), rfp_meta={"title": truth.title, "buyer": truth.buyer}))
    if status is None:
        return out
    state = status["state"]
    extraction = M.score_extraction(state.get("rfp_requirements", []), truth)
    item_by_req = extraction.pop("item_by_req")
    out["extraction"] = extraction
    out["summary"] = M.score_summary(state.get("rfp_answer", ""), truth)
    out["warnings"] = status.get("warnings", [])

    status = step("match", lambda: wf.resume(pid, {"note": "evaluation: approve as extracted"}))
    if status is None:
        return out
    out["matching"] = M.score_matching(status["state"].get("matched_items", []), truth, item_by_req, products)

    status = step("price", lambda: wf.resume(pid, {"accept_all": True, "note": "evaluation: accept all"}))
    if status is None:
        return out
    st = status["state"]
    out["pricing"] = M.score_pricing(st.get("line_items", []), st.get("pricing_summary", {}), PRICING_RULES,
                                     by_id, truth, item_by_req)
    if not with_proposal:
        return out

    status = step("draft", lambda: wf.resume(pid, {"note": "evaluation: keep pricing"}))
    if status is None:
        return out
    st = status["state"]
    sections = st.get("proposal_sections", [])
    out["proposal"] = M.score_proposal(sections, st.get("matched_items", []), amount_warnings(sections))

    status = step("render", lambda: wf.resume(pid, {"note": "evaluation: keep draft"}))
    if status is not None:
        pdf = status["state"].get("report_pdf_path")
        out["proposal"]["pdf_rendered"] = bool(pdf and Path(pdf).is_file())
    return out


def run_live(samples: Dict[str, Any], model: str, gemini_model: str, with_proposal: bool, tmp: Path) -> Dict[str, Any]:
    import rfp_rag
    import workflow

    os.environ["LLM_MODEL"] = model
    os.environ["GEMINI_MODEL"] = gemini_model
    rfp_rag.clear_rfp_tools()
    ckpt = tmp / f"checkpoints-{uuid.uuid4().hex[:6]}.db"
    with patch("workflow.PROPOSALS_DIR", str(tmp / "proposals")):
        wf = workflow.ProjectWorkflow(checkpoint_path=str(ckpt))
        try:
            results = {}
            for name, (path, truth) in samples.items():
                print(f"  [{model}] {name} …", flush=True)
                t = time.perf_counter()
                results[name] = live_sample(name, path, truth, wf, with_proposal)
                results[name]["total_seconds"] = _secs(t)
        finally:
            wf.close()
    return {"llm_model": model, "gemini_model": gemini_model, "samples": results,
            "aggregate": aggregate(results)}


def aggregate(results: Dict[str, Any]) -> Dict[str, Any]:
    def pick(*keys: str) -> List[Any]:
        vals = []
        for r in results.values():
            v: Any = r
            for k in keys:
                v = v.get(k) if isinstance(v, dict) else None
            vals.append(v)
        return vals

    def total(stage: Optional[str], field: str) -> Any:
        vals = []
        for r in results.values():
            stages = r.get("stages", {})
            for s, info in stages.items():
                if stage is None or s == stage:
                    vals.append(info["llm"]["total"].get(field) or 0)
        return round(sum(vals), 6)

    stages = ["extract", "match", "price", "draft", "render"]
    return {
        "samples": len(results),
        "failed": [n for n, r in results.items() if r.get("failed_at")],
        "extraction_item_recall": M.mean(pick("extraction", "item_recall")),
        "extraction_qty_accuracy": M.mean(pick("extraction", "qty_accuracy")),
        "extraction_term_recall": M.mean(pick("extraction", "term_recall")),
        "extraction_precision_proxy": M.mean(pick("extraction", "precision_proxy")),
        "summary_fact_recall": M.mean(pick("summary", "fact_recall")),
        "matching_top1": M.mean(pick("matching", "top1_accuracy")),
        "matching_top3": M.mean(pick("matching", "top3_accuracy")),
        "matching_high_conf_accuracy": M.mean(pick("matching", "accuracy_by_confidence", "high", "accuracy")),
        "matching_auto_accept_errors": sum(v or 0 for v in pick("matching", "auto_accept_errors")),
        "gaps_detected": f"{sum(v or 0 for v in pick('matching', 'gaps_detected'))}/"
                         f"{sum(v or 0 for v in pick('matching', 'gaps_expected'))}",
        "pricing_arithmetic_exact": M.mean(pick("pricing", "arithmetic_exact")),
        "pricing_qty_matches_rfp": M.mean(pick("pricing", "qty_matches_rfp")),
        "proposal_amount_mentions": sum(v or 0 for v in pick("proposal", "amount_mentions")),
        "proposal_unconfirmed_claims": sum(v or 0 for v in pick("proposal", "unconfirmed_claims")),
        "seconds_total": round(sum(v or 0 for v in pick("total_seconds")), 1),
        "seconds_by_stage": {s: round(sum((r.get("stages", {}).get(s) or {}).get("seconds", 0)
                                          for r in results.values()), 1) for s in stages},
        "llm_calls": total(None, "calls"),
        "llm_failed_attempts": total(None, "failed_attempts"),
        "llm_fallback_calls": total(None, "fallback_calls"),
        "prompt_tokens": total(None, "prompt_tokens"),
        "completion_tokens": total(None, "completion_tokens"),
        "cost_usd": total(None, "cost_usd"),
        "cost_usd_by_stage": {s: total(s, "cost_usd") for s in stages},
        "calls_by_stage": {s: total(s, "calls") for s in stages},
    }


# ─── Report ───────────────────────────────────────────────────────────────────

def _pct(v: Any) -> str:
    return "–" if v is None else f"{v * 100:.1f}%" if isinstance(v, float) else str(v)


def write_report(result: Dict[str, Any], path: Path) -> None:
    lines = [f"# Synapse evaluation · {result['started']}", ""]
    lines += ["## Offline (no LLM)", "", "| Sample | Pages | Tables | Table reader qty | Old chunker (upper bound) | "
              "Row hit@6 | Fact hit@6 | Parse s | Cold index s | Retrieval ms |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, o in result["offline"]["samples"].items():
        lines.append(f"| {name} | {o['pages']} | {len(o['tables'])} | {_pct(o['table_reader_qty_accuracy'])} | "
                     f"{_pct(o['old_chunker_upper_bound'])} | {_pct(o['retrieval_row_hit_at_6'])} | "
                     f"{_pct(o['retrieval_fact_hit_at_6'])} | {o['parse_s']} | {o['index_build_cold_s']} | "
                     f"{o['retrieval_latency_mean_ms']} |")
    lm = result["offline"]["local_models"]
    lines += ["", f"Local models: embedder load {lm['embedder_load_s']} s, {lm['embed_ms_per_chunk']} ms/chunk; "
                  f"reranker load {lm['reranker_load_s']} s, {lm['rerank_ms_per_20_pairs']} ms per 20 pairs.", ""]
    for run in result.get("live", []):
        a = run["aggregate"]
        lines += [f"## Live · {run['llm_model']} (proposal: {run['gemini_model']})", ""]
        lines += ["| Metric | Value |", "|---|---|"]
        for k, v in a.items():
            if isinstance(v, dict):
                v = ", ".join(f"{s}: {x}" for s, x in v.items())
            elif isinstance(v, float) and k.startswith(("extraction", "summary", "matching", "pricing")):
                v = _pct(v)
            lines.append(f"| {k} | {v} |")
        for name, s in run["samples"].items():
            lines += ["", f"### {name}", ""]
            for key in ("extraction", "summary", "matching", "pricing", "proposal"):
                if key in s:
                    lines.append(f"- **{key}**: " + json.dumps(s[key], ensure_ascii=False)[:1500])
            if s.get("failed_at"):
                lines.append(f"- **failed at** {s['failed_at']}: {s['stages'][s['failed_at']]['error']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Evaluate the Synapse agents on the sample RFPs.")
    parser.add_argument("--samples", default="hotel,school,office")
    parser.add_argument("--llm-models", default=os.environ.get("LLM_MODEL") or LLM_MODEL,
                        help="comma-separated OpenRouter model ids to compare")
    parser.add_argument("--gemini-model", default=os.environ.get("GEMINI_MODEL") or GEMINI_MODEL)
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--no-proposal", action="store_true")
    parser.add_argument("--out", default=str(Path(VAR_DIR) / "eval" / datetime.now().strftime("%Y%m%d-%H%M%S")))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="synapse-eval-"))
    wanted = [s.strip() for s in args.samples.split(",") if s.strip()]
    samples = {k: v for k, v in build_all(tmp / "samples").items() if k in wanted}

    result: Dict[str, Any] = {"started": datetime.now().isoformat(timespec="seconds"), "samples": list(samples)}
    print("Offline evaluation …", flush=True)
    result["offline"] = {
        "local_models": local_models(),
        "samples": {n: offline_sample(n, p, t, tmp / "cache") for n, (p, t) in samples.items()},
    }
    if not args.offline_only:
        result["live"] = []
        for model in [m.strip() for m in args.llm_models.split(",") if m.strip()]:
            print(f"Live evaluation with {model} …", flush=True)
            result["live"].append(run_live(samples, model, args.gemini_model, not args.no_proposal, tmp))

    (out / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_report(result, out / "report.md")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nResults: {out / 'results.json'}\nReport:  {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
