#!/usr/bin/env python3
"""
Evaluate the stage-1 RAG chat pipeline against eval/testset.json.

Metrics
  retrieval_accuracy  share of questions whose evidence passage is in the top-k chunks
  doc_hit_rate        share whose expected *document* appears anywhere in the top-k
  mrr                 mean reciprocal rank of the first chunk containing the evidence
  answer_accuracy     share of answers containing the key fact (text check or Claude grader)
  citation_accuracy   share of answers that cite the expected document's filename
  avg_ttft_s          mean seconds from question to first streamed token (also p50/p95)

Examples (run from backend/)
  python eval/run_eval.py --retrieval-only                       # fast, no API key needed
  python eval/run_eval.py                                        # full run, needs ANTHROPIC_API_KEY
  python eval/run_eval.py --grader claude                        # Claude grades the answers
  python eval/run_eval.py --config eval/configs/rerank.json      # try a variant
  python eval/run_eval.py --sweep eval/configs/*.json            # compare variants (retrieval)
  python eval/run_eval.py --retrieval-only --compare eval/results/baseline_retrieval.json --max-drop 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # backend/ -> import rag

from rag import ANTHROPIC_API_URL, RAGConfig, Retriever, answer_question  # noqa: E402

import requests  # noqa: E402

ACCURACY_METRICS = ["retrieval_accuracy", "doc_hit_rate", "mrr", "answer_accuracy", "citation_accuracy"]
PASSAGE_OVERLAP = 0.5  # a chunk "contains" the evidence if it covers >= 50% of it


# --------------------------------------------------------------------------
# Test set + corpus
# --------------------------------------------------------------------------

def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to one space; return the string and, for each
    output char, its index in the original text."""
    out, idx, prev_space = [], [], False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            prev_space = True
        else:
            out.append(ch)
            prev_space = False
        idx.append(i)
    return "".join(out), idx


def locate(doc_text: str, evidence: str) -> tuple[int, int] | None:
    norm_doc, idx = _normalize_with_map(doc_text)
    norm_ev = " ".join(evidence.split())
    pos = norm_doc.find(norm_ev)
    if pos < 0:
        return None
    return idx[pos], idx[pos + len(norm_ev) - 1] + 1


def load_testset(path: Path, limit: int | None = None):
    data = json.loads(path.read_text(encoding="utf-8"))
    corpus_dir = (path.parent / data.get("corpus_dir", "corpus")).resolve()
    docs = {p.name: p.read_text(encoding="utf-8") for p in sorted(corpus_dir.iterdir()) if p.is_file()}
    questions = data["questions"][:limit] if limit else data["questions"]
    problems = []
    for q in questions:
        if q["expected_document"] not in docs:
            problems.append(f"{q['id']}: document {q['expected_document']} not in {corpus_dir}")
            continue
        span = locate(docs[q["expected_document"]], q["evidence"])
        if span is None:
            problems.append(f"{q['id']}: evidence not found verbatim in {q['expected_document']}")
        q["_span"] = span
    if problems:
        raise SystemExit("Test set problems:\n  " + "\n  ".join(problems))
    return questions, docs


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def passage_rank(hits: list[dict], expected_doc: str, span: tuple[int, int]) -> int | None:
    """1-based rank of the first hit that covers the evidence passage."""
    ev_start, ev_end = span
    for rank, h in enumerate(hits, 1):
        m = h["metadata"]
        if m["filename"] != expected_doc:
            continue
        overlap = min(ev_end, m["end_pos"]) - max(ev_start, m["start_pos"])
        if overlap <= 0:
            continue
        denom = min(ev_end - ev_start, m["end_pos"] - m["start_pos"])
        if overlap / denom >= PASSAGE_OVERLAP:
            return rank
    return None


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def text_check(answer: str, key_fact: str, aliases: list[str]) -> bool:
    a = _norm(answer)
    return any(_norm(k) in a for k in [key_fact, *aliases] if k)


def doc_names(filename: str) -> list[str]:
    stem = filename.rsplit(".", 1)[0]
    return [filename.lower(), stem.lower()]


def cites(answer: str, filename: str) -> bool:
    a = answer.lower()
    return any(n in a for n in doc_names(filename))


GRADER_PROMPT = """You are grading a research assistant's answer against a reference.

Question: {question}
Reference passage: {evidence}
Key fact the answer must convey: {key_fact}

Answer to grade:
<answer>
{answer}
</answer>

Does the answer correctly convey the key fact (paraphrases count; extra correct detail is fine; a contradiction or omission is not)?
Reply with only JSON: {{"correct": true or false, "reason": "<one short sentence>"}}"""


def claude_grade(q: dict, answer: str, api_key: str, model: str) -> tuple[bool, str]:
    prompt = GRADER_PROMPT.format(question=q["question"], evidence=q["evidence"],
                                  key_fact=q["key_fact"], answer=answer)
    for attempt in range(4):
        r = requests.post(
            ANTHROPIC_API_URL,
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": model, "max_tokens": 200, "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        if r.status_code in (429, 500, 502, 503, 529):
            time.sleep(2 ** attempt * 2)
            continue
        r.raise_for_status()
        text = "".join(b.get("text", "") for b in r.json()["content"])
        m = re.search(r"\{.*\}", text, re.S)
        try:
            verdict = json.loads(m.group(0)) if m else {}
            return bool(verdict.get("correct")), str(verdict.get("reason", ""))
        except json.JSONDecodeError:
            return False, f"unparseable grader output: {text[:120]}"
    return False, "grader failed after retries"


def answer_with_retries(retriever: Retriever, question: str, api_key: str) -> dict:
    for attempt in range(4):
        out = answer_question(retriever, question, api_key)
        err = out["error"]
        if not err:
            return out
        retryable = isinstance(err, dict) and (
            err.get("error", {}).get("type") in ("overloaded_error", "rate_limit_error", "api_error")
            or err.get("status") in (429, 500, 502, 503, 529)
        )
        if not retryable or attempt == 3:
            return out
        time.sleep(2 ** attempt * 3)
    return out


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------

def build_index(cfg: RAGConfig, docs: dict[str, str]) -> tuple[Retriever, int, float]:
    t0 = time.perf_counter()
    retriever = Retriever(cfg)
    n_chunks = 0
    for doc_id, (name, text) in enumerate(docs.items(), 1):
        n_chunks += len(retriever.add_document(doc_id, name, text))
    return retriever, n_chunks, time.perf_counter() - t0


def pct(xs: list[bool]) -> float:
    return round(100.0 * sum(xs) / len(xs), 1) if xs else 0.0


def run(cfg: RAGConfig, questions, docs, *, retrieval_only: bool, grader: str,
        api_key: str | None, grader_model: str, verbose: bool = True) -> dict:
    retriever, n_chunks, index_s = build_index(cfg, docs)
    if verbose:
        print(f"[{cfg.name}] indexed {len(docs)} docs into {n_chunks} chunks in {index_s:.1f}s", file=sys.stderr)

    rows = []
    for i, q in enumerate(questions, 1):
        row = {"id": q["id"], "question": q["question"], "expected_document": q["expected_document"]}
        if retrieval_only:
            t0 = time.perf_counter()
            hits = retriever.search(q["question"])
            row["retrieval_s"] = time.perf_counter() - t0
        else:
            out = answer_with_retries(retriever, q["question"], api_key)
            hits = out["hits"]
            row.update({k: out[k] for k in ("answer", "error", "retrieval_s", "ttft_s", "total_s")})

        rank = passage_rank(hits, q["expected_document"], q["_span"])
        row["retrieved"] = [
            {"filename": h["metadata"]["filename"], "chunk_index": h["metadata"]["chunk_index"],
             "distance": round(h["distance"], 4), **({"rerank_score": round(h["rerank_score"], 3)} if "rerank_score" in h else {})}
            for h in hits
        ]
        row["passage_rank"] = rank
        row["retrieval_hit"] = rank is not None
        row["doc_hit"] = any(h["metadata"]["filename"] == q["expected_document"] for h in hits)

        if not retrieval_only:
            answer = row.get("answer") or ""
            row["answer_text_check"] = text_check(answer, q["key_fact"], q.get("aliases", []))
            if grader == "claude" and answer:
                row["answer_correct"], row["grader_reason"] = claude_grade(q, answer, api_key, grader_model)
            else:
                row["answer_correct"] = row["answer_text_check"]
            row["citation_correct"] = cites(answer, q["expected_document"])
            row["cited_other_docs"] = [d for d in docs if d != q["expected_document"] and cites(answer, d)]

        rows.append(row)
        if verbose:
            mark = "✓" if row["retrieval_hit"] else "✗"
            extra = ""
            if not retrieval_only:
                extra = (f" answer={'✓' if row['answer_correct'] else '✗'}"
                         f" cite={'✓' if row['citation_correct'] else '✗'}"
                         f" ttft={row['ttft_s']:.2f}s" if row.get("ttft_s") else f" ERROR {row.get('error')}")
            print(f"  [{i:>2}/{len(questions)}] {q['id']:<11} retrieval={mark}{extra}", file=sys.stderr)

    metrics = {
        "n_questions": len(rows),
        "retrieval_accuracy": pct([r["retrieval_hit"] for r in rows]),
        "doc_hit_rate": pct([r["doc_hit"] for r in rows]),
        "mrr": round(100.0 * statistics.mean(1 / r["passage_rank"] if r["passage_rank"] else 0 for r in rows), 1),
        "n_chunks": n_chunks,
        "index_build_s": round(index_s, 2),
        "avg_retrieval_ms": round(1000 * statistics.mean(r["retrieval_s"] for r in rows), 1),
    }
    if not retrieval_only:
        ttfts = sorted(r["ttft_s"] for r in rows if r.get("ttft_s"))
        metrics.update({
            "answer_accuracy": pct([r["answer_correct"] for r in rows]),
            "answer_accuracy_text_check": pct([r["answer_text_check"] for r in rows]),
            "citation_accuracy": pct([r["citation_correct"] for r in rows]),
            "errors": sum(1 for r in rows if r.get("error")),
            "avg_ttft_s": round(statistics.mean(ttfts), 3) if ttfts else None,
            "p50_ttft_s": round(statistics.median(ttfts), 3) if ttfts else None,
            "p95_ttft_s": round(ttfts[min(len(ttfts) - 1, int(0.95 * len(ttfts)))], 3) if ttfts else None,
        })
    return {"metrics": metrics, "rows": rows}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


METRIC_LABELS = {
    "retrieval_accuracy": "Retrieval accuracy (passage in top-k)",
    "doc_hit_rate": "Right document in top-k",
    "mrr": "MRR (×100)",
    "answer_accuracy": "Answer accuracy",
    "citation_accuracy": "Citation accuracy",
    "avg_ttft_s": "Avg time to first token (s)",
    "p95_ttft_s": "p95 time to first token (s)",
}


def compare(current: dict, baseline: dict, max_drop: float) -> tuple[list[str], list[str]]:
    lines, failures = [], []
    for key, label in METRIC_LABELS.items():
        cur, base = current.get(key), baseline.get(key)
        if cur is None or base is None:
            continue
        delta = cur - base
        sign = "+" if delta >= 0 else ""
        unit = "s" if key.endswith("_s") else " pts"
        lines.append(f"| {label} | {base} | {cur} | {sign}{round(delta, 3)}{unit} |")
        if key in ACCURACY_METRICS and key != "mrr" and -delta > max_drop:
            failures.append(f"{label} fell {abs(round(delta, 1))} pts ({base} → {cur}); allowed drop is {max_drop}")
    return lines, failures


def to_markdown(result: dict, baseline: dict | None, failures: list[str], cmp_lines: list[str]) -> str:
    m, cfg = result["metrics"], result["config"]
    out = [f"## RAG eval — `{cfg['name']}`",
           f"{m['n_questions']} questions · {m['n_chunks']} chunks · chunker `{cfg['chunker']}` "
           f"{cfg['chunk_size']}/{cfg['chunk_overlap']} · top-{cfg['top_k']} · rerank `{cfg['rerank']}` · "
           f"embedder `{cfg['embedding_model']}` · grader `{result['grader']}`", ""]
    if baseline and cmp_lines:
        out += ["| Metric | Baseline | This run | Δ |", "|---|---|---|---|", *cmp_lines, ""]
    else:
        out += ["| Metric | Value |", "|---|---|"]
        out += [f"| {label} | {m[k]} |" for k, label in METRIC_LABELS.items() if m.get(k) is not None]
        out.append("")
    out.append("**❌ Regression:** " + "; ".join(failures) if failures else "✅ No regression beyond the allowed drop.")
    misses = [r for r in result["rows"] if not r["retrieval_hit"]]
    if misses:
        out += ["", f"<details><summary>{len(misses)} retrieval misses</summary>", ""]
        out += [f"- `{r['id']}` {r['question']} → got {', '.join(h['filename'] for h in r['retrieved'])}" for r in misses]
        out += ["", "</details>"]
    wrong = [r for r in result["rows"] if r.get("answer_correct") is False]
    if wrong:
        out += ["", f"<details><summary>{len(wrong)} answers missing the key fact</summary>", ""]
        out += [f"- `{r['id']}` {r['question']}" + (f" — {r['grader_reason']}" if r.get("grader_reason") else "") for r in wrong]
        out += ["", "</details>"]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testset", default=str(HERE / "testset.json"))
    ap.add_argument("--config", help="RAG config JSON (default: production config from rag.py / RAG_* env)")
    ap.add_argument("--sweep", nargs="+", metavar="CONFIG", help="compare several configs (retrieval only)")
    ap.add_argument("--retrieval-only", action="store_true", help="skip Claude; no API key needed")
    ap.add_argument("--grader", choices=["text", "claude"], default="text")
    ap.add_argument("--grader-model", default=os.getenv("EVAL_GRADER_MODEL"))
    ap.add_argument("--embedder", help="override embedding model (e.g. 'hashing' for offline smoke tests)")
    ap.add_argument("--limit", type=int, help="only the first N questions")
    ap.add_argument("--out", help="write full results JSON here")
    ap.add_argument("--markdown", help="write a Markdown summary here (e.g. $GITHUB_STEP_SUMMARY)")
    ap.add_argument("--compare", help="baseline results JSON to compare against")
    ap.add_argument("--max-drop", type=float, default=3.0, help="fail if an accuracy metric drops more than this many points")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(HERE.parent / ".env")
    except ImportError:
        pass

    questions, docs = load_testset(Path(args.testset), args.limit)

    # ---- sweep mode: retrieval metrics for several configs side by side
    if args.sweep:
        results = []
        for path in args.sweep:
            cfg = RAGConfig.from_file(path)
            if args.embedder:
                cfg.embedding_model = args.embedder
            res = run(cfg, questions, docs, retrieval_only=True, grader="text", api_key=None,
                      grader_model="", verbose=not args.quiet)
            results.append({"config": cfg.to_dict(), "metrics": res["metrics"]})
        results.sort(key=lambda r: (r["metrics"]["retrieval_accuracy"], r["metrics"]["mrr"]), reverse=True)
        table = ["| Config | Chunker | Size/overlap | Rerank | Chunks | Retrieval acc. | Doc hit | MRR | ms/query |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for r in results:
            c, m = r["config"], r["metrics"]
            table.append(f"| {c['name']} | {c['chunker']} | {c['chunk_size']}/{c['chunk_overlap']} | {c['rerank']} | "
                         f"{m['n_chunks']} | {m['retrieval_accuracy']}% | {m['doc_hit_rate']}% | {m['mrr']} | {m['avg_retrieval_ms']} |")
        md = f"## Retrieval sweep ({len(questions)} questions, top-k as configured)\n\n" + "\n".join(table) + "\n"
        print(md)
        if args.markdown:
            with open(args.markdown, "a", encoding="utf-8") as fh:
                fh.write(md)
        if args.out:
            Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        return 0

    # ---- single run
    cfg = RAGConfig.from_file(args.config) if args.config else RAGConfig.from_env()
    if args.embedder:
        cfg.embedding_model = args.embedder
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not args.retrieval_only and not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set. Use --retrieval-only, or export the key.")

    res = run(cfg, questions, docs, retrieval_only=args.retrieval_only, grader=args.grader,
              api_key=api_key, grader_model=args.grader_model or cfg.llm_model, verbose=not args.quiet)
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "mode": "retrieval" if args.retrieval_only else "full",
        "grader": "none" if args.retrieval_only else args.grader,
        "config": cfg.to_dict(),
        **res,
    }

    baseline, failures, cmp_lines = None, [], []
    if args.compare and Path(args.compare).exists():
        baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        cmp_lines, failures = compare(res["metrics"], baseline["metrics"], args.max_drop)
    elif args.compare:
        print(f"(no baseline at {args.compare}; reporting without comparison)", file=sys.stderr)

    md = to_markdown(result, baseline, failures, cmp_lines)
    print(md)
    if args.markdown:
        with open(args.markdown, "a", encoding="utf-8") as fh:
            fh.write(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
