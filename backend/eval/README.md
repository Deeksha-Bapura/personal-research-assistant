# RAG evaluation suite (stage 1 chat)

Measures how well the document-chat pipeline in `backend/rag.py` (used by
`server.py`'s `/api/chat`) finds and uses the right passage.

## What's here

| Path | What it is |
|---|---|
| `corpus/` | 9 Python Enhancement Proposals (PEPs 8, 257, 343, 484, 492, 498, 572, 636, 3333), all public domain. Overlapping topics (typing, async, strings) make retrieval genuinely hard. |
| `testset.json` | 47 questions. Each has `expected_document`, a verbatim `evidence` passage from it, and a `key_fact` (+ `aliases`) the answer must contain. |
| `run_eval.py` | Runs every question through the pipeline and scores it. |
| `configs/` | Pipeline variants to compare (chunk size, overlap, sentence-aware chunking, cross-encoder reranking). `baseline.json` = current production settings. |
| `results/` | Committed baseline results the CI gate compares against. |

## Metrics

- **Retrieval accuracy** – the evidence passage is in the top 3 chunks (a chunk counts if it covers ≥ 50% of the passage, so this works for any chunk size).
- **Answer accuracy** – the answer contains the key fact. `--grader text` (default) does a case-insensitive substring check against the key fact and aliases; `--grader claude` has Claude judge it (handles paraphrases). Both are recorded.
- **Citation accuracy** – the answer names the expected document's filename (the system prompt asks Claude to cite filenames).
- **Time to first token** – seconds from the question arriving to the first streamed text, including retrieval. Mean, p50 and p95.
- Also reported: right *document* in top 3, MRR, chunk count, ms per query.

## Running it (from `backend/`)

```bash
pip install -r eval/requirements.txt
python eval/run_eval.py --retrieval-only                 # ~1 min, no API key
export ANTHROPIC_API_KEY=sk-ant-...
python eval/run_eval.py --grader claude                  # full run, ~47 Claude calls (+47 grader calls)
python eval/run_eval.py --sweep eval/configs/*.json      # compare all variants on retrieval
```

Useful flags: `--config FILE`, `--limit N`, `--out results.json`, `--markdown summary.md`,
`--compare baseline.json --max-drop 3` (exits 1 on regression).

## The before/after experiment

1. Record the baseline (current production settings):
   ```bash
   python eval/run_eval.py --retrieval-only --out eval/results/baseline_retrieval.json
   python eval/run_eval.py --grader claude  --out eval/results/baseline_full.json
   ```
2. Find the best change: `python eval/run_eval.py --sweep eval/configs/*.json`
3. Run the winner fully: `python eval/run_eval.py --grader claude --config eval/configs/<winner>.json --compare eval/results/baseline_full.json`
4. Make it the production default by changing the `RAGConfig` defaults in `rag.py` (or set `RAG_CONFIG=eval/configs/<winner>.json` for the server), then re-record the baselines in `results/` so CI guards the new level.

## CI

`.github/workflows/eval.yml` runs on every PR touching `rag.py`, `server.py` or `eval/`:
the retrieval eval always runs and fails the PR if accuracy drops > 3 points below
`results/baseline_retrieval.json`; the full Claude eval runs when the repo has an
`ANTHROPIC_API_KEY` secret (fails on a > 5 point drop vs `results/baseline_full.json`).
Results appear in the run's summary page and as downloadable artifacts. Trigger it
manually from the Actions tab to also run the config sweep.

## Adding your own documents

Drop `.txt`/`.md` files into `corpus/` and add questions to `testset.json`. The
`evidence` must be copied verbatim from the document (whitespace differences are
fine); the script refuses to run if it can't find it.
