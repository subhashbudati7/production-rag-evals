# production-rag-evals

A production-grade retrieval-augmented generation (RAG) service built to
answer questions over a document corpus with **citations, measured quality,
and known cost** — the things that separate a demo from something you can
actually ship.

## Problem

Naive RAG (embed everything, top-k cosine, stuff into a prompt) fails in
production in predictable ways: it retrieves the wrong chunks, the model
answers from memory instead of the retrieved context, latency spikes on long
contexts, and nobody can tell you what an answer costs. This project fixes
each of those with boring, measurable engineering:

- **Hybrid retrieval** — BM25 keyword search fused with dense vector search,
  so exact terms (error codes, names, IDs) and semantic matches both count.
- **Cross-encoder reranking** — a second, stronger model re-scores the top
  candidates before generation.
- **Citations** — every factual claim in an answer points at the chunk(s)
  it came from; ungrounded claims are flagged, not hidden.
- **An eval suite that runs in CI** — faithfulness, answer relevance,
  p50/p95 latency, and cost per 1k tokens, tracked over time so regressions
  are caught, not discovered.

## Architecture

```
                    ┌──────────────┐
  documents ──▶     │    ingest    │  loaders + overlap chunking
                    └──────┬───────┘
                           ▼
              ┌────────────────────────┐
              │   hybrid retrieval     │  BM25 ─┐
              │  (top_k candidates)    │        ├─▶ score fusion
              │                        │  vector ┘
              └───────────┬────────────┘
                          ▼
              ┌────────────────────────┐
              │  cross-encoder rerank  │  keep top rerank_top_k
              └───────────┬────────────┘
                          ▼
              ┌────────────────────────┐
              │      generation        │  answer + per-claim citations
              └───────────┬────────────┘
                          ▼
                   ┌─────────────┐
                   │  eval suite │  faithfulness / relevance /
                   │             │  latency / cost — every change measured
                   └─────────────┘
```

Served as a FastAPI app (`rag_service/api.py`), packaged with Docker.

## Eval results

Tracked per release; the numbers below are from the latest eval run against
the golden question set (`evals/golden.jsonl`).

| metric | target | latest (2026-10-03, n=6, extractive baseline) |
|---|---|---|
| faithfulness (grounding proxy) | ≥ 0.90 | 1.000 — baseline quotes evidence verbatim; discriminating once a generative LLM is used |
| answer relevance (keyword coverage) | ≥ 0.85 | 0.833 (5/6; one miss from stopword-counting sentence scorer — fix planned) |
| retrieval p50 latency | ≤ 300 ms | not yet measured separately from end-to-end |
| end-to-end p95 latency | ≤ 1500 ms | 498024.7 ms incl. one-time model download; steady-state 0.7–4.2 s on CPU (over budget — GPU/smaller reranker is the lever) |
| cost per 1k tokens (blended) | ≤ $0.004 | $0.00020 (modeled embed cost; no LLM configured) |

Full run notes: `evals/baseline_report.md`. Rerun the suite (needs the
`embeddings` extra; uses the extractive baseline generator unless you
point `--llm-module` at a real client or set `RAG_LLM_API_KEY`):

```bash
python scripts/run_eval.py --docs evals/docs --eval evals/golden.jsonl
```

## Cost model

Cost is a first-class output of the eval suite, not an afterthought.
`rag_service/eval.py` prices every run from measured token counts and the
configured per-1k-token rates, so a "better" model that doubles cost shows up
as a tradeoff, not a silent regression.

## Layout

```
rag_service/      the service (config, ingest, retrieval, rerank, generate, api, eval)
tests/            pytest suite — one module per service module + end-to-end pipeline test
evals/            golden dataset, sample docs corpus, eval runner outputs
scripts/          run_eval.py — golden eval set through the full pipeline
Dockerfile        production image (uvicorn)
```

## Quickstart

```bash
pip install -e ".[embeddings,dev]"
pytest
uvicorn rag_service.api:app --host 0.0.0.0 --port 8000
```

## Status

October 2026 build — working branch `rag/october-build`, merged to `main`
at deploy time (Oct 27). See commit history for the build log.
