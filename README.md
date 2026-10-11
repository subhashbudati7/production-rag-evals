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
| answer relevance (keyword coverage) | ≥ 0.85 | 1.000 (6/6) — the one miss was fixed by stopword-aware sentence scoring (regression test `test_extractive_llm_scores_content_words_over_stopwords`) |
| retrieval p50 latency | ≤ 300 ms | 0.1 ms pipeline overhead with the offline deterministic stack (`scripts/measure_latency.py`, n=6); production model inference not included |
| end-to-end p95 latency | ≤ 1500 ms | pipeline overhead 10.1 ms (offline stack). Real stack: 498024.7 ms incl. one-time model download; steady-state 0.7–4.2 s on CPU (over budget — GPU/smaller reranker is the lever) |
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

## Configuration

Every knob is an environment variable prefixed `RAG_` (or a `.env` file) —
no code changes needed to retune retrieval, swap models, or tighten
budgets. The authoritative list lives in `rag_service/config.py`.

| variable | default | what it does |
|---|---|---|
| `RAG_ENVIRONMENT` | `development` | `development` / `staging` / `production`; surfaced on `/health` |
| `RAG_CHUNK_SIZE` | `800` | target chunk size in characters |
| `RAG_CHUNK_OVERLAP` | `120` | overlap between consecutive chunks |
| `RAG_MIN_CHUNK_CHARS` | `50` | drop trailing chunks shorter than this |
| `RAG_TOP_K` | `20` | candidates pulled from hybrid retrieval |
| `RAG_RERANK_TOP_K` | `5` | candidates kept after reranking (≤ `RAG_TOP_K`) |
| `RAG_BM25_K1` | `1.2` | BM25 term-frequency saturation |
| `RAG_BM25_B` | `0.75` | BM25 length normalization |
| `RAG_HYBRID_ALPHA` | `0.6` | weight of the vector score in fusion; BM25 gets `1 - alpha` |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | dense embedding model |
| `RAG_RERANKER_MODEL` | `BAAI/bge-reranker-base` | cross-encoder reranker model |
| `RAG_GENERATOR_MODEL` | `gpt-4o-mini` | OpenAI-compatible chat model |
| `RAG_LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible chat endpoint |
| `RAG_LLM_API_KEY` | *(unset)* | when unset, the endpoint serves evidence citations without generated answers |
| `RAG_LLM_TIMEOUT_S` | `60` | generation request timeout |
| `RAG_RETRIEVAL_P50_BUDGET_MS` | `300` | budget the eval suite checks retrieval p50 against |
| `RAG_END_TO_END_P95_BUDGET_MS` | `1500` | budget the eval suite checks end-to-end p95 against |
| `RAG_COST_PER_1K_INPUT_TOKENS` | `0.00015` | USD per 1k input tokens for the cost model |
| `RAG_COST_PER_1K_OUTPUT_TOKENS` | `0.0006` | USD per 1k output tokens for the cost model |
| `RAG_COST_PER_1K_EMBED_TOKENS` | `0.00002` | USD per 1k embed tokens for the cost model |
| `RAG_DOCS_DIR` | `docs` | documents indexed at startup (mount here in Docker) |
| `RAG_AUTO_BUILD_INDEX_ON_STARTUP` | `true` | lifespan builds any uninjected pipeline stage |
| `RAG_WARMUP_MODELS_ON_STARTUP` | `false` | eagerly load weights at startup (slow boot, fast first query) |

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

Probes: `GET /health` is liveness (always 200 while the process serves);
`GET /ready` is readiness — 200 only once the index is built, 503 before
that. Every response carries an `X-Request-ID` header (propagated from
the client when sent, minted otherwise) for log correlation.

CI (`.github/workflows/ci.yml`) runs ruff + the deterministic test suite
+ the offline latency probe on every push. The model-backed eval doubles
as a release gate: `python scripts/run_eval.py --fail-on-budget` exits 1
when grounding, keyword coverage, p95 latency, or cost breach the budgets
in the eval table above.

## Status

October 2026 build — working branch `rag/october-build`, merged to `main`
at deploy time (Oct 27). See commit history for the build log.
