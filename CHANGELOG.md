# Changelog

## v0.1.0 — 2026-10-27 (unreleased)

Production RAG with evals — the October flagship project.

### What it is

A FastAPI service (`rag_service/api.py`) that answers questions over a
help-center corpus with cited evidence: hybrid BM25 + vector retrieval,
cross-encoder reranking, and an evidence-numbered generation step with
dangling-citation verification. Ships with Docker and a GitHub Actions CI
pipeline (ruff + pytest + offline latency probe).

### Eval numbers (golden set, n=10, dated 2026-10-17)

| metric | target | result |
|---|---|---|
| faithfulness (grounding proxy) | ≥ 0.90 | 1.000 |
| answer relevance (keyword coverage) | ≥ 0.85 | 1.000 (10/10) |
| end-to-end p95 latency | ≤ 1500 ms | 6445.7 ms on CPU — first-query model warmup; steady-state 691–1820 ms |
| cost per 1k tokens (blended) | ≤ $0.004 | $0.00019 (modeled embed cost; no LLM configured) |

Full run notes: `evals/baseline_report.md`.

### Highlights this release

- Cross-encoder reranker behind a pluggable `Scorer` protocol
  (sentence-transformers in production, deterministic fake in tests).
- Citation-grounded answers: `verify_citations` flags dangling references.
- `/ready` readiness probe distinct from `/health` liveness; X-Request-ID
  middleware echoed on every response.
- Eval budget gate (`--fail-on-budget`): grounding/coverage floors, p95
  latency, and cost budgets — exits 1 for CI.
- `--warmup-question` on `scripts/run_eval.py` isolates steady-state
  latency from first-query model warmup.
- One-command Docker demo runbook (`scripts/demo.sh`).

### Known limitations

- p95 latency misses the 1500 ms budget on CPU (bge-reranker-base on the
  critical path). The lever for v0.2.0: GPU inference or a smaller reranker.
- Generation defaults to the extractive baseline; a generative LLM is
  configured via `RAG_LLM_API_KEY`.
