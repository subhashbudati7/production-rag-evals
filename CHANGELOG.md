# Changelog

## v0.1.0 — 2026-10-27 (unreleased)

Production RAG with evals — the October flagship project.

### What it is

A FastAPI service (`rag_service/api.py`) that answers questions over a
help-center corpus with cited evidence: hybrid BM25 + vector retrieval,
cross-encoder reranking, and an evidence-numbered generation step with
dangling-citation verification. Ships with Docker and a GitHub Actions CI
pipeline (ruff + pytest + offline latency probe).

### Eval numbers (golden set, n=10, dated 2026-10-25, warmup excluded)

| metric | target | result |
|---|---|---|
| faithfulness (grounding proxy) | ≥ 0.90 | 1.000 |
| answer relevance (keyword coverage) | ≥ 0.85 | 1.000 (10/10) |
| end-to-end p95 latency | ≤ 1500 ms | 1443.0 ms on CPU (p50 708.4 ms) — steady-state with `--warmup-question`; first-query warmup excluded |
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

- p95 latency is within budget on CPU only with warmup excluded
  (1443.0 ms vs the 1500 ms budget — thin headroom; a slower machine
  could breach it). The lever for v0.2.0: GPU inference or a smaller
  reranker.
- Generation defaults to the extractive baseline; a generative LLM is
  configured via `RAG_LLM_API_KEY`.
