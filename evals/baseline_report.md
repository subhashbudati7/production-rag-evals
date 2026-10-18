# Baseline eval report — 2026-10-03

First model-backed run of the golden set (n=6) through the full pipeline:
hybrid BM25 + dense retrieval, cross-encoder rerank, extractive baseline
generator. No API keys; everything ran on CPU in this environment.

## Configuration

- Embeddings: `BAAI/bge-small-en-v1.5` via sentence-transformers
- Reranker: `BAAI/bge-reranker-base` cross-encoder
- Generator: extractive baseline (`ExtractiveLLM`, 2 sentences by query-token overlap)
- Corpus: `evals/docs/help-center.md` (733 bytes → 1 chunk at 800-char chunks)
- Retrieval: `top_k=20`, `rerank_top_k=5`, `hybrid_alpha=0.6`
- Command: `python scripts/run_eval.py --docs evals/docs --eval evals/golden.jsonl`

## Results

| question | grounding | keyword coverage | latency ms | cost $/1k tok |
|---|---|---|---|---|
| How long do I have to request a refund? | 1.000 | 0.000 | 498024.7 | 0.00020 |
| Are digital downloads refundable? | 1.000 | 1.000 | 4176.2 | 0.00019 |
| How fast is standard shipping? | 1.000 | 1.000 | 3909.1 | 0.00020 |
| What does the hardware warranty cover? | 1.000 | 1.000 | 1589.7 | 0.00020 |
| Does the warranty cover accidental damage? | 1.000 | 1.000 | 758.1 | 0.00020 |
| How do I protect my account? | 1.000 | 1.000 | 708.4 | 0.00020 |

**Aggregate (n=6):** grounding 1.000, keyword coverage 0.833,
latency p50 2749.4ms / p95 498024.7ms, cost $0.00020/1k tokens.

## Reading the numbers

- **grounding = 1.000 is a baseline artifact, not a model achievement.**
  The extractive generator quotes evidence sentences verbatim, so every
  content token trivially overlaps the evidence. The metric becomes
  discriminating once a generative LLM is plugged in (`--llm-module` or
  `RAG_LLM_API_KEY`).
- **keyword coverage missed 1 of 6.** "How long do I have to request a
  refund?" returned a warranty sentence instead of the "thirty days"
  sentence: the baseline's sentence scorer counts stopwords, so sharing
  "to"/"a" outranked sharing "refund". Documented failure mode of the
  baseline — not of retrieval (the right chunk was retrieved and reranked
  first). Fix planned: stopword-aware sentence scoring.
- **latency p95 (498s) is the one-time cross-encoder weight download**
  inside the first timed query, not serving latency. Steady-state CPU
  latencies on this box were 0.7–4.2s per query — over the 1500ms
  end-to-end budget, as expected for CPU cross-encoder reranking of 20
  candidates. Levers: GPU, a smaller reranker, or fewer rerank candidates.
- **cost is modeled, not billed:** $0.00020/1k tokens at the configured
  embedding rate — two orders of magnitude under the $0.004 budget.
  Real generator pricing applies once an LLM is configured.

## Addendum — 2026-10-10

- **Keyword coverage is now 6/6.** The stopword-aware sentence scorer
  (committed 2026-10-04, pinned by regression test
  `test_extractive_llm_scores_content_words_over_stopwords`) fixed the
  refund-question miss, so answer relevance on the golden set is 1.000.
- **Per-stage latency is now measured.** `/query` reports
  `retrieval_ms` / `rerank_ms` / `generation_ms`, and
  `scripts/measure_latency.py` runs the golden set through the real
  endpoint with a deterministic offline stack: retrieval p50 0.1ms /
  p95 9.6ms, rerank p50 0.1ms, generation p50 0.2ms, end-to-end p50
  0.3ms / p95 10.1ms. These are pipeline-overhead numbers (no model
  inference); they hold the service accountable for its own code while
  the real-stack CPU latencies above hold it accountable for its models.

## Addendum — 2026-10-11

- **Golden set expanded 6 → 10 cases.** The corpus gained four
  help-center sections (order tracking, gift cards, support hours,
  international shipping) with one golden question each, so retrieval
  and generation are now exercised beyond refunds/shipping/warranty.
  The table above remains the n=6 model-backed run; a fresh n=10 run
  with real models is part of the Oct 27 deploy checklist.
- **The eval suite is now a gate, not just a report.**
  `EvalReport.budget_violations` checks grounding, keyword coverage,
  p95 latency, and cost against the documented budgets, and
  `scripts/run_eval.py --fail-on-budget` exits 1 on any violation.

## Addendum — 2026-10-17

- **Offline probe refreshed for n=10.** `scripts/measure_latency.py` over
  the expanded golden set (deterministic offline stack, no model
  inference): retrieval p50 0.1ms / p95 0.3ms, rerank p50 0.1ms,
  generation p50 0.2ms / p95 0.5ms, end-to-end p50 0.4ms / p95 0.9ms.
  The README eval table now reports these instead of the n=6 numbers.

## Addendum — 2026-10-17 (n=10 model-backed rerun)

Fresh full-pipeline run over all 10 golden cases with real models on CPU
(extractive baseline generator, no API keys). Command:

```bash
export HF_HUB_CACHE=/home/hatch/.hf-cache/hub HF_HUB_OFFLINE=1
export NO_PROXY="localhost,127.0.0.1,::1,198.19.0.1,198.19.0.2" no_proxy="$NO_PROXY"
python scripts/run_eval.py --docs evals/docs --eval evals/golden.jsonl
```

- Embeddings: `BAAI/bge-small-en-v1.5`; reranker: `BAAI/bge-reranker-base`.
- **No downloads were needed.** Both weight sets were already cached at
  `/home/hatch/.hf-cache/hub` — this VM's HF CDN endpoint
  (`cdn-lfs.huggingface.co`) is blocked at the egress proxy, and the
  stock `no_proxy` value additionally crashes huggingface_hub's httpx
  (`InvalidURL: Invalid port: ':1]'`), so future runs must export the
  sanitized `NO_PROXY` above and point `HF_HUB_CACHE` at the existing
  cache; `HF_HUB_OFFLINE=1` proves nothing is fetched.
- Results: grounding **1.000**, keyword coverage **1.000** (10/10),
  latency p50 **1318.8ms** / p95 **6445.7ms**, cost **$0.00019**/1k tokens.
- The p95 is the first question (6445.7ms — model warmup); steady-state
  per-question latency is 691–1820ms. p95 remains over the 1500ms budget
  on CPU — same known condition as the Oct 3 run; GPU or a smaller
  reranker is the lever. `--fail-on-budget` would exit 1 on p95 alone.
