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
