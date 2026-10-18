#!/usr/bin/env python3
"""Run the golden eval set through the full RAG pipeline and print a report.

Usage:
    python scripts/run_eval.py --docs evals/docs --eval evals/golden.jsonl

Retrieval embeds with the ``embeddings`` extra (bge-small-en-v1.5); rerank
uses the cross-encoder unless --skip-rerank is passed. Generation uses the extractive baseline, so the suite runs deterministically
without API keys — set RAG_LLM_API_KEY to use the OpenAI-compatible
generator instead, or swap in a real generator by pointing --llm-module at a
dotted path to an LLMClient instance (e.g. ``myapp.llm:client``).

The first query through a model-backed pipeline pays one-time warmup cost
(weight loading, kernel init — on CPU it can dominate the p95). Pass
--warmup-question to run one question through the pipeline untimed before
the measured loop, so the reported latencies reflect steady state.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_service.api import build_retriever
from rag_service.config import Settings
from rag_service.eval import EvalCase, EvalReport, run_eval
from rag_service.generate import LLMClient, build_prompt, generate_answer
from rag_service.ingest import load_documents
from rag_service.models import HFEmbedder, OpenAILLM
from rag_service.rerank import CrossEncoderScorer, RerankedChunk, Reranker


def load_llm(dotted: str | None, sentences: int, settings: Settings) -> LLMClient:
    from rag_service.generate import ExtractiveLLM

    if dotted is not None:
        module_name, _, attr = dotted.partition(":")
        if not attr:
            raise ValueError("--llm-module must look like 'module:attr'")
        return getattr(importlib.import_module(module_name), attr)
    if settings.llm_api_key:
        return OpenAILLM(
            settings.generator_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout_s=settings.llm_timeout_s,
        )
    return ExtractiveLLM(sentences=sentences)


def run_eval_cases(
    cases: list[EvalCase],
    docs,
    settings: Settings,
    embedder,
    reranker: Reranker | None,
    llm: LLMClient,
    *,
    warmup_question: str | None = None,
    time_fn=None,
) -> EvalReport:
    """Build the pipeline from the given stages and run every case.

    ``warmup_question`` runs through the pipeline once before the measured
    loop (untimed, excluded from the report) so first-query model warmup
    does not pollute the p95. Testable without models by injecting fakes.
    """
    retriever = build_retriever(docs, settings, embedder)

    def pipeline(question: str) -> tuple[str, list[str], int, int]:
        hits = retriever.search(question, settings.top_k)
        if reranker is not None:
            evidence = reranker.rerank(question, hits, settings.rerank_top_k)
        else:
            evidence = [
                RerankedChunk(h.chunk_id, h.text, h.score, i + 1)
                for i, h in enumerate(hits[: settings.rerank_top_k])
            ]
        prompt = build_prompt(question, evidence)
        result = generate_answer(question, evidence, llm)
        return result.answer, [c.text for c in evidence], len(prompt), len(result.answer)

    if warmup_question is not None:
        pipeline(warmup_question)
    return run_eval(cases, pipeline, settings, time_fn=time_fn)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the golden RAG eval set.")
    parser.add_argument("--docs", default="evals/docs")
    parser.add_argument("--eval", default="evals/golden.jsonl")
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--llm-module", default=None)
    parser.add_argument("--extractive-sentences", type=int, default=2)
    parser.add_argument(
        "--warmup-question",
        default=None,
        help="run this question through the pipeline once before the timed "
        "loop (untimed, excluded from the report) so first-query model "
        "warmup does not pollute the p95",
    )
    parser.add_argument(
        "--fail-on-budget",
        action="store_true",
        help="exit 1 when the report violates a quality/latency/cost budget (CI gate)",
    )
    args = parser.parse_args()

    settings = Settings()
    docs = load_documents(args.docs)
    if not docs:
        print(f"no documents found under {args.docs}", file=sys.stderr)
        return 2
    cases = [
        EvalCase(c["question"], c.get("expected_keywords", []))
        for line in Path(args.eval).read_text().splitlines()
        if line.strip()
        for c in [json.loads(line)]
    ]
    if not cases:
        print(f"no eval cases found in {args.eval}", file=sys.stderr)
        return 2

    retriever_embedder = HFEmbedder(settings.embedding_model)
    reranker = None if args.skip_rerank else Reranker(CrossEncoderScorer(settings.reranker_model))
    llm = load_llm(args.llm_module, args.extractive_sentences, settings)

    report = run_eval_cases(
        cases,
        docs,
        settings,
        retriever_embedder,
        reranker,
        llm,
        warmup_question=args.warmup_question,
        time_fn=time.perf_counter,
    )
    print(report.markdown_table())
    if args.fail_on_budget:
        violations = report.budget_violations(settings)
        if violations:
            for violation in violations:
                print(f"budget violation: {violation}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
