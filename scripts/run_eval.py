#!/usr/bin/env python3
"""Run the golden eval set through the full RAG pipeline and print a report.

Usage:
    python scripts/run_eval.py --docs evals/docs --eval evals/golden.jsonl

Retrieval embeds with the ``embeddings`` extra (bge-small-en-v1.5); rerank
uses the cross-encoder unless --skip-rerank is passed. Generation uses the
extractive baseline, so the suite runs deterministically without API keys —
swap in a real generator by pointing --llm-module at a dotted path to an
LLMClient instance (e.g. ``myapp.llm:client``).
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
from rag_service.eval import EvalCase, run_eval
from rag_service.generate import LLMClient, build_prompt, generate_answer
from rag_service.ingest import load_documents
from rag_service.rerank import CrossEncoderScorer, RerankedChunk, Reranker


class HFEmbedder:
    """Production embedder backed by sentence-transformers."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed; run "
                "pip install -e '.[embeddings]' first"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def __call__(self, texts: list[str]):
        return self._model.encode(texts, convert_to_numpy=True)


def load_llm(dotted: str | None, sentences: int) -> LLMClient:
    from rag_service.generate import ExtractiveLLM

    if dotted is None:
        return ExtractiveLLM(sentences=sentences)
    module_name, _, attr = dotted.partition(":")
    if not attr:
        raise ValueError("--llm-module must look like 'module:attr'")
    return getattr(importlib.import_module(module_name), attr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the golden RAG eval set.")
    parser.add_argument("--docs", default="evals/docs")
    parser.add_argument("--eval", default="evals/golden.jsonl")
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--llm-module", default=None)
    parser.add_argument("--extractive-sentences", type=int, default=2)
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

    retriever = build_retriever(docs, settings, HFEmbedder(settings.embedding_model))
    reranker = None if args.skip_rerank else Reranker(CrossEncoderScorer(settings.reranker_model))
    llm = load_llm(args.llm_module, args.extractive_sentences)

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

    report = run_eval(cases, pipeline, settings, time_fn=time.perf_counter)
    print(report.markdown_table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
