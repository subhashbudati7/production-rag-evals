#!/usr/bin/env python3
"""Probe per-stage latency of the /query pipeline over the golden set.

Usage:
    python scripts/measure_latency.py --docs evals/docs --eval evals/golden.jsonl

Runs every golden question through the real FastAPI /query endpoint and
reports p50/p95/max for each reported stage (retrieval, rerank,
generation) plus end-to-end latency. The pipeline is built on a
deterministic, fully offline stack (keyword-count embedder, token-overlap
rerank scorer, extractive baseline generator), so this measures the
*pipeline's own* overhead and serves as a regression check — production
inference latencies need the real models and are tracked separately in
the eval table (README.md).

``--json`` prints the same summary as JSON for machine consumption.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from rag_service.api import build_retriever, create_app
from rag_service.config import Settings
from rag_service.generate import ExtractiveLLM
from rag_service.ingest import load_documents
from rag_service.rerank import Reranker


class KeywordEmbedder:
    """Deterministic keyword-count embedder; vocab comes from the eval set."""

    def __init__(self, vocab: list[str]):
        self.vocab = vocab

    def __call__(self, texts):
        return np.array(
            [[float(text.lower().count(word)) for word in self.vocab] for text in texts],
            dtype=np.float64,
        )


class OverlapScorer:
    """Deterministic token-overlap scorer for the reranker."""

    def score(self, query, texts):
        query_tokens = set(query.lower().split())
        return [float(len(query_tokens & set(text.lower().split()))) for text in texts]


#: Timed stages reported by the /query endpoint, in pipeline order.
STAGES = ("retrieval_ms", "rerank_ms", "generation_ms", "latency_ms")


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile of a non-empty value list."""
    if not values:
        raise ValueError("percentile() needs at least one value")
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int((pct / 100.0) * len(ordered))))
    return ordered[rank]


def load_questions(path: str) -> list[str]:
    return [
        json.loads(line)["question"] for line in Path(path).read_text().splitlines() if line.strip()
    ]


def _build_vocab(eval_path: str) -> list[str]:
    words: set[str] = set()
    for line in Path(eval_path).read_text().splitlines():
        if not line.strip():
            continue
        for keyword in json.loads(line).get("expected_keywords", []):
            words.add(str(keyword).lower())
    return sorted(words)


def _collect(
    questions: list[str], docs, settings: Settings, vocab: list[str]
) -> dict[str, list[float]]:
    retriever = build_retriever(docs, settings, KeywordEmbedder(vocab))
    reranker = Reranker(OverlapScorer())
    llm = ExtractiveLLM()
    app = create_app(
        settings=settings,
        retriever=retriever,
        reranker=reranker,
        llm=llm,
    )
    latencies: dict[str, list[float]] = {stage: [] for stage in STAGES}
    with TestClient(app) as client:
        for question in questions:
            response = client.post("/query", json={"query": question})
            response.raise_for_status()
            body = response.json()
            for stage, values in latencies.items():
                values.append(float(body[stage]))
    return latencies


def summarize(latencies: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    return {
        stage: {
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
            "max": percentile(values, 100),
        }
        for stage, values in latencies.items()
    }


def render_table(summary: dict[str, dict[str, float]], n: int) -> str:
    lines = [
        f"per-stage latency over {n} golden questions (ms, offline deterministic stack):",
        "",
        "| stage        | p50    | p95    | max    |",
        "|--------------|--------|--------|--------|",
    ]
    for stage, stats in summary.items():
        lines.append(
            f"| {stage:<12} | {stats['p50']:>6.1f} | {stats['p95']:>6.1f} | {stats['max']:>6.1f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe /query per-stage latency.")
    parser.add_argument("--docs", default="evals/docs")
    parser.add_argument("--eval", default="evals/golden.jsonl")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args()

    questions = load_questions(args.eval)
    if not questions:
        print(f"no eval cases found in {args.eval}", file=sys.stderr)
        return 2
    settings = Settings(
        top_k=10,
        rerank_top_k=3,
        auto_build_index_on_startup=False,
    )
    docs = load_documents(args.docs)
    if not docs:
        print(f"no documents found under {args.docs}", file=sys.stderr)
        return 2
    vocab = _build_vocab(args.eval)
    latencies = _collect(questions, docs, settings, vocab)
    summary = summarize(latencies)
    if args.json:
        print(json.dumps({"n": len(questions), **summary}, indent=2))
    else:
        print(render_table(summary, len(questions)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
