"""End-to-end pipeline test: docs -> retrieval -> rerank -> generate -> eval.

Uses deterministic fakes for the embedder, reranker, and LLM so the full
pipeline shape is exercised in CI; the model-backed variants are covered
by scripts/run_eval.py against the same golden set.
"""

import json
from pathlib import Path

import numpy as np

from rag_service.api import build_retriever
from rag_service.config import Settings
from rag_service.eval import EvalCase, run_eval
from rag_service.generate import ExtractiveLLM, build_prompt, generate_answer
from rag_service.ingest import load_documents
from rag_service.rerank import RerankedChunk, Reranker

EVALS_DIR = Path(__file__).resolve().parent.parent / "evals"


class DocVocabEmbedder:
    VOCAB = (
        "refund",
        "shipping",
        "warranty",
        "account",
        "days",
        "support",
        "order",
        "damage",
    )

    def __call__(self, texts):
        mat = np.zeros((len(texts), len(self.VOCAB)))
        for i, t in enumerate(texts):
            low = t.lower()
            for j, w in enumerate(self.VOCAB):
                mat[i, j] = low.count(w)
        return mat


class OverlapScorer:
    def score(self, query, texts):
        q = set(query.lower().split())
        return [float(len(q & set(t.lower().split()))) for t in texts]


def _pipeline_parts():
    settings = Settings(top_k=10, rerank_top_k=3)
    retriever = build_retriever(
        load_documents(str(EVALS_DIR / "docs")), settings, DocVocabEmbedder()
    )
    reranker = Reranker(OverlapScorer())
    llm = ExtractiveLLM()
    return settings, retriever, reranker, llm


def test_full_pipeline_produces_valid_cited_answers():
    settings, retriever, reranker, llm = _pipeline_parts()
    for line in (EVALS_DIR / "golden.jsonl").read_text().splitlines():
        case = json.loads(line)
        hits = retriever.search(case["question"], settings.top_k)
        assert hits, f"no retrieval hits for: {case['question']}"
        evidence = reranker.rerank(case["question"], hits, settings.rerank_top_k)
        result = generate_answer(case["question"], evidence, llm)
        assert result.citation_check.valid, f"dangling citations for: {case['question']}"
        assert result.answer


def test_full_pipeline_eval_report_is_sane():
    settings, retriever, reranker, llm = _pipeline_parts()
    cases = [
        EvalCase(**json.loads(line))
        for line in (EVALS_DIR / "golden.jsonl").read_text().splitlines()
    ]

    def pipeline(question: str):
        hits = retriever.search(question, settings.top_k)
        evidence = reranker.rerank(question, hits, settings.rerank_top_k)
        prompt = build_prompt(question, evidence)
        result = generate_answer(question, evidence, llm)
        return result.answer, [c.text for c in evidence], len(prompt), len(result.answer)

    report = run_eval(cases, pipeline, settings)
    assert report.n == 6
    assert report.grounding > 0.6
    assert report.keyword_coverage > 0.5
    assert report.latency_p95() >= 0
    print("\n" + report.markdown_table())


def test_reranked_chunk_shape_matches_generate_contract():
    c = RerankedChunk("doc:0", "some text", 1.0, 1)
    assert c.rank == 1
    assert build_prompt("q", [c]).startswith("Answer the question")
