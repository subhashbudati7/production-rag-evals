"""Tests for rag_service.rerank."""

import pytest

from rag_service.rerank import RerankedChunk, Reranker
from rag_service.retrieval import RetrievedChunk


class KeywordScorer:
    """Deterministic fake: score = shared keyword count with the query."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(query.lower().split())
        return [float(len(q & set(t.lower().split()))) for t in texts]


def _candidates() -> list[RetrievedChunk]:
    return [
        RetrievedChunk("c1", "refund policy thirty days", 0.9, 0.9, 0.9),
        RetrievedChunk("c2", "company values and synergy", 0.8, 0.8, 0.8),
        RetrievedChunk("c3", "contact support for refund help", 0.7, 0.7, 0.7),
    ]


def test_rerank_puts_most_relevant_first():
    ranked = Reranker(KeywordScorer()).rerank("refund policy", _candidates(), top_k=3)
    assert [c.chunk_id for c in ranked] == ["c1", "c3", "c2"]
    assert all(isinstance(c, RerankedChunk) for c in ranked)


def test_rerank_assigns_one_based_ranks():
    ranked = Reranker(KeywordScorer()).rerank("refund policy", _candidates(), top_k=2)
    assert [c.rank for c in ranked] == [1, 2]
    assert ranked[0].rerank_score >= ranked[1].rerank_score


def test_rerank_limits_to_top_k():
    ranked = Reranker(KeywordScorer()).rerank("refund", _candidates(), top_k=1)
    assert len(ranked) == 1
    assert ranked[0].rank == 1


def test_rerank_empty_query_or_candidates_returns_empty():
    r = Reranker(KeywordScorer())
    assert r.rerank("   ", _candidates(), top_k=3) == []
    assert r.rerank("refund", [], top_k=3) == []


def test_rerank_rejects_non_positive_top_k():
    with pytest.raises(ValueError, match="top_k must be positive"):
        Reranker(KeywordScorer()).rerank("refund", _candidates(), top_k=0)


def test_rerank_rejects_mismatched_scorer_output():
    class BadScorer:
        def score(self, query, texts):
            return [1.0]

    with pytest.raises(ValueError, match="scores for"):
        Reranker(BadScorer()).rerank("refund", _candidates(), top_k=3)


def test_rerank_ties_break_deterministically_on_chunk_id():
    tied = [
        RetrievedChunk("c9", "same words here", 0.5, 0.5, 0.5),
        RetrievedChunk("c1", "same words here", 0.5, 0.5, 0.5),
    ]
    ranked = Reranker(KeywordScorer()).rerank("same words here", tied, top_k=2)
    assert [c.chunk_id for c in ranked] == ["c1", "c9"]
