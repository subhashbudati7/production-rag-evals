"""Tests for rag_service.retrieval."""

import numpy as np
import pytest

from rag_service.retrieval import (
    BM25Index,
    HybridRetriever,
    InMemoryVectorStore,
    tokenize,
)


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, WORLD! error_code 404.") == [
        "hello",
        "world",
        "error",
        "code",
        "404",
    ]


def test_bm25_ranks_term_repetition_first():
    docs = [
        "the cat sat on the mat",
        "dogs dogs dogs are great pets for dogs",
        "a completely unrelated sentence about oceans",
    ]
    idx = BM25Index().fit(docs)
    hits = idx.search("dogs", top_k=3)
    assert hits[0][0] == 1
    assert all(s >= 0 for _, s in hits)


def test_bm25_exact_term_match_beats_semantic_noise():
    docs = [
        "refund policy: contact support within thirty days for a refund",
        "our company values collaboration and synergy going forward",
    ]
    idx = BM25Index().fit(docs)
    assert idx.search("refund policy", top_k=2)[0][0] == 0


def test_bm25_empty_query_scores_zero():
    idx = BM25Index().fit(["some text here"])
    assert idx.search("", top_k=5) == []
    assert idx.scores("") == [0.0]


def test_bm25_is_deterministic():
    docs = ["alpha beta gamma", "beta beta delta"]
    idx = BM25Index().fit(docs)
    assert idx.scores("beta") == idx.scores("beta")


def test_bm25_rejects_bad_params():
    with pytest.raises(ValueError):
        BM25Index(k1=0)
    with pytest.raises(ValueError):
        BM25Index(b=1.5)


class KeywordEmbedder:
    """Deterministic test embedder: normalized keyword-count vectors."""

    VOCAB = ("refund", "policy", "dog", "ocean", "collaboration")

    def __call__(self, texts):
        mat = np.zeros((len(texts), len(self.VOCAB)))
        for i, t in enumerate(texts):
            toks = t.lower()
            for j, w in enumerate(self.VOCAB):
                mat[i, j] = toks.count(w)
        return mat


def _store(ids_texts):
    store = InMemoryVectorStore(KeywordEmbedder())
    ids, texts = zip(*ids_texts)
    store.add(list(ids), list(texts))
    return store


def test_vector_store_finds_semantic_match():
    store = _store([("a", "refund policy details here"), ("b", "ocean waves and tides")])
    hits = store.search("how do refunds work", top_k=2)
    assert hits[0][0] == "a"
    assert len(store) == 2


def test_vector_store_empty_query_returns_nothing():
    store = _store([("a", "refund policy")])
    assert store.search("   ", top_k=2) == []


def test_vector_store_rejects_mismatched_ids():
    store = InMemoryVectorStore(KeywordEmbedder())
    with pytest.raises(ValueError):
        store.add(["a"], ["one", "two"])


def _hybrid(alpha):
    docs = [
        ("c1", "refund policy: thirty day refund window for all purchases"),
        ("c2", "our values emphasize collaboration across teams"),
        ("c3", "the ocean covers most of the planet surface"),
    ]
    ids, texts = zip(*docs)
    return HybridRetriever(
        BM25Index(),
        InMemoryVectorStore(KeywordEmbedder()),
        list(ids),
        list(texts),
        alpha=alpha,
    )


def test_hybrid_alpha_one_matches_vector_only():
    hybrid = _hybrid(alpha=1.0)
    vec = InMemoryVectorStore(KeywordEmbedder())
    vec.add(
        ["c1", "c2", "c3"],
        [
            "refund policy: thirty day refund window for all purchases",
            "our values emphasize collaboration across teams",
            "the ocean covers most of the planet surface",
        ],
    )
    assert [c.chunk_id for c in hybrid.search("refund", 3)] == [
        i for i, s in vec.search("refund", 3) if s > 0
    ]


def test_hybrid_alpha_zero_matches_bm25_only():
    hybrid = _hybrid(alpha=0.0)
    bm25 = BM25Index().fit(
        [
            "refund policy: thirty day refund window for all purchases",
            "our values emphasize collaboration across teams",
            "the ocean covers most of the planet surface",
        ]
    )
    assert [c.chunk_id for c in hybrid.search("refund policy", 3)] == [
        f"c{i + 1}" for i, _ in bm25.search("refund policy", 3)
    ]


def test_hybrid_scores_bounded_and_ranked():
    hybrid = _hybrid(alpha=0.6)
    hits = hybrid.search("refund ocean", top_k=2)
    assert [c.chunk_id for c in hits] == ["c1", "c3"]
    assert all(0.0 < c.score <= 1.0 for c in hits)
    assert hits[0].score >= hits[1].score
    assert hits[0].text  # chunk text is carried through
    assert hits[0].bm25_score > 0 and hits[0].vector_score > 0


def test_hybrid_empty_query_returns_nothing():
    assert _hybrid(alpha=0.6).search("   ", top_k=5) == []


def test_hybrid_rejects_bad_alpha():
    with pytest.raises(ValueError):
        _hybrid(alpha=1.5)
