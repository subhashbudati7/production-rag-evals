"""Tests for rag_service.api."""

import numpy as np
from fastapi.testclient import TestClient

from rag_service.api import build_retriever, create_app, index_docs_dir
from rag_service.config import Settings
from rag_service.ingest import Document
from rag_service.rerank import Reranker


class KeywordEmbedder:
    VOCAB = ("refund", "policy", "ocean")

    def __call__(self, texts):
        mat = np.zeros((len(texts), len(self.VOCAB)))
        for i, t in enumerate(texts):
            low = t.lower()
            for j, w in enumerate(self.VOCAB):
                mat[i, j] = low.count(w)
        return mat


class ReverseScorer:
    """Fake reranker: reverses candidate order so rerank effects are visible."""

    def score(self, query, texts):
        return [float(-i) for i in range(len(texts))]


class StubLLM:
    def __init__(self):
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return "Refunds are allowed for thirty days [1]."


def _docs():
    return [
        Document(
            id="d1",
            title="Refunds",
            source="s",
            text="refund policy: thirty day refund window for purchases. "
            "more refund policy details here to force a second chunk. "
            "even more refund text to overflow the chunk size limit.",
        ),
        Document(
            id="d2", title="Oceans", source="s", text="the ocean covers most of the planet surface"
        ),
    ]


def _full_client():
    settings = Settings(top_k=10, rerank_top_k=2)
    retriever = build_retriever(_docs(), settings, KeywordEmbedder())
    reranker = Reranker(ReverseScorer())
    return TestClient(create_app(settings, retriever, reranker, StubLLM()))


def test_query_with_reranker_reorders_evidence():
    resp = _full_client().post("/query", json={"query": "refund policy"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["generated"] is True
    assert body["answer"] == "Refunds are allowed for thirty days [1]."
    assert body["citations_valid"] is True


def test_query_rerank_top_k_caps_citations():
    resp = _full_client().post("/query", json={"query": "refund"})
    assert resp.status_code == 200
    assert len(resp.json()["citations"]) <= 2


def test_query_without_llm_returns_evidence_only():
    settings = Settings()
    retriever = build_retriever(_docs(), settings, KeywordEmbedder())
    resp = TestClient(create_app(settings, retriever)).post("/query", json={"query": "refund"})
    body = resp.json()
    assert resp.status_code == 200
    assert body["answer"] is None
    assert body["generated"] is False
    assert body["citations"], "expected evidence even without an LLM"


def _client():
    settings = Settings()
    retriever = build_retriever(_docs(), settings, KeywordEmbedder())
    return TestClient(create_app(settings, retriever))


def test_health():
    resp = TestClient(create_app(Settings())).get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"


def test_query_returns_citations_without_generation():
    resp = _client().post("/query", json={"query": "refund policy"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "refund policy"
    assert body["answer"] is None
    assert body["generated"] is False
    assert body["citations"], "expected at least one citation"
    assert body["citations"][0]["chunk_id"].startswith("d1")
    assert body["latency_ms"] >= 0


def test_query_without_index_is_503():
    resp = TestClient(create_app(Settings())).post("/query", json={"query": "refund"})
    assert resp.status_code == 503


def test_query_rejects_empty_string():
    resp = _client().post("/query", json={"query": ""})
    assert resp.status_code == 422


def test_query_top_k_is_respected():
    resp = _client().post("/query", json={"query": "refund", "top_k": 1})
    assert resp.status_code == 200
    assert len(resp.json()["citations"]) == 1


def test_index_docs_dir_reads_files(tmp_path):
    (tmp_path / "a.md").write_text("# Refunds\nrefund policy text here")
    retriever = index_docs_dir(str(tmp_path), Settings(), KeywordEmbedder())
    hits = retriever.search("refund", top_k=5)
    assert hits
    assert "refund" in hits[0].text


def test_index_docs_dir_requires_embedder(tmp_path):
    try:
        index_docs_dir(str(tmp_path), Settings(), None)
    except RuntimeError as exc:
        assert "no embedder configured" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
