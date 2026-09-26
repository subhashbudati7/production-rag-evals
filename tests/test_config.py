"""Tests for rag_service.config."""

import pytest

from rag_service.config import Settings


def test_defaults_are_sane():
    s = Settings()
    assert s.chunk_size == 800
    assert s.chunk_overlap < s.chunk_size
    assert s.rerank_top_k <= s.top_k
    assert 0 <= s.hybrid_alpha <= 1
    assert 0 <= s.bm25_b <= 1


def test_env_override(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "30")
    monkeypatch.setenv("RAG_GENERATOR_MODEL", "gpt-4o")
    s = Settings()
    assert s.top_k == 30
    assert s.generator_model == "gpt-4o"


def test_rerank_top_k_cannot_exceed_top_k():
    with pytest.raises(ValueError):
        Settings(top_k=5, rerank_top_k=10)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        Settings(chunk_size=100, chunk_overlap=100)


def test_unknown_env_vars_are_ignored(monkeypatch):
    monkeypatch.setenv("RAG_SOMETHING_WEIRD", "1")
    Settings()  # should not raise


def test_environment_is_constrained():
    with pytest.raises(ValueError):
        Settings(environment="moon")
    assert Settings(environment="production").environment == "production"
