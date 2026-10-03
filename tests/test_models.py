"""Tests for production model adapters (all network/model access faked)."""

from __future__ import annotations

import sys

import httpx
import pytest

from rag_service.models import HFEmbedder, OpenAILLM


def _mock_llm(content: str, status: int = 200) -> OpenAILLM:
    def handler(request: httpx.Request) -> httpx.Response:
        import json

        assert request.headers["Authorization"] == "Bearer sk-test"
        body = json.loads(request.read().decode())
        assert body["model"] == "gpt-4o-mini"
        assert body["messages"] == [{"role": "user", "content": "q"}]
        return httpx.Response(status, json={"choices": [{"message": {"content": content}}]})

    llm = OpenAILLM("gpt-4o-mini", api_key="sk-test")
    llm._client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.openai.com/v1",
        headers={"Authorization": "Bearer sk-test"},
        trust_env=False,
    )
    return llm


def test_openai_llm_returns_content():
    assert _mock_llm("The refund window is 30 days [1].").complete("q") == (
        "The refund window is 30 days [1]."
    )


def test_openai_llm_strips_whitespace():
    assert _mock_llm("  answer [1]\n").complete("q") == "answer [1]"


def test_openai_llm_rejects_empty_api_key():
    with pytest.raises(ValueError, match="api_key"):
        OpenAILLM("gpt-4o-mini", api_key="")


def test_openai_llm_raises_on_http_error():
    llm = _mock_llm("x", status=429)
    with pytest.raises(httpx.HTTPStatusError):
        llm.complete("q")


def test_openai_llm_raises_on_malformed_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    llm = OpenAILLM("gpt-4o-mini", api_key="sk-test")
    llm._client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.openai.com/v1",
        headers={"Authorization": "Bearer sk-test"},
        trust_env=False,
    )
    with pytest.raises(RuntimeError, match="unexpected chat-completions"):
        llm.complete("q")


def test_openai_llm_raises_on_empty_completion():
    with pytest.raises(RuntimeError, match="empty completion"):
        _mock_llm("   ").complete("q")


def test_hf_embedder_lazy_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    embedder = HFEmbedder("BAAI/bge-small-en-v1.5")
    with pytest.raises(RuntimeError, match="embeddings"):
        embedder(["some text"])
