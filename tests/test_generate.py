"""Tests for rag_service.generate."""

import pytest

from rag_service.generate import (
    ExtractiveLLM,
    build_prompt,
    generate_answer,
    verify_citations,
)
from rag_service.rerank import RerankedChunk


class EchoLLM:
    """Deterministic fake: answers with fixed text citing [1] and [2]."""

    def __init__(self, text: str = "Refunds are allowed within thirty days [1]. Contact support [2]."):
        self.text = text
        self.last_prompt = ""

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.text


def _evidence() -> list[RerankedChunk]:
    return [
        RerankedChunk("doc:0", "Refund policy: thirty days.", 9.1, 1),
        RerankedChunk("doc:1", "Contact support to start a refund.", 8.4, 2),
    ]


def test_build_prompt_numbers_evidence_and_includes_query():
    prompt = build_prompt("What is the refund policy?", _evidence())
    assert "[1] (id: doc:0)" in prompt
    assert "[2] (id: doc:1)" in prompt
    assert "What is the refund policy?" in prompt
    assert "ONLY the evidence" in prompt


def test_generate_answer_returns_grounded_answer():
    llm = EchoLLM()
    result = generate_answer("What is the refund policy?", _evidence(), llm)
    assert result.answer == llm.text
    assert result.citations_used == ["doc:0", "doc:1"]
    assert result.citation_check.valid
    assert "What is the refund policy?" in llm.last_prompt


def test_verify_citations_flags_dangling_numbers():
    check = verify_citations("Claim [1] and invented [9].", _evidence())
    assert check.cited_ids == ["doc:0"]
    assert check.dangling == [9]
    assert not check.valid


def test_verify_citations_dedupes_repeated_refs():
    check = verify_citations("A [2] and B [2] and C [1].", _evidence())
    assert check.cited_ids == ["doc:1", "doc:0"]


def test_verify_citations_handles_uncited_answer():
    check = verify_citations("The evidence does not answer this.", _evidence())
    assert check.cited_ids == []
    assert check.dangling == []
    assert check.valid


def test_generate_answer_rejects_empty_query():
    with pytest.raises(ValueError, match="query must not be empty"):
        generate_answer("  ", _evidence(), EchoLLM())


def test_generate_answer_requires_minimum_evidence():
    with pytest.raises(ValueError, match="need at least 2 evidence"):
        generate_answer("q", _evidence()[:1], EchoLLM(), min_evidence=2)


def test_generate_answer_rejects_empty_llm_output():
    with pytest.raises(ValueError, match="empty answer"):
        generate_answer("q", _evidence(), EchoLLM("   "))


def test_extractive_llm_quotes_top_evidence_with_citations():
    answer = ExtractiveLLM().complete(build_prompt("What is the refund policy?", _evidence()))
    assert "[1]" in answer
    assert "thirty days" in answer.lower()


def test_extractive_llm_generates_verifiable_answer():
    evidence = _evidence()
    answer = ExtractiveLLM().complete(build_prompt("q", evidence))
    check = verify_citations(answer, evidence)
    assert check.valid
    result = generate_answer("q", evidence, ExtractiveLLM())
    assert result.citation_check.valid
    assert result.citations_used == ["doc:0", "doc:1"]
