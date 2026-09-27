"""Citation-grounded answer generation.

The generator composes an answer over reranked evidence. The prompt
instruction demands inline citations like ``[2]`` after every factual claim,
and ``verify_citations`` enforces the contract on the way out: a citation
number must point at one of the evidence chunks that was actually shown to
the model. Citations the model invents (numbers beyond the evidence range)
are reported as dangling instead of being passed off as grounded.

The ``LLMClient`` protocol keeps tests deterministic: production plugs in an
OpenAI-compatible client, tests inject a fake.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from rag_service.rerank import RerankedChunk
from rag_service.retrieval import stem, tokenize

_CITATION_RE = re.compile(r"\[(\d+)\]")


class LLMClient(Protocol):
    """Completes a chat prompt; production uses an OpenAI-compatible client."""

    def complete(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class CitationCheck:
    """Result of checking an answer's inline citations against the evidence."""

    cited_ids: list[str]
    """Evidence chunk ids actually referenced by the answer, in first-use order."""
    dangling: list[int]
    """Citation numbers that point beyond the provided evidence."""

    @property
    def valid(self) -> bool:
        return not self.dangling


@dataclass(frozen=True)
class GeneratedAnswer:
    """An answer plus the evidence that grounds it."""

    answer: str
    citations_used: list[str]
    citation_check: CitationCheck
    evidence: list[RerankedChunk] = field(repr=False)


def build_prompt(query: str, evidence: list[RerankedChunk]) -> str:
    """Assemble the generation prompt with numbered evidence chunks."""
    sources = "\n\n".join(
        f"[{c.rank}] (id: {c.chunk_id})\n{c.text}" for c in evidence
    )
    return (
        "Answer the question using ONLY the evidence below. "
        "Put an inline citation like [1] after every factual claim, "
        "referring to the numbered evidence. "
        "If the evidence does not answer the question, say so explicitly "
        "and cite nothing.\n\n"
        f"Question: {query}\n\nEvidence:\n{sources}\n\nAnswer:"
    )


def verify_citations(answer: str, evidence: list[RerankedChunk]) -> CitationCheck:
    """Check that every citation in the answer points at real evidence."""
    by_number = {c.rank: c.chunk_id for c in evidence}
    seen: list[str] = []
    dangling: list[int] = []
    for n in (int(m) for m in _CITATION_RE.findall(answer)):
        if n in by_number:
            cid = by_number[n]
            if cid not in seen:
                seen.append(cid)
        elif n not in dangling:
            dangling.append(n)
    return CitationCheck(cited_ids=seen, dangling=sorted(dangling))


def generate_answer(
    query: str,
    evidence: list[RerankedChunk],
    llm: LLMClient,
    min_evidence: int = 1,
) -> GeneratedAnswer:
    """Generate a citation-grounded answer over the reranked evidence."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if len(evidence) < min_evidence:
        raise ValueError(
            f"need at least {min_evidence} evidence chunk(s), got {len(evidence)}"
        )
    answer = llm.complete(build_prompt(query, evidence)).strip()
    if not answer:
        raise ValueError("LLM returned an empty answer")
    check = verify_citations(answer, evidence)
    return GeneratedAnswer(
        answer=answer,
        citations_used=check.cited_ids,
        citation_check=check,
        evidence=evidence,
    )


_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]")


class ExtractiveLLM:
    """Zero-API-key baseline ``LLMClient`` for evals and smoke tests.

    Picks the evidence sentences with the most (stemmed) token overlap with
    the question and quotes them with their citation numbers. Not a
    substitute for a real generator — it exists so the eval suite runs
    deterministically in CI without credentials, and so any real LLM can
    be compared against it.
    """

    def __init__(self, sentences: int = 2) -> None:
        self.sentences = sentences

    def complete(self, prompt: str) -> str:
        evidence = _parse_evidence(prompt)
        question = _parse_question(prompt)
        q_tokens = {stem(t) for t in tokenize(question)}
        scored: list[tuple[float, int, int, str, int]] = []
        for rank, text in evidence:
            for pos, sentence in enumerate(_SENTENCE_RE.findall(text)):
                clean = sentence.strip()
                s_tokens = {stem(t) for t in tokenize(clean)}
                overlap = len(q_tokens & s_tokens)
                scored.append((overlap, -rank, -pos, clean, rank))
        scored.sort(key=lambda s: (s[0], s[1], s[2]), reverse=True)
        parts = [f"{clean} [{rank}]" for _, _, _, clean, rank in scored[: self.sentences]]
        return " ".join(parts) if parts else "The evidence does not answer this question."


def _parse_question(prompt: str) -> str:
    m = re.search(r"^Question: (.*)$", prompt, re.MULTILINE)
    return m.group(1) if m else ""


def _parse_evidence(prompt: str) -> list[tuple[int, str]]:
    """Recover the numbered evidence chunks from a ``build_prompt`` prompt.

    Chunk texts may contain blank lines, so headers are located with a
    regex and each chunk runs until the next header (or the Answer marker).
    """
    marker = "Evidence:\n"
    body = prompt.split(marker, 1)[1] if marker in prompt else ""
    tail = body.split("\n\nAnswer:", 1)[0]
    headers = list(re.finditer(r"\[(\d+)\] \(id: [^)]*\)\n", tail))
    chunks: list[tuple[int, str]] = []
    for i, match in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(tail)
        chunks.append((int(match.group(1)), tail[match.end() : end].strip()))
    return chunks
