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
