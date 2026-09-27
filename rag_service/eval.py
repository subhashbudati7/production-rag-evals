"""Evaluation: dataset schema, metrics, and report assembly.

The eval table is the portfolio's headline artifact: faithfulness, answer
relevance, p50/p95 latency, and cost per 1k tokens per query, then
aggregated. Two metrics are deterministic proxies that run without an LLM
judge — documented as such — until the LLM eval harness (December project)
lands a judge-based suite:

- ``grounding`` (faithfulness proxy): fraction of the answer's content
  tokens that appear in the cited evidence. A fully cited, fully grounded
  answer scores 1.0.
- ``keyword_coverage`` (answer-relevance proxy): fraction of the eval
  case's expected keywords present in the answer.

Latency is measured wall-clock per case; token counts are estimated at
~4 characters per token and priced from ``Settings``.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field

from rag_service.config import Settings
from rag_service.retrieval import tokenize

_CHARS_PER_TOKEN = 4.0

_STOPWORDS = frozenset(
    ["the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were", "be", "been", "on", "for", "with", "as", "by", "at", "from", "it", "its", "this", "that", "these", "those", "i", "you", "he", "she", "we", "they", "them", "his", "her", "our", "their", "not", "no", "yes", "if", "then", "than", "so", "such", "can", "could", "will", "would", "should", "may", "might", "do", "does", "did", "have", "has", "had", "what", "which", "who", "whom", "when", "where", "why", "how", "does"]
)


@dataclass(frozen=True)
class EvalCase:
    """One eval query with its expected answer signal."""

    question: str
    expected_keywords: list[str]
    """Keywords a correct answer should contain; relevance = coverage fraction."""


@dataclass(frozen=True)
class EvalRun:
    """One pipeline execution over an eval case."""

    answer: str
    evidence_texts: list[str]
    latency_ms: float
    input_chars: int
    output_chars: int


@dataclass(frozen=True)
class EvalRow:
    """Per-case metrics row for the eval table."""

    question: str
    grounding: float
    keyword_coverage: float
    latency_ms: float
    cost_usd_per_1k_tokens: float


@dataclass(frozen=True)
class EvalReport:
    """Full eval run: per-case rows plus aggregates."""

    rows: list[EvalRow] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.rows)

    def _col(self, attr: str) -> list[float]:
        return [getattr(r, attr) for r in self.rows]

    @property
    def grounding(self) -> float:
        return statistics.fmean(self._col("grounding")) if self.rows else 0.0

    @property
    def keyword_coverage(self) -> float:
        return statistics.fmean(self._col("keyword_coverage")) if self.rows else 0.0

    def latency_p50(self) -> float:
        return statistics.median(self._col("latency_ms")) if self.rows else 0.0

    def latency_p95(self) -> float:
        vals = sorted(self._col("latency_ms"))
        if not vals:
            return 0.0
        idx = min(int(0.95 * len(vals)), len(vals) - 1)
        return vals[idx]

    @property
    def cost_usd_per_1k_tokens(self) -> float:
        return statistics.fmean(self._col("cost_usd_per_1k_tokens")) if self.rows else 0.0

    def markdown_table(self) -> str:
        """Render the eval table as markdown for the README."""
        header = (
            "| question | grounding | keyword coverage | latency ms | "
            "cost $/1k tok |\n"
            "|---|---|---|---|---|\n"
        )
        body = "".join(
            f"| {r.question} | {r.grounding:.3f} | {r.keyword_coverage:.3f} | "
            f"{r.latency_ms:.1f} | {r.cost_usd_per_1k_tokens:.5f} |\n"
            for r in self.rows
        )
        summary = (
            f"\n**Aggregate (n={self.n}):** grounding {self.grounding:.3f}, "
            f"keyword coverage {self.keyword_coverage:.3f}, "
            f"latency p50 {self.latency_p50():.1f}ms / p95 {self.latency_p95():.1f}ms, "
            f"cost ${self.cost_usd_per_1k_tokens:.5f}/1k tokens.\n"
        )
        return header + body + summary


def _stem(tok: str) -> str:
    """Naive plural stemmer so 'refunds' matches 'refund' in grounding."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("es"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s"):
        return tok[:-1]
    return tok


def grounding(answer: str, evidence_texts: list[str]) -> float:
    """Faithfulness proxy: content-token overlap between answer and evidence."""
    evidence_tokens = set()
    for text in evidence_texts:
        evidence_tokens.update(
            _stem(t) for t in tokenize(text) if t not in _STOPWORDS
        )
    answer_tokens = [
        _stem(t)
        for t in tokenize(re.sub(r"\[\d+\]", " ", answer))
        if t not in _STOPWORDS
    ]
    if not answer_tokens:
        return 0.0
    if not evidence_tokens:
        return 0.0
    return sum(1 for t in answer_tokens if t in evidence_tokens) / len(answer_tokens)


def keyword_coverage(answer: str, expected: list[str]) -> float:
    """Answer-relevance proxy: fraction of expected keywords in the answer."""
    if not expected:
        return 1.0
    tokens = {_stem(t) for t in tokenize(answer)}
    return sum(1 for k in expected if _stem(k.lower()) in tokens) / len(expected)


def estimate_cost_usd(input_chars: int, output_chars: int, settings: Settings) -> float:
    """Estimated request cost in USD from ~4-chars-per-token pricing."""
    in_k = (input_chars / _CHARS_PER_TOKEN) / 1000
    out_k = (output_chars / _CHARS_PER_TOKEN) / 1000
    return in_k * settings.cost_per_1k_input_tokens + out_k * settings.cost_per_1k_output_tokens


Pipeline = Callable[[str], tuple[str, list[str], int, int]]
"""question -> (answer, evidence_texts, input_chars, output_chars)."""


def run_eval(
    cases: list[EvalCase],
    pipeline: Pipeline,
    settings: Settings | None = None,
    time_fn=None,
) -> EvalReport:
    """Run every eval case through the pipeline and assemble the report.

    ``time_fn`` is injectable for deterministic tests; production passes
    ``time.perf_counter``. Latency is measured in milliseconds.
    """
    import time as _time

    settings = settings or Settings()
    clock = time_fn or _time.perf_counter
    rows: list[EvalRow] = []
    for case in cases:
        start = clock()
        answer, evidence, in_chars, out_chars = pipeline(case.question)
        latency_ms = (clock() - start) * 1000
        token_total_k = (in_chars + out_chars) / _CHARS_PER_TOKEN / 1000
        rows.append(
            EvalRow(
                question=case.question,
                grounding=grounding(answer, evidence),
                keyword_coverage=keyword_coverage(answer, case.expected_keywords),
                latency_ms=latency_ms,
                cost_usd_per_1k_tokens=(
                    estimate_cost_usd(in_chars, out_chars, settings) / token_total_k
                    if token_total_k > 0
                    else 0.0
                ),
            )
        )
    return EvalReport(rows=rows)
