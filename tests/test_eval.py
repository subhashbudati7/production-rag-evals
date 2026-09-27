"""Tests for rag_service.eval."""

import pytest

from rag_service.config import Settings
from rag_service.eval import (
    EvalCase,
    EvalReport,
    EvalRow,
    estimate_cost_usd,
    grounding,
    keyword_coverage,
    run_eval,
)

EVIDENCE = ["refund policy: thirty day window for purchases"]


def test_grounding_scores_fully_cited_answer_high():
    assert grounding("Refunds are allowed for thirty days [1].", EVIDENCE) > 0.7


def test_grounding_penalizes_ungrounded_claims():
    grounded = grounding("Refunds are allowed for thirty days [1].", EVIDENCE)
    hallucinated = grounding(
        "Refunds are allowed and the moon is made of cheese [1].", EVIDENCE
    )
    assert hallucinated < grounded


def test_grounding_handles_empty_inputs():
    assert grounding("", EVIDENCE) == 0.0
    assert grounding("some answer [1]", []) == 0.0


def test_keyword_coverage_counts_expected_terms():
    assert keyword_coverage("thirty day refund window", ["refund", "thirty", "day"]) == 1.0
    assert keyword_coverage("thirty day window", ["refund", "thirty", "day"]) == 2 / 3
    assert keyword_coverage("anything", []) == 1.0


def test_estimate_cost_usd_uses_settings_rates():
    settings = Settings(cost_per_1k_input_tokens=0.1, cost_per_1k_output_tokens=0.2)
    # 4000 input chars = 1k tokens, 4000 output chars = 1k tokens
    assert estimate_cost_usd(4000, 4000, settings) == pytest.approx(0.3)


def _pipeline(question: str):
    return ("Refunds are allowed for thirty days [1].", EVIDENCE, 100, 40)


def _clock():
    _clock.t += 0.05
    return _clock.t


_clock.t = 0.0


def test_run_eval_assembles_rows_and_aggregates():
    cases = [
        EvalCase("What is the refund policy?", ["refund", "thirty"]),
        EvalCase("How long is the refund window?", ["refund", "window"]),
    ]
    report = run_eval(cases, _pipeline, Settings(), time_fn=_clock)
    assert report.n == 2
    assert report.grounding > 0.7
    assert report.keyword_coverage > 0.5
    assert report.latency_p50() == pytest.approx(50.0)
    assert report.latency_p95() == pytest.approx(50.0)
    assert report.cost_usd_per_1k_tokens > 0
    table = report.markdown_table()
    assert "| question |" in table
    assert "Aggregate (n=2)" in table


def test_run_eval_empty_cases_gives_zero_aggregates():
    report = run_eval([], _pipeline, Settings(), time_fn=_clock)
    assert report.n == 0
    assert report.grounding == 0.0
    assert report.latency_p50() == 0.0


def test_eval_report_p95_uses_ceiling_index():
    rows = [EvalRow(f"q{i}", 1.0, 1.0, float(i), 0.0) for i in range(20)]
    report = EvalReport(rows=rows)
    assert report.latency_p95() == 19.0
    assert report.latency_p50() == 9.5
