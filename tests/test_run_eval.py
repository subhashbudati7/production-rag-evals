"""Tests for scripts/run_eval.py's run_eval_cases (warmup + eval loop)."""

import importlib.util
from pathlib import Path

import numpy as np

from rag_service.config import Settings
from rag_service.eval import EvalCase
from rag_service.generate import ExtractiveLLM
from rag_service.ingest import load_documents

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "run_eval.py"
DOCS = load_documents(str(REPO_ROOT / "evals" / "docs"))


def _load_module():
    spec = importlib.util.spec_from_file_location("run_eval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ConstEmbedder:
    """Deterministic embedder; every text maps to the same vector."""

    def __call__(self, texts):
        return np.zeros((len(texts), 4))


def _cases():
    return [
        EvalCase("How long do I have to request a refund?", ["refund"]),
        EvalCase("Do you ship internationally?", ["shipping"]),
    ]


def _run(module, cases, **kwargs):
    settings = Settings(top_k=10, rerank_top_k=3)
    return module.run_eval_cases(
        cases,
        DOCS,
        settings,
        _ConstEmbedder(),
        None,  # no reranker: hybrid-score pass-through
        ExtractiveLLM(),
        **kwargs,
    )


def test_warmup_runs_first_and_is_excluded_from_report():
    module = _load_module()
    report = _run(module, _cases(), warmup_question="warmup: what is the refund policy?")
    # the report only contains the timed cases
    assert report.n == 2
    assert {r.question for r in report.rows} == {
        "How long do I have to request a refund?",
        "Do you ship internationally?",
    }


def test_no_warmup_by_default_runs_cases_only():
    module = _load_module()
    report = _run(module, _cases())
    assert report.n == 2


def test_warmup_consumes_no_clock_time():
    """The warmup call is untimed: the clock is consulted exactly twice per
    measured case (start/stop), never for the warmup."""
    module = _load_module()
    ticks = iter(range(1000))
    clock_calls = []

    def fake_clock():
        clock_calls.append(1)
        return float(next(ticks))

    report = _run(
        module,
        _cases(),
        warmup_question="warmup: what is the refund policy?",
        time_fn=fake_clock,
    )
    assert report.n == 2
    assert len(clock_calls) == 2 * 2
