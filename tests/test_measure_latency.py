"""Tests for scripts/measure_latency.py."""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "measure_latency.py"
EVALS_DIR = REPO_ROOT / "evals"


def _load_module():
    spec = importlib.util.spec_from_file_location("measure_latency", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def probe():
    return _load_module()


def test_percentile_basic(probe):
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert probe.percentile(values, 50) == 3.0
    assert probe.percentile(values, 100) == 5.0
    assert probe.percentile(values, 0) == 1.0


def test_percentile_rejects_empty(probe):
    with pytest.raises(ValueError):
        probe.percentile([], 50)


def test_build_vocab_from_golden(probe):
    vocab = probe._build_vocab(str(EVALS_DIR / "golden.jsonl"))
    assert "thirty" in vocab
    assert "digital" in vocab
    assert vocab == sorted(set(vocab))


def test_collect_runs_golden_set_through_query(probe):
    from rag_service.config import Settings
    from rag_service.ingest import load_documents

    settings = Settings(top_k=10, rerank_top_k=3, auto_build_index_on_startup=False)
    questions = probe.load_questions(str(EVALS_DIR / "golden.jsonl"))
    assert len(questions) == 10
    docs = load_documents(str(EVALS_DIR / "docs"))
    vocab = probe._build_vocab(str(EVALS_DIR / "golden.jsonl"))
    latencies = probe._collect(questions, docs, settings, vocab)

    for stage in ("retrieval_ms", "rerank_ms", "generation_ms", "latency_ms"):
        assert len(latencies[stage]) == len(questions)
        assert all(v >= 0 for v in latencies[stage])
    summary = probe.summarize(latencies)
    assert summary["retrieval_ms"]["p50"] <= summary["latency_ms"]["p50"]


def test_render_table_has_all_stages(probe):
    summary = {
        "retrieval_ms": {"p50": 1.0, "p95": 2.0, "max": 3.0},
        "rerank_ms": {"p50": 1.0, "p95": 2.0, "max": 3.0},
        "generation_ms": {"p50": 1.0, "p95": 2.0, "max": 3.0},
        "latency_ms": {"p50": 3.0, "p95": 6.0, "max": 9.0},
    }
    table = probe.render_table(summary, 6)
    assert "6 golden questions" in table
    for stage in summary:
        assert stage in table
