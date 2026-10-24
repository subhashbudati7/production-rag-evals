"""Tests for scripts/check_release.py's step runner."""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_release.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_release", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_step_passes_on_zero_exit(capsys):
    module = _load_module()
    assert module.run_step("ok", ["true"]) is True
    assert "PASS: ok" in capsys.readouterr().out


def test_run_step_fails_on_nonzero_exit(capsys):
    module = _load_module()
    assert module.run_step("bad", ["false"]) is False
    assert "FAIL: bad" in capsys.readouterr().out


def test_steps_covers_lint_tests_and_probe():
    module = _load_module()
    names = [name for name, _ in module.steps()]
    assert names == ["ruff", "pytest", "latency probe"]


def test_main_exits_zero_when_all_steps_pass(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "run_step", lambda name, argv: True)
    assert module.main() == 0


def test_main_exits_nonzero_when_a_step_fails(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module, "run_step", lambda name, argv: name != "pytest"
    )
    assert module.main() == 1
