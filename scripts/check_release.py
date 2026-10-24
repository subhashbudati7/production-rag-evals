#!/usr/bin/env python3
"""One-command pre-merge verification for the release (DEPLOY.md §1).

Usage:
    python scripts/check_release.py

Runs ruff, the test suite, and the offline latency probe, printing a
PASS/FAIL line per step. Exits 0 only if every step passes, so the
Oct 27 deploy run can gate the merge on a single command. Run from the
repo root.
"""

from __future__ import annotations

import subprocess
import sys


def run_step(name: str, argv: list[str]) -> bool:
    """Run one check step, print its verdict, return True on success."""
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    print(f"{'PASS' if result.returncode == 0 else 'FAIL'}: {name}")
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
    return result.returncode == 0


def steps() -> list[tuple[str, list[str]]]:
    """The release checks, in order: lint, tests, latency probe."""
    python = sys.executable
    return [
        ("ruff", [python, "-m", "ruff", "check", "rag_service", "tests", "scripts"]),
        ("pytest", [python, "-m", "pytest", "-q"]),
        (
            "latency probe",
            [
                python,
                "scripts/measure_latency.py",
                "--docs",
                "evals/docs",
                "--eval",
                "evals/golden.jsonl",
            ],
        ),
    ]


def main() -> int:
    """Run every step; return 0 only if all pass."""
    return 0 if all(run_step(name, argv) for name, argv in steps()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
