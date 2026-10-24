# Deploy runbook — project 1 (production-rag-evals), target 2026-10-27

Release = merge `rag/october-build` into `main` (no squash, so every commit
counts on the contribution graph) + a live demo URL.

## 1. Pre-merge verification

- `ruff check rag_service tests scripts` clean, `pytest -q` all green.
- `python scripts/measure_latency.py` runs; README eval table matches its output.
- Every local commit on `rag/october-build` is mirrored to GitHub —
  `git ls-remote` head must equal the local head. No unmirrored commits.
- Model-backed eval is fresh: `python scripts/run_eval.py
  --docs evals/docs --eval evals/golden.jsonl` over all 10 golden cases;
  README eval table date/numbers and `evals/baseline_report.md` updated.

## 2. Demo verification (needs a Docker host)

This VM has no Docker, so run this on a machine that does:

```bash
bash -n scripts/demo.sh            # syntax
bash scripts/demo.sh               # full run: build → index → golden question
DEMO_QUESTION_INDEX=7 bash scripts/demo.sh        # a different question
RAG_LLM_API_KEY=... bash scripts/demo.sh         # generated answer path
```

Expect: image builds, container serves `evals/docs`, `/health` reports
`index_built: true`, the golden question returns an answer with numbered
citations and per-stage timings, container is removed on exit.

## 3. Merge and release

```bash
git checkout main
git merge --no-ff rag/october-build -m "release: production RAG with evals (October)"
git push origin main
```

Verify the CI workflow is green on `main`, then tag the release
(e.g. `v0.1.0`) so the eval numbers are pinned to a ref.

## 4. Live demo URL — DECISION NEEDED

Hosting is not chosen yet. Candidates (prices verified October 2026):

- **Hugging Face Spaces** (Docker SDK): purpose-built for model demos,
  repo can live under his HF account. The free CPU tier is dead for new
  accounts — creating a new Docker Space needs PRO ($9/mo) or added
  credits. Still the recommendation, but only after he sees the real price.
- **Google Cloud Run** (warm always-on): ~$25–30/mo; no account decision
  beyond his existing Google account.
- **Render / Fly.io**: container hosting; verify current free-tier
  availability before choosing — do not assume a free tier exists.
- **Vercel**: not a fit — it doesn't run Docker containers.

Pick one, deploy the production image, and put the URL at the top of the
README. This is the one deploy step that needs Subhash's call (account +
hosting choice), so it travels through the report channel, not a silent run.

## 5. Report

One summary after the deploy: merge commit, CI status, live demo URL,
final eval numbers (n=10), commit count for the project.
