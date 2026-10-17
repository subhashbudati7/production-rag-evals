#!/usr/bin/env bash
# One-command local demo of the production image: build, serve the sample
# corpus, warm up the models, ask a golden question, print the answer with
# citations and per-stage timings, then tear down.
#
#   bash scripts/demo.sh
#
# Needs Docker and a network connection (model weights download into the
# container on first boot). Pass RAG_LLM_API_KEY=... for a generated
# answer; without it the endpoint returns evidence citations without
# pretending to generate. Pass DEMO_QUESTION_INDEX=N to ask the Nth
# golden question (default 0).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${DEMO_IMAGE:-rag-demo}"
CONTAINER="${DEMO_CONTAINER:-rag-demo-run}"
PORT="${DEMO_PORT:-8000}"
TIMEOUT_S="${DEMO_TIMEOUT_S:-900}"

cd "$REPO_ROOT"

echo "==> building image $IMAGE"
docker build -t "$IMAGE" .

cleanup() {
  echo "==> stopping container"
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "==> starting container (docs: evals/docs, model warmup on)"
docker run -d --name "$CONTAINER" \
  -p "$PORT:$PORT" \
  -v "$REPO_ROOT/evals/docs:/app/docs:ro" \
  -e "RAG_WARMUP_MODELS_ON_STARTUP=true" \
  ${RAG_LLM_API_KEY:+-e "RAG_LLM_API_KEY=$RAG_LLM_API_KEY"} \
  "$IMAGE" >/dev/null

echo "==> waiting for the index to build (up to ${TIMEOUT_S}s)"
deadline=$((SECONDS + TIMEOUT_S))
while true; do
  if health=$(curl -sf "http://localhost:$PORT/health" 2>/dev/null); then
    if echo "$health" | grep -q '"index_built":true'; then
      echo "$health"
      break
    fi
  fi
  if ((SECONDS > deadline)); then
    echo "timed out waiting for the index; container logs:" >&2
    docker logs "$CONTAINER" >&2 || true
    exit 1
  fi
  sleep 5
done

QUESTION_INDEX="${DEMO_QUESTION_INDEX:-0}"
QUESTION=$(python3 - "$QUESTION_INDEX" <<'EOF'
import json, sys
lines = [line for line in open("evals/golden.jsonl") if line.strip()]
idx = int(sys.argv[1])
if not 0 <= idx < len(lines):
    sys.exit(f"DEMO_QUESTION_INDEX {idx} out of range (0..{len(lines) - 1})")
print(json.loads(lines[idx])["question"])
EOF
)
echo "==> asking: $QUESTION"
curl -s "http://localhost:$PORT/query" \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "import json,sys; print(json.dumps({'query': sys.stdin.read()}))" <<<"$QUESTION")" \
  | python3 -m json.tool
