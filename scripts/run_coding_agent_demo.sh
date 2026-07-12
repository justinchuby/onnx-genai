#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8090}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}/v1}"
MODEL_DIR="${MODEL_DIR:-models/qwen2.5-0.5b}"
MODEL_ID="${MODEL_ID:-qwen2.5-0.5b}"
WORKDIR="${WORKDIR:-target/coding-agent-workspace}"

if [ ! -d "$MODEL_DIR" ]; then
  echo "Model directory $MODEL_DIR is missing; run scripts/build_qwen.sh first." >&2
  exit 1
fi

if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is already in use; set PORT to a free port." >&2
  exit 1
fi

cargo build -p onnx-genai-server
./target/debug/onnx-genai-server --model "$MODEL_DIR" --model-id "$MODEL_ID" --addr "127.0.0.1:${PORT}" &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null

python3 scripts/coding_agent.py --base-url "$BASE_URL" --model "$MODEL_ID" --workdir "$WORKDIR" --clean "$@"
