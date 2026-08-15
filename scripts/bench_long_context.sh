#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${MODE:-static}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
BUCKETS="${BUCKETS:-64,256,1024,2048}"
RSS_EVERY="${RSS_EVERY:-128}"

if [[ "$MODE" == "static" || "$MODE" == "auto" ]]; then
  MODEL_DIR="${MODEL_DIR:-$ROOT/models/qwen2.5-0.5b-scatter}"
  if [[ ! -d "$MODEL_DIR" ]]; then
    if [[ "${BUILD_MODEL:-0}" == "1" ]]; then
      STATIC_CACHE=1 MAX_SEQ_LEN="${MAX_SEQ_LEN:-$MAX_TOKENS}" OUT_DIR="$MODEL_DIR" scripts/build_qwen.sh
    else
      echo "Missing static-cache model directory: $MODEL_DIR" >&2
      echo "Build it with: BUILD_MODEL=1 MAX_SEQ_LEN=$MAX_TOKENS scripts/bench_long_context.sh" >&2
      exit 1
    fi
  fi
else
  MODEL_DIR="${MODEL_DIR:-$ROOT/models/qwen2.5-0.5b}"
fi

cargo run -p onnx-genai-ort --example long_context_bench -- \
  --model "$MODEL_DIR" \
  --mode "$MODE" \
  --max-tokens "$MAX_TOKENS" \
  --buckets "$BUCKETS" \
  --rss-every "$RSS_EVERY"
