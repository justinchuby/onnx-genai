#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUNS="${RUNS:-5}"
WARMUPS="${WARMUPS:-1}"
MAX_TOKENS="${MAX_TOKENS:-64}"
DATE="${BENCHMARK_DATE:-$(date +%Y-%m-%d)}"
HOST="${BENCHMARK_HOST:-$(hostname -s)}"
OUTPUT="${OUTPUT:-docs/benchmarks/${DATE}-${HOST}.md}"

ONNX_RUNTIME="${ONNX_RUNTIME:-onnx-genai|http://127.0.0.1:8080/v1|qwen2.5-0.5b|ONNX f32, dynamic KV cache|CPU EP; ORT default threads}"
OLLAMA_RUNTIME="${OLLAMA_RUNTIME:-Ollama (llama.cpp)|http://127.0.0.1:11434/v1|qwen2.5:0.5b-q8-bench|GGUF Q8_0|Metal/default threads}"
LM_STUDIO_RUNTIME="${LM_STUDIO_RUNTIME:-LM Studio|http://127.0.0.1:1234/v1|qwen2.5-0.5b-q8-bench|same GGUF Q8_0|Metal GPU offload; context=2048; parallel=1}"

cargo run --release -p onnx-genai-bench --bin compare -- \
  --runs "$RUNS" \
  --warmups "$WARMUPS" \
  --max-tokens "$MAX_TOKENS" \
  --output "$OUTPUT" \
  --runtime "$ONNX_RUNTIME" \
  --runtime "$OLLAMA_RUNTIME" \
  --runtime "$LM_STUDIO_RUNTIME"
