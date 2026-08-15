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

ONNX_RUNTIME="${ONNX_RUNTIME:-onnx-genai|http://127.0.0.1:8080/v1|qwen2.5-0.5b-f16-webgpu|ONNX fp16 weights; fp32 logits/KV API casts|WebGPU EP; ORT default threads}"
LM_STUDIO_RUNTIME="${LM_STUDIO_RUNTIME:-LM Studio|http://127.0.0.1:1234/v1|qwen05-q4-webgpu-bench|GGUF Q4_0|Metal GPU offload=max; context=2048; parallel=1; speculation=off}"

cargo run --release -p onnx-genai-bench --bin compare -- \
  --runs "$RUNS" \
  --warmups "$WARMUPS" \
  --max-tokens "$MAX_TOKENS" \
  --output "$OUTPUT" \
  --runtime "$ONNX_RUNTIME" \
  --runtime "$LM_STUDIO_RUNTIME"
