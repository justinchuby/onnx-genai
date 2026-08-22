#!/usr/bin/env bash
# Capture greedy token goldens from the current runtime for parity auditing.
set -u
OUT="$1"; shift
export ONNX_GENAI_ORT_LIB=/home/justinchu/.ort129/onnxruntime/capi/libonnxruntime.so.1.29.0
BIN=/home/justinchu/.cargo-target-1723/debug/examples/greedy_token_dump
declare -A MODELS=(
  [phi35_mini_int4_cpu]="/home/justinchu/.foundry/cache/models/Microsoft/Phi-3.5-mini-instruct-generic-cpu-2/v2"
  [qwen3_0_6b_cpu]="/home/justinchu/.foundry/cache/models/Microsoft/qwen3-0.6b-generic-cpu-4/v4"
  [qwen35_0_8b_hybrid_cpu]="/home/justinchu/.foundry/cache/models/Microsoft/qwen3.5-0.8b-generic-cpu-2/v2"
  [qwen25_0_5b_cuda]="/home/justinchu/.foundry/cache/models/Microsoft/qwen2.5-0.5b-instruct-cuda-gpu-4/v4-bs128"
  [qwen25_1_5b_cuda]="/home/justinchu/.foundry/cache/models/Microsoft/qwen2.5-1.5b-instruct-cuda-gpu-4/v4-bs128"
  [gpt_oss_20b_cpu]="/home/justinchu/.foundry/cache/models/Microsoft/gpt-oss-20b-generic-cpu-1/v1"
  [phi4_mini_cuda]="/home/justinchu/.foundry/cache/models/Microsoft/Phi-4-mini-instruct-cuda-gpu-5/v5"
)
: > "$OUT"
for name in "${!MODELS[@]}"; do
  dir="${MODELS[$name]}"
  if [ ! -d "$dir" ]; then echo "$name\tMISSING_DIR\t$dir" >> "$OUT"; continue; fi
  ids=$(timeout 900 "$BIN" "$dir" "Hello" 24 2>/dev/null | tail -1)
  if [ -z "$ids" ]; then ids="ERROR"; fi
  printf '%s\t%s\t%s\n' "$name" "$dir" "$ids" >> "$OUT"
done
sort -o "$OUT" "$OUT"
cat "$OUT"
