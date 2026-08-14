#!/usr/bin/env bash
# Canonical paired Muse workflow-vs-native conformance runner (stock ORT 1.28, H200).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANON="${MUSE_CANONICAL_ROOT:-$ROOT/../metadata-redesign/target}"
export ONNX_GENAI_ORT_LIB="${ONNX_GENAI_ORT_LIB:-/home/justinchu/.conda/envs/onnx/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.28.0}"
export ORT_ENABLE_CUDNN_FLASH_ATTENTION=0
export ONNX_GENAI_CUDA_GRAPH=1
export ONNX_GENAI_EP=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ONNX_GENAI_VRAM_LIMIT="${ONNX_GENAI_VRAM_LIMIT:-75161927680}"
export ONNX_GENAI_LINKED_MODEL_CACHE="${ONNX_GENAI_LINKED_MODEL_CACHE:-$ROOT/target/onnx-genai-linked}"
export ONNX_GENAI_MUSE_WORKFLOW_PACKAGE="${ONNX_GENAI_MUSE_WORKFLOW_PACKAGE:-$ROOT/target/muse-bench}"
export ONNX_GENAI_MUSE_PROMPT_IDS="${ONNX_GENAI_MUSE_PROMPT_IDS:-$CANON/muse-native-harness/benchmarks/muse_prompt_ids.json}"
export ONNX_GENAI_MUSE_NATIVE_BENCHMARK="${ONNX_GENAI_MUSE_NATIVE_BENCHMARK:-$CANON/muse-real-package/native-benchmark.json}"
export ONNX_GENAI_WORKFLOW_PERF_SAMPLES="${ONNX_GENAI_WORKFLOW_PERF_SAMPLES:-5}"
cd "$ROOT"
exec cargo test --release -p onnx-genai-engine --features cuda,cuda-13000 \
  --test workflow_performance_conformance -- --ignored --nocapture \
  real_muse_policy_chain_matches_direct_ort "$@"
