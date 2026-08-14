#!/usr/bin/env bash
# Alternating paired native/workflow samples on one machine, one session.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N="${1:-3}"
export PYTHONPATH="$ROOT/target/genai-sm90-build/Release/wheel"
export LD_LIBRARY_PATH=/home/justinchu/.conda/envs/onnx/lib:/home/justinchu/.conda/envs/onnx/lib/python3.12/site-packages/onnxruntime/capi
export ORT_ENABLE_CUDNN_FLASH_ATTENTION=0
export ONNX_GENAI_CUDA_GRAPH=1
export CUDA_VISIBLE_DEVICES=0
for i in $(seq 1 "$N"); do
  ( cd "$ROOT/scratch/native-harness" && /home/justinchu/.conda/envs/onnx/bin/python \
      "$ROOT/scripts/benchmark_muse_native_local.py" --model "$ROOT/target/muse-bench" \
      --config "$ROOT/target/muse_workflow_h200_5x_gated.json" \
      --output "$ROOT/target/evidence/native-alt-$i.json" >/dev/null 2>&1 )
  NAT=$(/home/justinchu/.conda/envs/onnx/bin/python -c "import json;print(round(json.load(open('$ROOT/target/evidence/native-alt-$i.json'))['metrics']['throughput_tok_s'],3))")
  WF=$("$ROOT/run_muse_paired.sh" 2>&1 | grep -oP 'workflow=\K[0-9.]+' | head -1)
  echo "pair $i: native=$NAT tok/s  workflow=$WF tok/s  ratio=$(/home/justinchu/.conda/envs/onnx/bin/python -c "print(round($WF/$NAT,4))")"
done
