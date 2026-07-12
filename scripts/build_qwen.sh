#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOBIUS_DIR="${MOBIUS_DIR:-/Users/justinc/Documents/GitHub/mobius}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
OUT_DIR="${OUT_DIR:-$ROOT/models/qwen2.5-0.5b}"
DTYPE="${DTYPE:-f32}"

mkdir -p "$ROOT/models/.hf_cache" "$ROOT/models/.scratch"

HF_HOME="${HF_HOME:-$ROOT/models/.hf_cache}" \
HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}" \
TMPDIR="${TMPDIR:-$ROOT/models/.scratch}" \
PYTHONPATH="$MOBIUS_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
python -m mobius build \
  --model "$MODEL_ID" \
  "$OUT_DIR" \
  --dtype "$DTYPE" \
  --runtime ort-genai
