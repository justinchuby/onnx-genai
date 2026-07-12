#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TARGET_MODEL_ID="${TARGET_MODEL_ID:-roneneldan/TinyStories-33M}"
DRAFT_MODEL_ID="${DRAFT_MODEL_ID:-roneneldan/TinyStories-1M}"
TARGET_DIR="${ONNX_GENAI_SPEC_TARGET:-$ROOT/models/tinystories-33m}"
DRAFT_DIR="${ONNX_GENAI_SPEC_DRAFT:-$ROOT/models/tinystories-1m}"
TARGET_LOCAL_DIR="${TARGET_LOCAL_DIR:-$ROOT/models/tinystories-33m-local}"
DRAFT_LOCAL_DIR="${DRAFT_LOCAL_DIR:-$ROOT/models/tinystories-1m-local}"
SPEC_K="${ONNX_GENAI_SPEC_K:-4}"
MAX_NEW_TOKENS="${ONNX_GENAI_SPEC_MAX_NEW_TOKENS:-32}"

if [[ "${BUILD_MODELS:-0}" == "1" ]]; then
  MODEL_ID="$TARGET_MODEL_ID" LOCAL_DIR="$TARGET_LOCAL_DIR" OUT_DIR="$TARGET_DIR" \
    "$ROOT/scripts/build_real_model.sh"
  MODEL_ID="$DRAFT_MODEL_ID" LOCAL_DIR="$DRAFT_LOCAL_DIR" OUT_DIR="$DRAFT_DIR" \
    "$ROOT/scripts/build_real_model.sh"
fi

if [[ ! -d "$TARGET_DIR" || ! -d "$DRAFT_DIR" ]]; then
  cat >&2 <<EOF
Missing benchmark models.

Expected:
  target: $TARGET_DIR
  draft:  $DRAFT_DIR

Build them with:
  BUILD_MODELS=1 scripts/bench_speculative.sh

Or point at existing models with ONNX_GENAI_SPEC_TARGET and ONNX_GENAI_SPEC_DRAFT.
EOF
  exit 2
fi

python - "$TARGET_DIR" "$DRAFT_DIR" <<'PY'
import hashlib
import sys
from pathlib import Path

target = Path(sys.argv[1])
draft = Path(sys.argv[2])
for name in ("tokenizer.json", "vocab.json", "merges.txt"):
    target_file = target / name
    draft_file = draft / name
    if target_file.exists() and draft_file.exists():
        t = hashlib.sha256(target_file.read_bytes()).hexdigest()
        d = hashlib.sha256(draft_file.read_bytes()).hexdigest()
        if t != d:
            raise SystemExit(f"tokenizer mismatch for {name}: target={t} draft={d}")
PY

export ONNX_GENAI_SPEC_TARGET="$TARGET_DIR"
export ONNX_GENAI_SPEC_DRAFT="$DRAFT_DIR"
export ONNX_GENAI_SPEC_K="$SPEC_K"
export ONNX_GENAI_SPEC_MAX_NEW_TOKENS="$MAX_NEW_TOKENS"

cargo test -p onnx-genai-engine --test speculative_speedup \
  speculative_decoding_exceeds_required_speedup_when_models_are_present \
  -- --ignored --nocapture
