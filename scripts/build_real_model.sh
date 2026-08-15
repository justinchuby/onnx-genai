#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOBIUS_ROOT="${MOBIUS_ROOT:-/Users/justinc/Documents/GitHub/mobius}"
MODEL_ID="${MODEL_ID:-roneneldan/TinyStories-1M}"
LOCAL_DIR="${LOCAL_DIR:-$ROOT/models/tinystories-local}"
OUT_DIR="${OUT_DIR:-$ROOT/models/tinystories}"
SCRATCH_DIR="$ROOT/.squad/.scratch/tmp"
HF_CACHE_DIR="${HF_HOME:-$ROOT/models/.hf_cache}"

mkdir -p "$SCRATCH_DIR" "$HF_CACHE_DIR" "$LOCAL_DIR" "$OUT_DIR"

export TMPDIR="$SCRATCH_DIR"
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export PYTHONPATH="$MOBIUS_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python - "$MODEL_ID" "$LOCAL_DIR" <<'PY'
from pathlib import Path
import shutil
import sys

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

model_id = sys.argv[1]
out = Path(sys.argv[2])
src = Path(
    snapshot_download(
        model_id,
        allow_patterns=[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "pytorch_model.bin",
        ],
    )
)
out.mkdir(parents=True, exist_ok=True)
for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
    shutil.copy2(src / name, out / name)
state = torch.load(src / "pytorch_model.bin", map_location="cpu")
save_file(state, out / "model.safetensors")
PY

python -m mobius build --config "$LOCAL_DIR" --runtime ort-genai "$OUT_DIR"

cat <<EOF

Built $MODEL_ID at $OUT_DIR.
Smoke test:
  cargo run -p onnx-genai --bin onnx-genai -- generate --model "$OUT_DIR" --max-new-tokens 30 "Once upon a time"
EOF
