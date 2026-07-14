#!/usr/bin/env bash
# Milestone A: target-only greedy decode on CUDA for a merged ONNX causal-LM
# package (built/verified against Gemma4 E2B, but MODEL-AGNOSTIC: it only points
# the runtime at a directory + prompt; no model-specific logic lives here).
#
# Why a "target-only" view dir: the shipped merged package carries a
# `speculative:` block in inference_metadata.yaml. With the default engine config
# that auto-selects the SharedKv speculative path (Milestone B). To prove the
# target export alone (Milestone A), we point the runtime at a sibling directory
# that symlinks the same model.onnx(.data)/tokenizer but ships a STRIPPED
# inference_metadata.yaml with NO `speculative:` block and NO share-buffer hints,
# which selects the plain contiguous past->present KV decode path (head_dim
# agnostic — required for this model's heterogeneous per-layer head_dim 256/512).
#
# NOTE ON <bos>: this package's tokenizer does not auto-prepend a BOS token
# (tokenizer_config.json has no add_bos_token and the tokenizer.json
# post-processor adds nothing). Gemma degenerates without BOS, so the prompt
# below prepends the literal "<bos>" special token. Fix the package tokenizer to
# drop this workaround.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${SRC:-$HOME/gemma4-e2b-onnx}"
TARGET_DIR="${TARGET_DIR:-$HOME/gemma4-e2b-onnx-target}"
PROMPT="${PROMPT:-<bos>The capital of France is}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

# --- Build env (engine/CLI need the full ORT CUDA env) ---
export LIBCLANG_PATH="${LIBCLANG_PATH:-$HOME/.local/lib/python3.12/site-packages/clang/native}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"  # .cudaenv.sh appends to this under set -u
# shellcheck disable=SC1091
source "$ROOT/.cudaenv.sh"
export BINDGEN_EXTRA_CLANG_ARGS="${BINDGEN_EXTRA_CLANG_ARGS:--I/usr/lib/gcc/x86_64-pc-linux-gnu/13.2.0/include -I/usr/include}"

# --- Force the CUDA execution provider (else ORT may fall back to CPU) ---
export ONNX_GENAI_EP=cuda

# --- Build the target-only view directory (symlinks + stripped metadata) ---
if [[ ! -f "$TARGET_DIR/model.onnx" ]]; then
  mkdir -p "$TARGET_DIR"
  ln -sf "$SRC/model.onnx"            "$TARGET_DIR/model.onnx"
  ln -sf "$SRC/model.onnx.data"       "$TARGET_DIR/model.onnx.data"
  ln -sf "$SRC/tokenizer.json"        "$TARGET_DIR/tokenizer.json"
  ln -sf "$SRC/tokenizer_config.json" "$TARGET_DIR/tokenizer_config.json"
fi
# Stripped, target-only metadata: no speculative block, no share-buffer hints ->
# plain contiguous past->present decode (head_dim agnostic).
cat > "$TARGET_DIR/inference_metadata.yaml" <<'YAML'
required_capabilities:
  - grouped_query_attention
model:
  attention:
    type: group_query_attention
    num_kv_heads: 1
    num_attention_heads: 8
    head_dim: 256
YAML

# --- Build (needs the CUDA feature; propagates to onnx-genai-ort/cuda) ---
cargo build --release -p onnx-genai --bin onnx-genai --features cuda

# --- Generate ---
exec "$ROOT/target/release/onnx-genai" generate \
  --model "$TARGET_DIR" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  "$PROMPT"
