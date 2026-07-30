#!/usr/bin/env bash
#
# Build the demo model: Qwen/Qwen2.5-0.5B-Instruct exported to ONNX via Mobius.
#
# This is the script a fresh clone runs first, so it validates its environment
# up front and fails with instructions rather than a Python traceback.
#
# Usage:
#     scripts/build_qwen.sh                       # models/qwen2.5-0.5b
#     STATIC_CACHE=1 scripts/build_qwen.sh        # models/qwen2.5-0.5b-scatter
#     scripts/build_qwen.sh --help
#
# Compatibility: runs on bash 3.2 (stock macOS /bin/bash). Notably, under
# `set -u` bash 3.2 aborts on `"${arr[@]}"` when `arr` is empty, so array
# expansions here use the `${arr[@]+"${arr[@]}"}` idiom.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck source-path=SCRIPTDIR
# shellcheck source=lib/mobius_env.sh
. "$ROOT/scripts/lib/mobius_env.sh"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
DTYPE="${DTYPE:-f32}"
EP="${EP:-default}"
STATIC_CACHE="${STATIC_CACHE:-${SCATTER_CACHE:-0}}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<USAGE
Build the demo model (Qwen2.5-0.5B-Instruct) into an ONNX package.

Usage:
  scripts/build_qwen.sh [--help]

Requires Mobius, the ONNX exporter, which lives in a separate repository:
  $MOBIUS_REPO_URL
Install it with:
  $MOBIUS_INSTALL_HINT

Environment variables:
  MODEL_ID      HuggingFace model to export      (default: Qwen/Qwen2.5-0.5B-Instruct)
  OUT_DIR       Output directory                 (default: models/qwen2.5-0.5b,
                                                  or models/qwen2.5-0.5b-scatter
                                                  when STATIC_CACHE=1)
  DTYPE         Weight dtype: f32|f16|bf16       (default: f32)
  EP            Target execution provider        (default: default)
  STATIC_CACHE  1 to export static KV buffers    (default: 0)
  MAX_SEQ_LEN   Static cache capacity            (default: 2048; STATIC_CACHE only)
  MOBIUS_DIR    Path to a Mobius checkout        (default: auto-detected)
  PYTHON        Python interpreter to use        (default: auto-detected)
  DRY_RUN       1 to print the build command and exit without building

Examples:
  scripts/build_qwen.sh
  STATIC_CACHE=1 MAX_SEQ_LEN=8192 scripts/build_qwen.sh
  MOBIUS_DIR=/path/to/mobius DTYPE=f16 scripts/build_qwen.sh
USAGE
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  "")
    ;;
  *)
    printf 'error: unexpected argument: %s\n\n' "$1" >&2
    usage >&2
    exit 2
    ;;
esac

# `truthy 1|true|yes|on` - accepted spellings for the boolean env vars.
truthy() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

CACHE_ARGS=()
if truthy "$STATIC_CACHE"; then
  OUT_DIR="${OUT_DIR:-$ROOT/models/qwen2.5-0.5b-scatter}"
  CACHE_ARGS=(--static-cache --max-seq-len "$MAX_SEQ_LEN")

  # A static-cache graph drives TensorScatter through integer control ports
  # (write_indices / kv_sequence_length) that are shape-indistinguishable, so
  # the runtime refuses to guess them and requires an explicit
  # `model.io.static_cache` declaration. That declaration lives in
  # inference_metadata.yaml, which only `--runtime onnx-genai` emits -
  # `ort-genai` writes genai_config.json, which cannot express it, and the
  # resulting model fails to load. See scripts/lib/write_static_cache_metadata.py.
  RUNTIME_TARGET="onnx-genai"
  REQUIRED_ARTIFACTS="inference_metadata.yaml model.onnx tokenizer.json tokenizer_config.json"

  case "$MAX_SEQ_LEN" in
    ''|*[!0-9]*)
      printf 'error: MAX_SEQ_LEN must be a positive integer, got: %s\n' "$MAX_SEQ_LEN" >&2
      exit 2
      ;;
  esac
else
  OUT_DIR="${OUT_DIR:-$ROOT/models/qwen2.5-0.5b}"
  RUNTIME_TARGET="ort-genai"
  REQUIRED_ARTIFACTS="genai_config.json model.onnx tokenizer.json tokenizer_config.json"
fi

# Locate an interpreter that can import Mobius. Prints install instructions
# and exits non-zero if it cannot find one.
mobius_resolve "$ROOT"

mkdir -p "$ROOT/models/.hf_cache" "$ROOT/models/.scratch" "$(dirname "$OUT_DIR")"

printf 'Building %s\n' "$MODEL_ID"
printf '  output   : %s\n' "$OUT_DIR"
printf '  dtype    : %s\n' "$DTYPE"
printf '  ep       : %s\n' "$EP"
if truthy "$STATIC_CACHE"; then
  printf '  kv cache : static (max_seq_len=%s)\n' "$MAX_SEQ_LEN"
else
  printf '  kv cache : dynamic\n'
fi
printf '  runtime  : %s\n' "$RUNTIME_TARGET"
printf '  mobius   : %s\n' "$MOBIUS_SOURCE"
printf '  python   : %s\n' "$MOBIUS_PYTHON"
printf '\n'

if truthy "$DRY_RUN"; then
  printf 'DRY_RUN=1, not building. Command would be:\n'
  printf '%s -m mobius build --model %s %s --dtype %s --ep %s %s--runtime %s\n' \
    "$MOBIUS_PYTHON" "$MODEL_ID" "$OUT_DIR" "$DTYPE" "$EP" \
    "$(if [ ${#CACHE_ARGS[@]} -gt 0 ]; then printf '%s ' "${CACHE_ARGS[@]}"; fi)" \
    "$RUNTIME_TARGET"
  if truthy "$STATIC_CACHE"; then
    printf '%s %s %s\n' \
      "$MOBIUS_PYTHON" "$ROOT/scripts/lib/write_static_cache_metadata.py" "$OUT_DIR"
    printf '%s %s %s %s\n' \
      "$MOBIUS_PYTHON" "$ROOT/scripts/lib/write_tokenizer_assets.py" "$MODEL_ID" "$OUT_DIR"
  fi
  exit 0
fi

# The export downloads weights and writes multi-GB intermediates; keep both
# inside the repo's models/ directory so they are easy to find and delete.
# TMPDIR is overridden (not defaulted) because macOS always presets it to a
# small per-user volume that the scratch files can overflow.
HF_HOME="${HF_HOME:-$ROOT/models/.hf_cache}" \
HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}" \
TMPDIR="$ROOT/models/.scratch" \
PYTHONPATH="$MOBIUS_PYTHONPATH" \
"$MOBIUS_PYTHON" -m mobius build \
  --model "$MODEL_ID" \
  "$OUT_DIR" \
  --dtype "$DTYPE" \
  --ep "$EP" \
  ${CACHE_ARGS[@]+"${CACHE_ARGS[@]}"} \
  --runtime "$RUNTIME_TARGET"

# Mobius does not emit the `io:` block for static-cache exports, so the model
# it just wrote cannot be loaded yet. Derive the declaration from the graph.
# The onnx-genai runtime target also writes only tokenizer.json, dropping the
# tokenizer_config.json that carries this model's stop token and chat template.
if truthy "$STATIC_CACHE"; then
  PYTHONPATH="$MOBIUS_PYTHONPATH" \
  "$MOBIUS_PYTHON" "$ROOT/scripts/lib/write_static_cache_metadata.py" "$OUT_DIR"

  HF_HOME="${HF_HOME:-$ROOT/models/.hf_cache}" \
  PYTHONPATH="$MOBIUS_PYTHONPATH" \
  "$MOBIUS_PYTHON" "$ROOT/scripts/lib/write_tokenizer_assets.py" "$MODEL_ID" "$OUT_DIR"
fi

# A partial export is worse than a failed one: the runtime would load it and
# fail later with a confusing error. Verify the package is complete.
missing=""
for required in $REQUIRED_ARTIFACTS; do
  if [ ! -f "$OUT_DIR/$required" ]; then
    missing="$missing $required"
  fi
done

if [ -n "$missing" ]; then
  printf '\nerror: Mobius exited successfully but the model package is incomplete.\n' >&2
  printf '       %s is missing:%s\n' "$OUT_DIR" "$missing" >&2
  printf '       Delete the directory and re-run to retry:\n' >&2
  printf '           rm -rf %s && scripts/build_qwen.sh\n' "$OUT_DIR" >&2
  exit 1
fi

# The whole point of a static-cache build is that it loads and engages
# continuous batching. Confirm the declaration actually landed rather than
# trusting that the helper ran.
if truthy "$STATIC_CACHE" &&
  ! grep -q 'static_cache:' "$OUT_DIR/inference_metadata.yaml"; then
  printf '\nerror: %s/inference_metadata.yaml has no model.io.static_cache block.\n' \
    "$OUT_DIR" >&2
  printf '       The runtime will reject this model at load time. Re-run:\n' >&2
  printf '           %s scripts/lib/write_static_cache_metadata.py %s\n' \
    "$MOBIUS_PYTHON" "$OUT_DIR" >&2
  exit 1
fi

cat <<DONE

Built $MODEL_ID at $OUT_DIR.
Smoke test:
  cargo run --release -p onnx-genai-cli --bin onnx-genai -- generate "$OUT_DIR" \\
    --max-new-tokens 32 --prompt "Write a short Rust hello-world program."
DONE
