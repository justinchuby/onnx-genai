#!/usr/bin/env bash
# Verify that a built model is actually usable, not merely present.
#
# A zero exit code from a build script is a claim about the script, not about
# the artifact. This script makes the claim about the artifact, and it checks
# the two things that have silently broken before:
#
#   1. Continuous batching ENGAGES. `run_engine_driver`
#      (crates/onnx-genai-server/src/driver.rs:415-421) probes
#      `continuous_batch_manager(max_batch).is_ok()` and, when that fails,
#      logs at INFO and quietly serves from the per-request path. The server
#      starts fine and answers correctly, so "no error" proves nothing. The
#      success criterion is the log line.
#
#   2. Generation STOPS on its own. If a model directory lacks
#      tokenizer_config.json, `load_eos_token_ids`
#      (crates/onnx-genai-ort/src/tokenizer.rs:103) falls back to a fixed
#      token list. For Qwen2.5 the real stop token <|im_end|> lives only in
#      that file, and the fallback <|endoftext|> does exist in the vocabulary,
#      so nothing errors -- the model just runs to the token limit every time.
#      A finish_reason of "length" on a short prompt is that bug.
#
# This script never invokes cargo: it requires an already-built server binary
# and says exactly how to produce one if it is missing. That keeps it safe to
# run while other work holds the shared cargo target lock.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: verify_model.sh MODEL_DIR [--port PORT] [--max-batch N] [--timeout SECONDS]

Boots the server against MODEL_DIR and verifies that continuous batching
engages and that generation terminates on its own.

Environment:
  CARGO_TARGET_DIR   Where to find the prebuilt server binary.
  ONNX_GENAI_EP      Execution provider (default: cpu).
  SERVER_BIN         Explicit path to the onnx-genai-server binary.
USAGE
}

MODEL_DIR=""
PORT=8123
MAX_BATCH=4
# A cold f32 load of Qwen2.5-0.5B on CPU is roughly 50s; allow generous margin.
TIMEOUT=300

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --port) PORT="$2"; shift 2 ;;
    --max-batch) MAX_BATCH="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    -*) printf 'error: unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
    *)
      if [ -n "$MODEL_DIR" ]; then
        printf 'error: unexpected extra argument: %s\n' "$1" >&2
        exit 2
      fi
      MODEL_DIR="$1"; shift ;;
  esac
done

if [ -z "$MODEL_DIR" ]; then
  usage >&2
  exit 2
fi

# The CLI coerces a config file path to its parent directory; the server does
# not, and fails with a much less obvious message. Catch it here.
if [ -f "$MODEL_DIR" ]; then
  printf 'error: --model takes a DIRECTORY, not a file. Did you mean:\n  %s\n' \
    "$(dirname "$MODEL_DIR")" >&2
  exit 2
fi
if [ ! -d "$MODEL_DIR" ]; then
  printf 'error: no such model directory: %s\n' "$MODEL_DIR" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
SERVER_BIN="${SERVER_BIN:-$TARGET_DIR/release/onnx-genai-server}"

if [ ! -x "$SERVER_BIN" ]; then
  cat >&2 <<EOF
error: no server binary at $SERVER_BIN

This script deliberately does not run cargo, because the workspace target
directory takes an exclusive lock and concurrent builds serialize silently.
Build it yourself first, then re-run:

  CARGO_TARGET_DIR=$TARGET_DIR cargo build --release -p onnx-genai-server
EOF
  exit 1
fi

MODEL_ID="verify-$$"
LOG_FILE="$(mktemp -t verify_model_log)"
SERVER_PID=""

# shellcheck disable=SC2329  # invoked by the EXIT trap below
cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -f "$LOG_FILE"
}
trap cleanup EXIT

printf 'model     : %s\n' "$MODEL_DIR"
printf 'server    : %s\n' "$SERVER_BIN"
printf 'max_batch : %s\n\n' "$MAX_BATCH"

ONNX_GENAI_EP="${ONNX_GENAI_EP:-cpu}" \
RUST_LOG="${RUST_LOG:-info}" \
  "$SERVER_BIN" \
  --model "$MODEL_DIR" \
  --model-id "$MODEL_ID" \
  --addr "127.0.0.1:$PORT" \
  >"$LOG_FILE" 2>&1 &
SERVER_PID=$!

printf 'waiting for the model to load (up to %ss)...\n' "$TIMEOUT"
ready=""
elapsed=0
while [ "$elapsed" -lt "$TIMEOUT" ]; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    printf '\nerror: the server exited before becoming ready. Log:\n\n' >&2
    cat "$LOG_FILE" >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
  elapsed=$((elapsed + 2))
done

if [ -z "$ready" ]; then
  printf '\nerror: server did not become ready within %ss. Log:\n\n' "$TIMEOUT" >&2
  cat "$LOG_FILE" >&2
  exit 1
fi
printf 'server ready after ~%ss\n\n' "$elapsed"

FAILURES=0

# --- Check 1: continuous batching actually engaged -------------------------
# Both outcomes log at INFO, so absence of an error means nothing here.
if grep -q "continuous batch driver enabled" "$LOG_FILE"; then
  printf 'ok   continuous batch driver enabled\n'
elif grep -q "continuous batch driver disabled" "$LOG_FILE"; then
  printf 'FAIL continuous batching did NOT engage\n' >&2
  printf '     The server logged: continuous batch driver disabled; using per-request engine path\n' >&2
  printf '     This model loads and answers correctly but is NOT batching. Usually it\n' >&2
  printf '     is a dynamic-cache model, or a static-cache model whose\n' >&2
  printf '     inference_metadata.yaml has no model.io.static_cache block.\n' >&2
  FAILURES=$((FAILURES + 1))
else
  printf 'FAIL could not determine batching state from the log\n' >&2
  printf '     Expected a "continuous batch driver ..." line at INFO. Is RUST_LOG filtering it?\n' >&2
  FAILURES=$((FAILURES + 1))
fi

# --- Check 2: generation stops on its own ----------------------------------
# A model missing tokenizer_config.json runs to the token limit instead.
response="$(
  curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: hello\"}],\"max_tokens\":64}" \
    2>&1
)" || {
  printf 'FAIL chat completion request failed\n' >&2
  printf '     %s\n' "$response" >&2
  FAILURES=$((FAILURES + 1))
  response=""
}

if [ -n "$response" ]; then
  finish_reason="$(
    printf '%s' "$response" | python3 -c '
import json, sys
try:
    payload = json.load(sys.stdin)
    print(payload["choices"][0].get("finish_reason", "<absent>"))
except Exception as error:
    print(f"<unparseable: {error}>")
'
  )"
  case "$finish_reason" in
    stop)
      printf 'ok   generation stopped on its own (finish_reason=stop)\n' ;;
    length)
      printf 'FAIL generation ran to the token limit (finish_reason=length)\n' >&2
      printf '     The model never emitted a stop token. Its directory is probably missing\n' >&2
      printf '     tokenizer_config.json, so the runtime fell back to a default stop-token\n' >&2
      printf '     list that does not include this model true stop token.\n' >&2
      FAILURES=$((FAILURES + 1)) ;;
    *)
      printf 'FAIL unexpected finish_reason: %s\n' "$finish_reason" >&2
      FAILURES=$((FAILURES + 1)) ;;
  esac
fi

printf '\n'
if [ "$FAILURES" -eq 0 ]; then
  printf 'PASS %s is loadable, batching, and terminates correctly.\n' "$MODEL_DIR"
  exit 0
fi

printf '%d check(s) failed. Do not promote this model.\n' "$FAILURES" >&2
exit 1
