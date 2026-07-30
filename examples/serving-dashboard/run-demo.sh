#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
#
# Launch the onnx-genai serving dashboard.
#
# THIS FILE IS THE CANONICAL LAUNCH COMMAND.
#
# The same command string appears in three other places: README.md, the
# file:// guard in index.html, and the two blocking failure states rendered by
# ui/launch-command.js. Four copies of a string that must agree is a rename
# waiting to break a demo, so this script is the source and the others quote it.
# `node --test check-launch-command.test.js` asserts they still agree.
#
# Why two servers: continuous batching and paged KV are mutually exclusive in
# this runtime today. ContinuousBatchManager never touches engine.kv_cache
# (crates/onnx-genai-engine/src/engine/batched.rs:101-110), so a static-cache
# model gets batching and no page table, and a dynamic model gets the page
# table and no batching. One server would have to leave half the dashboard
# pinned at zero. Two servers show both, honestly. See README.md.

set -euo pipefail

SCATTER_PORT="${SCATTER_PORT:-8123}"
DYNAMIC_PORT="${DYNAMIC_PORT:-8124}"

# Loopback by default, matching examples/diffusion-demo's security posture.
# The server has no authentication and --enable-debug-endpoints widens its
# surface, so it must not be reachable off-box without a deliberate choice.
BIND_HOST="${BIND_HOST:-127.0.0.1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Honour CARGO_TARGET_DIR: this repo's target dir is large enough that
# contributors routinely share one across worktrees.
TARGET_DIR="${CARGO_TARGET_DIR:-${REPO_ROOT}/target}"
SERVER_BIN="${TARGET_DIR}/release/onnx-genai-server"

# Models are gitignored (.gitignore:2) and are not part of a clone. They also
# commonly live in a primary checkout rather than a worktree, so fall back to a
# sibling checkout before giving up. MODELS_DIR overrides both.
default_models_dir() {
  local candidate
  for candidate in "${REPO_ROOT}/models" "${REPO_ROOT}/../onnx-genai/models"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s' "${candidate}"
      return 0
    fi
  done
  printf '%s' "${REPO_ROOT}/models"
}

MODELS_DIR="${MODELS_DIR:-$(default_models_dir)}"
SCATTER_MODEL="${SCATTER_MODEL:-${MODELS_DIR}/qwen2.5-0.5b-scatter-v2}"
DYNAMIC_MODEL="${DYNAMIC_MODEL:-${MODELS_DIR}/qwen2.5-0.5b}"

# Model load is ~50 s cold on CPU for Qwen2.5-0.5B, so this is generous on
# purpose. A timeout here should mean "something is wrong", not "be patient".
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-240}"

scatter_pid=""
dynamic_pid=""

cleanup() {
  # Both servers die with the script. A stray server holding port 8123 is the
  # single most confusing failure mode for the next run.
  for pid in "${scatter_pid}" "${dynamic_pid}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

fail() {
  printf '\nerror: %s\n' "$1" >&2
  shift
  for line in "$@"; do printf '  %s\n' "${line}" >&2; done
  exit 1
}

require_model() {
  local dir="$1" label="$2" why="$3"
  [[ -d "${dir}" ]] && return 0
  fail "the ${label} model directory does not exist: ${dir}" \
    "" \
    "${why}" \
    "" \
    "Models are gitignored and are not part of a clone. Build it, or point" \
    "this script at a checkout that already has it:" \
    "" \
    "  MODELS_DIR=/path/to/onnx-genai/models ./examples/serving-dashboard/run-demo.sh" \
    "" \
    "See examples/serving-dashboard/README.md, 'Getting the models'."
}

wait_until_ready() {
  local port="$1" label="$2" pid="$3"
  local deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
  printf 'waiting for the %s server on :%s (first load takes ~50 s)' "${label}" "${port}"
  while ((SECONDS < deadline)); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      printf '\n'
      fail "the ${label} server exited before it became ready." \
        "Its output is above — the server prints why it could not load the model."
    fi
    if curl --silent --fail --max-time 2 "http://${BIND_HOST}:${port}/health" >/dev/null 2>&1; then
      printf ' ready\n'
      return 0
    fi
    printf '.'
    sleep 1
  done
  printf '\n'
  fail "the ${label} server did not become ready within ${READY_TIMEOUT_SECONDS}s."
}

command -v curl >/dev/null 2>&1 || fail "curl is required to check server readiness."

# Check the models BEFORE building. The build is slow, and being told about a
# missing model directory only after waiting through it is a bad first run.
require_model "${SCATTER_MODEL}" "static-cache (scatter)" \
  "Continuous batching engages ONLY on static-cache models, so this one drives the batching scenario."
require_model "${DYNAMIC_MODEL}" "dynamic" \
  "The paged KV allocator and the prefix cache live on the dynamic path, so this one drives those scenarios."

if [[ ! -x "${SERVER_BIN}" ]]; then
  printf 'building onnx-genai-server (release)...\n'
  ( cd "${REPO_ROOT}" && cargo build --release -p onnx-genai-server )
fi
[[ -x "${SERVER_BIN}" ]] || fail "the server binary is missing after the build: ${SERVER_BIN}"

# --enable-debug-endpoints carries the KV and prefix-cache fields (/v1/debug/kv)
# and the context length shown on the model card (/v1/debug/config). The
# dashboard polls both, so without this flag those fields correctly degrade to
# unavailable and the demo has much less to show.
#
# --demo-assets-dir is passed explicitly. The server otherwise looks for
# ./examples/serving-dashboard relative to its WORKING DIRECTORY, so a server
# started from anywhere but the repository root would serve a helpful error at
# /demo instead of the dashboard. Passing the script's own directory makes this
# work from any cwd.
#
# No CORS flag is needed, and no CORS code exists in the server. Each page only
# ever talks to the origin that served it: switching to a scenario hosted by the
# other server NAVIGATES there rather than fetching across origins. A
# cross-origin request is never made, so there is nothing to authorise.
#
# --enable-admin-endpoints is deliberately NOT passed. Nothing the demo does
# calls /v1/admin/*, and the server has no authentication, so enabling it would
# widen the attack surface for no demonstrated capability.

printf 'starting the static-cache server on :%s (continuous batching)\n' "${SCATTER_PORT}"
ONNX_GENAI_EP="${ONNX_GENAI_EP:-cpu}" "${SERVER_BIN}" \
  --model "${SCATTER_MODEL}" \
  --model-id qwen-scatter \
  --addr "${BIND_HOST}:${SCATTER_PORT}" \
  --demo-assets-dir "${SCRIPT_DIR}" \
  --enable-debug-endpoints &
scatter_pid=$!

printf 'starting the dynamic server on :%s (paged KV, prefix caching)\n' "${DYNAMIC_PORT}"
ONNX_GENAI_EP="${ONNX_GENAI_EP:-cpu}" "${SERVER_BIN}" \
  --model "${DYNAMIC_MODEL}" \
  --model-id qwen-dynamic \
  --addr "${BIND_HOST}:${DYNAMIC_PORT}" \
  --demo-assets-dir "${SCRIPT_DIR}" \
  --enable-debug-endpoints &
dynamic_pid=$!

wait_until_ready "${SCATTER_PORT}" "static-cache" "${scatter_pid}"
wait_until_ready "${DYNAMIC_PORT}" "dynamic" "${dynamic_pid}"

cat <<EOF

  Open the demo:  http://${BIND_HOST}:${SCATTER_PORT}/demo/

  Both servers serve the page. Each scenario reads telemetry from the server
  that can actually measure it, and switching scenarios moves you between them:

    continuous batching          http://${BIND_HOST}:${SCATTER_PORT}/demo/
    paged KV / prefix caching    http://${BIND_HOST}:${DYNAMIC_PORT}/demo/

  Ctrl-C stops both servers.

EOF

wait
