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
#
# A candidate must CONTAIN the models, not merely exist. This directory is a
# shared dumping ground -- `.hf_cache`, and a `.scratch` that torchinductor
# creates on its own -- so an empty-but-present `models/` is the normal state of
# a fresh worktree, and testing `[[ -d ]]` would select it and defeat the
# fallback entirely. That is not hypothetical: this script ran green earlier
# tonight, then began failing with no edit to it, because an unrelated tool
# created models/.scratch and the worktree directory started winning.
default_models_dir() {
  local candidate
  for candidate in "${REPO_ROOT}/models" "${REPO_ROOT}/../onnx-genai/models"; do
    if [[ -d "${candidate}/qwen2.5-0.5b-scatter-v2" ]] || [[ -d "${candidate}/qwen2.5-0.5b" ]]; then
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

port_is_free() {
  # A health check cannot tell OUR server from a stranger's. If something is
  # already listening, `wait_until_ready` will happily curl it, get a 200, and
  # report green while our own server is dead -- which is exactly what happened
  # when this was written: another agent's server held :8123, ours exited with
  # "Address already in use", and the script printed the success banner anyway.
  ! lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

require_free_port() {
  local port="$1" label="$2"
  port_is_free "${port}" && return 0
  fail "port ${port} (${label}) is already in use." \
    "" \
    "Something is already listening there, so this script cannot start its own" \
    "${label} server -- and a health check could not tell the difference." \
    "It would report success while serving you an unrelated process with" \
    "unknown flags, an unknown model, and an unknown build of the dashboard." \
    "" \
    "Find it, or pick another port:" \
    "" \
    "  lsof -nP -iTCP:${port} -sTCP:LISTEN" \
    "  SCATTER_PORT=9123 DYNAMIC_PORT=9124 ./examples/serving-dashboard/run-demo.sh"
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
      # Re-check liveness AFTER the health probe. A 200 proves something is
      # listening, not that it is ours; if our process died during the probe,
      # the answer came from whatever took the port.
      if ! kill -0 "${pid}" 2>/dev/null; then
        printf '\n'
        fail "the ${label} server on :${port} answered /health, but our own" \
          "server process is not running — the reply came from a DIFFERENT" \
          "process holding that port. Refusing to report success."
      fi
      printf ' ready\n'
      return 0
    fi
    printf '.'
    sleep 1
  done
  printf '\n'
  fail "the ${label} server did not become ready within ${READY_TIMEOUT_SECONDS}s."
}

require_static_cache() {
  local dir="$1" metadata="$1/inference_metadata.yaml"

  # The directory check above is name-based, and a name cannot describe
  # content. A model exported WITHOUT STATIC_CACHE=1 into a directory called
  # `-scatter-v2` passes it while being the wrong kind of model entirely --
  # and that is the failure worth catching here, because it does not look like
  # a failure. The server starts, loads it, serves it, and reports the
  # batching path as not-applicable, so the batching scenario correctly shows
  # that it never batches. Nothing is fabricated and nothing errors; the demo
  # simply cannot demonstrate the thing it exists to demonstrate, and the
  # honest `n/a` is indistinguishable from the feature being absent.
  [[ -f "${metadata}" ]] && grep -q 'static_cache' "${metadata}" && return 0

  fail "the static-cache (scatter) model has no static-cache declaration: ${dir}" \
    "" \
    "The directory exists, but ${metadata##*/} does not declare" \
    "model.io.static_cache -- so it was exported WITHOUT STATIC_CACHE=1." \
    "" \
    "This would not have failed loudly. The server would start and serve it," \
    "and the batching scenario would honestly report that this path never" \
    "batches -- which looks exactly like continuous batching not existing." \
    "" \
    "Rebuild it, including STATIC_CACHE=1:" \
    "" \
    "  STATIC_CACHE=1 MAX_SEQ_LEN=4096 OUT_DIR=${dir} \\" \
    "    scripts/build_qwen.sh" \
    "" \
    "See examples/serving-dashboard/README.md, 'Getting the models'."
}

require_tokenizer_assets() {
  local dir="$1" label="$2" rebuild="$3"
  local missing=""

  for required in tokenizer.json tokenizer_config.json; do
    [[ -f "${dir}/${required}" ]] || missing="${missing} ${required}"
  done
  [[ -n "${missing}" ]] || return 0

  # scripts/build_qwen.sh names both files in REQUIRED_ARTIFACTS for BOTH
  # runtime targets, so a directory it produced cannot reach this line. What
  # reaches it is a directory built before that check existed, or one assembled
  # by hand, or a MODELS_DIR pointed at another checkout -- and models are
  # gitignored, so those are the ordinary ways to obtain one.
  #
  # This is checked here rather than left to the server because the server does
  # not treat it as an error. ChatTemplate::from_model_dir
  # (crates/onnx-genai-ort/src/chat_template.rs:150) returns Ok with a generic
  # role-tagged DEFAULT_CHAT_TEMPLATE when tokenizer_config.json is absent, and
  # load_eos_token_ids (crates/onnx-genai-ort/src/tokenizer.rs:103) reads stop
  # ids from generation_config.json and then tokenizer_config.json, treating
  # both as optional. Qwen's stop token lives in tokenizer_config.json.
  #
  # So the whole run stays green while being wrong: the server starts, /health
  # answers 200, wait_until_ready prints `ready`, and the dashboard fills in.
  # The replies are the only symptom -- prompted with the wrong template and
  # missing the token that ends a turn, they run to the token limit. A visitor
  # reads that as this model is bad, which is the reading we can least afford.
  fail "the ${label} model directory is missing tokenizer assets:${missing}" \
    "" \
    "  ${dir}" \
    "" \
    "tokenizer_config.json carries this model's chat template and its stop" \
    "token. Without it the server still starts and still answers /health, so" \
    "this script would report success and the dashboard would look correct." \
    "Only the replies would be wrong: untemplated, and running to the token" \
    "limit because nothing tells generation where a turn ends." \
    "" \
    "Rebuild the model -- the build script requires these files, so a" \
    "completed build always has them:" \
    "" \
    "  ${rebuild}" \
    "" \
    "See examples/serving-dashboard/README.md, 'Getting the models'."
}

command -v curl >/dev/null 2>&1 || fail "curl is required to check server readiness."

# Check the models BEFORE building. The build is slow, and being told about a
# missing model directory only after waiting through it is a bad first run.
require_model "${SCATTER_MODEL}" "static-cache (scatter)" \
  "Continuous batching engages ONLY on static-cache models, so this one drives the batching scenario."
require_model "${DYNAMIC_MODEL}" "dynamic" \
  "The paged KV allocator lives on the dynamic path, so this one drives those scenarios."
require_tokenizer_assets "${SCATTER_MODEL}" "static-cache (scatter)" \
  "STATIC_CACHE=1 MAX_SEQ_LEN=4096 OUT_DIR=${SCATTER_MODEL} scripts/build_qwen.sh"
require_tokenizer_assets "${DYNAMIC_MODEL}" "dynamic" \
  "OUT_DIR=${DYNAMIC_MODEL} scripts/build_qwen.sh"
require_static_cache "${SCATTER_MODEL}"

# Check the ports BEFORE starting anything, for the same reason the models are
# checked first: a clear refusal now beats an ambiguous success later.
require_free_port "${SCATTER_PORT}" "static-cache (scatter)"
require_free_port "${DYNAMIC_PORT}" "dynamic"

# Always build. Do NOT branch on whether the binary exists.
#
# This was `if [[ ! -x "${SERVER_BIN}" ]]`, and `-x` tests a MODE BIT: it
# answers "is this file executable", never "which tree was it compiled from".
# A binary built from a checkout that lacks tonight's fixes satisfies it
# exactly as well as a correct one, and the launcher then runs the stale
# binary and prints nothing. That is not hypothetical: servers started from a
# sibling checkout ran for five hours disclosing the operator's home directory
# from `/v1/models` while the fix sat committed and unshipped, and the source
# read as fixed the entire time.
#
# `cargo build` is a no-op when the tree is already fresh, which is the whole
# argument -- cargo ALREADY tracks the thing the `-x` test was guessing at, so
# the branch bought nothing and cost the demo its correctness. Let the tool
# that tracks source freshness answer the question about source freshness.
# Captured BEFORE the build, deliberately. Reading it afterwards would defeat
# the check: HEAD can advance during a 20-40s build, and a post-build read would
# then be a commit the build never saw, so a correct binary would fail against
# it. This is the tree we are asking the build to produce.
PRE_BUILD_SHA="$(cd "${REPO_ROOT}" && git rev-parse --short=8 HEAD 2>/dev/null || echo unknown)"
printf 'building onnx-genai-server (release)...\n'
( cd "${REPO_ROOT}" && cargo build --release -p onnx-genai-server )
[[ -x "${SERVER_BIN}" ]] || fail "the server binary is missing after the build: ${SERVER_BIN}"

# Refuse to launch a binary that cannot say which commit built it, and refuse
# one that disagrees with this checkout.
#
# Rebuilding unconditionally makes the binary fresh with respect to REPO_ROOT.
# It cannot make it fresh with respect to the tree the operator THINKS they are
# running, because CARGO_TARGET_DIR is shared across worktrees here: two
# checkouts write the same path, so `${SERVER_BIN}` names a directory rather
# than a history. The build above closes the stale-binary hole; this closes the
# wrong-tree hole, and they are different holes.
#
# The server stamps its own build commit into /v1/status (build.rs ->
# ONNX_GENAI_BUILD_SHA). Comparing that against HEAD is the only check here
# that reads the binary's PROVENANCE rather than its file metadata.
# The test is "did the build we just ran produce this binary", and the direction
# is the opposite of the obvious one.
#
# First attempt was equality against HEAD. Wrong: this branch moves while you
# work and a release build takes 20-40s, so a commit landing mid-build makes the
# stamp differ through no fault of the build. A launcher that refuses at random
# gets disabled by whoever is about to present, which costs us the check.
#
# Second attempt was "stamp must be an ancestor of HEAD". ALSO WRONG, and the
# control is what caught it: the sibling checkout was branched FROM this
# history, so its HEAD is an ancestor too. That check passes the exact binaries
# that served the operator's home directory for five hours -- it scored
# identically on the fixed tree and the leaking one, which is no test at all.
#
# The correct predicate runs the ancestry the OTHER WAY. We record HEAD before
# building, then require that commit to be an ancestor of what the binary
# reports. A binary the build above produced is stamped at that commit or newer,
# so it passes. Anything OLDER -- a stale artefact, or one from a sibling
# checkout sharing this CARGO_TARGET_DIR -- cannot contain it, and fails. This
# tolerates the branch advancing and still rejects every binary that predates
# the tree we are launching from.
BUILT_SHA="$("${SERVER_BIN}" --version 2>/dev/null | awk '{print $NF}')"
if [[ -z "${BUILT_SHA}" || "${BUILT_SHA}" == "unknown" ]]; then
  # Not fatal: the stamp is newer than some binaries, and refusing here would
  # make an older-but-fine checkout unlaunchable. Say so out loud instead --
  # silence is what let the stale binaries run.
  printf 'warning: this server cannot report its build commit; provenance is unverifiable\n' >&2
elif [[ "${PRE_BUILD_SHA}" == "unknown" ]]; then
  printf 'warning: could not read this checkout HEAD; binary reports %s, unverified\n' \
    "${BUILT_SHA}" >&2
elif ( cd "${REPO_ROOT}" && git merge-base --is-ancestor "${PRE_BUILD_SHA}" "${BUILT_SHA}" 2>/dev/null ); then
  printf 'server binary provenance: %s (contains checkout %s)\n' "${BUILT_SHA}" "${PRE_BUILD_SHA}"
else
  fail "the server binary reports build ${BUILT_SHA}, which does NOT contain this
  checkout (${PRE_BUILD_SHA}) -- it predates the tree you are launching from.
  ${SERVER_BIN} is shared between worktrees, so the build above may have
  written somewhere else, or another checkout may own this path. Build in
  this checkout, or point CARGO_TARGET_DIR somewhere this checkout owns."
fi

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

printf 'starting the dynamic server on :%s (paged KV)\n' "${DYNAMIC_PORT}"
ONNX_GENAI_EP="${ONNX_GENAI_EP:-cpu}" "${SERVER_BIN}" \
  --model "${DYNAMIC_MODEL}" \
  --model-id qwen-dynamic \
  --addr "${BIND_HOST}:${DYNAMIC_PORT}" \
  --demo-assets-dir "${SCRIPT_DIR}" \
  --enable-debug-endpoints &
dynamic_pid=$!

wait_until_ready "${SCATTER_PORT}" "static-cache" "${scatter_pid}"
wait_until_ready "${DYNAMIC_PORT}" "dynamic" "${dynamic_pid}"

SCATTER_ORIGIN="http://${BIND_HOST}:${SCATTER_PORT}"
DYNAMIC_ORIGIN="http://${BIND_HOST}:${DYNAMIC_PORT}"

# The page must not hard-code a port, and must never assume its peer sits on a
# conventional one. Guessing would make a scenario poll the wrong engine and
# render that engine's structural zeros as measurements -- the exact failure
# this demo exists to argue against. THIS is the process that bound the ports,
# so it passes both addresses in the URL it prints; scenario-origins.js reads
# them and carries them across scenario navigations.
TOPOLOGY="scatter-origin=${SCATTER_ORIGIN}&dynamic-origin=${DYNAMIC_ORIGIN}"

cat <<EOF

  Open the demo:  ${SCATTER_ORIGIN}/demo/?${TOPOLOGY}

  Both servers serve the page. Each scenario reads telemetry from the server
  that can actually measure it, and switching scenarios moves you between them:

    continuous batching   ${SCATTER_ORIGIN}/demo/?${TOPOLOGY}&scenario=continuous-batching
    paged KV block table  ${DYNAMIC_ORIGIN}/demo/?${TOPOLOGY}&scenario=paged-kv

  There is deliberately no prefix-caching link: the scenario was cut rather
  than shipped as a tab.

  Opening /demo/ without those parameters still works. The scenarios backed by
  the other server then report that it is not configured, rather than quietly
  reporting this server's numbers under the other server's name.

  Ctrl-C stops both servers.

EOF

wait
