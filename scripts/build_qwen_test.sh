#!/usr/bin/env bash
#
# Tests for scripts/build_qwen.sh and scripts/lib/mobius_env.sh.
#
# These cover the environment handling that a fresh clone depends on: Mobius
# discovery, the resulting `mobius build` command line, and the failure
# messages. They use DRY_RUN=1 so nothing is downloaded or exported.
#
# Run:  scripts/build_qwen_test.sh
#
# Deliberately executed under /bin/bash as well as the default bash, because
# the stock macOS /bin/bash is 3.2 and has different `set -u` array semantics
# than bash 5 - a difference that previously made the default build path fail
# for everyone without Homebrew bash on PATH.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/scripts/build_qwen.sh"

# The model-fidelity checks need a real export, and models are gitignored, so
# in a worktree they used to skip unconditionally -- meaning the branch that
# gets reviewed was structurally incapable of running its own strongest
# evidence. Resolve against a sibling checkout too. Every use below is
# read-only (`--check` never writes), so pointing at a primary checkout cannot
# disturb a perf baseline.
# shellcheck source=scripts/lib/models_dir.sh
. "$ROOT/scripts/lib/models_dir.sh"

TESTS_RUN=0
TESTS_FAILED=0
TESTS_SKIPPED=0
# Skips are not equal. The model-fidelity checks are this file's strongest
# evidence, so their absence gets named specifically in the summary rather than
# folded into a total where three missing proofs look like three missing
# conveniences.
MODEL_CHECKS_SKIPPED=0

# A Python interpreter that cannot import Mobius, used to simulate a machine
# where Mobius was never installed. Falls back to skipping those cases.
PYTHON_WITHOUT_MOBIUS=""
for candidate in /opt/homebrew/bin/python3 /usr/bin/python3 python3; do
  if command -v "$candidate" >/dev/null 2>&1 &&
    ! PYTHONPATH="" "$candidate" -c \
      'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("mobius") else 1)' \
      >/dev/null 2>&1; then
    PYTHON_WITHOUT_MOBIUS="$candidate"
    break
  fi
done

fail() {
  TESTS_FAILED=$((TESTS_FAILED + 1))
  printf 'FAIL: %s\n' "$1" >&2
  if [ -n "${2:-}" ]; then
    printf '      %s\n' "$2" >&2
  fi
}

pass() {
  printf 'ok   %s\n' "$1"
}

# A skipped check is not a passing check, but "N tests, 0 failed" renders both
# identically — so a run that verified less looks exactly as strong as one that
# verified more, and the reader has no way to tell without reading every line.
# Count skips and report them in the summary, so a weaker green is visible in
# the same place a reader already looks.
skip() {
  TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
  printf 'skip %s\n' "$1"
}

# skip_model <reason> - a skip that also records lost model-fidelity evidence.
skip_model() {
  MODEL_CHECKS_SKIPPED=$((MODEL_CHECKS_SKIPPED + 1))
  skip "$1"
}

# assert_contains <name> <haystack> <needle>
assert_contains() {
  TESTS_RUN=$((TESTS_RUN + 1))
  case "$2" in
    *"$3"*) pass "$1" ;;
    *) fail "$1" "expected output to contain: $3" ;;
  esac
}

# assert_not_contains <name> <haystack> <needle>
assert_not_contains() {
  TESTS_RUN=$((TESTS_RUN + 1))
  case "$2" in
    *"$3"*) fail "$1" "expected output NOT to contain: $3" ;;
    *) pass "$1" ;;
  esac
}

# assert_status <name> <expected> <actual>
assert_status() {
  TESTS_RUN=$((TESTS_RUN + 1))
  if [ "$2" = "$3" ]; then
    pass "$1"
  else
    fail "$1" "expected exit status $2, got $3"
  fi
}

# Run build_qwen.sh under a specific bash, capturing stdout+stderr.
# Usage: run_build <bash-binary> [env assignments...]
run_build() {
  local bash_bin="$1"
  shift
  env -u PYTHONPATH DRY_RUN=1 "$@" "$bash_bin" "$SCRIPT" 2>&1
}

printf 'Testing %s\n\n' "$SCRIPT"

# ---------------------------------------------------------------------------
# Regression: the default (dynamic KV) path builds an empty extra-args array.
# On bash 3.2 with `set -u`, expanding an empty array as "${arr[@]}" aborts
# with "unbound variable", which broke the default documented invocation on
# stock macOS. Both bash versions must now succeed.
# ---------------------------------------------------------------------------
for bash_bin in /bin/bash "$(command -v bash)"; do
  [ -x "$bash_bin" ] || continue
  # Single quotes are intentional: the version must expand in the *inner* bash.
  # shellcheck disable=SC2016
  bash_label="$bash_bin ($("$bash_bin" -c 'echo ${BASH_VERSINFO[0]}.${BASH_VERSINFO[1]}'))"

  output="$(run_build "$bash_bin")"
  status=$?
  assert_status "default build succeeds under $bash_label" 0 "$status"
  assert_not_contains "no unbound-variable error under $bash_label" \
    "$output" "unbound variable"
  assert_contains "default targets models/qwen2.5-0.5b under $bash_label" \
    "$output" "models/qwen2.5-0.5b"
done

# ---------------------------------------------------------------------------
# Command-line construction
# ---------------------------------------------------------------------------
output="$(run_build /bin/bash)"
assert_contains "default omits --static-cache" "$output" "--runtime ort-genai"
assert_not_contains "default omits --static-cache flag" "$output" "--static-cache"
assert_contains "default reports dynamic kv cache" "$output" "kv cache : dynamic"

output="$(run_build /bin/bash STATIC_CACHE=1 MAX_SEQ_LEN=8192)"
assert_contains "STATIC_CACHE=1 passes --static-cache" "$output" "--static-cache"
assert_contains "STATIC_CACHE=1 passes --max-seq-len" "$output" "--max-seq-len 8192"
assert_contains "STATIC_CACHE=1 retargets output dir" "$output" "models/qwen2.5-0.5b-scatter"

# A static-cache model is only loadable if it declares model.io.static_cache,
# which lives in inference_metadata.yaml. Only --runtime onnx-genai emits that
# file; --runtime ort-genai writes genai_config.json, which cannot express it,
# and the runtime then rejects the model at load time.
assert_contains "STATIC_CACHE=1 targets the onnx-genai runtime" \
  "$output" "--runtime onnx-genai"
assert_not_contains "STATIC_CACHE=1 does not use the ort-genai runtime" \
  "$output" "--runtime ort-genai"
assert_contains "STATIC_CACHE=1 writes the static_cache declaration" \
  "$output" "write_static_cache_metadata.py"

# The onnx-genai target writes only tokenizer.json. tokenizer_config.json
# carries Qwen's real stop token (<|im_end|>) and its chat template; without it
# the model loads but never stops generating.
assert_contains "STATIC_CACHE=1 restores the tokenizer companion files" \
  "$output" "write_tokenizer_assets.py"

# The dynamic path is a different contract: genai_config.json is what the
# runtime loads there, and it is known to work.
output="$(run_build /bin/bash)"
assert_contains "dynamic build targets the ort-genai runtime" \
  "$output" "--runtime ort-genai"
assert_not_contains "dynamic build does not write a static_cache declaration" \
  "$output" "write_static_cache_metadata.py"
# ort-genai already copies the tokenizer companions itself.
assert_not_contains "dynamic build does not re-copy tokenizer assets" \
  "$output" "write_tokenizer_assets.py"

# Legacy alias kept for existing docs and benchmark scripts.
output="$(run_build /bin/bash SCATTER_CACHE=1)"
assert_contains "SCATTER_CACHE=1 alias still enables static cache" "$output" "--static-cache"

# Boolean spellings
output="$(run_build /bin/bash STATIC_CACHE=true)"
assert_contains "STATIC_CACHE=true is truthy" "$output" "--static-cache"
output="$(run_build /bin/bash STATIC_CACHE=yes)"
assert_contains "STATIC_CACHE=yes is truthy" "$output" "--static-cache"
output="$(run_build /bin/bash STATIC_CACHE=0)"
assert_not_contains "STATIC_CACHE=0 is falsy" "$output" "--static-cache"

# Pass-through env vars
output="$(run_build /bin/bash DTYPE=f16 EP=webgpu)"
assert_contains "DTYPE is forwarded" "$output" "--dtype f16"
assert_contains "EP is forwarded" "$output" "--ep webgpu"

output="$(run_build /bin/bash OUT_DIR=/tmp/custom-qwen-out)"
assert_contains "OUT_DIR overrides the default" "$output" "/tmp/custom-qwen-out"

output="$(run_build /bin/bash MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct)"
assert_contains "MODEL_ID is forwarded" "$output" "--model Qwen/Qwen2.5-1.5B-Instruct"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
output="$(run_build /bin/bash STATIC_CACHE=1 MAX_SEQ_LEN=lots)"
status=$?
assert_status "non-numeric MAX_SEQ_LEN is rejected" 2 "$status"
assert_contains "non-numeric MAX_SEQ_LEN explains itself" \
  "$output" "MAX_SEQ_LEN must be a positive integer"

output="$(env -u PYTHONPATH DRY_RUN=1 /bin/bash "$SCRIPT" --bogus 2>&1)"
status=$?
assert_status "unknown argument is rejected" 2 "$status"

output="$(env -u PYTHONPATH /bin/bash "$SCRIPT" --help 2>&1)"
status=$?
assert_status "--help exits 0" 0 "$status"
assert_contains "--help documents MOBIUS_DIR" "$output" "MOBIUS_DIR"
assert_contains "--help documents STATIC_CACHE" "$output" "STATIC_CACHE"
assert_contains "--help links the real Mobius repo" \
  "$output" "github.com/onnxruntime/mobius"

# ---------------------------------------------------------------------------
# Mobius discovery failures must be actionable, never a Python traceback.
# ---------------------------------------------------------------------------
output="$(run_build /bin/bash MOBIUS_DIR=/nonexistent/mobius)"
status=$?
assert_status "invalid MOBIUS_DIR fails" 1 "$status"
assert_contains "invalid MOBIUS_DIR names the offending path" \
  "$output" "/nonexistent/mobius"
assert_contains "invalid MOBIUS_DIR says what it looked for" \
  "$output" "src/mobius/__init__.py"
assert_not_contains "invalid MOBIUS_DIR does not leak a traceback" \
  "$output" "Traceback"
assert_not_contains "invalid MOBIUS_DIR does not reach python -m mobius" \
  "$output" "No module named"

# An explicitly requested MOBIUS_DIR must never silently fall back to an
# installed Mobius: that would build from code the user did not ask for.
assert_not_contains "invalid MOBIUS_DIR does not fall back silently" \
  "$output" "DRY_RUN=1, not building"

if [ -n "$PYTHON_WITHOUT_MOBIUS" ]; then
  # Simulate a genuinely fresh machine: an interpreter without Mobius, a HOME
  # with no checkout, and a repo root with no sibling checkout.
  fresh_root="$(mktemp -d)"
  fresh_home="$(mktemp -d)"
  mkdir -p "$fresh_root/repo/scripts"
  cp -R "$ROOT/scripts/lib" "$fresh_root/repo/scripts/lib"
  cp "$SCRIPT" "$fresh_root/repo/scripts/build_qwen.sh"

  output="$(env -u PYTHONPATH -u MOBIUS_DIR -u MOBIUS_ROOT DRY_RUN=1 \
    HOME="$fresh_home" PYTHON="$PYTHON_WITHOUT_MOBIUS" \
    /bin/bash "$fresh_root/repo/scripts/build_qwen.sh" 2>&1)"
  status=$?

  assert_status "fresh clone without Mobius fails cleanly" 1 "$status"
  assert_contains "fresh clone explains Mobius is required" \
    "$output" "could not find Mobius"
  assert_contains "fresh clone gives an install command" \
    "$output" "git+https://github.com/onnxruntime/mobius"
  assert_contains "fresh clone offers the MOBIUS_DIR escape hatch" \
    "$output" "MOBIUS_DIR=/path/to/mobius"
  # `pip install mobius` grabs an unrelated PyPI package; warn explicitly.
  assert_contains "fresh clone warns about the PyPI name collision" \
    "$output" "mobius-ai"
  assert_not_contains "fresh clone does not leak a traceback" \
    "$output" "Traceback"

  rm -rf "$fresh_root" "$fresh_home"
else
  skip 'fresh-clone cases (every interpreter here already has mobius)'
fi

# ---------------------------------------------------------------------------
# The helper must be usable standalone by other build scripts.
# ---------------------------------------------------------------------------
TESTS_RUN=$((TESTS_RUN + 1))
if /bin/bash -c "set -euo pipefail
  ROOT='$ROOT'
  . '$ROOT/scripts/lib/mobius_env.sh'
  mobius_resolve \"\$ROOT\"
  [ -n \"\$MOBIUS_PYTHON\" ] && [ -n \"\$MOBIUS_SOURCE\" ]" >/dev/null 2>&1; then
  pass "mobius_env.sh is sourceable and exports MOBIUS_PYTHON/MOBIUS_SOURCE"
else
  fail "mobius_env.sh is sourceable and exports MOBIUS_PYTHON/MOBIUS_SOURCE"
fi

# ---------------------------------------------------------------------------
# The static_cache metadata generator. A static-cache model that does not
# declare model.io.static_cache is rejected at load time, so this block is what
# makes the STATIC_CACHE=1 output usable at all.
# ---------------------------------------------------------------------------
GENERATOR="$ROOT/scripts/lib/write_static_cache_metadata.py"
GENERATOR_PYTHON="${MOBIUS_PYTHON:-}"
if [ -z "$GENERATOR_PYTHON" ]; then
  # Reuse the script's own discovery so this works wherever the build does.
  GENERATOR_PYTHON="$(
    . "$ROOT/scripts/lib/mobius_env.sh" >/dev/null 2>&1
    mobius_resolve "$ROOT" >/dev/null 2>&1 && printf '%s' "$MOBIUS_PYTHON"
  )"
fi

# assert_derives_io <name> <model-dir> <ground-truth-yaml>
# Derives the io block from the ONNX graph and compares it to a known-good
# inference_metadata.yaml. --check is read-only, so this never mutates a model.
assert_derives_io() {
  TESTS_RUN=$((TESTS_RUN + 1))
  local name="$1" model_dir="$2" truth="$3"
  local derived
  # A missing committed input and a wrong generator are opposite diagnoses that
  # otherwise land as the same opaque traceback: one means "the ground truth is
  # gone", the other means "the code under test is broken". Name which input
  # vanished, or the reader debugs the generator while the fixture is the fault.
  if [ ! -d "$model_dir" ]; then
    fail "$name" "model dir is absent, so nothing was compared: $model_dir"
    return
  fi
  if [ ! -f "$truth" ]; then
    fail "$name" "ground-truth metadata is absent, so nothing was compared: $truth"
    return
  fi
  if ! derived="$("$GENERATOR_PYTHON" "$GENERATOR" "$model_dir" --check 2>&1)"; then
    fail "$name" "generator failed: $derived"
    return
  fi
  if printf '%s' "$derived" | "$GENERATOR_PYTHON" -c '
import sys, yaml
derived = yaml.safe_load(sys.stdin)["io"]
truth = yaml.safe_load(open(sys.argv[1]))["model"]["io"]
sys.exit(0 if derived == truth else 1)
' "$truth"; then
    pass "$name"
  else
    fail "$name" "derived io block does not match $truth"
  fi
}

# These two cover the precondition branches above. They deliberately sit OUTSIDE
# the GENERATOR_PYTHON guard: the checks run before the generator is invoked, so
# a fresh clone with no onnx/yaml installed still proves the diagnostics work.
assert_contains "absent model dir is reported as a missing input, not a generator bug" \
  "$( (assert_derives_io "probe" "$ROOT/no-such-model-dir" "$ROOT/no-such-truth.yaml") 2>&1 )" \
  "model dir is absent"

assert_contains "absent ground truth is named, and distinguished from a mismatch" \
  "$( (assert_derives_io "probe" "$ROOT/scripts" "$ROOT/no-such-truth.yaml") 2>&1 )" \
  "ground-truth metadata is absent"

if [ -n "$GENERATOR_PYTHON" ] &&
  "$GENERATOR_PYTHON" -c 'import onnx, yaml' >/dev/null 2>&1; then

  # The committed fixture is the ground truth available to a fresh clone.
  assert_derives_io "generator reproduces the tiny-llm-scatter fixture io block" \
    "$ROOT/tests/fixtures/tiny-llm-scatter" \
    "$ROOT/tests/fixtures/tiny-llm-scatter/inference_metadata.yaml"

  # A real 24-layer export, when one has been built locally. This is the case
  # that catches lexical-vs-numeric layer ordering: sorting the port names as
  # strings puts key_cache.10 before key_cache.2 and silently mis-pairs every
  # buffer past the ninth layer, which the 1-layer fixture cannot detect.
  real_scatter="$(resolve_model_dir "$ROOT" qwen2.5-0.5b-scatter-v2 || true)"
  if [ -n "$real_scatter" ] && [ -f "$real_scatter/model.onnx" ] &&
    [ -f "$real_scatter/inference_metadata.yaml" ]; then
    assert_derives_io "generator reproduces the 24-layer scatter model io block" \
      "$real_scatter" "$real_scatter/inference_metadata.yaml"
  else
    skip_model "24-layer generator check (no qwen2.5-0.5b-scatter-v2 in any candidate models dir)"
  fi

  # Ordering is a correctness property, not a cosmetic one. Reuse the --check
  # output captured in the shell; never invoke the generator without --check
  # from the tests, since without it the generator WRITES to the model dir.
  if [ -n "$real_scatter" ] && [ -f "$real_scatter/model.onnx" ]; then
    TESTS_RUN=$((TESTS_RUN + 1))
    derived_block="$("$GENERATOR_PYTHON" "$GENERATOR" "$real_scatter" --check 2>&1)"
    if printf '%s' "$derived_block" | "$GENERATOR_PYTHON" -c '
import sys, yaml
caches = yaml.safe_load(sys.stdin)["io"]["static_cache"]["key_cache_inputs"]
indices = [int(name.split(".")[1]) for name in caches]
sys.exit(0 if indices == list(range(len(indices))) else 1)
'; then
      pass "cache ports are ordered numerically by layer, not lexically"
    else
      fail "cache ports are ordered numerically by layer, not lexically" \
        "got: $(printf '%s' "$derived_block" | grep -c 'key_cache\.') entries out of order"
    fi
  else
    skip_model 'numeric layer ordering check (no local scatter model)'
  fi

  # A dynamic-cache model has no scatter ABI; say so instead of emitting a
  # bogus declaration.
  TESTS_RUN=$((TESTS_RUN + 1))
  dynamic_model="$(resolve_model_dir "$ROOT" qwen2.5-0.5b || true)"
  if [ -n "$dynamic_model" ] && [ -f "$dynamic_model/model.onnx" ]; then
    if output="$("$GENERATOR_PYTHON" "$GENERATOR" "$dynamic_model" --check 2>&1)"; then
      fail "generator rejects a dynamic-cache model"
    else
      case "$output" in
        *"does not expose a static-cache scatter ABI"*)
          pass "generator rejects a dynamic-cache model" ;;
        *) fail "generator rejects a dynamic-cache model" "unhelpful error: $output" ;;
      esac
    fi
  else
    TESTS_RUN=$((TESTS_RUN - 1))
    skip_model "dynamic-model rejection check (no qwen2.5-0.5b in any candidate models dir)"
  fi
else
  skip 'static_cache generator checks (onnx/pyyaml unavailable)'
fi

# ---------------------------------------------------------------------------
# scripts/lib/write_tokenizer_assets.py
#
# Mobius's onnx-genai runtime target writes only tokenizer.json. These cases
# use a local source directory so they are hermetic and need no network.
# ---------------------------------------------------------------------------
TOKENIZER_ASSETS="$ROOT/scripts/lib/write_tokenizer_assets.py"
ASSETS_PYTHON="${GENERATOR_PYTHON:-}"
[ -n "$ASSETS_PYTHON" ] || ASSETS_PYTHON="$(command -v python3 || true)"

if [ -n "$ASSETS_PYTHON" ]; then
  assets_tmp="$(mktemp -d)"
  trap 'rm -rf "$assets_tmp"' EXIT
  mkdir -p "$assets_tmp/src" "$assets_tmp/out"
  printf '{"eos_token": "<|im_end|>", "chat_template": "x"}\n' \
    >"$assets_tmp/src/tokenizer_config.json"
  printf '{}\n' >"$assets_tmp/src/vocab.json"
  printf 'a b\n' >"$assets_tmp/src/merges.txt"
  # The exporter already wrote this one; it must not be replaced.
  printf 'EXPORTER\n' >"$assets_tmp/out/merges.txt"

  # --check must never write. An earlier revision of the sibling generator's
  # test dropped this flag and mutated a shared model directory.
  TESTS_RUN=$((TESTS_RUN + 1))
  check_output="$("$ASSETS_PYTHON" "$TOKENIZER_ASSETS" "$assets_tmp/src" "$assets_tmp/out" --check 2>&1)"
  if [ -f "$assets_tmp/out/tokenizer_config.json" ]; then
    fail "--check writes nothing" "it created tokenizer_config.json"
  else
    pass "--check writes nothing"
  fi
  assert_contains "--check reports what it would copy" "$check_output" "would copy"

  TESTS_RUN=$((TESTS_RUN + 1))
  if copy_output="$("$ASSETS_PYTHON" "$TOKENIZER_ASSETS" "$assets_tmp/src" "$assets_tmp/out" 2>&1)"; then
    pass "copies missing tokenizer companions"
  else
    fail "copies missing tokenizer companions" "$copy_output"
  fi

  TESTS_RUN=$((TESTS_RUN + 1))
  if [ -f "$assets_tmp/out/tokenizer_config.json" ] && [ -f "$assets_tmp/out/vocab.json" ]; then
    pass "tokenizer_config.json and vocab.json land in the output dir"
  else
    fail "tokenizer_config.json and vocab.json land in the output dir"
  fi

  # Whatever the exporter wrote is authoritative.
  TESTS_RUN=$((TESTS_RUN + 1))
  if [ "$(cat "$assets_tmp/out/merges.txt")" = "EXPORTER" ]; then
    pass "existing exporter output is never overwritten"
  else
    fail "existing exporter output is never overwritten"
  fi

  # tokenizer.json is the exporter's own fast-format copy and is referenced by
  # inference_metadata.yaml; copying the source one over it would be wrong.
  TESTS_RUN=$((TESTS_RUN + 1))
  if [ -f "$assets_tmp/out/tokenizer.json" ]; then
    fail "does not copy tokenizer.json" "the exporter owns that file"
  else
    pass "does not copy tokenizer.json"
  fi

  # A source without tokenizer_config.json yields a model that loads but never
  # stops generating, so it must fail loudly rather than silently.
  TESTS_RUN=$((TESTS_RUN + 1))
  mkdir -p "$assets_tmp/empty_src" "$assets_tmp/empty_out"
  if output="$("$ASSETS_PYTHON" "$TOKENIZER_ASSETS" "$assets_tmp/empty_src" "$assets_tmp/empty_out" 2>&1)"; then
    fail "missing tokenizer_config.json is an error"
  else
    case "$output" in
      *"never stopping"*|*"never stop generating"*)
        pass "missing tokenizer_config.json is an error" ;;
      *) fail "missing tokenizer_config.json is an error" "unhelpful error: $output" ;;
    esac
  fi

  TESTS_RUN=$((TESTS_RUN + 1))
  if output="$("$ASSETS_PYTHON" "$TOKENIZER_ASSETS" "$assets_tmp/src" "$assets_tmp/nonexistent" 2>&1)"; then
    fail "a missing output directory is an error"
  else
    case "$output" in
      *"no such directory"*) pass "a missing output directory is an error" ;;
      *) fail "a missing output directory is an error" "unhelpful error: $output" ;;
    esac
  fi

  rm -rf "$assets_tmp"
  trap - EXIT
else
  skip 'tokenizer asset checks (no python3)'
fi

# ---------------------------------------------------------------------------
# Static-cache detection, driven by SYNTHETIC graphs.
#
# "If the graph has no static-cache scatter ABI it is not a static-cache model,
# and continuous batching will not engage" is the highest-frequency confusion on
# this project, so it must be checked mechanically rather than remembered.
#
# The real-model rejection case above needs models/qwen2.5-0.5b, which exists
# only in the main checkout - so it SKIPS on the branch the PR is cut from,
# leaving the assertion unverified exactly where it gets reviewed. These cases
# build throwaway ONNX graphs instead: no weights, no network, a few kilobytes,
# and they run in every checkout.
#
# The predicate is deliberately IDENTICAL to the runtime's fail-closed guard
# (crates/onnx-genai-ort/src/decode/io.rs:198): the presence of a write_indices
# or nonpad_kv_seqlen INPUT. Neither side ever inspects TensorScatter *nodes* -
# checking for the op would be testing a different property than the one the
# loader actually enforces.
#
# The last case is a POSITIVE control. Without it, all three rejection cases
# would still pass if the generator rejected every graph unconditionally.
# ---------------------------------------------------------------------------
if [ -n "${GENERATOR_PYTHON:-}" ] &&
  "$GENERATOR_PYTHON" -c 'import onnx, yaml' >/dev/null 2>&1; then

  synth_tmp="$(mktemp -d)"
  trap 'rm -rf "$synth_tmp"' EXIT

  # make_synthetic_graph <kind> <dir>
  make_synthetic_graph() {
    "$GENERATOR_PYTHON" - "$1" "$2" <<'PY'
import sys
import onnx
from onnx import TensorProto, helper

kind, out_dir = sys.argv[1], sys.argv[2]

def f32(name):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1, 1, 1])

def i64(name):
    return helper.make_tensor_value_info(name, TensorProto.INT64, [1])

inputs = [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 1])]
outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1])]
nodes = [helper.make_node("Cast", ["input_ids"], ["logits"], to=TensorProto.FLOAT)]

# Everything except "plain" carries the integer scatter control ports that the
# runtime keys on.
if kind != "plain":
    inputs += [i64("write_indices"), i64("nonpad_kv_seqlen")]

# TWELVE layers, not two, and the count is load-bearing. The runtime pairs the
# four cache lists POSITIONALLY, so a lexical sort mis-binds every buffer past
# layer 9 ("key_cache.10" sorts before "key_cache.2") with no crash and no
# wrong-looking output. Below ten layers, lexical and numeric order are
# IDENTICAL - an ordering assertion over two layers cannot fail on the very
# defect it exists to catch.
LAYERS = list(range(12)) if kind == "complete" else [0, 1]

if kind in ("mispaired", "complete"):
    for layer in LAYERS:
        inputs += [f32("key_cache.%d" % layer), f32("value_cache.%d" % layer)]
        for role in ("key", "value"):
            # "mispaired" omits updated_key_cache.1, so the four lists no longer
            # line up positionally - the silent KV mis-binding this guards.
            if kind == "mispaired" and role == "key" and layer == 1:
                continue
            src = "%s_cache.%d" % (role, layer)
            dst = "updated_%s_cache.%d" % (role, layer)
            outputs.append(f32(dst))
            nodes.append(helper.make_node("Identity", [src], [dst]))

graph = helper.make_graph(nodes, "synthetic-%s" % kind, inputs, outputs)
onnx.save(helper.make_model(graph), "%s/model.onnx" % out_dir)
PY
  }

  # assert_generator_rejects <name> <kind> <expected substring>
  assert_generator_rejects() {
    TESTS_RUN=$((TESTS_RUN + 1))
    local name="$1" kind="$2" needle="$3"
    local dir="$synth_tmp/$kind" output
    mkdir -p "$dir"
    if ! make_synthetic_graph "$kind" "$dir" >/dev/null 2>&1; then
      fail "$name" "could not build the synthetic '$kind' graph"
      return
    fi
    if output="$("$GENERATOR_PYTHON" "$GENERATOR" "$dir" --check 2>&1)"; then
      fail "$name" "generator ACCEPTED a '$kind' graph and emitted: $output"
      return
    fi
    case "$output" in
      *"$needle"*) pass "$name" ;;
      *) fail "$name" "rejected, but the message would not tell you why: $output" ;;
    esac
  }

  assert_generator_rejects \
    "a graph with no scatter ABI is rejected, not silently declared static" \
    plain "does not expose a static-cache scatter ABI"

  assert_generator_rejects \
    "scatter control ports without cache ports are rejected as incomplete" \
    control_only "no key_cache.N inputs"

  assert_generator_rejects \
    "a graph missing one layer's cache output is rejected before it mis-pairs" \
    mispaired "mis-pair"

  # Positive control, and the ordering guard. The expected list is spelled out to
  # layer 11 on purpose: it is only past layer 9 that a lexical sort diverges
  # from a numeric one, so this is the assertion that can actually go red on the
  # silent mis-pairing bug.
  TESTS_RUN=$((TESTS_RUN + 1))
  complete_dir="$synth_tmp/complete"
  mkdir -p "$complete_dir"
  if make_synthetic_graph complete "$complete_dir" >/dev/null 2>&1 &&
    complete_out="$("$GENERATOR_PYTHON" "$GENERATOR" "$complete_dir" --check 2>&1)"; then
    if printf '%s' "$complete_out" | "$GENERATOR_PYTHON" -c '
import sys, yaml
spec = yaml.safe_load(sys.stdin)["io"]["static_cache"]
expected = ["key_cache.%d" % i for i in range(12)]
ok = (
    spec["write_indices_input"] == "write_indices"
    and spec["kv_sequence_length_input"] == "nonpad_kv_seqlen"
    and spec["key_cache_inputs"] == expected
    and spec["value_cache_inputs"] == ["value_cache.%d" % i for i in range(12)]
    and spec["key_cache_outputs"] == ["updated_key_cache.%d" % i for i in range(12)]
    and spec["value_cache_outputs"] == ["updated_value_cache.%d" % i for i in range(12)]
)
if not ok:
    print("got key_cache_inputs=%r" % (spec.get("key_cache_inputs"),), file=sys.stderr)
sys.exit(0 if ok else 1)
' 2>"$synth_tmp/order.err"; then
      pass "cache ports pair by NUMERIC layer index, not lexically (12 layers)"
    else
      fail "cache ports pair by NUMERIC layer index, not lexically (12 layers)" \
        "$(cat "$synth_tmp/order.err" 2>/dev/null)"
    fi
  else
    fail "cache ports pair by NUMERIC layer index, not lexically (12 layers)" \
      "generator rejected a valid graph: ${complete_out:-<build failed>}"
  fi

  rm -rf "$synth_tmp"
  trap - EXIT
else
  skip 'synthetic static-cache detection checks (onnx/pyyaml unavailable)'
fi

# ---------------------------------------------------------------------------
# OUT_DIR safety (D4/D5).
#
# The motivating case is real: the demo README's copy-pasteable build command
# targets models/qwen2.5-0.5b-scatter-v2, which is the perf-baseline model. A
# build there would overwrite 2 GB of reference weights in place, with no
# confirmation and no backup.
# ---------------------------------------------------------------------------
outdir_tmp="$(mktemp -d)"

TESTS_RUN=$((TESTS_RUN + 1))
mkdir -p "$outdir_tmp/occupied"
printf 'pretend weights\n' >"$outdir_tmp/occupied/model.onnx"
if output="$(OUT_DIR="$outdir_tmp/occupied" STATIC_CACHE=1 \
  "$SCRIPT" 2>&1)"; then
  fail "refuses to build into a non-empty OUT_DIR"
else
  case "$output" in
    *"already exists and is not empty"*)
      pass "refuses to build into a non-empty OUT_DIR" ;;
    *) fail "refuses to build into a non-empty OUT_DIR" "unhelpful error: $output" ;;
  esac
fi

# The refusal must say what to do next, not merely say no.
assert_contains "the non-empty refusal names a way forward" \
  "$(OUT_DIR="$outdir_tmp/occupied" STATIC_CACHE=1 "$SCRIPT" 2>&1 || true)" \
  "FORCE=1"

# A model you still need must be nameable as such, or the message reads as
# bureaucracy and gets overridden reflexively.
assert_contains "the non-empty refusal warns about overwriting a real model" \
  "$(OUT_DIR="$outdir_tmp/occupied" STATIC_CACHE=1 "$SCRIPT" 2>&1 || true)" \
  "perf baseline"

TESTS_RUN=$((TESTS_RUN + 1))
if output="$(OUT_DIR="$outdir_tmp/occupied" STATIC_CACHE=1 FORCE=1 DRY_RUN=1 \
  "$SCRIPT" 2>&1)"; then
  case "$output" in
    *"FORCE=1 given, building anyway"*)
      pass "FORCE=1 overrides the non-empty refusal" ;;
    *) fail "FORCE=1 overrides the non-empty refusal" "no override notice: $output" ;;
  esac
else
  fail "FORCE=1 overrides the non-empty refusal" "still refused: $output"
fi

# An EMPTY directory is fine - the guard must not block the ordinary re-run.
TESTS_RUN=$((TESTS_RUN + 1))
mkdir -p "$outdir_tmp/empty"
if output="$(OUT_DIR="$outdir_tmp/empty" STATIC_CACHE=1 DRY_RUN=1 "$SCRIPT" 2>&1)"; then
  pass "an empty OUT_DIR is accepted"
else
  fail "an empty OUT_DIR is accepted" "refused an empty directory: $output"
fi

# D5: the package suffixes route loading down a different code path.
for suffix in ortpackage nxpackage; do
  TESTS_RUN=$((TESTS_RUN + 1))
  if output="$(OUT_DIR="$outdir_tmp/model.$suffix" STATIC_CACHE=1 DRY_RUN=1 \
    "$SCRIPT" 2>&1)"; then
    fail "rejects an OUT_DIR named .$suffix"
  else
    case "$output" in
      *"must not end in .ortpackage"*)
        pass "rejects an OUT_DIR named .$suffix" ;;
      *) fail "rejects an OUT_DIR named .$suffix" "unhelpful error: $output" ;;
    esac
  fi
done

# A scatter-named directory promises a static-cache model. Building a dynamic
# one into it produces a model that loads, serves, and never batches, with no
# error anywhere downstream — so the producer refuses it.
TESTS_RUN=$((TESTS_RUN + 1))
if output="$(OUT_DIR="$outdir_tmp/qwen2.5-0.5b-scatter-v2" DRY_RUN=1 \
  "$SCRIPT" 2>&1)"; then
  fail "refuses a scatter-named OUT_DIR without STATIC_CACHE" \
    "built a dynamic model into a scatter-named directory: $output"
else
  case "$output" in
    *"STATIC_CACHE is not set"*)
      pass "refuses a scatter-named OUT_DIR without STATIC_CACHE" ;;
    *) fail "refuses a scatter-named OUT_DIR without STATIC_CACHE" \
        "unhelpful error: $output" ;;
  esac
fi

# The same name WITH STATIC_CACHE=1 is the intended path and must stay open.
TESTS_RUN=$((TESTS_RUN + 1))
if output="$(OUT_DIR="$outdir_tmp/qwen2.5-0.5b-scatter-v2" STATIC_CACHE=1 \
  DRY_RUN=1 "$SCRIPT" 2>&1)"; then
  pass "allows a scatter-named OUT_DIR when STATIC_CACHE is set"
else
  fail "allows a scatter-named OUT_DIR when STATIC_CACHE is set" \
    "the guard rejected the correct invocation: $output"
fi

rm -rf "$outdir_tmp"

# The Mobius error points readers at a README heading. A pointer to a heading
# that has been renamed is worse than no pointer: it sends someone to search a
# document for text that is not there. Assert the anchor still exists.
TESTS_RUN=$((TESTS_RUN + 1))
cited_heading="Build a model with Mobius"
if grep -q "^#\{1,4\} $cited_heading" "$ROOT/README.md"; then
  pass "the README heading cited by the Mobius error still exists"
else
  fail "the README heading cited by the Mobius error still exists" \
    "scripts/lib/mobius_env.sh sends readers to README.md section '$cited_heading', which no longer exists; update both together"
fi

# ---------------------------------------------------------------------------
# scripts/lib/models_dir.sh
#
# The resolver decides whether this file's strongest checks RUN or SKIP, so a
# bug here is silent by construction: it does not fail anything, it just makes
# evidence quietly stop existing. These use synthetic trees under a temp dir,
# never the real models directory.
# ---------------------------------------------------------------------------

models_fixture="$(mktemp -d)"
mkdir -p "$models_fixture/repo/models/.hf_cache" \
  "$models_fixture/repo/models/.scratch" \
  "$models_fixture/onnx-genai/models/qwen2.5-0.5b-scatter-v2"
: >"$models_fixture/onnx-genai/models/qwen2.5-0.5b-scatter-v2/model.onnx"

# THE TRAP THIS FILE EXISTS FOR: an empty-but-present models/ holding only
# .hf_cache and .scratch is the normal state of a fresh worktree. A `[[ -d ]]`
# test on the directory selects it and defeats the sibling fallback, which is
# exactly how run-demo.sh started failing tonight with no edit to it.
TESTS_RUN=$((TESTS_RUN + 1))
resolved="$(MODELS_DIR='' resolve_model_dir "$models_fixture/repo" qwen2.5-0.5b-scatter-v2 || true)"
# Compare against the physical path: the fixture root itself may be a symlink
# (macOS /tmp is), so a literal string compare fails for a reason that has
# nothing to do with the behaviour under test.
expected_sibling="$(cd "$models_fixture/onnx-genai/models/qwen2.5-0.5b-scatter-v2" && pwd -P)"
case "$resolved" in
  "$expected_sibling")
    pass "an empty-but-present models/ does not defeat the sibling fallback" ;;
  *)
    fail "an empty-but-present models/ does not defeat the sibling fallback" \
      "resolved to '$resolved'; a dir holding only .hf_cache/.scratch must lose to a sibling that has the model" ;;
esac

# A dotfile-only directory must not read as "contains a model". Without this,
# the check above could pass for the wrong reason on some other machine.
TESTS_RUN=$((TESTS_RUN + 1))
if models_dir_contains_model "$models_fixture/repo/models"; then
  fail "a models dir holding only dotfiles does not count as containing a model" \
    "an empty-but-present dir was accepted, so the fallback would never be reached"
else
  pass "a models dir holding only dotfiles does not count as containing a model"
fi

# An explicit override must never silently fall through to somewhere the caller
# did not name -- that would run a check against a model they did not choose
# and report it under their command.
TESTS_RUN=$((TESTS_RUN + 1))
if MODELS_DIR="$models_fixture/repo/models" \
  resolve_model_dir "$models_fixture/repo" qwen2.5-0.5b-scatter-v2 >/dev/null 2>&1; then
  fail "MODELS_DIR does not fall back to a sibling checkout" \
    "an explicit MODELS_DIR that lacks the model resolved anyway, so the run silently used a model the caller did not name"
else
  pass "MODELS_DIR does not fall back to a sibling checkout"
fi

# Resolution is per MODEL, not per directory: a checkout may hold the scatter
# model and not the dynamic one, and a whole-directory match would send both
# lookups to a root that satisfies only one.
TESTS_RUN=$((TESTS_RUN + 1))
if MODELS_DIR='' resolve_model_dir "$models_fixture/repo" qwen2.5-0.5b >/dev/null 2>&1; then
  fail "a model missing from every candidate resolves to nothing" \
    "resolved a model that exists nowhere, so a check would run against the wrong export"
else
  pass "a model missing from every candidate resolves to nothing"
fi

rm -rf "$models_fixture"

# The loud banner is the only thing standing between a narrow run and a reader
# who believes it was a full one, and it lives at the very end of this file
# where it is easy to lose in an edit. Guard its two load-bearing properties:
# that a model-fidelity skip is counted separately, and that the banner names
# the command that restores the evidence. A banner that reported only a TOTAL
# would say "3 skipped" for three missing conveniences and three missing proofs
# alike, which is the exact ambiguity it exists to remove.
TESTS_RUN=$((TESTS_RUN + 1))
# Written to a file rather than piped into grep: under `set -o pipefail`,
# `producer | grep -q` lets grep exit early on a match, the producer takes
# SIGPIPE, and the pipeline reports 141 -- turning a MATCH into a failure,
# nondeterministically, depending on which process wins the race.
self_src_file="$(mktemp)"
grep -v '^[[:space:]]*#' "$ROOT/scripts/build_qwen_test.sh" >"$self_src_file"
missing_banner_parts=""
# Each pattern uses a glob char class so it does NOT literally contain the
# string it searches for. Without that, these three patterns are themselves
# occurrences, the guard matches its own source, and it passes green with the
# entire banner deleted. That has now happened three times in this repo; a
# self-inspecting test must be written so it cannot see itself.
# Each pattern uses a regex char class so it does NOT literally contain the
# string it searches for. Without that, these patterns are themselves
# occurrences, the guard matches its own source, and it passes green with the
# entire banner deleted. That has now happened three times in this repo; a
# self-inspecting test must be written so it cannot see itself.
# shellcheck disable=SC2016  # the $(( )) is a literal being searched for, not an expansion
if ! grep -q 'MODEL_CHECK[S]_SKIPPED=\$((MODEL_CHECKS_SKIPPED + 1))' "$self_src_file"; then
  missing_banner_parts="$missing_banner_parts skip_model-does-not-count"
fi
if ! grep -q 'MODEL[S]_DIR=/path/to/onnx-genai/models' "$self_src_file"; then
  missing_banner_parts="$missing_banner_parts banner-omits-the-enabling-command"
fi
# shellcheck disable=SC2016  # literal search string, not an expansion
if ! grep -q 'i[f] \[ "\$MODEL_CHECKS_SKIPPED" -gt 0 \]' "$self_src_file"; then
  missing_banner_parts="$missing_banner_parts banner-does-not-branch-on-model-skips"
fi

rm -f "$self_src_file"
if [ -z "$missing_banner_parts" ]; then
  pass "the skip banner names the lost model evidence and how to restore it"
else
  fail "the skip banner names the lost model evidence and how to restore it" \
    "missing:$missing_banner_parts; without these a narrower run reports a bare total and reads as a full pass"
fi

# A raw skip-printf bypasses the counter, so the summary would under-report and
# the weaker green becomes invisible again. Comments are stripped before the
# count: this guard is about code, and prose that merely discusses the pattern
# must never be able to turn it red. (It did, twice, while being written.)
TESTS_RUN=$((TESTS_RUN + 1))
raw_skips="$(grep -v '^[[:space:]]*#' "$ROOT/scripts/build_qwen_test.sh" |
  grep -c "printf 'ski[p]" || true)"
if [ "$raw_skips" -eq 1 ]; then
  pass "every skip goes through the counted skip() helper"
else
  fail "every skip goes through the counted skip() helper" \
    "found $raw_skips raw skip-printf sites, expected exactly 1 (inside skip() itself); a raw skip is not counted and hides a narrower run"
fi

TESTS_RUN=$((TESTS_RUN + 1))
skip_body="$(sed -n '/^skip() {/,/^}/p' "$ROOT/scripts/build_qwen_test.sh")"
case "$skip_body" in
  *"TESTS_SKIPPED=\$((TESTS_SKIPPED + 1))"*)
    pass "skip() actually increments the counter, not just prints" ;;
  *)
    fail "skip() actually increments the counter, not just prints" \
      "skip() printed a skip line without counting it, so the summary would report 0 skipped while skips scrolled past" ;;
esac

printf '\n%d tests, %d failed, %d skipped\n' \
  "$TESTS_RUN" "$TESTS_FAILED" "$TESTS_SKIPPED"
if [ "$TESTS_SKIPPED" -gt 0 ]; then
  # A reviewer must not be able to finish this run without seeing what did not
  # execute. "Visible if you look" is not visible: relying on a reader noticing
  # skip lines scrolled past a hundred ok lines is how a weaker green passes for
  # a full one. So this is a block, at the end, after the count they already
  # read, naming the missing evidence and the exact command that restores it.
  printf '\n'
  printf '=======================================================================\n'
  printf '  %d CHECK(S) DID NOT RUN. THIS PASS IS NARROWER THAN A FULL RUN.\n' "$TESTS_SKIPPED"
  printf '=======================================================================\n'
  if [ "$MODEL_CHECKS_SKIPPED" -gt 0 ]; then
    printf '  MODEL-FIDELITY EVIDENCE DID NOT RUN (%d check(s)).\n' "$MODEL_CHECKS_SKIPPED"
    printf '  Not verified: that the generator reproduces the real 24-layer\n'
    printf '  export byte-for-byte, and that cache ports pair by NUMERIC layer\n'
    printf '  index. Lexical ordering puts key_cache.10 before key_cache.2 and\n'
    printf '  silently mis-pairs every buffer past the ninth layer -- the\n'
    printf '  1-layer fixture cannot detect it. These are the strongest checks\n'
    printf '  in this file and they are the ones that just did not happen.\n'
    printf '\n'
    printf '  Cause: models are gitignored, so no clone or worktree has them.\n'
    # Report what was ACTUALLY searched. MODELS_DIR suppresses the candidate
    # list entirely, so printing the candidates under an override would put a
    # false statement inside the block whose entire purpose is accuracy.
    if [ -n "${MODELS_DIR:-}" ]; then
      printf '  Searched: %s only (MODELS_DIR is set, so no fallback was tried)\n' "$MODELS_DIR"
    else
      printf '  Searched: %s\n' "$(models_dir_candidates "$ROOT" | tr '\n' ' ')"
    fi
    printf '  Enable with ONE command, pointing at a checkout that has them:\n'
    printf '    MODELS_DIR=/path/to/onnx-genai/models scripts/build_qwen_test.sh\n'
  fi
  printf '  Read every skip line above before citing this run as green.\n'
  printf '=======================================================================\n'
fi
[ "$TESTS_FAILED" -eq 0 ]
