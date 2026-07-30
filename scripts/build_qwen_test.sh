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

TESTS_RUN=0
TESTS_FAILED=0

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
  printf 'skip  fresh-clone cases (every interpreter here already has mobius)\n'
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

printf '\n%d tests, %d failed\n' "$TESTS_RUN" "$TESTS_FAILED"
[ "$TESTS_FAILED" -eq 0 ]
