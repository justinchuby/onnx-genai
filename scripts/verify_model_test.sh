#!/usr/bin/env bash
# Tests for scripts/verify_model.sh.
#
# These exercise only the argument handling and preflight guards, which is
# everything that runs before a server is started. Nothing here boots a
# server, loads a model, or invokes cargo, so the suite is safe to run at any
# time and takes well under a second.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERIFY="$ROOT/scripts/verify_model.sh"

TESTS_RUN=0
TESTS_FAILED=0

pass() { printf 'ok   %s\n' "$1"; }

fail() {
  TESTS_FAILED=$((TESTS_FAILED + 1))
  printf 'FAIL: %s\n' "$1" >&2
  if [ -n "${2:-}" ]; then
    printf '      %s\n' "$2" >&2
  fi
}

# assert_contains <name> <haystack> <needle>
assert_contains() {
  TESTS_RUN=$((TESTS_RUN + 1))
  case "$2" in
    *"$3"*) pass "$1" ;;
    *) fail "$1" "expected output to contain: $3" ;;
  esac
}

# assert_fails <name> <command...>  -- asserts a non-zero exit
assert_fails() {
  local name="$1"
  shift
  TESTS_RUN=$((TESTS_RUN + 1))
  if "$@" >/dev/null 2>&1; then
    fail "$name" "expected a non-zero exit"
  else
    pass "$name"
  fi
}

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/model"
: >"$tmp/model/model.onnx"
: >"$tmp/a_config_file.yaml"

# A missing binary must never be papered over by shelling out to cargo: the
# shared target directory is an exclusive lock and a surprise build here would
# stall whatever else is running. Assert this behaviourally with a shim on
# PATH rather than by grepping, since the script legitimately prints a cargo
# command as guidance.
mkdir -p "$tmp/bin"
cat >"$tmp/bin/cargo" <<EOF
#!/bin/sh
touch "$tmp/cargo_was_invoked"
exit 0
EOF
chmod +x "$tmp/bin/cargo"

TESTS_RUN=$((TESTS_RUN + 1))
rm -f "$tmp/cargo_was_invoked"
PATH="$tmp/bin:$PATH" SERVER_BIN=/nonexistent/onnx-genai-server \
  "$VERIFY" "$tmp/model" >/dev/null 2>&1 || true
PATH="$tmp/bin:$PATH" "$VERIFY" /nonexistent-model-dir >/dev/null 2>&1 || true
PATH="$tmp/bin:$PATH" "$VERIFY" --help >/dev/null 2>&1 || true
if [ -e "$tmp/cargo_was_invoked" ]; then
  fail "verify_model.sh never invokes cargo" "a cargo shim on PATH was called"
else
  pass "verify_model.sh never invokes cargo"
fi

output="$("$VERIFY" --help 2>&1)"
assert_contains "--help documents usage" "$output" "Usage: verify_model.sh"
assert_contains "--help documents CARGO_TARGET_DIR" "$output" "CARGO_TARGET_DIR"

assert_fails "no arguments is an error" "$VERIFY"

output="$("$VERIFY" /nonexistent-model-dir 2>&1 || true)"
assert_contains "a missing model directory is named explicitly" \
  "$output" "no such model directory"

# The CLI silently coerces a file to its parent directory but the server does
# not, so this mistake otherwise surfaces as a confusing load failure.
output="$("$VERIFY" "$tmp/a_config_file.yaml" 2>&1 || true)"
assert_contains "passing a file suggests the parent directory" \
  "$output" "takes a DIRECTORY"
assert_contains "passing a file names the directory to use" "$output" "$tmp"

assert_fails "an unknown option is an error" "$VERIFY" "$tmp/model" --bogus

# The whole point of the preflight: say what is missing and how to get it,
# rather than building it as a side effect.
output="$(SERVER_BIN=/nonexistent/onnx-genai-server "$VERIFY" "$tmp/model" 2>&1 || true)"
assert_contains "a missing server binary is reported clearly" \
  "$output" "no server binary at"
assert_contains "a missing server binary gives the build command" \
  "$output" "cargo build --release -p onnx-genai-server"
assert_contains "a missing server binary explains why it does not build for you" \
  "$output" "exclusive lock"

# Both driver outcomes log at INFO, so the script must test for the failure
# string too rather than treating "no error" as success.
TESTS_RUN=$((TESTS_RUN + 1))
if grep -q "continuous batch driver enabled" "$VERIFY" &&
  grep -q "continuous batch driver disabled" "$VERIFY"; then
  pass "checks for both the enabled and disabled driver log lines"
else
  fail "checks for both the enabled and disabled driver log lines"
fi

# A model that never stops generating returns finish_reason=length.
TESTS_RUN=$((TESTS_RUN + 1))
if grep -q "finish_reason" "$VERIFY" && grep -q "length)" "$VERIFY"; then
  pass "treats finish_reason=length as a failure"
else
  fail "treats finish_reason=length as a failure"
fi

printf '\n%d tests, %d failed\n' "$TESTS_RUN" "$TESTS_FAILED"
[ "$TESTS_FAILED" -eq 0 ]
