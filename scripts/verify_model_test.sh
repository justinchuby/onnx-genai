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

# --- port occupancy -------------------------------------------------------
#
# Regression test for the worst defect this script has had: it probed /health,
# got an instant 200 from a server another agent had left on the port, and ran
# every check against a process that never loaded the model. It reported FAIL
# that time -- but against a HEALTHY stranger it reports PASS and tells you to
# promote a model it never opened.
fake_bin="$tmp/fake-server"
printf '#!/bin/sh\nsleep 60\n' >"$fake_bin"
chmod +x "$fake_bin"

# A stand-in for "another agent's server": anything that accepts a connection.
# A raw socket rather than http.server keeps this dependency-light and, more
# importantly, tests the guard's ACTUAL contract -- "something is listening" --
# instead of the narrower "something serves HTTP". It self-terminates so the
# suite never leaks a listener.
cat >"$tmp/listener.py" <<'PYEOF'
import socket, sys, time

s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", int(sys.argv[1])))
s.listen(16)

# The listener MUST actually accept and close, not just bind. A socket that
# binds but never accepts fills its backlog after a handful of probes, after
# which the OS REFUSES new connections -- and a refused connection is exactly
# what "nothing is listening" looks like. An earlier version of this test slept
# instead of accepting, so the first probe saw the port as busy and every later
# one saw it as FREE, which made the guard look broken when it was correct.
s.settimeout(0.5)
deadline = time.time() + 45
while time.time() < deadline:
    try:
        conn, _ = s.accept()
        conn.close()
    except (socket.timeout, OSError):
        pass
PYEOF
python3 "$tmp/listener.py" 8271 >/dev/null 2>&1 &

# Probe with the same semantics the guard uses: curl exit 7 means "nothing is
# listening". Asserting exit 0 here would be wrong -- a bare socket never
# speaks HTTP -- and that mismatch is what made the first version of this test
# report a spurious failure.
squatter_ready=""
for _ in 1 2 3 4 5 6 7 8 9 10; do
  probe_rc=0
  curl -s --connect-timeout 1 -o /dev/null "http://127.0.0.1:8271/" 2>/dev/null || probe_rc=$?
  if [ "$probe_rc" -ne 7 ]; then
    squatter_ready=1
    break
  fi
  sleep 0.3
done

TESTS_RUN=$((TESTS_RUN + 1))
if [ -z "$squatter_ready" ]; then
  fail "could not start a listener on 8271 to test port-occupancy detection"
else
  pass "test listener is up on 8271"

  output="$(SERVER_BIN="$fake_bin" "$VERIFY" "$tmp/model" --port 8271 2>&1 || true)"

  assert_contains "refuses to run when the port is already in use" \
    "$output" "port 8271 is already in use"
  # The danger is silent misattribution, so the message must name the model
  # that would NOT have been verified.
  assert_contains "names the model that would not have been verified" \
    "$output" "$tmp/model"
  assert_contains "explains that results would describe the other server" \
    "$output" "rather than"
  assert_contains "offers a concrete free port rather than just complaining" \
    "$output" "--port 827"

  TESTS_RUN=$((TESTS_RUN + 1))
  if SERVER_BIN="$fake_bin" "$VERIFY" "$tmp/model" --port 8271 >/dev/null 2>&1; then
    fail "exits non-zero when the port is occupied"
  else
    pass "exits non-zero when the port is occupied"
  fi
fi

# The post-readiness re-check exists because the liveness probe runs BEFORE the
# curl and loses the race on the first iteration.
TESTS_RUN=$((TESTS_RUN + 1))
if grep -q "address already in use" "$VERIFY" &&
  grep -q "came from a different process" "$VERIFY"; then
  pass "re-asserts our own process owns the port after readiness"
else
  fail "re-asserts our own process owns the port after readiness"
fi

# --- ANSI-styled log output -------------------------------------------------
# tracing's fmt layer emits ANSI styling even when stdout is redirected to a
# FILE, and it wraps structured field NAMES. The bytes are:
#   ...enabled <ESC>[3mmax_batch<ESC>[0m<ESC>[2m=<ESC>[0m4
# so "max_batch=4" is never contiguous. A human reading the log sees exactly
# that string; grep finds nothing. This cost a real false FAIL, and the
# symptom was "batching engaged at the wrong width" on a correct server.
esc="$(printf '\033')"
styled_line="${esc}[2m2026-07-30T07:57:28Z${esc}[0m ${esc}[32m INFO${esc}[0m ${esc}[2monnx_genai_server::driver${esc}[0m${esc}[2m:${esc}[0m continuous batch driver enabled ${esc}[3mmax_batch${esc}[0m${esc}[2m=${esc}[0m4"
printf '%s\n' "$styled_line" >"$tmp/styled.log"

# The bug must be real, or the fix guards nothing.
TESTS_RUN=$((TESTS_RUN + 1))
if grep -q "continuous batch driver enabled max_batch=4" "$tmp/styled.log"; then
  fail "raw tracing output does NOT contain a contiguous max_batch=4 (fixture is wrong)"
else
  pass "raw tracing output does NOT contain a contiguous max_batch=4"
fi

# And the strip must recover it.
sed "s/${esc}\[[0-9;]*m//g" "$tmp/styled.log" >"$tmp/styled.plain"
TESTS_RUN=$((TESTS_RUN + 1))
if grep -q "continuous batch driver enabled max_batch=4" "$tmp/styled.plain"; then
  pass "stripping ANSI makes max_batch=4 greppable"
else
  fail "stripping ANSI makes max_batch=4 greppable"
fi

# The message-only assertions survive styling because tracing appends fields
# AFTER the message. Recording that here so nobody "simplifies" the strip away
# on the grounds that the batching check still passes without it.
TESTS_RUN=$((TESTS_RUN + 1))
if grep -q "continuous batch driver enabled" "$tmp/styled.log"; then
  pass "message-only patterns match even unstripped (why this hid so long)"
else
  fail "message-only patterns match even unstripped (why this hid so long)"
fi

# Structural guard: no assertion may read the RAW log, or it silently starts
# matching against escape sequences again. The stripper itself is exempt --
# it is the one line allowed to read LOG_FILE, because it writes PLAIN_LOG.
TESTS_RUN=$((TESTS_RUN + 1))
# shellcheck disable=SC2016  # matching the literal text "$LOG_FILE" in the source
raw_asserts="$(grep -nE '(grep|sed)[^|]*"\$LOG_FILE"' "$VERIFY" | grep -v 'PLAIN_LOG' || true)"
if [ -z "$raw_asserts" ]; then
  pass "no assertion greps the raw, ANSI-styled log"
else
  fail "no assertion greps the raw, ANSI-styled log"
  printf '     offending line(s): %s\n' "$raw_asserts" >&2
fi

# The stripped copy must be refreshed before the checks read it, or PLAIN_LOG
# is an empty file and every grep below it silently reports "not found".
TESTS_RUN=$((TESTS_RUN + 1))
if grep -q '^refresh_plain_log$' "$VERIFY"; then
  pass "refresh_plain_log is actually invoked, not just defined"
else
  fail "refresh_plain_log is actually invoked, not just defined"
fi

# --max-batch must reach the server. Until it did, the value was printed in the
# header and passed nowhere -- a decorative number that read as a claim.
TESTS_RUN=$((TESTS_RUN + 1))
# shellcheck disable=SC2016  # matching the literal text "$MAX_BATCH" in the source
if grep -q -- '--max-batch "\$MAX_BATCH"' "$VERIFY"; then
  pass "--max-batch is forwarded to the server, not just printed"
else
  fail "--max-batch is forwarded to the server, not just printed"
fi

printf '\n%d tests, %d failed\n' "$TESTS_RUN" "$TESTS_FAILED"
[ "$TESTS_FAILED" -eq 0 ]
