#!/usr/bin/env bash
# miri_step_test.sh — conformance suite for run_miri's arity guard
#
# `run_miri` is a gate whose failure mode is a green report: if it ever lets a
# step that executed nothing report success, nothing downstream notices,
# because the evidence it was supposed to collect is exactly what is missing.
# That is not reviewable by reading, so every cell below runs the real
# `run_miri` against a fake step command whose output is scripted.
#
# The suite is hermetic and takes well under a second: no cargo, no Miri, no
# compilation. The fakes are `printf`/`exit` one-liners, which is the point --
# what is under test is the wrapper's reading of a step's output, not any step.
#
# One cell is a NEGATIVE CONTROL: it runs the pre-guard implementation, inlined
# verbatim, against the empty-filter output and asserts that it PASSES. Without
# it, the cells asserting the guard fires would still pass if `run_miri` failed
# for some unrelated reason -- a test that cannot distinguish "the guard caught
# it" from "something else broke" is not testing the guard. (#1897 merged a fix
# for exactly this shape: a negative control that survived the removal of the
# thing it controlled for.)
#
# One cell asserts the workflow actually SOURCES this file. A guard that is
# written, tested, and never called is the same green report by another route.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
ROOT="$(pwd)"
WORK="$ROOT/.miri-step-selftest"

rm -rf "$WORK"
mkdir -p "$WORK"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

# Keep the wrapper's own log out of the checkout.
export MIRI_STEP_LOG_DIR="$WORK"

# shellcheck source=scripts/miri_step.sh
. "$ROOT/scripts/miri_step.sh"

PASS=0
FAIL=0

ok() {
    PASS=$((PASS + 1))
    printf '  ok   %s\n' "$1"
}

bad() {
    FAIL=$((FAIL + 1))
    printf '  FAIL %s\n' "$1"
    [ $# -gt 1 ] && printf '       %s\n' "$2"
}

check() {
    # check <description> <0-if-passing> [detail]
    if [ "$2" = "0" ]; then ok "$1"; else bad "$1" "${3:-}"; fi
}

# ─── Fake steps ───────────────────────────────────────────────────────────
# Each emits libtest-shaped output and an exit status. `run_miri` never knows
# it is not cargo.

fake() {
    # fake <exit-status> <line>...
    local rc="$1"
    shift
    printf '%s\n' "$@"
    return "$rc"
}

RESULT_3="test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s"
RESULT_EMPTY="test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 2 filtered out; finished in 0.00s"
RESULT_IGNORED="test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s"
RESULT_FAILED="test result: FAILED. 2 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s"

# ─── Cells ────────────────────────────────────────────────────────────────

echo "run_miri arity guard"

OUT="$(run_miri "normal step" fake 0 "running 3 tests" "$RESULT_3" 2>&1)"
STATUS=$?
check "a step that ran 3 tests succeeds" \
    "$([ "$STATUS" = "0" ] && echo 0 || echo 1)" "got $STATUS: $OUT"
check "...and reports how many it ran" \
    "$(echo "$OUT" | grep -q "MIRI_EXECUTED normal step: 3" && echo 0 || echo 1)" "$OUT"

OUT="$(run_miri "empty filter" fake 0 "running 0 tests" "$RESULT_EMPTY" 2>&1)"
STATUS=$?
check "a filter matching nothing FAILS despite libtest reporting ok/0" \
    "$([ "$STATUS" != "0" ] && echo 0 || echo 1)" "got $STATUS: $OUT"
check "...and says so as a GitHub error annotation" \
    "$(echo "$OUT" | grep -q "^::error::Miri step 'empty filter' executed 0 tests" && echo 0 || echo 1)" "$OUT"
check "...and names the three ways a filter goes stale" \
    "$(echo "$OUT" | grep -q "renamed, moved, or" && echo 0 || echo 1)" "$OUT"

OUT="$(run_miri "all ignored" fake 0 "running 0 tests" "$RESULT_IGNORED" 2>&1)"
STATUS=$?
check "an all-ignored step FAILS: ignored is not executed" \
    "$([ "$STATUS" != "0" ] && echo 0 || echo 1)" "got $STATUS: $OUT"

# A single cargo invocation prints one `test result:` per target, and auxiliary
# targets legitimately report zero. The guard must sum, not require every line.
OUT="$(run_miri "two targets" fake 0 "$RESULT_EMPTY" "$RESULT_3" 2>&1)"
STATUS=$?
check "a 0-test auxiliary target alongside a real one still succeeds" \
    "$([ "$STATUS" = "0" ] && echo 0 || echo 1)" "got $STATUS: $OUT"
check "...counting only the tests that actually ran" \
    "$(echo "$OUT" | grep -q "MIRI_EXECUTED two targets: 3" && echo 0 || echo 1)" "$OUT"

# Propagation: the guard must not swallow or reshape a real failure.
OUT="$(run_miri "genuine failure" fake 101 "$RESULT_FAILED" 2>&1)"
STATUS=$?
check "a step whose command fails propagates its exact exit status" \
    "$([ "$STATUS" = "101" ] && echo 0 || echo 1)" "got $STATUS: $OUT"
check "...and is annotated as a failure, not as an empty filter" \
    "$(echo "$OUT" | grep -q "^::error::Miri step 'genuine failure' failed (exit 101)" && echo 0 || echo 1)" "$OUT"

# `$?` after a pipeline is `tee`'s status, and `tee` succeeds at copying a
# failure. This cell runs with `pipefail` OFF deliberately, and that is the
# whole cell: with `pipefail` ON -- which both this suite and miri.yml set --
# `$?` gives the rightmost nonzero status, so it coincidentally equals
# PIPESTATUS[0] and a `$?` implementation passes. Verified by mutation:
# `rc=$?` survives this cell with pipefail on and fails it with pipefail off.
# So the cell as first written asserted nothing, and `run_miri`'s comment
# claiming it does not depend on `$?` semantics would have been true only by
# the grace of a `set -o` two files away. Read this as: the guard must hold on
# its own terms, not on its caller's shell options.
LONG=$(for i in $(seq 1 200); do echo "chatter line $i"; done)
OUT="$(set +o pipefail; run_miri "loud failure" fake 7 "$LONG" "$RESULT_FAILED" 2>&1)"
STATUS=$?
check "a failing step is detected without relying on the caller's pipefail" \
    "$([ "$STATUS" = "7" ] && echo 0 || echo 1)" "got $STATUS: $OUT"

# The log group and the durable MIRI_TIMING line are consumed by humans and by
# the per-crate timing record; a failing step is when you most need them, and
# the pre-guard version lost both because `set -e` fired before they printed.
OUT="$(run_miri "grouping" fake 0 "$RESULT_3" 2>&1)"
check "a passing step opens and closes its log group" \
    "$(echo "$OUT" | grep -q '^::group::Miri: grouping' && echo "$OUT" | grep -q '^::endgroup::' && echo 0 || echo 1)" "$OUT"
check "...and prints MIRI_TIMING" \
    "$(echo "$OUT" | grep -q '^MIRI_TIMING grouping: ' && echo 0 || echo 1)" "$OUT"

OUT="$(run_miri "grouping on failure" fake 3 "$RESULT_FAILED" 2>&1)"
check "a FAILING step also closes its group and reports its duration" \
    "$(echo "$OUT" | grep -q '^::endgroup::' && echo "$OUT" | grep -q '^MIRI_TIMING grouping on failure: ' && echo 0 || echo 1)" "$OUT"

OUT="$(run_miri "streaming" fake 0 "a distinctive line from the step" "$RESULT_3" 2>&1)"
check "the step's own output is not swallowed" \
    "$(echo "$OUT" | grep -q "a distinctive line from the step" && echo 0 || echo 1)" "$OUT"

# ─── Negative control ─────────────────────────────────────────────────────
# The pre-guard implementation, verbatim. If this FAILS the empty-filter input,
# then the cells above are not measuring the guard and this suite proves
# nothing. It must pass, which is precisely the defect being fixed.

run_miri_before_the_guard() {
    local label="$1"
    shift
    local start end
    start=$(date +%s)
    echo "::group::Miri: ${label}"
    "$@"
    end=$(date +%s)
    echo "MIRI_TIMING ${label}: $((end - start))s"
    echo "::endgroup::"
}

OUT="$(run_miri_before_the_guard "control" fake 0 "running 0 tests" "$RESULT_EMPTY" 2>&1)"
STATUS=$?
check "NEGATIVE CONTROL: the pre-guard wrapper reports success on 0 tests" \
    "$([ "$STATUS" = "0" ] && echo 0 || echo 1)" \
    "the control did not reproduce the defect, so the cells above prove nothing: $OUT"

# ─── Wiring ───────────────────────────────────────────────────────────────
# A guard that is never called is the same green report by another route.

check "the Miri workflow sources this wrapper" \
    "$(grep -q 'scripts/miri_step\.sh' "$ROOT/.github/workflows/miri.yml" && echo 0 || echo 1)" \
    "miri.yml does not source scripts/miri_step.sh -- the guard is inert"

check "...and no longer defines its own run_miri" \
    "$(grep -q '^\s*run_miri() {' "$ROOT/.github/workflows/miri.yml" && echo 1 || echo 0)" \
    "miri.yml still defines run_miri inline, which would shadow the guarded one"

echo ""
EXPECTED=18
TOTAL=$((PASS + FAIL))
if [ "$TOTAL" -ne "$EXPECTED" ]; then
    echo "✗ $TOTAL assertions ran, expected $EXPECTED — a cell was added or lost" >&2
    exit 1
fi

if [ "$FAIL" -ne 0 ]; then
    echo "✗ $FAIL of $TOTAL assertions failed" >&2
    exit 1
fi

echo "✓ $PASS/$TOTAL assertions passed"
