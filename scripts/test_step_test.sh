#!/usr/bin/env bash
# test_step_test.sh — conformance suite for the step wrapper's arity guard
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
WORK="$ROOT/.test-step-selftest"

rm -rf "$WORK"
mkdir -p "$WORK"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

# Keep the wrapper's own log out of the checkout.
export TEST_STEP_LOG_DIR="$WORK"

# shellcheck source=scripts/test_step.sh
. "$ROOT/scripts/test_step.sh"

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
    "$(grep -q 'scripts/test_step\.sh' "$ROOT/.github/workflows/miri.yml" && echo 0 || echo 1)" \
    "miri.yml does not source scripts/test_step.sh -- the guard is inert"

check "...and no longer defines its own run_miri" \
    "$(grep -q '^\s*run_miri() {' "$ROOT/.github/workflows/miri.yml" && echo 1 || echo 0)" \
    "miri.yml still defines run_miri inline, which would shadow the guarded one"

# ─── run_test_step: the general entry point ───────────────────────────────
# Same guard, different lane and different log tokens. Re-run rather than
# assumed-by-inspection: `run_miri` and `run_test_step` are two call sites of
# one implementation, and "it delegates, so it must behave the same" is exactly
# the inference this file exists to refuse.

OUT="$(run_test_step "normal step" fake 0 "running 3 tests" "$RESULT_3" 2>&1)"
STATUS=$?
check "run_test_step passes a step that ran tests" \
    "$([ "$STATUS" = "0" ] && echo 0 || echo 1)" "$OUT"
check "...and reports the count under its own token" \
    "$(echo "$OUT" | grep -q "^TEST_STEP_EXECUTED normal step: 3" && echo 0 || echo 1)" "$OUT"
check "...and does not emit MIRI_ tokens from a non-Miri lane" \
    "$(echo "$OUT" | grep -q "MIRI_" && echo 1 || echo 0)" "$OUT"

OUT="$(run_test_step "empty filter" fake 0 "running 0 tests" "$RESULT_EMPTY" 2>&1)"
STATUS=$?
check "run_test_step FAILS a step whose filter matched nothing" \
    "$([ "$STATUS" = "1" ] && echo 0 || echo 1)" \
    "a name filter matching nothing exits 0 in libtest; the wrapper must not: $OUT"
check "...and says so as a workflow error" \
    "$(echo "$OUT" | grep -q '^::error::Test step .empty filter. executed 0 tests' && echo 0 || echo 1)" "$OUT"

OUT="$(run_test_step "all ignored" fake 0 "running 0 tests" "$RESULT_IGNORED" 2>&1)"
STATUS=$?
check "run_test_step FAILS an all-ignored step" \
    "$([ "$STATUS" = "1" ] && echo 0 || echo 1)" \
    "ignored must not count as executed: $OUT"

OUT="$(run_test_step "genuine failure" fake 101 "$RESULT_FAILED" 2>&1)"
STATUS=$?
check "run_test_step propagates a real failure's exit status" \
    "$([ "$STATUS" = "101" ] && echo 0 || echo 1)" "$OUT"

# ─── Wiring: the three MLAS steps (#2055) ─────────────────────────────────
# These are the only steps in the pipeline that build onnx-runtime-ep-cpu with
# --features mlas, they select by name/module path, and libtest exits 0 on a
# filter that matches nothing. Asserting the guard exists is not enough; assert
# each filter is actually routed through it.

CI="$ROOT/.github/workflows/ci.yml"

check "the Rust quality lane sources this wrapper" \
    "$(grep -q 'scripts/test_step\.sh' "$CI" && echo 0 || echo 1)" \
    "ci.yml does not source scripts/test_step.sh -- the MLAS steps are unguarded"

for filter in \
    optimization_registry_excludes_nchwc_without_cnn_ops \
    kernels::moe:: \
    kernels::qlinear_matmul::
do
    # The filter must appear inside a `run_test_step` invocation, not merely
    # somewhere in the file: a step that still shells out to a bare `cargo
    # test` would satisfy a naive grep for the filter alone.
    check "the '$filter' MLAS step runs through run_test_step" \
        "$(awk -v f="$filter" '
            /run_test_step/ { armed = 1 }
            /^      - name:/ { armed = 0 }
            armed && index($0, f) { found = 1 }
            END { exit(found ? 0 : 1) }
        ' "$CI" && echo 0 || echo 1)" \
        "ci.yml runs '$filter' outside the guard, so an empty filter there is still green"
done

# The three cells above pin the filters that exist today. This one is the
# forward-looking half: ANY `cargo test` step in ci.yml that carries a bare
# name filter must be routed through the guard. Without it, #2055 is fixed for
# exactly three steps and reopens the moment a fourth is added -- which is how
# the hole got here in the first place.
#
# A "bare name filter" is a positional argument: a token that is not an option
# and not the value of one. That is the only selector libtest does not police;
# `-p` and `--test` both exit 101 when they match nothing, so an option-only
# step needs no guard.

unguarded_filtered_steps() {
    awk '
        function flush(   i, n, tok, parts, skip) {
            if (block == "" || block !~ /cargo test/) return
            if (block ~ /run_test_step/) return
            # `$(...)` is opaque to a static scan. Blank it out here so it is
            # not misread as a positional -- and note that doing so would open
            # a quiet hole, which the cell below closes by EXECUTING each one
            # and proving it expands to options only.
            gsub(/\$\([^)]*\)/, " ", block)
            n = split(block, parts, /[ \t]+/)
            skip = 0
            for (i = 1; i <= n; i++) {
                tok = parts[i]
                if (tok == "" || tok == "\\" || tok == "|" || tok == ">-") continue
                if (skip) { skip = 0; continue }
                if (tok ~ /^-/) {
                    # Options that consume the following token as their value.
                    if (tok ~ /^(-p|--package|--test|--bench|--example|--bin|--features|--target|--target-dir|--manifest-path|--profile|-j|--jobs|--color|--message-format|--exclude)$/)
                        skip = 1
                    continue
                }
                # Words that are part of the invocation itself, not selectors.
                if (tok ~ /^(cargo|test|env|bash|sudo|run:|[A-Z_]+=.*)$/) continue
                print name
                return
            }
        }
        /^      - name:/ { flush(); block = ""; name = $0; sub(/^ *- name: */, "", name); inrun = 0; next }
        /^        run:/ { inrun = 1; block = block " " $0; sub(/^ *run: */, "", block); next }
        {
            if (!inrun) next
            line = $0
            sub(/^[ \t]+/, "", line)
            if (line ~ /^#/) next
            block = block " " line
        }
        END { flush() }
    ' "$1"
}

STRAY="$(unguarded_filtered_steps "$CI")"
check "no cargo test step in ci.yml carries an unguarded name filter" \
    "$([ -z "$STRAY" ] && echo 0 || echo 1)" \
    "these steps select tests by name outside run_test_step, so an empty filter is green: $STRAY"

# And the detector itself, by mutation: a synthetic unguarded filtered step
# must be found. A scanner that reports "nothing unguarded" because it parses
# nothing is the same vacuous pass this whole file exists to refuse.
cat > "$WORK/mutant.yml" <<'YAML'
      - name: A guarded step
        run: |
          . ./scripts/test_step.sh
          run_test_step "ok" cargo test -p some-crate some::filter::
      - name: An option-only step
        run: cargo test --locked -p some-crate --test some_target
      - name: A build, not a test
        run: cargo build --locked -p some-crate --features mlas
      - name: An unguarded filtered step
        run: >-
          cargo test --locked
          -p some-crate
          --features mlas
          kernels::something::
YAML
MUT="$(unguarded_filtered_steps "$WORK/mutant.yml")"
check "MUTATION: the detector finds an unguarded filtered step" \
    "$([ "$MUT" = "An unguarded filtered step" ] && echo 0 || echo 1)" \
    "expected exactly 'An unguarded filtered step', got: [$MUT]"

# The scanner blanks out `$(...)`, so a generator that emitted a bare word
# would slip past it. Close that by running each generator ci.yml actually
# uses and proving its expansion is options only. Executed, not assumed: the
# whole point of this file is that "it looks like it only emits -p" is the
# kind of claim that is wrong exactly when it matters.
#
# Only `cargo-args` substitutions are generators: their expansion lands *in* a
# cargo command line, which is what makes a bare word there a silent test
# filter. The same script also answers `verify`, which is a checker -- prose on
# stdout, the exit status is the signal, never substituted into cargo. Running
# one of those as if it were a generator reports a defect that is not there
# (it did: a `verify` call carrying a loop variable is unexpandable here, so
# this cell read "did not run" and failed a green tree). So classify first and
# execute only the generators -- and refuse any subcommand this suite has not
# been taught to classify, so a new generator cannot arrive unexamined.
# shellcheck disable=SC2016  # the $( is literal text being searched for
python_subs() { grep -o '\$(python[^)]*)' "$1" | sort -u; }

# Substitutions in the wrong position for their subcommand. Which one a
# substitution is cannot be read off its name -- it is decided by where the
# expansion lands. A `verify` substituted into a cargo command line would
# inject prose as test filters, and a generator outside one is not covered by
# the execution cell below. So classify by the step block, and print anything
# that does not match. Printed, not tolerated: silence here is how the next
# generator would skip that cell.
misclassified_subs() {
    awk -v list="${2:-}" '
        function flush(   i, n, parts, body) {
            if (block == "") return
            n = split(block, parts, /\$\(python/)
            for (i = 2; i <= n; i++) {
                body = parts[i]
                sub(/\).*/, "", body)
                if (list) { print "$(python" body ")"; continue }
                if (block ~ /cargo /) {
                    if (body !~ /cargo-args /)
                        print name ": in a cargo step but not a generator: $(python" body ")"
                } else if (body !~ /verify/) {
                    print name ": outside a cargo step and not a known checker: $(python" body ")"
                }
            }
        }
        /^      - name:/ { flush(); block = ""; name = $0; sub(/^ *- name: */, "", name); inrun = 0; next }
        /^        run:/ { inrun = 1; block = block " " $0; sub(/^ *run: */, "", block); next }
        {
            if (!inrun) next
            line = $0
            sub(/^[ \t]+/, "", line)
            if (line ~ /^#/) next
            block = block " " line
        }
        END { flush() }
    ' "$1"
}

SUBS="$(python_subs "$CI")"
ARGS_SUBS="$(printf '%s\n' "$SUBS" | grep -F 'cargo-args ' || true)"
check "ci.yml's package-list generators are actually enumerated" \
    "$([ -n "$ARGS_SUBS" ] && echo 0 || echo 1)" \
    "found no \$(python ... cargo-args ...) substitutions in ci.yml -- the cell below would be vacuous"

BAD_SUB=""
while IFS= read -r sub; do
    [ -n "$sub" ] || continue
    cmd="${sub#\$(}"
    cmd="${cmd%)}"
    # CI resolves `python`; this box only has `python3`.
    cmd="${cmd/#python /python3 }"
    # shellcheck disable=SC2086
    expansion="$(cd "$ROOT" && eval $cmd 2>&1)" || { BAD_SUB="$BAD_SUB [did not run: $sub]"; continue; }
    prev=""
    for tok in $expansion; do
        case "$tok" in
            -*) prev="$tok"; continue ;;
        esac
        case "$prev" in
            -p|--package|--exclude) prev=""; continue ;;
        esac
        BAD_SUB="$BAD_SUB [$sub emits bare token: $tok]"
        break
    done
done <<< "$ARGS_SUBS"

check "...and every one expands to options only, never a name filter" \
    "$([ -z "$BAD_SUB" ] && echo 0 || echo 1)" \
    "a generator can inject a bare test filter past the scanner:$BAD_SUB"

UNCLASSIFIED="$(misclassified_subs "$CI")"
check "...and every other \$(python ...) substitution sits where its subcommand belongs" \
    "$([ -z "$UNCLASSIFIED" ] && echo 0 || echo 1)" \
    "a substitution is in the wrong position for its subcommand: $UNCLASSIFIED"

# The cell above passes by printing nothing -- which is also what a block
# parser that reads nothing prints. Control for that: the classifier's own
# enumeration must match, substitution for substitution, what a flat grep
# finds. A pass earned by parsing zero steps is the failure mode this file
# exists to refuse, and it is invisible from the verdict alone.
CLASSIFIER_SAW="$(misclassified_subs "$CI" list | sort -u)"
check "...and the classifier actually parsed them (not a pass by reading nothing)" \
    "$([ "$CLASSIFIER_SAW" = "$SUBS" ] && echo 0 || echo 1)" \
    "classifier enumerated [$CLASSIFIER_SAW] but ci.yml contains [$SUBS]"

# And that classifier by mutation, both directions. Without this cell,
# narrowing the execution above to `cargo-args` would be indistinguishable
# from deleting it: a generator under a new subcommand, or a checker spliced
# into a cargo command line, would simply go unchecked.
cat > "$WORK/mutant_sub.yml" <<'YAML'
      - name: A generator in a cargo step
        run: cargo test --locked $(python .github/scripts/workspace_test_packages.py cargo-args lint)
      - name: A checker outside a cargo step
        run: |
          out=$(python .github/scripts/workspace_test_packages.py verify $sim 2>&1) && rc=0 || rc=$?
      - name: A subcommand nobody classified
        run: echo $(python .github/scripts/workspace_test_packages.py frobnicate --lane lint)
      - name: A checker spliced into a cargo command
        run: cargo test --locked $(python .github/scripts/workspace_test_packages.py verify)
YAML
MUT_SUB="$(misclassified_subs "$WORK/mutant_sub.yml")"
# shellcheck disable=SC2016  # the $( is literal text being matched, not expanded
MUT_WANT='A subcommand nobody classified: outside a cargo step and not a known checker: $(python .github/scripts/workspace_test_packages.py frobnicate --lane lint)
A checker spliced into a cargo command: in a cargo step but not a generator: $(python .github/scripts/workspace_test_packages.py verify)'
check "MUTATION: both a new subcommand and a misplaced checker are refused" \
    "$([ "$MUT_SUB" = "$MUT_WANT" ] && echo 0 || echo 1)" \
    "expected exactly the two offenders, got: [$MUT_SUB]"

echo ""
EXPECTED=36
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
