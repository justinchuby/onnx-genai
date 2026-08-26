#!/usr/bin/env bash
# Conformance suite for scripts/merge_when_green.sh.
#
# Hermetic: `gh` is a shim on PATH backed by fixture files, so every branch is
# reachable without a network, without a repository, and without merging
# anything. The shim records its own argv, which is how the "never --admin"
# cells are asserted behaviourally rather than by reading the source.
#
# The cell this suite exists for is `a required check that never appeared is
# not a passing one`. Every other refusal here is one a naive poller would
# also get right; that one is the defect (#1966), and it is invisible to
# `gh pr checks | grep -v fail` because there is no row to grep.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

MW=scripts/merge_when_green.sh
WORK=${CARGO_TARGET_TMPDIR:-target}/mwg-test.$$
REPO=acme/widget

pass=0
fail=0

cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

chk() {
    if [ "$2" = "$3" ]; then
        echo "  PASS  $1"
        pass=$((pass + 1))
    else
        echo "  FAIL  $1"
        echo "          got:  $2"
        echo "          want: $3"
        fail=$((fail + 1))
    fi
}

# --- the shim ---------------------------------------------------------------
#
# Dispatches on the shape of the call, not on a full argument parse, because a
# faithful `gh` is not the point: the point is that the script under test sees
# exactly the bytes a real `gh -q` would have printed.
make_shim() {
    mkdir -p "$WORK/bin"
    cat >"$WORK/bin/gh" <<'SHIM'
#!/usr/bin/env bash
FIX=$MWG_FIX
printf '%s\n' "$*" >>"$FIX/argv.log"

args="$*"
case "$args" in
    *"pr merge"*)
        printf '%s\n' "$*" >>"$FIX/merge.log"
        exit "$(cat "$FIX/merge_rc" 2>/dev/null || echo 0)"
        ;;
    *"pr view"*)
        n=$(cat "$FIX/poll" 2>/dev/null || echo 1)
        echo $((n + 1)) >"$FIX/poll"
        if [ -f "$FIX/pr_view.fail" ]; then exit 1; fi
        # Fail a WINDOW of reads rather than all of them, so a cell can let the
        # opening snapshot succeed and break the connection afterwards -- which
        # is the only way to reach the in-loop retry counter.
        from=$(cat "$FIX/fail_from" 2>/dev/null || echo 0)
        to=$(cat "$FIX/fail_to" 2>/dev/null || echo 0)
        if [ "$from" -gt 0 ] && [ "$n" -ge "$from" ] && { [ "$to" = 0 ] || [ "$n" -le "$to" ]; }; then
            exit 1
        fi
        if [ -f "$FIX/pr.$n.json" ]; then cat "$FIX/pr.$n.json"; else cat "$FIX/pr.json"; fi
        exit 0
        ;;
    *rulesets/*)
        case "$args" in
            *bypass_actors*) cat "$FIX/bypass" 2>/dev/null; exit 0 ;;
            *) cat "$FIX/contexts" 2>/dev/null; exit 0 ;;
        esac
        ;;
    *rulesets*)
        if [ -f "$FIX/ruleset_ids.fail" ]; then exit 1; fi
        cat "$FIX/ruleset_ids" 2>/dev/null
        exit 0
        ;;
esac
exit 0
SHIM
    chmod +x "$WORK/bin/gh"
}

# A fresh fixture set, green by default; each cell perturbs one thing.
fixture() {
    FIX="$WORK/fix.$1"
    rm -rf "$FIX"
    mkdir -p "$FIX"
    printf '20017687\n' >"$FIX/ruleset_ids"
    printf 'Fast (Linux x86_64)\nRust quality\n' >"$FIX/contexts"
    printf '1\n' >"$FIX/bypass"
    printf '0\n' >"$FIX/merge_rc"
    : >"$FIX/argv.log"
    rollup "$FIX/pr.json" OPEN deadbeef \
        '{"name":"Fast (Linux x86_64)","status":"COMPLETED","conclusion":"SUCCESS"}' \
        '{"name":"Rust quality","status":"COMPLETED","conclusion":"SUCCESS"}'
}

rollup() {
    local out=$1 state=$2 head=$3
    shift 3
    local checks=""
    for c in "$@"; do checks="${checks:+$checks,}$c"; done
    cat >"$out" <<EOF
{"state":"$state","headRefOid":"$head","isDraft":false,
 "mergeStateStatus":"BLOCKED","statusCheckRollup":[$checks]}
EOF
}

# `wc -l <missing` prints a shell error that looks like a suite fault; the
# absence of the file IS the assertion here, so name it.
merges() { [ -f "$FIX/merge.log" ] && wc -l <"$FIX/merge.log" || echo 0; }

run_mw() {
    MWG_FIX="$FIX" PATH="$WORK/bin:$PATH" \
        bash "$MW" 7 --repo "$REPO" --timeout "${T:-0}" --poll 1 "$@" \
        >"$FIX/out" 2>&1
    echo $?
}

make_shim

echo "== the green path =="
fixture green
chk "all required checks green merges" "$(run_mw)" "0"
chk "and it merged exactly once" "$(merges)" "1"
# Behavioural, not a grep of the source: the shim saw every argument the
# script ever passed, and --admin is not among them.
chk "without --admin, ever" "$(grep -c -- '--admin' "$FIX/argv.log")" "0"
chk "and with the method it was told to use" \
    "$(grep -c -- '--squash' "$FIX/merge.log")" "1"

fixture method
chk "--method rebase is honoured" "$(run_mw --method rebase)" "0"
chk "and rebase is what gh was asked for" \
    "$(grep -c -- '--rebase' "$FIX/merge.log")" "1"

echo
echo "== the refusals =="

# THE cell. A required context with no row in the rollup is the case a poller
# that greps for failure reports as success: there is nothing to grep.
fixture absent
rollup "$FIX/pr.json" OPEN deadbeef \
    '{"name":"Fast (Linux x86_64)","status":"COMPLETED","conclusion":"SUCCESS"}'
chk "a required check that never appeared is not a passing one" "$(run_mw)" "3"
chk "and nothing was merged" "$(merges)" "0"
chk "and it names the context it never saw" \
    "$(grep -c 'Rust quality *ABSENT' "$FIX/out")" "1"

# The same shape, dressed up: plenty of green, none of it the required one.
# Guards against a future rewrite that counts successes instead of matching
# names -- which passes the cell above by accident.
fixture absent_but_busy
rollup "$FIX/pr.json" OPEN deadbeef \
    '{"name":"Fast (Linux x86_64)","status":"COMPLETED","conclusion":"SUCCESS"}' \
    '{"name":"Rust coverage (Linux x86_64)","status":"COMPLETED","conclusion":"SUCCESS"}' \
    '{"name":"EP conformance (Linux x86_64)","status":"COMPLETED","conclusion":"SUCCESS"}' \
    '{"name":"codecov/patch","status":"COMPLETED","conclusion":"SUCCESS"}'
chk "four unrelated green checks do not stand in for the missing one" "$(run_mw)" "3"

fixture failed
rollup "$FIX/pr.json" OPEN deadbeef \
    '{"name":"Fast (Linux x86_64)","status":"COMPLETED","conclusion":"FAILURE"}' \
    '{"name":"Rust quality","status":"COMPLETED","conclusion":"SUCCESS"}'
chk "a failed required check stops immediately" "$(run_mw)" "2"
chk "and nothing was merged" "$(merges)" "0"

fixture pending
rollup "$FIX/pr.json" OPEN deadbeef \
    '{"name":"Fast (Linux x86_64)","status":"IN_PROGRESS","conclusion":null}' \
    '{"name":"Rust quality","status":"COMPLETED","conclusion":"SUCCESS"}'
chk "pending is not pass" "$(run_mw)" "3"
chk "and nothing was merged" "$(merges)" "0"

# A rebase under an armed wait is exactly how #1957 nearly merged unchecked.
fixture moved
cp "$FIX/pr.json" "$FIX/pr.1.json"
rollup "$FIX/pr.2.json" OPEN cafebabe \
    '{"name":"Fast (Linux x86_64)","status":"COMPLETED","conclusion":"SUCCESS"}' \
    '{"name":"Rust quality","status":"COMPLETED","conclusion":"SUCCESS"}'
chk "a head that moves invalidates the wait, green or not" "$(T=30 run_mw)" "4"
chk "and nothing was merged" "$(merges)" "0"

echo
echo "== fail closed on an undetermined gate =="
# An empty required set reads identically to a satisfied one. It must not be
# treated as satisfied -- this is the hardcoded-list drift the script's header
# refuses to risk.
fixture noctx
: >"$FIX/contexts"
chk "no required contexts is a refusal, not a green light" "$(run_mw)" "5"
chk "and nothing was merged" "$(merges)" "0"

fixture apierr
touch "$FIX/ruleset_ids.fail"
chk "an unreadable ruleset API is a refusal too" "$(run_mw)" "5"

echo
echo "== a rollup that cannot be read is not an empty one =="
# Both of these merged before review. The decision was `[ -z "$notgreen" ]`,
# which is satisfied by a verdict with NO ROWS -- and a parser fault, a blank
# required set and a silently dropped row all look exactly like that. A merge
# gate that merges on an unhandled exception is #1817 relocated into the gate.
fixture malformed
cat >"$FIX/pr.json" <<'EOF'
{"state":"OPEN","headRefOid":"deadbeef","isDraft":false,
 "mergeStateStatus":"BLOCKED","statusCheckRollup":[null]}
EOF
chk "a rollup entry that is not an object refuses, it does not merge" "$(run_mw)" "5"
chk "and nothing was merged" "$(merges)" "0"

fixture blankctx
printf '   \n\t\n' >"$FIX/contexts"
chk "a required set of nothing but whitespace is not a required set" "$(run_mw)" "5"
chk "and nothing was merged" "$(merges)" "0"

echo
echo "== a name that appears twice is folded pessimistically =="
# A re-run creates a fresh check run with the same name. Keyed naively, the
# LAST entry in the array wins -- so whether we merge past a failing required
# check is decided by an array order GitHub does not document as chronological.
fixture dup_fail_then_pass
rollup "$FIX/pr.json" OPEN deadbeef \
    '{"name":"Fast (Linux x86_64)","status":"COMPLETED","conclusion":"SUCCESS"}' \
    '{"name":"Rust quality","status":"COMPLETED","conclusion":"FAILURE"}' \
    '{"name":"Rust quality","status":"COMPLETED","conclusion":"SUCCESS"}'
chk "a later SUCCESS does not overwrite an earlier FAILURE of the same name" "$(run_mw)" "2"
chk "and nothing was merged" "$(merges)" "0"

fixture dup_pass_then_fail
rollup "$FIX/pr.json" OPEN deadbeef \
    '{"name":"Fast (Linux x86_64)","status":"COMPLETED","conclusion":"SUCCESS"}' \
    '{"name":"Rust quality","status":"COMPLETED","conclusion":"SUCCESS"}' \
    '{"name":"Rust quality","status":"COMPLETED","conclusion":"FAILURE"}'
chk "and the verdict does not depend on which order they arrived in" "$(run_mw)" "2"

fixture dup_all_green
rollup "$FIX/pr.json" OPEN deadbeef \
    '{"name":"Fast (Linux x86_64)","status":"COMPLETED","conclusion":"SUCCESS"}' \
    '{"name":"Rust quality","status":"COMPLETED","conclusion":"SUCCESS"}' \
    '{"name":"Rust quality","status":"COMPLETED","conclusion":"SUCCESS"}'
chk "two green runs of the same name still merge" "$(run_mw)" "0"

echo
echo "== refusals that will never go green stop, rather than wait =="
fixture errored
rollup "$FIX/pr.json" OPEN deadbeef \
    '{"context":"Fast (Linux x86_64)","state":"ERROR"}' \
    '{"name":"Rust quality","status":"COMPLETED","conclusion":"SUCCESS"}'
chk "a legacy status of ERROR fails fast instead of timing out" "$(T=30 run_mw)" "2"

fixture unreadable
touch "$FIX/pr_view.fail"
chk "a pull request that cannot be read at all fails before it polls" "$(T=60 run_mw)" "1"
chk "and nothing was merged" "$(merges)" "0"

# The case the retry counter actually exists for, and which the cell above
# does NOT reach: the opening read succeeds, then the token is revoked. Every
# later poll comes back empty, and reporting that as "timed out" invites the
# caller to retry an environment fault as if it were a slow queue.
fixture revoked
printf '2\n' >"$FIX/fail_from"
chk "a read that breaks mid-wait is an environment fault, not a slow check" "$(T=600 run_mw)" "1"
chk "and nothing was merged" "$(merges)" "0"

# ... but a few dropped reads are a flaky API, not a revoked token. Four
# misses then a recovery must still merge, or the counter has just turned
# every transient 502 into a refusal.
fixture flaky
printf '2\n' >"$FIX/fail_from"
printf '5\n' >"$FIX/fail_to"
chk "four dropped reads then a recovery still merges" "$(T=600 run_mw)" "0"

echo
echo "== states that are not about checks =="
fixture closed
rollup "$FIX/pr.json" CLOSED deadbeef
chk "a closed pull request is refused before any polling" "$(run_mw)" "6"

fixture draft
python3 - "$FIX/pr.json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
d["isDraft"] = True
json.dump(d, open(p, "w"))
PY
chk "a draft is refused" "$(run_mw)" "6"

fixture refused
printf '1\n' >"$FIX/merge_rc"
chk "green checks but a refused merge is not reported as success" "$(run_mw)" "6"

echo
echo "== the caller cannot ask for a bypass =="
fixture admin
chk "--admin is rejected as an unknown option" "$(run_mw --admin)" "1"
chk "and it never reached gh" "$(grep -c -- '--admin' "$FIX/argv.log")" "0"
# Not "the string never appears" -- it appears in the header and in the
# rejection message, and pinning prose is how a cell starts failing for the
# wrong reason. Pin the thing that matters: no line that invokes gh carries it.
chk "no gh invocation in the script carries --admin" \
    "$(grep -n 'gh pr merge\|gh api\|gh repo' "$MW" | grep -c -e '--admin')" "0"

echo
echo "== anti-vacuity =="
# Every cell above runs against a fixture. If the parse drifted away from what
# the real API returns, all of them would stay green while the script refused
# every real repository. So: read the real ruleset. Both branches assert, so
# the count is invariant -- a probe that skips silently is the same defect
# this file is about.
if [ -n "${MWG_LIVE:-}" ] && gh auth status >/dev/null 2>&1; then
    live=$(gh api repos/justinchuby/onnx-genai/rulesets \
        -q '.[] | select(.target == "branch") | select(.enforcement == "active") | .id' 2>/dev/null |
        while read -r id; do
            gh api "repos/justinchuby/onnx-genai/rulesets/$id" \
                -q '.rules[]? | select(.type == "required_status_checks")
                    | .parameters.required_status_checks[]?.context' 2>/dev/null
        done | sort -u)
    chk "the real ruleset really does yield required contexts" \
        "$([ -n "$live" ] && echo yes || echo no)" "yes"
else
    # No network here, so pin the property that makes the fixtures meaningful:
    # the required set comes from the API at run time. A literal list in this
    # file would satisfy every cell above and then quietly shrink -- and the
    # failure mode of a required set that lost a name is a green report, which
    # is the whole subject of #1966. Two calls: the ruleset index, then each
    # ruleset's rules.
    # shellcheck disable=SC2016  # matching the literal source text, not expanding it
    chk "the required set is read from the ruleset API, not a list in the script" \
        "$(sed -n '/^required_contexts() {/,/^}/p' "$MW" | grep -c 'gh api "repos/\$REPO/rulesets')" \
        "2"
fi

echo
echo "== SKIPPED is a third outcome =="

# A docs-only change skips the heavy lanes by design, and GitHub's ruleset
# accepts a skipped required check -- so the pull request is merge-ready by
# the repository's own rules while never reporting SUCCESS. Without a way
# through, this script waits for a verdict that is not coming and its operator
# reaches for `--admin`, which is the hazard it exists to remove. Without a
# GATE on the way through, a job skipped by a broken `if:` merges unvalidated.
# Hence: distinct outcome, explicit decision, and still a refusal if GitHub
# says the skip does not satisfy the ruleset.
skipped_fixture() {
    fixture "$1"
    rollup "$FIX/pr.json" OPEN deadbeef \
        '{"name":"Fast (Linux x86_64)","status":"COMPLETED","conclusion":"SKIPPED"}' \
        '{"name":"Rust quality","status":"COMPLETED","conclusion":"SKIPPED"}'
    # `rollup` writes BLOCKED; a repository that accepts the skip reports
    # UNSTABLE, and that difference is the gate two cells below.
    sed -i 's/"mergeStateStatus":"BLOCKED"/"mergeStateStatus":"UNSTABLE"/' "$FIX/pr.json"
}

skipped_fixture skip_default
chk "a SKIPPED required check does not merge by default" "$(run_mw)" "7"
chk "and nothing was merged" "$(merges)" "0"
chk "and the refusal names the outcome, not just a generic wait" \
    "$(grep -c 'concluded SKIPPED' "$FIX/out")" "1"
chk "and it points at the flag, not at --admin" \
    "$(grep -c -- '--allow-skipped' "$FIX/out")" "1"

skipped_fixture skip_allowed
chk "--allow-skipped merges when the skip is the only thing missing" \
    "$(run_mw --allow-skipped)" "0"
chk "and still without --admin" "$(grep -c -- '--admin' "$FIX/argv.log")" "0"

# The gate. `--allow-skipped` is the caller saying the skip is by design; it is
# NOT permission to overrule the repository. BLOCKED means GitHub does not
# accept the skip as satisfying the requirement.
skipped_fixture skip_blocked
sed -i 's/"mergeStateStatus":"UNSTABLE"/"mergeStateStatus":"BLOCKED"/' "$FIX/pr.json"
chk "--allow-skipped does not merge a pull request GitHub reports BLOCKED" \
    "$(run_mw --allow-skipped)" "6"
chk "and nothing was merged" "$(merges)" "0"

# The negative control for the flag: it must widen SKIPPED and nothing else.
# A required context that is ABSENT or still running is not skipped, and the
# absent case is the defect this whole suite exists for (#1966).
fixture skip_not_absent
rollup "$FIX/pr.json" OPEN deadbeef \
    '{"name":"Fast (Linux x86_64)","status":"COMPLETED","conclusion":"SKIPPED"}'
sed -i 's/"mergeStateStatus":"BLOCKED"/"mergeStateStatus":"UNSTABLE"/' "$FIX/pr.json"
chk "--allow-skipped does not excuse a required check with no row at all" \
    "$(run_mw --allow-skipped)" "3"
chk "and nothing was merged" "$(merges)" "0"

fixture skip_not_pending
rollup "$FIX/pr.json" OPEN deadbeef \
    '{"name":"Fast (Linux x86_64)","status":"COMPLETED","conclusion":"SKIPPED"}' \
    '{"name":"Rust quality","status":"IN_PROGRESS","conclusion":null}'
sed -i 's/"mergeStateStatus":"BLOCKED"/"mergeStateStatus":"UNSTABLE"/' "$FIX/pr.json"
chk "--allow-skipped does not excuse a required check still running" \
    "$(run_mw --allow-skipped)" "3"
chk "and nothing was merged" "$(merges)" "0"

fixture skip_not_failure
rollup "$FIX/pr.json" OPEN deadbeef \
    '{"name":"Fast (Linux x86_64)","status":"COMPLETED","conclusion":"SKIPPED"}' \
    '{"name":"Rust quality","status":"COMPLETED","conclusion":"FAILURE"}'
sed -i 's/"mergeStateStatus":"BLOCKED"/"mergeStateStatus":"UNSTABLE"/' "$FIX/pr.json"
chk "--allow-skipped does not excuse a required check that failed" \
    "$(run_mw --allow-skipped)" "2"
chk "and nothing was merged" "$(merges)" "0"

# And the flag must not change the ordinary path in either direction: an
# all-green pull request merges with it, and reports SUCCESS rather than
# claiming a skip it did not have.
fixture skip_absent_is_green
chk "--allow-skipped still merges an all-green pull request" \
    "$(run_mw --allow-skipped)" "0"
chk "and reports SUCCESS, not a skip" \
    "$(grep -c 'every required check reported SUCCESS' "$FIX/out")" "1"

echo
# An assertion that quietly stops running is indistinguishable from one that
# passes. Both branches of the probe above assert, so this total is invariant
# across environments.
chk "every assertion in this file ran" "$((pass + fail + 1))" "57"

echo
echo "passed=${pass} failed=${fail}"
[ "$fail" -eq 0 ]
