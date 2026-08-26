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

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P) || exit 1
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P) || exit 1
cd "$ROOT" || exit 1

MW="$ROOT/scripts/merge_when_green.sh"
REPO=acme/widget

pass=0
fail=0

resolve_work_root() {
    local candidate=${CARGO_TARGET_TMPDIR:-"$ROOT/target"}
    case "$candidate" in
        [A-Za-z]:[\\/]*)
            command -v cygpath >/dev/null 2>&1 || {
                echo "merge_when_green_test.sh: Windows temp path requires cygpath: $candidate" >&2
                return 1
            }
            candidate=$(cygpath -u "$candidate") || return 1
            ;;
        /*) ;;
        *) candidate="$ROOT/$candidate" ;;
    esac
    mkdir -p "$candidate" || return 1
    CDPATH= cd -- "$candidate" && pwd -P
}

WORK_ROOT=$(resolve_work_root) || exit 1
WORK=$(mktemp -d "$WORK_ROOT/mwg-test.XXXXXX") || exit 1
case "$WORK" in
    "" | / | "$ROOT")
        echo "merge_when_green_test.sh: refusing unsafe work root: $WORK" >&2
        exit 1
        ;;
esac

cleanup() { rm -rf -- "$WORK"; }
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

read_optional_uint() {
    local path=$1 default=$2 value
    if [ ! -e "$path" ]; then
        printf '%s\n' "$default"
        return
    fi
    [ -f "$path" ] && [ -r "$path" ] || return 90
    IFS= read -r value <"$path" || return 90
    case "$value" in
        0 | [1-9] | [1-9][0-9]*) ;;
        *) return 90 ;;
    esac
    [ "${#value}" -le 10 ] || return 90
    [ "$((10#$value))" -le 2147483647 ] || return 90
    printf '%s\n' "$value"
}

read_exit_code() {
    local path=$1 value
    [ -f "$path" ] && [ -r "$path" ] || return 90
    IFS= read -r value <"$path" || return 90
    case "$value" in
        0 | [1-9] | [1-9][0-9]*) ;;
        *) return 90 ;;
    esac
    [ "${#value}" -le 3 ] || return 90
    [ "$((10#$value))" -le 255 ] || return 90
    printf '%s\n' "$value"
}

printf '%s\n' "$*" >>"$FIX/argv.log" || exit 90

args="$*"
case "$args" in
    *"pr merge"*)
        merge_rc=$(read_exit_code "$FIX/merge_rc") || exit 90
        printf '%s\n' "$*" >>"$FIX/merge.log" || exit 90
        exit "$merge_rc"
        ;;
    *"pr view"*)
        n=$(read_optional_uint "$FIX/poll" 1) || exit 90
        printf '%s\n' "$((n + 1))" >"$FIX/poll" || exit 90
        if [ -f "$FIX/pr_view.fail" ]; then exit 1; fi
        # Fail a WINDOW of reads rather than all of them, so a cell can let the
        # opening snapshot succeed and break the connection afterwards -- which
        # is the only way to reach the in-loop retry counter.
        from=$(read_optional_uint "$FIX/fail_from" 0) || exit 90
        to=$(read_optional_uint "$FIX/fail_to" 0) || exit 90
        if [ "$from" -gt 0 ] && [ "$n" -ge "$from" ] && { [ "$to" = 0 ] || [ "$n" -le "$to" ]; }; then
            exit 1
        fi
        if [ -f "$FIX/pr.$n.json" ]; then
            cat "$FIX/pr.$n.json" || exit 90
        else
            cat "$FIX/pr.json" || exit 90
        fi
        ;;
    *rulesets/*)
        case "$args" in
            *bypass_actors*) cat "$FIX/bypass" || exit 90 ;;
            *) cat "$FIX/contexts" || exit 90 ;;
        esac
        ;;
    *rulesets*)
        if [ -f "$FIX/ruleset_ids.fail" ]; then exit 1; fi
        cat "$FIX/ruleset_ids" || exit 90
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
    require_fixture "$FIX" || exit 1
}

rollup() {
    local out=$1 state=$2 head=$3
    shift 3
    local checks=""
    for c in "$@"; do checks="${checks:+$checks,}$c"; done
    if ! cat >"$out" <<EOF
{"state":"$state","headRefOid":"$head","isDraft":false,
 "mergeStateStatus":"BLOCKED","statusCheckRollup":[$checks]}
EOF
    then
        echo "merge_when_green_test.sh: HARNESS ERROR: could not write $out" >&2
        exit 1
    fi
}

set_merge_state() {
    local path=$1 value=$2
    if ! python3 - "$path" "$value" <<'PY'
import json
import sys

path, value = sys.argv[1:]
with open(path, encoding="utf-8") as source:
    document = json.load(source)
if value == "__MISSING__":
    document.pop("mergeStateStatus", None)
else:
    document["mergeStateStatus"] = value
with open(path, "w", encoding="utf-8", newline="\n") as output:
    json.dump(document, output, separators=(",", ":"))
    output.write("\n")
PY
    then
        echo "merge_when_green_test.sh: HARNESS ERROR: could not update $path" >&2
        exit 1
    fi
}

require_fixture() {
    local root=$1 file failed=0 merge_rc
    for file in ruleset_ids contexts bypass merge_rc argv.log pr.json; do
        if [ ! -e "$root/$file" ]; then
            echo "merge_when_green_test.sh: HARNESS ERROR: fixture '$root' is missing $file" >&2
            failed=1
        elif [ ! -f "$root/$file" ] || [ ! -r "$root/$file" ]; then
            echo "merge_when_green_test.sh: HARNESS ERROR: fixture '$root/$file' is not a regular readable file" >&2
            failed=1
        fi
    done
    if [ -f "$root/argv.log" ] && [ ! -w "$root/argv.log" ]; then
        echo "merge_when_green_test.sh: HARNESS ERROR: fixture '$root/argv.log' is not writable" >&2
        failed=1
    fi
    if [ -f "$root/merge_rc" ] && [ -r "$root/merge_rc" ]; then
        if ! IFS= read -r merge_rc <"$root/merge_rc"; then
            echo "merge_when_green_test.sh: HARNESS ERROR: merge_rc is empty or unreadable" >&2
            failed=1
        else
            case "$merge_rc" in
                0 | [1-9] | [1-9][0-9]*)
                    if [ "${#merge_rc}" -gt 3 ] || [ "$((10#$merge_rc))" -gt 255 ]; then
                        echo "merge_when_green_test.sh: HARNESS ERROR: merge_rc is outside exit range 0..255: $merge_rc" >&2
                        failed=1
                    fi
                    ;;
                *)
                    echo "merge_when_green_test.sh: HARNESS ERROR: merge_rc must be an integer in exit range 0..255, got '$merge_rc'" >&2
                    failed=1
                    ;;
            esac
        fi
    fi
    [ "$failed" -eq 0 ]
}

# `wc -l <missing` prints a shell error that looks like a suite fault; the
# absence of the file IS the assertion here, so name it.
merges() {
    if [ ! -e "$FIX/merge.log" ]; then
        echo 0
    elif [ ! -f "$FIX/merge.log" ] || [ ! -r "$FIX/merge.log" ]; then
        echo "merge_when_green_test.sh: HARNESS ERROR: merge.log is not a regular readable file" >&2
        return 1
    else
        wc -l <"$FIX/merge.log"
    fi
}

run_mw() {
    if ! require_fixture "$FIX" >"$FIX/out" 2>&1; then
        echo 99
        return
    fi
    MWG_FIX="$FIX" PATH="$WORK/bin:$PATH" \
        bash "$MW" 7 --repo "$REPO" --timeout "${T:-0}" --poll 1 "$@" \
        >"$FIX/out" 2>&1
    echo $?
}

make_shim

fixture_contract_refuses() {
    local label=$1 pattern=$2 rc merge_count
    rc=$(run_mw)
    merge_count=$(merges) || exit 1
    if [ "$rc" != 99 ] || [ "$merge_count" != 0 ] || ! grep -q "$pattern" "$FIX/out"; then
        echo "  FAIL  $label"
        echo "          rc=$rc merges=$merge_count pattern=$pattern"
        cat "$FIX/out" >&2 || exit 1
        exit 1
    fi
    echo "  PASS  $label (harness refusal; not part of the 75 gate-verdict assertions)"
}

echo "== harness integrity =="
fixture missing_merge_rc
rm "$FIX/merge_rc"
fixture_contract_refuses "a missing merge_rc fails before the production gate" \
    "HARNESS ERROR:.*missing merge_rc"

fixture unreadable_merge_rc
rm "$FIX/merge_rc"
mkdir "$FIX/merge_rc"
fixture_contract_refuses "a merge_rc read failure fails before the production gate" \
    "HARNESS ERROR:.*merge_rc.*not a regular readable file"

fixture empty_merge_rc
: >"$FIX/merge_rc"
fixture_contract_refuses "an empty merge_rc fails before the production gate" \
    "HARNESS ERROR: merge_rc is empty or unreadable"

fixture noninteger_merge_rc
printf 'success\n' >"$FIX/merge_rc"
fixture_contract_refuses "a non-integer merge_rc fails before the production gate" \
    "HARNESS ERROR: merge_rc must be an integer"

fixture negative_merge_rc
printf '%s\n' '-1' >"$FIX/merge_rc"
fixture_contract_refuses "a negative merge_rc fails before the production gate" \
    "HARNESS ERROR: merge_rc must be an integer"

fixture out_of_range_merge_rc
printf '256\n' >"$FIX/merge_rc"
fixture_contract_refuses "an out-of-range merge_rc fails before the production gate" \
    "HARNESS ERROR: merge_rc is outside exit range 0..255"

fixture missing_pr_json
rm "$FIX/pr.json"
chk "a missing pr.json is a harness error, never a gate verdict" "$(run_mw)" "99"
chk "and a broken fixture cannot record a merge" "$(merges)" "0"
chk "and the harness error names the missing contract file" \
    "$(grep -c 'HARNESS ERROR:.*missing pr.json' "$FIX/out")" "1"

echo
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
    set_merge_state "$FIX/pr.json" UNSTABLE
    require_fixture "$FIX" || exit 1
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
set_merge_state "$FIX/pr.json" BLOCKED
chk "--allow-skipped does not merge a pull request GitHub reports BLOCKED" \
    "$(run_mw --allow-skipped)" "6"
chk "and nothing was merged" "$(merges)" "0"

# The negative control for the flag: it must widen SKIPPED and nothing else.
# A required context that is ABSENT or still running is not skipped, and the
# absent case is the defect this whole suite exists for (#1966).
fixture skip_not_absent
rollup "$FIX/pr.json" OPEN deadbeef \
    '{"name":"Fast (Linux x86_64)","status":"COMPLETED","conclusion":"SKIPPED"}'
set_merge_state "$FIX/pr.json" UNSTABLE
chk "--allow-skipped does not excuse a required check with no row at all" \
    "$(run_mw --allow-skipped)" "3"
chk "and nothing was merged" "$(merges)" "0"

fixture skip_not_pending
rollup "$FIX/pr.json" OPEN deadbeef \
    '{"name":"Fast (Linux x86_64)","status":"COMPLETED","conclusion":"SKIPPED"}' \
    '{"name":"Rust quality","status":"IN_PROGRESS","conclusion":null}'
set_merge_state "$FIX/pr.json" UNSTABLE
chk "--allow-skipped does not excuse a required check still running" \
    "$(run_mw --allow-skipped)" "3"
chk "and nothing was merged" "$(merges)" "0"

fixture skip_not_failure
rollup "$FIX/pr.json" OPEN deadbeef \
    '{"name":"Fast (Linux x86_64)","status":"COMPLETED","conclusion":"SKIPPED"}' \
    '{"name":"Rust quality","status":"COMPLETED","conclusion":"FAILURE"}'
set_merge_state "$FIX/pr.json" UNSTABLE
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
echo "== a skip must be the only thing wrong with its name =="

# The cells above give each bad conclusion its OWN required name, which the
# pre-existing failure path already caught -- they never reach the fold. A
# re-run puts two runs under ONE name, and the fold then has to choose which
# one to report. While the only question was green-or-not that choice was
# free; once SKIPPED is mergeable it decides whether a failure is visible.
# Both orders, because the hazard is that rollup array order alone decides.
dup_fixture() {
    fixture "$1"
    rollup "$FIX/pr.json" OPEN deadbeef \
        '{"name":"Fast (Linux x86_64)","status":"COMPLETED","conclusion":"SUCCESS"}' \
        "$2" "$3"
    set_merge_state "$FIX/pr.json" UNSTABLE
    require_fixture "$FIX" || exit 1
}

RQ_SKIP='{"name":"Rust quality","status":"COMPLETED","conclusion":"SKIPPED"}'
RQ_FAIL='{"name":"Rust quality","status":"COMPLETED","conclusion":"FAILURE"}'
RQ_RUN='{"name":"Rust quality","status":"IN_PROGRESS","conclusion":null}'

dup_fixture dup_skip_then_fail "$RQ_SKIP" "$RQ_FAIL"
chk "a name that skipped AND failed is a failure, skip listed first" \
    "$(run_mw --allow-skipped)" "2"
chk "and nothing was merged" "$(merges)" "0"

dup_fixture dup_fail_then_skip "$RQ_FAIL" "$RQ_SKIP"
chk "and the same, skip listed second -- array order decides nothing" \
    "$(run_mw --allow-skipped)" "2"

# Without the flag this input must not report itself as a skip either: exit 7
# tells the operator to re-run with --allow-skipped, so a masked failure here
# would route them into merging it.
dup_fixture dup_default_not_seven "$RQ_SKIP" "$RQ_FAIL"
chk "and without the flag it is still a failure, not an offer to allow it" \
    "$(run_mw)" "2"

dup_fixture dup_skip_then_running "$RQ_SKIP" "$RQ_RUN"
chk "a name that skipped and is also still running keeps waiting" \
    "$(run_mw --allow-skipped)" "3"
chk "and nothing was merged" "$(merges)" "0"

echo
echo "== the merge state is an allowlist, not a BLOCKED denylist =="

# Refusing only BLOCKED would make every other value -- including GitHub
# saying it has not worked out mergeability yet, and including the field
# vanishing in an API change -- count as the repository accepting the skip.
state_fixture() {
    fixture "$1"
    rollup "$FIX/pr.json" OPEN deadbeef \
        '{"name":"Fast (Linux x86_64)","status":"COMPLETED","conclusion":"SKIPPED"}' \
        '{"name":"Rust quality","status":"COMPLETED","conclusion":"SKIPPED"}'
    set_merge_state "$FIX/pr.json" "$2"
    require_fixture "$FIX" || exit 1
}

state_fixture state_dirty DIRTY
chk "--allow-skipped does not merge a pull request with conflicts" \
    "$(run_mw --allow-skipped)" "6"
chk "and nothing was merged" "$(merges)" "0"

state_fixture state_behind BEHIND
chk "--allow-skipped does not merge a pull request that is BEHIND" \
    "$(run_mw --allow-skipped)" "6"

state_fixture state_unknown UNKNOWN
chk "an uncomputed merge state is not an accepting one -- it waits" \
    "$(run_mw --allow-skipped)" "3"
chk "and nothing was merged" "$(merges)" "0"

skipped_fixture state_missing
set_merge_state "$FIX/pr.json" __MISSING__
chk "and the field being absent altogether waits too, not merges" \
    "$(run_mw --allow-skipped)" "3"
chk "and nothing was merged" "$(merges)" "0"
chk "the fixture really did drop the field" \
    "$(grep -c mergeStateStatus "$FIX/pr.json")" "0"

state_fixture state_clean CLEAN
chk "a CLEAN pull request with a skipped required check merges" \
    "$(run_mw --allow-skipped)" "0"

echo
# A `chk` assertion that quietly stops running is indistinguishable from one
# that passes. The fail-fast fixture-contract mutations above deliberately do
# not change this established 75-cell gate-verdict census.
chk "all 75 gate-verdict assertions ran" "$((pass + fail + 1))" "75"

echo
echo "passed=${pass} failed=${fail}"
[ "$fail" -eq 0 ]
