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
# An assertion that quietly stops running is indistinguishable from one that
# passes. Both branches of the probe above assert, so this total is invariant
# across environments.
chk "every assertion in this file ran" "$((pass + fail + 1))" "27"

echo
echo "passed=${pass} failed=${fail}"
[ "$fail" -eq 0 ]
