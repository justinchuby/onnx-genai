#!/usr/bin/env bash
# Merge a pull request only after its REQUIRED checks have actually reported
# success -- and refuse, loudly, in every case where that cannot be shown.
#
# Why this exists (#1966)
# ----------------------
# `gh pr merge --auto` does not mean "merge when the required checks are
# green". It means "merge as soon as the mergeable state allows". For an actor
# covered by a ruleset bypass -- this repository's `merge rules` ruleset has a
# single bypass actor, `RepositoryRole` id 5 (admin), with `bypass_mode:
# always` -- the required contexts are advisory, so the mergeable state allows
# it as soon as there is nothing visibly wrong.
#
# The trap is that "nothing visibly wrong" includes "the check has not been
# created yet". Under runner saturation a context can take 15-60 minutes to
# appear. #1958 was armed at 08:05 and merged at 08:05:37; its `Fast (Linux
# x86_64)` did not START until 08:09:40 and then failed. Nobody typed
# `--admin`. The same command, armed twenty minutes later against the same
# ruleset, waits correctly -- which is worse than a consistently broken gate,
# because it teaches you to trust it.
#
# So the property this script establishes, and that `--auto` does not:
#
#   every context the ruleset names as required is PRESENT in this PR's check
#   rollup, for THIS head commit, and every one of them concluded SUCCESS.
#
# Three words in that sentence are load-bearing and each corresponds to a
# defect we have already shipped:
#
#   PRESENT   absent is not pending, and pending is not pass. A poller that
#             greps for failure reports success for a check that does not
#             exist. That is the vacuous-guard class (#1817) relocated into
#             the gate that guards everything else.
#   THIS      a force-push resets the head SHA and the old contexts do not
#             carry over, but an armed `--auto` does. Rebasing under an armed
#             auto-merge re-opens the window from the top.
#   RULESET   the required set is read from the API, not from a list in this
#             file. A hardcoded list drifts silently into a shorter one, and
#             the failure mode of that is a green report.
#
# It never passes `--admin` and has no flag that would. If the checks are not
# green this script does not merge; it tells you why and exits non-zero.
#
# Usage:
#   scripts/merge_when_green.sh <pr-number> [options]
#
#     --repo OWNER/NAME     default: resolved from the checkout
#     --method M            merge|squash|rebase   (default: squash)
#     --timeout SECONDS     total wait budget     (default: 3600)
#     --poll SECONDS        interval              (default: 30)
#     --dry-run             report the decision, never merge
#
# Exit codes -- deliberately distinct, because "did not merge" is not one
# outcome and a caller that cannot tell them apart will retry the wrong ones:
#
#   0  merged
#   1  usage / environment fault
#   2  a required check concluded failure -- will never go green, stop
#   3  timed out waiting (still pending, or never appeared)
#   4  the head commit moved while waiting -- the wait was invalidated
#   5  the required set could not be determined -- FAIL CLOSED, never merge
#   6  the pull request is not open, or is not mergeable for a reason that is
#      not about checks (conflicts, draft, review)

set -uo pipefail

PROG=${0##*/}

die() {
    echo "$PROG: $*" >&2
    exit 1
}

usage() {
    sed -n '/^# Usage:/,/^#   6 /p' "$0" | sed 's/^# \{0,1\}//'
}

PR=""
REPO=""
METHOD=squash
TIMEOUT=3600
POLL=30
DRY_RUN=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --repo | --method | --timeout | --poll)
            [ "$#" -ge 2 ] || die "$1 requires a value"
            ;;
    esac
    case "$1" in
        --repo)
            REPO=$2
            shift 2
            ;;
        --method)
            METHOD=$2
            shift 2
            ;;
        --timeout)
            TIMEOUT=$2
            shift 2
            ;;
        --poll)
            POLL=$2
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        -*)
            die "unknown option: $1 (there is deliberately no --admin)"
            ;;
        *)
            [ -z "$PR" ] || die "expected one pull request number, got '$PR' and '$1'"
            PR=$1
            shift
            ;;
    esac
done

[ -n "$PR" ] || {
    usage >&2
    exit 1
}
case "$PR" in
    '' | *[!0-9]*) die "pull request must be a number, got: '$PR'" ;;
esac
case "$METHOD" in
    merge | squash | rebase) ;;
    *) die "--method takes merge, squash or rebase, got: '$METHOD'" ;;
esac
case "$TIMEOUT" in '' | *[!0-9]*) die "--timeout takes seconds" ;; esac
case "$POLL" in '' | *[!0-9]*) die "--poll takes seconds" ;; esac
[ "$POLL" -gt 0 ] || die "--poll must be positive"

command -v gh >/dev/null 2>&1 || die "gh is not on PATH"

if [ -z "$REPO" ]; then
    REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null)
    [ -n "$REPO" ] || die "could not resolve the repository; pass --repo OWNER/NAME"
fi

# ---------------------------------------------------------------------------
# The required set, from the ruleset API.
#
# Collected across every ACTIVE branch ruleset rather than the one that looks
# most relevant. Over-collecting is the safe direction: it can only make this
# script wait for a check it did not have to, never merge past one it did.
# Under-collecting is the failure this script exists to prevent.
# ---------------------------------------------------------------------------
required_contexts() {
    local ids id
    ids=$(gh api "repos/$REPO/rulesets" \
        -q '.[] | select(.target == "branch") | select(.enforcement == "active") | .id' 2>/dev/null) || return 1
    [ -n "$ids" ] || return 1
    for id in $ids; do
        gh api "repos/$REPO/rulesets/$id" \
            -q '.rules[]? | select(.type == "required_status_checks")
                | .parameters.required_status_checks[]?.context' 2>/dev/null
    done | sort -u
}

# Report a bypass, because it is the reason this script is not just `--auto`.
warn_if_bypassed() {
    local ids id actors
    ids=$(gh api "repos/$REPO/rulesets" \
        -q '.[] | select(.target == "branch") | select(.enforcement == "active") | .id' 2>/dev/null) || return 0
    for id in $ids; do
        actors=$(gh api "repos/$REPO/rulesets/$id" \
            -q '[.bypass_actors[]? | select(.bypass_mode == "always")] | length' 2>/dev/null)
        if [ -n "$actors" ] && [ "$actors" != "0" ]; then
            echo "$PROG: note: ruleset $id has $actors always-bypass actor(s), so these checks are advisory for some actors -- which is why this script exists rather than --auto (#1966)" >&2
            return 0
        fi
    done
}

REQUIRED=$(required_contexts)
if [ -z "$REQUIRED" ]; then
    echo "$PROG: could not determine any required status check from $REPO's rulesets." >&2
    echo "$PROG: refusing to merge. An empty required set is indistinguishable from" >&2
    echo "$PROG: a green one, and this script will not treat it as green." >&2
    exit 5
fi
warn_if_bypassed

echo "$PROG: required checks for $REPO:"
printf '%s\n' "$REQUIRED" | sed 's/^/  /'

# ---------------------------------------------------------------------------
# Pin the head. A force-push invalidates everything we are about to observe.
# ---------------------------------------------------------------------------
pr_json() {
    gh pr view "$PR" --repo "$REPO" \
        --json state,headRefOid,isDraft,mergeStateStatus,statusCheckRollup 2>/dev/null
}

SNAP=$(pr_json) || die "could not read pull request #$PR in $REPO"
[ -n "$SNAP" ] || die "could not read pull request #$PR in $REPO"

state=$(printf '%s' "$SNAP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["state"])')
HEAD=$(printf '%s' "$SNAP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["headRefOid"])')
draft=$(printf '%s' "$SNAP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["isDraft"])')

if [ "$state" != "OPEN" ]; then
    echo "$PROG: #$PR is $state, not OPEN -- nothing to do." >&2
    exit 6
fi
if [ "$draft" = "True" ] || [ "$draft" = "true" ]; then
    echo "$PROG: #$PR is a draft. Refusing." >&2
    exit 6
fi

echo "$PROG: watching #$PR at $HEAD (timeout ${TIMEOUT}s, poll ${POLL}s)"

# ---------------------------------------------------------------------------
# The verdict.
#
# Emitted as one line per required context so the reason is in the log rather
# than in somebody's memory of what the loop was doing:
#
#   <context> <status> <conclusion>
#
# with a synthesised `ABSENT -` for a context the rollup does not mention at
# all. That row is the whole point: it is the one `gh pr checks | grep -v
# fail` silently reports as fine.
# ---------------------------------------------------------------------------
verdict() {
    printf '%s' "$1" | REQ="$REQUIRED" python3 -c '
import json, os, sys

snap = json.load(sys.stdin)
rollup = {c.get("name") or c.get("context"): c
          for c in (snap.get("statusCheckRollup") or [])}

for name in os.environ["REQ"].split("\n"):
    name = name.strip()
    if not name:
        continue
    c = rollup.get(name)
    if c is None:
        print("%s\tABSENT\t-" % name)
        continue
    status = c.get("status") or c.get("state") or "-"
    concl = c.get("conclusion") or c.get("state") or "-"
    print("%s\t%s\t%s" % (name, status, concl))
'
}

deadline=$((SECONDS + TIMEOUT))
while :; do
    SNAP=$(pr_json)
    if [ -z "$SNAP" ]; then
        echo "$PROG: could not read #$PR; retrying" >&2
        sleep "$POLL"
        [ "$SECONDS" -lt "$deadline" ] || { exit 3; }
        continue
    fi

    now_state=$(printf '%s' "$SNAP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["state"])')
    now_head=$(printf '%s' "$SNAP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["headRefOid"])')

    if [ "$now_head" != "$HEAD" ]; then
        echo "$PROG: head moved $HEAD -> $now_head while waiting." >&2
        echo "$PROG: the checks observed so far were for a commit that is no longer" >&2
        echo "$PROG: the tip. Refusing to merge; re-run against the new head." >&2
        exit 4
    fi
    if [ "$now_state" != "OPEN" ]; then
        echo "$PROG: #$PR became $now_state while waiting." >&2
        exit 6
    fi

    V=$(verdict "$SNAP")
    echo "-- $(date -u +%H:%M:%S) --"
    printf '%s\n' "$V" | while IFS="$(printf '\t')" read -r n s c; do
        printf '  %-28s %-12s %s\n' "$n" "$s" "$c"
    done

    failed=$(printf '%s\n' "$V" | awk -F'\t' '$3 == "FAILURE" || $3 == "TIMED_OUT" || $3 == "CANCELLED" || $3 == "ACTION_REQUIRED" || $3 == "STARTUP_FAILURE" { print $1 }')
    if [ -n "$failed" ]; then
        echo "$PROG: required check(s) did not pass:" >&2
        printf '%s\n' "$failed" | sed 's/^/  /' >&2
        echo "$PROG: not merging." >&2
        exit 2
    fi

    notgreen=$(printf '%s\n' "$V" | awk -F'\t' '$3 != "SUCCESS" { print $1 }')
    if [ -z "$notgreen" ]; then
        echo "$PROG: every required check reported SUCCESS at $HEAD."
        if [ "$DRY_RUN" = 1 ]; then
            echo "$PROG: --dry-run, not merging."
            exit 0
        fi
        # No --admin, and no code path that could add one.
        if gh pr merge "$PR" --repo "$REPO" "--$METHOD"; then
            echo "$PROG: merged #$PR."
            exit 0
        fi
        echo "$PROG: checks were green but the merge was refused (conflicts, review, or protection)." >&2
        exit 6
    fi

    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "$PROG: timed out after ${TIMEOUT}s. Still not green:" >&2
        printf '%s\n' "$notgreen" | sed 's/^/  /' >&2
        echo "$PROG: not merging. An unfinished check is not a passing one, and a" >&2
        echo "$PROG: check that never appeared is not a passing one either." >&2
        exit 3
    fi
    sleep "$POLL"
done
