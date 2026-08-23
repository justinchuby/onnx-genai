#!/usr/bin/env bash
# hostlock_test.sh — self-test for hostlock.sh
#
# Run: scripts/hostlock_test.sh
#
# Costs no measurable CPU (it is mkdir/rename traffic and short sleeps), so it
# is safe to run on a busy host -- which matters, because the whole point of
# the thing under test is that the host is busy.
#
# The test that earned its place is the atomicity watcher. An earlier
# implementation acquired with `mkdir` and then wrote its metadata as a
# second step, and it was caught handing the host to THREE simultaneous
# winners: between the mkdir and the finished meta file there is a window
# where the lock exists with no readable anchor pid, and a competitor looking
# during that window reaps it as stale. A lock that can be held twice is
# worse than no lock, because people trust it.
#
# That original catch was luck, and the 40-way race that produced it does not
# reproduce it on demand -- when the bug was deliberately reintroduced, RACE A
# passed. The reason is in the comment above that test. The watcher checks the
# same invariant deterministically and does fail against the reintroduced bug,
# so it is the one to trust. RACE A and RACE B are kept as cheap smoke tests
# for the ordinary exclusion and reaping paths.

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
LOCK="$(pwd)/.hostlock-selftest"
export HOSTLOCK_DIR="$LOCK"
HL=scripts/hostlock.sh

pass=0
fail=0

cleanup() { rm -rf "$LOCK" "$LOCK".reaper "$LOCK".dead.* "$LOCK".stage.* "$LOCK".dead.* "$LOCK".gate 2>/dev/null; }
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

# Deliver a signal to a pid we spawned ourselves.
sig() { python3 -c 'import os,sys,signal; os.kill(int(sys.argv[1]), int(sys.argv[2]))' "$1" "$2" 2>/dev/null; }

# Launch N acquirers that all reach mkdir at the same instant.
#
# The obvious `for i in $(seq 40); do acquire & done` does NOT do this: each
# fork costs a millisecond or so, so acquirer 1 has finished long before
# acquirer 40 starts and they never actually collide. That version of this
# test passed against a deliberately reintroduced lock-doubling bug, which is
# the only thing worse than not having the test.
#
# So gate them on a FIFO. Every child blocks opening it for read, which costs
# no CPU, and all of them unblock the moment the parent opens it for write.
race() {
    local n=$1 owner_prefix=$2 log=$3 gate="$LOCK.gate"
    rm -f "$gate" "$log"
    mkfifo "$gate"
    for i in $(seq 1 "$n"); do
        (
            exec 3<"$gate"
            cat <&3 >/dev/null
            $HL acquire --owner "${owner_prefix}${i}" --ttl 600
        ) >>"$log" 2>&1 &
    done
    sleep 1
    exec 4>"$gate"
    exec 4>&-
    wait
    rm -f "$gate"
}

# Is this pid running and not already a zombie? Avoids `kill -0`, and treats
# a reaped-but-unwaited child as dead, which it is.
alive() {
    local st
    st=$(sed 's/.*) //' "/proc/$1/stat" 2>/dev/null | awk '{print $1}')
    [ -n "$st" ] && [ "$st" != Z ]
}

# Wait for a child, but never forever.
#
# A runner that fails to die when signalled is exactly the defect these tests
# exist to catch, and a bare `wait` on one hangs the suite instead of
# reporting it -- which is how the SIGHUP falsifier ran for ten minutes and
# said nothing. Bound it, kill the straggler, and return failure.
wait_bounded() {
    local pid=$1 limit=${2:-25} n=0
    while alive "$pid" && [ "$n" -lt "$limit" ]; do
        sleep 1
        n=$((n + 1))
    done
    if alive "$pid"; then
        sig "$pid" 9
        wait "$pid" 2>/dev/null
        return 1
    fi
    wait "$pid" 2>/dev/null
    return 0
}

# Read one field from `status --porcelain`. Parsing positional fields of the
# human-readable output made these tests fail for reasons that had nothing to
# do with the lock, so the tests use the machine-readable contract instead.
st() { $HL status --porcelain | sed -n "s/^$1=//p"; }

# A pid that is guaranteed dead, for forging a stale lock.
dead_pid() { bash -c 'echo $$'; }

forge_stale_lock() {
    cleanup
    mkdir -p "$LOCK"
    printf 'anchor_pid=%s\nstart_time=999999\nowner=ghost\nreason=crashed\nacquired_at=x\nacquired_epoch=1\nttl=600\n' \
        "$(dead_pid)" >"$LOCK/meta"
}

echo "== lifecycle =="
cleanup
chk "free host reports FREE" "$(st state)" "FREE"
$HL acquire --owner leon --reason "softmax matrix" --ttl 600 >/dev/null 2>&1
chk "after acquire reports HELD" "$(st state)" "HELD"
chk "status names the owner" "$(st owner)" "leon"
chk "status carries the reason" "$(st reason)" "softmax matrix"
$HL release >/dev/null 2>&1
chk "after release reports FREE" "$(st state)" "FREE"

echo "== the lock outlives the process that took it =="
# The bug this catches: anchoring to the script's own pid makes the lock stale
# the instant the script returns to the caller's shell.
cleanup
bash -c "HOSTLOCK_DIR='$LOCK' $HL acquire --owner leon --ttl 600" >/dev/null 2>&1
chk "lock taken by an exited script is still HELD" "$(st state)" "HELD"

echo "== mutual exclusion =="
cleanup
$HL acquire --owner leon --ttl 600 >/dev/null 2>&1
out=$($HL acquire --owner roy --ttl 600 2>&1)
rc=$?
chk "second acquirer exits 2 (busy)" "$rc" "2"
chk "second acquirer is told it is busy" "$(echo "$out" | grep -c BUSY)" "1"
chk "second acquirer is told who holds it" "$(echo "$out" | grep -c 'by leon')" "1"
cleanup

echo "== atomicity: a lock is never visible without its metadata =="
# This is the deterministic version of the mutual-exclusion property, and it
# is the test that actually has power over the lock-doubling bug.
#
# Racing 40 shell processes cannot reliably hit the window: each acquirer is
# a separate ~10ms bash startup, so even released simultaneously from a FIFO
# barrier they arrive at mkdir spread over tens of milliseconds, while the
# window itself is ~2ms. Both falsifiers below passed 40-way RACE A.
#
# So observe the invariant directly instead of trying to lose a race against
# it. A watcher polls the lock directory as fast as it can while acquires and
# releases run, and asserts it NEVER sees a lock that exists but has no
# complete metadata -- because such a lock is exactly what a competitor would
# reap out from under a live holder.
cleanup
atomicity_violations=$(
    python3 - "$LOCK" <<'PYEOF' &
import os, sys, time

# A violation is a lock directory that EXISTS and has no complete metadata.
#
# The naive check -- isdir() then open(meta) -- reports a violation every
# time a release lands between those two syscalls, because the open fails
# with ENOENT. That is the watcher being non-atomic, not the lock, and it
# produced 39 false positives against a correct implementation. So on any
# read failure, re-confirm the directory is still there and try once more;
# only a lock that is still present and still unreadable counts.
lock = sys.argv[1]
meta = os.path.join(lock, "meta")
COMPLETE = "runnable_at_acquire="


def incomplete():
    try:
        with open(meta) as fh:
            return COMPLETE not in fh.read()
    except OSError:
        return None


bad = 0
deadline = time.time() + 60
while time.time() < deadline:
    if os.path.isdir(lock):
        verdict = incomplete()
        if verdict is None or verdict:
            if os.path.isdir(lock):
                second = incomplete()
                if second is True or (second is None and os.path.isdir(lock)):
                    bad += 1
    if os.path.exists(lock + ".watchdone"):
        break
print(bad)
PYEOF
    watcher=$!
    for _ in $(seq 1 150); do
        $HL acquire --owner leon --ttl 600 >/dev/null 2>&1
        $HL release >/dev/null 2>&1
    done
    touch "$LOCK.watchdone"
    wait $watcher
)
chk "no lock ever observed without complete metadata" "$atomicity_violations" "0"
rm -f "$LOCK.watchdone"
cleanup

echo "== RACE A: 40 simultaneous acquirers on a free lock =="
for trial in 1 2 3; do
    cleanup
    log="$LOCK.racea.log"
    race 40 a "$log"
    chk "trial $trial: exactly one winner" "$(grep -c 'outcome=acquired' "$log")" "1"
    rm -f "$log"
done

echo "== RACE B: 40 simultaneous acquirers on a stale lock =="
for trial in 1 2 3; do
    forge_stale_lock
    log="$LOCK.raceb.log"
    race 40 b "$log"
    chk "trial $trial: exactly one winner" "$(grep -c 'outcome=acquired' "$log")" "1"
    chk "trial $trial: reaped exactly once" "$(grep -c 'reaping stale' "$log")" "1"
    rm -f "$log"
done

echo "== crash recovery =="
forge_stale_lock
chk "dead holder reported STALE" "$(st state)" "STALE"
$HL acquire --owner roy --ttl 600 >/dev/null 2>&1
chk "dead holder is reaped and lock acquired" "$(st owner)" "roy"
cleanup

echo "== an unreadable lock degrades to busy, not free =="
# A lock written by a different version of this script must not be mistaken
# for an abandoned one.
cleanup
mkdir -p "$LOCK"
: >"$LOCK/meta"
rc=0
$HL acquire --owner leon --ttl 600 >/dev/null 2>&1 || rc=$?
chk "empty metadata is treated as held" "$rc" "2"
cleanup

echo "== ttl expiry =="
cleanup
$HL acquire --owner leon --ttl 1 >/dev/null 2>&1
sleep 2
chk "expired lock is reported EXPIRED" "$(st state)" "EXPIRED"
out=$($HL acquire --owner roy --ttl 600 2>&1)
chk "expired lock is taken over" "$(st owner)" "roy"
chk "takeover of a live holder warns loudly" "$(echo "$out" | grep -c 'WARNING')" "2"
cleanup

echo "== run: releases on every exit path =="
$HL run --owner leon -- true >/dev/null 2>&1
chk "released after success" "$(st state)" "FREE"

rc=0
$HL run --owner leon -- bash -c 'exit 42' >/dev/null 2>&1 || rc=$?
chk "propagates the command's exit code" "$rc" "42"
chk "released after failure" "$(st state)" "FREE"

$HL run --owner leon --reason "long bench" -- sleep 60 >/dev/null 2>&1 &
runner=$!
sleep 2
chk "held while the command runs" "$(st state)" "HELD"
sig "$runner" 15
chk "runner terminates on SIGTERM" "$(wait_bounded "$runner" && echo yes || echo no)" "yes"
sleep 1
chk "released after SIGTERM" "$(st state)" "FREE"

# SIGKILL cannot be trapped, so the lock survives; the anchor is the runner's
# own pid, so the next acquirer must reap it. This is the case the pid anchor
# exists for.
$HL run --owner leon --reason "hard kill" -- sleep 60 >/dev/null 2>&1 &
runner=$!
sleep 2
sig "$runner" 9
wait_bounded "$runner" >/dev/null 2>&1
sleep 1
chk "SIGKILL leaves the lock behind" "$(st state)" "STALE"
$HL acquire --owner roy --ttl 600 >/dev/null 2>&1
chk "next acquirer reaps the killed holder" "$(st owner)" "roy"
cleanup

echo "== run does not clobber a successor's lock =="
# Review of #1806 demonstrated this one: `run` used to tear the lock down
# unconditionally, so if its TTL expired mid-command and somebody else
# legitimately took over, finishing the command deleted THEIR live lock and
# the next acquirer was handed a host two people were already using.
cleanup
$HL run --owner leon --ttl 2 -- sleep 8 >/dev/null 2>&1 &
runner=$!
sleep 4
$HL acquire --owner roy --ttl 600 >/dev/null 2>&1
chk "successor takes over the expired run" "$(st owner)" "roy"
wait_bounded "$runner" >/dev/null 2>&1
chk "finished run leaves the successor holding it" "$(st state)" "HELD"
chk "finished run did not steal the lock from the successor" "$(st owner)" "roy"
cleanup

echo "== a leaked reaper guard does not wedge stale recovery =="
# The reaper guard is the one directory with no anchor and no owner, so a
# crash inside its critical section would make every dead lock permanently
# un-reapable and every acquirer permanently BUSY.
forge_stale_lock
mkdir -p "$LOCK.reaper"
touch -d '10 minutes ago' "$LOCK.reaper"
$HL acquire --owner roy --ttl 600 >/dev/null 2>&1
chk "an abandoned reaper guard is cleared" "$(st owner)" "roy"
cleanup

echo "== corrupt metadata fails safe =="
cleanup
mkdir -p "$LOCK"
printf 'anchor_pid=%s\nstart_time=%s\nowner=leon\nreason=r\nacquired_epoch=NOTANUMBER\nttl=ALSONOTANUMBER\nrunnable_at_acquire=1\n' \
    "$PPID" "$(sed 's/.*) //' "/proc/$PPID/stat" | awk '{print $20}')" >"$LOCK/meta"
err=$($HL status 2>&1 >/dev/null)
chk "corrupt ttl/epoch produce no shell errors" "$(echo -n "$err" | wc -c)" "0"
chk "corrupt metadata is treated as held, not expired" "$(st state)" "HELD"
cleanup

echo "== interrupting a run releases promptly, and stops the command =="
# The property: release must not wait for the wrapped command to finish on
# its own. With `"$@"` inline, bash defers the trap until the foreground
# command completes, so signalling a `sleep 60` released only after the full
# 60s -- and the old unbounded test PASSED by simply blocking that long.
cleanup
$HL run --owner leon --reason "prompt" -- sleep 47 >/dev/null 2>&1 &
runner=$!
sleep 2
# Resolve the wrapped command's real pid and check THAT.
#
# `pgrep -f 'sleep 47'` cannot be used here: it matches this very test
# script's own command line, which contains the string, so it reported the
# command as still running no matter what. It also matched the orphaned
# `sleep` left behind by the SIGKILL case above, which by design outlives its
# wrapper. Neither has anything to do with the property under test.
wrapped=$(pgrep -P "$runner" 2>/dev/null | head -1)
started=$(date +%s)
sig "$runner" 15
wait_bounded "$runner" 30 >/dev/null 2>&1
elapsed=$(($(date +%s) - started))
chk "released within 10s of the signal, not after the command" \
    "$([ "$elapsed" -lt 10 ] && echo yes || echo "no (${elapsed}s)")" "yes"
chk "prompt release actually freed the lock" "$(st state)" "FREE"
chk "the wrapped command was stopped too" \
    "$(alive "$wrapped" && echo running || echo stopped)" "stopped"
cleanup

echo "== SIGHUP releases =="
cleanup
$HL run --owner leon --reason "hup" -- sleep 60 >/dev/null 2>&1 &
runner=$!
sleep 2
sig "$runner" 1
chk "runner terminates on SIGHUP" "$(wait_bounded "$runner" && echo yes || echo no)" "yes"
sleep 1
chk "released after SIGHUP" "$(st state)" "FREE"
cleanup

echo "== wait =="
cleanup
chk "wait returns immediately on a free host" "$($HL wait --timeout 5 >/dev/null 2>&1; echo $?)" "0"
$HL acquire --owner leon --ttl 600 >/dev/null 2>&1
chk "wait times out (3) while held" "$($HL wait --timeout 6 >/dev/null 2>&1; echo $?)" "3"
cleanup
$HL acquire --owner leon --ttl 1 >/dev/null 2>&1
sleep 2
chk "wait does not block on an expired lock" "$($HL wait --timeout 5 >/dev/null 2>&1; echo $?)" "0"
cleanup

echo "== gate =="
cleanup
$HL acquire --owner leon --gate 10000 --ttl 600 >/dev/null 2>&1
chk "a trivially satisfied gate does not block" "$(st state)" "HELD"
chk "a satisfied gate is recorded" "$($HL provenance | sed -n 's/^gate=//p' | cut -d: -f1)" "satisfied"
cleanup

# The gate must FAIL CLOSED. A gate that warns and proceeds returns success
# for a satisfied precondition and for an abandoned one, so every row emitted
# afterwards is labelled gated either way -- it converts contamination into a
# label, which is worse than not gating. This is a real defect observed in a
# hand-rolled gate on this host, whose `for ... sleep 10; done; return 0`
# proceeded on expiry.
rc=$($HL acquire --owner leon --gate 0 --gate-timeout 3 --ttl 600 >/dev/null 2>&1; echo $?)
chk "an unsatisfiable gate fails closed (5)" "$rc" "5"
chk "a failed gate leaves no lock behind" "$(st state)" "FREE"
cleanup

# NOT `out=$(...)`: command substitution runs the acquirer in a subshell that
# exits immediately, so $PPID -- the liveness anchor -- is dead on arrival and
# the lock reads STALE rather than HELD. Redirect to a file so the anchor is
# this script's own shell, which is what a real caller has.
$HL acquire --owner leon --gate 0 --gate-timeout 3 --ttl 600 --on-gate-timeout proceed >"$LOCK.out" 2>&1
chk "proceeding past the gate requires asking for it" "$(grep -c 'gate timed out' "$LOCK.out")" "1"
chk "and still acquires" "$(st state)" "HELD"
rm -f "$LOCK.out"
chk "and is recorded as timed_out_proceeded" "$($HL provenance | sed -n 's/^gate=//p' | cut -d: -f1)" "timed_out_proceeded"
cleanup

chk "--on-gate-timeout rejects a bogus value" \
    "$($HL acquire --owner leon --on-gate-timeout maybe >/dev/null 2>&1; echo $?)" "1"
cleanup

echo "== a reaped acquire is not the same event as a clean one =="
# Reclaiming a lock does not stop the dead holder's benchmark, so "somebody
# died holding this" has to be distinguishable from "the box was free".
cleanup
$HL acquire --owner leon --ttl 600 >/dev/null 2>&1
chk "a clean acquire records takeover=none" "$($HL provenance | sed -n 's/^takeover=//p')" "none"
cleanup

forge_stale_lock
out=$($HL acquire --owner leon --ttl 600 2>&1)
chk "a reaping acquire says so on stdout" "$(echo "$out" | grep -c 'outcome=acquired_after_reap')" "1"
chk "and records which kind of takeover" "$($HL provenance | sed -n 's/^takeover=//p')" "stale_pid"
cleanup

forge_stale_lock
rc=$($HL acquire --owner leon --ttl 600 --strict-reap >/dev/null 2>&1; echo $?)
chk "--strict-reap refuses a takeover (4)" "$rc" "4"
chk "and does not hold the lock it refused" "$(st state)" "FREE"
cleanup

echo "== provenance =="
cleanup
chk "provenance on a free host reports no claim" \
    "$($HL provenance | sed -n 's/^declared=//p')" "no"
$HL acquire --owner leon --reason "moe matrix" --ttl 600 >/dev/null 2>&1
chk "provenance names the holder" "$($HL provenance | sed -n 's/^held_by=//p')" "leon"
chk "provenance carries the reason into the row" \
    "$($HL provenance | sed -n 's/^reason=//p')" "moe matrix"
chk "provenance marks a held lock as declared" \
    "$($HL provenance | sed -n 's/^declared=//p')" "yes"
# No universal threshold is honest, so absent an expectation the answer is
# "unknown" rather than a guess.
chk "contended is unknown without an expectation" \
    "$($HL provenance | sed -n 's/^contended=//p')" "unknown"
chk "contended is answerable when the caller supplies one" \
    "$($HL provenance --expect-runnable 100000 | sed -n 's/^contended=//p')" "no"
chk "and reports contention when the expectation is not met" \
    "$($HL provenance --expect-runnable 0 | sed -n 's/^contended=//p')" "yes"
chk "--oneline emits exactly one line" "$($HL provenance --oneline | wc -l)" "1"
chk "--oneline carries the same fields" \
    "$($HL provenance --oneline | tr ' ' '\n' | grep -c '^held_by=leon$')" "1"
chk "--expect-runnable rejects a non-integer" \
    "$($HL provenance --expect-runnable two >/dev/null 2>&1; echo $?)" "1"
cleanup

echo
echo "passed=${pass} failed=${fail}"
[ "$fail" -eq 0 ]
