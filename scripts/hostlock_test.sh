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

cleanup() { rm -rf "$LOCK" "$LOCK".reaper "$LOCK".stage.* "$LOCK".dead.* "$LOCK".gate 2>/dev/null; }
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
    chk "trial $trial: exactly one winner" "$(grep -c 'acquired by' "$log")" "1"
    rm -f "$log"
done

echo "== RACE B: 40 simultaneous acquirers on a stale lock =="
for trial in 1 2 3; do
    forge_stale_lock
    log="$LOCK.raceb.log"
    race 40 b "$log"
    chk "trial $trial: exactly one winner" "$(grep -c 'acquired by' "$log")" "1"
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
wait "$runner" 2>/dev/null
sleep 1
chk "released after SIGTERM" "$(st state)" "FREE"

# SIGKILL cannot be trapped, so the lock survives; the anchor is the runner's
# own pid, so the next acquirer must reap it. This is the case the pid anchor
# exists for.
$HL run --owner leon --reason "hard kill" -- sleep 60 >/dev/null 2>&1 &
runner=$!
sleep 2
sig "$runner" 9
wait "$runner" 2>/dev/null
sleep 1
chk "SIGKILL leaves the lock behind" "$(st state)" "STALE"
$HL acquire --owner roy --ttl 600 >/dev/null 2>&1
chk "next acquirer reaps the killed holder" "$(st owner)" "roy"
cleanup

echo "== wait =="
cleanup
chk "wait returns immediately on a free host" "$($HL wait --timeout 5 >/dev/null 2>&1; echo $?)" "0"
$HL acquire --owner leon --ttl 600 >/dev/null 2>&1
chk "wait times out (3) while held" "$($HL wait --timeout 6 >/dev/null 2>&1; echo $?)" "3"
cleanup

echo "== gate =="
cleanup
$HL acquire --owner leon --gate 10000 --ttl 600 >/dev/null 2>&1
chk "a trivially satisfied gate does not block" "$(st state)" "HELD"
cleanup
out=$($HL acquire --owner leon --gate 0 --gate-timeout 3 --ttl 600 2>&1)
chk "an unsatisfiable gate proceeds and warns" "$(echo "$out" | grep -c 'gate timed out')" "1"
cleanup

echo
echo "passed=${pass} failed=${fail}"
[ "$fail" -eq 0 ]
