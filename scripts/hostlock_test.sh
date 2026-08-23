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
    chk "trial $trial: exactly one winner" "$(grep -cE 'outcome=acquired( |_after_reap )' "$log")" "1"
    rm -f "$log"
done

echo "== RACE B: 40 simultaneous acquirers on a stale lock =="
for trial in 1 2 3; do
    forge_stale_lock
    log="$LOCK.raceb.log"
    race 40 b "$log"
    chk "trial $trial: exactly one winner" "$(grep -cE 'outcome=acquired( |_after_reap )' "$log")" "1"
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
kids_before=$(pgrep -P "$runner" 2>/dev/null | wc -l)
chk "the wrapped command is a child of the runner" \
    "$([ "${kids_before:-0}" -ge 1 ] && echo yes || echo no)" "yes"
sig "$runner" 15
chk "runner terminates on SIGTERM" "$(wait_bounded "$runner" && echo yes || echo no)" "yes"
sleep 1
chk "released after SIGTERM" "$(st state)" "FREE"
# run_teardown verifies the child's START TIME before signalling, because a
# recycled pid would otherwise be signalled by a process that never started
# it. The verification must not be so strict that it stops terminating the
# real child -- a `sleep 60` still running here means the teardown refused a
# legitimate kill and the wrapped benchmark is now an orphan on the cores.
chk "the wrapped command is really stopped, not left orphaned" \
    "$(pgrep -P "$runner" 2>/dev/null | wc -l)" "0"

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

echo "== a released-then-reacquired lock belongs to its new owner =="
# Both new release paths (gate failure, strict-reap refusal) must release only
# a lock we still own. `remove_lock` unconditionally satisfies "the lock is
# FREE afterwards", so an assertion on FREE has no power over the anti-theft
# guard: a successor has to exist for the guard to be tested at all.
cleanup
sleep 60 &
succ=$!
(
    # alice's TTL lapses during her own gate, so bob legitimately takes over
    # while alice is still waiting -- then alice's gate fails.
    $HL acquire --owner alice --pid $succ --ttl 2 --gate 0 --gate-timeout 12 >/dev/null 2>&1
    echo "alice_rc=$?" >"$LOCK.alice"
) &
alice=$!
sleep 5
$HL acquire --owner bob --wait --timeout 20 --ttl 600 >"$LOCK.bob" 2>&1
chk "a successor can take over a TTL-expired lock" "$(st owner)" "bob"
wait_bounded "$alice" 25
chk "the loser's failing gate returns 5" "$(sed -n 's/^alice_rc=//p' "$LOCK.alice")" "5"
chk "and does NOT release the successor's lock" "$(st state)" "HELD"
chk "which is still the successor's" "$(st owner)" "bob"
sig "$succ" 9
wait "$succ" 2>/dev/null
rm -f "$LOCK.alice" "$LOCK.bob"
cleanup

# The same shape, but with the loser PROCEEDING past its gate, so it reaches
# meta_set while a successor owns the lock. Without the anchor guard the loser
# stamps its own abandoned-gate result onto the successor's live lock, and the
# successor's rows then carry a gate outcome from somebody else's run.
sleep 60 &
succ=$!
(
    $HL acquire --owner alice --pid $succ --ttl 2 --gate 0 --gate-timeout 12 \
        --on-gate-timeout proceed >/dev/null 2>&1
    echo "alice_rc=$?" >"$LOCK.alice"
) &
alice=$!
sleep 5
$HL acquire --owner bob --wait --timeout 20 --ttl 600 >/dev/null 2>&1
chk "successor holds it while the loser is still gating" "$(st owner)" "bob"
wait_bounded "$alice" 25
chk "the loser proceeds past its own gate (0)" "$(sed -n 's/^alice_rc=//p' "$LOCK.alice")" "0"
chk "but cannot stamp its gate result on the successor's lock" \
    "$($HL provenance | sed -n 's/^gate=//p')" "none"
chk "and the successor still owns it" "$(st owner)" "bob"
sig "$succ" 9
wait "$succ" 2>/dev/null
rm -f "$LOCK.alice"
cleanup

# The anchor guard is NOT what makes meta_set safe against a concurrent reap:
# staging inside $LOCK_DIR is. remove_lock renames the whole directory away,
# so a staged file inside it goes with the corpse and the final mv fails
# harmlessly. Every other temp path in the script is a sibling of $LOCK_DIR,
# so "normalising" this one for consistency is a plausible future edit -- and
# it would land our metadata on the successor's live lock, which is theft plus
# a leak, since their remove_lock_if_mine would then refuse to release. That
# safety is positional and invisible, so assert it structurally.
# shellcheck disable=SC2016  # matching the literal source text, not expanding it
chk "meta_set stages inside the lock directory, not beside it" \
    "$(sed -n '/^meta_set()/,/^}/p' "$HL" | grep -c 'tmp="\${LOCK_DIR}/')" "1"

echo "== strict-reap =="
# The refusal must beat the gate. The other order sits on the very lock it was
# told to refuse for the whole gate timeout, blocking everyone, and then
# reports the gate failure instead -- and the dead holder's orphaned load is
# the likeliest reason that gate could not be met in the first place.
forge_stale_lock
out=$($HL acquire --owner rv --ttl 600 --strict-reap --gate 0 --gate-timeout 20 2>/dev/null; echo "rc=$?")
chk "strict-reap beats the gate (4, not 5)" "$(echo "$out" | sed -n 's/^rc=//p')" "4"
chk "and does not linger on the lock it refused" "$(st state)" "FREE"
# The machine-readable channel must not report a success for a refused
# acquire: a harness grepping outcome= is the consumer this PR exists for.
chk "stdout says reap_refused, not acquired" \
    "$(echo "$out" | grep -c 'outcome=reap_refused (stale_pid)')" "1"
chk "and never says acquired_after_reap" \
    "$(echo "$out" | grep -c 'outcome=acquired_after_reap')" "0"
cleanup

# TTL expiry is the other takeover kind, and was asserted nowhere.
sleep 60 &
victim=$!
$HL acquire --owner alice --pid $victim --ttl 1 >/dev/null 2>&1
sleep 2
out=$($HL acquire --owner leon --ttl 600 2>&1; echo "rc=$?")
chk "a TTL takeover is reported as its own kind" \
    "$(echo "$out" | grep -c 'outcome=acquired_after_reap (ttl_expired)')" "1"
chk "and recorded as ttl_expired" "$($HL provenance | sed -n 's/^takeover=//p')" "ttl_expired"
sig "$victim" 9
wait "$victim" 2>/dev/null
cleanup

sleep 60 &
victim=$!
$HL acquire --owner alice --pid $victim --ttl 1 >/dev/null 2>&1
sleep 2
chk "strict-reap also refuses a TTL takeover" \
    "$($HL acquire --owner leon --ttl 600 --strict-reap >/dev/null 2>&1; echo $?)" "4"
sig "$victim" 9
wait "$victim" 2>/dev/null
cleanup

echo "== metadata is only writable by the lock's own holder =="
# meta_set's anchor guard: a peer must not be able to stamp its own takeover
# or gate result onto a lock somebody else holds.
cleanup
sleep 60 &
holder=$!
$HL acquire --owner alice --pid $holder --ttl 600 >/dev/null 2>&1
chk "the holder's gate result is recorded" "$($HL provenance | sed -n 's/^gate=//p')" "none"
# A second acquirer with a different anchor loses the mkdir and must leave the
# metadata alone; assert the owner survives a failed acquire attempt.
$HL acquire --owner mallory --ttl 600 >/dev/null 2>&1
chk "a losing acquirer does not rewrite the holder's owner" "$(st owner)" "alice"
chk "nor its takeover field" "$($HL provenance | sed -n 's/^takeover=//p')" "none"
sig "$holder" 9
wait "$holder" 2>/dev/null
cleanup

echo "== value-taking options require values =="
# `${2:-default}` + a silently-failing `shift 2` spins this loop on one core
# forever, on the box whose contention the script exists to control.
for flag in --gate --gate-timeout --ttl --timeout --pid --owner --reason --on-gate-timeout --expect-runnable; do
    rc=$(timeout 5 $HL acquire --owner leon "$flag" >/dev/null 2>&1; echo $?)
    chk "$flag with no value fails fast" "$rc" "1"
done
chk "--gate rejects a non-integer" \
    "$(timeout 5 $HL acquire --owner leon --gate abc >/dev/null 2>&1; echo $?)" "1"
chk "--gate-timeout rejects a non-integer" \
    "$(timeout 5 $HL acquire --owner leon --gate 0 --gate-timeout abc >/dev/null 2>&1; echo $?)" "1"
cleanup

echo "== provenance does not assert facts it does not have =="
# A lock written by an older revision of this script has no takeover/gate
# keys. Reporting `none` for them asserts "no takeover, no gate" about a run
# that may have reaped a corpse and abandoned its gate -- the same
# mislabelling this subcommand exists to prevent, relocated into the data.
cleanup
mkdir -p "$LOCK"
printf 'owner=oldagent\nanchor_pid=%s\nstart_time=%s\nacquired_epoch=%s\nttl=600\nreason=oldrun\n' \
    "$$" "$(sed 's/.*) //' "/proc/$$/stat" | awk '{print $20}')" "$(date +%s)" >"$LOCK/meta"
chk "an old lock reports takeover=unknown, not none" \
    "$($HL provenance | sed -n 's/^takeover=//p')" "unknown"
chk "and gate=unknown, not none" "$($HL provenance | sed -n 's/^gate=//p')" "unknown"
cleanup

# A live holder past its TTL is this design's steady state, not an edge case.
sleep 60 &
holder=$!
$HL acquire --owner alice --pid $holder --ttl 1 --reason "long sweep" >/dev/null 2>&1
sleep 2
chk "an expired-but-live lock is still EXPIRED" "$(st state)" "EXPIRED"
chk "and is still declared -- the box IS claimed" \
    "$($HL provenance | sed -n 's/^declared=//p')" "yes"
sig "$holder" 9
wait "$holder" 2>/dev/null
cleanup

# lock_age defaults a missing epoch to 0, making held_secs the current epoch.
mkdir -p "$LOCK"
: >"$LOCK/meta"
chk "an unparseable lock has no age rather than a 56-year one" \
    "$($HL provenance | sed -n 's/^held_secs=//p')" "unknown"
cleanup

echo "== a peer's free text cannot travel in the one-line form =="
# `reason` is unquoted, unterminated among space-separated fields, and written
# by whoever holds this shared fixed-path lock. A two-word reason truncates
# the field; anything shell-active is worse.
cleanup
# SC2016 is the point: the text must reach the metadata unexpanded, so that
# the assertions below are about what the SCRIPT does with it, not the shell.
# shellcheck disable=SC2016
$HL acquire --owner leon --ttl 600 --reason 'gemm $(touch '"$LOCK"'.pwned) sweep' >/dev/null 2>&1
chk "--oneline omits reason entirely" \
    "$($HL provenance --oneline | grep -c 'reason=')" "0"
chk "--oneline is still exactly one line with a multi-word reason" \
    "$($HL provenance --oneline | wc -l)" "1"
# shellcheck disable=SC2016
chk "the multi-line form carries the reason whole" \
    "$($HL provenance | sed -n 's/^reason=//p')" 'gemm $(touch '"$LOCK"'.pwned) sweep'
chk "and nothing executed it" "$([ -e "$LOCK.pwned" ] && echo yes || echo no)" "no"
cleanup

echo "== the no-argument help covers the options it documents =="
# `sed -n '3,50p'` silently stopped covering the header as it grew, so the
# flags added by a change were never in the help text of that change.
help=$($HL 2>&1)
for flag in --strict-reap --on-gate-timeout --expect-runnable --oneline --ttl --pid; do
    chk "help mentions $flag" "$(echo "$help" | grep -c -- "$flag" | head -1 | awk '{print ($1>0)?1:0}')" "1"
done

echo
echo "== REQUIREMENTS CONFORMANCE =="
# The seven properties this lock exists to guarantee, each asserted directly
# and named after the requirement rather than after the implementation. Three
# of them (R1, R6, R7) had no test at all before this block: they were
# properties of the design that nothing would have noticed the loss of.

echo "-- R1: the holder is the outer harness, spanning every arm --"
# The collision this encodes: an observer ran `ps`, saw the benchmark gone,
# concluded the host was clear, and started work -- but that process had
# exited only because the harness had advanced from one arm of an interleaved
# A/B to the next.
#
# So the test has to be two harnesses that differ in EXACTLY one thing -- which
# pid the lock is anchored to -- whose arms exit NORMALLY, sampled in the gap
# between arms. An earlier version of this block passed `--pid` explicitly and
# asserted only that the lock stayed HELD; it was inert (it passed unchanged
# with the arms deleted entirely) because it tested `--pid` rather than
# anchoring, and no edit to the script could have failed it.
cleanup
rm -f "$LOCK.gap"

# (a) The REJECTED design: anchored to the arm. Arm 1 ends by itself.
bash -c 'sleep 3' &
arm=$!
$HL acquire --owner per-arm --pid "$arm" --ttl 600 >/dev/null 2>&1
gap_child_during=$(st state)
wait "$arm" 2>/dev/null
gap_child=$(st state)
cleanup

# (b) The SHIPPED design: anchored to the harness that spans both arms.
(
    $HL acquire --owner harness --pid "$BASHPID" --ttl 600 >/dev/null 2>&1
    bash -c 'sleep 1' &
    a=$!
    wait "$a"
    $HL status --porcelain | sed -n 's/^state=//p' >"$LOCK.gap"
    $HL provenance --oneline | tr ' ' '\n' | sed -n 's/^held_by=//p' >>"$LOCK.gap"
    bash -c 'sleep 1' &
    a=$!
    wait "$a"
    $HL release >/dev/null 2>&1
)
gap_harness=$(sed -n 1p "$LOCK.gap")
owner_harness=$(sed -n 2p "$LOCK.gap")
rm -f "$LOCK.gap"

chk "a per-arm anchor holds the box while its arm runs" "$gap_child_during" "HELD"
chk "and LOSES it the moment that arm ends normally -- the hole" "$gap_child" "STALE"
chk "a harness anchor still holds it in the same gap" "$gap_harness" "HELD"
chk "and the row emitted in the gap names the harness" "$owner_harness" "harness"
# Without this the two halves could agree and the block would assert nothing.
chk "the two anchorings genuinely disagree" \
    "$([ "$gap_child" != "$gap_harness" ] && echo differ || echo same)" "differ"
cleanup

echo "-- R2: emitted rows carry identity, state and occupancy --"
cleanup
$HL acquire --owner leon --ttl 600 --reason "moe matrix" >/dev/null 2>&1
row=$($HL provenance --oneline --expect-runnable 100000)
for field in hostlock_state declared held_by held_uid held_pid held_secs takeover gate runnable_at_acquire runnable contended sampled_at; do
    chk "the row carries $field" \
        "$(echo "$row" | tr ' ' '\n' | grep -c "^${field}=")" "1"
done
chk "the row's state is the real state" \
    "$(echo "$row" | tr ' ' '\n' | sed -n 's/^hostlock_state=//p')" "HELD"
chk "the row's owner is the real owner" \
    "$(echo "$row" | tr ' ' '\n' | sed -n 's/^held_by=//p')" "leon"
# `held_by` is self-declared free text: `--owner roy` from my shell produces a
# row that says roy. The only identity the kernel vouches for is the anchor's
# uid, so a published row must carry that too, and it must be the REAL uid --
# not an echo of the declared string.
chk "identity is corroborated against the kernel, not just declared" \
    "$(echo "$row" | tr ' ' '\n' | sed -n 's/^held_uid=//p')" "$(id -u)"
chk "occupancy is a number, not a label" \
    "$(echo "$row" | tr ' ' '\n' | sed -n 's/^runnable=//p' | grep -c '^[0-9][0-9]*$')" "1"
# The row must be able to bracket the measured window, not just report the
# instant it was written -- provenance runs AFTER the measurement, so a single
# sample describes a moment when the window has already closed.
chk "the row brackets the window with occupancy at acquire" \
    "$(echo "$row" | tr ' ' '\n' | sed -n 's/^runnable_at_acquire=//p' | grep -c '^[0-9][0-9]*$')" "1"

# The assertion with actual power. "Is a number" is satisfied by a hardcoded
# constant -- and a constant would not merely corrupt this column, it would
# silently satisfy every `--gate N` for N>=1 on a saturated box, disabling the
# ONLY mechanism that sees load from agents who never took the lock. That is
# the 8.6x corruption this tool exists to prevent, wearing a green stamp.
#
# Bounded deliberately: 6 of 32 logical CPUs for ~2s, which is below the
# threshold the host protocol asks to be announced.
spin() {
    local end=$((SECONDS + 3))
    while [ "$SECONDS" -lt "$end" ]; do :; done
}
spinners=""
for _ in 1 2 3 4 5 6; do
    spin &
    spinners="$spinners $!"
done
sleep 1
busy=$($HL provenance --oneline | tr ' ' '\n' | sed -n 's/^runnable=//p')
for sp in $spinners; do sig "$sp" 9; done
wait 2>/dev/null
chk "occupancy tracks load the test itself created" \
    "$([ "${busy:-0}" -ge 6 ] && echo tracks || echo "constant:${busy}")" "tracks"
cleanup

echo "-- R3: FREE, HELD, STALE and EXPIRED are four distinct states --"
# STALE (holder dead) and EXPIRED (holder alive, TTL lapsed) are the pair that
# must not collapse: one means the box may still be loaded by an orphan, the
# other means somebody is actively using it and has overrun.
cleanup
s_free=$(st state)
sleep 30 &
h=$!
$HL acquire --owner alice --pid $h --ttl 600 >/dev/null 2>&1
s_held=$(st state)
sig "$h" 9
wait "$h" 2>/dev/null
s_stale=$(st state)
cleanup
sleep 30 &
h=$!
$HL acquire --owner alice --pid $h --ttl 1 >/dev/null 2>&1
sleep 2
s_expired=$(st state)
sig "$h" 9
wait "$h" 2>/dev/null
cleanup
chk "the four states are FREE/HELD/STALE/EXPIRED" \
    "${s_free}/${s_held}/${s_stale}/${s_expired}" "FREE/HELD/STALE/EXPIRED"
chk "and all four are distinct" \
    "$(printf '%s\n' "$s_free" "$s_held" "$s_stale" "$s_expired" | sort -u | wc -l)" "4"

echo "-- R4: both expiries fail closed --"
cleanup
# Admission expiry (the gate).
chk "admission expiry fails closed (5)" \
    "$($HL acquire --owner leon --gate 0 --gate-timeout 3 --ttl 600 >/dev/null 2>&1; echo $?)" "5"
chk "and holds nothing afterwards" "$(st state)" "FREE"
cleanup
# Wait expiry, on the acquire path a harness actually uses -- `wait` alone was
# covered, `acquire --wait` was not, and `run` goes through the latter.
# `--timeout 6` costs ~10s, not 6: the deadline is checked at the top of each
# iteration and then `sleep 5`, so it rounds up. Two of those is 20s, and a
# 30s incumbent would leave 10s of margin -- if it expired first the waiter
# would REAP and succeed, flipping both assertions on a loaded box. A suite
# that certifies measurements taken under load must not itself flake under it.
sleep 300 &
h=$!
$HL acquire --owner alice --pid $h --ttl 600 >/dev/null 2>&1
chk "wait expiry on acquire fails closed (3)" \
    "$($HL acquire --owner leon --wait --timeout 6 --ttl 600 >/dev/null 2>&1; echo $?)" "3"
# `st owner` reads the meta and answers identically for HELD and STALE, so on
# its own this passes even if the incumbent died.
chk "and the incumbent still holds it" "$(st owner)" "alice"
chk "and is genuinely still alive, not merely named" "$(st state)" "HELD"
chk "run inherits the same failure, it does not run the command" \
    "$($HL run --owner leon --timeout 6 -- bash -c 'echo RAN' 2>/dev/null | grep -c RAN)" "0"
sig "$h" 9
wait "$h" 2>/dev/null
cleanup

echo "-- R5: reaping compares start time, not just the pid --"
# A recycled pid must not be able to masquerade as a live holder. Forge a lock
# whose anchor pid is THIS script -- alive -- but whose recorded start time is
# wrong, which is exactly what pid reuse looks like.
cleanup
mkdir -p "$LOCK"
printf 'owner=ghost\nanchor_pid=%s\nstart_time=1\nacquired_epoch=%s\nttl=600\nreason=recycled\ntakeover=none\n' \
    "$$" "$(date +%s)" >"$LOCK/meta"
chk "a live pid with the wrong start time is STALE, not HELD" "$(st state)" "STALE"
$HL acquire --owner leon --ttl 600 >/dev/null 2>&1
chk "and is reaped as a stale pid" "$($HL provenance | sed -n 's/^takeover=//p')" "stale_pid"
cleanup
# The converse: the same pid with the RIGHT start time is a live holder and
# must never be reaped.
mkdir -p "$LOCK"
printf 'owner=ghost\nanchor_pid=%s\nstart_time=%s\nacquired_epoch=%s\nttl=600\nreason=live\ntakeover=none\n' \
    "$$" "$(sed 's/.*) //' "/proc/$$/stat" | awk '{print $20}')" "$(date +%s)" >"$LOCK/meta"
chk "the same pid with the right start time is HELD" "$(st state)" "HELD"
chk "and is not reapable" "$($HL acquire --owner leon --ttl 600 >/dev/null 2>&1; echo $?)" "2"
cleanup
# "Cannot verify liveness" must not be read as "the holder is dead". A meta
# with content but no start_time is exactly the older-or-newer-version case
# the grace window exists for, and the grace used to apply only to a ZERO-BYTE
# meta -- so this fell through to `! holder_alive` and a LIVE holder's box was
# taken. Agents on this host run different revisions of this script side by
# side, so it is a live risk rather than a hypothetical.
sleep 300 &
h=$!
mkdir -p "$LOCK"
printf 'owner=alice\nanchor_pid=%s\nacquired_epoch=%s\nttl=3600\nreason=live run\n' \
    "$h" "$(date +%s)" >"$LOCK/meta"
chk "a live holder with no start_time keeps its box" \
    "$($HL acquire --owner leon --ttl 600 >/dev/null 2>&1; echo $?)" "2"
chk "and is still the owner afterwards" "$(st owner)" "alice"
# The label has to agree with the behaviour: reporting STALE for a lock the
# tool refuses to reap tells a human the box is abandoned, and they take it.
chk "and is not labelled STALE while being treated as busy" "$(st state)" "HELD"
sig "$h" 9
wait "$h" 2>/dev/null
cleanup

echo "-- R6: a SIGKILLed holder is recoverable --"
# Not a forged lock: a real holder, really SIGKILLed, which is the failure the
# design promises to survive because it is the one signal it cannot trap.
cleanup
sleep 300 &
holder=$!
$HL acquire --owner alice --pid $holder --ttl 600 --reason "killed mid-run" >/dev/null 2>&1
chk "held while the anchor lives" "$(st state)" "HELD"
sig "$holder" 9
wait "$holder" 2>/dev/null
chk "SIGKILL leaves the lock behind, reported STALE" "$(st state)" "STALE"
chk "the next acquirer recovers the box" \
    "$($HL acquire --owner leon --ttl 600 >/dev/null 2>&1; echo $?)" "0"
chk "and is told it was a takeover, not a free box" \
    "$($HL provenance | sed -n 's/^takeover=//p')" "stale_pid"
cleanup

# The case that actually wedges the box, and the reason the block above is not
# sufficient on its own: `wait` there reaps the zombie, and only THEN does
# /proc/<pid> disappear -- so it measures bash's reaping, not the lock's
# recovery. A SIGKILLed process whose parent has not wait()ed is a ZOMBIE:
# /proc/<pid> still exists and its start time still matches, so a liveness
# check built from those two facts alone reports HELD on a corpse, forever.
# Every agent harness here launches long commands via Popen without an
# immediate wait(), and `run` sets ttl=0, so there is no expiry escape hatch.
# This test file's own alive() helper has excluded state Z since it was
# written; the implementation had not.
rm -f "$LOCK.zpid"
python3 -c '
import os, subprocess, time, sys
c = subprocess.Popen(["sleep", "300"])
open(sys.argv[1], "w").write(str(c.pid))
time.sleep(6)
c.kill()          # SIGKILL, and deliberately no wait(): the child stays a zombie
time.sleep(90)
' "$LOCK.zpid" &
zparent=$!
for _ in 1 2 3 4 5 6 7 8 9 10; do
    [ -s "$LOCK.zpid" ] && break
    sleep 0.5
done
zchild=$(cat "$LOCK.zpid" 2>/dev/null)
$HL acquire --owner alice --pid "$zchild" --ttl 0 --reason "run form, no expiry" >/dev/null 2>&1
chk "held while the run-form anchor lives" "$(st state)" "HELD"
wait_bounded_z=0
for _ in $(seq 1 20); do
    [ "$(sed 's/.*) //' "/proc/${zchild}/stat" 2>/dev/null | awk '{print $1}')" = Z ] && { wait_bounded_z=1; break; }
    sleep 1
done
chk "the anchor really did become a zombie" "$wait_bounded_z" "1"
chk "a zombie anchor is STALE, not HELD" "$(st state)" "STALE"
chk "and the box is recoverable despite ttl=0" \
    "$($HL acquire --owner leon --ttl 600 >/dev/null 2>&1; echo $?)" "0"
sig "$zparent" 9
wait "$zparent" 2>/dev/null
rm -f "$LOCK.zpid"
cleanup

echo "-- R7: the lock never kills anything it did not start --"
# Reclaiming a lock does not stop the load: the dead holder's benchmark is
# still on the cores. The tempting "fix" is for the reaper to kill it, which
# would make this tool capable of destroying a colleague's forty-minute run on
# the strength of a misparsed pid. It must never do that -- it warns instead.
cleanup
# Named to look like a benchmark, so a reaper that "helpfully" pattern-kills
# the dead holder's load (pkill -f) is caught behaviourally too, not only by
# the structural assertion below.
bash -c 'exec -a onnx-genai-bench-orphan sleep 300' &
orphan=$!
sleep 300 &
holder=$!
$HL acquire --owner alice --pid $holder --ttl 600 >/dev/null 2>&1
sig "$holder" 9
wait "$holder" 2>/dev/null
$HL acquire --owner leon --ttl 600 >/dev/null 2>&1
chk "reaping leaves the dead holder's orphaned load running" \
    "$(alive "$orphan" && echo alive || echo killed)" "alive"
sig "$orphan" 9
wait "$orphan" 2>/dev/null
cleanup
# Structural, because the behavioural test above can only cover the reap path.
# `grep -c '^[^#]*\bkill\b'` had three independent holes, each verified to
# leave the suite green: \b does not match pkill/killall/killpg (no word
# boundary between p and k); -c counts LINES, so a second kill appended to the
# sanctioned line was invisible; and ^[^#]* cannot cross a `#`, so a kill
# after any `#` on a line -- including one inside a string -- was hidden, a
# false-PASS direction. Strip comments first, count occurrences not lines, and
# match the whole family. Comments are stripped only at `#` that begins a word,
# so parameter expansions like ${x##* } are not mistaken for comments.
kills=$(sed -E 's/(^|[[:space:]])#.*//' "$HL" | grep -oE '\b(p?kill(all)?|killpg)\b' | wc -l)
chk "exactly one call in the kill family anywhere in the script" "$kills" "1"
# shellcheck disable=SC2016  # matching source text literally, not expanding it
chk "and it targets the command the script itself started" \
    "$(grep -c 'kill -TERM "\$child"' "$HL")" "1"

echo
echo "passed=${pass} failed=${fail}"
[ "$fail" -eq 0 ]
