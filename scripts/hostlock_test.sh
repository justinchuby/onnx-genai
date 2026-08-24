#!/usr/bin/env bash
# hostlock_test.sh — self-test for hostlock.sh
#
# Run: scripts/hostlock_test.sh
#
# Almost free: it is mkdir/rename traffic and short sleeps. There are two
# exceptions, both deliberate and both unavoidable, because each one exists to
# prove that a reported number tracks real load rather than being a constant:
# R2 starts 6 spinners and kills them once occupancy has been read, about a
# second later, and R2b runs three ~1.5s single-cpu cells. That is roughly
# ten core-seconds in total -- 6 x ~1s plus 3 x ~1.5s -- and it is not zero.
# (The spinners are written with a 5s cap they never reach; the cap is a
# deadman, not the cost.) Without them, `runnable_now() { echo 1; }` silently
# disables every --gate and a constant efficiency silently certifies every
# run. Do not run this during a benchmark you intend to publish, and take the
# real lock first, exactly as this tool asks everyone else to.
#
# EVERY ASSERTION IN THIS FILE MUST HOLD ON A CO-TENANTED HOST. Where a cell
# needs to know what a run achieved, it measures this host and derives its
# threshold, rather than asserting a number that only a quiet box can produce
# (see R2b). Verified by re-running those cells with a competitor pinned to
# the same cpu: the old fixed 0.8 threshold reported `contended` at a
# measured efficiency of 0.503, i.e. the suite would have failed for the
# reason it exists to detect in others.
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

cleanup() { rm -rf "$LOCK" "$LOCK".reaper "$LOCK".reaper.stage.* "$LOCK".reaper.dead.* "$LOCK".reaper.rel.* "$LOCK".dead.* "$LOCK".stage.* "$LOCK".gate "$LOCK".warn "$LOCK".zombie.* "$LOCK".zpid "$LOCK".ran 2>/dev/null; }
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

# Spawn a REAL zombie and echo "<zombie_pid> <keeper_pid>".
#
# A pid that is present-and-dead cannot be produced by killing a child of this
# shell: bash reaps it, /proc/<pid> disappears, and what is left is an ABSENT
# pid, which exercises a different branch. The only way to hold a corpse
# visible in /proc is a parent that deliberately never wait()s -- which is
# also exactly how every agent harness here launches work (Popen, no wait).
# The caller must kill the keeper when finished.
spawn_zombie() {
    local f="$LOCK.zombie.$$" z="" par
    rm -f "$f"
    # >/dev/null is load-bearing, not tidiness: this runs inside $(...), and a
    # background child that inherits the substitution's stdout keeps the pipe
    # open, so the caller blocks until the keeper exits -- by which time init
    # has reaped the corpse and the fixture is an ABSENT pid, silently testing
    # the other branch.
    python3 -c '
import subprocess, sys, time
c = subprocess.Popen(["sleep", "300"])
open(sys.argv[1], "w").write(str(c.pid))
c.kill()          # SIGKILL, and deliberately no wait(): the child stays a zombie
time.sleep(120)
' "$f" >/dev/null 2>&1 &
    par=$!
    for _ in $(seq 1 20); do
        [ -s "$f" ] && { z=$(cat "$f"); break; }
        sleep 0.5
    done
    for _ in $(seq 1 20); do
        [ "$(sed 's/.*) //' "/proc/${z}/stat" 2>/dev/null | awk '{print $1}')" = Z ] && break
        sleep 0.5
    done
    rm -f "$f"
    echo "$z $par"
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
$HL run --owner leon --reason "run releases" -- true >/dev/null 2>&1
chk "released after success" "$(st state)" "FREE"

rc=0
$HL run --owner leon --reason "run propagates status" -- bash -c 'exit 42' >/dev/null 2>&1 || rc=$?
chk "propagates the command's exit code" "$rc" "42"
chk "released after failure" "$(st state)" "FREE"

$HL run --owner leon --reason "long bench" -- sleep 60 >/dev/null 2>&1 &
runner=$!
sleep 2
chk "held while the command runs" "$(st state)" "HELD"
kids_before=$(pgrep -P "$runner" 2>/dev/null | wc -l)
chk "the wrapped command is a child of the runner" \
    "$([ "${kids_before:-0}" -ge 1 ] && echo yes || echo no)" "yes"
# Capture the wrapped pid BEFORE signalling. Asking `pgrep -P "$runner"` after
# the runner has exited and been reaped answers 0 whether or not the child was
# terminated, because an orphan is reparented to pid 1 -- the assertion would
# be inert, passing with the teardown kill removed entirely.
wrapped_kid=$(pgrep -P "$runner" 2>/dev/null | head -1)
chk "and the test actually found it" \
    "$([ -n "$wrapped_kid" ] && echo found || echo missing)" "found"
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
    "$(alive "$wrapped_kid" && echo orphaned || echo stopped)" "stopped"

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
$HL run --owner leon --reason "ttl holder" --ttl 2 -- sleep 8 >/dev/null 2>&1 &
runner=$!
sleep 4
$HL acquire --owner roy --ttl 600 >/dev/null 2>&1
chk "successor takes over the expired run" "$(st owner)" "roy"
wait_bounded "$runner" >/dev/null 2>&1
chk "finished run leaves the successor holding it" "$(st state)" "HELD"
chk "finished run did not steal the lock from the successor" "$(st owner)" "roy"
cleanup

echo "== a leaked reaper guard does not wedge stale recovery =="
# An UNATTRIBUTABLE guard -- no anchor, no owner -- is the only class the age
# backstop still covers. This script cannot produce one (the guard is
# published populated, by rename), but an older version or a stray mkdir can.
forge_stale_lock
mkdir -p "$LOCK.reaper"
touch -d '10 minutes ago' "$LOCK.reaper"
$HL acquire --owner roy --ttl 600 >/dev/null 2>&1
chk "an abandoned reaper guard is cleared" "$(st owner)" "roy"
cleanup

echo "== R8: the reaper guard cannot wedge the box =="
# The guard providing mutual exclusion for reaping used to be a bare mkdir
# with no anchor and no owner. One SIGKILL between its mkdir and its rmdir
# orphaned a directory nothing could attribute, and from then on every
# genuinely dead lock was un-reapable and every acquirer was told BUSY --
# indistinguishable from legitimate occupancy. These four cells are the
# falsifiers for the fix.

# R8.1 -- KILL IN THE WINDOW, the defect exactly as reported.
#
# Deterministic, not opportunistic: the seam holds the critical section open
# and publishes the pid that is inside it, so the kill lands in the window
# every run rather than when the scheduler cooperates. Nothing here calls
# cleanup between the kill and the recovery -- a cleanup there would sweep
# the orphaned guard and the test would pass against the defect it exists to
# catch, which is how this class of test usually dies.
forge_stale_lock
rm -f "$LOCK.reaper/stalled_pid"
HOSTLOCK_REAPER_STALL=30 $HL acquire --owner victim --ttl 600 >/dev/null 2>&1 &
stall_parent=$!
stalled=""
for _ in $(seq 1 40); do
    [ -s "$LOCK.reaper/stalled_pid" ] && { stalled=$(cat "$LOCK.reaper/stalled_pid"); break; }
    sleep 0.25
done
chk "a reaper is reachable inside its critical section" "$([ -n "$stalled" ] && echo yes)" "yes"
sig "$stalled" 9
wait "$stall_parent" 2>/dev/null
# Assert the ORPHAN EXISTS before asserting recovery. Without this the next
# two checks pass just as well when the guard was never leaked at all, and a
# vacuous conformance test is worse than none: it reports coverage it does
# not have.
chk "killing it in the window really does orphan the guard" "$([ -d "$LOCK.reaper" ] && echo yes)" "yes"
chk "and the orphan still names its dead owner" "$(sed -n 's/^anchor_pid=//p' "$LOCK.reaper/meta" 2>/dev/null)" "$stalled"
$HL acquire --owner roy --ttl 600 >/dev/null 2>&1
chk "a guard orphaned by SIGKILL does not wedge the next acquirer" "$(st owner)" "roy"
chk "and recovery is immediate, not after REAPER_GRACE" "$([ -d "$LOCK.reaper" ] && echo leaked || echo clear)" "clear"
# Checked BEFORE cleanup, deliberately: cleanup sweeps these globs, so an
# assertion placed after it would pass whether or not the reap cycle tidies
# up after itself. Rename-then-remove leaves a `.dead.`/`.rel.` directory
# behind if the second half is ever dropped, and on a shared box that is
# unbounded growth in the lock's own parent directory.
chk "a reap cycle leaves no guard litter behind" \
    "$(find "$(dirname "$LOCK")" -maxdepth 1 -name "$(basename "$LOCK").reaper.*" 2>/dev/null | wc -l)" "0"
cleanup

# R8.2 -- the mirror image: age is NOT evidence of death.
#
# The previous rule cleared any guard older than REAPER_GRACE. A reaper that
# is merely slow -- loaded box, qemu, a stalled stat -- would have its guard
# taken while still inside the critical section, which is the double reap the
# guard exists to prevent. Two agents would then both believe they own the
# host, which is worse than a wedge because it is silent.
forge_stale_lock
sleep 120 &
live_holder=$!
mkdir -p "$LOCK.reaper"
printf 'anchor_pid=%s\nstart_time=%s\nowner=slowpoke\nclaimed_epoch=1\n' \
    "$live_holder" "$(sed 's/.*) //' "/proc/$live_holder/stat" | awk '{print $20}')" >"$LOCK.reaper/meta"
touch -d '10 minutes ago' "$LOCK.reaper"
$HL acquire --owner thief --ttl 600 >/dev/null 2>&1
chk "an ancient guard held by a LIVE reaper is not stolen" "$(st owner)" "ghost"
chk "and the live reaper still holds its guard" "$([ -d "$LOCK.reaper" ] && echo yes)" "yes"
sig "$live_holder" 9
wait "$live_holder" 2>/dev/null
cleanup

# R8.3 -- a reclaimed reaper must not delete its successor's guard.
#
# If a guard is reclaimed while its owner is descheduled, that owner wakes up
# holding nothing. Releasing unconditionally would delete a successor's guard
# mid-reap and hand the box to two agents at once -- the same failure as R8.2
# reached from the other side.
forge_stale_lock
rm -f "$LOCK.reaper/stalled_pid"
HOSTLOCK_REAPER_STALL=4 $HL acquire --owner victim --ttl 600 >/dev/null 2>&1 &
stall_parent=$!
stalled=""
for _ in $(seq 1 40); do
    [ -s "$LOCK.reaper/stalled_pid" ] && { stalled=$(cat "$LOCK.reaper/stalled_pid"); break; }
    sleep 0.25
done
chk "a second reaper is reachable inside its critical section" "$([ -n "$stalled" ] && echo yes)" "yes"
sleep 120 &
successor=$!
rm -rf "$LOCK.reaper"
mkdir -p "$LOCK.reaper"
printf 'anchor_pid=%s\nstart_time=%s\nowner=successor\nclaimed_epoch=1\n' \
    "$successor" "$(sed 's/.*) //' "/proc/$successor/stat" | awk '{print $20}')" >"$LOCK.reaper/meta"
wait "$stall_parent" 2>/dev/null
chk "a reaper whose guard was reclaimed leaves the successor's guard alone" \
    "$(sed -n 's/^anchor_pid=//p' "$LOCK.reaper/meta" 2>/dev/null)" "$successor"
sig "$successor" 9
wait "$successor" 2>/dev/null
cleanup

# R8.3b -- and "mine" means pid AND start time.
#
# ~1.5M pids in four days on this box, so a successor landing on our number
# is a recycled pid, not us. A pid-only ownership check deletes that
# successor's guard; the same forged-successor shape as above, with the
# stalled reaper's own pid and a start time that is not its, is the
# falsifier.
forge_stale_lock
rm -f "$LOCK.reaper/stalled_pid"
HOSTLOCK_REAPER_STALL=4 $HL acquire --owner victim --ttl 600 >/dev/null 2>&1 &
stall_parent=$!
stalled=""
for _ in $(seq 1 40); do
    [ -s "$LOCK.reaper/stalled_pid" ] && { stalled=$(cat "$LOCK.reaper/stalled_pid"); break; }
    sleep 0.25
done
chk "a third reaper is reachable inside its critical section" "$([ -n "$stalled" ] && echo yes)" "yes"
rm -rf "$LOCK.reaper"
mkdir -p "$LOCK.reaper"
printf 'anchor_pid=%s\nstart_time=1\nowner=recycled\nclaimed_epoch=1\n' "$stalled" >"$LOCK.reaper/meta"
wait "$stall_parent" 2>/dev/null
chk "a guard on our pid but not our start time is somebody else's" \
    "$(sed -n 's/^owner=//p' "$LOCK.reaper/meta" 2>/dev/null)" "recycled"
cleanup

# R8.5 -- the guard is only ever published complete.
#
# Not reachable through behaviour: killing between a mkdir and the meta write
# is a microsecond window and a seam wide enough to test it would be wider
# than the bug. Asserted structurally instead, which is weaker evidence but
# honest about being weaker: the guard directory must never be created in
# place, only renamed into place already populated, so no kill can leave an
# unattributable guard behind (that class is the one the age backstop covers,
# and it should stay unreachable from this script).
# shellcheck disable=SC2016  # matching source text literally, not expanding it
chk "the guard is never created in place" \
    "$(grep -cE 'mkdir[^#]*\$REAP_DIR"' "$HL")" "0"
# shellcheck disable=SC2016  # matching source text literally, not expanding it
chk "it is renamed into place already populated" \
    "$(grep -cF 'mv -T "$stage" "$REAP_DIR"' "$HL")" "1"

# R8.4 -- the seam itself is inert unless asked for.
#
# A test seam in production code earns its place only if its absence is
# asserted: an `if` that fired by default would stall every real reap.
forge_stale_lock
t0=$(date +%s)
$HL acquire --owner roy --ttl 600 >/dev/null 2>&1
t1=$(date +%s)
# Bounded well under the 30s seam used above, but not so tight that a loaded
# host fails it: this asserts the seam is inert, not that the box is quiet.
chk "the stall seam does nothing when unset" "$([ "$((t1 - t0))" -lt 10 ] && echo fast || echo stalled)" "fast"
chk "and the reap still happened" "$(st owner)" "roy"
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
# The human branch has to say so too, and this is the arm that gets it wrong:
# an EXPIRED holder is by definition ALIVE, so "is gone; next acquire will
# reap it" is the same false-abandonment message one arm over. Read through
# `status` rather than `acquire`, because an expired lock is reapable and
# acquire would simply take it.
exp_text=$($HL status 2>&1 | tr '\n' ' ')
chk "the EXPIRED message does not say the holder is gone" \
    "$(echo "$exp_text" | grep -c 'is gone')" "0"
chk "and it says EXPIRED" "$(echo "$exp_text" | grep -c 'EXPIRED')" "1"
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
#
# Compared against the ANCHOR's uid read independently, not against the
# reader's own `id -u`. Those are the same number when the test acquires as
# itself, so `uid=$(id -u)` in the writer -- reporting whoever READ the lock
# rather than whoever HOLDS it -- passed the earlier form of this check while
# destroying the entire point of the field, which is attribution when you are
# looking at somebody else's lock.
anchor_of_row=$(echo "$row" | tr ' ' '\n' | sed -n 's/^held_pid=//p')
chk "identity is corroborated against the kernel, not just declared" \
    "$(echo "$row" | tr ' ' '\n' | sed -n 's/^held_uid=//p')" \
    "$(awk '/^Uid:/ { print $2; exit }' "/proc/${anchor_of_row}/status" 2>/dev/null)"
# Everything on this box runs as one user, so the check above cannot tell
# "the holder's uid" from "the reader's uid" -- and `uid=$(id -u)` in the
# writer, which reports whoever READ the lock, passes it while destroying the
# only thing the field is for: attribution when you are looking at somebody
# else's lock. Anchor to pid 1 instead, whose uid is derived here rather than
# assumed: hardcoding 0 fails as root (the default in most CI containers) and
# under a rootless or user-namespaced pid 1, for reasons that have nothing to
# do with hostlock. Skipped entirely when pid 1 is unreadable (hidepid=) or is
# us, because then it discriminates nothing.
pid1_uid=$(awk '/^Uid:/ { print $2; exit }' /proc/1/status 2>/dev/null)
if [ -n "$pid1_uid" ] && [ "$pid1_uid" != "$(id -u)" ]; then
    cleanup
    $HL acquire --owner leon --pid 1 --ttl 600 --reason "foreign anchor" >/dev/null 2>&1
    chk "held_uid is the HOLDER's uid, not the reader's" \
        "$($HL provenance --oneline | tr ' ' '\n' | sed -n 's/^held_uid=//p')" "$pid1_uid"
else
    echo "  SKIP  held_uid vs a foreign anchor (pid 1 uid unreadable or same as ours)"
    # A SKIP must not silently reduce the assertion count: in a root container
    # this branch is taken, and with nothing here the only check that pins
    # held_uid to the kernel disappears with no total to notice it. Assert the
    # weaker thing that is still true, so the count stays invariant and the
    # pinned total at the end of this file keeps its power.
    chk "held_uid is at least present and numeric when the foreign anchor is unavailable" \
        "$($HL provenance --oneline | tr ' ' '\n' | grep -c '^held_uid=[0-9]')" "1"
fi
cleanup
$HL acquire --owner leon --ttl 600 --reason "moe matrix" >/dev/null 2>&1
row=$($HL provenance --oneline --expect-runnable 100000)
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
# Bounded deliberately: 6 of 32 logical CPUs for ~3s. This is the ONE part of
# the suite that is not free (see the header note).
spin() {
    local end=$((SECONDS + 5))
    while [ "$SECONDS" -lt "$end" ]; do :; done
}
spinners=""
for _ in 1 2 3 4 5 6; do
    spin &
    spinners="$spinners $!"
done
sleep 1
busy=$($HL provenance --oneline | tr ' ' '\n' | sed -n 's/^runnable=//p')
# Acquire a SECOND lock while the load is still up, so runnable_at_acquire is
# sampled under known load and read back after it has gone. That is the
# bracketing property the field claims, and it is the only assertion here that
# a hardcoded `runnable_at_acquire=1` cannot satisfy.
cleanup
$HL acquire --owner leon --ttl 600 >/dev/null 2>&1
for sp in $spinners; do sig "$sp" 9; done
wait 2>/dev/null
sleep 1
row_after=$($HL provenance --oneline)
acq_busy=$(echo "$row_after" | tr ' ' '\n' | sed -n 's/^runnable_at_acquire=//p')
chk "occupancy tracks load the test itself created" \
    "$([ "${busy:-0}" -ge 6 ] && echo tracks || echo "constant:${busy}")" "tracks"
chk "runnable_at_acquire is sampled while the window is open" \
    "$([ "${acq_busy:-0}" -ge 6 ] && echo bracketed || echo "constant:${acq_busy}")" "bracketed"
# The load-based assertion above cannot distinguish "stored at acquire" from
# "re-read at print time" on a host that is busy for other reasons, and this
# host is. Forge the stored value instead: a reader that re-samples prints the
# real count, a reader that reports what the holder recorded prints 4242.
# Deterministic, and independent of what anyone else is running.
sed -i 's/^runnable_at_acquire=.*/runnable_at_acquire=4242/' "$LOCK/meta"
chk "and it is the holder's stored sample, not a reading taken now" \
    "$($HL provenance --oneline | tr ' ' '\n' | sed -n 's/^runnable_at_acquire=//p')" "4242"
chk "while the live occupancy column is still a live reading" \
    "$($HL provenance --oneline | tr ' ' '\n' | sed -n 's/^runnable=//p' | grep -c '^4242$')" "0"
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
    "$($HL run --owner leon --reason "blocked run" --timeout 6 -- bash -c 'echo RAN' 2>/dev/null | grep -c RAN)" "0"
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
# The human-readable branch has to agree too, and it is the one that matters
# here: cmd_acquire dumps it on the BUSY path, so this text is what a blocked
# agent actually reads. Telling them "is gone; next acquire will reap it"
# about a box the tool refuses to reap is how the box gets taken by hand.
busy_text=$($HL acquire --owner leon --ttl 600 2>&1 >/dev/null | tr '\n' ' ')
chk "the BUSY message does not tell a human the box is abandoned" \
    "$(echo "$busy_text" | grep -c 'is gone')" "0"
chk "and it names the real holder" \
    "$(echo "$busy_text" | grep -c 'HELD by alice')" "1"
# ...and it must STAY its box. The grace is 300s; the runs it protects are 40
# minutes. Routing a live-but-unverifiable holder through an age window does
# not prevent the theft, it schedules it. Backdate the lock well past the
# grace: the anchor pid is readable and running, which is evidence the clock
# cannot override.
touch -d "@$(( $(date +%s) - 4000 ))" "$LOCK"
chk "a live holder with no start_time keeps its box past the grace too" \
    "$($HL acquire --owner leon --ttl 600 >/dev/null 2>&1; echo $?)" "2"
chk "and is still the owner after the grace lapses" "$(st owner)" "alice"
# ...but the anti-theft guard must disable the CLOCK-BASED grace only, not the
# expiry the holder itself declared. Returning "not reapable" unconditionally
# here replaced a bounded wedge with an unbounded one: the lock would outlive
# any ttl forever, and `wait` would block on a lock everyone agrees has
# expired. Guard against a vacuous pass first: if any assertion above has
# already let the lock be reaped, the acquire below succeeds for the wrong
# reason and this -- the primary assertion for the primary fix -- passes
# without exercising anything.
chk "the ttl fixture starts from a lock that is still alice's" "$(st owner)" "alice"
printf 'owner=alice\nanchor_pid=%s\nacquired_epoch=%s\nttl=1\nreason=live but overrun\n' \
    "$h" "$(( $(date +%s) - 4000 ))" >"$LOCK/meta"
# The LABEL for that class has to be right too, and this is the third arm of
# the same defect: HELD said "is gone" (fixed), EXPIRED said "is gone"
# (fixed), and then the reaper announced "reaping stale lock from dead pid N"
# about a pid the guard immediately above had just verified was RUNNING --
# suppressing the "still alive ... both sets of numbers are now suspect"
# warning in exactly the case it applies to. A takeover from a live holder is
# the one event in this tool that can corrupt somebody's benchmark, so it is
# the one message that must never be downgraded to routine housekeeping.
chk "an overrun live holder with no start_time is EXPIRED, not STALE" "$(st state)" "EXPIRED"
exp_text=$($HL status 2>&1 | tr '\n' ' ')
chk "and the human text does not call that live pid gone" \
    "$(echo "$exp_text" | grep -c 'is gone')" "0"
chk "and it says the ttl ran out" \
    "$(echo "$exp_text" | grep -c '^EXPIRED')" "1"
rm -f "$LOCK.warn"
$HL acquire --owner leon --ttl 600 >/dev/null 2>"$LOCK.warn"
chk "a live holder with no start_time still honours its own ttl" "$?" "0"
chk "and the takeover warns the holder is STILL ALIVE" \
    "$(grep -c 'still alive' "$LOCK.warn")" "1"
chk "and warns that both sets of numbers are suspect" \
    "$(grep -c 'both sets of numbers' "$LOCK.warn")" "1"
chk "and never claims it reaped a dead pid" \
    "$(grep -c 'dead pid' "$LOCK.warn")" "0"
rm -f "$LOCK.warn"
sig "$h" 9
wait "$h" 2>/dev/null
cleanup
# The converse of the guard above: an anchor pid that is DEAD, with no
# start_time. Nothing else in this file has that shape -- every other fixture
# either carries a start_time or has no anchor_pid at all -- so deleting the
# liveness test from that guard made every unparseable lock permanently
# unreapable, which is the exact wedge this commit exists to avoid, and left
# the suite green.
#
# dead_pid() is fully reaped, so /proc/<pid> does NOT exist: this is the
# ABSENT-and-dead shape. The present-and-dead shape is a zombie and is tested
# immediately below; both must reach the clock, by different routes.
dead=$(dead_pid)
mkdir -p "$LOCK"
printf 'owner=ghost\nanchor_pid=%s\nacquired_epoch=%s\nttl=3600\nreason=dead anchor, old script\n' \
    "$dead" "$(date +%s)" >"$LOCK/meta"
touch -d "@$(( $(date +%s) - 200 ))" "$LOCK"
chk "a dead anchor with no start_time is still protected inside the grace" \
    "$($HL acquire --owner leon --ttl 600 >/dev/null 2>&1; echo $?)" "2"
touch -d "@$(( $(date +%s) - 400 ))" "$LOCK"
chk "a dead anchor with no start_time is reapable past the grace" \
    "$($HL acquire --owner leon --ttl 600 >/dev/null 2>&1; echo $?)" "0"
chk "and that takeover is labelled unverifiable too" \
    "$($HL provenance | sed -n 's/^takeover=//p')" "unverifiable"
cleanup
# The present-and-dead shape, which is the one that actually wedges the box.
# A ZOMBIE anchor passes `[ -d /proc/$pid ]`, so writing the anti-theft guard
# in terms of the directory rather than in terms of real liveness hands a
# corpse the same protection a live holder gets -- and on the `run` path,
# where ttl=0, holder_expired is never true, so nothing bounds it: the grace
# that would have released the box is disabled on behalf of a dead process,
# forever. That is a strictly worse wedge than the one the guard replaced.
zline=$(spawn_zombie); zapid=${zline%% *}; zakeeper=${zline##* }
chk "the fixture anchor really is a zombie" \
    "$(sed 's/.*) //' "/proc/${zapid}/stat" 2>/dev/null | awk '{print $1}')" "Z"
mkdir -p "$LOCK"
printf 'owner=ghost\nanchor_pid=%s\nacquired_epoch=%s\nttl=0\nreason=zombie anchor, meta caught mid-write\n' \
    "$zapid" "$(date +%s)" >"$LOCK/meta"
touch -d "@$(( $(date +%s) - 200 ))" "$LOCK"
chk "a zombie anchor with no start_time is protected inside the grace" \
    "$($HL acquire --owner leon --ttl 600 >/dev/null 2>&1; echo $?)" "2"
touch -d "@$(( $(date +%s) - 400 ))" "$LOCK"
chk "a zombie anchor with no start_time is reapable past the grace, ttl=0 and all" \
    "$($HL acquire --owner leon --ttl 600 >/dev/null 2>&1; echo $?)" "0"
sig "$zakeeper" 9
wait "$zakeeper" 2>/dev/null
cleanup
# The grace itself must be pinned. Every assertion above creates its lock
# immediately before checking, so now-mtime is 0 and ANY grace >= 1 passes --
# UNPARSEABLE_GRACE could be cut from 300 to 2 with the suite still green.
# Bracketed at 200s and 400s, so the constant is pinned to within a factor of
# two rather than the two orders of magnitude a 30s/4000s pair allows.
mkdir -p "$LOCK"
printf 'owner=ghost\nacquired_epoch=%s\nttl=3600\nreason=unreadable\n' "$(date +%s)" >"$LOCK/meta"
touch -d "@$(( $(date +%s) - 200 ))" "$LOCK"
chk "an unreadable lock inside the grace is held" \
    "$($HL acquire --owner leon --ttl 600 >/dev/null 2>&1; echo $?)" "2"
touch -d "@$(( $(date +%s) - 400 ))" "$LOCK"
chk "an unreadable lock past the grace is reapable" \
    "$($HL acquire --owner leon --ttl 600 >/dev/null 2>&1; echo $?)" "0"
# ...but the takeover must not CLAIM the holder was dead. The script never
# established that; `stale_pid` is a positive claim on evidence it does not
# have, and it is the reassuring direction.
chk "and the takeover is labelled unverifiable, not stale_pid" \
    "$($HL provenance | sed -n 's/^takeover=//p')" "unverifiable"
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
# The STALE arm of the human branch had no assertion at all: replacing its
# whole body with `echo "FREE (nobody is here)"` left the suite green, because
# every STALE check in this file went through porcelain. That is the one arm
# where saying FREE is actively dangerous -- a stale lock still covers a host
# that may be loaded, and "next acquire will reap it" is the sentence that
# tells a human to go through the tool instead of round it.
stale_text=$($HL status 2>&1 | tr '\n' ' ')
# 2>&1 deliberately: this assertion is anchored, so anything the script leaks
# on stderr breaks it. That is how the departed-pid redirection noise was
# found -- `<"/proc/$pid/stat" 2>/dev/null` silences tr but not bash's own
# "No such file or directory", because redirections apply left to right, so
# every status on a stale lock printed two lines of shell error first.
chk "the STALE human text says STALE, with nothing leaked in front of it" \
    "$(echo "$stale_text" | grep -c '^STALE')" "1"
chk "and status writes nothing at all on stderr for a departed pid" \
    "$($HL status 2>&1 >/dev/null | wc -c)" "0"
chk "and names the owner it is about to reap" \
    "$(echo "$stale_text" | grep -c 'holder alice')" "1"
chk "and says a reap is what happens next" \
    "$(echo "$stale_text" | grep -c 'is gone')" "1"
chk "and still reports occupancy, because a stale lock does not stop the load" \
    "$(echo "$stale_text" | tr ' ' '\n' | grep -c '^runnable=[0-9]')" "1"
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

# The converse, and the more dangerous error of the two. State Z is NOT proof
# of death: when a thread-group leader exits via pthread_exit() while other
# threads keep running, /proc/<tgid>/stat reports Z for a fully LIVE process
# (`ps` shows `Zl ... <defunct>`). Reading that as dead reaps a live holder
# mid-benchmark -- one agent stealing another's box, which is the outcome this
# whole tool exists to prevent, whereas the zombie bug above only wedged it.
# Reachable via `acquire --pid <harness>`, which is the flag R1 requires.
rm -f "$LOCK.zlpid"
python3 -c '
import ctypes, os, sys, threading, time
open(sys.argv[1], "w").write(str(os.getpid()))
def worker():
    time.sleep(120)
t = threading.Thread(target=worker, daemon=False)
t.start()
time.sleep(2)
# Exit the thread-group LEADER only. The process stays alive via the worker.
ctypes.CDLL(None).pthread_exit(None)
' "$LOCK.zlpid" &
zlparent=$!
for _ in $(seq 1 20); do
    [ -s "$LOCK.zlpid" ] && break
    sleep 0.5
done
zlpid=$(cat "$LOCK.zlpid" 2>/dev/null)
$HL acquire --owner alice --pid "$zlpid" --ttl 600 --reason "threaded harness" >/dev/null 2>&1
zl_is_z=0
for _ in $(seq 1 20); do
    [ "$(sed 's/.*) //' "/proc/${zlpid}/stat" 2>/dev/null | awk '{print $1}')" = Z ] && { zl_is_z=1; break; }
    sleep 1
done
chk "a live threaded harness can report state Z" "$zl_is_z" "1"
chk "and it still has more than one thread" \
    "$([ "$(awk '/^Threads:/ { print $2; exit }' "/proc/${zlpid}/status" 2>/dev/null || echo 0)" -gt 1 ] \
        && echo threaded || echo single)" "threaded"
chk "a Z leader with live threads keeps its box" "$(st state)" "HELD"
chk "and is not reaped out from under the run" \
    "$($HL acquire --owner leon --ttl 600 >/dev/null 2>&1; echo $?)" "2"
chk "and the owner is unchanged" "$(st owner)" "alice"
# KNOWN GAP, recorded rather than hidden: the "unreadable status means no
# evidence of death, so the lock stands" direction is NOT pinned. Producing a
# process that is Z with a readable /proc/<pid>/stat and an unreadable
# /proc/<pid>/status needs hidepid= or a pid namespace, or losing a
# microsecond race between the two reads; the alternative is a HOSTLOCK_PROC
# test hook in production, which would also be a way to spoof liveness and
# steal a lock. Mutation `[ "${threads:-1}" -le 1 ]` leaves this suite green.
# What makes that acceptable: every ACCIDENTAL form of the regression is safe
# by construction. Dropping the `-n` guard gives `[ "" -le 1 ]`, which exits 2
# ("integer expression expected"), so the `if` is false and the holder is
# still treated as alive. Only writing an explicit numeric default flips the
# direction, and that is a deliberate rewrite rather than a drift.
sig "$zlparent" 9
wait "$zlparent" 2>/dev/null
rm -f "$LOCK.zlpid"
cleanup

echo "-- R2b: a run that did not get its cores must say so --"
# The occupancy snapshot in a published row is an ADMISSION control: it says
# what the host looked like at two instants. Roy gated on exactly that,
# sampled before and after each arm, and it reported "peak 2-4, clean" for
# runs that were in fact getting 50-70% of a core -- a 2s arm has ample room
# for a burst that starts after the opening sample and ends before the closing
# one. His A/A null was 52%: the same binary against itself, disagreeing by
# half. So the row can say `contended=no` about a ruined measurement, and
# nothing in this tool would have contradicted it.
#
# --min-efficiency measures the run itself instead: CPU-seconds actually
# consumed over cores x wall. It needs no quiet host and it is not a proxy.
#
# THESE CELLS MUST PASS ON A BUSY HOST. An assertion of the form "this run
# achieved 0.99 of a core" is an exclusive-host assumption wearing a test's
# clothes: it holds on a quiet box and fails on the shared, co-tenanted one
# that is the actual deployment target -- and a suite that only passes when
# nobody else is working is a suite people learn to re-run until it is green.
# So the acceptance threshold is DERIVED from what this environment actually
# delivered, and the falsifiers that need an absolute number are the ones
# that stay true under any load: a sleeping command consumes no CPU whatever
# the neighbours do, and one core of work judged against two is half
# efficient by construction.
cleanup
busy1s='import time
t = time.time()
while time.time() - t < 1.5:
    pass'
# Measure first, judge second. `run` without a threshold still emits the
# number, which is what makes this possible at all.
out=$($HL run --owner leon --reason "cpu probe" --expect-cores 1 -- python3 -c "$busy1s" 2>&1)
eff_user=$(echo "$out" | sed -n 's/.*efficiency_frac=\([0-9.]*\).*/\1/p' | head -1)
chk "the efficiency of a real run is measured, not labelled" \
    "$(awk -v e="${eff_user:-0}" 'BEGIN { print (e > 0 && e <= 1.05) ? "plausible" : "implausible" }')" \
    "plausible"
thr=$(awk -v e="${eff_user:-0}" 'BEGIN { t = e * 0.9; if (t < 0.05) t = 0.05; printf "%.3f", t }')
out=$($HL run --owner leon --reason "cpu ok" --expect-cores 1 --min-efficiency "$thr" \
    -- python3 -c "$busy1s" 2>&1); rc=$?
chk "a run that got what this host had to give passes the gate" "$rc" "0"
chk "and is reported as ok" "$(echo "$out" | grep -c 'verdict=ok')" "1"
# The number has to be a MEASUREMENT, not a label: a constant 1.000 would pass
# the assertion above forever. A sleeping command consumes no CPU at all, so
# the same threshold must reject it.
out=$($HL run --owner leon --reason "cpu idle" --expect-cores 1 --min-efficiency 0.8 \
    -- sleep 1 2>&1); rc=$?
chk "a run that never touched its core fails the efficiency gate" "$rc" "6"
chk "and is reported as contended" "$(echo "$out" | grep -c 'verdict=contended')" "1"
chk "and says which direction to disbelieve" \
    "$(echo "$out" | grep -c 'treat its numbers as untrusted')" "1"
chk "and names the gate that cannot see it" \
    "$(echo "$out" | grep -c 'samples instants')" "1"
chk "and the lock is released even when the run is rejected" "$(st state)" "FREE"
# Kernel time counts too. Every cell above is a pure user-mode busy loop, so
# `CPU_TICKS=$((cu))` -- dropping cstime -- leaves them all passing while
# under-measuring anything syscall-bound by the whole system half. A real
# benchmark here loads models, reads tensors and writes results, so that is
# not a hypothetical shape. This cell spends most of its time in the kernel.
syscall1s='import os, time
fd = os.open("/dev/null", os.O_WRONLY)
buf = b"x" * 64
t = time.time()
while time.time() - t < 1.5:
    for _ in range(200):
        os.write(fd, buf)'
# Compared against the user-mode probe taken moments ago on this same host,
# not against a fixed 0.8: co-tenancy scales both down together, so the RATIO
# survives a busy box while still collapsing to ~0.23 the moment kernel time
# stops being counted.
out=$($HL run --owner leon --reason "cpu sys" --expect-cores 1 -- python3 -c "$syscall1s" 2>&1)
eff_sys=$(echo "$out" | sed -n 's/.*efficiency_frac=\([0-9.]*\).*/\1/p' | head -1)
chk "a run whose time is mostly kernel time still counts as using its core" \
    "$(awk -v s="${eff_sys:-0}" -v u="${eff_user:-1}" 'BEGIN { print (u > 0 && s / u >= 0.7) ? "counted" : "under-measured" }')" \
    "counted"
out=$($HL run --owner leon --reason "cpu sys judged" --expect-cores 1 --min-efficiency "$thr" \
    -- python3 -c "$syscall1s" 2>&1); rc=$?
chk "and is reported as ok rather than under-measured" \
    "$(echo "$out" | grep -c 'verdict=ok')" "1"
chk "and it passes the same threshold the user-mode run did" "$rc" "0"

# The denominator must be the declared core count, not a constant 1. A
# one-core load judged against two cores is half-efficient by construction,
# so ignoring --expect-cores flips this cell and only this cell.
out=$($HL run --owner leon --reason "cpu half" --expect-cores 2 --min-efficiency 0.8 \
    -- python3 -c "$busy1s" 2>&1); rc=$?
chk "one core of work judged against two cores is contended" "$rc" "6"
chk "and the row records the denominator it used" \
    "$(echo "$out" | grep -c 'cores_expected=2')" "1"
# A failing command's own status is the more important signal. Reporting 6
# for a crashed benchmark buries the crash under a host-quality complaint.
out=$($HL run --owner leon --reason "cpu fail" --expect-cores 1 --min-efficiency 0.8 \
    -- sh -c 'exit 42' 2>&1); rc=$?
chk "a failing command keeps its own exit status" "$rc" "42"
# Without a threshold the measurement is still emitted -- an unjudged number
# in the log is what lets somebody re-examine a suspicious row later -- but
# nothing is enforced and `run`'s documented contract is unchanged.
out=$($HL run --owner leon --reason "cpu unjudged" -- sleep 1 2>&1); rc=$?
chk "without a threshold the run is not judged" "$rc" "0"
chk "but the measurement is still recorded" "$(echo "$out" | grep -c 'verdict=unjudged')" "1"
chk "and it does not invent a denominator it was not given" \
    "$(echo "$out" | grep -c 'cores_expected=unspecified')" "1"
# Fail closed, twice over. A threshold with no denominator cannot be
# evaluated, and defaulting it to 1 would pass every multi-threaded run; a run
# too short to measure cannot be verified either. Both must refuse rather than
# assume, and the first must refuse BEFORE the command runs.
rm -f "$LOCK.ran"
out=$($HL run --owner leon --reason "no denominator" --min-efficiency 0.8 \
    -- touch "$LOCK.ran" 2>&1); rc=$?
chk "a threshold with no denominator is a usage error" "$rc" "1"
chk "and the command never ran" "$([ -e "$LOCK.ran" ] && echo ran || echo no)" "no"
chk "and no lock was left behind" "$(st state)" "FREE"
rm -f "$LOCK.ran"
out=$($HL run --owner leon --reason "too short" --expect-cores 1 --min-efficiency 0.8 \
    -- true 2>&1); rc=$?
chk "a run too short to measure is not certified as clean" "$rc" "6"
chk "and says it could not be evaluated rather than that it passed" \
    "$(echo "$out" | grep -c 'NOT verified')" "1"
# ...and the same two knobs must not be silently accepted where they do
# nothing. An inert knob that is believed in is worse than an absent one.
out=$($HL status --expect-cores 4 2>&1); rc=$?
chk "--expect-cores is refused outside run" "$rc" "1"
out=$($HL acquire --owner leon --min-efficiency 0.5 --expect-cores 1 2>&1); rc=$?
chk "--min-efficiency is refused outside run" "$rc" "1"
chk "and refusing it did not leave a lock" "$(st state)" "FREE"
cleanup

echo "-- R2c: the measurement names its unit, and a bound inside the command is not a ceiling --"
# Three properties, each of which was believed to hold and did not.
#
# (a) `efficiency` was one field name for two quantities: CPU-seconds per wall
#     second without a denominator, and the fraction of --expect-cores held
#     with one. They differ by a factor of N for the SAME run, so a log mixing
#     both is not comparable to itself, and neither reading announces which it
#     is. The field now names its unit.
#
# (b) CPU accounting covers this shell's whole child tree, so a `taskset`
#     applied INSIDE the wrapped command bounds only its own descendants
#     while everything else in the tree is still counted. The measurement is
#     then a superset of what the bound constrains and legitimately exceeds
#     it. This is asserted rather than documented alone because the figure it
#     produces invites the conclusion "affinity leaked" -- an alarm actually
#     raised, and retracted, on this host.
#
# (c) `run` accepted a missing or whitespace-only --reason silently. It works
#     perfectly for the person who started it and tells whoever it blocks
#     nothing, which is the failure mode that does not announce itself.
cleanup
# (a) needs no bound at all: which name is printed is a function of whether a
# denominator was given, not of the value or of how the work was placed. So
# assert the naming unguarded, and spend the taskset-dependent budget on (b).
spin2='for i in 1 2; do ( end=$((SECONDS+2)); while [ $SECONDS -lt $end ]; do :; done ) & done; wait'
out=$($HL run --owner leon --reason "unit naming, undenominated" -- bash -c "$spin2" 2>&1)
chk "without --expect-cores the field is named in cores" \
    "$(echo "$out" | grep -c 'efficiency_cores=')" "1"
# The denominator changes the quantity, not merely its scale -- the same run
# reads N times larger undenominated -- so the two forms must not share a name,
# and the unused one must be absent rather than also printed.
out=$($HL run --owner leon --reason "unit naming, denominated" --expect-cores 1 -- \
    bash -c "$spin2" 2>&1)
chk "with --expect-cores the field is named as a fraction" \
    "$(echo "$out" | grep -c 'efficiency_frac=')" "1"
chk "and the cores-form field is then absent, so neither can be misread" \
    "$(echo "$out" | grep -c 'efficiency_cores=')" "0"

# (b) needs a real bound. Bind to a cpu this process is actually ALLOWED to
# use, read from the kernel rather than assumed: `taskset -c 0` fails outright
# when the suite is itself run under an outer bind (`taskset -c 16-23 ...`),
# which is exactly the placement this PR's docs now tell people to use. A
# hardcoded cpu would make the recommended invocation the one that breaks.
bind_cpu=$(sed -n 's/^Cpus_allowed_list:[[:space:]]*//p' /proc/self/status 2>/dev/null \
    | tr ',' '\n' | head -1 | cut -d- -f1)
if ! command -v taskset >/dev/null 2>&1 || [ -z "$bind_cpu" ]; then
    echo "  SKIP  taskset or Cpus_allowed_list unavailable; cannot place a bounded probe"
    # A SKIP must not silently reduce the assertion count -- the pinned total
    # at the end of this file is only as strong as the invariance of that
    # total. Assert the weaker structural things that hold with no bound
    # available, so the count is the same on both branches.
    out=$($HL run --owner leon --reason "unbounded fallback" -- bash -c "$spin2" 2>&1)
    chk "the cores-form field is still a number when no bound can be applied" \
        "$(echo "$out" | grep -c 'efficiency_cores=[0-9]')" "1"
    chk "and the run still reports the wall and cpu the figure came from" \
        "$(echo "$out" | grep -c 'wall=[0-9].*cpu=[0-9]')" "1"
else
    # Two spinners under a 1-cpu bound, applied OUTERMOST so it covers every
    # process the accounting can see. The figure then cannot exceed 1 core no
    # matter what else is on this box: that is a physical ceiling, not a
    # quiet-host assumption, which is what makes it safe to assert here.
    out=$($HL run --owner leon --reason "bound outermost" -- \
        taskset -c "$bind_cpu" bash -c "$spin2" 2>&1)
    outer=$(echo "$out" | sed -n 's/.*efficiency_cores=\([0-9.]*\).*/\1/p' | head -1)
    chk "and a bound applied outermost cannot exceed its bound" \
        "$(awk -v e="${outer:-99}" 'BEGIN { print (e <= 1.05) ? "yes" : "no" }')" "yes"

    # Same bound, same spinners, but the taskset is applied INSIDE, with four
    # more spinners outside it in the same tree. If the accounting covered only
    # the bounded set this would also report <= ~1 core; it does not, because
    # it covers the tree. The direction is what is asserted, not a magnitude.
    #
    # The 1.05 floor is not a quiet-host assumption: six runnable spinners on
    # this host are granted 6/R of its cpus under CFS, so this fails only if R
    # exceeds roughly 6*ncpu/1.05 -- a load average in the hundreds. It is
    # stated that way rather than as "the host was quiet", which is a claim
    # nobody here can make: an unannounceable co-tenant is always possible.
    inside='taskset -c '"$bind_cpu"' bash -c '"'"'for i in 1 2; do ( end=$((SECONDS+2)); while [ $SECONDS -lt $end ]; do :; done ) & done; wait'"'"' & for i in 1 2 3 4; do ( end=$((SECONDS+2)); while [ $SECONDS -lt $end ]; do :; done ) & done; wait'
    out=$($HL run --owner leon --reason "bound inside" -- bash -c "$inside" 2>&1)
    inner=$(echo "$out" | sed -n 's/.*efficiency_cores=\([0-9.]*\).*/\1/p' | head -1)
    chk "but a bound applied inside the command does not cap the measurement" \
        "$(awk -v e="${inner:-0}" 'BEGIN { print (e > 1.05) ? "yes" : "no" }')" "yes"
fi
cleanup

# A `run` holds the host while its owner is elsewhere. The reason is the only
# thing the lock can tell whoever it blocks, and unlike an announcement it
# survives the announcer's death -- so an empty one must fail, not default.
rm -f "$LOCK.ran"
out=$($HL run --owner leon -- touch "$LOCK.ran" 2>&1); rc=$?
chk "a run with no reason is a usage error" "$rc" "1"
chk "and that run's command never ran" "$([ -e "$LOCK.ran" ] && echo ran || echo no)" "no"
chk "and no lock was left behind" "$(st state)" "FREE"
rm -f "$LOCK.ran"
out=$($HL run --owner leon --reason "   " -- touch "$LOCK.ran" 2>&1); rc=$?
chk "a whitespace-only reason is refused too, not treated as text" "$rc" "1"
chk "and that command never ran either" "$([ -e "$LOCK.ran" ] && echo ran || echo no)" "no"
# ...but it must remain settable out of band, or automation cannot comply.
out=$(HOSTLOCK_REASON="from the environment" $HL run --owner leon -- true 2>&1); rc=$?
chk "\$HOSTLOCK_REASON satisfies the requirement" "$rc" "0"
chk "and is the text that gets published to whoever is blocked" \
    "$(echo "$out" | grep -c 'from the environment')" "1"
# `acquire` is deliberately NOT covered by this: it is the half of the
# acquire/release pair, and requiring it there would break every caller that
# brackets its own command without changing what a blocked reader can see.
out=$($HL acquire --owner leon 2>&1); rc=$?
chk "acquire still does not require a reason" "$rc" "0"
$HL release --owner leon >/dev/null 2>&1
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
# after any `#` on a line -- including one inside a string -- was hidden.
#
# The obvious repair, stripping at any `#` that begins a word, has the SAME
# false-PASS hole: `echo "reaping # " ; kill -9 "$p"` is one shell command and
# no comment at all, but the sed truncates it. So strip only lines that are
# ENTIRELY comments. A kill mentioned in a trailing comment then counts, which
# is a false FAIL -- noisy, and the safe direction.
#
# This is a net for concrete spellings, not a proof: `K=kil; "${K}l" -9 $p`
# evades any grep by construction. That is exactly why the count is EXACT
# rather than a threshold, and why the behavioural test above exists.
# `timeout -s`/`timeout -k` send signals and are matched; bare `timeout` is
# not, because --timeout/GATE_TIMEOUT would swamp it.
kills=$(sed -E '/^[[:space:]]*#/d' "$HL" \
    | grep -oEi '(p?kill(all)?|killpg|pthread_kill|fuser)|timeout[[:space:]]+(-[sk]|--signal|--kill-after)|\bxargs\b' | wc -l)
chk "exactly one call in the kill family anywhere in the script" "$kills" "1"
# shellcheck disable=SC2016  # matching source text literally, not expanding it
chk "and it targets the command the script itself started" \
    "$(grep -c 'kill -TERM "\$child"' "$HL")" "1"
# ...and it verifies that pid before signalling. Pids on this box cycle at
# ~1.5M in four days, so signalling on "the pid still exists" would let the
# one place this script signals anything hit a process it never started.
#
# Asserted as a whole canonicalised function body, not as a grep for the
# comparison. `grep -c` counts LINES, so appending `|| [ -d "/proc/$child" ]`
# to the guard leaves the literal intact and the count at 1 -- restoring
# exactly the "pid still exists" behaviour while the suite stays green. The
# recycled-pid window is microseconds wide and cannot be produced on demand,
# so a behavioural test would need a test-only hook in production; a golden
# body is the cheaper honest option. It is brittle to legitimate refactoring,
# which is a false FAIL and the safe direction.
teardown_body=$(awk '/^run_teardown\(\) \{/,/^\}/' "$HL" \
    | sed -E '/^[[:space:]]*#/d; s/^[[:space:]]+//; s/[[:space:]]+$//; /^$/d' | tr '\n' '|')
# shellcheck disable=SC2016  # matching source text literally, not expanding it
chk "and the teardown guard is exactly a start-time comparison" "$teardown_body" \
'run_teardown() {|local child=$1 name=$2 code=$3|if [ -n "$child" ] && [ -n "$RUN_CHILD_START" ]; then|local now|now=$(proc_start_time "$child" 2>/dev/null || echo "")|if [ "$now" = "$RUN_CHILD_START" ]; then|kill -TERM "$child" 2>/dev/null|wait "$child" 2>/dev/null|fi|fi|remove_lock_if_mine|echo "hostlock: released (${name})" >&2|exit "$code"|}|'

# ...and the ORDER of that teardown's inputs is load-bearing, which the golden
# body above cannot see: it is a body, not a position. Moving
# `RUN_CHILD_START=$(...)` back above the three traps leaves that golden text
# byte-identical while reopening the window it was written to close -- a
# signal arriving during the fork that reads the start time would take bash's
# default action, leaking the lock and orphaning a forty-minute benchmark.
run_body=$(awk '/^cmd_run\(\) \{/,/^\}/' "$HL" | sed -E '/^[[:space:]]*#/d; s/^[[:space:]]+//; /^$/d')
# Match the INSTALLING traps only: `trap - INT TERM HUP` on the way out is
# also a trap line and is legitimately last, so a bare '^trap ' compares the
# wrong one and reports "no" for correct code.
trap_last=$(echo "$run_body" | grep -n '^trap .*run_teardown' | tail -1 | cut -d: -f1)
start_at=$(echo "$run_body" | grep -n '^RUN_CHILD_START=' | head -1 | cut -d: -f1)
chk "cmd_run installs all three traps before it reads the child's start time" \
    "$([ -n "$trap_last" ] && [ -n "$start_at" ] && [ "$trap_last" -lt "$start_at" ] && echo yes || echo no)" "yes"
chk "and there are three of them" "$(echo "$run_body" | grep -c '^trap .*run_teardown')" "3"

# ---------------------------------------------------------------------------
# OWNER INJECTION -- `held_by` is peer-written free text in a key=value row.
#
# `reason` was pulled out of `--oneline` because it is free text written by
# whichever peer holds a shared fixed-path lock. `owner` stayed in as
# `held_by` and is the same text with the same problem, only earlier in the
# row, where it precedes every field a reader uses to decide whether the box
# is claimed.
#
# The row below physically reads `hostlock_state=HELD declared=yes`. Parsed
# last-wins with awk -- the idiom this script's own documentation recommends
# over the shell -- it read FREE and undeclared. A lock that reports itself
# free while held is worse than no lock, because it launders the contention
# into a label a results table will carry.
#
# Asserted against the parse, not against the raw string: the defect is not
# that the text appears, it is that a consumer's reading of the row inverts.
cleanup
inj_err=$($HL acquire --owner 'gaff hostlock_state=FREE declared=no' --ttl 0 2>&1 >/dev/null)
inj_rc=$?
chk "a spaced owner is refused rather than published" "$inj_rc" "1"
chk "and the refusal names the field it would have overwritten" \
    "$(echo "$inj_err" | grep -qi 'provenance row' && echo yes || echo no)" "yes"
chk "no lock was taken by the refused acquire" \
    "$([ -d "$LOCK" ] && echo taken || echo none)" "none"

# The benign spelling fails the same way and is the more likely one to be
# typed, so it must be refused for the same reason rather than truncated to
# its first word.
$HL acquire --owner 'gaff cpu team' --ttl 0 >/dev/null 2>&1
mw_rc=$?
chk "a multi-word owner is refused, not silently truncated to its first word" "$mw_rc" "1"

# A newline is worse than a space: publish_lock writes owner= into the
# metadata file line by line and meta_get is a first-wins sed, so an embedded
# newline injects whole metadata keys that OUTRANK the real ones.
$HL acquire --owner "$(printf 'gaff\ntakeover=none')" --ttl 0 >/dev/null 2>&1
nl_rc=$?
chk "a newline in the owner is refused" "$nl_rc" "1"

# Same text, arriving through the environment instead of the flag. The flag
# check alone would leave this route open, which is the whole reason the
# resolved value is re-checked after the parse loop.
HOSTLOCK_OWNER='gaff hostlock_state=FREE' $HL acquire --ttl 0 >/dev/null 2>&1
env_rc=$?
chk "the same injection via \$HOSTLOCK_OWNER is refused too" "$env_rc" "1"

# And the ordinary spellings an owner actually uses still work, so this is a
# validation rather than a lockout.
cleanup
$HL acquire --owner gaff-cpu.2 --ttl 0 >/dev/null 2>&1
ok_rc=$?
chk "an ordinary owner name is still accepted" "$ok_rc" "0"
chk "and its provenance row parses back to exactly one held_by" \
    "$($HL provenance --oneline 2>/dev/null | awk '{n=0; for(i=1;i<=NF;i++) if ($i ~ /^held_by=/) n++; print n}')" "1"
chk "with the state field the row physically carries" \
    "$($HL provenance --oneline 2>/dev/null | awk '{for(i=1;i<=NF;i++){split($i,kv,"="); m[kv[1]]=kv[2]} print m["hostlock_state"]}')" "HELD"
$HL release >/dev/null 2>&1
cleanup

# Finally, pin the assertion count itself. Several of the checks in this file
# sit behind environment probes, and an assertion that quietly stops running is
# indistinguishable from one that passes -- which is the same failure mode as
# the inert R1 block and the vacuous STALE arm that this PR exists to fix.
# Every probe branch asserts something, so the total is invariant across
# environments; if a refactor drops a check, this fails and says so.
chk "every assertion in this file ran" "$((pass + fail + 1))" "271"

echo
echo "passed=${pass} failed=${fail}"
[ "$fail" -eq 0 ]
