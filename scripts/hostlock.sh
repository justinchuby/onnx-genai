#!/usr/bin/env bash
# hostlock.sh — advisory whole-host lock for benchmark runs
#
# Several agents share one machine. A benchmark that runs while somebody
# else is saturating the cores does not produce a slow number, it produces a
# meaningless one: the same cell has been measured at 197.2 and 22.8 tok/s in
# two windows on this host, an 8.6x swing, with intra-run spreads as tight as
# 6% in the corrupted samples. A tight spread only says the contention was
# steady, not that the host was quiet.
#
# Announce-before/announce-after works right up to the moment somebody is
# heads-down, which is exactly when it is needed. This is the mechanical
# version of the same protocol.
#
# It is ADVISORY. It cannot stop a process from using cores and does not try
# to. It answers one question honestly -- "is somebody benchmarking right
# now, and who?" -- and it makes the answer cheap enough to check that there
# is no excuse for not checking.
#
# Reclaiming a lock does NOT stop the load that lock was covering. A holder
# killed with SIGKILL leaves its benchmark orphaned and still burning cores,
# and the next acquirer will reap the lock and start measuring on a host that
# is not actually quiet. That is what `--gate` is for: the lock tells you
# whether a participant claims the box, the runnable count tells you whether
# anything is actually running on it. They answer different questions and you
# want both.
#
# It assumes every participant runs as the same UID. /tmp is sticky, so under
# mixed UIDs one user cannot rename or remove another's lock and takeover,
# reaping and release all fail with EPERM.
#
# Usage:
#   hostlock.sh status [--porcelain]    # who holds it, is that holder alive
#   hostlock.sh acquire [opts]          # take it, or fail / wait
#   hostlock.sh release                 # give it back (only if you hold it)
#   hostlock.sh wait [--timeout S]      # block until free, do not take it
#   hostlock.sh run [opts] -- CMD...    # acquire, run CMD, always release
#
# Options for acquire/run:
#   --reason TEXT     what you are running (shown to whoever is blocked)
#   --owner NAME      defaults to $HOSTLOCK_OWNER, else $USER
#   --wait            block until free instead of failing immediately
#   --timeout S       give up after S seconds (default 3600 with --wait)
#   --gate N          after taking the lock, also wait until the instantaneous
#                     runnable count is <= N, so non-participating load (other
#                     agents' builds, a stray editor) is drained too
#   --gate-timeout S  give up gating after S seconds (default 900) and proceed
#                     anyway, printing the runnable count actually reached
#   --ttl S           hard expiry: a lock older than this is reapable by the
#                     next acquirer, which prints a loud warning naming you.
#                     Default 3600 for `acquire`, 0 (never) for `run`.
#   --pid N           liveness anchor; defaults to the invoking shell ($PPID)
#
# Exit codes:
#   0 ok   1 usage/error   2 busy (acquire without --wait)   3 timed out
#
# The `run` form is the one to prefer: it releases on success, on failure and
# on Ctrl-C, and it stops the wrapped command when interrupted. Signals it
# cannot catch (SIGKILL) leave the lock in place, where the pid anchor lets
# the next acquirer reclaim it -- so a crash costs one stale lock, not a
# wedged box.
#
# Liveness, and why there are two mechanisms rather than one.
#
# The lock is a directory (mkdir is atomic on POSIX). Inside it is an anchor
# pid AND that pid's start time from /proc/<pid>/stat. A crashed holder is
# reaped automatically by the next acquirer. The start time is what makes
# that safe: pids are recycled, so "is there a process with pid 12345" is not
# the same question as "is the process that took this lock still alive".
# Reaping is guarded by a second mkdir lock, so two acquirers racing on the
# same dead lock cannot both reap and both believe they won.
#
# The anchor cannot be this script's own pid for `acquire`, because that pid
# dies the moment the script returns to your shell -- the lock would be stale
# on arrival and the next acquirer would reap it instantly. So `acquire`
# anchors to the invoking shell ($PPID). That shell is typically an agent
# session alive for days, which fixes the stale-on-arrival bug and creates
# the opposite one: a lock you forget to release survives for days. Hence the
# TTL, which is a hard expiry and not merely a label. `run` is the exception
# and the reason to prefer it: its anchor is its own pid, which is exact, so
# it needs no expiry. Note the distinction between the anchor dying and the
# lock being released -- on SIGKILL the directory survives, and it is the
# dead anchor that lets the next acquirer reclaim it.

set -uo pipefail

# Deliberately a fixed path, NOT $TMPDIR. A whole-host lock is only worth
# anything if every agent on the box resolves it to the same directory, and
# TMPDIR is per-session: one agent with it set would get a private lock,
# acquire it instantly every time, and never once collide with anybody --
# coordination that silently does nothing is worse than none, because it is
# believed. Override with HOSTLOCK_DIR only for testing.
LOCK_DIR="${HOSTLOCK_DIR:-/tmp/onnx-genai-hostlock}"
UNPARSEABLE_GRACE=300
REAPER_GRACE=60
REAP_DIR="${LOCK_DIR}.reaper"
META="${LOCK_DIR}/meta"

die() {
    echo "hostlock: $*" >&2
    exit 1
}

# Start time (field 22) of a pid, robust to a comm containing spaces or
# parentheses: everything up to and including the last ')' is dropped, which
# removes fields 1 and 2, so field 22 becomes field 20 of the remainder.
proc_start_time() {
    local pid=$1 rest
    rest=$(tr -d '\0' <"/proc/${pid}/stat" 2>/dev/null | sed 's/.*) //') || return 1
    [ -n "$rest" ] || return 1
    awk '{print $20}' <<<"$rest"
}

runnable_now() {
    cut -d' ' -f4 /proc/loadavg | cut -d/ -f1
}

# Metadata is a file on a shared filesystem, so treat it as untrusted: a
# corrupt or truncated `ttl=`/`acquired_epoch=` line must not make arithmetic
# throw. Non-numeric degrades to the default, which fails safe (not expired).
num_or() {
    case "${1:-}" in
        '' | *[!0-9]*) echo "$2" ;;
        *) echo "$1" ;;
    esac
}

meta_get() {
    [ -f "$META" ] || return 1
    sed -n "s/^$1=//p" "$META" | head -1
}

# 0 if the recorded anchor process is still running, 1 otherwise.
holder_alive() {
    local pid start now
    pid=$(meta_get anchor_pid) || return 1
    start=$(meta_get start_time) || return 1
    [ -n "$pid" ] && [ -d "/proc/${pid}" ] || return 1
    now=$(proc_start_time "$pid") || return 1
    # A recycled pid has a different start time, so this is not just "does
    # some process with this number exist".
    [ "$now" = "$start" ]
}

lock_age() {
    local epoch now
    epoch=$(num_or "$(meta_get acquired_epoch || echo 0)" 0)
    now=$(date +%s)
    echo $((now - epoch))
}

# 0 if the lock has outlived its declared TTL. ttl=0 means never expires,
# which is only safe when the anchor is exact (the `run` form).
holder_expired() {
    local ttl
    ttl=$(num_or "$(meta_get ttl || echo 0)" 0)
    [ "$ttl" -gt 0 ] || return 1
    [ "$(lock_age)" -gt "$ttl" ]
}

# 0 if the lock is reapable: its anchor died, or it outlived its TTL.
#
# A lock with no readable metadata is NOT evidence of a dead holder -- it is
# most likely a lock from an older or newer version of this script, or one
# being written by a peer. Refuse to reap it until it is clearly abandoned,
# so that an unparseable lock degrades to "busy" rather than "free".
reapable() {
    if [ ! -s "$META" ]; then
        local mtime now
        mtime=$(stat -c %Y "$LOCK_DIR" 2>/dev/null || echo 0)
        now=$(date +%s)
        [ "$((now - mtime))" -gt "$UNPARSEABLE_GRACE" ]
        return $?
    fi
    ! holder_alive && return 0
    holder_expired
}

# Remove a reapable lock. Guarded so that two acquirers racing on the same
# dead lock cannot both conclude they reaped it.
reap_if_dead() {
    # The reaper guard is the one directory here with no anchor and no owner,
    # so a crash inside the critical section below would leak it and make
    # every genuinely dead lock permanently un-reapable -- acquirers would be
    # told BUSY forever with nothing holding anything. Bound it by age.
    if [ -d "$REAP_DIR" ]; then
        local rm_age rm_now
        rm_age=$(stat -c %Y "$REAP_DIR" 2>/dev/null || echo 0)
        rm_now=$(date +%s)
        if [ "$((rm_now - rm_age))" -gt "$REAPER_GRACE" ]; then
            rmdir "$REAP_DIR" 2>/dev/null
        fi
    fi
    mkdir "$REAP_DIR" 2>/dev/null || return 1
    local reaped=1
    if [ -d "$LOCK_DIR" ] && reapable; then
        local pid owner
        pid=$(meta_get anchor_pid 2>/dev/null || echo '?')
        owner=$(meta_get owner 2>/dev/null || echo '?')
        if holder_alive; then
            echo "hostlock: WARNING taking over a lock held by ${owner} (pid ${pid}, still alive) after its $(meta_get ttl)s TTL expired" >&2
            echo "hostlock: WARNING if ${owner} is still benchmarking, both sets of numbers are now suspect" >&2
        else
            echo "hostlock: reaping stale lock from dead pid ${pid} (owner ${owner})" >&2
        fi
        remove_lock
        reaped=0
    fi
    rmdir "$REAP_DIR" 2>/dev/null
    return $reaped
}

# Publish the lock atomically, complete with its metadata.
#
# The obvious implementation -- mkdir, then write meta into it -- has a race
# that cost this script three simultaneous winners in testing. mkdir itself
# is atomic, but between it and the finished meta file there is a window in
# which the lock exists with no readable anchor pid. A competing acquirer
# that looks during that window sees a lock whose holder cannot be shown to
# be alive, concludes it is stale, reaps it, and takes the host from someone
# who is already benchmarking on it.
#
# So build the lock in a private staging directory, fill in the metadata, and
# move it into place with rename(2). rename onto an existing NON-EMPTY
# directory fails with ENOTEMPTY, which is exactly the exclusion we want, and
# the lock becomes visible only in its finished state. There is no window.
#
# The exclusion rests on an invariant, not on rename alone: rename onto an
# EMPTY directory SUCCEEDS, so this is only safe because a published lock is
# never empty -- it always contains `meta`, and remove_lock takes the whole
# directory away in one step rather than emptying it first. Anything that
# creates $LOCK_DIR directly, with a bare mkdir anywhere, breaks that
# invariant and reintroduces the three-winner bug.
publish_lock() {
    local start stage
    start=$(proc_start_time "$ANCHOR_PID") || die "cannot read start time of anchor pid ${ANCHOR_PID}"
    stage="${LOCK_DIR}.stage.$$"
    rm -rf "$stage"
    mkdir -p "$stage" || die "cannot create staging dir ${stage}"
    {
        echo "anchor_pid=${ANCHOR_PID}"
        echo "start_time=${start}"
        echo "script_pid=$$"
        echo "owner=${OWNER}"
        echo "reason=${REASON}"
        echo "acquired_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "acquired_epoch=$(date +%s)"
        echo "ttl=${TTL}"
        echo "runnable_at_acquire=$(runnable_now)"
    } >"${stage}/meta"
    if mv -T "$stage" "$LOCK_DIR" 2>/dev/null; then
        return 0
    fi
    rm -rf "$stage"
    return 1
}

# One state word, for scripts: FREE, HELD, EXPIRED or STALE.
lock_state() {
    [ -d "$LOCK_DIR" ] || { echo FREE; return 0; }
    if holder_alive; then
        if holder_expired; then echo EXPIRED; else echo HELD; fi
    else
        echo STALE
    fi
}

# Remove the lock atomically.
#
# `rm -rf "$LOCK_DIR"` is not safe here for the mirror image of the reason
# publish_lock stages: rm unlinks `meta` first and the directory second, so
# for a moment the lock exists with no metadata -- indistinguishable, to a
# concurrent acquirer, from a lock being published. Rename it out of the way
# first; the name disappears in one step, and the corpse can be deleted at
# leisure.
remove_lock() {
    local doomed="${LOCK_DIR}.dead.$$"
    rm -rf "$doomed" 2>/dev/null
    if mv -T "$LOCK_DIR" "$doomed" 2>/dev/null; then
        rm -rf "$doomed" 2>/dev/null
        return 0
    fi
    rm -rf "$LOCK_DIR" 2>/dev/null
    return 0
}

# Release, but only if the lock is still ours.
#
# `run` used to tear down unconditionally. If its TTL expired mid-command and
# somebody else legitimately took over, finishing the command then deleted
# THEIR live lock -- and the next acquirer would be handed a host that two
# people were already using. That is the exact doubling this tool exists to
# prevent, arriving through the teardown path instead of the acquire path.
remove_lock_if_mine() {
    local pid
    pid=$(meta_get anchor_pid 2>/dev/null || echo '')
    if [ -n "$pid" ] && [ "$pid" != "$ANCHOR_PID" ]; then
        echo "hostlock: NOT releasing -- the lock is now held by pid ${pid} ($(meta_get owner 2>/dev/null))." >&2
        echo "hostlock: we were taken over mid-run, so both sets of numbers are suspect." >&2
        return 1
    fi
    remove_lock
}

cmd_status() {
    if [ "$PORCELAIN" = 1 ]; then
        echo "state=$(lock_state)"
        echo "runnable=$(runnable_now)"
        if [ -d "$LOCK_DIR" ]; then
            echo "owner=$(meta_get owner || echo '?')"
            echo "anchor_pid=$(meta_get anchor_pid || echo '?')"
            echo "reason=$(meta_get reason || echo '')"
            echo "ttl=$(meta_get ttl || echo 0)"
            echo "age=$(lock_age)"
        fi
        return 0
    fi
    if [ ! -d "$LOCK_DIR" ]; then
        echo "FREE  (runnable=$(runnable_now))"
        return 0
    fi
    local owner reason at pid age ttl
    owner=$(meta_get owner || echo '?')
    reason=$(meta_get reason || echo '?')
    at=$(meta_get acquired_at || echo '?')
    pid=$(meta_get anchor_pid || echo '?')
    ttl=$(meta_get ttl || echo 0)
    age=$(lock_age)
    if holder_alive; then
        local state="HELD"
        if holder_expired; then
            state="EXPIRED (held ${age}s > ttl ${ttl}s; next acquire will take it over)"
        fi
        echo "${state} by ${owner} pid=${pid} for ${age}s since ${at}"
        echo "  reason: ${reason}"
        echo "  runnable=$(runnable_now)"
        return 0
    fi
    echo "STALE  holder ${owner} pid=${pid} is gone; next acquire will reap it"
    return 0
}

cmd_acquire() {
    local deadline=$((SECONDS + TIMEOUT))
    while :; do
        if publish_lock; then
            gate_on_runnable
            echo "hostlock: acquired by ${OWNER} (anchor pid ${ANCHOR_PID})${REASON:+ — ${REASON}}"
            return 0
        fi
        # Existing lock: reap it if its anchor died OR it outlived its TTL,
        # then retry immediately. Testing `! holder_alive` here instead of
        # `reapable` silently disabled TTL expiry altogether -- status would
        # report EXPIRED and the next acquirer would still be told BUSY.
        if reapable; then
            reap_if_dead && continue
        fi
        if [ "$DO_WAIT" != 1 ]; then
            echo "hostlock: BUSY" >&2
            cmd_status >&2
            return 2
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "hostlock: timed out after ${TIMEOUT}s waiting for the lock" >&2
            cmd_status >&2
            return 3
        fi
        sleep 5
    done
}

# Holding the lock only excludes other participants. It says nothing about a
# build, an editor, or an agent that never took it, so optionally wait for the
# machine itself to go quiet as well.
gate_on_runnable() {
    [ -n "$GATE" ] || return 0
    local deadline=$((SECONDS + GATE_TIMEOUT)) r
    while :; do
        r=$(runnable_now)
        if [ "$r" -le "$GATE" ]; then
            [ "$SECONDS" -gt 0 ] && echo "hostlock: host quiet (runnable=${r} <= ${GATE})"
            return 0
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "hostlock: WARNING gate timed out after ${GATE_TIMEOUT}s; proceeding at runnable=${r} (wanted <= ${GATE}). Treat results from this run as suspect." >&2
            return 0
        fi
        sleep 5
    done
}

cmd_release() {
    [ -d "$LOCK_DIR" ] || {
        echo "hostlock: not held; nothing to release"
        return 0
    }
    local pid
    pid=$(meta_get anchor_pid || echo '')
    # Only the holder may release, so a stray `release` in someone else's
    # shell cannot hand the box away mid-run.
    if [ -n "$pid" ] && [ "$pid" != "$ANCHOR_PID" ] && [ "$pid" != "$$" ] && holder_alive; then
        if [ "${HOSTLOCK_FORCE:-0}" = 1 ]; then
            echo "hostlock: forcing release of a live lock held by pid ${pid}" >&2
        else
            echo "hostlock: refusing to release a lock held by live pid ${pid}" >&2
            echo "hostlock: set HOSTLOCK_FORCE=1 if you are certain" >&2
            return 1
        fi
    fi
    remove_lock
    echo "hostlock: released"
}

cmd_wait() {
    local deadline=$((SECONDS + TIMEOUT))
    # Wait only while somebody actually has a live claim. An EXPIRED lock is
    # one the next acquirer would take over, so blocking on it would disagree
    # with what `status` and `acquire` both say.
    while [ "$(lock_state)" = HELD ]; do
        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "hostlock: still held after ${TIMEOUT}s" >&2
            return 3
        fi
        sleep 5
    done
    echo "hostlock: free (runnable=$(runnable_now))"
}

# Kill the wrapped command, release, and exit with the conventional code.
run_teardown() {
    local child=$1 name=$2 code=$3
    if [ -n "$child" ] && [ -e "/proc/${child}" ]; then
        kill -TERM "$child" 2>/dev/null
        wait "$child" 2>/dev/null
    fi
    remove_lock_if_mine
    echo "hostlock: released (${name})" >&2
    exit "$code"
}

cmd_run() {
    [ "$#" -gt 0 ] || die "run needs a command after --"
    DO_WAIT=1
    cmd_acquire || return $?

    # Run the command in the BACKGROUND and wait for it, rather than inline.
    #
    # Bash does not run a trap until the current foreground command finishes.
    # With `"$@"` inline, Ctrl-C during a forty-minute benchmark would not
    # release the lock until that benchmark ended by itself -- prompt release
    # failing in precisely the case it exists for. `wait` is interruptible,
    # so this makes the signal handlers actually prompt. It also lets us stop
    # the wrapped command, which the inline form left running.
    local child rc
    "$@" &
    child=$!

    trap 'run_teardown "$child" SIGINT 130' INT
    trap 'run_teardown "$child" SIGTERM 143' TERM
    trap 'run_teardown "$child" SIGHUP 129' HUP

    wait "$child"
    rc=$?
    trap - INT TERM HUP
    remove_lock_if_mine
    echo "hostlock: released (command exit ${rc})"
    return $rc
}

OWNER="${HOSTLOCK_OWNER:-${USER:-unknown}}"
REASON=""
DO_WAIT=0
TIMEOUT=3600
GATE=""
GATE_TIMEOUT=900
TTL=""
ANCHOR_PID=""
PORCELAIN=0

[ "$#" -ge 1 ] || {
    sed -n '3,50p' "$0" >&2
    exit 1
}
SUB=$1
shift

while [ "$#" -gt 0 ]; do
    case "$1" in
        --reason)
            REASON=${2:-}
            shift 2
            ;;
        --owner)
            OWNER=${2:-}
            shift 2
            ;;
        --wait)
            DO_WAIT=1
            shift
            ;;
        --timeout)
            TIMEOUT=${2:-3600}
            shift 2
            ;;
        --gate)
            GATE=${2:-}
            shift 2
            ;;
        --gate-timeout)
            GATE_TIMEOUT=${2:-900}
            shift 2
            ;;
        --ttl)
            TTL=${2:-0}
            shift 2
            ;;
        --pid)
            ANCHOR_PID=${2:-}
            shift 2
            ;;
        --porcelain)
            PORCELAIN=1
            shift
            ;;
        --)
            shift
            break
            ;;
        *) die "unknown option: $1" ;;
    esac
done

# `run` anchors to itself: that pid is exact and dies with the command on
# every exit path, so it needs no expiry. `acquire` anchors to the invoking
# shell, which can outlive the benchmark by days, so it gets a default TTL.
if [ "$SUB" = run ]; then
    : "${ANCHOR_PID:=$$}"
    : "${TTL:=0}"
else
    : "${ANCHOR_PID:=$PPID}"
    : "${TTL:=3600}"
fi
[ -d "/proc/${ANCHOR_PID}" ] || die "anchor pid ${ANCHOR_PID} is not running"

case "$SUB" in
    status) cmd_status ;;
    acquire) cmd_acquire ;;
    release) cmd_release ;;
    wait) cmd_wait ;;
    run) cmd_run "$@" ;;
    *) die "unknown subcommand: ${SUB}" ;;
esac
