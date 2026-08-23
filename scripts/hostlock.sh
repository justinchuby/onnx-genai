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
#   hostlock.sh provenance [opts]       # lock state as fields, to record WITH
#                                       # each measured row (see below)
#
# Options for acquire/run:
#   --reason TEXT     what you are running (shown to whoever is blocked)
#   --owner NAME      defaults to $HOSTLOCK_OWNER, else $USER
#   --wait            block until free instead of failing immediately
#   --timeout S       give up after S seconds (default 3600 with --wait)
#   --gate N          after taking the lock, also wait until the instantaneous
#                     runnable count is <= N, so non-participating load (other
#                     agents' builds, a stray editor) is drained too.
#                     N=0 is unsatisfiable by construction: field 4 of
#                     /proc/loadavg counts the process doing the reading, so
#                     the floor is 1 even on a perfectly idle box.
#                     The gate is a SECONDARY signal. The lock -- declared
#                     intent -- is the primary one. A bounded good citizen
#                     using 4 of 32 cpus reads runnable 4-5 and trips
#                     `--gate 3`, while a single-threaded 100% hog reads ~1
#                     and sails through. Do not gate tightly and call it
#                     admission control.
#   --gate-timeout S  give up gating after S seconds (default 900)
#   --on-gate-timeout fail|proceed
#                     what to do when the gate expires. Default `fail`: release
#                     the lock and exit 5, because a gate that warns and
#                     proceeds labels contaminated rows as gated. `proceed`
#                     measures anyway and records `gate=timed_out_proceeded`.
#   --strict-reap     refuse (exit 4, releasing) an acquire that had to reap a
#                     stale or TTL-expired lock, instead of taking it over.
#                     Reclaiming a lock does not stop the dead holder's
#                     benchmark, so for an unattended harness "somebody died
#                     holding this" is a reason to stop. Decided before the
#                     gate, so it never sits on a lock it is about to refuse.
#   --ttl S           hard expiry: a lock older than this is reapable by the
#                     next acquirer, which prints a loud warning naming you.
#                     Default 3600 for `acquire`, 0 (never) for `run`.
#   --pid N           liveness anchor; defaults to the invoking shell ($PPID)
#
# Options for run:
#   --expect-cores N      how many cores the wrapped command should be able to
#                         keep busy. Required by --min-efficiency, and
#                         deliberately not defaulted: defaulting it to 1 would
#                         report a 16-thread benchmark at efficiency 16.0 and
#                         pass every threshold, which is the reassuring
#                         direction and the wrong one.
#   --min-efficiency F    after the command finishes, compare its measured CPU
#                         time against N cores x wall time, and FAIL if the
#                         ratio is below F (see "Believing a run" below).
#
# Options for provenance:
#   --expect-runnable N   judge `contended=yes/no` against N; omitted means
#                         `contended=unknown`, which is honest rather than a
#                         guessed threshold
#   --oneline             one space-separated line, to append to a result row
#
# Exit codes (acquire/release/wait):
#   0 ok            1 usage/error       2 busy (acquire without --wait)
#   3 timed out     4 reap refused      5 gate not satisfied
#   6 CPU efficiency below --min-efficiency (`run` only, and only when the
#     wrapped command itself succeeded -- a failing command's status wins,
#     because it is the more important signal). Note this collides with a
#     wrapped command that exits 6 by itself, exactly as 5 collides with the
#     gate: `run` multiplexes two status spaces onto one, and the `cpu ...
#     verdict=` line, not the exit code, is what tells them apart.
#
# `run` otherwise does NOT use this table: it returns the wrapped command's
# own status, so a command exiting 5 is indistinguishable from a gate failure.
# When you need to tell hostlock's failures apart from the command's, use
# `acquire` and `release` around the command rather than `run`. Interrupted
# `run` returns 128+signal (130 SIGINT, 143 SIGTERM) like any shell. Passing
# --min-efficiency is an explicit request to override that contract for one
# case; without it, `run`'s status is unchanged.
#
# Believing a run, as opposed to starting one. The lock decides whether to
# START and --gate decides whether the host looks quiet enough; neither can
# tell you afterwards whether the numbers are any good. They are admission
# controls, and admission controls sample instants:
#
#   Roy gated on the instantaneous runnable count, sampled before and after
#   each arm, and it reported "peak 2-4, clean" for runs that were in fact
#   getting 50-70% of a core. A 2-second arm has ample room for a burst that
#   begins after the opening sample and ends before the closing one. His A/A
#   null was 52% -- the same binary against itself, disagreeing by half.
#
# --min-efficiency measures the thing itself rather than a proxy for it: a
# process that owns its cores spends ~1.00 CPU-seconds per core per wall
# second, and anything less means it did not have them, whatever the runnable
# count said at either end. It needs no quiet host; it tells you which reps to
# throw away.
#
# It measures "did not have the cores", which is NOT the same claim as
# "somebody stole them": a benchmark that sleeps, blocks on I/O, or leaves a
# deliberate inter-token gap is legitimately below 1.0 and is not contended.
# Only you know which your workload is, which is why the threshold is yours to
# set and why there is no default. Set it from a measured quiet-host run, not
# from an ideal. With it, that same collection went from a 52%
# null to 0.04-0.56%. The two compose -- the gate decides whether to start,
# this decides whether to believe -- and the credit for the technique is
# Roy's.
#
#   hostlock.sh run --owner leon --reason "moe 6-width matrix" \
#       --expect-cores 16 --min-efficiency 0.90 -- ./bench.sh
#
# CPU time comes from this shell's own reaped-children accounting
# (/proc/self/stat cutime+cstime), so it covers the wrapped command and every
# descendant it waited for, and costs nothing: no sampling, no extra process.
#
# Recording provenance. Emitting the lock state to a terminal helps whoever is
# watching; embedding it in the rows is what helps six weeks later. Every
# contaminated run on this host was caught, when it was caught, by somebody
# having an A/A null arm -- which is luck, not method. So:
#
#   printf '%s\t%s\n' "$ms" "$(scripts/hostlock.sh provenance --oneline \
#                                --expect-runnable 2)" >> results.tsv
#
# Append the line to the row; do not `eval` it. The fields are `key=value` and
# every value in the one-line form is a bounded token, but `reason` is free
# text written by whichever peer holds this shared, fixed-path lock, so it is
# omitted from `--oneline` entirely and only appears in the multi-line form,
# where the newline delimits it. Parse with awk/split on `=`, never with the
# shell.
#
# Record `hostlock_state` and `takeover` alongside `held_by` -- `held_by` names
# the holder of a STALE lock just as readily as a live one, so on its own it
# can attribute a row to somebody who was already dead. A row that cannot say
# what the host was doing is not a measurement, it is an anecdote with a number
# in it.
#
# (end of usage summary)
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
# Reaping is guarded by a second lock, so two acquirers racing on the same
# dead lock cannot both reap and both believe they won. That guard is built
# the same way as the lock -- published atomically by rename, owned by pid +
# start time, reclaimed from a dead owner immediately and never taken from a
# live one -- because a guard without liveness is just a smaller wedge in a
# place nobody looks. See "The reaper guard" below.
#
# PORTABILITY: Linux only, deliberately and unavoidably. Every liveness claim
# here comes from /proc/<pid>/stat and /proc/<pid>/status; the atomic publish
# needs `mv -T` (GNU coreutils), the age backstop needs `stat -c %Y`, and the
# occupancy field needs /proc/loadavg. There is no BSD/macOS fallback and no
# attempt at one: a fallback that silently degrades liveness to `kill -0`
# would reap live holders on the platform that took it, which is the one
# error this script must never make.
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
REAP_META="${REAP_DIR}/meta"
META="${LOCK_DIR}/meta"

die() {
    echo "hostlock: $*" >&2
    exit 1
}

# Start time (field 22) of a pid, robust to a comm containing spaces or
# parentheses: everything up to and including the last ')' is dropped, which
# removes fields 1 and 2, so field 22 becomes field 20 of the remainder.
# NOTE the 2>/dev/null precedes the input redirection. Bash performs
# redirections left to right, so with the usual `<file 2>/dev/null` ordering
# the "No such file or directory" for a departed pid is emitted BEFORE stderr
# is silenced -- which put two lines of shell noise in front of every `status`
# on a stale lock, and in front of anything parsing it.
proc_start_time() {
    local pid=$1 rest
    rest=$(tr -d '\0' 2>/dev/null <"/proc/${pid}/stat" | sed 's/.*) //') || return 1
    [ -n "$rest" ] || return 1
    awk '{print $20}' <<<"$rest"
}

# Process state (field 3) and start time (field 22) in one read. Both come
# from the same remainder after the comm field is dropped, so they cost one
# open between them and cannot disagree about which process they describe.
proc_state_and_start() {
    local pid=$1 rest
    rest=$(tr -d '\0' 2>/dev/null <"/proc/${pid}/stat" | sed 's/.*) //') || return 1
    [ -n "$rest" ] || return 1
    awk '{print $1, $20}' <<<"$rest"
}

runnable_now() {
    cut -d' ' -f4 /proc/loadavg | cut -d/ -f1
}

# CPU time consumed by every child this shell has REAPED, in clock ticks.
# Sets the global CPU_TICKS; deliberately not an echo, because the value must
# be read in THIS process. fork() zeroes a child's RUSAGE_CHILDREN, so
# `t=$(children_cpu_ticks)` -- and equally `times | awk` -- reads a freshly
# zeroed counter in the subshell and always answers 0. `read < file` is a
# builtin with a redirection and does not fork, so this is exact and free.
read_children_cpu_ticks() {
    local line rest fields cu cs
    CPU_TICKS=""
    read -r line < /proc/self/stat 2>/dev/null || return 1
    # comm (field 2) is parenthesised and may contain spaces; everything from
    # the last ") " is positional. First token after it is field 3 (state),
    # so cutime (16) and cstime (17) are tokens 14 and 15.
    rest=${line##*") "}
    # shellcheck disable=SC2206  # deliberate word splitting: this is a field list
    fields=($rest)
    cu=${fields[13]:-}
    cs=${fields[14]:-}
    # Both must be present and numeric. The ':' separator means an empty
    # field shows up as a leading or trailing colon, so those are the empty
    # cases; an empty pattern could never match this word.
    case "${cu}:${cs}" in
        *[!0-9:]* | :* | *:) return 1 ;;
    esac
    CPU_TICKS=$((cu + cs))
}

wall_now() {
    date +%s.%N
}

# Report, and optionally judge, how much CPU the wrapped command actually got.
#
# Prints one line whatever happens, because the number is worth recording even
# when nothing is being enforced -- an unjudged measurement in the log is what
# lets somebody re-examine a suspicious row later. Returns non-zero only when
# a --min-efficiency was asked for and is not met, or cannot be evaluated.
check_cpu_efficiency() {
    local t0=$1 t1=$2 w0=$3 w1=$4 hz cpu wall verdict eff
    hz=$(getconf CLK_TCK 2>/dev/null || echo 100)
    if [ -z "$t0" ] || [ -z "$t1" ]; then
        # Unreadable /proc accounting. Say so; do not print a number.
        echo "hostlock: cpu unmeasurable (could not read this shell's child CPU accounting)" >&2
        [ -n "$MIN_EFFICIENCY" ] || return 0
        echo "hostlock: WARNING --min-efficiency ${MIN_EFFICIENCY} cannot be evaluated, so this run is NOT verified" >&2
        return 1
    fi
    cpu=$(awk -v a="$t0" -v b="$t1" -v hz="$hz" 'BEGIN { printf "%.3f", (b - a) / hz }')
    wall=$(awk -v a="$w0" -v b="$w1" 'BEGIN { printf "%.3f", b - a }')
    # Below ~50ms the clock-tick quantum (usually 10ms) is a large fraction of
    # the measurement, so a ratio computed from it is noise wearing a decimal
    # point. Refuse to publish one rather than publish a bad one.
    if awk -v w="$wall" 'BEGIN { exit !(w < 0.05) }'; then
        echo "hostlock: cpu wall=${wall}s cpu=${cpu}s efficiency=unmeasurable (run too short to judge)" >&2
        [ -n "$MIN_EFFICIENCY" ] || return 0
        echo "hostlock: WARNING --min-efficiency ${MIN_EFFICIENCY} cannot be evaluated on a ${wall}s run, so this run is NOT verified" >&2
        return 1
    fi
    if [ -n "$EXPECT_CORES" ]; then
        eff=$(awk -v c="$cpu" -v w="$wall" -v n="$EXPECT_CORES" 'BEGIN { printf "%.3f", c / (w * n) }')
    else
        eff=$(awk -v c="$cpu" -v w="$wall" 'BEGIN { printf "%.3f", c / w }')
    fi
    verdict=unjudged
    local rc=0
    if [ -n "$MIN_EFFICIENCY" ]; then
        if awk -v e="$eff" -v m="$MIN_EFFICIENCY" 'BEGIN { exit !(e < m) }'; then
            verdict=contended
            rc=1
        else
            verdict=ok
        fi
    fi
    echo "hostlock: cpu wall=${wall}s cpu=${cpu}s cores_expected=${EXPECT_CORES:-unspecified} efficiency=${eff} verdict=${verdict}" >&2
    if [ "$rc" -ne 0 ]; then
        echo "hostlock: WARNING measured CPU efficiency ${eff} is below --min-efficiency ${MIN_EFFICIENCY}" >&2
        echo "hostlock: WARNING the command did not have ${EXPECT_CORES} core(s) to itself for the whole run; treat its numbers as untrusted" >&2
        echo "hostlock: WARNING the runnable gate cannot see this -- it samples instants, this measures the whole window" >&2
    fi
    return "$rc"
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

# Returns 1 when the key is ABSENT, not just when the file is. The two are
# different facts and collapsing them is how a lock written by an older
# version of this script reports `takeover=none gate=none` -- an assertion --
# for a takeover and a gate it never recorded. Absent must surface as
# `unknown`, for the same reason `contended` is `unknown` without an
# expectation.
meta_get() {
    [ -f "$META" ] || return 1
    grep -q "^$1=" "$META" 2>/dev/null || return 1
    sed -n "s/^$1=//p" "$META" | head -1
}

# Update one key in the metadata of a lock we hold.
#
# Written to a sibling file and renamed, so a concurrent reader sees either
# the old value or the new one and never a half-written file -- the same
# reason publish_lock stages and renames. Refuses to touch a lock whose
# anchor is not ours, so a takeover race cannot have the loser rewriting the
# winner's metadata.
#
# `tmp` MUST stay INSIDE $LOCK_DIR. That is what actually makes this safe
# against a concurrent reap, not the anchor check: remove_lock renames the
# whole directory away, so a staged file inside it goes with the corpse and
# the final mv fails harmlessly. Moved to a sibling path (as every other temp
# path in this file is) it would survive the reap and land on the SUCCESSOR's
# meta -- writing our metadata over their live lock, which is theft plus a
# leak, since their remove_lock_if_mine would then refuse to release.
meta_set() {
    local key=$1 value=$2 tmp cur
    [ -f "$META" ] || return 1
    cur=$(sed -n 's/^anchor_pid=//p' "$META" | head -1)
    [ "$cur" = "$ANCHOR_PID" ] || return 1
    tmp="${LOCK_DIR}/.meta.$$"
    { grep -v "^${key}=" "$META" 2>/dev/null; echo "${key}=${value}"; } >"$tmp" 2>/dev/null || return 1
    mv -f "$tmp" "$META" 2>/dev/null || { rm -f "$tmp"; return 1; }
}

# 0 if the recorded anchor process is still running, 1 otherwise.
# True when a pid names a process that is genuinely still running.
#
# A SIGKILLed process whose parent has not wait()ed for it is a ZOMBIE:
# /proc/<pid> still exists and its start time still matches, so a liveness
# check built from those two facts alone reports HELD on a corpse forever.
# That is worst on the `run` path, which sets ttl=0 on the reasoning that its
# own pid is exact -- exact, and still resolving, to a dead process. Every
# agent harness here launches long commands without an immediate wait(), so
# this is the common shape, not the exotic one.
#
# But state Z is NOT proof of death. When a thread-group leader exits via
# pthread_exit() while other threads keep running, /proc/<tgid>/stat reports Z
# for a fully live process (`ps` shows `Zl ... <defunct>`). Treating that as
# dead reaps a LIVE holder mid-benchmark, which is the worse of the two errors
# by a wide margin. Threads: is signal->nr_threads -- non-reaped tasks in the
# group -- so a true zombie reads 1 and a live leader reads more. If status
# cannot be read, or reads as something that is not a number, we have no
# evidence of death and the process is treated as alive: there is deliberately
# no numeric default here, because a default is a constant nothing tests.
pid_is_live() {
    local pid=$1 info state threads
    [ -n "$pid" ] || return 1
    info=$(proc_state_and_start "$pid") || return 1
    state=${info%% *}
    [ "$state" = Z ] || return 0
    threads=$(awk '/^Threads:/ { print $2; exit }' "/proc/${pid}/status" 2>/dev/null)
    case "$threads" in
        '' | *[!0-9]*) return 0 ;;
        *) [ "$threads" -gt 1 ] ;;
    esac
}

# True when the lock's anchor is running but the lock cannot be VERIFIED,
# because it carries no start_time. Kept separate from holder_alive: for a
# lock that does have a start_time, a live pid with the wrong one is a
# recycled pid and must NOT count as alive.
unverifiable_live_anchor() {
    local a s
    a=$(meta_get anchor_pid) || return 1
    s=$(meta_get start_time) || s=""
    [ -n "$a" ] && [ -z "$s" ] || return 1
    pid_is_live "$a"
}

# Is the process identified by (pid, start_time) still running?
#
# One predicate, used for the lock's holder AND for the reaper guard's holder.
# The last time two call sites each decided this question for themselves, the
# answers disagreed and two of the four defects in #1830 came out of the gap.
anchor_alive() {
    local pid=$1 start=$2 info now
    [ -n "$pid" ] && [ -n "$start" ] || return 1
    pid_is_live "$pid" || return 1
    info=$(proc_state_and_start "$pid") || return 1
    now=${info##* }
    # A recycled pid has a different start time, so this is not just "does
    # some process with this number exist".
    [ "$now" = "$start" ]
}

holder_alive() {
    local pid start
    pid=$(meta_get anchor_pid) || return 1
    start=$(meta_get start_time) || return 1
    anchor_alive "$pid" "$start"
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
# A lock with no readable metadata is NOT evidence of a dead holder. The
# reachable cause is a lock caught MID-WRITE by a peer, or one damaged on
# disk; a future version that stops writing a field would be another. It is
# not a past version: start_time has been written by every revision since
# this script landed, so no release has ever produced an anchor without one.
# Refuse to reap such a lock until it is clearly abandoned, so that an
# unparseable lock degrades to "busy" rather than "free".
reapable() {
    # "Cannot verify liveness" must not be treated as "the holder is dead".
    # The grace below used to apply only to a ZERO-BYTE meta, so a lock with
    # content but no readable anchor_pid/start_time -- a lock caught mid-write
    # being the reachable way to get one -- skipped it, fell through to
    # `! holder_alive`, and had a LIVE holder's box stolen out from under it.
    # The protection did not cover its own rationale.
    local a s
    a=$(meta_get anchor_pid) || a=""
    s=$(meta_get start_time) || s=""
    # Use the evidence we actually have before falling back to a clock. If
    # the anchor pid is readable and that pid is running, the holder is not
    # demonstrably dead -- unverifiable, but not dead -- and the age grace
    # must not apply. The grace is 300s and the runs it protects are 40
    # minutes, so routing a live-but-unverifiable holder through it does not
    # prevent the theft, it schedules it.
    # Note the predicate is unverifiable_live_anchor and not `[ -d /proc/$a ]`:
    # a ZOMBIE anchor passes the directory test, and with ttl=0 -- the `run`
    # path's default -- holder_expired is never true, so a corpse would hold
    # the box forever with the bounding grace disabled on its behalf. A dead
    # anchor therefore falls through to the clock, which is what bounds it.
    if unverifiable_live_anchor; then
        # ...but do not void the holder's OWN declared expiry while doing it.
        # Returning unconditionally here replaced a bounded wedge with an
        # unbounded one: a lock with a live anchor and no readable start_time
        # would outlive any ttl forever and `wait` would block on a lock
        # everyone agrees has expired. The anti-theft property only needs the
        # CLOCK-BASED grace disabled, not the expiry the holder asked for.
        holder_expired
        return $?
    fi
    if [ ! -s "$META" ] || [ -z "$a" ] || [ -z "$s" ]; then
        local mtime now
        mtime=$(stat -c %Y "$LOCK_DIR" 2>/dev/null || echo 0)
        now=$(date +%s)
        [ "$((now - mtime))" -gt "$UNPARSEABLE_GRACE" ]
        return $?
    fi
    ! holder_alive && return 0
    holder_expired
}

# True when a lock is reapable only because it aged out of the grace, i.e.
# the script never established that the holder was dead. `stale_pid` is a
# positive claim and must not be made on this evidence.
reap_was_unverifiable() {
    local a s
    a=$(meta_get anchor_pid) || a=""
    s=$(meta_get start_time) || s=""
    [ ! -s "$META" ] || [ -z "$a" ] || [ -z "$s" ]
}

# ---------------------------------------------------------------------------
# The reaper guard
#
# `reap_if_dead` must be mutually exclusive: two acquirers racing on the same
# dead lock must not both conclude they reaped it and both start benchmarking.
# The guard that provides that exclusion used to be a bare `mkdir` with no
# anchor and no owner, which put the wedge this whole script exists to prevent
# INSIDE the mechanism that prevents it. One SIGKILL between the mkdir and the
# rmdir orphaned a directory that nothing could ever attribute, after which
# every genuinely dead lock was permanently un-reapable and `acquire` reported
# BUSY forever -- indistinguishable, to the operator, from legitimate
# occupancy. A benchmark box that can only be unwedged by hand is not a
# coordination mechanism, it is a second thing to page someone about.
#
# It is now the same construction as the lock itself, for the same reasons:
#
#   * PUBLISHED ATOMICALLY, complete with metadata, by staging a populated
#     directory and `mv -T`. There is no instant at which the guard exists
#     but is unattributable, so a kill cannot create one. rename(2) onto a
#     NON-EMPTY directory fails with ENOTEMPTY, which is what makes this a
#     test-and-set rather than a "last writer wins".
#   * OWNED, by pid + start_time, checked through the one `anchor_alive`
#     predicate. A dead owner's guard is reclaimed IMMEDIATELY by the next
#     contender -- no grace, no waiting, nothing to wedge behind.
#   * NEVER STOLEN FROM A LIVE OWNER. Age is not evidence of death. The old
#     rule cleared any guard older than REAPER_GRACE, so a reaper merely slow
#     -- a loaded box, a qemu run, a stalled NFS stat -- could have its guard
#     taken while it was still inside the critical section, producing exactly
#     the double reap the guard exists to prevent.
#   * RELEASED ONLY BY ITS OWNER, so a process whose guard was reclaimed
#     while it was descheduled cannot delete its successor's guard on the way
#     out.
#
# REAPER_GRACE survives as a backstop for one residual class only: a guard
# that exists but carries no readable anchor. Staging makes that unreachable
# for guards this script writes; it remains reachable for a stray directory
# at that path from an older version or another hand. For that class there is
# no liveness evidence at all, and age is the only instrument left.
# ---------------------------------------------------------------------------

reap_meta_get() {
    [ -f "$REAP_META" ] || return 1
    grep -q "^$1=" "$REAP_META" 2>/dev/null || return 1
    sed -n "s/^$1=//p" "$REAP_META" | head -1
}

# Clear the guard if -- and only if -- nothing is holding it.
# Returns 0 when the path is free to claim, 1 when someone else holds it.
reaper_clear_if_dead() {
    [ -d "$REAP_DIR" ] || return 0
    local pid start age now dead
    pid=$(reap_meta_get anchor_pid) || pid=""
    start=$(reap_meta_get start_time) || start=""
    if [ -n "$pid" ] && [ -n "$start" ]; then
        # Attributable. Liveness decides, and it decides both ways: a live
        # owner is never disturbed, a dead one is reclaimed on the spot.
        if anchor_alive "$pid" "$start"; then
            return 1
        fi
    else
        # Unattributable. No liveness evidence exists, so age is all there is.
        age=$(stat -c %Y "$REAP_DIR" 2>/dev/null || echo 0)
        now=$(date +%s)
        [ "$((now - age))" -gt "$REAPER_GRACE" ] || return 1
    fi
    # Rename before removing, so that two contenders reclaiming the same
    # abandoned guard cannot delete each other's pieces: the loser's rename
    # fails and it simply retries the claim below.
    dead="${REAP_DIR}.dead.$$"
    if mv -T "$REAP_DIR" "$dead" 2>/dev/null; then
        rm -rf "$dead"
    fi
    return 0
}

reaper_claim() {
    local start stage
    start=$(proc_start_time "$$") || return 1
    stage="${REAP_DIR}.stage.$$"
    rm -rf "$stage"
    mkdir -p "$stage" 2>/dev/null || return 1
    {
        echo "anchor_pid=$$"
        echo "start_time=${start}"
        echo "owner=${OWNER:-?}"
        echo "claimed_epoch=$(date +%s)"
    } >"${stage}/meta" 2>/dev/null || { rm -rf "$stage"; return 1; }
    if mv -T "$stage" "$REAP_DIR" 2>/dev/null; then
        return 0
    fi
    rm -rf "$stage"
    return 1
}

# Release only what we still own. If our guard was reclaimed while we were
# descheduled, the directory at that path now belongs to a successor who may
# be mid-reap; deleting it would hand the box to two agents at once.
#
# Ownership is pid AND start time, not pid alone, for the same reason the
# lock's is: this box is at ~1.5M pids after four days, so a successor that
# happens to land on our number is a recycled pid, not us. Checking only $$
# here while `reaper_clear_if_dead` checks both would leave the asymmetry
# exactly where the consequence is worst -- deleting a live successor's
# guard rather than merely failing to clean up our own.
reaper_release() {
    local pid start dead
    pid=$(reap_meta_get anchor_pid) || return 0
    [ "$pid" = "$$" ] || return 0
    start=$(reap_meta_get start_time) || return 0
    [ "$start" = "$(proc_start_time "$$")" ] || return 0
    dead="${REAP_DIR}.rel.$$"
    if mv -T "$REAP_DIR" "$dead" 2>/dev/null; then
        rm -rf "$dead"
    fi
}

# Remove a reapable lock, under the guard above.
reap_if_dead() {
    reaper_clear_if_dead || return 1
    reaper_claim || return 1
    # Test seam, inert unless set: hold the critical section open so the
    # conformance suite can kill this process INSIDE it deterministically,
    # which is the only way to test that an orphaned guard is recoverable
    # without waiting REAPER_GRACE for a class that no longer uses it.
    if [ -n "${HOSTLOCK_REAPER_STALL:-}" ]; then
        echo "$$" >"${REAP_DIR}/stalled_pid" 2>/dev/null
        sleep "$HOSTLOCK_REAPER_STALL"
    fi
    local reaped=1
    if [ -d "$LOCK_DIR" ] && reapable; then
        local pid owner
        pid=$(meta_get anchor_pid 2>/dev/null || echo '?')
        owner=$(meta_get owner 2>/dev/null || echo '?')
        if holder_alive || unverifiable_live_anchor; then
            echo "hostlock: WARNING taking over a lock held by ${owner} (pid ${pid}, still alive) after its $(meta_get ttl)s TTL expired" >&2
            echo "hostlock: WARNING if ${owner} is still benchmarking, both sets of numbers are now suspect" >&2
        else
            echo "hostlock: reaping stale lock from dead pid ${pid} (owner ${owner})" >&2
        fi
        remove_lock
        reaped=0
    fi
    reaper_release
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
    local start stage uid
    start=$(proc_start_time "$ANCHOR_PID") || die "cannot read start time of anchor pid ${ANCHOR_PID}"
    # `owner` is self-declared free text and corroborates nothing -- anyone can
    # pass --owner roy. `anchor_uid` is read from the kernel, so a row can at
    # least say which account really holds the box. This is an ADVISORY lock;
    # the identity is for attribution in a published row, not enforcement.
    uid=$(awk '/^Uid:/ { print $2; exit }' "/proc/${ANCHOR_PID}/status" 2>/dev/null || echo "")
    stage="${LOCK_DIR}.stage.$$"
    rm -rf "$stage"
    mkdir -p "$stage" || die "cannot create staging dir ${stage}"
    {
        echo "anchor_pid=${ANCHOR_PID}"
        echo "start_time=${start}"
        echo "anchor_uid=${uid}"
        echo "script_pid=$$"
        echo "owner=${OWNER}"
        echo "reason=${REASON}"
        echo "acquired_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "acquired_epoch=$(date +%s)"
        echo "ttl=${TTL}"
        echo "runnable_at_acquire=$(runnable_now)"
        echo "takeover=none"
        echo "gate=${GATE:+requested:${GATE}}"
    } >"${stage}/meta"
    if mv -T "$stage" "$LOCK_DIR" 2>/dev/null; then
        return 0
    fi
    rm -rf "$stage"
    return 1
}

# One state word, for scripts: FREE, HELD, EXPIRED or STALE.
# The four states, and nothing else -- FREE, HELD, STALE, EXPIRED.
#
# STALE is a claim about the world ("the holder is dead, the next acquire
# will reap it"), so it must not be reported for a lock this script is
# actually refusing to reap. An unverifiable lock inside its grace window
# reads STALE by the liveness test and BUSY by `acquire`, and of those two
# the reassuring one is STALE: it tells a human the box is abandoned, and
# they take it. Report what the tool will actually do.
lock_state() {
    [ -d "$LOCK_DIR" ] || { echo FREE; return 0; }
    # unverifiable_live_anchor is the class this tool cannot verify but can
    # SEE running. Routing it through the STALE arm made `status` say "is
    # gone" about a pid the reaper had just confirmed present -- the same
    # false-abandonment message the HELD and EXPIRED arms were fixed for.
    if holder_alive || unverifiable_live_anchor; then
        if holder_expired; then echo EXPIRED; else echo HELD; fi
    elif reapable; then
        echo STALE
    else
        echo HELD
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

# Emit the lock state as fields to be recorded WITH each measurement, not
# just printed to a console nobody keeps.
#
# Every contaminated run this tool exists to prevent was caught, when it was
# caught at all, because somebody happened to have an A/A null arm. A row that
# carries who held the box and what the runnable count was is self-describing
# weeks later, when the console scrollback is gone and the only thing left is
# a table in a merged document. This is the same lesson as reporting
# `decode_width realized=16 as_requested` while placing those 16 workers on 8
# physical cores: the number that was reported was not the number that was
# wrong, and the one that mattered was never emitted at all.
#
# `contended` is only answered when the caller supplies the expectation it
# should be judged against (`--expect-runnable N`), because there is no honest
# universal threshold -- a bounded 4-of-32-CPU neighbour and a 32-way build
# are both "runnable > 1" and only one of them ruins a measurement. With no
# expectation the field is `unknown`, which is a fact, unlike a guess.
cmd_provenance() {
    local state owner pid age reason takeover gate r r_acq contended uid
    state=$(lock_state)
    r=$(runnable_now)
    owner=none ; pid=none ; age=unknown ; reason='' ; takeover=unknown ; gate=unknown
    r_acq=unknown ; uid=unknown
    if [ -d "$LOCK_DIR" ]; then
        owner=$(meta_get owner) || owner=unknown
        pid=$(meta_get anchor_pid) || pid=unknown
        # `held_by` is whatever the holder typed. `held_uid` came from the
        # kernel. When a published row has to be trusted months later, they
        # are not the same kind of fact and should not look like it.
        uid=$(meta_get anchor_uid) || uid=unknown
        reason=$(meta_get reason) || reason=''
        # An absent key stays `unknown`. A lock published by an older
        # version of this script has neither of these, and answering `none`
        # for it asserts "no takeover, no gate" about a run that may have
        # reaped a corpse and abandoned its gate -- moving the very
        # mislabelling this subcommand exists to prevent out of the console
        # and into the data, where it outlives the person who could correct it.
        takeover=$(meta_get takeover) || takeover=unknown
        gate=$(meta_get gate) || gate=unknown
        # `acquired_epoch` absent makes lock_age default it to 0, i.e. the
        # current epoch -- a 56-year age that looks like a datum if anyone
        # aggregates the column.
        if meta_get acquired_epoch >/dev/null; then age=$(lock_age); fi
        # publish_lock already sampled occupancy when the window OPENED, and
        # dropping it left the row with a single sample taken after the window
        # CLOSED -- the moment the row was written, not the interval it
        # describes. Emitting both lets a reader bracket the measurement for
        # the cost of one line.
        r_acq=$(meta_get runnable_at_acquire) || r_acq=unknown
    fi
    contended=unknown
    if [ -n "$EXPECT_RUNNABLE" ]; then
        if [ "$r" -le "$EXPECT_RUNNABLE" ]; then contended=no; else contended=yes; fi
    fi
    # EXPIRED means the TTL lapsed, NOT that the box is unclaimed: the holder
    # can be alive and mid-benchmark. With a 3600s default TTL and an anchor
    # that is an agent session alive for days, a live holder past its TTL is
    # this design's steady state, not an edge case. `declared` is the one
    # boolean a reader will filter on, so answering `no` for a claimed, busy
    # box is the reassuring direction and the wrong one. The lapse is not
    # hidden -- it is in hostlock_state.
    local declared=no
    case "$state" in HELD | EXPIRED) declared=yes ;; esac
    local fields=(
        "hostlock_state=${state}"
        "declared=${declared}"
        "held_by=${owner:-none}"
        "held_uid=${uid:-unknown}"
        "held_pid=${pid:-none}"
        "held_secs=${age}"
        "takeover=${takeover:-none}"
        "gate=${gate:-none}"
        "runnable_at_acquire=${r_acq:-unknown}"
        "runnable=${r}"
        "contended=${contended}"
        "sampled_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    )
    if [ "$ONELINE" = 1 ]; then
        # `reason` is deliberately NOT in the one-line form. It is free text
        # written by whoever held the lock, it is unquoted and unterminated
        # among space-separated fields, and the shared fixed path means the
        # text comes from a peer. A two-word reason silently truncates the
        # field; anything shell-active is worse. Multi-line output carries it
        # on its own line, where a newline is the delimiter.
        echo "${fields[*]}"
    else
        printf '%s\n' "${fields[@]}"
        printf 'reason=%s\n' "${reason:-}"
    fi
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
    # Drive the human branch from lock_state, not from holder_alive directly.
    # Keying off holder_alive made this print "STALE ... is gone; next acquire
    # will reap it" for a lock that lock_state calls HELD and that acquire
    # refuses to take -- and cmd_acquire dumps this text on the BUSY path, so
    # the contradiction is exactly what a blocked agent reads. Telling a human
    # the box is abandoned is how the box gets taken.
    case "$(lock_state)" in
        HELD)
            echo "HELD by ${owner} pid=${pid} for ${age}s since ${at}"
            echo "  reason: ${reason}"
            echo "  runnable=$(runnable_now)"
            ;;
        EXPIRED)
            echo "EXPIRED (held ${age}s > ttl ${ttl}s; next acquire will take it over) by ${owner} pid=${pid} for ${age}s since ${at}"
            echo "  reason: ${reason}"
            echo "  runnable=$(runnable_now)"
            ;;
        STALE)
            echo "STALE  holder ${owner} pid=${pid} is gone; next acquire will reap it"
            echo "  reason: ${reason}"
            echo "  runnable=$(runnable_now)"
            ;;
        *)
            # lock_state saw no lock dir; it was released under us since the
            # test above. Report what is true now rather than a stale label.
            echo "FREE  (runnable=$(runnable_now))"
            ;;
    esac
    return 0
}

# Give back a lock we took but are not going to use.
#
# `remove_lock_if_mine`, never `remove_lock`: by the time an acquire decides to
# abandon, our own TTL may have lapsed and a successor may legitimately hold
# the lock -- which is not hypothetical, it is what happens whenever --ttl is
# shorter than --gate-timeout. Unconditional removal would delete a live
# holder's lock while their benchmark runs, and it satisfies every "the lock
# is FREE afterwards" assertion identically, so the guard has to be here and
# tested here rather than duplicated at each call site.
abandon_lock() {
    remove_lock_if_mine
}

cmd_acquire() {
    local deadline=$((SECONDS + TIMEOUT)) rc
    while :; do
        if publish_lock; then
            meta_set takeover "$TAKEOVER"
            # Refuse BEFORE gating. The other order holds the reaped lock --
            # the one we were told to refuse -- for the whole --gate-timeout
            # (default 900s), blocks every other acquirer for that long, and
            # then reports the gate failure instead. The two are causally
            # linked: the dead holder's orphaned benchmark is the likeliest
            # reason the gate cannot be met, so the case where --strict-reap
            # has something to say is exactly the case that suppressed it.
            if [ "$TAKEOVER" != none ] && [ "$STRICT_REAP" = 1 ]; then
                echo "hostlock: outcome=reap_refused (${TAKEOVER}) by ${OWNER}"
                echo "hostlock: --strict-reap given: refusing an acquire that required reaping a ${TAKEOVER} lock." >&2
                echo "hostlock: the previous holder's load may still be running; check the host before retrying." >&2
                abandon_lock
                return 4
            fi
            gate_on_runnable
            rc=$?
            if [ "$rc" -ne 0 ]; then
                abandon_lock
                return "$rc"
            fi
            # An acquire that had to reap is NOT the same event as one that
            # found the box free: something died holding it, and whatever it
            # was running may still be on the cores (reclaiming a lock does
            # not stop the load). Report it as its own outcome so a caller
            # can decide, rather than folding it into a silent success.
            if [ "$TAKEOVER" != none ]; then
                echo "hostlock: outcome=acquired_after_reap (${TAKEOVER}) by ${OWNER} (anchor pid ${ANCHOR_PID})${REASON:+ — ${REASON}}"
            else
                echo "hostlock: outcome=acquired by ${OWNER} (anchor pid ${ANCHOR_PID})${REASON:+ — ${REASON}}"
            fi
            return 0
        fi
        # Existing lock: reap it if its anchor died OR it outlived its TTL,
        # then retry immediately. Testing `! holder_alive` here instead of
        # `reapable` silently disabled TTL expiry altogether -- status would
        # report EXPIRED and the next acquirer would still be told BUSY.
        # TAKEOVER has to survive from the iteration that reaps to the one
        # that publishes, so it cannot be reset at the top of the loop. It
        # must still be dropped the moment it stops describing us: if a peer
        # won the publish race after our reap and now holds the lock legitimately,
        # our eventual acquire is a CLEAN one. Carrying the stale value would
        # label it acquired_after_reap and, under --strict-reap, abort an
        # unattended harness over a takeover that did not happen.
        if reapable; then
            if holder_alive; then
                TAKEOVER=ttl_expired
            elif reap_was_unverifiable; then
                # The lock aged out of the grace with unreadable metadata. We
                # never established the holder was dead, so `stale_pid` would
                # be a claim the script has no evidence for -- the same
                # mislabelling `takeover=unknown` exists to prevent, one level
                # down.
                TAKEOVER=unverifiable
            else
                TAKEOVER=stale_pid
            fi
            if reap_if_dead; then continue; fi
            TAKEOVER=none
        else
            TAKEOVER=none
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
#
# The gate is a SECONDARY signal and deliberately not the admission decision.
# The lock is declared intent and is the thing to trust; the runnable count is
# a physical fact about the box that answers a different question. Runnable is
# a poor admission test on its own in both directions: a well-behaved neighbour
# deliberately bounded to 4 of 32 CPUs shows runnable 4-5 and fails `--gate 3`,
# while a single-threaded process pinning one core at 100% shows ~1 and sails
# through. Use the lock to decide, the gate to drain stragglers, and record
# both (see `provenance`).
#
# On timeout this FAILS by default, and that default is the whole point. A gate
# that warns and proceeds returns the same status for a satisfied precondition
# and an abandoned one, so every row it emits is labelled gated either way --
# it launders contamination into a label, which is worse than not gating at
# all. `--on-gate-timeout proceed` is available, but it has to be asked for,
# and it is recorded in the lock metadata so the resulting rows can say so.
gate_on_runnable() {
    [ -n "$GATE" ] || return 0
    local deadline=$((SECONDS + GATE_TIMEOUT)) r
    while :; do
        r=$(runnable_now)
        if [ "$r" -le "$GATE" ]; then
            [ "$SECONDS" -gt 0 ] && echo "hostlock: host quiet (runnable=${r} <= ${GATE})"
            meta_set gate "satisfied:${r}<=${GATE}"
            return 0
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            if [ "$ON_GATE_TIMEOUT" = proceed ]; then
                echo "hostlock: WARNING gate timed out after ${GATE_TIMEOUT}s; proceeding at runnable=${r} (wanted <= ${GATE}) because --on-gate-timeout proceed was given. Treat results from this run as suspect." >&2
                meta_set gate "timed_out_proceeded:${r}>${GATE}"
                return 0
            fi
            echo "hostlock: gate NOT satisfied after ${GATE_TIMEOUT}s: runnable=${r}, wanted <= ${GATE}" >&2
            echo "hostlock: releasing the lock and failing rather than measuring on a busy host." >&2
            echo "hostlock: pass --on-gate-timeout proceed to measure anyway (it will be recorded)." >&2
            return 5
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
    # Verify the start time before signalling. The window between the child
    # exiting and this trap firing is small, but pids on this box are cycling
    # at ~1.5M in four days, so "small" is not "empty" -- and this is the one
    # place the script signals anything at all. If the pid has been recycled,
    # leave it alone and just release the lock.
    if [ -n "$child" ] && [ -n "$RUN_CHILD_START" ]; then
        local now
        now=$(proc_start_time "$child" 2>/dev/null || echo "")
        if [ "$now" = "$RUN_CHILD_START" ]; then
            kill -TERM "$child" 2>/dev/null
            wait "$child" 2>/dev/null
        fi
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
    local child rc wall0 wall1 cpu0 cpu1
    wall0=$(wall_now)
    "$@" &
    child=$!

    # Traps first. Reading the child's start time forks, and a signal arriving
    # in that window would take the script's default action -- leaking the
    # lock and orphaning the benchmark, the two failures this block exists to
    # prevent. run_teardown tolerates an empty RUN_CHILD_START by declining to
    # signal, which is the safe direction for a window this narrow.
    trap 'run_teardown "$child" SIGINT 130' INT
    trap 'run_teardown "$child" SIGTERM 143' TERM
    trap 'run_teardown "$child" SIGHUP 129' HUP
    RUN_CHILD_START=$(proc_start_time "$child" 2>/dev/null || echo "")

    # Baseline the child-CPU counter as late as possible: it accumulates only
    # when a child is REAPED, so every fork above (the start-time read, the
    # acquire's own helpers) is already folded into this reading and cancels
    # out of the difference.
    read_children_cpu_ticks || CPU_TICKS=""
    cpu0=$CPU_TICKS
    wait "$child"
    rc=$?
    read_children_cpu_ticks || CPU_TICKS=""
    cpu1=$CPU_TICKS
    wall1=$(wall_now)
    trap - INT TERM HUP
    remove_lock_if_mine
    echo "hostlock: released (command exit ${rc})"
    # A failing command's own status always wins: it is the more important
    # signal, and reporting "the cores were busy" for a run that crashed would
    # bury the crash. Only a command that SUCCEEDED can be overridden, and
    # only when --min-efficiency was explicitly asked for.
    if ! check_cpu_efficiency "$cpu0" "$cpu1" "$wall0" "$wall1" && [ "$rc" -eq 0 ]; then
        rc=6
    fi
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
ON_GATE_TIMEOUT=fail
STRICT_REAP=0
TAKEOVER=none
EXPECT_RUNNABLE=""
ONELINE=0
RUN_CHILD_START=""
EXPECT_CORES=""
MIN_EFFICIENCY=""
CPU_TICKS=""

[ "$#" -ge 1 ] || {
    awk '/^# \(end of usage summary\)/ { exit } NR >= 3' "$0" >&2
    exit 1
}
SUB=$1
shift

# Every flag that takes a value must be given one, and numeric flags must be
# numeric.
#
# `--gate` with a missing value is not a typo that costs a warning: `${2:-}`
# supplies a default, so validation passes, and then `shift 2` with one
# argument left FAILS SILENTLY (there is no `set -e`). `$#` never decreases
# and this loop spins with no syscall in it -- a core pinned at 100% forever,
# on the one box whose contention this script exists to control.
#
# `--gate abc` used to cost a warning and proceed. Now that the gate fails
# closed it holds the lock for the full --gate-timeout and exits 5, and
# `--gate-timeout abc` evaluates to 0 inside $((SECONDS + GATE_TIMEOUT)),
# quietly turning the gate into "give up immediately".
require_uint() {
    case "$2" in
        '' | *[!0-9]*) die "$1 takes a non-negative integer, got: '$2'" ;;
    esac
}

require_ufloat() {
    case "$2" in
        '' | *[!0-9.]* | *.*.* | .) die "$1 takes a non-negative number, got: '$2'" ;;
    esac
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --reason | --owner | --timeout | --gate | --gate-timeout | --ttl | --pid | --on-gate-timeout | --expect-runnable | --expect-cores | --min-efficiency)
            [ "$#" -ge 2 ] || die "$1 requires a value"
            ;;
    esac
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
            TIMEOUT=$2
            require_uint "$1" "$TIMEOUT"
            shift 2
            ;;
        --gate)
            GATE=$2
            require_uint "$1" "$GATE"
            shift 2
            ;;
        --gate-timeout)
            GATE_TIMEOUT=$2
            require_uint "$1" "$GATE_TIMEOUT"
            shift 2
            ;;
        --ttl)
            TTL=$2
            require_uint "$1" "$TTL"
            shift 2
            ;;
        --pid)
            ANCHOR_PID=$2
            require_uint "$1" "$ANCHOR_PID"
            shift 2
            ;;
        --porcelain)
            PORCELAIN=1
            shift
            ;;
        --on-gate-timeout)
            ON_GATE_TIMEOUT=$2
            case "$ON_GATE_TIMEOUT" in
                fail | proceed) ;;
                *) die "--on-gate-timeout takes 'fail' or 'proceed', got: ${ON_GATE_TIMEOUT}" ;;
            esac
            shift 2
            ;;
        --strict-reap)
            STRICT_REAP=1
            shift
            ;;
        --expect-runnable)
            EXPECT_RUNNABLE=$2
            require_uint "$1" "$EXPECT_RUNNABLE"
            shift 2
            ;;
        --expect-cores)
            EXPECT_CORES=$2
            require_uint "$1" "$EXPECT_CORES"
            [ "$EXPECT_CORES" -ge 1 ] || die "--expect-cores takes a positive integer, got: '$EXPECT_CORES'"
            shift 2
            ;;
        --min-efficiency)
            MIN_EFFICIENCY=$2
            require_ufloat "$1" "$MIN_EFFICIENCY"
            shift 2
            ;;
        --oneline)
            ONELINE=1
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
    # A threshold with no denominator cannot be evaluated, and defaulting the
    # denominator to 1 would pass every multi-threaded run. Fail at parse time
    # rather than at the end of a forty-minute benchmark.
    if [ -n "$MIN_EFFICIENCY" ] && [ -z "$EXPECT_CORES" ]; then
        die "--min-efficiency requires --expect-cores N (how many cores the command should keep busy)"
    fi
else
    # These two do nothing outside `run`, which has no wrapped command to
    # measure. Accepting them silently is how a knob comes to be believed in
    # while being inert -- the exact defect filed against the EP's affinity
    # environment variable, where every setting produced identical placement.
    if [ -n "$EXPECT_CORES" ] || [ -n "$MIN_EFFICIENCY" ]; then
        die "--expect-cores/--min-efficiency apply to \`run\` only; ${SUB} has no command to measure"
    fi
    : "${ANCHOR_PID:=$PPID}"
    : "${TTL:=3600}"
fi
[ -d "/proc/${ANCHOR_PID}" ] || die "anchor pid ${ANCHOR_PID} is not running"

case "$SUB" in
    status) cmd_status ;;
    provenance) cmd_provenance ;;
    acquire) cmd_acquire ;;
    release) cmd_release ;;
    wait) cmd_wait ;;
    run) cmd_run "$@" ;;
    *) die "unknown subcommand: ${SUB}" ;;
esac
