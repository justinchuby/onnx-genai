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
# WHERE THE LOCK LIVES
#   Default: /tmp/onnx-genai-hostlock, and everyone on the box must resolve
#   the same path or the lock coordinates nothing.
#
#   To move it (host with an unwritable, noexec, or per-service /tmp), write a
#   MACHINE-LOCAL config, which every invocation by every agent reads:
#       ~/.config/onnx-genai/hostlock.conf     (or $XDG_CONFIG_HOME, or
#                                               $HOSTLOCK_CONF)
#       lock_dir=/var/lib/onnx-genai/hostlock
#   Absolute paths only; the file is parsed, never sourced. While a config is
#   in effect, `acquire`/`run` still consult the old /tmp path READ-ONLY and
#   refuse (exit 2) while a live holder is there, because a peer who has not
#   re-read the config cannot see the new lock.
#
#   $HOSTLOCK_DIR also overrides the path, and is NOT the same thing: it is
#   per process, so it does not move peers with you. It is a PRIVATE lock,
#   which acquires instantly every time and collides with nobody, and every
#   invocation says so on stderr unless HOSTLOCK_PRIVATE_OK=1 acknowledges it.
#   `status --porcelain` and `provenance` emit lock_dir/lock_scope/
#   lock_dir_source, so a recorded row says which lock its `declared=yes` is
#   a claim about.
#
# Usage:
#   hostlock.sh status [--porcelain]    # who holds it, is that holder alive
#   hostlock.sh acquire [opts]          # take it, or fail / wait
#   hostlock.sh release                 # give it back (only if you hold it)
#   hostlock.sh wait [--timeout S]      # block until free, do not take it
#   hostlock.sh run --reason TEXT [opts] -- CMD...
#                                       # acquire, run CMD, always release
#   hostlock.sh provenance [opts]       # lock state as fields, to record WITH
#                                       # each measured row (see below)
#
# Options for acquire/run:
#   --reason TEXT     what you are running (shown to whoever is blocked).
#                     REQUIRED, and must be non-empty, for `run`; optional for
#                     `acquire`. Defaults to $HOSTLOCK_REASON. `run` holds the
#                     host while its owner is elsewhere, so this is the only
#                     channel that can tell whoever it blocks what they are
#                     waiting for -- and unlike an announcement it survives
#                     the announcer's death.
#   --owner NAME      defaults to $HOSTLOCK_OWNER, else $USER.
#                     `run` exports a DECLARED owner (--owner, or an inherited
#                     $HOSTLOCK_OWNER) to the wrapped command, so a harness
#                     that reads the lock can recognise its own. A $USER
#                     default is never exported: every agent on a shared box
#                     runs as the same unix user, so that would make every
#                     lock read as ours. See #1929.
#   --wait            block until free instead of failing immediately
#   --timeout S       give up after S seconds (default 3600). Only `wait`,
#                     `run`, and `acquire --wait` ever enter a wait loop, so
#                     passing it anywhere else is refused rather than ignored.
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
#                     REFUSED on `run` when finite (exit 1); `--ttl 0` is the
#                     `run` default and is accepted, because 0 means "never
#                     expires" and so arms no takeover at all. A TTL does not mean "release
#                     this if I abandon it" -- it means "release this on the
#                     clock, whether or not I am still running", and the
#                     takeover path fires on a holder that is verifiably alive.
#                     So a finite TTL on a long job hands the host to a second
#                     measurer mid-run and contaminates BOTH sets of numbers.
#                     `run` cannot leak a lock in the first place: it anchors
#                     to its own pid and start time, which die on every exit
#                     path, and a zombie anchor is caught by `pid_is_live`.
#                     Nor can it declare the box free while its own job runs:
#                     a SIGKILLed run cannot reap its tree, so liveness also
#                     consults the wrapped command's process group and the
#                     lock stays held until that group is empty.
#                     To bound the JOB rather than the CLAIM, bound the
#                     process tree -- setsid + process group + a hard
#                     `timeout` + a verified reap (`pgrep -g`). A lock TTL
#                     bounds neither: it would relabel the host as free while
#                     the runaway kept burning cores.
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
# The measurement line names its own unit, because the ratio is a different
# quantity depending on whether you gave a denominator:
#
#   cpu wall=..s cpu=..s cores_expected=unspecified efficiency_cores=8.718 ...
#   cpu wall=..s cpu=..s cores_expected=8           efficiency_frac=1.090 ...
#
# `efficiency_cores` is CPU-seconds per wall second (0..ncpu); `efficiency_frac`
# is the fraction of --expect-cores actually held (0..1 when healthy). They
# differ by a factor of N for the same run, so never compare one against the
# other, and grep for the specific field rather than for `efficiency`.
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
#   7 lock dir unusable -- no lock can be created at LOCK_DIR on this host, so
#     the answer is neither "yours" nor "somebody else's". It is separate from
#     1 because a misconfigured or unwritable host is not a bad argument and
#     must not be retried as one, and separate from 2 and 3 because those two
#     assert a peer holds the box. See lock_dir_problem.
#   8 unsupported platform -- not Linux, or /proc is unreadable, so process
#     liveness cannot be established at all. Separate from 7: a usable lock
#     directory is not the problem, the host is. No lock is ever taken on this
#     path, so the caller can treat it as "this box cannot participate" rather
#     than as contention. See require_supported_platform.
#   9 nested against our own holder -- this process is already inside a `run`
#     holding THIS lock, so the holder we would wait for cannot release until
#     we return. Separate from 2 and 3 for the same reason 7 is: those say a
#     peer has the box, and reporting contention for a lock we ourselves hold
#     sends the caller looking for a co-tenant who does not exist (#1977).
#     See nested_under_own_run.
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
# --min-efficiency measures something closer to the thing itself than the
# runnable count is: a process that owns its cores spends ~1.00 CPU-seconds
# per core per wall second, and anything less means it did not have them,
# whatever the runnable count said at either end. It integrates the whole
# interval instead of sampling its ends, and it took that same collection from
# a 52% A/A null to 0.04-0.56%.
#
# It is still SUPPLEMENTARY, and it does not license running unlocked. What it
# measures is how much of the wall clock the process spent SCHEDULED, which is
# not how much work it got done, and there are two ordinary ways to lose the
# second without losing the first:
#
#   An SMT sibling never deschedules you. A competitor on the other hyperthread
#   shares the front end and the execution ports, so you keep running,
#   efficiency sits at ~1.00, and throughput falls anyway.
#
#   Neither does a neighbour off-core. Memory bandwidth, LLC occupancy and
#   turbo headroom are shared box-wide; a process saturating DRAM elsewhere
#   slows you without ever touching your runqueue.
#
# The A/A null has the mirror-image blind spot: it is a VARIANCE measurement,
# so contention that is steady across both arms depresses both equally, the
# null comes out small, and the published ratio is a ratio of two equally
# contaminated numbers. Neither instrument can establish exclusivity, because
# exclusivity is a declaration, not a measurement -- which is what the lock is
# and why it is the one that is mandatory.
#
# It measures "did not have the cores", which is NOT the same claim as
# "somebody stole them": a benchmark that sleeps, blocks on I/O, or leaves a
# deliberate inter-token gap is legitimately below 1.0 and is not contended.
# Only you know which your workload is, which is why the threshold is yours to
# set and why there is no default. Set it from a measured quiet-host run, not
# from an ideal. The two compose -- the gate decides whether to start, this
# decides whether to disbelieve -- and the credit for the technique is Roy's.
#
#   hostlock.sh run --owner leon --reason "moe 6-width matrix" \
#       --expect-cores 16 --min-efficiency 0.90 -- ./bench.sh
#
# CPU time comes from this shell's own reaped-children accounting
# (/proc/self/stat cutime+cstime), so it covers the wrapped command and every
# descendant it waited for, and costs nothing: no sampling, no extra process.
#
# BECAUSE it covers the whole child tree, put `taskset` OUTERMOST:
#
#   hostlock.sh run --reason "..." -- taskset -c 16-23 cargo test        # yes
#   hostlock.sh run --reason "..." -- bash -c '... taskset -c 16-23 cargo test ... | grep'   # NO
#
# In the second form the bound covers `cargo` and its descendants, while the
# outer `bash` and the `grep` draining its output are unbound and still fully
# counted. The measurement is then a SUPERSET of what the bound constrains, so
# the reported figure can and does exceed the bound -- a real run of the
# second form measured 8.718 cores against a `taskset -c 16-23` 8-cpu bound.
# Nothing leaked: the bound held, and the number was correctly measured over a
# wider set of processes than the bound applied to.
#
# This matters because of the conclusion it invites. A figure above the bound
# reads as "affinity escaped" or "the tool is broken", and on this host the
# first of those was raised as an alarm and then retracted. When CPU exceeds
# the bound there are three candidates, not two: the bound leaked, the bound
# was never applied, or -- this one -- the bound was applied to a subset of
# what was measured. Only the third leaves both the bound and the instrument
# blameless, and it is the only one you can fix by moving a word.
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
# reaped automatically by the next acquirer -- unless the job it started
# outlived it, in which case the lock stays held until that job is gone. The
# anchor is the CLAIM; the wrapped command's process group is the JOB, and
# only the job holds cores. `run` publishes `child_pgid` for exactly this, and
# a SIGKILLed run leaves both a dead anchor and a live tree, because the one
# signal it cannot trap is also the one that stops it reaping. See
# orphan_group_pids. The start time is what makes
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

die() {
    echo "hostlock: $*" >&2
    exit 1
}

# The options are already documented, exhaustively, in this file's header --
# but only to someone who opens a 2000-line shell script. Four agents spent a
# night arguing to build this tool while it sat in scripts/ with a test suite,
# and `--help` answering "unknown subcommand" is a large part of why. So print
# the header rather than a second copy of it: a curated duplicate would drift,
# and the drift would land on exactly the flags people get wrong (`--wait` is
# a bare flag, `--reason` is mandatory for `run`).
#
# The end anchor is `# (end of usage summary)`, the same one the no-argument
# path uses -- deliberately not a second boundary of my own. That marker exists
# because an earlier `sed -n '3,50p'` here silently stopped covering the header
# as it grew, so flags added by a change were absent from that change's own
# help. A private anchor would reintroduce exactly that drift, one copy along.
# Both anchor lines are load-bearing; renaming either breaks help, loudly, in
# the conformance suite.
#
# Defined here, next to `die`, rather than beside its `cmd_*` siblings, for one
# reason: everything below refuses, and help must outrank every refusal. See
# the interception directly beneath this function.
cmd_help() {
    sed -n '/^# Usage:/,/^# (end of usage summary)/p' "$0" | sed '$d' | sed 's/^#\{1\} \{0,1\}//'
    echo "Full notes, including how to read a gated row: $0"
}

# Answer help before anything is validated -- and "anything" is meant
# literally, which is why this sits at the top of the file and not at the
# dispatch `case` where a subcommand would naturally be handled.
#
# Four separate gates below can refuse, and every one of them refuses in
# exactly the situation that sends someone to `--help`:
#
#   * `require_supported_platform` (below) exits 8 on a host without /proc --
#     macOS, native Windows, a stripped container. "What does this tool need?"
#     is the question those users have, and the answer is in the header.
#   * the lock_dir config resolution dies on a relative or empty `lock_dir` --
#     the mistake made by someone following the header's own instructions for
#     moving the lock, who then cannot read those instructions to see it.
#   * `require_name` rejects a $HOSTLOCK_OWNER or $USER containing a space.
#   * the anchor-pid check rejects a stale `--pid`.
#
# Asking how the tool works must never depend on holding it correctly, so the
# only thing allowed to run first is the thing that prints the answer.
# `cmd_help` needs no state: `$0` and `sed`, both available at this point.
case "${1:-}" in
    help|-h|--help) cmd_help; exit 0 ;;
esac

# Refuse a platform this script cannot make its liveness claims on.
#
# The PORTABILITY note above says Linux only. Until now that was a comment,
# and a comment does not stop anything: on a host without /proc every
# `proc_start_time` call returns empty, so every lock would be published with
# an empty `start_time` and the whole file would run permanently in the
# degraded no-start_time mode -- the grace-window path meant for the rare
# unreadable anchor, silently promoted to the normal case. That mode cannot
# tell a recycled pid from the original, which is the one error this script
# must never make. Failing is strictly better than holding a lock that cannot
# be trusted, so this is a hard refusal with its own exit code rather than a
# warning.
#
# The probe is a capability check, not a name check: `uname` says Linux on
# WSL, which is genuinely fine, and a stripped container without /proc says
# Linux while being unusable. Both are decided correctly by asking whether the
# files this script actually reads are there.
#
# Windows (git-bash reports MINGW64_NT) and macOS (Darwin) therefore exit 8
# with a message naming the missing capability, rather than failing later and
# obscurely inside an arithmetic or `stat -c` error.
require_supported_platform() {
    local sys
    sys=$(uname -s 2>/dev/null || echo unknown)
    case "$sys" in
        Linux) ;;
        *)
            echo "hostlock: unsupported platform '${sys}' -- this lock needs" \
                 "Linux /proc for its liveness claims, and degrading them" \
                 "would let it reap a live holder. No lock was taken." >&2
            exit 8
            ;;
    esac
    if [ ! -r /proc/self/stat ] || [ ! -r /proc/loadavg ]; then
        echo "hostlock: /proc is not readable on this host, so process" \
             "liveness cannot be verified. No lock was taken." >&2
        exit 8
    fi
}
require_supported_platform

# WHERE THE LOCK LIVES, and why it is not simply $TMPDIR or simply a constant.
#
# A whole-host lock is only worth anything if every agent on the box resolves
# it to the same directory. TMPDIR is per-session: one agent with it set would
# get a private lock, acquire it instantly every time, and never once collide
# with anybody -- coordination that silently does nothing is worse than none,
# because it is believed.
#
# HOSTLOCK_DIR has exactly that failure mode, and the reason it is still here
# is that the test suite needs a lock it cannot deadlock the real host with.
# So the two overrides are deliberately NOT equivalent:
#
#   env HOSTLOCK_DIR  -- PRIVATE. It is set per process, so two agents on one
#                        box do not converge on it. Announced loudly (see
#                        warn_if_private) because a private lock reports
#                        FREE/HELD in bytes identical to the shared one.
#   config lock_dir=  -- BOX-WIDE. A file on the machine is read by every
#                        invocation by every agent, so redirecting the lock
#                        there moves all of them together. This is the
#                        supported way off /tmp for hosts where /tmp is
#                        unwritable, noexec, per-service (systemd
#                        PrivateTmp=), or where a policy forbids writing it.
#
# The config file is PARSED, never sourced: it is a fixed path that any
# process on the box can write, so `.` would make it an execution vector for
# every hostlock invocation by every agent.
# The built-in path, and the seam that makes the legacy-holder consult
# testable. The consult below reads whatever path this box used BEFORE a
# config moved the lock -- which for every real host is /tmp. A conformance
# suite cannot fabricate a live holder there: it would have to write the real
# shared lock and deadlock the actual machine, so that branch would be the one
# piece of this design that ships untested. HOSTLOCK_LEGACY_DIR redirects it,
# and every row emits `legacy_dir=` so a run made with the seam active says so
# rather than looking like a run that consulted the real path.
HOSTLOCK_BUILTIN_DIR=/tmp/onnx-genai-hostlock
HOSTLOCK_LEGACY_PATH="${HOSTLOCK_LEGACY_DIR:-$HOSTLOCK_BUILTIN_DIR}"
HOSTLOCK_CONF_PATH="${HOSTLOCK_CONF:-${XDG_CONFIG_HOME:-${HOME:-/nonexistent}/.config}/onnx-genai/hostlock.conf}"

# Echo the configured lock_dir, or return 1 when there is no usable one.
# Comments and surrounding whitespace are stripped; the first key wins.
conf_lock_dir() {
    local v
    [ -f "$HOSTLOCK_CONF_PATH" ] || return 1
    v=$(sed -n 's/^[[:space:]]*lock_dir[[:space:]]*=[[:space:]]*//p' "$HOSTLOCK_CONF_PATH" 2>/dev/null | head -1)
    v=${v%%#*}
    v=$(printf '%s' "$v" | sed 's/[[:space:]]*$//')
    [ -n "$v" ] || return 1
    printf '%s\n' "$v"
}

# SHARED_LOCK_DIR is where peers on this box coordinate, whatever THIS process
# was told to use. Resolved unconditionally so the private-lock warning can
# name the path the caller is failing to coordinate on -- "this is private" is
# only actionable alongside "and everyone else is over there".
if SHARED_LOCK_DIR=$(conf_lock_dir); then
    case "$SHARED_LOCK_DIR" in
        /*) ;;
        *) SHARED_LOCK_DIR="" ;;
    esac
else
    SHARED_LOCK_DIR=""
fi

if [ -n "${HOSTLOCK_DIR:-}" ]; then
    LOCK_DIR="$HOSTLOCK_DIR"
    LOCK_DIR_SOURCE="env"
    LOCK_SCOPE=private
elif [ -n "$SHARED_LOCK_DIR" ]; then
    LOCK_DIR="$SHARED_LOCK_DIR"
    LOCK_DIR_SOURCE=config
    LOCK_SCOPE=box
else
    # A config that exists but whose lock_dir is unusable is a configuration
    # ERROR, not a reason to quietly use /tmp: the admin who wrote it may have
    # done so because /tmp does not work here, and half the box silently
    # falling back is the lock-splitting defect this whole block exists to
    # prevent. Only the env path (tests) skips this, because it never consults
    # the config in the first place.
    if [ -f "$HOSTLOCK_CONF_PATH" ] && grep -q '^[[:space:]]*lock_dir[[:space:]]*=' "$HOSTLOCK_CONF_PATH" 2>/dev/null; then
        die "${HOSTLOCK_CONF_PATH}: lock_dir must be a non-empty absolute path (got '$(conf_lock_dir || true)')"
    fi
    LOCK_DIR="$HOSTLOCK_BUILTIN_DIR"
    LOCK_DIR_SOURCE=default
    LOCK_SCOPE=box
fi
[ -n "$SHARED_LOCK_DIR" ] || SHARED_LOCK_DIR="$HOSTLOCK_BUILTIN_DIR"

UNPARSEABLE_GRACE=300
REAPER_GRACE=60
REAP_DIR="${LOCK_DIR}.reaper"
REAP_META="${REAP_DIR}/meta"
META="${LOCK_DIR}/meta"

# Say so, on every invocation, when this process is coordinating with nobody.
#
# The defect this closes: with HOSTLOCK_DIR set, `status` printed "FREE" and
# `status --porcelain` printed `state=FREE` in bytes IDENTICAL to a genuinely
# free shared host -- while a peer held the real one. The output was not
# wrong about what it measured; it was silent about what it measured. Nothing
# downstream, and no human reading a scrollback, could tell the two apart.
#
# HOSTLOCK_PRIVATE_OK=1 silences it, and the suite sets it: three lines of
# stderr per invocation across hundreds of invocations is how a warning gets
# deleted for being noise. It is an explicit acknowledgement, which is the
# same shape as --gate's timeout decision: the caller must say the word.
warn_if_private() {
    [ "$LOCK_SCOPE" = private ] || return 0
    [ "${HOSTLOCK_PRIVATE_OK:-0}" = 1 ] && return 0
    echo "hostlock: WARNING: HOSTLOCK_DIR is set, so this is a PRIVATE lock at ${LOCK_DIR}." >&2
    echo "hostlock: it coordinates with NOBODY -- peers on this host use ${SHARED_LOCK_DIR}." >&2
    echo "hostlock: set HOSTLOCK_PRIVATE_OK=1 to acknowledge and silence (the test suite does)." >&2
}

# Is somebody still holding the OLD default path?
#
# Only asked when a config has moved this box off /tmp, and answered strictly
# READ-ONLY: never reap it, never write to it, never touch its metadata. A
# peer holding the legacy path is by definition running a version or a session
# that has not re-read the config, so it will not see us move it and it cannot
# be negotiated with -- it can only be waited out.
#
# Echoes "<owner> <pid>" for a live legacy holder, returns 1 otherwise.
legacy_holder() {
    local m pid start owner mtime age
    [ "$LOCK_DIR_SOURCE" = config ] || return 1
    [ "$LOCK_DIR" != "$HOSTLOCK_LEGACY_PATH" ] || return 1
    [ -d "$HOSTLOCK_LEGACY_PATH" ] || return 1
    m="${HOSTLOCK_LEGACY_PATH}/meta"
    pid=$(sed -n 's/^anchor_pid=//p' "$m" 2>/dev/null | head -1)
    start=$(sed -n 's/^start_time=//p' "$m" 2>/dev/null | head -1)
    owner=$(sed -n 's/^owner=//p' "$m" 2>/dev/null | head -1)
    if [ -z "$pid" ] || [ -z "$start" ]; then
        # Unreadable metadata is not evidence of a dead holder (same rule as
        # `reapable`), but an abandoned corpse on a path we will never reap
        # must not block this host forever. Busy while fresh, ignored once it
        # has clearly aged out.
        mtime=$(stat -c %Y "$HOSTLOCK_LEGACY_PATH" 2>/dev/null || echo 0)
        age=$(( $(date +%s) - mtime ))
        [ "$age" -le "$UNPARSEABLE_GRACE" ] && { printf 'unknown ?\n'; return 0; }
        return 1
    fi
    anchor_alive "$pid" "$start" || return 1
    printf '%s %s\n' "${owner:-unknown}" "$pid"
}

# Fail closed while a legacy holder is live. Called before any acquire.
refuse_if_legacy_held() {
    local h
    h=$(legacy_holder) || return 0
    echo "hostlock: BUSY (legacy path)" >&2
    echo "hostlock: ${HOSTLOCK_LEGACY_PATH} is held by ${h% *} (pid ${h#* }), which is the path this host used" >&2
    echo "hostlock: before ${HOSTLOCK_CONF_PATH} moved the lock to ${LOCK_DIR}. That holder cannot see our lock," >&2
    echo "hostlock: so taking this one would put two benchmarks on the box. Wait for it to release." >&2
    return 2
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

# Process group id (field 5) of a pid, from the same remainder as the reads
# above: after the comm field is dropped, $1 is state, $3 is pgrp, $20 is
# start time.
proc_pgid() {
    local pid=$1 rest
    rest=$(tr -d '\0' 2>/dev/null <"/proc/${pid}/stat" | sed 's/.*) //') || return 1
    [ -n "$rest" ] || return 1
    awk '{print $3}' <<<"$rest"
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
    local t0=$1 t1=$2 w0=$3 w1=$4 hz cpu wall verdict eff eff_field
    hz=$(getconf CLK_TCK 2>/dev/null || echo 100)
    # One field name for two different quantities is a unit error waiting to
    # be made: without a denominator this ratio is CPU-seconds per wall second
    # (i.e. cores, 0..ncpu), and with one it is the fraction of those cores
    # that were actually held (0..1). The same 8-core run reads 8.000 in the
    # first form and 1.000 in the second, so a log mixing both is not
    # comparable to itself. Name the unit in the field rather than expecting
    # the reader to notice `cores_expected=` first.
    if [ -n "$EXPECT_CORES" ]; then
        eff_field=efficiency_frac
    else
        eff_field=efficiency_cores
    fi
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
        echo "hostlock: cpu wall=${wall}s cpu=${cpu}s ${eff_field}=unmeasurable (run too short to judge)" >&2
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
    echo "hostlock: cpu wall=${wall}s cpu=${cpu}s cores_expected=${EXPECT_CORES:-unspecified} ${eff_field}=${eff} verdict=${verdict}" >&2
    if [ "$rc" -ne 0 ]; then
        echo "hostlock: WARNING measured ${eff_field} ${eff} is below --min-efficiency ${MIN_EFFICIENCY}" >&2
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

# Flatten a value so it cannot forge metadata.
#
# The store is one `key=value` per line and `meta_get` takes the FIRST match,
# so a newline inside an operator-supplied value does not corrupt the file --
# it *appends fields*, and any key written after the injection point wins over
# the real one. Measured before this existed:
#
#   --reason $'x\nttl=999999'   with --ttl 5   ->   status reports ttl=999999
#
# `ttl` is not decoration: `holder_expired` reads it, so a reason string could
# make a lock outlive its own declared expiry while every report looked
# ordinary. `takeover`, `gate`, `runnable_at_acquire` and `acquired_epoch` sit
# after `reason` too, and those are exactly the fields harnesses now copy into
# published results as provenance -- a forged one is a benchmark row claiming
# a quiet uncontended host it never had.
#
# The safety-critical anchor fields (`anchor_pid`, `start_time`, `anchor_uid`)
# were never exposed, because they are written before any operator value and
# first-match wins. So this is a provenance and expiry defect, not a way to
# steal a lock -- which is the only reason it is a bug fix and not an
# emergency.
#
# Newlines become spaces rather than being deleted, so a pasted multi-line
# reason stays readable instead of silently welding two words together, and
# NULs and CRs are dropped outright.
meta_value() {
    printf '%s' "$1" | tr -d '\000\r' | tr '\n' ' '
}

# The repo checkout the holder is working in.
#
# Several worktrees of this repo share one machine and therefore one lock, so
# "who holds it" is only half an answer -- `owner=roy` does not say which tree
# is being benchmarked, and the reason string is prose the operator typed
# rather than an observed fact. This is observed.
holder_worktree() {
    local top
    # Bounded, because this runs inside publish_lock's stage-and-rename
    # critical section. `git rev-parse` is local and fast normally, but it
    # walks parent directories looking for .git, and on a stalled NFS/FUSE
    # mount that walk blocks without ever returning an exit code -- leaving a
    # staging directory created and never published, by a process that cannot
    # be reaped because the lock does not exist yet. A descriptive field is
    # never worth that, so it gets two seconds and then we fall back.
    top=$(timeout 2 git -C "$PWD" rev-parse --show-toplevel 2>/dev/null) || top=""
    [ -n "$top" ] || top="$PWD"
    printf '%s' "$top"
}

# The command actually holding the lock.
#
# `run` knows its wrapped command exactly and sets HOLDER_CMD. `acquire`
# anchors to the invoking shell, so the best available answer is that shell's
# own argv, read from /proc -- NUL-separated, hence the tr. Either way this is
# what is running, not what the operator said was running: a reason of "quick
# smoke test" attached to a four-hour matrix is precisely the drift worth
# catching, and it is the same argument as recording realized placement rather
# than trusting a width label.
#
# This records argv, so do not put secrets on a benchmark command line. That
# was already true of `ps` on a shared box -- every agent here runs as the same
# unix user -- but this writes it to a file that outlives the process, so it is
# worth saying rather than discovering.
holder_cmd() {
    if [ -n "${HOLDER_CMD:-}" ]; then
        printf '%s' "$HOLDER_CMD"
        return 0
    fi
    local c=""
    if [ -n "${ANCHOR_PID:-}" ] && [ -r "/proc/${ANCHOR_PID}/cmdline" ]; then
        # `2>/dev/null` FIRST: redirections are applied left to right, and a
        # failure to open the input file is reported on whatever stderr is
        # current at that moment. Written the other way round the guard is
        # applied too late to silence the thing it exists to silence. The
        # `-r` test above does not close the window -- the holder can exit
        # between the test and the read, which is exactly when this runs.
        c=$(tr '\000' ' ' 2>/dev/null <"/proc/${ANCHOR_PID}/cmdline")
    fi
    printf '%s' "$c"
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

# Processes still alive in the WRAPPED COMMAND's process group, for use when
# the anchor itself is gone. Prints them; returns 1 when there are none.
#
# The wedge this closes, reproduced on this host before it was fixed: `run` is
# SIGKILLed, so its EXIT trap never fires and it never reaps the tree it
# started. The anchor pid dies, `holder_alive` says no, `reapable` says yes,
# and `status` prints "STALE ... next acquire will reap it" -- while the
# wrapped command's children still hold every core they had. The next
# acquirer takes a box that is not free and measures against a live
# competitor it cannot see, which is the one outcome this lock exists to
# prevent.
#
# This is the same failure the `--ttl` text at the top of this file already
# names as the reason `run` refuses a finite TTL -- "it would relabel the host
# as free while the runaway kept burning cores" -- arriving through a door
# nobody checked. The anchor is the CLAIM. The process group is the JOB. The
# job is what holds the cores, so the job is what liveness has to mean.
#
# It matters because the holder is usually NOT the process doing the damage:
# a `cargo test` that forks qemu children, a driver that spawns arm binaries.
# Anchor-only liveness asks whether the bookkeeper is alive, not the workers.
#
# `child_pgid` is recorded only when the child leads its own group (see the
# guard in `run`), so this can never name this script's own group, and it is
# only ever consulted once the anchor is dead -- so it cannot keep a lock
# alive on the strength of the holder.
#
# ZOMBIES DO NOT COUNT: `live_pids` already excludes them, and it must, or a
# not-yet-reaped child would read as a core-holder on every clean teardown.
#
# A pgid can be recycled, and a recycled one would hold the lock against a
# stranger's processes. That is the conservative direction -- BUSY when it is
# free, never FREE when it is busy -- which is the direction this file takes
# everywhere else. `status` names the surviving pids so an operator can see in
# one line whether they are the job or a coincidence.
orphan_group_pids() {
    local pgid live
    pgid=$(meta_get child_pgid) || return 1
    [ -n "$pgid" ] || return 1
    case "$pgid" in '' | *[!0-9]*) return 1 ;; esac
    live=$(live_pids "-$pgid")
    [ -n "$live" ] || return 1
    printf '%s' "$live"
}

# Are we running INSIDE a `run` that holds this very lock?
#
# `run` always sets DO_WAIT, so a nested acquire against the same lock path
# waits for a holder that is its own ancestor: the parent cannot release until
# the wrapped command returns, and the wrapped command is the thing waiting.
# That is not contention, it is a cycle, and it burns the full --timeout
# (default 3600s) before reporting a peer that was never there (#1977).
#
# Both halves are load-bearing. The path alone is not enough: a `run` whose
# wrapped command daemonises leaves the variable set in a process that outlives
# the lock, and refusing that process's later, legitimate acquire would be a
# fail-closed bug of our own making. So the live lock's own anchor must still
# be the one we exported -- if the parent released, or was reaped and the lock
# republished by somebody else, the anchors differ and this is an ordinary
# acquire against an ordinary peer.
#
# Deliberately NOT a pid-ancestry walk. /proc/<pid>/stat ppid chains break the
# moment anything reparents (a daemonised harness, a subreaper, an agent whose
# shell exits), and the failure direction there is to stop recognising our own
# holder -- back to the hang. The exported pair is exact for the case that
# actually deadlocks, and silent for every case that does not.
nested_under_own_run() {
    [ -n "${HOSTLOCK_HELD_DIR:-}" ] || return 1
    [ -n "${HOSTLOCK_HELD_ANCHOR:-}" ] || return 1
    [ "$HOSTLOCK_HELD_DIR" = "$LOCK_DIR" ] || return 1
    [ "$(meta_get anchor_pid 2>/dev/null || echo "")" = "$HOSTLOCK_HELD_ANCHOR" ] || return 1
    return 0
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
    ! holder_alive && {
        # A dead anchor is not a free host if the job it started is still
        # running. Checked only here, on the path that would otherwise hand
        # the box to the next acquirer.
        orphan_group_pids >/dev/null && return 1
        return 0
    }
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
        echo "owner=$(meta_value "${OWNER}")"
        echo "reason=$(meta_value "${REASON}")"
        echo "worktree=$(meta_value "$(holder_worktree)")"
        echo "cmd=$(meta_value "$(holder_cmd)")"
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
# Echo why no lock can be taken at LOCK_DIR and return 0; return 1 when one can.
#
# FREE is not a fact about the directory, it is a promise to the caller: "the
# host is yours, go ahead". On a host where the lock cannot be CREATED that
# promise is false, and until this existed `status` made it anyway -- printing
# `FREE  (runnable=4)`, and `state=FREE` in porcelain, in bytes IDENTICAL to a
# genuinely free and usable host. That is the same defect `warn_if_private`
# was written for one level up (#1942): output that is not wrong about what it
# measured, and silent about what it measured.
#
# It is not hypothetical here. This box's default lock_dir is under /tmp, and
# at least one agent on it is under a hard prohibition on writing /tmp at all;
# `acquire` failed for that agent with a raw `mkdir: Permission denied` and
# exit 1 -- "usage/error", the code for a bad argument -- while `status` went
# on saying FREE. An unattended harness reading either one proceeds unlocked,
# which is precisely what the lock became mandatory to prevent.
#
# What is tested is the PARENT, never LOCK_DIR itself: publish_lock stages at
# `${LOCK_DIR}.stage.$$`, a SIBLING, and renames it into place. A lock dir that
# exists but is unwritable is therefore still perfectly publishable, and an
# absent one whose parent is writable is fine too -- so the question is always
# "can we create entries next to it", answered at the nearest ancestor that
# exists, because `mkdir -p` will make the rest.
#
# `-w` lies under CAP_DAC_OVERRIDE, so root sees "usable" where a mortal would
# not. That is the safe direction: it is exactly today's behaviour, and root
# can in fact write there.
lock_dir_problem() {
    local p
    if [ -e "$LOCK_DIR" ] && [ ! -d "$LOCK_DIR" ]; then
        printf '%s\n' "${LOCK_DIR} exists and is not a directory"
        return 0
    fi
    case "$LOCK_DIR" in
        */*) p=${LOCK_DIR%/*}; [ -n "$p" ] || p=/ ;;
        *) p=. ;;
    esac
    while [ ! -e "$p" ]; do
        case "$p" in
            */*) p=${p%/*}; [ -n "$p" ] || p=/ ;;
            *) p=. ; break ;;
        esac
    done
    if [ ! -d "$p" ]; then
        printf '%s\n' "${p} exists and is not a directory"
        return 0
    fi
    if [ ! -w "$p" ] || [ ! -x "$p" ]; then
        printf '%s\n' "${p} is not writable by $(id -un 2>/dev/null || echo "uid $(id -u 2>/dev/null)")"
        return 0
    fi
    return 1
}

# Say the UNUSABLE part out loud, on stderr, wherever a caller was about to be
# told the host is available. Three lines, because one is not enough to stop an
# agent that has been told the lock is mandatory and has just read "free".
explain_unusable() {
    local problem=$1
    echo "hostlock: UNUSABLE: ${problem}" >&2
    echo "hostlock: no lock can be created at ${LOCK_DIR} (${LOCK_DIR_SOURCE}, ${LOCK_SCOPE}), so this host cannot participate." >&2
    echo "hostlock: set lock_dir in ${HOSTLOCK_CONF_PATH} to a path you can write; do NOT run saturating benchmarks unlocked." >&2
}

lock_state() {
    if [ ! -d "$LOCK_DIR" ]; then
        if lock_dir_problem >/dev/null; then echo UNUSABLE; else echo FREE; fi
        return 0
    fi
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
    local state owner pid age reason takeover gate r r_acq contended uid legacy
    local holder_wt=unknown holder_cl=unknown
    state=$(lock_state)
    r=$(runnable_now)
    legacy=$(legacy_holder) || legacy=""
    legacy=${legacy%% *}
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
        # Absent on locks published before these fields existed. `unknown` is
        # the honest answer; `none` would assert the holder was in no worktree.
        holder_wt=$(meta_get worktree) || holder_wt=unknown
        holder_cl=$(meta_get cmd) || holder_cl=unknown
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
        # WHICH lock this row is about. Without it a private lock's row is
        # byte-identical to a shared one, so a table of measurements taken
        # with HOSTLOCK_DIR set -- coordinating with nobody -- reads exactly
        # like a table taken under the real host lock. `declared=yes` is a
        # claim about a host; it is only checkable if the row says which
        # directory the claim was made in.
        "lock_dir=${LOCK_DIR}"
        "lock_scope=${LOCK_SCOPE}"
        "lock_dir_source=${LOCK_DIR_SOURCE}"
        "legacy_dir=$(legacy_consult_path)"
        "legacy_held_by=${legacy:-none}"
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
        # Same rule as `reason`, for the same reason: a path can contain
        # spaces and a command almost always does, so neither is safe among
        # space-separated fields. One per line, newline-delimited.
        printf 'held_worktree=%s\n' "${holder_wt:-unknown}"
        printf 'held_cmd=%s\n' "${holder_cl:-unknown}"
    fi
}

# Which path the legacy consult actually reads, or `none` when no consult
# applies. Emitted into every machine-readable row: `declared=yes` measured on
# a box mid-migration means something different from `declared=yes` on a box
# that never moved, and a row that does not say which cannot be re-checked.
legacy_consult_path() {
    if [ "$LOCK_DIR_SOURCE" = config ] && [ "$LOCK_DIR" != "$HOSTLOCK_LEGACY_PATH" ]; then
        printf '%s\n' "$HOSTLOCK_LEGACY_PATH"
    else
        printf 'none\n'
    fi
}

# The human `status` says WHICH lock it is about whenever that is not the
# default. "FREE" is the answer most likely to be acted on and the one whose
# meaning depends entirely on the directory it was measured in.
status_lock_dir_note() {
    local legacy=$1
    [ "$LOCK_DIR_SOURCE" = default ] || echo "  lock: ${LOCK_DIR} (${LOCK_DIR_SOURCE}, ${LOCK_SCOPE})"
    [ "$legacy" != none ] && echo "  legacy: ${HOSTLOCK_LEGACY_PATH} is still held by ${legacy%% *} (pid ${legacy##* }); acquire will refuse"
    return 0
}

print_free_or_unusable() {
    local problem
    if problem=$(lock_dir_problem); then
        echo "UNUSABLE  ${problem}"
        # Exactly one lock line either way: status_lock_dir_note prints it for
        # every non-default source, and the default source is the case that
        # needs it most -- /tmp is where this actually bites.
        [ "$LOCK_DIR_SOURCE" = default ] && echo "  lock: ${LOCK_DIR} (${LOCK_DIR_SOURCE}, ${LOCK_SCOPE})"
        echo "  no lock can be created here; this host cannot participate"
        echo "  runnable=$(runnable_now)"
        return 0
    fi
    echo "FREE  (runnable=$(runnable_now))"
    return 0
}

cmd_status() {
    local legacy
    legacy=$(legacy_holder) || legacy="none"
    if [ "$PORCELAIN" = 1 ]; then
        echo "state=$(lock_state)"
        echo "runnable=$(runnable_now)"
        echo "lock_dir=${LOCK_DIR}"
        echo "lock_scope=${LOCK_SCOPE}"
        echo "lock_dir_source=${LOCK_DIR_SOURCE}"
        # Always emitted, empty when there is none, so the key set does not
        # change shape between hosts -- a consumer that has to test for a key's
        # PRESENCE to learn the state is one `grep` away from reading absence
        # as "fine", which is the failure this whole field exists to report.
        echo "lock_dir_problem=$(lock_dir_problem || true)"
        echo "legacy_dir=$(legacy_consult_path)"
        echo "legacy_held_by=${legacy%% *}"
        if [ -d "$LOCK_DIR" ]; then
            echo "owner=$(meta_get owner || echo '?')"
            echo "anchor_pid=$(meta_get anchor_pid || echo '?')"
            echo "reason=$(meta_get reason || echo '')"
            # `unknown` rather than empty for locks published before these
            # fields existed: empty would read as "held from no worktree".
            echo "worktree=$(meta_get worktree || echo 'unknown')"
            echo "cmd=$(meta_get cmd || echo 'unknown')"
            echo "ttl=$(meta_get ttl || echo 0)"
            echo "age=$(lock_age)"
        fi
        return 0
    fi
    if [ ! -d "$LOCK_DIR" ]; then
        print_free_or_unusable
        status_lock_dir_note "$legacy"
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
            # Distinguish a live holder from a crashed one whose job survived
            # it. Both are HELD -- the box is busy either way, which is why
            # lock_state does not gain a state here and `wait` keeps waiting --
            # but "pid=N" against a pid that no longer exists reads as a bug
            # unless the surviving job is named. See orphan_group_pids.
            local orph
            if ! holder_alive && ! unverifiable_live_anchor && orph=$(orphan_group_pids); then
                echo "HELD (ORPHANED)  holder ${owner} pid=${pid} is gone, but the job it started is still running"
                echo "  reason: ${reason}"
                echo "  still alive in pgid $(meta_get child_pgid || echo '?'): ${orph}"
                echo "  the host is NOT free; the lock stays held until these exit"
                echo "  runnable=$(runnable_now)"
            else
                echo "HELD by ${owner} pid=${pid} for ${age}s since ${at}"
                echo "  reason: ${reason}"
                echo "  runnable=$(runnable_now)"
            fi
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
            print_free_or_unusable
            ;;
    esac
    status_lock_dir_note "$legacy"
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
    local deadline=$((SECONDS + TIMEOUT)) rc problem announced_wait=0
    # Refuse BEFORE anything else, including the legacy consult. An acquire
    # that cannot possibly publish must not spend --timeout looking busy: with
    # --wait (which `run` always sets) the unusable host reported "timed out
    # after 900s waiting for the lock", i.e. it blamed peers for contention
    # that did not exist, and returned 3 -- a code whose documented meaning is
    # that somebody else has the box. Distinct outcomes, or the label launders
    # the fault.
    if problem=$(lock_dir_problem); then
        explain_unusable "$problem"
        return 7
    fi
    # Refuse a cycle before refusing a peer, and before waiting for either.
    # The order matters for the same reason the unusable-host check comes
    # first: a nested acquire that falls through to the wait loop spends
    # --timeout (3600s by default under `run`) and then reports code 3, which
    # says a co-tenant held the box. There was no co-tenant. See #1977.
    if nested_under_own_run; then
        echo "hostlock: outcome=nested by ${OWNER}" >&2
        echo "hostlock: this process is already inside a \`run\` holding ${LOCK_DIR} (anchor pid ${HOSTLOCK_HELD_ANCHOR})." >&2
        echo "hostlock: that holder cannot release until this command returns, so waiting for it can only time out." >&2
        echo "hostlock: run the inner command directly -- the host is already held for it -- or give the inner lock its own path via HOSTLOCK_DIR." >&2
        cmd_status >&2
        return 9
    fi
    refuse_if_legacy_held || return $?
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
        # Say so, once, on the first pass that finds the lock held.
        #
        # A silent wait is indistinguishable from a slow build, which is the
        # worst way for a lock to fail on a box where "is this wedged?" is the
        # question people are already asking (#1977). The holder's identity is
        # the answer, and it is already printed by two other paths (BUSY and
        # timeout); the wait path was the only one that withheld it. Once, not
        # per iteration: a line every 5s for an hour is how a message gets
        # filtered out.
        #
        # This sits BEFORE the deadline check on purpose, and the ordering is
        # not cosmetic. With it after, whether you were told who held the lock
        # depended on whether the first pass happened to cross the deadline --
        # so on a loaded box, the one condition that makes a wait worth
        # explaining is the one that silences the explanation. It was measured:
        # the cell below failed 1 run in 4 at --timeout 1 until the
        # announcement moved above the check. It is deliberately not gated on
        # the timeout value either, so that the ordering is observable at
        # --timeout 0 without waiting for a clock.
        if [ "$announced_wait" != 1 ]; then
            announced_wait=1
            echo "hostlock: waiting up to ${TIMEOUT}s for the lock" >&2
            cmd_status >&2
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
    local deadline=$((SECONDS + TIMEOUT)) problem
    # An unusable lock dir is never HELD, so the loop below falls straight
    # through and announces "free" -- the one word this caller is waiting to
    # hear, on the one host where it cannot be true.
    if problem=$(lock_dir_problem); then
        explain_unusable "$problem"
        return 7
    fi
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

# Where `run` spawns the wrapped command, so teardown can signal its whole
# process group. Empty when setsid is unavailable, which is handled rather
# than assumed away: stop_wrapped_tree verifies leadership before signalling.
SETSID_BIN=$(command -v setsid 2>/dev/null || true)

# The pids still RUNNING under a target: a pid, or "-PGID" for a whole process
# group -- the same spelling the signals below use, so the thing verified is
# exactly the thing signalled.
#
# ZOMBIES DO NOT COUNT. An unreaped child is a zombie, a zombie is still a
# member of its process group, and one that read as live would fire the
# escalation and the warning on every clean teardown. A zombie holds no cores,
# and cores are the only thing this lock is about.
#
# Read from /proc directly rather than `pgrep -g`, which is shorter and wrong
# in the one way this file cannot afford. If pgrep were missing its failure is
# an EMPTY result, which reads as "nothing is alive": the poll would exit on
# its first pass, the escalation to KILL and the survivor warning would both be
# skipped, and a stubborn group would leak in silence. A missing dependency
# would have turned a loud failure into a quiet one, which is the exact defect
# class this script exists to close. /proc is already its source of truth for
# every other pid question here, and the loop is a bash builtin read -- no fork
# per pid, so scanning it twenty times over a ten-second poll is free.
#
# Deliberately not `kill -0` either. This is a liveness QUESTION, not a signal,
# and the suite pins the number of calls in the kill family to an exact count
# so that this script's signalling surface stays countable by eye -- the
# property that keeps a host lock structurally incapable of stopping anything
# it did not start. A probe spelled as a kill would inflate that count and
# blunt the guard for no gain.
live_pids() {
    local t=$1 pgid d p line rest out=""
    [ -n "$t" ] || { printf ''; return 0; }
    if [ "${t#-}" != "$t" ]; then
        pgid=${t#-}
        for d in /proc/[0-9]*; do
            p=${d#/proc/}
            # `2>/dev/null` precedes the input redirect deliberately: bash
            # applies redirections left to right and reports a failed open on
            # the stderr in force at that point, so the other order silences
            # nothing. This scan races every exit on the host by construction
            # -- it globs /proc and then reads each entry -- so with the guard
            # mis-ordered a teardown poll prints one "No such file" per pid
            # that ends mid-pass, onto the lock's own stderr.
            read -r line 2>/dev/null <"$d/stat" || continue
            # Longest match, as everywhere else here: the comm field is the
            # only parenthesised one, so this lands on state ($1) and pgrp
            # ($3) even for a command name containing a bracket.
            rest=${line##*") "}
            # shellcheck disable=SC2086  # deliberate splitting of stat fields
            set -- $rest
            [ "${3:-}" = "$pgid" ] && [ "${1:-Z}" != Z ] && out="${out}${p} "
        done
    else
        read -r line 2>/dev/null <"/proc/${t}/stat" || { printf ''; return 0; }
        rest=${line##*") "}
        # shellcheck disable=SC2086  # deliberate splitting of stat fields
        set -- $rest
        [ "${1:-Z}" != Z ] && out="${t} "
    fi
    printf '%s' "$out"
    return 0
}

target_alive() {
    [ -n "$(live_pids "$1")" ]
}

# The pids still running under a target, for the warning below.
live_under() {
    live_pids "$1"
}

# Stop the wrapped command AND everything it started.
#
# `kill -TERM "$child"` stops exactly one pid. Every runaway this box has had
# to account for was a GRANDchild: a harness runs `cargo`, cargo runs the test
# binary, and signalling cargo alone leaves that binary spinning -- reparented
# to init -- while this script prints "released" and the host is declared
# free. Measured here, not reasoned about: a wrapped
# `bash -c 'sleep 300 & wait'` left the sleep running with PPID 1 after the
# runner took SIGTERM and released the lock. That is precisely the orphaned
# load this file's header warns about ("reclaiming a lock does not stop the
# load the lock was covering"), and it was reachable through the lock's own
# teardown. The suite asserted the direct child was stopped, which is the one
# depth the defect was never at.
#
# Signal the process GROUP, which is why `run` spawns under setsid. A group is
# atomic where walking `pgrep -P` is not: a process that forks between
# enumeration and signalling escapes the walk, and the tree most in need of
# stopping is a build system that is actively spawning.
#
# TERM first so a benchmark can flush partial results, then KILL, because the
# payloads that leak are exactly the ones that do not die on TERM. Then VERIFY
# rather than assume: "a signal was sent" and "the cores are free" are
# different claims, and the whole purpose of this script is to make the second.
#
# The grace between the two is TEN SECONDS (20 polls of 0.5s), and it is a
# ceiling, not a suggestion: a wrapped command that traps TERM to flush more
# than ten seconds of partial results will be KILLed mid-flush. Stated here
# because a wrapper cannot discover it by reading its own code, and a longer
# window is not obviously right -- these signals are sent when a run is being
# aborted, usually because somebody needs the box back.
#
# Every wait here is BOUNDED. The first draft of this function reaped the child
# with a plain `wait` before polling, which is correct for a child that dies
# and an unbounded hang for one that does not: a `mut_hl.sh` run with
# `trap ... TERM` that returns instead of exiting held teardown in `wait`
# forever, so the escalation below was never reached and the lock was never
# released. That is the same defect as the one being fixed -- a bound that can
# itself block is not a bound -- so the child is reaped only once it is known
# to be dead or a zombie, and never blocked on.
stop_wrapped_tree() {
    local child=$1 pgid target tries survivors state

    pgid=$(proc_pgid "$child" 2>/dev/null || echo "")
    if [ -n "$pgid" ] && [ "$pgid" = "$child" ]; then
        target="-$pgid"
    else
        # Not a group leader -- setsid missing, or it forked instead of
        # exec'ing. Signalling "-$pgid" here would hit OUR OWN group, taking
        # down this script and its caller along with the benchmark. Falling
        # back to the single pid keeps the old, narrower reach rather than
        # trading a leak for a much worse failure.
        target="$child"
    fi

    kill -TERM "$target" 2>/dev/null

    tries=0
    while [ "$tries" -lt 20 ] && target_alive "$target"; do
        sleep 0.5
        tries=$((tries + 1))
    done

    if target_alive "$target"; then
        kill -KILL "$target" 2>/dev/null
        sleep 0.5
        if target_alive "$target"; then
            survivors=$(live_under "$target")
            echo "hostlock: WARNING the wrapped command (${target}) survived both signals: ${survivors:-unknown}. The lock is being released, but this load is still on the cores -- the host is NOT free." >&2
        fi
    fi

    # Reap, but never BLOCK on it: `wait` on something that outlived KILL
    # (uninterruptible sleep, a stopped process) would hang here, which is
    # exactly what this function exists to stop happening. If it is still
    # alive the warning above already said so; leaving a zombie for init to
    # collect is strictly better than a silent hang.
    state=$(proc_state_and_start "$child" 2>/dev/null | awk '{print $1}')
    if [ -z "$state" ] || [ "$state" = Z ]; then
        wait "$child" 2>/dev/null
    fi
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
            stop_wrapped_tree "$child"
        fi
    fi
    remove_lock_if_mine
    echo "hostlock: released (${name})" >&2
    exit "$code"
}

cmd_run() {
    [ "$#" -gt 0 ] || die "run needs a command after --"
    HOLDER_CMD="$*"
    DO_WAIT=1
    cmd_acquire || return $?

    # Hand the declared identity to the wrapped command, so a harness that
    # reads the lock into its rows can tell its own parent's declaration from
    # a co-tenant's. Without this, `--owner leon` is only a shell variable and
    # the child reports `host_lock=held:leon` -- honest, but unattributed and
    # deliberately not treated as certifying -- while the environment form
    # reports `mine:leon`. The obvious invocation was the one that silently
    # measured weaker (#1929).
    #
    # Only when the owner was actually declared: see OWNER_DECLARED.
    if [ "$OWNER_DECLARED" = 1 ]; then
        export HOSTLOCK_OWNER="$OWNER"
    fi

    # Tell the wrapped command which lock we are holding for it, and with which
    # anchor. A harness that calls back into hostlock.sh would otherwise wait
    # on its own parent until --timeout expires (#1977); nested_under_own_run
    # reads exactly this pair. Unconditional, unlike HOSTLOCK_OWNER above: that
    # export is withheld on the default path because a $USER-derived owner
    # would manufacture an attribution nobody made (#1929), whereas the lock
    # path and the anchor pid are facts about this invocation, attribute
    # nothing to anybody, and are wrong to withhold -- the deadlock does not
    # care whether --owner was passed.
    export HOSTLOCK_HELD_DIR="$LOCK_DIR"
    export HOSTLOCK_HELD_ANCHOR="$ANCHOR_PID"

    # Run the command in the BACKGROUND and wait for it, rather than inline.
    #
    # Bash does not run a trap until the current foreground command finishes.
    # With `"$@"` inline, Ctrl-C during a forty-minute benchmark would not
    # release the lock until that benchmark ended by itself -- prompt release
    # failing in precisely the case it exists for. `wait` is interruptible,
    # so this makes the signal handlers actually prompt. It also lets us stop
    # the wrapped command, which the inline form left running.
    local child rc wall0 wall1 cpu0 cpu1 monitor_was_on
    wall0=$(wall_now)
    # Job control OFF across the spawn, and restored immediately after.
    #
    # `setsid` execs IN PLACE when its caller is not a process-group leader,
    # and FORKS when it is. A bash background job is not a leader -- unless
    # monitor mode is on, which puts every background job in a group of its
    # own. Then `$!` is the short-lived setsid parent: `wait` returns at once,
    # this script prints "released", frees the lock, and the benchmark runs on
    # an unlocked host. Measured, not feared: with `SHELLOPTS=monitor` in the
    # environment (bash imports it, so no edit to this file is needed to reach
    # this), `run -- sh -c 'sleep 3'` reported `wall=0.010s` and released while
    # the sleep was still running.
    #
    # Restored rather than left off, because `-m` is the caller's setting and
    # this script is not entitled to keep it; by then the child is already in
    # its own session and cannot be moved back.
    monitor_was_on=0
    case "$-" in *m*) monitor_was_on=1 ;; esac
    set +m
    # Spawn into its OWN process group so teardown can stop the tree rather
    # than one pid (see stop_wrapped_tree). `$!` is then the new group leader
    # and its pgid equals its own pid -- verified before any group signal,
    # never assumed.
    #
    # Two consequences of the new session, neither of which affects a
    # benchmark: the command loses the controlling terminal, so a payload that
    # opens /dev/tty or checks isatty sees a difference and a Ctrl-C reaches it
    # only through the trap above (which is the whole point -- that path stops
    # the tree, the terminal's would not); and `run -- <shell builtin>` is now
    # a not-found error rather than silently doing nothing useful, because
    # setsid execs a program.
    if [ -n "$SETSID_BIN" ]; then
        "$SETSID_BIN" "$@" &
    else
        "$@" &
    fi
    child=$!
    if [ "$monitor_was_on" = 1 ]; then
        set -m
    fi

    # Traps first. Reading the child's start time forks, and a signal arriving
    # in that window would take the script's default action -- leaking the
    # lock and orphaning the benchmark, the two failures this block exists to
    # prevent. run_teardown tolerates an empty RUN_CHILD_START by declining to
    # signal, which is the safe direction for a window this narrow.
    trap 'run_teardown "$child" SIGINT 130' INT
    trap 'run_teardown "$child" SIGTERM 143' TERM
    trap 'run_teardown "$child" SIGHUP 129' HUP
    RUN_CHILD_START=$(proc_start_time "$child" 2>/dev/null || echo "")
    # Recorded only when the child really leads its own group, so every later
    # `-$RUN_CHILD_PGID` cannot possibly name this script's own group.
    RUN_CHILD_PGID=$(proc_pgid "$child" 2>/dev/null || echo "")
    [ "$RUN_CHILD_PGID" = "$child" ] || RUN_CHILD_PGID=""

    # Publish the group, so that a run which is KILLED -- and therefore never
    # runs a trap, never reaps its tree, and leaves the lock anchored to a
    # dead pid -- still declares the box busy for as long as its children
    # hold cores. See orphan_group_pids.
    #
    # An append, not a publish_lock field: the lock is taken before the child
    # exists, so the group is not knowable at publish time. Safe because
    # `meta_get` is first-match-wins and nothing else ever writes this key, so
    # this cannot shadow or forge an earlier field; and a reader that catches
    # the append mid-write simply fails to match, falling back to anchor-only
    # liveness, which is the behaviour that existed before this line.
    if [ -n "$RUN_CHILD_PGID" ] && [ -d "$LOCK_DIR" ]; then
        printf 'child_pgid=%s\n' "$RUN_CHILD_PGID" >>"$META" 2>/dev/null || true
    fi

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
    # A command that returned is not the same as a host that is quiet: it can
    # leave a background grandchild behind, and the next line declares the box
    # free. Report that rather than releasing over it in silence. Deliberately
    # a warning and not a kill -- on this path the command RETURNED rather
    # than being signalled (at any exit status; this is not gated on rc), so
    # whatever it left behind it left deliberately, and stopping it would be
    # this script overruling that. On the SIGNAL path the run is being aborted
    # and stop_wrapped_tree does kill, which is the opposite situation.
    if [ -n "$RUN_CHILD_PGID" ] && target_alive "-$RUN_CHILD_PGID"; then
        echo "hostlock: WARNING the command exited but left its process group (${RUN_CHILD_PGID}) running: $(live_under "-$RUN_CHILD_PGID"). Releasing anyway; the host is not necessarily idle." >&2
    fi
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

# Whether the owner was DECLARED, or merely defaulted from the unix user.
#
# `run` exports a declared owner to its child (#1929) so a harness reading the
# lock can tell its own parent's declaration from a co-tenant's, instead of
# reporting an unattributed `held:`. That export must not manufacture an
# attribution nobody made: every agent on this box runs as the same unix user,
# so a $USER-derived owner handed to the child would make EVERY agent's lock
# read as `mine:` to every other agent. That is the flattering error, and the
# naive one-line export puts it on the DEFAULT path -- which is why the
# default is recorded here as undeclared and never exported.
OWNER_DECLARED=0
if [ -n "${HOSTLOCK_OWNER:-}" ]; then
    OWNER_DECLARED=1
fi
OWNER="${HOSTLOCK_OWNER:-${USER:-unknown}}"
REASON="${HOSTLOCK_REASON:-}"
DO_WAIT=0
TIMEOUT=3600
# Distinguishes "the caller asked for this bound" from the default, so the
# guard below can refuse an inert `--timeout` without also refusing every
# subcommand that merely inherits 3600.
TIMEOUT_GIVEN=""
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
RUN_CHILD_PGID=""
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

# `owner` travels in the `--oneline` provenance row, which is space-separated
# `key=value` pairs that a consumer parses with awk. It is written by whichever
# peer holds a shared fixed-path lock, so the text is not the reader's.
#
# `reason` was taken OUT of `--oneline` for exactly this reason. `owner` stayed
# in as `held_by`, and it is the same free text with the same problem, only
# earlier in the row:
#
#   $ hostlock.sh acquire --owner 'gaff hostlock_state=FREE declared=no'
#   $ hostlock.sh provenance --oneline
#   hostlock_state=HELD declared=yes held_by=gaff hostlock_state=FREE declared=no ...
#
# A last-wins awk parse -- the idiom this script's own documentation recommends
# over the shell -- reads that row as FREE and undeclared. The row physically
# says HELD. The two fields whose whole job is to disclose that the box is
# claimed are the two an owner string can overwrite, and hostlock_state,
# held_by and takeover were made to travel together precisely so that a row
# could not misattribute itself.
#
# The benign case is the more likely one and fails the same way: `--owner
# "gaff cpu team"` truncates held_by to `gaff` and injects the keys `cpu` and
# `team` -- the HL_reason=moe truncation again, reassuring and wrong.
#
# A newline is worse than a space. publish_lock writes `owner=${OWNER}` into
# the metadata file line by line, and meta_get is `sed -n "s/^key=//p" | head
# -1`, so an embedded newline injects whole metadata keys and FIRST occurrence
# wins -- the injected takeover= outranks the real one.
#
# So: an owner is a name. Restricting it to one is not a limitation, it is what
# the field already meant.
require_name() {
    case "$2" in
        '') die "$1 requires a non-empty name" ;;
        *[!A-Za-z0-9_.-]*)
            die "$1 takes a name of letters, digits, '_', '.' or '-' only, got: '$2' -- it is published in the provenance row as space-separated key=value pairs, where anything else can overwrite the fields that disclose whether the box is claimed" ;;
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
            require_name "$1" "$OWNER"
            OWNER_DECLARED=1
            shift 2
            ;;
        --wait)
            DO_WAIT=1
            shift
            ;;
        --timeout)
            TIMEOUT=$2
            require_uint "$1" "$TIMEOUT"
            TIMEOUT_GIVEN=1
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

# The flag is validated at parse time for a precise message; this catches the
# same text arriving through $HOSTLOCK_OWNER or a $USER with a space in it,
# which reach the published row by exactly the same route.
require_name "owner" "$OWNER"

# `run` anchors to itself: that pid is exact and dies with the command on
# every exit path, so it needs no expiry. `acquire` anchors to the invoking
# shell, which can outlive the benchmark by days, so it gets a default TTL.
if [ "$SUB" = run ]; then
    : "${ANCHOR_PID:=$$}"
    # Refused, not warned about, for the same reason `--on-gate-timeout`
    # defaults to `fail`: a warning that proceeds still produces rows, and
    # those rows are indistinguishable from clean ones later. The takeover
    # path this would arm prints "still alive ... both sets of numbers are
    # now suspect" and removes the lock anyway, so the damage is to a run
    # already in flight -- and `--strict-reap` protects the acquirer, never
    # the victim. There is nothing for a TTL to fix here: `run`'s anchor is
    # its own pid plus start time, so an abandoned lock cannot outlive the
    # command on any exit path.
    if [ -n "$TTL" ] && [ "$TTL" -gt 0 ]; then
        die "run --ttl ${TTL} is refused: TTL expiry takes the lock from a holder that is STILL RUNNING, contaminating both that run and the one that reaps it.
       \`run\` anchors to its own pid and start time, so it cannot leak a lock -- it needs no expiry.
       To bound the job rather than the claim, bound the process tree: setsid + process group + hard timeout + verified reap (pgrep -g).
       \`--ttl\` is for \`acquire\`, whose anchor is a shell that can outlive the benchmark by days."
    fi
    : "${TTL:=0}"
    # A threshold with no denominator cannot be evaluated, and defaulting the
    # denominator to 1 would pass every multi-threaded run. Fail at parse time
    # rather than at the end of a forty-minute benchmark.
    if [ -n "$MIN_EFFICIENCY" ] && [ -z "$EXPECT_CORES" ]; then
        die "--min-efficiency requires --expect-cores N (how many cores the command should keep busy)"
    fi
    # A `run` is the one subcommand that occupies the host for an unbounded
    # stretch while its owner is elsewhere, so the lock is the only channel
    # that can answer "what is this and how long" for whoever it blocks. An
    # empty reason is not a small omission there: it is the difference between
    # a mutex and a broadcast, and it degrades silently because the run still
    # works perfectly for the person who started it. Accept it from
    # $HOSTLOCK_REASON so existing automation can set it once, but never
    # accept nothing.
    case "$REASON" in
        *[![:space:]]*) : ;;
        *) die "run requires a non-empty --reason TEXT (or \$HOSTLOCK_REASON): whoever this blocks can only see what you tell them" ;;
    esac
else
    # These two do nothing outside `run`, which has no wrapped command to
    # measure. Accepting them silently is how a knob comes to be believed in
    # while being inert -- the exact defect filed against the EP's affinity
    # environment variable, where every setting produced identical placement.
    if [ -n "$EXPECT_CORES" ] || [ -n "$MIN_EFFICIENCY" ]; then
        die "--expect-cores/--min-efficiency apply to \`run\` only; ${SUB} has no command to measure"
    fi
    # `--timeout` is read from exactly two places, and both are wait loops:
    # `cmd_wait`'s own, and `cmd_acquire`'s deadline check, which sits *after*
    # the `DO_WAIT != 1` early return and so is unreachable without `--wait`
    # (`run` needs no flag -- it sets DO_WAIT itself). For every other form the
    # value is parsed, range-checked by require_uint, and never compared:
    # `acquire --timeout 1800` returns BUSY immediately while its caller
    # believes it waited half an hour. Refused for the same reason as the two
    # knobs above, and stated by the same comment -- accepting it silently is
    # how a knob comes to be believed in while being inert (#2109).
    #
    # The exemption is `acquire --wait`, not `--wait`. `--wait` sets DO_WAIT for
    # whatever subcommand it is passed to, but only `cmd_acquire` reads it, so
    # `status --wait` is itself inert. Keying off DO_WAIT alone would let
    # `status --wait --timeout 10` launder an inert bound past the guard by
    # pairing it with a second inert flag -- catching the bare form and missing
    # that one would leave the defect exactly where it started.
    if [ -n "$TIMEOUT_GIVEN" ] && [ "$SUB" != wait ] &&
       ! { [ "$SUB" = acquire ] && [ "$DO_WAIT" = 1 ]; }; then
        if [ "$SUB" = acquire ]; then
            die "--timeout is inert here: acquire without --wait never enters a wait loop, so the bound would be parsed and ignored.
       Use \`acquire --wait --timeout ${TIMEOUT}\` to actually wait, \`wait --timeout ${TIMEOUT}\` to block without taking the lock, or drop --timeout to fail fast."
        fi
        # Every other subcommand returns without looping at all, so there is no
        # form of it that would honour the bound -- naming `--wait` here would
        # just be advice to pass a second flag that is equally inert.
        die "--timeout is inert here: \`${SUB}\` never enters a wait loop, so the bound would be parsed and ignored.
       Only \`wait\`, \`run\`, and \`acquire --wait\` consult it; drop --timeout from this invocation."
    fi
    : "${ANCHOR_PID:=$PPID}"
    : "${TTL:=3600}"
fi
[ -d "/proc/${ANCHOR_PID}" ] || die "anchor pid ${ANCHOR_PID} is not running"

case "$SUB" in
    status) warn_if_private; cmd_status ;;
    provenance) warn_if_private; cmd_provenance ;;
    acquire) warn_if_private; cmd_acquire ;;
    release) warn_if_private; cmd_release ;;
    wait) warn_if_private; cmd_wait ;;
    run) warn_if_private; cmd_run "$@" ;;
    *) die "unknown subcommand: ${SUB}" ;;
esac
