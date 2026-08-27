#!/usr/bin/env bash
#
# hostcensus.sh -- attribute the CPU load that the host lock does NOT cover.
#
# `hostlock.sh status` answers "who holds the lock, and is that holder alive".
# It cannot answer "what else is running", and that is the question every false
# HOST FREE announcement on this box has actually turned on. Three separate
# agents in one day published or nearly published a measurement into load they
# could not see: two announcements that were true about the sender and false
# about the machine, and one point-in-time reading quoted 74 minutes after it
# was taken.
#
# This is the other half of #2252 ("a dead holder with a live job is not a free
# host"), one step further out: NO holder is not a free host either.
#
#   ./scripts/hostcensus.sh                 # human table
#   ./scripts/hostcensus.sh --porcelain     # key=value rows
#   ./scripts/hostcensus.sh --fail-on-unlocked 0.5
#                                           # exit 3 if >= 0.5 unlocked cores
#
# Options:
#   --interval S            sampling window, default 2. CPU is measured as a
#                           DELTA across it (see "Why a delta" below).
#   --root DIR              attribution root, default /workspace/dev. Processes
#                           are grouped by their first path component under it.
#   --porcelain             machine-readable output
#   --fail-on-unlocked N    exit 3 when at least N cores of UNLOCKED load are
#                           running: attributable, under --root, and not in the
#                           lock holder's own worktree. A held lock therefore
#                           excuses the holder's own load and nobody else's,
#                           which is the case that actually bites -- one agent
#                           holding the lock while another's forgotten
#                           `cargo test` saturates a different worktree.
#   --hostlock PATH         hostlock.sh to consult; default: next to this script
#
# Exit codes:
#   0 ok      1 usage/error      3 unlocked load over --fail-on-unlocked
#
# WHAT THIS CANNOT DO -- read this before using it as a gate.
#
# A census is a measurement of one instant. It can prove the box is NOT free.
# It can never prove the box will STAY free, because nothing stops a peer from
# starting a 16-thread build one second after you read it. That is precisely
# how the stale readings on this box misled people: not because the reading was
# wrong when taken, but because it was quoted as a state after it had expired.
#
# Only HOLDING THE LOCK for the duration of a run makes a claim that survives
# its own measurement. Use this to catch load the lock does not cover -- never
# as a substitute for taking the lock.
#
# WHY A DELTA, AND NOT `ps` %CPU OR CUMULATIVE TIME
#
# `/proc/PID/stat` utime+stime is CPU consumed SINCE THE PROCESS STARTED, and
# `ps` %CPU is that total divided by elapsed -- both are lifetime averages. They
# fail in both directions for "is this running right now":
#
#   * a process that burned 40 minutes of CPU and has since gone idle still
#     reports a large total, so it looks like load that is not there;
#   * a 16-thread build started three seconds ago reports ~48 CPU-seconds
#     against a 3-second lifetime, and averaging hides how much it is taking
#     from you at this moment.
#
# The second is the reassuring-direction error -- it understates a job that is
# about to ruin your measurement -- so this samples twice and differences.
#
# A process that STARTS during the window has no first sample to difference
# against. It is not dropped: a process that started inside the window has
# spent its whole lifetime inside it, so its lifetime CPU is exactly its
# in-window CPU. Dropping them is not a theoretical concern -- an earlier
# version did, and a 4-core burner launched 1.5s into a 3s read appeared in no
# bucket at all while the summary printed `unlocked_cores=0.000`.
#
# A pid RECYCLED onto a new process mid-window is detected by comparing field
# 22 (`starttime`) between samples and counted as an arrival rather than
# differenced against a stranger's counters.
#
# WHAT THIS CANNOT SEE, STATED RATHER THAN SWALLOWED
#
# `/proc/PID/cwd` and `/exe` are only readable for our own processes. On a
# shared box most entries belong to other users and cannot be attributed. A
# scan that quietly dropped them would under-report load -- again the
# reassuring direction -- so their CPU is summed into `(unattributable)` and
# printed. A large `unattributable_cores` means "I cannot see whose this is",
# NOT "the box is quiet".
#
# `unreadable_procs` is the separate, smaller case: a `/proc/PID/stat` that
# could not be read at all, almost always a process that exited between the
# directory listing and the open. It is reported for the same reason -- so that
# a scan which saw less than it should says so.
#
# Load that ARRIVES mid-window is counted (see the arrivals branch below).
# Load that DEPARTS mid-window is not: a process seen in the first sample and
# gone by the second has an in-window CPU that cannot be recovered, since the
# second reading it would be differenced against does not exist. It is dropped
# rather than guessed. That is a real under-count, and it is the one place this
# tool errs in the reassuring direction on purpose -- but it under-reports load
# that has already stopped competing with you, whereas an arrival is load that
# is competing with you right now and will still be there when you start.
set -eu

interval=2
root=/workspace/dev
porcelain=0
fail_on_unlocked=""
hostlock=""

die() { printf 'hostcensus: %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --interval) [ $# -ge 2 ] || die "--interval needs a value"; interval=$2; shift 2 ;;
        --root) [ $# -ge 2 ] || die "--root needs a value"; root=$2; shift 2 ;;
        --porcelain) porcelain=1; shift ;;
        --fail-on-unlocked)
            [ $# -ge 2 ] || die "--fail-on-unlocked needs a value"
            fail_on_unlocked=$2; shift 2 ;;
        --hostlock) [ $# -ge 2 ] || die "--hostlock needs a value"; hostlock=$2; shift 2 ;;
        -h|--help) sed -n '2,/^set -eu/p' "$0" | sed '$d' | sed 's/^#\{1\} \{0,1\}//'; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

case "$interval" in
    ''|*[!0-9.]*|.) die "--interval must be a number, got: $interval" ;;
esac
awk -v v="$interval" 'BEGIN { exit !(v > 0) }' </dev/null \
    || die "--interval must be greater than zero, got: $interval"
if [ -n "$fail_on_unlocked" ]; then
    case "$fail_on_unlocked" in
        ''|*[!0-9.]*|.) die "--fail-on-unlocked must be a number, got: $fail_on_unlocked" ;;
    esac
fi
case "$root" in
    /*) ;;
    *) die "--root must be absolute, got: $root" ;;
esac
root=${root%/}

if [ -z "$hostlock" ]; then
    hostlock="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)/hostlock.sh"
fi

ticks=$(getconf CLK_TCK 2>/dev/null || echo 100)

# Everything this tool's own work does is either excluded or structurally
# invisible, and the distinction matters because the obvious guard is wrong.
#
# Excluding by PROCESS GROUP is the tempting version and it under-counts: with
# job control off -- a wrapper script, a CI step, `hostcensus.sh &` -- the
# census shares its caller's group, so it would exclude every SIBLING in that
# group. A benchmark launched next to the census by the same driver is exactly
# the load a user most needs to see, and pgid exclusion hides it. Verified: an
# identical burner is reported at 4.120 cores when started with `setsid` and
# vanishes entirely when started as a plain `&` sibling.
#
# So only this process is skipped -- and for that to be sufficient, this
# process has to be the only one doing the scanning. It is: `sample` writes
# into a variable in the CURRENT shell rather than being called as `$(sample)`,
# so no subshell ever exists while /proc is being read, and the one pid that
# accrues the scan's cost is `$$`.
#
# That is not a stylistic preference, it is the whole guarantee. An earlier
# version ran each scan in `$(sample)`, and argued that the subshell needed no
# guard because a pid present in only ONE sample has no delta. That argument
# was true until mid-window arrivals started being counted -- an arrival IS a
# pid present in the second sample only, which is exactly what the second
# scan's subshell is. Review caught it; the census was charging its own scan
# cost to `(unattributable)`, and therefore to `ungated_cores`, on every run.
#
# It went unnoticed because it only fires when the sampler reads its own
# /proc entry LATE in the scan, and `/proc/[0-9]*` globs in LEXICOGRAPHIC
# order, so a 7-digit pid beginning `10` is read very early and had accrued
# under one tick. Reversing the scan order made it appear immediately and
# reproducibly at 4 ticks -- 0.04 cores at `--interval 1`, against a
# significance floor of 0.05. The safety was an accident of pid ordering,
# which is not a safety at all.
self_pid=$$

# One sample: "pid starttime cputicks" per line, plus a count of what we could
# not read. Only `/proc/PID/stat` is touched here -- no `readlink`, no `exec`
# per process -- because attribution is deferred to the few PIDs that turn out
# to have moved. On this box that is ~1200 stat reads versus ~1200 execs.
# Sets `_rows` (and must NOT be called in a command substitution -- see above).
sample() {
    _unreadable=0
    _rows=""
    for _p in /proc/[0-9]*; do
        _pid=${_p#/proc/}
        [ "$_pid" = "$self_pid" ] && continue
        _line=""
        # The redirection, not just `read`, has to be inside the silenced
        # group: a process that exits between the glob and the open makes the
        # SHELL print the error, and `read ... 2>/dev/null` does not cover it.
        # shellcheck disable=SC2162
        if ! { read _line < "$_p/stat"; } 2>/dev/null || [ -z "$_line" ]; then
            _unreadable=$((_unreadable + 1))
            continue
        fi
        # `comm` is parenthesised and may itself contain spaces and ')', so
        # split after the LAST ')' rather than tokenising from the left.
        _rest=${_line##*') '}
        # shellcheck disable=SC2086
        set -- $_rest
        # $1 is overall field 3 (state), so utime/stime/starttime (14/15/22)
        # are $12/$13/$20 here.
        [ $# -ge 20 ] || { _unreadable=$((_unreadable + 1)); continue; }
        # Accumulated in-shell, not printed: `$(sample)` would fork.
        _rows="$_rows$_pid ${20} $(( ${12} + ${13} ))
"
    done
    _rows="${_rows}UNREADABLE $_unreadable"
}

sample; first=$_rows
sleep "$interval"
sample; second=$_rows

lock_state=unknown
lock_holder=
lock_reason=
lock_group=
if [ -x "$hostlock" ]; then
    lock_out=$("$hostlock" status --porcelain 2>/dev/null || true)
    lock_state=$(printf '%s\n' "$lock_out" | sed -n 's/^state=//p' | head -1)
    lock_holder=$(printf '%s\n' "$lock_out" | sed -n 's/^owner=//p' | head -1)
    [ -n "$lock_holder" ] || lock_holder=$(printf '%s\n' "$lock_out" | sed -n 's/^held_by=//p' | head -1)
    lock_reason=$(printf '%s\n' "$lock_out" | sed -n 's/^reason=//p' | head -1)
    lock_wt=$(printf '%s\n' "$lock_out" | sed -n 's/^worktree=//p' | head -1)
    [ -n "$lock_state" ] || lock_state=unknown
    # The holder's worktree is what makes "unlocked" mean something more than
    # "the lock is free". A lock held by worktree X does not account for a
    # suite running in worktree Y, and that exact case -- one agent holding
    # while another's forgotten `cargo test` ran elsewhere -- is what put a
    # contaminated measurement one minute from publication on this box.
    case "$lock_wt" in
        "$root"/*) lock_group=${lock_wt#"$root"/}; lock_group=${lock_group%%/*} ;;
    esac
fi

# Difference the two samples, then attribute only the PIDs that actually moved.
# The samples are tagged rather than inferred from order: the END block has to
# know which set a pid was missing from, and "whichever one I saw it in" is not
# recoverable from the concatenation.
moved=$(
    {
        printf '%s\n' "$first" | sed 's/^/A /'
        printf '%s\n' "$second" | sed 's/^/B /'
    } | awk -v ticks="$ticks" -v iv="$interval" '
        $2 == "UNREADABLE" { unreadable[$1] = $3; next }
        $1 == "A" { startA[$2] = $3; cpuA[$2] = $4; next }
        {
            pid = $2
            if (pid in startA && startA[pid] == $3) {
                # Present in both, same process: an ordinary delta.
                d = $4 - cpuA[pid]
                if (d > 0) printf "%s %.4f\n", pid, d / ticks / iv
                next
            }
            # Either the pid appeared during the window, or it was RECYCLED
            # onto a new process (different starttime) -- differencing those
            # two counters would invent load out of a stranger. Both are
            # arrivals.
            #
            # An earlier version dropped these, and that was the tool lying in
            # its own reassuring direction: a peer launching a 16-thread build
            # 1.5s into a 2s read landed in no bucket at all, and the box
            # certified as unlocked=0.000. Measured, it was 4.1 cores.
            #
            # They can be measured exactly rather than merely disclosed: a
            # process that started inside the window has spent its ENTIRE
            # lifetime inside it, so its lifetime CPU *is* its in-window CPU.
            # This errs in exactly one way: a process that already existed but
            # whose sample-1 stat could not be read has its whole lifetime
            # charged to this window. That over-reports, which is the safe
            # direction for a tool whose job is to stop people trusting a quiet
            # reading. (It used to err in a second way: a subshell running
            # the second scan is itself a pid present in the second sample
            # only, so the census was charged its own scan cost. That is why
            # `sample` now runs in this shell and not in `$(...)`.)
            if ($4 > 0) printf "%s %.4f\n", pid, $4 / ticks / iv
        }
        END {
            u = (unreadable["A"] > unreadable["B"] ? unreadable["A"] : unreadable["B"])
            printf "UNREADABLE %d\n", u
        }
    '
)

unreadable=$(printf '%s\n' "$moved" | awk '$1 == "UNREADABLE" { print $2 }')
[ -n "$unreadable" ] || unreadable=0

# Attribution pass: readlink only the movers.
attributed=$(
    printf '%s\n' "$moved" | while read -r pid cores; do
        [ "$pid" = "UNREADABLE" ] && continue
        path=$(readlink "/proc/$pid/cwd" 2>/dev/null || true)
        [ -n "$path" ] || path=$(readlink "/proc/$pid/exe" 2>/dev/null || true)
        if [ -z "$path" ]; then
            group="(unattributable)"
        else
            case "$path" in
                "$root"/*)
                    rel=${path#"$root"/}
                    group=${rel%%/*}
                    ;;
                *) group="(outside $root)" ;;
            esac
        fi
        printf '%s\t%s\t%s\n' "$group" "$cores" "$pid"
    done
)

summary=$(
    printf '%s\n' "$attributed" | awk -F'\t' '
        NF < 2 { next }
        { cores[$1] += $2; procs[$1] += 1 }
        END { for (g in cores) printf "%.3f\t%d\t%s\n", cores[g], procs[g], g }
    ' | sort -rn
)

total=$(printf '%s\n' "$summary" | awk -F'\t' '{ t += $1 } END { printf "%.3f", t + 0 }')
unattr=$(printf '%s\n' "$summary" | awk -F'\t' '$3 == "(unattributable)" { t += $1 } END { printf "%.3f", t + 0 }')
outside=$(printf '%s\n' "$summary" | awk -F'\t' -v o="(outside $root)" '$3 == o { t += $1 } END { printf "%.3f", t + 0 }')

# Unlocked load: attributable, under --root, and not the lock holder's own
# worktree. The two excluded buckets above are reported but never gated -- a
# gate that fired on the ambient CPU of processes it cannot even identify would
# fire on every run, and a gate that fires on every run is one nobody passes.
# The cost of that choice is stated in the summary rather than hidden: a clean
# `unlocked_cores` is not a certificate, it is the absence of load THIS TOOL
# CAN SEE.
unlocked=$(
    printf '%s\n' "$summary" | awk -F'\t' \
        -v skip="$lock_group" -v held="$lock_state" -v o="(outside $root)" '
        NF < 3 { next }
        $3 == "(unattributable)" || $3 == o { next }
        held == "HELD" && skip != "" && $3 == skip { next }
        { t += $1 }
        END { printf "%.3f", t + 0 }
    '
)
loadavg=$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo "? ? ?")

# Load that is real, measured, and NOT judged by the gate. Reported as its own
# number because the alternative is a headline `unlocked=0.000` printed while
# 1.5 cores burn outside --root -- a reassuring summary standing in front of
# the evidence that contradicts it, which is the whole failure this tool was
# written to end.
ungated=$(awk -v a="$outside" -v b="$unattr" 'BEGIN { printf "%.3f", a + b }' </dev/null)

if [ "$porcelain" -eq 1 ]; then
    printf 'lock_state=%s\n' "$lock_state"
    printf 'lock_holder=%s\n' "${lock_holder:-none}"
    printf 'lock_reason=%s\n' "${lock_reason:-}"
    printf 'lock_group=%s\n' "${lock_group:-none}"
    printf 'interval=%s\n' "$interval"
    printf 'root=%s\n' "$root"
    printf 'active_cores=%s\n' "$total"
    printf 'unlocked_cores=%s\n' "$unlocked"
    printf 'ungated_cores=%s\n' "$ungated"
    printf 'outside_root_cores=%s\n' "$outside"
    printf 'unattributable_cores=%s\n' "$unattr"
    printf 'unreadable_procs=%s\n' "$unreadable"
    printf 'loadavg=%s\n' "$loadavg"
    printf '%s\n' "$summary" | awk -F'\t' 'NF >= 3 { printf "group cores=%s procs=%s name=%s\n", $1, $2, $3 }'
else
    printf 'host census  (%ss window, root %s)\n' "$interval" "$root"
    printf '  lock: %s' "$lock_state"
    [ -n "$lock_holder" ] && [ "$lock_holder" != "none" ] && printf ' (held by %s%s)' \
        "$lock_holder" "${lock_reason:+ -- $lock_reason}"
    printf '\n  loadavg: %s\n\n' "$loadavg"
    if [ -z "$summary" ]; then
        printf '  no process consumed measurable CPU during the window\n'
    else
        printf '  %8s  %5s  %s\n' cores procs where
        printf '%s\n' "$summary" | awk -F'\t' -v skip="$lock_group" -v held="$lock_state" '
            NF >= 3 {
                tag = (held == "HELD" && skip != "" && $3 == skip) ? "   <- lock holder" : ""
                printf "  %8s  %5s  %s%s\n", $1, $2, $3, tag
            }'
    fi
    printf '\n  unlocked: %s cores   (outside root: %s   unattributable: %s   unreadable /proc entries: %s)\n' \
        "$unlocked" "$outside" "$unattr" "$unreadable"
    if awk -v u="$ungated" 'BEGIN { exit !(u >= 0.05) }' </dev/null; then
        printf '  NOTE: a further %s cores are running that this gate does not judge\n' "$ungated"
        printf '        (outside --root, or not attributable to any worktree).\n'
        printf '        The unlocked figure is a verdict on THIS root, not on the box.\n'
    fi
    if [ "$lock_state" = "FREE" ]; then
        printf '  NOTE: the lock is FREE, so nothing above is accounted for by a holder.\n'
    fi
fi

if [ -n "$fail_on_unlocked" ]; then
    if awk -v t="$unlocked" -v f="$fail_on_unlocked" 'BEGIN { exit !(t >= f) }' </dev/null; then
        printf 'hostcensus: %s unlocked cores active, threshold %s -- the host is not free\n' \
            "$unlocked" "$fail_on_unlocked" >&2
        exit 3
    fi
fi
exit 0
