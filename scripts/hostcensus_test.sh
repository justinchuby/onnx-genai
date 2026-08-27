#!/usr/bin/env bash
# hostcensus_test.sh -- self-test for hostcensus.sh
#
# Run: scripts/hostcensus_test.sh
#
# Cost: three ~2s single-cpu spinners, so roughly six core-seconds. That is not
# zero, and it is deliberate for the same reason `hostlock_test.sh` pays for its
# R2 cells: a census that always reported `active_cores=0.000` would pass every
# argument-parsing test in this file and silently certify every contended host
# as free. The load cells are the only thing standing between this tool and
# that failure, so they measure a REAL spinner and prove the number tracks it.
#
# EVERY ASSERTION HERE MUST HOLD ON A CO-TENANTED HOST. No cell asserts an
# absolute core count that only a quiet box can produce. The load cells assert
# a RELATIVE fact -- the same group reads higher with a spinner in it than
# without -- which is immune to whatever else the box is doing, plus a floor
# (0.05 cores) an order of magnitude above the idle noise measured here.
#
# The spinners are pinned to one cpu and capped with `timeout`, so a failure
# anywhere in this file cannot leave a burner behind. That cap is a deadman,
# not the cost.
#
# NOT TESTED, AND SAID SO RATHER THAN FAKED: the PID-reuse guard (a second
# sample whose `starttime` differs is dropped). Forcing a pid to be recycled
# onto a new process inside a window we control is not something a test can do
# deterministically, and a test that re-derived the guard's own arithmetic
# would assert itself rather than the script. That branch is carried by
# reasoning and by the comment at its site.

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
# Absolute, deliberately. A relative path silently resolves to nothing in any
# cell that `cd`s first -- and a census that produced no output at all passed
# every "this group is absent" assertion, because absent-because-broken and
# absent-because-excluded are the same string. One cell was already vacuous
# for exactly this reason before it was noticed.
CENSUS="$(pwd)/scripts/hostcensus.sh"
WORK="$(pwd)/.hostcensus-selftest"

pass=0
fail=0

cleanup() {
    rm -rf "$WORK" 2>/dev/null
}
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

# Start a spinner whose cwd is $1, capped at $2 seconds. Echoes its pid.
#
# The cap is what makes this suite safe to abort: every spinner dies on its own
# whether or not the cell that started it reaches its cleanup.
spin_in() {
    local dir="$1" cap="$2" pid
    mkdir -p "$dir"
    ( cd "$dir" && exec timeout "$cap" taskset -c 0 \
        sh -c 'while :; do :; done' ) >/dev/null 2>&1 &
    pid=$!
    echo "$pid"
}

reap() {
    kill "$1" 2>/dev/null
    wait "$1" 2>/dev/null
}

# Extract one porcelain field.
field() {
    printf '%s\n' "$1" | sed -n "s/^$2=//p" | head -1
}

# Whether group $2 appears at all in porcelain output $1.
#
# Distinct from `group_cores` and the distinction is load-bearing: a group
# emitted with `cores=0.000` reads as zero to any numeric comparison, so a
# census that listed every idle process at zero cores would satisfy
# "cores == 0" while doing exactly the thing that assertion forbids. Mutation
# M2 (`d >= 0`) proved that: it passed 38/38 against the numeric form.
group_present() {
    printf '%s\n' "$1" | awk -v want="$2" '
        {
            if (match($0, /^group cores=[0-9.]+ procs=[0-9]+ name=/)) {
                if (substr($0, RLENGTH + 1) == want) { print "present"; found = 1; exit }
            }
        }
        END { if (!found) print "absent" }
    '
}

# Cores attributed to group $2 in porcelain output $1, or 0 when absent.
#
# Exact string comparison in awk rather than a `sed` substitution: group names
# are paths -- `(outside /some/dir)` -- and a `/` inside a `s///` pattern is a
# delimiter, not a character. The first version of this helper did use `sed`,
# and the cell it broke was the one asserting that load outside --root is not
# dropped. It reported "dropped", i.e. it accused the script of exactly the
# reassuring-direction error the script was written to avoid, and the fault was
# in the test.
group_cores() {
    printf '%s\n' "$1" | awk -v want="$2" '
        {
            if (match($0, /^group cores=[0-9.]+ procs=[0-9]+ name=/)) {
                name = substr($0, RLENGTH + 1)
                if (name == want) {
                    sub(/^group cores=/, "")
                    sub(/ procs=.*$/, "")
                    print
                    found = 1
                    exit
                }
            }
        }
        END { if (!found) print 0 }
    '
}

echo "== usage =="

out=$("$CENSUS" --interval 0 2>&1; echo "rc=$?")
chk "a zero interval is refused, not divided by" \
    "$(printf '%s' "$out" | sed -n 's/.*rc=//p')" "1"
chk "and says which argument was wrong" \
    "$(printf '%s' "$out" | grep -c 'interval must be greater than zero')" "1"

out=$("$CENSUS" --interval abc 2>&1; echo "rc=$?")
chk "a non-numeric interval is refused" \
    "$(printf '%s' "$out" | sed -n 's/.*rc=//p')" "1"

out=$("$CENSUS" --interval 2>&1; echo "rc=$?")
chk "a flag missing its value is refused rather than silently defaulted" \
    "$(printf '%s' "$out" | sed -n 's/.*rc=//p')" "1"

out=$("$CENSUS" --root relative/path 2>&1; echo "rc=$?")
chk "a relative --root is refused: attribution is prefix matching on /proc paths" \
    "$(printf '%s' "$out" | sed -n 's/.*rc=//p')" "1"

out=$("$CENSUS" --fail-on-unlocked nope 2>&1; echo "rc=$?")
chk "a non-numeric threshold is refused" \
    "$(printf '%s' "$out" | sed -n 's/.*rc=//p')" "1"

out=$("$CENSUS" --nonsense 2>&1; echo "rc=$?")
chk "an unknown argument is refused rather than ignored" \
    "$(printf '%s' "$out" | sed -n 's/.*rc=//p')" "1"

out=$("$CENSUS" --help 2>&1; echo "rc=$?")
chk "--help exits clean" "$(printf '%s' "$out" | sed -n 's/.*rc=//p')" "0"
chk "and carries the warning that a census cannot certify a quiet future" \
    "$(printf '%s' "$out" | grep -c 'never prove the box will STAY free')" "1"

echo
echo "== attribution and the load it must actually see =="

# The central cell. Without it every other assertion in this file passes
# against a census that reports nothing at all.
mkdir -p "$WORK/proj-a/nested"
burner=$(spin_in "$WORK/proj-a/nested" 25)
sleep 0.5
hot=$("$CENSUS" --interval 2 --root "$WORK" --porcelain 2>/dev/null)
hot_cores=$(group_cores "$hot" "proj-a")
reap "$burner"
sleep 0.5
cold=$("$CENSUS" --interval 2 --root "$WORK" --porcelain 2>/dev/null)
cold_cores=$(group_cores "$cold" "proj-a")

chk "a spinner in <root>/proj-a is attributed to proj-a, not to its subdirectory" \
    "$(awk -v c="$hot_cores" 'BEGIN { print (c >= 0.05) ? "seen" : "missed" }')" "seen"
chk "and the reading falls once it stops -- the number tracks load, not process count" \
    "$(awk -v h="$hot_cores" -v c="$cold_cores" 'BEGIN { print (h > c) ? "fell" : "flat" }')" "fell"
chk "active_cores is non-zero while it runs" \
    "$(awk -v c="$(field "$hot" active_cores)" 'BEGIN { print (c >= 0.05) ? "load" : "none" }')" "load"

# A process that exists but does no work must not appear. This is what
# separates a census from `ps`: idle processes are not contention.
mkdir -p "$WORK/proj-idle"
( cd "$WORK/proj-idle" && exec timeout 12 sleep 10 ) >/dev/null 2>&1 &
idler=$!
sleep 0.5
idle_out=$("$CENSUS" --interval 2 --root "$WORK" --porcelain 2>/dev/null)
reap "$idler"
chk "an idle process is not reported at all, not reported as zero" \
    "$(group_present "$idle_out" "proj-idle")" "absent"

# A burner that shares the census's PROCESS GROUP must still be counted.
#
# This cell exists because of the guard it forbids. Excluding by process group
# looks like the right way to stop the instrument reading itself, and it
# silently hides SIBLINGS: with job control off -- a wrapper script, a CI step,
# `hostcensus.sh &` -- the census shares its caller's group, so a benchmark
# launched next to it by the same driver vanishes from the reading. That is the
# load a user most needs to see.
#
# Note the ordinary spinners above cannot test this: `timeout` puts its child
# in a NEW process group, which is exactly why they are visible either way. So
# this one is self-limiting instead of `timeout`-capped, and the precondition
# -- that it really did land in our group -- is asserted rather than assumed.
# It is `bash -c`, not `sh -c`, because `$SECONDS` is a bash builtin: under
# dash the loop exits immediately, the burner is gone before the census reads
# it, and the cell goes green for the wrong reason. That is not hypothetical:
# the first version of this cell did exactly that.
mkdir -p "$WORK/proj-pg"
# shellcheck disable=SC2016  # $SECONDS must expand in the burner, not here
( cd "$WORK/proj-pg" && exec taskset -c 0 \
    bash -c 'e=$((SECONDS+9)); while [ $SECONDS -lt $e ]; do :; done' ) >/dev/null 2>&1 &
pg_burner=$!
own_pgid=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
burner_pgid=$(ps -o pgid= -p "$pg_burner" 2>/dev/null | tr -d ' ')
chk "precondition: the burner really is in this shell's process group" \
    "$burner_pgid" "$own_pgid"

sleep 0.5
same_pg=$("$CENSUS" --interval 2 --root "$WORK" --porcelain 2>/dev/null)
reap "$pg_burner"
chk "a sibling in the census's own process group is counted, not hidden" \
    "$(awk -v c="$(group_cores "$same_pg" "proj-pg")" 'BEGIN { print (c >= 0.05) ? "seen" : "hidden" }')" \
    "seen"

# Load that ARRIVES during the sampling window has no first sample to
# difference against, and an earlier version dropped it entirely: a burner
# started 1.5s into a 3s read appeared in no bucket at all while the summary
# printed unlocked_cores=0.000. That is the tool's own failure mode -- a
# reassuringly low number on a busy box -- so it gets its own cell.
mkdir -p "$WORK/proj-late"
arrive_out="$WORK/arrive.txt"
( "$CENSUS" --interval 4 --root "$WORK" --porcelain >"$arrive_out" 2>/dev/null ) &
census_bg=$!
sleep 2
late_burner=$(spin_in "$WORK/proj-late" 12)
wait "$census_bg" 2>/dev/null
late_read=$(cat "$arrive_out")
reap "$late_burner"
chk "load that arrives mid-window is counted, not dropped" \
    "$(awk -v c="$(group_cores "$late_read" "proj-late")" 'BEGIN { print (c >= 0.05) ? "counted" : "dropped" }')" \
    "counted"

# Work outside --root is still counted, in its own bucket. Dropping it would
# under-report the box in the reassuring direction.
outside2=$(spin_in "$WORK/proj-a" 30)
sleep 0.5
narrow=$("$CENSUS" --interval 2 --root "$WORK/proj-b" --porcelain 2>/dev/null)
chk "load outside --root is bucketed, not dropped" \
    "$(awk -v c="$(group_cores "$narrow" "(outside $WORK/proj-b)")" 'BEGIN { print (c >= 0.05) ? "bucketed" : "dropped" }')" \
    "bucketed"

# ...and the human summary must SAY that its headline number does not cover it.
# A gate that reports `unlocked: 0.000` above a line showing 1.5 cores burning
# outside its root is a reassuring summary standing in front of the evidence
# that contradicts it.
narrow_human=$("$CENSUS" --interval 2 --root "$WORK/proj-b" 2>/dev/null)
chk "and the summary warns that the gate did not judge it" \
    "$(printf '%s\n' "$narrow_human" | grep -c 'cores are running that this gate does not judge')" "1"

# `ungated` is the sum the human summary warns about, so the thing to assert is
# that it really is that sum -- not that it exceeds some number.
#
# An absolute threshold here was satisfied by ambient `unattributable` load
# whether or not the outside path worked at all; review demonstrated it by
# removing outside-bucketing entirely and leaving this cell green. The obvious
# repair -- assert it is higher with the spinner than without -- fights ambient
# noise from the other side, and did in fact flake once on a contended box.
# The identity is exact, needs no second measurement, and still fails any
# mutation that drops a component: the cell above independently pins the
# outside bucket at >= 0.05 on this same reading, so a version summing only
# `unattributable` cannot satisfy it.
chk "ungated is exactly the load the gate excluded: outside + unattributable" \
    "$(awk -v u="$(field "$narrow" ungated_cores)" \
           -v o="$(field "$narrow" outside_root_cores)" \
           -v n="$(field "$narrow" unattributable_cores)" \
           'BEGIN { print (u - (o + n) < 0.002 && (o + n) - u < 0.002) ? "exact" : "wrong" }')" \
    "exact"
reap "$outside2"

echo
echo "== porcelain contract =="

for key in lock_state lock_holder lock_group interval root active_cores \
           unlocked_cores ungated_cores outside_root_cores unattributable_cores \
           unreadable_procs loadavg; do
    chk "porcelain emits $key" \
        "$(printf '%s\n' "$idle_out" | grep -c "^$key=")" "1"
done
chk "porcelain echoes the interval it actually used" "$(field "$idle_out" interval)" "2"
chk "porcelain echoes the root it actually used" "$(field "$idle_out" root)" "$WORK"

echo
echo "== the gate judges load the lock does not cover =="

gate_burner=$(spin_in "$WORK/proj-a" 40)
sleep 0.5

# Stub holders, so the two cases that matter can be driven deterministically.
# A real second worktree cannot be conjured inside a test, and `--hostlock`
# exists precisely so the census's reading of a lock can be exercised without
# one. The cell below this group re-runs the same logic against the REAL
# hostlock.sh, so these stubs cannot quietly drift from what the tool actually
# emits.
mkdir -p "$WORK/stub"
make_stub() {
    cat > "$WORK/stub/hostlock.sh" <<EOF
#!/bin/sh
echo "state=$1"
echo "owner=$2"
echo "reason=stubbed"
echo "worktree=$3"
EOF
    chmod +x "$WORK/stub/hostlock.sh"
}

make_stub HELD peer "$WORK/proj-a"
out=$("$CENSUS" --interval 2 --root "$WORK" --hostlock "$WORK/stub/hostlock.sh" \
    --fail-on-unlocked 0.05 2>&1; echo "rc=$?")
chk "a holder's own worktree is excused" \
    "$(printf '%s' "$out" | sed -n 's/.*rc=//p')" "0"

# The incident this tool exists for: the lock is held, honestly, by someone
# whose own load is accounted for -- and a forgotten suite is saturating a
# DIFFERENT worktree. `hostlock.sh status` reports HELD and is correct; the
# box is still not quiet.
make_stub HELD peer "$WORK/proj-b"
out=$("$CENSUS" --interval 2 --root "$WORK" --hostlock "$WORK/stub/hostlock.sh" \
    --fail-on-unlocked 0.05 2>&1; echo "rc=$?")
chk "load in a worktree the holder does not own still trips the gate" \
    "$(printf '%s' "$out" | sed -n 's/.*rc=//p')" "3"
chk "and says the host is not free" \
    "$(printf '%s' "$out" | grep -c 'the host is not free')" "1"

held_other=$("$CENSUS" --interval 1 --root "$WORK" --hostlock "$WORK/stub/hostlock.sh" \
    --porcelain 2>/dev/null)
chk "the holder's group is reported even when it is not the one running" \
    "$(field "$held_other" lock_group)" "proj-b"

make_stub FREE none "$WORK/proj-a"
out=$("$CENSUS" --interval 2 --root "$WORK" --hostlock "$WORK/stub/hostlock.sh" \
    --fail-on-unlocked 0.05 2>&1; echo "rc=$?")
chk "with no holder at all, the same load trips the gate" \
    "$(printf '%s' "$out" | sed -n 's/.*rc=//p')" "3"

out=$("$CENSUS" --interval 2 --root "$WORK" --hostlock "$WORK/stub/hostlock.sh" \
    --fail-on-unlocked 9999 2>&1; echo "rc=$?")
chk "load under the threshold exits 0" \
    "$(printf '%s' "$out" | sed -n 's/.*rc=//p')" "0"

# Same reading, real tool. A private lock dir: a suite that took the real host
# lock would deadlock the box it runs on. Same device as hostlock_test.sh.
export HOSTLOCK_DIR="$WORK/lock"
export HOSTLOCK_PRIVATE_OK=1
scripts/hostlock.sh acquire --owner census-test --reason "census gate cell" \
    --pid $$ >/dev/null 2>&1
held=$("$CENSUS" --interval 1 --root "$WORK" --porcelain 2>/dev/null)
scripts/hostlock.sh release >/dev/null 2>&1
reap "$gate_burner"

chk "the real hostlock's HELD state is read" "$(field "$held" lock_state)" "HELD"
chk "and its owner field is the one the census reports as the holder" \
    "$(field "$held" lock_holder)" "census-test"

free_out=$("$CENSUS" --interval 1 --root "$WORK" 2>/dev/null)
chk "a FREE lock says nothing listed is accounted for" \
    "$(printf '%s\n' "$free_out" | grep -c 'the lock is FREE, so nothing above is accounted for')" "1"

cleanup

# The sampler must not run in a command substitution.
#
# This is a white-box assertion, deliberately, and the reason is worth stating
# because "assert behaviour, not structure" is otherwise the right rule.
#
# The defect it guards is the census charging its own scan cost to
# `(unattributable)`: a subshell running the second scan is a pid present in
# the second sample only, which is precisely the shape of a mid-window
# arrival. Measured, that is 4 ticks -- 0.04 cores at `--interval 1`. Ambient
# `unattributable` on a shared box moves by more than that between two reads
# seconds apart, so ANY behavioural cell for this would be measuring noise and
# would pass or fail regardless of the bug. A behavioural test here would be a
# tautology, and this file already shipped three of those.
#
# Worse, the bug was invisible even to a direct probe: `/proc/[0-9]*` globs
# LEXICOGRAPHICALLY, so a 7-digit pid beginning `10` reads its own entry near
# the start of the scan having accrued under a tick. It only reproduces when
# the scan order is reversed. A behavioural cell would have been green on this
# box for the wrong reason -- the same failure as a test that does not run on
# the platform whose green check is being cited.
#
# So the structural fact is the guarantee, and the structural fact is what is
# pinned.
chk "the sampler does not run in a subshell that the census would then count" \
    "$(grep -v '^[[:space:]]*#' "$CENSUS" \
       | grep -cE '\$\([[:space:]]*sample[[:space:]]*\)')" "0"

# Pin the assertion count. Several cells above are load-dependent, and an
# assertion that quietly stops running is indistinguishable from one that
# passes -- which is the whole defect this tool exists to catch, one level up.
chk "every assertion in this file ran" "$((pass + fail + 1))" "44"

echo
echo "passed=${pass} failed=${fail}"
[ "$fail" -eq 0 ]
