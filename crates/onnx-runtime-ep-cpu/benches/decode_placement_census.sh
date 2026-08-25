#!/usr/bin/env bash
# Categorical *placement* census: which CPU each SPMD decode worker is pinned to.
#
# This answers a claim that timing cannot: a cross-agent report (2026-08-24) had a
# default unpinned launch pinning 16 workers to cpus 0-15 -- 8 physical cores and a
# single 32 MiB L3 -- leaving half the machine idle. That report predates #1729
# (`6e8c31ebd`, merged 2026-08-23T01:11:35Z), which routes every shard-planning
# path through `decode_affinity::order_pin_targets` (leaders first). This reads
# the realized placement out of /proc on the tree actually built.
#
# It is a `Cpus_allowed_list` read, not a benchmark: it does not need a quiet
# host and no number here is a timing. It does spin up the pool three times at
# width 16/8, so it still runs under the hostlock as a courtesy to whoever is
# measuring -- see the re-exec below. That sentence was here before the lock
# call was, and nothing in the tree contradicted it, which is the reason
# `test_gate_conformance.py` now reads this directory's shell files too.
#
# Three configurations, because they answer three different questions:
#   1. default, no taskset      -- the configuration the report was taken in
#   2. THREADS=16, even mask    -- the configuration all my w=16 rows were taken in
#   3. THREADS=8,  even mask    -- ditto w=8 (tests the single-L3 confound)
#
# Observed on `0a668d54b` (AMD EPYC 9V74, 16 physical / 32 logical, siblings
# adjacent, L3 in two 32 MiB instances over cpus 0-15 and 16-31):
#
#   default, no taskset, no env   15 workers on 0,2,...,28   8 x L3#0 + 7 x L3#1
#   THREADS=16 under even mask    15 workers on 0,2,...,28   8 x L3#0 + 7 x L3#1
#   THREADS=8  under even mask     7 workers on 0,2,...,12   7 x L3#0
#
# One worker per physical core in every case, with the reserved dispatcher CPU
# (30 at width 16, 14 at width 8) left clear. The "two workers per core on one
# L3" report is what a pre-#1729 build does, not this one.
#
# The width-8 row is the one worth keeping in view when reading a width sweep:
# `THREADS=8` confines the process to [0,2,4,6,8,10,12,14], which is entirely
# inside one L3 instance, so a t=8 -> t=16 doubling on this host doubles cache
# and memory-controller reach as well as cores. `acc0_gap_matrix.native_pin`
# gives the ORT arm the same CPUs at each width, so the comparison is symmetric
# -- but "twice the threads" is not all that changes.
set -u
SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
cd "$(dirname "$0")/../../.." || exit 1

# Take the host lock once, for the whole census, by re-executing under it.
#
# Once, and around all three arms, because a lock taken per arm releases
# between them -- which is exactly the gap #1803 was filed over: a peer who
# sampled the host in that window read it as clear while an interleaved
# comparison was mid-flight. `hostlock.sh run` anchors the claim to its own
# pid and always releases, so a SIGKILL here cannot leak it.
#
# The sentinel is what stops the second entry from recursing. `--wait`
# because this census is not timing-sensitive: blocking behind somebody
# else's matrix costs nothing here and starting three pools on top of it
# costs them their numbers.
if [ -z "${HOSTLOCK_CENSUS_HELD:-}" ]; then
  export HOSTLOCK_CENSUS_HELD=1
  exec ./scripts/hostlock.sh run \
    --reason "decode placement census: three pool launches (default/w16/w8)" \
    --wait --timeout 1800 -- "$SELF" "$@"
fi

BIN=""
for cand in target/release/deps/int4_decode_loop_ab-*; do
  case "$cand" in *.d) continue ;; esac
  [ -x "$cand" ] && BIN="$cand" && break
done
[ -n "$BIN" ] || { echo "no bench binary built" >&2; exit 1; }
PIN=$(seq -s, 0 2 30)

L3_0=$(cat /sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list 2>/dev/null)
L3_1=$(cat /sys/devices/system/cpu/cpu16/cache/index3/shared_cpu_list 2>/dev/null)
SIB0=$(cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list 2>/dev/null)

# Every spmd thread's pinned CPU set, one per line.
#
# The name is Linux's 15-byte `comm` truncation of what the pool actually
# spawns ("onnx-genai-spmd-n0-1"), and it is duplicated here because a shell
# script cannot read the Rust const. Keep it equal to SPMD_THREAD_NAME_PREFIX
# in crates/onnx-runtime-ep-cpu/src/decode_spmd.rs: this is a *filter*, so a
# rename there empties the census silently rather than failing it.
worker_cpus() {
  local pid=$1
  for t in /proc/"$pid"/task/*; do
    [ "$(cat "$t/comm" 2>/dev/null)" = "onnx-genai-spmd" ] || continue
    grep -m1 '^Cpus_allowed_list:' "$t/status" 2>/dev/null | awk '{print $2}'
  done
}

# Which L3 instance a single-CPU mask belongs to (workers are pinned 1:1, so a
# mask like "0-31" means "not pinned" and is reported as such).
l3_of() {
  case "$1" in
    *-*|*,*) echo "unpinned" ;;
    *) [ "$1" -lt 16 ] && echo L3#0 || echo L3#1 ;;
  esac
}

run() {
  local label="$1" pin="$2"; shift 2
  local out; out=$(mktemp -p . .place.XXXXXX)
  local pfx=""
  [ -n "$pin" ] && pfx="taskset -c $pin"
  # shellcheck disable=SC2086
  env PROBE_MODEL=llama PROBE_BLOCK=32 PROBE_ACCURACY=0 PROBE_SESSIONS=1 \
      PROBE_TOKENS=6000 PROBE_REPS=1 "$@" $pfx "$BIN" > "$out" 2>&1 &
  local p=$!
  sleep 3
  local cpus; cpus=$(worker_cpus "$p" | sort -n | tr '\n' ' ')
  local n; n=$(worker_cpus "$p" | wc -l)
  local doms; doms=$(for c in $cpus; do l3_of "$c"; done | sort | uniq -c | tr '\n' ' ')
  kill "$p" 2>/dev/null; wait "$p" 2>/dev/null

  echo "== $label"
  if [ "$n" -eq 0 ]; then
    echo "   spawned spmd threads : 0  <-- NOTHING MATCHED. Either the pool"
    echo "                              spawned no workers, or the thread name"
    echo "                              no longer starts with onnx-genai-spmd."
    echo "                              An empty census is not evidence of a"
    echo "                              placement; do not read the rows below."
  else
    echo "   spawned spmd threads : $n"
  fi
  echo "   pinned cpus          : $cpus"
  echo "   l3 spread            : $doms"
  grep -m1 'confined the process to' "$out" | sed 's/^/   confinement          : /'
  grep -m1 '^decode_width' "$out" | sed 's/^/   /'
  rm -f "$out"
}

echo "host: L3#0=$L3_0  L3#1=$L3_1  siblings(cpu0)=$SIB0"
echo "binary: $BIN"
echo
run "default (no taskset, no THREADS env)" ""
run "THREADS=16 under even mask" "$PIN" ONNX_GENAI_CPU_DECODE_THREADS=16
run "THREADS=8 under even mask"  "$PIN" ONNX_GENAI_CPU_DECODE_THREADS=8
