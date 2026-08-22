# Decode pool width scaling: `ONNX_GENAI_CPU_DECODE_THREADS` is not inert at `=2`

**Date:** 2026-08-22
**Host:** AMD EPYC 9V74, 32 vCPU = 16 physical cores (SMT siblings adjacent, so
physical cores are the even CPUs), two 32 MiB L3 instances, single NUMA node.
**Build:** `int4_decode_loop_ab`, release, pure native CPU EP (no MLAS, no ORT
fallback). Main at `3e2be5e92`.
**Workload:** qwen, block 32, `PROBE_ACCURACY=0`, `PROBE_SESSIONS=1`, pinned
`taskset -c 0,2,...,30`.

## Why this exists

Two merged documents recorded that `ONNX_GENAI_CPU_DECODE_THREADS=2` produced
timings identical to `=1` and concluded that the second worker was parked rather
than computing — "a dispatch/wakeup defect in the persistent decode pool at small
worker counts, not a measurement artifact".

It is a measurement artifact. Under control the knob scales.

## Result

One process per launch, arms interleaved round-robin, per-rep load guard on the
instantaneous runnable count (`/proc/loadavg` field 4), cells over the guard
discarded rather than kept (7 discarded in the published run).

| width | ms/token (min) | tok/s | speedup vs `=1` | reps | spread |
|---:|---:|---:|---:|---:|---:|
| 1 | 40.039 | 24.7 | 1.00x | 2 | 2.1% |
| 2 | **20.447** | **48.6** | **1.96x** | 6 | 1.7% |
| 4 | 10.442 | 94.8 | 3.83x *(provisional)* | 2 | **46.4%** |
| 8 | 5.326 | 187.8 | 7.52x | 3 | 3.1% |
| 16 | 4.324 | 231.3 | 9.26x | 2 | 1.2% |

**A/A null control (both arms at width 2): 0.6%.** The 1.96x clears it by more
than two orders of magnitude. `=2` reproduced at 20.447 / 20.575 / 20.562 /
20.636 ms/token across three independent windows, including one contended one.

**The `t=4` cell is not measured and is marked provisional.** Its two reps were
10.442 and 15.283 — a 46.4% spread, bimodal in the way this host is known to be
per process launch. The minimum is quoted for consistency with the other rows,
but by the standard this repository already sets — "do not pick the flattering
statistic, extend the run", and a cell whose estimators disagree in direction has
not been measured — 3.83x is a waypoint awaiting reps, not a result. It is not
load-bearing for anything below.

The shape of the curve is therefore: a solid 1.96x at `t=2`, solid points at
`t=8` and `t=16`, and a knee into `t=16` (7.52x to 9.26x — only 1.23x for the
last doubling) consistent with the memory-bandwidth plateau the pool default is
sized for. Whether the `t=1`-to-`t=8` segment is *linear* depends on the
unmeasured `t=4` point and is not claimed here.

## The parked-worker reading is falsified directly

Per-thread attribution from `/proc/<pid>/task/*/stat` and `status`, deltas taken
across a steady window only so pool construction and model load cannot
contaminate them:

| width | SPMD workers | CPU-s per 6 s window | busy | voluntary ctxsw |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0.00 | 0% | 0 |
| 2 | 2 | 11.91 | **99%** | 62 |
| 4 | 4 | 23.00 | 96% | 180 |

At `=2` both workers are 99% busy with 62 voluntary context switches over six
seconds. A parked worker shows the inverse: near-zero CPU and *high* voluntary
ctxsw. They are computing.

**`t=1` is the row that is genuinely special.** Its worker is 0% busy and the
dispatcher does all the work, because `dispatch_output_rows` takes its
`self.total_workers <= 1` serial short-circuit on every op. `t=1` therefore
measures "serial on the dispatcher", not "the pool with one worker", and it
spawns a pinned worker that never receives a dispatch. Comparisons against `t=1`
are comparisons against a different code path.

## Why the original reading looked so convincing

`Percent of CPU` from `/usr/bin/time` is `(user+sys)/wall`, so it is
**wall-derived** and degrades under contention exactly like wall time. It is not
an independent check on a wall-time result. The `user` column *is* robust, and it
was flat across widths in the original data — which is consistent with correct
work division, not with a stall.

The stronger clue was arithmetic: the original pair, 23.529 vs 23.527 ms/token,
agree to **four significant figures**. Contention is random and does not produce
that; two configurations reading the same number to 0.008% are the same
configuration. That points at the width not reaching the measured process, not at
the scheduler.

## Non-vacuity check (categorical, works on a loaded host)

Thread construction does not depend on load, so this is valid at any load and
takes thirty seconds:

```bash
ONNX_GENAI_CPU_DECODE_THREADS=$w taskset -c 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30 $BIN &
sleep 6; for t in /proc/$P/task/*; do cat "$t/comm"; done | grep -c onnx-genai-spmd
```

`w` in must give `w` out. Verified 1→1, 2→2, 4→4, 8→8. `NXRT_CALIB_DEBUG=1`
additionally prints the selected path (`persistent SPMD pool` at every width
here), which rules out a silent fallback.

## Two configurations where the knob *is* genuinely vacuous

Neither applies to the runs above, but both are real and worth a guard:

1. **In-process width sweeps.** `static POOLS: OnceLock<Option<SpmdDecodePools>>`
   builds the pool **once per process**, at first decode, from whatever the
   environment said at that moment. A harness that loops widths inside one
   process reports the first width for every later cell. Launch a fresh process
   per cell. (`crates/onnx-runtime-ep-cpu/benches/acc0_gap_matrix.py` already
   does — its `native()` shells out per cell.)

2. **`THREADS=N` on an exactly-N-CPU cpuset**, via
   `reserve_single_group_headroom(N, N) == N-1`, which reserves one allowed CPU
   for the dispatcher. At `N=2` that yields **one** worker, which then triggers
   the `total_workers <= 1` serial short-circuit above: all work runs on the
   dispatcher while a pinned worker sits parked, with no diagnostic. Benchmarking
   inside a small container hits this.

## Addendum (#1746): the container case is not container-only

**Date:** 2026-08-22. Added by Sebastian after this document merged.

Vacuity case 2 above — `THREADS=N` on an exactly-N-CPU cpuset, via
`reserve_single_group_headroom(N, N) == N-1` — is described here as something
*"benchmarking inside a small container hits"*. It is not confined to
containers. **The shipping EP builds that cpuset itself**, so the condition
fires on every explicit budget in production.

`provider.rs:340` — `EpFactory::initialize()`, the earliest per-session hook —
calls `bound_process_to_decode_budget()`, which confines the process to exactly
`N` CPUs so that "a user who caps cores disturbs at most N CPUs". The pool is
then built lazily at first decode, reads `allowed_cpus()`, finds exactly `N`,
and reserves one for the dispatcher.

Measured through the production sequence, under the *same* outer
`taskset -c 0,2,...,30` this document's runs used:

```
onnx-genai: CPU decode budget 2 confined the process to 2 CPUs [0, 2]
PROD budget=2 allowed_before=Some(16) allowed_after=Some(2) spmd_threads=1
PROD budget=4 allowed_before=Some(16) allowed_after=Some(4) spmd_threads=3
PROD budget=8 allowed_before=Some(16) allowed_after=Some(8) spmd_threads=7
```

`allowed_before=16 -> allowed_after=N` is the mechanism. At `N=2` the single
remaining worker trips the `total_workers <= 1` serial short-circuit this
document already identifies as special-casing `t=1` — so in production `=2` and
`=1` really are the same code path.

### This does not retract anything above

The measurements in this document are correct **for the harness they were taken
on**, and the 1.96x stands. `crates/onnx-runtime-ep-cpu/benches/int4_decode_loop_ab.rs`
never calls `EpFactory::initialize()`; it constructs `CpuExecutionProvider` and
enters the decode scope through `with_decode_pool_scope` directly. So
`bound_process_to_decode_budget()` never runs there, the process keeps the 16
CPUs the outer `taskset` gave it, `reserve_single_group_headroom(2, 16) = 2`,
and the bench genuinely gets two busy workers. The per-thread attribution table
is sound.

That is also why the non-vacuity check passed: `w` in gave `w` out because the
reservation never fired in that binary.

**The finding is the divergence, not an error in either measurement.** The
decode bench does not reproduce the production thread topology, so a width sweep
run through it is structurally unable to observe this class of defect. Any
thread-scaling curve taken from `int4_decode_loop_ab` describes the harness.

### One correction to the table

The `t=16` row is a **15-worker** measurement. Under
`taskset -c 0,2,...,30` the bench has 16 allowed CPUs, so
`reserve_single_group_headroom(16, 16) = 15` fires even without the production
confinement. The `1/2/4/8` non-vacuity checks could not have caught this,
because the reservation only triggers at exactly full subscription. The knee
into `t=16` is therefore measured at 15 lanes, not 16 — which slightly
*understates* the plateau rather than overstating it, so the conclusion drawn
from it is unaffected.

### Fix

#1746 gives the reserved CPU a compute lane instead of only a spinning
dispatcher: `N-1` pinned worker threads (so the 20-60x starvation cliff that
motivated the reservation stays fixed) plus the dispatcher computing the
remaining shard = `N` lanes on `N` CPUs. Verified `N` in -> `N` lanes for
`N = 2/4/8/16` through the production sequence, with a subprocess regression
test per budget.
