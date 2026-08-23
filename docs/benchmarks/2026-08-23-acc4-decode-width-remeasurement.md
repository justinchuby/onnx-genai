# acc4 int4 decode — width curve re-measured, and why `w=16` resists measurement here

**Date:** 2026-08-23 · **Owner:** Roy (CPU MatMul) · **Host:** AMD EPYC 9V74,
16 physical / 32 logical, single NUMA node, AVX2+FMA+F16C, shared.

Re-measurement of the decode width curve recorded in
[`CPU_MATMUL_ASSIGNMENT.md` §20](../performance/CPU_MATMUL_ASSIGNMENT.md), which
gave `22.949/22.893/11.584/5.866/3.302 ms/token at 1/2/4/8/16 threads`, an
`8→16` ratio of `1.77x ±0.7%`, and recorded `t=1 ≡ t=2` as "unexplained ... an
open question".

Dispositions: the `t=1 ≡ t=2` cell is **wrong**; the `±0.7%` interval is
**unverifiable on this host** and withdrawn without being disproven; the
description of what width 1 actually runs was **wrong in the source material
and in my own earlier accounts of it**, and is corrected here.

The trigger was #1740, which retracted the same `t=1 ≡ t=2` signature in three
other locations after a controlled re-measurement showed `=2` is 1.96x `=1` on
acc0. That correction missed §20, and §20 is the one place the claim was framed
as an open research question rather than an observation — the framing most
likely to stop the next person from measuring it. In the same spirit, this
change also corrects two sites that *this* document's first draft missed: the
bench header in `int4_decode_loop_ab.rs` and §4 of the ntile record, both of
which still asserted the withdrawn `±0.7%`.

## Method

Same protocol that took A/A nulls to 0.04% on #1809.

* **One fresh process per cell.** The decode pool is a process-lifetime
  `OnceLock`, so a width loop *inside* one process reports the first width for
  every subsequent cell. Every cell here is its own `taskset` + `execve`.
* **`realized=` read back from the binary as a hard precondition.** #1833 made
  the bench print `decode_width requested=N realized=M path=P as_requested`. A
  cell whose `realized` differs from its request, or which is not
  `as_requested`, aborts the run rather than being reported. This supersedes
  thread-name census: Linux `comm` truncates at 15 characters and
  `onnx-genai-spmd` is exactly 15, so `grep -c onnx-genai-spmd` both
  over-matches and hides a second pool, and it returns `w-1` on this harness
  while returning `w` on others. A check whose answer depends on the harness
  cannot be a categorical gate; the runtime's own answer can.
* **Per-rep CPU-efficiency guard** from `os.wait4` rusage: `(utime+stime)/wall`,
  discarding reps below `0.95 x median` for that arm. The threshold must be
  median-relative — at width `w` a healthy run spends about `w` CPU-seconds per
  wall-second, so a fixed floor does not transfer across widths.
* **Interleaved arms**, round-robin, so drift hits every arm equally.
* **An A/A null**: width 2 is run as two independent arms (`2` and `2b`). Any
  difference between them is pure measurement noise and bounds what the
  matrix can resolve.

Workload: `int4_decode_loop_ab`, `PROBE_ACCURACY=4 PROBE_BLOCK=32
PROBE_SESSIONS=1`, llama shapes, 384-token steady window (1536 for the `w=8`
and `w=16` work), pinned `taskset -c 0,2,...,30`. Checksums are asserted
identical across every cell of an arm, so no cell is comparing different
arithmetic.

**Pin correctness.** The even-CPU pin is 16 *distinct physical cores* on this
box: `thread_siblings_list` is `0-1`, `2-3`, ... so the odd CPUs are precisely
the hardware siblings of the pinned ones. This was verified rather than
assumed, because on hosts where siblings are `(0,16),(1,17),...` the identical
pin would silently be 8 cores and every number below would be wrong.

## Result 1 — `t=2` is 1.96x `t=1`, and the A/A null is 0.00%

| width | path | ms/token | tok/s | vs `t=1` | reps | spread |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `flat` | 14.300 | 69.9 | 1.00x | 5 | 2.25% |
| 2 | `spmd-pool` | **7.278** | 137.4 | **1.96x** | 5 | 1.73% |
| 4 | `spmd-pool` | 3.784 | 264.3 | 3.78x | 5 | 2.60% |
| 2b (A/A null) | `spmd-pool` | 7.278 | 137.4 | — | 5 | 1.29% |

**A/A null: 0.00%.** The two width-2 arms returned the same median to four
significant figures, so the 1.96x is not close to the noise floor.

**The 1.96x is not itself remarkable, and should not be sold as if it were.**
The decode budget confines the process to `w` CPUs (the run banner says so:
`CPU decode budget 2 confined the process to 2 CPUs [0, 2]`), so `t=2` has
twice the hardware of `t=1` and roughly 2x is the *expected* outcome. §27
records 1.96x for acc0/qwen and this is acc4/llama, which is worth noting, but
both are a doubling of lanes measured against the same serial baseline through
the same harness — they are corroborating, not independent. The finding here is
**negative**: the previously recorded curve reported *no* speedup where the
ordinary one was.

Note the irony worth recording: four-significant-figure agreement is what
originally exposed the bug — `23.529` vs `23.527` between two configurations
that should have differed. Here the same agreement appears between two arms
that *should* be identical, which is what it is supposed to mean.

## Result 2 — `t=1` does not build a pool at all

At width 1 the decode budget confines the process to a **single CPU**, and
`build_from_env` then declines to construct the SPMD pool: with one CPU there
is no core to run the inline dispatcher alongside a spinning worker, so the
pool would starve itself. The fallback is explicit in `decode_spmd.rs` (the
`allowed.len() == 1` branch, which calls `report_spmd_fallback` and returns
`None`), and the crate's own test records the consequence — *"the smallest
budget that builds a pool is 2"*. Decode runs on the flat path, and the bench
reports `path=flat` at width 1 and `path=spmd-pool` from width 2 up.

Two corrections to how this has been described previously, including by me:

* It is **not** the `total_workers <= 1` serial short-circuit in
  `dispatch_output_rows` that fires here. That short-circuit is real, but at
  width 1 there is no pool and no worker for it to act on.
* There is therefore **no "spawned worker sitting at 0% busy"** under this
  harness — nothing is spawned. Any account of width 1 that describes an idle
  pinned worker does not apply to a budget-confined run.

The conclusion is unchanged and is the part that matters: **`t=1` is a
different code path from every other column**, so a speedup quoted against it
means "vs the serial flat path", not "vs a one-worker pool".

## Result 3 — `w=16` is bimodal here, and no narrow interval on it is supportable

The original `8→16 = 1.77x ±0.7% over three interleaved repetitions` is
withdrawn. Six independent launches, alternating widths, sampling per-CPU busy
time from `/proc/stat` across each run:

| launch | width | ms/token | sibling (odd CPU) busy% |
|---:|---:|---:|---:|
| 0 | 16 | 1.600 | 10.6 |
| 1 | 8 | 3.225 | 8.8 |
| 2 | 16 | 1.476 | 16.1 |
| 3 | 8 | 3.195 | 8.9 |
| 4 | 16 | 1.511 | 12.1 |
| 5 | 8 | 3.220 | 9.0 |
| 6 | 16 | **9.064** | 21.3 |
| 7 | 8 | 3.391 | 16.3 |
| 8 | 16 | **4.820** | 15.7 |
| 9 | 8 | 3.389 | 15.6 |
| 10 | 16 | **8.995** | 22.1 |
| 11 | 8 | 3.509 | 15.7 |

| width | n | min | median | max | spread |
|---:|---:|---:|---:|---:|---:|
| 8 | 6 | 3.195 | 3.307 | 3.509 | **9.8%** |
| 16 | 6 | 1.476 | 3.210 | 9.064 | **514%** |

**What is established:** `w=8` is stable to 9.8% across independent launches
and `w=16` is not, spanning a factor of six. That alone retires the `±0.7%`.

**What is *not* established: the cause.** The obvious reading is SMT
contention — `w=16` holds all 16 physical cores, so a co-tenant lands on a
hardware sibling — and the correlation looks strong at Pearson **r = 0.91**.
That figure should not be trusted: it rests entirely on two leverage points
(the ~9 ms launches), and the rank correlation, which does not, is Spearman
**0.54** on n = 6 — nowhere near significant.

The data contains a direct counterexample. Launch 8 is **slow (4.820 ms) at
15.7%** sibling occupancy while launch 2 is the **fastest of all (1.476 ms) at
16.1%** — slightly *more* sibling load, 3.3x faster. The two modes' sibling
ranges overlap (fast {10.6, 12.1, 16.1}, slow {15.7, 21.3, 22.1}), so no
threshold on sibling occupancy separates them.

And odd-CPU occupancy is a proxy for **whole-socket load**, not specifically
for SMT. On a single-NUMA 16-core socket a co-tenant contends for shared L3 and
memory bandwidth wherever it is scheduled. This measurement cannot separate
"sibling steals issue slots" from "neighbour steals bandwidth", because it only
has an aggregate odd-CPU number. Distinguishing them needs per-core-pair
attribution, which is not done here.

So: bimodality **confirmed**, external-load involvement **likely**, SMT as the
specific mechanism **unproven**.

**On the withdrawn interval specifically.** The 2026-08-21 `t=16` reps
(3.297 / 3.328 / 3.344) are not directly comparable to today's: the absolute
level moved between builds, with `t=8` going from 5.87 to 3.31 over the same
period. So this is *not* a claim that those three repetitions were secretly
bimodal and got lucky — that cannot be determined now, and the arithmetic does
not support it either, since the old `t=16` sits near today's `w=8`. The claim
is narrower and sufficient: **a ±0.7% interval on `w=16` is not something this
host can support**, so the figure should not stand.

### Three things ruled out, each checked rather than argued

1. **Not a configuration failure.** Every one of these launches reported
   `realized=16 path=spmd-pool as_requested`. The slow runs are genuinely
   16-wide.
2. **Not the efficiency guard failing to fire — the guard cannot see this.**
   Rusage efficiency `(utime+stime)/wall` across ten `w=16` launches, split by
   mode:

   | mode | ms/token | CPU-s per wall-s |
   |---|---:|---:|
   | fast | 1.884, 1.888, 1.901, 1.897, 1.897, 1.893 | 13.21–14.74 (mean 14.4) |
   | slow | 3.392, 3.374, 3.408, 3.375 | 13.82–14.45 (mean 14.1) |

   The slow mode bills **the same CPU-seconds** as the fast one while taking
   1.8x the wall time. That is the signature of reduced instructions-per-cycle,
   not of missing or parked workers: the affected thread is never descheduled,
   so it stays runnable and burns its full CPU-second. A CPU-time guard is
   therefore blind to this class of interference **by construction**, as are
   voluntary-context-switch counts. `/usr/bin/time`'s `Percent of CPU` is blind
   twice over, being wall-derived as well.
3. **Not a bad pin.** `thread_siblings_list` reads `0-1`, `2-3`, … on this box,
   so the even-CPU pin really is 16 distinct physical cores. Verified rather
   than assumed: on a host where siblings are `(0,16),(1,17),…` the identical
   pin would silently be 8 cores and every number here would be wrong.

**Not ruled out:** which specific external resource is contended. See above.

## Practice this changes

* **A CPU-efficiency guard does not detect throughput interference.** It catches
  missing workers and descheduled threads; it cannot catch a co-tenant halving
  your IPC, because the victim still bills full CPU-time. Measured directly
  above: identical CPU-seconds across a 1.8x wall-time gap.
* **Prefer `w=8` for A/B work on this host.** It is 9.8% stable against `w=16`'s
  514%, and an A/B that cannot distinguish its arms from the host is not an A/B.
  The relevant boundary is **physical cores, not vCPUs** — an earlier record
  drew it at "all-vCPU" (t=32) and so treated t=16 as safe, which it is not.
* **Report distributions over independent launches, not repetitions within one.**
  The bimodality here is invisible within a single launch: reps *inside* one
  process agree closely and are jointly wrong, because the mode is fixed for
  that launch's lifetime. This is the case that shows why the practice matters.
* **Read `realized=`/`path=` from the binary.** It is the only non-vacuity check
  in this codebase that is not harness-dependent — and note it would *not* have
  caught the problem in Result 3, which is why it is necessary and not
  sufficient.
* **When an interval is withdrawn, say whether it was wrong or merely
  unverifiable.** These are different claims and only the weaker one is
  supported here.
