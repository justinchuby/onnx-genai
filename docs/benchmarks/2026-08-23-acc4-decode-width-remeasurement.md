# acc4 int4 decode — width curve re-measured, and why `w=16` is not measurable here

**Date:** 2026-08-23 · **Owner:** Roy (CPU MatMul) · **Host:** AMD EPYC 9V74,
16 physical / 32 logical, single NUMA node, AVX2+FMA+F16C, shared.

Re-measurement of the decode width curve recorded in
[`CPU_MATMUL_ASSIGNMENT.md` §20](../performance/CPU_MATMUL_ASSIGNMENT.md), which
gave `22.949/22.893/11.584/5.866/3.302 ms/token at 1/2/4/8/16 threads`, an
`8→16` ratio of `1.77x ±0.7%`, and recorded `t=1 ≡ t=2` as "unexplained ... an
open question". Two of those three are wrong. This document replaces them.

The trigger was #1740, which retracted the same `t=1 ≡ t=2` signature in three
other locations after a controlled re-measurement showed `=2` is 1.96x `=1` on
acc0. That correction missed §20, and §20 is the one place the claim was framed
as an open research question rather than an observation — the framing most
likely to stop the next person from measuring it.

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

This reproduces §27's acc0 figure of 1.96x on a *different* configuration
(acc4 rather than acc0, llama rather than qwen). Two unrelated shapes landing
on the same factor is much stronger evidence than either alone.

Note the irony worth recording: four-significant-figure agreement is what
originally exposed the bug — `23.529` vs `23.527` between two configurations
that should have differed. Here the same agreement appears between two arms
that *should* be identical, which is what it is supposed to mean.

## Result 2 — `t=1` is not "the pool with one worker"

The bench reports `path=flat` at width 1 and `path=spmd-pool` from width 2 up.
At `total_workers <= 1`, `dispatch_output_rows` takes a serial short-circuit and
runs every op on the **dispatcher**; the spawned worker receives no dispatch.

So any speedup quoted against `t=1` is **"vs the serial dispatcher"**, not "vs a
one-worker pool". This is a footnote every thread-scaling table on this codebase
needs, including the ones in §20 and §27.

## Result 3 — `w=16` is not measurable on this host while it is shared

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

`w=8` is stable. `w=16` spans a factor of six, and the slow launches are the
ones with the busiest SMT siblings. The mechanism is headroom: at `w=16` the
run holds all 16 physical cores, so a co-tenant scheduled on the hardware
siblings takes throughput directly out of the measurement; at `w=8` half the
cores are free to absorb it.

Three things this rules out, each checked rather than argued:

1. **Not a configuration failure.** Every one of these launches reported
   `realized=16 path=spmd-pool as_requested`. The slow runs are genuinely
   16-wide.
2. **Not the efficiency guard failing to fire.** In the slow mode rusage
   efficiency is ~13-14 CPU-s/wall-s — the threads *are* runnable and running.
   SMT contention is invisible to a CPU-time guard by construction, because the
   victim thread is never descheduled; it just retires fewer instructions per
   cycle. This is a real limitation of the guard and worth knowing.
3. **Not a bandwidth knee.** A plateau is monotone and reproducible. This is
   bimodal, and it correlates with an external variable.

The `±0.7%` was three repetitions that happened to land in one mode. That is
precisely the "do not quote the flattering statistic" failure the acc4 protocol
warns about, applied to my own number, and the correct response is to leave
`8→16` unquoted rather than to publish a narrower interval than the host can
support.

## Practice this changes

* **A CPU-efficiency guard does not detect SMT contention.** It catches missing
  workers and descheduled threads; it cannot catch a co-tenant halving your IPC.
  For any run that occupies all physical cores, sample sibling occupancy from
  `/proc/stat` and report it alongside the timing, or measure at half width
  where there is headroom.
* **Prefer `w=8` for A/B work on this host.** It is 9.8% stable against `w=16`'s
  514%, and an A/B that cannot distinguish its arms from the host is not an A/B.
* **Report distributions over independent launches, not repetitions within one.**
  The bimodality here is invisible within a single launch: reps 1-4 of one
  process agree closely and are jointly wrong.
* **Read `realized=`/`path=` from the binary.** It is the only non-vacuity check
  in this codebase that is not harness-dependent.
