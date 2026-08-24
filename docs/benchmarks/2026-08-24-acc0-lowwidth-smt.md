# The t=2 sub-one-core CPU reading is not SMT co-location — and does not reproduce

Date: 2026-08-24 · Owner: Roy · Base: `7bf32c5f4`
Probe: `crates/onnx-runtime-ep-cpu/benches/acc0_lowwidth_smt.py`

## The claim under test

An earlier table of mine reported `Percent of CPU` 98 / 71 / 186 at decode
widths 1 / 2 / 4. The t=2 cell is the anomaly: 71% is *less than one core* while
two workers exist, so adding the second worker apparently reduced total CPU. I
attributed it provisionally to wake/park latency.

A cross-agent hypothesis (2026-08-24) is that this is placement: that the two
workers sat on cpus 0 and 1, SMT siblings of one physical core, so "t=2" was
really one core.

## Why the arithmetic already argued against it

The same message supplied the refutation. A pinned scalar probe measured cpu 1
delivering 55% of the work of an uncontended CPU while reporting a CPU *share*
of 1.000, because its sibling cpu 0 was busy. That is the defining property of
SMT: **it steals throughput without stealing CPU-time.**

Two spinning workers on one core should therefore read ~200% of CPU with halved
throughput. SMT contention cannot produce a reading *below* one core. A
sub-one-core reading means the workers were not on-CPU at all.

That is a deduction from someone else's numbers, so it was measured with the
co-location forced directly rather than inferred.

## Result 1 — REJECT, by forcing the co-location

Three placement arms per width, `ONNX_GENAI_CPU_DECODE_THREADS` always explicit,
3 launches per cell interleaved, quiet host under `hostlock.sh`. `cores` =
taskset to even CPUs (one per physical core); `siblings` = taskset to adjacent
CPUs (w=2 → cpus 0,1 = one core); `default` = no taskset.

| arm | w | cores | ms/tok | lanes | pct_cpu |
|---|---|---|---|---|---|
| cores | 1 | 1 | 36.348 | 1.00 | 100.2 |
| default | 1 | 1 | 36.244 | 0.99 | 99.0 |
| cores | 2 | **2** | 18.549 | 1.97 | 196.5 |
| **siblings** | 2 | **1** | **33.708** | **2.00** | **200.0** |
| default | 2 | 2 | 18.096 | 2.00 | 199.6 |
| cores | 4 | 4 | 9.230 | 4.00 | 399.5 |
| siblings | 4 | 2 | 17.553 | 3.85 | 384.5 |
| default | 4 | 4 | 9.195 | 4.00 | 400.3 |

Forcing both workers onto one physical core costs **1.86x throughput and
nothing at all in CPU-time** — 200.0% of CPU, indistinguishable from the
two-core arm's 196.5%. The pre-registered REJECT threshold was lanes >= 1.5.

**VERDICT: REJECT.** SMT co-location does not suppress CPU-time, so it cannot be
what a sub-one-core reading is made of.

The work-completed ratio is the cross-check: **0.550** siblings/cores. The
independent scalar probe measured 15499/28154 = **0.5505** on the same host.
Two unrelated workloads agree on the SMT throughput factor to three decimals,
which is a strong argument that both instruments are reading the real thing.

## Result 2 — the anomaly does not reproduce under its own instrument

The table above uses steady-phase CPU accounting, not the whole-process,
wall-derived `Percent of CPU` that produced 98/71/186. Comparing across
instruments proves nothing, so the retired instrument was re-pointed at current
main, unchanged:

| width | old reading | `/usr/bin/time -v` on `7bf32c5f4` |
|---|---|---|
| 1 | 98% | 99 / 99 / 99 |
| 2 | **71%** | **196 / 196 / 196** |
| 4 | 186% | 365 / 378 / 379 |

The same instrument now reads 196% where it read 71%. So the anomaly was **not**
an instrument artifact — it was a real property of a tree that no longer exists.
w=1 reproduces exactly (98 → 99), which is what makes the w>=2 rows a change in
the runtime rather than a change in the host or the measurement.

Both the original wake-latency attribution and the SMT-co-location hypothesis
are therefore moot: there is no anomaly left on this tree to attribute. Decode
CPU now scales 99 / 196 / 372 against an ideal 100 / 200 / 400.

## Structural fact found along the way

**Decode width `w` builds `w − 1` threads named `onnx-genai-spmd` and runs the
w-th lane on the dispatcher thread itself.** Width 16 shows 15 spmd threads
while measuring ~16.0 lanes of CPU; width 2 shows 1; width 1 builds no pool at
all and runs inline.

Any instrument that counts `onnx-genai-spmd` threads and calls that the width is
off by one — including a cross-agent `obs/disp` column reading 16.00 on one arm
and 15.00 on another, where the difference is the dispatcher-reserve path, not a
missing worker.

## Three sample-instant defects in one probe

All three were the same bug wearing different clothes, and each would have
produced a confident wrong answer rather than an error:

1. **Worker masks read too early.** Accepting the first `/proc` read with `w`
   entries raced the EP's own `sched_setaffinity`: a thread sampled between
   spawn and pinning still carries the inherited mask, so the two-worker arm
   reported one pinned CPU and the distinct-core count came out 1 instead of 2 —
   voiding the contrast gate on a probe whose entire job is counting cores.
   Fixed by requiring every mask to be a single CPU.
2. **Process cpuset read too early.** Sampling once at the first poll caught
   `taskset` before it applied the mask, reporting `0-31` for every arm.
   Fixed by re-reading and keeping the latest.
3. **Width 1 has no pool**, so an empty mask list is the correct observation
   there, not a failed sample. It was originally discarded as a failure.

This is the third record in a row where the instrument's *sampling instant* was
the defect (cf. the width-16 placement probe, whose fixed 4.0 s sample was later
than a fast launch's entire 1.13 s lifetime and so selected for slow launches).
The general rule: **check every sample instant against the shortest arm, not the
longest, and against the setup path, not just the steady phase.**

A fourth, non-timing defect: the first run lost five minutes of clean data to a
formatting error in the verdict block, because the JSON dump came last. Persist
before summarising.
