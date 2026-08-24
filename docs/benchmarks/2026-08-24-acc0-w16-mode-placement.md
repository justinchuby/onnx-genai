# acc0 width-16 bimodality is not a worker-placement difference

Date: 2026-08-24 · Owner: Roy · Base: `eee5da8c6`
Probe: `crates/onnx-runtime-ep-cpu/benches/acc0_w16_mode_placement.py`

## Question

The width-16 int4 decode loop is bimodal per process launch (~4.0 vs ~6.0
ms/token, 1.69x, `migrations=0` throughout). A standing hypothesis was that a
launch's mode is decided by *where* its pool lands — that the fast mode is a
correctly-placed pool and the slow mode is a confined one.

Prior placement evidence could not answer this: `decode_placement_census.sh`
kills each launch before it times, so it reads placement without a mode, and
the A/B harness reads a mode without placement. This probe reads **both from
the same process**.

## Pre-registered rule

Written before the first launch, in the probe docstring:

- **REJECT** (mode is not placement) if both modes appear on a byte-identical
  pinned-CPU set.
- **REPORT NOTHING** if fewer than 10 trusted launches, or if only one mode was
  sampled — a run that never drew the slow mode cannot speak about it.

## The probe selected for the mode it was classifying

The first version slept a fixed 4.0 s and then read `/proc/<tid>/Cpus_allowed_list`
for every `onnx-genai-spmd` thread. It discarded 11 of 14 launches, and every
launch it *did* sample was slow.

That looked like a workload failure. It was not. At `--tokens 192 --reps 2` a
fast launch completes in `wall_s = 1.13` and has torn its pool down long before
t = 4 s; a slow launch is still running. **The sampling instant was later than a
fast launch's entire lifetime, so the probe could only ever observe slow
launches** — and would have "found" that every sampled launch shared one
placement while silently never seeing the other mode.

This is the sharpest instance yet of a defect class already in this ledger: a
harness that reports "no data" identically for a workload failure and for an
instrument that structurally cannot see the case of interest. The fix is to
poll from t=0 at 20 ms and take the first read holding the full pool, which
removes the duration-dependence entirely.

## Result — REJECT

16 trusted launches, quiet host under `hostlock.sh`:

| modes sampled | distinct placements | verdict |
|---|---|---|
| fast 15, slow 1 | **1** | REJECT |

Every launch, both modes: **15 workers on `0,2,4,…,28`** — one per physical
core, 8/7 split across the two 32 MiB L3 instances, dispatcher reserved on
cpu 30.

```
launch 15  fast  ms=  4.041  lanes=15.47  workers=15  cores=15  8/7 L3  cpus=0,2,...,28
launch 16  SLOW  ms=  6.022  lanes=12.19  workers=15  cores=15  8/7 L3  cpus=0,2,...,28
```

Pooling the two earlier (biased, slow-only) runs, **21 launches across three
runs and both modes have now produced exactly one placement string.** The bias
in those runs makes their slow-mode rows *more* informative here, not less:
they are a concentrated sample of precisely the mode the hypothesis was about.

A 4.0 ms launch and a 6.0 ms launch are byte-identical in worker count,
physical-core count, L3 distribution and CPU set. Placement is excluded.

## What is now excluded, and what is left

| candidate | status |
|---|---|
| THP / page backing | REJECT (2026-08-24 null record) |
| foreign load on the pinned set | REJECT — ≤1.7% of pinned busy vs 23% needed |
| spare-tile static steal | REJECT — 24 launches, ratio 0.988 |
| **worker placement** | **REJECT — this record** |
| weight-arena placement across L3/CCX | open |
| per-launch clock/boost state | open |

The mode difference remains a **waiting** difference, not a work difference:
user CPU/token spans 11% across both modes while wall spans 1.69x, and the slow
mode is +170% sys. The remaining 22.2-point straggler wait is not explained by
where the workers are.

Because a slow worker is slow for reasons of its own — not because it holds
more work — a *static* redistribution cannot collect this. The evidence points
at a dynamic, measurement-driven steal.
