# The dynamic decode claim was unreachable in production, and what it is worth

**Date:** 2026-08-24
**Owner:** Roy (CPU MatMul / MatMulNBits)
**Host:** 32 logical CPUs (16 physical cores, SMT), single NUMA node, two 32 MiB
L3 domains. Quiet, held under `scripts/hostlock.sh --gate 6`.
**Binary:** `int4_decode_loop_ab`, release, default features (**no `mlas`**).
**Probe:** `crates/onnx-runtime-ep-cpu/benches/acc0_w16_steal_ab.py`
**Raw data:** `bb/steal_gran.json` (8/arm), `bb/steal_gran60.json` (60/arm)

## Summary

Two separate results, and they should not be conflated.

1. **A defect, fixed here.** `ONNX_GENAI_CPU_DECODE_SCHEDULE=steal` was
   documented as selecting a dynamic tile claim, and was **inert in every
   production build**. The parse arms recognising `steal` were gated on
   `#[cfg(feature = "mlas")]`, while the scheduling path they select is
   pure-native. `mlas` is off by default and is documented as *"never linked
   into, activated in, or routed to by a default build"*, so the only builds
   that could reach the switch were the ones nobody ships.

2. **A measurement, which does *not* license a default change.** With the
   selector working, the dynamic claim removes **~20% of decode latency inside
   the slow mode** of the width-16 bimodality (replicated at n=24 and n=60,
   A/A null 0.43%). It does **not** produce a resolvable end-to-end gain,
   because the mode fraction is noise-dominated on this host and the A/A null
   on the mode-weighted expectation is **6.55%**. The default is unchanged.

## 1. The selector, and why no test caught it

`decode_schedule_from_raw` mapped `"steal"` / `"work-stealing"` to
`DecodeSchedule::Steal` only under `#[cfg(feature = "mlas")]`, falling through
to `Fixed` otherwise. But the thing that arm selects has **two** implementations:

* with `mlas`, the *executor* is swapped for `mlas_sys::WorkStealingThreadPool`;
* without it, `dispatch_output_rows` and `dispatch_rows_work_stealing` run an
  `AtomicUsize::fetch_add` cursor over the tile table **on the ordinary native
  SPMD pool**.

The second path is complete, safe, and carries no MLAS dependency. It was simply
unreachable: the *policy* was gated on a feature only its optional *executor*
needs.

It survived because the existing `work_stealing_*` tests construct
`DecodeSchedule::Steal` **directly**, bypassing the parser. The implementation
was covered; its reachability never was. The test added here
(`work_stealing_is_reachable_from_the_env_string_without_mlas`) starts from the
env string a user actually sets and asserts parse → build → dispatch, with every
output claimed exactly once. Against the unfixed code both it and
`decode_schedule_parses_env_values` fail with `left: Fixed, right: Steal`; that
control was run before the fix was kept.

This is the third member of a family on this project: #1792
(`ONNX_GENAI_CPU_DECODE_AFFINITY` entirely inert), the latched-`OnceLock` A/B,
and now this. **A knob is not verified until an observable changes when you turn
it.** Here the observable is the width line, which now reports honestly:

```
ONNX_GENAI_CPU_DECODE_SCHEDULE=fixed  ->  path=spmd-pool
ONNX_GENAI_CPU_DECODE_SCHEDULE=steal  ->  path=work-stealing-pool
```

The first smoke run of the A/B was three arms all printing `spmd-pool` — an
experiment comparing a configuration with itself, three times. It produced no
data and its `CONTROL 1` is what caught this.

## 2. Granularity: predicted a priori not to help at the default

`work_stealing_segments_aligned` computes `target = total_workers *
STEAL_TILES_PER_WORKER`, and with the default `STEAL_TILES_PER_WORKER = 1` it
returns `worker_row_segments_aligned` — **the same segments the fixed split
uses**. With one tile per worker a dynamic claim can absorb a lane that *wakes
late* (an awake worker takes the absent one's tile) but cannot absorb a lane that
*executes slowly*, because that lane still holds one whole tile to the end.

My straggler measurements say the victim usually computes longer
(`straggler_idx == slowest_idx` at 0.667/0.667/0.684 across three experiments,
chance 0.067), so `steal4`
(`ONNX_GENAI_CPU_DECODE_STEAL_TILES_PER_WORKER=4`) carries the hypothesis and
`steal1` is the granularity control. This was written into the probe before the
first launch.

## 3. Result, n=60 per arm, four interleaved arms with an A/A null

Every arm is bimodal, with modes ~2.25x apart and an empty band between them. A
median over a bimodal mixture estimates **how many launches landed in each
mode**, not the effect, so the pooled verdict is void and the report stratifies
per arm. (Rule amendment 2, recorded in the probe, applied to the data that
motivated it.)

| arm | n_fast | med_fast | n_slow | med_slow | frac_fast |
|---|---|---|---|---|---|
| fixed  | 32 | 1.6045 | 28 | 3.6090 | 0.533 |
| steal1 | 26 | 1.6735 | 34 | 2.9075 | 0.433 |
| steal4 | 30 | 1.6100 | 30 | 2.8725 | 0.500 |
| null   | 28 | 1.6570 | 32 | 3.6245 | 0.467 |

**Slow mode** — A/A null **0.0043**:

| arm | vs fixed |
|---|---|
| steal1 | **+19.44%** |
| steal4 | **+20.41%** |

Replicated: an independent n=24 run gave +19.72% / +21.46% with an A/A null of
0.0223. Both granularities win, so the effect is not granularity-specific — the
`steal1`-vs-`steal4` discrimination I predicted did **not** appear, and the
"absorbs a slow executor" reading is therefore *not* supported over "absorbs a
late waker".

**Fast mode** — A/A null **0.0327**, above the 0.025 bar: unresolvable, no claim.
(At n=24 `steal1` appeared 11% *worse* in fast mode; at n=60 that is not
judgeable. It was noise.)

**Mode fraction** — the quantity the verdict turns on:

| comparison | delta | z | |
|---|---|---|---|
| A/A floor \|fixed − null\| | 0.067 | — | the harness's own noise |
| steal1 vs fixed | 0.100 | 1.10 | UNRESOLVED |
| steal4 vs fixed | 0.033 | 0.37 | UNRESOLVED |

At n=24 `steal4`'s fraction looked shifted by 0.243 (z=1.63) and, had I stopped
there and weighted by it, the arm would have read as *neutral overall*. At n=60
the shift is 0.033 and the A/A floor is **larger than it**. The n=24 fraction
gap was noise.

**Mode-weighted expectation:** fixed 2.5399, steal1 2.3728, steal4 2.2412, null
2.7063 — but the **A/A null on the expectation is 0.0655**, so no end-to-end
statement is licensed. The bimodality dominates the expectation and this harness
cannot resolve a 5% effect through it at n=60.

## 4. Disposition

**No default change.** The pre-registered rule requires a resolvable end-to-end
gain and there isn't one. `decode_schedule_from_raw` still falls through to
`Fixed`.

**The selector fix ships on contract grounds, not benchmark grounds.** A
documented switch that silently does nothing is wrong independently of whether
turning it on is faster, exactly as #1729 shipped on the grounds that
`realized=16` while using 8 cores is wrong rather than slow.

**What this buys the straggler investigation.** The slow mode sits ~2.25x above
the fast mode; the dynamic claim removes ~20% of it, i.e. roughly **37% of the
slow mode's excess over the fast mode**. So a substantial minority of the
width-16 slow mode is *work-distribution* cost that scheduling can recover, and
the majority is not. That is the first mechanism to take a measurable bite out
of the straggler after five rejected hypotheses (assignment, lane index, CPU
placement, virtual layout, physical page backing) — and it bites without
identifying the selector, which remains unknown.

**Caveat carried forward.** Both granularities winning equally in the slow mode
undercuts the late-waker/slow-executor discrimination this probe was built to
make. A `steal1` win is consistent with absorbing a *late-waking* lane, which my
`straggler_idx == slowest_idx` evidence argues against. That tension is recorded,
not resolved.
