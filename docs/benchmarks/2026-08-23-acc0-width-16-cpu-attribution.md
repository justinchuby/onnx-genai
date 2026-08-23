# Where native loses at width 16: CPU-seconds attribution, and a spin-wait that hides a third of the pool

**Date:** 2026-08-23
**Harnesses:** `crates/onnx-runtime-ep-cpu/benches/acc0_w8_w16_cpu_split.py`,
`crates/onnx-runtime-ep-cpu/benches/acc0_w16_blocktime_ab.py`
**Instrument:** `benches/common/mod.rs::process_cpu_time`,
`benches/int4_decode_loop_ab.rs`, `benches/ort_matmulnbits_baseline.py`
**Answers the open question in:** [2026-08-23-acc0-gap-at-width-16.md](2026-08-23-acc0-gap-at-width-16.md)

## Summary

[2026-08-23-acc0-gap-at-width-16.md](2026-08-23-acc0-gap-at-width-16.md)
established that the acc0 gap is ~1.78x at width 16 versus 1.12x at width 8,
and left the mechanism open: **when the width doubling fails to double
throughput, are the extra workers idle, or are they busy and inefficient?**
Wall-clock timing cannot answer that. This adds CPU-seconds attribution to
both arms and answers it.

Two findings, and the second one changes how the first must be read.

1. **At width 16 native burns 1.45x the CPU per token it burns at width 8,
   while ORT on the same host in the same window burns 1.07x.** The inflation
   is ours, not a property of the machine. This kills the
   "DRAM/bandwidth plateau" attribution that had been the default explanation
   for the width-16 knee.

2. **`busy` is blind to spin-wait, and the shipped wait path spins.**
   `decode_spmd`'s worker wait spins then `sched_yield`s for up to 500 us
   before parking, and a yielding thread accrues system time exactly like a
   working thread accrues user time. With the ramp switched off, at
   statistically identical throughput, the width-16 busy fraction falls from
   **0.953 to 0.692**. Roughly a third of the pool is not working, and the
   default configuration reports it as fully occupied.

Under the first (default-configuration) measurement the pre-registered rule
returns **BURN-DOMINATED**. That verdict is correct as measured and is
**not** the conclusion of this document, because the quantity it keys on is
contaminated by the spin. Re-measured with the spin unmasked, the same
unchanged rule returns **MIXED**: at width 16 native does ~30% more real CPU
work per token than at width 8 **and** leaves ~40% of the sixteen cores idle.

## The instrument

Every measured cell reports, for the same window that produces `wall`:

- `user_s`, `sys_s` from `/proc/self/stat` (native, thread-group totals, ticks
  from `onnx_runtime_hostmon::clock_tick_hz()`) and from
  `getrusage(RUSAGE_SELF)` (ORT). Identical field names on both arms so one
  parser reads both.
- `cpu_s_per_token = (user_s + sys_s) / tokens` — CPU cost of the work.
- `busy = (user_s + sys_s) / (wall_s * w)` — occupancy of the `w` cores the
  arm was confined to.
- `sys_frac = sys_s / (user_s + sys_s)`.

These are read directly rather than via `/usr/bin/time`, because
`/usr/bin/time`'s `Percent of CPU` is `(user+sys)/wall` — it is **not** an
independent check on a wall-time result, it is the same measurement divided by
itself, and it degrades exactly when wall does. `utime`/`stime` are
contention-robust: a neighbour steals our wall clock but does not add to our
CPU seconds.

### The decomposition is an identity, so every cell self-tests

```
tps(16) / tps(8)  ==  2 * R_busy / R_cpu
```

where `R_cpu = cpu_s_per_token(16) / cpu_s_per_token(8)` and
`R_busy = busy(16) / busy(8)`. This is algebra, not a model: substituting the
definitions cancels everything. Residual is therefore 0.00% by construction,
which makes it a free per-cell test of the instrument. Any cell whose identity
error exceeds 5% is discarded as an instrument fault rather than reported.

**That test earned its place before it was ever used in anger.** The first run
of the harness returned `REPORT NOTHING (n_trusted = 1 < 6)` and was honoured —
nothing was quoted from it, including the one trusted cell. Diagnosing why
found a defect in *my* instrument, not in the host: each quantity was being
reduced by its own independent median. Because `tps = tokens / wall`, sorting
by `tps` and sorting by `wall` are reversed orders, so at an even repetition
count the two medians select **different repetitions**. The published row
described no run that had actually happened, and `busy` carried the whole
repetition spread as bias — identity errors of **4.4% to 29.7%** against a
quantity that is algebraically zero.

Both producers now emit every CPU field, plus `tps_rep`, from the **single**
repetition whose throughput is the median. **No threshold in the pre-registered
rule was changed** — not the counts, not the ratios, not the sign fraction.
The instrument that feeds the rule changed, and it changed because a self-test
fired before anything was scored, not because the numbers were disappointing.
Re-smoked on the exact failing case: identity error **0.01%**.

(Related trap worth recording: Rust's `median()` here is `sorted[len/2]`, while
Python's `statistics.median` *averages* the two middle values. At even `n` the
two conventions disagree, so a Rust-side and a Python-side "median" of the same
data are not the same number.)

## Finding 1 — native inflates CPU per token at width 16; ORT does not

llama3-8B projection chain, block 32, `accuracy_level = 0`, one session,
384 tokens/rep, 3 reps/cell, both arms pinned to the same 16 physical cores.
14 launches, **13 trusted**, identity error **0.00% on every trusted cell**.

| arm | speedup 8→16 | `R_cpu` | `R_busy` | busy@8 | busy@16 | cpu_s/tok @8 | @16 | sys_frac @8 | @16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native | 1.445 | **1.449** | 1.057 | 0.900 | 0.966 | 0.04112 | 0.05932 | 0.062 | **0.212** |
| ORT | 1.860 | **1.074** | 0.999 | 1.000 | 0.999 | 0.03360 | 0.03604 | 0.000 | 0.000 |

Verdict from the pre-registered rule: **BURN-DOMINATED**
(`R_cpu = 1.449 >= 1.25`, `R_busy = 1.057 >= 0.90`, sign consistency 100%).

**The ORT control is what makes this interpretable.** ORT ran interleaved with
native, same launches, same 16 cores, same minutes. If the width-16 knee were a
memory-bandwidth ceiling on this host, ORT would hit it too. ORT's CPU per
token rises 7% and its occupancy is 1.000 at both widths. Native's rises 45%.
Whatever the ceiling is, native reaches it and ORT does not, at the same width
on the same silicon.

Rough achieved bandwidth for context (the chain moves ~136.3 MB/token):
native ~24.8 GB/s at width 8 and ~36.1 GB/s at width 16; ORT ~64 GB/s at
width 16. None of these is near a DDR5 EPYC wall.

`sys_frac` rising 0.062 → 0.212 is **named as a contributor here, not
explained** — that is what finding 2 is for. ORT's is 0.000 at both widths.

## Finding 2 — the wait ramp costs ~20% of process CPU and buys nothing here

`ONNX_GENAI_CPU_DECODE_BLOCKTIME_US` (default **500 us**) controls
`decode_spmd`'s KMP_BLOCKTIME-style spin → yield → park wait.

### Dose-response first (non-vacuity)

Before the A/B, a three-point dose-response, because an env var that never
reaches the child would produce a beautifully consistent null. Width 16,
192 tokens:

| blocktime | `user_s` | `sys_s` |
|---:|---:|---:|
| 0 | 9.28 | **0.13** |
| 500 (default) | 9.27 | **3.28** |
| 20000 | 8.99 | **3.12** |

The knob is live, the ramp is already saturated by 500 us, and — the load-
bearing part — **`user_s` is invariant to it.** The system time is not work
being moved around; it is pure `sched_yield` overhead on top of an unchanged
amount of real computation. (The value is latched in a `OnceLock` at first
`worker_wait`, so it is per-process; every cell is a fresh subprocess.)

### The A/B

Pre-registered, three arms per width (control 500 / test 0 / A-A null), arm
order rotated per launch, width 8 carried as a regression guard. 10 launches,
**7 trusted**; 3 discarded at runnable peaks of 55/64/56 caused by another
agent's build.

| w | tps @500 | tps @0 | ratio | A/A null | busy @500 | busy @0 | cpu_s/tok @500 | @0 | user_s/tok @500 | @0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 252.6 | 225.6 | **0.9960** | 0.9937 | 0.953 | **0.692** | 0.06037 | 0.04838 | 0.04844 | 0.04760 |
| 8 | 187.8 | 193.1 | 0.9906 | 1.0186 | 0.957 | 0.916 | 0.04078 | 0.03794 | 0.03831 | 0.03789 |

**Throughput: REJECT.** Ratio 0.9960 with 43% sign consistency against a 5.24%
A/A half-width. Removing the ramp does not make this workload faster, and no
amount of favourable-looking mechanism data changes that. **Regression at
width 8: none** (0.9906, below the 0.95 regression threshold).

**Mechanism: total and 100% consistent.** `sys_frac` collapses 0.208 → 0.016
and `cpu_s_per_token` falls 19.9%, while `user_s/token` moves 1.7%. The rule
makes the mechanism claim conditional on the throughput claim, so this is
recorded as **UNPROVEN as a throughput intervention** — but it is not
ambiguous as a measurement of what the ramp costs: **~20% of process CPU at
width 16, on top of an unchanged amount of real work.**

This independently corroborates, on a second workload, the ~20% figure
Sebastian reported for the `worker_wait` yield ramp.

### Scope limit, stated up front

This is a **zero-gap decode loop** — the next token is requested the instant
the previous one lands. That is precisely the workload where parking early
looks free, because there is never a gap during which a parked worker must be
woken. **Nothing here licenses changing the shipped default**, which exists to
protect latency when there *are* gaps. That question needs the gap-aware
harness (#1395) and is not answered by this document.

## What finding 2 does to finding 1

`busy` counts a spinning thread as a working thread. At the shipped default,
width-16 `busy` reads 0.953; with the spin removed, at statistically identical
throughput, it reads 0.692. **Twenty-six points of "occupancy" were
`sched_yield`.**

So the BURN-DOMINATED verdict — which keys on `R_busy >= 0.90`, i.e. "the
workers are not idle" — was reading a masked quantity. Re-running the same
harness with the same unchanged rule at `--blocktime 0`, 12 launches,
**9 trusted**, identity error **0.00% on every trusted cell**:

| arm | `R_cpu` | `R_busy` | busy@8 | busy@16 | cpu_s/tok @8 | @16 | sys_frac @8 | @16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native | **1.304** | **0.652** | 0.938 | **0.595** | 0.03789 | 0.04896 | 0.002 | 0.017 |
| ORT | 1.109 | 0.999 | 1.000 | 0.999 | 0.03337 | 0.03625 | 0.000 | 0.000 |

**VERDICT: MIXED** (`R_cpu = 1.304 >= 1.15` **and** `R_busy = 0.652 <= 0.90`),
sign consistency **100% on both** ratios across all 9 trusted cells.

Side by side, the same width-16 occupancy under the two configurations:

| configuration | busy@16 | reported as |
|---|---:|---|
| default (500 us spin/yield ramp) | 0.966 | pool nearly fully occupied → BURN |
| ramp off (`blocktime = 0`) | **0.595** | **40% of the pool is not working** |

Thirty-seven points of "occupancy" were `sched_yield`. **The correct
attribution for the width-16 wall is mixed: native does ~30% more real CPU work
per token at width 16 than at width 8, *and* it leaves ~40% of the sixteen
cores unused.** Both halves are real, both are ours, and neither alone accounts
for the knee.

### Why the CPU-based verdict survives a window the wall-clock numbers do not

This run was taken in a noisier window than the first (peaks of 20–36; three
launches discarded, and ORT's own scaling collapsed in three *trusted* cells).
The wall-derived per-launch speedup swings **0.78x to 1.37x** — a 1.75x range,
useless on its own. Over the identical cells:

| quantity | per-launch range | sign consistency |
|---|---|---:|
| wall speedup 8→16 | 0.78 – 1.37 | — |
| `R_cpu` | 1.10 – 1.34 | **100%** |
| `R_busy` | 0.52 – 0.81 | **100%** |

That is the designed-for behaviour and the reason the instrument exists: a
competing process steals our wall clock but does not add to our CPU seconds.
The median wall speedup here (1.000) is **not** comparable to the first run's
1.445 and no conclusion is drawn from it — the throughput question was settled
separately by the A/B (ratio 0.9960 against a 5.24% null), which is a
same-window paired comparison rather than a cross-run one.

## Where this leaves the width-16 gap

- The knee is **not** a host bandwidth ceiling. The ORT control settles that.
- It is **not** a single cause. Both halves are real and both are ours.
- The idle half is a **dispatch/partition** problem: at width 16 the pool
  leaves ~40% of the cores unused on these shapes (`busy` 0.938 → 0.595), and
  the shipped configuration reports that as 0.966.
- The burn half is a **kernel-efficiency** problem: ~30% more CPU seconds per
  token at the same numerics and the same shapes, which in CPU-seconds terms
  means memory-stall cycles rather than extra instructions.

ORT, measured in the same launches on the same cores, holds `busy = 0.999` at
width 16 and pays 11% more CPU per token. Both of our losses are addressable.

Two follow-ups, in priority order:

1. **Localise the idleness.** `ONNX_GENAI_CPU_DECODE_WORKER_PROFILE` and the
   per-worker straggler attribution from #1859 exist for exactly this — decide
   between load imbalance (one worker finishing late every op) and dispatch/
   wake latency (all workers late off the line).
2. **Localise the burn.** Per-op attribution at width 8 versus width 16 on the
   same shapes, against the 136.3 MB/token figure.

## Reproducing

```bash
scripts/hostlock.sh run --owner <you> --reason "acc0 cpu split" --wait --gate 6 -- \
  python3 crates/onnx-runtime-ep-cpu/benches/acc0_w8_w16_cpu_split.py \
    --binary target/release/deps/int4_decode_loop_ab-<hash> \
    --blocktime 0 --launches 12 --tokens 384 --reps 3

scripts/hostlock.sh run --owner <you> --reason "blocktime ab" --wait --gate 6 -- \
  python3 crates/onnx-runtime-ep-cpu/benches/acc0_w16_blocktime_ab.py \
    --binary target/release/deps/int4_decode_loop_ab-<hash> --launches 10
```

Both harnesses accept `--replay <json>` to re-score archived data without
running anything, and both carry their acceptance rule in the module docstring
where it was written before the first run.

### If you sweep a thread-count or wait knob, check it is not vacuous

Categorical, 30 seconds, works on a loaded host:

```bash
ONNX_GENAI_CPU_DECODE_THREADS=$w taskset -c 0,2,4,...,30 $BIN &
sleep 6; for t in /proc/$P/task/*; do cat "$t/comm"; done | grep -c onnx-genai-spmd
```

`w` in must give `w` out. Add `NXRT_CALIB_DEBUG=1` to confirm the decode path.
Note also that `ONNX_GENAI_CPU_DECODE_THREADS=1` is **not** "the pool with one
worker": `dispatch_output_rows` takes a serial short-circuit at
`total_workers <= 1` and runs everything on the dispatcher, so any speedup
quoted against width 1 is "versus serial".
