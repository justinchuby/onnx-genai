# The width-16 idle is a barrier wait, not a wake problem: 22 of 41 points are one slow worker

**Date:** 2026-08-23
**Harness:** `crates/onnx-runtime-ep-cpu/benches/acc0_w16_worker_split.py`
**Instrument:** `SpmdWorkerProfile` deltas emitted by
`benches/int4_decode_loop_ab.rs`
**Answers the follow-up in:** [2026-08-23-acc0-width-16-cpu-attribution.md](2026-08-23-acc0-width-16-cpu-attribution.md)

## Summary

The CPU-attribution study found that at width 16 native leaves ~40% of the
sixteen cores unused once the spin-wait mask is removed (`busy` 0.938 → 0.595).
That is *what*, not *why*, and three mechanisms produce identical aggregate
idleness while wanting opposite fixes: wake latency, load imbalance, and
dispatcher-serial time.

Per-worker attribution, 10 launches, **10 trusted**, both pre-registered
conditions firing at 100% sign consistency:

- **Wake latency is not the problem.** `wake_frac` is 0.006 at width 8 and
  **0.051** at width 16. The pre-registered WAKE-BOUND condition did **not**
  fire. A perfect wait/wake path would recover at most 5 points.
- **The pool is DISPATCH-BOUND and IMBALANCE-BOUND.** `resid_frac` 0.107 →
  **0.458**; `work_skew` 0.035 → **0.452**.
- **Most of the recoverable loss is one slow worker.** The average worker
  spends **22.2%** of the window parked at the barrier waiting for a straggler
  that is doing ~45% more work than the mean.

The mean worker's window at width 16, decomposed:

| | width 8 | width 16 |
|---|---:|---:|
| useful work | 0.886 | **0.492** |
| straggler wait | 0.031 | **0.222** |
| wake latency | 0.006 | 0.051 |
| dispatcher / serial | 0.077 | 0.235 |

## Not everything idle is a defect: the Amdahl calibration

The dispatcher/serial share nearly triples, which looks alarming and mostly
is not. Holding width 8's serial time constant and halving its parallel time —
Amdahl with no defect anywhere — predicts mean useful work of 0.796 at width
16, i.e. a serial share of **0.204**. Observed is **0.235**.

So of the ~50 points of non-work at width 16:

| component | points | recoverable? |
|---|---:|---|
| straggler wait | 22.2 | **yes** — partitioning or stealing |
| dispatcher/serial in excess of Amdahl | 3.1 | maybe |
| wake latency | 5.1 | partly — this is the wait-path budget |
| Amdahl-predicted serial | 20.4 | **no** — not a defect |

**Quoting the 46% residual as though it were all recoverable would be wrong by
more than a factor of two.** The honest recoverable figure at this width is
~25 points, and 22 of those are the barrier wait.

Note also that this **rules out a pure Amdahl explanation** for the width-16
knee, which was the other standing candidate: constant-serial scaling predicts
0.796 useful work and the pool delivers 0.492.

## The straggler

Per-worker dump, width 16, 256 tokens, blocktime 0 (`work_ms` and last-arrival
counts over the steady window):

```
idx=7  cpu=14  work_ms=879.7  wake_ms= 3.6  last_arrivals=928  parks= 50
idx=4  cpu=8   work_ms=687.2  wake_ms=19.3  last_arrivals=106  parks=547
idx=3  cpu=6   work_ms=680.3  wake_ms=23.0  last_arrivals= 96  parks=578
...
idx=12 cpu=24  work_ms=551.9  wake_ms=19.4  last_arrivals=  0  parks=927
```

One worker holds **72.5%** of the last-arrivals against a chance share of
1/15 = 6.7%, does **1.5x** the median work, and almost never parks (50 parks
against 550–930 for everyone else) because it is always the one still running.

`last_arrivals` is the sharp instrument here: summed over a node it equals that
node's dispatch count exactly, so the chance share is exactly `1/w` and any
concentration is real. It detects the condition long before `work_ns` does —
at width 8 the straggler holds 66% of arrivals while its shard is only 3.5%
above the mean.

### What it is not

Two mechanisms were tested and both are **negative**:

- **Not a fixed partitioning bug.** `worker_row_segments` divides evenly for
  every shape in this chain (4096 and 6144 and 14336 rows over 2 nodes x 8
  shards all divide exactly), and the straggler's *identity moves between
  launches* — worker 10 in only 4 of 10 launches. A static mis-partition would
  pin the same index every time.
- **Not the dispatcher colliding with a worker.** The dispatcher thread is
  unpinned while the 15 spawned workers hold one even CPU each, leaving CPU 30
  free — so a collision was the obvious candidate. Sampling the dispatcher's
  CPU during four profiled launches against the straggler's CPU in the same
  launch: `30 → 22`, `6 → 24`, `6 → 20`, `{30,20,18} → 18`. One partial match
  in four. The dispatcher migrates and does not select the straggler.

The mechanism is therefore **not yet identified**, and this document does not
claim one. What is established is its size, its signature, and that it is
dynamic rather than structural.

## Why no worker helps the straggler: stealing is off by default

`work_stealing_segments_aligned` computes
`target = total_workers * steal_tiles_per_worker()` and then:

```rust
if target <= self.total_workers {
    return self.worker_row_segments_aligned(n, align);
}
```

with `DEFAULT_STEAL_TILES_PER_WORKER = 1`. So at the shipped default
`target == total_workers` on every shape and the dynamic path **always** falls
back to static equal segments. There are no spare tiles, so a straggler can
never be helped, which is exactly consistent with the 22.2 points measured
above.

The in-tree comment records that finer 2x/3x tiling "split Qwen3 projection
shards too narrowly and regressed measured throughput" — a finding made before
the straggler cost was known, and at widths where that cost is 3.1 points
rather than 22.2.

### Turning stealing on: a large effect the instrument cannot certify

Pre-registered A/B on `ONNX_GENAI_CPU_DECODE_STEAL_TILES_PER_WORKER`, 1 (the
shipped default) versus 2, blocktime held at the shipped 500 µs in **both**
arms, arm order rotated, width 8 carried as a regression guard, A/A null in the
same invocation. 10 launches, **8 trusted**.

| w | tps ratio (2 ÷ 1) | sign | A/A ratio | sys_frac 1 → 2 | cpu_s/tok 1 → 2 |
|---:|---:|---:|---:|---:|---:|
| 16 | **1.2327** | 88% | 1.1301 | 0.280 → 0.192 | 0.06730 → 0.05944 |
| 8 | 1.0463 | — | 1.0211 | 0.038 → 0.030 | 0.04068 → 0.03967 |

**VERDICT: REJECT, and the verdict is honoured.** The ratio clears the 1.10
threshold and the 80% sign floor, but the rule also requires the effect to be
at least 3x the A/A half-width, and the A/A half-width in this very run is
**0.2154**. The requirement is therefore +0.6462 and the effect is +0.2327.
**No point estimate is claimed and the change is not proposed.**

Two things are nonetheless worth recording, because they are not what a null
looks like:

- The **mechanism moves in the predicted direction**: `sys_frac` falls
  0.280 → 0.192 at 88% sign consistency and CPU per token falls 11.7%. Less
  time spinning at the barrier is exactly what removing a straggler wait should
  produce, and it is the same signature, at a different knob, as the one this
  study predicted.
- **The width-8 regression guard does not fire** (ratio 1.0463, down-sign 38%),
  which is a genuine surprise given the in-tree comment. One early launch read
  0.56 at width 8 and it did not survive the other seven — a reminder of why
  single launches are not evidence here.

**The binding constraint is now the instrument, not the kernel.** The width-16
A/A null — two *identical* arms disagreeing by ±21.5% in this run, and ±35% in
the earlier width-16 study — is large enough to make a +23% candidate
unprovable. Until that is understood, no width-16 improvement of realistic size
can clear a pre-registered bar. That promotes the A/A instability from a
known oddity to **the next thing to fix**, ahead of any further kernel work at
this width.

The temptation here is to re-run until the null happens to come in narrow. That
is choosing the sample that licenses the conclusion, and it is not done: the
rule needs a tighter instrument, not more attempts at the same one.

> **SUPERSEDED, 2026-08-24 — there is no +23% effect.** The instrument was
> tightened as this section demanded, and the candidate was re-run 24 launches
> deep: **ratio 0.9889 against an A/A half-width of 0.0323**, an instrument 4.6x
> sharper than this one, able to resolve anything above +9.7%. The +23% here is
> **accounted for by a mode imbalance**: the control arm drew the slow mode in
> 4 of 8 launches and the test arm in 1 of 8, and at the 1.687x mode ratio that
> alone manufactures up to **+0.2576** of ratio — more than the +0.2327 observed.
> The `sys_frac` fall reported above as the mechanism moving in the predicted
> direction inverts too (+0.0111 at 46% sign at n=24): the slow mode *is* the
> high-`sys` mode, so the mechanism agreed with the effect because both were the
> same artifact. **The paragraph above is right that the instrument was the
> binding constraint, and wrong to read the point estimate as a candidate.**
> The 22.2-point straggler wait measured in this document stands; what is
> disposed of is the claim that spare tiles collect it. Full record:
> [`2026-08-24-acc0-steal-tiles-retest.md`](2026-08-24-acc0-steal-tiles-retest.md).


## Method

`ONNX_GENAI_CPU_DECODE_WORKER_PROFILE=1` populates `wake_ns`/`work_ns` at a
cost of two clock reads per worker per op (~4% at width 12), so **both widths
are profiled and no throughput number from this harness may be compared with an
unprofiled arm**; the harness refuses to print a tps ratio for that reason.

Measured at `blocktime = 0`. At the shipped 500 µs the workers spin through
what would otherwise be barrier wait, and the residual and wake fractions are
not interpretable — this is the same masking that made the first CPU-split run
score BURN-DOMINATED.

The per-worker identity `wall == work_ns + wake_ns + residual` holds by
construction over the same window that produces `wall`, so a residual outside
`[-0.02, 1.02]` is an instrument fault and discards the launch rather than
being clamped. Zero launches were discarded on that ground here.

The acceptance rule — thresholds, sign-consistency floor, and the
`REPORT NOTHING` / `UNATTRIBUTED` outcomes — is in the harness docstring, where
it was written before the first run. `UNATTRIBUTED` is a permitted outcome and
would have been reported as such.

## Reproducing

```bash
scripts/hostlock.sh run --owner <you> --reason "worker split" --wait --gate 6 -- \
  python3 crates/onnx-runtime-ep-cpu/benches/acc0_w16_worker_split.py \
    --binary target/release/deps/int4_decode_loop_ab-<hash> \
    --blocktime 0 --launches 10 --tokens 256 --reps 2 --out split.json

python3 crates/onnx-runtime-ep-cpu/benches/acc0_w16_worker_split.py \
    --binary x --replay split.json     # re-score, run nothing
```

## Next

1. **Fix or characterise the width-16 A/A null.** At ±21.5% it gates every
   width-16 result, including a +23% candidate this study could not certify.
   It is symmetric across arms and implementations, so it points at the host or
   the pinning rather than at either kernel.
2. **Find the straggler's mechanism.** It is dynamic, it moves between
   launches, and it is worth ~22 points of the machine at width 16. Candidates
   not yet excluded: cross-node weight placement for the second node's workers,
   a per-worker first-touch/page-fault asymmetry, and interference from the
   parked prefill/task pools that share the same 16 CPUs.
3. **Make stealing reachable without regressing width 8** — the straggler
   cannot be helped while `target == total_workers` on every shape.
