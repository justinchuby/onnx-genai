# The width-16 idle half is straggler-bound, not dispatch-bound — a correction, and the instrument defect that produced the wrong answer

**Date:** 2026-08-26
**Harness:** `crates/onnx-runtime-ep-cpu/benches/acc0_w16_worker_split.py`
(unchanged apparatus; the reporting bug fixed here)
**Instrument:** `worker …` rows emitted by `benches/int4_decode_loop_ab.rs` under
`ONNX_GENAI_CPU_DECODE_WORKER_PROFILE=1`
**Data:** the same two datasets as the original run, replayed. No new host time.
**Corrects:** [#2071](https://github.com/justinchuby/onnx-genai/issues/2071)
and [#1801](https://github.com/justinchuby/onnx-genai/issues/1801) comments of
2026-08-26.

## Verdict first

I published **DISPATCH-BOUND + IMBALANCE-BOUND** for the shipped configuration,
and a **+0.086 "serial excess over Amdahl"** which I then named as the largest
recoverable term and the highest-value remaining target in this lane.

Both are wrong, from the same cause. The harness's pre-registered rule says
*"Compare w=16 against w=8"*. The code took the pair from `min(--widths)` and
`max(--widths)`. I ran it with `--widths 2,8,16` to get an extra descriptive
column, which silently re-based every verdict and the whole Amdahl calibration
onto **w=2**.

Re-scored against the pre-registered baseline, from the identical bytes:

| | published (w=2 baseline) | pre-registered (w=8 baseline) |
|---|---|---|
| `blocktime=0` verdict | DISPATCH-BOUND | **DISPATCH-BOUND** (unchanged) |
| `blocktime=500` (shipped) verdict | DISPATCH-BOUND + IMBALANCE-BOUND | **IMBALANCE-BOUND only** |
| serial excess over Amdahl, `bt=0` | **+0.086** | **−0.009** |
| serial excess over Amdahl, `bt=500` | **+0.082** | **−0.064** |
| recoverable-in-principle, `bt=500` | 0.146 of the window | **0.064 of the window** |

The sign of the headline term flips. At the shipped blocktime the dispatcher's
serial share is **below** what Amdahl predicts from w=8 — there is no serial
excess to attack. The only recoverable term at w=16 is the **straggler wait,
0.064 of the window**, and the surviving verdict points at the partitioner.

One conclusion is unaffected and worth restating because it was the surprising
one: **WAKE-BOUND does not fire on either baseline, at either blocktime.** The
futex-park hypothesis — mine — stays falsified at sessions=1.

## Why w=2 is the wrong baseline, mechanically

The Amdahl calibration takes the narrow width's `work_frac` as "serial /
parallel with no defect" and divides the parallel part by the width ratio. It is
therefore only as good as the narrow width, and it is *sharply* sensitive to it.

At w=2 the pool spawns **one** worker (`reserve_single_group_headroom` leaves a
CPU for the dispatcher, and the dispatcher owns a shard inline). The mean worker
is busy 0.997 of the window, because there is barely a barrier to be serial *at*
— one worker cannot wait for a straggler, and the dispatcher's own shard is
concurrent with it, not serial to it. Extrapolating from `work_frac = 0.997`
predicts a serial term of **0.021** at w=16. Nothing real could clear that bar,
so any observed serial term reads as "excess".

At w=8 the mean worker is busy 0.909 — a barrier with seven participants and a
measurable dispatch gap. That predicts **0.167** at w=16, against **0.103**
observed. Same dataset, opposite conclusion.

This is not a threshold that wants tuning. w=8 was pre-registered *before* the
first run, and it is the defensible choice for exactly the reason above.

## Full corrected decomposition, w=8 → w=16, 7 trusted launches per arm

`blocktime=0` (parking exposed):

```
VERDICT: DISPATCH-BOUND (resid_frac 0.064 -> 0.184, +0.120)

              stat         w8        w16      delta
         wake_frac      0.008      0.022     +0.014
         work_frac      0.933      0.795     -0.139
        resid_frac      0.064      0.184     +0.120
         work_skew      0.017      0.084     +0.067
  straggler_excess      2.253      1.875     -0.378
  parks_per_worker     99.571    345.867   +246.295

       useful work      0.933      0.795
    straggler wait      0.016      0.067
      wake latency      0.008      0.022
 dispatcher/serial      0.042      0.116
  serial excess over Amdahl: -0.009 | straggler wait: 0.067 | wake: 0.022
```

`blocktime=500` (the shipped default):

```
VERDICT: IMBALANCE-BOUND (straggler_excess 3.03x chance, work_skew 0.027 -> 0.078)

              stat         w8        w16      delta
         wake_frac      0.009      0.014     +0.004
         work_frac      0.909      0.819     -0.090
        resid_frac      0.082      0.173     +0.091
         work_skew      0.027      0.078     +0.051
  straggler_excess      2.450      3.031     +0.581
  parks_per_worker      4.857      0.533     -4.324

       useful work      0.909      0.819
    straggler wait      0.025      0.064
      wake latency      0.009      0.014
 dispatcher/serial      0.057      0.103
  serial excess over Amdahl: -0.064 | straggler wait: 0.064 | wake: 0.014
```

DISPATCH-BOUND misses at the shipped blocktime by **0.009** on a **0.10**
threshold (`+0.091` rise). That is a miss, not a near-hit to be argued around: a
pre-registered threshold that only binds when it agrees with you is not a
threshold. What the two arms together *do* show is that the residual is real and
that spinning masks part of it (0.184 → 0.173) — but at the shipped setting it
does not clear the bar the rule set for calling it the mechanism.

## The `parks_per_worker` row is the other durable number

At the shipped 500 µs the median worker parks **0.533 times** across a whole
launch at w=16. With the window removed it parks **345.9** times, and the wake
latency that buys costs **0.022** of the window.

So the spin window is doing exactly what its doc comment claims — it eliminates
essentially all parking during a generation. The part of the comment that is not
supported is the *stakes*: it says parking on every barrier "would tank
throughput", and the measured price of parking on essentially every barrier is
**2.2% of the window** in wake latency. Correcting that comment is
[#2071](https://github.com/justinchuby/onnx-genai/issues/2071)'s business and is
not done here; this record only supplies the number.

## The instrument defect, and the fix

`verdict()` and `report()` each computed `narrow, wide = min(widths),
max(widths)`. The pre-registered pair is a **clause of the rule**, so taking it
from the command line means the printed verdict answers a different question
from the one the rule asks — with nothing in the output to say so. The header
line, the threshold banner and the verdict text were byte-identical in shape
between the correct run and the re-based one.

Fixed by pinning `RULE_NARROW = 8` / `RULE_WIDE = 16` in code:

- the rule pair is no longer derived from `--widths`;
- extra widths are still measured and printed, under an explicit
  `descriptive only, NOT used by the rule or the decomposition` banner;
- a run that does not measure **both** members of the pair now prints
  `REPORT NOTHING (the pre-registered rule compares w=16 against w=8; this run
  measured 2,16)` instead of substituting whatever it has;
- the report prints `# verdict pair: w8 vs w16 (pre-registered; not taken from
  --widths)` on every run, so the provenance is in the output rather than in the
  reader's memory.

`--self-test` proves the pinning against a synthetic dataset built so that the
w=8 rise is **+0.09** (below threshold) and the w=2 rise is **+0.17** (above):
DISPATCH must not fire under `--widths 8,16`, `2,8,16`, `16,8` or `2,16,8,4`. It
also covers the REPORT NOTHING paths and the `n_trusted` floor. Before the fix
the `2,8,16` case fires DISPATCH, so the test is proved against the defect it
exists for.

`--binary` is no longer required for `--replay` (it was demanded and never used,
which is why re-scoring an archived dataset was more awkward than it should have
been — and awkward re-scoring is how a re-based verdict survives).

## What this changes about where the lane goes next

The target I named yesterday does not exist. Priorities on the corrected data:

1. **Straggler wait, 0.064 of the window at w=16**, with
   `straggler_excess = 3.03x` chance and a `work_skew` of 0.078 — this is
   [#2017](https://github.com/justinchuby/onnx-genai/issues/2017)'s
   lane-versus-chunk question and it is now the *only* term the rule attributes.
   Note the straggler is **not** a fixed worker: the modal holder at w=16 holds
   it in 2 of 7 launches, barely above chance for 15 workers, which argues
   against a fixed bad shard and towards per-launch placement.
2. **Wake latency is not a target.** 0.022 of the window with parking fully
   exposed.
3. **Dispatcher serial is not a target at w=8 → w=16.** It tracks Amdahl.

## Method note

Three of the four errors this campaign has caught this week share a shape: a
harness that selects an arm or a baseline by a parameter nobody re-checked
(`PIN-ON`, a string the code never emits; a `OnceLock` that latches the first
arm; and now a rule baseline taken from `argv`). In every case the output looked
exactly like a valid measurement. The general defence is the one applied here:
**anything the pre-registered rule names must be pinned in code and printed in
the output**, and the harness must have a self-test that fails against the
defect rather than merely exercising the happy path.
