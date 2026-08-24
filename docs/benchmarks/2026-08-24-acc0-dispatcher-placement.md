# The decode pool reserves a CPU for its dispatcher and never binds it — an opt-in pin, REJECTED by its own bar

**Date:** 2026-08-24
**Harness:** `crates/onnx-runtime-ep-cpu/benches/acc0_w16_blocktime_ab.py` (reused
unchanged), `crates/onnx-runtime-ep-cpu/benches/acc0_w16_dispersion.py` (new)
**Instrument:** `dispatcher …` row emitted by `benches/int4_decode_loop_ab.rs`
**Unblocks:** nothing yet — see the verdict.
**Follows:** [2026-08-23-acc0-width-16-worker-attribution.md](2026-08-23-acc0-width-16-worker-attribution.md)

## Verdict first

The decode pool leaves one CPU empty for its dispatcher and then never puts the
dispatcher on it. Binding it there (`ONNX_GENAI_CPU_DECODE_DISPATCHER_PIN=1`) is
faster in **15 of 15** trusted launches on current main — and the median gain is
**9.5%**, under the **10%** its pre-registered rule requires. **REJECT.** A
second, separately pre-registered rule about measurement dispersion **failed its
own self-test** and returned REPORT NOTHING. An earlier 6-launch run of the same
knob returned ACCEPT at 19.1%; it **does not replicate** at n=15 on the current
tree, and this record supersedes it.

The knob merges **off**, as apparatus and as a recorded negative. What is worth
keeping is not the number — it is that the reservation exists without a binding,
that the knob demonstrably takes the reserved CPU, and that three separate ways
of measuring "where is the dispatcher" gave confident wrong answers first.

## Why this study exists

The worker-attribution study ended by naming the **measurement**, not the
kernel, as the binding constraint at width 16. A steal-tiles change measured
**+23%** with exactly the mechanism it predicted (`sys_frac` 0.280 → 0.192) and
no width-8 regression, and was nonetheless **REJECTED**, because the A/A null in
the same run — two arms that differ in nothing at all — was **±21.5%**. No
improvement of a realistic size can clear a pre-registered bar against a null
that wide. Until the null is understood, width-16 work cannot ship.

This study is about the null.

## The null is not width-16-specific. The *kind* of slow arm is.

First, from archived data only, at zero host cost — pooling
`bb/steal_ab.json` and `bb/blocktime_ab.json`, 20 launches, 120 arms:

| | width 8 | width 16 |
|---|---:|---:|
| mean \|aa − 1\| (steal run) | 18.7% | 21.6% |
| mean \|aa − 1\| (blocktime run) | 16.0% | 10.9% |
| worst single A/A deviation | **+90%, +105%** | +49% |

**This corrected my own earlier localisation.** I had been calling this a
width-16 instability; it is as bad at width 8, and the two worst single
deviations in the whole archive are at width 8. What actually differs is the
*shape* of a slow arm. Splitting arms slower than 1.3× the launch's best by
execution position and by intra-arm rep spread:

| | width 8 | width 16 |
|---|---|---|
| slow arms | 5 / 45 | **15 / 45** |
| position | 1 and 2 only, never 3 | evenly 5 / 5 / 5 |
| intra-arm spread | **≥23%** every one (median 6.9%) | tight — one ran 1.3× slow with 0.5% internal spread |

Width 8's slow arms are warmup-shaped, position-dependent, and **self-detectable**
— the rep spread announces them. Width 16's are none of those things. An arm
that is uniformly slow for every rep, with a tight internal spread, is not being
disturbed: it is in a **different state for its whole life**. That is a
per-process property, and it points at what the process got at startup rather
than at what happened to it during the run.

## What the process gets at startup

- **NUMA is not it.** `numactl -H` reports **one** node, CPUs 0–31. First-touch
  placement cannot vary between launches when there is nowhere else to place.
  One command, hypothesis closed.
- **But the cache topology is not flat.** L3 is 64 MiB in **two** instances:
  CPUs 0–15 and CPUs 16–31. The pool's `node_worker_counts = [8, 8]` maps onto
  these two L3 complexes — not onto NUMA nodes, of which there is one.
- **The workers are pinned, and it is not `decode_affinity`.** On a single-node
  host `decide_affinity` returns `Off`, yet every worker's
  `/proc/<tid>/status:Cpus_allowed_list` is a single CPU. The pinning comes from
  the CPU-decode-budget path, which also confines the whole process to `w` CPUs.
- **At width 16 one core is deliberately left empty.** Workers 0–14 take even
  CPUs 0–28, one each. **CPU 30 is free.** `DISPATCHER_RESERVED_CPUS = 1`, and
  `reserve_single_group_headroom` caps workers at `core_count − 1` inside the
  physical-core budget. The in-tree justification is a measured **1.57×**
  (16 workers at 4.41 ms/token vs 15 workers at 2.81 ms/token): a dispatcher
  sharing a core turns that core's worker into the straggler everyone waits for.

**And the dispatcher was never put on it.** The reservation frees the core and
then nothing binds anything to it. The dispatcher is left to the scheduler,
which may place it on the free core, or on a worker's core, and may move it.
Where it lands is decided once per process, early, by the scheduler — which is
exactly the signature the archived data pointed at.

## Three instrument failures, all in the same 20 lines

I record these because two of them produced *confident wrong answers*, not
noise, and the class is general.

**1. Reading the wrong thread inverted the sign.** The first reporter called
`sched_getcpu()` on the *reporting* thread. Pin off, it read CPU 30; pin on, it
read 18. That is exactly backwards, and it is not a coincidence: the reporter is
idle, so with the pin off the scheduler parks it on the one core nobody is
using — CPU 30 — and with the pin on the dispatcher has **evicted** it from
there. A wrong-thread reading does not add variance to a placement measurement;
it can report the negative of the truth with a straight face.

**2. The dispatcher is transient.** The second version read the dispatcher's own
`/proc/self/task/<tid>/stat` (field index verified) and returned `none`. The
dispatcher is neither the process main thread nor the pool's builder, and it has
usually **exited by the time a bench can report**. Any question of the form
"where is the dispatcher" has to be answered from *inside* the dispatch path.

**3. A process dispatches from more than one thread over its life.** A
migration counter built from consecutive samples reported 2–7 moves on **pinned**
runs, which is impossible. The samples were coming from different dispatching
threads, each of which had correctly taken the reserved CPU. Fixed by recording
the first dispatching thread's tid and sampling **only** that thread — with the
baseline sample taken *after* the bind, so a pinned dispatcher reads exactly
zero. The *pin* behaviour was deliberately left unchanged (every dispatching
thread still takes the reserved CPU) so that the A/B already run was not
invalidated by a diagnostics fix.

The earlier "dispatcher/worker CPU collision was tested and excluded (one
partial match in four launches)" claim in the ledger came from a probe that
sampled `/proc/<pid>/stat` — the **main thread**. That claim rested on failure
mode 1 and every statement derived from it was removed from the code before
commit. It is not re-asserted here in either direction.

## The intervention

`ONNX_GENAI_CPU_DECODE_DISPATCHER_PIN=1` binds the dispatching thread to the CPU
the reservation already freed (`shards.last().cpus[shards.last().workers]` — the
reserved core belongs to the **last** node, because that is the shard
`node_worker_counts` adds the dispatcher to). Default **off**. One `sched_setaffinity`
per dispatching thread, from a thread-local one-shot inside the single `fn dispatch`
funnel.

## Pre-registered rules

Two, both written down before the first measurement.

1. **Throughput** — the existing validated single-knob A/B, reused byte-identical
   and pointed at the new knob via `--env-name/--control/--test`:
   median ratio ≥ **1.10**, sign consistency ≥ **80%**, effect > **3×** the A/A
   half-width, no width-8 regression below 0.95, ≥ 6 trusted launches.
2. **Dispersion** — a *new* file with its own rule, rather than an edit to the
   validated instrument: D = (p90 − p10) / median over per-launch throughput must
   fall by ≥ **2×**. Replay-only, self-tested, and it refuses to score a run whose
   test arm does not report `PIN-TOOK`.

## Result: both rules say no

Current main `7e274a4e2`, 16 launches, **15 trusted**, qwen / acc0 / block 32 /
384 tokens / 3 reps, arms interleaved with the order rotated per launch and the
A/A taken in the same launch as the effect it has to clear.

| launch | peak | ratio w16 | A/A w16 | sys_frac control | sys_frac test | ratio w8 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 19 | 1.0117 | 0.9454 | 0.165 | 0.170 | 1.0685 |
| 1 | 23 | 1.6715 | 1.0164 | 0.333 | 0.192 | 0.9895 |
| 2 | 22 | 1.3442 | 0.9954 | 0.232 | 0.148 | 0.9993 |
| 3 | 22 | 1.0174 | 1.0157 | 0.193 | 0.188 | 0.9931 |
| 4 | 22 | 1.1980 | 1.1731 | 0.201 | 0.164 | 1.0510 |
| 5 | 22 | 1.1572 | 1.0181 | 0.214 | 0.144 | 1.1607 |
| 6 | 19 | 1.0401 | 0.6822 | 0.140 | 0.154 | 1.0248 |
| 7 | 23 | 1.0008 | 1.0028 | 0.150 | 0.156 | 0.6670 |
| 8 | 24 | 1.0953 | 0.9299 | 0.192 | 0.170 | 1.0048 |
| 9 | 23 | 1.0704 | 1.0172 | 0.198 | 0.168 | 1.0440 |
| 10 | 22 | 1.0299 | 1.0245 | 0.150 | 0.145 | 1.6965 |
| 11 | 22 | 1.5696 | 1.0255 | 0.276 | 0.170 | 1.5189 |
| 12 | 23 | 1.0863 | 0.8645 | 0.191 | 0.214 | 1.9997 |
| 13 | 23 | 1.1811 | 1.0354 | 0.222 | 0.173 | 1.0865 |
| 14 | 25 | 1.2158 | 0.7295 | 0.223 | 0.170 | 1.0258 |
| 15 | 75 | *(discarded — runnable peak 75)* | | | | |

**THROUGHPUT: REJECT.** Median ratio **1.0953**, below the pre-registered
**1.10**. The other two conditions passed — sign consistency is **100%** (the
pinned arm was faster in **15 of 15** trusted launches, against a bar of 80%),
and the effect **+0.0953** clears 3× the A/A half-width, **0.0765**. The
composite rule is nonetheless REJECT, and REJECT is the verdict. The bar was
written down before the first measurement and is not being moved now that a
result has landed 0.005 underneath it.

**MECHANISM: UNPROVEN.** `sys_frac` falls 0.198 → 0.170, but the shift holds in
only 73% of launches, under the 80% the same rule requires.

**REGRESSION at width 8: none.** Ratio 1.0440, down-sign 27%.

**DISPERSION: REPORT NOTHING.** The scorer's own self-test failed:
|D(A/A) − D(A/A's reference arm)| = **0.1432**, over the allowed 0.5 × 0.2591 =
0.1296. Two arms of identical configuration produced dispersion estimates that
differ by more than the estimator's tolerance, so the estimator cannot reproduce
itself at n=15 and is not entitled to compare anything. D(control) = 0.3610 and
D(test) = 0.0780 are recorded here as **unscored observations only** — they are
what the rule refused to certify, not a result.

### The earlier ACCEPT does not replicate, and this supersedes it

An earlier run of the same knob against the same throughput rule returned
**ACCEPT** — ratio 1.1910, effect 0.1910 vs a 3× A/A half-width of 0.1477 — and
the dispersion rule returned **PIN-STABILISES**, 0.2781 → 0.0416. That run had
**6** trusted launches and was taken on `d5e585d2a`, before #1868 landed. The
run above has 15 and is on the tree this branch actually merges into. Two things
moved:

- **n.** Six launches put the median 19% up; fifteen put it 9.5% up with the
  same sign in every launch. The six-launch median was optimistic, which is the
  ordinary behaviour of a median over a heavy-tailed sample, not a defect in
  either run.
- **The baseline.** #1868 fixed the spin deadline at two yield sites, and
  control `sys_frac` at width 16 fell from 0.257 to 0.198 between the two runs.
  Some of what the pin was recovering has already been recovered upstream.

The larger, on-tree run wins. **The pin does not clear its bar on current main.**

## What is established, and what is not

**Established, and not by timing:**

- The pool reserves a CPU for the dispatcher (`DISPATCHER_RESERVED_CPUS = 1`,
  justified in-tree by a measured 1.57×) and **binds nothing to it**. At width
  16 on this host that is CPU 30, and with the knob off the dispatcher is an
  ordinary unpinned thread.
- The knob does what it says. Direct measurement, 4 launches per arm,
  interleaved: pinned reports `observed_cpu=30` and **0 migrations** in every
  launch; unpinned reports 1, 1, 1 and 0, and in one launch was last seen on
  **CPU 2** — a worker's core — rather than the reserved one.
- Native's width-16 A/A instability is **not** width-16-specific in magnitude,
  and the width-16 slow arms are internally consistent, which makes them a
  per-process state rather than a disturbance.

**Not established:**

- **That the pin helps.** Directionally consistent 15/15 and below its bar.
- **Why it would.** The migration counter samples once per 1024 dispatches —
  roughly 150 samples per launch — and sees at most one change per unpinned
  launch. That rate is far too low to make steady-state migration the
  explanation, so the counter's honest reading is that **migration is not the
  mechanism**, not that it is. The leading remaining candidate is *wakeup*
  placement rather than residence: the dispatcher parks and is woken hundreds of
  times per token, and a single-CPU affinity mask lets the kernel skip the
  idle-sibling search on each wake. `sys_frac` falling is consistent with that
  and is not evidence for it at 73% sign. **Nothing here should be cited as a
  mechanism.**
- **That the A/A null is fixed.** It is the thing that blocked the +23%
  steal-tiles candidate, and the dispersion rule declined to certify any change
  in it.

## Why the knob ships off, and what flipping it would need

Beyond it not having cleared its bar: the dispatcher is the **session thread**,
and `sched_setaffinity` is not scoped to a decode. A thread pinned during decode
keeps that mask afterwards, so a subsequent **prefill** on the same thread would
run one CPU wide. This harness measures decode only and cannot see that. Any
proposal to flip the default needs prefill in the matrix, not more decode
launches.


## Reproduce

```bash
cargo build --release -p onnx-runtime-ep-cpu --benches
BIN=$(ls target/release/deps/int4_decode_loop_ab-* | grep -v '\.d$' | head -1)
./scripts/hostlock.sh run --wait --gate 8 --reason "acc0 dispatcher pin ab" -- \
  python3 crates/onnx-runtime-ep-cpu/benches/acc0_w16_blocktime_ab.py \
    --binary "$BIN" --env-name ONNX_GENAI_CPU_DECODE_DISPATCHER_PIN \
    --control 0 --test 1 --launches 16 --out pin_ab.json
python3 crates/onnx-runtime-ep-cpu/benches/acc0_w16_dispersion.py --replay pin_ab.json
```

Non-vacuity is checked by the harness itself: the `dispatcher …` row must read
`PIN-OFF` on the control arm and `PIN-TOOK` on the test arm, and the dispersion
scorer aborts if it does not.
