# The last bimodality candidate was malformed; the straggler is a work-imbalance lead

Date: 2026-08-24 · Owner: Roy · Base: `7bf32c5f4` (+ `squad/roy-clock-state`)
Probe: `crates/onnx-runtime-ep-cpu/benches/acc0_w16_mode_worker_split.py`

## 1. "Weight-arena placement across the two L3/CCX domains" cannot exist here

I carried this candidate through two records. It is malformed on this host:

```
$ numactl --hardware
available: 1 nodes (0)
node 0 cpus: 0 1 2 ... 31
node distances:  0:  10
```

**One NUMA node**, all 32 CPUs, single distance. There is no second memory
domain to place an arena in, and an L3 is a cache rather than an allocation
target — a read-only weight stream is simply replicated into whichever L3 reads
it. I should have checked that the host could express the hypothesis before
listing it twice as the remaining candidate. Dropped.

That empties the candidate list: placement, THP/page backing, foreign load,
static spare-tile steal, clock/boost and now arena placement are all closed.

## 2. Stratifying the worker-split instrument by mode — REPORT NOTHING

The outside-in evidence had narrowed to a statement about *participation*: in a
slow launch realized lanes fall ~15.5 → ~12.2 while both user and sys CPU per
token stay flat, so the missing lanes are neither running nor in the kernel.
The in-EP `SpmdWorkerProfile` counters (`work_ns`, `wake_ns`, `last_arrivals`)
measure exactly that, and had never been read split by mode.

This adds no instrument: it runs the validated `acc0_w16_worker_split.py` and
imports its `trusted()` and derived fields rather than reimplementing them.

16 launches, 14 trusted. **Both gates fired:**

| gate | value |
|---|---|
| launches per mode | fast 12, **slow 2** (need ≥3) |
| between-mode gap vs within-mode spread | gap 0.106 s vs spread **0.292 s** (need 3x) |

The second is the more informative one: under worker profiling the width-16
`wall_s` values are **one broad distribution, not two**. Two clock reads per
worker per op are enough to dissolve the bimodality. That is a real constraint
on this line of attack — the counters that would explain the modes perturb the
system enough to remove them — and it is why the pre-registered rule refused to
name a mechanism.

## 3. What the aggregate run did show — a work-imbalance lead

Not stratified, so not an answer to the bimodality, but large and in my domain:

| quantity | value | chance value |
|---|---|---|
| straggler wait | **0.313 of the window** | — |
| max share of `last_arrivals` held by one worker | **0.565** | 0.067 (1/15) |
| `straggler_excess` (share × n) | **8.47** | 1.0 |
| `work_skew` (max/mean − 1) | **0.562** | 0.0 |

One worker does ~56% more work than the mean and is systematically last, and
every other worker waits for it on every one of those ops. 31.3% of the width-16
window is spent waiting at the barrier.

## 4. It is not contention on cpu 0

A cross-agent report attributed straggling to a "permanent external competitor
on cpu 0" (50.3% of a core). Measured directly, 1.5 s pinned scalar loop per
CPU, quiet host:

```
cpu= 0 iters=2122 rel=0.965      cpu= 8 iters=2266 rel=1.031
cpu= 1 iters=2204 rel=1.003      cpu=16 iters=2191 rel=0.997
cpu= 2 iters=2138 rel=0.973      cpu=28 iters=2198 rel=1.000
cpu= 4 iters=2152 rel=0.979      cpu=30 iters=2269 rel=1.032
cpu= 6 iters=2133 rel=0.970      cpu=31 iters=2202 rel=1.002
```

**cpu 0 is inside the 0.965–1.032 band of every CPU tested.** There is no
competitor on it now. Whatever was measured earlier was transient or was the
observer's own load. So the straggler is not an external-contention artifact,
and the imbalance is ours.

This also repairs a gap in my own foreign-load falsifier, which bounded
*aggregate* foreign CPU across all 16 pinned CPUs at ≤1.7%. A barrier makes
that bound weaker than it looks: load concentrated on a single lane is
sufficient to hold every other lane, so the aggregate form would not have caught
a one-CPU competitor. The bound stands for this host only because the direct
per-CPU census above shows there is no such competitor.

## 5. Explicitly not diagnosed

`output_chunk_len_for(threads, n, k)` returns `n.div_ceil(tasks)` with
`tasks ≤ threads`, and every llama projection width divides evenly by 16, so a
static reading says the split is exact and predicts **no** skew. The measurement
says 0.562. I am recording that contradiction rather than resolving it from
source, because reasoning a mechanism out of code and publishing it without
measurement is the specific error this ledger has caught three times (a
dispatcher-yield "contention signal" that saturates at 1.0 per dispatch, a
"realized width" that never asserted placement, and my own spare-tile candidate).

The next step is to measure *which* lane is last and *how much* work it holds
per op, not to argue about `div_ceil`.
