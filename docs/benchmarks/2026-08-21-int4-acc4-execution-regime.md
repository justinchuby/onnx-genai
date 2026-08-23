# Execution regime of the packed-nibble int4 `accuracy_level = 4` decode kernel

**Question asked:** is the current kernel instruction/front-end/latency bound or
memory/LLC-bandwidth bound, and what is the projected benefit bound for
**(a)** an N-column tile that shares activation work, versus **(b)** bf16 scales
plus a block-major prepack that removes ~10% of the byte traffic?

**Answer: instruction-bound up to 8 threads, and neither lever is worth
building.** In the production (DRAM-resident) regime lever (a) measures
**0.94x** — a loss — and lever (b) measures **0.96x at tile 1 / 1.10x at tile
8**. The N-tile's attractive number only appears when the benchmark working set
fits in L3, which the real decode never does.

**What is worth building is a third thing neither lever names.** The kernel
still pays per-block bookkeeping — two bounds-checked slices constructed per
block, plus a group count recovered from `packed.len()` on every call. Removing
it is worth **1.69x at t=1 and 1.64x at t=4** at block 32, **1.38x at block 64**,
and **1.24x at t=16**, bit-identical output. It is a **wash at t=8** and at
**block 16** (which never enters the wide path). That patch is in this branch;
see [The recommendation](#the-recommendation).

> **Re-measured on latest main by Roy (2026-08-21), with corrections.** The
> original draft of this document claimed a flat "1.61x-1.63x at 1-8 threads"
> plus 1.212x at block 64. **Four of its nine rows do not reproduce**, and two
> of my own first-pass rows did not survive re-measurement against current main
> either. Current standing: the win is real and *larger* than the draft claimed
> where it exists (1.686x at t=1), it extends to **block 64 at 1.380x** —
> my earlier "block-32-specific" reading was itself an artifact of only ever
> measuring block 64/128 at t=8 — and it is **a wash at t=8** on current main,
> where the baseline itself got 7.8% faster. The t=16 row was *pessimistic* in
> the draft; the true figure is ~1.24x, not 1.065x. The full corrected matrix,
> the re-measurement against current main, the reason the original t=8 baseline
> was inflated, and a methodology defect in the multi-session rows are in
> [Confirmed on the real kernel](#confirmed-on-the-real-kernel).

> **Superseded width labels (2026-08-22, #1771).** Every `t=N` row in this
> document was taken before **#1766** (`11cb8e5f3`), on a process that never
> called `CpuExecutionProvider::initialize()`. The decode budget sized the SPMD
> pool but did not *bound the process*: affinity stayed at all logical CPUs and
> the global Rayon pool ran full width at every budget. So a row labelled `t=8`
> means "8 SPMD workers inside an unbounded process", not "8-wide execution".
>
> Be precise about what that does and does not invalidate. Each speedup here is
> a **paired A/B whose two arms shared the same broken topology**, so the
> *ratios* survive — 1.686x at t=1 and 1.380x at block 64 are still the measured
> effect of removing per-block bookkeeping. What does not survive is any claim
> that attributes an effect **to a width**, because the width on the label was
> not the width that ran. Specifically: the "wash at t=8", the t=16 rows, "the
> one place bandwidth does bind is full width", and every cross-width comparison
> in [The regime](#the-regime-instruction-bound-not-bandwidth-bound) need
> re-deriving on post-#1766 main before they are built on or re-quoted.
>
> Rows taken from now on carry their realized width inline (#1764, #1770):
> `decode_width requested=2 realized=2 path=spmd-pool as_requested`. The
> contract behind that label is asserted in CI at every benchmarked width by
> `every_benchmarked_decode_width_realizes_the_worker_count_it_requests`
> (#1747), so a future sweep cannot silently print a width it never ran.

This extends rather than contradicts #1619, which already found that "most of
the win is not the byte saving" but "removing per-block overhead". The finding
here is that a 1.6x of per-block overhead was still left on the table.

## Host, and two corrections to the #1619 record

AMD EPYC 9V74, 32 vCPU. Both corrections change how results must be read, so
they come first.

1. **L3 is 32 MiB per CCX, not 64 MiB shared.** `lscpu`/sysfs show two
   16-vCPU CCXs (CPUs 0-15 and 16-31) with a 32 MiB victim L3 each. #1619's
   residency argument — that a 58.7 MB expanded weight is "L3-resident on a
   64 MiB L3" — does not hold: no single core can see 64 MiB, and the figure
   that matters for a decode token is the **136.3 MB** of packed weights plus
   scales for the whole projection set, which is ~4x one CCX's L3.
2. **SMT siblings are adjacent pairs** — (0,1), (2,3) ... (30,31) — so the
   physical-core set is the **even** CPUs. Any experiment that pins to
   `0..N` is really using `N/2` cores with both siblings loaded.

**There is no PMU.** Every hardware event reads `<not supported>`; this was
confirmed to be virtualization, not permissions, by lowering
`kernel.perf_event_paranoid` from 4 to 1 and re-testing (restored to 4
afterwards). So no cycles/instructions/LLC-miss counters exist on this host, the
regime had to be established **behaviourally**, and **no roofline is quoted** —
the bandwidth denominator below is measured, not from a spec sheet.

### Measured bandwidth ceiling (the denominator)

`bwprobe`, T threads over a partitioned shared buffer, 8 accumulators, pinned to
physical cores:

| working set | 1t | 4t | 8t | 16t |
|---|---|---|---|---|
| 2 GiB | 31.1 | 37.1 | 36.4 | 71.5 GB/s |
| 136 MiB (a real token's traffic) | 32.0 | 36.1 | 33.1 | **56.6** (peak 72.5) GB/s |

A single CCX saturates at ~36 GB/s from only ~4 cores. #1619's "75.8 GB/s DRAM"
is reachable only across both CCXs.

## Method

The host is shared and bursty (load 1.6-11 during this work, other agents'
processes resident). Three controls were needed before any number was allowed to
count, and each of them rejected something:

- **Round-robin interleaving.** Timing arm A to completion, then arm B, lets a
  co-tenant episode land entirely inside one arm. Every arm is run once per
  cycle instead, and the median is taken over cycles. This was not a
  precaution: sequential timing produced a **bimodal** result for identical
  code — 1.79 ms and 3.05 ms at the same load average — because CPU 0's SMT
  sibling is CPU 1, and a co-tenant landing there costs ~1.7x.
- **In-run A/A null control.** Two arms of every experiment are the *same*
  code. A run whose two identical arms disagree by >2-3% cannot support a claim
  about arms differing by less, and is discarded whole. This rejected 1 of 5
  microbenchmark replicates and 3 of 17 real-kernel A/B runs, including one at
  87% apart.
- **A correctness gate before any timing.** Every variant is compared
  element-by-element against the baseline first. All report
  `max_rel = 0.000e0` — bit-identical, not merely close.

Knob used for thread count is `ONNX_GENAI_CPU_DECODE_THREADS` (the harness
documents that `RAYON_NUM_THREADS` does **not** size this pool), affinity via
`taskset` over even CPUs, `PROBE_ACCURACY=4`.

One negative result about the tooling: **block_size 16 cannot be measured with
this microbenchmark** and is excluded. At `blob = 8` the wide path's
`groups = blob/16` is 0, so both arms skip the inner loop entirely and agree at
zero work. Production has a separate narrow path; the numbers a naive run
produces for block 16 are meaningless.

## The regime: instruction-bound, not bandwidth-bound

Five independent lines of evidence, all pointing the same way.

| evidence | measured | expected if bandwidth-bound |
|---|---|---|
| single-thread byte rate | 5.86 GB/s = **18%** of the 32 GB/s 1-thread ceiling | at/near ceiling |
| scaling 1->16 cores | **7.3x** | <=1.77x (the measured bandwidth curve)* |
| block sweep 16->128 at **fixed weight bytes** | **3.73x** | 1.41x (163.6 -> 115.9 MB) |
| `time = blocks*A + weights*B` fit | A = 2.850 ns/block, B = 24.15 ps/weight; per-block tail is **79%** of time at block 32 | tail ~0 |
| per-column overhead (4096x14336 vs 14336x4096: identical weights *and* identical block count, 3.5x the columns) | wide-n shape **2.4% faster** | n/a |

The fit independently reproduces #1619's own model (8.5 cyc/block,
13.4 weights/cyc) at an assumed ~3 GHz.

> \* **Note (2026-08-23):** the `1.77x` comparator in row 2 comes from a width
> curve whose error bar has since been withdrawn — `w=16` is bimodal on a
> shared host (see
> [2026-08-23-acc4-decode-width-remeasurement.md](2026-08-23-acc4-decode-width-remeasurement.md)).
> The conclusion drawn from that row is unaffected: the gap between 7.3x and
> the bandwidth curve is far larger than any plausible revision of the
> comparator, so the not-bandwidth-bound reading stands. Row 2 should not be
> re-quoted as a precise figure.

**The one place bandwidth does bind is full width.** At t=16 the kernel demands
42.7 GB/s against a measured 56.6 GB/s ceiling (~75%), and that is exactly where
the improvements below stop paying — see the t=16 row of the results table.

The last row of the table is the direct falsifier of lever (a)'s premise: an
N-tile amortizes fixed per-column cost, and **there is no fixed per-column cost
to amortize.**

## Lever (a): the N-column tile

`ntile`, an AVX2 microbenchmark replicating `nibble_outputs_avx2` exactly (its
per-block cost lands within 7% of the real kernel's, so it is a faithful
stand-in). Two forms were tested, because the weak one is what the phrase
"N-tile" usually means and the strong one is the actual proposal:

- `tile<T>` — shares only the scale/`block_sum` loads across T columns.
- `shared<T>` — additionally hoists the activation `even`/`odd` loads out of the
  column loop, so loads per block fall from `3*T` to `T+2`. This is the version
  that "shares activation work".

Medians over 4 A/A-passing replicates, block 32, one physical core:

| arm | **DRAM-resident** (40 MB, production-like) | L3-resident (10 MB, artifact) |
|---|---|---|
| baseline `tile<1>` | 3.334 ns/block — 1.00x | 3.321 ns/block — 1.00x |
| `tile<2>` / `tile<4>` / `tile<8>` | 0.87x / 0.92x / 1.01x | 1.01x / 1.02x / 1.13x |
| `shared<1>` | **1.53x** | **1.52x** |
| `shared<8>` | 1.44x | 1.87x |
| **N-tile effect alone** (`shared<8>` / `shared<1>`) | **0.94x — a loss** | 1.23x |

And under full-width pressure — one probe copy per physical core, so aggregate
demand scales exactly as it does with 16 decode threads:

| arm | ms, median of 16 copies |
|---|---|
| `shared<1>` | **1.149** |
| `shared<8>` | 1.512 (**0.76x** — a 24% loss) |

**Lever (a) is an L3-residency artifact.** It is worth +23% when the whole
working set fits in a 32 MiB L3, 0.94x when it streams from DRAM as production
does, and 0.76x at 16 cores. The mechanism is that TILE=8 turns one sequential
weight stream per thread into eight, and at 16 threads that is 128 concurrent
streams. **Do not build it.**

## Lever (b): bf16 scales / block-major prepack

The traffic claim is right. At block 32, scales are 27.26 MB of the 136.31 MB a
token moves — **20%** — so halving them removes **10.0%** of the stream. But the
share is strongly block-size dependent (block 16: 33%, block 32: 20%, block 64:
10%, block 128: 5.9%), so the lever shrinks as block size grows.

Measured with the same gate, an added `bf16<T>` arm holding scales as `u16` and
rebuilding the f32 with one unpack plus one shift:

| comparison | DRAM-resident | L3-resident |
|---|---|---|
| `bf16<1>` vs `shared<1>` | **0.96x** | 0.97x |
| `bf16<8>` vs `shared<8>` | 1.10x | 1.01x |

**Lever (b) is a wash.** A 10% traffic cut buys nothing when the kernel is using
18% of the bandwidth available to it; the two extra uops per tile cost slightly
more than the bytes save. Its 1.10x at tile 8 is real but is only recovering
part of what the N-tile lost, and tile 8 is itself a loss.

The one condition under which (b) could pay is the one place bandwidth binds —
t=16, at ~75% of ceiling. It is not recommended on that basis, because the
measured t=16 behaviour (below) shows the decode loop there is limited by the
runtime, not by the kernel.

## What actually pays: per-block bookkeeping

`nibble_outputs_avx2` constructs **two bounds-checked slices per block**:

```rust
nibble_block_acc_avx2(
    &weights[block * blob..(block + 1) * blob],
    &activation.values[block * block_size..(block + 1) * block_size],
    group,
)
```

and the callee then recovers `let groups = packed.len() / (WIDE_GROUP / 2)` and
re-branches on `group < WIDE_GROUP` — **once per block**. At block 32 a block is
a single 10-uop group, so this bookkeeping is not amortized by anything.

An attribution arm settles where the cost is. `rawptr<T>` is structurally
identical to the baseline — same per-block call boundary, same 4-lane array,
same hadd tree, **no** activation sharing — and differs only in using raw
pointers and a hoisted group count:

| arm | DRAM-resident |
|---|---|
| `rawptr<1>` | **1.51x** |
| `shared<1>` | 1.53x |

`rawptr<1>` recovers the entire gap. **None of it is loop restructuring or
activation sharing; all of it is per-block slice construction and the redundant
group recompute.**

### Confirmed on the real kernel

Not a microbenchmark claim. `int4_decode_loop_ab`, `PROBE_ACCURACY=4`,
1 layer, interleaved A/B with the baseline binary entered twice as an A/A
control.

**The table below is the re-measurement on latest main (`f8eb8a3e2`), not the
original draft's numbers.** Both arms are built from one tree, differing only in
`int4_nibble.rs`; arm B is the final shipped form (see [Safe formulations were
tried first](#safe-formulations-were-tried-first)). Estimator is **min over 6+
interleaved repetitions** — this host takes intermittent one-sided interference
spikes, and contention can only ever *add* time, so the minimum is the robust
estimate of uncontended cost. Every row carries its own A/A; rows failing the
2% gate are not reported as results. **All rows in this table are against base
`f8eb8a3e2`**; they are retained as the original correction record. The
current-main re-measurement is the table in [the next
section](#re-measured-on-current-main-and-two-rows-changed), and it supersedes
these rows at t=8 and at block 64/128.

| threads | sessions | block | baseline | patched | speedup | A/A | original draft claimed |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 32 | 23.525 ms/tok | 14.016 | **1.678x** | 0.01% | 1.626x — reproduces |
| 4 | 1 | 32 | 7.963 | 4.777 | **1.667x** | 0.05% | 1.633x — reproduces |
| 8 | 1 | 32 | 3.542 | 3.104 | **1.141x** | 0.23% | 1.609x — **does not reproduce** |
| 16 | 1 | 32 | 1.801 | 1.447 | **1.245x** | 0.83% | 1.065x — **draft was pessimistic** |
| 8 | 1 | 64 | 2.762 | 2.747 | 1.005x | 0.43% | 1.212x — **does not reproduce** |
| 8 | 1 | 128 | 2.534 | 2.527 | 1.003x | 0.04% | 1.031x — **does not reproduce** |
| 4 | 2 | 32 | 179.9 tok/s | 255.3 | **1.419x** | 1.75% | 1.497x (different metric) |
| 2 | 4 | 32 | 175.9 tok/s | 248.9 | **1.415x** | 0.11% | 1.137x (different metric) |

### Re-measured on current main, and two rows changed

The table above was taken against `f8eb8a3e2`. Main moved eleven commits (the
native-MTP series) before this could land, so it was re-measured end to end
against `2f94cba4d` with a freshly built baseline arm. **Stale green is not
evidence**, and re-running found one reversal and one under-claim:

| threads | block | baseline | patched | speedup | A/A | vs. previous |
|---|---|---|---|---|---|---|
| 1 | 32 | 23.535 ms/tok | 13.962 | **1.686x** | 0.00% | 1.678x — holds |
| 4 | 32 | 6.100 | 3.720 | **1.640x** | 0.16% | 1.667x — holds |
| 8 | 32 | 3.264 | 3.214 | **1.016x** | 0.03% | 1.141x — **reversed, now a wash** |
| 16 | 32 | 1.804 | 1.452 | **1.242x** | 0.33% | 1.245x — holds |
| 1 | 16 | 44.003 | 42.803 | 1.028x | 0.06% | not previously measured |
| 1 | 64 | 14.515 | 10.517 | **1.380x** | 0.03% | 1.005x @ t=8 — **was under-claimed** |
| 1 | 128 | 10.251 | 9.392 | **1.091x** | 0.10% | 1.003x @ t=8 — **was under-claimed** |

**The t=8 win is gone.** Two independent runs (1.009x at 240 tokens, 1.016x at
400 tokens, A/A 0.52% and 0.03%) agree. What moved is visible in the arms: the
*baseline* got 7.8% faster on new main (3.542 → 3.264) while the patched arm
barely moved, so the cell closed on its own. **Why** it closed is not
established: the rebase window contains the native-MTP series, but those are
CUDA-graph and speculative-decode commits with no stated path into a CPU int4
decode kernel, so attributing it to them would be a guess. Recorded as
correlated with the window, not explained. Either way this change no longer buys
anything at t=8 and that row is reported as a wash.

**"Block-32-specific" was wrong, and the error was methodological.** Block 64
and 128 had only ever been measured *at t=8* — the one thread count where
nothing shows. Measured at t=1, where the mechanism is visible, block 64 gets
**1.380x** and block 128 **1.091x**. The win covers the two most common
configurations, not one.

The ladder is consistent with what the mechanism predicts. `wide` requires
`group >= WIDE_GROUP` (32), and `wide_groups = blob / 16`, so the removed fixed
per-block cost is amortized over 1, 2 and 4 groups at block 32, 64 and 128:

| block | wide? | groups | predicted `1 + F/(n·G)` | measured |
|---|---|---|---|---|
| 16 | **no** — `group < WIDE_GROUP` | — | 1.0x (path not taken) | 1.028x |
| 32 | yes | 1 | (fit: `F = 0.686·G`) | 1.686x |
| 64 | yes | 2 | 1.343x | 1.380x |
| 128 | yes | 4 | 1.172x | 1.091x |

Fitting `F` from the block-32 row predicts block 64 within 3% and block 128
within 8% — the residual is because `G` is not constant across block sizes (the
absolute baseline cost per group falls as blocks get larger and the activation
is re-read less often). That second, unfitted variable is why this is
*consistent with* amortization rather than a clean prediction from it: it is a
one-parameter fit checked at two points. Block 16 is the stronger evidence — it
provably never enters the wide path, and it measures nothing.


Output is bit-identical. The t=16 row uses a 240-token steady window (see
[below](#the-t16-row-needed-a-longer-window)); the multi-session rows use
aggregate throughput, for a reason that is itself a finding:

#### The multi-session rows were measured with an invalid statistic

Column 2 of the harness is the **median per-token latency pooled across
sessions**. Sessions are spawned together but finish independently, so as soon
as one session exits, the survivor's remaining tokens run *uncontended* and pull
the pooled median down. The statistic therefore measures session
desynchronisation as much as it measures the kernel, and it breaks the
one-sidedness that makes a minimum meaningful — a baseline repetition read
**9.348 ms against a 15.9 ms population**, and the resulting A/A was **52.7%**.
Under this statistic the 2-session cell "measured" 1.591x on one run and failed
its control on the next.

Aggregate throughput (`sessions * tokens / wall`) is defined over the whole
window and is insensitive to which session finishes first, so the multi-session
rows use it, with the **maximum** as the robust estimator. Re-measured that way
the two cells agree closely with each other (**1.419x** and **1.415x**) and pass
their controls. The honest multi-session number is therefore ~1.42x, **not** the
~1.59x the latency statistic suggested.

#### Why the original t=8 row was inflated

At t=8 the *baseline* arm is bimodal — 3.54 ms or ~5.2 ms — while the patched
arm is stable at 3.10-3.68 ms. The original draft's baseline of **5.880 ms is
its slow mode**; against the reproducible 3.542 ms fast mode the speedup is
1.141x, not 1.609x. This is not a dispute about noise handling: the A/A control
passes at 0.23% because *both* baseline arms reach 3.54 ms reproducibly. Taking
medians instead would report 1.541x here, which is why the estimator has to be
stated — the patched kernel is genuinely more robust under load, but that is a
different claim from being 1.6x faster uncontended, and only the second one is
what a "1.6x" headline is read as.

#### The t=16 row needed a longer window

At the default 60-token window t=16 gave contradictory estimators — min 1.234x,
median 0.730x — because the patched arm was bimodal with only 1 sample of 10 in
its fast mode. Re-run with a **240-token** steady window both estimators agree
(min 1.245x, median 1.412x, A/A 0.83%), and the figure reproduces across three
independent runs (1.227x, 1.234x, 1.245x). A cell whose estimators disagree in
*direction* is a cell that has not been measured yet; the fix is a longer
window, not a choice of statistic.

#### Where the win is, and is not

The effect scales inversely with the group count, exactly as the mechanism
implies. At block 32 with the wide path a block is exactly one 32-element group
(`wide_groups == 1`), so the removed bookkeeping — two bounds-checked slice
constructions and a group recount — is amortized over a single ~10-uop group and
dominates it. At block 64 and 128 the same fixed cost is spread over 2 and 4
groups, giving **1.380x and 1.091x**. At block 16 the kernel does not take the
wide path at all (`group < WIDE_GROUP`), and it measures **1.028x: nothing**.

**An earlier revision of this document called the win "specific to block_size
32". That was wrong, and it is worth recording why**, because the error is the
same class as the three it was written to correct: block 64 and 128 had only
ever been measured at **t=8**, which is precisely the thread count where this
change buys nothing on current main. Two independent variables were confounded
in one cell. Measured at t=1, block 64 is a 1.380x win, not a wash.

Any future summary of this work that says "1.6x on the int4 acc4 decode kernel"
without saying "at block 32 or 64, and not at 8 threads" is overstating it.

#### Null control: `accuracy_level = 0` is untouched

`accuracy_level = 4` is the only value that reaches the packed-nibble route, so
the same A/B run at `PROBE_ACCURACY=0` is a disclosed null control — it exercises
the whole harness, both binaries and the same thread counts, and must show
nothing. It does:

| threads | block | baseline | patched | speedup | A/A |
|---|---|---|---|---|---|
| 1 | 32 | 56.345 ms/tok | 56.373 | 1.000x | 0.06% |
| 4 | 32 | 18.843 | 18.887 | 0.998x | 0.27% |
| 8 | 32 | 8.147 | 8.151 | 1.000x | 0.43% |

This is the control that makes the acc4 rows mean something: the same
measurement apparatus that reports 1.678x at acc4 reports 1.000x at acc0.
(It also shows acc4 is ~2.4x *faster* than acc0 at t=1 on this shape — 23.5 vs
56.3 ms/token — which is #1619's result, not this one's.)

## Where this leaves the ORT gap

A speedup against our own previous self is not the number the assignment asks
for. `benches/ort_baseline.py` only covers f32 ops, so I built the ORT
counterpart of this harness directly: one ONNX graph holding the same five
llama3-8B `MatMulNBits` projections, same `K`/`N`, same `block_size=32`, same
`accuracy_level`, `m=1`, CPU EP, `intra_op_num_threads` matched, and **one
`Run` == one decode token** so the unit is the same. Median per token,
min-of-3-reps, same host and same session as the native numbers.

The attribute is honoured rather than assumed: ORT at `accuracy_level=0` costs
30.632 ms against 7.822 ms at `accuracy_level=4`, a 3.9x separation, so ORT is
genuinely running its int8-activation path in the acc4 rows.

Both native columns below are the **current-main (`2f94cba4d`)** arms from the
re-measurement above — not the superseded `f8eb8a3e2` numbers. Mixing the two
bases would credit this change with the t=8 win that the re-measurement
retracts.

| threads | ORT acc4 | native, main | native, this PR | gap before | **gap after** |
|---|---|---|---|---|---|
| 1 | 7.822 ms/tok | 23.535 | 13.962 | 3.01x | **1.78x** |
| 4 | 2.154 | 6.100 | 3.720 | 2.83x | **1.73x** |
| 8 | 1.249 | 3.264 | 3.214 | 2.61x | 2.57x |
| 16 | 1.227 | 1.804 | 1.452 | 1.47x | **1.18x** |

So this change takes the single-thread deficit from **3.01x to 1.78x**, which is
the largest single step any int4 decode change has produced so far, and at t=16
brings us to **1.18x** — near parity, though see the caveat below about *why*
that row flatters us.

Three things this table says that the headline speedup does not:

- **ORT saturates by t=8** (1.249 → 1.227 from 8 to 16 threads, 2%). Our t=16
  row looks good partly because ORT has stopped scaling, not only because we
  got faster. Parity at t=16 is worth much less than the same ratio at t=1.
- **The worst remaining row is t=8 at 2.57x**, and it barely moves (2.61x →
  2.57x) because this change is a wash at t=8 on current main. That cell is now
  the top of the list, and nothing in this PR addresses it.
- **Zero-points cost ORT ~26%** — 9.885 ms with per-block zero-points against
  7.816 without, at t=1, each the min over three windows. (An earlier revision
  said 28% from a single 10.041 ms window; more reps moved it down. The
  zero-point arm is noticeably noisier than the plain one — a short 3-rep/32-token
  window read 12.2 ms — so this cell needs the full rep count to converge, and
  the figure is quoted as approximate.) Every row above is measured *without*
  zero-points on **both** sides, which is ORT's fastest configuration and
  therefore the harder comparison. Our kernel handles the zero-point case in the
  same loop; comparing there would make us look better and is not the number I
  am reporting.

Caveat on comparability: ORT's figure includes per-`Run` binding and allocation
for a five-node graph, and ours includes `with_decode_pool_scope` entry. Neither
is free and they are not the same overhead. At t=1, where a token costs 8-14 ms,
that framing cost is small relative to the gap; at t=16, where a token costs
~1.2 ms, it is not obviously small, so **the 1.18x row carries more uncertainty
than the 1.79x row.** I am not claiming parity at t=16 on this evidence.

### The default path is untouched and is now the worse gap

This kernel is gated to `accuracy_level = 4`. Production default is
`accuracy_level = 0`, and measuring both sides there on the same host:

| threads | ORT acc0 | native acc0 | gap |
|---|---|---|---|
| 1 | 30.632 ms/tok | 56.307 | **1.84x** |
| 8 | — | 14.091 | — |

> **Scope note (2026-08-23).** The `1.84x` is a **`threads = 1`** figure and is
> the only acc0 cell with an ORT baseline — the t=8 row has no ORT number, so
> **the acc0 gap at production width is unmeasured**. This is not a pedantic
> label: on the acc4 table in this same document the gap moves from 3.01x at
> t=1 to 1.47x at t=16, so width is among the largest effects here and 1.84x
> should not be assumed to survive to t=8/16. At t=1 the native side also runs
> `path=flat`, confined by the decode budget to one CPU with no pool built
> (see
> [2026-08-23-acc4-decode-width-remeasurement.md](2026-08-23-acc4-decode-width-remeasurement.md)),
> so this row compares native-serial against ORT-single-thread specifically.
> Measuring ORT acc0 at t=4/8 is the prerequisite for sizing acc0 work.

So the honest summary is that we have moved the *opt-in* path from 3.01x to
1.79x and left the *default* path at 1.84x, where it was. Those two numbers
being nearly equal is a coincidence of this shape, not a shared cause: the acc4
gap is now dominated by the t=8 scaling anomaly, while the acc0 gap is a
different kernel entirely. **Closing acc0 is a separate, larger piece of work
and nothing here should be read as progress on it.**

Two implementation details cost more than they saved and are recorded so they
are not re-tried:

- **Const-generic group counts** (`match wide_groups { 1 => ...::<1>, ... }`)
  put a `match` inside the block loop and gave **0.945x** — worse than the
  baseline. The trip count is not the problem.
- The pointer computations must be hoisted **above** the `if wide` branch. With
  them inside, block 64 regressed to 0.876x. (The paired "hoisted, it is 1.212x"
  figure does not reproduce — hoisted block 64 measures 1.005x on latest main,
  see the table above. What survives is the *ordering*: inside-the-branch is
  worse than hoisted, so keep them hoisted.)

### Safe formulations were tried first

The shipped kernel uses raw pointers inside `unsafe`. That is only defensible if
a safe formulation is actually slower, so three were built and measured before
the raw form was accepted. All are bit-identical; all A/A gates <= 0.05%;
single physical core, block 32, min of 6 interleaved repetitions. **Base
`f8eb8a3e2`** — these are relative comparisons among formulations measured in
one session, so the base affects the absolute ms but not the ranking, which is
the result being used here.

| form | ms/token | vs main |
|---|---|---|
| main (two bounds-checked slices per block, group recount per call) | 23.525 | 1.000x |
| **E** — tile `chunks_exact` + bounds-checked index sub-slicing | 17.556 | 1.340x |
| **C** — nested `chunks_exact` (tile, then block), slice accumulator | 15.254 | 1.542x |
| **D** — as C, but hoisted group count and an index loop inside | 15.254 | 1.542x |
| **B** — raw pointers, hoisted group count (the draft's form) | 14.527 | 1.619x |
| **F** — B, plus `#[inline]` on the split-out kernel and call-site prevalidation | **14.016** | **1.678x** |

Two things follow. **C and D are identical to three decimal places**, which
rules out the inner accumulator loop's shape as the cost — the entire difference
between the safe and raw forms is caller-side slice construction, exactly where
the original attribution said it was. And the safe forms are measurably slower
than the shipped one: the best of them (C/D) runs **8.8% slower** than F and so
gives up **13.0% of the win** (F saves 9.509 ms/token over main, C/D saves
8.271); E runs 25.3% slower and gives up 37.2%. Codegen is therefore
demonstrably *not* equivalent, and the safe version cannot simply be preferred
on the "if performance is unchanged, prefer safe" rule.

The shipped form is therefore raw-pointer, but with the unsafe confined to one
`#[inline]` function whose contract is stated as a loop invariant, and with a
**`validate_nibble_outputs` prevalidation pass on the safe entry point** so the
pointer derivations rest on checked dimensional arithmetic rather than on a
caller's good behaviour. F is 1.036x faster than B; that margin is the
`#[inline]`, which lets the kernel inline back into the tile loop it was split
out of. There are no integer-to-pointer round trips in either form.

### What the tests did not cover

Before this change the tiled AVX2 path **was not executed by a single test in
its own module**. `the_kernel_tracks_the_float64_contract` drives
`k_blocks in {1,2,3}`, and the tile loop runs `k_blocks / BLOCK_TILE` iterations
with `BLOCK_TILE = 4` — so the trip count was always **zero** and every case
fell through to the scalar tail. The path had coverage only from four
`matmul_nbits` integration tests one module away. "1607 tests pass" was true and
said nothing about this loop.

Six mutations were used to check that the new tests actually constrain the
kernel, each applied to the shipped source and run against the suite:

| mutation | caught by |
|---|---|
| `srli` shift 4 -> 2 in the wide kernel | 2 tests |
| swap even/odd activation halves | 2 tests |
| load odd half at `+WIDE_GROUP` instead of `+WIDE_GROUP/2` | 2 tests |
| `groups` -> `groups - 1` (drop the last group) | tiled-path oracle |
| `tiled_blocks + 1` (tail loop starts one block late) | 3 tests |
| delete the `validate_nibble_outputs` call | fail-before-unsafe test |

Two notes on honesty here. The first `tiles - 1` mutant used was **equivalent** —
reducing the tile count only migrates work to the scalar tail and the result
stays correct — so it was replaced with `tiled_blocks + 1`, which is not.
And the fail-before-unsafe test initially asserted only `is_err()`, which the
`novalidate` mutant survived: the malformed inputs also trip an incidental slice
bounds check *after* the pointers are formed, so `is_err()` proved nothing about
ordering. It now asserts the **panic message** names the validator.

### Memory-safety validation, and what Miri did and did not cover

`cargo +nightly miri test -p onnx-runtime-ep-cpu --lib int4_nibble` with
`-Zmiri-strict-provenance` passes all 13 tests. That statement is worthless
without the following scope check, which was run rather than assumed:

**Miri reports `is_x86_feature_detected!("avx2") == false`.** A default Miri run
therefore takes the generic path and the two vector tests early-return — the
unsafe code under review is never executed, and the run is vacuous. Building
with `RUSTFLAGS="-C target-feature=+avx2"` makes the detection macro fold to
`true` at compile time and Miri then interprets the AVX2 intrinsics directly.

That the AVX2 path is genuinely covered was then *proved*, not assumed, by
injecting faults and confirming Miri catches them:

| injected fault | Miri result |
|---|---|
| `for index in 0..groups + 1` in `nibble_block_acc_avx2_wide_raw` | UB: "attempting to access 16 bytes, but got `alloc…+0x10` which is at or beyond the end of the allocation of size 16" |
| `activation.values.as_ptr().add(block * block_size + 8)` in the tile loop | UB: "attempting to access 32 bytes … only 16 bytes from the end of the allocation" |

Both raw-pointer derivations in the change — the callee's and the caller's — are
therefore inside Miri's coverage under strict provenance, and both are clean as
shipped. No sanitizer beyond Miri was available on this host.

### Where the ceiling actually binds

The original draft placed the boundary at t=16 ("1.6x at 1-8 threads,
1.065x at 16") and read that as the kernel handing off to the memory system and
the parallel runtime. The re-measurement moves the boundary **one octave
earlier and makes it non-monotonic**: on current main, 1.686x / 1.640x at
t=1/t=4, then a **wash at t=8 (1.016x)**, then back up to **1.242x at t=16**.

The dip at t=8 is not explained by this data and is recorded as open. On the
older base it read 1.141x with a superlinear baseline step from t=4 to t=8
(7.963 -> 3.542 ms, **2.25x from 2x the cores**) against 1.54x for the patched
arm; on current main the baseline improved further at that cell (to 3.264 ms)
and the gap closed entirely. A working-set/aggregation effect — 8 cores span
more of one CCX's 32 MiB L3 — is the natural shape for a superlinear baseline
step, but with **no PMU on this host** that is a hypothesis, not a finding, and
it is left labelled as one.

What does survive from the original reading is the direction: **the kernel is no
longer the sole binding constraint at full width.** It is simply not true that
the handoff happens cleanly at 16 threads, and the 1.6x headline does not
survive past 4.

## The recommendation

1. **Do not build lever (a).** 0.94x DRAM-resident, 0.76x at full width. Its
   positive number is an L3-residency artifact of small benchmark shapes.
   **Retired** — see
   [`2026-08-21-int4-acc4-ntile-design.md`](2026-08-21-int4-acc4-ntile-design.md),
   which is marked retired rather than deleted so the analytic bound is not
   re-derived.
2. **Do not build lever (b) for throughput.** 0.96x/1.10x. The traffic
   arithmetic is correct and the kernel is simply not traffic-limited where it
   would help. If bf16 scales are wanted for *footprint*, that is a separate
   and defensible case — but it should not be sold as speed. **Retired.**
3. **Take the per-block bookkeeping instead.** On current main: **1.686x at
   t=1, 1.640x at t=4, a wash (1.016x) at t=8, 1.242x at t=16, ~1.42x
   multi-session**; across block sizes at t=1, **1.380x at block 64, 1.091x at
   block 128, and nothing (1.028x) at block 16**. Bit-identical, all gates
   green. Shipped in #1628.
4. **Do not tune this kernel further at block 16, or at t=8, on the strength of
   this result** — block 16 provably never enters the wide path, and at t=8 the
   decode loop is no longer purely kernel-bound. Block 64 *does* benefit and an
   earlier revision of this document wrongly said it did not, because it had
   only been measured at t=8.
5. **Do not quote a single headline multiplier for this change.** Four of the
   nine rows in the first draft of this document did not reproduce, in both
   directions, and the multi-session rows moved by 0.17x once the statistic was
   corrected. Quote the cell.

## Reproducing

```bash
# real-kernel A/B (arms A0/A1 identical = A/A control; discard if >2% apart)
PROBE_ACCURACY=4 PROBE_BLOCK=32 PROBE_SESSIONS=1 PROBE_TOKENS=60 PROBE_LAYERS=1 \
  ONNX_GENAI_CPU_DECODE_THREADS=8 \
  cargo bench -p onnx-runtime-ep-cpu --bench int4_decode_loop_ab

# the ORT side of the gap table, same shapes / block size / accuracy_level,
# one Run == one decode token
python3 crates/onnx-runtime-ep-cpu/benches/ort_matmulnbits_baseline.py \
  --threads 8 --block 32 --accuracy 4 --tokens 48 --reps 3
```

Pin to even CPUs only (`taskset -c 0,2,4,...`); interleave arms within a cycle
(A0, B, A1 per repetition, never arm-at-a-time); reject any repetition set whose
A/A arms disagree by more than 2%. Do not report a single sequential timing on
this host.

> **Superseded in part by §27 of the assignment ledger (#1712).** Both arms of
> this A/B were later found not to be computing the same statistic, and the ORT
> baseline invoked above no longer prints `steady_median_ms=`: `run()` now routes
> every session count through `run_concurrent` and reports the **median**
> aggregate throughput, because reporting `min`/`max` over repetitions on one
> arm while the other ran single-shot was one of the four biases §27 documents.
> Requirement 1 below ("use the minimum over >= 6 repetitions") is therefore
> retained only as the historical record of this document's protocol; it is
> **not** the current one. Prefer `benches/acc0_gap_matrix.py`, which enforces
> the unified definition, per-cell A/A, and a quiet-host gate.

Five protocol requirements were learned the hard way during the re-measurement
and are not optional:

1. **Use the minimum over >= 6 repetitions for single-session latency**, not the
   median. Interference on this host is one-sided, so the minimum estimates the
   uncontended cost; but 3 repetitions is not enough — a 3-rep t=16 run produced
   an apparent **0.815x regression** that 6 reps showed to be **1.227x**.
2. **Use aggregate throughput (column 4), not pooled median latency (column 2),
   for any `PROBE_SESSIONS > 1` row.** Column 2 mixes contended and uncontended
   tokens depending on when each session exits; it produced a 52.7% A/A.
3. **Lengthen the window when estimators disagree in direction.** t=16 at 60
   tokens gave min 1.234x and median 0.730x; at 240 tokens it gives 1.245x and
   1.412x. Do not pick the flattering statistic — extend the run.
4. **Verify that a knob moves what it claims to.**
   The t=2 row in the first draft of this document read *identical* to `=1`
   (23.529 vs 23.527 ms/token) and was dropped rather than restated, and the
   same identity was reproduced during #1712 (23.6 vs 23.6 tok/s, pinned, qwen,
   acc0). Dropping the row was right; the **explanation** first offered for it
   was not.

   **Correction (2026-08-22): the knob is not inert, and the timings above are
   the artifact.** Under a controlled re-measurement on a quiet host — one
   process per launch, arms interleaved, per-rep load guard, over-guard cells
   discarded — `=2` runs at **20.447 ms/token against 40.039 for `=1`, a 1.96x
   speedup**, with an **A/A null of 0.6%**. Per-thread attribution shows both
   `=2` workers **99% busy** with 62 voluntary context switches over six
   seconds, which is the opposite of parked. The "71% of a single core" figure
   came from `/usr/bin/time`'s `Percent of CPU`, which is `(user+sys)/wall` and
   therefore wall-derived — it degrades under contention exactly like the wall
   time it was being used to corroborate. See
   [2026-08-22-decode-width-scaling.md](2026-08-22-decode-width-scaling.md).

   The protocol point stands and is strengthened: verify the knob *structurally*
   rather than by timing. `w` in must give `w` SPMD threads out, which is a
   categorical check valid even on a loaded host. Two configurations where this
   knob really is vacuous — in-process width sweeps (the pool is a process-wide
   `OnceLock`) and `THREADS=N` on an exactly-N-CPU cpuset — are documented there.
5. **Run `PROBE_ACCURACY=0` as a null control** for any claim about this kernel.
   It must read 1.000x; if it does not, the harness is measuring something other
   than the route under test.

Block size 16 still cannot be measured with this microbenchmark (see
[Method](#method)).
