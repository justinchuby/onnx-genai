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
it is **1.61x-1.63x on the real decode loop at 1-8 threads**, bit-identical
output, ~40 lines. That patch is included in this branch and is offered to
whoever owns the kernel next; see [The recommendation](#the-recommendation).

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
| scaling 1->16 cores | **7.3x** | <=1.77x (the measured bandwidth curve) |
| block sweep 16->128 at **fixed weight bytes** | **3.73x** | 1.41x (163.6 -> 115.9 MB) |
| `time = blocks*A + weights*B` fit | A = 2.850 ns/block, B = 24.15 ps/weight; per-block tail is **79%** of time at block 32 | tail ~0 |
| per-column overhead (4096x14336 vs 14336x4096: identical weights *and* identical block count, 3.5x the columns) | wide-n shape **2.4% faster** | n/a |

The fit independently reproduces #1619's own model (8.5 cyc/block,
13.4 weights/cyc) at an assumed ~3 GHz.

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
control. **Every row below passed its A/A gate** (worst 1.53%); three further
runs failed theirs and were discarded rather than reported.

| threads | sessions | block | baseline ms/token | patched | speedup |
|---|---|---|---|---|---|
| 1 | 1 | 32 | 22.963 | 14.126 | **1.626x** |
| 2 | 1 | 32 | 23.051 | 14.531 | **1.586x** |
| 4 | 1 | 32 | 11.545 | 7.071 | **1.633x** |
| 8 | 1 | 32 | 5.880 | 3.655 | **1.609x** |
| 16 | 1 | 32 | 4.622 | 4.339 | 1.065x |
| 8 | 1 | 64 | 3.715 | 3.064 | 1.212x |
| 8 | 1 | 128 | 2.948 | 2.858 | 1.031x |
| 8 | 2 | 32 | 5.904 | 3.945 | **1.497x** |
| 8 | 4 | 32 | 5.964 | 5.247 | 1.137x |

Output is bit-identical; `cargo test -p onnx-runtime-ep-cpu --release` passes
1607 tests, `cargo fmt --check` and `cargo clippy --all-targets -D warnings`
are clean.

Two implementation details cost more than they saved and are recorded so they
are not re-tried:

- **Const-generic group counts** (`match wide_groups { 1 => ...::<1>, ... }`)
  put a `match` inside the block loop and gave **0.945x** — worse than the
  baseline. The trip count is not the problem.
- The pointer computations must be hoisted **above** the `if wide` branch. With
  them inside, block 64 regressed to 0.876x; hoisted, it is 1.212x.

### The t=16 result is the interesting one

The patch is worth 1.6x at 1-8 threads and **1.065x at 16**. That is not a
disappointment, it is a boundary: after the fix, t=8 (3.655) is already within
**1.09x** of t=16 (4.339 baseline-paired, 3.51 on a quiet host), whereas before
the fix t=8 -> t=16 still gained 1.27x. **Removing the kernel overhead moves the
binding constraint off the kernel and onto the memory system and the parallel
runtime.** This is the same ceiling as the single-session underutilization in
the scheduler lane, reached from the other side, and it means further *kernel*
micro-optimization at full width has little left to win.

## The recommendation

1. **Do not build lever (a).** 0.94x DRAM-resident, 0.76x at full width. Its
   positive number is an L3-residency artifact of small benchmark shapes.
2. **Do not build lever (b) for throughput.** 0.96x/1.10x. The traffic
   arithmetic is correct and the kernel is simply not traffic-limited where it
   would help. If bf16 scales are wanted for *footprint*, that is a separate
   and defensible case — but it should not be sold as speed.
3. **Take the per-block bookkeeping instead.** 1.61x-1.63x at 1-8 threads,
   ~40 lines, bit-identical, all gates green. The patch is on this branch.
4. **Do not tune the kernel further at t=16 without first moving the runtime
   ceiling** — at full width the decode loop is no longer kernel-bound.

## Reproducing

```bash
# real-kernel A/B (arms A0/A1 identical = A/A control; discard if >2% apart)
PROBE_ACCURACY=4 PROBE_BLOCK=32 PROBE_SESSIONS=1 PROBE_TOKENS=10 PROBE_LAYERS=1 \
  ONNX_GENAI_CPU_DECODE_THREADS=8 \
  cargo bench -p onnx-runtime-ep-cpu --bench int4_decode_loop_ab
```

Pin to even CPUs only (`taskset -c 0,2,4,...`); interleave arms within a cycle
and take medians across cycles; reject any cycle set whose A/A arms disagree by
more than 2%. Do not report a single sequential timing on this host.
