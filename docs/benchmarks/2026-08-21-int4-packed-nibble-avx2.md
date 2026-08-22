# Packed-nibble int4 x int16 AVX2 decode kernel (`accuracy_level = 4`)

**Verdict: a win, merged.** Consuming the ONNX int4 weight at its wire density of
0.5 B/weight — instead of expanding every nibble to a whole `i8` — is **1.2x to
2.4x** faster than the route it replaces, on every shape, block size and thread
count measured, with no losses. Against ONNX Runtime the same cells go from
**~5.8-8.1x behind to ~3.3-5.6x behind**. The remaining gap is real and is not
claimed to be closed.

Most of the win is not the byte saving. Two thirds of it came from removing
**per-block overhead** that the first working version paid and the byte saving
alone could not overcome — see [What actually paid](#what-actually-paid), which
is the transferable part of this result.

Host: AMD EPYC 9V74, 32 vCPU (16 cores x 2 SMT), AVX2 + FMA + F16C, **no
AVX-512, no VNNI, no AMX**. L1d 32 KiB/core, L2 1 MiB/core, L3 64 MiB shared,
75.8 GB/s DRAM. Shared machine; `perf` unavailable.

> **Correction (2026-08-21, from the execution-regime study).** Two of the host
> figures above are wrong and are corrected in
> [`2026-08-21-int4-acc4-execution-regime.md`](2026-08-21-int4-acc4-execution-regime.md):
> **L3 is 32 MiB per CCX, not 64 MiB shared** (two 16-vCPU CCXs, so no single
> core sees 64 MiB), and **75.8 GB/s is not an achievable figure on this host**
> — measured ceilings are 31-36 GB/s within one CCX and ~56.6 GB/s (peak 72.5)
> across both. **SMT siblings are adjacent pairs**, so the physical-core set is
> the even CPUs; any experiment here that pinned to `0..N` used `N/2` cores with
> both siblings loaded. See [the affected
> paragraph](#why-this-contradicts-the-2026-08-20-rejection) for what this changes.

## The opportunity

On x86-64 the existing int4-direct decode route requires VNNI, `block_size = 32`
and no zero points (`DotKernel::supports_int4_direct`). This host has no VNNI,
so a `bits = 4, accuracy_level = 4` node fell through to `prepack_int8_weight`
-> `int8_matmul`, which **expands every 4-bit weight into a whole `i8` byte**.

Decode reads each weight exactly once and reuses nothing, so that expansion
doubles the only stream that matters. The wire layout is already what a
`vpmaddwd` kernel wants; nothing needed unpacking ahead of time.

## The arithmetic

Per `k` block, with `a` the int8-quantized activation, `w` the raw unsigned
nibble and `z` the block zero point:

```
sum_k a_k * (w_k - z)  =  (sum_k a_k * w_k)  -  z * (sum_k a_k)
```

The right-hand term is **activation-only**. It is computed once per block and
reused across all `N` outputs, so the inner loop never subtracts a zero point
per element, and an absent zero-point tensor costs nothing (ONNX's default is
8 — and a *signed* int4 weight simply is the unsigned nibble with `z = 8`, which
is how signed and unsigned interpretation share one path with no dead branch).

`vpmaddwd` takes `i16` lanes and produces `i32` lanes: exact, never saturating.
With `|a| <= 128` and `w <= 15`, a product is `<= 1920`, a pair `<= 3840`, and a
`block_size = 128` block sums 64 pairs — under `2^18`. There is no `i32`
overflow path at any block size the operator admits.

**Layout.** Packed byte `j` holds `k = 2j` in its low nibble and `k = 2j+1` in
its high nibble. One `cvtepu8_epi16` plus `and 0x0F` yields a group's *even* `k`;
a `srli 4` yields the *odd*. Rather than shuffle the weights back into `k` order
once per output row, the **activation is deinterleaved once per row** into
`[even | odd]` halves. The block dot sums over the whole block, so permuting both
sides identically cannot change the result.

Group size is `min(32, block_size)`, and a block size is a power of two `>= 16`,
so a block is always a whole number of groups — the inner loop has no tail. A
short *final* `k` block needs no masking either: the activation is zero-padded to
`padded_k`, so padding contributes `0 * w = 0` to the dot and `0` to the block
sum.

## What actually paid

The first working version was **1-5%** faster than the expanded-`i8` route —
essentially nothing, and far short of what halving the weight stream predicts.
`ONNX_GENAI_PROFILE_OPS=1` put `MatMulNBits` at **99.95%** of the run, so there
was no Amdahl dilution to blame: the kernel itself had barely moved.

The diagnostic was a block-size sweep at `t = 1` on one geometry
(`llama3_8b_mlp`, `k = 4096`, `n = 14336`). **The weight bytes are identical at
every block size** — only the number of blocks changes:

| `block_size` | blocks | v1 ms | v2 ms | v3 ms |
| --- | --- | --- | --- | --- |
| 16 | 3.67 M | 22.653 | 19.780 | 11.903 |
| 32 | 1.835 M | 12.584 | 10.388 | 6.216 |
| 64 | 0.917 M | 7.097 | 6.016 | 3.864 |
| 128 | 0.459 M | 4.371 | 3.765 | 2.767 |

Time tracked `1/block_size` almost exactly. Fitting `time = blocks*A +
weights*B` from the two ends:

| version | per-block `A` | inner loop `B` |
| --- | --- | --- |
| v1 per-block feature probe | 17.1 cycles/block | 11.1 weights/cycle |
| v2 probe hoisted out of the loop | 15.0 cycles/block | 13.3 weights/cycle |
| v3 four-block tile | **8.5 cycles/block** | **13.4 weights/cycle** |

The uop budget predicts 12.8 weights/cycle (10 uops per 32 weights at 4
uops/cycle), so **the vector inner loop was always at budget**. The entire
shortfall was per-block cost, and at `block_size = 32` — the common case — a
block is only 32 weights, so 17 cycles of overhead sat on top of ~2.5 cycles of
arithmetic.

Two changes removed two thirds of it:

1. **`is_x86_feature_detected!` was being evaluated per block.** It caches in an
   atomic, so it is cheap in isolation, but it also sits on a non-`target_feature`
   call boundary that stops the AVX2 body from inlining, forcing the accumulator
   through memory every 32 weights. Hoisting the probe to the row driver and
   marking that driver `#[target_feature(enable = "avx2")]` let the block dot
   inline. (v1 -> v2)
2. **Four blocks now share one reduction and one vector tail.** A block dot ended
   in a full horizontal reduction, then a scalar tail: zero-point correction,
   `i32`->`f32` convert, two multiplies and an add. Four consecutive blocks are
   contiguous in *every* array that tail reads — weight scales, activation
   scales, block sums, and the zero-point nibbles (four nibbles are two adjacent
   bytes, and a tile always starts on an even block) — so four blocks reduce
   through one `hadd` tree and the tail runs once, on four lanes. (v2 -> v3)

The lesson worth carrying: **at decode block sizes the per-block tail, not the
multiply-accumulate, is the kernel.** A byte-density change is invisible until
the tail is out of the way.

## Why this contradicts the 2026-08-20 rejection

[`2026-08-20-int4-nibble-i16-negative.md`](2026-08-20-int4-nibble-i16-negative.md)
built this same idea one day earlier, measured it **1.5x-2.2x slower** in every
cell, and concluded "the idea is dead on AVX2". That document is superseded on
its verdict. Both measurements are real; the kernels are not the same kernel.

**Two of its three load-bearing claims survive, one does not.**

*Survives:* `madd_epi16` retires 16 products per instruction where `maddubs_epi16`
retires 32, and int16 activations need two loads per 32 weights where int8 needs
one. Both are true and this kernel pays both. They are not, however, decisive —
they cost ~4 uops per 32 weights against a per-block tail that cost ~17 cycles.

*Survives:* its own note that the rejected kernel's inner loop was ~14 uops per
32 weights. That is where the two designs diverge. It spent `and` + `srli` + `and`
+ two `unpack` + two `cvtepu8_epi16` **restoring `k` order in the weights**. This
kernel never restores it: one `cvtepu8_epi16` on the packed bytes plus `and` and
`srli` yields the even and odd `k` of a group in place, and the **activation** is
deinterleaved to match, once per row, amortized over all `N` outputs. That is
14 uops -> 10, before any of the per-block work discussed above.

*Does not survive:* **"the incumbent is already at the memory roofline."** That
conclusion rests on the `acc4_int8` arm reaching 98-102% of the 75.8 GB/s DRAM
figure at block 128. The direct refutation is that this kernel is **2.37x faster
than that arm** on that cell; nothing can be 2.37x faster than a genuinely
bandwidth-saturated kernel reading half its bytes.

> **Correction (2026-08-21).** The *reasoning* originally given for this
> refutation was wrong, even though the refutation itself stands. It argued that
> `llama3_8b_mlp` expanded to `i8` is 58.7 MB against "a **64 MiB L3**", so the
> weight was L3-resident and DRAM never bound. L3 is in fact **32 MiB per CCX**,
> so a 58.7 MB working set is *not* L3-resident and does stream — that argument
> is void. What survives, and is now the supported reason, is that **75.8 GB/s
> is not an achievable denominator on this host at all**: measured ceilings are
> 31-36 GB/s within one CCX and ~56.6 GB/s across both, so an arm reported at
> "98-102% of 75.8 GB/s" was being scored against a number ~2x its real ceiling.
> The warning the original document gave itself — "a roofline percentage means
> nothing until you check whether the denominator is the binding constraint" —
> applied to the *denominator*, not to residency.

Its rejected mitigation is also worth separating from this one. It tried folding
each activation group's `i32` partial into an `f32x8` accumulator *inside* a
block, which scales eight lanes separately and so **gives up the integer dot's
exactness** — correctly judged not worth 1-3 pp. The four-block tile here does not
touch the integer dot: each block's `i32` sum is still exact and still reduced
exactly. Only the **cross-block `f32` accumulation** is reassociated into four
lanes, which the contract already permits, since the bound was always a tolerance
against the fp32 dequantized reference and is validated against an `f64` oracle.

The transferable lesson is not that the earlier measurement was careless — it was
more careful than most — but that **"instruction-bound" was diagnosed without
separating per-block cost from per-weight cost.** Fitting `blocks*A + weights*B`
is what distinguishes "this loop cannot issue fast enough" from "this loop is
fine and its tail is 6x its body", and those two have opposite fixes.

## Method

Two prebuilt `bench_generic` binaries, alternated: `final` is the branch, `base`
is the identical tree with the route's branch condition dead, so `base` is
exactly today's expanded-`i8` behaviour. Models from `scripts/ort_ab/gen_gemm.py`
(llama/qwen projection geometries). All native-vs-native numbers are
`--native-only`: ORT's intra-op pool spin-waits and steals cores from a
co-resident native arm.

**Three independent null controls**, all disclosed:

| control | what it is | reading |
| --- | --- | --- |
| `null` arm in `ab.py` | the `final` binary run again under a second name | -21.9% .. +23.8% per cell |
| `accuracy_level = 0` cells | code-identical in both arms (acc0 cannot reach this route) | -4.5% .. +5.5% |
| `m = 128` with the gate at 64 | route gated off, both arms identical | +0.1% |

The `null` arm's per-cell estimate is itself noisy and on unlucky cells is worse
than the effect it is supposed to bound. **The honest floor for this host at
`t = 8` is the `accuracy_level = 0` block: about +/-5%.** Every result below is
far outside it, but that is the number to judge them against, not the +/-0.1%
the null arm reports on lucky cells.

## Results

### Block size x shape, `m = 1`, `t = 8`, ratio = base/final (>1 favours the kernel)

| shape | bs=16 | bs=32 | bs=64 | bs=128 |
| --- | --- | --- | --- | --- |
| `llama3_8b_mlp` | 2.38x | 1.89x | 1.93x | 2.37x |
| `llama3_8b_qkv` | 1.92x | 1.85x | 1.55x | 1.36x |
| `qwen3_0p6b_mlp` | 1.27x | 1.82x | 1.54x | 1.32x |
| `qwen3_0p6b_qkv` | 1.61x | 1.68x | 1.41x | 1.21x |
| `qwen3_8b_square` | 1.80x | 1.78x | 1.42x | 1.40x |

The bs=16/64/128 columns are direct ms measurements; the bs=32 column is derived
from the `ab.py` matrix (3 trials, interleaved arms, null control) and is the
only column measured under that harness.

No cell in the matrix is a loss. The smallest win (1.21x) is the smallest
geometry at the widest block — the case with the least weight traffic to save.

### Row crossover, `qwen3_8b_square`, bs=32, `t = 8`

Its `m = 1` row reads **1.66x** where the table above reads **1.78x** for the same
cell. Same shape, two harnesses: the block-size table is an `ab.py` run with the
two arms interleaved rep-by-rep, this one is a direct timing of each binary in
turn. A 7% spread between harnesses on one cell is above the +-5.5% single-cell
floor, so neither number is quotable to two digits on its own -- which is why the
verdict rests on 20 cells with no losing one, not on any single cell. Both
harnesses agree on the sign and on roughly-1.7x for this shape.


| `m` | final ms | base ms | ratio |
| --- | --- | --- | --- |
| 1 | 0.248 | 0.412 | 1.66x |
| 2 | 0.386 | 0.753 | 1.95x |
| 4 | 0.751 | 1.457 | 1.94x |
| 8 | 1.407 | 2.881 | 2.05x |
| 16 | 2.790 | 5.704 | 2.04x |
| 32 | 5.588 | 11.755 | 2.10x |
| 64 | 11.099 | 22.566 | 2.03x |
| 128 | 45.064 | 45.010 | 1.00x (gated off — null control) |

`INT4_NIBBLE_MAX_ROWS` was **swept, not asserted**. The gate ships at **64**: the
largest `m` measured to win. Past it there is no measurement, so the route stays
off rather than being extrapolated. The driver parallelizes over rows with no
weight reuse, so a real prefill geometry wants a blocked kernel, not a wider gate.

### Against ONNX Runtime, bs=32, `m = 1`, 8 threads

| shape | base native/ort | final native/ort |
| --- | --- | --- |
| `llama3_8b_mlp` | 7.47 | 3.27 |
| `llama3_8b_qkv` | 7.94 | 4.60 |
| `qwen3_0p6b_mlp` | 6.74 | 4.81 |
| `qwen3_0p6b_qkv` | 5.85 | 5.60 |
| `qwen3_8b_square` | 8.12 | 4.53 |

These are **paired** runs, so the native arm is depressed by ORT's spinning pool
(the same cell reads 0.875 ms native-only versus 1.653 ms paired). Both arms
suffer it equally, so the *improvement* is sound; the absolute ratio is
pessimistic. Taking `final`'s native-only time against ORT's undisturbed time
puts `llama3_8b_mlp` at ~1.7x rather than 3.27x — quoted only to bound the error,
not as a measurement.

**We are still behind ORT on every one of these cells.** ORT's MLAS SQNBit
CompInt8 kernel is the target and has not been reached.

### Two cells re-measured

`qwen3_0p6b_mlp` `m=16` and `qwen3_8b_square` `m=8` first read -1.9% and +2.1%,
contradicting every other cell of their own shape. Re-measured, three reps each:

| cell | final ms | base ms | median ratio |
| --- | --- | --- | --- |
| `qwen3_0p6b_mlp_t16` | 1.473 / 1.459 / **2.318** | 2.791 / 2.859 / 3.000 | 1.89x |
| `qwen3_8b_square_t8` | 1.465 / 1.433 / **2.552** | 2.916 / 3.284 / 2.870 | 1.99x |

The bolded third rep of each is a contended sample on a shared host — disclosed
rather than dropped, and the medians are reported over all three.

## Correctness

The kernel owes a tolerance, not an equality: it quantizes the activation
*signed* (matching the existing VNNI int4 route), whereas the expanded-`i8` route
uses the `+128` unsigned-offset spelling with a `-128 * block_sum` correction.
The two are not expected to agree bit for bit, so the bound is against the **fp32
dequantized reference**, never against the other route.

- Vector path equals its readable scalar reference on every nibble at every block
  size, and **exhaustively over all 256 packed byte values** crossed with
  activation extremes.
- Tracked against an **f64 oracle** at block sizes 16/32/64/128, with and without
  a zero-point tensor, including short final blocks and `k` not a multiple of the
  block size.
- Zero points: absent means 8; packing is two per byte along `k` blocks with each
  output's run padded to a whole byte; asserted per block and per output.
- Overflow headroom asserted at the widest block at full range.
- `accuracy_level` `None/0/1/2/3` assert the **route counter is zero** — a
  tolerance would pass for a route that merely happened to be accurate on one
  fixture.
- 1/2/4 concurrent sessions on one constant weight: every session sees the same
  prepack pointer, every run takes the route, and no per-`Run` repack occurs.

**Every one of these was mutation-checked.** Seven mutations — widening the
`accuracy_level == 4` gate, disabling the branch, swapping `and`/`srli`, breaking
the deinterleave stride, dropping `is_power_of_two`, dropping the zero-point row
`div_ceil`, and ignoring zero points in the driver — each fails exactly the test
that claims it. The first run of that battery reported all seven as *undetected*;
the battery itself was wrong (`--exact` with a bare test name runs zero tests and
exits 0). The harness bug is recorded here because the failure mode is silent.

The zero-point mutation is the reason the envelope test is trustworthy:
`run_decode_case` passes `None` for the zero-point shape, so an asymmetric
fixture silently ran against the default of 8 while its reference used the real
zero points. A decode helper that wires the tensor was added; without it that
test could not detect a kernel that ignored zero points at all.

## Gating

The route requires `bits == 4 && accuracy_level == 4 && !weight_prepacked &&
group_indices.is_none() && m <= 64 && !dot_kernel.uses_vnni_int4_direct() &&
int4_nibble::supported(block_size)` — the last being a power-of-two block `>= 16`
and runtime AVX2.

`accuracy_level == 4` is the reduced-precision-activation contract and is
enforced by the route counter, not by a tolerance. `supported()` is deliberately
**not** a function of `accuracy_level`: the caller owns that gate, and a second
copy could be satisfied while the first was not.

The **VNNI guard is a deliberate refusal to extrapolate.** On a host with
`vpdpbusd` the expanded-`i8` route gets a 1-uop-per-32-MAC dot, so trading
arithmetic for bytes may not pay — and this host cannot measure it. Such hosts
keep today's behaviour unchanged.

Under `--features mlas` with no native int8 dot, MLAS SQNBit CompInt8 takes
accuracy-4 decode for a constant weight *before* this route. That is pre-existing
precedence and is left alone; the shipped artifact is MLAS-free
(`default_artifacts_are_mlas_free`), so production takes this route.

## What this does and does not establish

**Does:** on AVX2-without-VNNI x86-64, consuming the packed nibble directly beats
expanding to `i8`, by 1.2-2.4x across four block sizes, five geometries and
`m = 1..64`, measured against a controlled baseline with three null controls.

**Does not:**

- **Anything about VNNI or aarch64 hosts.** Both are gated out. Untested, not
  claimed.
- **That 8.5 cycles/block is a floor.** It is 3.4x the inner loop's own cost.
  Widening the tile past 4, or tiling over `N` so one activation block serves
  several outputs, was not tried. The `N` tile is the more promising one and is
  harder: weights, scales and zero points are all `N`-major, so it needs a
  block-major prepack of the scale and zero-point arrays.
- **That this closes the ORT gap.** It roughly halves it. ~3.3-5.6x remains.
- **That `m > 64` would lose.** It was not measured, which is why the gate is
  where it is.
- **Anything under Miri.** Miri cannot execute x86 SIMD intrinsics, so it is a
  no-op gate for this module — *no coverage*, not a pass.

## Reproducing

```bash
python3 scripts/ort_ab/gen_gemm.py --out models/b32a4 --block-size 32 \
    --accuracy-level 4 --tokens 1 2 4 8 16
python3 scripts/ort_ab/ab.py --arms final=./bench_final base=./bench_base \
    --models models/b32a4/*.onnx --threads 8 --trials 3 --runs 15 --warmups 5 \
    --native-only --null-control --csv out.csv
```

`base` is built by making the route's branch condition `false` in
`matmul_nbits.rs`; everything else is identical.
