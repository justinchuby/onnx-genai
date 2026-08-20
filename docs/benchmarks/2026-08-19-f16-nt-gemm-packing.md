# f16 `Gemm` `transB = 1` prefill: the transposed-`B` packer was a scalar strided gather

**Date:** 2026-08-19
**Host:** AMD EPYC 9V74, 32 vCPU (16 physical cores x 2 SMT), L1d 32 KiB/core, L2 1 MiB/core,
L3 2 x 32 MiB. AVX2 + FMA + F16C; **no AVX-512, no VNNI**. Bundled ONNX Runtime 1.27.0.
**Build:** default pure-native, `--features bench-native`. **No `mlas` feature**, no ORT CPU
fallback, no deferral.
**Base commit:** `e13460af6`.

## The claim being closed

`docs/benchmarks/2026-08-19-f16-gemm-transb-decode.md` (not yet on `main`; it lands with #1417)
closed the `M = 1` half of this cell and left the prefill half explicitly open:

> **`transB = 1` f16 *prefill* is still badly broken** [...] at `M = 128` it measures 156 ms
> against ORT's 39 ms at 1 thread (4.04x p50) and 101 ms against 6.0 ms at 8 (16.95x p50).
> [...] Fixing it needs a packed **NT** half GEMM.

That reproduces on `e13460af6`: **173.96 ms vs 38.83 (4.48x) at one thread, 95.78 vs 5.99
(15.99x) at eight.** This document is that fix.

The diagnosis in the quoted text was wrong in an instructive way. The problem was not the
*micro-kernel* — there was nothing wrong with the NT arithmetic, because there was no NT
arithmetic. It was the **packer**.

## The measurement that localised it in one step

Every explanation offered for this cell so far ("we need an NT kernel", "it is the fork/join",
"it is bandwidth") is a claim about the GEMM. None of them can be tested by a ratio against ORT,
because ORT differs from us in *every* respect at once.

So `scripts/ort_ab/gen_f16_nt.py` emits each shape **twice**:

* `nt` — `Gemm` with `transB = 1` over a `[n, k]` initializer.
* `nn` — `Gemm` with `transB = 0` over that same array physically transposed to `[k, n]`.

Same product, same numbers, same micro-kernel, same thread pool, same everything — the two
differ *only* in the `MatrixLayout` that `half_gemm::pack_b` receives. So `nn` is a true control:
whatever it costs is the cost of the GEMM, and **`nt` minus `nn` is the cost of the layout alone**.

Baseline, `K = N = 3584`, native-only `p50`, ms:

| cell | t=1 | t=8 |
|---|---:|---:|
| `nt` | 173.96 | 95.78 |
| `nn` | 110.65 | 33.76 |
| **`nt` - `nn`** | **63.3** | **62.0** |

The penalty is **~62 ms at both thread counts**. A constant that does not shrink when you add
8x the cores is not arithmetic — arithmetic parallelises. It is work whose *total* grows exactly
as fast as the pool does.

That is precisely what `gemm_block` does. `pack_b` is called **once per row-block**, and
`gemm_impl` sizes row-blocks so that their *count* scales with the thread count:
`m = 128` gives 2 blocks at `t=1` (`mc = MAX_MC = 64`) and 16 at `t=8`. So B is packed 2x
serially, or 16x across 8 workers — 2 pack-times of wall clock either way. The model predicts a
flat penalty, and a flat penalty is what the control measures.

## Root cause

`MatrixLayout::transposed(k)` is `row_stride = 1, column_stride = k`. `pack_b` had two branches,
and transposed `B` fell into the general one:

```rust
for depth in 0..panel_depth {
    if layout.column_stride == 1 {
        T::pack_contiguous(...);              // row-major: SIMD widen of a contiguous run
    } else {
        for (column, output) in destination.iter_mut().enumerate() {
            let source_index = (depth_start + depth) * layout.row_stride
                + (column_start + column) * layout.column_stride;
            *output = T::to_f32(source[source_index]);   // <-- scalar, stride k
        }
    }
}
```

The inner loop walks `column`, so consecutive iterations are `column_stride = k` elements apart:
**a separate cache line per element, and a scalar `to_f32` per element**, for all
`panel_depth * panel_columns` of them. F16C is never reached. At `k = 3584` that is a 7 KiB
stride, so it is also a fresh page every ~9 columns.

The irony is that transposed `B` is the *easy* layout to read: it stores each logical column
contiguously. The packer was walking the one axis that strides.

Packing all of `[3584, 3584]` this way costs ~31 ms, which is the ~62 ms flat penalty over the
two effective pack-times. That is ~2.4 ns/element, about 6 cycles — the right order for a
line-missing scalar gather.

## The fix

`pack_b_transposed` reads **along the stored direction**: for each column, one contiguous
`panel_depth`-long run through the same `T::pack_contiguous` the row-major path uses, so it
gets F16C on x86 and the FP16/bf16 widening on NEON, with no new intrinsics and no new
architecture gates.

Filling a `[depth][column]` panel from column-major input is a transpose, so the stores are
strided instead. Columns are therefore processed `TRANSPOSED_PACK_GROUP` at a time, making each
store `GROUP` adjacent `f32` wide.

### Choosing the group width

`GROUP` trades store width against scratch: `GROUP * KC * 4` bytes of scratch share a 32 KiB L1d
with the 32 KiB packed panel being written. Bigger groups write wider but evict more.
`f16gemm_nt_qwen3_8b_m128`, native-only `p50` ms, two independent runs (9 runs/3 warmups, then
15/5):

| GROUP | scratch | t=1 | t=8 | t=16 |
|---:|---:|---:|---:|---:|
| 1 | 0.5 KiB | 141.5 / 144.4 | 46.1 / 48.5 | 39.8 / 40.8 |
| **4** | **2 KiB** | **124.6 / 121.8** | **35.6 / 36.7** | **30.5 / 30.6** |
| 8 | 4 KiB | 126.3 / 123.2 | 38.4 / 38.7 | 32.0 / 33.3 |
| 16 | 8 KiB | 125.9 / 121.7 | 36.9 / 38.3 | 33.6 / 33.0 |

4 wins `t=8` and `t=16` in both runs and is within 2.9 ms at `t=1`. The informative row is
`GROUP = 1` — contiguous loads, still-scattered stores — which already captures roughly
three-quarters of the win. **The loads were the dominant cost;** blocking the stores is a real
but secondary 10-20% on top. An 8x8 SIMD register transpose would only be attacking that
secondary part, so it is not worth the architecture-specific code.

## Numerics: bit-identical

Widening `f16`/`bf16` to `f32` is exact and elementwise, so the *order* the packed panel is
filled in cannot change a single bit of it, and the micro-kernel then sees an identical panel.
This is not a tolerance argument, so the test does not use one:
`transposed_b_is_bit_identical_to_pre_transposed_row_major` asserts `to_bits()` equality between
the `transB = 1` route and the pre-transposed row-major route, across 10 shapes that straddle
`TRANSPOSED_PACK_GROUP` (4), `NR` (8), `NC` (64) and `KC` (128), both formats, and **every**
execution path including `Scalar`. End-to-end `parity=PASS` against ORT on the paired harness.

`a_layout_strided_in_both_directions_still_packs_correctly` pins the general branch, so the fast
path cannot quietly become the only column-strided route.

One behaviour genuinely does change, and it is worth stating rather than leaving to be discovered:
transposed `B` now converts through the same SIMD widening as the row-major path instead of the
scalar `to_f32`. Both are exact for every finite value, subnormal and infinity, so they can differ
only on NaN encodings — and sweeping all 65536 patterns on this host, `f16` scalar and F16C
`_mm256_cvtph_ps` agree on **every** pattern including all NaNs, so the f16 path is bit-identical
outright. The one real divergence is `bf16`, on **126** signalling NaNs: scalar `bf16::to_f32`
quiets them (`0x7f81 -> 0x7fc1_0000`) where the `<< 16` widening preserves the signalling bit. The
result is NaN either way, no numeric contract is affected, and the change makes transposed `B`
agree with the row-major path, which has always used the shift. The bit-identity test's `sin`/`cos`
data never generates NaN, so the test does not cover this; the claim it makes is the fill-order
one, which is the claim that matters for the change.

## Results

27 NT cells, three geometries x `M` x threads. Native-only `p50`, ms. `null` is the base binary
under a second name in the same invocation — this host's noise floor for that cell. 5 interleaved
trials, 21 runs / 7 warmups each.

| cell | t | base | new | **speedup** | null | ORT | base/ORT | **new/ORT** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_8b m32 | 1 | 52.76 | 38.28 | **1.38x** | 56.74 | 10.28 | 5.13 | 3.72 |
| qwen3_8b m32 | 8 | 36.62 | 16.20 | **2.26x** | 36.29 | 1.59 | 23.03 | 10.19 |
| qwen3_8b m32 | 16 | 71.76 | 31.42 | **2.28x** | 71.67 | 1.13 | 63.50 | 27.81 |
| qwen3_8b m128 | 1 | 160.96 | 124.67 | **1.29x** | 161.12 | 38.00 | 4.24 | 3.28 |
| qwen3_8b m128 | 8 | 79.86 | 39.78 | **2.01x** | 79.66 | 5.70 | 14.01 | 6.98 |
| qwen3_8b m128 | 16 | 73.73 | 34.08 | **2.16x** | 73.93 | 3.60 | 20.48 | 9.47 |
| qwen3_8b m512 | 1 | 644.87 | 500.44 | **1.29x** | 645.69 | 150.35 | 4.29 | 3.33 |
| qwen3_8b m512 | 8 | 116.15 | 76.63 | **1.52x** | 115.94 | 22.57 | 5.15 | 3.40 |
| qwen3_8b m512 | 16 | 94.23 | 55.34 | **1.70x** | 93.69 | 13.82 | 6.82 | 4.01 |
| llama3_8b m32 | 1 | 82.27 | 50.76 | **1.62x** | 82.31 | 14.00 | 5.88 | 3.63 |
| llama3_8b m32 | 8 | 57.28 | 21.95 | **2.61x** | 57.38 | 2.03 | 28.22 | 10.81 |
| llama3_8b m32 | 16 | 112.21 | 42.26 | **2.66x** | 112.61 | 1.29 | 86.98 | 32.76 |
| llama3_8b m128 | 1 | 225.72 | 164.64 | **1.37x** | 227.27 | 50.58 | 4.46 | 3.26 |
| llama3_8b m128 | 8 | 120.20 | 51.94 | **2.31x** | 120.46 | 7.35 | 16.35 | 7.07 |
| llama3_8b m128 | 16 | 114.48 | 45.23 | **2.53x** | 115.06 | 5.23 | 21.89 | 8.65 |
| llama3_8b m512 | 1 | 905.20 | 661.44 | **1.37x** | 907.01 | 199.55 | 4.54 | 3.31 |
| llama3_8b m512 | 8 | 168.10 | 101.55 | **1.66x** | 168.07 | 28.55 | 5.89 | 3.56 |
| llama3_8b m512 | 16 | 139.87 | 71.97 | **1.94x** | 140.68 | 18.55 | 7.54 | 3.88 |
| qwen3_0p6b m32 | 1 | 4.69 | 3.06 | **1.53x** | 4.70 | 0.83 | 5.65 | 3.69 |
| qwen3_0p6b m32 | 8 | 3.04 | 1.39 | **2.19x** | 3.03 | 0.26 | 11.69 | 5.35 |
| qwen3_0p6b m32 | 16 | 5.95 | 2.71 | **2.20x** | 7.11 | 0.21 | 28.33 | 12.90 |
| qwen3_0p6b m128 | 1 | 13.22 | 10.04 | **1.32x** | 13.28 | 3.15 | 4.20 | 3.19 |
| qwen3_0p6b m128 | 8 | 6.65 | 3.32 | **2.00x** | 6.65 | 0.73 | 9.11 | 4.55 |
| qwen3_0p6b m128 | 16 | 6.30 | 2.96 | **2.13x** | 6.25 | 0.56 | 11.25 | 5.29 |
| qwen3_0p6b m512 | 1 | 53.18 | 40.17 | **1.32x** | 52.98 | 12.50 | 4.25 | 3.21 |
| qwen3_0p6b m512 | 8 | 10.07 | 6.80 | **1.48x** | 10.04 | 2.79 | 3.61 | 2.44 |
| qwen3_0p6b m512 | 16 | 8.30 | 4.99 | **1.66x** | 8.20 | 2.15 | 3.86 | 2.32 |

**27 cells, 27 wins, 1.29x - 2.66x** (as opened; see the latest-main re-measurement below). 25 of the 27 nulls are within 1.6% of base. Two are not —
`qwen3_8b m32 t=1` (7.5%) and `qwen3_0p6b m32 t=16` (19.5%) — and both still win by far more than
their own noise floor, but they are the two cells to distrust.

The headline row, the one the previous document left open: **`M = 128`, `t = 8`, 14.0x -> 7.0x.**

That before/after pair is the `base` and `new` columns of the *same* invocation, which is the only
comparison this harness licenses. The 15.99x quoted at the top of this document came from the
separate reproduction run above; ORT's own measured time differs between the two sessions (5.99 vs
5.70 ms), which is exactly why cells from different invocations must not be pasted into one row.
The change is the same size either way; only the denominator moved.

### Re-measured on latest main, immediately before merge

The table above was taken when the branch was opened. `main` has moved a long way since (the half
GEMV register tile, the decode GEBP handover, the plugin thread-local change), so the whole sweep
was re-run on the merged head. Same-tree A/B: only `half_gemm.rs` is reverted for the `base` arm,
so the harness is byte-identical in both arms. Native-only `p50`, 5 interleaved trials.

| cell | t | base ms | new ms | delta | speedup | null |
|---|---:|---:|---:|---:|---:|---:|
| llama3_8b_m128 | 1 | 235.55 | 164.03 | -30.4% | 1.44x | 2.8% |
| llama3_8b_m128 | 8 | 132.66 | 47.61 | -64.1% | 2.79x | 9.2% |
| llama3_8b_m128 | 16 | 170.78 | 64.09 | -62.5% | 2.66x | 0.2% |
| llama3_8b_m32 | 1 | 83.16 | 49.35 | -40.7% | 1.68x | 0.2% |
| llama3_8b_m32 | 8 | 57.18 | 20.01 | -65.0% | 2.86x | 0.0% |
| llama3_8b_m32 | 16 | 127.50 | 56.10 | -56.0% | 2.27x | 0.6% |
| llama3_8b_m512 | 1 | 904.50 | 640.38 | -29.2% | 1.41x | 0.3% |
| llama3_8b_m512 | 8 | 169.65 | 98.41 | -42.0% | 1.72x | 0.4% |
| llama3_8b_m512 | 16 | 203.15 | 98.97 | -51.3% | 2.05x | 0.0% |
| qwen3_0p6b_m128 | 1 | 13.27 | 9.75 | -26.5% | 1.36x | 0.1% |
| qwen3_0p6b_m128 | 8 | 6.64 | 3.11 | -53.2% | 2.14x | 0.0% |
| qwen3_0p6b_m128 | 16 | 8.34 | 4.07 | -51.3% | 2.05x | 0.8% |
| qwen3_0p6b_m32 | 1 | 4.75 | 2.96 | -37.6% | 1.60x | 0.1% |
| qwen3_0p6b_m32 | 8 | 4.08 | 2.41 | -40.9% | 1.69x | 46.3% **null unusable** |
| qwen3_0p6b_m32 | 16 | 7.72 | 3.57 | -53.8% | 2.16x | 1.0% |
| qwen3_0p6b_m512 | 1 | 53.17 | 39.39 | -25.9% | 1.35x | 0.1% |
| qwen3_0p6b_m512 | 8 | 10.09 | 6.53 | -35.3% | 1.55x | 0.3% |
| qwen3_0p6b_m512 | 16 | 11.85 | 6.52 | -45.0% | 1.82x | 3.8% |
| qwen3_8b_m128 | 1 | 160.56 | 121.32 | -24.4% | 1.32x | 0.2% |
| qwen3_8b_m128 | 8 | 79.82 | 35.78 | -55.2% | 2.23x | 0.1% |
| qwen3_8b_m128 | 16 | 94.52 | 43.90 | -53.6% | 2.15x | 0.0% |
| qwen3_8b_m32 | 1 | 57.10 | 36.65 | -35.8% | 1.56x | 0.5% |
| qwen3_8b_m32 | 8 | 36.53 | 14.69 | -59.8% | 2.49x | 0.5% |
| qwen3_8b_m32 | 16 | 106.15 | 41.58 | -60.8% | 2.55x | 0.6% |
| qwen3_8b_m512 | 1 | 654.63 | 496.33 | -24.2% | 1.32x | 0.5% |
| qwen3_8b_m512 | 8 | 117.02 | 75.70 | -35.3% | 1.55x | 0.0% |
| qwen3_8b_m512 | 16 | 137.19 | 73.66 | -46.3% | 1.86x | 0.1% |

**26 of 27 prefill cells improve beyond their own noise floor, -24.2% to -65.0% (1.32x - 2.86x).**
That is *stronger* than the 1.29x - 2.66x this PR originally claimed. The 27th cell,
`qwen3_0p6b_m32 t=8`, reads -40.9% but its null moved 46.3%, so it is not claimed at all.

#### The `M = 1` cells, and a measurement artifact worth recording

The sweep also covers `M = 1`, which this PR does not claim: at `M = 1` with `transB = 1` the
`Gemm` path takes `gemv_f16_nk` and never reaches `pack_b`, so no code this PR touches runs.

At 5 trials x 15 runs three `M = 1` cells nevertheless read as regressions *above* their null —
`qwen3_0p6b_m1 t=8` +10.20% against a 2.04% null, `t=16` +6.10% against 1.22%, and `qwen3_8b_m1
t=1` +1.85% against 0.12%. Those are the shapes that run in tens of microseconds, where a null that
tight is not credible. Re-measured at **11 trials x 40 runs**:

| cell | t | base ms | new ms | delta | null | verdict |
|---|---:|---:|---:|---:|---:|---|
| qwen3_0p6b_m1 | 1 | 0.095 | 0.095 | +0.00% | 0.00% | within noise |
| qwen3_0p6b_m1 | 8 | 0.044 | 0.047 | +6.82% | 11.36% | within noise |
| qwen3_0p6b_m1 | 16 | 0.095 | 0.092 | -3.16% | 11.58% | within noise |
| qwen3_8b_m1 | 1 | 0.808 | 0.813 | +0.62% | 0.74% | within noise |
| qwen3_8b_m1 | 8 | 0.212 | 0.212 | +0.00% | 0.00% | within noise |
| qwen3_8b_m1 | 16 | 0.272 | 0.266 | -2.21% | 5.15% | within noise |

All six are within noise once the null is allowed to show its real width — the longer run widened
the nulls from 0.1-2% to 5-11.6%, and the deltas did not follow. **No `M = 1` regression is
claimed, and none is supported.** The short-run signal is recorded here because it is a good
example of a null control that is too tight to trust: on a cell this fast, a 5-trial null can
under-report the noise floor by an order of magnitude and manufacture a regression out of nothing.

### The negative control: row-major `B` is untouched

The row-major branch of `pack_b` is not modified, so the `nn` cells must not move. `qwen3_8b`:

| cell | t | base | new | delta | null |
|---|---:|---:|---:|---:|---:|
| nn m32 | 1 | 29.72 | 29.68 | -0.1% | 29.63 |
| nn m32 | 8 | 6.49 | 6.65 | +2.5% | 6.75 |
| nn m32 | 16 | 11.84 | 12.36 | +4.4% | 11.92 |
| nn m128 | 1 | 109.27 | 109.41 | +0.1% | 109.31 |
| nn m128 | 8 | 20.55 | 20.48 | -0.3% | 20.41 |
| nn m128 | 16 | 15.31 | 14.07 | -8.1% | 14.50 |
| nn m512 | 1 | 440.12 | 440.27 | 0.0% | 438.37 |
| nn m512 | 8 | 63.51 | 63.77 | +0.4% | 63.33 |
| nn m512 | 16 | 38.05 | 37.80 | -0.7% | 37.79 |

Seven of nine are within 0.5%. The two that are not are both at `t = 16`, have **opposite signs**,
and sit on top of a null that moved by a comparable amount. That is noise, not an effect.

## How much of the penalty is gone

`nt - nn`, the layout cost in ms, `qwen3_8b`:

| cell | before | after | removed |
|---|---:|---:|---:|
| m32 t=1 | 23.0 | 8.6 | 63% |
| m32 t=8 | 30.1 | 9.6 | 68% |
| m32 t=16 | 59.9 | 19.1 | 68% |
| m128 t=1 | 51.7 | 15.3 | 70% |
| m128 t=8 | 59.3 | 19.3 | 67% |
| m128 t=16 | 58.4 | 20.0 | 66% |
| m512 t=1 | 204.8 | 60.2 | 71% |
| m512 t=8 | 52.6 | 12.9 | 76% |
| m512 t=16 | 56.2 | 17.5 | 69% |

**Roughly two-thirds to three-quarters of the layout penalty, and no more.** The `nn` control is
the honest ceiling for a packing change, and it is still 2.6-3.3x behind ORT on its own.

## What this does not fix, stated plainly

* **The GEMM itself is still 2.6-3.3x behind ORT even with free packing.** Every `nn` cell above
  says so. This change removes a layout tax; it does not make the half GEMM competitive. That is
  the *same* gap section 1 of the work list describes for f32, and it is untouched here.
* **`B` is still re-packed once per row-block.** The residual 9-20 ms is mostly this: the pack is
  now ~3x cheaper but still runs `m / mc` times. Hoisting the packed `B` panel out of the row-block
  loop and sharing it across workers is the next structural fix, and it would help the row-major
  path equally. It is a change to `gemm_impl`'s parallel decomposition, not to a kernel, so it is
  deliberately not bundled here.
* **`M = 1` `transB = 1` overlaps #1417 and that PR's fix is the better one.** This change takes
  the `M = 1` cell from 36.11 to 15.87 ms (2.28x) because on `e13460af6` it still falls into the
  blocked GEMM. #1417 routes `M = 1` to the f16 GEMV instead and reaches ~1.2 ms. The two are
  complementary — #1417 removes the cell from this kernel; this change fixes the kernel for the
  `M > 1` cells that stay. They do not conflict, and the `M = 1` row is excluded from the matrix
  above for that reason.
* **The `Gemm` f16 `M = 128` rows in the work-list matrix (1.03-2.24) do not describe the shipped
  default build.** On the default native build I measure `transB = 1` at 4.24x and even the
  row-major control at 2.88x at `M = 128 / t = 1`. Those rows are in the same category the work
  list already flags for `MatMul` f32 `M = 1` and `QLinearMatMul` — measured on a research build.
  I have not re-measured every row, so I have added a note rather than rewriting them.
* **Anti-scaling from `t = 8` to `t = 16` is still present** on the small-`M` cells (`m32` gets
  *slower*: 16.20 -> 31.42 ms). This change does not address it and the ratios at `t = 16` remain
  the worst in the table.
* **AVX-512 / VNNI untested** — this host has neither. `TRANSPOSED_PACK_GROUP` is a compile-time
  constant tuned for a 32 KiB L1d.

## Reproducing

```sh
python3 scripts/ort_ab/gen_f16_nt.py --out bench-models/f16nt
cargo build --release -p onnx-genai-bench --features bench-native --bin bench_generic

# native arm and ORT arm are measured SEPARATELY, on purpose (see below)
./target/release/bench_generic --model bench-models/f16nt/f16gemm_nt_qwen3_8b_m128.onnx \
  --native-only --runs 21 --warmups 7 --native-threads 8
./target/release/bench_generic --model bench-models/f16nt/f16gemm_nt_qwen3_8b_m128.onnx \
  --ort-only --runs 21 --warmups 7 --ort-intra-threads 8
```

Separate arms are **not** optional here. ORT's intra-op pool spin-waits, so a paired run depresses
the native arm on exactly these small-`M` cells; the co-residency effect has been measured at up to
6x elsewhere in this work. Run the paired mode only to confirm `parity=PASS`.

`--ort-only` previously refused any `f16` graph — it synthesized only f32/i32/i64 inputs while the
paired path already handled Float16, Uint8 and Int8. That gap made the separate-arm method
unavailable for precisely the half-precision kernels that most need it, so `run_ort_only` now uses
the same synthesizers as the paired path and is fed byte-identical inputs.
