# Decision: AVX2 LayerNorm — Small-N Threshold + Welford SIMD

**Author:** Resch (Intel CPU Optimization)
**Date:** 2026-08-11
**PR:** https://github.com/microsoft/onnxruntime/pull/31973
**Host:** AMD EPYC 9V74 (AVX2/FMA, no AVX-512)

## Task 1: Small-N Size Threshold

### Measured crossover

Microbenchmark comparing AVX2 kernel vs scalar `ComputeJob` baseline (3M iterations, `noinline`, AMD EPYC 9V74):

**RMSNorm** (same algorithm both sides — sum-of-squares):
| N | Scalar(ns) | AVX2(ns) | Ratio |
|---|-----------|---------|-------|
| 1 | 4.9 | 5.9 | 1.22 |
| 4 | 6.5 | 7.7 | 1.19 |
| 7 | 10.4 | 10.7 | 1.03 |
| 8 | 7.5 | 6.5 | 0.87 |
| 16 | 11.5 | 7.4 | 0.64 |

**LayerNorm** (Welford scalar vs Welford SIMD):
| N | Scalar(ns) | AVX2(ns) | Ratio |
|---|-----------|---------|-------|
| 1 | 6.5 | 8.4 | 1.29 |
| 2 | 9.1 | 9.7 | 1.06 |
| 3 | 12.6 | 11.0 | 0.88 |
| 8 | 27.1 | 8.0 | 0.29 |

### Chosen threshold: NormSize < 8

Placed in `layernorm.cpp` (the MLAS dispatch function), returning `false` so the caller's scalar `ComputeJob` path executes instead. Rationale:

1. **N < 8 means zero SIMD iterations.** The AVX2 kernel's vector loop condition is `i + 8 <= n`. Below 8 elements, the entire computation falls into the scalar tail — no vectorization occurs by definition.
2. **Dispatch-level avoids call overhead.** Returning `false` before calling the kernel means short rows never pay function-call overhead, branch prediction cold-start, or vector register setup.
3. **Natural, defensible boundary.** 8 = one ymm register width. Any reviewer can verify: below this, no 256-bit instruction executes.

### Test impact

9 of 39 MLAS tests now fail with `REACHABILITY FAILURE` at N∈{1,7} and the ZeroVariance edge test (N=1). These tests assert `MlasLayerNormF32` must return `true` on AVX2 hardware. **Chew must update the assertion** to accept `false` for N < 8 — the scalar fallback is correct and tested separately. All 30 tests with N≥8 pass.

## Task 2: Numerics — Welford-Preserving SIMD (Option a)

### Decision

Replaced the two-pass `sum + sum_of_squares` variance formulation in the full LayerNorm path with **Welford's online algorithm using 8 parallel accumulators** (one per AVX2 lane), merged with the standard pairwise combine formula.

RMSNorm retains two-pass sum-of-squares (numerically equivalent to the scalar RMS path — no cancellation risk since there's no mean subtraction in the variance).

### Performance (Welford SIMD vs scalar Welford baseline)

| NormSize | Scalar(ns) | Welford SIMD(ns) | Speedup |
|----------|-----------|-----------------|---------|
| 16 | 64.1 | 40.4 | 1.6× |
| 128 | 786.4 | 151.8 | 5.2× |
| 1024 | 6254.1 | 886.1 | 7.1× |
| 4096 | 25252.3 | 3663.5 | 6.9× |

Welford SIMD is ~2.5-3× slower than two-pass SIMD (due to per-iteration `_mm256_div_ps`), but still a **5-7× win over the scalar Welford baseline** at typical LLM hidden dimensions. The per-element division is the price of numerical stability, and it's amortised across 8 lanes.

### Numerical accuracy (adversarial: mean~1e6, spread~1, N=4096)

| Method | max_abs_output_err | max_abs_mean_err |
|--------|-------------------|-----------------|
| Two-pass (fp32) | **4.18** | **30.0** |
| Welford SIMD | 0.049 | 0.015 |
| Welford scalar | 0.227 | 0.172 |
| fp64 reference | 0.0 | 0.0 |

Two-pass suffers **catastrophic cancellation** when computing `Var = E[X²] - E[X]²` with mean >> spread. The Welford SIMD kernel is actually more accurate than scalar Welford for adversarial inputs (the 8-way lane decomposition acts like a mild form of compensated summation).

For normal-range inputs (mean~0, spread~1): all methods are within 1e-7, essentially equivalent.

### What I need from Chew

1. **Update test assertions** for N < 8: `MlasLayerNormF32` now legitimately returns `false` for short rows.
2. **Fix the `worst_welford` unused-variable error** in `test_layernorm.cpp:656` (blocks the build with `-Werror`).
3. His adversarial precision tests should confirm the Welford SIMD numbers above. I'd like him to test with mean offsets of 1e4, 1e6, 1e8 and N = 128, 1024, 4096 to characterize the fp32 precision boundary.

## Files changed

- `onnxruntime/core/mlas/lib/layernorm.cpp` — NormSize < 8 threshold
- `onnxruntime/core/mlas/lib/layernorm_kernel_avx2.cpp` — Welford SIMD for full LayerNorm, restructured RMSNorm path

## Task 3: RMSNorm — Skip mean accumulation when MeanOut is null (Gaff S1)

### Premise verification

Confirmed: in the RMSNorm (Simplified) path, `vsum` accumulates the element sum
purely to compute `mean_val`, which is **only** written to `*MeanOut` at the
bottom of the function. The Simplified normalization pass multiplies
`Input[i] * inv_denom * Scale[i]` — it never subtracts the mean. Therefore
`vsum` is dead work when `MeanOut == nullptr`.

### Change shape

Per-row runtime check `if (MeanOut != nullptr)` **outside** the inner vector
loop, splitting into two loop bodies:
- **MeanOut requested:** accumulate both `vsum` and `vsumsq` (original behavior).
- **MeanOut null:** accumulate `vsumsq` only, skip `vsum` + horizontal reduce.

Rejected alternatives:
- **Template/constexpr split:** Would require a second kernel symbol + dispatch
  changes in `platform.cpp`/`mlasi.h`. Too invasive for one `vaddps`.
- **Runtime branch inside inner loop:** Could cost more than the `vaddps` it
  removes due to branch misprediction overhead.

The per-row check is free — the function is called once per normalization row,
so the branch is predicted correctly after the first row.

### Measured speedup

Micro-benchmark on AMD EPYC 9V74, 500k iterations, RMSNorm only:

| NormSize | With MeanOut (ns) | Without MeanOut (ns) | Speedup |
|----------|------------------|---------------------|---------|
| 8        | 8.5              | 8.0                 | 6.0%    |
| 16       | 9.6              | 8.9                 | 7.8%    |
| 32       | 12.4             | 11.3                | 9.2%    |
| 64       | 17.6             | 16.8                | 4.8%    |
| 128      | 31.1             | 29.9                | 3.7%    |
| 256      | 74.9             | 75.5                | -0.8%   |
| 512      | 132.7            | 131.3               | 1.0%    |
| 1024     | 254.7            | 253.6               | 0.4%    |
| 4096     | 1166.7           | 1166.7              | 0.0%    |

**Honest assessment:** Gaff's "~15%" estimate was overstated. The real speedup is
**5-9% for small N (8-64), negligible (<1%) for N≥256.** At LLM-typical hidden
dimensions (768-4096), the savings are in the noise. The change stands on
**code clarity** — the intent that the sum is dead when MeanOut is null is now
explicit — not on a performance claim.

### N2: dispatch-contract assert

The `assert(!Simplified || Bias == nullptr)` on line 47 is invisible in release
builds. I agree with Gaff's observation and also agree it should stay as-is.
This matches MLAS convention: `assert` documents the contract for developers;
the dispatch layer in `layernorm.cpp` enforces it structurally by passing
`nullptr` for Bias when Simplified is true. Adding a runtime check would
penalise every call for a condition that's architecturally impossible.
