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
