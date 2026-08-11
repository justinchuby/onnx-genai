# Iran: LayerNorm B1/B2 kernel fix — centered two-pass with double-sum

**Date:** 2026-08-11
**PR:** microsoft/onnxruntime#31973
**Branch:** `nxrt/mlas-avx2-layernorm` in `/workspace/upstream/ort-fork`
**Status:** Kernel code revised; 2 precision tests need Pris update (algorithm change from Welford to centered two-pass)

## B1 — Reproduced: AVX2 lane-parallel Welford is badly inaccurate

**Confirmed.** Standalone reproducer with fp64 oracle across adversarial sweep:

| base | spread | N | Scalar Welford rel err | AVX2 Welford rel err | Ratio |
|------|--------|---|------------------------|----------------------|-------|
| 1e5 | 1e-2 | 512 | 4.73e-03 | 2.82e-01 | 59.6× |
| 1e5 | 1e-2 | 1024 | 2.03e-03 | 2.68e-01 | 131.7× |
| 1e5 | 1e-2 | 4096 | 3.35e-05 | 2.71e-01 | 8096× |
| 1e6 | 1e-1 | 4096 | 3.06e-04 | 1.92e-01 | 627.8× |

Worst-case AVX2 Welford: **28.2% relative error** on inv_std_dev.
Root cause confirmed: per-lane fp32 means accumulate rounding errors proportional to N/8; the merge cannot recover.

## B2 — Centered two-pass with double-sum: faster AND more accurate

Evaluated three alternatives against fp64 oracle:

| Algorithm | Worst rel err (inv_std) | Speed (N=1024) | vs scalar |
|-----------|------------------------|----------------|-----------|
| Scalar Welford | 5.03e-02 | 6096 ns | 1× |
| AVX2 Welford (old) | 2.82e-01 | 751 ns | 8.1× |
| Centered 2P fp32-sum | **1.00e+00** | ~400 ns | ~15× |
| **Centered 2P double-sum** | **5.95e-03** | **427 ns** | **14.3×** |

**Decision:** Replaced Welford with centered two-pass, double accumulation for the first-pass sum.

- **fp32 sum was NOT sufficient.** At base=1e7/N=4096, fp32 sum rounds the mean enough that var=sum((x-mean)^2) collapses to zero → 100% error.
- **Double sum is critical.** The 4-wide `_mm256_cvtps_pd` + `_mm256_add_pd` loop costs ~10% throughput vs 8-wide fp32, but eliminates the mean rounding that destroys variance.
- **Centered squaring eliminates cancellation.** Unlike the uncentered E[x²]-mean² that produced NaN in earlier testing, subtracting the accurate mean before squaring is numerically standard and safe.
- The Welford inner loop's `_mm256_div_ps` (latency 11-14 cycles) is the bottleneck that makes centered two-pass 1.8× faster.

## N2 — RVV suppression fixed

The `NormSize < 8` threshold was in shared `layernorm.cpp` dispatch, gating ALL platforms. Moved it inside `#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_IX86)` so RVV (and any future non-x86 kernel) sees no gate.

## N3 — RMSNorm mean skip

The RMSNorm path already skipped the mean sum when `MeanOut == nullptr` (prior commit). Updated the RMSNorm mean path to also use double-precision accumulation for consistency when the caller does request it.

## N4 — Windows `/arch:AVX2`

`layernorm_kernel_avx2.cpp` was listed in `target_sources` but NOT in the GLOB'd `mlas_platform_srcs_avx2` variable, so `set_source_files_properties` for `/arch:AVX2` did not reach it on MSVC. Added an explicit `set_source_files_properties` call.

## Files changed

- `onnxruntime/core/mlas/lib/layernorm_kernel_avx2.cpp` — full LayerNorm: Welford → centered two-pass with double-sum; RMSNorm MeanOut path upgraded to double-sum
- `onnxruntime/core/mlas/lib/layernorm.cpp` — NormSize<8 gate scoped to x86 only
- `cmake/onnxruntime_mlas.cmake` — `/arch:AVX2` for layernorm on MSVC

## Test results

- **39/41 LayerNorm tests pass** (all 32 functional tests + 3 precision tests)
- **2 precision tests fail** (`CatastrophicCancellationPasses`, `Fp64ParitySweep`) — these compare SIMD output to scalar Welford parity. With the algorithm change to centered two-pass, parity with Welford is no longer the contract. **Pris must update these to compare against fp64 oracle instead.**

## Entry points for Sebastian's benchmark

```cpp
// The single public entry point (dispatches to platform kernel):
bool MlasLayerNormF32(Input, Scale, Bias, Output, MeanOut, InvStdDevOut, NormSize, Epsilon, Simplified);

// The AVX2 kernel directly (for micro-benchmarks):
void MlasLayerNormKernelAvx2(Input, Scale, Bias, Output, MeanOut, InvStdDevOut, NormSize, Epsilon, Simplified);
```
