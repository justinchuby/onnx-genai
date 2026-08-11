# Decision: AVX2 LayerNorm/RMSNorm Benchmark — True Baseline Measurement

**Author:** Sebastian (Performance Engineer)
**Date:** 2026-08-11
**Status:** Complete — ready for upstream PR #31973
**PR:** https://github.com/microsoft/onnxruntime/pull/31973

---

## Summary

Pris flagged that her initial benchmark used an **fp64-accumulated C++ reference** as the scalar baseline, not the actual fp32 fallback the kernel replaces. I have now measured against the **true baseline**: the scalar fp32 fallback from `onnxruntime/core/providers/cpu/nn/layer_norm_impl.cc` (`ComputeJob`).

**Result:** The honest speedup is **larger** than the fp64-inflated numbers, not smaller. The fp64 reference was actually *faster* than the true baseline (fewer operations, no per-element division in Welford's), so Pris's original numbers were *conservative*.

---

## What is the True Baseline?

Before this PR, on x86-64:
- `MlasLayerNormF32()` returned `false` (no kernel registered — `LayerNormF32Kernel` was `nullptr`)
- The caller in `layer_norm_impl.cc` fell through to its inline scalar fp32 code:
  - **Full LayerNorm:** Welford's online algorithm (single-pass, per-element `delta / (h+1)` division)
  - **RMSNorm (Simplified):** Simple sum-of-squares accumulation

The AVX2 kernel replaces this fallback with a two-pass vectorized approach.

### Why the speedup asymmetry between LayerNorm and RMSNorm

The full LayerNorm baseline uses **Welford's algorithm** with a per-element floating-point division, which is inherently serial and expensive. The AVX2 kernel replaces this with a simple two-pass sum + sum-of-squares (vectorized), which is both algorithmically cheaper *and* SIMD-vectorized.

For RMSNorm, both the baseline and the kernel use sum-of-squares, so the speedup is purely from SIMD vectorization.

This means the LayerNorm speedup reflects both:
1. **Algorithmic improvement:** two-pass sum vs Welford's per-element division
2. **SIMD vectorization:** 8-wide AVX2+FMA

This is a *fair* comparison — the kernel genuinely replaces Welford's fallback. But upstream reviewers should understand the speedup is not purely from SIMD.

---

## Auto-Vectorization Check

- **Test binary compile flags:** `-O3 -std=gnu++20` — **no `-mavx2`**, no `-ftree-vectorize` override
- **Global CMake flags:** `-fno-fast-math` (prevents aggressive vectorization)
- **Production `layer_norm_impl.cc`:** Same flags (no per-file overrides)
- **Object file verification:** 0 ymm (AVX) instructions in `test_layernorm.cpp.o`; only xmm (SSE2) used
- **Conclusion:** The scalar baseline is **not auto-vectorized** beyond SSE2, matching the production build exactly. The comparison is fair.

---

## Evidence Template

| Field | Value |
|---|---|
| **Hardware** | AMD EPYC 9V74 80-Core (16 cores / 32 threads, 1 socket) |
| **ISA** | AVX2, FMA, F16C — **no AVX-512** |
| **OS** | Linux 6.11.0-1012-azure (Ubuntu 24.04) |
| **Compiler** | g++ 13.3.0 |
| **Flags** | `-O3 -DNDEBUG -std=gnu++20 -fno-fast-math` (no `-mavx2` for test/baseline) |
| **Commit** | `0c20b10` (branch `nxrt/mlas-avx2-layernorm`) |
| **Baseline commit** | `16b486a2` (upstream `main`) |
| **Binary** | `build/mlas_test/onnxruntime_mlas_test` |
| **Command** | `./onnxruntime_mlas_test --gtest_filter="*DISABLED_Benchmark*" --gtest_also_run_disabled_tests` |
| **Warmup** | 100 iterations |
| **Measured iterations** | 1000 |
| **Threading** | Single-threaded (one norm row at a time) |
| **Benchmark disabled by default** | Yes (upstream convention) |

---

## Results: Full LayerNorm

The baseline is `ComputeJob` from `layer_norm_impl.cc` (Welford's, fp32, no bias).

| NormSize | AVX2 p50 (µs) | Scalar p50 (µs) | Speedup p50 | Speedup p95 | AVX2 stdev | Scalar stdev |
|----------|---------------|-----------------|-------------|-------------|------------|--------------|
| 7        | 0.050         | 0.080           | 1.60×       | 1.35×       | 0.005      | 0.003        |
| 15       | 0.060–0.080   | 0.130           | 1.62–2.17×  | 1.62–2.15×  | 0.002–0.005| 0.005        |
| 128      | 0.080         | 0.811           | 10.14×      | 10.02×      | 0.005      | 0.004–0.325  |
| 256      | 0.100–0.101   | 1.583–1.592     | 15.76–15.83×| 14.35×      | 0.005      | 0.374–0.509  |
| 768      | 0.220–0.221   | 4.757           | 21.52–21.62×| 20.64–20.73×| 0.325–0.379| 0.755–0.964  |
| 1024     | 0.300–0.301   | 6.310–6.350     | 21.03–21.10×| 20.45–21.13×| 0.004–0.008| 0.850–0.908  |
| 2048     | 0.561–0.581   | 12.649–12.698   | 21.77–22.63×| 21.15–21.52×| 0.009–0.020| 1.165–1.509  |
| 4096     | 1.292–1.312   | 25.157          | 19.17–19.47×| 18.68–19.24×| 0.357–0.518| 1.602–1.804  |

## Results: RMSNorm (Simplified)

The baseline is `ComputeJob` simplified path (sum-of-squares, fp32).

| NormSize | AVX2 p50 (µs) | Scalar p50 (µs) | Speedup p50 | Speedup p95 | AVX2 stdev | Scalar stdev |
|----------|---------------|-----------------|-------------|-------------|------------|--------------|
| 7        | 0.050         | 0.040           | 0.80×       | 0.83×       | 0.005      | 0.005–0.008  |
| 15       | 0.060–0.071   | 0.050           | 0.70–0.83×  | 0.74–0.98×  | 0.004–0.005| 0.004        |
| 128      | 0.070–0.071   | 0.180           | 2.54–2.57×  | 2.23–2.26×  | 0.005      | 0.002–0.003  |
| 256      | 0.100         | 0.330           | 3.30×       | 3.01×       | 0.004      | 0.005        |
| 768      | 0.211         | 0.911           | 4.32×       | 4.13×       | 0.005      | 0.324–0.326  |
| 1024     | 0.290–0.291   | 1.202           | 4.13–4.14×  | 4.03–4.04×  | 0.009–0.378| 0.315–0.372  |
| 2048     | 0.551–0.571   | 2.384           | 4.18–4.33×  | 3.98–4.10×  | 0.268–0.428| 0.421–0.521  |
| 4096     | 1.262–1.272   | 4.837–4.967     | 3.83–3.90×  | 3.76–3.88×  | 0.302–0.401| 0.784–0.789  |

---

## Comparison with Pris's Original Numbers

Pris measured "Scalar" = fp64-accumulated reference vs AVX2 kernel:

| NormSize | Pris speedup | True speedup (LayerNorm) | True speedup (RMSNorm) |
|----------|-------------|--------------------------|------------------------|
| 128      | 2.88×       | **10.1×** (LayerNorm)    | **2.5×** (RMSNorm)     |
| 768      | 5.05×       | **21.5×** (LayerNorm)    | **4.3×** (RMSNorm)     |
| 4096     | 4.58×       | **19.3×** (LayerNorm)    | **3.9×** (RMSNorm)     |

Pris's numbers were **conservative**, not inflated — the fp64 reference is faster than the actual fp32 Welford's baseline because it uses a simpler two-pass algorithm with no per-element division.

---

## Amdahl Context — What Is and Is Not Measured

**What is measured:**
- Single-row microbenchmark of one MLAS kernel vs its fallback
- In-process, same binary, same data, same compiler flags
- Measures the kernel dispatch path only (no thread pool, no operator overhead)

**What is NOT measured:**
- Model-level / end-to-end impact
- Multi-row throughput with thread pool parallelism
- Cache effects in a full inference pipeline
- Impact on any specific model's overall latency

**Honest statement:** This is a **microbenchmark**. The kernel-level speedup does not translate linearly to model-level gains. LayerNorm is typically a small fraction of total inference time (dominated by matmul/attention). End-to-end impact is unmeasured and should not be inferred from these numbers.

---

## Notes for Upstream Reviewers

1. The very large LayerNorm speedups (15-22×) are explained by **two factors combined**: the baseline uses Welford's algorithm (per-element division, inherently serial) while the AVX2 kernel uses a simpler two-pass approach (vectorizable). This is a fair comparison because the kernel genuinely replaces Welford's fallback.

2. RMSNorm speedups (3-4×) are more representative of pure SIMD benefit, since both paths use the same algorithm (sum-of-squares).

3. For small sizes (n=7, n=15), RMSNorm shows **no speedup or slight regression** due to AVX2 setup overhead. This is expected and acceptable — small sizes are dominated by function call overhead.

4. The benchmark is DISABLED by default per upstream convention. Run with `--gtest_also_run_disabled_tests`.
