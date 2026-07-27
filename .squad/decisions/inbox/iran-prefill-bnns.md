# Decision: BNNS prefill + first-decode spike elimination

**Author:** Iran (Mac CPU Optimization Engineer)
**Date:** 2026-07-27
**Status:** Implemented (PR #275, branch `squad/mac-prefill-bnns`)
**Commits:** `f0cbd786`, `aa219b4b`, `9f1e7684`, `3ab6999a`, `17be7087`

## Context

PR #227 won decode (1.42× ORT) but prefill was 9.1–11.9× worse than ORT,
making end-to-end 0.464× ORT. The structural opportunity: Apple's BNNS
`BNNSMatMul` reaches AMX for fp16→f32 GEMM, while ORT's MLAS only reaches
NEON/KleidiAI on macOS.

## Decisions

### 1. Three-regime MatMul dispatch
- **M=1:** NEON GEMV from f16 mmap'd weights (bandwidth-bound, unchanged from #227)
- **M≥2 on macOS:** BNNS `BNNSMatMul` f16→f32 via AMX (compute-bound)
- **M≥2 elsewhere:** portable `half_gemm.rs` NEON fallback

### 2. Column-major B zero-copy for both BNNS and GEMV
- lm_head weight is column-major B[K,N] with strides [1,K]
- Memory layout is B^T[N,K] row-major — exactly what BNNS `trans_b` and GEMV need
- Zero-copy: pass raw mmap data directly, no transpose or contiguous copy

### 3. Global weight-transpose cache with eager pre-transpose
- Process-global `LazyLock<Mutex<HashMap<usize, Arc<Vec<u16>>>>>` keyed by data pointer
- Survives kernel-cache shape evictions (M=40→M=1 shape change creates new kernel instances)
- Eager pre-transpose during model load populates cache for all f16 MatMul/FusedMatMulBias weights
- Adds ~7ms to model load (still 14.6× faster than ORT), saves ~30ms on first decode
- **Lifetime contract:** `clear_weight_transpose_caches()` called from `Executor::Drop`
  prevents address-reuse staleness and memory leaks

### 4. Correctness fixes (rubber-duck review findings)
- **Non-constant non-contiguous B guard:** The rescue block (M≥2, non-contiguous B) now
  requires `constant_inputs[1]`. Without it, non-constant activations (Transpose views)
  entered the block, `contiguous_b_f16()` returned None, output was silently all zeros.
- **Cache lifetime management:** `clear_weight_transpose_caches()` in `Executor::Drop`
  prevents stale data if mmap address is reused by a subsequent model load.
- **Poison recovery:** `.lock().unwrap_or_else(|e| e.into_inner())` at all cache lock
  sites prevents cascade aborts if a thread panics while holding the lock.
- **Buffer density assert:** `debug_assert_eq!` in `precompute_f16_weight_transpose`
  verifies buffer is dense (no padding/gaps).
- **M≥2 threshold documented:** Categorical GEMV-vs-GEMM, not a tuned crossover.

## Results (measured, load ~12–20)

| metric | before | after | ORT | vs ORT |
|---|---:|---:|---:|---:|
| TTFT | 989 ms | 167 ms | 109 ms | 1.53× |
| decode | 57.6 tok/s | 70.6 tok/s | 42.2 tok/s | 1.67× |
| end-to-end | 17.7 tok/s | 57.8 tok/s | 38.7 tok/s | **1.50×** |
| total time | ~2800 ms | 865 ms | 1293 ms | **1.50×** |

End-to-end reconciles: 170 + 49×14.2 = 865 ≈ 865 measured. The ~967ms
first-decode spike is gone.

## Root cause of the ~1s spike

Shape-keyed kernel cache: prefill (M=40) → decode (M=1) shape change creates
new kernel instances with cold OnceLock caches. 169 kernels re-transpose
~776 MB of weights. Additionally, the lm_head (K=896, N=151936, column-major)
fell through to f32 densification—a 544 MB allocation costing ~960 ms alone.

## Constraints upheld
- ✅ Decode unregressed: 70.6 vs 42.2 = 1.67× ORT (at low load)
- ✅ Guard tests green: `fp16_m1_decode_reaches_neon_gemv_not_half_gemm`,
  `fp16_m_ge2_prefill_reaches_bnns_not_half_gemm`, `bf16_m_ge2_does_not_reach_bnns`,
  `f16_non_constant_non_contiguous_b_produces_correct_result`,
  `f16_constant_non_contiguous_b_enters_rescue_block`
- ✅ No BNNS/Accelerate from inside Rayon parallel region
- ✅ x86_64 cross-compilation clean (clippy --all-targets -D warnings)
- ✅ Runtime feature detection, no compile-time constants
- ✅ `check_platform_naming.py` passes
