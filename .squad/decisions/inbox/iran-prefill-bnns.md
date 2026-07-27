# Decision: BNNS prefill + first-decode spike elimination

**Author:** Iran (Mac CPU Optimization Engineer)
**Date:** 2026-07-27
**Status:** Implemented (PR #275, branch `squad/mac-prefill-bnns`)
**Commits:** `a855f826`, `58bafd0d`, `f0cbd786`, `aa219b4b`, `9f1e7684`

## Context

Decode was won in PR #227 (native 1.42× vs ORT). Prefill remained 9.6–11.9× worse than ORT (TTFT 1034–1314 ms vs 103–110 ms), making end-to-end performance 0.40–0.45× of ORT. Prefill is compute-bound (arithmetic intensity ≈20 FLOP/byte) — the opposite of decode's bandwidth-bound character.

## Decisions

### 1. Three-regime MatMul dispatch

| M | Path | Bound |
|---|---|---|
| M = 1 | NEON GEMV (unchanged) | bandwidth |
| M ≥ 2, macOS | **BNNS `BNNSMatMul` fp16→f32 via AMX** | compute |
| M ≥ 2, non-Mac | `half_gemm.rs` NEON | portable fallback |

### 2. Column-major B zero-copy for both BNNS and GEMV

The lm_head vocab projection weight (896×151936, 272 MB) is stored column-major (non-contiguous). Column-major B[K,N] in memory is row-major B^T[N,K]:
- **BNNS path (M≥2):** `trans_b: true` lets BNNS read the raw mmap'd data directly
- **GEMV path (M=1):** Raw data IS B_T[N,K], exactly what `neon_gemv_f16_col_parallel` needs — route directly, zero-copy

Without this, the lm_head falls through to f32 densification (544 MB alloc, ~960 ms).

### 3. Global weight-transpose cache with eager pre-transpose

The kernel cache is shape-keyed: prefill M=40 → decode M=1 creates new kernel instances with cold OnceLock caches. 169 kernels would re-transpose ~776 MB.

- Process-global `LazyLock<Mutex<HashMap<usize, Arc<Vec<u16>>>>>` keyed by data pointer
- Survives kernel-cache shape evictions via Arc sharing
- Eager pre-transpose during model load: +7ms load time, saves ~30ms on first decode
- Model load still 14.6× faster than ORT (114ms vs 1671ms)

### 4. BNNS filter cache (thread-local)

`BNNSFilterCreateLayerBroadcastMatMul` costs 3–19 ms cold. A `HashMap<(M,K,N,trans_b), BNNSFilter>` in thread-local storage amortises to zero for subsequent calls. Filters cleaned up via `Drop`.

## Why BNNS

- Standard BLAS has no half-precision GEMM (`sgemm` is f32, `dgemm` is f64)
- Apple's fp16 matrix path is BNNS, which reaches AMX
- Measured 2451 GFLOPS at M=128 vs 52 GFLOPS for NEON blocked GEMM (~47×)
- ORT links no Accelerate — structural advantage they cannot match

## Constraints upheld

1. **No BNNS/Accelerate from Rayon parallel regions** — calls from dispatch level only
2. **Decode unregressed** — guard test `fp16_m1_decode_reaches_neon_gemv_not_half_gemm` passes; 70.6 tok/s (1.67× ORT)
3. **Runtime feature detection** — `bnns_matmul_available()` probes at startup
4. **One implementation** — `half_gemm.rs` remains portable fallback
5. **Cross-compilation** — clippy clean on both aarch64 and x86_64 with `--all-targets -D warnings`

## Final results

M1 Max 10-core, 64 GB, `qwen2.5-0.5b-f16`, 40-token prompt, 50 gen tokens, load ~12.

| metric | before campaign | after | ORT | vs ORT |
|---|---:|---:|---:|---:|
| TTFT | 989 ms | **170 ms** | 109 ms | 1.56× |
| decode | 57.6 tok/s | **70.6 tok/s** | 42.2 tok/s | **1.67×** |
| end-to-end | 17.7 tok/s | **57.8 tok/s** | 38.7 tok/s | **1.50×** |
| model load | 105 ms | 114 ms | 1671 ms | 0.068× |
| total time | ~2800 ms | **865 ms** | 1293 ms | **1.50×** |

End-to-end arithmetic reconciles: 170 + 49×14.2 = 865 ≈ 865 measured ✓

## Evolution

1. Initial BNNS dispatch: null result (989ms unchanged) — non-contiguous weights bypassed BNNS
2. Filter cache + contiguous_b_f16: TTFT 989→348ms
3. trans_b zero-copy: TTFT 348→167ms
4. Global cache + column-major GEMV: eliminated ~967ms first-decode spike, end-to-end reconciles

## Remaining leads

1. TTFT gap: 170ms vs ORT 109ms (1.56×). BNNS production at 260–346 GFLOPS vs 2451 microbenchmark
2. Non-GEMM overhead: ~55ms (LayerNorm, SoftMax, RoPE, graph dispatch)
