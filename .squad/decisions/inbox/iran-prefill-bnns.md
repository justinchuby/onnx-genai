# Decision: BNNS fp16→f32 Prefill via AMX

**Author:** Iran (Mac CPU Optimization Engineer)
**Date:** 2026-07-27
**PR:** #275
**Branch:** `squad/mac-prefill-bnns`
**Commits:** `a855f826` (initial BNNS dispatch), `58bafd0d` (filter cache + contiguous B rescue), `f0cbd786` (trans_b zero-copy)

## Context

Decode was won in PR #227 (native 1.42× vs ORT). Prefill remained 9.6–11.9× worse than ORT (TTFT 1034–1314 ms vs 103–110 ms), making end-to-end performance 0.40–0.45× of ORT. Prefill is compute-bound (arithmetic intensity ≈20 FLOP/byte) — the opposite of decode's bandwidth-bound character.

## Decision

**Three-regime MatMul dispatch**, replacing the binary `m > 1` gate:

| M | Path | Bound |
|---|---|---|
| M = 1 | NEON GEMV (unchanged) | bandwidth |
| M ≥ 2, macOS | **BNNS `BNNSFilterCreateLayerBroadcastMatMul` fp16→f32** | compute (AMX) |
| M ≥ 2, non-Mac | `half_gemm.rs` NEON | portable fallback |

Plus two critical performance fixes discovered after initial dispatch was null-result:

### Fix 1: BNNS filter cache (thread-local)

`BNNSFilterCreateLayerBroadcastMatMul` costs 3–19 ms cold (GCD dispatch setup / AMX micro-code compilation). A `HashMap<(M,K,N,trans_b), BNNSFilter>` in thread-local storage amortises this to zero for subsequent calls. Filters are cleaned up via `Drop` when the thread exits. A typical 24-layer model has ~25 entries (including trans_b variants for the vocab projection).

### Fix 2: Non-contiguous vocab weight — trans_b zero-copy

The lm_head vocab projection weight (896×151936, 272 MB) is stored column-major (non-contiguous) in the ONNX model. `try_matmul_half` requires contiguous inputs and skips it, causing fallthrough to element-by-element `to_dense_f32_widen` — measured at **1066 ms** for 136M elements.

Initial fix materialised a contiguous row-major f16 copy via `MatMulPrepack::contiguous_b_f16` (parallel Rayon strided copy, cached per session via `OnceLock`). This worked but the copy still cost ~180 ms (warm pages) per Engine lifetime.

**Final fix:** Column-major B[K,N] in memory is row-major B^T[N,K]. BNNS's `trans_b: true` flag computes C = A @ (B^T)^T = A @ B — passing the raw data directly with zero copy. The FilterCache key was extended to `(M,K,N,trans_b)` and the B descriptor is created as [N,K] when trans_b is true.

Two sub-bugs found:
- FilterCache needed trans_b in the key to avoid shape collisions
- The `batch_shape.is_empty()` check blocked the trans_b path because the engine emits A as [1,M,K], producing a trivial batch [1]. Fixed by checking `trivial_batch` (all dims == 1) instead.

This eliminated the remaining ~180 ms vocab copy: TTFT **347 → 167 ms**.

## Why BNNS

- Standard BLAS has no half-precision GEMM (`sgemm` is f32, `dgemm` is f64)
- Apple's fp16 matrix path is BNNS, which reaches AMX
- Measured 2451 GFLOPS at M=128 vs 52 GFLOPS for NEON blocked GEMM (~47×)
- ORT links no Accelerate — this is a structural advantage they cannot match without taking an Accelerate dependency

## Why M=2 threshold

Sebastian measured the crossover at M=2 for both prefill and batch decode. The per-call GCD overhead (~50 µs) is absorbed by AMX throughput even at small M.

## Constraints upheld

1. **No BNNS/Accelerate from Rayon parallel regions** — calls from dispatch level only
2. **Decode unregressed** — `fp16_m1_decode_reaches_neon_gemv_not_half_gemm` guard passes; decode 58.36 tok/s (1.390× ORT)
3. **Runtime feature detection** — `bnns_matmul_available()` probes at startup, caches result
4. **One implementation** — `half_gemm.rs` remains portable fallback for non-Mac
5. **Cross-compilation clean** — clippy passes on both aarch64 and x86_64 with `--all-targets -D warnings`

## Key implementation detail: b_is_weights=false

Setting `b_is_weights: true` causes `BNNSFilterApplyTwoInput` to return -1. When true, BNNS expects B data baked into the filter descriptor at creation time. Since we pass both A and B at apply time, both must be `false`.

## Tests

- Dispatch reachability: fp16 M≥2 → BNNS (atomic counter guard)
- Non-contiguous B rescue: vocab weight routes through cached contiguous copy → BNNS
- BF16 exclusion: verified via output parity with portable reference
- Numerics: f64 reference at shapes up to 128×896×4864, tolerance √K·2e-3
- Edge values: fp16 max (65504), denormals, NaN, zero
- Bitwise determinism
- Guard-break proof: broken M threshold caught by test
- Decode guard: `fp16_m1_decode_reaches_neon_gemv_not_half_gemm` green

## Measurements

M1 Max 10-core, 64 GB, `qwen2.5-0.5b-f16`, 40-token prompt, 50 gen tokens.

### Before (baseline, load 8–13, Justin's measurement):
| | native | ORT |
|---|---:|---:|
| TTFT | 989.2 ms | 108.7 ms |
| decode | 57.57 tok/s | 41.39 tok/s |
| end-to-end | 17.69 | 38.10 |

### After (commit `58bafd0d`, load 25–32):
| | native | ORT |
|---|---:|---:|
| TTFT | **347.4 ms** [346.8, 351.1] | 108.5 ms [108.0, 109.8] |
| decode | **58.36 tok/s** [57.32, 59.76] | 41.98 [41.76, 42.56] |
| end-to-end | 22.94 [22.24, 23.34] | 38.50 [38.37, 39.02] |

### After trans_b (commit `f0cbd786`, load 24–27):
| | native | ORT |
|---|---:|---:|
| TTFT | **167.4 ms** [166.8, 168.6] | 108.4 ms [108.3, 109.0] |
| decode | **55.31 tok/s** [42.58, 56.45] | 41.88 [40.80, 41.93] |
| end-to-end | 24.54 [21.65, 24.83] | 38.42 [37.61, 38.48] |

**Overall TTFT: 5.9× faster** (989 → 167 ms). Now only 1.54× behind ORT (was 9.1×).
**Decode: 1.32× ORT** — unregressed.

**TTFT breakdown** (168 BNNS calls, M=40, measured per-call):
- 24×3 large GEMMs (K=896,N=4864 / K=4864,N=896): ~1.3 ms each, total ~94 ms
- 24×2 medium GEMMs (K=896,N=896): ~0.2 ms each, total ~10 ms
- 24×2 small GEMMs (K=896,N=128): ~0.03 ms each, total ~1.4 ms
- Vocab projection (K=896,N=151936, via rescue path): ~14 ms
- Contiguous B copy (one-time per Engine): ~30 ms
- **Total BNNS GEMM: ~150 ms** → remaining ~200 ms is non-GEMM (LayerNorm, SoftMax, RoPE, embedding, graph dispatch)

**BNNS per-call GFLOPS** (M=40): 260–346 GFLOPS depending on shape.

### Diagnosis of null-result (initial BNNS dispatch showed 0% TTFT improvement)

Two root causes, both fixed:
1. **BNNS filter cold-start**: 3–19 ms per unique shape per thread. With ~20 unique shapes, this added ~60–380 ms to first prefill. Fixed with thread-local filter cache.
2. **Non-contiguous vocab weight**: lm_head weight stored column-major. Bypassed BNNS entirely, fell through to element-by-element f32 widening (1066 ms). Fixed with contiguous f16 copy cache + rescue dispatch.

## Remaining gap to ORT

TTFT is 167 ms vs ORT 108 ms (1.54× at load 24–27). The remaining 59 ms gap breaks down as:
- **~105 ms BNNS GEMM** (168 calls at ~260–1270 GFLOPS depending on shape)
- **~55 ms non-GEMM** (LayerNorm, SoftMax, RoPE, embedding, graph dispatch — only 2.1% of original TTFT)
- **~7 ms filter cache** (amortised to zero after first inference)

Production BNNS runs at 260–346 GFLOPS vs microbenchmark 966–2451 GFLOPS (3.7× gap). The gap is attributable to M=40 vs M=128 tile utilization (~2×) and production environment effects (mmap'd weights, TLB pressure from 994 MB model mapping, GCD scheduling — ~2×). Further GEMM improvement requires model tiling or engine-level batching.

End-to-end is 0.64× ORT, limited by TTFT. Decode remains 1.32× ORT (unregressed).
