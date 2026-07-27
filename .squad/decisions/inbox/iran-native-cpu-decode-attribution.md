# Native CPU Decode Attribution — Iran

**Date:** 2026-07-27 (updated)
**Model:** Qwen2.5-0.5B-Instruct (100% fp32 dense, 1.93 GB, 496M params)
**Hardware:** Apple M1 Max, 8P+2E cores, 32 GiB unified memory

## Baseline (before changes)
| Configuration | Load | TTFT | Decode | Effective BW |
|---|---|---|---|---|
| ORT + CPU | 1343 ms | 118 ms | **45.87 tok/s** | ~88 GB/s |
| Native + CPU | 125 ms | 1253 ms | **3.26 tok/s** | ~6 GB/s |

## Root Cause (two bugs, both universal)

### Bug 1: Accelerate is an unwired placeholder
`CpuBackend::auto_detect()` returns `Accelerate` on macOS (`backend.rs:83`).
`gemm_with_backend()` had only `Mlas` and `SimdX86` arms — `Accelerate`
fell to `_ => gemm_generic()` (`matmul.rs:169`). Every Mac was running the
pure-Rust correctness baseline (scalar 4×4 tiled GEMM).

### Bug 2: gemm_generic has zero M=1 parallelism
`gemm_generic()` parallelizes over `M` rows via `par_chunks_mut(mc * n)`.
At M=1 (decode), mc=1 → 1 chunk → single-threaded on a 10-core machine.

### Combined effect
Single-threaded scalar GEMV at ~6 GB/s on a 197 GB/s machine = **2.9% of roofline**.

## Per-Op Decode Attribution (169 ops/token, steady-state)

| Op Type | Count | ms/token | % of decode |
|---|---|---|---|
| MatMul | 49 | 16.9 | 47% |
| FusedMatMulBias | 120 | 12.4 | 35% |
| Attention | 24 | 2.9 | 8% |
| Swish | 24 | 1.1 | 3% |
| Other (RMSNorm, RotaryEmb, etc.) | ~200 | ~1.2 | 3% |
| Session overhead | — | ~0.8 | 2% |
| **Total** | **~417** | **~35.8** | **= p50 decode latency** |

Key weight shapes: [896,4864]×48, [4864,896]×24, [896,896]×48, [896,128]×48, [896,151936]×1

## Per-Shape GEMV Bandwidth (current)

| Shape | Weight MB | Route | p50 µs | GB/s |
|---|---|---|---|---|
| [896,4864] gate/up | 17.5 | NEON-MT 8T | ~260 | 66 |
| [4864,896] down | 17.4 | NEON-MT 8T | ~260 | 66 |
| [896,896] q/o | 3.2 | Accelerate sgemv | ~30 | 107 |
| [896,128] k/v | 0.46 | Accelerate sgemv | ~4 | 129 |
| [896,151936] lm_head | 545 | NEON-MT 8T | ~5500 | 99 |

## Fixes Applied

### Fix A: Column-parallel gemm_generic (arch-neutral)
When M < threads, partition over N instead of M. Helps all backends.

### Fix B: Wire Accelerate arm
- M>1 prefill: `cblas_sgemm` via Accelerate (reaches AMX, 2449 GFLOPS)
- M=1 decode: hybrid dispatch based on L2 residency

### Fix C: NEON GEMV with 4-row batched inner kernel
- Cache B_T[N,K] (transpose of weight B[K,N]) in MatMulPrepack OnceLock
- Each Rayon thread: 4-row-batched NEON dot products on contiguous B_T rows
- 4-row batching improves ILP (8 independent FMA chains vs 4)
- Hybrid L2-aware dispatch: `sgemv_accelerate` for L2-resident, NEON for DRAM-bound

### Fix D: Hybrid L2-aware dispatch (Accelerate for small, NEON for large)
- Runtime L2 cache query via `sysctl("hw.perflevel0.l2cachesize")`
- `is_l2_resident()` threshold = L2_bytes / 2
- Accelerate sgemv for L2-resident (106-156 GB/s)
- NEON col-parallel for DRAM-bound (66 GB/s)

## After Changes
| Configuration | Load | TTFT | p50 ms | Steady-state tok/s | Overall tok/s |
|---|---|---|---|---|---|
| ORT + CPU | 1343 ms | 118 ms | 22.0 | 45.5 | 45.87 |
| Native + CPU | 125 ms | 1120 ms | 35.8 | **27.9** | 18.9* |

*Overall includes one-time ~830 ms transpose on first decode token.

Improvement from baseline: **3.26 → 27.9 tok/s** steady-state (**8.6× speedup**).

## Remaining Gap Analysis (27.9 vs 45.5 tok/s)

Target p50: 22 ms. Current: 35.8 ms. Need to save **13.8 ms**.

### GEMV bandwidth gap: 66 vs 88 GB/s (saves 7.3 ms)
- GEMV takes 29.3 ms at 66 GB/s. At ORT's 88 GB/s: 22 ms.
- Our 66 GB/s = 33% of 197 GB/s DRAM roof; ORT's 88 = 45%.
- Root causes investigated:
  - E-core scheduling: tested with `taskpolicy -c utility`, no effect
  - Thread count: saturates at 6-8 threads, more doesn't help
  - 4-row batched kernel: 35% faster single-threaded, neutral at 8T (DRAM-limited)
  - Rayon per-call overhead: ~5 µs × 169 = 0.85 ms (3% of GEMV time)
  - ORT uses MLAS packed weight format + persistent pool, achieving higher BW

### Non-GEMV op time: 6.5 ms (saves 6.5 ms if eliminated)
- ORT has near-zero non-GEMV overhead (~0.1 ms) due to op fusion
- Our Attention (2.9 ms), Swish (1.1 ms), RMSNorm/RotaryEmb/etc. (2.5 ms) = 6.5 ms
- These are real computations, not reducible by kernel optimization alone
- Op fusion (graph-level) would amortize dispatch and eliminate intermediate buffers

### Breakdown of achievable gains

| Optimization | Saves | Classification |
|---|---|---|
| Match ORT GEMV BW (66→88 GB/s) | 7.3 ms | Needs MLAS-like packing or better pool |
| Op fusion (Attention, SiLU, etc.) | 5-6 ms | Graph-level, outside CPU EP scope |
| FP16 weights (halves bytes moved) | ~13 ms | Doubles ceiling to ~100 tok/s |
| First-token transpose elimination | 830 ms one-time | Background precompute |

### Path to beating ORT (45.5 tok/s)

**FP32 alone cannot reach 45.5 on kernel optimizations only.** Even at 88 GB/s (ORT's BW) + current 6.5 ms non-GEMV = 28.5 ms → 35.1 tok/s. Matching ORT requires BOTH higher GEMV BW AND eliminating non-GEMV overhead through graph-level fusion.

**FP16 weights would clear ORT comfortably.** At 100 GB/s effective FP16 BW, GEMV = 9.7 ms + 6.5 ms non-GEMV = 16.2 ms → 61.7 tok/s. NEON FP16 arithmetic is universal across Apple Silicon; notably Accelerate has no FP16 GEMV, so this would be our custom path.

## Answers to Five Questions

1. **Dtype**: 100% fp32 dense. Zero MatMulNBits ops. No quantization engaged.
2. **Multithreading**: Was single-threaded (M=1 → 1 Rayon chunk). Now 8-thread parallel via Rayon dense decode pool.
3. **NEON**: `simd_gemm.rs` is `cfg(x86)`-gated only. New `accelerate_gemm.rs` has NEON intrinsics (4-row batched).
4. **Accelerate/AMX**: Was unwired placeholder. Now wired: sgemm for M>1, sgemv for L2-resident M=1, NEON for DRAM-bound M=1.
5. **TTFT/prefill**: Was 38× slower because prefill also ran scalar single-threaded GEMM. Now uses Accelerate sgemm. TTFT ~1120 ms vs ORT's 118 ms — still 10× slower, attributed to non-MLAS GEMM path.

## SPMD Pool Investigation

Attempted activating the persistent SPMD decode pool for FP32:
- Default SPMD pool: 5 workers (`available/2`), dispatcher doesn't compute → worse than Rayon 8T
- Even at 10 SPMD workers: 22.15 tok/s vs Rayon's 27.9 tok/s
- Root cause: SPMD is designed for int4 (compute-bound), not FP32 (bandwidth-bound)
- Decision: keep SPMD for quantized models, use Rayon dense pool for FP32
