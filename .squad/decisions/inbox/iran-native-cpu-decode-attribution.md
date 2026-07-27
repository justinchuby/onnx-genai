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

Improvement from baseline: **3.26 → 27.96 tok/s** steady-state (**8.6× speedup**).

## Revised Roofline (Pris's harness, authoritative)

| Metric | Value |
|---|---|
| Measured achievable BW (8T, 256 MiB/thread) | **121.9 GB/s** |
| FP32 decode ceiling | **61.41 tok/s** |
| ORT decode | 45.83 tok/s = **74.6% of roof** |
| Native decode | 27.96 tok/s = **45.5% of roof** |

Note: Sebastian's 197 GB/s was a pure sequential-stream measurement; the achievable
GEMV bandwidth is 121.9 GB/s (Pris's probe), consistent with Sebastian's own
"achievable MT GEMV" figure of 112 GB/s. The FP32 opportunity is much thinner
than earlier estimates suggested.

## The FP32 Wall — Cannot Beat ORT with Kernel Changes Alone

| Scenario | GEMV ms | + Non-GEMV | Total | tok/s | Roof % |
|---|---|---|---|---|---|
| **Current** | 35.8 | 6.5 | 42.3 | **27.96** | 45.5% |
| Match ORT GEMV BW (91 GB/s) | 21.8 | 6.5 | 28.3 | 35.3 | 57.5% |
| **100% GEMV roof (121.9 GB/s)** | 16.3 | 6.5 | 22.8 | **43.9** | **71.5%** |
| ORT (for reference) | ~21.8 | ~0.1 | 21.9 | 45.83 | 74.6% |

**Even at theoretical maximum GEMV bandwidth, non-GEMV overhead (6.5 ms)
caps the native EP at 43.9 tok/s — below ORT's 45.83.**

ORT achieves near-zero non-GEMV overhead through op fusion (MatMul+Bias, fused
attention, fused activation). Our native EP executes 417 ops/token individually,
each with dispatch overhead and intermediate buffer allocation.

## FP16 is the Lever

| Scenario | GEMV ms | + Non-GEMV | Total | tok/s |
|---|---|---|---|---|
| FP16 @ current BW (55.5 GB/s) | 17.9 | 6.5 | 24.4 | **41.0** |
| FP16 @ ORT BW (91 GB/s) | 10.9 | 6.5 | 17.4 | **57.4** |

FP16 halves the bytes moved per token. At ORT-level GEMV bandwidth, FP16 clears
ORT by 25%. NEON FP16 arithmetic (FMLA/half) is ARMv8.2 baseline — universal
across all Apple Silicon. Accelerate has no FP16 GEMV, so this path is ours
regardless.

## Remaining Gap Analysis (27.96 vs 45.83 tok/s)

### GEMV bandwidth: 55.5 vs ~91 GB/s effective
- Pure GEMV at ~66 GB/s; total effective 55.5 GB/s (diluted by non-GEMV)
- 45.5% of achievable roof vs ORT's 74.6%
- Root causes investigated:
  - E-core scheduling: tested with `taskpolicy -c utility`, no effect
  - Thread count: saturates at 6-8 threads, more doesn't help
  - 4-row batched kernel: 35% faster single-threaded, neutral at 8T (DRAM-limited)
  - Rayon per-call overhead: ~5 µs × 169 = 0.85 ms (3% of GEMV time)
  - ORT uses MLAS packed weight format + persistent pool, achieving higher BW

### Non-GEMV op time: 6.5 ms (the hard wall)
- ORT has near-zero non-GEMV overhead (~0.1 ms) due to op fusion
- Our Attention (2.9 ms), Swish (1.1 ms), RMSNorm/RotaryEmb/etc. (2.5 ms) = 6.5 ms
- Not reducible by kernel optimization alone
- Op fusion (graph-level) would amortize dispatch and eliminate intermediate buffers

### Ranked fixes

| # | Fix | Tok/s gain | Classification |
|---|---|---|---|
| 1 | **FP16 weight GEMV** | +13-30 tok/s | Universal, THE lever |
| 2 | **Op fusion** (gate+up, QKV, MatMul+bias+act) | +5-8 tok/s | Universal, graph-level |
| 3 | **MLAS-like packed weights** (higher GEMV BW) | +4-7 tok/s | Universal |
| 4 | **Background weight transpose** | -830 ms TTFT | Universal, one-time |
| 5 | **Prefill opt** (TTFT 1105→102 ms) | 10× TTFT | Universal |

### Path to beating ORT (45.83 tok/s)

**FP32 alone cannot reach ORT with kernel-only changes.** Even at 100% GEMV roof
(121.9 GB/s) + current 6.5 ms non-GEMV = 22.8 ms → 43.9 tok/s < ORT 45.83.
This is a hard wall: the non-GEMV overhead is 6.5 ms that ORT doesn't pay
because it fuses those ops into the GEMM dispatch.

**FP16 weights clears ORT.** At current 55.5 GB/s effective BW, FP16 gives
41.0 tok/s. At ORT-level BW (91 GB/s), FP16 gives 57.4 tok/s. NEON FP16
(FMLA half-precision) is ARMv8.2 baseline — universal on Apple Silicon.
Accelerate has no FP16 GEMV, so this would be our custom kernel path.

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
