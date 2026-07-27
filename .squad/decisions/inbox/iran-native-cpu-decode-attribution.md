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

## Session 3: SDPA NEON + Dispatch Simplification

### Changes
1. **NEON SDPA fast path** (sdpa.rs):
   - Added `dot_neon()` and `axpy_neon()` — 4×-unrolled NEON intrinsics for aarch64
   - New `sdpa_f32_neon()` function using NEON dot/AXPY for QK and AttnV inner loops
   - Attention: 111 µs/call → 75 µs/call (32% faster), saving **0.86 ms per token**
   - Same bug class as the original GEMV scalar fallback: `dot_f32` and `axpy_f32` 
     had AVX2 paths for x86 but fell through to scalar on aarch64

2. **Unified GEMV dispatch** (matmul.rs):
   - Removed Accelerate sgemv L2-resident path
   - Measured: Accelerate sgemv has ~30-50 µs GCD thread wake-up overhead, making it
     equivalent to Rayon NEON for L2-resident matrices
   - All M=1 decode now routes to NEON col-parallel (neutral on performance, simpler)

### Updated Per-Op Attribution (post-session 3)

| Op Type | Count | ms/token | µs/call | % of decode | Change |
|---|---|---|---|---|---|
| MatMul | 49 | 16.9 | 345 | 49% | — |
| FusedMatMulBias | 120 | 12.6 | 105 | 37% | — |
| **Attention** | **24** | **1.8** | **75** | **5%** | **-0.86 ms** |
| Swish | 24 | 1.0 | 43 | 3% | — |
| Other | ~200 | ~1.2 | — | 3% | — |
| Session overhead | — | ~0.8 | — | 2% | — |
| **Total** | **~417** | **~34.5** | — | **100%** | **-1.2 ms** |

### Updated Measurements (Pris compare harness, 5 runs, median)

| Configuration | p50 ms | tok/s | Roof % | Effective GB/s |
|---|---|---|---|---|
| Native + CPU (session 3) | 34.3 | **29.17** | 47.6% | ~58 |
| Native + CPU (session 2) | 35.7 | 27.96 | 45.5% | ~55.5 |
| ORT + CPU | 22.0 | **45.82** | 74.7% | ~91 |

### Updated FP32 Wall (with session 3 non-GEMV reduction)

| Scenario | GEMV ms | + Non-GEMV | Total | tok/s | Roof % |
|---|---|---|---|---|---|
| **Current** | 29.5 | 5.0 | 34.5 | **29.0** | 47.3% |
| Match ORT GEMV BW (91 GB/s) | 21.8 | 5.0 | 26.8 | 37.3 | 60.8% |
| **100% GEMV roof (121.9 GB/s)** | 16.3 | 5.0 | 21.3 | **46.9** | **76.5%** |
| ORT (for reference) | ~21.8 | ~0.1 | 21.9 | 45.83 | 74.6% |

Progress: reduced non-GEMV from 6.5 → 5.0 ms. At 100% GEMV roof, native EP 
would now reach **46.9 tok/s — just barely above ORT's 45.83**. But achieving 
100% GEMV roof requires closing a 30% gap (66 → 95+ GB/s), which is limited by
Rayon fork-join overhead vs ORT's MLAS persistent pool.

### What Didn't Work This Session

1. **Accelerate sgemv for L2-resident**: 30-50 µs GCD overhead per call makes it 
   equivalent to NEON multi-threaded for small matrices. Not a win.
2. **L2-aware single-threaded threshold**: Routing q/o [896,896] to single-threaded 
   NEON was slightly WORSE than multi-threaded. L2-resident matrices are still 
   large enough that 8T parallelism helps.
3. **Persistent barrier pool (GCD/pthread)**: Deadlocked in standalone test, but the 
   concept is sound — Rayon's ~5 µs per fork-join × 169 calls = 0.85 ms overhead.


## Session 4 Update — Dispatch Overhead Reduction

**Authoritative harness result: 31.30 tok/s (50.7% of roof) — up from 29.17 tok/s (+7.3%)**

### Optimizations Applied

| Change | Savings | Scope |
|---|---|---|
| f32 memcpy fast path in `write_dense_f32_narrow` | ~1.5 ms/token | Universal (all architectures) |
| NEON SiLU vectorization (Cephes exp, ~1 ULP) | ~0.8 ms/token | aarch64 (scalar fallback elsewhere) |
| Swish(1.0) → Silu canonicalization | ~0.2 ms/token | Universal |
| Redundant `matmul_geometry` elimination | ~0.1 ms/token | Universal |
| FMB fast 1-D bias add | ~0.1 ms/token | Universal |
| **Total** | **~2.7 ms/token** | |

### Updated Per-Op Breakdown (31.5 ms/token steady-state)

| Op Type | Count | ms/token | % of decode | vs Session 3 |
|---|---|---|---|---|
| MatMul | 49 | 16.5 | 54.2% | -0.4 ms |
| FusedMatMulBias | 120 | 11.3 | 37.2% | -1.3 ms |
| Attention | 24 | 1.26 | 4.2% | -0.55 ms |
| RMSNormalization | 49 | 0.36 | 1.2% | -0.10 ms |
| Swish | 24 | 0.25 | 0.8% | -0.78 ms |
| Other | 151 | 0.87 | 2.9% | ~same |
| **Total** | **417** | **30.5** | **100%** | **-3.1 ms** |

### FP32 Wall Analysis (revised)

| Scenario | GEMV ms | Non-GEMV ms | Total ms | tok/s | Roofline % |
|---|---|---|---|---|---|
| **Current** | 27.8 | 3.5 | 31.3 | **31.9** | **51.7%** |
| Non-GEMV → 1 ms | 27.8 | 1.0 | 28.8 | 34.7 | 56.2% |
| GEMV at ORT's 91 GB/s | 21.8 | 3.5 | 25.3 | 39.5 | 64.0% |
| GEMV at 91 GB/s + non-GEMV → 1 ms | 21.8 | 1.0 | 22.8 | 43.9 | 71.1% |
| GEMV at 100% roof (122.5 GB/s) + 1 ms | 16.2 | 1.0 | 17.2 | 58.1 | 94.1% |
| ORT (for reference) | ~21.8 | ~0.1 | ~21.9 | 45.96 | 74.5% |

### Gap to ORT — Two Independent Bottlenecks

1. **GEMV BW: 62 GB/s vs 91 GB/s (68% of ORT)**
   - MLAS uses hand-tuned ARM assembly GEMV kernels
   - ORT's intra-op thread pool has lower fork-join overhead than Rayon (~2 µs vs ~5 µs per dispatch)
   - ORT fuses gate+up projections into single GEMV, halving dispatches

2. **Non-GEMV overhead: 3.5 ms vs ~0.1 ms (35× worse)**
   - ORT fuses entire subgraphs (attention, norm, activation) into mega-ops
   - Our EP dispatches 417 individual ops with per-op executor overhead
   - Not fixable at the kernel level — requires graph-level fusion

### Conclusion

**FP32 native decode is unlikely to beat ORT (45.96 tok/s) without graph-level op fusion.**

Even at 100% GEMV roof AND non-GEMV reduced to 1 ms, we reach 58 tok/s. But our GEMV realistically caps at ~80-85 GB/s (without MLAS-quality kernels), giving ~40 tok/s even with perfect non-GEMV.

**The honest FP32 ceiling with current architecture: ~35-40 tok/s.** This requires GEMV at 80+ GB/s (via better prefetching, reduced Rayon overhead, or graph-level GEMV batching) + non-GEMV reduced to ~1 ms (via op fusion).

### Next Lever: FP16

FP16 model exists at `models/qwen2.5-0.5b-f16` (959 MB — half the bytes).
Sebastian measured FP16 NEON at 46.3 tok/s with pthread spawn, ~97 tok/s projected with persistent pool.
Must compare native-FP16 vs ORT-FP16 per Justin's fairness rule.

---

## Session 6 — batch_shape dispatch bug fix + FMB direct output

**Date:** 2026-07-27T09:25Z

### Critical Discovery: batch_shape dispatch bug

The Accelerate M=1 GEMV fast path in `matmul_dense_into_with_backend` checked
`geom.batch_shape.is_empty()`, but during decode with input shape [1,1,K],
`batch_shape = [1]` (not empty). This caused ALL GEMV calls to fall through
to `gemm_with_backend` → `neon_gemv_parallel` (outer product approach) instead
of the optimized `neon_gemv_col_parallel` (dot product with pre-transposed B_T).

**Evidence:** CPU sampling confirmed 672/672 GEMV samples in `neon_gemv_parallel`,
0 in `neon_gemv_col_parallel`. After fix: 247/247 samples in `neon_gemv_col_parallel`.

**Fix:** `numel(&geom.batch_shape) <= 1` treats single-element batch shapes as
non-batched. Also applied to the general non-batched path (line 826).

### Performance results (commit d65e5c38)

| | p50 ms/tok | tok/s | Eff. GB/s | Roof % |
|---|---|---|---|---|
| Before (session 5) | 32.5 | 30.8 | 61 | 55% |
| **After (session 6)** | **29.7** | **33.7** | **65** | **60%** |
| ORT | 22.2 | 45.0 | 87 | 78% |
| Ceiling | 17.3 | 56.7 | 112 | 100% |

### Also: FMB direct output path

FusedMatMulBias now writes directly into the output tensor when eligible
(contiguous f32, no alias), skipping Vec<f32> allocation + write_dense_f32_narrow
copy for 120 calls/token. Measured at parity — the allocation overhead was
already small (~200 µs), but eliminates unnecessary allocation traffic.

### Accelerate sgemv experiment — NEGATIVE result

Tested Accelerate cblas_sgemv for L2-resident attention projections. Result:
GCD wake-up overhead (~30-50 µs per call) dominates compute saving. For [896,896]
at 3.2 MB: Accelerate 58 µs (18 µs compute + 40 µs wake-up) vs single-thread
NEON 49 µs. Net negative — reverted.

### Remaining gap analysis (29.7 ms vs ORT 22.2 ms = 7.5 ms gap)

1. **GEMV bandwidth:** ~75 GB/s (col-parallel NEON) vs ~91 GB/s (MLAS) = 4.5 ms gap
   - MLAS uses hand-written aarch64 assembly with explicit prefetch
   - Our NEON intrinsics generate good code (5 ldp + 8 fmla) but ~20% lower BW
2. **Non-GEMV overhead:** ~3.5 ms vs ~1 ms = 2.5 ms gap
   - Graph executor dispatches 168 individual ops per token
   - ORT fuses subgraphs into fewer mega-ops
3. **Both must improve to beat ORT in FP32**

### Updated fix ranking

| Priority | Fix | Est. gain | Status |
|---|---|---|---|
| 1 | MLAS-quality GEMV kernel (prefetch, tile) | 3-5 ms/tok | Not started |
| 2 | Graph-level op fusion (reduce 168→~50 ops) | 2-3 ms/tok | Architecture change |
| 3 | FP16 NEON GEMV (halve bytes moved) | 2× ceiling | Next lever |

---

## Session 7: Final FP32 attribution — GEMV sequence benchmark

**Date:** 2026-07-26

### Pure GEMV sequence benchmark (isolating kernel from framework)

Ran a standalone benchmark simulating the full Qwen2.5 decode pattern:
169 GEMV calls (24 layers × 7 projections + LM head) through a Rayon pool,
no graph executor, no tensor binding, no shape resolution.

| Measurement | Time | GB/s | Roof % |
|---|---|---|---|
| **Full 169-call GEMV sequence** | **24.35 ms** | **81.1** | **72%** |
| 48× gate/up [896,4864] isolated | 7.58 ms | 110.4 | 99% |
| 1× gate [896,4864] isolated | 0.175 ms | 99.8 | 89% |

**Key finding: the NEON GEMV kernel achieves 81 GB/s when measured
without framework overhead — within 10% of ORT's ~89 GB/s.**

The drop from 99 GB/s (single call) to 81 GB/s (full sequence) is due to:
- Small matrices ([896,128] K/V projections) that don't parallelise well
- The massive LM head ([896,151936]) that dominates with less bandwidth efficiency
- Inter-call Rayon dispatch overhead across 169 calls

### Per-op decode breakdown (ONNX_GENAI_PROFILE_OPS=1, steady-state token)

| Op | Calls/token | Total ms | % of decode |
|---|---|---|---|
| MatMul | 49 | 14.80 | 52.3% |
| FusedMatMulBias | 120 | 11.16 | 39.4% |
| Attention | 24 | 1.21 | 4.3% |
| RMSNormalization | 49 | 0.28 | 1.0% |
| Swish | 24 | 0.23 | 0.8% |
| RotaryEmbedding | 48 | 0.19 | 0.7% |
| Mul | 24 | 0.18 | 0.6% |
| Constant | 96 | 0.10 | 0.4% |
| **Total (executor)** | **434** | **28.32** | **100%** |

### Gap decomposition (native 30 ms vs ORT 22 ms = 8 ms gap)

| Source | Our cost | ORT cost | Gap | % of gap |
|---|---|---|---|---|
| GEMV pure bandwidth | 24.4 ms | ~21.7 ms | 2.7 ms | 34% |
| Per-op framework overhead | 1.6 ms | ~0 ms | 1.6 ms | 20% |
| Non-GEMV computation | 2.3 ms | ~0.5 ms | 1.8 ms | 23% |
| Non-graph overhead | 1.7 ms | ~0 ms | 1.7 ms | 21% |

ORT's near-zero non-GEMV cost comes from fused kernels (MatMul+Bias+Activation
in MLAS handles bias/activation while data is still in cache) and significantly
fewer graph nodes (~50 fused ops vs our 434 individual ops).

### Experiments attempted and abandoned (session 7)

1. **Accelerate cblas_sgemv for L2-resident attention projections** — NEGATIVE
   - GCD wake-up overhead (~30-50 µs/call) exceeds compute savings
   - [896,896]: Accelerate 58 µs vs NEON 49 µs
2. **L2-based single-thread threshold** — NEGATIVE
   - Col-parallel with 8 threads beats single-thread even for L2-resident shapes
3. **Software prefetch (prfm pldl1strm)** — NEGATIVE
   - M1's hardware prefetcher handles sequential access better; SW prefetch 40% slower
4. **Persistent spin-wait pool (session 5, re-confirmed session 7)** — NEGATIVE (~3% improvement)
   - Rayon IS a persistent pool with ~3 µs per-call cost (not 30-50 µs as initially projected)
   - Custom sense-reversing barrier pool only 3% better

### Conclusions

**FP32 native cannot beat ORT without two structural changes:**
1. MLAS-quality GEMV assembly (~10% bandwidth gap, 2.7 ms)
2. Graph-level op fusion (~3.4 ms from framework + non-GEMV overhead)

**The GEMV kernel is NOT the bottleneck.** At 81 GB/s pure, it's at 72% of roof.
The bottleneck is the graph executor dispatching 434 individual ops per token
vs ORT's ~50 fused ops.

**Recommended next step: FP16 NEON GEMV.**
- Model at 959 MB → GEMV ceiling at 81 GB/s would give ~85 tok/s
- ORT on FP16: ~42 tok/s (widens to FP32)
- Path to 2× over ORT is clear
- FP16 storage + FP32 accumulate for numerics safety
