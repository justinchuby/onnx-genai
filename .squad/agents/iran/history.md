# Iran — History

## Project Context (joined day)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **State when joined:** Native CUDA EP beats/parity ORT on several Foundry models; correctness suite green (int8/block32 f64-adjudicated in #190). Team reorganized into pods; CPU & Edge pod formed to broaden hardware coverage beyond CUDA/Metal.
- **Role:** Mac CPU Optimization Engineer — Apple Silicon CPU-EP perf (NEON, Accelerate/AMX), aarch64-apple-darwin GEMV/GEMM hot paths.
- **Requested by:** Justin Chu
- **Joined:** 2026-07-26

## 2026-07-26 — Joined the team
Cast into the CPU & Edge pod. Standing directive: optimizations must be portable (consumer/edge hardware, not just H200); every perf claim backed by a benchmark; SIMD/NPU paths must match the scalar/f64 reference within a justified tolerance and be locked with regression tests.

## 2026-07-27: Pre-transposed Column-Parallel NEON GEMV Implementation

### Changes committed (squad/mac-cpu-ep-roofline, commit 487d1aff)
1. **`accelerate_gemm.rs` (new)**: Apple Accelerate BLAS FFI + column-parallel NEON GEMV
   - `sgemm()`: cblas_sgemm for M>1 prefill (reaches AMX coprocessor)
   - `neon_gemv_col_parallel()`: Rayon-parallel NEON dot products on pre-transposed B_T[N,K]
   - `neon_gemv_parallel()`: Row-parallel fallback when transposed B unavailable
   - 4 internal tests (sgemm parity, col-parallel small/model-scale, row-parallel)

2. **`matmul.rs` modifications**:
   - `MatMulPrepack.transposed_b`: OnceLock cache for B[K,N]→B_T[N,K] transpose
   - Accelerate arm in `gemm_with_backend`: sgemm for M>1, neon_gemv_parallel for M=1
   - Pre-transposed GEMV dispatch in `matmul_dense_into_with_backend`
   - `gemm_generic_col_parallel`: column-parallel generic GEMM for small M (arch-neutral)
   - `prepack` parameter now unconditional (not behind cfg(mlas))
   - 3 parity tests (accelerate_sgemm, accelerate_decode_gemv, col_parallel_gemm)

## 2026-07-27 (session 2): Hybrid L2 dispatch + 4-row batched kernel

### Changes (on top of 487d1aff, uncommitted)
1. **Hybrid L2-aware dispatch** (`matmul.rs:810-834`):
   - Runtime L2 cache query via `sysctl("hw.perflevel0.l2cachesize")`
   - L2-resident matrices → `sgemv_accelerate()` (106-156 GB/s via AMX)
   - DRAM-bound matrices → `neon_gemv_col_parallel()` (66 GB/s 8T NEON)
   - Fallback when no pretranspose → `neon_gemv_parallel()`

2. **`sgemv_accelerate()`** (`accelerate_gemm.rs`): cblas_sgemv FFI wrapper

3. **4-row batched NEON inner kernel** (`neon_gemv_batch`):
   - Processes 4 output rows simultaneously, sharing x reads across rows
   - 8 independent FMA chains (2 per row) for better ILP
   - 35% faster single-threaded, neutral at 8T (DRAM-limited)
   - Correct scalar tail handling for arbitrary K

4. **`spmd_decode_active()` made pub(crate)** for SPMD pool dispatch

### Performance results (Qwen2.5-0.5B, M1 Max)
- Steady-state p50: **35.8 ms (27.9 tok/s)** — 8.6× over 3.26 baseline
- Overall decode: 18.9 tok/s (includes 830 ms first-token transpose)
- ORT reference: 45.87 tok/s / 22.0 ms p50
- All 126 matmul tests pass, fmt clean, clippy clean

### Investigation results
- **SPMD pool not effective for FP32**: 5 workers (avail/2) + no dispatcher compute → worse than Rayon 8T
- **E-core scheduling**: no effect (tested with taskpolicy)
- **Thread count saturation**: 6-8 threads optimal, more doesn't help
- **Accelerate sgemv pathology**: 35-59 GB/s for DRAM-bound (91% of weights), 137-156 GB/s L2-resident only
- **Standalone micro-benchmark**: 82-89 GB/s achievable with 4-row NEON 8T, but in-model only 66 GB/s (Rayon overhead + inter-op cache effects)

### Remaining gap (27.9 vs 45.5 tok/s) — see attribution report
- GEMV BW: 66 vs ORT's 88 GB/s (saves 7.3 ms if matched)
- Non-GEMV ops: 6.5 ms (ORT ~0.1 ms due to op fusion)
- **FP32 alone cannot reach 45.5 with kernel-only changes** — needs graph-level op fusion or FP16 weights
- FP16 projected: ~62 tok/s (doubles ceiling, NEON FP16 universal on Apple Silicon)

## 2026-07-27: Session 3 — SDPA NEON + Dispatch Simplification (commit c1dbc71f)

### Changes
1. **NEON SDPA fast path** (sdpa.rs): Added `dot_neon()`, `axpy_neon()`, and `sdpa_f32_neon()`
   - Same bug class as original GEMV scalar fallback: `dot_f32`/`axpy_f32` had AVX2 for x86 but scalar on aarch64
   - Attention: 111 µs/call → 75 µs/call, saving 0.86 ms per token
2. **Unified M=1 GEMV dispatch** (matmul.rs): Removed Accelerate sgemv L2-resident path
   - Measured: Accelerate sgemv has ~30-50 µs GCD overhead, equivalent to Rayon NEON
   - All M=1 decode now uses NEON col-parallel (simpler, no oversubscription hazard)

### Results (Pris compare harness, 5 runs, median)
- **Native: 29.17 tok/s** (was 27.96, +4.3%), 47.6% of roof
- ORT: 45.82 tok/s, 74.7% of roof
- 904 unit tests pass, cargo fmt clean, clippy clean

### Key finding: FP32 wall is now tighter
- Non-GEMV reduced from 6.5 → 5.0 ms (Attention savings)
- At 100% GEMV roof: 16.3 + 5.0 = 21.3 ms → 46.9 tok/s — barely clears ORT
- But reaching 100% GEMV requires closing 30% BW gap (66 → 95+ GB/s)
- **Conclusion: FP16 remains the lever to definitively beat ORT**

## Session 4 — Dispatch Overhead Reduction (2026-07-27)

**Campaign:** PR #227 (squad/mac-cpu-ep-roofline)

### Result
- **31.30 tok/s** (50.7% of roof), up from 29.17 tok/s (+7.3%)
- **9.6× improvement** from baseline 3.26 tok/s
- Still 0.681× ORT (45.96 tok/s)

### Changes (commit `77296fab`)
- `dtype.rs`: Contiguous f32→f32 memcpy fast path in `write_dense_f32_narrow`
- `activations.rs`: NEON SiLU (Cephes exp polynomial) + Swish(1.0)→Silu canonicalization
- `matmul.rs`: Eliminate redundant `matmul_geometry` computation, remove dead code
- `fused_matmul_bias.rs`: Fast 1-D bias add path

### Key Findings
- `write_dense_f32_narrow` was doing redundant Vec copy + per-element strided write for f32→f32: **1.5 ms/token waste**
- Swish(alpha=1.0) wasn't using the SiLU fast path: **0.8 ms/token from scalar exp()**
- `broadcast_apply` in FMB does multi-dim index walk for what should be simple vector add
- **FP32 wall confirmed**: even at 100% GEMV roof + 1 ms non-GEMV, native EP reaches ~58 tok/s ceiling. But realistic FP32 achievable is ~35-40 tok/s due to GEMV BW (~80 GB/s max) + op dispatch overhead.
- Gap to ORT is TWO independent bottlenecks: GEMV kernel quality (62 vs 91 GB/s) AND non-GEMV dispatch (3.5 vs 0.1 ms)

### Tests
- 920 tests pass (904 + 10 + 1 + 1 + 4)
- `cargo fmt --all -- --check` clean
- `cargo clippy -p onnx-runtime-ep-cpu` clean

## 2026-07-27T09:25:00Z — Session 6: batch_shape dispatch bug + FMB direct output

**Context:** Investigating why GEMV bandwidth was 69 GB/s (55% of roof) despite the NEON
kernel benchmarking at 75-86 GB/s in isolation.

**Critical finding:** CPU sampling revealed ALL decode GEMV calls were going through
`neon_gemv_parallel` (outer product, non-transposed B) instead of `neon_gemv_col_parallel`
(dot product, pre-transposed B_T). Root cause: `batch_shape.is_empty()` check excluded
inputs with shape [1,1,K] (which have batch_shape = [1], not empty).

**Fix:** Changed to `numel(&batch_shape) <= 1` in both the Accelerate M=1 fast path and
the general non-batched path. Also added FusedMatMulBias direct output (GEMV into output
tensor, bias in-place, skip Vec alloc + copy for 120 calls/token).

**Results:** p50 32.5 → 29.7 ms, 30.8 → 33.7 tok/s (+9.4%), 55% → 60% of roof.
**Commit:** d65e5c38

**Negative result:** Accelerate sgemv for L2-resident matrices — GCD wake-up overhead
(~40 µs) dominates compute saving for all our matrix sizes. Reverted.
