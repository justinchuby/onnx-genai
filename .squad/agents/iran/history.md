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

## 2026-07-27T12:30:00Z — Session 7: Final FP32 attribution + GEMV sequence benchmark

**Context:** Coordinator asked for persistent spin-wait pool and gate+up fusion.
Session 5 already showed Rayon IS persistent (~3 µs/call, not 30-50 µs). This
session focused on isolating where the remaining 8 ms gap (30 ms native vs 22 ms ORT)
actually lives.

**Key measurement: Pure GEMV sequence benchmark**
Ran standalone 169-call GEMV sequence (same shapes as full model decode) through Rayon:
- Full sequence: **81.1 GB/s** (72% of roof) = 24.35 ms
- 48× gate/up isolated: **110.4 GB/s** (99% of roof)
- 1× gate isolated: **99.8 GB/s** (89% of roof)

**Insight:** The NEON GEMV kernel is within 10% of ORT's bandwidth when framework
overhead is removed. The remaining gap is NOT in the kernel — it is in the graph
executor dispatching 434 individual ops per token vs ORT's ~50 fused ops.

**Per-op decode breakdown (profiled with ONNX_GENAI_PROFILE_OPS=1):**
- MatMul + FMB: 26.0 ms (91.8% of decode)
- Attention: 1.2 ms (4.3%)
- All other (RMSNorm, Swish, Mul, RotaryEmb, Constants): 1.1 ms (3.9%)
- Total executor: 28.3 ms

**Gap decomposition (8 ms gap = native 30 ms vs ORT 22 ms):**
1. GEMV bandwidth (81 vs 89 GB/s): 2.7 ms (34%)
2. Per-op framework overhead (168 dispatches × 9.5 µs): 1.6 ms (20%)
3. Non-GEMV computation (Attention, RMSNorm, etc.): 1.8 ms (23%)
4. Non-graph overhead (KV, sampling, etc.): 1.7 ms (21%)

**Experiments — all NEGATIVE:**
- Accelerate sgemv for L2-resident matrices: GCD overhead dominates
- L2-based threshold: col-parallel beats single-thread even for small matrices
- Software prefetch: 40% slower (M1 HW prefetcher is better)
- Persistent spin-wait pool: ~3% improvement over Rayon (not the bottleneck)

**Conclusion:** FP32 cannot beat ORT without MLAS-quality assembly AND graph-level
fusion. Both require significant infrastructure changes. Recommended FP16 as next lever
(model at 959 MB → projected ~85 tok/s, ORT on FP16 gets ~42 tok/s).

**Commits this session:** 708a672c (docs only, attribution update)

## 2026-07-27T09:50:00Z — Session 8: SPMD pool for FP32 + cleanup

**Context:** Coordinator insisted on persistent pool despite session 5 showing only 3%
improvement. Key insight: session 5 tested BEFORE batch_shape fix → GEMVs went through
wrong path (outer product). After batch_shape fix, SPMD pool IS effective for col-parallel.

**The breakthrough: SPMD pool with P_cores - 1 workers**
- Tested SPMD pool thread counts: 5=37, 7=43.5, 8=41, 9=36 tok/s
- Optimal: 7 workers (P_cores - 1 = 8 - 1) — dispatcher + 7 workers = 8 P-cores
- Added `performance_core_count()` via hw.perflevel0.physicalcpu for Apple Silicon
- Enabled auto-calibration for dense FP32 models (was restricted to quantized only)

**Results (M1 Max, Qwen2.5-0.5B FP32):**
| | p50 ms | tok/s | GB/s | Roof% |
|---|---|---|---|---|
| ORT (baseline) | 22.1 | 45.2 | ~87 | 78% |
| **Native (SPMD)** | **23.5** | **42.6** | **~82** | **73%** |
| Native (Rayon, before) | 30.0 | 33.3 | ~65 | 58% |
| Native (original) | ~307 | 3.26 | ~6 | 5% |

**Improvement: 13× from baseline (3.26 → 42.6 tok/s), 94% of ORT.**

**Chew review items folded in:**
- C1: Fixed SiLU docstring (was ~1 ULP, measured ~28 ULP)
- C3: Removed 7 dead Accelerate sgemv items + 3 dead tests
- C4: Removed unused `half` variable

**Commit:** 3a88ba8c

## 2026-07-27T10:15:00Z — Session 8 cont: FP16 GEMV + NEON bulk conversion

**THE WIN: Native FP16 is 27% faster than ORT's best.**

| Backend | Model | p50 ms | Steady tok/s | vs ORT best |
|---|---|---|---|---|
| ORT | FP32 | 22.2 | 45.0 | — |
| ORT | FP16 | 24.5 | 40.8 | — |
| Native | FP32 | 24.2 | 41.3 | 0.92× |
| **Native** | **FP16** | **17.4** | **57.5** | **1.27×** |

**Why ORT cannot compete on FP16:** ORT's CPU EP widens FP16→FP32 before GEMM
(no native FP16 kernel). It pays the conversion cost and gets no bandwidth
benefit — actually 8% SLOWER than FP32. Our kernel reads FP16 directly from
mmap via NEON fcvtl (ARMv8 base, all Apple Silicon), halving memory bandwidth.

**Implementation:**
1. `neon_gemv_f16_col_parallel`: f16 weight GEMV, same dispatch as f32 (SPMD pool)
2. `load_f16x4_to_f32x4`: inline-asm fcvtl wrapper (stable Rust, no nightly)
3. `MatMulPrepack::transposed_b_f16`: lazy f16 transpose cache
4. `neon_f16_to_f32_bulk` / `neon_f32_to_f16_bulk`: bulk conversion for non-GEMV ops
5. Dispatch in both MatMulKernel and FusedMatMulBiasKernel

**Critical finding: NEON bulk f16↔f32 was load-bearing.** Without it, non-GEMV
ops (RMSNorm, Swish, Attention) fell through to scalar conversion, adding
~1.8 ms/token overhead that erased the GEMV bandwidth savings. FP16 was
actually SLOWER than FP32 before adding the bulk conversion.

**906 tests pass, cargo fmt clean, clippy clean.**
**Commit:** 75311827

### Session 8 — Profile regeneration + README update

Regenerated all CPU profiles and added FP16 pair:

| Backend | Decode tok/s | Model |
|---|---|---|
| ORT+CPU | 44.6 | FP32 |
| ORT+CPU f16 | 40.6 | FP16 |
| native | 33.6 | FP32 |
| **native f16** | **43.9** | **FP16** |

Native FP16 beats ORT FP16 (43.9 vs 40.6, like-for-like). Updated
`check_profile_table.py` to validate 6 columns. Rewrote README prose:
documented FP16 architectural advantage (direct mmap read vs widen-before-GEMM),
honest TTFT weakness (1174–1291 ms vs 109–124 ms for ORT), further headroom
notes (gate+up fusion, op fusion, prefill, Q4).

Machine was under heavy load from concurrent agents. Used
`ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=1` to force the SPMD pool (auto-calibrator
falls back to flat under load). Used 200 tokens to dilute the ~1s pool init spike.

`check_profile_table.py` passes (6 samples × 4 rows). `cargo fmt --check` clean.

### Session 8 — FP16 Discrepancy Resolution

Coordinator stopped docs work: Fact Checker measured native FP16 at 36.1 tok/s
vs my 57.5 claim. Three of four cells reproduced; only native FP16 diverged.

**Root cause: auto-calibrator under load.** Under system contention, the SPMD
pool auto-calibrator selects the flat (single-threaded) path. This specifically
devastates native FP16 (loses multi-threaded bandwidth advantage) while ORT
(MLAS, no auto-calibrator) and native FP32 are barely affected.

**Re-measurement on quiet machine (load avg <6), FC's exact protocol:**

| Backend | Decode tok/s | Spread |
|---|---|---|
| Native FP16 | **59.78** | [58.77, 59.81] |
| ORT FP16 | 42.33 | [42.06, 42.41] |
| Native FP32 | 42.07 | [41.96, 42.24] |
| ORT FP32 | 45.91 | [45.84, 46.06] |

Native FP16/ORT FP16 = **1.41×**. Native FP16/ORT FP32 = **1.30×**.
<2% CoV across 5 runs. Number exceeds original 57.5 claim.

**Metric clarification:** Original 57.5 was 1000/p50_ms. Correct throughput
(tokens/total_time from compare harness) is 59.78. The p50 metric underestimated
because it ignores the distribution shape.

**500-token non-determinism: cannot reproduce on quiet machine.** Both auto-cal
and forced pool produce byte-identical tokens at 500 tokens. FC's non-determinism
was from auto-calibrator path-switching under load (flat vs pool → different
floating-point reduction order → different logits → different argmax).

**TTFT still ~10× worse** (1070 ms vs 107 ms). Known, documented weakness.

## Session 9 — Calibrator freeze + profile regeneration (2026-07-27)

**Priority 1 — Load testing (complete):** Measured forced-pool vs auto-cal vs
forced-flat under moderate load (4 `yes` processes, ~25% idle). Forced flat
wins at 32.55 tok/s; forced pool worst at 19.43 (spin-wait steals CPU).
Conclusion: auto-calibrator IS correct; pool cannot be default under load.

**Priority 2 — Calibrator freeze (commit `177e8a73`):**
- Removed `CALIB_RECAL_PERIOD` re-probe mechanism — path frozen permanently
  once committed
- Fixed false "token-exact" claims in module and `AutoPath` docs
- Replaced re-probe test with permanent-commitment test
- All 906 tests pass, `cargo fmt --check` clean

**Priority 3 — Profile regeneration (commit `d8793f33`):**
- Regenerated all 4 CPU profiles (ORT FP32/FP16, native FP32/FP16) on quiet
  machine after calibrator freeze
- Updated README table and prose with verified numbers
- Native FP16: 43.6 tok/s decode (p50 steady-state 57.8 tok/s) vs ORT FP16
  40.5 tok/s — architectural win via direct FP16 read
- TTFT weakness documented: 1023-1366 ms vs 114-119 ms (~10× worse)
- `check_profile_table.py` passes: 6 samples × 4 rows all agree
