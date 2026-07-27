# Chew — PR #227 Numerics Review

**Branch:** `squad/mac-cpu-ep-roofline`
**Date:** 2026-07-27T01:30:00-07:00
**Author under review:** Iran
**Reviewer:** Chew (Numerics)

---

## Verdict: **APPROVE with concerns**

The four commits introduce NEON-vectorized SiLU, SDPA, and GEMV kernels plus an Accelerate sgemm integration for the native CPU EP on Apple Silicon. All 904 unit tests pass. `cargo fmt --check` passes (BLOCKING gate). `cargo clippy` passes (warnings only — dead code, cosmetic). End-to-end generation on Qwen 2.5-0.5B produces 30 tokens at ~30 tok/s without crashes or panics on M1 Max.

The numerics are **sound for production inference** but several documentation claims are inaccurate, and the SDPA NEON path lacks direct test coverage. None of the concerns below are blocking for merge, but they should be tracked for follow-up.

---

## Per-item findings

### 1. Vectorized SiLU (`activations.rs:357-436`) — NON-BLOCKING CONCERN

**The "~1 ULP" claim (line 353) is incorrect.** Measured accuracy of the Cephes-style polynomial (simulated with hardware FMA to match NEON `vfmaq_f32`):

| Range | Max ULP | Max relative error |
|---|---:|---:|
| Practical [-10, 10] | 28.0 | 3.31e-6 |
| Wide [-20, 20] | 28.3 | 3.34e-6 |
| Near zero [-0.01, 0.01] | 1.5 | 1.47e-7 |
| Clamped region [-100, -87.3] | 12.5M | ~0 abs (subnormal) |
| Positive > 88.7 | 0.0 | 0.0 |

**Assessment:** ~28 ULP in the practical range is acceptable for f32 transformer inference (effective ~17 bits of precision). The extreme-negative clamped region produces subnormal-magnitude results where the absolute error is negligible (~1e-37). **But the docstring must be corrected from "1 ULP" to "~28 ULP" or "< 1e-5 relative error".**

- `half` variable (line 372) is declared but never used — dead code from the original Cephes formulation where `floor(x+0.5)` was used for rounding; Iran replaced it with `vrndnq_f32` (NEON round-to-nearest) but didn't remove the constant.
- Non-finite fix-up (lines 423-429) is correct: re-scans the NEON-computed region and delegates NaN/Inf to the scalar reference.
- **Path verification PASSED:** inserted `panic!` at `silu_f32_neon` entry; `silu_contiguous_matches_reference` test hit it, confirming the NEON path is compiled and executed on this machine.

**Assignee for correction:** Deckard or Sapper (not Iran — locked out).

### 2. Swish→SiLU canonicalization (`activations.rs:234-246`) — SAFE

```rust
let activation = if alpha == 1.0 { Activation::Silu } else { Activation::Swish { alpha } };
```

- Uses **exact f32 equality** (`alpha == 1.0`), not epsilon. This is correct.
- Default is `unwrap_or(1.0)` — exactly 1.0f32.
- A near-1.0 alpha (e.g., 0.99999994) will NOT canonicalize to SiLU. No silent misrouting.
- Mathematically, Swish(x, β=1) = x·σ(x) = SiLU(x). Identity confirmed.

### 3. NEON SDPA (`sdpa.rs:744-820`) — NON-BLOCKING CONCERN

**Numerics are sound:**
- Softmax uses max-subtraction stability (line 502) — correct.
- `sdpa_f32_neon` reuses the existing `softmax_in_place` scalar path, so softmax stability is inherited.
- `dot_neon` and `axpy_neon` use 4×-unrolled FMA accumulators with correct tail handling (scalar fallback for remainder).
- Masked/-inf entries handled correctly: `scores.fill(0.0)` in softmax when all scores are `-inf`, and `probability == 0.0` skip in V-weighted accumulation (line 815).
- GQA grouping (`heads_per_kv`) is correct.

**Test coverage gap:** All 11 SDPA tests call `sdpa_f32_scalar` directly — **no test exercises `sdpa_f32_neon`**. Inserted a `panic!` at `sdpa_f32_neon` entry; all SDPA tests passed without hitting it. This means a bug in the NEON SDPA path would go undetected.

**Recommendation:** Add a parity test that calls `sdpa_f32(...)` (the dispatcher) and compares against `sdpa_f32_scalar(...)` for a representative set of shapes including non-power-of-2 head dims.

**Assignee for follow-up:** Pris (test owner).

### 4. GEMV correctness (`accelerate_gemm.rs`, `matmul.rs`) — SOUND

**Transpose:** The pre-transpose in `MatMulPrepack::transposed_b` (matmul.rs:100-133) produces `B_T[N,K]` row-major from `B[K,N]` row-major via `bt[j*k + i] = b[i*n + j]`. Correct. The `neon_gemv_col_parallel` kernel then computes `y[j] = dot(B_T[j,:], x)` = `Σ_i B[i,j]*x[i]` — which is `y = B^T @ x`, i.e., the correct decode GEMV.

**Tail handling:**
- `neon_gemv_batch`: processes 4 output rows at a time with 8 accumulators (2 per row, 8-wide K loop). K-tail handled by scalar fallback. N-remainder via `neon_dot` for individual rows. Correct.
- `neon_dot`: 16-element unrolled loop, 4-element secondary loop, scalar tail. Correct.
- `neon_outer_product_unrolled`: 4-row K-unrolled outer product with NEON N-vectorized inner loop, scalar N-tail. Correct.

**Accumulation:** All accumulations are f32 throughout (NEON `float32x4_t` → `vaddvq_f32` horizontal sum → f32 scalar tail).

**Accelerate sgemv removed from dispatch:** Confirmed. The `matmul_dense_into_with_backend` function at matmul.rs:795-822 dispatches M=1 to `neon_gemv_col_parallel` or `neon_gemv_parallel`, never to `sgemv_accelerate`. The `gemm_with_backend` at matmul.rs:217-224 dispatches M=1 to `neon_gemv_parallel`. No dead branch in the dispatch chain.

**Dead code in module:** `sgemv_accelerate`, `is_l2_resident`, `l2_cache_bytes`, `query_sysctl_usize`, `CBLAS_TRANS`, `cblas_sgemv` are declared but never called from production code. The compiler emits dead_code warnings. These are from the removed Accelerate sgemv path. Non-blocking: remove or mark `#[allow(dead_code)]` with a justification.

**Test guard-break PASSED:** Zeroed `y[i]` in `neon_gemv_batch` → `col_parallel_gemv_matches_reference` test failed with error 0.997. Tests are sensitive to GEMV breakage.

**Model-scale tolerance:** The `accelerate_decode_gemv_matches_generic_at_model_scale` test uses 2% relative tolerance. Measured actual max_rel: 0.018% for [1,896,896], 0.39% for [1,896,4864], 1.57% for [1,4864,896]. The 1.57% is a legitimate f32 accumulation-order difference (row-parallel outer-product reduction vs sequential tiled GEMM). The 2% tolerance accommodates this but is loose enough to mask real bugs. Non-blocking: consider tightening or comparing against a f64 reference.

### 5. `dtype.rs` f32 memcpy fast path (dtype.rs:643-664) — SAFE

Guard conditions:
1. `out.dtype == DataType::Float32` — exact dtype match
2. `out.is_contiguous()` — verified: calls `onnx_runtime_ir::is_contiguous(shape, strides)` which checks strides match computed contiguous strides exactly

A strided-to-contiguous case cannot take this path — the strides check prevents it. The length check (`data.len() != n`) prevents buffer overrun. The `validate()` call checks the output tensor invariants. Safe.

### 6. `matmul_nbits.rs` visibility change (line 1902) — SAFE

`fn spmd_decode_active()` changed from private to `pub(crate)`. This allows `accelerate_gemm.rs` to call it (line ~186) to prefer the persistent SPMD pool when active. The function only reads thread-local state (`IN_SPMD_SCOPE`) — no side effects, no new computation. Safe.

---

## Structural checks

### Silent-fallback bug class

- **NEON SiLU path:** Verified reachable. `cfg(all(not(feature = "mlas"), target_arch = "aarch64"))` is active on this M1 Max. Panic-probe confirmed.
- **NEON SDPA path:** `cfg(target_arch = "aarch64")` is active, and the dispatch at sdpa.rs:291-294 reaches it when `qk.is_none()`. However, **no unit test exercises this path** (all tests call `sdpa_f32_scalar` directly). The path IS reachable in production (the engine calls `sdpa_f32`), but it has no dedicated test coverage.
- **Accelerate GEMM paths:** `cfg(any(target_os = "macos", target_os = "ios"))` is active. Tests exercise both sgemm and NEON GEMV paths.

### Apple Silicon generality

- **No hardcoded thread counts or cache sizes.** L2 threshold is queried at runtime via `hw.perflevel0.l2cachesize` sysctl with 4 MB fallback. Thread counts come from `rayon::current_num_threads()`.
- **NEON intrinsics are ARMv8 baseline only.** All intrinsics used: `vfmaq_f32`, `vld1q_f32`, `vst1q_f32`, `vdupq_n_f32`, `vaddq_f32`, `vaddvq_f32`, `vmulq_f32`, `vnegq_f32`, `vrndnq_f32`, `vdivq_f32`, `vmaxq_f32`, `vminq_f32`, `vsubq_f32`, `vcvtq_s32_f32`, `vshlq_n_s32`, `vreinterpretq_f32_s32`. No dotprod, no FP16 arithmetic, no SME, no SVE, no BF16 intrinsics. Works on M1/M2/M3/M4 all trims.
- **500,000 element threshold** for single-thread dispatch is a heuristic. Not chip-specific.
- **TILE=64** in the transpose (matmul.rs:120) is a cache-blocking parameter, not tied to a specific L1 size.

### One implementation, no arch fork

The NEON kernels are guarded by `cfg(target_arch = "aarch64")` with scalar fallbacks on other architectures. The Accelerate integration is guarded by `cfg(any(target_os = "macos", target_os = "ios"))` with the generic GEMM fallback. This is **runtime branching behind cfg, not a fork of the kernel tree**. Intel (Resch) and ARM/QNN (Luba) share the same `gemm_with_backend` dispatcher and the same scalar reference. Acceptable.

### Parity tests

SiLU: `silu_contiguous_matches_reference` and `silu_in_range_region_is_bit_close` test the NEON path against f64 reference with ≤2e-6 / ≤1e-5 tolerances. Guard-break not directly tested on SiLU NEON (the test goes through `silu_f32_slice` which dispatches to NEON — confirmed reachable via panic probe).

GEMV: `col_parallel_gemv_matches_reference`, `row_parallel_gemv_matches_reference`, `accelerate_sgemm_matches_generic_for_small_shapes`, `accelerate_decode_gemv_matches_generic_at_model_scale`, `col_parallel_matches_at_model_scale`. Guard-break test PASSED for GEMV.

SDPA: **No parity test for the NEON path.** Gap.

---

## End-to-end coherence

Generated 30 tokens with native CPU EP on Qwen 2.5-0.5B at ~30 tok/s with prompt "The capital of France is" (temperature implicitly greedy). No crash, no panic. The compare tool does not report generated text, so direct text comparison against ORT output could not be performed. Both backends generated exactly 30 tokens.

---

## Summary of concerns (non-blocking, for tracking)

| # | Severity | Item | File:Line | Assignee |
|---|---|---|---|---|
| C1 | Low | SiLU docstring claims "1 ULP", measured ~28 ULP | activations.rs:353 | Deckard |
| C2 | Medium | SDPA NEON path has zero test coverage | sdpa.rs:291-294, 744-820 | Pris |
| C3 | Low | 7 dead code items from removed Accelerate sgemv | accelerate_gemm.rs:17,38,84,101,127,136 | Deckard |
| C4 | Low | Unused `half` variable | activations.rs:372 | Deckard |
| C5 | Low | GEMV model-scale tolerance is 2% (actual max 1.57%) | matmul.rs:1887 | Pris |
| C6 | Info | Compare tool doesn't report generated text for coherence verification | bench compare.rs | Sebastian |
