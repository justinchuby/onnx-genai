# Chew — PR #353 Thin-M GEMM Review

**Date:** 2026-07-28T12:45:00Z
**PR:** #353 (`squad/thin-m-gemm-prefill`)
**Author:** Iran
**Verdict:** 🟢 **APPROVE** with one non-blocking documentation nit (C1)

## Scope

Two changes: (1) NEON column-parallel thin-M GEMM for f32 prefill M=2..16 on
Apple Silicon; (2) f32 weight transpose precomputation at model load.

## Findings

### 1. Numerical equivalence — SOUND

The `neon_dot4_bt` kernel uses ARMv8 baseline `vfmaq_f32` (fused multiply-add)
with two 128-bit accumulators per dot product, merged via `vaddq_f32 + vaddvq_f32`.
Remainder elements use scalar multiply-accumulate.

Test `f32_thin_m_numerics_match_cblas_reference` passes all three shapes against
an f64 reference with `rel_err < 1e-5`. Shapes tested: (7,64,65536),
(4,128,50257), (16,32,100000). Standard error analysis for K=768 f32 FMA
accumulation gives worst-case relative error ~4.6e-5 (767 × 2⁻²⁴), but
practical error on model weights is consistently ~1e-7 (measured in PR #227
GEMV review). The tolerance is acceptable for inference.

### 2. Threshold correctness — CORRECT

- `thin_m_gemm_eligible`: `m >= 2 && m <= 16 && k.saturating_mul(n) > 4_000_000`
- M=1 excluded (handled by existing NEON GEMV). M=17+ falls through to cblas_sgemm.
- K×N overflow handled by `saturating_mul`. K=1 or N=1 cannot exceed 4M threshold.
- Precompute threshold uses `(k as u64) * (n as u64) <= THRESHOLD as u64` — consistent
  with the eligibility check. Zero-element guard present.
- Non-constant B returns None from `transposed_b`, falling through to cblas_sgemm. ✓
- Batched shapes excluded by `numel(&geom.batch_shape) <= 1` guard. ✓
- Only rank-2 B by `geom.b_promoted_rank == 2` guard. ✓

### 3. FP16 BNNS path — INDEPENDENTLY VERIFIED UNAFFECTED

Test `fp16_m_ge2_prefill_reaches_bnns_not_half_gemm` passes (counter fires).
Structural confirmation: the thin-M path lives in `matmul_dense_into_with_backend`
(f32 only), reached only after `try_matmul_half` has already handled/rejected f16
inputs. The f32 precompute loop in `build.rs` filters on
`DataType::Float32`, so f16 weights are never touched.

### 4. Transpose precompute lifecycle — CORRECT

- `WEIGHT_TRANSPOSE_F32` is process-global `LazyLock<Mutex<HashMap<usize, Arc<Vec<f32>>>>>`.
- `clear_weight_transpose_caches()` clears both f16 and f32 caches on `Executor::Drop`.
- Precompute inserts `Arc<Vec<f32>>`; kernel `transposed_b` OnceLock holds a clone.
  Both are released on Drop — no leak, no dangling pointer, no unbounded growth across
  Engine create/destroy cycles.
- Duplicate-insert guard via `contains_key` before transpose. ✓
- `debug_assert_eq!` on buffer size vs expected k×n×4. ✓

### 5. Dispatch discipline — COMPLIANT

- `THIN_M_GEMM_TEST_HITS` counter: present, correctly gated on
  `cfg(all(test, target_arch = "aarch64", any(target_os = "macos", target_os = "ios")))`.
- `dispatch_manifest.toml` row: present, correctly scoped to
  `op = "MatMul"`, `variant = "f32_thin_m"`, `platform = "aarch64-apple-darwin"`.
- Test `f32_thin_m_prefill_reaches_neon_col_parallel` verifies the counter fires. ✓

### 6. Reachability — VERIFIED BY EXPERIMENT

Ran both thin-M tests (reachability + numerics parity): both pass.
Ran `fp16_m_ge2_prefill_reaches_bnns_not_half_gemm`: passes.
Full ep-cpu test suite: **1004 passed, 0 failed, 6 ignored**.

### 7. Portability — ACCEPTABLE

- All NEON intrinsics are ARMv8.0 baseline (`vfmaq_f32`, `vld1q_f32`,
  `vaddvq_f32`, `vaddq_f32`, `vdupq_n_f32`). No v8.2+ features.
- Scalar fallback provided for non-aarch64 via `#[cfg(not(target_arch = "aarch64"))]`.
- Threshold labeled as fitted with measured bracket [2M, 4M elements] on M1 Max,
  M_MAX labeled as fitted with measured bracket [16, 24].
- No hardcoded thread counts or cache-line sizes.

## Non-blocking concern

**C1: Documentation factual error in `THIN_M_LARGE_B_THRESHOLD` comment.**

Line 862–863 of `accelerate_gemm.rs`:
> "4M (16 MB) … is well below the smallest SLC (8 MB on M1 base)"

16 MB > 8 MB — this is numerically inverted. The threshold is *above* the
smallest SLC, not below it. The mechanism conclusion (B must stream from DRAM
at this size) is actually correct *precisely because* 16 MB exceeds the 8 MB
SLC. The text states the opposite relationship. Same defect class as PR #347's
L1 cache error. **Fix the comment before merge** — the standing directive says
"a wrong rationale is worse than no rationale."

Suggested replacement:
> "We use 4M (16 MB) as the portable floor because it exceeds both the
> per-cluster L2 (~12 MB on base chips) and the smallest SLC (8 MB on M1 base),
> ensuring B must stream from DRAM on every Apple Silicon variant."

## Test results

```
cargo test -p onnx-runtime-ep-cpu --lib
test result: ok. 1004 passed; 0 failed; 6 ignored
```
