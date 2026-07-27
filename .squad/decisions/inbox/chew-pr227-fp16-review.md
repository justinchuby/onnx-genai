# Chew — PR #227 FP16 Path Numerics Review (Second Pass)

**Branch:** `squad/mac-cpu-ep-roofline`
**Date:** 2026-07-27T02:00:00-07:00
**Commits under review:** `75311827` (FP16 storage GEMV + bulk conversion), `3a88ba8c` (SPMD pool for FP32 GEMV + cleanup)
**Author under review:** Iran
**Reviewer:** Chew (Numerics)

---

## Verdict: **APPROVE**

The FP16 storage GEMV kernel and bulk f16↔f32 conversion are numerically sound. All 922 tests pass. `cargo fmt --check` clean (BLOCKING). `cargo clippy` produces only pre-existing cosmetic warnings in `activations.rs` (not new code). Two non-blocking concerns are noted below.

---

## Per-item findings

### 1. Inline assembly for `fcvtl` — SOUND, NON-BLOCKING CONCERN

**File:** `accelerate_gemm.rs:419-433` (`load_f16x4_to_f32x4`)

The asm block:
```asm
"ldr {v:d}, [{ptr}]"       // load 8 bytes (4 × f16) into low 64 bits of Vn
"fcvtl {v:v}.4s, {v:v}.4h" // widen low 4 × f16 → 4 × f32
```

**Correctness assessment:**
- **Constraints correct.** `ptr = in(reg)` for the base address, `v = out(vreg)` for the NEON result. Using the same register for input/output of `fcvtl` is valid — the instruction reads from the low half and writes to the full register.
- **Options correct.** `nostack` (no stack use), `readonly` (no memory writes), `pure` (no side effects). All accurate — the block only reads memory and produces a register value.
- **Clobbers correct.** No additional clobbers needed; `v` is declared as `out(vreg)` which already tells the compiler it's modified.
- **`volatile` correctly absent.** The `pure, readonly` combination allows the compiler to CSE/LICM the load, which is desirable for the GEMV inner loop.

**Verified bit-exact against scalar `half::f16::to_f32()`** for: normal values (1.0–65504.0), denormals (0x0001, 0x03FF), ±inf (0x7C00, 0xFC00), NaN (0x7E00), ±zero (0x0000, 0x8000). All lanes match bit-for-bit.

**Concern C1 (non-blocking):** The rationale for inline asm over intrinsics is that `vcvt_f32_f16` requires Rust's unstable `f16` type, which needs nightly. This is a valid practical reason today. However, Rust's `f16` type is on track for stabilization (RFC 3453). **Recommend: add a `// TODO: replace with vcvt_f32_f16 intrinsic when f16 stabilizes` comment.** Inline asm in a shared kernel that Resch (Intel) and Luba (ARM) also maintain is a maintenance hazard — the intrinsic should replace it as soon as feasible.

**Assignee for C1:** Deckard or Sapper (not Iran).

### 2. FP32 accumulation — VERIFIED SOUND

**Files:** `accelerate_gemm.rs:474-554` (`neon_gemv_f16_batch`), `accelerate_gemm.rs:558-601` (`neon_dot_f16`)

Accumulation is genuinely f32 throughout:
- Accumulators `a0..a3`, `b0..b3` are `float32x4_t`, initialized via `vdupq_n_f32(0.0)`.
- `vfmaq_f32` is f32 fused-multiply-add — operates entirely in f32.
- Horizontal reduction via `vaddvq_f32` (f32).
- Scalar tail accumulates into `s0..s3` (f32 locals) using `half::f16::from_bits(x).to_f32()` followed by f32 multiply.

**This is NOT native FP16 accumulate.** It is the correct FP16-storage-f32-accumulate pattern.

**Measured error vs f64 reference:**

| Shape (name) | K | N | max abs | max rel | max ULP |
|---|---:|---:|---:|---:|---:|
| gate_proj | 896 | 4864 | 3.46e-6 | 1.52e-7 | 2 |
| down_proj | 4864 | 896 | 2.95e-5 | 2.38e-7 | 4 |
| q_proj | 896 | 896 | 3.37e-6 | 1.53e-7 | 2 |
| kv_proj | 896 | 128 | 2.64e-6 | 1.15e-7 | 1 |
| 1×1 | 1 | 1 | 4.49e-10 | 2.14e-8 | 0 |
| 1×4 | 1 | 4 | 1.58e-9 | 4.86e-8 | 0 |
| odd_tail | 67 | 9 | 2.28e-7 | 1.13e-7 | 1 |

Max relative error across all shapes: **2.38e-7**. This is well within the FP32-accumulate envelope (~1e-7 relative from FMA ordering). The doc claim of "~2.3e-4 max relative error" is conservative — actual measured error is 1000× better than claimed, which is consistent with FP32 accumulation (the 2.3e-4 figure would be for FP16 accumulation).

**FP16 GEMV vs F32 GEMV with identical f16-quantized weights:** max relative error **1.73e-6**. This confirms the accumulation is truly f32 — if it were FP16 accumulate, this would be ~1e-3 or worse.

### 3. Tail handling — VERIFIED CORRECT

**Main loop:** processes 8 elements per iteration (2 × 4 `fcvtl` loads per row) in `neon_gemv_f16_batch`, 16 elements per iteration in `neon_dot_f16`.

**Tail:** scalar loop `while j < k` widens each f16 individually via `half::f16::from_bits().to_f32()`.

**Verified at:** K=67 (not divisible by 8 or 16), N=9 (not divisible by 4). Both produce correct results with max abs error 2.28e-7 vs f64 reference. Also verified K=1, N=1 and K=1, N=4 — all correct.

The `neon_gemv_f16_batch` outer loop processes 4 output rows at a time, with a `while i < n` scalar tail that calls `neon_dot_f16` per remaining row. This tail is also correct at N=9 (processes 2 groups of 4, then 1 remaining row).

### 4. Transpose cache (`transposed_b_f16`) — THREAD-SAFE

**File:** `matmul.rs:161-205`

- Uses `OnceLock<Vec<u16>>`, which is Rust's standard thread-safe lazy initialization. Only one thread will execute `get_or_init`; all others block until initialization completes. No torn reads possible.
- The transpose itself uses Rayon `par_chunks_mut` — each thread writes to a disjoint slice of `bt`, so no data races.
- The `unsafe` for `from_raw_parts` is justified: the view is validated as contiguous Float16 with exactly `numel` elements; `half::f16` is `repr(transparent)` over `u16`.
- Transpose logic verified correct: `bt_chunk[jj * k + i] = src[i * n + j]` where `j = j0 + jj`. This maps `src[K,N]` row-major → `bt[N,K]` row-major, which is the correct transposition.

### 5. Bulk conversion (`neon_f16_to_f32_bulk` / `neon_f32_to_f16_bulk`) — SOUND

**File:** `dtype.rs:774-828`

**Widen (`fcvtl`, line 775-797):** Same asm block as `load_f16x4_to_f32x4`, correctly annotated `readonly, pure`. Scalar tail uses `half::f16::from_bits().to_f32()`. Verified bit-exact against scalar for all edge cases.

**Narrow (`fcvtn`, line 803-828):**
- Asm block correctly does NOT have `readonly` or `pure` — it writes to memory via `str`.
- `options(nostack)` only — correct, since it has a memory side effect.
- `src = in(vreg)` for the f32x4 input, `ptr = in(reg)` for the output address, `v = out(vreg) _` for the scratch register. Constraints are correct.

**Rounding mode:** `fcvtn` uses IEEE round-to-nearest-even (the ARM default FPCR.RMode). Verified: `neon_f32_to_f16_bulk` produces bit-identical output to `half::f16::from_f32()` for all tested values including tie-breaking cases.

**Overflow to inf:** Values > 65504 (e.g. 65520, 65536, 100000) correctly narrow to `±inf` (0x7C00/0xFC00). This matches `half::f16::from_f32()` behavior.

**Denormal handling:** Values in the f16 denormal range (e.g. 6.0e-8) are correctly narrowed with gradual underflow, matching scalar.

**NaN preservation:** NaN inputs produce NaN outputs (bit patterns may differ in payload, which is IEEE-compliant).

**Non-multiple-of-4 tail:** Tested with n=21 (21 elements). Scalar tail correctly handles the remaining 1 element.

### 6. SPMD pool correctness (`3a88ba8c`) — SOUND

**File:** `matmul_nbits.rs:1463-1488`

- `perf_cores.saturating_sub(1).max(1).min(available)` — correctly handles:
  - 1 P-core → `max(0, 1) = 1` → 1 worker (safe minimum)
  - 2 P-cores → `max(1, 1) = 1` → 1 worker (conservative)
  - 8 P-cores (this M1 Max) → `min(7, 10) = 7` workers
  - `.min(available)` prevents exceeding logical CPU count
  - `saturating_sub` prevents underflow
  - Cannot produce zero or negative — `.max(1)` is the floor

- `performance_core_count()` (line 1632-1662) returns `None` on Intel Macs or VMs where `hw.perflevel0.physicalcpu` doesn't exist, causing the override block to be skipped entirely — falling back to the generic `available/2` default. Safe.

- The existing `with_decode_pool_scope` change (line 2243-2258) correctly gates SPMD pool eligibility: without MLAS, the pool is eligible for both quantized and dense models; with MLAS, only quantized models use it (avoiding contention with MLAS's own Rayon pool).

### 7. Silent-fallback audit — PASSED

The `constant_weight_prepack_reuses_weight_and_keeps_activation_live` test (matmul.rs:1700-1745) asserts `kernel.prepack.transposed_b_f16.get().is_some()` on macOS, proving the FP16 GEMV path is compiled and executed. The test uses `Owned::f16` weights with M=1, which matches the dispatch condition. Result `[2., 6.]` and `[8., 15.]` are numerically exact (f16-representable values).

### 8. Apple Silicon generality — CORRECT

- `fcvtl` and `fcvtn` are ARMv8 base FP instructions, not FEAT_FP16. They are present on ALL aarch64 CPUs, not just Apple Silicon.
- The entire `accelerate_gemm` module is gated by `#[cfg(any(target_os = "macos", target_os = "ios"))]` — Luba's ARM Linux code never enters this module.
- Non-aarch64 scalar fallbacks exist at lines 551 and 606.
- Thread count is derived at runtime from `hw.perflevel0.physicalcpu` with sane fallback — no hardcoded tile sizes or cache assumptions.

### 9. Test coverage — ADEQUATE

**New tests in `accelerate_gemm.rs`:**
- `f16_col_parallel_gemv_matches_reference` (K=64, N=128, max abs < 1e-3)
- `f16_col_parallel_matches_at_model_scale` (K=896, N=4864, max rel < 2%)
- `f16_gemv_odd_k_tail` (K=67, N=9, exercises scalar tail)

**Updated tests in `matmul.rs`:**
- `constant_weight_prepack_reuses_weight_and_keeps_activation_live` — updated to assert f16 cache path on macOS

**Concern C2 (non-blocking):** The model-scale test threshold of `max_rel < 0.02` (2%) is very loose for what should be FP32-accumulate accuracy. Measured actual error is ~2.4e-7 (1e5× below threshold). Recommend tightening to `max_rel < 1e-4` to catch genuine FP16-accumulate regressions. Similarly, the `f16_col_parallel_gemv_matches_reference` threshold of `max_abs < 1e-3` should be `< 1e-5`.

**Assignee for C2:** Deckard or Sapper (not Iran).

---

## Summary

| Item | Status |
|---|---|
| Inline asm `fcvtl` correctness | ✅ Sound (bit-exact vs scalar) |
| FP32 accumulation preserved | ✅ Verified (max rel 2.38e-7) |
| FP16 GEMV numerical parity | ✅ Within f32-accumulate envelope |
| Tail handling (K, N non-aligned) | ✅ Correct at K=67/N=9/K=1/N=1 |
| Transpose cache thread safety | ✅ OnceLock + disjoint par_chunks |
| Bulk conversion rounding/overflow/NaN | ✅ Bit-exact with half::f16 |
| SPMD pool edge cases | ✅ Cannot produce ≤0 workers |
| Path reachability | ✅ Test proves f16 GEMV is hit |
| Apple Silicon generality | ✅ ARMv8 base, correct gating |
| Test coverage | ✅ Adequate (3 new + 1 updated) |

**Non-blocking concerns:**
- **C1:** Add TODO to replace inline asm with intrinsics when `f16` stabilizes.
- **C2:** Tighten test error thresholds from 2%/1e-3 to 1e-4/1e-5 to guard against accumulation regressions.
