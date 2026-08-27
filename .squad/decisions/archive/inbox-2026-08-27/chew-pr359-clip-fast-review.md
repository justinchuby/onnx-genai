# PR #359 — NEON SIMD fast-path for contiguous f32 Clip

**Reviewer:** Chew  
**Date:** 2026-07-28T14:18:00Z  
**Verdict:** 🟢 **APPROVE**

## Summary

Adds `clip_contiguous_f32_fast()` — a zero-allocation NEON SIMD clamp for contiguous
f32 tensors, eliminating the ~42 MB/inference wasted bandwidth from two full copies
per Clip node. Measured 3.75–3.91× MobileNetV2 speedup against a 3.93× Amdahl
projection.

## Findings

### 1. NaN handling — **not a blocking divergence** (pre-existing)

Empirically verified: the new path uses `f32::max`/`f32::min` (NEON `vmaxq_f32`/`vminq_f32`)
which suppress NaN (return the non-NaN operand), while the generic `clip_typed` reference
propagates NaN via comparison operators.

However, this divergence is **pre-existing**. The MLAS path (`mlas_compute_activation`
activation type 5) uses SIMD min/max with identical NaN-suppressing semantics. The new
path targets exactly the same case (contiguous f32) that MLAS handled, so no new
behavioral split is introduced. The dispatch priority is:

```
MLAS (if feature) → fast NEON (new) → clip_typed (generic)
```

Non-contiguous f32 and non-f32 dtypes always fall through to `clip_typed` regardless.

### 2. Signed zero — same analysis

`f32::max(-0.0, 0.0)` = +0.0 (NEON FMAX same). Reference preserves -0.0 because
`-0.0 < 0.0` is false in IEEE 754. Pre-existing divergence matching MLAS behavior.

### 3. Optional/absent bounds — **correct**

- `inputs.len() > 1 && !inputs[1].is_absent()` matches the MLAS path exactly.
- Falls back to `kernel.min`/`kernel.max` attributes (pre-opset-11) or ±∞ defaults.
- MobileNetV2-12 uses opset 12 (inputs, not attributes) — correctly handled.

### 4. min > max — **correct, defensive error**

Explicit `EpError::KernelFailed` returned. Matches both MLAS and reference paths.

### 5. Tail handling — **verified correct**

- `bulk_end = n & !15`: rounds down correctly.
- Middle tail: `while i + 4 <= n` handles 4-wide chunks.
- Final tail: `while i < n` scalar.
- Verified: n=1 (one scalar), n=15 (3×4 + 3 scalar), n=16 (one bulk), n=17 (one bulk + 1 scalar).
- Test uses n=300 (not a multiple of 16), exercising tail.

### 6. Contiguity guard — **sound**

Both `inputs[0].is_contiguous()` and `output.is_contiguous()` checked; returns
`Ok(false)` on failure. No non-contiguous tensor can reach the SIMD path.

### 7. Aliasing / in-place — **acceptable**

Unlike the MLAS path, no explicit overlap check. The operation is element-wise and
positional (read i, write i), so even if src==dst it produces correct values. The Rust
aliasing UB concern (simultaneous `&[f32]` and `&mut [f32]`) is mitigated by the runtime
contract that input and output tensors are distinct allocations. Non-blocking.

### 8. Dispatch discipline — **met**

- `CLIP_F32_FAST_TEST_HITS` counter: present, incremented on success.
- `dispatch_manifest.toml` row: op=Clip, variant=contiguous_f32, platform=all, counter named.
- Test `clip_f32_fast_path_fires_on_contiguous_input`: asserts counter increment.

### 9. Portability — **correct**

- NEON is ARMv8 baseline (all Apple Silicon).
- Scalar fallback for non-aarch64 via `#[cfg(not(target_arch = "aarch64"))]`.
- No hardcoded constants, no hardware-specific thresholds.
- No comments requiring the #353 citation rule.

### 10. Format — **clean**

`cargo fmt --all -- --check` passes.

## Non-blocking concerns (do not block merge)

1. **Missing overlap check**: The MLAS path defensively checks input/output overlap.
   The new path omits this. Suggest adding for parity, but runtime contract prevents
   the scenario.

2. **NaN documentation**: Recommend a test with `f32::NAN` input documenting that the
   fast path intentionally suppresses NaN (matching MLAS), not propagates it. This
   prevents future confusion if someone compares against `clip_typed`.

## Verdict

**🟢 APPROVE.** Implementation is numerically sound, consistent with the existing MLAS
fast path, correctly guarded, correctly dispatched, and portable. The measured speedup
aligns with the Amdahl projection within noise.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
