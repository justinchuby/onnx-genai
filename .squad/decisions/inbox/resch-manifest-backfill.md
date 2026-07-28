# Decision: Manifest Backfill + Inverse Check

**Author:** Resch (Intel CPU Optimization)  
**Date:** 2026-07-27  
**Status:** Proposed (PR pending review)  
**Depends on:** #323 (dispatch manifest), #324 (ResNet non-Conv ops)

---

## What

1. Backfill manifest claims for PR #324's three new optimizations.
2. Add an **inverse check** — flagging optimization counters that lack manifest rows.
3. Fix `check_dispatch_reachability.py` to recognize `AtomicU64` counters (PR #324 used `AtomicU64` instead of `AtomicUsize`).
4. Document BatchNorm as a known-good-but-unguarded fusion, not a dispatch tier.

## New Claim Rows

| Op | Variant | Platform | Tier | Counter |
|----|---------|----------|------|---------|
| MatMul | f16_m1_colmaj | aarch64-apple-darwin | tier2 | GEMV_F16_COLMAJ_TEST_HITS |
| MaxPool | 2d_undilated | aarch64-apple-darwin | tier1 | POOL_BNNS_TEST_HITS |
| Add | contiguous_f32 | aarch64-apple-darwin | tier2 | ADD_VDSP_TEST_HITS |

## New Exclusion Rows

| Op | Variant | Platform | Reason |
|----|---------|----------|--------|
| MaxPool | dilated | aarch64-apple-darwin | BNNS doesn't support dilation |
| BatchNormalization | fusion_elimination | all | Graph-level, no dispatch counter |

## The BatchNorm Question

BatchNorm's optimization is **op elimination by fusion**, not "dispatch reached tier N". The manifest's schema is `(op, platform) → minimum_tier`, which assumes the op *executes* and we prove *which path* ran. BatchNorm should not execute at all for fused CNN models.

**Decision: Do not force it into [[claim]].** A claim row would need to point at a counter in the optimizer, which doesn't exist. Forcing it would either:
- Create a fake "tier" meaning "didn't run" (semantic pollution)
- Point at a non-existent counter (would fail the lint)

Instead, document it as an `[[exclusion]]` with the reason explaining it's a known gap in enforcement. When the optimizer gains fusion-fired counters, extend the manifest schema.

This is the tenth instance of the structural bug — and it's a **new failure mode** (opset registration, not cfg gate). The manifest-as-designed cannot catch registration errors. That requires either:
- An optimizer-level counter proving fusion fired
- An integration test asserting BN nodes are absent from the final execution plan

Both are worth building but are separate work.

## The Inverse Check

**Assessment: Sound, worth having, implemented.**

The rule: any `_TEST_HITS` counter whose name does NOT contain `SCALAR`, `FALLBACK`, `RESCUE`, or `REF` is an optimization counter. If it has no manifest `[[claim]]` row, CI fails.

**False positive analysis:**
- `PREBIND_FALLBACK_TEST_HITS` → contains "FALLBACK" → excluded ✅
- `CONV_SCALAR_REF_TEST_HITS` → contains "SCALAR" and "REF" → excluded ✅
- `POOL_SCALAR_TEST_HITS` → contains "SCALAR" → excluded ✅
- `ADD_SCALAR_TEST_HITS` → contains "SCALAR" → excluded ✅
- `NONCONTIG_RESCUE_TEST_HITS` → contains "RESCUE" → excluded ✅

All optimization counters (`GEMV_F16_*`, `BNNS_F16_*`, `CONV_BNNS_*`, `SDPA_NEON_*`, `POOL_BNNS_*`, `ADD_VDSP_*`, `PREBIND_FAST_PATH_*`) are correctly flagged when unclaimed.

**What it catches that the original didn't:**
- Would have immediately blocked PR #324 from merging without manifest rows
- Would have caught all three of PR #324's paths automatically
- Closes the "human must remember to add a row" process gap

**Residual gap:** Optimizations that DON'T use counters (e.g. BatchNorm opset registration) remain invisible.

## Guard-Break Proofs

1. **MaxPool claim**: Renamed `POOL_BNNS_TEST_HITS` → lint failed with `MaxPool/2d_undilated on aarch64-apple-darwin — CLAIM UNSATISFIED`. Restored → passes.

2. **Inverse check**: Added `FAKE_SIMD_OPT_TEST_HITS` counter with no manifest row → lint failed with `FAKE_SIMD_OPT_TEST_HITS in crates/.../add.rs` and prescriptive fix message. Removed → passes.

## AtomicU64 Fix

PR #324 used `pub static FOO: AtomicU64` instead of `static FOO: AtomicUsize`. Both lints now match `Atomic(?:Usize|U64)` and `pub` visibility. Without this fix, the reachability lint only found 8/12 counters — the 4 new ones were invisible.
