# Fix tautological cache test + SiLU accuracy doc conflict

**Date:** 2026-07-27
**Author:** Deckard
**PR:** #227 (squad/mac-cpu-ep-roofline)

## Context

GitHub Copilot's automated reviewer flagged two issues in `onnx-runtime-ep-cpu`:

1. **matmul.rs** — the `constant_weight_prepack_reuses_weight_and_keeps_activation_live` test compared a cache pointer to itself (tautological: always passes). This is the fifth instance of the "test that cannot fail" bug class on this campaign.

2. **activations.rs** — the `silu_f32_slice` doc comment claimed "1 ULP accuracy" for the NEON exp polynomial, contradicting the measured "~28 ULP" stated in the implementation comment below it. Partially fixed earlier; the higher-level doc was missed.

## Decision

- **Fix 1:** Restructured the cache-reuse test to capture the prepack pointer *before* the second `execute()` call. The comparison now spans the second call, proving the first call populated the cache and the second reused it. Guard-break confirmed: substituting a fresh kernel (simulating cache invalidation) makes the test fail with distinct pointers.

- **Fix 2:** Updated the slice-level doc to state "~28 ULP worst-case on [-87, 88]", matching the measured value. Grep-verified: no other "1 ULP" claim remains in `activations.rs`. The other "1 ULP" references in the crate (`decode_spmd.rs`, `matmul_nbits.rs`) are about N-tile boundary drift, not exp accuracy — left as-is.

## Verification

- `cargo fmt --all -- --check` ✅
- `cargo clippy -p onnx-runtime-ep-cpu --lib -- -D warnings` (aarch64) ✅
- `cargo clippy -p onnx-runtime-ep-cpu --lib --target x86_64-apple-darwin -- -D warnings` ✅
- Full CPU EP test suite: 906 passed, 0 failed ✅
- `sdpa_dispatcher_reaches_neon_on_aarch64 ... ok` ✅
