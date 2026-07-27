# Pris — SDPA NEON coverage follow-up

Date: 2026-07-27
Campaign: PR #227 (`squad/mac-cpu-ep-roofline`)
Owner: Pris (Tester)

## Decision

`sdpa_f32_neon` now has direct aarch64 coverage instead of relying on scalar-only SDPA tests.

The new coverage in `crates/onnx-runtime-ep-cpu/src/kernels/sdpa.rs` compares NEON against both `sdpa_f32_scalar` and an f64 reference on decode-relevant shapes:

- Qwen-style GQA decode: batch 1, 14 query heads, 2 KV heads, q_seq 1, kv_seq 257, dh/dv 64.
- Odd/tail dimensions: dh 133, dv 65, q_seq 3, kv_seq 129, causal, softcap, bias, mask, and a fully masked query.
- Large-score stability: magnitude 48 inputs to exercise softmax max-subtraction, with masked entries and odd dimensions.

Tolerance is intentionally not exact: NEON uses 4x-unrolled/tree accumulation while the scalar path is sequential. The guard accepts NEON-vs-scalar max abs <= 5e-4, relative <= 2e-3 with a 1e-4 denominator floor, and NEON-vs-f64 max abs <= 1e-3.

A dispatcher reach test increments a test-only hit counter in `sdpa_f32_neon` and asserts `sdpa_f32(...)` reaches that path on aarch64 when the MLAS feature is not selected.

## Guard-break proof

Probe applied: deliberately skipped `dot_neon` scalar tail handling by setting `j = n` before the final `while j < n` tail loop.

Expected failure observed:

```text
test kernels::sdpa::tests::sdpa_neon_matches_scalar_and_f64_reference_on_decode_shapes ... FAILED
odd-dh-dv-tail-masked: NEON vs scalar max_abs=9.221658e-4 max_rel=2.034264e0
```

After restoring the tail loop, the focused test passed:

```text
running 1 test
test kernels::sdpa::tests::sdpa_neon_matches_scalar_and_f64_reference_on_decode_shapes ... ok
```

The aarch64 dispatcher reach check also passed:

```text
test kernels::sdpa::tests::sdpa_dispatcher_reaches_neon_on_aarch64 ... ok
```

## GEMV tolerance follow-up

Chew measured the model-scale GEMV max relative drift at 1.57% for `[1,4864,896]`, with smaller cases below that. The `accelerate_decode_gemv_matches_generic_at_model_scale` threshold was tightened from 2.0% to 1.8%, leaving modest cross-machine headroom for legitimate f32 accumulation-order drift while catching larger regressions.
