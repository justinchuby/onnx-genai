# Gaff Delta Review — PR #762

**Date:** 2026-08-12  
**Scope:** Commits 2106ac0f7, d1421ac88, 7a2268021, 3826e1127, 8b3197ea6  
**Verdict:** Ready to leave draft — no blockers.

## Item-by-Item Status

| Item | Status | Notes |
|------|--------|-------|
| Gate coverage (fail-loud) | ✅ Genuinely closed | All three conditions (ORT lib missing, EP cdylib missing, fixture missing, dlopen failure) panic when `NXRT_REQUIRE_ORT_TESTS=1` |
| CI lane (`CLI ORT Linux x86_64`) | ✅ Genuinely closed | `NXRT_REQUIRE_ORT_TESTS=1` set at job level (ci.yml:544); lane builds ORT via onnx-genai-ort-sys |
| `scratch_alloc_bytes` shared | ✅ Genuinely closed | Single definition at compute.rs:601; canaries call it directly; `scratch_buffer_detects_oversized_write` would trip on formula regression |
| `validate_write_dtype` | ⚠️ Partially closed | Tests-only — never called in production. See SUBSTANTIVE below |
| Compile-time routing | ✅ Genuinely closed | Dual-role values correctly rejected as unrepresentable; conservative (ORT falls back) |
| CUDA `end_version: i32::MAX` | ✅ Genuinely closed with caveat | See NITS below |
| Vtable release guard | ✅ Genuinely closed | Test is non-vacuous (see below) |

## Findings

### BLOCKING

None.

### SUBSTANTIVE

1. **`validate_write_dtype` is dead code in production** — `crates/onnx-runtime-ep-api/src/tensor.rs:317`  
   The function exists and is tested, but zero production call sites invoke it. Coco's judgement ("runtime enforcement infeasible without restructuring the raw-pointer API") is *honest* — the kernel writes through raw pointers that bypass `TensorMut` — but the result is a validator that cannot validate anything at runtime. It is effectively documentation-as-code. This is acceptable if acknowledged; it should have a `// NOTE: enforced by tests only` comment or be `#[cfg(test)]` gated. As-is it gives false confidence. **Not a blocker** because the scratch sizing (`max(byte_size, 8)`) is the actual safety net and it is proven by canaries.

2. **`find_ort_lib_dir` duplication across 3 test files** — `optional_slots.rs:77`, `plugin_ort_e2e.rs:107`, `layernorm_dynamic_axis.rs:33`  
   Two copies are the refactored `ort_discovery` module (identical). The third (`layernorm_dynamic_axis.rs`) is a hand-rolled older version that hard-codes `"libonnxruntime.so"` (not platform-aware) and lacks `CARGO_TARGET_DIR` fallback. **This WILL drift** — it already has. Not a blocker for leaving draft but should be tracked as a follow-up.

### NITS

3. **CUDA `i32::MAX` over-claim** — `crates/onnx-runtime-ep-cuda-plugin/src/lib.rs:65`  
   The comment ("kernels are version-agnostic") is correct for the current op list (MatMul, Softmax, LayerNorm, etc.) whose semantics haven't changed across opsets — only default attribute values changed, and those are resolved by ORT's schema before reaching the EP. For ops like `Resize` or `Pad` (not in CUDA_COVERED_OPS), semantics DID change across opsets, but since those aren't registered, no over-claim exists today. The guard is that ops are only registered if in `CUDA_COVERED_OPS`. Safe as-is; the comment could note this invariant.

## Key Verification Questions

**Is the undersized-vtable negative test non-vacuous?**  
Yes. `loader.rs:373-445`: The test (a) constructs an undersized factory, (b) runs the guarded path → confirms `RELEASE_CALLED` stays false, (c) then directly calls `release` without the guard → confirms the flag flips. Both arms execute; the test would fail if either the guard or the release function were broken.

**Does `i32::MAX` on CUDA over-claim any op?**  
No, for the current `CUDA_COVERED_OPS` list. All registered ops have stable semantics across opset versions for the dtypes served (f32/f16/bf16). The risk is future additions — if someone adds an op whose semantics changed (e.g. `Resize`), the `i32::MAX` would silently over-claim. The existing `CUDA_COVERED_OPS` const acts as the control surface.

## Ready to Leave Draft?

**Yes.** No blocking defects. The two substantive items are honest trade-offs (dead validator, test helper duplication) that don't affect correctness or safety.

## What I Verified vs. Took on Trust

**Verified myself:**
- `scratch_alloc_bytes`: single definition, 5 call sites, canary tests call production function directly
- `validate_write_dtype`: zero production call sites (grep across all crates)
- `find_ort_lib_dir`: 3 copies, one already drifted (`layernorm_dynamic_axis.rs` not platform-aware)
- Vtable test: read full test body, confirmed both guarded and unguarded paths exercise the atomic flag
- CUDA `end_version`: cross-referenced `CUDA_COVERED_OPS` list — no ops with semantics that changed across opsets
- Compile-time routing: traced `build_subgraph_routing` logic; dual-role values correctly return `None`
- CI lane: `NXRT_REQUIRE_ORT_TESTS=1` at ci.yml:544, job name matches "CLI ORT (Linux x86_64)"
- Gate panics: all four conditions in `optional_slots.rs` panic when env var is set

**Took on trust:**
- That the CI lane actually has ORT available at runtime (can't run CI from here; inferred from the ort-sys build step)
- That the `ort_discovery` scan logic actually resolves at CI time (the logic looks correct but I didn't execute it)
- Rachael's claim that Rust integration tests cannot share code across files (true per Cargo's compilation model for `tests/` — each file is a separate crate)
