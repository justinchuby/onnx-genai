# Pris: Trait ↔ C-ABI Parity — Integration Tests

**Date:** 2026-08-10
**Branch:** `squad/ep-plugin-parity-cuda`
**Author:** Pris (Tester)

## Capability-parity rule (encoded and pinned)

```text
C_ABI_claims = trait_claims ∩ { nodes where ShapeInference::for_node ≠ Declined }
```

The C ABI `GetCapability` path in `ep.rs` applies a **fail-closed shape-inference filter** on top of the trait's `supports_op` / `supports_node` result. Every node the C ABI claims is also trait-supported, but the converse is not true: attribute-dependent ops (Squeeze, Unsqueeze, ReduceMean, Conv, Gemm, etc.) return `ShapeInference::Declined` without node attributes, causing the C ABI to exclude them even when the trait's kernel registry supports the op.

This is **intentional** — it prevents over-claiming ops whose output shapes cannot be inferred at compile time. The divergence lives in `crates/onnx-runtime-ep-plugin/src/ep.rs` (the `claims.into_iter().filter(...)` block in `ep_get_capability_inner`).

## What is now proven (once lib compiles)

| Assertion | Test |
|-----------|------|
| Supported ops with known shapes → both paths claim | `capability_parity_supported_ops_with_known_shapes` |
| Supported but shape-Declined → trait claims, C ABI does NOT | `capability_parity_supported_but_shape_declined` |
| Unsupported ops → both decline (all domains) | `capability_parity_unsupported_ops` |
| Mixed graph → C ABI claims correct subset | `capability_parity_mixed_graph` |
| com.microsoft unknown ops → both decline | `capability_parity_com_microsoft_domain` |
| Memory roundtrip bit-exact | `numerical_parity_memory_roundtrip` |
| Device-to-device copy bit-exact | `numerical_parity_device_copy` |
| Conv without attrs → Declined by C ABI only | `error_parity_declined_shape_inference_is_cabi_only` |
| Unknown op → both decline | `error_parity_unknown_op_declined_by_both` |

## Blockers — teammate in-flight changes

The `onnx-runtime-ep-plugin` lib does not currently compile due to:

1. **`ep.rs:114`** — `ep_get_kernel_registry` not found (owner: **Deckard**)
2. **`ep.rs:406`** — `cleanup_partial_infos` not found (owner: **Deckard**)
3. **`device.rs:121,221`** — lifetime errors on `ep as *const dyn ExecutionProvider` (owner: **Nabil**)
4. **`device.rs:466,473,556,563`** — methods not in `ExecutionProvider` trait (owner: **Nabil**)

Once these resolve, `cargo test -p onnx-runtime-ep-plugin --test trait_cabi_parity` will validate.

## What remains unproven

- **Full kernel-level numerical parity**: driving `get_kernel` → `execute` through both paths for a complete op requires the session layer's tensor-view machinery. The memory-path tests prove allocate/copy/deallocate parity, but op-level compute parity can only be fully proven via the ORT e2e path (milestone 1's conformance tests cover this).
- **Concurrent kernel dispatch**: not tested (single-threaded assertions only).
- **EPContext save/load**: deferred in the adapter (fail-closed, not faked).
