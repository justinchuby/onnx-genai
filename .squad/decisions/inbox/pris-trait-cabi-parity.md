# Pris: Trait ↔ C-ABI Parity — Integration Tests

**Date:** 2026-08-10
**Branch:** `squad/ep-plugin-parity-cuda`
**Author:** Pris (Tester)
**Updated:** 2026-08-10 — tests now compile and pass

## Capability-parity rule (proven in code)

```text
C_ABI_claims = trait_claims ∩ { nodes where ShapeInference::for_node ≠ Declined }
```

The C ABI `GetCapability` path in `ep.rs` applies a **fail-closed shape-inference filter**
on top of the trait's `supports_op` / `supports_node` result. Every node the C ABI claims
is also trait-supported, but the converse is not true.

### Important finding from making tests pass

The rule **holds**, but the set of "Declined" ops is **smaller than the initial test assumed**:

- `ShapeInference::for_node` (full node-reading) is smarter than `for_op` (defaults only).
  It can infer shapes for Squeeze (empty axes = no-op), ReduceMean, and even Conv (from
  weight shape) without explicit attributes. Only ops with **truly data-dependent output
  shapes** are Declined by `for_node`.
- The concrete confirmed-Declined case: **`Unsqueeze` at opset ≥ 13** where `axes` come
  from `input[1]` (a runtime tensor), not from an attribute. `for_node` correctly returns
  `Declined { op_type: "Unsqueeze", ... }` for this case.
- The initial test used `DataType::Float` (wrong) instead of `DataType::Float32` — fixed.
- Graphs constructed without opset imports have `effective_opset() = 0`, causing
  `supports_op` to decline all ops. All test graphs now import opset 13.

## What is now proven (tests compiled and pass — 9/9)

| Assertion | Test | Verdict |
|-----------|------|---------|
| Supported ops with known shapes at opset 13 → both paths claim | `capability_parity_supported_ops_with_known_shapes` | ✅ PASS |
| Unsqueeze opset-13 no-attr-axes: trait may support, for_node declines, C ABI filter excludes | `capability_parity_supported_but_shape_declined` | ✅ PASS |
| Unsupported ops → both decline (all domains) | `capability_parity_unsupported_ops` | ✅ PASS |
| Mixed graph → trait claims Add, not FakeOp; query_capabilities claims Add | `capability_parity_mixed_graph` | ✅ PASS |
| com.microsoft unknown ops → both decline | `capability_parity_com_microsoft_domain` | ✅ PASS |
| Memory roundtrip bit-exact | `numerical_parity_memory_roundtrip` | ✅ PASS |
| Device-to-device copy bit-exact | `numerical_parity_device_copy` | ✅ PASS |
| Unsqueeze (opset-13, no axes attr) → C ABI filter predicate = false | `error_parity_declined_shape_inference_is_cabi_only` | ✅ PASS |
| Unknown op → both decline | `error_parity_unknown_op_declined_by_both` | ✅ PASS |

## Final test counts

- `cargo test -p onnx-runtime-ep-plugin`: **127 passed; 0 failed; 9 integration tests (all pass)**
- `cargo test -p onnx-runtime-ep-cpu-plugin`: **15 passed; 0 failed; 2 ignored (f16/bf16, blocked)**

## f16/bf16 status

**Blocked.** Deckard has not landed `registry_entries()` on `CpuExecutionProvider`
(`crates/onnx-runtime-ep-cpu/src/provider.rs`). Without this, ORT does not route
Float16/BFloat16 nodes to our EP via `GetKernelRegistry`.

Tests written and `#[ignore]`-d with precise reason:
- `conformance_add_float16` — uses `tests/fixtures/add_float16/model.onnx`
- `conformance_add_bfloat16` — uses `tests/fixtures/add_bfloat16/model.onnx`

Both tests assert **exact bit-pattern equality** on f16/bf16 outputs (independently
computed expected values, not derived from the implementation). Remove `#[ignore]` when
`registry_entries()` lands and test under `-- --ignored`.

## What remains unproven

- **Full kernel-level numerical parity**: driving `get_kernel` → `execute` through both paths
  for a complete op requires the session layer. Memory-path tests prove
  allocate/copy/deallocate parity; op-level compute parity is covered by the ORT e2e suite.
- **Concurrent kernel dispatch**: not tested (single-threaded assertions only).
- **EPContext save/load**: fail-closed by design, not faked.
