# Pris: Trait ↔ C-ABI Parity — Integration Tests

**Date:** 2026-08-10
**Branch:** `squad/ep-plugin-parity-cuda`
**Author:** Pris (Tester)
**Updated:** 2026-08-10 (lint fix + f16/bf16 verdict)

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
| Unsqueeze (opset-13, no axes attr): trait claims it AND C ABI filter removes it (divergence proven) | `error_parity_declined_shape_inference_is_cabi_only` | ✅ PASS |
| Unknown op → both decline | `error_parity_unknown_op_declined_by_both` | ✅ PASS |

## Final test counts (2026-08-10, commit 577047a74)

- `cargo test -p onnx-runtime-ep-plugin`: **132 passed; 0 failed; 9 integration tests (all pass)**
- `cargo test -p onnx-runtime-ep-cpu-plugin -- --include-ignored`: **23 passed; 0 failed; 0 ignored**

## f16/bf16 status — **OUTCOME 1: PASSES. #[ignore] removed.**

**Verdict (2026-08-10, commit 577047a74):**

Both `conformance_add_float16` and `conformance_add_bfloat16` pass with **numerically
correct, exact bit-pattern output**. The `#[ignore]` has been removed from both tests.

**Evidence:**
```
test conformance_add_float16  ... ok   (f16: [0x4000, 0x4400, 0x4600, 0x4800] ✓)
test conformance_add_bfloat16 ... ok   (bf16: [0x4000, 0x4080, 0x40C0, 0x4100] ✓)
```

**Why they now pass:**
- Deckard landed `build_cpu_registry_with_descriptors()` in `crates/onnx-runtime-ep-cpu`.
- The cpu-plugin shim wires it through `GetKernelRegistry` in
  `crates/onnx-runtime-ep-cpu-plugin/src/lib.rs` via `build_kernel_registry_entries()`.
- `Float16` and `BFloat16` are included in the kernel descriptor supported-dtype lists,
  so ORT's type-constraint metadata matches and routes f16/bf16 nodes to our EP.
- The kernel produces correct output (1.0+1.0=2.0, 2.0+2.0=4.0, 3.0+3.0=6.0, 4.0+4.0=8.0
  for both Float16 and BFloat16 exactly).

**This is outcome 1** (our EP accelerates f16/bf16), not outcome 2 (ORT fallback).
The kernel registry wiring confirms our EP claims and executes the nodes.

## What remains unproven

- **Full kernel-level numerical parity**: driving `get_kernel` → `execute` through both paths
  for a complete op requires the session layer. Memory-path tests prove
  allocate/copy/deallocate parity; op-level compute parity is covered by the ORT e2e suite.
- **Concurrent kernel dispatch**: not tested (single-threaded assertions only).
- **EPContext save/load**: fail-closed by design, not faked.
