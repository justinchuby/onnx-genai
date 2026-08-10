# EP Capability ↔ Shape-Inference Agreement Contract

**By:** Nabil  
**Date:** 2026-08-10  

## Problem

Our EP claimed nodes in `GetCapability` based solely on `ep.supports_op()` (registry-based), but the Compile/Compute path used `ShapeInference::for_op()` which returned `Declined` for attribute-dependent ops. This created two failure modes:

1. **Over-claiming**: Ops like `NonZero` (data-dependent output shape) were claimed but could not be correctly executed, causing runtime failures instead of graceful fallback.
2. **Dead rules**: Deckard's 22 attribute-aware shape-inference rules in `compute.rs` were unreachable because the live path only knew op names, not attributes.

## Contract

**Claim predicate and shape-inference capability now agree by construction:**

1. In `ep_get_capability_inner`, after `query_capabilities()` returns claims, each claimed node is checked with `ShapeInference::for_node(node, input_shapes, num_outputs)`. If the result is `Declined`, the entire claim containing that node is dropped.

2. In `ep_compile_inner`, the same `ShapeInference::for_node()` is used (no longer `for_op()`), so the compile-time shape inference is guaranteed to match what was validated at claim time.

**Result:** A node is claimed ↔ its shape inference is resolved. No mismatch can recur because both paths use the same `for_node` function with the same attribute data.

## Attributes and Initializers: Owned Copies at Compile Time

ORT's header forbids caching `OrtGraph*`/`OrtNode*` beyond the Compile call. We solve this by:

1. **Attribute reading** (`graph_reader.rs:read_node_attributes`): During `from_ort_graph()`, all node attributes are read via `Node_GetNumAttributes`/`Node_GetAttributes`/`ReadOpAttr` and copied into owned `onnx_runtime_ir::Attribute` values stored in `Node.attributes`. No ORT pointers are retained.

2. **Initializer reading** (`graph_reader.rs:read_initializers_int64`): Small int64 initializer tensors (≤64 elements, 1-D) are read via `Graph_GetInitializers`/`ValueInfo_GetInitializerValue`/`GetTensorData` and copied into an owned `HashMap<String, Vec<i64>>`. This enables resolving opset-13 Unsqueeze/Squeeze axes from constant inputs.

3. **Opset-13 Unsqueeze/Squeeze**: When `since_version >= 13` and no `axes` attribute exists, the reader checks if `input[1]` matches a known initializer name. If so, the initializer values are injected as a synthetic `axes` attribute, allowing `for_node` to produce `Unsqueeze { axes }` instead of `Declined`.

## SubgraphRouting for Multi-Node Fused Subgraphs

When a compiled subgraph has more than one node, `build_subgraph_routing()` constructs a `SubgraphRouting` table:

- Graph input values → `NodeInputSource::Ort(idx)`
- Prior node outputs consumed by later nodes → `NodeInputSource::Buffer(buf_idx)`  
- Node outputs that are graph outputs → `NodeOutputSink::Ort(idx)`
- Internal intermediate outputs → `NodeOutputSink::Buffer(buf_idx)`

This is attached via `info.set_routing(routing)` so the Compute path can thread intermediates in topological order.

## Fail-Closed Principle

- If attribute reading fails for any reason, we get fewer attributes → `for_node` may return `Declined` → node is not claimed → ORT keeps it on its own EP. Safe.
- If initializer reading fails, opset-13 Unsqueeze/Squeeze axes stay missing → `Declined` → not claimed. Safe.
- If routing construction fails (e.g. unexpected graph topology), `build_subgraph_routing` returns `None` → no routing is set → multi-node Compute will error explicitly (existing invariant in `compute.rs`).
