# EP Plugin Shape-Inference Contract
**Date:** 2026-08-10  
**Author:** Deckard (Systems Dev)  
**Files:** `crates/onnx-runtime-ep-plugin/src/compute.rs`

---

## Shape-Inference Contract

Every op that appears in a fused subgraph **must** have an explicit `ShapeInference`
variant. The enum now has **22 variants** covering all high-value CPU EP ops:

| Variant | Ops covered |
|---------|-------------|
| `ElementwiseBroadcast` | Add, Sub, Mul, Div, Pow, Where, Max, Min, … |
| `SameAsInput(idx)` | Relu, Sigmoid, Cast, Identity, Dropout, Softmax, LayerNorm, … |
| `SameAsInputMultiOutput { idx, count }` | Any op returning N copies of the same shape |
| `ShapePreservingNorm { num_outputs }` | LayerNormalization, SkipLayerNorm, RMSNorm (1–3 outputs) |
| `MatMul` | MatMul, MatMulNBits (1-D dot, 2-D, batched-ND with broadcast) |
| `Gemm { trans_a, trans_b }` | Gemm |
| `Concat { axis }` | Concat |
| `Transpose { perm }` | Transpose (explicit or default reverse) |
| `Gather { axis }` | Gather |
| `GatherND { batch_dims }` | GatherND |
| `GatherBlockQuantized` | GatherBlockQuantized (treated as GatherND(0)) |
| `ShapeOp { start, end }` | Shape |
| `Squeeze { axes }` | Squeeze (all-ones or explicit axes) |
| `Unsqueeze { axes }` | Unsqueeze (attribute-driven; data-dependent case → Declined) |
| `ReshapeData { allowzero }` | Reshape (reads shape tensor at runtime) |
| `SliceData` | Slice (reads starts/ends/axes/steps at runtime) |
| `Reduction { keepdims, axes, … }` | ReduceMean, ReduceSum, ReduceProd, ReduceMax, ReduceMin, … |
| `ReductionFromInput { keepdims, … }` | Same ops, opset-18+ (axes from input[1]) |
| `Conv { out_channels, per_axis }` | Conv, ConvInteger (NOTSET auto_pad only) |
| `MultiHeadAttention { num_heads, … }` | MultiHeadAttention (1–3 outputs) |
| `GroupQueryAttention { num_heads, … }` | GroupQueryAttention |
| `RotaryEmbedding` | RotaryEmbedding |
| `Declined { op_type, domain }` | Any unmodelled op — **fail-closed** |

---

## Fail-Closed Policy

**Before this change:** `ShapeInference::for_op("UnknownOp")` returned `SameAsInput(0)`,
silently producing tensors with wrong shapes.

**After this change:** `for_op` and `for_node` both return `Declined { op_type, domain }`
for any op with no modelled rule. When `infer_shapes` hits `Declined`, it returns an
`Err` with an actionable message:

```
Op 'FooBar' (domain 'some.domain') has no shape-inference rule.
Call ShapeInference::for_node(node, input_shapes, num_outputs) instead of for_op
to enable attribute-driven inference. If the op is not yet modelled, add a variant
to ShapeInference and handle it in infer_shapes.
```

This bubbles up to `compute_execute` → `fail_status(...)` → ORT receives an error
status, not silently-wrong output tensors.

### Which ops are deliberately `Declined` from `for_op`

`for_op` also returns `Declined` for ops that *require* attributes to compute shape:
`Unsqueeze`, `ReduceMean`/`ReduceSum`/… (all reductions), `Conv`, `Gemm`,
`MultiHeadAttention`, `GroupQueryAttention`. These must go through `for_node`.

### Required change in `ep.rs` (Nabil's file)

The Compile path at ~line 281 currently calls:
```rust
ShapeInference::for_op(&node.op_type)
```
This must be changed to:
```rust
ShapeInference::for_node(node, &input_shapes, num_outputs)
```
where `input_shapes` is the static shape of each input (from the graph IR) and
`num_outputs = view.node_outputs(node_idx).len()`. Until this change is made, any
attribute-dependent op that reaches `compute_execute` will produce a `Declined` error
at runtime, which is the correct fail-closed behaviour.

---

## Data-Dependent Shapes

Two ops have shapes determined by tensor *values* (not just input ranks):

**Reshape (`ReshapeData`):**
- `input[1]` is the target shape tensor (int64). `infer_shapes` reads it with
  `read_i64_tensor` (unsafe raw pointer read — caller must ensure it is CPU-resident
  and contiguous, which is guaranteed for initialiser-backed shape inputs).
- Handles `0` (copy from input dim), `-1` (infer), and `allowzero`.
- If the shape tensor pointer is null, returns an error.

**Slice (`SliceData`):**
- `inputs[1..=4]` are starts, ends, axes, steps. Each is read with `read_i64_tensor`.
- Absent optional inputs (null pointer) default to full-range / unit-step.
- Output dim formula mirrors `slice_plan` in the CPU kernel exactly, including
  clamping and negative-step semantics.

**Unsqueeze with opset-13+ axes-as-input:**
- If axes come from `input[1]` (data-dependent) and the attribute `axes` is absent,
  `for_node` returns `Declined`. The correct fix is to make ep.rs check whether
  `input[1]` is a static initialiser and pass its values as `axes` to `for_node`.

---

## Multi-Node Subgraph Correctness

Added `SubgraphRouting` + `NodeInputSource` + `NodeOutputSink` to describe how each
node's inputs/outputs map to ORT context slots or intermediate heap buffers.

`ExportedComputeInfo` now has:
```rust
pub routing: Option<SubgraphRouting>
pub fn set_routing(&mut self, routing: SubgraphRouting)
```

`compute_execute` has three paths:

1. **Single-kernel (fast path):** no routing needed; reads all ORT inputs, allocates
   all ORT outputs, runs the one kernel.
2. **Routed multi-node path:** for each node in topological order, resolves inputs from
   ORT or from `IntermediateBuf` slots, allocates outputs to ORT or to heap buffers,
   runs the kernel. Only true graph outputs reach ORT.
3. **Multi-kernel without routing:** explicit error —
   *"Compute: multi-node subgraph requires SubgraphRouting"*. This forces ep.rs to
   wire up the routing table rather than silently misbehaving.

`IntermediateBuf` is a `Vec<u8>` + shape/strides/dtype owned by the Compute call.
All intermediate buffers are freed when `compute_execute` returns (RAII).

### Required change in `ep.rs` for multi-node fused subgraphs

After building the `ExportedComputeInfo`, call:
```rust
info.set_routing(SubgraphRouting {
    input_sources: vec![/* per-node input sources */],
    output_sinks:  vec![/* per-node output sinks */],
    num_intermediate_buffers: N,
});
```

---

## Test Coverage (66 tests pass)

New tests in `compute::tests` cover:
- `for_op_unknown_returns_declined` — fail-closed fallback
- `for_op_attribute_dependent_returns_declined` — Unsqueeze, reductions, Conv, Gemm
- `declined_infer_gives_actionable_error` — error message content
- `elementwise_broadcast_*` — same shape, numpy rules, scalar broadcast, no-input error
- `same_as_input_*` — roundtrip, OOB error, multi-output
- `matmul_*` — 2D, batched, batch-broadcast, 1D dot product, mat-vec
- `gemm_*` — no-transpose, transB
- `concat_*` — axis 0, axis 1, negative axis
- `transpose_*` — default reverse, explicit perm, wrong-length perm error
- `gather_*` — axis 0, axis 1 with matrix indices
- `gather_nd_basic`
- `shape_op_*` — full, sliced, negative indices
- `squeeze_*` — removes all size-1, specific axis
- `unsqueeze_*` — front insertion, multiple axes
- `reshape_*` — static shape, -1 inference, zero-copy dims
- `slice_*` — basic, negative step reverse
- `reduction_*` — keepdims, no-keepdims, all-axes, noop-empty-axes
- `conv_*` — 1D no padding, 2D with padding
- `intermediate_buf_view_roundtrip` — IntermediateBuf lifecycle
- `subgraph_routing_structure` — multi-node chain ORT→Buffer→ORT
- `create_and_release_state_lifecycle` — CreateState/ReleaseState
- `create_state_null_out_does_not_panic`
- `for_op_*` coverage — elementwise, unary, matmul, safe defaults

Result: `test result: ok. 66 passed; 0 failed`
