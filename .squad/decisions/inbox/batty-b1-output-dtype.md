# B1: Output dtype sourced from ORT graph value info, not guessed from inputs

**Date:** 2026-08-11
**Author:** Batty (Engine Dev)
**Status:** Implemented (pending Chew's test updates)

## Problem

`CompiledKernelEntry.output_dtype` was a single `DataType` guessed from the first input's dtype. This produces silently wrong tensors for:
- `Cast(f32 → i64)` — would allocate f32 output instead of i64
- `Where(bool, f32, f32)` — first input is bool, output should be f32
- `Shape(f32) → i64` — output is always i64 regardless of input
- Any multi-output op where outputs differ in type

## Solution

1. **`CompiledKernelEntry.output_dtypes: Vec<DataType>`** — per-output dtype vector read from the ORT graph's value info at Compile time. Never inferred from inputs.

2. **Claim-time output producibility validation** in `GetCapability`: any node with an `Undefined` output dtype is declined before the registry dtype filter runs. Fail-closed: a declined node is correct-and-slower; a mis-typed claimed node is silently wrong.

3. **Compute path** uses per-output dtypes for both the single-kernel fast path and the routed multi-node path.

4. **No `OrtGraph*`/`OrtNode*` pointers cached past Compile** — dtypes are copied into owned `Vec<DataType>` during `ep_compile_inner` via `view.value(val_idx).dtype`.

## Files changed

- `crates/onnx-runtime-ep-plugin/src/compute.rs` — `CompiledKernelEntry` field rename, compute paths use per-output dtypes
- `crates/onnx-runtime-ep-plugin/src/ep.rs` — Compile reads per-output dtypes from IR graph; GetCapability adds Undefined-output filter

## Chew action required

`crates/onnx-runtime-ep-cpu-plugin/tests/plugin_export_abi.rs` lines 363 and 429: change `output_dtype: DataType::Float32` → `output_dtypes: vec![DataType::Float32]`.

## Follow-up: Multi-output shape inference fix (2026-08-11)

### Bug

`ShapePreservingNorm` variant emitted `vec![input_shape; num_outputs]` — all outputs
got input[0]'s full shape. Per ONNX spec, LayerNormalization's Mean and InvStdDev
outputs have shape `[d[0]..d[axis-1], 1, .., 1]` (keepdims reduction over normalised
axes). For input `[2,4]` with `axis=-1`, Mean should be `[2,1]` not `[2,4]`.

### Fix — Structural, not per-op

Replaced `ShapePreservingNorm` with `LayerNorm { axis, num_outputs, full_shape_outputs }`.

- `axis` is resolved to a non-negative index at `for_node` time (handles negative axis).
- Output 0 always gets input[0]'s shape.
- Outputs 1+ get the reduced shape (dims from axis onward → 1), unless listed in
  `full_shape_outputs` (e.g. SkipLayerNormalization's 4th output).
- `for_op_domain` now declines these ops (requires axis + input shapes), ensuring
  fail-closed: if axis can't be resolved, the node is declined rather than claimed
  with wrong shapes.

### Ops checked

| Op | axis source | Outputs checked |
|----|-------------|-----------------|
| LayerNormalization | `axis` attr (default -1) | Y (full), Mean (reduced), InvStdDev (reduced) |
| SimplifiedLayerNormalization | `axis` attr (default -1) | Y (full), InvStdDev (reduced) |
| RMSNormalization | `axis` attr (default -1) | Y (full), InvStdDev (reduced) |
| SkipLayerNormalization | last axis (contrib, no attr) | output (full), mean (reduced), inv_std_dev (reduced), input_skip_bias_sum (full) |
| SkipSimplifiedLayerNormalization | last axis (contrib, no attr) | output (full), inv_std_dev (reduced) |

### Chew action required

`conformance_layer_norm_multi_output` in `plugin_ort_e2e.rs` can have its `#[ignore]` removed — it passes with `--include-ignored`.
