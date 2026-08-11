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
