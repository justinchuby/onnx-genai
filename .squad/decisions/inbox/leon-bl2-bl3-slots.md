# BL2 + BL3: Optional Slot Positional Integrity

**Date:** 2026-08-11
**Author:** Leon (Engine Dev)
**PR:** #762

## Decision

Adopted the **preferred slot-map fix** for BL2 (output positions) rather than
fail-closed. For BL3 (input positions), added `NodeInputSource::Absent` variant
with a genuine absent `TensorView` in the compute path.

## BL2 — Omitted Optional Outputs

**Root cause:** `graph_reader.rs` used `filter_map` to drop empty-named output
slots, compacting the output vector. ONNX indexes outputs by position, so
downstream kernels wrote to wrong positions (e.g., mean into the sum slot).

**Fix:** Replace `filter_map` with positional preservation. Empty-named output
slots get placeholder `ValueId`s with `DataType::Undefined`. In the compute
fast path, `Undefined`-dtype slots receive local scratch buffers (so the kernel
sees full output arity) while only present slots are allocated through ORT's
kernel context with sequential ORT indices.

## BL3 — Omitted Optional Inputs (partial)

**Root cause in ep.rs (not mine):** `build_subgraph_routing` maps `None` inputs
to `NodeInputSource::Ort(0)`, aliasing absent inputs to the first ORT input.

**Fix (compute.rs only):** Added `NodeInputSource::Absent` variant. The compute
loop passes `TensorView::absent(DataType::Undefined)` for these, which kernels
detect via `is_absent()`.

**Remaining:** `ep.rs:597` still emits `Ort(0)` for None inputs — Sebastian's
pass must change it to `Absent`. Until then, single-node fast-path SkipLayerNorm
with omitted beta/bias works because the kernel checks `inputs.len()` and
`is_absent()`, but fused multi-node subgraphs with absent intermediate inputs
will still alias incorrectly.

## Nonblocker — unwrap_or(DataType::Float32)

Removed all three `unwrap_or(DataType::Float32)` fallbacks in compute.rs. Now
fails closed with an explicit error when the `output_dtypes` vector is too short.

## Impact

- No changes to `ep.rs` (Sebastian's domain).
- No changes to `nxrt-abi`, `cuda-plugin`, or existing `plugin_ort_e2e.rs`.
- 4 new conformance tests prove correctness against real ORT 1.27.
