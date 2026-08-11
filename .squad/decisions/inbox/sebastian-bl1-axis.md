# BL1: LayerNorm axis resolution deferred to runtime

**Date:** 2026-08-11
**Author:** Sebastian (Performance Engineer)
**Status:** Implemented

## Decision

LayerNorm's `axis` attribute is no longer pre-resolved at claim/compile time against a truncated static shape. Instead, the raw `axis` (possibly negative) is stored in `ShapeInference::LayerNorm { raw_axis, .. }` and resolved against the **runtime input rank** in `infer_shapes()`.

## Rationale

The old code used `filter_map(|d| d.as_static())` to build input shapes, which drops symbolic (dynamic) dimensions. For a model with `[B, S, H]` inputs where B and S are symbolic, this collapsed the shape to `[H]` (rank 1). Resolving `axis=-1` against rank 1 yields index 0, making the reduced shape `[1]` instead of the correct `[B, S, 1]`.

The fix follows the "fail closed or defer to runtime" principle: when static information is incomplete, don't guess — wait until runtime provides the true shape.

## Scope

- `crates/onnx-runtime-ep-plugin/src/compute.rs`: `ShapeInference::LayerNorm` variant now stores `raw_axis: i64`; `infer_shapes` resolves it.
- `crates/onnx-runtime-ep-plugin/src/ep.rs`: BL3 carry-over (absent inputs → `NodeInputSource::Absent`); registry build surfaces failures.
