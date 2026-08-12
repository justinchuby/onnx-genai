# Decision: Shared ort_discovery module via #[path]

**Date:** 2026-08-12
**Author:** Freysa (testing)
**PR:** #762

## Context

Three integration test files in `onnx-runtime-ep-cpu-plugin` each needed identical ORT
discovery logic (`find_ort_lib_dir`). Two had been refactored to inline `mod ort_discovery`
blocks; the third (`layernorm_dynamic_axis.rs`) still held a stale, non-platform-aware copy.

## Decision

Use `#[path = "common/ort_discovery.rs"] mod ort_discovery;` in all three test files,
pointing at a single canonical implementation in `tests/common/ort_discovery.rs`.

Rust's integration-test model makes `tests/common/mod.rs` awkward (it gets compiled as its
own test binary unless excluded). The `#[path]` attribute avoids that entirely — the module
is included textually at compile time, no extra binary, no `[[test]]` exclusion needed.

## Also

`validate_write_dtype` in `onnx-runtime-ep-api/src/tensor.rs` was documented to clarify it
is a test-only contract helper, not a runtime guard. `scratch_alloc_bytes` is the actual
production guard.
