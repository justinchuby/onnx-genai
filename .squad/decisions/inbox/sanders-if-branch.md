### 2026-07-23: Preserve dynamic rank for mismatched If branches
**By:** Sanders
**What:** Updated `crates/onnx-runtime-shape-inference/src/infer.rs` so standard ONNX `If` outputs with matching element types but different branch ranks no longer fail inference. Such outputs retain the branch element type, are marked shape-unknown, have no `TypeInfo` (the shape-inference crate's established unknown-rank representation), and remain listed as unresolved. The equal-rank per-dimension merge is unchanged, including parent-symbol preservation and fresh dimensions for incompatible dimensions. Added `Graph::mark_value_type_known` in `crates/onnx-runtime-ir/src/graph.rs` so dtype-only results can replace placeholder element types without inventing a rank. No executor change was needed: `exec_if` runs only the selected subgraph, then `store_output_tensor`/`store_output_bytes` resize storage and record the selected tensor's runtime dtype and concrete shape in `value_dtypes`, `buffer_shapes`, and `resolved`.
**Why:** ONNX requires `If` branch element types to match, but permits branch shapes and ranks to differ. Rejecting rank-2/rank-3 and rank-4/rank-5 branches blocked valid Nemotron VAD and Whisper jump-times graphs even though runtime execution already supports the selected branch's actual shape.

Tests added/verified:
- rank 2 versus rank 3 succeeds with Float16 dtype and unknown rank;
- rank 4 versus rank 5 succeeds with Float16 dtype and unknown rank;
- equal-rank concrete and symbolic merge tests remain green;
- mismatched branch element types still return `ShapeInferError::Invalid`.
- `cargo test -p onnx-runtime-shape-inference`: 210 passed (14 unit + 17 graph + 178 op-rule + 1 doctest).
- `cargo build --release -p onnx-runtime-ep-cpu --features mlas`: passed.
- The requested default-feature `cargo test -p onnx-runtime-session` is blocked by pre-existing unguarded `mlas_sys` references in NCHWc code while `mlas-sys` is optional. With MLAS enabled, 62 library tests, 1 BERT conformance test, 3 optimized parity tests, and the focused selected-branch `If` executor test passed. The full control-flow target has two pre-existing failures: an optimizer `MissingProducer(ValueId(3))` in the mismatched-output-count fixture and the dtype-mismatch fixture expecting a runtime error although shape inference correctly rejects it during session construction.

Whisper jump-times and Nemotron VAD model files were not available in this worktree, so validation is at the general unit-test level. Their rank-4/rank-5 and rank-2/rank-3 `If` patterns now pass shape inference with dynamic-rank outputs.
