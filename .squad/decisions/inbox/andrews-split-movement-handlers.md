### 2026-07-27: Split movement shape handlers by operator family
**By:** Andrews
**What:** Replaced the 1,809-line `handlers/movement.rs` with:
- `movement/mod.rs` (114 lines): shared helpers and the unchanged registration facade.
- `movement/transform.rs` (409 lines): Transpose, Reshape, Flatten, Squeeze, Unsqueeze, Expand.
- `movement/resize.rs` (302 lines): Resize.
- `movement/concat_slice.rs` (394 lines): Concat, Slice.
- `movement/split_gather.rs` (380 lines): Split, Gather, GatherElements, GatherND.
- `movement/scatter.rs` (137 lines): Scatter, ScatterElements, ScatterND, Trilu.
- `movement/space_depth.rs` (132 lines): DepthToSpace, SpaceToDepth.

The split totals 1,868 lines including module-local imports. Registration order, operator/opset mappings, handler bodies, shape rules, and diagnostic text are unchanged.

**Why:** Cohesive operator-family modules reduce navigation and review cost while keeping this change mechanical and behavior-preserving. `cargo fmt -p onnx-runtime-shape-inference`, shape-inference build/tests (225 tests plus one doctest), clippy with `-D warnings`, and downstream `onnx-runtime-session` build all pass.
