### 2026-08-04: GraphIoMetadata excludes initializer-backed graph inputs

**By:** Quaid

**What:** In `crates/onnx-genai-ort/src/loader.rs`, `graph_io_from_model_path` now
collects `graph.initializer` names into a `HashSet` and skips any `graph.input`
entry whose name is an initializer before building `GraphIoMetadata.inputs`.
Added a hermetic regression test in
`crates/onnx-genai-ort/tests/pipeline_loader.rs`
(`graph_io_metadata_excludes_initializers_and_matches_session_geometry`) that
authors an in-test decoder with (a) a weight listed in both `graph.input` and
`graph.initializer` and (b) rank-4 fp16 paged-KV I/O, then asserts metadata
input/output names, fp16 dtypes, and KV geometry `[-1,2,-1,4]`, cross-checking
byte-for-byte against a real ORT `Session` over the same model.

**Why:** ONNX permits (mandatory pre-IR-4, still legal IR>=4) an initializer to
also appear in `graph.input`. Both ORT's `Session` (GetInputCount) and this
repo's native graph loader (`onnx-runtime-loader::graph_builder`, §2) exclude
those. Without the exclusion, `GraphIoMetadata.inputs()`/`input_names()` leaked
weight tensors, which would (1) trip the decode float-rank>=3 native-load guard
in `decode/resolved_io.rs::from_structure`, falsely rejecting a valid decoder,
and (2) route leaked weights as spurious unroutable ports in
`pipeline/routing.rs::build_step_bindings`. Fix is general — no op/model
special-casing. Resolves Harry's [MAJOR] rejection on PR #625.
