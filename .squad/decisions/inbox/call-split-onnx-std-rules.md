### 2026-07-27: Split ONNX validation rules by model layer
**By:** Call
**What:** Split the former 5,316-line `crates/onnx-std/src/check/rules.rs` into a `rules/mod.rs` facade and five private rule-family modules:
- `graph_topology.rs` — 368 lines; opset imports, duplicate names, graph input/output connectivity, and acyclicity.
- `schema_types.rs` — 1,217 lines; schema conformance, type constraints, initializer declarations, metadata, attributes, and retained protobuf types.
- `ir_version_functions.rs` — 1,147 lines; IR version/feature gates and local function validation. The two existing `#[allow(clippy::too_many_arguments)]` attributes remain on their original functions.
- `tensor_sparse_payloads.rs` — 558 lines; dense tensor payload and sparse tensor validation.
- `multi_device.rs` — 393 lines; device configuration and sharding validation.
- `mod.rs` — 1,711 lines; public facade, shared diagnostic helpers, and unchanged tests.

**Why:** Cohesive private modules reduce the validation implementation's file-level entropy while preserving the flat public API. Rule ORDER is unchanged because `check/mod.rs` and its 17 `checker.add_rule(...)` calls were not modified. Violation WORDING is unchanged: all 579 Rust string literals were compared as multisets before formatting and preserved exactly; the non-author reviewer independently approved the split and found rule implementations/helper logic unchanged.

**Gates:** `cargo fmt -p onnx-std` passed; `cargo build -p onnx-std` passed; `cargo test -p onnx-std` passed (126 unit tests, 23 integration tests, 1 doc-test); `cargo clippy -p onnx-std --all-targets -- -D warnings` passed. Non-author review: approved with no blocking findings.
