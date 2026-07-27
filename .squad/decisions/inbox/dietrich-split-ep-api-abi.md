### 2026-07-27: Split the plugin EP ABI bridge by responsibility
**By:** Dietrich
**What:** Moved `crates/onnx-runtime-ep-api/src/abi.rs` to `abi/mod.rs` and split implementation details into `runtime.rs`, `host.rs`, `ffi_helpers.rs`, and `weights.rs`. The facade retains `OrtGraphView`, `SubgraphClaim`, and `PluginExecutionPlan`; it re-exports `PluginCompiledKernel` at the unchanged `abi::PluginCompiledKernel` path. The host projection test is colocated in `host.rs`.
**Why:** The 2,512-line plugin-EP ORT C-ABI boundary was difficult to review safely. This is pure code motion with only minimally scoped `pub(super)` visibility needed between sibling modules.

Module breakdown:
- `abi/mod.rs`: facade, stable public surface, graph view, claims, execution plan — 429 LOC.
- `abi/runtime.rs`: plugin runtime ownership, shared kernel state, compiled kernel — 304 LOC.
- `abi/host.rs`: ORT host projections, C-ABI vtables/callbacks, and host projection test — 1,618 LOC.
- `abi/ffi_helpers.rs`: raw-pointer conversions and plugin-device accessors — 127 LOC.
- `abi/weights.rs`: mapped external-weight cache and initializer projection — 103 LOC.

Invariant counts across the ABI module tree:
- ABI root LOC: 2,512 before; 429 after.
- `unsafe {` blocks: 82 before; 82 after.
- `#[cfg(...)]` attributes: 1 before; 1 after.
- `extern "C"` occurrences: 59 before; 59 after.
- `#[no_mangle]` attributes: 0 before; 0 after.

Public API is unchanged: all prior bare-`pub` items and methods retain their paths and signatures; `PluginCompiledKernel` is re-exported from the facade.

Validation:
- `cargo build -p onnx-runtime-ep-api`: passed.
- `cargo test -p onnx-runtime-ep-api`: passed (38 unit tests, 7 integration tests, 0 failures).
- `cargo clippy -p onnx-runtime-ep-api --all-targets -- -D warnings`: passed.
- `cargo build -p onnx-runtime-session`: passed.
- `cargo fmt -p onnx-runtime-ep-api`: passed.
