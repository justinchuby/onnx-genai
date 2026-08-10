# Decision: ORT Plugin-EP Export Adapter (Outbound ABI)

**Author:** Nabil  
**Date:** 2026-08-10T20:12:35.793+00:00  
**Status:** Implemented (v1 — CPU EP)

## Context

Justin requested that all in-repo EPs be callable by upstream ORT as real plugin EPs. The inbound direction (nxrt hosts foreign plugin EPs) existed; the outbound direction did not.

## Decision

### Architecture

Two new crates:
- **`onnx-runtime-ep-plugin`** (lib) — shared adapter owning 100% of unsafe FFI. Projects any `ExecutionProvider` through the ORT plugin-EP C ABI via the `export_ep_factories!()` macro.
- **`onnx-runtime-ep-cpu-plugin`** (cdylib + lib) — thin shim that constructs `CpuExecutionProvider` and invokes the macro.

### Export symbol name

**Assumed: `CreateEpFactories`** — matching the inbound loader. Pris's test plan suggests `CreateEpApiFactories`. The name is behind `onnx_runtime_ep_plugin::EXPORT_SYMBOL_CREATE` constant for a one-line fix when Challenger's verdict arrives.

### Workspace integration

Both crates are workspace **members** but NOT in `default-members`. A bare `cargo build` does not build them and does not require an ORT C library.

### `OrtPluginExport` placeholder removed

The empty `OrtPluginExport` struct and `as_ort_plugin()` trait method were removed per Justin's directive. The real export mechanism is the external cdylib crate, not a trait method.

## What landed (v1)

- `CreateEpFactories` / `ReleaseEpFactory` C ABI exports ✓
- `OrtEpFactory` vtable: `ort_version_supported`, `GetName`, `GetSupportedDevices`, `CreateEp`, `ReleaseEp` ✓
- `OrtEp` vtable: `GetCapability`, `Compile`, `ReleaseNodeComputeInfos` ✓
- `OrtNodeComputeInfo` vtable: `CreateState`, `Compute` (returns NOT_IMPLEMENTED — fail-closed), `ReleaseState` ✓
- Reuses `OrtGraphView::query_capabilities()`, `UnionFind`, `SubgraphClaim` from the inbound path ✓
- L2 dlopen integration test ✓

## What remains

- **Compute path** — output shape inference bridging needed for kernel execution
- **CUDA EP** — device memory, stream ownership, cuBLAS/cuDNN handle binding
- **Allocator callbacks** — needed for device EPs
- **EPContext save/load** — deferred
- **Custom ops** — orthogonal

## Alternatives considered

- Feature-gating the export in the EP crate itself (rejected: pollutes the EP crate with cdylib concerns)
- Proc-macro instead of declarative macro (rejected: unnecessary complexity for a simple pattern)
