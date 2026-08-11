# B2: ReleaseEpFactory signature corrected to OrtStatus*

**Date:** 2026-08-11
**Author:** Sapper
**Status:** Implemented (pending shim updates by file owners)

## Decision

Fixed `ReleaseEpFactory` in the `export_ep_factories!` macro to return
`*mut OrtStatus` instead of `void`, matching the ORT header:

```c
// onnxruntime_ep_c_api.h:2669
typedef OrtStatus* (*ReleaseEpApiFactoryFn)(_In_ OrtEpFactory* factory);
```

## What changed (B2 first pass — macro only)

- **`lib.rs` macro**: `ReleaseEpFactory` now returns `-> *mut OrtStatus`.
  On success, returns the status from `factory::release_ep_factory` (nullptr).
  On caught panic, returns an error status via `panic_to_fail_status` instead
  of silently swallowing.
- **`status.rs`**: Clarified `fail_status` doc re: null-as-success pre-init window.

## What still needed fixing — CPU shim (B2 follow-up, now complete)

### `ReleaseEpFactory` — FIXED (2026-08-11)

`crates/onnx-runtime-ep-cpu-plugin/src/lib.rs`: hand-written `ReleaseEpFactory`
returned `void`. Updated to return `*mut OrtStatus`, catching panics and
surfacing them as error statuses. The function stays hand-written (cannot use
the macro because `CreateEpFactories` calls
`factory::create_ep_factories_with_registry`, a path not covered by the macro).
A comment marks the body as a mirror of the macro arm with a keep-in-sync note.

### `CreateEpFactories` — signature verified, no fix needed

The hand-written `CreateEpFactories` already returned `*mut OrtStatus` and its
parameter list matches the macro and header exactly. No drift found.

## CUDA shim (Iran's file — not touched)

`crates/onnx-runtime-ep-cuda-plugin/src/lib.rs` still has the same void-return
bug. Iran owns that file and is fixing it separately.

## Chew — ABI test update needed

The ABI test at `crates/onnx-runtime-ep-cpu-plugin/tests/plugin_export_abi.rs:71`
currently declares:
```rust
type ReleaseEpFactory = unsafe extern "C" fn(*mut ort::OrtEpFactory);
```
Must be changed to:
```rust
type ReleaseEpFactory = unsafe extern "C" fn(*mut ort::OrtEpFactory) -> *mut ort::OrtStatus;
```
And line 118 (`// Release factory — ReleaseEpFactory returns void`) must be
updated to check the returned status is null (success).
