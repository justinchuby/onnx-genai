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

## What changed

- **`lib.rs` macro**: `ReleaseEpFactory` now returns `-> *mut OrtStatus`.
  On success, returns the status from `factory::release_ep_factory` (nullptr).
  On caught panic, returns an error status via `panic_to_fail_status` instead
  of silently swallowing.
- **`status.rs`**: Clarified `fail_status` doc re: null-as-success pre-init window.

## What still needs fixing (not my files)

The CPU shim (`crates/onnx-runtime-ep-cpu-plugin/src/lib.rs:98`) and CUDA shim
(`crates/onnx-runtime-ep-cuda-plugin/src/lib.rs:163`) have hand-written
`ReleaseEpFactory` with `void` return. They do NOT use the macro. Their owners
must update to `-> *mut OrtStatus` and propagate the `release_ep_factory` return.

## CreateEpFactories verification

`CreateEpFactories` in the macro already matches the header parameter-for-parameter:
```
Header:  OrtStatus* (const char*, const OrtApiBase*, const OrtLogger*, OrtEpFactory**, size_t, size_t*)
Macro:   *mut OrtStatus (c_char, OrtApiBase, OrtLogger, *mut OrtEpFactory, usize, *mut usize)
```
No second mismatch found. ✓

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
