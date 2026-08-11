# B4 — CUDA Plugin Fail-Closed Implementation

**Date:** 2026-08-11
**Author:** Iran (CPU/systems optimization)
**Status:** Implemented (pending commit)

## Decision

The CUDA plugin (`onnx-runtime-ep-cuda-plugin`) now returns **zero EP factories** in both feature configurations (`cuda` ON and OFF). This is a fail-closed gate: the plugin will not advertise a GPU EP to ORT until the implementation defects are resolved.

## Rationale

The Opus reviewer correctly identified that the CUDA EP was **implementation-blocked, not hardware-blocked**. Even with a GPU in hand, the code could not work as written. Advertising a broken GPU EP is worse than not shipping one — ORT would route real work to it and get silent corruption.

## Four Implementation Defects (Specification for Future Work)

### Defect #1: Separate CUDA Runtime/Context per Component
The EP, allocator (`DeviceAllocator`), and stream (`DeviceSyncStream`) each construct independent CUDA runtime/context instances. **Required fix:** Share a single `CUcontext` + `cudaStream_t` across all three, passed at construction time.

### Defect #2: CreateDataTransfer Returns Non-Functional Transfer
The `DeviceDataTransfer` lacks access to `OrtApi` (needed to extract tensor data pointers from `OrtValue`) and has no shared CUDA stream for `cudaMemcpyAsync`. `CopyTensors` returns an error unconditionally. **Required fix:** Store `OrtApi` and shared `cudaStream_t` at construction; implement actual `cudaMemcpyAsync`-based copies.

### Defect #3: GetHandle Returns NULL Stream Handle
`stream_get_handle` returns `null_mut()` — there is no `cudaStream_t`. ORT consumers that call `GetHandle` to order work on the EP's stream receive NULL. **Required fix:** Return the shared `cudaStream_t` from defect #1.

### Defect #4: Free Passes size=0 Violating Allocator Contract
`device_free` reconstructs a `DeviceBuffer` with `size=0` because allocation sizes are not tracked. **Required fix:** Track allocation sizes in a side table (`HashMap<*mut c_void, usize>`) populated by `device_alloc`, or make free-by-pointer an explicit EP contract.

## CanCopy Fail-Closed Fix

Both `transfer_can_copy` and `transfer_full_can_copy` previously returned `true` unconditionally for device EPs — a fail-open bug. They now return `false`, letting ORT fall back. When defect #2 is resolved, restore direction-aware `CanCopy` using `CopyDirection::classify + is_supported`.

## Files Changed

- `crates/onnx-runtime-ep-cuda-plugin/src/lib.rs` — fail-closed gate + defect documentation
- `crates/onnx-runtime-ep-plugin/src/transfer.rs` — `CanCopy` returns false for device EPs
- `crates/onnx-runtime-ep-plugin/src/device.rs` — documented defect #4 in `device_free`

## What Chew Should Assert

For the CUDA plugin fail-closed behavior, tests should verify:
1. `CreateEpFactories` sets `*out_num = 0` in both feature configs
2. `CreateEpFactories` returns a non-null error status (when ORT API is loaded)
3. The error message contains "IMPLEMENTATION-BLOCKED" (with `cuda` feature)
4. The error message contains "without `cuda` feature" (without `cuda` feature)
5. `CanCopy` returns `false` for any `DeviceDataTransfer` backed by a device EP
6. `CanCopy` returns `false` for any `DeviceDataTransferFull` backed by a device EP

## Note for Sapper (factory.rs)

No changes needed in `factory.rs` for fail-closed — the gate is in the CUDA plugin's `CreateEpFactories` which now never calls `create_ep_factories_with_device_support`. The factory code remains correct for when a working EP is eventually wired.
