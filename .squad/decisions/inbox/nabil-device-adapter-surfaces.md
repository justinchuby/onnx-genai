# Device Adapter Surfaces — Ownership Contracts & Design

**Date:** 2026-08-10
**Author:** Nabil
**Branch:** `squad/ep-plugin-parity-cuda`

## What was implemented

### 1. `device.rs` — Generalized device surface module

New module `crates/onnx-runtime-ep-plugin/src/device.rs` providing:

- **Device type mapping**: `device_type_to_ort_hardware()` maps nxrt `DeviceType` → `OrtHardwareDeviceType` (CPU/GPU/NPU).
- **`DeviceSupport` config struct**: Declares which hardware types an EP serves, its allocator name, vendor ID, stream-awareness, and host-accessibility. Provides `cpu_only()` and `gpu()` constructors.
- **`DeviceAllocator`** (`#[repr(C)]`): Projects EP `allocate`/`deallocate` through ORT's `OrtAllocator` vtable. Panic-guarded. Null-checked.
- **`DeviceSyncStream`** (`#[repr(C)]`): Projects EP `sync()` through ORT's `OrtSyncStreamImpl` vtable with `Release`/`GetHandle`/`Flush`/`OnSessionRunEnd`.
- **Fail-closed validators**: `validate_device_support`, `validate_allocator_request`, `validate_stream_request` — each returns `OrtStatus` error on mismatch.

### 2. `onnx-runtime-ep-cuda-plugin` shim crate

Mirrors `onnx-runtime-ep-cpu-plugin`. Without `cuda` feature (default): compiles to a stub that panics in the constructor (caught by `export_ep_factories!` panic guard → returns 0 factories + error status). With `cuda` feature: projects `CudaExecutionProvider` through the ABI.

Workspace member but NOT a default-member. `cargo check --workspace` succeeds without CUDA toolkit.

## Ownership contracts from headers

### OrtMemoryInfo → EpDevice_AddAllocatorInfo (line 1092–1111)
ORT stores the raw pointer. Does NOT copy. Must outlive OrtEpDevice. ORT releases via `ReleaseEpDevice`. Do NOT call `ReleaseMemoryInfo` after successful `AddAllocatorInfo`.

### OrtSyncStreamImpl (line 204–258)
ORT calls `Release` on the vtable when done. Implementation must free resources in its `Release` callback. `ort_version_supported` field must be `ORT_API_VERSION`.

### OrtAllocator (line 2821–2835, factory; 2312–2329, EP)
Created by factory or EP. Released by `OrtEpFactory::ReleaseAllocator`. The `Info()` vtable field must return a valid `OrtMemoryInfo*` for the lifetime of the allocator.

### OrtHardwareDevice (line 1225–1241)
Created by EP inside `GetSupportedDevices` via `OrtEpApi::CreateHardwareDevice`. ORT takes ownership of returned `OrtEpDevice` array entries.

## What factory.rs needs (integration)

`factory.rs` currently hard-codes `OrtHardwareDeviceType_CPU` filtering. To adopt the new surface:

1. Replace the `if dev_type != ort::OrtHardwareDeviceType_CPU { continue; }` check with:
   ```rust
   let support = device::DeviceSupport::from_ep(&ep); // or pass config
   if !support.serves(dev_type) { continue; }
   ```

2. Use `support.memory_device_type()` and `support.allocator_name` in the `CreateMemoryInfo_V2` call instead of hard-coded `OrtMemoryInfoDeviceType_CPU` / `"Cpu"`.

3. For GPU EPs, add a second `EpDevice_AddAllocatorInfo` with `OrtDeviceMemoryType_HOST_ACCESSIBLE` (pinned host memory).

4. Replace `factory_is_stream_aware` → return `support.stream_aware`.

5. Replace `factory_create_sync_stream` → use `DeviceSyncStream::new(ep_ptr)` when stream-aware.

6. Replace `factory_create_allocator` → use `DeviceAllocator::new(ep_ptr, mem_info)` for device EPs.

## What is now mechanical vs hardware-gated

| Surface | Status | Notes |
|---------|--------|-------|
| Device enumeration for GPU/NPU | ✅ Ready | `DeviceSupport` + validators. `factory.rs` integration is a 1-function change. |
| Allocator vtable projection | ✅ Ready | `DeviceAllocator` tested with mock. Real CUDA EP just needs `allocate`/`deallocate` to call cuMemAlloc/cuMemFree. |
| Stream vtable projection | ✅ Ready | `DeviceSyncStream` tested with mock. Real CUDA EP wraps cudaStream_t in `GetHandle`, calls `cudaStreamSynchronize` in `Flush`. |
| Data transfer | 🔶 Design only | Needs `OrtDataTransferImpl` vtable (similar pattern). Blocked on needing real device memory for integration testing. |
| Kernel execution with device pointers | 🔶 Requires hardware | Opaque device pointers in `OrtValue` need real CUDA context. |
| Multi-GPU ordinal selection | ✅ Ready | `CreateHardwareDevice(GPU, vendor_id, device_id, ...)` maps to EP's `device_id().index`. |

## Tests added

~26 unit tests in `device.rs`:
- Device type mapping (Cuda→GPU, Rocm→GPU, Cpu→CPU, Qnn→NPU, WebGpu→None)
- EP hardware support validation (positive + negative paths)
- DeviceSupport config (serves/not-serves, stream-aware, host-accessible)
- Fail-closed validators (device mismatch, allocator mismatch, non-stream-aware)
- Allocator alloc/free roundtrip with MockGpuEp
- Stream flush/release/null-safety with MockGpuEp
- Hardware→memory type mapping

All tests exercise real code paths via mock EPs (MockGpuEp implementing `ExecutionProvider` with host-malloc pretending to be device memory).

## Blocked

- `ep.rs:114` references undefined `ep_get_kernel_registry` (Deckard's concurrent work). Causes `cargo build -p onnx-runtime-ep-plugin` to fail. `cargo check` passes because it skips monomorphization of that path. Tests cannot run until that's resolved.
