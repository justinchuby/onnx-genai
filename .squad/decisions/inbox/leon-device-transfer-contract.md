# Leon: Device Data-Transfer Contract (2026-08-11)

## Summary

Implemented `crates/onnx-runtime-ep-plugin/src/transfer.rs` — the ORT `OrtDataTransferImpl` adapter projecting Rust EP `copy`/`copy_async`/`copy_from_host`/`copy_to_host` through the C ABI.

## Transfer Contract (from `onnxruntime_ep_c_api.h`)

1. **ORT calls `CreateDataTransfer` once per factory** → returns `OrtDataTransferImpl*` owned by ORT.
2. **`CanCopy(src_device, dst_device)`** — advisory; `CopyTensors` error is definitive.
3. **`CopyTensors(src[], dst[], streams[], n)`** — stream-ordered if `streams[i]` non-null.
4. **`Release`** — ORT calls exactly once when done.

## Copy-Direction Matrix

| Direction              | Supported | Method            |
|------------------------|-----------|-------------------|
| Host → Device          | ✓         | `copy_from_host`  |
| Device → Host          | ✓         | `copy_to_host`    |
| Device → Same Device   | ✓         | `copy`            |
| Device → Other Device  | ✗         | fail closed       |
| Host → Host            | ✗         | ORT handles       |

## Device-Pointer Safety

- CUDA `DeviceBuffer` is tagged `DeviceId::cuda(N)` → `is_host_accessible() == false`.
- `kernel_ctx.rs` rejects null data pointers ("device-only memory not supported").
- EP trait default `copy_from_host`/`copy_to_host` checks `is_host_accessible()` before any deref.
- The adapter never dereferences raw pointers from OrtValue — delegates to EP methods.

## Stream-Ordering Guarantee

- When `streams[i]` is non-null, `copy_async` is used → returns a `Fence`.
- `wait_fence` orders the compute stream after the transfer.
- Consumer reads after stream flush are guaranteed to observe the copied data.
- Synchronous EPs return `Fence::signalled()` → immediate visibility.

## Ownership

- `DeviceDataTransfer` is `Box::into_raw` → ORT owns → `Release` does `Box::from_raw` → drop.
- EP pointer is **borrowed** (not freed on release). ORT guarantees release before factory release.
- `DeviceDataTransferFull` additionally stores `*const OrtApi` for real tensor data extraction.

## Constructor Signature for `factory.rs` (Deckard)

```rust
// Basic (no tensor-data extraction — fail-closed CopyTensors for real device):
let transfer = unsafe { DeviceDataTransfer::new(ep_ptr, support.clone()) };
let raw = Box::into_raw(transfer) as *mut OrtDataTransferImpl;
*out_data_transfer = raw;

// Full (with OrtApi for actual copy operations):
let transfer = unsafe { DeviceDataTransferFull::new(ep_ptr, support.clone(), api_ptr) };
let raw = Box::into_raw(transfer) as *mut OrtDataTransferImpl;
*out_data_transfer = raw;
```

Where:
- `ep_ptr: *const dyn ExecutionProvider` — same pointer used for allocator/stream
- `support: DeviceSupport` — factory's device-support config
- `api_ptr: *const OrtApi` — from `GetApi(ORT_API_VERSION)`

## What Is Genuinely Tested Without a GPU

- Copy-direction classification (all 5 combinations)
- `CanCopy` behavior: false for host-accessible, true for device EP, false on null
- `CopyTensors` null checks, zero-count success, fail-closed for host EP
- `DeviceDataTransfer`/`DeviceDataTransferFull` lifecycle (create → release, no leak)
- Mock device EP: host→device, device→host, device→device copy with data verification
- Async copy with non-signalled fence + wait_fence path
- Drop-counter leak detection
- Device-pointer non-host-accessibility assertion

## What Remains Hardware-Gated

- **Actual CUDA memory operations** — `cudaMemcpy*`, `cudaStreamSynchronize` — require toolkit + GPU.
- **OrtEpApi `Value_GetMemoryDevice`/`MemoryDevice_GetDeviceType`** — needed for proper src/dst classification in `CanCopy`; requires live ORT session.
- **End-to-end ORT data transfer** — `CopyTensors` with real OrtValues needs a full ORT session with our EP loaded on a GPU host.
- **Cross-device (multi-GPU) rejection** — cannot be tested without 2+ GPUs.

## ⚠️ Explicit disclaimer

Nothing here proves CUDA works. The adapter is structurally correct and compile-checked, the contract matches the header, and the mock exercises all code paths — but actual device memory operations are hardware-gated.
