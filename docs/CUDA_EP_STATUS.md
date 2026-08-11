# CUDA EP Status — Compiles, Unvalidated on Hardware

**Author:** Roy (Lead), Sapper (GPU/Systems)
**Updated:** 2026-08-11 (post-defect-resolution)
**Branch:** `squad/ep-plugin-parity-cuda` (draft PR #762)

---

## Summary — Four defects resolved; hardware validation pending

The B4 rubber-duck review found four implementation defects preventing the CUDA
plugin from functioning correctly. **All four have been resolved** in the shared
adapter (`onnx-runtime-ep-plugin`) and the CUDA plugin
(`onnx-runtime-ep-cuda-plugin`). The fixes are correct by construction but have
**not been validated on a physical CUDA GPU** — this host has no GPU.

**Current behavior:**
- With `cuda` feature ON: `CreateEpFactories` attempts to construct a
  `CudaExecutionProvider`. If a GPU is available, it advertises 1 factory. If
  no GPU is available, it returns 0 factories with an actionable error.
- With `cuda` feature OFF: returns 0 factories ("not available").
- The CPU path is unchanged and unaffected.

Issue [#768](https://github.com/justinchuby/onnx-genai/issues/768) tracks
hardware validation on a real CUDA GPU.

---

## 1. The Four Implementation Defects — Resolution Status

| # | Defect | Resolution | Status |
|---|--------|------------|--------|
| **1** | **Separate CUDA runtime/context per component.** | `ExportedFactory::shared_ep` holds a single `CudaExecutionProvider` wrapped in `Arc<Mutex<..>>`. All components (allocator, stream, data transfer) borrow this shared EP instead of constructing their own. Ownership tracked via `owns_ep` flag. | ✅ **Fixed** (unvalidated on hardware) |
| **2** | **`CreateDataTransfer` returns NULL.** | `factory_create_data_transfer` now creates a `DeviceDataTransferFull` for device EPs with: ORT API pointer for tensor data extraction, `OrtEpApi` pointer for `MemoryDevice_GetDeviceType`-based `CanCopy` direction classification, and the shared EP for actual copy operations. | ✅ **Fixed** (unvalidated on hardware) |
| **3** | **`GetHandle` returns NULL stream handle.** | `DeviceSyncStream` now carries a configurable `stream_handle` field. The CUDA plugin extracts the real `cudaStream_t` via `CudaRuntime::stream_ptr()` and stores it in `ExportedFactory::stream_handle`. `GetHandle` returns this handle. | ✅ **Fixed** (unvalidated on hardware) |
| **4** | **`Free` passes `size=0` violating the allocator contract.** | `DeviceAllocator` now tracks allocation sizes in a `Mutex<HashMap<usize, usize>>`. `Alloc` records `(pointer_addr, size)`. `Free` looks up the size and passes it to `deallocate`. | ✅ **Fixed** (unvalidated on hardware) |

**All four are resolved in code. None have been validated on GPU hardware.**

---

## 2. Implementation Details

### Shared EP Architecture (Defect #1)

**File:** `crates/onnx-runtime-ep-plugin/src/factory.rs`

```
ExportedFactory {
    shared_ep: Option<Arc<Mutex<Box<dyn ExecutionProvider + Send>>>>,
    stream_handle: *mut c_void,
    ...
}
```

When `shared_ep` is `Some`, factory callbacks (`CreateAllocator`,
`CreateSyncStreamForDevice`, `CreateDataTransfer`) borrow the shared EP
instead of calling the constructor. Each component tracks whether it owns
its EP reference (`owns_ep` flag) to prevent double-free.

The CPU path is unaffected: `shared_ep` defaults to `None` and
`host_accessible` is `true`, so CPU callbacks use ORT's built-in
allocator and never create device allocators/streams/transfers.

### Data Transfer (Defect #2)

**File:** `crates/onnx-runtime-ep-plugin/src/transfer.rs`

`DeviceDataTransferFull` now stores both `OrtApi*` and `OrtEpApi*`.
`CanCopy` classifies directions via `OrtEpApi::MemoryDevice_GetDeviceType`,
then applies the direction matrix:

| Direction | Supported | Method |
|-----------|-----------|--------|
| H→D | ✓ | `copy_from_host` |
| D→H | ✓ | `copy_to_host` |
| D→D (same) | ✓ | `copy` |
| D→D (cross) | ✗ | rejected |
| H→H | ✗ | ORT handles |

Without `OrtEpApi` (null), `CanCopy` returns `false` (fail-closed).

### Stream Handle (Defect #3)

**File:** `crates/onnx-runtime-ep-plugin/src/device.rs`

`DeviceSyncStream::stream_handle` is set to:
- `null_mut()` for non-stream-aware EPs (CPU)
- `CudaRuntime::stream_ptr() as *mut c_void` for CUDA

### Allocator Size Tracking (Defect #4)

**File:** `crates/onnx-runtime-ep-plugin/src/device.rs`

```
DeviceAllocator {
    alloc_sizes: Mutex<HashMap<usize, usize>>,
    owns_ep: bool,
    ...
}
```

`device_alloc` records `sizes.insert(ptr as usize, size)`.
`device_free` looks up `sizes.remove(&(p as usize))` and passes the
real size to `ep.deallocate()`.

---

## 3. Status of the Standalone CUDA EP (nxrt-native, not ORT plugin)

### Status levels

| Level | Meaning |
|---|---|
| **CODE EXISTS** | Source code compiles via `cargo check`. Correctness unproven. |
| **STUB** | Deliberate no-op or placeholder. Semantically honest. |
| **VALIDATED** | Run on a physical CUDA GPU with output compared to reference. **No capability has reached this level.** |

### Capability table

| Capability | Status | Notes |
|---|---|---|
| ORT plugin-EP export | **CODE EXISTS** | `CreateEpFactories` advertises 1 factory when GPU available. All four defects resolved. **Unvalidated on hardware.** |
| Device enumeration | CODE EXISTS | Returns `DeviceType::Cuda` and `DeviceId::new(Cuda, ordinal)`. |
| Initialize / bind device | CODE EXISTS | `runtime.bind()`. Not run on hardware. |
| Device allocator | CODE EXISTS | VMM arena or `cuMemAlloc`. Size-tracked in plugin adapter. |
| Data transfer (sync/async) | CODE EXISTS | Plugin adapter wired with OrtApi tensor access. |
| Stream synchronization | CODE EXISTS | `runtime.synchronize()`. |
| Op support query | CODE EXISTS | 109 entries in `src/kernels/mod.rs`. |
| Kernel execution | CODE EXISTS | cuBLASLt for GEMM; custom kernels. |
| `prefetch_lazy_weight` | **STUB** | Deferred to post-Phase-2a. Returns `Ok(false)`. |

**Every "CODE EXISTS" row is unvalidated.**

---

## 4. What Must Happen Before CUDA is Production-Ready

### Phase A: Hardware validation (Issue #768)

All four implementation defects are resolved. The remaining gate is
running the code on a physical CUDA GPU (compute ≥ 7.0, driver ≥ 535.x).
No self-hosted GPU runner exists in this repository.

### Phase B: ORT integration design gaps

| Gap | Description | Status |
|---|---|---|
| **Device-pointer ORT ABI marshaling** | ORT passes device pointers via `OrtKernelContext_GetInput`. | 🔴 Not designed |
| **Stream/context sharing with ORT** | ORT may provide its own context/stream. | 🔴 Not designed |
| **cuBLAS/cuDNN handle binding** | Handles bound to nxrt's stream. | 🔴 Not designed |
| **Weight paging in plugin model** | `CudaWeightResidency` manages its own VMM pool. | 🔴 Not designed |
| **CUDA graph capture** | Assumes stream ownership; incompatible with ORT. | 🔴 Not designed |

---

## 5. `prefetch_lazy_weight` — Stub Decision Record

**Location:** `crates/onnx-runtime-ep-cuda/src/provider.rs:564–573`

Returns `Ok(false)` — "no transfer enqueued." Deferred to post-Phase-2a.
Not a correctness bug; a functional gap.

---

## 6. Hardware Conformance Runner

`scripts/cuda_conformance_runner.sh` exits 2 (UNVALIDATED) on this host.
Every capability remains unvalidated. No self-hosted GPU workflow exists.
