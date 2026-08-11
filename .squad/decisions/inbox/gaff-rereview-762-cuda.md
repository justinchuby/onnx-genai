# Gaff Re-Review — PR #762 CUDA Plugin (Commits 2ca515eb7..7aba5cb93)

**Reviewer:** Gaff (adversarial, did not author this code)
**Date:** 2026-08-11
**Branch:** `squad/ep-plugin-parity-cuda`
**Head:** `7aba5cb93`

---

## BLOCKING

### B1: Shared EP raw pointer is dangling — use-after-free in all three callbacks

**Files:** `crates/onnx-runtime-ep-plugin/src/factory.rs:586-592, 686-692, 747-753`

In `factory_create_allocator`, `factory_create_data_transfer`, and
`factory_create_sync_stream`, the shared EP pointer is obtained by:

```rust
let guard = shared.lock().unwrap();
let ep_ref: &dyn ExecutionProvider = &**guard;
(ep_ref as *const dyn ExecutionProvider, false)
```

The `MutexGuard` (`guard`) is dropped at the closing brace of the `if let`
block. The raw pointer `ep_ref as *const dyn ExecutionProvider` becomes
**dangling immediately**. Every subsequent call through the allocator's `ep`,
the stream's `ep`, or the data transfer's `ep` is **undefined behavior** — a
use-after-free.

This is not theoretical. The pointer does not point into the `Arc`'s
heap-allocated `Mutex<Box<..>>` — it points into the `Box` *inside* the
`Mutex`, which is only safely accessible while the guard is held. Once the
guard drops, another thread could lock and move/swap the contents.

**Severity:** This is a soundness bug. On a real CUDA host, every
`device_alloc`, `device_free`, `stream_flush`, and `transfer_*_copy_tensors`
call dereferences this dangling pointer. Result: crash, corruption, or silent
wrong-device operations.

**Fix:** Store the `Arc<Mutex<..>>` itself (or a clone) in `DeviceAllocator`,
`DeviceSyncStream`, and `DeviceDataTransferFull`. Lock it on each use rather
than caching a raw pointer. Alternatively, use `Arc::as_ptr` on the inner
`Box` — but only if the `Box` is never swapped inside the `Mutex`.

### B2: `CanCopy` same-device check uses pointer equality on opaque `OrtMemoryDevice`

**File:** `crates/onnx-runtime-ep-plugin/src/transfer.rs:452`

```rust
let same_device = src_memory_device == dst_memory_device;
```

`OrtMemoryDevice` is opaque. ORT may pass two distinct `OrtMemoryDevice*`
pointers that represent the same GPU (device 0). Pointer equality fails →
`CopyDirection::DeviceToDifferentDevice` → `is_supported() == false` →
`CanCopy` returns false → ORT falls back. D2D copies on a single GPU would
never use this EP.

This is **fail-closed** (not fail-open), so it won't corrupt — but it will
cause functional failure on a real GPU: ORT will fail to find a data transfer
provider for device-to-same-device copies, and the session will error out.

**Fix:** Use `OrtEpApi::MemoryDevice_GetDeviceId` (if it exists) for
same-device comparison, or compare device type + device id fields rather than
pointer equality. If no API exists to compare, document this is a known D2D
gap and file a follow-up.

### B3: `CopyTensors` doesn't distinguish H→D vs D→H vs D→D

**File:** `crates/onnx-runtime-ep-plugin/src/transfer.rs:608-660`

`transfer_full_copy_tensors` wraps both src and dst as
`DeviceBuffer::from_borrowed_parts(ptr, ep.device_id(), ...)` — it treats
**both** as device buffers on the same device. Then it calls `ep.copy()` or
`ep.copy_async()`.

For a host→device copy, the src pointer is a **host** pointer. Wrapping it in
a `DeviceBuffer` with `ep.device_id()` (CUDA:0) and then calling `ep.copy()`
will pass a host pointer to `cudaMemcpyAsync(dst, src, ...,
cudaMemcpyDeviceToDevice)` — which is **UB on CUDA** (src is not a device
pointer). Similarly D→H.

The comments acknowledge this: *"The correct production approach: store
OrtEpApi and call Value_GetMemoryDevice"*. But the code **doesn't do it**,
even though `DeviceDataTransferFull` stores `ep_api` and `api` for exactly
this purpose.

**Severity:** On a real CUDA host, any H→D or D→H copy through this path will
crash or corrupt memory.

**Fix:** Use the stored `ep_api` to call `Value_GetMemoryDevice` /
`MemoryDevice_GetDeviceType` on each OrtValue to classify direction, then
dispatch to `ep.copy_from_host()`, `ep.copy_to_host()`, or `ep.copy()`
accordingly. The `CanCopy` callback already does this classification; apply the
same logic in `CopyTensors`.

---

## SUBSTANTIVE

### S1: `device_free` falls back to `size=0` for unknown pointers

**File:** `crates/onnx-runtime-ep-plugin/src/device.rs:184`

```rust
.unwrap_or(0);
```

If ORT frees a pointer we never tracked (or double-frees), we call
`ep.deallocate()` with `size=0`. For CUDA's `cuMemFree` this might be benign
(the pointer already encodes the allocation), but for a VMM arena allocator
it's a contract violation that could corrupt the free list.

**Fix:** Return early (no-op) if the pointer is not in the map, or log a
diagnostic. Passing a fabricated size is worse than skipping the free.

### S2: `Mutex::lock().unwrap()` across extern "C" boundary

**File:** `crates/onnx-runtime-ep-plugin/src/device.rs:155`

```rust
if let Ok(mut sizes) = alloc.alloc_sizes.lock() {
```

`device_alloc` correctly uses `if let Ok(..)`, but
`factory_create_allocator` (factory.rs:589) uses `.unwrap()` on the shared EP
mutex. If ORT calls `CreateAllocator` after a panic poisoned the mutex, this
panic crosses `extern "C"` — that's UB. The `catch_unwind` guard is present
on the outer `factory_create_allocator` so it's *technically* caught, but
`AssertUnwindSafe` suppresses the unwind safety lint, masking the real issue.

**Fix:** Use `.lock().map_err(|e| ...)` with a fail_status return instead of
`.unwrap()`.

### S3: `factory_create_ep` constructs a fresh EP, ignoring `shared_ep`

**File:** `crates/onnx-runtime-ep-plugin/src/factory.rs:494-516`

`factory_create_ep` always calls `(exported.constructor)()` — it never
checks `exported.shared_ep`. For the CUDA plugin, the constructor is a panic
bomb:

```rust
|| { panic!("CUDA EP constructor called but shared_ep should be used instead"); }
```

If ORT ever calls `CreateEp`, this panics inside `catch_unwind` and returns
an error. Not a crash (the guard works), but:
- ORT asked for an EP and got refused — the session won't run.
- The error message is a panic backtrace, not an actionable error.

**Fix:** `factory_create_ep` should check `shared_ep` and, if set, create an
`ExportedEp` wrapping the shared EP (with `owns_ep = false`).

### S4: CUDA plugin constructor panics in `create_ep_factories_with_device_support`

**File:** `crates/onnx-runtime-ep-cuda-plugin/src/lib.rs:143-148`

`create_ep_factories_with_device_support` delegates to
`create_ep_factories` which calls `constructor()` to get the EP name
(factory.rs:152: `let ep = constructor();`). For the CUDA plugin, this
constructor is the panic bomb. So on the *happy path* (GPU available), the
code:

1. Successfully constructs the EP via `construct_ep_with_stream()` ✓
2. Calls `create_ep_factories_with_device_support` with a panic-bomb
   constructor
3. Which calls `create_ep_factories` → calls `constructor()` → **panics**

This means **even on a CUDA host with a working GPU, CreateEpFactories will
always fail** due to the panic in step 3.

**Severity:** The claimed fix (advertising 1 factory on CUDA hosts) is
**impossible** with this code. The constructor is called unconditionally on
line 152 of factory.rs to read the EP name. It will always panic.

**Fix:** Either:
(a) Pass the EP name explicitly (new parameter to
`create_ep_factories_with_device_support`) so it doesn't need to call the
constructor, or
(b) Change the panic bomb to a real constructor that uses `shared_ep`, or
(c) Add a `create_ep_factories_for_shared_ep` variant that takes name + EP
directly.

---

## NITS

### N1: Comment says "mock" in production data transfer code

**File:** `crates/onnx-runtime-ep-plugin/src/transfer.rs:253-259`

Multiple comments reference "mock test path" and "not yet wired for real
device" in what is now supposed to be production code. Misleading.

### N2: `factory_get_vendor_id` always returns 0

**File:** `crates/onnx-runtime-ep-plugin/src/factory.rs:308`

Comment says "No PCI vendor ID for CPU EP" but this callback is shared by all
EPs including CUDA. Should read `exported.device_support.vendor_id`.

---

## DEFECT VERDICT

| # | Claimed Fix | Genuinely Fixed? | Notes |
|---|---|---|---|
| **1** (shared runtime) | `Arc<Mutex<..>>` shared EP | **NO** | Raw pointer derived from guard; dangling after guard drops (B1). Even if this were fixed, the constructor panic bomb means it never reaches this code (S4). |
| **2** (CreateDataTransfer) | `DeviceDataTransferFull` | **PARTIAL** | Struct exists and `CanCopy` direction logic is correct. But CopyTensors doesn't use direction classification (B3), and ep pointer is dangling (B1). |
| **3** (GetHandle) | `stream_ptr()` stored | **PARTIAL** | Handle is extracted and stored correctly. But the allocator/stream/transfer that use the shared EP have a dangling pointer (B1), so the stream is on the right handle but the EP behind it is garbage. |
| **4** (Free size tracking) | `HashMap<usize,usize>` | **YES** (design correct, impl has S1 nit) | Size tracking logic is sound. Falls back to 0 for unknown pointers (S1) but that's a graceful-degradation issue, not a soundness bug. |

## THE KEY QUESTION

**If this ran on a real CUDA host tomorrow, would it work or crash?**

**It would crash.** Two independent reasons:

1. **S4:** `create_ep_factories_with_device_support` calls the panic-bomb
   constructor to read the EP name. This panics inside `catch_unwind`, returns
   a non-null error status, and `CreateEpFactories` returns 0 factories. The
   plugin never loads. This is the *best* outcome — it fails closed.

2. **B1:** If S4 were somehow bypassed (e.g. constructor fixed), the dangling
   EP pointer from the dropped `MutexGuard` would cause use-after-free on every
   allocator, stream, and data transfer operation. This would manifest as
   crashes, CUDA context corruption, or silent wrong-device ops.

3. **B3:** If B1 were also fixed, CopyTensors would still pass host pointers
   to `cudaMemcpyDeviceToDevice`, corrupting memory.

**Recommendation:** The code is still fail-closed — but **by accident** (the
panic bomb), not by design. The four defects are not genuinely fixed. The code
should remain in draft, and the fail-closed guarantee should be restored by
construction (as it was before this commit range), not by happenstance.

## IS #762 READY TO LEAVE DRAFT?

**No.** The CUDA path has three blocking bugs (B1, B2, B3) and one
showstopper (S4) that prevents it from ever loading. The CPU path is
unaffected (I verified targeted tests pass — 9/9 in `ep-plugin`, 9/9 in
`ep-cuda-plugin` including fail-closed tests).

## WHAT I VERIFIED MYSELF vs TOOK ON TRUST

### Verified:
- `cargo check -p onnx-runtime-ep-cuda-plugin` — ✅ (both with and without `cuda` feature)
- `cargo clippy --workspace --all-targets -- -D warnings` — ❌ pre-existing failures in `onnx-genai-engine` (not from this PR)
- `cargo test -p onnx-runtime-ep-cuda-plugin -p onnx-runtime-ep-plugin` — ✅ 18 passed, 0 failed
- Full `cargo test --workspace` — timed out after >30min; not completed. The targeted tests cover the PR-relevant crates.
- Read all source in `factory.rs`, `device.rs`, `transfer.rs`, `cuda-plugin/src/lib.rs`
- Traced the `MutexGuard` lifetime through each callback (B1)
- Traced the constructor call chain through `create_ep_factories` (S4)
- Confirmed `CopyTensors` wraps both pointers as same-device (B3)
- Confirmed `CanCopy` uses pointer equality for same-device (B2)
- No `as *const i8` portability issues found — all uses go through `c_char`
- `ReleaseEpFactory` returns `*mut OrtStatus` ✓
- CPU path unchanged and unaffected ✓
- `docs/CUDA_EP_STATUS.md` says "unvalidated on hardware" throughout ✓

### Took on trust:
- The 2 pre-existing test failures are genuinely pre-existing (stated in instructions, not independently verified on `675b697bc` due to time)
- `CudaRuntime::stream_ptr()` returns a real `cudaStream_t` (can't verify without GPU)
- `cudarc` dynamic loading works at runtime (can't verify without GPU)
- The ORT plugin EP ABI contract as documented in comments matches the actual header
