# CUDA EP Status — Compiles, Unvalidated on Hardware

**Author:** Roy (Lead), Sapper (GPU/Systems), Nabil (FFI/Systems, B1/B3/S4 revision)
**Updated:** 2026-08-11 (post-Gaff re-review revision)
**Branch:** `squad/ep-plugin-parity-cuda` (draft PR #762)

---

## Summary — Three blockers fixed; hardware validation pending

Gaff's adversarial re-review (commit range `2ca515eb7..7aba5cb93`) identified
three blocking defects (B1: use-after-free, B3: wrong copy direction, S4: panic
bomb preventing factory creation) that invalidated the earlier "four defects
resolved" claim. **Those three are now fixed.** Defect #4 (Free size tracking)
was genuinely fixed previously.

**Current behavior:**
- With `cuda` feature ON: `CreateEpFactories` attempts to construct a
  `CudaExecutionProvider`. If a GPU is available, it advertises 1 factory. If
  no GPU is available, it returns 0 factories with an actionable `OrtStatus`.
  This is **fail-closed by design** (not by accidental panic).
- With `cuda` feature OFF: returns 0 factories ("not available").
- The CPU path is unchanged and unaffected.

Issue [#768](https://github.com/justinchuby/onnx-genai/issues/768) tracks
hardware validation on a real CUDA GPU.

---

## 1. The Four Implementation Defects — Resolution Status

| # | Defect | Resolution | Status |
|---|--------|------------|--------|
| **1** | **Shared CUDA runtime/context — use-after-free (B1).** Raw pointer derived from `MutexGuard` was dangling after guard dropped. | Components now hold `Arc<Mutex<..>>` clones via `EpRef::Shared`. Each use locks the mutex; the EP outlives all components by construction. No raw pointers from guards. | ✅ **Fixed** (unvalidated on hardware) |
| **2** | **`CopyTensors` doesn't classify direction (B3).** Both src/dst wrapped as device buffers — host pointers would be passed to `cudaMemcpyDeviceToDevice`. | `transfer_full_copy_tensors` now uses `Value_GetMemoryDevice` + `MemoryDevice_GetDeviceType` to classify each tensor, then dispatches to `copy_from_host`, `copy_to_host`, or `copy`. | ✅ **Fixed** (unvalidated on hardware) |
| **3** | **Panic bomb makes success unreachable (S4).** `create_ep_factories` called constructor to read EP name; CUDA's constructor was a panic bomb. | New `create_ep_factories_for_shared_ep` takes `ep_name` directly — no constructor call. The CUDA plugin passes the name from the pre-constructed EP. | ✅ **Fixed** (unvalidated on hardware) |
| **4** | **`Free` passes `size=0` violating the allocator contract.** | `DeviceAllocator` tracks sizes in `HashMap<usize,usize>`. Unknown pointers are no-op'd (S1 fix: skip rather than fabricate `size=0`). | ✅ **Fixed** (unvalidated on hardware) |

**All four are resolved in code. None have been validated on GPU hardware.**

---

## 2. Gaff's Substantive Items — Disposition

| # | Issue | Disposition |
|---|-------|-------------|
| **S1** | `device_free` falls back to `size=0` for unknown pointers | ✅ **Fixed.** Unknown pointers are now no-op'd (early return). |
| **S2** | `Mutex::lock().unwrap()` across extern "C" | ✅ **Fixed.** `EpRef::with_ep` uses `.lock().map_err()` — no unwrap. Poisoned mutex returns an actionable error. |
| **S3** | `factory_create_ep` ignores `shared_ep` | ✅ **Fixed.** Returns an actionable error status explaining the shared EP is used by components directly. |
| **S4** | CUDA constructor panics in factory creation | ✅ **Fixed.** See defect #3 above. |
| **B2** | `CanCopy` same-device uses pointer equality | ✅ **Fixed.** Now uses `MemoryDevice_GetDeviceId` (present in ORT 1.27 bindings, `OrtEpApi` offset 96) to compare device IDs when pointer equality fails. Same-device D2D copies are accepted; cross-device (peer-to-peer) copies fail closed with an actionable error status. If `GetDeviceId` is `None` at runtime, fails closed (cross-device). Compiles and type-checks; **unvalidated on hardware** — blocked on #768. |
| **N1** | Comments say "mock" in production code | ✅ **Fixed.** Comments updated. |
| **N2** | `factory_get_vendor_id` always returns 0 | ✅ **Fixed.** Now reads `exported.device_support.vendor_id`. |

---

## 3. Ownership Architecture (B1 fix)

```
ExportedFactory {
    shared_ep: Option<Arc<Mutex<Box<dyn ExecutionProvider + Send>>>>,
    ...
}

// Each component stores:
DeviceAllocator       { ep_ref: EpRef::Shared(Arc::clone(&shared_ep)), ... }
DeviceSyncStream      { ep_ref: EpRef::Shared(Arc::clone(&shared_ep)), ... }
DeviceDataTransferFull { ep_ref: EpRef::Shared(Arc::clone(&shared_ep)), ... }
```

**Why no Mutex deadlock:** Each callback locks the mutex only for the duration
of the EP method call (allocate, sync, copy). ORT's threading model ensures
these are not re-entrant on the same factory. The mutex is not held across
blocking CUDA calls — `ep.sync()` / `ep.copy()` execute with the lock held
briefly (the actual CUDA synchronization happens inside the EP implementation
which does not re-lock).

**Why Arc, not raw pointer:** A raw pointer extracted from a `MutexGuard` is
only valid while the guard is held. Once the guard drops, another thread could
lock and move the contents. By storing an `Arc` clone, each component holds
a strong reference that keeps the heap allocation alive independently.

---

## 4. What Remains Unknowable Without a GPU

- Whether `cudarc` dynamic loading actually finds `libcuda.so` and `libcudart.so`
- Whether `CudaRuntime::stream_ptr()` returns a valid `cudaStream_t`
- Whether ORT's `Value_GetMemoryDevice` returns the correct device type for
  tensors allocated by our allocator
- Whether the EP's `copy_from_host`/`copy_to_host`/`copy` implementations
  produce correct results on real CUDA memory
- Whether the Mutex lock duration is acceptable for performance (no contention
  profiling possible without hardware)

---

## 5. `prefetch_lazy_weight` — Stub Decision Record

**Location:** `crates/onnx-runtime-ep-cuda/src/provider.rs:564–573`

Returns `Ok(false)` — "no transfer enqueued." Deferred to post-Phase-2a.

---

## 6. Hardware Conformance Runner

`scripts/cuda_conformance_runner.sh` exits 2 (UNVALIDATED) on this host.
Every capability remains unvalidated. No self-hosted GPU workflow exists.
