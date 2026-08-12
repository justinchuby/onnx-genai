# CUDA EP Status — Compiles, Unvalidated on Hardware

**Author:** Roy (Lead), Sapper (GPU/Systems), Nabil (FFI/Systems, B1/B3/S4 revision),
Deckard (Systems Dev, `CreateEp` ownership-unification revision)
**Updated:** 2026-08-12 (post-#762 `CreateEp` fix revision)
**Branch:** `squad/cuda-plugin-runtime` (draft PR #830, follows merged #762)

---

## Summary — CreateEp now actually works for shared EPs; hardware validation still pending

PR #762 merged the CPU EP and native nxrt ABI, and left the CUDA plugin
deliberately fail-closed. Auditing the merged code turned up one remaining
defect that made the fail-closed CUDA path **permanently** fail-closed even
with real GPU hardware present: `factory_create_ep` unconditionally returned
an error status whenever `ExportedFactory::shared_ep` was set (this was S3's
"fix," see §2 below) — meaning **no ORT session could ever execute a compiled
subgraph through the CUDA plugin**, regardless of hardware availability. That
defect is now fixed (PR #830, commit `62a7d3547`): the shared EP's ownership
model was unified from `Arc<Mutex<Box<dyn ExecutionProvider + Send>>>` to a
lock-free `Arc<dyn ExecutionProvider>`, so `CreateEp` clones the same `Arc`
already used by the allocator/stream/transfer surfaces instead of failing.
See §3 for the updated architecture and §6 for new factory-level conformance
tests that exercise this fix through a real dlopen'd ORT.

**Current behavior:**
- With `cuda` feature ON and a GPU available: `CreateEpFactories` constructs
  a `CudaExecutionProvider`, advertises 1 factory, and — as of this revision —
  `CreateEp` actually succeeds, handing ORT a real, usable `OrtEp*` backed by
  the same CUDA runtime/context as the allocator, stream, and data transfer.
- With `cuda` feature ON and no GPU available: returns 0 factories with an
  actionable `OrtStatus`. This is **fail-closed by design** (not by accidental
  panic), and is exactly what this (no-GPU) development environment exercises.
- With `cuda` feature OFF: returns 0 factories ("not available").
- The CPU path is unchanged and unaffected.

Issue [#768](https://github.com/justinchuby/onnx-genai/issues/768) tracks
hardware validation on a real CUDA GPU. It was closed on the (incorrect)
premise that `scripts/cuda_conformance_runner.sh` did not exist — it was
added in the very commit (#762) the closure referenced. This revision
reopens/replaces that issue; see the PR #830 description for the corrected
record. **No hardware validation has been performed as part of this
revision** — no NVIDIA GPU is present in this development environment
(`nvidia-smi` is not installed).

---

## 1. Implementation Defects — Resolution Status

| # | Defect | Resolution | Status |
|---|--------|------------|--------|
| **1** | **Shared CUDA runtime/context — use-after-free (B1).** Raw pointer derived from `MutexGuard` was dangling after guard dropped. | Components now hold a lock-free `Arc<dyn ExecutionProvider>` clone via `EpRef::Shared` (revised again in the `CreateEp` fix below — see §3). The EP outlives all components by construction; no raw pointers from guards, and no mutex at all. | ✅ **Fixed** (unvalidated on hardware) |
| **2** | **`CopyTensors` doesn't classify direction (B3).** Both src/dst wrapped as device buffers — host pointers would be passed to `cudaMemcpyDeviceToDevice`. | `transfer_full_copy_tensors` now uses `Value_GetMemoryDevice` + `MemoryDevice_GetDeviceType` to classify each tensor, then dispatches to `copy_from_host`, `copy_to_host`, or `copy`. | ✅ **Fixed** (unvalidated on hardware; CPU-host conformance in §6) |
| **3** | **Panic bomb makes success unreachable (S4).** `create_ep_factories` called constructor to read EP name; CUDA's constructor was a panic bomb. | New `create_ep_factories_for_shared_ep` takes `ep_name` directly — no constructor call. The CUDA plugin passes the name from the pre-constructed EP. | ✅ **Fixed** (unvalidated on hardware) |
| **4** | **`Free` passes `size=0` violating the allocator contract.** | `DeviceAllocator` tracks sizes in `HashMap<usize,usize>`. Unknown pointers are no-op'd (S1 fix: skip rather than fabricate `size=0`). | ✅ **Fixed** (unvalidated on hardware; size-zero conformance in §6) |
| **5** | **`CreateEp` unconditionally fails for shared EPs** — found auditing the merged #762 code, not part of Gaff's original review. `factory_create_ep`'s shared-EP branch always returned a fail status (this was S3's "fix" below), so **no ORT session could ever execute a compiled subgraph** through a shared-EP factory (CUDA), even with real GPU hardware present. | `factory_create_ep` now clones the shared `Arc<dyn ExecutionProvider>` and hands it to a real `ExportedEp`, exactly like the allocator/stream/transfer surfaces. See §3 for the ownership model and §6 for conformance tests proving `CreateEp` now succeeds and shares the same runtime instance. | ✅ **Fixed** (PR #830, commit `62a7d3547`; unvalidated on hardware) |

**All five are resolved in code. None have been validated on GPU hardware.**

---

## 2. Gaff's Substantive Items — Disposition

| # | Issue | Disposition |
|---|-------|-------------|
| **S1** | `device_free` falls back to `size=0` for unknown pointers | ✅ **Fixed.** Unknown pointers are now no-op'd (early return). |
| **S2** | `Mutex::lock().unwrap()` across extern "C" | ✅ **Superseded.** The shared-EP path no longer uses a `Mutex` at all (see §3) — `EpRef::Shared` holds a lock-free `Arc<dyn ExecutionProvider>`, so there is nothing to lock or poison on this path. |
| **S3** | `factory_create_ep` ignores `shared_ep` | ⚠️ **Originally "fixed" by returning an actionable error status — this was itself defect #5 above.** Failing closed on every `CreateEp` call is not a fix for a plugin EP that must eventually run inference; it just fails safely instead of unsafely. `CreateEp` now actually succeeds for shared EPs (see §3, §6). |
| **S4** | CUDA constructor panics in factory creation | ✅ **Fixed.** See defect #3 above. |
| **B2** | `CanCopy` same-device uses pointer equality | ✅ **Fixed.** Now uses `MemoryDevice_GetDeviceId` (present in ORT 1.27 bindings, `OrtEpApi` offset 96) to compare device IDs when pointer equality fails. Same-device D2D copies are accepted; cross-device (peer-to-peer) copies fail closed with an actionable error status. If `GetDeviceId` is `None` at runtime, fails closed (cross-device). Compiles and type-checks; **unvalidated on hardware** — blocked on #768. CPU-host conformance for the CPU->GPU/GPU->CPU/GPU->GPU-same-device classification matrix is in §6. |
| **N1** | Comments say "mock" in production code | ✅ **Fixed.** Comments updated. |
| **N2** | `factory_get_vendor_id` always returns 0 | ✅ **Fixed.** Now reads `exported.device_support.vendor_id`. |

---

## 3. Ownership Architecture (B1 fix, revised again for the `CreateEp` fix)

The shared-EP ownership model was unified once more in the `CreateEp` fix: the
original B1 fix wrapped the EP in `Arc<Mutex<Box<dyn ExecutionProvider +
Send>>>`, which is what made `factory_create_ep` unable to hand ownership to a
new `ExportedEp` (there was no way to move the EP out of the `Mutex` while
other components still held clones). The fix replaces this with a **lock-free**
`Arc<dyn ExecutionProvider>`, shared verbatim by every surface — including,
now, `CreateEp`:

```
ExportedFactory {
    shared_ep: Option<Arc<dyn ExecutionProvider>>,
    ...
}

// Every surface clones the SAME Arc — no locking, no MutexGuard lifetime:
DeviceAllocator        { ep_ref: EpRef::Shared(Arc::clone(shared)), ... }
DeviceSyncStream       { ep_ref: EpRef::Shared(Arc::clone(shared)), ... }
DeviceDataTransferFull { ep_ref: EpRef::Shared(Arc::clone(shared)), ... }
ExportedEp             { ep: Arc::clone(shared), ... }   // <- CreateEp fix
```

**Why this is sound without a mutex:** `ExecutionProvider: Send + Sync`, and
every method used post-construction (`allocate`, `deallocate`, `copy`,
`copy_async`, `sync`, `get_kernel`, `supports_op`, …) takes `&self`. Only
`initialize`/`shutdown` need `&mut self`; `initialize` is called once, before
the EP is wrapped in the `Arc` (the caller's responsibility — see
`create_ep_factories_for_shared_ep`'s doc comment), and `shutdown` is called
from `factory_release_ep` only when `Arc::get_mut` succeeds — i.e. only when
that `ExportedEp` is the *sole* remaining owner. As long as the owning
factory is alive, it always holds one more `Arc` clone (`ExportedFactory
::shared_ep` is never taken/cleared), so `Arc::get_mut` on an `ExportedEp`'s
clone can never succeed while the factory lives — `shutdown()` is never
called on a runtime other surfaces (or the factory itself) still depend on.
Real CUDA resource teardown is not gated on `shutdown()` at all: it happens
via `Drop` impls (e.g. on `CudaRuntime`), which run exactly once when the
last `Arc` strong reference is dropped, regardless of whether `shutdown()`
was ever explicitly invoked.

**Why Arc, not a raw pointer or Mutex:** A raw pointer extracted from a
`MutexGuard` is only valid while the guard is held — the original B1 defect.
A `Mutex<Box<dyn ExecutionProvider>>` prevents ever moving the `Box` out
(needed by `CreateEp`) without either cloning the value (not always possible
for a trait object) or `Option::take`-ing it (which would strand every other
holder with a dangling `EpRef`). `Arc<dyn ExecutionProvider>` sidesteps both
problems: it is `Clone`, requires no lock for `&self`-only usage, and every
holder keeps the allocation alive independently.

`crates/onnx-runtime-ep-plugin/tests/shared_gpu_conformance.rs` proves this
end-to-end (via `Arc::ptr_eq`, not just matching EP names) against a real
dlopen'd ORT — see §6.

---

## 4. What Remains Unknowable Without a GPU

- Whether `cudarc` dynamic loading actually finds `libcuda.so` and `libcudart.so`
- Whether `CudaRuntime::stream_ptr()` returns a valid `cudaStream_t`
- Whether ORT's `Value_GetMemoryDevice` returns the correct device type for
  tensors allocated by our allocator
- Whether the EP's `copy_from_host`/`copy_to_host`/`copy` implementations
  produce correct results on real CUDA memory
- Whether `CreateEp` on the real `CudaExecutionProvider` actually compiles and
  runs a subgraph correctly end-to-end (the new conformance tests in §6 prove
  the ABI plumbing works with a mock EP; they cannot exercise real CUDA
  kernels, a real `OrtGraph`/`Compile` cycle, or real device memory)
- Whether the lock-free `Arc<dyn ExecutionProvider>` sharing model holds up
  under real concurrent ORT sessions (no contention/stress profiling possible
  without hardware)

---

## 5. `prefetch_lazy_weight` — Stub Decision Record

**Location:** `crates/onnx-runtime-ep-cuda/src/provider.rs:564–573`

Returns `Ok(false)` — "no transfer enqueued." Deferred to post-Phase-2a.

---

## 6. Factory-Level Conformance Tests (new, `onnx-runtime-ep-plugin/tests/shared_gpu_conformance.rs`)

Prior to this revision, `factory.rs` had **zero** direct tests of its vtable
callbacks — every existing test exercised `device.rs`/`transfer.rs` structs
directly, bypassing the actual `extern "C"` function pointers ORT calls. Two
new tests close that gap by dlopen'ing a real upstream ONNX Runtime and
driving the genuine `CreateAllocator`, `CreateSyncStreamForDevice`,
`CreateDataTransfer`, and `CreateEp` vtable entries against a
`MockCudaLikeEp` — an `ExecutionProvider` tagged `DeviceType::Cuda`
(`stream_aware: true`, `host_accessible: false`) but backed entirely by
ordinary host heap memory.

**This is CPU-host/mock conformance, not GPU hardware validation** — it
proves ABI wiring and single-shared-instance ownership, nothing about real
CUDA correctness, performance, or device-memory behavior:

- `create_ep_succeeds_for_shared_ep_and_shares_the_runtime_instance` — the
  narrowest reproduction of defect #5: confirms `CreateEp` now succeeds for a
  shared EP (previously always failed), that `GetName` reports the correct
  name, and — via `Arc::ptr_eq` against the (publicly exposed, for exactly
  this purpose) `ExportedEp::ep` field — that `CreateEp` hands back the
  *exact same* `Arc<dyn ExecutionProvider>` allocation used elsewhere, not
  merely an equivalently-named instance. Also confirms `ReleaseEp` correctly
  skips `shutdown()` while other `Arc` clones are alive, and that releasing
  the factory drops the last reference exactly once.
- `shared_ep_surfaces_all_dispatch_to_one_runtime_instance` — end-to-end
  walkthrough: allocator `Alloc`/`Free` round-trip including size-zero
  `Alloc(0)` (non-null pointer, no panic on `Free`); a caller-owned opaque
  stream handle round-tripping unchanged through `CreateSyncStreamForDevice`
  → `GetHandle` (non-null, correctly-owned — the adapter never frees it);
  `CanCopy` classifying H2D/D2H/H2H/GPU-same-device directions via real
  `OrtMemoryDevice` objects; `CopyTensors` performing genuine H2D and D2H
  byte copies through real `OrtValue` objects built via
  `CreateTensorWithDataAsOrtValue`; and `CreateEp` sharing the same `Arc`
  instance as every other surface (again via `Arc::ptr_eq`). A shared,
  per-instance-tagged call log cross-checks that every callback actually
  dispatched to the same concrete mock instance.

Run with `cargo test -p onnx-runtime-ep-plugin --test shared_gpu_conformance`.
Set `NXRT_REQUIRE_ORT_TESTS=1` to turn a "ORT not found" skip into a hard
failure (useful in CI once a prebuilt ORT is guaranteed available).

---

## 7. Hardware Conformance Runner

`scripts/cuda_conformance_runner.sh` exits 2 (UNVALIDATED) on this host.
Every capability remains unvalidated. No self-hosted GPU workflow exists.
