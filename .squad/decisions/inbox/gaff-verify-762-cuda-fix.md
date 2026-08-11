# Gaff Verification — PR #762 CUDA EP Fix (Commit d64a49d59)

**Reviewer:** Gaff (adversarial)
**Date:** 2026-08-11
**Head:** `d64a49d59`
**Previous review:** `.squad/decisions/inbox/gaff-rereview-762-cuda.md`

---

## Per-Blocker Verdicts

### B1: Use-after-free — **GENUINELY FIXED** ✅

`EpRef::Shared(Arc<Mutex<..>>)` in `device.rs:78-88`. Components hold `Arc`
clones; each callback locks via `with_ep` (`device.rs:95-108`), never extracts
a raw pointer from a guard. The `Arc::clone` at construction
(`device.rs:178`, `device.rs:312`, `transfer.rs:392`) means the heap
allocation outlives any one component. No remaining reference escapes a
`MutexGuard`.

Test `shared_ep_allocator_outlives_original_arc` is **non-vacuous**: it drops
the original `Arc`, then calls `Alloc` through the ORT vtable on the
allocator's `Arc` clone. If the old dangling-pointer bug existed, this would
segfault.

### B3: Copy direction — **GENUINELY FIXED** ✅

`transfer.rs:555-575`: `value_get_mem_device` + `get_dev_type` classify each
tensor. Dispatch at lines 664/684/700 routes to `copy_from_host` /
`copy_to_host` / `copy`. Unknown device types default to CPU
(`transfer.rs:558-559`) — fail-closed (falls into H→D or D→H rather than
wrong-kind cudaMemcpy).

Unsupported directions (D→D-different, H→H) return `fail_status` at line
578 — fail-closed. `CopyDirection::classify` logic at lines 80-98 is
exhaustive.

Test `copy_direction_h2d_and_d2h_both_supported` tests the `classify` function
and `is_supported` — it verifies the **classification logic** but NOT the
actual `CopyTensors` dispatch path. However, the classification is the part
that was wrong; the dispatch is mechanical, so this is acceptable.

### S4: Panic bomb — **GENUINELY FIXED** ✅

`factory.rs:311-356`: `create_ep_factories_for_shared_ep` takes `ep_name: &str`
directly. The CUDA plugin at `lib.rs:146-159` extracts the name from the
pre-constructed EP before wrapping in Arc. The old constructor closure at
`factory.rs:344` is unreachable in the shared-EP path and panics with an
actionable message if somehow called — fail-closed.

Test `fail_closed_by_status_not_panic` verifies no panic escapes FFI. It's
**marginal** — it would pass even with a different bug (e.g. segfault would
abort, not panic). But combined with
`cuda_plugin_returns_zero_factories_without_gpu` (which asserts `num == 0`),
the success path is genuinely reachable on a CUDA host.

---

## B2 Deferral: **NOT LEGITIMATE** ⚠️

Nabil claims `MemoryDevice_GetDeviceId` "may not exist in ORT 1.27." This is
**factually wrong**. I verified:

```
target/debug/build/onnx-genai-ort-sys-3b504ed789bb5e57/out/bindings.rs:6309:
    pub MemoryDevice_GetDeviceId:
        Option<unsafe extern "C" fn(memory_device: *const OrtMemoryDevice) -> u32>,
```

It's at offset 96 in `OrtEpApi`, present in the generated bindings from the
ORT headers this project builds against. The `Option<fn>` wrapper means it
could be `None` at runtime (if ORT didn't populate it), but the code already
handles `None` for `MemoryDevice_GetDeviceType` with a fail-closed path.

**Impact:** Pointer equality at `transfer.rs:463` will fail for same-GPU D2D
copies if ORT passes two distinct `OrtMemoryDevice*` for device 0. ORT falls
back (fail-closed) — no corruption, but functional failure for D2D on single
GPU.

**Verdict:** The deferral is defensible from a *safety* standpoint (no UB), but
the justification is dishonest. The API exists; the fix is trivial (check
`MemoryDevice_GetDeviceId`, if `Some`, compare device IDs). This should be a
**condition of leaving draft**, not a post-merge follow-up.

---

## Deadlock / Re-entrancy

`with_ep` holds the `Mutex` for the duration of each EP method call. For
`sync()` this means `cudaStreamSynchronize` blocks under the lock
(`provider.rs:1500-1502`). Other ORT threads calling `Alloc`, `Free`, or
`CopyTensors` will block until sync completes.

**Correctness:** No deadlock — there's no circular lock dependency. ORT does
not re-enter the same EP's allocator from inside `sync()`.

**Performance:** On a real GPU, `cudaStreamSynchronize` can block for
milliseconds. All allocations and copies are serialized behind it. This is a
throughput problem for multi-stream workloads. Noted but not blocking — a
`RwLock` or per-operation lock granularity is a follow-up.

---

## Fail-Closed Integrity

Both feature configs verified:
- `#[cfg(feature = "cuda")]`: constructs EP, if fails → `*out_num = 0` + status.
- `#[cfg(not(feature = "cuda"))]`: immediate `*out_num = 0` + status.
- Every `extern "C"` wrapped in `catch_unwind` → `fail_status` on panic.
- `with_ep` returns `Err` on poisoned mutex → status, not panic.
- Unknown pointers in `device_free` → no-op (lines 244-250).

**By design, not by accident.** ✅

---

## Test Quality Assessment

| Test | Non-vacuous? | Notes |
|------|-------------|-------|
| `shared_ep_allocator_outlives_original_arc` | **YES** | Actually exercises vtable call after Arc drop. Would catch B1. |
| `fail_closed_by_status_not_panic` | **Marginal** | Catches panic escape but not other failure modes. Paired with other tests it's adequate. |
| `copy_direction_h2d_and_d2h_both_supported` | **YES** (logic only) | Tests classification matrix exhaustively. Does not test the FFI dispatch path. |

---

## CPU Regression

No changes to CPU path. `onnx-runtime-ep-plugin` is shared but all new code is
behind `EpRef::Shared` which is only constructed by the CUDA plugin's
`CreateEpFactories`. The trait_cabi_parity tests (9 tests) pass, confirming the
shared crate is not broken.

---

## `docs/CUDA_EP_STATUS.md` Honesty

The doc now states:
- "Compiles, Unvalidated on Hardware" in title
- "None have been validated on GPU hardware" (§1 table)
- Lists what remains unknowable (§4)
- References #768 for hardware validation
- B2 marked as "Deferred" with "may not exist" — **this is the one dishonest claim** (see above)

Otherwise accurate.

---

## `as *const i8` / `c_char` / aarch64

No hardcoded `as *const i8`. All FFI string boundaries use `c_char` via
`std::ffi::CString`. Correct on both x86_64 (`c_char = i8`) and aarch64
(`c_char = u8`).

---

## If This Ran on a Real CUDA Host Tomorrow

**Would it work?** It would construct the EP and advertise a factory. ORT would
call `CreateAllocator`, `CreateDataTransfer`, `CreateSyncStream` successfully.
Allocations and H↔D copies would dispatch to the correct CUDA memcpy kind.

**Would it fail?** D2D copies on the same GPU would be rejected (B2 pointer
equality) — ORT falls back. Depending on ORT's fallback path, this is either
transparent degradation or a session-load error.

**Would it crash/corrupt?** No. The three UB paths (B1, B3, S4) are eliminated.
Remaining failure modes return `OrtStatus` errors.

**Summary:** Fail-closed on D2D same-device; otherwise functional with the
caveat that no one has verified the actual CUDA operations produce correct
results.

---

## NEW Blockers

**B2 should not be deferred** — the API exists, the fix is 5 lines. Without it,
any model that moves intermediate tensors between subgraphs on the same GPU
will fail. This is a common pattern (attention → FFN on device 0).

However, B2 fails *closed* (returns error to ORT, no UB), so it is not a
*safety* blocker. It is a *functionality* blocker.

---

## Verdict: Ready to Leave Draft?

**Conditional YES** — with the condition that B2 is fixed (using
`MemoryDevice_GetDeviceId`) before the PR leaves draft. This is ~5 lines of
code; the API is available. The justification for deferral was factually
incorrect.

If the team accepts "D2D same-device is broken until a follow-up," then YES
unconditionally — the code is sound, fail-closed, and won't corrupt.

---

## What I Verified vs Took on Trust

| Verified myself | Took on trust |
|----------------|---------------|
| B1 fix architecture: read every `EpRef` construction and `with_ep` call | That `ep.allocate()` / `ep.copy()` implementations in `onnx-runtime-ep-cuda` are correct (not exercised without GPU) |
| B3 direction classification logic and dispatch path | That ORT's `Value_GetMemoryDevice` returns correct device types for EP-allocated tensors |
| S4 factory creation path end-to-end | That `cudarc` runtime init works on real hardware |
| `MemoryDevice_GetDeviceId` existence in ORT 1.27 bindings | ORT actually populates the function pointer at runtime (could be `None`) |
| All 16 tests pass (7 cuda-plugin + 9 ep-plugin) | Baseline 4580/20/436 numbers (did not re-run full workspace) |
| Clippy clean on both EP crates | |
| No `as *const i8` | |
| Panic containment at every `extern "C"` boundary | |
| CUDA_EP_STATUS.md accuracy (except B2 claim) | |
