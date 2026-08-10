# EP Plugin Export — Security Audit

**Auditor:** Holden (Security Engineer)
**Date:** 2026-08-10T20:12:35.793+00:00
**Scope:** `crates/onnx-runtime-ep-plugin/src/{factory,ep,graph_reader,compute,kernel_ctx,status,lib}.rs`
**Status:** `compute.rs` and `kernel_ctx.rs` are in-flux (Nabil actively editing); findings on those are preliminary and must be re-checked at merge.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 1 |
| HIGH     | 3 |
| MEDIUM   | 2 |
| LOW      | 1 |

---

## CRITICAL

### C1. No `catch_unwind` on any `extern "C"` callback — unwinding across FFI is instant UB

**Files:** All 9 exported callbacks: `factory.rs:119` (`factory_get_name`), `factory.rs:127` (`factory_get_supported_devices`), `factory.rs:147` (`factory_create_ep`), `factory.rs:175` (`factory_release_ep`), `ep.rs:53` (`ep_get_capability`), `ep.rs:128` (`ep_compile`), `ep.rs:199` (`ep_release_node_compute_infos`), `compute.rs:58` (`compute_create_state`), `compute.rs:82` (`compute_execute`), `compute.rs:106` (`compute_release_state`).

**Scenario:** Any Rust panic (failed allocation, `Vec` index OOB in `claims[i].node_ids`, `unwrap()` on `CString::new`, `GraphViewCache::build` panicking, etc.) will unwind through the `extern "C"` boundary into ORT's C/C++ runtime. Per Rust reference this is **undefined behavior** — in practice it corrupts ORT's stack and crashes the host process silently.

**Specific panic sources observed:**
- `factory.rs:97`: `constructor()` — user-supplied closure may panic.
- `ep.rs:170-176`: `exported.ep.get_kernel(...)` — trait method on user-provided EP may panic.
- `ep.rs:73`: `reader.to_ir_graph()` and `GraphViewCache::build()` — complex code, can panic on OOM or internal assertion.
- `compute.rs:82-95`: `exported.kernels` index/iteration if any `Kernel::execute` panics (future Phase 2).

**Fix:** Wrap every `extern "C" fn` body in `std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| { ... }))`. On `Err(_)`, return `fail_status("internal panic")` (or for `-> ()` callbacks, just swallow). This is non-negotiable before shipping.

---

## HIGH

### H1. `static mut HOST_ORT_API` is a data race — unsound under concurrent `CreateEpFactories` calls

**File:** `status.rs:11-28`

**Scenario:** If ORT (or a host with multiple providers) calls `CreateEpFactories` for two plugin libs from different threads — or calls any callback while `set_host_api` is running — the `static mut` read/write is a data race (UB even if the value written is the same both times). Rust 2024 edition's stricter `unsafe` lint will flag this soon.

**Fix:** Replace with `static HOST_ORT_API: AtomicPtr<ort::OrtApi> = AtomicPtr::new(ptr::null_mut())` and use `store(Relaxed)` / `load(Acquire)`.

### H2. `graphs` pointer not null-checked before indexing in `ep_compile`

**File:** `ep.rs:143`

**Scenario:** If ORT passes `graphs == null` with `count > 0`, `*graphs.add(i)` dereferences a null pointer → segfault in ORT's process.

**Fix:** Add `if graphs.is_null() { return fail_status(...); }` alongside the existing `out_infos` null check.

### H3. `unsafe impl Send + Sync` for `OutboundGraphReader` is unsound if instance escapes callback scope

**File:** `graph_reader.rs:32-33`

**Scenario:** `OutboundGraphReader` stores raw `*const OrtNode` pointers whose validity is scoped to the current ORT callback invocation. The `Send + Sync` impls allow it to be moved to another thread or stored in a `Mutex` that outlives the callback. If any future code (or a user EP impl) captures the reader in a closure, spawn, or `Arc`, those pointers become dangling.

**Current risk:** Low today because usage is stack-local in `ep_get_capability` / `ep_compile`. But the blanket impls are a landmine.

**Fix:** Remove `unsafe impl Send for OutboundGraphReader` and `unsafe impl Sync for OutboundGraphReader`. The struct is only used within a single callback frame; it never needs to be `Send`/`Sync`. If the compiler later requires it for some composition, audit that specific path rather than granting blanket impls.

---

## MEDIUM

### M1. `check()` in `graph_reader.rs:465` releases the OrtStatus but discards the message

**File:** `graph_reader.rs:459-470`

**Scenario:** When an ORT API call fails, `check()` releases the status immediately and returns a generic `"ORT API call failed"` string. This loses the actual error message from ORT, making debugging impossible. More critically, the function calls `ReleaseStatus` — if ORT's `ReleaseStatus` expects the caller to have NOT already read the message (unlikely but possible ABI-dependent), this could misbehave.

**Fix:** Before `ReleaseStatus`, call `GetErrorMessage(status)` to extract the real message and include it in the `Err(...)`. This is a correctness/debuggability issue, not a memory-safety issue, so MEDIUM.

### M2. `ep_compile` does not clean up already-allocated `ExportedComputeInfo` on mid-loop error

**File:** `ep.rs:128-197`

**Scenario:** If `get_kernel` fails at subgraph index `i=3` (of 5), the function returns an error status immediately. Subgraphs 0..2 already have `ExportedComputeInfo` written into `out_infos`. ORT's contract for error returns from `Compile` is unclear — if ORT does NOT call `ReleaseNodeComputeInfos` on error, those allocations leak. If ORT DOES call it, this is fine.

**Fix:** Either (a) document that ORT always calls `ReleaseNodeComputeInfos` even on `Compile` error and cite the header, or (b) on error, iterate back over `out_infos[0..i]` and free any non-null entries before returning.

---

## LOW

### L1. `node_id_to_ort_index` returns 0 on miss — could silently report wrong node

**File:** `graph_reader.rs:186`

**Scenario:** If `node_id_to_ort_index` is called with a `NodeId` that doesn't exist in `ort_index_to_node_id` (bug elsewhere), it returns index 0 — silently reporting the first ORT node as a claim. This is a logic bug rather than memory-safety.

**Fix:** Return `Option<usize>` and propagate the error, or at least `debug_assert!` the lookup succeeds.

---

## Areas Verified Sound

- **Pointer ownership model** (factory.rs, ep.rs): `Box::into_raw` / `Box::from_raw` pairs are correctly matched between `create_ep_factories` ↔ `release_ep_factory`, `factory_create_ep` ↔ `factory_release_ep`, `ep_compile` ↔ `ep_release_node_compute_infos`, `compute_create_state` ↔ `compute_release_state`. No double-free or type mismatch.
- **`#[repr(C)]` vtable layout**: `ExportedFactory`, `ExportedEp`, `ExportedComputeInfo` all place the vtable as their first field. The cast from `*mut ExportedFoo` to `*mut OrtFoo` is sound because `#[repr(C)]` guarantees the first field is at offset 0.
- **String handling** (`factory_get_name`): Returns `exported.name_cstr.as_ptr()` which points into the still-live `ExportedFactory`. The pointer is valid for as long as the factory exists (until `ReleaseEpFactory`). No use-after-free.
- **Buffer overflow on `out_factories`**: `create_ep_factories` writes exactly 1 factory and checks `max_factories == 0` beforehand. Sound.
- **`kernel_ctx.rs`**: Empty (placeholder for Phase 2). No findings.
- **`compute.rs`**: Returns `NOT_IMPLEMENTED` — fail-closed. The only live paths are `CreateState`/`ReleaseState` which are simple Box round-trips. Sound for v1.

---

## In-Flux Files (re-check required)

`compute.rs` and `kernel_ctx.rs` are actively being edited by Nabil for Phase 2. When the `Compute` path becomes live, the following must be audited:

1. **Tensor size overflow**: `dims.iter().product::<usize>()` can overflow for attacker-controlled shapes.
2. **dtype validation**: Constructing a typed `TensorView<f32>` over a buffer that's actually `int8` is instant UB.
3. **Bounds on `KernelContext_GetOutput` shape**: The shape passed to allocate the output tensor must be validated; an EP that requests a 16-exabyte output from ORT would be a DoS.
4. **Panic safety in `Kernel::execute`**: Same C1 issue — `catch_unwind` is mandatory.

---

## Verdict

**CRITICAL C1 (panic safety) must be fixed before this code can ship.** A single panic in any code path reachable from these callbacks — including OOM, assertion failures deep in `onnx_runtime_ir`, or user-EP trait panics — is undefined behavior that corrupts ORT's process.

**HIGH H1–H3 should be fixed in the same PR.** H1 is a latent data race; H2 is a null-deref crash; H3 is a soundness hole that will bite when the code evolves.

MEDIUM/LOW items are hardening and can land in a follow-up.
