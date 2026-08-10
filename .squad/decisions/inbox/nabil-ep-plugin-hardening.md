# EP Plugin Export Hardening — Nabil decision record

**Date:** 2026-08-10T21:15:32Z
**Author:** Nabil
**Branch:** squad/ep-plugin-export
**Scope:** `crates/onnx-runtime-ep-plugin/src/{ep,factory,graph_reader,status}.rs`

---

## Compile error (E0560) — root cause and fix

**Error:** `struct OrtEp has no field named ValidateCompiledModelCompatibilityInfo`

**Root cause:** A previous session built the `OrtEp` vtable by listing
`ValidateCompiledModelCompatibilityInfo: None` — but that field belongs to
`OrtEpFactory`, not `OrtEp`. The real `OrtEp` has `GetCompiledModelCompatibilityInfo`
(a read-only getter) and no `Validate*` counterpart.

Additionally, the `OrtEp` struct in ORT 1.27.0 bindings contains 11 more optional
fields beyond those originally listed (`IsConcurrentRunSupported`, `Sync`,
`CreateProfiler`, `IsGraphCaptureEnabled`, `IsGraphCaptured`, `ReplayGraph`,
`GetGraphCaptureNodeAssignmentPolicy`, `GetAvailableResource`,
`OnSessionInitializationEnd`, `GetDefaultMemoryDevice`, `ReleaseCapturedGraph`).

**Fix (ep.rs):** The file on disk had already been updated to use
`..Default::default()` for the remaining fields, so the compile error was gone
by the time this session ran. I verified with `cargo check` and confirmed the
`OrtEp` struct fields against the generated bindings at
`target/debug/build/onnx-genai-ort-sys-*/out/bindings.rs`.

**ABI note:** `ValidateCompiledModelCompatibilityInfo` IS a real field on
`OrtEpFactory` (verified in bindings). It was simply put on the wrong struct
in the previous session.

---

## Security finding resolutions

### C1 — No `catch_unwind` on extern "C" callbacks (CRITICAL)

**Status: ALREADY FIXED in current codebase.**

Every `extern "C"` callback in `factory.rs` and `ep.rs` already wraps its
body in `std::panic::catch_unwind(AssertUnwindSafe(||{ ... }))`.
The `factory_release_ep` and `ep_release_node_compute_infos` (void-returning)
use `let _ = std::panic::catch_unwind(...)` to swallow panics safely.

Unit tests verifying the pattern are in:
- `status::tests::catch_unwind_prevents_panic_from_escaping`
- `ep::tests::catch_unwind_in_callback_wrapper_works`

### H1 — `static mut HOST_ORT_API` data race (HIGH)

**Status: ALREADY FIXED in current codebase.**

`status.rs` uses `static HOST_ORT_API: AtomicPtr<ort::OrtApi>` with
`store(Ordering::Release)` / `load(Ordering::Acquire)`. No `static mut` anywhere.

**`host_api()` signature:** `pub(crate) fn host_api() -> *const ort::OrtApi`
— NOT marked `unsafe`. Callers in `graph_reader.rs` had `unsafe {}` wrappers
that were incorrect (`unused_unsafe` warnings). Fixed: removed the wrappers.
Deckard's `compute.rs:135` had the same pattern; that is Deckard's file so it
was left alone per ownership rules.

### H2 — `graphs` pointer not null-checked in `ep_compile` (HIGH)

**Status: FIXED (both existing check confirmed and error code improved).**

`ep_compile_inner` in `ep.rs` checks:
```rust
if graphs.is_null() {
    return invalid_arg_status("Compile: graphs pointer is null");
}
```
Also updated the combined null-arg check at entry to use `invalid_arg_status`
(ORT_INVALID_ARGUMENT) instead of `fail_status` (ORT_FAIL), which is the
semantically correct error code for bad-pointer arguments.

Added `invalid_arg_status` and `status_with_code` helpers to `status.rs`.

Unit test: `ep::tests::compile_null_graphs_returns_status`.

### H3 — Blanket `unsafe impl Send + Sync` on `OutboundGraphReader` (HIGH)

**Status: FIXED.**

Removed:
```rust
unsafe impl Send for OutboundGraphReader {}
unsafe impl Sync for OutboundGraphReader {}
```

Replaced with a doc comment explaining why the struct must NOT be `Send`/`Sync`:
the raw `*const OrtNode` pointers it holds are only valid within the ORT
callback frame. The struct is always used stack-locally in `ep_get_capability`
and `ep_compile`.

**File:** `graph_reader.rs:24-28`.

---

## Additional fixes

### `unused_unsafe` warnings (graph_reader.rs:40, :501)

`host_api()` is not an `unsafe fn`, so `unsafe { crate::status::host_api() }`
was incorrect. Removed the `unsafe` wrapper at both call sites.
(Deckard's `compute.rs:135` has the same issue; left for Deckard per ownership.)

### M1 — `check()` discarded the ORT error message

`graph_reader.rs::check()` now calls `GetErrorMessage` before `ReleaseStatus`
and includes the real ORT error text in the `Err(String)` result.

---

## `host_api()` signature — note for Deckard

**No signature change.** `host_api()` remains:
```rust
pub(crate) fn host_api() -> *const ort::OrtApi
```
The only change is that callers must NOT wrap it in `unsafe {}` — the function
is safe. Deckard's `compute.rs:135` still uses `unsafe { host_api() }`, which
compiles with a warning; that warning is Deckard's to fix per ownership rules.

---

## Validation

```
cargo check -p onnx-runtime-ep-plugin    → 0 errors, 0 warnings
cargo clippy -p onnx-runtime-ep-plugin -- -D warnings  → clean (0 errors)
cargo test -p onnx-runtime-ep-plugin --lib → 23 passed, 0 failed
```
