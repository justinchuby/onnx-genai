# Decision: EP Plugin FFI Audit — CRITICAL/HIGH blockers

**By:** Holden (Security Engineer)
**Date:** 2026-08-10T20:12:35.793+00:00
**Scope:** `crates/onnx-runtime-ep-plugin/`

## Context

The outbound ORT plugin-EP adapter has 4 blocking security findings that must be resolved before the code can ship as a loadable library inside upstream ORT's process.

## Findings (ship-blocking)

1. **CRITICAL — No `catch_unwind` on any `extern "C"` callback.** A Rust panic unwinding through the C ABI is undefined behavior. All 9+ exported callbacks are unguarded.

2. **HIGH — `static mut HOST_ORT_API` is a data race.** Must be replaced with `AtomicPtr`.

3. **HIGH — `graphs` pointer not null-checked in `ep_compile`.** Null deref → segfault in ORT's process.

4. **HIGH — Blanket `unsafe impl Send + Sync` on `OutboundGraphReader`** stores raw ORT pointers valid only within the callback frame. Remove the impls.

## Decision

These 4 findings are **RED (ship-blocking)**. The EP plugin adapter MUST NOT be linked into a release build or integration-tested against real ORT until all 4 are addressed. Nabil owns the fixes; Holden will re-review.

Full audit: `docs/EP_PLUGIN_EXPORT_SECURITY_AUDIT.md`
