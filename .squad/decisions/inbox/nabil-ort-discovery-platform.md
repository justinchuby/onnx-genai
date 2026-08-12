# Decision: Unify ORT library filename via `ort_discovery::ort_lib_name()`

**Date:** 2026-08-12
**Author:** Nabil (via Copilot)
**Status:** Implemented (PR #762)

## Context

`plugin_ort_e2e.rs` hardcoded `"libonnxruntime.so"` in 7 call sites despite
`ort_discovery.rs` already providing the platform-correct `ort_lib_name()`.
This caused panics on Windows (`onnxruntime.dll`) and macOS (`libonnxruntime.dylib`).

## Decision

1. **Single source of truth for the ORT library filename:**
   All test files must use `ort_discovery::ort_lib_name()` — never a hardcoded string.

2. **Use `PathBuf::join` for all path construction** — never string concatenation —
   to avoid mixed separator issues on Windows.

3. **`skip_if_missing!` macro must be the ONLY skip mechanism** in EP e2e tests.
   It respects `NXRT_REQUIRE_ORT_TESTS=1` uniformly.  The `diag_ort_ep_api_nullcheck`
   test had a hand-rolled match that bypassed the fail-loud gate; fixed.

## Recurring Bug Families Addressed

- **Copy-drift:** eliminated 7 hardcoded `.so` literals.
- **Platform/arch assumption:** `cfg!`-based name selection already existed in
  `ort_discovery.rs`; callers now use it.

## Verification

- Linux x86_64: 40 tests pass, 0 skipped, all 7 previously-failing tests run and pass.
- Fail-loud gate: with all ORT libs renamed + `NXRT_REQUIRE_ORT_TESTS=1`, tests
  correctly panic with a clear message.
- Windows/macOS: **unverified at runtime** (no hardware). Correctness by construction
  (`cfg!(target_os)` + `PathBuf::join`). CI must confirm.
