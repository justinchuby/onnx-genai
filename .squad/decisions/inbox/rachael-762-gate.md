# Decision: NXRT_REQUIRE_ORT_TESTS gate hardening

**Date:** 2026-08-12
**Author:** Rachael (via Copilot)
**PR:** #762

## Context

The `NXRT_REQUIRE_ORT_TESTS=1` gate ensures ORT conformance tests fail loudly when ORT is absent, rather than silently skipping. Two holes existed:

1. `find_ort_lib_dir` did not honour `CARGO_TARGET_DIR`, causing ORT to be "not found" on any machine using a custom target directory.
2. Fixture-missing and dlopen-failure paths used `.ok()?` or bare `return None`, bypassing the gate entirely.

## Decision

- Unified `find_ort_lib_dir` resolution into an `ort_discovery` module (in both `optional_slots.rs` and `plugin_ort_e2e.rs`) that checks: `NXRT_ORT_LIB_DIR` → `CARGO_TARGET_DIR`/debug/build → workspace default.
- Routed every skip path (fixture-missing, dlopen failure) through the gate check.
- Enabled `NXRT_REQUIRE_ORT_TESTS=1` in the `CLI ORT (Linux x86_64)` CI lane, which already builds ORT via ort-sys.

## Trade-offs

- Two copies of `ort_discovery` exist (one per integration test file) rather than a shared crate. Acceptable because test helpers are file-scoped in Rust integration tests.
- Gate is Linux-only in CI for now (Windows ORT path detection uses DLL names; can be extended later).
