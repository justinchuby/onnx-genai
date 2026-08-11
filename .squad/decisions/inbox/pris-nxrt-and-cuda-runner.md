# Pris — nxrt ABI Round-Trip Tests and CUDA Conformance Runner

**Date:** 2026-08-11T00:40:00Z  
**By:** Pris (tester)  
**Branch:** `squad/ep-plugin-parity-cuda` (draft PR #762)  
**Commit:** 99560c876

## nxrt ABI Round-Trip: What Is Now Proven (Real ABI)

The test fixture (`crates/onnx-runtime-ep-nxrt-testplugin/`) has been **rebuilt on
Nabil's real shipped ABI** (`onnx-runtime-ep-nxrt-abi`). It exports `NxrtNegotiate` +
`NxrtCreateEpFactories` via the `export_nxrt_ep_factories!` macro with full panic
containment. The old private duplicate symbols are gone.

The crate is now a **workspace member** (not default-member), included in
`cargo check --workspace`.

10 integration tests in `crates/onnx-runtime-ep-nxrt-host/tests/nxrt_abi_roundtrip.rs`
exercise the full lifecycle via `libloading` against the real cdylib:

### Round-trip (positive):
1. **Negotiate:** `NxrtNegotiate` succeeds with matching major/minor, capability flags within `NXRT_CAP_KNOWN_MASK`.
2. **Full lifecycle:** negotiate → `NxrtCreateEpFactories` (1 factory, num_devices≥1) → factory name → `create_ep` → EP device_type/name → `get_capability` (claims 0 nodes, fail-closed) → release EP → release factory.
3. **Ownership/lifetime:** Create 3 EPs, release all, assert drop counter returns to zero.

### Negative (fail-closed, never crash or hang):
4. **Incompatible major version:** host_range [99,99] → `VersionMismatch` status.
5. **Plugin minor newer than host:** `validate_negotiation` rejects agreed_minor > host minor_max.
6. **Unknown capability bits:** bit 63 set → `validate_negotiation` rejects via `NXRT_CAP_KNOWN_MASK`.
7. **Missing library file:** `Library::new("/nonexistent/...")` fails gracefully.
8. **Missing/misspelled symbol:** `lib.get(b"NxrtNegotiate_TYPO")` fails gracefully.
9. **Panic containment:** `NXRT_TEST_PANIC=1` → macro catches, returns `InternalError`, zeroes `out_num`.
10. **Factory error (panic-based):** `NXRT_TEST_FACTORY_ERROR=1` → same containment path.

### Design note — host adapter not yet rewired

Isidore's host loader (`crates/onnx-runtime-ep-nxrt-host/src/loader.rs`) still uses the
old private `abi_contract.rs` symbols (`nxrt_abi_version`/`nxrt_create_ep`). The
integration tests bypass the host adapter and call the real ABI symbols directly. Once
Isidore re-points the loader at Nabil's ABI, the host adapter tests will also pass
through the real contract.

## CUDA Conformance Runner

`scripts/cuda_conformance_runner.sh` — single-command runner for GPU hosts:

- **Preconditions:** nvidia-smi reachable, libcuda.so + libcublasLt.so.13 loadable, ≥1 CUDA GPU, `cargo check --features cuda`
- **Exit codes:** 0 = VALIDATED, 1 = FAILED, 2 = UNVALIDATED
- **Phases:** CreateEpFactories → GetSupportedDevices → CreateEp → allocator → sync-stream → MatMul session Run (numeric) → weight offload
- **On this host:** exits 2 (UNVALIDATED) — "nvidia-smi not found"

## Status

- **CUDA remains UNVALIDATED on this host** (no GPU, no NVIDIA driver).
- nxrt ABI round-trip is proven end-to-end against the **shipped** `onnx-runtime-ep-nxrt-abi` surface.
- No regressions: `cargo test -p onnx-runtime-ep-plugin` 154 lib + 9 parity, `cargo test -p onnx-runtime-ep-cpu-plugin` 23.

---

## Update 2026-08-11T00:48:00Z — Integration gap closed, suite de-duplicated

### Fixture Resolution Fixed

The `testplugin_path()` in `nxrt_abi_roundtrip.rs` now resolves correctly after
the testplugin became a workspace member:

1. `NXRT_TESTPLUGIN_PATH` env override (asserts file exists)
2. `CARGO_TARGET_DIR` / `$PROFILE` / libname
3. Workspace root / `target` / `$PROFILE` / libname
4. **Auto-build fallback:** invokes `cargo build -p onnx-runtime-ep-nxrt-testplugin`
   and asserts success — the test **cannot silently pass** with a missing fixture.

`PROFILE` defaults to `debug`; set it to `release` for release-mode testing.

### Suite De-duplication

Deleted `crates/onnx-runtime-ep-nxrt-abi/tests/nxrt_roundtrip.rs` (the duplicate).
The authoritative nxrt round-trip suite lives in
`crates/onnx-runtime-ep-nxrt-host/tests/nxrt_abi_roundtrip.rs` because it exercises
the real shipped ABI types (`NxrtNegotiateRequest`, `NxrtEpFactoryVtable`, etc.)
directly.

### Env-var race fix

Added `ENV_MUTEX` serialization for tests that set `NXRT_TEST_PANIC` /
`NXRT_TEST_FACTORY_ERROR` and those that call `create_factories` (which reads
those vars from the shared library).

### Validation Results

- `cargo check --workspace` ✓
- `onnx-runtime-ep-nxrt-abi`: 30 unit passed, 0 failed
- `onnx-runtime-ep-nxrt-host`: 4 unit + 10 integration passed, 0 failed
- `onnx-runtime-ep-plugin`: 154 lib + 9 parity passed
- `onnx-runtime-ep-cpu-plugin`: 23 passed
- Clean-state (rm cdylib → re-run): 10 passed (auto-build triggered)
- **CUDA remains UNVALIDATED** — no GPU on this host.

---

## Update 2026-08-11T01:40:00Z — Stale-artifact false-pass eliminated from cpu-plugin tests

### Problem

Five CI lanes failed at `l1_nm_exported_symbols` because `plugin_export_abi.rs`
and `plugin_ort_e2e.rs` used hardcoded `target/debug/*.so` / `target/release/*.so`
paths — no `CARGO_TARGET_DIR` support, no profile-awareness, no auto-build, and
Linux-only `.so` extension (would also fail on Windows `.dll` / macOS `.dylib`).

Local passes were a false positive: a stale `libonnx_runtime_ep_cpu_plugin.so` from
an earlier manual build was sitting in `target/debug/`. Removing it reproduced the
CI failure exactly.

### Fix — shared `cdylib_resolve` helper

Created `crates/onnx-runtime-ep-cpu-plugin/tests/cdylib_resolve.rs` with the same
resolution pattern already proven in nxrt's `testplugin_path()`:

1. `NXRT_CPU_PLUGIN_PATH` env var (explicit override, asserts file exists)
2. `CARGO_TARGET_DIR` / `$PROFILE` / platform-libname
3. Workspace root / `target` / `$PROFILE` / platform-libname
4. Auto-build via `cargo build -p onnx-runtime-ep-cpu-plugin`; panic if build fails

Platform-appropriate filenames: `.so` (Linux), `.dylib` (macOS), `.dll` (Windows).

Both `plugin_export_abi.rs` and `plugin_ort_e2e.rs` now `mod cdylib_resolve;` and
delegate to the shared helper, eliminating the duplicated hardcoded logic.

### Rule

**Every test that loads a built artifact by path MUST:**
- Honor `CARGO_TARGET_DIR` and `$PROFILE`
- Use platform-appropriate library extensions
- Auto-build when absent and fail loudly if the build itself fails
- Never rely on stale artifacts from prior manual builds
