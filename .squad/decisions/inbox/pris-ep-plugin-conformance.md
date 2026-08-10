# EP Plugin Conformance Test Architecture

**Author:** Pris (Tester)  
**Date:** 2026-08-10  
**Branch:** squad/ep-plugin-export

## Summary

Built the real test harness for the `onnx-runtime-ep-cpu-plugin` cdylib proving
ABI correctness, lifecycle safety, fail-closed behavior, and ORT 1.27.0
compatibility at three layers.

## Test Architecture

### L1 — ABI Surface (no ORT required)

Tests: `l1_nm_exported_symbols`, `l1_readelf_dyn_syms`  
File: `crates/onnx-runtime-ep-cpu-plugin/tests/plugin_export_abi.rs`

- Parses `nm --dynamic --defined-only --extern-only` output from the built `.so`.
- Asserts `CreateEpFactories` and `ReleaseEpFactory` exist as `T` (text/code) symbols.
- Asserts no other un-mangled Rust function names leak into the public symbol table.
- Also verifies via `readelf --dyn-syms` that both are `FUNC GLOBAL` in `.dynsym`.

**Result: PASSES** — exactly 2 T symbols, no leakage.

### L2 — dlopen/dlsym Direct Drive (no ORT required)

Tests: `dlopen_and_create_factory`, `compute_add_end_to_end`, `compute_add_broadcast`,
       `l2_fail_closed_unsupported_api_version`  
File: `crates/onnx-runtime-ep-cpu-plugin/tests/plugin_export_abi.rs`

- `dlopen_and_create_factory`: resolves `CreateEpFactories` and `ReleaseEpFactory`,
  calls factory lifecycle, verifies `ort_version_supported == ORT_API_VERSION`,
  checks `GetName` returns `"cpu_ep"`.
- `compute_add_end_to_end`: mock OrtApi with thread-local tensor state, drives the
  Add kernel end-to-end; asserts `[1,2,3,4] + [10,20,30,40] = [11,22,33,44]`.
- `compute_add_broadcast`: drives Add with broadcasting `[2,3] + [3] → [2,3]`;
  asserts broadcast output numerically.
- `l2_fail_closed_unsupported_api_version`: calls `CreateEpFactories` with an
  `OrtApiBase` whose `GetApi` returns null for all versions. Asserts:
  - 0 factories returned (fail-closed)
  - factory slot remains null

**Result: ALL PASS**

### L3 — Real ORT 1.27.0 End-to-End

Tests: `ort_api_sanity`, `ort_register_ep_library` [ignored], 
       `ort_loads_our_ep_and_runs_model` [ignored],
       `ort_unsupported_op_declines_not_crashes` [ignored],
       `diag_ort_ep_api_nullcheck`  
File: `crates/onnx-runtime-ep-cpu-plugin/tests/plugin_ort_e2e.rs`

- `ort_api_sanity`: loads ORT 1.27.0 via `OrtGetApiBase()→GetApi(27)` and asserts all
  18 plugin-EP vtable slots are non-null. **PASSES**.
- `diag_ort_ep_api_nullcheck`: prints null-check audit for key ORT API functions.
  **PASSES** — all PRESENT.

**Blocked tests (3) — root cause: `factory.rs::GetSupportedDevices`**:  
`OrtEpFactory::GetSupportedDevices` in `crates/onnx-runtime-ep-plugin/src/factory.rs`
returns `*out_num = 0` (no devices). ORT 1.27.0 calls this function inside
`RegisterExecutionProviderLibrary` and segfaults (SIGSEGV, signal 11) when the EP
factory reports zero devices.

**Fix required (Nabil's file):** `factory_get_supported_devices` must call
`OrtEpApi::CreateEpDevice` (obtained via `(*api).GetEpApi()`) to create an
`OrtEpDevice` for each compatible CPU hardware device and populate the output array.

A second blocker for the Run-level tests: `compute.rs` returns `ORT_NOT_IMPLEMENTED`
(Deckard's fix pending).

## Fixtures

- `tests/fixtures/add_1x4/model.onnx` — Add([1,4] f32, [1,4] f32) → [1,4] f32 (pre-existing)
- `tests/fixtures/nonzero_1x4/model.onnx` — NonZero([1,4] f32) → [2,N] int64 (new; unsupported op)

## Bug Found

**`ep.rs` struct initialization bug** (fixed in this session):  
`crates/onnx-runtime-ep-plugin/src/ep.rs` had `ValidateCompiledModelCompatibilityInfo: None`
in the `OrtEp` struct initializer — this field belongs to `OrtEpFactory`, not `OrtEp`.
Compiler rejected it (E0560). Fixed by removing the stray field and adding
`..Default::default()` to handle optional fields forward-compatibly.

## What Is Genuinely Proven

| Claim | Proven? |
|---|---|
| cdylib exports `CreateEpFactories` and `ReleaseEpFactory` (and only these two) | ✅ L1 |
| Factory lifecycle (create → inspect → release) works without ORT | ✅ L2 |
| `ort_version_supported` == ORT_API_VERSION (27) | ✅ L2 |
| `GetName` returns `"cpu_ep"` | ✅ L2 |
| Add kernel computes correct f32 values (flat and broadcast) | ✅ L2 |
| Fail-closed: unsupported ORT version returns 0 factories, not garbage | ✅ L2 |
| ORT 1.27 exposes all plugin-EP vtable slots (non-null) | ✅ L3 diagnostic |
| ORT `RegisterExecutionProviderLibrary` completes without crash | ❌ BLOCKED (factory.rs bug) |
| EP appears in `GetEpDevices` device list | ❌ BLOCKED |
| `CreateSession` on Add model with our EP | ❌ BLOCKED |
| Correct `Run` output ([1,2,3,4]+[5,6,7,8]=[6,8,10,12]) | ❌ BLOCKED (also Deckard) |
| Unsupported-op model falls back gracefully without crash | ❌ BLOCKED |

## What Remains Unproven Until Fixes Land

1. **Nabil**: Fix `factory_get_supported_devices` to call `OrtEpApi::CreateEpDevice`.
2. **Deckard**: Fix `Compute` to implement Add instead of returning `ORT_NOT_IMPLEMENTED`.
3. **After both**: all 3 ignored L3 tests should turn green with `--include-ignored`.
