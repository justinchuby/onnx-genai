# Pris — nxrt ABI Round-Trip Tests and CUDA Conformance Runner

**Date:** 2026-08-11T00:35:00Z  
**By:** Pris (tester)  
**Branch:** `squad/ep-plugin-parity-cuda` (draft PR #762)

## nxrt ABI Round-Trip: What Is Now Proven

10 integration tests in `crates/onnx-runtime-ep-nxrt-abi/tests/nxrt_roundtrip.rs` exercise
the nxrt dynamic EP ABI end-to-end via a real `cdylib` test fixture
(`crates/onnx-runtime-ep-nxrt-testplugin/`):

### Round-trip (positive):
1. **Full lifecycle:** dlopen → `nxrt_abi_version` → `nxrt_create_ep` → `nxrt_ep_name` → `nxrt_device_count` → `nxrt_destroy_ep` — all succeed with correct values.
2. **Ownership/lifetime:** Create 5 EP instances, destroy in reverse order, assert live-count reaches zero (no leak).
3. **Library outlives EP:** Correct drop ordering proven (destroy EP before unloading library).

### Negative (fail-closed, never crash):
4. **Incompatible major version:** Plugin reports major=99 → host rejects.
5. **Missing symbol:** `dlsym` for nonexistent symbol fails with actionable error.
6. **Missing library:** `dlopen` of nonexistent path fails with actionable error.
7. **Factory error:** Plugin returns `InternalError` status, null handle.
8. **Zero devices:** Plugin reports 0 devices → detectable by host.
9. **Panic containment:** `catch_status_panic` converts panics to `InternalError` with message.
10. **Null handle:** `nxrt_device_count(NULL, ...)` returns `InvalidArgument`, never crashes.

### Design note — ABI contract mismatch

The nxrt-abi crate (Nabil) exports `NxrtNegotiate` + `NxrtCreateEpFactories` with vtable-based handles.
The nxrt-host crate (Isidore) expects `nxrt_abi_version` + `nxrt_create_ep` + `nxrt_destroy_ep` + `nxrt_ep_name` + `nxrt_device_count`.
**These are different contracts.** Tests are written against the host's expected symbols (what actually gets `dlopen`'d). Nabil and Isidore must reconcile.

## CUDA Conformance Runner

`scripts/cuda_conformance_runner.sh` — a single-command runner for GPU hosts:

- **Detects preconditions:** nvidia-smi, libcuda.so, `cargo check --features cuda`
- **Exit codes:** 0 = VALIDATED, 1 = FAILED, 2 = UNVALIDATED
- **Phases when GPU present:** allocator → stream/copies → matmul → full conformance sweep → attention
- **On this host:** exits 2 (UNVALIDATED) — "nvidia-smi not found"
- **Invocation:** `./scripts/cuda_conformance_runner.sh` or `CUDA_VISIBLE_DEVICES=0 ./scripts/cuda_conformance_runner.sh`

## Blocking Issue Found

**`onnx-runtime-ep-nxrt-host` does not compile** — missing `sync` method in `impl ExecutionProvider for NxrtExecutionProvider` at `crates/onnx-runtime-ep-nxrt-host/src/provider_adapter.rs:90`. **Owner: Isidore.**

## Status

- **CUDA remains UNVALIDATED on this host** (no GPU, no NVIDIA driver).
- nxrt ABI round-trip is proven for the host loader's symbol contract.
- No regressions: ep-plugin 154+9 tests, nxrt-abi 19+10 tests all pass.
