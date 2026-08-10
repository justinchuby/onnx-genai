# Milestone: ORT Plugin EP Export — CPU EP end-to-end

**By:** Roy (Lead)
**Date:** 2026-08-10T22:42:00Z
**Branch:** `squad/ep-plugin-export`

## Milestone Summary

The Rust CPU execution provider (`onnx-runtime-ep-cpu-plugin`) now loads, registers,
and executes as a real ORT plugin EP under upstream ORT 1.27.0. This closes the
"EPs unreachable from upstream ORT" gap that existed before this branch.

## Verified facts

All claims below were verified by Roy running the commands on 2026-08-10:

- `cargo check -p onnx-runtime-ep-plugin` exits 0 (1 unused-fn warning only)
- `cargo check -p onnx-runtime-ep-cpu-plugin` exits 0
- `cargo test -p onnx-runtime-ep-plugin --lib`: **82 passed, 0 failed**
- 10 integration tests individually pass:
  `ort_register_ep_library`, `ort_loads_our_ep_and_runs_model`,
  `ort_unsupported_op_declines_not_crashes`, `conformance_add_broadcast`,
  `conformance_add_dynamic_dim`, `conformance_add_int32`,
  `conformance_chain_add_mul`, `conformance_matmul_2d`,
  `conformance_mixed_partition`, `conformance_multiple_run_calls`
- `conformance_two_sessions` is `#[ignore]`d due to known OrtEpDevice corruption
  after ≥6 register cycles (Nabil, factory.rs)

## ORT Compatibility Boundary

| Item | Value |
|------|-------|
| ORT version | 1.27.0 (`ORT_API_VERSION = 27`) |
| Required exports | `CreateEpFactories`, `ReleaseEpFactory` |
| `OrtEp` fields implemented | `GetName`, `GetCapability`, `Compile`, `ReleaseNodeComputeInfos` |
| `OrtEp` fields set to `None` | 20 optional fields (1.25–1.27 additions) |
| Minimum ORT for plugin EP API | 1.22 |
| Host call sequence | `CreateEnv` → `RegisterExecutionProviderLibrary` → `GetEpDevices` → `SessionOptionsAppendExecutionProvider_V2` → `CreateSession` → `Run` |

## Hard-won ABI contracts (guidance for future EP authors)

**A. OrtMemoryInfo outlives OrtEpDevice.**
`EpDevice_AddAllocatorInfo` stores the pointer; ORT does NOT copy. Release only on
failure. Use `CreateMemoryInfo_V2`, not `CreateCpuMemoryInfo` (legacy API).

**B. OrtGraph*/OrtNode* must not be cached beyond Compile.**
Copy all needed data into owned Rust structures during `Compile`. ORT may free
these pointers after the callback returns.

## Provider readiness

| Provider | Status | Blocker |
|----------|--------|---------|
| CPU EP | **NEAR** (security-pending) | Holden N1: `compute_execute` unguarded |
| CUDA EP | **BLOCKED** | No CUDA toolkit/GPU on this host |

## Push credentials

This host has no GitHub push credentials. The coordinator must push the branch.
