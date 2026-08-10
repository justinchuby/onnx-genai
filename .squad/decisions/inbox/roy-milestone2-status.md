# EP Plugin Export — Milestone 2 Status

**Author:** Roy (Lead)  
**Date:** 2026-08-10T23:52Z  
**Branch:** `squad/ep-plugin-parity-cuda` at `5a5b40877`  
**Validation commit:** `5a5b40877`

---

## Milestone 1 — COMPLETE (🟡 YELLOW — may ship)

Branch `squad/ep-plugin-export`. All critical/high security blockers cleared.
23 conformance tests pass (6 plugin_export_abi + 17 e2e, including f16/bf16 which
landed in M2). M1 is independently mergeable.

## Milestone 2 — LARGELY COMPLETE (🟡 YELLOW — two pre-merge fixes required)

Three commits on `squad/ep-plugin-parity-cuda`: `2da0c4e7f`, `577047a74`, `5a5b40877`.

### What landed

1. **Trait↔C-ABI parity — PROVEN.** Pris's 9 parity tests in
   `crates/onnx-runtime-ep-plugin/tests/trait_cabi_parity.rs` all pass.
   Pinned rule: `C_ABI_claims = trait_claims ∩ { for_node != Declined }`.

2. **f16/bf16 end-to-end — PROVEN.** `conformance_add_float16` and
   `conformance_add_bfloat16` pass with exact bit-pattern assertions.
   This is our EP claiming and executing those nodes — not ORT fallback.

3. **Dtype-aware capability claiming.** `node_passes_dtype_filter()` + single
   `Vec<KernelRegistryEntry>` source of truth. Drift between claims and advertised
   types is structurally prevented.

4. **Device surfaces.** `DeviceSupport`, `DeviceAllocator` (OrtAllocator vtable),
   `DeviceSyncStream` (OrtSyncStreamImpl) — in `device.rs`, integrated into
   `factory.rs`, tested with mock EPs.

5. **`onnx-runtime-ep-cuda-plugin` crate.** Feature-gated behind `cuda`, workspace
   member, not default. `cargo check --workspace` passes without CUDA toolkit.

6. **`GetKernelRegistry` + `build_cpu_registry_with_descriptors()`.** Derives dtype
   support from real CPU registry via `RecordingOpRegistry`.

### Declined-set correction

The Declined set is **smaller than originally assumed**. `for_node` *can* infer
Squeeze (empty axes), ReduceMean, and Conv. Confirmed Declined: opset≥13 Unsqueeze
with data-dependent axes, and NonZero (data-dependent output shape). Any prior doc
claiming Squeeze/ReduceMean/Conv are Declined is wrong; corrected in PR doc.

### Pre-merge blockers (must fix before M2 PR)

| ID | Issue | Owner |
|----|-------|-------|
| CLIPPY | `ep.rs:1041,1047` — 2 `needless_borrows_for_generic_args` errors | Last M2 committer |
| M2-1 | EP leaked in `stream_release` — `DeviceSyncStream` has no Drop; `Box::into_raw` EP at `factory.rs:666–667` never freed | Leon |
| M2-2 | Misleading doc `device.rs:86` — says "Owned by this allocator; freed on drop" but allocator borrows ORT-owned pointer | Leon |

None of these block M1.

## Validation (Roy, 2026-08-10T23:52Z, commit `5a5b40877`)

| Command | Result |
|---------|--------|
| `cargo clippy -p onnx-runtime-ep-plugin --all-targets -- -D warnings` | **FAILS** — 2 errors `ep.rs:1041,1047` |
| `cargo test -p onnx-runtime-ep-plugin` | **141 pass** (132 lib + 9 parity), 0 fail, 1 doc-test ignored |
| `cargo test -p onnx-runtime-ep-cpu-plugin` | **23 pass** (6 + 17 e2e incl. f16/bf16), 0 fail |
| `cargo check --workspace` | **pass** (pre-existing unrelated warnings in `onnx-genai-bench`) |

## §524 Compliance Status

- C ABI: ✅ Complete and proven (23 ORT conformance tests)
- Rust trait: ✅ **Now proven** (9 parity tests)
- Fail-closed: ✅ Complete (Declined path + dtype filter)
- **Native nxrt dynamic ABI: 🔴 Not implemented.** Neither milestone addresses this.
  Remaining §524 gap must be tracked explicitly.

## CUDA Gap

The CUDA shim crate and device surfaces exist. What remains is **both** hardware-gated
(no CUDA toolkit/GPU on this host) **and** requires real engineering: device-pointer
crossing the ORT C ABI, cuMemAlloc/cuMemFree integration, stream/event mapping. Do not
describe this as purely hardware-blocked.

## PR Recommendation

Two stacked PRs (M1 independent, M2 stacked on M1). M1 may be pushed and opened
immediately by a user or CI runner with credentials. M2 requires fixing clippy
regression + M2-1 + M2-2 before the PR is opened.

**Push blocker:** No `GH_TOKEN`/`GITHUB_TOKEN`, no SSH key, `gh` not logged in on this
host. `git ls-remote origin` confirms neither branch exists remotely.
