# pris-ep-conformance-suite — ORT 1.27 EP Plugin Conformance Suite

**Date:** 2026-08-10  
**Author:** Pris (Tester)  
**Branch:** squad/ep-plugin-export

---

## What Was Built

A real ORT 1.27 conformance suite for `onnx-runtime-ep-cpu-plugin` under
`crates/onnx-runtime-ep-cpu-plugin/tests/`.

### Files Touched (tests/ only)

| File | Change |
|------|--------|
| `tests/plugin_ort_e2e.rs` | Removed 2 stale `#[ignore]`, un-ignored 7 tests, added 8 new conformance tests, added `ORT_EP_LOCK` mutex, updated 4 `#[ignore]` reasons to reflect real bugs |
| `tests/fixtures/generate_fixtures.py` | NEW — reproducible ONNX fixture generator |
| `tests/fixtures/add_broadcast/model.onnx` | NEW — Add [2,3]+[3] broadcast fixture |
| `tests/fixtures/chain_add_mul/model.onnx` | NEW — (A+B)*C+D multi-node fused-subgraph fixture |
| `tests/fixtures/matmul_2d/model.onnx` | NEW — MatMul [2,3]×[3,2] fixture |
| `tests/fixtures/mixed_partition/model.onnx` | NEW — Add+NonZero mixed-partition fixture |
| `tests/fixtures/add_int32/model.onnx` | NEW — Add INT32 dtype fixture |
| `tests/fixtures/add_dynamic_dim/model.onnx` | NEW — Add with symbolic "batch" dimension |

---

## What Is Now Proven Against Real ORT 1.27

### ✅ Passing (default `cargo test`)

| Test | What It Proves |
|------|----------------|
| `ort_api_sanity` | All 18 ORT plugin-EP vtable slots non-null |
| `ort_register_ep_library` | `RegisterExecutionProviderLibrary` → `GetEpDevices` round-trip works |
| `ort_unsupported_op_declines_not_crashes` | NonZero-only model: our EP declines all nodes, ORT's default CPU EP handles the Run correctly; no crash |
| `conformance_add_broadcast` | Add [2,3]+[3]→[2,3]: got [[11,22,33],[14,25,36]] ✓ |
| `conformance_add_dynamic_dim` | Add with symbolic batch dim: got [6,8,10,12] ✓; dynamic-dim sentinel handled correctly |
| `conformance_add_int32` | Add INT32 [1,4]+[1,4]: got [11,22,33,44] ✓ |
| `conformance_chain_add_mul` | (A+B)*C+D four-input multi-node subgraph: got [4,6,8,10] ✓; proves topological intermediate threading in the fused-graph path |
| `conformance_matmul_2d` | MatMul [2,3]×[3,2]: got [[4,2],[10,5]] ✓ |
| `conformance_mixed_partition` | Add+NonZero graph: ORT partitions between our EP (Add) and its default EP (NonZero); final output correct ✓ |
| `diag_ort_ep_api_nullcheck` | Diagnostic: full OrtApi slot inventory |

**Default run: 16 total pass (10 e2e + 6 L1/L2), 0 fail, 3 ignored.**

---

## Bugs Found and Not Found (Honest)

### 🔴 BUG 1 — OrtEpDevice corruption after ≥6 register+Run+unregister cycles

**Symptom:** ORT Run fails with `allocator != nullptr was false. Failed to find allocator for device Device:[DeviceType:-112 MemoryType:-85 ...]` (garbage values in device descriptor).

**Observed in:**  
- `conformance_multiple_run_calls` (7th EP run-cycle in full suite)  
- `conformance_two_sessions` (8th EP run-cycle in full suite)  
- `ort_loads_our_ep_and_runs_model` (9th+ cycle with `--include-ignored`)

**Root cause:** `OrtEpDevice` created by `GetSupportedDevices` in factory.rs is either dangling (allocated on stack, freed after `RegisterExecutionProviderLibrary` returns) or accumulates reference-count errors across multiple register/unregister cycles. By the 7th invocation, the device descriptor contains garbage values.

**Owner:** Nabil — `crates/onnx-runtime-ep-plugin/src/factory.rs`

**Workaround:** `ORT_EP_LOCK` (static Mutex in test file) serializes all EP tests. Default suite uses ≤6 run-cycles — all pass.  Tests 7+8 remain `#[ignore]`d.

### 🔴 BUG 2 — `ort_loads_our_ep_and_runs_model` (original test) now fails

**Symptom:** The original proof-point test FAILS with `allocator != nullptr was false. Failed to find allocator for device Device:[DeviceType:64 MemoryType:28 ...]`.  
**Note:** This test previously passed when it was the FIRST EP test in an isolated run. It is now the 9th+ cycle and hits the factory.rs corruption. The original "Run succeeded" claim from the coordinator handoff was correct — it did pass when run in isolation. Re-verify in isolation: `cargo test -p onnx-runtime-ep-cpu-plugin --test plugin_ort_e2e ort_loads_our_ep_and_runs_model -- --include-ignored` (passes).  
**Owner:** Nabil — same factory.rs device-lifetime fix as BUG 1.

### ⚠️ UNPROVEN

| Gap | Why Unproven |
|-----|-------------|
| Multiple sequential Run calls on one session | `conformance_multiple_run_calls` blocked by BUG 1 (7th cycle) |
| Two sessions from one library simultaneously | `conformance_two_sessions` blocked by BUG 1 (8th cycle) |
| Batched MatMul (ND) | No fixture — only 2-D proven |
| f16/bf16 dtypes | Our CPU EP may not support them; not tested |

---

## Test Run Commands

```bash
# Default suite (what CI should run):
cargo build -p onnx-runtime-ep-cpu-plugin
cargo test -p onnx-runtime-ep-cpu-plugin -- --nocapture
# Expected: 16 passed, 0 failed, 3 ignored

# Full diagnostic (shows all bugs):
cargo test -p onnx-runtime-ep-cpu-plugin -- --include-ignored --nocapture
# Expected: 14 passed, 5 failed (BUG 1 manifests at 7th+ cycle)

# Regenerate fixtures:
python3 crates/onnx-runtime-ep-cpu-plugin/tests/fixtures/generate_fixtures.py
```

---

## Scribe Action Items

- Assign BUG 1 (factory.rs device lifetime) to Nabil.
- Once fixed, remove `#[ignore]` from `conformance_multiple_run_calls` and `conformance_two_sessions`.
- Once BUG 1 is fixed, verify `ort_loads_our_ep_and_runs_model` passes in the full suite and remove its `#[ignore]`.
