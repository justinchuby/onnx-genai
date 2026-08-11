# Pris — Sixth Review Pass on PR #762

**Date:** 2026-08-11  
**Reviewer:** Pris (Tester)  
**Branch:** `squad/ep-plugin-parity-cuda` (head `e0ef1f0a8`)  
**Scope:** Test-integrity audit — are tests testing what they claim?

---

## 1. Are the 8 assignment assertions real?

**YES — they are genuine and correctly wired.**

`Session_GetEpGraphAssignmentInfo` (ORT ≥1.24) is called via `query_ep_assignment()` at `plugin_ort_e2e.rs:831–870`. The implementation:
1. Enables `session.record_ep_graph_assignment_info=1` unconditionally in `conformance_setup()` (line ~790).
2. Calls the real ORT API functions: `Session_GetEpGraphAssignmentInfo`, `EpAssignedSubgraph_GetEpName`, `EpAssignedSubgraph_GetNodes`, `EpAssignedNode_GetOperatorType`.
3. Filters assignments by `ep == "cpu_ep"` — which is **our** EP (defined at `crates/onnx-runtime-ep-cpu/src/provider.rs:120`).

**Name collision risk: NONE.** ORT's built-in CPU EP is named `"CPUExecutionProvider"`. Our EP is `"cpu_ep"`. These are distinct strings; the filter `ep == "cpu_ep"` cannot match ORT's built-in.

Tests with direct `assert_ops_assigned_to_our_ep` calls (8 total):
- `conformance_add_broadcast` — asserts `["Add"]`
- `conformance_chain_add_mul` — asserts `["Add", "Mul"]`
- `conformance_matmul_2d` — asserts `["MatMul"]`
- `conformance_add_int32` — asserts `["Add"]`
- `conformance_add_dynamic_dim` — asserts `["Add"]`
- `conformance_matmul_batched_nd` — asserts `["MatMul"]`
- `conformance_cast_f32_to_i64` — asserts `["Cast"]`
- `conformance_where_bool_f32` — asserts `["Where"]`

Additionally, `conformance_mixed_partition` and `conformance_shape_f32` call `query_ep_assignment` directly with softer assertions (appropriate for their semantics).

**Forced-failure check:** The `assert_ops_assigned_to_our_ep` function panics with `"Expected op '{op}' assigned to cpu_ep, but assignment was: {assignments:?}"`. Resch's reported forced failure (asserting `Relu` on the `add_1x4` model) is consistent with this code path — the assertion would print the actual assignment list (e.g., `[("cpu_ep", "Add")]`). I did NOT run this myself (no build environment available; read-only audit).

---

## 2. Test Classification Table

### Real-ORT tests in `plugin_ort_e2e.rs`:

| Test | Category | Notes |
|------|----------|-------|
| `ort_api_sanity` | N/A | Vtable probe, not an EP-exercises-model test |
| `ort_register_ep_library` | N/A | Registration only, no model run |
| `ort_loads_our_ep_and_runs_model` | (b) | Uses device-lookup but NO `disable_cpu_ep_fallback`, NO assignment assertion. Values verified. |
| `ort_unsupported_op_declines_not_crashes` | **(c)** | **Explicitly allows fallback**, no assignment check, no fallback-disable. Proves no crash but cannot prove our EP declined vs never loaded. |
| `conformance_add_broadcast` | **(a)** | Assignment assertion + fallback-disable + value check |
| `conformance_chain_add_mul` | **(a)** | Assignment assertion + fallback-disable + value check |
| `conformance_matmul_2d` | **(a)** | Assignment assertion + fallback-disable + value check |
| `conformance_mixed_partition` | Special | Fallback-disable OFF by design; has assignment query with defensive assertion (`NonZero` must NOT be on our EP) |
| `conformance_add_int32` | **(a)** | Assignment assertion + fallback-disable + value check |
| `conformance_add_dynamic_dim` | **(a)** | Assignment assertion + fallback-disable + value check |
| `conformance_multiple_run_calls` | **(b)** | Fallback-disable ON, values checked, but NO assignment assertion |
| `conformance_two_sessions` | **(b)** | Fallback-disable ON, values checked, but NO assignment assertion |
| `conformance_matmul_batched_nd` | **(a)** | Assignment assertion + fallback-disable + value check |
| `stress_register_run_unregister_cycles` | **(b)** | Fallback-disable OFF (not set), values checked per cycle, no assignment |
| `conformance_add_float16` | **(b)** | Fallback-disable ON, values checked, but NO assignment assertion |
| `conformance_add_bfloat16` | **(b)** | Fallback-disable ON, values checked, but NO assignment assertion |
| `conformance_cast_f32_to_i64` | **(a)** | Assignment assertion + fallback-disable + value check |
| `conformance_where_bool_f32` | **(a)** | Assignment assertion + fallback-disable + value check |
| `conformance_shape_f32` | **(a)** soft | Has assignment query (soft: logs if Shape was constant-folded), fallback-disable ON, values checked |
| `conformance_layer_norm_multi_output` | **(b)** | Fallback-disable ON, shape+value check, but NO assignment assertion |
| `conformance_layer_norm_neg_axis` | **(b)** | Fallback-disable ON, shape+value check, but NO assignment assertion |
| `conformance_rms_norm` | **(b)** | Fallback-disable ON, shape+value check, but NO assignment assertion |
| `diag_ort_ep_api_nullcheck` | N/A | Diagnostic only |

### Real-ORT tests in `optional_slots.rs`:

| Test | Category | Notes |
|------|----------|-------|
| `skip_layer_norm_output_sum_position` | **(b)** | Fallback-disable ON, values checked, no assignment assertion |
| `clip_omitted_min_with_max` | **(b)** | Fallback-disable ON, values checked, no assignment assertion |
| `skip_layer_norm_omitted_beta_bias` | **(b)** | Fallback-disable ON, values checked, no assignment assertion |
| `simplified_layer_norm_two_outputs_position` | **(b)** | Fallback-disable ON, values checked, no assignment assertion |

### Real-ORT tests in `layernorm_dynamic_axis.rs`:

| Test | Category | Notes |
|------|----------|-------|
| `layernorm_dynamic_axis_mean_invstddev_shape` | **(c)** | **No `disable_cpu_ep_fallback`, no assignment assertion.** Only proves shapes are correct — could pass if ORT's built-in CPU EP does LayerNorm. |

### CUDA plugin tests in `cuda_fail_closed.rs`:

| Test | Category | Notes |
|------|----------|-------|
| All 6 tests | N/A | No model run through ORT; test fail-closed behaviour, ABI, allocator, copy-direction |

---

## 3. `conformance_mixed_partition` — still valid?

**PARTIALLY VALID.** The test:
1. Asserts our EP is NEVER assigned `NonZero` (defensive negative assertion) ✓
2. Checks if ORT assigned `Add` to our EP — but logs `"ℹ ORT routed all to built-in CPUExecutionProvider (no partition)"` if it didn't

**The test's stated purpose — "prove ORT partitions the graph" — is NOT reliably demonstrated.** ORT may legitimately route everything to its built-in CPU EP when `disable_cpu_ep_fallback=false`, making the test's partition claim vacuous.

**Verdict:** The test is NOT worthless (it proves our EP doesn't claim NonZero and the final output is correct regardless of routing), but its partitioning demonstration is aspirational, not guaranteed. The `if ours.contains(&"Add")` branch may never execute.

---

## 4. `conformance_shape_f32` — can it fail?

**YES, it can fail — but only under narrow conditions.** With `disable_cpu_ep_fallback=1`, if ORT constant-folds Shape (removing it from the graph entirely), no EP assignment is needed. The test:
- Queries assignment, logs if Shape was folded (soft check) ✓
- Still verifies the output dtype is INT64 and values are `[3,4,5]` ✓

**The output/value check is NOT vacuous** — it confirms the session produces correct output. But the EP-coverage aspect (proving OUR code handles Shape) is indeed a soft check. If ORT always constant-folds Shape, our Shape kernel is never exercised in this test.

**This is a test that likely always passes regardless of our EP's Shape implementation.** It cannot distinguish "our kernel computed Shape correctly" from "ORT folded it at graph optimization time."

---

## 5. Historical Bug Regression Coverage

| Bug | Covered? | Test(s) |
|-----|----------|---------|
| **BL2: Compacted optional output slots** (`SkipLayerNormalization output,"","",sum` writing mean into sum) | ✅ YES | `skip_layer_norm_output_sum_position` — asserts sum contains `X+skip`, not mean |
| **BL3: Absent optional inputs aliased to input 0** | ✅ YES | `clip_omitted_min_with_max` + `skip_layer_norm_omitted_beta_bias` |
| **BL1: LayerNorm axis against truncated rank** (dynamic dims) | ✅ YES | `layernorm_dynamic_axis_mean_invstddev_shape` — BUT category (c), no proof our EP executes it |
| **Forgeable name-based sentinels** | ❓ NOT DIRECTLY | No test I can find that specifically exercises the sentinel-based absent-input detection. The `clip_no_min` and `skip_layer_norm_no_beta_bias` tests exercise absent inputs but don't test forgeability (model with an input literally named the sentinel string). |
| **Zero-substituted unknown dims** | ✅ YES | `conformance_add_dynamic_dim` — assignment-proven, with value check |
| **Use-after-free in factory** (cycle ≥6) | ✅ YES | `stress_register_run_unregister_cycles` — 25 cycles |

---

## 6. Fixtures

**22 `.onnx` files in `crates/onnx-runtime-ep-cpu-plugin/tests/fixtures/`** — all tracked in git (`git ls-files` confirms). No `.gitignore` in that directory. No untracked fixtures found. No tracked fixture is unused (each maps to a test).

The `layer_norm_dynamic_axis/model.onnx` fixture is referenced by `layernorm_dynamic_axis.rs` ✓.

---

## 7. Tests That Assert Success Rather Than Correctness

| Test | Issue |
|------|-------|
| `ort_unsupported_op_declines_not_crashes` | Asserts `check_status(Run)` succeeds and output is non-null, but **does NOT validate the NonZero output values**. It checks session builds and runs without crash, but doesn't verify correctness. |
| `stress_register_run_unregister_cycles` | Each cycle verifies output values ✓ — this is fine. |
| `ort_loads_our_ep_and_runs_model` | Verifies output values ✓ |

**Only `ort_unsupported_op_declines_not_crashes` asserts success without verifying output values** for its NonZero computation. However, this test's purpose is explicitly "must not crash" rather than correctness, so this is acceptable by intent.

---

## FINDINGS

### BLOCKING

**None.**

### SUBSTANTIVE

1. **`layernorm_dynamic_axis_mean_invstddev_shape` is category (c)** — no `disable_cpu_ep_fallback`, no assignment assertion. If ORT's built-in CPU EP handles LayerNorm (which it does), this test passes without our EP ever executing. This is the BL1 regression test and it may not actually exercise our code.
   - **File:** `crates/onnx-runtime-ep-cpu-plugin/tests/layernorm_dynamic_axis.rs`
   - **Owner:** Resch
   - **Fix:** Add `disable_cpu_ep_fallback=1` and `assert_ops_assigned_to_our_ep(api, session, &["LayerNormalization"], ...)` (same pattern as `conformance_add_broadcast`).

2. **`conformance_add_float16` and `conformance_add_bfloat16` lack assignment assertions** — they have fallback-disable (good) but don't directly prove via `Session_GetEpGraphAssignmentInfo` that the f16/bf16 nodes landed on our EP. The type-constraint routing should be confirmed.
   - **File:** `crates/onnx-runtime-ep-cpu-plugin/tests/plugin_ort_e2e.rs`
   - **Owner:** Resch
   - **Fix:** Add `assert_ops_assigned_to_our_ep(api, session, &["Add"], "add_float16")` (one line each).

3. **`conformance_layer_norm_multi_output`, `conformance_layer_norm_neg_axis`, `conformance_rms_norm` lack assignment assertions** — they have fallback-disable (proving the session would error if our EP declined) which is strong evidence, but adding assignment assertions would make them category (a) for completeness.
   - **File:** `crates/onnx-runtime-ep-cpu-plugin/tests/plugin_ort_e2e.rs`
   - **Owner:** Resch

4. **No regression test for forgeable name-based sentinels** — the bug where a model input named like an internal sentinel string could trick the absent-input detection. No test creates a model with such a name.
   - **Owner:** Resch (or whoever owns the sentinel-removal fix)

### NITS

5. **`conformance_shape_f32`** could log a warning if Shape is not in the assignment list, suggesting ORT constant-folded it. Currently it does — acceptable as-is.

6. **`ort_loads_our_ep_and_runs_model`** uses registration name `"cpu_ep_e2e"` but no fallback-disable and no assignment assertion. Adding these would promote it from (b) to (a) trivially.

---

## Forced-Failure Reproduction

**I did NOT reproduce the forced failure myself.** This is a read-only audit without a build environment. The code path (`assert_ops_assigned_to_our_ep` → panic with diagnostic message) is structurally sound based on code reading. The forced-failure output Resch reported (`"Expected op 'Relu' assigned to cpu_ep, but assignment was: [("cpu_ep", "Add")]"`) is consistent with the format string at line ~878 of `plugin_ort_e2e.rs`.

---

## Should #762 Leave Draft?

**YES — with one reservation.**

The test suite is in vastly better shape than any prior round. The 8 assignment assertions are real and correctly distinguish our EP from ORT's built-in. The historical bugs are regression-covered (5/6 confirmed; name-sentinel is the gap). The `disable_cpu_ep_fallback` mechanism is deployed across all conformance tests (except the two that intentionally allow fallback).

**The one reservation** is item (1) above: `layernorm_dynamic_axis` is category (c) and is the sole regression test for BL1. This is the same class of vulnerability that caused the catastrophic false-green two rounds ago. Adding `disable_cpu_ep_fallback=1` to that one test (literally 5 lines of code) would close the last hole.

**Shortest path to ready-for-review:** Add fallback-disable to `layernorm_dynamic_axis.rs`. Optionally add assignment assertions to f16/bf16 tests (one line each). The rest are nits.

---

## What I Verified vs Took on Trust

| Item | Method |
|------|--------|
| `Session_GetEpGraphAssignmentInfo` is genuinely called | Read code — traced through `conformance_setup` → config entry → `query_ep_assignment` |
| `"cpu_ep"` is our EP, not ORT's built-in | Read `provider.rs:120` (our EP returns `"cpu_ep"`); ORT's is `"CPUExecutionProvider"` |
| All 22 fixtures are tracked | `git ls-files` |
| No unused fixtures | Cross-referenced test code with fixture directory |
| Tests verify values (not just success) | Read each test's assertion block |
| `disable_cpu_ep_fallback` is set in `conformance_setup` | Read line ~769 |
| `layernorm_dynamic_axis` lacks fallback-disable | Confirmed via grep (zero matches) |
| EP crate test count (269 pass, 0 fail) | Taken on trust from established facts |
| Forced-failure output format | Read code, did not execute |
| CUDA tests don't overstate what they prove | Read all 6 — they explicitly state "without a CUDA GPU" context |
