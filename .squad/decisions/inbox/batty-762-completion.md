# Decision: #762 Corrective Wave Completion

**Author:** Batty (systems engineer)  
**Date:** 2026-08-12  
**PR:** #762 (squad/ep-plugin-parity-cuda)

## Changes

### 1. EP Graph Assignment Assertion (Nabil's skip fixed)

Added `Session_GetEpGraphAssignmentInfo` assertion to `add_skip_layer_norm_mul_routed` test in `optional_slots.rs`. The helper `assert_ops_assigned_to_our_ep` reuses the same pattern from `plugin_ort_e2e.rs` (not a parallel implementation). The assertion proves that **Add**, **SkipLayerNormalization**, and **Mul** are all assigned to `cpu_ep`, not ORT's built-in `CPUExecutionProvider`. Both `record_ep_graph_assignment_info=1` and `disable_cpu_ep_fallback=1` are now set in `optional_slots::setup()`.

Nabil's reason ("not exposed in Rust bindings") was false — `OrtApi::Session_GetEpGraphAssignmentInfo` exists and was already used in `layernorm_dynamic_axis.rs:155`.

### 2. Kernel Registry `end_version` Ranges

**Was incomplete.** The cpu-plugin `build_kernel_registry_entries()` set `end_version: since` — meaning each op matched only its introduction opset. Fixed to `end_version: i32::MAX`, matching ORT's convention for ops that remain valid across all future opsets.

### 3. Factory `struct_size` Validation in Loader

Added `struct_size` bounds checks in `loader.rs` before reading `factory.name` and calling `factory.release`/`factory.ctx`. Mirrors the precedent in `provider_adapter.rs:77-85`. A newer host talking to an older plugin with a smaller struct will now safely skip the fields it cannot reach.

### 4. `NXRT_REQUIRE_ORT_TESTS=1` Fail-Loud Gate

When set, real-ORT conformance tests panic instead of silently skipping if ORT or the EP cdylib is unavailable. Implemented in:
- `plugin_ort_e2e.rs`: `skip_if_missing!` macro and `conformance_setup` helper
- `optional_slots.rs`: `setup()` helper and `require_ort_or_skip()` function
- `layernorm_dynamic_axis.rs`: skip paths

Default (unset) preserves today's silent-skip behavior.

### 5. Initializer-Backed Fixture

Added `matmul_initializer_weights` fixture and `conformance_matmul_initializer_weights` test. Uses constant initializer for MatMul weight `W[4,3]` — routes through prepacking path rather than graph inputs. This was the gap the sibling upstream PR identified.

### 6. `.gitignore` Negations

Added missing `!` negations for 5 fixtures that were git-tracked but not listed: `clip_no_min`, `simplified_layer_norm_two_outputs`, `skip_layer_norm_no_beta_bias`, `skip_layer_norm_output_sum`, `matmul_initializer_weights`.

## Preserved (from earlier rejections)

All B1 and B2 fixes are intact: scratch dtype from `output_dtypes[slot]`, `RoutedSlotKind` per-slot view map, `absent_outputs: HashSet<ValueId>`, `NodeInputSource::Absent`, panic containment, `c_char` portability.

## Test Counts

**278 passed, 0 failed** (was 277 before this wave; +1 from `conformance_matmul_initializer_weights`).
