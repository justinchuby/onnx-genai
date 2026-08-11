# Decision: EP assignment assertions in all normalization and narrow-float tests

**Date:** 2026-08-11
**Author:** Rachael
**Context:** PR #762 final hardening before ready

## Decision

Every test that exercises our EP's execution of a node must assert EP ownership
via `Session_GetEpGraphAssignmentInfo`, not just correct output. This is
complementary to `disable_cpu_ep_fallback=1`.

## Tests hardened

| Test | Op asserted | File |
|------|------------|------|
| `layernorm_dynamic_axis_mean_invstddev_shape` | `LayerNormalization` | `layernorm_dynamic_axis.rs` |
| `conformance_add_float16` | `Add` | `plugin_ort_e2e.rs` |
| `conformance_add_bfloat16` | `Add` | `plugin_ort_e2e.rs` |
| `conformance_layer_norm_multi_output` | `LayerNormalization` | `plugin_ort_e2e.rs` |
| `conformance_layer_norm_neg_axis` | `LayerNormalization` | `plugin_ort_e2e.rs` |
| `conformance_rms_norm` | `RMSNormalization` | `plugin_ort_e2e.rs` |

## Non-vacuity proof

Forced `Relu` assertion in layernorm_dynamic_axis → immediate failure:
```
[layernorm_dynamic_axis] Expected op 'Relu' assigned to cpu_ep, but assignment was: [("cpu_ep", "LayerNormalization")]
```

Forced `Relu` in conformance_add_float16 → immediate failure:
```
[conformance_add_float16] Expected op 'Relu' assigned to cpu_ep, but assignment was: [("cpu_ep", "Add")]
```
