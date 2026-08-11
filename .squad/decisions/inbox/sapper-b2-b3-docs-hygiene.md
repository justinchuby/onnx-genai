# B2 & B3 — docs/OperatorKernels.md + internal leakage sweep

**Date:** 2026-08-11
**Author:** Sapper
**PR:** microsoft/onnxruntime#31974 (`nxrt/mlas-bf16-layernorm`)

## B2 — `docs/OperatorKernels.md` update

### Method: hand-edit (not regenerated)

The generator script (`tools/python/gen_opkernel_doc.py`) requires a built `onnxruntime` Python wheel (`onnxruntime.capi.onnxruntime_pybind11_state`), which is impractical in this environment. The doc was **hand-edited**, not regenerated.

**⚠️ Flag this on the PR** — a hand-edited generated file needs explicit reviewer acknowledgement.

### Changes (5 insertions, 5 deletions — zero unrelated churn)

| Line | Op (CPU EP) | Section | Change |
|------|-------------|---------|--------|
| 232 | LayerNormalization (ONNX, 17+) | T | Added `tensor(bfloat16)` |
| 233 | LayerNormalization (contrib, [1,16]) | T, U, V | Added `tensor(bfloat16)` to T and V; U changed `tensor(double), tensor(float), tensor(float16)` → `tensor(double), tensor(float)` |
| 457 | SimplifiedLayerNormalization (contrib, 1+) | T, U, V | Same as above |
| 627 | SkipLayerNormalization (contrib, 1+) | T | Added `tensor(bfloat16)` |
| 628 | SkipSimplifiedLayerNormalization (contrib, 1+) | T | Added `tensor(bfloat16)` |

### N1 assumption

The U constraint change (dropping `tensor(float16)` from contrib LayerNorm/SimplifiedLayerNorm) reflects the **current code on this branch**, where `REGISTER_CONTRIB_KERNELS(MLFloat16, float)` registers U=float instead of U=MLFloat16. If Iran's N1 decision reverts this change, the U lines for contrib LayerNorm and SimplifiedLayerNorm must be restored to include `tensor(float16)`.

## B3 — internal leakage sweep

### Findings

All leakage is confined to test files owned by Luv:

| File:Line | Content | Owner |
|-----------|---------|-------|
| `onnxruntime/test/contrib_ops/layer_norm_bf16_cpu_test.cc:22` | `Chew's empirical measurements` | Luv |
| `onnxruntime/test/contrib_ops/layer_norm_bf16_cpu_test.cc:28` | `accumulation error Chew measured` | Luv |
| `onnxruntime/test/mlas/unittest/test_layernorm_bf16.cpp:23` | `when Resch's implementation lands` | Luv |
| `onnxruntime/test/mlas/unittest/test_layernorm_bf16.cpp:672` | `path that Resch is implementing` | Luv |

Luv has been instructed to strip these.

**No leakage found** in any other changed files (`onnxruntime/core/**`, `onnxruntime/contrib_ops/**`, `docs/**`).

No upstream repo policy was changed (`.gitignore`, lint config, CI config).
