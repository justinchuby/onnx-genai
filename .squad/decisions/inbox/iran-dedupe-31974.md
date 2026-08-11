# Decision: Consolidate NarrowToFloat/FloatToNarrow helpers (PR #31974)

**Date:** 2026-08-11  
**Author:** Iran  
**Branch:** `nxrt/mlas-bf16-layernorm` @ `6dd19a6f56`

## What was duplicated

`NarrowToFloat<T>` and `FloatToNarrow<T>` — thin template wrappers dispatching to
`MlasConvertHalfToFloatBuffer`/`BFloat16ToFloat` and
`MlasConvertFloatToHalfBuffer`/`FloatToBFloat16` respectively — were copy-pasted
identically into:

- `onnxruntime/core/providers/cpu/nn/layer_norm_impl.cc`
- `onnxruntime/contrib_ops/cpu/skip_layer_norm.cc`

Both were added by this PR (confirmed: not present in `HEAD~4`).

## ConvertMLFloat16ToFloatIfNeeded

Pre-existed in both files before this PR (`HEAD~4`). Left untouched as instructed.

## No suitable upstream helper found

Searched `core/util/`, `core/framework/`, `core/common/`, `include/`. The underlying
bulk-conversion functions (`MlasConvertHalfToFloatBuffer`, `BFloat16ToFloat`, etc.)
exist in `mlas.h` and `float16.h`, but no template that dispatches across both narrow
types existed.

## Resolution

Created `onnxruntime/core/util/narrow_float_utils.h` with the two templates in
`namespace onnxruntime`. Both source files now include the new header; local
definitions removed.

## Validation

- Build: clean (no warnings-as-errors regression)
- `LayerNormBFloat16*`: 17/17 pass
- `*LayerNorm*`: 96/96 pass
- clang-format: clean on all three touched files
