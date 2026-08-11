# Decision: Fix contrib LayerNorm `U` type constraint for narrow-float types

**Date:** 2026-08-11  
**Author:** Deckard (Systems Dev)  
**PR:** microsoft/onnxruntime#31974 (MLAS BFloat16 LayerNorm)

## Finding

The contrib CPU `REGISTER_CONTRIB_KERNELS(T)` macro registered `U=T` for all types,
including MLFloat16 and the new BFloat16. The contrib schema constrains
`U` to `{tensor(float)}` only (Mean / InvStdDev outputs). This is a **pre-existing
mismatch** — MLFloat16 has shipped this way since the contrib op was introduced.

### Does it affect correctness?

**No.** The contrib `LayerNorm` constructor does not pass `contrib_op=true` to
`LayerNormImpl`, so `SrcDispatcher` always takes the non-contrib branch and calls
`ComputeImpl<T, float>`. Mean/InvStdDev are always float regardless of registration.

The CUDA contrib already handles this correctly: `REGISTER_KERNEL_TYPED(MLFloat16, float, MLFloat16)`.

## Decision: Option (b) — fix the macro for all narrow-float types

Changed the macro from `REGISTER_CONTRIB_KERNELS(T)` (one param, U=T) to
`REGISTER_CONTRIB_KERNELS(T, U)` (two params), and registered narrow types with `U=float`:

```
REGISTER_CONTRIB_KERNELS(float, float)
REGISTER_CONTRIB_KERNELS(double, double)
REGISTER_CONTRIB_KERNELS(MLFloat16, float)
REGISTER_CONTRIB_KERNELS(BFloat16, float)
```

### Why not (a) or (c)?

- **(a)** Fix only BFloat16: leaves MLFloat16 inconsistent with BFloat16 on adjacent lines,
  which a reviewer will question. Also leaves the MLFloat16 mismatch as a known-wrong registration.
- **(c)** Document and skip: this is a one-line fix with zero risk (the compute path already uses
  `U=float`). The CUDA EP already does it correctly. Not fixing it means deliberately leaving the
  CPU registration inconsistent with both the schema and CUDA.

### Duplication nits (N2/N3)

`NarrowToFloat`/`FloatToNarrow` are duplicated between `layer_norm_impl.cc` and
`skip_layer_norm.cc`. The `BFloat16 ComputeJob` and `BFloat16Math` parallel the MLFloat16
versions. Deduplication would require a shared header and touching multiple files — scope creep
for a registration-only PR. The duplicated code is short, obviously correct, and well-contained.
Recommend deferring to a follow-up cleanup PR.

## Validation

Build and all 10 `LayerNormBFloat16*` tests pass after the change.
