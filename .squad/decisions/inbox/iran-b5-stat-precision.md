# B5: Stats round-tripped through BFloat16 though U is float

**Date:** 2026-08-11
**Author:** Iran (CPU/numerics)
**PR:** microsoft/onnxruntime#31974

## Problem

The BFloat16 and MLFloat16 `ComputeJob` overloads in `layer_norm_impl.cc` wrote
`Mean`/`InvStdDev` statistics by narrowing the f32 values through `BFloat16(mean)`
or `MLFloat16(mean)`, even though the output type `U` is always `float`. This
silently degraded statistics to ~3-digit (bf16) or ~3.5-digit (fp16) precision,
contradicting the schema promise that these outputs are float.

## Fix

1. Both narrow-float `ComputeJob` overloads now call `WriteStat<U>(...)` instead
   of hardcoding the narrow type. Since `U = float`, stats are written at full
   f32 precision.

2. The `WriteStat` template previously had dead `if constexpr` branches for
   `MLFloat16` and `BFloat16`. These were reachable only if the compiler
   instantiated `ComputeImpl<T, T>` for narrow types — which happened because
   `SrcDispatcher` used a runtime `if (contrib_op)` that forced both branches
   to compile. Changed to `if constexpr` so the `ComputeImpl<T, T>` path is
   never instantiated for narrow types. The dead branches were then removed.

3. `NarrowToFloat`/`FloatToNarrow` comments updated in both `layer_norm_impl.cc`
   and `skip_layer_norm.cc` to honestly describe the conversions as portable
   scalar widen/narrow — no hardware bf16 instructions are used on AVX2.

## N1 decision: MLFloat16 U=float registration

**Decision: Keep.** The commit `142cb563c5` already changed the contrib macro
from `(T)` to `(T, U)` and registered `MLFloat16` with `U=float`. This is:
- Correct per the contrib schema (U is constrained to float)
- Consistent with the CUDA contrib kernels (which already register U=float)
- Declaration-only at runtime (SrcDispatcher always uses `ComputeImpl<T, float>`)
- Adjacent to the BFloat16 registration, so consistency beats splitting

Splitting it out would leave two adjacent registrations with different U
constraints, which is the exact inconsistency this change fixed.

## Deduplication: NarrowToFloat / FloatToNarrow

**Not deduplicated.** These ~10-line helpers are duplicated between
`layer_norm_impl.cc` (anonymous namespace) and `skip_layer_norm.cc` (anonymous
namespace). Deduplicating requires a shared header, which is scope creep for a
bug-fix. The code is short, identical, and obviously correct. Filed as
follow-up.

## Test results

Build: clean with `-Werror` (no `--compile_no_warning_as_error`).

```
[==========] Running 17 tests from 1 test suite.  (LayerNormBFloat16*)
[  PASSED  ] 17 tests.

[==========] Running 96 tests from 8 test suites.  (*LayerNorm*)
[  PASSED  ] 96 tests.
```

All 17 BFloat16 tests, 23 fp16 tests, and the full 96-test LayerNorm suite pass.
