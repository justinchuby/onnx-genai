# Review: PR #31974 — Register BFloat16 LayerNorm/RMSNorm kernels on the CPU EP

**Reviewer:** Luv (Code Reviewer)
**Date:** 2026-08-11
**Verdict:** CONDITIONAL APPROVE — one substantive issue, no blockers

---

## BLOCKING findings

None.

## SUBSTANTIVE findings

### S1. Contrib `LayerNormalization`/`SimplifiedLayerNormalization` register `U=BFloat16` but schema constrains `U` to `{float, double}`

**Files:** `contrib_ops/cpu/layer_norm.cc:29` (macro `REGISTER_CONTRIB_KERNELS(BFloat16)`)

The `REGISTER_CONTRIB_KERNELS` macro sets `.TypeConstraint("U", DataTypeImpl::GetTensorType<T>())` — so for `T=BFloat16`, the kernel declares it handles `U=BFloat16`. However, the contrib schema (`contrib_defs.cc:3180,3327`) constrains `U` to `{"tensor(float)", "tensor(double)"}` only. BFloat16 is **not** in the `U` constraint.

**Mitigating context:** This is a **pre-existing issue** — the existing `MLFloat16` registration has the same mismatch (`U=MLFloat16` vs schema `U={float,double}`). The `U` outputs (Mean, InvStdDev) are optional and rarely requested in inference. The `stash_type` attribute defaults to `float`. In practice, models don't request `U=bf16` outputs for these ops.

**Risk:** Low — but the kernel registration is technically illegal per the schema. A future validator could reject it. Consider either:
- Fixing the macro to use `U=float` for narrow-float types (correct fix), or
- Documenting the mismatch as pre-existing and tracking it separately.

**This is not a blocker** because it follows the established pattern and has no runtime impact.

### S2. Duplicated `NarrowToFloat`/`FloatToNarrow` helpers

**Files:** `layer_norm_impl.cc:30-50`, `skip_layer_norm.cc:118-148`

Two identical pairs of `NarrowToFloat<T>`/`FloatToNarrow<T>` template functions are defined in anonymous namespaces in two files. If a third op needs bf16 support, this becomes three copies.

**Suggestion:** Extract to a shared header (e.g., `narrow_float_utils.h`). Not blocking.

## NITS

### N1. Comment says `ORT_DISABLE_ALL` but code relies on `TransformerLevel::Default`

**File:** `test/contrib_ops/layer_norm_bf16_cpu_test.cc:12`

The header comment says "sets graph_optimization_level = ORT_DISABLE_ALL" but `ConfigEp` + `RunWithConfig` relies on `BaseTester` defaulting to `TransformerLevel::Default` (which is indeed off). The comment is misleading — the test doesn't explicitly set `ORT_DISABLE_ALL`.

### N2. BFloat16 `ComputeJob` is a full copy of the MLFloat16 specialization

**File:** `layer_norm_impl.cc:227-296`

The BFloat16 `ComputeJob` is a verbatim copy of the MLFloat16 version with types substituted. Both widen to f32, accumulate, and narrow back. This could be unified into a single template for `is_narrow_float_v<T>` types, reducing ~70 lines of duplication.

### N3. BFloat16Math is a near-clone of HalfMath

**File:** `layer_norm_impl.cc:370-404`

Same pattern — could be templatized.

## Independently verified

1. **Schema legality of T=BFloat16:** Verified all four op families:
   - Contrib `LayerNormalization` (1-16): T ✅ (`contrib_defs.cc:3176`), V ✅ (`3184`), **U ⚠️ pre-existing mismatch** (`3180`)
   - Contrib `SimplifiedLayerNormalization`: T ✅ (`3323`), V ✅ (`3331`), **U ⚠️ same** (`3327`)
   - ONNX `LayerNormalization` (17): T ✅, U=float ✅ (kernel correctly constrains U=float)
   - `SkipLayerNormalization` / `SkipSimplifiedLayerNormalization`: T ✅ (`bert_defs.cc:2178,2228`), U=float ✅ (schema U={float})

2. **No fp16-specific overflow/clamping:** Confirmed no `65504`, `HALF_MAX`, or fp16-range clamping in the code paths. The `NarrowToFloat`/`FloatToNarrow` dispatch uses `BFloat16ToFloat`/`FloatToBFloat16` which are independent of the fp16 path.

3. **Rounding is round-to-nearest-even:** Verified `BFloat16(float)` constructor (`float16.h:105-150`) uses `kRoundToNearest` with `(upper_bits & 1)` tie-breaking — this is standard RNE. Every narrowing path goes through `BFloat16(float)` (via `FloatToBFloat16` loop at `float16.h:291`).

4. **Prepacking is correct:** `ConvertMLFloat16ToFloatIfNeeded` (both in `layer_norm_impl.cc` and `skip_layer_norm.cc`) now handles `BFloat16` via `BFloat16ToFloat`. Prepacked data is stored as f32. The prepacked buffers are consumed by the same f32-accumulation compute path. Type-safe.

5. **Anti-fallback testing is sound:** `ConfigEp(DefaultCpuExecutionProvider())` provides only the CPU EP. `BaseTester::RunWithConfig` uses `TransformerLevel::Default` (no graph optimizations, no Cast insertion). With a single EP and no optimizations, if no bf16 kernel exists, `InferenceSession::Initialize` fails with "no kernel found" — cannot silently pass via Cast fallback.

6. **Tolerance:** 0.016 (2 bf16 ULP at unit scale) — appropriate for bf16 (7-bit mantissa, 1 ULP ≈ 0.0078).

## Taken on trust

- The 45 MLAS kernel tests and 10 operator tests claimed passing — not independently built/run (build infrastructure not available in this review environment).
- PR text's claim about CUDA EP already having bf16 registration (`cuda_contrib_kernels.cc:178`) — not verified.

---

# Re-Review: S1 fix commit `142cb563c5` (Register U=float for narrow-float contrib kernels)

**Reviewer:** Luv (Code Reviewer)
**Date:** 2026-08-11
**Verdict:** APPROVE — no blockers, no new substantive findings. S1 is fully resolved.

## Change summary

Commit `142cb563c5` modifies `contrib_ops/cpu/layer_norm.cc`:
- Macro `REGISTER_CONTRIB_KERNELS(T)` → `REGISTER_CONTRIB_KERNELS(T, U)`, with `U` controlling the `.TypeConstraint("U", ...)` registration.
- `float,float` and `double,double` retain `U=T` — no change.
- `MLFloat16,float` and `BFloat16,float` now register `U=float` instead of `U=MLFloat16`/`U=BFloat16`.

## Scrutiny results

### 1. Is the "declaration-only, no runtime change" claim true? — **YES**

Traced the full path:
- Contrib `LayerNorm` constructor (`layer_norm.h:14-15`): calls `LayerNormImpl(op_kernel_info, simplified)` — **two args**, so `contrib_op` defaults to `false`.
- `LayerNormImpl::Compute` (`layer_norm_impl.cc:697-703`): dispatches via `SrcDispatcher`.
- `SrcDispatcher` (`layer_norm_impl.h:44-62`): checks `contrib_op`. When `contrib_op=false`, it calls `ComputeImpl<T, float>` regardless of the declared `U`.
- Therefore for the contrib `LayerNorm<false/true>` kernel, `U` in the kernel def is **declaration-only** — it affects kernel matching but not runtime computation. `Mean`/`InvStdDev` outputs were always emitted as `float`.

**Conclusion: no runtime behaviour change for any type, including MLFloat16.** The claim is accurate.

### 2. Does the kernel registry key change? — **YES, but no breakage risk**

The `U` TypeConstraint participates in kernel matching. For a model with opset 1-16 `LayerNormalization` where the graph has `T=MLFloat16`:
- **Before:** kernel declared `U=MLFloat16`. If a model's `U` output edge was typed `float` (the schema-correct type), the kernel's `U=MLFloat16` constraint would *not match* — but this was already the case, so no regression.
- **After:** kernel declares `U=float`. If a model's `U` output edge is typed `float`, the kernel now *correctly matches*.
- If a model had `U=MLFloat16` edges (schema-violating), it previously matched and now won't. But such a model was already schema-violating, so this is a correctness improvement, not a regression.

**Net effect: existing valid models that worked before will continue to work. Models with U outputs typed as float will now match correctly (an improvement). Schema-violating models may stop matching, which is the correct behaviour.**

### 3. Is the CUDA parity claim accurate? — **YES**

Verified in `contrib_ops/cuda/layer_norm.cc:30-35`:
```
REGISTER_KERNEL_TYPED(float, float, float)
REGISTER_KERNEL_TYPED(double, double, double)
REGISTER_KERNEL_TYPED(MLFloat16, float, MLFloat16)
REGISTER_KERNEL_TYPED(float, float, MLFloat16)
REGISTER_KERNEL_TYPED(MLFloat16, float, float)
REGISTER_KERNEL_TYPED(BFloat16, float, BFloat16)
```
CUDA registers `U=float` for all narrow-float types. The CPU contrib fix now matches.

### 4. Did the macro change miss or over-apply? — **NO**

All four expansions verified:
- `REGISTER_CONTRIB_KERNELS(float, float)` → U=float ✅ (same as before)
- `REGISTER_CONTRIB_KERNELS(double, double)` → U=double ✅ (same as before)
- `REGISTER_CONTRIB_KERNELS(MLFloat16, float)` → U=float ✅ (fixed)
- `REGISTER_CONTRIB_KERNELS(BFloat16, float)` → U=float ✅ (fixed)

The `SkipLayerNormalization`/`SkipSimplifiedLayerNormalization` macros (`skip_layer_norm.cc:17-35`) have no `U` constraint — unaffected. The ONNX-domain opset-17 kernel (`core/providers/cpu/nn/layer_norm.cc`) already hardcodes `U=float` — unaffected.

### 5. Scope recommendation: **KEEP COMBINED**

Reasons:
- The bf16 registration *introduced* a schema-violating `U=BFloat16` that this commit fixes.
- Fixing MLFloat16 in the same commit prevents inconsistency between two adjacent lines of the same macro.
- The fix is 10 lines in one file with zero runtime impact — splitting it would add review overhead with no safety benefit.
- An ORT maintainer who wants it separated can cherry-pick the commit trivially, but I recommend against it.

### 6. Prior findings re-confirmed

- **4 op registrations** (LayerNorm, SimplifiedLayerNorm, SkipLayerNorm, SkipSimplifiedLayerNorm): all bf16 registrations still present and correct.
- **Prepacking:** `ConvertMLFloat16ToFloatIfNeeded` still handles BFloat16.
- **Rounding:** BFloat16 constructor still uses RNE.
- **Anti-fallback property:** verified by running tests — 10/10 pass.
- **Tests:** `./onnxruntime_provider_test --gtest_filter="LayerNormBFloat16*"` — 10 tests passed independently.

## Verified vs taken on trust

**Verified directly:**
- SrcDispatcher dispatch path (contrib_op=false → U=float always)
- CUDA contrib registrations (U=float for narrow types)
- All 4 macro expansions
- 10 bf16 tests pass
- No MLFloat16-specific tests exist in this test suite to exercise the changed registration

**Taken on trust:**
- No existing MLFloat16 LayerNorm tests break (full test suite not run — only bf16 filter). **Recommendation:** PR CI should confirm the full `LayerNormalization` test suite passes for MLFloat16.
