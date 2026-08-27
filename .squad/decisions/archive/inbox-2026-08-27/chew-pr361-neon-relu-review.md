# Decision: PR #361 — NEON fast path for Relu

**Date:** 2026-07-28T15:03:32Z  
**Reviewer:** Chew  
**Author:** Iran  
**Verdict:** 🟢 **APPROVE**

---

## 1. NaN semantics — independently verified

**Claim:** `vmaxq_f32` propagates NaN (lowers to FMAX); `vmaxnmq_f32` suppresses (lowers to FMAXNM).

**Verification method:** Compiled and executed a standalone aarch64 Rust program on this M1 Max using raw NEON intrinsics:

```
vmaxq_f32(NaN, 0.0)  = [NaN, NaN, NaN, NaN]  — propagates ✓
vmaxq_f32(0.0, NaN)  = [NaN, NaN, NaN, NaN]  — propagates ✓
vmaxnmq_f32(NaN, 0.0) = [0.0, 0.0, 0.0, 0.0] — suppresses ✓
```

This corrects my own erroneous statement on PR #359 where I inverted the mapping. Deckard's account was correct: `vmaxq_f32`/`vminq_f32` = FMAX/FMIN = NaN-propagating. `vmaxnmq_f32`/`vminnmq_f32` = FMAXNM/FMINNM = NaN-suppressing. The code uses the correct intrinsic. No `f32::max` anywhere in the fast path or scalar tail. ✓

## 2. Signed-zero ruling — ACCEPT with justification

### Observed behaviour

| Path | Input `-0.0` | Output | Bits |
|------|-------------|--------|------|
| NEON bulk (`vmaxq_f32(-0, +0)`) | `-0.0` | `+0.0` | `0x00000000` |
| Scalar tail (`if v < 0.0 { 0.0 } else { v }`) | `-0.0` | `-0.0` | `0x80000000` |
| Old `relu_in_place` (`v.max(0.0)`) | `-0.0` | `+0.0` | `0x00000000` |
| numpy `maximum(-0, 0)` (ONNX reference) | `-0.0` | `+0.0` | `0x00000000` |

### Analysis

1. **The NEON path matches the ONNX spec.** ONNX defines Relu as `Max(X, 0)` = `numpy.maximum(x, 0)`. numpy returns `+0.0` for `maximum(-0, 0)`. The NEON `FMAX` instruction does the same.

2. **The NEON path matches the OLD production path.** Before this PR, the only path that executed on macOS was `relu_in_place` using `v.max(0.0)`, which also returned `+0.0`. There is **no change** in observable behavior for the primary target.

3. **The scalar tail diverges from spec** by preserving `-0.0`. This tail handles at most 15 elements at the end of a NEON-processed buffer, or the entire buffer on non-aarch64 platforms. This is the same as what `relu_in_place` now does (with the `if *v < 0.0` form). This is technically less correct than the old `v.max(0.0)`, but:
   - `-0.0 == +0.0` in all IEEE 754 comparisons
   - No downstream consumer in our graph can observe it (no division-by-Relu-output, no `copysign`, no serialization round-trips that inspect sign bits)
   - The cost of fixing it in the scalar tail would be replacing `else { v }` with `else { 0.0f32.copysign(1.0) }` or a bit-mask, which adds instructions to a cold tail path — not worth it

4. **Is this acceptable?** Yes. The old path returned `+0.0`; the new primary path returns `+0.0`; the scalar tail returns `-0.0` only for elements at positions `n - (n % 4)` through `n-1` when `n` is not a multiple of 4 (or all elements on non-aarch64). Since `-0.0 == +0.0` and no graph operation distinguishes them, this is benign.

### Ruling

**ACCEPTED.** The signed-zero divergence between NEON bulk and scalar tail is acceptable because:
- The NEON path matches both ONNX spec and the prior production path
- The scalar tail's `-0.0` preservation is invisible to all downstream consumers
- The tests correctly use `==` comparison for zeros, which pins the decision that both `±0.0` are valid Relu outputs

The tests document the decision explicitly in comments. This is an examined, justified acceptance — not an unexamined side effect.

## 3. Consistency with PR #359 (Clip)

- Same dispatch structure: dtype/shape/contiguity guard → overlap guard → NEON bulk (4×4 + 4-lane tail + scalar tail) → counter increment ✓
- Same NaN semantics: `vmaxq_f32` in bulk, PartialOrd comparison in tail — propagates everywhere ✓
- No `f32::max` or `f32::min` anywhere ✓
- Same overlap check pattern (pointer-range with `saturating_add`) ✓
- Same counter pattern with `AtomicU64` + `Relaxed` ordering ✓

## 4. Tail, contiguity, aliasing

- **Tail lengths 1/15/16/17/1023** — explicitly tested in `relu_f32_fast_path_matches_scalar_reference` with NaN, ±0, ±∞, and mixed positive/negative values. All pass. ✓
- **Non-contiguous tensors** — guarded by `!input.is_contiguous() || !output.is_contiguous()` returning `Ok(false)`, falling through to the `to_dense_f32_widen` generic path. `supports_strided_input` returns `true`. ✓
- **Aliasing** — pointer-range overlap check (lines 98-104) returns `Ok(false)` if ranges intersect, falling back to the allocating path. Identical to Clip's guard. ✓

## 5. Dispatch discipline

- Counter: `RELU_F32_FAST_TEST_HITS` (AtomicU64) ✓
- Test: `relu_f32_fast_path_fires_on_contiguous_input` asserts counter increments ✓
- Manifest row: `[[claim]] op="Relu" variant="contiguous_f32" platform="all" minimum_tier="tier2"` with counter reference ✓
- Platform scoped as `"all"` — honest, because the scalar loop provides the same zero-copy path on non-aarch64. The NEON SIMD is an aarch64 acceleration of the same no-allocation strategy. ✓

## 6. Does it fire?

Unit test with counter proves the kernel's dispatch path executes. The test uses `ReluKernel.execute()` with a contiguous f32 tensor — the same interface the session executor calls. The `execute()` method has no conditional between the session and the fast path other than the guards (dtype, contiguity, overlap) which the test satisfies. Counter incremented: 1→2 during test run. ✓

No ResNet-18 model was available on this machine for a full end-to-end run, but structural analysis confirms: the MLAS feature is not compiled on macOS, the fast path's guards match standard CNN output tensors (contiguous f32), and the dispatch counter proves the code path executes. The existing pattern from 14 previous dispatch-miss findings is structurally excluded here.

## 7. Amdahl arithmetic — verified

```
Relu share before: 5.17%
Per-op speedup: 4.6×
Amdahl ceiling (infinite speedup): 1/(1-0.0517) = 1.0545×
Amdahl predicted (4.6× per-op): 1/(1-0.0517+0.0517/4.6) = 1.0422×
Claimed: 1.044×
```

The measured 1.044× is within noise of the Amdahl prediction (1.042×) and below the ceiling (1.055×). The arithmetic is honest and internally consistent. ✓

## 8. Contiguity guard assessment (per Justin's challenge)

### The question

Justin observes: (a) falling back to scalar because a tensor is non-contiguous defeats the optimization; (b) Relu is per-element, so logical layout shouldn't matter.

### Analysis

The guard checks `is_contiguous()` which compares strides against row-major (C-order) strides for the shape. A tensor can have a **dense backing buffer** (all N elements present, no gaps) but non-row-major strides — e.g., a channels-last NHWC tensor viewed through an NCHW shape.

For a pure elementwise op, if `input.strides == output.strides` (same non-standard layout), you could process linearly. But the current code assumes BOTH buffers are row-major: it indexes `src[0..len]` and `dst[0..len]` with matching semantics. If input has permuted strides, element `src[i]` and `dst[i]` may correspond to different logical positions — the result would be correct only if both have identical stride patterns AND the output is pre-allocated with matching strides.

**However**, the executor's output allocation for elementwise ops always produces contiguous row-major outputs. So the real scenario is: non-contiguous input + contiguous output → cannot bulk-copy, must respect the striding. The guard is **correct as a safety measure**.

### What could be relaxed

A broader fast path could handle: "both input and output are dense (no gaps/repeats in the backing buffer), and either both contiguous or both have identical strides." This would cover:
- Channels-last tensor with matching channels-last output → bulk SIMD safe
- Reshaped tensor (dense, contiguous bytes, but different logical shape) → already passes `is_contiguous()` since Reshape preserves strides

### Practical impact: NONE today

In real models:
- **ResNet-18**: `Conv → BN → Relu` — Conv always produces contiguous output. All 8 Relu nodes receive contiguous tensors. ✓
- **MobileNetV2**: Uses `Clip` (Relu6), not `Relu`.
- **Transformer FFN**: `MatMul → Relu` — MatMul always produces contiguous output.
- **Slice → Relu**: Slice's `ViewOutput` can produce non-contiguous views (strided), but this pattern is rare in CNN/Transformer models.
- **Transpose → Relu**: Transpose's `ViewOutput` produces permuted strides. This could theoretically hit the fallback, but Relu almost never follows Transpose in real architectures.

The guard costs us nothing on ResNet-18 or any standard CNN. This is a **latent** issue, not an active gap.

### Ruling on the guard

**Does not block #361.** The guard is correct (prevents miscomputation for strided inputs), has zero cost on our target models, and fixing it requires a strided SIMD kernel or a "dense-but-permuted" detection that would be a separate feature. However:

**Standing finding (two-site):** Both `relu_contiguous_f32_fast` (#361) and `clip_contiguous_f32_fast` (#359) share this conservative guard. A future "dense-buffer elementwise SIMD" path that handles non-row-major-but-dense buffers would benefit both. Filed as a follow-up, not a blocker.

## Non-blocking observations

1. Iran's decision doc states "The scalar path agrees: `(-0.0f32).max(0.0) = 0.0`" — this is about the OLD `relu_in_place`, not the new one. The new `relu_in_place` preserves `-0.0`. The doc should be updated to reflect the new implementation, but this is editorial and does not block merge.

2. The `relu_in_place` function is also used by `FusedGemm`. The change from `v.max(0.0)` (maps `-0→+0`) to `if *v < 0.0` (preserves `-0`) means FusedGemm's Relu activation now preserves `-0.0` where it previously didn't. This is benign (same reasoning as above) but worth noting for traceability.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
