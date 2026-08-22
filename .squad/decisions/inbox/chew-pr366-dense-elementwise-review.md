# Chew — PR #366 Dense Elementwise SIMD Review

**Date:** 2026-07-28T23:55:00Z  
**PR:** #366 (`squad/dense-elementwise-simd`)  
**Author:** Iran  
**Verdict:** 🔴 **REJECT**  
**Revision agent:** Pris  

---

## Performance Regression Assessment (Primary Ask)

**No regression.** Measured under bench lock (load 1.78–2.12):

| Metric | Main | PR | Delta |
|---|---|---|---|
| Guard overhead (`is_contiguous` vs `stride_match+is_dense`) | 28.6 ns | 28.1 ns | −2% (noise) |
| Relu f32 802K elements throughput (3 runs) | 68.9–69.3 µs | 68.6–70.7 µs | ±2% (noise) |
| Effective bandwidth | ~93 GB/s | ~93 GB/s | identical |

The contiguous common-case code path is functionally identical: one extra slice
comparison (`strides != strides`), `is_dense()` costs the same as `is_contiguous()`
(both allocate a Vec), and the NEON inner loop is byte-for-byte the same.

---

## Blocking Findings

### B1: `is_dense` accepts negative strides — **soundness bug**

`is_dense()` at `crates/onnx-runtime-ir/src/layout.rs:44` uses `s.unsigned_abs() as i64`,
stripping the sign from strides. A tensor with `shape=[5], strides=[-1]` (produced by
`Slice` with `step=-1`, confirmed in test `view_output_negative_step_is_negative_stride`
at `slice.rs:557`) passes `is_dense` → returns `true`.

Subsequently, `dispatch_dense_f32` calls:
```rust
let src = std::slice::from_raw_parts(input.data_ptr::<f32>(), numel);
```

With `byte_offset=16` (pointing to the last element) and `numel=5`, this reads 20 bytes
**forward** from `data_ptr`, but elements are at **negative** offsets. This is UB —
out-of-bounds read.

**Fix:** Either:
- `is_dense` must reject negative strides: `if strides.iter().any(|&s| s < 0) { return false; }`
- Or the dispatch entry must guard: `if input.strides.iter().any(|&s| s < 0) { return Ok(false); }`

The simpler fix is in `is_dense` itself, since a negative-stride tensor is not
"dense" in the sense this module requires (linear forward scan of `data_ptr..data_ptr+numel`).

### B2: `DENSE_ELEM_NON_DENSE_FALLBACK_HITS` has no manifest row

Per the dispatch-manifest inverse rule: every counter needs a manifest row and a test
proving it fires. The fallback counter exists and is wired but has no `[[claim]]` entry
in `dispatch_manifest.toml`. This is a merge-blocker per standing directives.

---

## Non-Blocking Findings (Verified Sound)

### NB1: Broadcast correctly rejected by `is_dense`

Zero stride with dim > 1 produces `abs_stride = 0`, which sorts first, fails
`pairs[0].0 != 1`. Binary ops in `elementwise.rs` additionally require
`inputs[1].shape == output.shape` — broadcast operands (different shape) never enter.

### NB2: vcvt baseline claim — VERIFIED on device

`vcvt_f32_f16` / `vcvt_f16_f32` compile and execute correctly on M1 Max without any
`target_feature = "fp16"` annotation. These map to `FCVTL`/`FCVTN` (base AdvSIMD),
not the optional FEAT_FP16 arithmetic extension. Verified NaN, subnormal, and infinity
preservation through conversion.

### NB3: f16 double-rounding — NOT an issue

Relu/Clip are selection operations (max/min). Widen f16→f32 is exact (f16 ⊂ f32),
operation selects between input value and constant, narrow is single-round. No
double-rounding.

### NB4: NaN propagation — VERIFIED by direct experiment

- `vmaxq_f32(NaN, 0)` → NaN ✓ (both operand positions)
- `vminq_f32(NaN, 0)` → NaN ✓
- f16 NaN preserved through full widen→relu→narrow path ✓
- `FMAX(+0, -0) = +0` (accepted divergence, matches ONNX/ORT) ✓
- No `f32::max` / `f32::min` anywhere in the module ✓
- Scalar uses PartialOrd (NaN compares false → passes through) ✓

### NB5: "Latent only" claim is accurate

The dense-but-permuted path (new acceptance region) does not fire in current models
because all tensors at Relu/Clip nodes are already contiguous. The f16/bf16 paths
similarly don't fire because current CPU EP models use f32. The existing f32 contiguous
path fires today (same as before, just with `is_dense` as guard).

---

## Required Changes for Revision (assigned to Pris)

1. Fix `is_dense` to reject negative strides (single line guard at function entry).
2. Add a test proving negative-stride input does NOT enter the dense path.
3. Add a `[[claim]]` manifest row for `DENSE_ELEM_NON_DENSE_FALLBACK_HITS` with a test
   proving it fires (e.g. on a strided non-dense tensor).
4. Consider whether `try_dense_elementwise` should also have a guard at entry rejecting
   negative strides (defense-in-depth), independent of `is_dense`.
