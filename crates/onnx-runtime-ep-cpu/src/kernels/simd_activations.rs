//! Vectorised transcendental primitives for the *approximate* activation
//! family (`Tanh`, `Sigmoid`, `FastGelu`, `QuickGelu`).
//!
//! # Why this exists
//!
//! The scalar kernels evaluate one `libm` transcendental per element. On this
//! class of hardware a dependent `tanhf` is ~13 ns and `f64::tanh` ~25 ns per
//! element, which is roughly two orders of magnitude above what ONNX Runtime
//! achieves for the same op. ORT's advantage is not algorithmic: MLAS ships
//! hand-written FMA3 kernels (`lib/x86_64/{Tanh,Logistic,Erf}KernelFma3.S`)
//! that evaluate a *rational polynomial* over a clamped range, eight lanes at
//! a time, with no branches and no libm call.
//!
//! This module reproduces those two rational approximations in safe-ish Rust
//! over `core::arch::x86_64` intrinsics, so the default build (which does
//! **not** enable the `mlas` feature) gets the same throughput.
//!
//! # Numerical contract
//!
//! * The polynomials and their evaluation order are taken verbatim from
//!   `MlasTanhConstants` / `MlasLogisticConstants` (which MLAS in turn took
//!   from Eigen), so a build using this path tracks ORT's own CPU output.
//! * MLAS clamps its input to the polynomial's valid band and returns the
//!   polynomial value at the clamp point; this module *saturates* instead,
//!   substituting the exact limit outside the band. On this host the two
//!   agree bit-for-bit everywhere they were probed, `±Inf` included: the
//!   vendored `MlasComputeLogistic` returns exactly `0.0` for `sigmoid(-Inf)`
//!   and for every `x <= -18`. That is not an underflow: at the clamp point
//!   the rational evaluates to `-5.96e-8`, and `logistic.cpp`'s own *output*
//!   clamp (`std::clamp((p / q) + 0.5f, 0.0f, 1.0f)`) pins that negative
//!   value to `0.0`. Saturation is therefore an *equivalent* formulation that
//!   makes the endpoints exact by construction rather than by that accident —
//!   it is not a correctness win, and earlier revisions of this comment
//!   wrongly claimed clamping leaked `1.5e-8` at `-Inf`. It is still the
//!   formulation worth keeping: it is exact for any future constant set, and
//!   it costs nothing measurable. Note that saturation is *not* correctly
//!   rounded at the very edge of the `tanh` band: `sigmoid(18) = 1 - 1.523e-8`
//!   does round to `1.0f32`, but `tanh(9) = 1 - 3.046e-8` rounds to
//!   `0.99999994` (`0x3F7FFFFF`) because `3.046e-8` exceeds the `2.98e-8`
//!   half-ulp threshold below `1.0`. `tanh` only rounds to `1.0f32` from
//!   `|x| >= 9.010914` (the first f32 whose `tanh` does), so the substituted
//!   `±1` is one ulp high on `9 < |x| < 9.010914`. MLAS and ORT return `1.0`
//!   there too, the scaled error is `6.62e-9` against the module's asserted
//!   `4e-7` bound, and the alternative in that range is the rational's own
//!   out-of-range overshoot, so this is accepted rather than special-cased.
//! * `NaN` propagates unchanged. Both the clamp (`maxps`/`minps` with the
//!   value as the *second* operand, matching MLAS) and the saturation selects
//!   (ordered compares, which are false for `NaN`) preserve it. Measured
//!   against ORT 1.28.0 CPU, the payload and sign survive exactly
//!   (`0x7FC01234 -> 0x7FC01234`) and a signalling `NaN` is quieted with its
//!   payload kept (`0x7F800001 -> 0x7FC00001`), on both the vector and the
//!   scalar path.
//! * Signed zero is preserved: the numerator is odd, so `p * (-0.0) = -0.0`
//!   and `-0.0 / q = -0.0`.
//! * **Two deliberate divergences from ORT**, both verified by running ORT
//!   1.28.0 on identical inputs. Over 140 probed `(function, special value)`
//!   pairs these are the only two disagreements; in 32 of the remaining pairs
//!   this vector path matches ORT where the exact-libm scalar fallback does
//!   not.
//!     1. `tanh` is pinned to `[-1, 1]` (see the note at the pin site); ORT
//!        returns `1.0000001` at `x = 8.442762`.
//!     2. `tanh_gelu(-Inf)` and `quick_gelu(-Inf)` return `+0.0`, the
//!        mathematical limit; ORT evaluates `-Inf * 0` and returns `NaN`.
//!        ONNX does not specify either. The limit is pinned by
//!        `gelu_special_values`; the scalar fallback and the f64 references
//!        agree only because they carry the same explicit `-Inf` guard, so
//!        that agreement documents intent rather than corroborating it.
//!
//! # ISA dependence
//!
//! Dispatch is by runtime feature detection and only ever *adds* a path:
//! without AVX2+FMA the caller's exact scalar closure is used unchanged, so no
//! existing target regresses. As with ORT, this means results can differ by
//! ~1e-7 relative between an AVX2 host and a non-AVX2 host. Tests therefore
//! assert an error bound against an `f64` reference rather than bit equality.

#![allow(clippy::excessive_precision)]

use crate::dtype::{output_direct_write_eligible, slice_byte_range, write_dense_f32_narrow};
use onnx_runtime_ep_api::{Result, TensorMut};

// ---------------------------------------------------------------------------
// Direct-write plumbing
// ---------------------------------------------------------------------------

/// Apply `f` to `x` and land the result in `out`, writing straight into the
/// output tensor's storage whenever that is sound.
///
/// The obvious spelling — `let mut y = vec![0.0; n]; f(&x, &mut y);
/// write_dense_f32_narrow(op, out, &y)` — makes three passes over the data
/// (zero the scratch, compute into it, copy it out) plus an allocation. At
/// prefill sizes the activation itself is a handful of cycles per element, so
/// those extra passes, not the arithmetic, set the runtime: they cost more than
/// the kernel.
///
/// So when [`output_direct_write_eligible`] says the output is a contiguous,
/// host-visible, correctly-sized `f32` buffer that does *not* alias the input we
/// still have to read, `f` writes into it in place and the scratch disappears.
/// Any other case — f16/bf16/f64 output, a strided view, a device pointer, or
/// an in-place `y = act(y)` node where the ranges do overlap — falls back to the
/// owned buffer, which is exactly the situation `write_dense_f32_narrow` exists
/// to handle. Correctness never depends on which arm runs.
pub(crate) fn write_mapped<F>(op: &str, out: &mut TensorMut, x: &[f32], f: F) -> Result<()>
where
    F: FnOnce(&[f32], &mut [f32]),
{
    write_mapped_reading(op, out, x, &[], f)
}

/// [`write_mapped`] for closures that read a slice *besides* `x`.
///
/// The disjointness check has to cover every buffer the closure still reads
/// once we start writing, not just the primary input. FastGelu's bias is the
/// motivating case: it is a `Cow::Borrowed` view of the bias tensor whenever
/// that tensor is contiguous `f32`, so it is live borrowed storage, and the
/// fused kernel re-reads it for every row. Were it ever handed to us aliasing
/// the output, writing row 0 would corrupt the bias that every later row still
/// depends on — and we would be holding `&mut` and `&` over the same bytes.
/// Callers declare those extra ranges here and the direct-write arm is skipped
/// when any of them overlaps the output.
pub(crate) fn write_mapped_reading<F>(
    op: &str,
    out: &mut TensorMut,
    x: &[f32],
    also_read: &[core::ops::Range<usize>],
    f: F,
) -> Result<()>
where
    F: FnOnce(&[f32], &mut [f32]),
{
    let n = x.len();
    let eligible = if also_read.is_empty() {
        output_direct_write_eligible(out, n, &[slice_byte_range(x)])
    } else {
        let mut reads = Vec::with_capacity(1 + also_read.len());
        reads.push(slice_byte_range(x));
        reads.extend_from_slice(also_read);
        output_direct_write_eligible(out, n, &reads)
    };
    if eligible {
        out.validate()?;
        if n == 0 {
            return Ok(());
        }
        // SAFETY: `output_direct_write_eligible` confirmed a validated,
        // contiguous, host-accessible Float32 tensor holding exactly `n`
        // elements, and that its bytes are disjoint from every slice the
        // closure still reads (`x` plus `also_read`).
        let dst = unsafe { std::slice::from_raw_parts_mut(out.data_ptr_mut::<f32>(), n) };
        f(x, dst);
        return Ok(());
    }
    let mut y = vec![0.0f32; n];
    f(x, &mut y);
    write_dense_f32_narrow(op, out, &y)
}

/// Smallest slice length for which the vector path is worth its dispatch
/// overhead. Below this the scalar loop wins (measured: crossover sits between
/// 8 and 32 elements; 32 is the conservative side of it).
pub(crate) const SIMD_MIN_LEN: usize = 32;

// ---------------------------------------------------------------------------
// MLAS constants
// ---------------------------------------------------------------------------

/// `MlasTanhConstants`, `onnxruntime/core/mlas/lib/tanh.cpp`.
mod tanh_c {
    pub(super) const LOWER: f32 = -9.0;
    pub(super) const UPPER: f32 = 9.0;
    pub(super) const ALPHA_13: f32 = -2.76076847742355e-16;
    pub(super) const ALPHA_11: f32 = 2.00018790482477e-13;
    pub(super) const ALPHA_9: f32 = -8.60467152213735e-11;
    pub(super) const ALPHA_7: f32 = 5.12229709037114e-08;
    pub(super) const ALPHA_5: f32 = 1.48572235717979e-05;
    pub(super) const ALPHA_3: f32 = 6.37261928875436e-04;
    pub(super) const ALPHA_1: f32 = 4.89352455891786e-03;
    pub(super) const BETA_6: f32 = 1.19825839466702e-06;
    pub(super) const BETA_4: f32 = 1.18534705686654e-04;
    pub(super) const BETA_2: f32 = 2.26843463243900e-03;
    pub(super) const BETA_0: f32 = 4.89352518554385e-03;
}

/// `MlasLogisticConstants`, `onnxruntime/core/mlas/lib/logistic.cpp`.
mod logistic_c {
    pub(super) const LOWER: f32 = -18.0;
    pub(super) const UPPER: f32 = 18.0;
    pub(super) const ALPHA_9: f32 = 4.37031012579801e-11;
    pub(super) const ALPHA_7: f32 = 1.15627324459942e-07;
    pub(super) const ALPHA_5: f32 = 6.08574864600143e-05;
    pub(super) const ALPHA_3: f32 = 8.51377133304701e-03;
    pub(super) const ALPHA_1: f32 = 2.48287947061529e-01;
    pub(super) const BETA_10: f32 = 6.10247389755681e-13;
    pub(super) const BETA_8: f32 = 5.76102136993427e-09;
    pub(super) const BETA_6: f32 = 6.29106785017040e-06;
    pub(super) const BETA_4: f32 = 1.70198817374094e-03;
    pub(super) const BETA_2: f32 = 1.16817656904453e-01;
    pub(super) const BETA_0: f32 = 9.93151921023180e-01;
}

/// `√(2/π)` and the cubic coefficient of the tanh GELU approximation, rounded
/// to `f32`. Matches ORT's `contrib_ops/cpu/bert/fast_gelu.cc`.
const GELU_B: f32 = 0.7978845608028654;
const GELU_C: f32 = 0.044715;

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Returns `true` when the AVX2+FMA vector kernels in this module are live.
#[inline]
pub(crate) fn vector_path_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Dispatch helper: run `vector` when AVX2+FMA is present and the slice is
/// long enough to amortise it, otherwise map `scalar` element-wise.
///
/// `input` and `output` must have equal length; the caller guarantees it.
macro_rules! dispatch {
    ($input:expr, $output:expr, $scalar:expr, $vector:expr) => {{
        let input: &[f32] = $input;
        let output: &mut [f32] = $output;
        debug_assert_eq!(input.len(), output.len());
        #[cfg(target_arch = "x86_64")]
        {
            if input.len() >= SIMD_MIN_LEN && vector_path_available() {
                // SAFETY: guarded by the runtime AVX2+FMA detection above.
                unsafe { $vector(input, output) };
                return;
            }
        }
        let scalar = $scalar;
        for (o, &i) in output.iter_mut().zip(input) {
            *o = scalar(i);
        }
    }};
}

/// `y = tanh(x)`.
pub(crate) fn tanh_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, tanh_scalar, tanh_avx2);
}

/// `y = √x`.
///
/// Unlike every other kernel in this module this is **not** an approximation.
/// `vsqrtps` and `sqrtss` are the same correctly-rounded IEEE-754 square root,
/// so the vector body, the scalar tail and the pre-existing `f32::sqrt` kernel
/// this replaces all produce bit-identical results — `-0.0 -> -0.0`,
/// `x < 0 -> NaN`, `+Inf -> +Inf`, subnormals exact. Nothing here trades
/// accuracy for speed; the win is eight lanes per instruction plus the caller
/// no longer materialising an intermediate `Vec` per call.
///
/// Both instructions read the same `MXCSR`, so the equivalence also holds under
/// flush-to-zero / denormals-are-zero: if the host process has set `FTZ`/`DAZ`
/// then subnormals are flushed by the replacement exactly as they were by the
/// code it replaces. "Subnormals exact" above is a statement about the default
/// `MXCSR`, not a guarantee this kernel adds or removes.
pub(crate) fn sqrt_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, f32::sqrt, sqrt_avx2);
}

/// `y = 1 / (1 + e^-x)`.
pub(crate) fn sigmoid_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, sigmoid_scalar, sigmoid_avx2);
}

/// `y = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`, the tanh GELU
/// approximation used by `FastGelu`.
pub(crate) fn tanh_gelu_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, tanh_gelu_scalar, tanh_gelu_avx2);
}

/// `y = x·sigmoid(alpha·x)`, the `QuickGelu` / Swish form.
pub(crate) fn quick_gelu_f32_slice(input: &[f32], output: &mut [f32], alpha: f32) {
    debug_assert_eq!(input.len(), output.len());
    #[cfg(target_arch = "x86_64")]
    {
        if input.len() >= SIMD_MIN_LEN && vector_path_available() {
            // SAFETY: guarded by the runtime AVX2+FMA detection above.
            unsafe { quick_gelu_avx2(input, output, alpha) };
            return;
        }
    }
    for (o, &i) in output.iter_mut().zip(input) {
        *o = quick_gelu_scalar(i, alpha);
    }
}

/// `tanh_gelu(x[i] + bias[i % width])` for every element, without ever
/// materialising `x + bias`.
///
/// FastGelu's bias is a broadcast over the last dimension, so folding it into a
/// scratch row before the transcendental costs a full extra write *and* read of
/// the activation tensor — at prefill sizes that is more traffic than the GELU
/// itself. Adding it in-register instead keeps the whole op at one read of `x`,
/// one write of `y`, and repeated reads of a `width`-element bias row that stays
/// resident in L1.
///
/// `width` must be non-zero and `bias.len()` must equal `width`. A trailing
/// partial row is written too, consuming the matching prefix of the bias, which
/// is what ONNX's `bias[i % width]` broadcast means. Results are bit-identical
/// to folding `x + bias` first and mapping over it, because the in-register add
/// is the same IEEE `f32` addition in the same order.
pub(crate) fn tanh_gelu_bias_f32_slice(
    input: &[f32],
    bias: &[f32],
    width: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(bias.len(), width);
    debug_assert!(width != 0);
    #[cfg(target_arch = "x86_64")]
    {
        // Gated on total length, exactly as `tanh_gelu_f32_slice` is: whether a
        // FastGelu node carries a bias must not change which polynomial its
        // elements go through. `map_bias_ps` handles `width < 8` through its
        // masked tail, so narrow rows stay correct (if unexciting) here.
        if input.len() >= SIMD_MIN_LEN && vector_path_available() {
            // SAFETY: guarded by the runtime AVX2+FMA detection above; the
            // debug asserts above are the caller's contract.
            unsafe { tanh_gelu_bias_avx2(input, bias, width, output) };
            return;
        }
    }
    for (row_in, row_out) in input.chunks(width).zip(output.chunks_mut(width)) {
        for ((o, &v), &b) in row_out.iter_mut().zip(row_in).zip(bias) {
            *o = tanh_gelu_scalar(v + b);
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar reference implementations
//
// These are the *exact* libm forms, not the polynomial. They are what runs
// when AVX2+FMA is unavailable, so a legacy target keeps today's accuracy and
// today's speed rather than inheriting a slow software-`fma` polynomial.
// ---------------------------------------------------------------------------

#[inline]
fn tanh_scalar(x: f32) -> f32 {
    x.tanh()
}

#[inline]
fn sigmoid_scalar(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
fn tanh_gelu_scalar(x: f32) -> f32 {
    if x == f32::NEG_INFINITY {
        return 0.0;
    }
    let xf = x as f64;
    let inner = f64::from(GELU_B) * (xf + f64::from(GELU_C) * xf * xf * xf);
    (0.5 * xf * (1.0 + inner.tanh())) as f32
}

#[inline]
fn quick_gelu_scalar(x: f32, alpha: f32) -> f32 {
    if x == f32::NEG_INFINITY {
        return 0.0;
    }
    x * sigmoid_scalar(alpha * x)
}

// ---------------------------------------------------------------------------
// AVX2 + FMA kernels
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{GELU_B, GELU_C, logistic_c, tanh_c};
    use core::arch::x86_64::*;

    /// `[-1; 7] ++ [0; 8]`. Loading 8 lanes at offset `7 - rem` yields a mask
    /// with exactly `rem` active lanes for `rem` in `1..=7`.
    #[rustfmt::skip]
    static MASK_TABLE: [i32; 15] = [
        -1, -1, -1, -1, -1, -1, -1,
        0, 0, 0, 0, 0, 0, 0, 0,
    ];

    /// Mask selecting the low `rem` lanes. `rem` must be in `1..=7`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn tail_mask(rem: usize) -> __m256i {
        debug_assert!((1..=7).contains(&rem));
        // SAFETY: `7 - rem` is in `0..=6`, so the 8-lane read stays inside the
        // 15-element table.
        unsafe { _mm256_loadu_si256(MASK_TABLE.as_ptr().add(7 - rem).cast()) }
    }

    /// MLAS's NaN-preserving two-step clamp. `maxps`/`minps` return their
    /// *second* operand when either input is NaN, so passing the value second
    /// lets NaN through.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn clamp_nan_preserving(v: __m256, lower: f32, upper: f32) -> __m256 {
        let v = _mm256_max_ps(_mm256_set1_ps(lower), v);
        _mm256_min_ps(_mm256_set1_ps(upper), v)
    }

    /// `tanh` over 8 lanes, following `MlasTanhKernel` but saturating to `±1`
    /// outside `[-9, 9]` instead of returning the polynomial at the clamp
    /// point.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn tanh_ps(x: __m256) -> __m256 {
        unsafe {
            let v = clamp_nan_preserving(x, tanh_c::LOWER, tanh_c::UPPER);
            let v2 = _mm256_mul_ps(v, v);

            let mut p = _mm256_fmadd_ps(
                v2,
                _mm256_set1_ps(tanh_c::ALPHA_13),
                _mm256_set1_ps(tanh_c::ALPHA_11),
            );
            p = _mm256_fmadd_ps(p, v2, _mm256_set1_ps(tanh_c::ALPHA_9));
            p = _mm256_fmadd_ps(p, v2, _mm256_set1_ps(tanh_c::ALPHA_7));
            p = _mm256_fmadd_ps(p, v2, _mm256_set1_ps(tanh_c::ALPHA_5));
            p = _mm256_fmadd_ps(p, v2, _mm256_set1_ps(tanh_c::ALPHA_3));
            p = _mm256_fmadd_ps(p, v2, _mm256_set1_ps(tanh_c::ALPHA_1));
            p = _mm256_mul_ps(p, v);

            let mut q = _mm256_fmadd_ps(
                v2,
                _mm256_set1_ps(tanh_c::BETA_6),
                _mm256_set1_ps(tanh_c::BETA_4),
            );
            q = _mm256_fmadd_ps(q, v2, _mm256_set1_ps(tanh_c::BETA_2));
            q = _mm256_fmadd_ps(q, v2, _mm256_set1_ps(tanh_c::BETA_0));

            // The rational overshoots `tanh`'s mathematical range. Sweeping
            // every f32 in `[8, 9]` through this exact FMA evaluation order,
            // `p/q` exceeds `1.0` for 57 437 of them, spanning
            // `[8.127431, 8.999997]` and peaking at `1.0000002` near
            // `|x| = 8.4755`. ORT ships the overshoot — ORT 1.28.0 CPU
            // `Tanh(8.442762)` returns `1.0000001` — and downstream code is
            // entitled to assume `|tanh| <= 1`, so we pin to `[-1, 1]`. This
            // is a deliberate, measured divergence from ORT in favour of the
            // mathematical range. (The counts are FMA-specific: the same
            // constants evaluated without fusion overshoot on only 26 503
            // points over `[8.052297, 8.999964]`.)
            let poly = clamp_nan_preserving(_mm256_div_ps(p, q), -1.0, 1.0);

            // Saturate. Ordered compares are false for NaN, so NaN keeps the
            // polynomial result, which is NaN.
            let above = _mm256_cmp_ps(x, _mm256_set1_ps(tanh_c::UPPER), _CMP_GT_OQ);
            let below = _mm256_cmp_ps(x, _mm256_set1_ps(tanh_c::LOWER), _CMP_LT_OQ);
            let r = _mm256_blendv_ps(poly, _mm256_set1_ps(1.0), above);
            _mm256_blendv_ps(r, _mm256_set1_ps(-1.0), below)
        }
    }

    /// `sigmoid` over 8 lanes, following `MlasLogisticKernel` but saturating
    /// to `0` / `1` outside `[-18, 18]`.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn sigmoid_ps(x: __m256) -> __m256 {
        unsafe {
            let v = clamp_nan_preserving(x, logistic_c::LOWER, logistic_c::UPPER);
            let v2 = _mm256_mul_ps(v, v);

            let mut p = _mm256_fmadd_ps(
                v2,
                _mm256_set1_ps(logistic_c::ALPHA_9),
                _mm256_set1_ps(logistic_c::ALPHA_7),
            );
            p = _mm256_fmadd_ps(p, v2, _mm256_set1_ps(logistic_c::ALPHA_5));
            p = _mm256_fmadd_ps(p, v2, _mm256_set1_ps(logistic_c::ALPHA_3));
            p = _mm256_fmadd_ps(p, v2, _mm256_set1_ps(logistic_c::ALPHA_1));
            p = _mm256_mul_ps(p, v);

            let mut q = _mm256_fmadd_ps(
                v2,
                _mm256_set1_ps(logistic_c::BETA_10),
                _mm256_set1_ps(logistic_c::BETA_8),
            );
            q = _mm256_fmadd_ps(q, v2, _mm256_set1_ps(logistic_c::BETA_6));
            q = _mm256_fmadd_ps(q, v2, _mm256_set1_ps(logistic_c::BETA_4));
            q = _mm256_fmadd_ps(q, v2, _mm256_set1_ps(logistic_c::BETA_2));
            q = _mm256_fmadd_ps(q, v2, _mm256_set1_ps(logistic_c::BETA_0));

            let poly = _mm256_add_ps(_mm256_div_ps(p, q), _mm256_set1_ps(0.5));
            let poly = clamp_nan_preserving(poly, 0.0, 1.0);

            let above = _mm256_cmp_ps(x, _mm256_set1_ps(logistic_c::UPPER), _CMP_GT_OQ);
            let below = _mm256_cmp_ps(x, _mm256_set1_ps(logistic_c::LOWER), _CMP_LT_OQ);
            let r = _mm256_blendv_ps(poly, _mm256_set1_ps(1.0), above);
            _mm256_blendv_ps(r, _mm256_set1_ps(0.0), below)
        }
    }

    /// `0.5·x·(1 + tanh(B·(x + C·x³)))` over 8 lanes.
    ///
    /// `x = -Inf` is pinned to `0` to match the scalar kernel: the natural
    /// evaluation gives `0.5·(-Inf)·(1 - 1) = NaN`.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn tanh_gelu_ps(x: __m256) -> __m256 {
        unsafe {
            let x2 = _mm256_mul_ps(x, x);
            // B·x + B·C·x³, arranged so a single fma covers the cubic term.
            let inner = _mm256_mul_ps(
                _mm256_set1_ps(GELU_B),
                _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(GELU_C), x2), x, x),
            );
            let t = tanh_ps(inner);
            let y = _mm256_mul_ps(
                _mm256_mul_ps(_mm256_set1_ps(0.5), x),
                _mm256_add_ps(_mm256_set1_ps(1.0), t),
            );
            let neg_inf = _mm256_cmp_ps(x, _mm256_set1_ps(f32::NEG_INFINITY), _CMP_EQ_OQ);
            _mm256_blendv_ps(y, _mm256_setzero_ps(), neg_inf)
        }
    }

    /// `x·sigmoid(alpha·x)` over 8 lanes, with `x = -Inf` pinned to `0`.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn quick_gelu_ps(x: __m256, alpha: __m256) -> __m256 {
        unsafe {
            let s = sigmoid_ps(_mm256_mul_ps(alpha, x));
            let y = _mm256_mul_ps(x, s);
            let neg_inf = _mm256_cmp_ps(x, _mm256_set1_ps(f32::NEG_INFINITY), _CMP_EQ_OQ);
            _mm256_blendv_ps(y, _mm256_setzero_ps(), neg_inf)
        }
    }

    /// Apply an 8-lane kernel across a slice. The `< 8` remainder is processed
    /// through the *same* kernel via a masked load/store, so every element of
    /// the output — tail included — is computed identically.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    /// Like [`map_ps`], but adds a `width`-element bias row to each `width`-element
    /// slab of the input before applying `kernel`.
    pub(super) unsafe fn map_bias_ps(
        input: &[f32],
        bias: &[f32],
        width: usize,
        output: &mut [f32],
        kernel: impl Fn(__m256) -> __m256,
    ) {
        unsafe {
            let bptr = bias.as_ptr();
            // `chunks`, not `chunks_exact`: a tensor that is not a whole number
            // of rows must still get every element written. ONNX broadcasts the
            // bias as `bias[i % width]`, so a short final row consumes the
            // matching prefix of the bias — which is what the shorter row length
            // produces here.
            for (row_in, row_out) in input.chunks(width).zip(output.chunks_mut(width)) {
                let len = row_in.len();
                let body = len & !7;
                let rem = len - body;
                let src = row_in.as_ptr();
                let dst = row_out.as_mut_ptr();
                let mut i = 0;
                while i < body {
                    let v =
                        _mm256_add_ps(_mm256_loadu_ps(src.add(i)), _mm256_loadu_ps(bptr.add(i)));
                    _mm256_storeu_ps(dst.add(i), kernel(v));
                    i += 8;
                }
                if rem != 0 {
                    let mask = tail_mask(rem);
                    let v = _mm256_add_ps(
                        _mm256_maskload_ps(src.add(body), mask),
                        _mm256_maskload_ps(bptr.add(body), mask),
                    );
                    _mm256_maskstore_ps(dst.add(body), mask, kernel(v));
                }
            }
        }
    }

    pub(super) unsafe fn map_ps(
        input: &[f32],
        output: &mut [f32],
        kernel: impl Fn(__m256) -> __m256,
    ) {
        unsafe {
            let n = input.len();
            let src = input.as_ptr();
            let dst = output.as_mut_ptr();
            let body = n & !7;
            let mut i = 0;
            while i < body {
                _mm256_storeu_ps(dst.add(i), kernel(_mm256_loadu_ps(src.add(i))));
                i += 8;
            }
            let rem = n - body;
            if rem != 0 {
                let mask = tail_mask(rem);
                let v = _mm256_maskload_ps(src.add(body), mask);
                _mm256_maskstore_ps(dst.add(body), mask, kernel(v));
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tanh_avx2(input: &[f32], output: &mut [f32]) {
    unsafe { avx2::map_ps(input, output, |v| avx2::tanh_ps(v)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sigmoid_avx2(input: &[f32], output: &mut [f32]) {
    unsafe { avx2::map_ps(input, output, |v| avx2::sigmoid_ps(v)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sqrt_avx2(input: &[f32], output: &mut [f32]) {
    unsafe { avx2::map_ps(input, output, |v| core::arch::x86_64::_mm256_sqrt_ps(v)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tanh_gelu_bias_avx2(input: &[f32], bias: &[f32], width: usize, output: &mut [f32]) {
    unsafe { avx2::map_bias_ps(input, bias, width, output, |v| avx2::tanh_gelu_ps(v)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tanh_gelu_avx2(input: &[f32], output: &mut [f32]) {
    unsafe { avx2::map_ps(input, output, |v| avx2::tanh_gelu_ps(v)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn quick_gelu_avx2(input: &[f32], output: &mut [f32], alpha: f32) {
    unsafe {
        let a = core::arch::x86_64::_mm256_set1_ps(alpha);
        avx2::map_ps(input, output, |v| avx2::quick_gelu_ps(v, a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- f64 references, rounded once to f32 --------------------------------

    fn tanh_ref(x: f32) -> f32 {
        f64::from(x).tanh() as f32
    }

    fn sigmoid_ref(x: f32) -> f32 {
        let x = f64::from(x);
        if x >= 0.0 {
            (1.0 / (1.0 + (-x).exp())) as f32
        } else {
            let e = x.exp();
            (e / (1.0 + e)) as f32
        }
    }

    fn tanh_gelu_ref(x: f32) -> f32 {
        if x == f32::NEG_INFINITY {
            return 0.0;
        }
        let xf = f64::from(x);
        let inner = f64::from(GELU_B) * (xf + f64::from(GELU_C) * xf * xf * xf);
        (0.5 * xf * (1.0 + inner.tanh())) as f32
    }

    fn quick_gelu_ref(x: f32, alpha: f32) -> f32 {
        if x == f32::NEG_INFINITY {
            return 0.0;
        }
        let xf = f64::from(x);
        let z = f64::from(alpha) * xf;
        let s = if z >= 0.0 {
            1.0 / (1.0 + (-z).exp())
        } else {
            let e = z.exp();
            e / (1.0 + e)
        };
        (xf * s) as f32
    }

    /// Error normalised by the documented contract: absolute error scaled by
    /// `max(1, |x|)`. `tanh`/`sigmoid` are bounded in magnitude so the scale is
    /// 1; the GELU forms multiply by `x`, so their error scales with `|x|`.
    fn scaled_err(got: f32, want: f32, x: f32) -> f64 {
        if (got.is_nan() && want.is_nan()) || got == want {
            return 0.0;
        }
        (f64::from(got) - f64::from(want)).abs() / f64::from(x).abs().max(1.0)
    }

    fn grid(lo: f32, hi: f32, n: usize, extra: &[f32]) -> Vec<f32> {
        let mut v: Vec<f32> = (0..n)
            .map(|i| lo + (hi - lo) * (i as f32) / (n as f32 - 1.0))
            .collect();
        v.extend_from_slice(extra);
        v
    }

    fn check(
        values: &[f32],
        got: &[f32],
        reference: impl Fn(f32) -> f32,
        bound: f64,
        name: &str,
    ) -> f64 {
        let mut worst = 0.0f64;
        let mut worst_at = 0.0f32;
        for (&x, &g) in values.iter().zip(got) {
            let e = scaled_err(g, reference(x), x);
            if e > worst {
                worst = e;
                worst_at = x;
            }
        }
        assert!(
            worst <= bound,
            "{name}: worst scaled error {worst:e} at x={worst_at:e} exceeds {bound:e}"
        );
        worst
    }

    // ---- accuracy sweeps ----------------------------------------------------

    /// Documented bound: `|err| <= 4e-7 * max(1, |x|)`.
    const TANH_BOUND: f64 = 4e-7;
    /// Documented bound: `|err| <= 2e-7` (output is in `[0, 1]`).
    const SIGMOID_BOUND: f64 = 2e-7;
    /// Documented bound: `|err| <= 4e-7 * max(1, |x|)`.
    const GELU_BOUND: f64 = 4e-7;

    #[test]
    fn tanh_dense_sweep_matches_f64_reference() {
        let extra = [
            -9.0,
            9.0,
            (-9.0f32).next_down(),
            9.0f32.next_up(),
            (-9.0f32).next_up(),
            9.0f32.next_down(),
            1e-3,
            -1e-3,
        ];
        let x = grid(-14.0, 14.0, 400_003, &extra);
        let mut out = vec![0.0f32; x.len()];
        tanh_f32_slice(&x, &mut out);
        let worst = check(&x, &out, tanh_ref, TANH_BOUND, "tanh");
        eprintln!("tanh worst scaled error: {worst:e}");
    }

    #[test]
    fn sigmoid_dense_sweep_matches_f64_reference() {
        let extra = [
            -18.0,
            18.0,
            (-18.0f32).next_down(),
            18.0f32.next_up(),
            (-18.0f32).next_up(),
            18.0f32.next_down(),
        ];
        let x = grid(-26.0, 26.0, 400_003, &extra);
        let mut out = vec![0.0f32; x.len()];
        sigmoid_f32_slice(&x, &mut out);
        let worst = check(&x, &out, sigmoid_ref, SIGMOID_BOUND, "sigmoid");
        eprintln!("sigmoid worst scaled error: {worst:e}");
    }

    #[test]
    fn tanh_gelu_dense_sweep_matches_f64_reference() {
        let x = grid(-25.0, 25.0, 400_003, &[]);
        let mut out = vec![0.0f32; x.len()];
        tanh_gelu_f32_slice(&x, &mut out);
        let worst = check(&x, &out, tanh_gelu_ref, GELU_BOUND, "tanh_gelu");
        eprintln!("tanh_gelu worst scaled error: {worst:e}");
    }

    #[test]
    fn quick_gelu_dense_sweep_matches_f64_reference() {
        for alpha in [1.0f32, 1.702, 0.5, -1.0, 2.0] {
            let x = grid(-25.0, 25.0, 200_003, &[]);
            let mut out = vec![0.0f32; x.len()];
            quick_gelu_f32_slice(&x, &mut out, alpha);
            // The sigmoid argument is `alpha * x`, so the error scales with
            // `|x|` only after accounting for the extra `alpha` factor.
            let bound = GELU_BOUND * f64::from(alpha.abs()).max(1.0);
            let worst = check(&x, &out, |v| quick_gelu_ref(v, alpha), bound, "quick_gelu");
            eprintln!("quick_gelu(alpha={alpha}) worst scaled error: {worst:e}");
        }
    }

    // ---- special values -----------------------------------------------------

    /// Index of the first special value; the preceding lanes exist purely to
    /// push the slice over `SIMD_MIN_LEN` so the vector path is taken.
    const PAD: usize = SIMD_MIN_LEN;

    fn special_inputs() -> Vec<f32> {
        std::iter::repeat_n(0.0f32, PAD)
            .chain([
                f32::NEG_INFINITY,
                -0.0,
                0.0,
                f32::INFINITY,
                f32::NAN,
                -f32::MAX,
                f32::MAX,
                -1e30,
                1e30,
            ])
            .collect()
    }

    /// `sqrt` is not an approximation: the AVX2 body, the scalar tail and a
    /// plain `f32::sqrt` must agree **bit for bit** on every input, including
    /// the ones where a "fast" reciprocal-sqrt implementation would not. Every
    /// length in `0..=64` is covered so the 8-wide body and the masked tail are
    /// both exercised at every offset, and lengths below `SIMD_MIN_LEN` prove
    /// the scalar dispatch arm agrees too. Negative inputs are interleaved so
    /// the NaN-producing lanes run through the vector body as well, not only
    /// through `sqrt_special_values`.
    #[test]
    fn sqrt_is_bit_identical_to_scalar_at_every_length() {
        let base: Vec<f32> = (0..64)
            .map(|i| {
                // A spread that includes exact squares, non-squares, subnormals
                // and values whose sqrt is not representable.
                let v = match i % 8 {
                    0 => i as f32,
                    1 => 1.0 / (i as f32 + 1.0),
                    2 => (i as f32) * 1e-30,
                    3 => (i as f32) * 1e30,
                    4 => f32::from_bits(i as u32 + 1), // subnormals
                    5 => (i * i) as f32,
                    6 => 2.0f32.powi(i - 32),
                    _ => (i as f32) + 0.5,
                };
                // Every third lane is negative, so the vector body has to
                // produce NaN in arbitrary lane positions.
                if i % 3 == 1 { -v } else { v }
            })
            .collect();
        for len in 0..=64usize {
            let x = &base[..len];
            let mut got = vec![0.0f32; len];
            sqrt_f32_slice(x, &mut got);
            for (i, (&g, &v)) in got.iter().zip(x).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    v.sqrt().to_bits(),
                    "sqrt mismatch at len={len} index={i} input={v:e}"
                );
            }
        }
    }

    #[test]
    fn sqrt_special_values() {
        let mut x = special_inputs();
        // `special_inputs` is signed-symmetric, which is what matters for
        // `sqrt`: every negative input must produce NaN.
        x.push(1.0);
        x.push(4.0);
        let mut o = vec![0.0f32; x.len()];
        sqrt_f32_slice(&x, &mut o);
        assert!(o[PAD].is_nan(), "sqrt(-Inf) is NaN");
        assert_eq!(
            o[PAD + 1].to_bits(),
            (-0.0f32).to_bits(),
            "sqrt(-0) is -0, not NaN"
        );
        assert_eq!(o[PAD + 2].to_bits(), 0.0f32.to_bits(), "sqrt(+0) is +0");
        assert_eq!(o[PAD + 3], f32::INFINITY, "sqrt(+Inf)");
        assert!(o[PAD + 4].is_nan(), "sqrt(NaN)");
        assert!(o[PAD + 5].is_nan(), "sqrt(-MAX) is NaN");
        assert_eq!(o[PAD + 6], f32::MAX.sqrt());
        assert!(o[PAD + 7].is_nan(), "sqrt(-1e30) is NaN");
        assert_eq!(o[PAD + 8], 1e30f32.sqrt());
        assert_eq!(o[x.len() - 2], 1.0);
        assert_eq!(o[x.len() - 1], 2.0);
    }

    /// A NaN input must come back as a NaN with its payload and sign intact —
    /// the same contract the transcendental kernels above hold.
    #[test]
    fn sqrt_preserves_nan_payload() {
        let mut x = vec![1.0f32; PAD];
        x.push(f32::from_bits(0x7FC0_1234));
        x.push(f32::from_bits(0xFFC0_1234));
        let mut o = vec![0.0f32; x.len()];
        sqrt_f32_slice(&x, &mut o);
        assert_eq!(o[PAD].to_bits(), 0x7FC0_1234);
        assert_eq!(o[PAD + 1].to_bits(), 0xFFC0_1234);
    }

    /// `y = sqrt(y)` is a legal graph, and ORT hands us the same buffer for
    /// both. `write_mapped`'s disjointness check has to reject the direct-write
    /// arm there, exactly as it does for `Tanh` — this pins that it does for the
    /// new `Sqrt` caller too, and that the narrowing (non-f32 output) arm agrees.
    #[test]
    fn sqrt_write_mapped_agrees_between_direct_and_aliased_outputs() {
        let n = 257; // not a multiple of 8, so the masked tail runs too
        let src: Vec<f32> = (0..n).map(|i| i as f32 * 0.37).collect();

        let mut expected = vec![0.0f32; n];
        sqrt_f32_slice(&src, &mut expected);

        let mut disjoint = Owned::f32(&[n], &vec![0.0f32; n]);
        write_mapped("Sqrt", &mut disjoint.view_mut(), &src, sqrt_f32_slice).unwrap();
        assert_eq!(disjoint.to_f32(), expected);

        let mut aliased = Owned::f32(&[n], &src);
        {
            let mut view = aliased.view_mut();
            // SAFETY: `view` addresses `n` contiguous f32; the slice is only
            // read while `write_mapped` decides which arm to take, which is
            // exactly the aliasing `output_direct_write_eligible` must detect.
            let borrowed: &[f32] =
                unsafe { std::slice::from_raw_parts(view.data_ptr_mut::<f32>(), n) };
            let borrowed: &[f32] = unsafe { std::mem::transmute(borrowed) };
            write_mapped("Sqrt", &mut view, borrowed, sqrt_f32_slice).unwrap();
        }
        assert_eq!(aliased.to_f32(), expected);

        let mut narrowed = Owned::f16(&[n], &vec![0.0f32; n]);
        write_mapped("Sqrt", &mut narrowed.view_mut(), &src, sqrt_f32_slice).unwrap();
        let got = narrowed.to_u16_bits();
        for (g, e) in got.iter().zip(&expected) {
            assert_eq!(*g, half::f16::from_f32(*e).to_bits());
        }
    }

    #[test]
    fn tanh_special_values() {
        let x = special_inputs();
        let mut o = vec![0.0f32; x.len()];
        tanh_f32_slice(&x, &mut o);
        assert_eq!(o[PAD], -1.0, "tanh(-Inf)");
        assert_eq!(
            o[PAD + 1].to_bits(),
            (-0.0f32).to_bits(),
            "tanh(-0) keeps sign"
        );
        assert_eq!(
            o[PAD + 2].to_bits(),
            0.0f32.to_bits(),
            "tanh(+0) keeps sign"
        );
        assert_eq!(o[PAD + 3], 1.0, "tanh(+Inf)");
        assert!(o[PAD + 4].is_nan(), "tanh(NaN)");
        assert_eq!(o[PAD + 5], -1.0);
        assert_eq!(o[PAD + 6], 1.0);
        assert_eq!(o[PAD + 7], -1.0);
        assert_eq!(o[PAD + 8], 1.0);
    }

    #[test]
    fn sigmoid_special_values() {
        let x = special_inputs();
        let mut o = vec![0.0f32; x.len()];
        sigmoid_f32_slice(&x, &mut o);
        assert_eq!(o[PAD].to_bits(), 0.0f32.to_bits(), "sigmoid(-Inf) = +0");
        assert_eq!(o[PAD + 1], 0.5, "sigmoid(-0)");
        assert_eq!(o[PAD + 2], 0.5, "sigmoid(+0)");
        assert_eq!(o[PAD + 3], 1.0, "sigmoid(+Inf)");
        assert!(o[PAD + 4].is_nan(), "sigmoid(NaN)");
        assert_eq!(o[PAD + 5], 0.0);
        assert_eq!(o[PAD + 6], 1.0);
        assert_eq!(o[PAD + 7], 0.0);
        assert_eq!(o[PAD + 8], 1.0);
    }

    #[test]
    fn gelu_special_values() {
        let x = special_inputs();
        let mut o = vec![0.0f32; x.len()];

        tanh_gelu_f32_slice(&x, &mut o);
        assert_eq!(o[PAD], 0.0, "tanh_gelu(-Inf) is pinned to the limit 0");
        assert_eq!(o[PAD + 1].to_bits(), (-0.0f32).to_bits(), "tanh_gelu(-0)");
        assert_eq!(o[PAD + 2].to_bits(), 0.0f32.to_bits(), "tanh_gelu(+0)");
        assert_eq!(o[PAD + 3], f32::INFINITY, "tanh_gelu(+Inf)");
        assert!(o[PAD + 4].is_nan(), "tanh_gelu(NaN)");
        assert_eq!(o[PAD + 5], 0.0, "tanh_gelu(-MAX)");
        assert_eq!(o[PAD + 6], f32::MAX, "tanh_gelu(MAX)");
        assert_eq!(o[PAD + 7], 0.0);
        assert_eq!(o[PAD + 8], 1e30);

        quick_gelu_f32_slice(&x, &mut o, 1.702);
        assert_eq!(o[PAD], 0.0, "quick_gelu(-Inf)");
        assert_eq!(o[PAD + 1].to_bits(), (-0.0f32).to_bits(), "quick_gelu(-0)");
        assert_eq!(o[PAD + 2].to_bits(), 0.0f32.to_bits(), "quick_gelu(+0)");
        assert_eq!(o[PAD + 3], f32::INFINITY, "quick_gelu(+Inf)");
        assert!(o[PAD + 4].is_nan(), "quick_gelu(NaN)");
        assert_eq!(o[PAD + 5], 0.0, "quick_gelu(-MAX)");
        assert_eq!(o[PAD + 6], f32::MAX);
        assert_eq!(o[PAD + 7], 0.0);
        assert_eq!(o[PAD + 8], 1e30);
    }

    /// A `NaN` anywhere in a vector must not perturb its neighbours: the
    /// polynomial is branch-free, but `blendv` masks and `min`/`max` operand
    /// order both have to be right for this to hold.
    #[test]
    fn nan_does_not_contaminate_neighbouring_lanes() {
        let mut x: Vec<f32> = (0..SIMD_MIN_LEN + 8)
            .map(|i| (i as f32 - 16.0) * 0.5)
            .collect();
        let clean = x.clone();
        for poison in [3usize, 8, SIMD_MIN_LEN, SIMD_MIN_LEN + 5] {
            x.copy_from_slice(&clean);
            x[poison] = f32::NAN;
            let mut a = vec![0.0f32; x.len()];
            let mut b = vec![0.0f32; x.len()];
            tanh_f32_slice(&x, &mut a);
            tanh_f32_slice(&clean, &mut b);
            for i in 0..x.len() {
                if i == poison {
                    assert!(a[i].is_nan(), "poison lane {i} lost its NaN");
                } else {
                    assert_eq!(
                        a[i].to_bits(),
                        b[i].to_bits(),
                        "lane {i} perturbed by NaN at {poison}"
                    );
                }
            }
        }
    }

    // ---- lengths, tails, aliasing ------------------------------------------

    /// Every length from 0 up to four vectors past the dispatch threshold, so
    /// the masked tail is exercised at all eight residues and the short
    /// (scalar) path is compared against the same reference.
    #[test]
    fn all_lengths_match_reference() {
        let base: Vec<f32> = (0..SIMD_MIN_LEN + 40)
            .map(|i| (i as f32 - 30.0) * 0.37)
            .collect();
        for len in 0..base.len() {
            let x = &base[..len];
            let mut o = vec![0.0f32; len];

            tanh_f32_slice(x, &mut o);
            check(x, &o, tanh_ref, TANH_BOUND, &format!("tanh len={len}"));
            sigmoid_f32_slice(x, &mut o);
            check(
                x,
                &o,
                sigmoid_ref,
                SIGMOID_BOUND,
                &format!("sigmoid len={len}"),
            );
            tanh_gelu_f32_slice(x, &mut o);
            check(x, &o, tanh_gelu_ref, GELU_BOUND, &format!("gelu len={len}"));
            quick_gelu_f32_slice(x, &mut o, 1.702);
            check(
                x,
                &o,
                |v| quick_gelu_ref(v, 1.702),
                GELU_BOUND * 1.702,
                &format!("quick len={len}"),
            );
        }
    }

    /// The masked tail must not write past the requested length.
    #[test]
    fn masked_tail_does_not_overwrite_neighbours() {
        const GUARD: u32 = 0xDEAD_BEEF;
        for len in SIMD_MIN_LEN..SIMD_MIN_LEN + 16 {
            let x: Vec<f32> = (0..len).map(|i| i as f32 * 0.1 - 3.0).collect();
            let mut out = vec![f32::from_bits(GUARD); len + 16];
            tanh_f32_slice(&x, &mut out[..len]);
            for (i, &v) in out[len..].iter().enumerate() {
                assert_eq!(v.to_bits(), GUARD, "len {len} clobbered slot +{i}");
            }
        }
    }

    /// Misaligned inputs and outputs (the kernels use unaligned loads/stores).
    #[test]
    fn unaligned_slices_match_aligned() {
        let backing: Vec<f32> = (0..SIMD_MIN_LEN + 32)
            .map(|i| (i as f32 - 20.0) * 0.31)
            .collect();
        let n = SIMD_MIN_LEN + 8;
        let mut aligned = vec![0.0f32; n];
        tanh_f32_slice(&backing[..n], &mut aligned);
        for off in 1..8 {
            let mut out = vec![0.0f32; n + off];
            tanh_f32_slice(&backing[..n], &mut out[off..][..n]);
            for i in 0..n {
                assert_eq!(
                    out[off + i].to_bits(),
                    aligned[i].to_bits(),
                    "offset {off} lane {i}"
                );
            }
        }
    }

    // ---- documented deviations ---------------------------------------------

    /// Monotonicity.
    ///
    /// Both functions are non-decreasing in exact arithmetic. In the interior
    /// of their range — where the derivative is large relative to the output
    /// resolution — the vector kernels reproduce that exactly. Near the
    /// asymptotes they do not, for two reasons that are both inherited from
    /// MLAS/Eigen and are present in ORT as well:
    ///
    /// * `sigmoid` is evaluated as `p/q + 0.5`. In the far negative tail that
    ///   sum cancels, so the result is quantised to multiples of
    ///   `ulp(0.5) = 6e-8` and can step backwards by a few of those.
    /// * `tanh`'s `p/q` rounds to exactly `±1` slightly before the `±9`
    ///   saturation point, so a one-ulp backwards step can occur there.
    ///
    /// Both deviations are bounded by the documented absolute error, which is
    /// itself two orders of magnitude inside the ONNX conformance tolerance
    /// (`atol = 1e-5`). This test pins that: strict monotonicity where the
    /// output is informative, bounded regression everywhere else.
    #[test]
    fn monotonicity_within_documented_slack() {
        /// Region where the output is far enough from its asymptotes that
        /// strict monotonicity is required.
        fn informative(v: f32, lo: f32, hi: f32) -> bool {
            let span = hi - lo;
            v > lo + span * 1e-2 && v < hi - span * 1e-2
        }
        fn check_pair(name: &str, x: f32, prev: f32, cur: f32, lo: f32, hi: f32, slack: f64) {
            if informative(prev, lo, hi) {
                assert!(cur >= prev, "{name} not monotone at {x}: {cur} < {prev}");
            } else {
                assert!(
                    f64::from(cur) >= f64::from(prev) - slack,
                    "{name} regressed beyond the documented bound at {x}: {cur} < {prev}"
                );
            }
            assert!(
                (lo..=hi).contains(&cur),
                "{name}({x}) = {cur} escaped [{lo}, {hi}]"
            );
        }

        let x: Vec<f32> = (0..60_001).map(|i| -30.0 + i as f32 * 0.001).collect();
        let mut t = vec![0.0f32; x.len()];
        let mut s = vec![0.0f32; x.len()];
        tanh_f32_slice(&x, &mut t);
        sigmoid_f32_slice(&x, &mut s);
        for i in 1..x.len() {
            check_pair("tanh", x[i], t[i - 1], t[i], -1.0, 1.0, 2.0 * TANH_BOUND);
            check_pair(
                "sigmoid",
                x[i],
                s[i - 1],
                s[i],
                0.0,
                1.0,
                2.0 * SIGMOID_BOUND,
            );
        }
    }

    /// Known deviation: `tanh`'s numerator is `x * poly(x^2)`, and for
    /// subnormal `x` that product underflows, so the result is a signed zero
    /// rather than `x`. The absolute error is bounded by `f32::MIN_POSITIVE`
    /// (1.2e-38) and the sign is preserved. `sigmoid` is unaffected.
    #[test]
    fn subnormal_inputs_underflow_to_signed_zero() {
        let d = f32::from_bits(1);
        let x: Vec<f32> = std::iter::repeat_n(d, PAD)
            .chain([d, -d, f32::MIN_POSITIVE, -f32::MIN_POSITIVE])
            .collect();
        let mut o = vec![0.0f32; x.len()];

        tanh_f32_slice(&x, &mut o);
        for (i, (&got, &inp)) in o.iter().zip(&x).enumerate() {
            assert!(
                (f64::from(got) - f64::from(inp)).abs() <= f64::from(f32::MIN_POSITIVE),
                "lane {i}: tanh({inp:e}) = {got:e} exceeds the documented subnormal bound"
            );
            assert_eq!(
                got.is_sign_negative(),
                inp.is_sign_negative(),
                "lane {i}: tanh({inp:e}) lost the sign"
            );
        }

        sigmoid_f32_slice(&x, &mut o);
        assert!(o.iter().all(|&v| v == 0.5), "sigmoid near zero must be 0.5");
    }

    /// The `< SIMD_MIN_LEN` scalar path and the vector path must agree to
    /// within the sum of their documented bounds, so a caller cannot observe a
    /// discontinuity purely from tensor size.
    #[test]
    fn scalar_and_vector_paths_agree() {
        if !vector_path_available() {
            return;
        }
        let x: Vec<f32> = (0..SIMD_MIN_LEN * 4)
            .map(|i| (i as f32 - 60.0) * 0.21)
            .collect();
        let mut vector = vec![0.0f32; x.len()];
        tanh_f32_slice(&x, &mut vector);
        for (i, (&xi, &v)) in x.iter().zip(&vector).enumerate() {
            let scalar = tanh_scalar(xi);
            let e = scaled_err(v, scalar, xi);
            assert!(
                e <= TANH_BOUND,
                "lane {i}: vector {v} vs scalar {scalar}, err {e:e}"
            );
        }
        sigmoid_f32_slice(&x, &mut vector);
        for (i, (&xi, &v)) in x.iter().zip(&vector).enumerate() {
            let scalar = sigmoid_scalar(xi);
            let e = scaled_err(v, scalar, xi);
            assert!(
                e <= SIGMOID_BOUND,
                "lane {i}: vector {v} vs scalar {scalar}, err {e:e}"
            );
        }
    }

    // ── direct-write plumbing ─────────────────────────────────────────────

    use crate::kernels::testutil::Owned;

    /// `write_mapped` must produce the same bytes whether it took the
    /// direct-write arm or the owned-scratch arm. The interesting case is an
    /// in-place node (`y = tanh(y)`), where the widened input borrows the
    /// output's own storage: writing through would corrupt the tail of the
    /// input mid-kernel, so `output_direct_write_eligible` has to reject it and
    /// send us to the scratch buffer.
    #[test]
    fn write_mapped_agrees_between_direct_and_aliased_outputs() {
        let n = 257; // not a multiple of 8, so the masked tail runs too
        let src: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.11).collect();

        let mut expected = vec![0.0f32; n];
        tanh_f32_slice(&src, &mut expected);

        // Disjoint output: takes the direct-write arm.
        let mut disjoint = Owned::f32(&[n], &vec![0.0f32; n]);
        write_mapped("Tanh", &mut disjoint.view_mut(), &src, |x, y| {
            tanh_f32_slice(x, y)
        })
        .unwrap();
        assert_eq!(disjoint.to_f32(), expected);

        // Aliased output: the input slice *is* the output storage.
        let mut aliased = Owned::f32(&[n], &src);
        {
            let mut view = aliased.view_mut();
            // SAFETY: `view` addresses `n` contiguous f32; the slice is only
            // read while `write_mapped` decides which arm to take, which is
            // exactly the aliasing `output_direct_write_eligible` must detect.
            let borrowed: &[f32] =
                unsafe { std::slice::from_raw_parts(view.data_ptr_mut::<f32>(), n) };
            let borrowed: &[f32] = unsafe { std::mem::transmute(borrowed) };
            write_mapped("Tanh", &mut view, borrowed, tanh_f32_slice).unwrap();
        }
        assert_eq!(aliased.to_f32(), expected);

        // A non-f32 output can never take the direct arm; it must still narrow
        // correctly.
        let mut narrowed = Owned::f16(&[n], &vec![0.0f32; n]);
        write_mapped("Tanh", &mut narrowed.view_mut(), &src, |x, y| {
            tanh_f32_slice(x, y)
        })
        .unwrap();
        let got = narrowed.to_u16_bits();
        for (g, e) in got.iter().zip(&expected) {
            assert_eq!(*g, half::f16::from_f32(*e).to_bits());
        }
    }

    /// Fusing FastGelu's bias into the vector kernel must be bit-identical to
    /// materialising `x + bias` and mapping over it — that equivalence is the
    /// whole justification for the fused path.
    #[test]
    fn bias_fusion_is_bit_identical_to_folding_first() {
        for width in [1usize, 7, 8, 31, 32, 33, 64, 129] {
            let rows = 5;
            let n = rows * width;
            let x: Vec<f32> = (0..n).map(|i| (i as f32).sin() * 7.0).collect();
            let bias: Vec<f32> = (0..width).map(|i| (i as f32).cos() * 3.0).collect();

            let mut folded_in = vec![0.0f32; n];
            for (row_in, row_out) in x.chunks(width).zip(folded_in.chunks_mut(width)) {
                for ((o, &v), &b) in row_out.iter_mut().zip(row_in).zip(&bias) {
                    *o = v + b;
                }
            }
            let mut want = vec![0.0f32; n];
            tanh_gelu_f32_slice(&folded_in, &mut want);

            let mut got = vec![0.0f32; n];
            tanh_gelu_bias_f32_slice(&x, &bias, width, &mut got);

            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "width={width} i={i}: fused {g} != folded {w}"
                );
            }
        }
    }

    /// Special values must survive the fused bias add: `+inf + finite` stays
    /// `+inf`, `-inf` maps to `0`, and a NaN bias poisons the row.
    #[test]
    fn bias_fusion_handles_special_values() {
        let width = 40;
        let mut x = vec![1.0f32; width * 2];
        x[0] = f32::INFINITY;
        x[1] = f32::NEG_INFINITY;
        x[2] = f32::NAN;
        let mut bias = vec![0.5f32; width];
        bias[3] = f32::NAN;
        bias[4] = f32::INFINITY;

        let mut got = vec![0.0f32; x.len()];
        tanh_gelu_bias_f32_slice(&x, &bias, width, &mut got);

        assert_eq!(got[0], f32::INFINITY);
        assert_eq!(got[1], 0.0);
        assert!(got[2].is_nan());
        assert!(got[3].is_nan());
        assert_eq!(got[4], f32::INFINITY);
    }

    /// A bias that aliases the output must force the scratch arm. Writing the
    /// first row through would otherwise corrupt the bias that every later row
    /// still needs, so this is a data-corruption regression test.
    #[test]
    fn write_mapped_reading_rejects_an_aliasing_extra_read() {
        let width = 40;
        let n = width * 4;
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 8.0).collect();
        let bias: Vec<f32> = (0..width).map(|i| (i as f32) * 0.05 - 1.0).collect();

        let mut want = vec![0.0f32; n];
        tanh_gelu_bias_f32_slice(&x, &bias, width, &mut want);

        // Place the bias inside the output buffer itself.
        let mut seeded = vec![0.0f32; n];
        seeded[..width].copy_from_slice(&bias);
        let mut out = Owned::f32(&[n], &seeded);
        {
            let mut view = out.view_mut();
            // SAFETY: `view` addresses `n` contiguous f32, the first `width` of
            // which hold the bias. This is exactly the overlap that the extra
            // read range must make `write_mapped_reading` detect.
            let aliased: &[f32] =
                unsafe { std::slice::from_raw_parts(view.data_ptr_mut::<f32>(), width) };
            let aliased: &[f32] = unsafe { std::mem::transmute(aliased) };
            write_mapped_reading(
                "FastGelu",
                &mut view,
                &x,
                &[crate::dtype::slice_byte_range(aliased)],
                |x, y| tanh_gelu_bias_f32_slice(x, aliased, width, y),
            )
            .unwrap();
        }
        assert_eq!(out.to_f32(), want);
    }

    /// A tensor whose length is not a whole number of bias rows must still have
    /// every element written, with the bias broadcast as `bias[i % width]`.
    #[test]
    fn bias_fusion_writes_a_trailing_partial_row() {
        let width = 48;
        for n in [1usize, width + 1, width * 2 + 7, width * 3 - 1] {
            let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03 - 5.0).collect();
            let bias: Vec<f32> = (0..width).map(|i| (i as f32) * 0.02 - 0.4).collect();

            let mut want = vec![f32::NAN; n];
            for (row_in, row_out) in x.chunks(width).zip(want.chunks_mut(width)) {
                let folded: Vec<f32> = row_in.iter().zip(&bias).map(|(v, b)| v + b).collect();
                tanh_gelu_f32_slice(&folded, row_out);
            }

            let mut got = vec![f32::NAN; n];
            tanh_gelu_bias_f32_slice(&x, &bias, width, &mut got);

            for (i, g) in got.iter().enumerate() {
                assert!(!g.is_nan(), "n={n} i={i}: element never written");
            }
            // A short final row can land on the other side of the length
            // threshold from the reference, so bound rather than bit-compare.
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                let scale = x[i].abs().max(1.0);
                assert!(
                    f64::from((g - w).abs()) / f64::from(scale) <= GELU_BOUND,
                    "n={n} i={i}: {g} vs {w}"
                );
            }
        }
    }
}
