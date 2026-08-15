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
//!   polynomial value at the clamp point. That is wrong for `±Inf`
//!   (`sigmoid(-Inf)` would leak `1.5e-8` instead of `0`), so this module
//!   *saturates* instead: outside the band the exact limit is substituted.
//!   Because `tanh(9) = 1 - 3.0e-8` and `sigmoid(18) = 1 - 1.5e-8` both round
//!   to `1.0f32`, saturation is strictly more accurate than clamping, not
//!   less.
//! * `NaN` propagates unchanged. Both the clamp (`maxps`/`minps` with the
//!   value as the *second* operand, matching MLAS) and the saturation selects
//!   (ordered compares, which are false for `NaN`) preserve it.
//! * Signed zero is preserved: the numerator is odd, so `p * (-0.0) = -0.0`
//!   and `-0.0 / q = -0.0`.
//!
//! # ISA dependence
//!
//! Dispatch is by runtime feature detection and only ever *adds* a path:
//! without AVX2+FMA the caller's exact scalar closure is used unchanged, so no
//! existing target regresses. As with ORT, this means results can differ by
//! ~1e-7 relative between an AVX2 host and a non-AVX2 host. Tests therefore
//! assert an error bound against an `f64` reference rather than bit equality.

#![allow(clippy::excessive_precision)]

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

            // The rational overshoots `tanh`'s mathematical range near the
            // clamp point — `p/q` reaches `1.0000001` around `|x| = 8.99` —
            // so pin it to `[-1, 1]`. MLAS ships the overshoot; downstream
            // code is entitled to assume `|tanh| <= 1`, so we do not.
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
}
