//! Vectorised transcendental primitives for the activation family (`Tanh`,
//! `Sigmoid`, `Erf`, `Gelu` in both its exact and tanh forms, `FastGelu`,
//! `QuickGelu`, `BiasGelu`).
//!
//! # Why this exists
//!
//! The scalar kernels evaluate one `libm` transcendental per element. On this
//! class of hardware a dependent `tanhf` is ~13 ns and `f64::tanh` ~25 ns per
//! element, which is roughly two orders of magnitude above what ONNX Runtime
//! achieves for the same op. ORT's advantage is not algorithmic: MLAS ships
//! hand-written FMA3 kernels (`lib/x86_64/{Tanh,Logistic,Erf}KernelFma3.S`)
//! that evaluate a *polynomial* over a clamped range, eight lanes at a time,
//! with no branches and no libm call.
//!
//! This module reproduces those approximations in safe-ish Rust over
//! `core::arch::x86_64` intrinsics, so the default build (which does **not**
//! enable the `mlas` feature) gets the same throughput.
//!
//! # Numerical contract
//!
//! * The polynomials and their evaluation order are taken verbatim from
//!   `MlasTanhConstants` / `MlasLogisticConstants` (which MLAS in turn took
//!   from Eigen) and `MlasErfConstants`, so a build using this path tracks
//!   ORT's own CPU output.
//! * `erf` is the one member of the family whose scalar fallback is *more*
//!   accurate than its vector path: `libm::erf` is correctly rounded, MLAS's
//!   polynomial is only faithfully rounded (measured worst error 5.96e-8 =
//!   1 ulp below 1.0, over a 400 003-point sweep of `[-6, 6]` plus both branch
//!   boundaries). That is the same trade ORT itself makes, and 1 ulp is two
//!   orders of magnitude inside the conformance suite's `rtol=1e-4`.
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
///
/// Only the `x86_64` dispatch arms consult this, but the tests below use it as
/// a length unit on every architecture, so it stays compiled rather than
/// `cfg`-gated.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) const SIMD_MIN_LEN: usize = 32;

// ---------------------------------------------------------------------------
// MLAS constants
// ---------------------------------------------------------------------------

/// `MlasTanhConstants`, `onnxruntime/core/mlas/lib/tanh.cpp`.
///
/// Consumed exclusively by the AVX2 module below, so it follows that module's
/// gating: on a non-`x86_64` target these would be unreferenced constants, and
/// CI builds with `-D warnings`.
#[cfg(target_arch = "x86_64")]
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
///
/// `x86_64`-only for the same reason as [`tanh_c`].
#[cfg(target_arch = "x86_64")]
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

/// `MlasErfConstants`, `onnxruntime/core/mlas/lib/erf.cpp`.
///
/// MLAS took the algorithm and coefficients from the "efficient faithfully
/// rounded implementation of erff" reference cited at the top of that file.
/// `x86_64`-only for the same reason as [`tanh_c`].
///
/// The two polynomials split at `|x| = 0.921875`:
///
/// * below it, `erf(x) ≈ x·(1 + P(x²))` with `SMALL_P5_MINUS_ONE` folded so
///   the final step is a single `fma(r, x, x)`;
/// * above it, `erf(x) = 1 - exp(-R(|x|))` with `R` a degree-7 polynomial in
///   `|x|` (again with the leading `1` folded into `BIG_P6_MINUS_ONE`), and
///   `exp` evaluated by the standard range-reduce/`2^k` scheme whose constants
///   follow.
#[cfg(target_arch = "x86_64")]
mod erf_c {
    /// `erf` is within half an ulp of `±1` past this, so MLAS clamps `|x|` here
    /// and lets the big-branch polynomial return exactly `1`.
    pub(super) const UPPER_ABS_RANGE: f32 = 3.925;
    pub(super) const SPLIT_BOUNDARY: f32 = 0.921875;

    pub(super) const SMALL_P0: f32 = -5.99104969e-4;
    pub(super) const SMALL_P1: f32 = 4.99339588e-3;
    pub(super) const SMALL_P2: f32 = -2.67667342e-2;
    pub(super) const SMALL_P3: f32 = 1.12818025e-1;
    pub(super) const SMALL_P4: f32 = -3.76124859e-1;
    pub(super) const SMALL_P5_MINUS_ONE: f32 = 1.28379151e-1;

    pub(super) const BIG_P0: f32 = 1.72948930e-5;
    pub(super) const BIG_P1: f32 = -3.83208680e-4;
    pub(super) const BIG_P2: f32 = 3.88393435e-3;
    pub(super) const BIG_P3: f32 = -2.42545605e-2;
    pub(super) const BIG_P4: f32 = 1.06777847e-1;
    pub(super) const BIG_P5: f32 = 6.34846687e-1;
    pub(super) const BIG_P6_MINUS_ONE: f32 = 1.28717512e-1;

    // Independent `exp` parameters, used only by the big branch.
    pub(super) const EXP_LOWER_RANGE: f32 = -88.376_262_664_794_9;
    /// MLAS spells this `1.44269504088896341f`, which rounds to exactly
    /// `f32::consts::LOG2_E`.
    pub(super) const EXP_LOG2_RECIPROCAL: f32 = std::f32::consts::LOG2_E;
    pub(super) const EXP_LOG2_HI: f32 = -6.93145752e-1;
    pub(super) const EXP_LOG2_LO: f32 = -1.42860677e-6;
    pub(super) const EXP_P0: f32 = 1.38319808e-3;
    pub(super) const EXP_P1: f32 = 8.37550033e-3;
    pub(super) const EXP_P2: f32 = 4.16689515e-2;
    pub(super) const EXP_P3: f32 = 1.66664466e-1;
    pub(super) const EXP_P4: f32 = 4.99999851e-1;
    pub(super) const EXP_P5: f32 = 1.0;
    pub(super) const EXP_P6: f32 = 1.0;
    /// `1.5 · 2^23`: adding then subtracting it rounds a float to the nearest
    /// integer under the default rounding mode without a `roundps`.
    pub(super) const EXP_C: f32 = 1.25829120e7;
}

/// `MlasExpConstants` (`onnxruntime/core/mlas/lib/compute.cpp`), the parameter
/// set behind `MlasComputeExp`.
///
/// These are *not* the `erf_c::EXP_*` values above. `erf_c`'s copy is the older
/// polynomial extracted from `MlasErfConstants`, valid only on
/// `[-88.376, 0]` — enough for `erf`'s big branch, which never sees a positive
/// argument. A standalone `Exp` operator has to cover the whole `f32` line, so
/// MLAS uses a separately refined polynomial plus XNNPACK's two-piece exponent
/// reconstruction, which extends the representable output range down to
/// `-103.972` (subnormal results) and up to `88.776` (overflow to `+Inf`).
#[cfg(target_arch = "x86_64")]
mod exp_c {
    /// Below this every result is `0`; `-Inf` clamps here.
    pub(super) const LOWER_RANGE: f32 = -103.9720840454;
    /// Above this every result overflows `f32`; `+Inf` clamps here and the
    /// reconstruction below still evaluates to `+Inf`.
    pub(super) const UPPER_RANGE: f32 = 88.7762626647950;
    /// `1.5 · 2^23`, the round-to-nearest-integer magic constant. Also reused
    /// as the raw bit source for the exponent reconstruction.
    pub(super) const ROUNDING_BIAS: f32 = 1.25829120e7;
    pub(super) const LOG2_RECIPROCAL: f32 = std::f32::consts::LOG2_E;
    pub(super) const LOG2_HIGH: f32 = -6.93145752e-1;
    pub(super) const LOG2_LOW: f32 = -1.42860677e-6;
    pub(super) const P0: f32 = f32::from_bits(0x3AB4_A000);
    pub(super) const P1: f32 = f32::from_bits(0x3C09_2F6E);
    pub(super) const P2: f32 = f32::from_bits(0x3D2A_ADAD);
    pub(super) const P3: f32 = f32::from_bits(0x3E2A_AA28);
    pub(super) const P4: f32 = f32::from_bits(0x3EFF_FFFB);
    /// MLAS stores a single `poly_56` field because the degree-5 and degree-6
    /// coefficients are both exactly `1.0`. In `MlasComputeExpVector` — the
    /// variant ported here — only one of them appears in the Horner chain; the
    /// other is merged into the overflow-exponent multiply/add below, following
    /// XNNPACK. (The reduced-range helper further down `compute.cpp` instead
    /// applies `poly_56` twice, because it has no overflow term to fold into.)
    pub(super) const P56: f32 = 1.0;
    /// Exponent field clamps for the two-piece `2^m` reconstruction.
    pub(super) const MINIMUM_EXPONENT: i32 = 0xC100_0000u32 as i32;
    pub(super) const MAXIMUM_EXPONENT: i32 = 0x3F80_0000;
}

/// `√(2/π)` and the cubic coefficient of the tanh GELU approximation, rounded
/// to `f32`. Matches ORT's `contrib_ops/cpu/bert/fast_gelu.cc`.
const GELU_B: f32 = 0.7978845608028654;
const GELU_C: f32 = 0.044715;

/// `1/√2`, the inner scale of exact GELU, rounded to `f32`. ORT's
/// `Gelu(approximate="none")` CPU kernel scales by `M_SQRT1_2` in `float`
/// before calling `MlasComputeErf`, so this matches its evaluation order.
#[cfg(target_arch = "x86_64")]
const FRAC_1_SQRT_2_F32: f32 = std::f32::consts::FRAC_1_SQRT_2;

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Returns `true` when the AVX2+FMA vector kernels in this module are live.
///
/// Deliberately answers on every architecture (`false` off `x86_64`) so tests
/// can branch on it portably; only the `x86_64` dispatch arms call it, hence
/// the conditional `dead_code` allowance.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
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

/// `y = e^x`.
///
/// The vector path is a port of `MlasComputeExpVector`
/// (`onnxruntime/core/mlas/lib/compute.cpp`), the same polynomial MLAS uses
/// inside its own softmax and logistic kernels. The scalar fallback is
/// `f32::exp`, which is correctly rounded, so — exactly as for [`erf_f32_slice`]
/// — a value can differ by ~1 ulp depending on whether the tensor was long
/// enough to reach [`SIMD_MIN_LEN`]. `Exp` has no bit-exactness contract in
/// ONNX and ORT's own CPU kernel is a different (Eigen) approximation again, so
/// this seam is a documented accuracy property, not a correctness bug. Over a
/// 65536-point sweep of `[-110, 89]` the worst observed error against an `f64`
/// reference is **1 ulp**, including through the subnormal range.
///
/// Special values match `f32::exp` and ORT: `NaN` in gives `NaN` out (see
/// [`avx2::exp_full_ps`] — it falls out of the clamp's operand order rather
/// than a mask), `+Inf` gives `+Inf`, `-Inf` gives `+0`, arguments above
/// `88.7762626647950` overflow to `+Inf`, and arguments below
/// `-103.9720840454` flush to `+0` through the subnormal range.
pub(crate) fn exp_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, f32::exp, exp_avx2);
}

/// `y = erf(x)`, the Gauss error function.
///
/// The vector path is a port of `MlasErfKernel` (`onnxruntime/core/mlas/lib/
/// erf.cpp`), which is what ORT's own CPU `Erf` and `Gelu(approximate="none")`
/// kernels evaluate via `MlasComputeErf`. It is a *faithfully rounded* `f32`
/// approximation, not the correctly-rounded `libm::erf` the scalar fallback
/// uses: see the module-level note on ISA dependence. Measured against an
/// `f64` reference over a dense sweep the worst observed error is 5.96e-8
/// (1 ulp below `1.0`), and against ORT 1.28.0 on identical inputs the two
/// agree bit-for-bit over 4M+ probed points — which is the point of porting
/// MLAS's coefficients rather than inventing a polynomial.
///
/// # The `SIMD_MIN_LEN` seam
///
/// Slices shorter than [`SIMD_MIN_LEN`] take the correctly-rounded scalar
/// fallback, so the *same* input value can differ by up to 1 ulp depending on
/// how many elements the tensor has — measured at 286 of 2000 random values
/// between a 31-element and a 40-element tensor. `Tanh` and `Sigmoid` have had
/// exactly this seam since they were vectorised, and ORT has it too (MLAS
/// dispatches its own scalar tail below the vector width). It is accepted for
/// the same reason: 1 ulp is three orders of magnitude inside the conformance
/// tolerance, and the alternative is making short tensors 20x slower to buy
/// bit-stability nothing depends on.
pub(crate) fn erf_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, erf_scalar, erf_avx2);
}

/// `y = 0.5·x·(1 + erf(x/√2))`, the exact (`approximate="none"`) GELU.
///
/// Fused rather than composed out of [`erf_f32_slice`] so the intermediate
/// `x/√2` is never written to memory.
pub(crate) fn erf_gelu_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, erf_gelu_scalar, erf_gelu_avx2);
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

/// `erf_gelu(x[i] + bias[i % width])`, the `BiasGelu` contrib op.
///
/// Same in-register bias fold as [`tanh_gelu_bias_f32_slice`], and the same
/// contract: `width` non-zero, `bias.len() == width`, a trailing partial row
/// consumes the matching bias prefix.
pub(crate) fn erf_gelu_bias_f32_slice(
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
        if input.len() >= SIMD_MIN_LEN && vector_path_available() {
            // SAFETY: guarded by the runtime AVX2+FMA detection above; the
            // debug asserts above are the caller's contract.
            unsafe { erf_gelu_bias_avx2(input, bias, width, output) };
            return;
        }
    }
    for (row_in, row_out) in input.chunks(width).zip(output.chunks_mut(width)) {
        for ((o, &v), &b) in row_out.iter_mut().zip(row_in).zip(bias) {
            *o = erf_gelu_scalar(v + b);
        }
    }
}

// ---------------------------------------------------------------------------
// Exact (bit-identical) elementwise kernels
// ---------------------------------------------------------------------------
//
// Unlike the rest of this module these are **not** approximations. Each is a
// sign-bit mask, an IEEE-754 division, or `vroundps` in an explicit rounding
// mode, so the vector body, the masked tail and the scalar fallback all produce
// bit-identical results — `-0.0`, `±Inf`, subnormals and NaN payloads included.
// There is no `SIMD_MIN_LEN` accuracy seam here, only a throughput one.
//
// They needed their own kernels because `UnaryMathKernel::execute_f32` used to
// hand `write_mapped` a closure that ran `MathOp::apply` — a 24-arm `match` —
// *inside* the element loop. LLVM could not hoist it, so even `Neg`, which is
// one `vxorps`, compiled to a per-element jump table. Measured against ORT that
// put `Neg` at 0.049x at 1M elements while the plugin EP was claiming the op.
// ---------------------------------------------------------------------------

/// `y = -x`. Flips the sign bit; exact for every input, `NaN` keeps its payload
/// with the sign flipped and `-0.0 -> +0.0`.
pub(crate) fn neg_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, |v: f32| -v, neg_avx2);
}

/// `y = |x|`. Clears the sign bit; exact, and `|NaN|` keeps its payload.
pub(crate) fn abs_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, f32::abs, abs_avx2);
}

/// `y = 1 / x`.
///
/// `vdivps`, **not** `vrcpps`: the reciprocal approximation carries only ~12
/// bits and would break the bit-exactness this group promises. Division is
/// correctly rounded, so this matches `1.0 / x` exactly, including
/// `1/+0 = +Inf` and `1/-0 = -Inf`.
pub(crate) fn reciprocal_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, |v: f32| 1.0 / v, reciprocal_avx2);
}

/// `y = floor(x)`, `vroundps` mode 1 — identical to `f32::floor`.
pub(crate) fn floor_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, f32::floor, floor_avx2);
}

/// `y = ceil(x)`, `vroundps` mode 2 — identical to `f32::ceil`.
pub(crate) fn ceil_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, f32::ceil, ceil_avx2);
}

/// `y = round-half-to-even(x)`, `vroundps` mode 0.
///
/// ONNX `Round` is banker's rounding, which is `f32::round_ties_even` and
/// `_MM_FROUND_TO_NEAREST_INT` — *not* `f32::round`, which rounds halves away
/// from zero.
pub(crate) fn round_ties_even_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, f32::round_ties_even, round_ties_even_avx2);
}

/// ONNX `Sign`: `-1` / `0` / `+1`, with `sign(±0) = 0` and `sign(NaN) = NaN`.
///
/// Built from two *ordered* compares, which are false for `NaN`, so a `NaN`
/// input falls through both selects and is returned unchanged — matching the
/// scalar `is_nan()` branch bit-for-bit, payload included.
pub(crate) fn sign_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, sign_scalar, sign_avx2);
}

/// `y = x / (1 + |x|)`, ONNX `Softsign`.
///
/// One `vandps` and one `vdivps`, both exact, so this is bit-identical to the
/// scalar form rather than an approximation of it.
pub(crate) fn softsign_f32_slice(input: &[f32], output: &mut [f32]) {
    dispatch!(input, output, softsign_scalar, softsign_avx2);
}

/// ONNX `Sign`: `-1` / `0` / `+1`, with `sign(±0) = +0` and `sign(NaN) = NaN`.
///
/// The `NaN` input is returned **unchanged**, payload and sign bit included.
/// This function used to return the canonical `f32::NAN` instead, which
/// silently rewrote `0xFFC01234` to `0x7FC01234`. ORT 1.28.0's CPU `Sign`
/// preserves the input bit pattern (verified on both signs and a non-default
/// payload), and so does the AVX2 path below — its ordered compares are all
/// false for `NaN`, so the lane falls through every select. Canonicalising was
/// the odd one out, and it is what made the two paths disagree.
#[inline]
pub(crate) fn sign_scalar(x: f32) -> f32 {
    if x.is_nan() {
        x
    } else if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// ONNX `Softsign`: `x / (1 + |x|)`.
#[inline]
pub(crate) fn softsign_scalar(x: f32) -> f32 {
    x / (1.0 + x.abs())
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

/// Correctly-rounded `erf`, the fallback when AVX2+FMA is absent. `libm::erf`
/// is `f64`, so this is what makes the non-x86 path both exact and slow; see
/// [`erf_f32_slice`].
#[inline]
fn erf_scalar(x: f32) -> f32 {
    crate::kernels::elementwise::erf(f64::from(x)) as f32
}

/// Exact GELU on the scalar fallback, in `f64` throughout to match the
/// pre-existing `kernels::gelu::exact_gelu`.
#[inline]
fn erf_gelu_scalar(x: f32) -> f32 {
    if x == f32::NEG_INFINITY {
        return 0.0;
    }
    let xf = f64::from(x);
    (0.5 * xf * (1.0 + crate::kernels::elementwise::erf(xf * std::f64::consts::FRAC_1_SQRT_2)))
        as f32
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
    use super::{GELU_B, GELU_C, erf_c, exp_c, logistic_c, tanh_c};
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
            // The `[-1, 1]` clamp is also the saturation. Because the input was
            // already clamped to `[-9, 9]`, the largest argument the rational
            // ever sees is `±9`, where `p/q` evaluates to `±1.0000001`; the
            // clamp turns that into exactly `±1`. So every `|x| >= 9` — up to
            // and including `±Inf` — already leaves here as exactly `±1`
            // without a separate saturation step, and `NaN` survives because
            // `maxps`/`minps` return their second operand on an unordered
            // compare. `tanh_saturation_blend_is_redundant` proves this by
            // sweeping all 1 047 527 424 finite `f32` with `|x| >= 9`.
            clamp_nan_preserving(_mm256_div_ps(p, q), -1.0, 1.0)
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

            // As in `tanh_ps`, the `[0, 1]` clamp is also the saturation: the
            // rational only ever sees `±18`, where `p/q + 0.5` lands outside
            // `[0, 1]` on both ends, so `|x| >= 18` and `±Inf` already leave
            // here as exactly `0` or `1`.
            // `sigmoid_saturation_blend_is_redundant` proves this over all
            // 1 039 138 816 finite `f32` with `|x| >= 18`.
            clamp_nan_preserving(poly, 0.0, 1.0)
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
            // Blending against zero is an `andnot` of the compare mask, which
            // is one uop where `vblendvps` is two on Zen. The mask is all-ones
            // or all-zeros, so the two are exactly equivalent here.
            let neg_inf = _mm256_cmp_ps(x, _mm256_set1_ps(f32::NEG_INFINITY), _CMP_EQ_OQ);
            _mm256_andnot_ps(neg_inf, y)
        }
    }

    /// `x·sigmoid(alpha·x)` over 8 lanes, with `x = -Inf` pinned to `0`.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn quick_gelu_ps(x: __m256, alpha: __m256) -> __m256 {
        unsafe {
            let s = sigmoid_ps(_mm256_mul_ps(alpha, x));
            let y = _mm256_mul_ps(x, s);
            // Blending against zero is an `andnot` of the compare mask, which
            // is one uop where `vblendvps` is two on Zen. The mask is all-ones
            // or all-zeros, so the two are exactly equivalent here.
            let neg_inf = _mm256_cmp_ps(x, _mm256_set1_ps(f32::NEG_INFINITY), _CMP_EQ_OQ);
            _mm256_andnot_ps(neg_inf, y)
        }
    }

    /// `erf` over 8 lanes, following `MlasErfKernel` step for step.
    ///
    /// Both branches are evaluated for every lane and merged with `or`, which
    /// is what MLAS does: the inactive branch is forced to `+0.0` — the small
    /// branch by `andnot(split, ..)`, the big branch because zeroing its input
    /// collapses the polynomial to `1 - exp(-0) = 0` — so the `or` acts as a
    /// select. Being branch-free is why this beats a scalar `libm::erf` by far
    /// more than the 8× the SIMD width alone would give.
    ///
    /// The sign is stripped up front (`erf` is odd) and re-applied at the end
    /// with an `or`, so `erf(-0.0) = -0.0` and a negative `NaN` stays negative.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn erf_ps(x: __m256) -> __m256 {
        unsafe {
            let neg_zero = _mm256_set1_ps(-0.0);
            let sign = _mm256_and_ps(x, neg_zero);
            // `minps` returns its *second* operand when either is NaN, so a
            // NaN input survives the clamp — matching MLAS's operand order.
            let abs = _mm256_min_ps(
                _mm256_set1_ps(erf_c::UPPER_ABS_RANGE),
                _mm256_andnot_ps(neg_zero, x),
            );
            let sq = _mm256_mul_ps(abs, abs);

            // |x| <= 0.921875: erf(x) = |x|·(1 + P(x²)).
            let mut small = _mm256_fmadd_ps(
                _mm256_set1_ps(erf_c::SMALL_P0),
                sq,
                _mm256_set1_ps(erf_c::SMALL_P1),
            );
            small = _mm256_fmadd_ps(small, sq, _mm256_set1_ps(erf_c::SMALL_P2));
            small = _mm256_fmadd_ps(small, sq, _mm256_set1_ps(erf_c::SMALL_P3));
            small = _mm256_fmadd_ps(small, sq, _mm256_set1_ps(erf_c::SMALL_P4));
            small = _mm256_fmadd_ps(small, sq, _mm256_set1_ps(erf_c::SMALL_P5_MINUS_ONE));
            small = _mm256_fmadd_ps(small, abs, abs);

            // Ordered `>`, so a NaN lane is *false* here and therefore keeps
            // the small branch, whose polynomial already produced that NaN.
            let split = _mm256_cmp_ps(abs, _mm256_set1_ps(erf_c::SPLIT_BOUNDARY), _CMP_GT_OQ);
            let small = _mm256_andnot_ps(split, small);

            // |x| > 0.921875: erf(x) = 1 - exp(-R(|x|)).
            let abs = _mm256_and_ps(split, abs);
            let mut big = _mm256_fmadd_ps(
                _mm256_set1_ps(erf_c::BIG_P0),
                abs,
                _mm256_set1_ps(erf_c::BIG_P1),
            );
            big = _mm256_fmadd_ps(big, abs, _mm256_set1_ps(erf_c::BIG_P2));
            big = _mm256_fmadd_ps(big, abs, _mm256_set1_ps(erf_c::BIG_P3));
            big = _mm256_fmadd_ps(big, abs, _mm256_set1_ps(erf_c::BIG_P4));
            big = _mm256_fmadd_ps(big, abs, _mm256_set1_ps(erf_c::BIG_P5));
            big = _mm256_fmadd_ps(big, abs, _mm256_set1_ps(erf_c::BIG_P6_MINUS_ONE));
            big = _mm256_fmadd_ps(big, abs, abs);

            let neg_big = _mm256_max_ps(
                _mm256_set1_ps(erf_c::EXP_LOWER_RANGE),
                _mm256_xor_ps(big, neg_zero),
            );
            let y = _mm256_sub_ps(_mm256_set1_ps(1.0), exp_ps(neg_big));

            _mm256_or_ps(_mm256_or_ps(small, y), sign)
        }
    }

    /// `exp` over 8 lanes for arguments already clamped to
    /// `[EXP_LOWER_RANGE, 0]`, using `MlasErfConstants`' `exp` parameters.
    ///
    /// Range-reduces `x = k·ln2 + f` with `k` obtained by the add-then-subtract
    /// round-to-integer trick, evaluates a degree-6 polynomial on `f`, then
    /// scales by `2^k` built directly in the exponent field. Only the `erf` big
    /// branch calls this, so it deliberately does *not* handle overflow, `Inf`
    /// or `NaN`: those lanes are masked off before they reach it.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn exp_ps(x: __m256) -> __m256 {
        unsafe {
            let magic = _mm256_set1_ps(erf_c::EXP_C);
            let k = _mm256_fmadd_ps(_mm256_set1_ps(erf_c::EXP_LOG2_RECIPROCAL), x, magic);
            let k = _mm256_sub_ps(k, magic);

            let mut f = _mm256_fmadd_ps(k, _mm256_set1_ps(erf_c::EXP_LOG2_HI), x);
            f = _mm256_fmadd_ps(k, _mm256_set1_ps(erf_c::EXP_LOG2_LO), f);

            let mut p = _mm256_fmadd_ps(
                _mm256_set1_ps(erf_c::EXP_P0),
                f,
                _mm256_set1_ps(erf_c::EXP_P1),
            );
            p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(erf_c::EXP_P2));
            p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(erf_c::EXP_P3));
            p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(erf_c::EXP_P4));
            p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(erf_c::EXP_P5));
            p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(erf_c::EXP_P6));

            _mm256_mul_ps(p, power_of_2_ps(k))
        }
    }

    /// Full-range `exp` over 8 lanes: a port of `MlasComputeExpVector`.
    ///
    /// Unlike [`exp_ps`], which the `erf` big branch calls with arguments that
    /// are already clamped to `[-88.376, 0]`, this covers the entire `f32`
    /// line. Two differences buy that:
    ///
    /// * `2^m` is reconstructed in **two** pieces (XNNPACK's refinement). One
    ///   exponent field cannot express the full `[-150, 128]` range a
    ///   general-purpose `exp` needs, so the biased exponent is split into a
    ///   clamped `normal` part and an `overflow` remainder, and the two are
    ///   applied at different points in the Horner chain. This is what lets
    ///   subnormal results come out right instead of flushing early.
    /// # `NaN` survives without a mask, and the clamp's operand order is why
    ///
    /// The reconstruction reinterprets `biased` as an integer, and for a `NaN`
    /// argument that integer is meaningless — so it is worth being explicit
    /// about why a `NaN` cannot come out finite. `MINPS`/`MAXPS` return their
    /// **second** operand when either input is unordered. Both clamp steps put
    /// the value being clamped second (`min(UPPER, x)`, then `max(LOWER, v)`),
    /// so a `NaN` argument passes through the clamp unchanged; the first
    /// polynomial `fmadd(P0, v, P1)` then carries it into `p`, and every later
    /// step multiplies or adds `p`, so the `NaN` reaches the result. It holds
    /// for every payload and both signs, quiet and signalling alike, because
    /// `NaN * anything` is `NaN` — including `NaN * 0` and `NaN * Inf`, the two
    /// values the integer path can produce.
    ///
    /// **The operand order in the clamp is therefore load-bearing.** Rewriting
    /// `min(UPPER, x)` as `min(x, UPPER)` would silently replace every `NaN`
    /// with `UPPER_RANGE` and turn `exp(NaN)` into `+Inf`.
    /// [`exp_tests::nan_propagates_instead_of_becoming_finite`] pins this.
    ///
    /// An explicit `_CMP_UNORD_Q` compare plus `blendv` was measured as the
    /// alternative and cost about 10% of throughput at 256 K elements for no
    /// behavioural difference.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn exp_full_ps(x: __m256) -> __m256 {
        let mut v = _mm256_min_ps(_mm256_set1_ps(exp_c::UPPER_RANGE), x);
        v = _mm256_max_ps(_mm256_set1_ps(exp_c::LOWER_RANGE), v);

        let bias = _mm256_set1_ps(exp_c::ROUNDING_BIAS);
        let biased = _mm256_fmadd_ps(v, _mm256_set1_ps(exp_c::LOG2_RECIPROCAL), bias);
        let m = _mm256_sub_ps(biased, bias);

        v = _mm256_fmadd_ps(m, _mm256_set1_ps(exp_c::LOG2_HIGH), v);
        v = _mm256_fmadd_ps(m, _mm256_set1_ps(exp_c::LOG2_LOW), v);

        let max_exp = _mm256_set1_epi32(exp_c::MAXIMUM_EXPONENT);
        let min_exp = _mm256_set1_epi32(exp_c::MINIMUM_EXPONENT);
        let raw = _mm256_slli_epi32::<23>(_mm256_castps_si256(biased));
        let normal = _mm256_max_epi32(_mm256_min_epi32(raw, max_exp), min_exp);
        let overflow = _mm256_add_epi32(_mm256_sub_epi32(raw, normal), max_exp);
        let normal = _mm256_add_epi32(normal, max_exp);
        let overflow = _mm256_castsi256_ps(overflow);
        let normal = _mm256_castsi256_ps(normal);

        let mut p = _mm256_set1_ps(exp_c::P0);
        p = _mm256_fmadd_ps(p, v, _mm256_set1_ps(exp_c::P1));
        p = _mm256_fmadd_ps(p, v, _mm256_set1_ps(exp_c::P2));
        p = _mm256_fmadd_ps(p, v, _mm256_set1_ps(exp_c::P3));
        p = _mm256_fmadd_ps(p, v, _mm256_set1_ps(exp_c::P4));
        p = _mm256_fmadd_ps(p, v, _mm256_set1_ps(exp_c::P56));

        v = _mm256_mul_ps(v, overflow);
        p = _mm256_fmadd_ps(p, v, overflow);
        p = _mm256_mul_ps(p, normal);

        p
    }

    /// `2^k` for integer-valued `k`, built by biasing and shifting into the
    /// exponent field (`MlasPowerOf2Float32x4`).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn power_of_2_ps(k: __m256) -> __m256 {
        let e = _mm256_add_epi32(_mm256_cvttps_epi32(k), _mm256_set1_epi32(127));
        _mm256_castsi256_ps(_mm256_slli_epi32::<23>(e))
    }

    /// `0.5·x·(1 + erf(x/√2))` over 8 lanes.
    ///
    /// `x = -Inf` is pinned to `0` for the same reason as [`tanh_gelu_ps`]:
    /// the natural evaluation is `0.5·(-Inf)·(1 - 1) = NaN`.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn erf_gelu_ps(x: __m256) -> __m256 {
        unsafe {
            let e = erf_ps(_mm256_mul_ps(x, _mm256_set1_ps(super::FRAC_1_SQRT_2_F32)));
            let y = _mm256_mul_ps(
                _mm256_mul_ps(_mm256_set1_ps(0.5), x),
                _mm256_add_ps(_mm256_set1_ps(1.0), e),
            );
            // Blending against zero is an `andnot` of the compare mask, which
            // is one uop where `vblendvps` is two on Zen. The mask is all-ones
            // or all-zeros, so the two are exactly equivalent here.
            let neg_inf = _mm256_cmp_ps(x, _mm256_set1_ps(f32::NEG_INFINITY), _CMP_EQ_OQ);
            _mm256_andnot_ps(neg_inf, y)
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
unsafe fn exp_avx2(input: &[f32], output: &mut [f32]) {
    unsafe { avx2::map_ps(input, output, |v| avx2::exp_full_ps(v)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sqrt_avx2(input: &[f32], output: &mut [f32]) {
    unsafe { avx2::map_ps(input, output, |v| core::arch::x86_64::_mm256_sqrt_ps(v)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn neg_avx2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;
    // XOR the sign bit. `_mm256_sub_ps(zero, v)` would turn `-0.0` into `+0.0`
    // correctly but map `NaN` through an arithmetic op; the mask is exact.
    let sign = _mm256_set1_ps(-0.0);
    unsafe { avx2::map_ps(input, output, |v| _mm256_xor_ps(v, sign)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn abs_avx2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;
    let mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    unsafe { avx2::map_ps(input, output, |v| _mm256_and_ps(v, mask)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn reciprocal_avx2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;
    let one = _mm256_set1_ps(1.0);
    unsafe { avx2::map_ps(input, output, |v| _mm256_div_ps(one, v)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn floor_avx2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;
    unsafe { avx2::map_ps(input, output, |v| _mm256_floor_ps(v)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn ceil_avx2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;
    unsafe { avx2::map_ps(input, output, |v| _mm256_ceil_ps(v)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn round_ties_even_avx2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;
    unsafe {
        avx2::map_ps(input, output, |v| {
            _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(v)
        })
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sign_avx2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;
    unsafe {
        let zero = _mm256_setzero_ps();
        let one = _mm256_set1_ps(1.0);
        let minus_one = _mm256_set1_ps(-1.0);
        avx2::map_ps(input, output, |v| {
            // `_CMP_GT_OQ`/`_CMP_LT_OQ` are *ordered*: both are false for NaN,
            // so a NaN lane selects neither `±1` nor `0` and `v` survives.
            let pos = _mm256_cmp_ps::<_CMP_GT_OQ>(v, zero);
            let neg = _mm256_cmp_ps::<_CMP_LT_OQ>(v, zero);
            // Zero (either sign) is ordered-equal to zero, so it takes this
            // arm and yields `+0.0`, which is what ONNX `Sign` specifies.
            let eq = _mm256_cmp_ps::<_CMP_EQ_OQ>(v, zero);
            let out = _mm256_blendv_ps(v, zero, eq);
            let out = _mm256_blendv_ps(out, one, pos);
            _mm256_blendv_ps(out, minus_one, neg)
        })
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn softsign_avx2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;
    unsafe {
        let one = _mm256_set1_ps(1.0);
        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
        avx2::map_ps(input, output, |v| {
            _mm256_div_ps(v, _mm256_add_ps(one, _mm256_and_ps(v, abs_mask)))
        })
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn erf_avx2(input: &[f32], output: &mut [f32]) {
    unsafe { avx2::map_ps(input, output, |v| avx2::erf_ps(v)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn erf_gelu_avx2(input: &[f32], output: &mut [f32]) {
    unsafe { avx2::map_ps(input, output, |v| avx2::erf_gelu_ps(v)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tanh_gelu_bias_avx2(input: &[f32], bias: &[f32], width: usize, output: &mut [f32]) {
    unsafe { avx2::map_bias_ps(input, bias, width, output, |v| avx2::tanh_gelu_ps(v)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn erf_gelu_bias_avx2(input: &[f32], bias: &[f32], width: usize, output: &mut [f32]) {
    unsafe { avx2::map_bias_ps(input, bias, width, output, |v| avx2::erf_gelu_ps(v)) }
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

    fn erf_ref(x: f32) -> f32 {
        if x.is_nan() {
            return f32::NAN;
        }
        libm::erf(f64::from(x)) as f32
    }

    fn erf_gelu_ref(x: f32) -> f32 {
        if x == f32::NEG_INFINITY {
            return 0.0;
        }
        let xf = f64::from(x);
        (0.5 * xf * (1.0 + libm::erf(xf * std::f64::consts::FRAC_1_SQRT_2))) as f32
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

    /// Documented bound: `|err| <= 3e-7` (output is in `[-1, 1]`). MLAS's own
    /// reference calls this polynomial "faithfully rounded", i.e. within one
    /// ulp of the correctly-rounded `f32` result; one ulp just below `1.0` is
    /// `5.96e-8`, so `3e-7` is a ~5x margin that still fails loudly if a
    /// coefficient is mistyped.
    const ERF_BOUND: f64 = 3e-7;
    /// Documented bound: `|err| <= 4e-7 * max(1, |x|)`, as for tanh GELU.
    const ERF_GELU_BOUND: f64 = 4e-7;

    #[test]
    fn erf_dense_sweep_matches_f64_reference() {
        // Both sides of the 0.921875 branch split, both sides of the 3.925
        // saturation clamp, and the near-zero region where the small
        // polynomial's leading term dominates.
        let extra = [
            0.921875,
            -0.921875,
            0.921875f32.next_up(),
            0.921875f32.next_down(),
            (-0.921875f32).next_up(),
            (-0.921875f32).next_down(),
            3.925,
            -3.925,
            3.925f32.next_up(),
            3.925f32.next_down(),
            1e-7,
            -1e-7,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
        ];
        let x = grid(-6.0, 6.0, 400_003, &extra);
        let mut out = vec![0.0f32; x.len()];
        erf_f32_slice(&x, &mut out);
        let worst = check(&x, &out, erf_ref, ERF_BOUND, "erf");
        eprintln!("erf worst scaled error: {worst:e}");
    }

    /// The interesting band is `|x| <= 1`, where `erf` is steep and the small
    /// polynomial runs; sample it far more densely than the wide sweep can.
    #[test]
    fn erf_dense_sweep_near_origin() {
        let x = grid(-1.5, 1.5, 400_003, &[]);
        let mut out = vec![0.0f32; x.len()];
        erf_f32_slice(&x, &mut out);
        let worst = check(&x, &out, erf_ref, ERF_BOUND, "erf near origin");
        eprintln!("erf near-origin worst scaled error: {worst:e}");
    }

    #[test]
    fn erf_gelu_dense_sweep_matches_f64_reference() {
        let x = grid(-25.0, 25.0, 400_003, &[]);
        let mut out = vec![0.0f32; x.len()];
        erf_gelu_f32_slice(&x, &mut out);
        let worst = check(&x, &out, erf_gelu_ref, ERF_GELU_BOUND, "erf_gelu");
        eprintln!("erf_gelu worst scaled error: {worst:e}");
    }

    /// `erf` saturates to exactly `±1` in `f32` well before the `3.925` clamp,
    /// so the clamp must not be observable: every input past it has to return
    /// exactly `±1.0`, not `1.0 - epsilon`.
    #[test]
    fn erf_saturates_to_exactly_one_past_the_clamp() {
        let mut x: Vec<f32> = vec![0.0; PAD];
        x.extend([
            3.925,
            3.93,
            4.0,
            5.0,
            10.0,
            1e10,
            f32::MAX,
            f32::INFINITY,
            -3.925,
            -3.93,
            -4.0,
            -5.0,
            -10.0,
            -1e10,
            f32::MIN,
            f32::NEG_INFINITY,
        ]);
        let mut out = vec![0.0f32; x.len()];
        erf_f32_slice(&x, &mut out);
        for (&v, &r) in x[PAD..].iter().zip(&out[PAD..]) {
            let want = if v > 0.0 { 1.0f32 } else { -1.0f32 };
            assert_eq!(
                r, want,
                "erf({v}) = {r}, expected exactly {want} (saturation is not exact)"
            );
        }
    }

    /// The scalar tail and the 8-lane body must agree, or a tensor's results
    /// would depend on its length modulo 8. `map_ps` routes the tail through
    /// the same kernel via a masked load/store, so this asserts bit equality
    /// rather than a tolerance.
    #[test]
    fn erf_tail_lanes_match_the_vector_body() {
        if !vector_path_available() {
            return;
        }
        let full: Vec<f32> = (0..SIMD_MIN_LEN + 8)
            .map(|i| -4.0 + 8.0 * (i as f32) / 39.0)
            .collect();
        let mut want = vec![0.0f32; full.len()];
        erf_f32_slice(&full, &mut want);
        for len in SIMD_MIN_LEN..full.len() {
            let mut got = vec![0.0f32; len];
            erf_f32_slice(&full[..len], &mut got);
            assert_eq!(got, want[..len], "erf length {len} disagrees with the body");
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

    #[test]
    fn erf_special_values() {
        let x = special_inputs();
        let mut o = vec![0.0f32; x.len()];
        erf_f32_slice(&x, &mut o);
        assert_eq!(o[PAD], -1.0, "erf(-Inf)");
        assert_eq!(
            o[PAD + 1].to_bits(),
            (-0.0f32).to_bits(),
            "erf(-0) keeps sign"
        );
        assert_eq!(o[PAD + 2].to_bits(), 0.0f32.to_bits(), "erf(+0) keeps sign");
        assert_eq!(o[PAD + 3], 1.0, "erf(+Inf)");
        assert!(o[PAD + 4].is_nan(), "erf(NaN)");
        assert_eq!(o[PAD + 5], -1.0, "erf(-MAX)");
        assert_eq!(o[PAD + 6], 1.0, "erf(MAX)");
        assert_eq!(o[PAD + 7], -1.0);
        assert_eq!(o[PAD + 8], 1.0);
    }

    #[test]
    fn erf_gelu_special_values() {
        let x = special_inputs();
        let mut o = vec![0.0f32; x.len()];
        erf_gelu_f32_slice(&x, &mut o);
        assert_eq!(o[PAD], 0.0, "erf_gelu(-Inf) is pinned to the limit 0");
        assert_eq!(o[PAD + 1].to_bits(), (-0.0f32).to_bits(), "erf_gelu(-0)");
        assert_eq!(o[PAD + 2].to_bits(), 0.0f32.to_bits(), "erf_gelu(+0)");
        assert_eq!(o[PAD + 3], f32::INFINITY, "erf_gelu(+Inf)");
        assert!(o[PAD + 4].is_nan(), "erf_gelu(NaN)");
        assert_eq!(o[PAD + 5], 0.0, "erf_gelu(-MAX)");
        assert_eq!(o[PAD + 6], f32::MAX, "erf_gelu(MAX)");
        assert_eq!(o[PAD + 7], 0.0);
        assert_eq!(o[PAD + 8], 1e30);
    }

    /// `erf` is odd. The sign is applied by an `or` at the very end of the
    /// kernel rather than being carried through the polynomial, so exact
    /// antisymmetry is a property worth pinning: any lane where it fails means
    /// the sign mask leaked into the arithmetic.
    #[test]
    fn erf_is_exactly_odd() {
        let pos: Vec<f32> = (0..4096)
            .map(|i| 1e-4 + 6.0 * (i as f32) / 4095.0)
            .collect();
        let neg: Vec<f32> = pos.iter().map(|v| -v).collect();
        let mut po = vec![0.0f32; pos.len()];
        let mut no = vec![0.0f32; neg.len()];
        erf_f32_slice(&pos, &mut po);
        erf_f32_slice(&neg, &mut no);
        for ((p, n), x) in po.iter().zip(&no).zip(&pos) {
            assert_eq!(*n, -*p, "erf({x}) and erf(-{x}) are not antisymmetric");
        }
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

#[cfg(test)]
mod exact_tests {
    use super::*;

    /// Every value a `f32` kernel can be asked about that is interesting to a
    /// sign-bit mask, a division, or a rounding mode.
    fn adversarial() -> Vec<f32> {
        let mut v = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            1.5,
            -1.5,
            2.5,
            -2.5,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::from_bits(0x8000_0001),
            f32::MAX,
            f32::MIN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
            f32::from_bits(0x7fc0_1234),
            f32::from_bits(0xffc0_1234),
            8_388_608.0,
            -8_388_608.0,
            16_777_216.0,
        ];
        let mut state = 0x1234_5678u32;
        for _ in 0..4096 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let f = f32::from_bits(state);
            if f.is_finite() {
                v.push(f);
            }
            v.push((state as i32 as f32) / 1024.0);
        }
        v
    }

    /// Bit-compare a vector kernel against its scalar reference over the
    /// adversarial set, at every length from 0 to `4 * SIMD_MIN_LEN` so the
    /// masked tail, the scalar-fallback seam and the aligned body are all hit.
    fn assert_bit_exact(
        name: &str,
        vector: impl Fn(&[f32], &mut [f32]),
        scalar: impl Fn(f32) -> f32,
    ) {
        let values = adversarial();
        for len in 0..=(4 * SIMD_MIN_LEN) {
            for start in [0usize, 1, 7, 13] {
                if start + len > values.len() {
                    continue;
                }
                let x = &values[start..start + len];
                let mut got = vec![0.0f32; len];
                vector(x, &mut got);
                for (i, (&g, &v)) in got.iter().zip(x).enumerate() {
                    let want = scalar(v);
                    assert_eq!(
                        g.to_bits(),
                        want.to_bits(),
                        "{name}: len={len} start={start} i={i} x={v:e} ({:#010x}): \
                         got {g:e} ({:#010x}), want {want:e} ({:#010x})",
                        v.to_bits(),
                        g.to_bits(),
                        want.to_bits()
                    );
                }
            }
        }
        // And once over the whole set, which is far longer than any threshold.
        let mut got = vec![0.0f32; values.len()];
        vector(&values, &mut got);
        for (i, (&g, &v)) in got.iter().zip(&values).enumerate() {
            assert_eq!(
                g.to_bits(),
                scalar(v).to_bits(),
                "{name}: full sweep i={i} x={v:e}"
            );
        }
    }

    #[test]
    fn neg_is_bit_exact() {
        assert_bit_exact("neg", neg_f32_slice, |v| -v);
    }

    #[test]
    fn abs_is_bit_exact() {
        assert_bit_exact("abs", abs_f32_slice, f32::abs);
    }

    #[test]
    fn reciprocal_is_bit_exact() {
        assert_bit_exact("reciprocal", reciprocal_f32_slice, |v| 1.0 / v);
    }

    #[test]
    fn floor_is_bit_exact() {
        assert_bit_exact("floor", floor_f32_slice, f32::floor);
    }

    #[test]
    fn ceil_is_bit_exact() {
        assert_bit_exact("ceil", ceil_f32_slice, f32::ceil);
    }

    #[test]
    fn round_is_bit_exact_and_ties_to_even() {
        assert_bit_exact(
            "round_ties_even",
            round_ties_even_f32_slice,
            f32::round_ties_even,
        );
        // The property that separates ONNX `Round` from `f32::round`: halves
        // go to the even neighbour, not away from zero. Long enough to be on
        // the vector path.
        let x: Vec<f32> = std::iter::repeat_n([-2.5f32, -1.5, -0.5, 0.5, 1.5, 2.5], 16)
            .flatten()
            .collect();
        let mut got = vec![0.0f32; x.len()];
        round_ties_even_f32_slice(&x, &mut got);
        for chunk in got.chunks(6) {
            assert_eq!(chunk, &[-2.0, -2.0, -0.0, 0.0, 2.0, 2.0]);
        }
    }

    #[test]
    fn sign_is_bit_exact_and_keeps_nan_payloads() {
        assert_bit_exact("sign", sign_f32_slice, sign_scalar);
        // Explicit: a NaN lane must come back as the *same* NaN, and both
        // zeros must come back as `+0.0`, on the vector path.
        let nan = f32::from_bits(0x7fc0_1234);
        let neg_nan = f32::from_bits(0xffc0_1234);
        let x: Vec<f32> = std::iter::repeat_n([nan, neg_nan, -0.0, 3.0, -3.0], 16)
            .flatten()
            .collect();
        let mut got = vec![0.0f32; x.len()];
        sign_f32_slice(&x, &mut got);
        for chunk in got.chunks(5) {
            // Matches ORT 1.28.0: the NaN comes back with its payload *and*
            // sign bit intact, not canonicalised.
            assert_eq!(chunk[0].to_bits(), nan.to_bits());
            assert_eq!(chunk[1].to_bits(), neg_nan.to_bits());
            assert_eq!(chunk[2].to_bits(), 0.0f32.to_bits());
            assert_eq!(chunk[3], 1.0);
            assert_eq!(chunk[4], -1.0);
        }
    }

    #[test]
    fn softsign_is_bit_exact() {
        assert_bit_exact("softsign", softsign_f32_slice, softsign_scalar);
    }

    /// The exact group must not have the `SIMD_MIN_LEN` accuracy seam the
    /// approximations do: a value's result cannot depend on how many elements
    /// share the tensor with it.
    /// Signalling NaN is the one input class where the scalar and AVX2 paths
    /// are **not** bit-identical, so it is deliberately excluded from
    /// [`adversarial`] and pinned here instead.
    ///
    /// The divergence is structural: `_mm256_round_ps`, `_mm256_div_ps` and
    /// the FMA in `softsign` are IEEE arithmetic and quiet a signalling NaN,
    /// while Rust's `f32::ceil`/`floor`/`round_ties_even` return the operand
    /// untouched. Neither is wrong -- ONNX does not specify NaN payload
    /// propagation -- and ORT is itself inconsistent across exactly the same
    /// split. Measured against ORT 1.28.0 CPU with input `0x7f801234`:
    ///
    /// | op | ORT | our scalar | our AVX2 |
    /// |----|-----|------------|----------|
    /// | `Ceil`       | quiets   | preserves | quiets    |
    /// | `Floor`      | quiets   | preserves | quiets    |
    /// | `Reciprocal` | quiets   | quiets    | quiets    |
    /// | `Softsign`   | quiets   | quiets    | quiets    |
    /// | `Round`      | preserves| preserves | quiets    |
    /// | `Sign`       | preserves| preserves | preserves |
    /// | `Neg`        | preserves| preserves | preserves |
    /// | `Abs`        | preserves| preserves | preserves |
    ///
    /// So on the vector path this EP matches ORT everywhere except `Round`,
    /// and on the scalar path it matches everywhere except `Ceil`/`Floor`.
    /// Closing the gap would need a per-element payload check that would cost
    /// the entire measured win, for an input class that does not occur in
    /// inference. This test exists so the behaviour cannot drift unnoticed.
    #[test]
    fn signalling_nan_behaviour_is_pinned() {
        const SNAN: u32 = 0x7f80_1234;
        const QUIETED: u32 = 0x7fc0_1234;
        let input = vec![f32::from_bits(SNAN); SIMD_MIN_LEN * 2];

        // Sign-bit-only kernels never touch the payload, on either path.
        for (name, kernel) in [
            ("sign", sign_f32_slice as fn(&[f32], &mut [f32])),
            ("neg", neg_f32_slice),
            ("abs", abs_f32_slice),
        ] {
            let mut out = vec![0.0; input.len()];
            kernel(&input, &mut out);
            assert_eq!(
                out[0].to_bits() & 0x7fff_ffff,
                SNAN,
                "{name} must leave a signalling NaN payload untouched",
            );
        }

        // Arithmetic kernels quiet it. Asserting this pins the behaviour that
        // `adversarial` cannot cover.
        for (name, kernel) in [
            ("ceil", ceil_f32_slice as fn(&[f32], &mut [f32])),
            ("floor", floor_f32_slice),
            ("round_ties_even", round_ties_even_f32_slice),
            ("reciprocal", reciprocal_f32_slice),
            ("softsign", softsign_f32_slice),
        ] {
            let mut out = vec![0.0; input.len()];
            kernel(&input, &mut out);
            assert!(
                out[0].is_nan(),
                "{name} of a signalling NaN must still be NaN",
            );
            if vector_path_available() {
                assert_eq!(
                    out[0].to_bits(),
                    QUIETED,
                    "{name} on the vector path is expected to quiet a signalling NaN",
                );
            }
        }
    }

    #[test]
    fn exact_kernels_have_no_length_seam() {
        let values = adversarial();
        type ExactKernel = fn(&[f32], &mut [f32]);
        let kernels: [(&str, ExactKernel); 8] = [
            ("neg", neg_f32_slice),
            ("abs", abs_f32_slice),
            ("reciprocal", reciprocal_f32_slice),
            ("floor", floor_f32_slice),
            ("ceil", ceil_f32_slice),
            ("round", round_ties_even_f32_slice),
            ("sign", sign_f32_slice),
            ("softsign", softsign_f32_slice),
        ];
        for (name, k) in kernels {
            // Below the threshold (scalar) and far above it (vector).
            let short = &values[..SIMD_MIN_LEN - 1];
            let mut got_short = vec![0.0f32; short.len()];
            k(short, &mut got_short);

            let long = &values[..SIMD_MIN_LEN * 8];
            let mut got_long = vec![0.0f32; long.len()];
            k(long, &mut got_long);

            for i in 0..short.len() {
                assert_eq!(
                    got_short[i].to_bits(),
                    got_long[i].to_bits(),
                    "{name}: element {i} (x={:e}) differs between a {}-element and a \
                     {}-element tensor — the exact group must have no length seam",
                    short[i],
                    short.len(),
                    long.len()
                );
            }
        }
    }
}

/// `Exp`'s vector path is a different approximation from `f32::exp`, so its
/// tests are error-bounded and special-value-pinned rather than bit-exact.
#[cfg(test)]
mod exp_tests {
    use super::*;

    /// Long enough to reach the vector path on every host that has one.
    const N: usize = 4096;

    fn vector_exp(values: &[f32]) -> Vec<f32> {
        assert!(
            values.len() >= SIMD_MIN_LEN,
            "input must be long enough to reach the vector path"
        );
        let mut out = vec![0.0f32; values.len()];
        exp_f32_slice(values, &mut out);
        out
    }

    /// Error against an `f64` reference over a dense sweep of the whole
    /// argument range, measured in ulp rather than relative error.
    ///
    /// Relative error is the wrong metric near the bottom of the range: below
    /// about `-87` the result is subnormal and carries only a handful of
    /// significand bits, so a *correctly rounded* answer can already be tens of
    /// percent away from the real value. Comparing representations counts what
    /// actually matters — how many representable `f32` steps separate us from
    /// the best possible answer — and stays meaningful through the subnormal
    /// range and at the overflow boundary.
    #[test]
    fn dense_sweep_stays_within_two_ulp_of_a_f64_reference() {
        let mut x = Vec::with_capacity(1 << 16);
        let (lo, hi) = (-110.0f64, 89.0f64);
        for i in 0..(1 << 16) {
            x.push((lo + (hi - lo) * (i as f64) / ((1 << 16) as f64 - 1.0)) as f32);
        }
        let got = vector_exp(&x);

        let mut worst = 0i64;
        let mut worst_at = 0.0f32;
        for (&v, &g) in x.iter().zip(&got) {
            let want = f64::from(v).exp() as f32;
            // Both are non-negative, so the bit patterns are monotonic in value
            // and their difference is exactly the number of representable steps
            // between them — including across the subnormal/normal boundary.
            let ulp = i64::from(g.to_bits()) - i64::from(want.to_bits());
            if ulp.abs() > worst {
                worst = ulp.abs();
                worst_at = v;
            }
        }
        assert!(
            worst <= 2,
            "worst error {worst} ulp at x={worst_at} (exp = {})",
            f64::from(worst_at).exp()
        );
    }

    /// The seam between the vector path and the correctly-rounded scalar
    /// fallback is allowed to move a result by a bounded amount, never more.
    #[test]
    fn vector_and_scalar_paths_agree_to_two_ulp() {
        let mut x = Vec::with_capacity(N);
        let mut state = 0x1234_5678u32;
        for _ in 0..N {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            x.push((state >> 8) as f32 / (1 << 24) as f32 * 60.0 - 30.0);
        }
        let vector = vector_exp(&x);
        for (&v, &g) in x.iter().zip(&vector) {
            let want = v.exp();
            let rel = ((f64::from(g) - f64::from(want)) / f64::from(want)).abs();
            assert!(
                rel <= 2.0 * f64::from(f32::EPSILON),
                "exp({v}): vector {g} vs scalar {want} (rel {rel:e})"
            );
        }
    }

    /// The reconstruction reads the biased exponent as an integer, so every
    /// saturating and non-finite argument has to be pinned explicitly. These
    /// answers were read off ORT 1.28.0's own CPU `Exp` kernel.
    #[test]
    fn special_values_match_ort() {
        let cases: [(f32, f32); 12] = [
            (f32::INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, 0.0),
            (100.0, f32::INFINITY),
            (89.0, f32::INFINITY),
            (88.7762626647950, f32::INFINITY),
            (88.3762626647949, 2.4061436e38),
            (-87.0, 1.6458115e-38),
            (-200.0, 0.0),
            (0.0, 1.0),
            (-0.0, 1.0),
            (1.0, std::f32::consts::E),
            (-1.0, 0.36787945),
        ];
        let mut x = vec![0.0f32; N];
        for (i, slot) in x.iter_mut().enumerate() {
            *slot = cases[i % cases.len()].0;
        }
        let got = vector_exp(&x);
        for (i, &g) in got.iter().enumerate() {
            let (arg, want) = cases[i % cases.len()];
            if want.is_infinite() || want == 0.0 || want == 1.0 {
                assert_eq!(g, want, "exp({arg}) = {g}, expected {want}");
            } else {
                let rel = ((f64::from(g) - f64::from(want)) / f64::from(want)).abs();
                assert!(
                    rel <= 2.0 * f64::from(f32::EPSILON),
                    "exp({arg}) = {g}, expected {want}"
                );
            }
        }
    }

    /// `NaN` survives the exponent reconstruction only because of the clamp's
    /// operand order — see [`avx2::exp_full_ps`]. Nothing in the arithmetic
    /// makes that obvious, so it gets its own test, across both signs and both
    /// quiet and signalling payloads.
    #[test]
    fn nan_propagates_instead_of_becoming_finite() {
        let payloads = [
            f32::NAN.to_bits(),
            0x7FC0_1234,
            0xFFC0_1234,
            0x7F80_0001, // signalling
        ];
        let mut x = vec![0.0f32; N];
        for (i, slot) in x.iter_mut().enumerate() {
            *slot = f32::from_bits(payloads[i % payloads.len()]);
        }
        let got = vector_exp(&x);
        for (i, &g) in got.iter().enumerate() {
            assert!(
                g.is_nan(),
                "exp(NaN 0x{:08X}) at {i} produced {g}",
                payloads[i % payloads.len()]
            );
        }
    }

    /// The tail is processed through the same kernel via a masked load, so a
    /// non-multiple-of-8 length must not change any answer.
    #[test]
    fn tail_lengths_are_computed_identically() {
        let base: Vec<f32> = (0..(SIMD_MIN_LEN + 8))
            .map(|i| (i as f32) * 0.37 - 6.0)
            .collect();
        let full = vector_exp(&base);
        for n in SIMD_MIN_LEN..base.len() {
            let mut out = vec![0.0f32; n];
            exp_f32_slice(&base[..n], &mut out);
            for (i, (&g, &f)) in out.iter().zip(&full).enumerate() {
                assert_eq!(g.to_bits(), f.to_bits(), "n={n} i={i}");
            }
        }
    }

    /// `y = exp(y)` is a legal graph; the kernel must tolerate one buffer.
    #[test]
    fn aliased_input_and_output_are_supported_by_the_slice_form() {
        let mut buf: Vec<f32> = (0..N).map(|i| (i as f32) * 0.001 - 2.0).collect();
        let want: Vec<f32> = {
            let mut o = vec![0.0f32; N];
            exp_f32_slice(&buf, &mut o);
            o
        };
        let copy = buf.clone();
        exp_f32_slice(&copy, &mut buf);
        for (i, (&g, &w)) in buf.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "i={i}");
        }
    }
}

/// Falsifiers for the saturation removed from `tanh_ps` / `sigmoid_ps`.
///
/// Both kernels used to follow the `[-1, 1]` / `[0, 1]` clamp with a pair of
/// `vcmpps` + `vblendvps` that forced the saturated constant for `|x|` beyond
/// the rational's clamp range. That step is redundant: the input clamp means
/// the rational never sees an argument past `±9` / `±18`, and at those points
/// the result is already outside the output clamp, so the clamp alone
/// saturates. These tests keep the old sequence as a reference and assert the
/// current kernels reproduce it bit for bit.
#[cfg(all(test, target_arch = "x86_64"))]
mod saturation_absorption {
    use super::*;
    use std::arch::x86_64::*;

    /// `tanh_ps` as it was written before the blends were removed.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn tanh_ps_with_blend(x: __m256) -> __m256 {
        unsafe {
            let poly = avx2::tanh_ps(x);
            let above = _mm256_cmp_ps(x, _mm256_set1_ps(tanh_c::UPPER), _CMP_GT_OQ);
            let below = _mm256_cmp_ps(x, _mm256_set1_ps(tanh_c::LOWER), _CMP_LT_OQ);
            let r = _mm256_blendv_ps(poly, _mm256_set1_ps(1.0), above);
            _mm256_blendv_ps(r, _mm256_set1_ps(-1.0), below)
        }
    }

    /// `sigmoid_ps` as it was written before the blends were removed.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn sigmoid_ps_with_blend(x: __m256) -> __m256 {
        unsafe {
            let poly = avx2::sigmoid_ps(x);
            let above = _mm256_cmp_ps(x, _mm256_set1_ps(logistic_c::UPPER), _CMP_GT_OQ);
            let below = _mm256_cmp_ps(x, _mm256_set1_ps(logistic_c::LOWER), _CMP_LT_OQ);
            let r = _mm256_blendv_ps(poly, _mm256_set1_ps(1.0), above);
            _mm256_blendv_ps(r, _mm256_set1_ps(0.0), below)
        }
    }

    /// Runs both forms over one vector of inputs and returns the two results.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn pair(
        lean: unsafe fn(__m256) -> __m256,
        reference: unsafe fn(__m256) -> __m256,
        input: &[f32; 8],
    ) -> ([f32; 8], [f32; 8]) {
        unsafe {
            let v = _mm256_loadu_ps(input.as_ptr());
            let mut a = [0.0f32; 8];
            let mut b = [0.0f32; 8];
            _mm256_storeu_ps(a.as_mut_ptr(), lean(v));
            _mm256_storeu_ps(b.as_mut_ptr(), reference(v));
            (a, b)
        }
    }

    fn assert_same(
        lean: unsafe fn(__m256) -> __m256,
        reference: unsafe fn(__m256) -> __m256,
        input: &[f32; 8],
        what: &str,
    ) {
        let (a, b) = unsafe { pair(lean, reference, input) };
        for i in 0..8 {
            assert_eq!(
                a[i].to_bits(),
                b[i].to_bits(),
                "{what}({:e}): without blend {:e} ({:08x}), with blend {:e} ({:08x})",
                input[i],
                a[i],
                a[i].to_bits(),
                b[i],
                b[i].to_bits(),
            );
        }
    }

    /// Walks every `f32` in `[lo, hi]` by ULP, eight at a time, plus the
    /// negation of each.
    fn sweep(
        lean: unsafe fn(__m256) -> __m256,
        reference: unsafe fn(__m256) -> __m256,
        lo: f32,
        hi: f32,
        what: &str,
    ) -> u64 {
        let (mut bits, end) = (lo.to_bits(), hi.to_bits());
        let mut seen = 0u64;
        while bits <= end {
            let mut pos = [0.0f32; 8];
            let mut neg = [0.0f32; 8];
            for slot in 0..8 {
                let b = (bits + slot as u32).min(end);
                pos[slot] = f32::from_bits(b);
                neg[slot] = -pos[slot];
            }
            assert_same(lean, reference, &pos, what);
            assert_same(lean, reference, &neg, what);
            seen += 8;
            bits += 8;
        }
        seen
    }

    fn avx2() -> bool {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }

    #[test]
    fn tanh_saturation_blend_is_redundant_near_the_boundary() {
        if !avx2() {
            return;
        }
        // Dense over the decade above the clamp, where an overshoot would show.
        let n = sweep(avx2::tanh_ps, tanh_ps_with_blend, 9.0, 128.0, "tanh");
        assert!(n > 32_000_000, "sweep covered only {n} values");
    }

    #[test]
    fn sigmoid_saturation_blend_is_redundant_near_the_boundary() {
        if !avx2() {
            return;
        }
        let n = sweep(
            avx2::sigmoid_ps,
            sigmoid_ps_with_blend,
            18.0,
            256.0,
            "sigmoid",
        );
        assert!(n > 20_000_000, "sweep covered only {n} values");
    }

    #[test]
    fn saturation_blend_is_redundant_for_special_values() {
        if !avx2() {
            return;
        }
        let specials = [
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
            f32::MAX,
            f32::MIN,
            1e30,
            -1e30,
        ];
        assert_same(avx2::tanh_ps, tanh_ps_with_blend, &specials, "tanh");
        assert_same(
            avx2::sigmoid_ps,
            sigmoid_ps_with_blend,
            &specials,
            "sigmoid",
        );

        // Exactly at, and one ULP either side of, both clamp boundaries.
        for &edge in &[tanh_c::LOWER, tanh_c::UPPER] {
            let e = edge.to_bits();
            let around = [
                f32::from_bits(e - 2),
                f32::from_bits(e - 1),
                edge,
                f32::from_bits(e + 1),
                f32::from_bits(e + 2),
                edge,
                edge,
                edge,
            ];
            assert_same(avx2::tanh_ps, tanh_ps_with_blend, &around, "tanh");
        }
        for &edge in &[logistic_c::LOWER, logistic_c::UPPER] {
            let e = edge.to_bits();
            let around = [
                f32::from_bits(e - 2),
                f32::from_bits(e - 1),
                edge,
                f32::from_bits(e + 1),
                f32::from_bits(e + 2),
                edge,
                edge,
                edge,
            ];
            assert_same(avx2::sigmoid_ps, sigmoid_ps_with_blend, &around, "sigmoid");
        }
    }

    /// The complete proof: every finite `f32` past the clamp, both signs.
    /// 1 047 527 424 values for `tanh` and 1 039 138 816 for `sigmoid`, which
    /// is minutes in an unoptimised test build, so it is not run by default.
    #[test]
    #[ignore = "exhaustive; run with --ignored"]
    fn saturation_blend_is_redundant_exhaustively() {
        if !avx2() {
            return;
        }
        sweep(avx2::tanh_ps, tanh_ps_with_blend, 9.0, f32::MAX, "tanh");
        sweep(
            avx2::sigmoid_ps,
            sigmoid_ps_with_blend,
            18.0,
            f32::MAX,
            "sigmoid",
        );
    }
}
