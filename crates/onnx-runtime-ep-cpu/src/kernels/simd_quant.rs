//! Vectorized per-block activation quantizers for the MatMulNBits decode path.
//!
//! Before each int4/int8 `MatMulNBits` GEMV on the M=1 decode step, the f32
//! activation row is quantized per K-block: symmetric int8 (feeding the VNNI
//! `u8 x i8` / `i8 x i8` dot products) or per-group int16 (feeding
//! `block_dot_u8_i16`). Those quantizers were ordinary scalar Rust loops that
//! LLVM only partially auto-vectorized (128-bit SSE for the max reduction, no
//! `ymm`/`zmm`), so they were the scalar straggler feeding the already-AVX-512
//! dot products. This module provides AVX-512 implementations, runtime-gated on
//! feature detection, with the exact scalar loops preserved as the portable
//! fallback (used on non-AVX-512 x86 and on aarch64).
//!
//! Numerics — **bit-identical** to the scalar path (this is stricter than the
//! f64-parity bar used for float reductions, and is achievable because the
//! output is integer codes):
//!   * The per-block `max_abs` scale is a `max` reduction. `max` is associative
//!     and commutative for the finite activation values here (each partial max
//!     is exactly one of the inputs), so the lane-parallel reduction returns the
//!     identical f32 bit pattern as the serial scalar fold — hence an identical
//!     `scale` and `inverse_scale`.
//!   * Rust's `f32::round` is round-half-away-from-zero. The AVX-512 path
//!     reproduces it exactly with `trunc(x) + copysign(1.0, x)` when
//!     `|x - trunc(x)| >= 0.5` (verified bit-identical to `f32::round` across
//!     the full f32 range that quantization can reach).
//!   * The clamp bounds, the int cast (truncation of an already-integer f32),
//!     and the unsigned `+128` offset all match the scalar code exactly.
//!
//! The unit tests assert bit-identical output between the SIMD and scalar paths
//! across random and edge-case inputs (zeros, saturating values, mixed signs,
//! block-boundary lengths), and only exercise the SIMD path when the host
//! actually supports the required features.

/// Symmetric int8 activation quantization of one K-block.
///
/// Writes `out[i] = round(src[i] * 127 / max_abs)` clamped to `[-127, 127]`,
/// returning `scale = max_abs / 127`. An all-zero block yields scale `0` and
/// all-zero codes. Dispatches to AVX-512 when the host supports it.
#[inline]
pub(crate) fn quantize_block_i8(src: &[f32], out: &mut [i8]) -> f32 {
    debug_assert_eq!(src.len(), out.len());
    #[cfg(target_arch = "x86_64")]
    {
        if have_avx512_quant() {
            // SAFETY: guarded by runtime avx512f+bw+vl detection; slice lengths
            // are asserted equal so every store stays in bounds.
            return unsafe { quantize_block_i8_avx512(src, out) };
        }
    }
    quantize_block_i8_scalar(src, out)
}

/// Symmetric int8 activation quantization stored offset by 128 (unsigned) so a
/// VNNI `u8 x i8` dot can consume it.
///
/// Writes `out[i] = round(src[i] * 127 / max_abs) + 128` (the int8 code offset
/// into `[1, 255]`), returning `scale = max_abs / 127`. An all-zero block yields
/// scale `0` and all-128 codes. Dispatches to AVX-512 when supported.
#[inline]
pub(crate) fn quantize_block_u8_offset(src: &[f32], out: &mut [u8]) -> f32 {
    debug_assert_eq!(src.len(), out.len());
    #[cfg(target_arch = "x86_64")]
    {
        if have_avx512_quant() {
            // SAFETY: guarded by runtime avx512f+bw+vl detection; slice lengths
            // are asserted equal so every store stays in bounds.
            return unsafe { quantize_block_u8_offset_avx512(src, out) };
        }
    }
    quantize_block_u8_offset_scalar(src, out)
}

/// Symmetric int16 activation quantization of one group.
///
/// Writes `out[i] = round(src[i] * 32767 / max_abs)` clamped to
/// `[-32767, 32767]`, returning `scale = max_abs / 32767`. An all-zero group
/// yields scale `0` and all-zero codes. Dispatches to AVX-512 when supported.
#[inline]
pub(crate) fn quantize_block_i16(src: &[f32], out: &mut [i16]) -> f32 {
    debug_assert_eq!(src.len(), out.len());
    #[cfg(target_arch = "x86_64")]
    {
        if have_avx512_quant() {
            // SAFETY: guarded by runtime avx512f+bw+vl detection; slice lengths
            // are asserted equal so every store stays in bounds.
            return unsafe { quantize_block_i16_avx512(src, out) };
        }
    }
    quantize_block_i16_scalar(src, out)
}

/// Runtime detection of the AVX-512 subset the quantizers use, cached.
///
/// Requires AVX512F (the f32 vector math and float→int conversions), AVX512BW +
/// AVX512VL (the narrowing byte/word conversions and 128/256-bit stores).
#[cfg(target_arch = "x86_64")]
fn have_avx512_quant() -> bool {
    use std::sync::OnceLock;
    static SUPPORTED: OnceLock<bool> = OnceLock::new();
    *SUPPORTED.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vl")
    })
}

/// Serial `max(|src|)` fold, initialized at 0.0 (matches the scalar quantizers).
fn max_abs_scalar(src: &[f32]) -> f32 {
    src.iter().map(|value| value.abs()).fold(0.0f32, f32::max)
}

fn quantize_block_i8_scalar(src: &[f32], out: &mut [i8]) -> f32 {
    let max_abs = max_abs_scalar(src);
    if max_abs == 0.0 {
        out.fill(0);
        return 0.0;
    }
    let scale = max_abs / 127.0;
    let inverse_scale = 127.0 / max_abs;
    for (code, &value) in out.iter_mut().zip(src) {
        *code = (value * inverse_scale).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

fn quantize_block_u8_offset_scalar(src: &[f32], out: &mut [u8]) -> f32 {
    let max_abs = max_abs_scalar(src);
    if max_abs == 0.0 {
        out.fill(128);
        return 0.0;
    }
    let scale = max_abs / 127.0;
    let inverse_scale = 127.0 / max_abs;
    for (code, &value) in out.iter_mut().zip(src) {
        let signed = (value * inverse_scale).round().clamp(-127.0, 127.0) as i8;
        *code = (signed as i16 + 128) as u8;
    }
    scale
}

fn quantize_block_i16_scalar(src: &[f32], out: &mut [i16]) -> f32 {
    let max_abs = max_abs_scalar(src);
    if max_abs == 0.0 {
        out.fill(0);
        return 0.0;
    }
    let scale = max_abs / 32767.0;
    let inverse_scale = 32767.0 / max_abs;
    for (code, &value) in out.iter_mut().zip(src) {
        *code = (value * inverse_scale).round().clamp(-32767.0, 32767.0) as i16;
    }
    scale
}

/// AVX-512 `max(|src|)` reduction, initialized at 0.0.
///
/// Bit-identical to [`max_abs_scalar`]: `max` returns exactly one of its inputs,
/// so the lane-parallel reduction and the serial fold agree bit-for-bit for the
/// finite activation values here.
///
/// # Safety
/// The host must support `avx512f`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn max_abs_avx512(src: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    // SAFETY: the caller guarantees `avx512f`. Full 16-wide loads are bounded by
    // `i + 16 <= n`; the remainder folds in with a scalar `max`.
    unsafe {
        let n = src.len();
        let ptr = src.as_ptr();
        let sign = _mm512_set1_ps(-0.0);
        let mut acc = _mm512_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            let value = _mm512_loadu_ps(ptr.add(i));
            let magnitude = _mm512_andnot_ps(sign, value);
            acc = _mm512_max_ps(acc, magnitude);
            i += 16;
        }
        let mut max_abs = _mm512_reduce_max_ps(acc);
        while i < n {
            max_abs = max_abs.max((*ptr.add(i)).abs());
            i += 1;
        }
        max_abs
    }
}

/// Round `values` half-away-from-zero to integers, reproducing `f32::round`.
///
/// `trunc(x) + copysign(1.0, x)` when `|x - trunc(x)| >= 0.5`, else `trunc(x)`.
///
/// # Safety
/// The host must support `avx512f`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn round_half_away_avx512(
    values: std::arch::x86_64::__m512,
) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
    // SAFETY: the caller guarantees `avx512f`.
    unsafe {
        let sign = _mm512_set1_ps(-0.0);
        let one = _mm512_set1_ps(1.0);
        let half = _mm512_set1_ps(0.5);
        // Round toward zero (truncate), suppressing precision exceptions.
        let truncated = _mm512_roundscale_ps::<0x0B>(values);
        let fraction = _mm512_sub_ps(values, truncated);
        let abs_fraction = _mm512_andnot_ps(sign, fraction);
        let at_or_past_half = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(abs_fraction, half);
        // copysign(1.0, values): borrow the sign bit of the original value.
        let signed_one = _mm512_or_ps(_mm512_and_ps(values, sign), one);
        let bump = _mm512_maskz_mov_ps(at_or_past_half, signed_one);
        _mm512_add_ps(truncated, bump)
    }
}

/// AVX-512 symmetric int8 quantization of one K-block. See [`quantize_block_i8`].
///
/// # Safety
/// The host must support `avx512f`, `avx512bw`, `avx512vl`; `src.len() ==
/// out.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn quantize_block_i8_avx512(src: &[f32], out: &mut [i8]) -> f32 {
    use std::arch::x86_64::*;
    // SAFETY: the caller guarantees the required features and equal lengths.
    // Full 16-wide stores are bounded by `i + 16 <= n`; the remainder uses the
    // scalar per-element formula.
    unsafe {
        let n = src.len();
        let max_abs = max_abs_avx512(src);
        if max_abs == 0.0 {
            out.fill(0);
            return 0.0;
        }
        let scale = max_abs / 127.0;
        let inverse_scale = 127.0 / max_abs;
        let inverse = _mm512_set1_ps(inverse_scale);
        let lower = _mm512_set1_ps(-127.0);
        let upper = _mm512_set1_ps(127.0);
        let src_ptr = src.as_ptr();
        let out_ptr = out.as_mut_ptr();
        let mut i = 0;
        while i + 16 <= n {
            let value = _mm512_loadu_ps(src_ptr.add(i));
            let scaled = _mm512_mul_ps(value, inverse);
            let rounded = round_half_away_avx512(scaled);
            let clamped = _mm512_min_ps(_mm512_max_ps(rounded, lower), upper);
            let as_i32 = _mm512_cvttps_epi32(clamped);
            let as_i8 = _mm512_cvtepi32_epi8(as_i32);
            _mm_storeu_si128(out_ptr.add(i).cast(), as_i8);
            i += 16;
        }
        while i < n {
            *out_ptr.add(i) = (*src_ptr.add(i) * inverse_scale).round().clamp(-127.0, 127.0) as i8;
            i += 1;
        }
        scale
    }
}

/// AVX-512 int8-offset-by-128 quantization. See [`quantize_block_u8_offset`].
///
/// # Safety
/// The host must support `avx512f`, `avx512bw`, `avx512vl`; `src.len() ==
/// out.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn quantize_block_u8_offset_avx512(src: &[f32], out: &mut [u8]) -> f32 {
    use std::arch::x86_64::*;
    // SAFETY: the caller guarantees the required features and equal lengths.
    unsafe {
        let n = src.len();
        let max_abs = max_abs_avx512(src);
        if max_abs == 0.0 {
            out.fill(128);
            return 0.0;
        }
        let scale = max_abs / 127.0;
        let inverse_scale = 127.0 / max_abs;
        let inverse = _mm512_set1_ps(inverse_scale);
        let lower = _mm512_set1_ps(-127.0);
        let upper = _mm512_set1_ps(127.0);
        let offset = _mm512_set1_epi32(128);
        let src_ptr = src.as_ptr();
        let out_ptr = out.as_mut_ptr();
        let mut i = 0;
        while i + 16 <= n {
            let value = _mm512_loadu_ps(src_ptr.add(i));
            let scaled = _mm512_mul_ps(value, inverse);
            let rounded = round_half_away_avx512(scaled);
            let clamped = _mm512_min_ps(_mm512_max_ps(rounded, lower), upper);
            let as_i32 = _mm512_cvttps_epi32(clamped);
            let offset_i32 = _mm512_add_epi32(as_i32, offset);
            let as_u8 = _mm512_cvtepi32_epi8(offset_i32);
            _mm_storeu_si128(out_ptr.add(i).cast(), as_u8);
            i += 16;
        }
        while i < n {
            let signed =
                (*src_ptr.add(i) * inverse_scale).round().clamp(-127.0, 127.0) as i8;
            *out_ptr.add(i) = (signed as i16 + 128) as u8;
            i += 1;
        }
        scale
    }
}

/// AVX-512 symmetric int16 quantization of one group. See [`quantize_block_i16`].
///
/// # Safety
/// The host must support `avx512f`, `avx512bw`, `avx512vl`; `src.len() ==
/// out.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn quantize_block_i16_avx512(src: &[f32], out: &mut [i16]) -> f32 {
    use std::arch::x86_64::*;
    // SAFETY: the caller guarantees the required features and equal lengths.
    unsafe {
        let n = src.len();
        let max_abs = max_abs_avx512(src);
        if max_abs == 0.0 {
            out.fill(0);
            return 0.0;
        }
        let scale = max_abs / 32767.0;
        let inverse_scale = 32767.0 / max_abs;
        let inverse = _mm512_set1_ps(inverse_scale);
        let lower = _mm512_set1_ps(-32767.0);
        let upper = _mm512_set1_ps(32767.0);
        let src_ptr = src.as_ptr();
        let out_ptr = out.as_mut_ptr();
        let mut i = 0;
        while i + 16 <= n {
            let value = _mm512_loadu_ps(src_ptr.add(i));
            let scaled = _mm512_mul_ps(value, inverse);
            let rounded = round_half_away_avx512(scaled);
            let clamped = _mm512_min_ps(_mm512_max_ps(rounded, lower), upper);
            let as_i32 = _mm512_cvttps_epi32(clamped);
            let as_i16 = _mm512_cvtepi32_epi16(as_i32);
            _mm256_storeu_si256(out_ptr.add(i).cast(), as_i16);
            i += 16;
        }
        while i < n {
            *out_ptr.add(i) =
                (*src_ptr.add(i) * inverse_scale).round().clamp(-32767.0, 32767.0) as i16;
            i += 1;
        }
        scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random activations with occasional large outliers,
    /// matching the realistic magnitudes exercised by the RMS SIMD tests.
    fn make_row(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = ((state >> 11) as f64 / (1u64 << 53) as f64) as f32;
                let base = (unit - 0.5) * 6.0;
                if state.is_multiple_of(97) {
                    base * 20.0
                } else {
                    base
                }
            })
            .collect()
    }

    /// Lengths spanning block/group sizes (32, 128), sub-16 tails, and the 0.6B
    /// hidden size, including non-multiples of 16 to exercise the scalar tail.
    const LENGTHS: [usize; 16] =
        [1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 48, 64, 127, 128, 129, 1024];

    #[test]
    fn quantize_block_i8_simd_matches_scalar_bit_identical() {
        for &len in &LENGTHS {
            for seed in 0..16u64 {
                let row = make_row(len, seed);
                let mut simd = vec![0i8; len];
                let mut scalar = vec![7i8; len];
                let scale_simd = quantize_block_i8(&row, &mut simd);
                let scale_scalar = quantize_block_i8_scalar(&row, &mut scalar);
                assert_eq!(scale_simd.to_bits(), scale_scalar.to_bits(), "len {len} seed {seed}");
                assert_eq!(simd, scalar, "len {len} seed {seed}");
            }
        }
    }

    #[test]
    fn quantize_block_u8_offset_simd_matches_scalar_bit_identical() {
        for &len in &LENGTHS {
            for seed in 0..16u64 {
                let row = make_row(len, seed);
                let mut simd = vec![0u8; len];
                let mut scalar = vec![7u8; len];
                let scale_simd = quantize_block_u8_offset(&row, &mut simd);
                let scale_scalar = quantize_block_u8_offset_scalar(&row, &mut scalar);
                assert_eq!(scale_simd.to_bits(), scale_scalar.to_bits(), "len {len} seed {seed}");
                assert_eq!(simd, scalar, "len {len} seed {seed}");
            }
        }
    }

    #[test]
    fn quantize_block_i16_simd_matches_scalar_bit_identical() {
        for &len in &LENGTHS {
            for seed in 0..16u64 {
                let row = make_row(len, seed);
                let mut simd = vec![0i16; len];
                let mut scalar = vec![7i16; len];
                let scale_simd = quantize_block_i16(&row, &mut simd);
                let scale_scalar = quantize_block_i16_scalar(&row, &mut scalar);
                assert_eq!(scale_simd.to_bits(), scale_scalar.to_bits(), "len {len} seed {seed}");
                assert_eq!(simd, scalar, "len {len} seed {seed}");
            }
        }
    }

    /// Edge cases: all zeros, saturating magnitudes, mixed signs, exact halves
    /// at the quantization boundary, and a single large outlier per block.
    #[test]
    fn quantize_edge_cases_bit_identical() {
        let mut rows: Vec<Vec<f32>> = Vec::new();
        rows.push(vec![0.0; 40]);
        rows.push(vec![-0.0; 40]);
        rows.push(vec![5.0; 40]);
        rows.push(vec![-5.0; 40]);
        // Mixed signs with the max at both ends.
        rows.push((0..40).map(|i| if i % 2 == 0 { 3.0 } else { -3.0 }).collect());
        // A single outlier forcing tiny codes elsewhere.
        {
            let mut row = vec![0.01f32; 40];
            row[19] = 100.0;
            row[20] = -100.0;
            rows.push(row);
        }
        // Values engineered to land on exact .5 codes: value = max_abs * k.5 / 127.
        {
            let max_abs = 127.0f32;
            let mut row = vec![0.0f32; 40];
            for (i, value) in row.iter_mut().enumerate() {
                *value = ((i as f32) - 20.0) + 0.5; // ..., -0.5, 0.5, 1.5, ...
            }
            row[0] = max_abs; // pin the block scale to 1.0
            rows.push(row);
        }

        for row in &rows {
            let mut i8_simd = vec![0i8; row.len()];
            let mut i8_scalar = vec![0i8; row.len()];
            assert_eq!(
                quantize_block_i8(row, &mut i8_simd).to_bits(),
                quantize_block_i8_scalar(row, &mut i8_scalar).to_bits()
            );
            assert_eq!(i8_simd, i8_scalar);

            let mut u8_simd = vec![0u8; row.len()];
            let mut u8_scalar = vec![0u8; row.len()];
            assert_eq!(
                quantize_block_u8_offset(row, &mut u8_simd).to_bits(),
                quantize_block_u8_offset_scalar(row, &mut u8_scalar).to_bits()
            );
            assert_eq!(u8_simd, u8_scalar);

            let mut i16_simd = vec![0i16; row.len()];
            let mut i16_scalar = vec![0i16; row.len()];
            assert_eq!(
                quantize_block_i16(row, &mut i16_simd).to_bits(),
                quantize_block_i16_scalar(row, &mut i16_scalar).to_bits()
            );
            assert_eq!(i16_simd, i16_scalar);
        }
    }
}
