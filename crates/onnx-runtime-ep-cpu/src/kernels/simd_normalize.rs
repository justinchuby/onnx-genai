//! Vectorized elementwise normalize-and-scale for RMS normalization kernels.

pub(crate) fn scale_shape_is_exact_identity(
    input_shape: &[usize],
    normalization_axis: usize,
    scale_shape: &[usize],
) -> bool {
    scale_shape == &input_shape[normalization_axis..]
}

/// Writes `(input[index] * inverse_rms) * scale[index]` for each element.
///
/// The two multiplications remain separate and in scalar evaluation order so
/// vectorized results are bit-identical to the portable fallback.
#[inline]
pub(crate) fn normalize_and_scale(
    input: &[f32],
    output: &mut [f32],
    inverse_rms: f32,
    scale: &[f32],
) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(input.len(), scale.len());

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: guarded by runtime AVX-512F detection above.
            unsafe {
                normalize_and_scale_avx512(input, output, inverse_rms, scale);
            }
            return;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime AVX2 detection above.
            unsafe {
                normalize_and_scale_avx2(input, output, inverse_rms, scale);
            }
            return;
        }
    }
    normalize_and_scale_scalar(input, output, inverse_rms, scale);
}

fn normalize_and_scale_scalar(input: &[f32], output: &mut [f32], inverse_rms: f32, scale: &[f32]) {
    for index in 0..input.len() {
        output[index] = input[index] * inverse_rms * scale[index];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn normalize_and_scale_avx512(
    input: &[f32],
    output: &mut [f32],
    inverse_rms: f32,
    scale: &[f32],
) {
    use std::arch::x86_64::*;

    // SAFETY: the caller guarantees AVX-512F support and equal slice lengths.
    // Full-width loads and stores are bounded by `input.len()`.
    unsafe {
        let inverse_rms_vector = _mm512_set1_ps(inverse_rms);
        let mut index = 0;
        while index + 16 <= input.len() {
            let normalized = _mm512_mul_ps(
                _mm512_loadu_ps(input.as_ptr().add(index)),
                inverse_rms_vector,
            );
            let scaled = _mm512_mul_ps(normalized, _mm512_loadu_ps(scale.as_ptr().add(index)));
            _mm512_storeu_ps(output.as_mut_ptr().add(index), scaled);
            index += 16;
        }
        normalize_and_scale_scalar(
            &input[index..],
            &mut output[index..],
            inverse_rms,
            &scale[index..],
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn normalize_and_scale_avx2(
    input: &[f32],
    output: &mut [f32],
    inverse_rms: f32,
    scale: &[f32],
) {
    use std::arch::x86_64::*;

    // SAFETY: the caller guarantees AVX2 support and equal slice lengths.
    // Full-width loads and stores are bounded by `input.len()`.
    unsafe {
        let inverse_rms_vector = _mm256_set1_ps(inverse_rms);
        let mut index = 0;
        while index + 8 <= input.len() {
            let normalized = _mm256_mul_ps(
                _mm256_loadu_ps(input.as_ptr().add(index)),
                inverse_rms_vector,
            );
            let scaled = _mm256_mul_ps(normalized, _mm256_loadu_ps(scale.as_ptr().add(index)));
            _mm256_storeu_ps(output.as_mut_ptr().add(index), scaled);
            index += 8;
        }
        normalize_and_scale_scalar(
            &input[index..],
            &mut output[index..],
            inverse_rms,
            &scale[index..],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_values(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as i32 - (1 << 23)) as f32 * (1.0 / (1 << 20) as f32)
            })
            .collect()
    }

    fn assert_bit_identical(
        input: &[f32],
        scale: &[f32],
        inverse_rms: f32,
        vectorized: impl FnOnce(&mut [f32]),
    ) {
        let mut expected = vec![0.0; input.len()];
        let mut actual = vec![0.0; input.len()];
        normalize_and_scale_scalar(input, &mut expected, inverse_rms, scale);
        vectorized(&mut actual);
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn dispatched_normalize_and_scale_is_bit_identical_to_scalar() {
        let sizes = [1, 7, 8, 15, 16, 17, 31, 32, 63, 64, 127, 1024, 1536];
        let epsilons = [0.0f32, 1e-6, 1e-5, 1e-2];
        for &len in &sizes {
            let input = make_values(len, len as u64 + 11);
            let scale = make_values(len, len as u64 + 29);
            let mean_square = input.iter().map(|value| value * value).sum::<f32>() / len as f32;
            for &epsilon in &epsilons {
                let inverse_rms = 1.0 / (mean_square + epsilon).sqrt();
                assert_bit_identical(&input, &scale, inverse_rms, |actual| {
                    normalize_and_scale(&input, actual, inverse_rms, &scale);
                });
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_normalize_and_scale_is_bit_identical_to_scalar() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let input = make_values(1027, 41);
        let scale = make_values(1027, 43);
        let inverse_rms = 0.731_234_55;
        assert_bit_identical(&input, &scale, inverse_rms, |actual| {
            // SAFETY: guarded by runtime AVX-512F detection above.
            unsafe { normalize_and_scale_avx512(&input, actual, inverse_rms, &scale) };
        });
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_normalize_and_scale_is_bit_identical_to_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let input = make_values(1027, 47);
        let scale = make_values(1027, 53);
        let inverse_rms = 0.731_234_55;
        assert_bit_identical(&input, &scale, inverse_rms, |actual| {
            // SAFETY: guarded by runtime AVX2 detection above.
            unsafe { normalize_and_scale_avx2(&input, actual, inverse_rms, &scale) };
        });
    }
}
