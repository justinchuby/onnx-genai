//! Vectorized sum-of-squares reductions for the RMS-normalization family.
//!
//! The RMS path (`RMSNormalization` / `SimplifiedLayerNormalization` /
//! `SkipSimplifiedLayerNormalization`) reduces a hidden-size row to
//! `Σ xᵢ²` once per group, ~113×/decode on the 0.6B model. This module
//! provides an AVX-512F (16-lane) reduction, runtime-gated on `avx512f`
//! detection, with a portable scalar fallback used on non-AVX-512 x86 and
//! on aarch64.
//!
//! Numerics: a 16-lane accumulator reduced with `_mm512_reduce_add_ps`
//! (a tree reduction) is typically **more** accurate than a serial scalar
//! sum, because partial sums stay closer in magnitude and the final
//! cross-lane combine is pairwise. Under the project's f64-reference parity
//! bar this is an accuracy-neutral-or-better change; see the unit tests in
//! this module which assert the AVX-512 result is closer-or-equal to a f64
//! reference than the old serial scalar sum.

/// `Σ valuesᵢ²` in f32. Dispatches to AVX-512F when the host supports it,
/// otherwise a lane-parallel scalar fallback.
#[inline]
pub(crate) fn sum_of_squares(values: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: guarded by runtime `avx512f` detection above.
            return unsafe { sum_of_squares_avx512(values) };
        }
    }
    sum_of_squares_scalar(values)
}

/// `sumᵢ = inputᵢ + skipᵢ (+ biasᵢ)` written into `sum`, returning `Σ sumᵢ²`.
///
/// Fuses the residual add with the reduction so the assembled row is only
/// traversed once, matching the pre-existing `SkipSimplifiedLayerNormalization`
/// contract. Dispatches to AVX-512F when available.
#[inline]
pub(crate) fn assemble_and_sum_of_squares(
    input: &[f32],
    skip: &[f32],
    bias: Option<&[f32]>,
    sum: &mut [f32],
) -> f32 {
    debug_assert_eq!(input.len(), skip.len());
    debug_assert_eq!(input.len(), sum.len());
    debug_assert!(bias.is_none_or(|bias| bias.len() == input.len()));

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: guarded by runtime `avx512f` detection above; the slice
            // lengths are asserted equal so every masked/full store stays in
            // bounds.
            return unsafe { assemble_and_sum_of_squares_avx512(input, skip, bias, sum) };
        }
    }
    assemble_and_sum_of_squares_scalar(input, skip, bias, sum)
}

/// Lane-parallel scalar `Σ valuesᵢ²` (portable fallback).
///
/// Uses an 8-wide accumulator array so the fallback is itself pairwise-ish
/// (better conditioned than a strictly serial sum) and stable across the
/// x86 non-AVX-512 and aarch64 targets.
fn sum_of_squares_scalar(values: &[f32]) -> f32 {
    const LANES: usize = 8;
    let mut lane_sums = [0.0f32; LANES];
    let bulk = values.len() / LANES * LANES;
    let mut base = 0;
    while base < bulk {
        for (lane, lane_sum) in lane_sums.iter_mut().enumerate() {
            let value = values[base + lane];
            *lane_sum += value * value;
        }
        base += LANES;
    }
    let mut total = lane_sums.into_iter().sum::<f32>();
    for &value in &values[bulk..] {
        total += value * value;
    }
    total
}

/// Scalar `sumᵢ = inputᵢ + skipᵢ (+ biasᵢ)` + `Σ sumᵢ²` (portable fallback).
fn assemble_and_sum_of_squares_scalar(
    input: &[f32],
    skip: &[f32],
    bias: Option<&[f32]>,
    sum: &mut [f32],
) -> f32 {
    const LANES: usize = 8;
    let mut lane_sums = [0.0f32; LANES];
    let bulk = input.len() / LANES * LANES;
    let mut base = 0;
    while base < bulk {
        for (lane, lane_sum) in lane_sums.iter_mut().enumerate() {
            let index = base + lane;
            let value = input[index] + skip[index] + bias.map_or(0.0, |bias| bias[index]);
            sum[index] = value;
            *lane_sum += value * value;
        }
        base += LANES;
    }
    let mut total = lane_sums.into_iter().sum::<f32>();
    for index in bulk..input.len() {
        let value = input[index] + skip[index] + bias.map_or(0.0, |bias| bias[index]);
        sum[index] = value;
        total += value * value;
    }
    total
}

/// AVX-512F 16-lane `Σ valuesᵢ²`.
///
/// # Safety
/// The host must support `avx512f`. Callers gate on
/// `is_x86_feature_detected!("avx512f")`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sum_of_squares_avx512(values: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    // SAFETY: the caller guarantees `avx512f` is present (every intrinsic
    // below requires it). All loads stay within `values` — the full loops
    // read 16-wide chunks bounded by `n`, and the tail uses a `rem`-bit mask
    // so masked-off lanes are neither read nor squared.
    unsafe {
        let n = values.len();
        let ptr = values.as_ptr();
        // Four independent accumulators hide the FMA latency for long rows.
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();

        let mut i = 0;
        while i + 64 <= n {
            let v0 = _mm512_loadu_ps(ptr.add(i));
            let v1 = _mm512_loadu_ps(ptr.add(i + 16));
            let v2 = _mm512_loadu_ps(ptr.add(i + 32));
            let v3 = _mm512_loadu_ps(ptr.add(i + 48));
            acc0 = _mm512_fmadd_ps(v0, v0, acc0);
            acc1 = _mm512_fmadd_ps(v1, v1, acc1);
            acc2 = _mm512_fmadd_ps(v2, v2, acc2);
            acc3 = _mm512_fmadd_ps(v3, v3, acc3);
            i += 64;
        }
        while i + 16 <= n {
            let v = _mm512_loadu_ps(ptr.add(i));
            acc0 = _mm512_fmadd_ps(v, v, acc0);
            i += 16;
        }
        if i < n {
            let rem = n - i;
            let mask = (1u16 << rem) - 1;
            let v = _mm512_maskz_loadu_ps(mask, ptr.add(i));
            acc0 = _mm512_fmadd_ps(v, v, acc0);
        }

        let acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        _mm512_reduce_add_ps(acc)
    }
}

/// AVX-512F fused residual-add + 16-lane `Σ sumᵢ²`.
///
/// # Safety
/// The host must support `avx512f`; `input`, `skip`, `sum` (and `bias` when
/// present) must all have the same length. Callers gate on
/// `is_x86_feature_detected!("avx512f")` and assert equal lengths.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn assemble_and_sum_of_squares_avx512(
    input: &[f32],
    skip: &[f32],
    bias: Option<&[f32]>,
    sum: &mut [f32],
) -> f32 {
    use std::arch::x86_64::*;

    // SAFETY: the caller guarantees `avx512f` is present and that `input`,
    // `skip`, `sum` (and `bias` when present) share the same length `n`. Every
    // load/store below is 16-wide bounded by `n`, and the tail uses a
    // `rem`-bit mask so only valid lanes are touched.
    unsafe {
        let n = input.len();
        let in_ptr = input.as_ptr();
        let skip_ptr = skip.as_ptr();
        let sum_ptr = sum.as_mut_ptr();
        let bias_ptr = bias.map(|bias| bias.as_ptr());

        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();

        let mut i = 0;
        while i + 32 <= n {
            let mut v0 = _mm512_add_ps(
                _mm512_loadu_ps(in_ptr.add(i)),
                _mm512_loadu_ps(skip_ptr.add(i)),
            );
            let mut v1 = _mm512_add_ps(
                _mm512_loadu_ps(in_ptr.add(i + 16)),
                _mm512_loadu_ps(skip_ptr.add(i + 16)),
            );
            if let Some(bias_ptr) = bias_ptr {
                v0 = _mm512_add_ps(v0, _mm512_loadu_ps(bias_ptr.add(i)));
                v1 = _mm512_add_ps(v1, _mm512_loadu_ps(bias_ptr.add(i + 16)));
            }
            _mm512_storeu_ps(sum_ptr.add(i), v0);
            _mm512_storeu_ps(sum_ptr.add(i + 16), v1);
            acc0 = _mm512_fmadd_ps(v0, v0, acc0);
            acc1 = _mm512_fmadd_ps(v1, v1, acc1);
            i += 32;
        }
        while i + 16 <= n {
            let mut v = _mm512_add_ps(
                _mm512_loadu_ps(in_ptr.add(i)),
                _mm512_loadu_ps(skip_ptr.add(i)),
            );
            if let Some(bias_ptr) = bias_ptr {
                v = _mm512_add_ps(v, _mm512_loadu_ps(bias_ptr.add(i)));
            }
            _mm512_storeu_ps(sum_ptr.add(i), v);
            acc0 = _mm512_fmadd_ps(v, v, acc0);
            i += 16;
        }
        if i < n {
            let rem = n - i;
            let mask = (1u16 << rem) - 1;
            let mut v = _mm512_add_ps(
                _mm512_maskz_loadu_ps(mask, in_ptr.add(i)),
                _mm512_maskz_loadu_ps(mask, skip_ptr.add(i)),
            );
            if let Some(bias_ptr) = bias_ptr {
                v = _mm512_add_ps(v, _mm512_maskz_loadu_ps(mask, bias_ptr.add(i)));
            }
            _mm512_mask_storeu_ps(sum_ptr.add(i), mask, v);
            // Masked-off lanes are zero and contribute nothing to the reduction.
            acc0 = _mm512_fmadd_ps(v, v, acc0);
        }

        _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serial scalar reference matching the ORIGINAL RMS reduction
    /// (`slice.iter().map(|v| v*v).sum::<f32>()`).
    fn old_serial_sum_of_squares(values: &[f32]) -> f32 {
        values.iter().map(|&v| v * v).sum::<f32>()
    }

    /// High-precision f64 reference (the parity oracle).
    fn f64_sum_of_squares(values: &[f32]) -> f64 {
        values.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>()
    }

    /// Deterministic pseudo-random realistic hidden vectors of a given size.
    fn make_row(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Realistic activation magnitudes with a few large outliers.
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

    #[test]
    fn sum_of_squares_matches_and_is_closer_to_f64_than_serial_scalar() {
        // Cover non-multiples of 16 (mask tail) and multiples across the
        // realistic hidden-size range for the 0.6B model (1024) and others.
        let sizes = [
            1, 3, 7, 15, 16, 17, 31, 32, 63, 64, 127, 128, 512, 896, 1024, 1536, 4096,
        ];
        for &len in &sizes {
            for seed in 0..8u64 {
                let row = make_row(len, seed);
                let f64_ref = f64_sum_of_squares(&row);
                let vectorized = sum_of_squares(&row) as f64;
                let serial = old_serial_sum_of_squares(&row) as f64;

                // (a) Tight absolute-or-relative tolerance vs f64 reference.
                let tol = 1e-4 + 1e-5 * f64_ref.abs();
                assert!(
                    (vectorized - f64_ref).abs() <= tol,
                    "len {len} seed {seed}: vectorized {vectorized} vs f64 {f64_ref}"
                );

                // (b) No accuracy regression: the vectorized reduction is
                // closer-or-equal to the f64 reference than the old serial
                // scalar sum. Allow a tiny slack for the rare case where both
                // round to essentially the same value.
                let err_vec = (vectorized - f64_ref).abs();
                let err_serial = (serial - f64_ref).abs();
                assert!(
                    err_vec <= err_serial + 1e-6 * f64_ref.abs().max(1.0),
                    "len {len} seed {seed}: vectorized err {err_vec} > serial err {err_serial} \
                     (regressed accuracy vs f64)"
                );
            }
        }
    }

    #[test]
    fn assemble_matches_reference_and_writes_sum() {
        let sizes = [1, 15, 16, 17, 31, 32, 33, 128, 1024];
        for &len in &sizes {
            let input = make_row(len, 11);
            let skip = make_row(len, 22);
            let bias = make_row(len, 33);
            for use_bias in [false, true] {
                let bias_opt = use_bias.then_some(bias.as_slice());
                let mut sum = vec![0.0f32; len];
                let got = assemble_and_sum_of_squares(&input, &skip, bias_opt, &mut sum);

                // Expected assembled row and its f64 sum-of-squares.
                let mut expected_sum = vec![0.0f32; len];
                let mut f64_ref = 0.0f64;
                for index in 0..len {
                    let value = input[index] + skip[index] + bias_opt.map_or(0.0, |b| b[index]);
                    expected_sum[index] = value;
                    f64_ref += (value as f64) * (value as f64);
                }
                assert_eq!(
                    sum, expected_sum,
                    "len {len} use_bias {use_bias}: sum row differs"
                );

                let tol = 1e-4 + 1e-5 * f64_ref.abs();
                assert!(
                    ((got as f64) - f64_ref).abs() <= tol,
                    "len {len} use_bias {use_bias}: got {got} vs f64 {f64_ref}"
                );
            }
        }
    }

    /// Load-guarded micro-bench of the reduction (scalar vs AVX-512F) at the
    /// 0.6B hidden size. Ignored by default; run with
    /// `cargo test -p onnx-runtime-ep-cpu --features mlas -- --ignored --nocapture bench_sum_of_squares`.
    #[test]
    #[ignore = "prints the f64-parity audit table; run explicitly"]
    fn audit_f64_table() {
        let sizes = [16usize, 64, 128, 512, 896, 1024, 1536, 4096];
        println!("len | max|vec-f64| | max|serial-f64| | max rel(vec) | vec_closer_or_equal");
        for &len in &sizes {
            let mut max_vec = 0.0f64;
            let mut max_serial = 0.0f64;
            let mut max_rel = 0.0f64;
            let mut all_closer = true;
            for seed in 0..64u64 {
                let row = make_row(len, seed);
                let f64_ref = f64_sum_of_squares(&row);
                let vec = sum_of_squares(&row) as f64;
                let serial = old_serial_sum_of_squares(&row) as f64;
                let e_vec = (vec - f64_ref).abs();
                let e_serial = (serial - f64_ref).abs();
                max_vec = max_vec.max(e_vec);
                max_serial = max_serial.max(e_serial);
                max_rel = max_rel.max(e_vec / f64_ref.abs().max(1.0));
                if e_vec > e_serial + 1e-9 {
                    all_closer = false;
                }
            }
            println!("{len:5} | {max_vec:.4e} | {max_serial:.4e} | {max_rel:.3e} | {all_closer}");
        }
    }

    #[test]
    #[ignore = "timing micro-bench; run explicitly"]
    fn bench_sum_of_squares() {
        use std::hint::black_box;
        use std::time::Instant;

        let row = make_row(1024, 7);
        let iters = 200_000;

        // Warm up.
        for _ in 0..10_000 {
            black_box(sum_of_squares_scalar(black_box(&row)));
            black_box(sum_of_squares(black_box(&row)));
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            black_box(sum_of_squares_scalar(black_box(&row)));
        }
        let scalar_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

        let t1 = Instant::now();
        for _ in 0..iters {
            black_box(sum_of_squares(black_box(&row)));
        }
        let dispatch_ns = t1.elapsed().as_nanos() as f64 / iters as f64;

        let avx512 = {
            #[cfg(target_arch = "x86_64")]
            {
                std::arch::is_x86_feature_detected!("avx512f")
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                false
            }
        };
        println!(
            "sum_of_squares(len=1024): scalar-lane {scalar_ns:.1} ns/call, \
             dispatched(avx512f={avx512}) {dispatch_ns:.1} ns/call"
        );
    }
}
