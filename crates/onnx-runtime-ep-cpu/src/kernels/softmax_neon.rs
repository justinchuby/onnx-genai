//! Four-lane native softmax, shared by identical/disjoint row entry points.
//! The degree-six exp fit is shared with AVX2; gradual underflow is retained.

use std::arch::aarch64::*;

use super::exp_coefficients as coeff;

#[cfg(test)]
static SOFTMAX_NEON_TEST_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Four exponentials for softmax differences (`x <= 0`, NaN or -infinity).
///
/// # Safety
/// NEON must be available. Arguments originate after subtracting the row max.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn exp4<const NEUTRAL_SCALE: bool>(x: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let zero = vdupq_n_f32(0.0);
    let nan = vmvnq_u32(vceqq_f32(x, x));
    // exp(-104) rounds to zero in f32. Mask these/NaN lanes BEFORE integer
    // range reduction; the final selects restore exact zero or a quiet NaN.
    let underflow = vcleq_f32(x, vdupq_n_f32(-104.0));
    let safe = vbslq_f32(vorrq_u32(nan, underflow), zero, x);
    let k = vrndnq_f32(vmulq_n_f32(safe, core::f32::consts::LOG2_E));
    let r = vfmsq_n_f32(safe, k, coeff::LN2_HI);
    let r = vfmsq_n_f32(r, k, coeff::LN2_LO);
    let p = vfmaq_f32(vdupq_n_f32(coeff::C5), vdupq_n_f32(coeff::C6), r);
    let p = vfmaq_f32(vdupq_n_f32(coeff::C4), p, r);
    let p = vfmaq_f32(vdupq_n_f32(coeff::C3), p, r);
    let p = vfmaq_f32(vdupq_n_f32(coeff::C2), p, r);
    let p = vfmaq_f32(one, p, r);
    let p = vfmaq_f32(one, p, r);
    let ki = vcvtq_s32_f32(k);
    let subnormal = vcltq_s32(ki, vdupq_n_s32(-126));
    // For k < -126 construct normal 2^(k+126), multiply by the polynomial,
    // then scale by 2^-126. This avoids constructing an invalid negative IEEE
    // exponent and preserves f32 subnormals down through round-to-zero.
    let bias = vbslq_s32(subnormal, vdupq_n_s32(253), vdupq_n_s32(127));
    let power = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(ki, bias)));
    let y = vmulq_f32(p, power);
    // Select the scale BEFORE multiplying: ordinary lanes must not execute
    // a discarded underflowing multiply. The compile-time false control is
    // instantiated only in tests to retain the initial NEON mechanism.
    let y = if NEUTRAL_SCALE {
        let scale = vbslq_f32(subnormal, vdupq_n_f32(f32::MIN_POSITIVE), one);
        vmulq_f32(y, scale)
    } else {
        vbslq_f32(subnormal, vmulq_n_f32(y, f32::MIN_POSITIVE), y)
    };
    let y = vbslq_f32(underflow, zero, y);
    let quiet = vreinterpretq_f32_u32(vorrq_u32(
        vreinterpretq_u32_f32(x),
        vdupq_n_u32(0x0040_0000),
    ));
    vbslq_f32(nan, quiet, y)
}

/// Max / exp-and-sum / normalize, with no allocation or intermediate copy.
///
/// # Safety
/// NEON must be available; src/dst cover d f32 elements and are identical or
/// disjoint. Each vector load precedes its corresponding destination store.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn softmax_row_impl<const NEUTRAL_SCALE: bool>(src: *const f32, dst: *mut f32, d: usize) {
    #[cfg(test)]
    SOFTMAX_NEON_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: all complete-vector and scalar accesses are bounded by d.
    unsafe {
        let mut maximum = vdupq_n_f32(f32::NEG_INFINITY);
        let mut i = 0;
        while i + 4 <= d {
            let value = vld1q_f32(src.add(i));
            // Ordered compare/select ignores quiet AND signalling NaNs, just
            // as the original scalar `if value > maximum` does.
            maximum = vbslq_f32(vcgtq_f32(value, maximum), value, maximum);
            i += 4;
        }
        let mut max = vmaxvq_f32(maximum);
        while i < d {
            let value = *src.add(i);
            if value > max {
                max = value;
            }
            i += 1;
        }
        let vmax = vdupq_n_f32(max);
        let mut sums = vdupq_n_f32(0.0);
        i = 0;
        while i + 4 <= d {
            let e = exp4::<NEUTRAL_SCALE>(vsubq_f32(vld1q_f32(src.add(i)), vmax));
            vst1q_f32(dst.add(i), e);
            sums = vaddq_f32(sums, e);
            i += 4;
        }
        let mut sum = vaddvq_f32(sums);
        while i < d {
            let e = (*src.add(i) - max).exp();
            *dst.add(i) = e;
            sum += e;
            i += 1;
        }
        let inverse = 1.0 / sum;
        let inv = vdupq_n_f32(inverse);
        i = 0;
        while i + 4 <= d {
            vst1q_f32(dst.add(i), vmulq_f32(vld1q_f32(dst.add(i)), inv));
            i += 4;
        }
        while i < d {
            *dst.add(i) *= inverse;
            i += 1;
        }
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn softmax_row(src: *const f32, dst: *mut f32, d: usize) {
    // SAFETY: forward the caller's NEON and identical/disjoint row contract.
    unsafe { softmax_row_impl::<true>(src, dst, d) };
}

#[cfg(all(test, feature = "mlas"))]
fn initial_neon(src: &[f32], dst: &mut [f32], _n: usize, d: usize) {
    for (input, output) in src.chunks(d).zip(dst.chunks_mut(d)) {
        // SAFETY: NEON test host; disjoint equal-length complete rows.
        unsafe { softmax_row_impl::<false>(input.as_ptr(), output.as_mut_ptr(), d) };
    }
}

#[cfg(all(test, feature = "mlas"))]
fn scalar_before(src: &[f32], dst: &mut [f32], _n: usize, d: usize) {
    for (input, output) in src.chunks(d).zip(dst.chunks_mut(d)) {
        // Match the native arm's test-only per-row bookkeeping.
        SOFTMAX_NEON_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: disjoint complete rows; exact historical scalar arithmetic.
        unsafe { super::softmax_row_scalar(input.as_ptr(), output.as_mut_ptr(), d) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    fn standard_fpcr() -> u64 {
        let value: u64;
        // SAFETY: reading FPCR changes no floating-point or memory state.
        unsafe {
            std::arch::asm!("mrs {value}, fpcr", value=out(reg) value, options(nomem, nostack, preserves_flags))
        };
        assert_eq!(
            value & ((1 << 1) | (3 << 22) | (1 << 24) | (1 << 25)),
            0,
            "NEON exp proof requires standard rounding/subnormal/NaN mode; restore FPCR (got {value:#x})"
        );
        value
    }

    #[test]
    fn exp4_preserves_specials_and_the_gradual_underflow_band() {
        standard_fpcr();
        let mut worst = 0;
        let mut checked = 0;
        // Sample the full negative f32 domain densely in bit space, and cover
        // EVERY f32 argument in [-104,-87], including the subnormal boundary.
        for bits in (0x8000_0000u32..=(-104.0f32).to_bits())
            .step_by(1097)
            .chain(((-87.0f32).to_bits()..=(-104.0f32).to_bits()).step_by(4))
        {
            let input = std::array::from_fn::<_, 4, _>(|i| {
                f32::from_bits((bits + i as u32).min((-104.0f32).to_bits()))
            });
            let mut output = [0.0; 4];
            // SAFETY: AArch64 NEON host, four initialized input/output lanes.
            unsafe { vst1q_f32(output.as_mut_ptr(), exp4::<true>(vld1q_f32(input.as_ptr()))) };
            let mut initial = [0.0; 4];
            // SAFETY: same four-lane contract, exact initial NEON control.
            unsafe {
                vst1q_f32(
                    initial.as_mut_ptr(),
                    exp4::<false>(vld1q_f32(input.as_ptr())),
                )
            };
            assert!(
                initial
                    .iter()
                    .zip(&output)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            for (x, y) in input.into_iter().zip(output) {
                let expected = f64::from(x).exp() as f32;
                assert!(y >= 0.0 && y.is_finite());
                let ulp = y.to_bits().abs_diff(expected.to_bits());
                worst = worst.max(ulp);
                assert!(
                    ulp <= 2,
                    "exp4 x={x:e} got={y:e} expected={expected:e} ulp={ulp}"
                );
                checked += 1;
            }
        }
        assert!(
            checked > 3_000_000,
            "normal and full subnormal-argument coverage must execute"
        );
        for lane in 0..4 {
            for value in [
                f32::NEG_INFINITY,
                -f32::MAX,
                f32::NAN,
                f32::from_bits(0x7f80_0001),
                f32::from_bits(0xffc0_1234),
                -0.0,
            ] {
                let mut input = [-1.0; 4];
                input[lane] = value;
                let mut output = [0.0; 4];
                // SAFETY: equal initialized vectors on the NEON host.
                unsafe { vst1q_f32(output.as_mut_ptr(), exp4::<true>(vld1q_f32(input.as_ptr()))) };
                if value.is_nan() {
                    assert_eq!(output[lane].to_bits(), value.to_bits() | 0x0040_0000);
                } else if value == 0.0 {
                    assert_eq!(output[lane], 1.0);
                } else {
                    assert_eq!(output[lane].to_bits(), 0);
                }
                for (i, y) in output.iter().enumerate() {
                    if i != lane {
                        assert!(y.to_bits().abs_diff((-1.0f32).exp().to_bits()) <= 2);
                    }
                }
            }
        }
        eprintln!("NEON exp4 checked={checked} max_ulp={worst}");
    }

    #[test]
    fn native_dispatch_reaches_neon_and_preserves_every_row_boundary() {
        standard_fpcr();
        let before = SOFTMAX_NEON_TEST_HITS.load(Ordering::Relaxed);
        let mut witness = false;
        for d in (1..=137).chain([257, 1024, 4099]) {
            for offset in 0..4 {
                let input: Vec<f32> = (0..d + 4)
                    .map(|i| -13.0 + (i * 7919 % 1009) as f32 * 0.021)
                    .collect();
                let input = &input[offset..offset + d];
                let mut backing = vec![-12345.5; d + 8];
                let mut scalar = vec![0.0; d];
                let mut direct = vec![0.0; d];
                super::super::softmax_rows_serial_out_native(input, &mut backing[4..4 + d], 1, d);
                // SAFETY: initialized equal-length slices, NEON host.
                unsafe {
                    super::super::softmax_row_scalar(input.as_ptr(), scalar.as_mut_ptr(), d);
                    if d >= 4 {
                        softmax_row(input.as_ptr(), direct.as_mut_ptr(), d);
                    } else {
                        super::super::softmax_row_scalar(input.as_ptr(), direct.as_mut_ptr(), d);
                    }
                }
                assert!(
                    backing[4..4 + d]
                        .iter()
                        .zip(&direct)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                witness |= d >= 4
                    && direct
                        .iter()
                        .zip(&scalar)
                        .any(|(a, b)| a.to_bits() != b.to_bits());
                let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum: f64 = input
                    .iter()
                    .map(|&x| (f64::from(x) - f64::from(max)).exp())
                    .sum();
                for (&x, &y) in input.iter().zip(&backing[4..4 + d]) {
                    let expected = (f64::from(x) - f64::from(max)).exp() / sum;
                    assert!((f64::from(y) - expected).abs() <= 1e-6);
                }
                let mut inplace = input.to_vec();
                super::super::softmax_rows_serial_native(&mut inplace, 1, d);
                assert!(
                    inplace
                        .iter()
                        .zip(&direct)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                assert!(
                    backing[..4]
                        .iter()
                        .chain(&backing[4 + d..])
                        .all(|&v| v == -12345.5)
                );
            }
        }
        assert!(
            witness,
            "native caller must expose a vector result distinct from scalar control"
        );
        assert!(SOFTMAX_NEON_TEST_HITS.load(Ordering::Relaxed) > before);
    }

    #[test]
    fn native_row_maximum_sees_each_vector_and_tail_position() {
        for d in 1..=65 {
            for position in 0..d {
                let mut input = vec![-3.0e38; d];
                input[position] = 3.0e38;
                let mut output = vec![f32::NAN; d];
                super::super::softmax_rows_serial_out_native(&input, &mut output, 1, d);
                for (i, value) in output.iter().enumerate() {
                    assert_eq!(
                        *value,
                        if i == position { 1.0 } else { 0.0 },
                        "row max omitted position={position}, d={d}, output lane={i}"
                    );
                }
            }
        }
    }

    #[test]
    fn row_special_values_and_tiny_probabilities_match_the_native_contract() {
        for d in 1..=37 {
            for position in 0..d {
                for special in [
                    f32::NEG_INFINITY,
                    f32::INFINITY,
                    f32::from_bits(0x7f80_0001),
                    f32::from_bits(0xffc0_1234),
                    -90.0,
                    -100.0,
                ] {
                    let mut input = vec![0.0; d];
                    input[position] = special;
                    let mut out = vec![f32::NAN; d];
                    super::super::softmax_rows_serial_out_native(&input, &mut out, 1, d);
                    let mut inplace = input.clone();
                    super::super::softmax_rows_serial_native(&mut inplace, 1, d);
                    assert!(
                        out.iter()
                            .zip(&inplace)
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                    );
                    if special.is_nan() || special == f32::INFINITY {
                        assert!(out.iter().all(|v| v.is_nan()));
                    } else if d > 1 {
                        assert!(out.iter().all(|v| v.is_finite()));
                        if special == f32::NEG_INFINITY {
                            assert_eq!(out[position].to_bits(), 0);
                        } else {
                            assert!(out[position] > 0.0, "tiny probability must not be flushed");
                        }
                    }
                }
            }
            let input = vec![f32::NEG_INFINITY; d];
            let mut out = vec![0.0; d];
            super::super::softmax_rows_serial_out_native(&input, &mut out, 1, d);
            assert!(out.iter().all(|v| v.is_nan()));
        }
    }
    #[cfg(feature = "mlas")]
    #[test]
    #[ignore = "wall diagnostic; run under the shared host lock"]
    fn neon_softmax_wall_ab() {
        use std::{hint::black_box, time::Instant};
        eprintln!("NEON_SOFTMAX_FPCR,{}", standard_fpcr());
        let arg = |name: &str, fallback| {
            std::env::var(name)
                .map_or(Ok(fallback), |v| v.parse::<usize>())
                .unwrap_or_else(|error| panic!("{name} must be an unsigned integer: {error}"))
        };
        let width = arg("NXRT_SOFTMAX_BENCH_WIDTH", 1);
        let reps = arg("NXRT_SOFTMAX_BENCH_REPS", 21);
        let order = arg("NXRT_SOFTMAX_BENCH_ORDER", 0);
        assert!(
            [1, 4, 8].contains(&width),
            "NXRT_SOFTMAX_BENCH_WIDTH must be 1, 4 or 8; got {width}"
        );
        assert!(
            (1..=101).contains(&reps),
            "NXRT_SOFTMAX_BENCH_REPS must be 1..101; got {reps}"
        );
        crate::task_runtime::set_task_thread_budget(Some(width)).unwrap();
        assert_eq!(crate::task_runtime::width(), width);
        assert!(rayon::current_thread_index().is_none() && !crate::task_runtime::in_task());
        let mut shapes = [
            (1usize, 4usize),
            (1, 7),
            (1, 257),
            (32, 1024),
            (128, 4099),
            (16, 65539),
        ];
        let rotate = order % shapes.len();
        shapes.rotate_left(rotate);
        for mode_index in 0..2 {
            let serial = (mode_index + order) % 2 == 0;
            let mode = if serial { "serial" } else { "runtime" };
            for (n, d) in shapes {
                let source: Vec<_> = (0..n * d)
                    .map(|i| -16.0 + (i.wrapping_mul(7919) % 100003) as f32 * (16.0 / 100002.0))
                    .collect();
                let mut output = vec![0.0; source.len()];
                let iters = (262144 / source.len()).clamp(1, 2048);
                let run = |arm, out: &mut [f32]| {
                    let input = black_box(source.as_slice());
                    let out = black_box(out);
                    let kernel: fn(&[f32], &mut [f32], usize, usize) = match arm {
                        0 => scalar_before,
                        1 | 2 => super::super::softmax_rows_serial_out_native,
                        3 => mlas_sys::compute_softmax,
                        4 => initial_neon,
                        _ => unreachable!(),
                    };
                    if serial {
                        kernel(input, out, n, d);
                    } else {
                        super::super::softmax_rows_with_kernel(input, out, n, d, kernel);
                    }
                };
                for arm in 0..5 {
                    let before = crate::task_runtime::testing::counters();
                    run(arm, &mut output);
                    let after = crate::task_runtime::testing::counters();
                    let workers = (after.tasks - before.tasks)
                        - (after.tasks_by_dispatcher - before.tasks_by_dispatcher);
                    let parallel =
                        !serial && width > 1 && super::super::fan_out_is_worthwhile(n, d);
                    assert_eq!(
                        workers > 0,
                        parallel,
                        "actual worker participation: {mode}/{n}/{d}/arm{arm}"
                    );
                    let mut worst = 0.0f64;
                    let mut digest = 0xcbf2_9ce4_8422_2325u64;
                    for (input, row) in source.chunks(d).zip(output.chunks(d)) {
                        let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let sum: f64 = input
                            .iter()
                            .map(|&x| (f64::from(x) - f64::from(max)).exp())
                            .sum();
                        for (&x, &y) in input.iter().zip(row) {
                            let expected = (f64::from(x) - f64::from(max)).exp() / sum;
                            let error = (f64::from(y) - expected).abs();
                            assert!(
                                y.is_finite() && error <= 1e-6,
                                "arm{arm} {n}x{d}: error={error}"
                            );
                            worst = worst.max(error);
                            digest =
                                (digest ^ u64::from(y.to_bits())).wrapping_mul(0x100_0000_01b3);
                        }
                    }
                    eprintln!(
                        "NEON_SOFTMAX_VALIDATE,{width},{mode},{n},{d},{arm},{digest},{worst},{},{workers}",
                        u8::from(parallel)
                    );
                    for _ in 0..3 {
                        run(arm, &mut output);
                    }
                }
                for rep in 0..reps {
                    for turn in 0..5 {
                        let arm = (rep + turn + order) % 5;
                        let start = Instant::now();
                        for _ in 0..iters {
                            run(arm, &mut output);
                        }
                        let ns = start.elapsed().as_nanos();
                        black_box(output.as_slice());
                        let token = [0, source.len() / 2, source.len() - 1].into_iter().fold(
                            0xcbf2_9ce4_8422_2325u64,
                            |h, i| {
                                (h ^ u64::from(output[i].to_bits())).wrapping_mul(0x100_0000_01b3)
                            },
                        );
                        println!(
                            "NEON_SOFTMAX_SAMPLE,{width},{mode},{n},{d},{iters},{rep},{arm},{ns},{token}"
                        );
                    }
                }
            }
        }
        eprintln!("NEON_SOFTMAX_FPCR,{}", standard_fpcr());
    }
}
