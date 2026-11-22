//! Four-lane ports of the shared MLAS-derived Erf polynomials and fused GELU.
//! Coefficients and attribution live in the parent module and MLAS_LICENSE.

use std::arch::aarch64::*;

use super::{FRAC_1_SQRT_2_F32, erf_c};

#[cfg(test)]
static ERF_NEON_TEST_HITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static GELU_NEON_TEST_HITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Arguments are finite and clamped to the shared Erf exp polynomial's band.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn exp_reduced(x: float32x4_t) -> float32x4_t {
    let c = vdupq_n_f32(erf_c::EXP_C);
    let k = vsubq_f32(vfmaq_f32(c, vdupq_n_f32(erf_c::EXP_LOG2_RECIPROCAL), x), c);
    let f = vfmaq_f32(x, k, vdupq_n_f32(erf_c::EXP_LOG2_HI));
    let f = vfmaq_f32(f, k, vdupq_n_f32(erf_c::EXP_LOG2_LO));
    let sq = vmulq_f32(f, f);
    let cube = vmulq_f32(sq, f);
    let hi = vfmaq_f32(vdupq_n_f32(erf_c::EXP_P1), vdupq_n_f32(erf_c::EXP_P0), f);
    let hi_lo = vfmaq_f32(vdupq_n_f32(erf_c::EXP_P3), vdupq_n_f32(erf_c::EXP_P2), f);
    let hi = vfmaq_f32(hi_lo, hi, sq);
    let lo = vfmaq_f32(vdupq_n_f32(erf_c::EXP_P5), vdupq_n_f32(erf_c::EXP_P4), f);
    let lo = vfmaq_f32(vdupq_n_f32(erf_c::EXP_P6), lo, f);
    let p = vfmaq_f32(lo, hi, cube);
    let exponent = vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(k), vdupq_n_s32(127)));
    vmulq_f32(p, vreinterpretq_f32_s32(exponent))
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn erf_lanes(x: float32x4_t) -> float32x4_t {
    // Mask NaNs before range reduction. Their result is explicitly quieted
    // from the original input below, retaining payload/sign on every lane.
    let ordered = vceqq_f32(x, x);
    let abs = vminq_f32(
        vabsq_f32(vbslq_f32(ordered, x, vdupq_n_f32(0.0))),
        vdupq_n_f32(erf_c::UPPER_ABS_RANGE),
    );
    let sq = vmulq_f32(abs, abs);
    let small = vfmaq_f32(
        vdupq_n_f32(erf_c::SMALL_P1),
        vdupq_n_f32(erf_c::SMALL_P0),
        sq,
    );
    let small = vfmaq_f32(vdupq_n_f32(erf_c::SMALL_P2), small, sq);
    let small = vfmaq_f32(vdupq_n_f32(erf_c::SMALL_P3), small, sq);
    let small = vfmaq_f32(vdupq_n_f32(erf_c::SMALL_P4), small, sq);
    let small = vfmaq_f32(vdupq_n_f32(erf_c::SMALL_P5_MINUS_ONE), small, sq);
    let small = vfmaq_f32(abs, small, abs);
    let split = vcgtq_f32(abs, vdupq_n_f32(erf_c::SPLIT_BOUNDARY));
    let big_abs = vbslq_f32(split, abs, vdupq_n_f32(0.0));
    let sq = vmulq_f32(big_abs, big_abs);
    let cube = vmulq_f32(sq, big_abs);
    let hi = vfmaq_f32(
        vdupq_n_f32(erf_c::BIG_P1),
        vdupq_n_f32(erf_c::BIG_P0),
        big_abs,
    );
    let hi_lo = vfmaq_f32(
        vdupq_n_f32(erf_c::BIG_P3),
        vdupq_n_f32(erf_c::BIG_P2),
        big_abs,
    );
    let hi = vfmaq_f32(hi_lo, hi, sq);
    let lo = vfmaq_f32(
        vdupq_n_f32(erf_c::BIG_P5),
        vdupq_n_f32(erf_c::BIG_P4),
        big_abs,
    );
    let lo = vfmaq_f32(vdupq_n_f32(erf_c::BIG_P6_MINUS_ONE), lo, big_abs);
    let big = vfmaq_f32(lo, hi, cube);
    let big = vfmaq_f32(big_abs, big, big_abs);
    let neg_big = vmaxq_f32(vnegq_f32(big), vdupq_n_f32(erf_c::EXP_LOWER_RANGE));
    // SAFETY: range-clamped finite argument; inactive lanes have argument zero.
    let big = vsubq_f32(vdupq_n_f32(1.0), unsafe { exp_reduced(neg_big) });
    let magnitude = vbslq_f32(split, big, small);
    let sign = vandq_u32(vreinterpretq_u32_f32(x), vdupq_n_u32(0x8000_0000));
    let result = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(magnitude), sign));
    vbslq_f32(ordered, result, vaddq_f32(x, x))
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn gelu_lanes(x: float32x4_t) -> float32x4_t {
    // SAFETY: caller establishes NEON; every lane is handled by erf_lanes.
    let e = unsafe { erf_lanes(vmulq_n_f32(x, FRAC_1_SQRT_2_F32)) };
    let y = vmulq_f32(vmulq_n_f32(x, 0.5), vaddq_f32(vdupq_n_f32(1.0), e));
    vbslq_f32(
        vceqq_f32(x, vdupq_n_f32(f32::NEG_INFINITY)),
        vdupq_n_f32(0.0),
        y,
    )
}

/// The short remainder uses the same vector arithmetic as complete vectors;
/// zero-filled stack lanes avoid reading/writing beyond either slice.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn map(
    input: &[f32],
    output: &mut [f32],
    mut kernel: impl FnMut(float32x4_t) -> float32x4_t,
) {
    assert_eq!(
        input.len(),
        output.len(),
        "NEON activation input/output lengths must match"
    );
    let body = input.len() & !3;
    for i in (0..body).step_by(4) {
        // SAFETY: each iteration has four initialized input/output elements.
        unsafe {
            vst1q_f32(
                output.as_mut_ptr().add(i),
                kernel(vld1q_f32(input.as_ptr().add(i))),
            )
        };
    }
    let tail = input.len() - body;
    if tail != 0 {
        let mut lanes = [0.0; 4];
        lanes[..tail].copy_from_slice(&input[body..]);
        // SAFETY: stack buffer has four lanes; only the live prefix is copied.
        unsafe { vst1q_f32(lanes.as_mut_ptr(), kernel(vld1q_f32(lanes.as_ptr()))) };
        output[body..].copy_from_slice(&lanes[..tail]);
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn erf(input: &[f32], output: &mut [f32]) {
    #[cfg(test)]
    ERF_NEON_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: caller establishes NEON; map checks lengths and bounds all accesses.
    unsafe { map(input, output, |x| erf_lanes(x)) };
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn gelu(input: &[f32], output: &mut [f32]) {
    #[cfg(test)]
    GELU_NEON_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: caller establishes NEON; map checks lengths and bounds all accesses.
    unsafe { map(input, output, |x| gelu_lanes(x)) };
}

/// Fused bias add and exact GELU, with no full-sized intermediate allocation.
#[target_feature(enable = "neon")]
pub(super) unsafe fn gelu_bias(input: &[f32], bias: &[f32], width: usize, output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "NEON bias GELU input/output lengths must match"
    );
    assert_eq!(
        bias.len(),
        width,
        "NEON bias GELU row width must match bias length"
    );
    assert_ne!(width, 0, "NEON bias GELU row width must be nonzero");
    #[cfg(test)]
    GELU_NEON_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if width < 4 {
        // A vector spans multiple narrow rows. Prepare each possible cyclic
        // bias phase once, so no lane is wasted and no per-element division
        // or full-sized bias expansion is needed in the hot loop.
        let mut patterns = [[0.0; 4]; 4];
        for (phase, pattern) in patterns.iter_mut().enumerate().take(width) {
            for (lane, value) in pattern.iter_mut().enumerate() {
                *value = bias[(phase + lane) % width];
            }
        }
        let advance = 4 % width;
        let mut phase = 0;
        // SAFETY: all patterns have four initialized lanes; map bounds the
        // source/destination and caller establishes the NEON feature.
        unsafe {
            map(input, output, |x| {
                let b = vld1q_f32(patterns[phase].as_ptr());
                phase += advance;
                if phase >= width {
                    phase -= width;
                }
                gelu_lanes(vaddq_f32(x, b))
            })
        };
        return;
    }
    for (row, result) in input.chunks(width).zip(output.chunks_mut(width)) {
        let body = row.len() & !3;
        for i in (0..body).step_by(4) {
            // SAFETY: this row and its bias have four initialized elements.
            unsafe {
                let x = vaddq_f32(
                    vld1q_f32(row.as_ptr().add(i)),
                    vld1q_f32(bias.as_ptr().add(i)),
                );
                vst1q_f32(result.as_mut_ptr().add(i), gelu_lanes(x));
            }
        }
        let tail = row.len() - body;
        if tail != 0 {
            let mut x = [0.0; 4];
            let mut b = [0.0; 4];
            x[..tail].copy_from_slice(&row[body..]);
            b[..tail].copy_from_slice(&bias[body..body + tail]);
            // SAFETY: both stack arrays have four initialized lanes.
            unsafe {
                vst1q_f32(
                    x.as_mut_ptr(),
                    gelu_lanes(vaddq_f32(vld1q_f32(x.as_ptr()), vld1q_f32(b.as_ptr()))),
                )
            };
            result[body..].copy_from_slice(&x[..tail]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        SIMD_MIN_LEN, erf_f32_slice_native, erf_gelu_f32_slice_native, erf_gelu_scalar, erf_scalar,
    };
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn neon_erf_and_gelu_match_independent_accuracy_bounds() {
        assert!(std::arch::is_aarch64_feature_detected!("neon"));
        let mut input: Vec<f32> = (0..400_003)
            .map(|i| -6.0 + 12.0 * i as f32 / 400_002.0)
            .collect();
        for center in [
            0.0f32,
            0.921875,
            3.925,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
        ] {
            for value in [center.next_down(), center, center.next_up()] {
                input.extend([value, -value]);
            }
        }
        let mut seed = 0x1234_abcd_u32;
        for _ in 0..100_003 {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            input.push(f32::from_bits(seed));
        }
        input.extend([
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7fc0_1234),
            f32::from_bits(0xff80_0123),
        ]);
        let mut output = vec![0.0; input.len()];
        let mut worst_erf = 0.0f64;
        let mut worst_gelu = 0.0f64;
        // SAFETY: NEON feature assertion above; the mapper bounds every lane.
        unsafe { erf(&input, &mut output) };
        for (&x, &y) in input.iter().zip(&output) {
            if x.is_nan() {
                assert_eq!(y.to_bits(), x.to_bits() | 0x0040_0000);
                continue;
            }
            if x.is_infinite() || x == 0.0 {
                assert_eq!(y.to_bits(), erf_scalar(x).to_bits());
                continue;
            }
            let error = (f64::from(y) - f64::from(erf_scalar(x))).abs();
            worst_erf = worst_erf.max(error);
            assert!(error <= 5.97e-8, "Erf x={x:e} y={y:e} error={error:e}");
        }
        // SAFETY: same feature and slice-length contract.
        unsafe { gelu(&input, &mut output) };
        for (&x, &y) in input.iter().zip(&output) {
            if x.is_nan() {
                assert_eq!(y.to_bits(), x.to_bits() | 0x0040_0000);
                continue;
            }
            if x.is_infinite() || x == 0.0 {
                assert_eq!(y.to_bits(), erf_gelu_scalar(x).to_bits());
                continue;
            }
            let error =
                (f64::from(y) - f64::from(erf_gelu_scalar(x))).abs() / f64::from(x).abs().max(1.0);
            worst_gelu = worst_gelu.max(error);
            assert!(
                y.is_finite() && error <= super::super::tests::ERF_GELU_BOUND,
                "GELU x={x:e} y={y:e} error={error:e}"
            );
        }
        eprintln!(
            "NEON Erf worst absolute error={worst_erf:e}; GELU worst scaled error={worst_gelu:e}"
        );
    }

    #[test]
    fn native_erf_and_gelu_dispatch_reach_neon_and_bound_every_tail() {
        let before = [
            ERF_NEON_TEST_HITS.load(Ordering::Relaxed),
            GELU_NEON_TEST_HITS.load(Ordering::Relaxed),
        ];
        let source: Vec<f32> = (0..100).map(|i| -3.0 + i as f32 * 0.061).collect();
        let mut witnesses = [false; 2];
        for offset in 0..4 {
            for n in 0..=96 {
                let input = &source[offset..offset + n];
                for (op, witness) in witnesses.iter_mut().enumerate() {
                    let mut backing = vec![-12345.5; n + 8];
                    let actual = &mut backing[4..4 + n];
                    let mut expected = vec![0.0; n];
                    if op == 0 {
                        erf_f32_slice_native(input, actual);
                        // SAFETY: AArch64 test host supports NEON; equal slices.
                        unsafe { erf(input, &mut expected) };
                    } else {
                        erf_gelu_f32_slice_native(input, actual);
                        // SAFETY: same feature and equal-slice contract.
                        unsafe { gelu(input, &mut expected) };
                    }
                    for ((&x, &got), &vector) in input.iter().zip(actual.iter()).zip(&expected) {
                        let scalar = if op == 0 {
                            erf_scalar(x)
                        } else {
                            erf_gelu_scalar(x)
                        };
                        let want = if n < SIMD_MIN_LEN { scalar } else { vector };
                        assert_eq!(
                            got.to_bits(),
                            want.to_bits(),
                            "op={op}, offset={offset}, n={n}, x={x}"
                        );
                        *witness |= n >= SIMD_MIN_LEN && vector.to_bits() != scalar.to_bits();
                    }
                    assert!(
                        backing[..4]
                            .iter()
                            .chain(&backing[4 + n..])
                            .all(|&v| v == -12345.5)
                    );
                }
            }
        }
        // The numerical witnesses make this proof caller-specific even when
        // another test concurrently increments the diagnostic counters.
        assert!(
            witnesses.into_iter().all(|w| w),
            "both dispatches must expose a polynomial result distinct from the scalar control"
        );
        assert!(ERF_NEON_TEST_HITS.load(Ordering::Relaxed) > before[0]);
        assert!(GELU_NEON_TEST_HITS.load(Ordering::Relaxed) > before[1]);
    }

    #[test]
    fn native_neon_erf_and_gelu_parallel_and_nested_routes_are_bit_invariant() {
        let n = 2 * super::super::PAR_MIN_LEN + 3;
        let input: Vec<f32> = (0..n)
            .map(|i| -6.0 + (i % 100_003) as f32 * 0.00012)
            .collect();
        for op in [erf_f32_slice_native, erf_gelu_f32_slice_native] {
            let mut baseline = vec![0.0f32; n];
            super::super::serial_scope(|| op(&input, &mut baseline));
            let mut parallel = vec![0.0f32; n];
            let before = super::super::PARALLEL_DISPATCHES.with(|c| c.get());
            op(&input, &mut parallel);
            if crate::task_runtime::width() > 1 {
                assert!(
                    super::super::PARALLEL_DISPATCHES.with(|c| c.get()) > before,
                    "native entry must actually reach the task scheduler"
                );
            }
            assert!(
                parallel
                    .iter()
                    .zip(&baseline)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            for width in [1, 2, 4, 8] {
                let mut output = vec![0.0f32; n];
                rayon::ThreadPoolBuilder::new()
                    .num_threads(width)
                    .build()
                    .unwrap()
                    .install(|| {
                        let before = super::super::PARALLEL_DISPATCHES.with(|c| c.get());
                        op(&input, &mut output);
                        assert_eq!(
                            super::super::PARALLEL_DISPATCHES.with(|c| c.get()),
                            before,
                            "nested Rayon calls must stay serial"
                        );
                    });
                assert!(
                    output
                        .iter()
                        .zip(&baseline)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "thread width {width} changed a polynomial or tail"
                );
            }
        }
    }
    #[test]
    fn neon_bias_gelu_matches_separate_add_including_partial_rows() {
        let input: Vec<f32> = (0..100).map(|i| -3.0 + i as f32 * 0.063).collect();
        for width in 1..=13 {
            let bias: Vec<f32> = (0..width).map(|i| i as f32 * 0.11 - 0.7).collect();
            for n in 0..=96 {
                let added: Vec<f32> = input[..n]
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| x + bias[i % width])
                    .collect();
                let mut expected = vec![0.0; n];
                let mut backing = vec![-12345.5; n + 8];
                erf_gelu_f32_slice_native(&added, &mut expected);
                super::super::erf_gelu_bias_f32_slice(
                    &input[..n],
                    &bias,
                    width,
                    &mut backing[4..4 + n],
                );
                assert!(
                    backing[4..4 + n]
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "width={width}, n={n}"
                );
                assert!(
                    backing[..4]
                        .iter()
                        .chain(&backing[4 + n..])
                        .all(|&x| x == -12345.5)
                );
            }
        }
    }
    /// Same-process wall-only comparison. Production uses native; MLAS and the
    /// exact former scalar body are explicit controls, never runtime toggles.
    #[cfg(feature = "mlas")]
    #[test]
    #[ignore = "wall diagnostic; run explicitly under the shared host lock"]
    fn neon_erf_wall_ab() {
        use std::hint::black_box;
        use std::time::Instant;
        let argument = |name: &str, default: usize| {
            std::env::var(name)
                .map_or(Ok(default), |v| v.parse::<usize>())
                .unwrap_or_else(|e| panic!("{name} must be an unsigned integer: {e}"))
        };
        let width = argument("NXRT_ERF_BENCH_WIDTH", 1);
        let reps = argument("NXRT_ERF_BENCH_REPS", 21);
        let order_offset = argument("NXRT_ERF_BENCH_ORDER", 0);
        assert!(
            [1, 4, 8].contains(&width),
            "benchmark width must be 1, 4 or 8"
        );
        assert!(
            (1..=101).contains(&reps),
            "benchmark repetitions must be 1..101"
        );
        let mut lengths = [32usize, 257, 4096, 65_537, 4_194_307];
        let rotate = order_offset % lengths.len();
        lengths.rotate_left(rotate);
        crate::task_runtime::set_task_thread_budget(Some(width)).unwrap();
        assert!(rayon::current_thread_index().is_none());
        assert!(!crate::task_runtime::in_task());
        assert_eq!(
            crate::task_runtime::width(),
            width,
            "diagnostic must own the requested native task-pool width"
        );
        {
            for mode_index in 0..2 {
                let serial = (mode_index + order_offset) % 2 == 0;
                let mode = if serial { "serial" } else { "runtime" };
                for op_index in 0..2 {
                    let op_index = (op_index + order_offset) % 2;
                    let op = if op_index == 0 { "erf" } else { "gelu" };
                    for n in lengths {
                        let input: Vec<f32> = (0..n)
                            .map(|i| {
                                -6.0 + (i.wrapping_mul(7919) % 100_003) as f32 * (12.0 / 100_002.0)
                            })
                            .collect();
                        let mut output = vec![0.0f32; n];
                        let iters = (1_048_576 / n).clamp(1, 8192);
                        let run = |arm: usize, out: &mut [f32]| {
                            let kernel = || {
                                let input = black_box(input.as_slice());
                                let out = black_box(out);
                                match (op_index, arm) {
                                    (0, 0) => super::super::erf_scalar_before(input, out),
                                    (1, 0) => super::super::erf_gelu_scalar_before(input, out),
                                    (0, 1 | 2) => erf_f32_slice_native(input, out),
                                    (1, 1 | 2) => erf_gelu_f32_slice_native(input, out),
                                    (0, 3) => super::super::erf_f32_slice(input, out),
                                    (1, 3) => super::super::erf_gelu_f32_slice(input, out),
                                    _ => unreachable!(
                                        "the wall diagnostic has two operations and four arms"
                                    ),
                                }
                            };
                            if serial {
                                super::super::serial_scope(kernel);
                            } else {
                                kernel();
                            }
                        };
                        for arm in 0..4 {
                            let before = super::super::PARALLEL_DISPATCHES.with(|c| c.get());
                            let tasks_before = crate::task_runtime::testing::counters();
                            run(arm, &mut output);
                            let tasks_after = crate::task_runtime::testing::counters();
                            let worker_tasks = (tasks_after.tasks - tasks_before.tasks)
                                - (tasks_after.tasks_by_dispatcher
                                    - tasks_before.tasks_by_dispatcher);
                            let dispatched =
                                super::super::PARALLEL_DISPATCHES.with(|c| c.get()) > before;
                            assert_eq!(
                                dispatched,
                                !serial && width > 1 && n >= super::super::PAR_MIN_LEN,
                                "{mode}/{op}/{n}/arm{arm}: measured scheduling path differs"
                            );
                            assert_eq!(
                                worker_tasks > 0,
                                dispatched,
                                "{mode}/{op}/{n}/arm{arm}: require actual worker participation for parallel evidence"
                            );
                            let mut worst = 0.0f64;
                            let mut digest = 0xcbf2_9ce4_8422_2325u64;
                            for (&x, &y) in input.iter().zip(&output) {
                                let oracle = if op_index == 0 {
                                    erf_scalar(x)
                                } else {
                                    erf_gelu_scalar(x)
                                };
                                let scale = if op_index == 0 {
                                    1.0
                                } else {
                                    f64::from(x).abs().max(1.0)
                                };
                                let error = (f64::from(y) - f64::from(oracle)).abs() / scale;
                                let bound = if op_index == 0 {
                                    super::super::tests::ERF_BOUND
                                } else {
                                    super::super::tests::ERF_GELU_BOUND
                                };
                                assert!(
                                    y.is_finite() && error <= bound,
                                    "{op} arm={arm}, x={x:e}, error={error:e}"
                                );
                                worst = worst.max(error);
                                digest =
                                    (digest ^ u64::from(y.to_bits())).wrapping_mul(0x100_0000_01b3);
                            }
                            eprintln!(
                                "NEON_ERF_VALIDATE,{width},{mode},{op},{n},{arm},{digest},{worst},{},{}",
                                u8::from(dispatched),
                                worker_tasks
                            );
                            for _ in 0..3 {
                                run(arm, &mut output);
                            }
                        }
                        for rep in 0..reps {
                            for order in 0..4 {
                                let arm = (rep + order + order_offset) % 4;
                                let start = Instant::now();
                                for _ in 0..iters {
                                    run(arm, &mut output);
                                }
                                let ns = start.elapsed().as_nanos();
                                black_box(output.as_slice());
                                // A lightweight output token, not a full hash:
                                // full-buffer digest/oracle checks are above.
                                let token = [0, n / 2, n - 1].into_iter().fold(
                                    0xcbf2_9ce4_8422_2325u64,
                                    |h, i| {
                                        (h ^ u64::from(output[i].to_bits()))
                                            .wrapping_mul(0x100_0000_01b3)
                                    },
                                );
                                println!(
                                    "NEON_ERF_SAMPLE,{width},{mode},{op},{n},{iters},{rep},{arm},{ns},{token}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
