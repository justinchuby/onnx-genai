use super::*;
use onnx_runtime_ir::dequantize_packed::Format;

fn code(format: BlockFormat, feature: usize, block: usize, lane: usize) -> i8 {
    if matches!(format, BlockFormat::Q8_0) {
        ((feature * 17 + block * 11 + lane * 31) % 256) as u8 as i8
    } else {
        ((feature * 13 + block * 7 + lane * 3) % 16) as i8 - 8
    }
}
fn scale(feature: usize, block: usize) -> f32 {
    let sign = if feature.is_multiple_of(2) { 1. } else { -1. };
    sign * 0.125 * (1 + (feature + block) % 4) as f32
}
fn weights(format: BlockFormat, k: usize, n: usize) -> Vec<u8> {
    let dot = DotFormat::from_format(format).unwrap();
    let blocks = k.div_ceil(QK);
    let mut packed = vec![0; n * blocks * dot.bytes()];
    for feature in 0..n {
        for block in 0..blocks {
            let bytes = &mut packed[(feature * blocks + block) * dot.bytes()
                ..(feature * blocks + block + 1) * dot.bytes()];
            bytes[..2].copy_from_slice(&half::f16::from_f32(scale(feature, block)).to_le_bytes());
            match dot {
                DotFormat::Q8_0 => {
                    for lane in 0..32 {
                        bytes[2 + lane] = code(format, feature, block, lane) as u8;
                    }
                }
                DotFormat::Q4_0 => {
                    for lane in 0..16 {
                        bytes[2 + lane] = (code(format, feature, block, lane) + 8) as u8
                            | (((code(format, feature, block, lane + 16) + 8) as u8) << 4);
                    }
                }
            }
        }
    }
    packed
}
fn reference(
    format: BlockFormat,
    x: &[f32],
    m: usize,
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
) -> Vec<(f64, f64)> {
    let mut result = Vec::new();
    for row in 0..m {
        for feature in 0..n {
            let mut sum = 0.;
            let mut magnitude = 0.;
            for index in 0..k {
                let block = index / 32;
                let start = row * k + block * 32;
                let input = &x[start..(start + 32).min((row + 1) * k)];
                let maximum = input.iter().fold(0f32, |m, x| m.max(x.abs()));
                let step = maximum / 127.;
                let q = if maximum == 0. {
                    0.
                } else {
                    (x[row * k + index] / step).round().clamp(-127., 127.)
                };
                let activation = half::f16::from_f32(step).to_f64() * f64::from(q);
                let weight = f64::from(scale(feature, block))
                    * f64::from(code(format, feature, block, index % 32));
                sum += activation * weight;
                magnitude += (activation * weight).abs();
            }
            let bias = f64::from(bias.map_or(0., |b| b[feature]));
            result.push((sum + bias, magnitude + bias.abs()));
        }
    }
    result
}
fn assert_reference(actual: &[f32], expected: &[(f64, f64)], blocks: usize) {
    for (&actual, &(expected, magnitude)) in actual.iter().zip(expected) {
        let bound = 8. * f64::from(f32::EPSILON) * (blocks + 1) as f64 * magnitude + 1e-7;
        assert!(
            (f64::from(actual) - expected).abs() <= bound,
            "{actual} != {expected}; bound {bound}"
        );
    }
}

#[test]
fn q8_and_q4_grouped_dots_use_caller_lanes_and_an_independent_oracle() {
    for format in [BlockFormat::Q8_0, BlockFormat::Onnx(Format::Q4_0)] {
        for (m, k, n) in [
            (1, 1, 1),
            (1, 32, 7),
            (3, 65, 7),
            (16, 96, 9),
            (2, 0, 3),
            (0, 65, 3),
            (3, 65, 0),
        ] {
            let x: Vec<_> = (0..m * k)
                .map(|i| ((i * 17 % 97) as f32 - 48.) / 29.)
                .collect();
            let packed = weights(format, k, n);
            let bias: Vec<_> = (0..n).map(|i| i as f32 * 0.25).collect();
            let plan = PackedProjectionPlan::for_current_pool(ActivationArithmetic::Q8_0);
            let layout = plan.layout(m, k, n).unwrap();
            let mut backing = vec![91.; layout.bytes() / 4 + 4];
            let mut actual = vec![37.; m * n];
            {
                let mut scratch =
                    ProjectionScratch::test_binding(&mut backing[..layout.bytes() / 4]);
                plan.project(
                    format,
                    &packed,
                    &x,
                    m,
                    k,
                    n,
                    Some(&bias),
                    &mut scratch,
                    &mut actual,
                )
                .unwrap();
            }
            assert_reference(
                &actual,
                &reference(format, &x, m, k, n, Some(&bias)),
                k.div_ceil(32),
            );
            assert!(backing[layout.bytes() / 4..].iter().all(|&v| v == 91.));
        }
    }
}

#[test]
fn conversion_once_per_group_is_independent_of_channels_and_selected_pool() {
    for threads in [1, 3] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let (m, k) = (3, 65);
                let x = vec![1.; m * k];
                for n in [1, 7, 31] {
                    let plan = PackedProjectionPlan::for_current_pool(ActivationArithmetic::Q8_0);
                    let mut storage = vec![0.; plan.layout(m, k, n).unwrap().bytes() / 4];
                    let mut scratch = ProjectionScratch::test_binding(&mut storage);
                    let mut out = vec![0.; m * n];
                    ACTIVATION_CONVERSIONS.with(|v| v.set(0));
                    plan.project(
                        BlockFormat::Q8_0,
                        &weights(BlockFormat::Q8_0, k, n),
                        &x,
                        m,
                        k,
                        n,
                        None,
                        &mut scratch,
                        &mut out,
                    )
                    .unwrap();
                    assert_eq!(
                        ACTIVATION_CONVERSIONS.with(std::cell::Cell::get),
                        m * k.div_ceil(32)
                    );
                    assert_reference(
                        &out,
                        &reference(BlockFormat::Q8_0, &x, m, k, n, None),
                        k.div_ceil(32),
                    );
                }
            });
    }
}

#[test]
fn dirty_padding_and_partial_conversion_failure_are_fully_rewritten_on_retry() {
    let (m, n) = (3, 7);
    let plan = PackedProjectionPlan::Q8_0;
    let mut storage = vec![17.; plan.layout(m, 96, n).unwrap().bytes() / 4];
    let mut scratch = ProjectionScratch::test_binding(&mut storage);
    let mut out = vec![0.; m * n];
    for k in [96, 65, 33, 1, 65] {
        let x: Vec<_> = (0..m * k).map(|i| (i % 31) as f32 / 13. - 1.).collect();
        let packed = weights(BlockFormat::Q8_0, k, n);
        plan.project(
            BlockFormat::Q8_0,
            &packed,
            &x,
            m,
            k,
            n,
            None,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        assert_reference(
            &out,
            &reference(BlockFormat::Q8_0, &x, m, k, n, None),
            k.div_ceil(32),
        );
    }
    let mut x = vec![1.; m * 65];
    x[35] = f32::NAN;
    let unchanged = out.clone();
    let packed = weights(BlockFormat::Q8_0, 65, n);
    let error = plan
        .project(
            BlockFormat::Q8_0,
            &packed,
            &x,
            m,
            65,
            n,
            None,
            &mut scratch,
            &mut out,
        )
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("row 0 block 1") && error.contains("float32"),
        "{error}"
    );
    assert_eq!(out, unchanged);
    x[35] = 2.;
    plan.project(
        BlockFormat::Q8_0,
        &packed,
        &x,
        m,
        65,
        n,
        None,
        &mut scratch,
        &mut out,
    )
    .unwrap();
    assert_reference(&out, &reference(BlockFormat::Q8_0, &x, m, 65, n, None), 3);
}

#[test]
fn short_scratch_refuses_before_activation_validation_or_destination_writes() {
    let mut storage = [19.; 8];
    let mut out = [37.];
    let mut scratch = ProjectionScratch::test_binding(&mut storage);
    let error = PackedProjectionPlan::Q8_0
        .project(
            BlockFormat::Q8_0,
            &weights(BlockFormat::Q8_0, 32, 1),
            &[f32::NAN; 32],
            1,
            32,
            1,
            None,
            &mut scratch,
            &mut out,
        )
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("scratch") && !error.contains("non-finite"),
        "{error}"
    );
    assert_eq!(out, [37.]);

    assert_eq!(storage, [19.; 8]);
}

#[test]
fn unsupported_formats_and_activation_or_weight_ranges_have_actionable_errors() {
    assert!(
        validate_format(ActivationArithmetic::Q8_0, BlockFormat::Mxfp4)
            .unwrap_err()
            .to_string()
            .contains("float32")
    );
    validate_format(ActivationArithmetic::Float32, BlockFormat::Mxfp4).unwrap();
    let plan = PackedProjectionPlan::Q8_0;
    let mut storage = vec![0.; plan.layout(1, 32, 1).unwrap().bytes() / 4];
    let mut scratch = ProjectionScratch::test_binding(&mut storage);
    let mut out = [71.];
    for value in [f32::NAN, f32::INFINITY, f32::MAX, f32::MIN_POSITIVE] {
        let message = plan
            .project(
                BlockFormat::Q8_0,
                &weights(BlockFormat::Q8_0, 32, 1),
                &[value; 32],
                1,
                32,
                1,
                None,
                &mut scratch,
                &mut out,
            )
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("row 0 block 0") && message.contains("float32"),
            "{message}"
        );
        assert_eq!(out, [71.]);
    }
    let mut w = weights(BlockFormat::Q8_0, 32, 1);
    w[..2].copy_from_slice(&half::f16::INFINITY.to_le_bytes());
    let message = plan
        .project(
            BlockFormat::Q8_0,
            &w,
            &[1.; 32],
            1,
            32,
            1,
            None,
            &mut scratch,
            &mut out,
        )
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("weight output 0 block 0") && message.contains("float32"),
        "{message}"
    );
}
