use crate::common::{Tensor, absent_input, assert_close, decode_floats, float_input, input};
use onnx_runtime_ir::{Attribute, DataType};

pub(crate) struct Case {
    pub(crate) opset: u64,
    pub(crate) inputs: Vec<Tensor>,
    pub(crate) output: (DataType, Vec<usize>),
    expected: Vec<f32>,
    pub(crate) attrs: Vec<(&'static str, Attribute)>,
}

pub(crate) fn cases() -> Vec<Case> {
    use DataType::{BFloat16, Float16, Float32, Float64};

    // ONNX backend test: test_basic_deform_conv_with_padding / without_padding
    // and test_deform_conv_with_mask_bias.
    let x = [0., 1., 2., 3., 4., 5., 6., 7., 8.];
    let mut pad_off = [0.; 128];
    pad_off[0] = 0.5;
    pad_off[86] = -0.1;
    let mut off = [0.; 32];
    off[0] = 0.5;
    off[21] = -0.1;
    let mut mask = [1.; 16];
    mask[11] = 0.2;
    let mut upstream_group_offset = [0.; 64];
    upstream_group_offset[0] = 0.5;
    upstream_group_offset[53] = -0.1;
    let mut v = vec![
        Case {
            opset: 19,
            inputs: vec![
                float_input(Float32, &[1, 1, 3, 3], &x),
                float_input(Float32, &[1, 1, 2, 2], &[1.; 4]),
                float_input(Float32, &[1, 8, 4, 4], &pad_off),
            ],
            output: (Float32, vec![1, 1, 4, 4]),
            expected: vec![
                0., 1., 3., 2., 3., 8., 11.9, 7., 9., 20., 24., 13., 6., 13., 15., 8.,
            ],
            attrs: vec![("pads", Attribute::Ints(vec![1; 4]))],
        },
        Case {
            opset: 19,
            inputs: vec![
                float_input(Float32, &[1, 1, 3, 3], &x),
                float_input(Float32, &[1, 1, 2, 2], &[1.; 4]),
                float_input(Float32, &[1, 8, 2, 2], &off),
                float_input(Float32, &[1], &[1.]),
                float_input(Float32, &[1, 4, 2, 2], &mask),
            ],
            output: (Float32, vec![1, 1, 2, 2]),
            expected: vec![10.5, 12.9, 21., 19.4],
            attrs: vec![("kernel_shape", Attribute::Ints(vec![2, 2]))],
        },
        Case {
            opset: 22,
            inputs: vec![
                float_input(
                    Float32,
                    &[1, 2, 3, 3],
                    &[
                        0., 1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
                    ],
                ),
                float_input(Float32, &[1, 2, 2, 2], &[1.; 8]),
                float_input(Float32, &[1, 16, 2, 2], &upstream_group_offset),
            ],
            output: (Float32, vec![1, 1, 2, 2]),
            expected: vec![33.5, 32.1, 32., 32.],
            attrs: vec![("offset_group", Attribute::Int(2))],
        },
        // 1-D, group boundary crosses an offset-group boundary. With a
        // one-tap kernel and +0.5 offsets, the right neighbor is zero-extended.
        Case {
            opset: 22,
            inputs: vec![
                float_input(
                    BFloat16,
                    &[1, 6, 2],
                    &[2., 4., 6., 8., 10., 12., 14., 16., 18., 20., 22., 24.],
                ),
                float_input(BFloat16, &[2, 3, 1], &[1.; 6]),
                float_input(BFloat16, &[1, 3, 2], &[0., 0., 0.5, 0.5, -1., -0.5]),
                absent_input(BFloat16),
                float_input(BFloat16, &[1, 3, 2], &[1., 2., 1., 1., 1., 1.]),
            ],
            output: (BFloat16, vec![1, 2, 2]),
            // At position 1, the last offset group samples at 1 - 0.5.
            expected: vec![19., 30., 15., 50.],
            attrs: vec![
                ("group", Attribute::Int(2)),
                ("offset_group", Attribute::Int(3)),
            ],
        },
        // The middle offset group straddles the two convolution groups.
        // Its samples are shared, but each group still reads its own channels.
        Case {
            opset: 22,
            inputs: vec![
                float_input(
                    Float32,
                    &[1, 6, 1, 3],
                    &[
                        2., 4., 6., 6., 8., 10., 10., 12., 14., 14., 16., 18., 18., 20., 22., 22.,
                        24., 26.,
                    ],
                ),
                float_input(
                    Float32,
                    &[2, 3, 1, 2],
                    &[1., 2., 1., 2., 1., 2., 1., 2., 1., 2., 1., 2.],
                ),
                float_input(
                    Float32,
                    &[1, 12, 1, 2],
                    &[
                        0., 0., 0., 0., 0., 0., -1., -1., 0., 0., 0.5, 0.5, 0., 0., 0., 0., 0., 0.,
                        -1., -0.5, 0., 0., 1., 1.,
                    ],
                ),
                absent_input(Float32),
                float_input(
                    Float32,
                    &[1, 6, 1, 2],
                    &[1., 2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                ),
            ],
            output: (Float32, vec![1, 2, 1, 2]),
            expected: vec![59., 89., 143., 95.],
            attrs: vec![
                ("group", Attribute::Int(2)),
                ("offset_group", Attribute::Int(3)),
            ],
        },
        // Trilinear at half-pixel in all axes of a singleton input: 8/8.
        Case {
            opset: 19,
            inputs: vec![
                float_input(Float16, &[1, 1, 1, 1, 1], &[8.]),
                float_input(Float16, &[1, 1, 1, 1, 1], &[2.]),
                float_input(Float16, &[1, 3, 1, 1, 1], &[0.5, -0.5, 0.5]),
            ],
            output: (Float16, vec![1, 1, 1, 1, 1]),
            expected: vec![2.],
            attrs: vec![],
        },
        Case {
            opset: 19,
            inputs: vec![
                input(Float64, &[1, 1, 1, 1], &[2.0_f64]),
                input(Float64, &[1, 1, 1, 1], &[3.0_f64]),
                input(Float64, &[1, 2, 1, 1], &[0.0_f64, -0.5]),
            ],
            output: (Float64, vec![1, 1, 1, 1]),
            expected: vec![3.],
            attrs: vec![],
        },
        // 1-D with dilation=2, stride=2, and asymmetric beginning padding.
        // The two sample pairs are (-1, 1) and (1, 3).
        Case {
            opset: 22,
            inputs: vec![
                float_input(Float32, &[1, 1, 4], &[2., 4., 8., 16.]),
                float_input(Float32, &[1, 1, 2], &[1., 2.]),
                float_input(Float32, &[1, 2, 2], &[0.; 4]),
                float_input(Float32, &[1], &[1.]),
            ],
            output: (Float32, vec![1, 1, 2]),
            expected: vec![9., 37.],
            attrs: vec![
                ("pads", Attribute::Ints(vec![1, 0])),
                ("dilations", Attribute::Ints(vec![2])),
                ("strides", Attribute::Ints(vec![2])),
            ],
        },
    ];
    // 2-D multi-output tile and tail, distinct rows, offset reused by all outputs.
    v.push(Case {
        opset: 22,
        inputs: vec![
            float_input(Float32, &[1, 1, 1, 2], &[4., 8.]),
            float_input(
                Float32,
                &[9, 1, 1, 1],
                &[1., 2., 3., 4., 5., 6., 7., 8., 9.],
            ),
            float_input(Float32, &[1, 2, 1, 2], &[0., 0., 0.5, 1.0]),
        ],
        output: (Float32, vec![1, 9, 1, 2]),
        expected: (1..=9).flat_map(|w| [6. * w as f32, 0.]).collect(),
        attrs: vec![],
    });
    v
}

pub(crate) fn compare(case: &Case, output: &[u8]) {
    let actual = if case.output.0 == DataType::Float64 {
        output
            .chunks_exact(8)
            .map(|b| f64::from_ne_bytes(b.try_into().unwrap()) as f32)
            .collect()
    } else {
        decode_floats(output, case.output.0)
    };
    assert_close("DeformConv", case.output.0, &actual, &case.expected, 2e-5);
}
