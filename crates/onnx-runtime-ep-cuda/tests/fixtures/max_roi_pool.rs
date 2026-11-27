#![allow(dead_code)]

use crate::common::{
    Tensor, build_graph, float_input, input, require_cuda, run_cpu, run_cuda, try_run_cpu,
    try_run_cuda,
};
use onnx_runtime_ep_api::{
    DeviceId, DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView,
};
use onnx_runtime_ir::{Attribute, DataType};

const OP: &str = "MaxRoiPool";

struct Case {
    opset: u64,
    inputs: [Tensor; 2],
    expected: Tensor,
    attrs: Vec<(&'static str, Attribute)>,
}

fn cases() -> Vec<Case> {
    use DataType::{BFloat16, Float16, Float32, Float64};

    let mut cases = Vec::new();
    for (opset, dtype) in
        (19..=29)
            .map(|v| (v, Float32))
            .chain([(19, Float16), (22, BFloat16), (29, BFloat16)])
    {
        cases.push(Case {
            opset,
            inputs: [
                float_input(
                    dtype,
                    &[2, 2, 3, 3],
                    &[
                        1., 2., 3., 4., 5., 6., 7., 8., 9., -1., -2., -3., -4., -5., -6., -7., -8.,
                        -9., 11., 12., 13., 14., 15., 16., 17., 18., 19., -11., -12., -13., -14.,
                        -15., -16., -17., -18., -19.,
                    ],
                ),
                float_input(dtype, &[2, 5], &[1., 0., 0., 2., 2., 0., 0., 0., 2., 2.]),
            ],
            attrs: vec![("pooled_shape", Attribute::Ints(vec![2, 2]))],
            // A 3x3 inclusive region splits into overlapping 2x2 bins.
            expected: float_input(
                dtype,
                &[2, 2, 2, 2],
                &[
                    15., 16., 18., 19., -11., -12., -14., -15., 5., 6., 8., 9., -1., -2., -4., -5.,
                ],
            ),
        });
    }

    cases.push(Case {
        opset: 22,
        inputs: [
            float_input(Float32, &[1, 1, 2, 3], &[-1., -2., -3., -4., -5., -6.]),
            float_input(
                Float32,
                &[4, 5],
                &[
                    0., -1., 1., 3., 1., 0., 1., 1., -3., -1., 0., 7., 1., 9., 1., 0., 1., -1., 3.,
                    -1.,
                ],
            ),
        ],
        attrs: vec![
            ("pooled_shape", Attribute::Ints(vec![1, 4])),
            ("spatial_scale", Attribute::Float(0.5)),
        ],
        // Half ties round away from zero. The first ROI starts at x=-1;
        // clipping happens after binning. The inverted ROI uses a 1x1 extent.
        // The last two ROIs lie entirely outside the feature map.
        expected: float_input(
            Float32,
            &[4, 1, 1, 4],
            &[
                0., -4., -5., -6., -5., -5., -5., -5., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
        ),
    });

    const EXACT: f64 = 16_777_217.0;
    cases.push(Case {
        opset: 21,
        inputs: [
            input(Float64, &[1, 1, 1, 2], &[EXACT - 1., EXACT]),
            input(
                Float64,
                &[2, 5],
                &[0., 0.499_999_99, 0., 0.499_999_99, 0., 0., 0.5, 0., 0.5, 0.],
            ),
        ],
        attrs: vec![("pooled_shape", Attribute::Ints(vec![1, 1]))],
        // Neither the maximum nor the ROI coordinates may be narrowed to f32.
        expected: input(Float64, &[2, 1, 1, 1], &[EXACT - 1., EXACT]),
    });
    for (x_shape, rois_shape, rois, y_shape, y) in [
        (
            [1, 1, 0, 3],
            [1, 5],
            &[0., 0., 0., 2., 2.][..],
            [1, 1, 1, 1],
            &[0.][..],
        ),
        ([1, 1, 0, 3], [0, 5], &[][..], [0, 1, 1, 1], &[][..]),
    ] {
        cases.push(Case {
            opset: 29,
            inputs: [
                float_input(Float32, &x_shape, &[]),
                float_input(Float32, &rois_shape, rois),
            ],
            expected: float_input(Float32, &y_shape, y),
            attrs: vec![("pooled_shape", Attribute::Ints(vec![1, 1]))],
        });
    }
    cases
}

fn invalid_cases() -> impl Iterator<Item = ([Tensor; 2], &'static str)> {
    [
        ([1., 0., 0., 0., 0.], "batch_id"),
        ([0.5, 0., 0., 0., 0.], "batch_id"),
        ([-1., 0., 0., 0., 0.], "batch_id"),
        ([0., f32::NAN, 0., 0., 0.], "coordinate"),
    ]
    .into_iter()
    .map(|(roi, reason)| {
        (
            [
                float_input(DataType::Float32, &[1, 1, 1, 1], &[1.]),
                float_input(DataType::Float32, &[1, 5], &roi),
            ],
            reason,
        )
    })
}

pub fn cpu_conformance() {
    for case in cases() {
        let output = [(case.expected.dtype, case.expected.shape)];
        assert_eq!(
            run_cpu(OP, "", case.opset, &case.inputs, &output, &case.attrs)[0],
            case.expected.bytes,
            "{output:?} at opset {}",
            case.opset
        );
    }
    for (inputs, reason) in invalid_cases() {
        let err = try_run_cpu(
            OP,
            "",
            22,
            &inputs,
            &[(DataType::Float32, vec![1; 4])],
            &[("pooled_shape", Attribute::Ints(vec![1, 1]))],
        )
        .unwrap_err();
        assert!(err.to_string().contains(reason), "{err}");
    }
}

pub fn strict_cuda_conformance() {
    let ep = require_cuda();
    for case in cases() {
        let output = [(case.expected.dtype, case.expected.shape)];
        assert_eq!(
            run_cuda(&ep, OP, "", case.opset, &case.inputs, &output, &case.attrs)[0],
            case.expected.bytes,
            "{output:?} at opset {}",
            case.opset
        );
    }
    for (inputs, reason) in invalid_cases() {
        let err = try_run_cuda(
            &ep,
            OP,
            "",
            22,
            &inputs,
            &[(DataType::Float32, vec![1; 4])],
            &[("pooled_shape", Attribute::Ints(vec![1, 1]))],
        )
        .unwrap_err();
        assert!(err.to_string().contains(reason), "{err}");
    }

    let dtype = DataType::Float32;
    let x = [1.0_f32];
    let roi = [0.0_f32; 5];
    let mut y = [0.0_f32];
    let (graph, id) = build_graph(
        OP,
        "",
        22,
        &[input(dtype, &[1; 4], &x), input(dtype, &[1, 5], &roi)],
        &[(dtype, vec![1; 4])],
        &[("pooled_shape", Attribute::Ints(vec![1, 1]))],
    );
    let kernel = ep
        .get_kernel(graph.node(id), &[vec![1; 4], vec![1, 5]], 22)
        .unwrap();
    let cpu = DeviceId::cpu();
    let err = kernel
        .execute(
            &[
                TensorView::new(DevicePtr(x.as_ptr().cast()), dtype, &[1; 4], &[1; 4], cpu),
                TensorView::new(DevicePtr(roi.as_ptr().cast()), dtype, &[1, 5], &[5, 1], cpu),
            ],
            &mut [TensorMut::new(
                DevicePtrMut(y.as_mut_ptr().cast()),
                dtype,
                &[1; 4],
                &[1; 4],
                cpu,
            )],
        )
        .unwrap_err();
    assert!(err.to_string().contains("X is on"), "{err}");
}
