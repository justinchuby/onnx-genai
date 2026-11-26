use crate::common::{
    assert_close, build_graph, decode_floats, float_input, input, run_cpu, run_cuda,
};
use onnx_runtime_ep_api::ExecutionProvider;
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ir::{Attribute, DataType, static_shape};

pub(crate) const OP: &str = "MeanVarianceNormalization";

pub(crate) fn goldens(cuda: Option<&CudaExecutionProvider>) {
    let check =
        |dtype, shape: &[usize], x: &[f32], attrs: &[(&str, Attribute)], expected: &[f32]| {
            let inputs = [float_input(dtype, shape, x)];
            let outputs = [(dtype, shape.to_vec())];
            let cpu = run_cpu(OP, "", 29, &inputs, &outputs, attrs);
            let cpu = decode_floats(&cpu[0], dtype);
            assert_close("CPU vs golden", dtype, &cpu, expected, 1e-6);
            if let Some(ep) = cuda {
                let gpu = run_cuda(ep, OP, "", 29, &inputs, &outputs, attrs);
                let gpu = decode_floats(&gpu[0], dtype);
                assert_close("CUDA vs golden", dtype, &gpu, expected, 1e-6);
                assert_close("CUDA vs CPU", dtype, &gpu, &cpu, 1e-6);
            }
        };

    // Default NCHW axes, population variance and cancellation: legacy
    // E[X²]-E[X]² rounds to zero for the first channel, not one.
    check(
        DataType::Float32,
        &[2, 2, 1, 1],
        &[9999., -2., 10001., 6.],
        &[],
        &[-1., -1., 1., 1.],
    );
    // Explicit epsilon of 2 goes outside the square root: 2 / (sqrt(4) + 2).
    check(
        DataType::Float32,
        &[2, 2],
        &[0., 4., 8., 4.],
        &[
            ("axes", Attribute::Ints(vec![-1])),
            ("epsilon", Attribute::Float(2.)),
        ],
        &[-0.5, 0.5, 0.5, -0.5],
    );
    // Empty axes reduce all dimensions. The small standard deviation makes
    // the default epsilon observable: 1e-9 / (1e-9 + 1e-9).
    let all = [("axes", Attribute::Ints(vec![]))];
    check(
        DataType::Float32,
        &[2, 1],
        &[-1e-9, 1e-9],
        &all,
        &[-0.5, 0.5],
    );
    check(DataType::Float32, &[], &[7.], &all, &[0.]);
    check(DataType::Float32, &[0, 3], &[], &all, &[]);
    // An f16 square of 1024 overflows; f32 intermediates do not.
    check(DataType::Float16, &[2], &[-1024., 1024.], &all, &[-1., 1.]);
    // In bf16 the mean 257 would round to 256 before the subtraction.
    check(DataType::BFloat16, &[2], &[256., 258.], &all, &[-1., 1.]);
    // Two separated reduction runs, more than one group, and more elements
    // per group than CUDA threads. Each group alternates -2 and +2 about its own mean.
    let values: Vec<f32> = (0..2 * 3 * 258)
        .map(|i| (i / 258 % 3) as f32 * 10. + if i % 2 == 0 { -2. } else { 2. })
        .collect();
    let expected: Vec<f32> = (0..values.len())
        .map(|i| if i % 2 == 0 { -1. } else { 1. })
        .collect();
    check(
        DataType::Float32,
        &[2, 3, 258],
        &values,
        &[("axes", Attribute::Ints(vec![0, -1]))],
        &expected,
    );

    // A real f64 check: casting either input to f32 would collapse the pair.
    let inputs = [input(DataType::Float64, &[2], &[1e12f64 - 2., 1e12 + 2.])];
    let outputs = [(DataType::Float64, vec![2])];
    let attrs = [
        ("axes", Attribute::Ints(vec![0])),
        ("epsilon", Attribute::Float(2.)),
    ];
    let cpu = run_cpu(OP, "", 29, &inputs, &outputs, &attrs);
    let expected: Vec<u8> = [-0.5f64, 0.5]
        .into_iter()
        .flat_map(f64::to_ne_bytes)
        .collect();
    assert_eq!(cpu[0], expected);
    if let Some(ep) = cuda {
        assert_eq!(run_cuda(ep, OP, "", 29, &inputs, &outputs, &attrs), cpu);
    }
}

pub(crate) fn claims(ep: &dyn ExecutionProvider) {
    for (opset, supported) in [(9, false), (13, false), (28, false), (29, true), (30, true)] {
        let (graph, id) = build_graph(
            OP,
            "",
            opset,
            &[input(DataType::Float64, &[2], &[0f64, 1.])],
            &[(DataType::Float64, vec![2])],
            &[("axes", Attribute::Ints(vec![0]))],
        );
        let claim = ep.supports_op(
            graph.node(id),
            opset,
            &[static_shape([2])],
            &[DataType::Float64],
            &[],
        );
        assert_eq!(
            claim.is_supported(),
            supported,
            "opset {opset}: {:?}",
            claim.reason()
        );
        assert!(
            !ep.supports_op(
                graph.node(id),
                opset,
                &[static_shape([2])],
                &[DataType::Int32],
                &[]
            )
            .is_supported()
        );
    }
    for (axes, reason) in [(vec![1], "out of range"), (vec![0, -1], "duplicate")] {
        let (graph, id) = build_graph(
            OP,
            "",
            29,
            &[input(DataType::Float32, &[2], &[0f32, 1.])],
            &[(DataType::Float32, vec![2])],
            &[("axes", Attribute::Ints(axes))],
        );
        let claim = ep.supports_op(
            graph.node(id),
            29,
            &[static_shape([2])],
            &[DataType::Float32],
            &[],
        );
        assert!(
            claim.reason().is_some_and(|text| text.contains(reason)),
            "{:?}",
            claim.reason()
        );
    }
}
