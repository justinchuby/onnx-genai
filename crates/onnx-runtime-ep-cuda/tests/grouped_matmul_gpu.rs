mod common;

use DataType::{BFloat16, Float16, Float32, Float64, Int32, Int64};
use common::{
    Tensor, absent_input, decode_floats, float_input, input, require_cuda, try_run_cpu,
    try_run_cuda_with_origin_offset,
};
use onnx_runtime_ep_api::ExecutionProvider;
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ir::{Attribute, DataType, Node, NodeId, ValueId, static_shape};

fn test_provider(ep: Option<&CudaExecutionProvider>, origin_offset: usize) {
    let run =
        |op, inputs: &[Tensor], outputs: &[(DataType, Vec<usize>)], attrs: &[(&str, Attribute)]| {
            match ep {
                Some(ep) => try_run_cuda_with_origin_offset(
                    ep,
                    op,
                    "",
                    29,
                    inputs,
                    outputs,
                    attrs,
                    origin_offset,
                ),
                None => try_run_cpu(op, "", 29, inputs, outputs, attrs),
            }
            .map(|mut y| y.remove(0))
        };
    let cpu = CpuExecutionProvider::new();
    let provider: &dyn ExecutionProvider = ep.map_or(&cpu, |ep| ep as _);
    let dtypes = [BFloat16, BFloat16, Int32];
    let shapes = [
        static_shape([4, 3]),
        static_shape([2, 3, 2]),
        static_shape([4, 1]),
    ];
    let node = Node::new(
        NodeId(0),
        "GroupedMatMul",
        vec![Some(ValueId(0)), Some(ValueId(1)), Some(ValueId(2))],
        vec![ValueId(4)],
    );
    assert!(
        provider
            .supports_op(&node, 29, &shapes, &dtypes, &[])
            .is_supported()
    );
    assert!(
        !provider
            .supports_op(&node, 28, &shapes, &dtypes, &[])
            .is_supported()
    );
    assert!(
        !provider
            .supports_op(&node, 29, &shapes, &[Int32; 3], &[])
            .is_supported()
    );

    for dtype in [Float16, BFloat16, Float32, Float64] {
        let f = |shape: &[usize], v: &[f32]| {
            if dtype == Float64 {
                input(
                    dtype,
                    shape,
                    &v.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
                )
            } else {
                float_input(dtype, shape, v)
            }
        };
        let expected = [24., 30., 0., 6., 24., 30., 50., 71., 92., 113., 8., 29.];
        let x = f(&[2, 3], &[1., 2., -1., 3., 0., 4.]);
        let w = f(&[3, 2, 3], &(0..18).map(|i| i as f32).collect::<Vec<_>>());
        for trans_b in [-1, 2] {
            let attrs = [("transB", Attribute::Int(trans_b))];
            let ids = input(Int64, &[2, 3], &[2i64, 0, 2, 1, 2, 0]);
            let inputs = [x.clone(), w.clone(), ids, absent_input(dtype)];
            let result = run("GroupedMatMul", &inputs, &[(dtype, vec![2, 3, 2])], &attrs).unwrap();
            assert_eq!(decode_floats(&result, dtype), expected);
        }
        let x = f(&[2, 2], &[1., 2., 3., 4.]);
        let w = f(&[2, 2, 2], &[1., 0., 0., 1., 0., 1., 1., 0.]);
        let ids = input(Int32, &[2, 2], &[0i32, 1, 1, 0]);
        let b = f(&[2, 2], &[0.5, 1., 1.5, 2.]);
        let result = run(
            "GroupedMatMul",
            &[x, w, ids, b],
            &[(dtype, vec![2, 2, 2])],
            &[],
        )
        .unwrap();
        assert_eq!(
            decode_floats(&result, dtype),
            [1.5, 3., 3.5, 3., 5.5, 5., 3.5, 5.]
        );
        if matches!(dtype, Float16 | BFloat16) {
            let half_ulp = if dtype == Float16 {
                1. / 2048.
            } else {
                1. / 256.
            };
            let out = run(
                "GroupedMatMul",
                &[
                    f(&[1, 2], &[1., half_ulp]),
                    f(&[1, 2, 1], &[1., 1.]),
                    input(Int32, &[1, 1], &[0i32]),
                    f(&[1, 1], &[-1.]),
                ],
                &[(dtype, vec![1, 1, 1])],
                &[],
            )
            .unwrap();
            assert_eq!(decode_floats(&out, dtype), [0.]);
        }
        let x = f(&[2, 0], &[]);
        let w = f(&[2, 0, 1], &[]);
        let ids = input(Int32, &[2, 1], &[1i32, 0]);
        let b = f(&[2, 1], &[0.5, -2.0]);
        let result = run(
            "GroupedMatMul",
            &[x.clone(), w.clone(), ids, b],
            &[(dtype, vec![2, 1, 1])],
            &[],
        )
        .unwrap();
        assert_eq!(decode_floats(&result, dtype), [-2., 0.5]);
        let ids = input::<i64>(Int64, &[2, 0], &[]);
        assert!(
            run(
                "GroupedMatMul",
                &[x, w, ids],
                &[(dtype, vec![2, 0, 1])],
                &[]
            )
            .unwrap()
            .is_empty()
        );
    }

    let error = run(
        "GroupedMatMul",
        &[
            float_input(Float32, &[1, 1], &[1.]),
            float_input(Float32, &[1, 1, 1], &[2.]),
            input(Int32, &[1, 1], &[-1i32]),
        ],
        &[(Float32, vec![1, 1, 1])],
        &[],
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("GroupedMatMul") && error.contains("group_indices"),
        "{error}"
    );
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires CUDA; enable gpu-tests")]
fn grouped_matmul_native_cpu_and_cuda() {
    let ep = require_cuda();
    test_provider(None, 0);
    test_provider(Some(&ep), 0);
    test_provider(Some(&ep), 1);
}
