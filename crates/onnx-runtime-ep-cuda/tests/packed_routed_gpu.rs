mod common;

use DataType::{BFloat16, Float16, Float32, Float64, Int32, Int64, Uint8, Uint32};
use common::{
    Tensor, absent_input, decode_floats, float_input, input, require_cuda, try_run_cpu,
    try_run_cuda,
};
use onnx_runtime_ep_api::ExecutionProvider;
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ir::{Attribute, DataType, Node, NodeId, ValueId, static_shape};

fn test_provider(ep: Option<&CudaExecutionProvider>) {
    let run =
        |op, inputs: &[Tensor], outputs: &[(DataType, Vec<usize>)], attrs: &[(&str, Attribute)]| {
            match ep {
                Some(ep) => try_run_cuda(ep, op, "", 29, inputs, outputs, attrs),
                None => try_run_cpu(op, "", 29, inputs, outputs, attrs),
            }
            .map(|mut y| y.remove(0))
        };
    let cpu = CpuExecutionProvider::new();
    let provider: &dyn ExecutionProvider = ep.map_or(&cpu, |ep| ep as _);
    for (name, dtypes, shapes) in [
        ("Pack", vec![Uint8], vec![static_shape([8])]),
        (
            "Unpack",
            vec![Uint32, Int64],
            vec![static_shape([2]), static_shape([])],
        ),
        (
            "GroupedMatMul",
            vec![BFloat16, BFloat16, Int32],
            vec![
                static_shape([4, 3]),
                static_shape([2, 3, 2]),
                static_shape([4, 1]),
            ],
        ),
    ] {
        let mut node = Node::new(
            NodeId(0),
            name,
            (0..dtypes.len()).map(|i| Some(ValueId(i as u32))).collect(),
            vec![ValueId(4)],
        );
        if name != "GroupedMatMul" {
            node.attributes.insert("bits".into(), Attribute::Int(3));
        }
        assert!(
            provider
                .supports_op(&node, 29, &shapes, &dtypes, &[])
                .is_supported(),
            "{name}"
        );
        assert!(
            !provider
                .supports_op(&node, 28, &shapes, &dtypes, &[])
                .is_supported(),
            "{name}"
        );
        assert!(
            !provider
                .supports_op(&node, 29, &shapes, &vec![Int32; dtypes.len()], &[])
                .is_supported(),
            "{name}"
        );
    }

    // Expected vectors from the ONNX opset-29 backend cases, commit
    // 622adf94d4e720142950aa8178f7357d52e63db0 (Apache-2.0).
    for (attrs, x, packed) in [
        (
            vec![("bits", Attribute::Int(3))],
            input(Uint8, &[8], &[0u8, 1, 2, 3, 4, 5, 6, 7]),
            input(Uint8, &[3], &[0x88u8, 0xc6, 0xfa]),
        ),
        (
            vec![("bits", Attribute::Int(5))],
            input(Uint8, &[2, 3], &[1u8, 2, 3, 31, 0, 16]),
            input(Uint8, &[2, 2], &[0x41u8, 0x0c, 0x1f, 0x40]),
        ),
        (
            vec![
                ("radix", Attribute::Int(3)),
                ("group_size", Attribute::Int(5)),
            ],
            input(Uint8, &[2, 5], &[0u8, 1, 2, 0, 1, 2, 2, 2, 2, 2]),
            input(Uint8, &[2, 1], &[102u8, 242]),
        ),
        (
            vec![("bits", Attribute::Int(5))],
            input::<u8>(Uint8, &[2, 0, 9], &[]),
            input::<u8>(Uint8, &[2, 0, 6], &[]),
        ),
    ] {
        let count = input(Int64, &[], &[*x.shape.last().unwrap() as i64]);
        assert_eq!(
            run(
                "Pack",
                std::slice::from_ref(&x),
                &[(Uint8, packed.shape.clone())],
                &attrs
            )
            .unwrap(),
            packed.bytes
        );
        assert_eq!(
            run("Unpack", &[packed, count], &[(Uint8, x.shape)], &attrs).unwrap(),
            x.bytes
        );
    }
    let attrs = [("bits", Attribute::Int(5)), ("axis", Attribute::Int(0))];
    let x = input(Uint8, &[7, 2], &[31u8; 14]);
    let mut pack_attrs = attrs.to_vec();
    pack_attrs.push(("output_type", Attribute::Int(12)));
    let expected = input(Uint32, &[2, 2], &[u32::MAX, u32::MAX, 7, 7]);
    assert_eq!(
        run(
            "Pack",
            std::slice::from_ref(&x),
            &[(Uint32, vec![2, 2])],
            &pack_attrs
        )
        .unwrap(),
        expected.bytes
    );
    // Nonzero padding is deliberately ignored by Unpack.
    let ones = input(Uint32, &[2, 2], &[u32::MAX; 4]);
    let count = input(Int64, &[], &[7i64]);
    assert_eq!(
        run("Unpack", &[ones, count], &[(Uint8, vec![7, 2])], &attrs).unwrap(),
        x.bytes
    );

    // 3^39 fits into signed 64 bits, but a group is 62 bits wide and crosses
    // multiple words. Use the independent base-3 identity for an all-2 group.
    let attrs = [
        ("radix", Attribute::Int(3)),
        ("group_size", Attribute::Int(39)),
    ];
    let mut codes = vec![2u8; 39];
    codes.extend([1, 2]);
    let all_twos = 3u128.pow(39) - 1;
    let stream = (all_twos | (7u128 << 62)).to_le_bytes();
    let expect = &stream[..16];
    assert_eq!(
        run(
            "Pack",
            &[input(Uint8, &[41], &codes)],
            &[(Uint8, vec![16])],
            &attrs
        )
        .unwrap(),
        expect
    );
    assert_eq!(
        run(
            "Unpack",
            &[input(Uint8, &[16], expect), input(Int64, &[], &[41i64])],
            &[(Uint8, vec![41])],
            &attrs
        )
        .unwrap(),
        codes
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

    for (op, inputs, out, attrs, reason) in [
        (
            "Pack",
            vec![input(Uint8, &[1], &[8u8])],
            (Uint8, vec![1]),
            vec![("bits", Attribute::Int(3))],
            "code",
        ),
        (
            "Unpack",
            vec![input(Uint8, &[1], &[255u8]), input(Int64, &[], &[5i64])],
            (Uint8, vec![5]),
            vec![
                ("radix", Attribute::Int(3)),
                ("group_size", Attribute::Int(5)),
            ],
            "group",
        ),
        (
            "Unpack",
            vec![input(Uint8, &[1], &[0u8]), input(Int64, &[], &[-1i64])],
            (Uint8, vec![2]),
            vec![("bits", Attribute::Int(3))],
            "count",
        ),
        (
            "Unpack",
            vec![input(Uint8, &[2], &[0u8, 0]), input(Int64, &[], &[2i64])],
            (Uint8, vec![2]),
            vec![("bits", Attribute::Int(3))],
            "exactly 1 words",
        ),
        (
            "GroupedMatMul",
            vec![
                float_input(Float32, &[1, 1], &[1.]),
                float_input(Float32, &[1, 1, 1], &[2.]),
                input(Int32, &[1, 1], &[-1i32]),
            ],
            (Float32, vec![1, 1, 1]),
            vec![],
            "group_indices",
        ),
    ] {
        let error = run(op, &inputs, &[out], &attrs).unwrap_err().to_string();
        assert!(error.contains(op) && error.contains(reason), "{error}");
    }
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires CUDA; enable gpu-tests")]
fn packed_routed_native_cpu_and_cuda() {
    let ep = require_cuda();
    test_provider(None);
    test_provider(Some(&ep));
}
