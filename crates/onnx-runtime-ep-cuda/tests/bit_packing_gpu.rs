mod common;

use DataType::{Int32, Int64, Uint8, Uint32};
use common::{Tensor, input, require_cuda, try_run_cpu, try_run_cuda_with_origin_offset};
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
    for (name, dtypes, shapes) in [
        ("Pack", vec![Uint8], vec![static_shape([8])]),
        (
            "Unpack",
            vec![Uint32, Int64],
            vec![static_shape([2]), static_shape([])],
        ),
    ] {
        let mut node = Node::new(
            NodeId(0),
            name,
            (0..dtypes.len()).map(|i| Some(ValueId(i as u32))).collect(),
            vec![ValueId(4)],
        );
        node.attributes.insert("bits".into(), Attribute::Int(3));
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
    ] {
        let error = run(op, &inputs, &[out], &attrs).unwrap_err().to_string();
        assert!(error.contains(op) && error.contains(reason), "{error}");
    }
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires CUDA; enable gpu-tests")]
fn bit_packing_native_cpu_and_cuda() {
    let ep = require_cuda();
    test_provider(None, 0);
    test_provider(Some(&ep), 0);
    test_provider(Some(&ep), 1);
}
