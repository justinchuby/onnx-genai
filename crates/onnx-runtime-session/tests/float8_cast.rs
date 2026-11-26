use onnx_runtime_ir::{DataType, DeviceType};
use onnx_runtime_loader::proto::onnx;
use onnx_runtime_session::{DevicePreference, InferenceSession, Tensor};
use prost::Message;

fn value(name: &str, dtype: DataType, count: usize) -> onnx::ValueInfoProto {
    use onnx::tensor_shape_proto::{Dimension, dimension::Value};
    onnx::ValueInfoProto {
        name: name.into(),
        r#type: Some(onnx::TypeProto {
            value: Some(onnx::type_proto::Value::TensorType(
                onnx::type_proto::Tensor {
                    elem_type: dtype as i32,
                    shape: Some(onnx::TensorShapeProto {
                        dim: vec![Dimension {
                            value: Some(Value::DimValue(count as i64)),
                            ..Default::default()
                        }],
                    }),
                },
            )),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn int(name: &str, i: i64) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.into(),
        i,
        r#type: onnx::attribute_proto::AttributeType::Int as i32,
        ..Default::default()
    }
}

fn mode(mode: &str) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: "round_mode".into(),
        s: mode.as_bytes().to_vec(),
        r#type: onnx::attribute_proto::AttributeType::String as i32,
        ..Default::default()
    }
}

fn check(
    device: &DevicePreference,
    opset: i64,
    like: bool,
    input: &Tensor,
    dst: DataType,
    mut attributes: Vec<onnx::AttributeProto>,
    expected: &[u8],
) {
    let op = if like { "CastLike" } else { "Cast" };
    let x = value("X", input.dtype, input.shape[0]);
    let y = value("Y", dst, input.shape[0]);
    let target = Tensor::zeros(dst, vec![1]).unwrap();
    let mut node_inputs = vec!["X".into()];
    let mut graph_inputs = vec![x];
    let mut feed = vec![("X", input)];
    if like {
        node_inputs.push("T".into());
        graph_inputs.push(value("T", dst, 1));
        feed.push(("T", &target));
    } else {
        attributes.push(int("to", i64::from(dst as i32)));
    }
    let model = onnx::ModelProto {
        ir_version: 14,
        opset_import: vec![onnx::OperatorSetIdProto {
            domain: String::new(),
            version: opset,
        }],
        graph: Some(onnx::GraphProto {
            name: format!("{op}-fp8"),
            input: graph_inputs,
            output: vec![y],
            node: vec![onnx::NodeProto {
                op_type: op.into(),
                input: node_inputs,
                output: vec!["Y".into()],
                attribute: attributes,
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let context = format!("{op}-{opset} {:?} -> {dst:?}", input.dtype);
    let mut session = InferenceSession::builder()
        .model_bytes(&model.encode_to_vec())
        .device(device.clone())
        .build()
        .unwrap_or_else(|e| panic!("{context}: {e:#}"));
    let expected_device = if matches!(device, DevicePreference::Cpu) {
        DeviceType::Cpu
    } else {
        DeviceType::Cuda
    };
    assert_eq!(
        session.device_id().device_type,
        expected_device,
        "{context}"
    );
    assert!(
        session.execution_provider_fallback_report().is_none(),
        "{context}"
    );
    if expected_device == DeviceType::Cuda {
        assert!(
            session.heterogeneous_placement_report().is_none(),
            "{context}: unexpected heterogeneous placement: {:?}",
            session.heterogeneous_placement_report()
        );
    }
    let outputs = session
        .run(&feed)
        .unwrap_or_else(|e| panic!("{context}: {e:#}"));
    assert_eq!(outputs[0].as_bytes(), expected, "{context}");
}

fn check_float8_casts(device: DevicePreference) {
    use DataType::*;
    // Values and exact encoding from the ONNX Cast E4M3FN backend fixture.
    let official = Tensor::from_f32(
        &[15],
        &[
            0.47892547,
            0.48033667,
            0.49968487,
            0.81910545,
            0.47031248,
            0.7229038,
            1e6,
            1e-7,
            f32::NAN,
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -1e-7,
            1e-7,
            -1e6,
        ],
    )
    .unwrap();
    check(
        &device,
        19,
        false,
        &official,
        Float8E4M3FN,
        vec![],
        &[
            47, 47, 48, 53, 47, 52, 126, 0, 127, 126, 126, 254, 128, 0, 254,
        ],
    );
    let signed =
        Tensor::from_f32(&[6], &[0., -0., 1., -1., f32::INFINITY, f32::NEG_INFINITY]).unwrap();
    for (dtype, opset, saturated, unsaturated) in [
        (
            Float8E4M3FN,
            19,
            [0, 128, 56, 184, 126, 254],
            [0, 128, 56, 184, 127, 255],
        ),
        (
            Float8E4M3FNUZ,
            21,
            [0, 0, 64, 192, 127, 255],
            [0, 0, 64, 192, 128, 128],
        ),
        (
            Float8E5M2,
            23,
            [0, 128, 60, 188, 123, 251],
            [0, 128, 60, 188, 124, 252],
        ),
        (
            Float8E5M2FNUZ,
            25,
            [0, 0, 64, 192, 127, 255],
            [0, 0, 64, 192, 128, 128],
        ),
    ] {
        check(&device, opset, false, &signed, dtype, vec![], &saturated);
        check(
            &device,
            opset,
            true,
            &signed,
            dtype,
            vec![int("saturate", 0)],
            &unsaturated,
        );
        let packed = Tensor::from_raw(dtype, vec![4], &saturated[..4]).unwrap();
        let half_zero = if saturated[1] == 128 {
            [0, 128]
        } else {
            [0, 0]
        };
        let expected = [0, 0, half_zero[0], half_zero[1], 0, 60, 0, 188];
        check(&device, 29, false, &packed, Float16, vec![], &expected);
    }
    let codes: Vec<_> = (0..=u8::MAX).collect();
    for dtype in [
        Float8E4M3FN,
        Float8E4M3FNUZ,
        Float8E5M2,
        Float8E5M2FNUZ,
        Float8E8M0,
    ] {
        let packed = Tensor::from_raw(dtype, vec![codes.len()], &codes).unwrap();
        for like in [false, true] {
            check(&device, 29, like, &packed, dtype, vec![], &codes);
        }
    }

    // ONNX backend test_cast_e8m0_FLOAT_to_FLOAT8E8M0.
    let x = Tensor::from_f32(&[8], &[0., 0.124, 0.25, 0.5, 1.1, 2., 4., 8.]).unwrap();
    check(
        &device,
        24,
        false,
        &x,
        Float8E8M0,
        vec![],
        &[0, 124, 125, 126, 128, 128, 129, 130],
    );

    let f64s = |xs: &[f64]| {
        let bytes: Vec<_> = xs.iter().flat_map(|x| x.to_le_bytes()).collect();
        Tensor::from_raw(Float64, vec![xs.len()], &bytes).unwrap()
    };
    let e8m0 = f64s(&[1., 1f64.next_up(), 1.5f64.next_down(), 1.5, 2.]);
    for (round, expected) in [
        ("up", [127, 128, 128, 128, 128]),
        ("down", [127, 127, 127, 127, 128]),
        ("nearest", [127, 127, 127, 128, 128]),
    ] {
        check(
            &device,
            28,
            false,
            &e8m0,
            Float8E8M0,
            vec![mode(round)],
            &expected,
        );
        check(
            &device,
            24,
            true,
            &e8m0,
            Float8E8M0,
            vec![mode(round)],
            &expected,
        );
    }
    let rn = f64s(&[
        1.0625f64.next_down(),
        1.0625,
        1.0625f64.next_up(),
        1.1875,
        464.0,
        464.0f64.next_up(),
    ]);
    check(
        &device,
        29,
        false,
        &rn,
        Float8E4M3FN,
        vec![int("saturate", 0)],
        &[56, 56, 57, 58, 126, 127],
    );
    let overflow = f64s(&[2f64.powi(127), 2f64.powi(127).next_up(), f64::NAN]);
    check(
        &device,
        29,
        false,
        &overflow,
        Float8E8M0,
        vec![int("saturate", 0)],
        &[254, 255, 255],
    );
    check(
        &device,
        29,
        true,
        &overflow,
        Float8E8M0,
        vec![],
        &[254, 254, 255],
    );
    let smallest = 2f64.powi(-127);
    let largest = 2f64.powi(127);
    let outside = f64s(&[
        0.,
        smallest.next_down(),
        smallest,
        largest,
        largest.next_up(),
        f64::INFINITY,
    ]);
    check(
        &device,
        29,
        false,
        &outside,
        Float8E8M0,
        vec![mode("nearest"), int("saturate", 0)],
        &[255, 255, 0, 254, 255, 255],
    );
    check(
        &device,
        29,
        true,
        &outside,
        Float8E8M0,
        vec![],
        &[0, 0, 0, 254, 254, 254],
    );

    // Code zero is a positive scale (2^-127), not floating-point zero.
    let codes = Tensor::from_raw(Float8E8M0, vec![3], &[0, 127, 254]).unwrap();
    let wide: Vec<_> = [2f32.powi(-127), 1.0, 2f32.powi(127)]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    check(&device, 29, false, &codes, Float32, vec![], &wide);
    check(
        &device,
        29,
        false,
        &codes,
        Float8E5M2,
        vec![],
        &[0, 60, 123],
    );

    // FP8 factories must still reach both packed-output CUDA entry points and
    // the packed-source load path after opset selection.
    let ints = Tensor::from_raw(Uint8, vec![5], &[0, 1, 7, 8, 255]).unwrap();
    check(
        &device,
        21,
        false,
        &ints,
        Uint4,
        vec![],
        &[0x10, 0x87, 0x0f],
    );
    let half = Tensor::from_raw(Float16, vec![3], &[0, 0x3c, 0, 0xc0, 0, 0x42]).unwrap();
    check(&device, 25, true, &half, Int2, vec![], &[0x39]);
    let packed = Tensor::from_raw(Int2, vec![3], &[0x39]).unwrap();
    check(&device, 28, false, &packed, Int8, vec![], &[1, 254, 255]);
}

#[test]
fn float8_cast_native_cpu() {
    check_float8_casts(DevicePreference::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn float8_cast_native_cuda() {
    check_float8_casts(DevicePreference::Explicit {
        device_type: DeviceType::Cuda,
        index: 0,
    });
}
