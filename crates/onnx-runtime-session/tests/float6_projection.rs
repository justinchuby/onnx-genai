use onnx_runtime_loader::{KeyedWeightFile, proto::onnx};
use onnx_runtime_session::{DevicePreference, InferenceSession, Tensor};
use prost::Message;

fn value(name: &str, dims: [i64; 2]) -> onnx::ValueInfoProto {
    use onnx::tensor_shape_proto::{Dimension, dimension::Value};
    onnx::ValueInfoProto {
        name: name.into(),
        r#type: Some(onnx::TypeProto {
            value: Some(onnx::type_proto::Value::TensorType(
                onnx::type_proto::Tensor {
                    elem_type: 1,
                    shape: Some(onnx::TensorShapeProto {
                        dim: dims
                            .into_iter()
                            .map(|dim| Dimension {
                                value: Some(Value::DimValue(dim)),
                                ..Default::default()
                            })
                            .collect(),
                    }),
                },
            )),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn check_projection<const K: usize, const N: usize>(device: DevicePreference, cuda: bool) {
    for (tag, kind, expected) in [
        (
            27,
            "F6_E2M3",
            [0.125_f32, -1.5, 7.5, -0.0, -0.25, 2.0, -7.5, 0.875],
        ),
        (
            28,
            "F6_E3M2",
            [0.0625_f32, -1.5, 28.0, -0.0, -0.125, 2.0, -28.0, 0.1875],
        ),
    ] {
        let packed = if tag == 27 {
            [0x01, 0xfb, 0x81, 0x22, 0xf4, 0x1f] // [1,44,31,32,34,16,63,7]
        } else {
            [0x81, 0xfb, 0x81, 0x22, 0xf4, 0x0f] // [1,46,31,32,34,16,63,3]
        };
        let len = K * N * 3 / 4;
        let mut header =
            format!(r#"{{"W":{{"dtype":"{kind}","shape":[{K},{N}],"data_offsets":[0,{len}]}}}}"#);
        while !header.len().is_multiple_of(8) {
            header.push(' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        // Vary the pattern across rows, so accidental row aliasing changes the
        // projection rather than reproducing the same tiled weight.
        for row in 0..K {
            let shift = row % 2 * 3;
            bytes.extend(packed.iter().cycle().skip(shift).take(N * 3 / 4));
        }
        let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
        let path = dir.join(format!(
            "float6-projection-{kind}-{cuda}-{K}x{N}-{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, bytes).unwrap();
        let model = onnx::ModelProto {
            ir_version: 14,
            opset_import: vec![onnx::OperatorSetIdProto {
                domain: String::new(),
                version: 28,
            }],
            graph: Some(onnx::GraphProto {
                name: "FLOAT6 projection".into(),
                input: vec![value("X", [1, K as i64])],
                output: vec![value("Y", [1, N as i64]), value("F", [K as i64, N as i64])],
                initializer: vec![onnx::TensorProto {
                    name: "W".into(),
                    dims: vec![K as i64, N as i64],
                    data_type: tag,
                    data_location: onnx::tensor_proto::DataLocation::External as i32,
                    ..Default::default()
                }],
                node: vec![
                    onnx::NodeProto {
                        op_type: "Cast".into(),
                        input: vec!["W".into()],
                        output: vec!["F".into()],
                        attribute: vec![onnx::AttributeProto {
                            name: "to".into(),
                            i: 1,
                            r#type: onnx::attribute_proto::AttributeType::Int as i32,
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                    onnx::NodeProto {
                        op_type: "MatMul".into(),
                        input: vec!["X".into(), "F".into()],
                        output: vec!["Y".into()],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut session = InferenceSession::builder()
            .model_bytes(&model.encode_to_vec())
            .device(device.clone())
            .keyed_weight_file(KeyedWeightFile::safetensors(&path))
            .build()
            .unwrap();
        assert_eq!(
            session.device_id().device_type == onnx_runtime_ir::DeviceType::Cuda,
            cuda
        );
        assert!(session.execution_provider_fallback_report().is_none());
        for (even, odd) in [(0.5, -0.25), (-0.25, 2.0), (1.0, 0.0)] {
            let x = Tensor::from_f32(&[1, K], &[even, odd].repeat(K / 2)).unwrap();
            let outputs = session.run(&[("X", &x)]).unwrap();
            if even == 0.5 {
                let widened = outputs[1].to_vec_f32();
                assert_eq!(widened.len(), K * N);
                for (row, values) in widened.chunks_exact(N).take(2).enumerate() {
                    for (col, value) in values.iter().enumerate() {
                        let reference = expected[(col + row * 4) % 8];
                        assert_eq!(value.to_bits(), reference.to_bits(), "{kind} [{row},{col}]");
                    }
                }
            }
            let out = outputs[0].to_vec_f32();
            assert_eq!(out.len(), N);
            for (i, &v) in out.iter().enumerate() {
                let reference =
                    (even * expected[i % 8] + odd * expected[(i + 4) % 8]) * (K / 2) as f32;
                assert_eq!(v, reference, "{kind} col {i}");
            }
        }
        let x_values: Vec<_> = (0..K)
            .map(|r| ((r * 173 % 257) as f32 - 128.0) / 151.0)
            .collect();
        let x = Tensor::from_f32(&[1, K], &x_values).unwrap();
        let out = session.run(&[("X", &x)]).unwrap()[0].to_vec_f32();
        assert_eq!(out.len(), N);
        for (col, &actual) in out.iter().enumerate() {
            let terms = x_values
                .iter()
                .enumerate()
                .map(|(row, &x)| f64::from(x) * f64::from(expected[(col + row % 2 * 4) % 8]));
            let reference: f64 = terms.clone().sum();
            let tolerance = 2e-6 * terms.map(f64::abs).sum::<f64>();
            assert!(
                (f64::from(actual) - reference).abs() <= tolerance,
                "{kind} col {col}: got {actual}, expected {reference} ± {tolerance}"
            );
        }
        drop(session);
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn float6_safetensors_projection_cpu() {
    check_projection::<1024, 2048>(DevicePreference::Cpu, false);
    check_projection::<14, 132>(DevicePreference::Cpu, false);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn float6_safetensors_projection_cuda() {
    let device = DevicePreference::Gpu { index: Some(0) };
    check_projection::<1024, 2048>(device.clone(), true);
    check_projection::<14, 132>(device, true);
}
