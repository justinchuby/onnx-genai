use onnx_runtime_ir::{DataType, DeviceType};
use onnx_runtime_loader::proto::onnx;
use onnx_runtime_session::{DevicePreference, InferenceSession, Tensor};
use prost::Message;

fn value(name: &str, dtype: DataType, shape: &[usize]) -> onnx::ValueInfoProto {
    use onnx::tensor_shape_proto::{Dimension, dimension::Value};
    onnx::ValueInfoProto {
        name: name.into(),
        r#type: Some(onnx::TypeProto {
            value: Some(onnx::type_proto::Value::TensorType(
                onnx::type_proto::Tensor {
                    elem_type: dtype as i32,
                    shape: Some(onnx::TensorShapeProto {
                        dim: shape
                            .iter()
                            .map(|&dim| Dimension {
                                value: Some(Value::DimValue(dim as i64)),
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

fn bytes(dtype: DataType, values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| match dtype {
            DataType::Float32 => v.to_le_bytes().to_vec(),
            DataType::Float16 => half::f16::from_f32(v).to_le_bytes().to_vec(),
            DataType::BFloat16 => half::bf16::from_f32(v).to_le_bytes().to_vec(),
            DataType::Int16 => (v as i16).to_le_bytes().to_vec(),
            DataType::Uint16 => (v as u16).to_le_bytes().to_vec(),
            _ => unreachable!(),
        })
        .collect()
}

struct Case<'a> {
    opset: i64,
    code: DataType,
    shape: &'a [usize],
    input: &'a [f32],
    scale_shape: &'a [usize],
    scale: &'a [f32],
    scale_type: DataType,
    zero: Option<&'a [f32]>,
    axis: i64,
    block: i64,
    precision: Option<DataType>,
    output: Option<DataType>,
    q: &'a [f32],
    y: &'a [f32],
}

impl Case<'_> {
    fn model(&self, opset: i64) -> Vec<u8> {
        let attr = |name: &str, i| onnx::AttributeProto {
            name: name.into(),
            i,
            r#type: onnx::attribute_proto::AttributeType::Int as i32,
            ..Default::default()
        };
        let node = |op: &str, data: &str, output: &str, mut attrs: Vec<onnx::AttributeProto>| {
            attrs.push(attr("axis", self.axis));
            if self.block != 0 {
                attrs.push(attr("block_size", self.block));
            }
            onnx::NodeProto {
                op_type: op.into(),
                // Include an explicit omitted slot when not providing zero_point.
                input: vec![
                    data.into(),
                    "S".into(),
                    if self.zero.is_some() {
                        "Z".into()
                    } else {
                        String::new()
                    },
                ],
                output: vec![output.into()],
                attribute: attrs,
                ..Default::default()
            }
        };
        let mut q_attrs = Vec::new();
        if self.zero.is_none() {
            q_attrs.push(attr("output_dtype", self.code as i64));
        }
        if let Some(precision) = self.precision {
            q_attrs.push(attr("precision", precision as i64));
        }
        let mut inputs = vec![
            value("X", DataType::Float32, self.shape),
            value("S", self.scale_type, self.scale_shape),
        ];
        if self.zero.is_some() {
            inputs.push(value("Z", self.code, self.scale_shape));
        }
        onnx::ModelProto {
            ir_version: 10,
            opset_import: vec![onnx::OperatorSetIdProto {
                domain: String::new(),
                version: opset,
            }],
            graph: Some(onnx::GraphProto {
                name: "int16_linear_quantization".into(),
                input: inputs,
                output: vec![
                    value("Q", self.code, self.shape),
                    value("Y", self.output.unwrap_or(self.scale_type), self.shape),
                ],
                node: vec![
                    node("QuantizeLinear", "X", "Q", q_attrs),
                    node(
                        "DequantizeLinear",
                        "Q",
                        "Y",
                        self.output
                            .iter()
                            .map(|&dtype| attr("output_dtype", dtype as i64))
                            .collect(),
                    ),
                ],
                ..Default::default()
            }),
            ..Default::default()
        }
        .encode_to_vec()
    }

    fn check(&self, device: DevicePreference, expected_device: DeviceType) {
        let mut session = InferenceSession::builder()
            .model_bytes(&self.model(self.opset))
            .device(device)
            .build()
            .unwrap();
        assert_eq!(session.device_id().device_type, expected_device);
        assert!(
            session.execution_provider_fallback_report().is_none(),
            "{:?}",
            session.execution_provider_fallback_report()
        );
        if expected_device == DeviceType::Cuda {
            assert!(
                session.heterogeneous_placement_report().is_none(),
                "expected CUDA-only placement, got {:?}",
                session.heterogeneous_placement_report()
            );
        }
        let x = Tensor::from_f32(self.shape, self.input).unwrap();
        let scale = Tensor::from_raw(
            self.scale_type,
            self.scale_shape.to_vec(),
            &bytes(self.scale_type, self.scale),
        )
        .unwrap();
        let mut inputs = vec![("X", &x), ("S", &scale)];
        let zero = self.zero.map(|z| {
            Tensor::from_raw(self.code, self.scale_shape.to_vec(), &bytes(self.code, z)).unwrap()
        });
        if let Some(zero) = &zero {
            inputs.push(("Z", zero));
        }
        let actual = session.run(&inputs).unwrap();
        let expected = [
            (self.code, self.q),
            (self.output.unwrap_or(self.scale_type), self.y),
        ];
        assert_eq!(actual.len(), expected.len());
        for (tensor, (dtype, values)) in actual.iter().zip(expected) {
            assert_eq!(tensor.dtype, dtype);
            assert_eq!(
                tensor.as_bytes(),
                bytes(dtype, values),
                "opset {}, {dtype:?}",
                self.opset
            );
        }
    }
}

fn cases() -> [Case<'static>; 5] {
    [
        Case {
            opset: 21,
            code: DataType::Int16,
            shape: &[6],
            input: &[-1e30, -1.5, -0.5, 0.5, 1.5, 1e30],
            scale_shape: &[],
            scale: &[1.0],
            scale_type: DataType::Float32,
            zero: None,
            axis: 1,
            block: 0,
            precision: None,
            output: None,
            q: &[-32768., -2., 0., 0., 2., 32767.],
            y: &[-32768., -2., 0., 0., 2., 32767.],
        },
        Case {
            opset: 21,
            code: DataType::Uint16,
            shape: &[2, 3],
            input: &[-1.5, 0.5, 1e30, -3., 1., 3.],
            scale_shape: &[2],
            scale: &[1., 2.],
            scale_type: DataType::Float32,
            zero: Some(&[65534., 3.]),
            axis: -2,
            block: 0,
            precision: None,
            output: None,
            q: &[65532., 65534., 65535., 1., 3., 5.],
            y: &[-2., 0., 1., -4., 0., 4.],
        },
        Case {
            opset: 29,
            code: DataType::Int16,
            shape: &[2, 3, 2],
            input: &[4.; 12],
            scale_shape: &[2, 2, 2],
            scale: &[1., 2., 4., 8., 2., 1., 8., 4.],
            scale_type: DataType::Float16,
            zero: None,
            axis: 1,
            block: 2,
            precision: None,
            output: Some(DataType::BFloat16),
            q: &[4., 2., 4., 2., 1., 0., 2., 4., 2., 4., 0., 1.],
            y: &[4., 4., 4., 4., 4., 0., 4., 4., 4., 4., 0., 4.],
        },
        Case {
            opset: 23,
            code: DataType::Uint16,
            shape: &[2],
            input: &[0.5001, 2.5001],
            scale_shape: &[],
            scale: &[1.],
            scale_type: DataType::BFloat16,
            zero: None,
            axis: 1,
            block: 0,
            precision: None,
            output: None,
            q: &[0., 2.],
            y: &[0., 2.],
        },
        Case {
            opset: 23,
            code: DataType::Uint16,
            shape: &[2],
            input: &[0.5001, 2.5001],
            scale_shape: &[],
            scale: &[1.],
            scale_type: DataType::BFloat16,
            zero: None,
            axis: 1,
            block: 0,
            precision: Some(DataType::Float32),
            output: Some(DataType::Float16),
            q: &[1., 3.],
            y: &[1., 3.],
        },
    ]
}

#[test]
fn int16_quantization_cpu() {
    for case in cases() {
        case.check(DevicePreference::Cpu, DeviceType::Cpu);
    }
}

#[test]
fn int16_quantization_rejects_opset_20() {
    let bytes = cases()[1].model(20);
    let error = InferenceSession::builder()
        .model_bytes(&bytes)
        .device(DevicePreference::Cpu)
        .build()
        .err()
        .expect("opset 20 does not define 16-bit integer quantization");
    let message = format!("{error:#}");
    assert!(
        message.contains("opset 21") || message.contains("uint16") || message.contains("UINT16"),
        "{message}"
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device; asserts native CUDA without CPU fallback"]
fn int16_quantization_cuda_no_fallback() {
    for case in cases() {
        case.check(
            DevicePreference::Explicit {
                device_type: DeviceType::Cuda,
                index: 0,
            },
            DeviceType::Cuda,
        );
    }
}
