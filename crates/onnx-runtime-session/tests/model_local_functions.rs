//! End-to-end proof that the runtime executes ONNX **model-local functions**.
//!
//! A `ModelProto` declares a `FunctionProto` and calls it. The loader inlines
//! the call into primitive ops at load time (see
//! `onnx_runtime_loader::function_inline`), so the executor — which only has
//! kernels for primitive ops — runs it and produces the same numbers as the
//! hand-computed reference.

use prost::Message;

use onnx_runtime_loader::proto::onnx;
use onnx_runtime_session::{InferenceSession, Tensor};

fn tensor_type(dims: &[i64]) -> onnx::TypeProto {
    use onnx::tensor_shape_proto::{dimension::Value as DV, Dimension};
    onnx::TypeProto {
        value: Some(onnx::type_proto::Value::TensorType(onnx::type_proto::Tensor {
            elem_type: 1, // FLOAT
            shape: Some(onnx::TensorShapeProto {
                dim: dims
                    .iter()
                    .map(|&n| Dimension {
                        value: Some(DV::DimValue(n)),
                        ..Default::default()
                    })
                    .collect(),
            }),
        })),
        ..Default::default()
    }
}

fn value_info(name: &str, dims: &[i64]) -> onnx::ValueInfoProto {
    onnx::ValueInfoProto {
        name: name.to_string(),
        r#type: Some(tensor_type(dims)),
        ..Default::default()
    }
}

fn f32_initializer(name: &str, dims: &[i64], data: &[f32]) -> onnx::TensorProto {
    onnx::TensorProto {
        name: name.to_string(),
        data_type: 1, // FLOAT
        dims: dims.to_vec(),
        raw_data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ..Default::default()
    }
}

fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> onnx::NodeProto {
    onnx::NodeProto {
        op_type: op.to_string(),
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: outputs.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    }
}

fn call(name: &str, domain: &str, inputs: &[&str], outputs: &[&str]) -> onnx::NodeProto {
    let mut n = node(name, inputs, outputs);
    n.domain = domain.to_string();
    n
}

#[test]
fn model_local_function_runs_end_to_end() {
    // Function: MyLinear(X, W, B) = Add(MatMul(X, W), B)
    let func = onnx::FunctionProto {
        name: "MyLinear".to_string(),
        domain: "custom.domain".to_string(),
        input: vec!["X".into(), "W".into(), "B".into()],
        output: vec!["Y".into()],
        node: vec![
            node("MatMul", &["X", "W"], &["xw"]),
            node("Add", &["xw", "B"], &["Y"]),
        ],
        opset_import: vec![onnx::OperatorSetIdProto {
            domain: String::new(),
            version: 17,
        }],
        ..Default::default()
    };

    // X: [2,3], W: [3,4], B: [4].
    let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w = [
        1.0f32, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0,
    ];
    let b = [10.0f32, 20.0, 30.0, 40.0];

    let graph = onnx::GraphProto {
        input: vec![value_info("input", &[2, 3])],
        output: vec![value_info("out", &[2, 4])],
        initializer: vec![
            f32_initializer("weight", &[3, 4], &w),
            f32_initializer("bias", &[4], &b),
        ],
        node: vec![call(
            "MyLinear",
            "custom.domain",
            &["input", "weight", "bias"],
            &["out"],
        )],
        ..Default::default()
    };

    let model = onnx::ModelProto {
        ir_version: 8,
        opset_import: vec![onnx::OperatorSetIdProto {
            domain: String::new(),
            version: 17,
        }],
        graph: Some(graph),
        functions: vec![func],
        ..Default::default()
    };
    let bytes = model.encode_to_vec();

    // The executor only has primitive kernels; if inlining failed it would
    // reject the unknown `custom.domain::MyLinear` op at load time.
    let mut session = InferenceSession::load_bytes(&bytes).expect("load + inline function");

    let x_tensor = Tensor::from_f32(&[2, 3], &x).unwrap();
    let outputs = session.run(&[("input", &x_tensor)]).expect("run");
    assert_eq!(outputs.len(), 1);

    // Reference: MatMul(X, W) + B, with W = [I3 | 0] so MatMul copies the first
    // 3 columns of each row and zero-fills the 4th, then B is added.
    let want = [
        1.0 + 10.0,
        2.0 + 20.0,
        3.0 + 30.0,
        0.0 + 40.0,
        4.0 + 10.0,
        5.0 + 20.0,
        6.0 + 30.0,
        0.0 + 40.0,
    ];
    let got = outputs[0].to_vec_f32();
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < 1e-4, "element {i}: got {g}, want {w}");
    }
    assert_eq!(outputs[0].shape, vec![2, 4]);
}

#[test]
fn passthrough_custom_function_without_default_opset_loads_and_runs() {
    // BUG 3 regression: a model whose ONLY node calls a custom-domain
    // pass-through function F(X) -> X, declaring ONLY the custom domain's opset
    // import and NO default (`""`) import — a valid ONNX model. Inlining
    // synthesizes a default-domain `Identity`; the loader must gain a
    // default-domain opset import so the previously-valid model still loads.
    let func = onnx::FunctionProto {
        name: "Passthrough".to_string(),
        domain: "custom.domain".to_string(),
        input: vec!["X".into()],
        output: vec!["X".into()],
        node: vec![], // pure pass-through: output aliases input
        opset_import: vec![onnx::OperatorSetIdProto {
            domain: "custom.domain".to_string(),
            version: 1,
        }],
        ..Default::default()
    };

    let graph = onnx::GraphProto {
        input: vec![value_info("input", &[2, 3])],
        output: vec![value_info("out", &[2, 3])],
        node: vec![call("Passthrough", "custom.domain", &["input"], &["out"])],
        ..Default::default()
    };

    // Deliberately declare ONLY the custom domain — NO default `""` import.
    let model = onnx::ModelProto {
        ir_version: 8,
        opset_import: vec![onnx::OperatorSetIdProto {
            domain: "custom.domain".to_string(),
            version: 1,
        }],
        graph: Some(graph),
        functions: vec![func],
        ..Default::default()
    };
    let bytes = model.encode_to_vec();

    // Before the fix the synthesized `Identity` (default domain) had no opset
    // import, so the loader rejected this previously-valid model.
    let mut session =
        InferenceSession::load_bytes(&bytes).expect("load + inline pass-through function");

    let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x_tensor = Tensor::from_f32(&[2, 3], &x).unwrap();
    let outputs = session.run(&[("input", &x_tensor)]).expect("run");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape, vec![2, 3]);
    // Pass-through: output equals input.
    let got = outputs[0].to_vec_f32();
    assert_eq!(got, x.to_vec());
}
