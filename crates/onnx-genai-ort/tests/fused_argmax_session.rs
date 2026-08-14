//! End-to-end checks for the fused last-axis argmax custom op.
//!
//! These build a real ONNX model whose only node is the custom op, create a
//! CUDA session through this crate (which is what registers the op's domain),
//! and compare against a host reference over the same data. They are the gate
//! that the op is wired correctly: the schema is accepted, ORT places the node
//! on CUDA, the kernel launches on ORT's stream, and the answer matches.

#![cfg(feature = "cuda")]

use onnx_genai_ort::env::Environment;
use onnx_genai_ort::fused_argmax::{DOMAIN, OP_NAME};
use onnx_genai_ort::session::{Session, SessionOptions, ep_selection};
use onnx_genai_ort::value::Value;
use onnx_runtime_loader::proto::onnx::{
    GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorShapeProto, TypeProto,
    ValueInfoProto, tensor_proto, tensor_shape_proto, type_proto,
};
use prost::Message;

fn tensor_type(elem: tensor_proto::DataType, dims: &[Option<i64>]) -> TypeProto {
    TypeProto {
        value: Some(type_proto::Value::TensorType(type_proto::Tensor {
            elem_type: elem as i32,
            shape: Some(TensorShapeProto {
                dim: dims
                    .iter()
                    .map(|dim| tensor_shape_proto::Dimension {
                        value: Some(match dim {
                            Some(size) => tensor_shape_proto::dimension::Value::DimValue(*size),
                            None => tensor_shape_proto::dimension::Value::DimParam("batch".into()),
                        }),
                        ..Default::default()
                    })
                    .collect(),
            }),
        })),
        ..Default::default()
    }
}

/// A model whose whole body is one fused argmax over `[batch, vocab]`.
fn fused_model(vocab: i64) -> Vec<u8> {
    ModelProto {
        ir_version: 8,
        producer_name: "fused-argmax-test".into(),
        opset_import: vec![
            OperatorSetIdProto {
                domain: String::new(),
                version: 17,
            },
            OperatorSetIdProto {
                domain: DOMAIN.to_string(),
                version: 1,
            },
        ],
        graph: Some(GraphProto {
            name: "fused_argmax".into(),
            node: vec![NodeProto {
                op_type: OP_NAME.to_string(),
                domain: DOMAIN.to_string(),
                name: "argmax".into(),
                input: vec!["logits".into()],
                output: vec!["token".into()],
                ..Default::default()
            }],
            input: vec![ValueInfoProto {
                name: "logits".into(),
                r#type: Some(tensor_type(
                    tensor_proto::DataType::Float,
                    &[None, Some(vocab)],
                )),
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: "token".into(),
                r#type: Some(tensor_type(tensor_proto::DataType::Int64, &[None])),
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    }
    .encode_to_vec()
}

/// Host oracle: lowest index of the maximum, ignoring NaN, 0 when nothing wins.
fn reference(row: &[f32]) -> i64 {
    let mut best = f32::NEG_INFINITY;
    let mut index = i64::MAX;
    for (position, &value) in row.iter().enumerate() {
        let position = position as i64;
        if !value.is_nan() && (value > best || (value == best && position < index)) {
            best = value;
            index = position;
        }
    }
    if index == i64::MAX { 0 } else { index }
}

/// Build a CUDA session over `model`, or `None` when this machine has no CUDA
/// execution provider. The model is written under Cargo's per-target temp
/// directory so the session is created from a path, which every build of this
/// crate supports.
fn cuda_session(model: &[u8], name: &str) -> Option<(Environment, Session)> {
    let directory = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("fused-argmax");
    std::fs::create_dir_all(&directory).expect("test model directory");
    let path = directory.join(format!("{name}.onnx"));
    std::fs::write(&path, model).expect("write test model");
    let environment = Environment::new("fused-argmax-test").ok()?;
    let options = SessionOptions::with_execution_provider(ep_selection("cuda"));
    let session = Session::new(&environment, &path, options).ok()?;
    Some((environment, session))
}

fn run(session: &Session, rows: usize, vocab: usize, data: &[f32]) -> Vec<i64> {
    let logits = Value::from_slice_f32(data, &[rows as i64, vocab as i64]).expect("input value");
    let outputs = session.run(&[("logits", &logits)]).expect("session run");
    outputs[0].to_vec_i64().expect("token output")
}

#[test]
fn fused_argmax_matches_the_host_reference_on_cuda() {
    let vocab = 202_048usize;
    let model = fused_model(vocab as i64);
    let Some((_environment, session)) = cuda_session(&model, "fused-argmax") else {
        eprintln!("skipping: no CUDA session available");
        return;
    };

    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 16_777_216.0 - 0.5
    };
    let mut row: Vec<f32> = (0..vocab).map(|_| next()).collect();
    row[180_559] = 12.5;
    assert_eq!(run(&session, 1, vocab, &row), vec![180_559]);
    assert_eq!(run(&session, 1, vocab, &row), vec![reference(&row)]);
}

#[test]
fn fused_argmax_handles_ties_nan_and_infinities() {
    // A width above the tiling threshold but small enough to enumerate.
    let vocab = 8_192usize;
    let model = fused_model(vocab as i64);
    let Some((_environment, session)) = cuda_session(&model, "fused-argmax-edges") else {
        eprintln!("skipping: no CUDA session available");
        return;
    };

    let cases: Vec<(&str, Vec<f32>)> = vec![
        ("all equal", vec![2.0; vocab]),
        ("tie across parts", {
            let mut row = vec![0.0; vocab];
            row[7] = 4.0;
            row[6_000] = 4.0;
            row
        }),
        ("leading nan", {
            let mut row = vec![f32::NAN; vocab];
            row[4_321] = 1.0;
            row
        }),
        ("all nan", vec![f32::NAN; vocab]),
        ("all negative infinity", vec![f32::NEG_INFINITY; vocab]),
        ("positive infinity last", {
            let mut row = vec![1.0; vocab];
            row[vocab - 1] = f32::INFINITY;
            row
        }),
        ("maximum at the very end", {
            let mut row = vec![-3.0; vocab];
            row[vocab - 1] = -2.0;
            row
        }),
    ];
    for (name, row) in cases {
        assert_eq!(
            run(&session, 1, vocab, &row),
            vec![reference(&row)],
            "case {name:?}"
        );
    }
}

#[test]
fn fused_argmax_is_batch_aware() {
    let vocab = 6_144usize;
    let model = fused_model(vocab as i64);
    let Some((_environment, session)) = cuda_session(&model, "fused-argmax-batch") else {
        eprintln!("skipping: no CUDA session available");
        return;
    };

    // Heterogeneous rows, including a NaN row and a constant row, with the
    // winner in a different part of every row.
    for rows in [1usize, 2, 5, 17] {
        let mut data = Vec::with_capacity(rows * vocab);
        let mut expected = Vec::with_capacity(rows);
        for row in 0..rows {
            let mut values = vec![0.0f32; vocab];
            match row % 4 {
                0 => values[row * 37 % vocab] = 5.0,
                1 => values.iter_mut().for_each(|value| *value = 1.0),
                2 => {
                    values.iter_mut().for_each(|value| *value = f32::NAN);
                    values[vocab - row - 1] = 0.5;
                }
                _ => values[vocab - 1] = 9.0,
            }
            expected.push(reference(&values));
            data.extend_from_slice(&values);
        }
        assert_eq!(run(&session, rows, vocab, &data), expected, "rows={rows}");
    }
}
