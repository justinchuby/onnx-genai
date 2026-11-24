use onnx_runtime_ir::{Attribute, DataType, DeviceId, Dim, Graph, Node, NodeId, static_shape};
use onnx_runtime_loader::{Model, encode_model};
use onnx_runtime_session::{DevicePreference, InferenceSession, SessionError, Tensor};

fn model(opset: u64, dtype: DataType, axis: Option<i64>, mask_len: usize) -> Vec<u8> {
    model_with_dtypes(opset, dtype, dtype, axis, mask_len)
}

fn model_with_dtypes(
    opset: u64,
    input_dtype: DataType,
    output_dtype: DataType,
    axis: Option<i64>,
    mask_len: usize,
) -> Vec<u8> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), opset);
    let x = graph.create_named_value("X", input_dtype, static_shape([2, 3]));
    let mask = graph.create_named_value("mask", DataType::Bool, static_shape([mask_len]));
    graph.add_input(x);
    graph.add_input(mask);
    let selected = Dim::Symbolic(graph.intern_symbol("selected"));
    let shape = if axis.is_some() {
        vec![Dim::Static(2), selected]
    } else {
        vec![selected]
    };
    let y = graph.create_named_value("Y", output_dtype, shape);
    let mut node = Node::new(NodeId(0), "Compress", vec![Some(x), Some(mask)], vec![y]);
    node.name = "select_values".into();
    if let Some(axis) = axis {
        node.attributes.insert("axis".into(), Attribute::Int(axis));
    }
    graph.insert_node(node);
    graph.add_output(y);
    encode_model(&Model::new(&graph)).expect("encode Compress model")
}

fn load(bytes: &[u8], device: DeviceId) -> Result<InferenceSession, SessionError> {
    InferenceSession::builder()
        .model_bytes(bytes)
        .device(DevicePreference::Explicit {
            device_type: device.device_type,
            index: device.index,
        })
        .build()
}

fn bf16_bytes(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn check_bf16_outputs(device: DeviceId) {
    let dtype = DataType::BFloat16;
    // Include a NaN payload, signed zero, and a subnormal so that conversion
    // instead of byte-preserving selection can change the reference output.
    let x = Tensor::from_raw(
        dtype,
        vec![2, 3],
        &bf16_bytes(&[0x3f80, 0x7fc5, 0x4000, 0x4040, 0x8000, 0x0001]),
    )
    .unwrap();
    for (axis, mask, shape, reference) in [
        (Some(-1), &[0, 1][..], &[2, 1][..], &[0x7fc5, 0x8000][..]),
        (
            None,
            &[0, 1, 0, 0, 0, 1][..],
            &[2][..],
            &[0x7fc5, 0x0001][..],
        ),
    ] {
        let model = model(28, dtype, axis, mask.len());
        let mut session = load(&model, device).expect("admit Compress-28 BF16");
        assert_eq!(session.device_id(), device, "axis={axis:?}");
        assert!(session.execution_provider_fallback_report().is_none());
        assert!(session.heterogeneous_placement_report().is_none());
        let mask = Tensor::from_raw(DataType::Bool, vec![mask.len()], mask).unwrap();
        let outputs = session.run(&[("X", &x), ("mask", &mask)]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].dtype, dtype, "axis={axis:?}");
        assert_eq!(outputs[0].shape, shape, "axis={axis:?}");
        assert_eq!(
            outputs[0].as_bytes(),
            bf16_bytes(reference),
            "axis={axis:?}"
        );
    }
}

#[test]
fn compress_28_bf16_cpu() {
    check_bf16_outputs(DeviceId::cpu());
}

#[test]
fn compress_pre_28_rejects_bf16_but_runs_legacy_dtype() {
    let device = DeviceId::cpu();
    for (opset, schema, input_dtype, port) in [
        (9, 9, DataType::BFloat16, "input"),
        (11, 11, DataType::BFloat16, "input"),
        (27, 11, DataType::BFloat16, "input"),
        (27, 11, DataType::Uint8, "output"),
    ] {
        let invalid = model_with_dtypes(opset, input_dtype, DataType::BFloat16, Some(-1), 2);
        let err = load(&invalid, device)
            .err()
            .expect("pre-28 Compress must not admit BF16");
        let message = err.to_string();
        for detail in [
            "select_values",
            &format!("{port} dtype BFloat16"),
            &format!("Compress-{schema}"),
            &format!("opset {opset}"),
            "28+",
        ] {
            assert!(message.contains(detail), "{message}");
        }
    }

    let legacy = model(27, DataType::Uint8, Some(-1), 2);
    let mut session = load(&legacy, device).expect("opset 27 still admits Uint8");
    let x = Tensor::from_raw(DataType::Uint8, vec![2, 3], &[1, 2, 3, 4, 5, 6]).unwrap();
    let mask = Tensor::from_raw(DataType::Bool, vec![2], &[0, 1]).unwrap();
    let outputs = session.run(&[("X", &x), ("mask", &mask)]).unwrap();
    assert_eq!(outputs[0].dtype, DataType::Uint8);
    assert_eq!(outputs[0].shape, [2, 1]);
    assert_eq!(outputs[0].as_bytes(), [2, 5]);
}

#[test]
fn compress_28_rejects_float8() {
    let invalid = model(28, DataType::Float8E4M3FN, None, 2);
    let err = load(&invalid, DeviceId::cpu())
        .err()
        .expect("Compress-28 uses the IR4 tensor set, not all fixed-width types");
    let message = err.to_string();
    for detail in ["select_values", "Float8E4M3FN", "Compress-28"] {
        assert!(message.contains(detail), "{message}");
    }
}

#[cfg(feature = "cuda")]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires a CUDA device")]
#[test]
fn compress_28_bf16_cuda() {
    check_bf16_outputs(DeviceId::cuda(0));
}
