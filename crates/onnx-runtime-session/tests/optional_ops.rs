use onnx_runtime_ir::onnx::{TypeProto, type_proto};
use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, ValueId, static_shape};
use onnx_runtime_session::{InferenceSession, Tensor};

fn graph(opset: u64) -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), opset);
    graph
}

fn tensor_type() -> TypeProto {
    TypeProto {
        value: Some(type_proto::Value::TensorType(type_proto::Tensor {
            elem_type: DataType::Int64.to_onnx(),
            shape: None,
        })),
        ..Default::default()
    }
}

fn op(
    g: &mut Graph,
    name: &str,
    inputs: Vec<Option<ValueId>>,
    dtype: DataType,
    shape: &[usize],
    attr: Option<TypeProto>,
) -> ValueId {
    let out = g.create_value(dtype, static_shape(shape.iter().copied()));
    let mut node = Node::new(NodeId(0), name, inputs, vec![out]);
    if let Some(tp) = attr {
        node.attributes
            .insert("type".into(), Attribute::TypeProto(tp));
    }
    g.insert_node(node);
    out
}

#[test]
fn tensor_round_trip_and_direct_operands() {
    let mut g = graph(28);
    let input = g.create_named_value("x", DataType::Int64, static_shape([2]));
    g.add_input(input);
    let wrapped = op(
        &mut g,
        "Optional",
        vec![Some(input)],
        DataType::Undefined,
        &[],
        None,
    );
    for operand in [wrapped, input] {
        let has = op(
            &mut g,
            "OptionalHasElement",
            vec![Some(operand)],
            DataType::Bool,
            &[],
            None,
        );
        let get = op(
            &mut g,
            "OptionalGetElement",
            vec![Some(operand)],
            DataType::Int64,
            &[2],
            None,
        );
        g.add_output(has);
        g.add_output(get);
    }
    let mut session = InferenceSession::from_graph(g).unwrap();
    for values in [[-17_i64, 42], [5, -12]] {
        let bytes: Vec<_> = values.into_iter().flat_map(i64::to_le_bytes).collect();
        let input = Tensor::from_raw(DataType::Int64, vec![2], &bytes).unwrap();
        let outputs = session.run(&[("x", &input)]).unwrap();
        for pair in outputs.chunks_exact(2) {
            assert_eq!(pair[0].dtype, DataType::Bool);
            assert!(pair[0].shape.is_empty());
            assert_eq!(pair[0].as_bytes(), &[1]);
            assert_eq!(pair[1].dtype, DataType::Int64);
            assert_eq!(pair[1].shape, [2]);
            assert_eq!(pair[1].as_bytes(), bytes);
        }
    }
}

#[test]
fn absent_is_distinct_from_present_empty_sequence() {
    let mut g = graph(28);
    let absent = op(
        &mut g,
        "Optional",
        vec![None],
        DataType::Undefined,
        &[],
        Some(tensor_type()),
    );
    let seq = op(
        &mut g,
        "SequenceEmpty",
        vec![],
        DataType::Undefined,
        &[],
        None,
    );
    let present = op(
        &mut g,
        "Optional",
        vec![Some(seq)],
        DataType::Undefined,
        &[],
        None,
    );
    for operand in [None, Some(absent), Some(present), Some(seq)] {
        let has = op(
            &mut g,
            "OptionalHasElement",
            operand.into_iter().map(Some).collect(),
            DataType::Bool,
            &[],
            None,
        );
        g.add_output(has);
    }
    let unwrapped = op(
        &mut g,
        "OptionalGetElement",
        vec![Some(present)],
        DataType::Undefined,
        &[],
        None,
    );
    let direct = op(
        &mut g,
        "OptionalGetElement",
        vec![Some(unwrapped)],
        DataType::Undefined,
        &[],
        None,
    );
    let len = op(
        &mut g,
        "SequenceLength",
        vec![Some(direct)],
        DataType::Int64,
        &[],
        None,
    );
    g.add_output(len);
    let outputs = InferenceSession::from_graph(g).unwrap().run(&[]).unwrap();
    assert_eq!(
        outputs
            .iter()
            .take(4)
            .map(|t| t.as_bytes()[0])
            .collect::<Vec<_>>(),
        [0, 0, 1, 1]
    );
    assert_eq!(outputs[4].as_bytes(), &0_i64.to_le_bytes());
}

#[test]
fn absent_get_reports_the_input_and_guard() {
    let mut g = graph(28);
    let absent = op(
        &mut g,
        "Optional",
        vec![],
        DataType::Undefined,
        &[],
        Some(tensor_type()),
    );
    let value = op(
        &mut g,
        "OptionalGetElement",
        vec![Some(absent)],
        DataType::Int64,
        &[],
        None,
    );
    g.add_output(value);
    let mut session = InferenceSession::from_graph(g).unwrap();
    let message = session.run(&[]).unwrap_err().to_string();
    assert!(
        message.contains("OptionalGetElement")
            && message.contains("absent")
            && message.contains("OptionalHasElement"),
        "{message}"
    );
}

#[test]
fn opset_15_rejects_direct_tensor_get() {
    let mut g = graph(15);
    let input = g.create_named_value("x", DataType::Int64, static_shape([2]));
    g.add_input(input);
    let value = op(
        &mut g,
        "OptionalGetElement",
        vec![Some(input)],
        DataType::Int64,
        &[2],
        None,
    );
    g.add_output(value);
    let message = InferenceSession::from_graph(g).err().unwrap().to_string();
    assert!(message.contains("before opset 18"), "{message}");
}

#[test]
fn standalone_optional_graph_interfaces_are_rejected() {
    let mut g = graph(28);
    let input = g.create_named_value("x", DataType::Undefined, vec![]);
    g.value_mut(input).declared_type = Some(TypeProto {
        value: Some(type_proto::Value::OptionalType(Box::new(
            type_proto::Optional {
                elem_type: Some(Box::new(tensor_type())),
            },
        ))),
        ..Default::default()
    });
    g.add_input(input);
    let output = op(
        &mut g,
        "OptionalHasElement",
        vec![Some(input)],
        DataType::Bool,
        &[],
        None,
    );
    g.add_output(output);
    let message = InferenceSession::from_graph(g).err().unwrap().to_string();
    assert!(
        message.contains("graph input") && message.contains("'x'"),
        "{message}"
    );

    let mut g = graph(28);
    let output = op(
        &mut g,
        "Optional",
        vec![],
        DataType::Undefined,
        &[],
        Some(tensor_type()),
    );
    g.add_output(output);
    let message = InferenceSession::from_graph(g).err().unwrap().to_string();
    assert!(
        message.contains("graph output") && message.contains("cannot return Optional"),
        "{message}"
    );
}

#[cfg(feature = "cuda")]
#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires a CUDA GPU")]
fn cuda_optional_preserves_sequence_payload_residency() {
    use onnx_runtime_ir::{DeviceType, TensorData, WeightRef};
    use onnx_runtime_loader::{Model, encode_model};
    use onnx_runtime_session::DevicePreference;

    let mut g = graph(28);
    g.ir_version = Some(14);
    let tensor = g.create_named_value("x", DataType::Int64, static_shape([2]));
    let bytes: Vec<_> = [-17_i64, 42]
        .into_iter()
        .flat_map(i64::to_le_bytes)
        .collect();
    g.set_initializer(
        tensor,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            vec![2],
            bytes.clone(),
        )),
    );
    let seq = op(
        &mut g,
        "SequenceConstruct",
        vec![Some(tensor)],
        DataType::Undefined,
        &[],
        None,
    );
    let wrapped = op(
        &mut g,
        "Optional",
        vec![Some(seq)],
        DataType::Undefined,
        &[],
        None,
    );
    let get = op(
        &mut g,
        "OptionalGetElement",
        vec![Some(wrapped)],
        DataType::Undefined,
        &[],
        None,
    );
    g.value_mut(get).name = Some("out".into());
    g.value_mut(get).declared_type = Some(TypeProto {
        value: Some(type_proto::Value::SequenceType(Box::new(
            type_proto::Sequence {
                elem_type: Some(Box::new(tensor_type())),
            },
        ))),
        ..Default::default()
    });
    g.add_output(get);
    let model = encode_model(&Model::new(&g)).unwrap();
    let mut session = InferenceSession::builder()
        .model_bytes(&model)
        .device(DevicePreference::Gpu { index: Some(0) })
        .build()
        .unwrap();
    assert_eq!(session.device_id().device_type, DeviceType::Cuda);
    let outputs = session.run_outputs(&[]).unwrap();
    let seq = outputs[0].as_sequence().unwrap();
    let elem = &seq.elements()[0];
    assert_eq!(elem.device().device_type, DeviceType::Cuda);
    assert_eq!(elem.dtype, DataType::Int64);
    assert_eq!(elem.contiguous_bytes().unwrap(), bytes);
}
