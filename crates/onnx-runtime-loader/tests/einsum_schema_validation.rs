use prost::Message;

use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, static_shape};
use onnx_runtime_loader::{LoaderError, load_model_bytes, proto::onnx, validate_model};

fn tensor_type(elem_type: i32, shape: Option<&[i64]>) -> onnx::TypeProto {
    use onnx::tensor_shape_proto::{Dimension, dimension::Value};

    onnx::TypeProto {
        value: Some(onnx::type_proto::Value::TensorType(
            onnx::type_proto::Tensor {
                elem_type,
                shape: shape.map(|shape| onnx::TensorShapeProto {
                    dim: shape
                        .iter()
                        .map(|&dimension| Dimension {
                            value: Some(Value::DimValue(dimension)),
                            ..Default::default()
                        })
                        .collect(),
                }),
            },
        )),
        ..Default::default()
    }
}

fn value_info(name: &str, elem_type: Option<i32>, shape: Option<&[i64]>) -> onnx::ValueInfoProto {
    onnx::ValueInfoProto {
        name: name.to_string(),
        r#type: elem_type.map(|elem_type| tensor_type(elem_type, shape)),
        ..Default::default()
    }
}

fn einsum_node(input: &str, output: &str, equation: &str) -> onnx::NodeProto {
    onnx::NodeProto {
        op_type: "Einsum".to_string(),
        input: vec![input.to_string()],
        output: vec![output.to_string()],
        attribute: vec![onnx::AttributeProto {
            name: "equation".to_string(),
            r#type: onnx::attribute_proto::AttributeType::String as i32,
            s: equation.as_bytes().to_vec(),
            ..Default::default()
        }],
        ..Default::default()
    }
}

fn model(
    opset: i64,
    input: onnx::ValueInfoProto,
    output: onnx::ValueInfoProto,
    nodes: Vec<onnx::NodeProto>,
) -> Vec<u8> {
    onnx::ModelProto {
        ir_version: 8,
        opset_import: vec![onnx::OperatorSetIdProto {
            domain: String::new(),
            version: opset,
        }],
        graph: Some(onnx::GraphProto {
            input: vec![input],
            output: vec![output],
            node: nodes,
            ..Default::default()
        }),
        ..Default::default()
    }
    .encode_to_vec()
}

fn find(graph: &Graph, name: &str) -> onnx_runtime_ir::ValueId {
    graph
        .values
        .iter()
        .find_map(|(id, value)| (value.name.as_deref() == Some(name)).then_some(id))
        .unwrap_or_else(|| panic!("value {name:?} was not loaded"))
}

fn einsum_graph(opset: u64, dtype: DataType) -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), opset);
    let input = graph.create_named_value("input", dtype, static_shape([]));
    let output = graph.create_named_value("output", dtype, static_shape([]));
    graph.add_input(input);
    graph.add_output(output);
    let mut node = Node::new(NodeId(0), "Einsum", vec![Some(input)], vec![output]);
    node.name = "einsum".to_string();
    node.attributes
        .insert("equation".to_string(), Attribute::String(b"->".to_vec()));
    graph.insert_node(node);
    graph
}

#[test]
fn loader_resolves_einsum_schema_from_imported_opset() {
    let error = validate_model(&einsum_graph(11, DataType::Float32)).unwrap_err();
    assert!(matches!(error, LoaderError::InvalidEinsum { .. }));
    assert!(error.to_string().contains("predates Einsum-12"));

    for opset in [12, 27] {
        let error = validate_model(&einsum_graph(opset, DataType::BFloat16)).unwrap_err();
        assert!(matches!(error, LoaderError::InvalidEinsum { .. }));
        assert!(error.to_string().contains("not admitted by Einsum-12"));
    }

    validate_model(&einsum_graph(28, DataType::BFloat16)).unwrap();
}

#[test]
fn omitted_interior_value_info_is_inferred_after_fail_fast_validation() {
    let graph = load_model_bytes(&model(
        12,
        value_info("X", Some(1), Some(&[2])),
        value_info("Y", Some(1), Some(&[2])),
        vec![
            einsum_node("X", "interior", "i->i"),
            einsum_node("interior", "Y", "i->i"),
        ],
    ))
    .unwrap();

    let interior = find(&graph, "interior");
    assert!(!graph.value_type_is_known(interior));
    assert!(!graph.value_shape_is_known(interior));
    assert_eq!(graph.value(interior).dtype, DataType::Float32);
    assert_eq!(graph.value(interior).shape, static_shape([2]));
}

#[test]
fn unannotated_f16_and_opset28_bf16_values_do_not_claim_placeholder_f32() {
    for (opset, raw_dtype, dtype) in [(12, 10, DataType::Float16), (28, 16, DataType::BFloat16)] {
        let graph = load_model_bytes(&model(
            opset,
            value_info("X", Some(raw_dtype), Some(&[3])),
            value_info("Y", None, None),
            vec![
                einsum_node("X", "interior", "i->i"),
                einsum_node("interior", "Y", "i->i"),
            ],
        ))
        .unwrap_or_else(|error| panic!("opset {opset}, dtype {dtype:?}: {error}"));

        for name in ["interior", "Y"] {
            let value = find(&graph, name);
            assert!(!graph.value_type_is_known(value), "{name}, opset {opset}");
            assert!(!graph.value_shape_is_known(value), "{name}, opset {opset}");
            assert_eq!(graph.value(value).dtype, dtype, "{name}, opset {opset}");
            assert_eq!(
                graph.value(value).shape,
                static_shape([3]),
                "{name}, opset {opset}"
            );
        }
    }
}

#[test]
fn tensor_type_without_shape_is_known_type_not_a_declared_scalar() {
    let graph = load_model_bytes(&model(
        12,
        value_info("X", Some(10), None),
        value_info("Y", Some(10), None),
        vec![einsum_node("X", "Y", "i->i")],
    ))
    .unwrap();

    for name in ["X", "Y"] {
        let value = find(&graph, name);
        assert!(graph.value_type_is_known(value), "{name}");
        assert!(!graph.value_shape_is_known(value), "{name}");
        assert_eq!(graph.value(value).dtype, DataType::Float16, "{name}");
        assert!(graph.value(value).shape.is_empty(), "{name}");
    }
}

#[test]
fn partial_metadata_still_rejects_every_known_invalid_fact() {
    let invalid_dtype = load_model_bytes(&model(
        27,
        value_info("X", Some(16), None),
        value_info("Y", None, None),
        vec![einsum_node("X", "Y", "i->i")],
    ))
    .unwrap_err();
    assert!(
        invalid_dtype
            .to_string()
            .contains("not admitted by Einsum-12")
    );

    let output_mismatch = load_model_bytes(&model(
        12,
        value_info("X", Some(10), None),
        value_info("Y", Some(1), None),
        vec![einsum_node("X", "Y", "i->i")],
    ))
    .unwrap_err();
    assert!(
        output_mismatch
            .to_string()
            .contains("does not match known homogeneous input dtype")
    );

    let malformed_equation = load_model_bytes(&model(
        12,
        value_info("X", None, None),
        value_info("Y", None, None),
        vec![einsum_node("X", "Y", "i$->i")],
    ))
    .unwrap_err();
    assert!(malformed_equation.to_string().contains("invalid character"));

    let mut known_shape_unknown_type = einsum_graph(12, DataType::Float32);
    let input = known_shape_unknown_type.inputs[0];
    let output = known_shape_unknown_type.outputs[0];
    known_shape_unknown_type.value_mut(input).shape = static_shape([2, 3]);
    known_shape_unknown_type.mark_value_type_unknown(input);
    known_shape_unknown_type.mark_value_type_unknown(output);
    let invalid_rank = validate_model(&known_shape_unknown_type).unwrap_err();
    assert!(invalid_rank.to_string().contains("rank 2 does not match"));

    let mut one_known_bad_shape = Graph::new();
    one_known_bad_shape.opset_imports.insert(String::new(), 12);
    let left =
        one_known_bad_shape.create_named_value("left", DataType::Float32, static_shape([2, 3]));
    let right = one_known_bad_shape.create_named_value("right", DataType::Float32, Vec::new());
    let result = one_known_bad_shape.create_named_value("result", DataType::Float32, Vec::new());
    one_known_bad_shape.mark_value_shape_unknown(right);
    one_known_bad_shape.mark_value_type_unknown(result);
    one_known_bad_shape.mark_value_shape_unknown(result);
    one_known_bad_shape.add_input(left);
    one_known_bad_shape.add_input(right);
    one_known_bad_shape.add_output(result);
    let mut node = Node::new(
        NodeId(0),
        "Einsum",
        vec![Some(left), Some(right)],
        vec![result],
    );
    node.attributes.insert(
        "equation".to_string(),
        Attribute::String(b"i,j->ij".to_vec()),
    );
    one_known_bad_shape.insert_node(node);
    let invalid_partial_rank = validate_model(&one_known_bad_shape).unwrap_err();
    assert!(
        invalid_partial_rank
            .to_string()
            .contains("input #0 rank 2 does not match")
    );
}
