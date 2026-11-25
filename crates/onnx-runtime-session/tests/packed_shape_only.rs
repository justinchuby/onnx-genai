#[path = "packed_shape_only/cases.rs"]
mod cases;

use cases::{
    CASES, IR9_FP8_TYPES, TYPES, assert_ir9_fp8_reshape, assert_transport, dynamic_reshape_graph,
    graph,
};
use onnx_runtime_ir::{DataType, DeviceId, Graph, Node, NodeId, static_shape};
use onnx_runtime_loader::{LoaderError, Model, encode_model, load_model_bytes, validate_model};
use onnx_runtime_session::InferenceSession;

#[test]
fn packed_shape_only_cpu_transport() {
    assert_transport(
        |graph| InferenceSession::from_graph(graph).unwrap(),
        DeviceId::cpu(),
    );
}

#[test]
fn reshape_19_dynamic_fp8_cpu_transport() {
    assert_ir9_fp8_reshape(
        |graph| InferenceSession::from_graph(graph).unwrap(),
        DeviceId::cpu(),
    );
}

#[test]
fn shape_only_ir9_fp8_opset_boundaries() {
    for dtype in IR9_FP8_TYPES {
        let Err(error) = InferenceSession::from_graph(dynamic_reshape_graph(18, dtype)) else {
            panic!("Reshape-18 accepted {dtype:?}");
        };
        let message = error.to_string();
        assert!(
            message.contains("Reshape")
                && message.contains(&format!("{dtype:?}"))
                && message.contains("at opset 18")
                && message.contains("use opset 19"),
            "{message}"
        );
        for case @ (op, ..) in CASES.into_iter().skip(1) {
            let Err(error) = InferenceSession::from_graph(graph(20, dtype, case)) else {
                panic!("{op} accepted {dtype:?} before opset 21");
            };
            let message = error.to_string();
            assert!(
                message.contains(op)
                    && message.contains(&format!("{dtype:?}"))
                    && message.contains("at opset 20")
                    && message.contains("use opset 21"),
                "{message}"
            );
        }
    }
}

#[test]
fn packed_shape_only_rejects_early_opsets() {
    for (dtype, since, _) in TYPES {
        for case @ (op, ..) in CASES {
            let opset = since - 1;
            let Err(error) = InferenceSession::from_graph(graph(opset, dtype, case)) else {
                panic!("{op} accepted {dtype:?} at opset {opset}, before opset {since}");
            };
            let message = error.to_string();
            assert!(
                message.contains(op) && message.contains(&format!("{dtype:?}")),
                "{message}"
            );
            assert!(
                message.contains(&format!("at opset {opset}"))
                    && message.contains(&format!("use opset {since}")),
                "{message}"
            );
        }
    }
}

#[test]
fn shape_only_serialized_admission_and_transport() {
    let encode = |mut graph: Graph| {
        graph.ir_version = Some(13);
        encode_model(&Model::new(&graph)).expect("encode shape-only fixture")
    };
    for (op, dtype, since, graph) in [
        (
            "Reshape",
            DataType::Float8E4M3FN,
            19,
            dynamic_reshape_graph(18, DataType::Float8E4M3FN),
        ),
        (
            "Reshape",
            DataType::Int4,
            21,
            graph(20, DataType::Int4, CASES[0]),
        ),
        (
            "Flatten",
            DataType::Int4,
            21,
            graph(20, DataType::Int4, CASES[1]),
        ),
        (
            "Squeeze",
            DataType::Int4,
            21,
            graph(20, DataType::Int4, CASES[2]),
        ),
        (
            "Unsqueeze",
            DataType::Int4,
            21,
            graph(20, DataType::Int4, CASES[3]),
        ),
    ] {
        assert!(
            matches!(
                load_model_bytes(&encode(graph)),
                Err(LoaderError::InvalidShapeOnlyDtype {
                    op: rejected_op,
                    dtype: rejected_dtype,
                    opset,
                    since: required,
                    ..
                }) if rejected_op == op && rejected_dtype == dtype && opset == since - 1 && required == since
            ),
            "{op}: serialized model must reject {dtype:?} before opset {since}"
        );
    }

    let build = |graph| {
        InferenceSession::builder()
            .model_bytes(&encode(graph))
            .build()
            .unwrap()
    };
    assert_transport(build, DeviceId::cpu());
    assert_ir9_fp8_reshape(build, DeviceId::cpu());
}

#[test]
fn shape_only_serialized_admission_checks_inferred_input_type() {
    let dtype = DataType::Float8E4M3FN;
    let mut graph = graph(20, dtype, CASES[1]);
    graph.ir_version = Some(13);
    let flatten = graph.nodes.keys().next().unwrap();
    let input = graph.inputs[0];
    let intermediate = graph.create_named_value("intermediate", dtype, static_shape([5, 1, 1]));
    graph.mark_value_type_unknown(intermediate);
    graph.insert_node(Node::new(
        NodeId(0),
        "Identity",
        vec![Some(input)],
        vec![intermediate],
    ));
    assert_eq!(
        graph.replace_input(flatten, 0, Some(intermediate)),
        Some(input)
    );
    let bytes =
        encode_model(&Model::new(&graph)).expect("encode fixture with untyped intermediate");
    assert!(matches!(
        load_model_bytes(&bytes),
        Err(LoaderError::InvalidShapeOnlyDtype {
            slot: "input",
            dtype: DataType::Float8E4M3FN,
            opset: 20,
            since: 21,
            ..
        })
    ));
}

#[test]
fn shape_only_admission_uses_node_version_and_known_output_type() {
    let mut graph = graph(21, DataType::Int4, CASES[0]);
    let reshape = graph.nodes.keys().next().unwrap();
    let input = graph.inputs[0];
    graph.mark_value_type_unknown(input);
    let node = graph.node_mut(reshape);
    node.domain = "ai.onnx".into();
    node.version = Some(20);
    graph.normalize_domains();
    assert!(matches!(
        validate_model(&graph),
        Err(LoaderError::InvalidShapeOnlyDtype {
            slot: "output",
            dtype: DataType::Int4,
            opset: 20,
            since: 21,
            ..
        })
    ));
    graph.node_mut(reshape).version = None;
    validate_model(&graph).expect("admit INT4 with the imported opset 21");
}
