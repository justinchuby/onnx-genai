//! CUDA placement regressions for constrained GLM standard operators.

use onnx_runtime_ep_api::{ExecutionProvider, KernelMatch};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, static_shape};

fn node(
    op_type: &str,
    input_dtypes: &[DataType],
    outputs: usize,
    omitted_input: Option<usize>,
    attrs: &[(&str, Attribute)],
) -> (Graph, NodeId) {
    let mut graph = Graph::new();
    let inputs = input_dtypes
        .iter()
        .enumerate()
        .map(|(index, &dtype)| {
            let value =
                graph.create_named_value(format!("input_{index}"), dtype, static_shape([1]));
            graph.add_input(value);
            if omitted_input == Some(index) {
                None
            } else {
                Some(value)
            }
        })
        .collect();
    let outputs = (0..outputs)
        .map(|index| {
            graph.create_named_value(
                format!("output_{index}"),
                DataType::Float32,
                static_shape([1]),
            )
        })
        .collect();
    let mut node = Node::new(NodeId(0), op_type, inputs, outputs);
    for (name, value) in attrs {
        node.attributes.insert((*name).into(), value.clone());
    }
    let id = graph.insert_node(node);
    (graph, id)
}

fn assert_rejected(
    ep: &CudaExecutionProvider,
    op_type: &str,
    opset: u64,
    input_dtypes: &[DataType],
    outputs: usize,
) {
    let (graph, id) = node(op_type, input_dtypes, outputs, None, &[]);
    assert!(
        matches!(
            ep.supports_op(graph.node(id), opset, &[], input_dtypes, &[]),
            KernelMatch::Unsupported { .. }
        ),
        "{op_type} must reject its unsupported input dtype at claim time"
    );
}

#[test]
fn glm_standard_claim_gates_reject_runtime_unsupported_input_dtypes() {
    let ep = CudaExecutionProvider::new_default().expect("CUDA runtime must be available");

    assert_rejected(
        &ep,
        "RMSNormalization",
        23,
        &[DataType::Float16, DataType::Float32],
        1,
    );
    assert_rejected(
        &ep,
        "RotaryEmbedding",
        23,
        &[
            DataType::Float32,
            DataType::Float32,
            DataType::Float32,
            DataType::Int32,
        ],
        1,
    );
    assert_rejected(&ep, "TopK", 24, &[DataType::Float32, DataType::Int32], 2);
    assert_rejected(&ep, "CumSum", 24, &[DataType::Float32, DataType::Int32], 1);
    assert_rejected(
        &ep,
        "Gather",
        24,
        &[DataType::Float32, DataType::Float32],
        1,
    );
    assert_rejected(
        &ep,
        "GatherElements",
        24,
        &[DataType::Float32, DataType::Int32],
        1,
    );
    assert_rejected(
        &ep,
        "ScatterElements",
        24,
        &[DataType::Float32, DataType::Int32, DataType::Float32],
        1,
    );
    assert_rejected(
        &ep,
        "Where",
        24,
        &[DataType::Int64, DataType::Float32, DataType::Float32],
        1,
    );
    assert_rejected(&ep, "Expand", 24, &[DataType::Float32, DataType::Int32], 1);
}

#[test]
fn optional_glm_inputs_distinguish_omission_from_wrong_dtype() {
    let ep = CudaExecutionProvider::new_default().expect("CUDA runtime must be available");
    let dtypes = [
        DataType::Float32,
        DataType::Float32,
        DataType::Float32,
        DataType::Undefined,
    ];
    let (omitted_graph, omitted_id) = node("RotaryEmbedding", &dtypes, 1, Some(3), &[]);
    assert!(
        ep.supports_op(omitted_graph.node(omitted_id), 23, &[], &dtypes, &[])
            .is_supported(),
        "an omitted RotaryEmbedding position_ids must be claimed"
    );

    let present_dtypes = [
        DataType::Float32,
        DataType::Float32,
        DataType::Float32,
        DataType::Int32,
    ];
    let (present_graph, present_id) = node("RotaryEmbedding", &present_dtypes, 1, None, &[]);
    assert!(matches!(
        ep.supports_op(
            present_graph.node(present_id),
            23,
            &[],
            &present_dtypes,
            &[]
        ),
        KernelMatch::Unsupported { ref reason } if reason.contains("position_ids")
    ));
}

#[test]
fn claim_gates_reject_attributes_cuda_would_otherwise_silently_coerce() {
    let ep = CudaExecutionProvider::new_default().expect("CUDA runtime must be available");
    for (op_type, opset, input_dtypes, outputs, attrs) in [
        (
            "RMSNormalization",
            23,
            vec![DataType::Float32, DataType::Float32],
            1,
            vec![("stash_type", Attribute::Int(16))],
        ),
        (
            "RotaryEmbedding",
            23,
            vec![DataType::Float32, DataType::Float32, DataType::Float32],
            1,
            vec![("num_heads", Attribute::Int(-1))],
        ),
        (
            "TopK",
            24,
            vec![DataType::Float32, DataType::Int64],
            2,
            vec![("largest", Attribute::Int(2))],
        ),
        (
            "CumSum",
            24,
            vec![DataType::Float32, DataType::Int64],
            1,
            vec![("exclusive", Attribute::Int(-1))],
        ),
        (
            "ScatterElements",
            24,
            vec![DataType::Float32, DataType::Int64, DataType::Float32],
            1,
            vec![("reduction", Attribute::String(b"overwrite".to_vec()))],
        ),
    ] {
        let (graph, id) = node(op_type, &input_dtypes, outputs, None, &attrs);
        assert!(
            matches!(
                ep.supports_op(graph.node(id), opset, &[], &input_dtypes, &[]),
                KernelMatch::Unsupported { .. }
            ),
            "{op_type} must reject an invalid attribute at claim time"
        );
    }
}
