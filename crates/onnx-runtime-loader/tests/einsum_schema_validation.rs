use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, static_shape};
use onnx_runtime_loader::{LoaderError, validate_model};

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
