use onnx_runtime_ir::{DataType, Graph, Node, NodeId, static_shape};
use onnx_runtime_loader::{Model, ModelMetadata, encode_model};
use onnx_runtime_session::InferenceSession;

#[test]
fn session_exposes_source_model_metadata() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let input = graph.create_named_value("X", DataType::Float32, static_shape([1]));
    let output = graph.create_named_value("Y", DataType::Float32, static_shape([1]));
    graph.add_input(input);
    graph.add_output(output);
    graph.insert_node(Node::new(
        NodeId(0),
        "Relu",
        vec![Some(input)],
        vec![output],
    ));

    let metadata = ModelMetadata {
        producer_name: "nxrt-test".into(),
        domain: "com.example".into(),
        model_version: 7,
        doc_string: Some("metadata regression test".into()),
        graph_name: "metadata_graph".into(),
        metadata_props: vec![("author".into(), "rachael".into())],
        ..Default::default()
    };
    let bytes = encode_model(&Model::new(&graph).with_metadata(metadata.clone())).unwrap();

    let session = InferenceSession::load_bytes(&bytes).unwrap();
    assert_eq!(session.model_metadata(), &metadata);
}
