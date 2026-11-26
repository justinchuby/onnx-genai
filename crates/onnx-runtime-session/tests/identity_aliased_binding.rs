use onnx_runtime_ir::{DataType, Graph, Node, NodeId, static_shape};
use onnx_runtime_session::InferenceSession;
#[test]
fn public_identity_binding_can_share_input_output_backing_without_workspace() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 13);
    let x = graph.create_named_value("X", DataType::Float32, static_shape([2, 4]));
    graph.add_input(x);
    let y = graph.create_named_value("Y", DataType::Float32, static_shape([2, 4]));
    graph.add_output(y);
    graph.insert_node(Node::new(NodeId(0), "Identity", vec![Some(x)], vec![y]));
    let model =
        onnx_runtime_loader::encode_model(&onnx_runtime_loader::Model::new(&graph)).unwrap();
    let mut session = InferenceSession::builder()
        .model_bytes(&model)
        .build()
        .unwrap();
    let mut binding = session
        .allocate_device_binding("X", Some("Y"), DataType::Float32, vec![2, 4], vec![2, 4])
        .unwrap();
    let bits = [
        0u32, 0x80000000, 0x7fc12345, 0xffc54321, 0x7f800000, 0xff800000, 0x3f800000, 1,
    ];
    let bytes = bits
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect::<Vec<_>>();
    for _ in 0..2 {
        binding.write_bytes(0, &bytes).unwrap();
        let report = session
            .prepare_with_device_bindings(&[], std::slice::from_mut(&mut binding))
            .unwrap();
        assert_eq!(report.require_complete().unwrap().step().bytes, 0);
        let output = session
            .run_with_device_bindings(&[], std::slice::from_mut(&mut binding))
            .unwrap();
        assert!(output[0].is_none());
        assert_eq!(binding.read_bytes().unwrap(), bytes);
    }
}
