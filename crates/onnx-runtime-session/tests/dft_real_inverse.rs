use std::sync::Arc;

use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, static_shape};
use onnx_runtime_session::{InferenceSession, Tensor};

#[test]
fn real_fft_roundtrip_loads_and_executes_with_real_inverse_output() {
    for version in [17, 20] {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), version);
        let input = graph.create_named_value("signal", DataType::Float32, static_shape([2, 8, 1]));
        let spectrum =
            graph.create_named_value("spectrum", DataType::Float32, static_shape([2, 5, 2]));
        let output =
            graph.create_named_value("recovered", DataType::Float32, static_shape([2, 8, 1]));
        graph.add_input(input);
        graph.add_output(output);
        for (id, x, y, inverse) in [(0, input, spectrum, 0), (1, spectrum, output, 1)] {
            let mut node = Node::new(NodeId(id), "DFT", vec![Some(x)], vec![y]);
            node.attributes.insert("onesided".into(), Attribute::Int(1));
            node.attributes
                .insert("inverse".into(), Attribute::Int(inverse));
            if version < 20 {
                node.attributes.insert("axis".into(), Attribute::Int(-2));
            }
            graph.insert_node(node);
        }
        let bytes =
            onnx_runtime_loader::encode_model(&onnx_runtime_loader::Model::new(&graph)).unwrap();
        let cpu = onnx_runtime_ep_cpu::CpuExecutionProvider::initialized_with_resources(
            Default::default(),
        )
        .unwrap();
        let mut session = InferenceSession::builder()
            .model_bytes(&bytes)
            .execution_provider(Arc::new(cpu))
            .build()
            .unwrap();
        let values = (0..16)
            .map(|i| (i as f32 * 0.375).sin() + i as f32 * 0.125)
            .collect::<Vec<_>>();
        let input = Tensor::from_f32(&[2, 8, 1], &values).unwrap();
        for _ in 0..3 {
            let outputs = session.run(&[("signal", &input)]).unwrap();
            assert_eq!(outputs[0].shape, [2, 8, 1]);
            for (&actual, &expected) in outputs[0].try_as_slice_f32().unwrap().iter().zip(&values) {
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "opset {version}: {actual} versus {expected}"
                );
            }
        }
    }
}
