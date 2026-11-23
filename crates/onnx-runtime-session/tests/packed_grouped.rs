use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, TensorData, WeightRef, dequantize_packed::Format,
    static_shape,
};
use onnx_runtime_session::{InferenceSession, Tensor};

fn graph(format: Format, expose_decoded: bool) -> Graph {
    let (width, bytes) = format.block();
    let (groups, n, k, m, selections) = (3, 2, 2 * width, 2, 2);
    let blocks = groups * n * k / width;
    let mut encoded = Vec::with_capacity(blocks * bytes);
    for block in 0..blocks {
        encoded.extend((0..bytes).map(|i| ((i * 13 + block * 17 + 97) % 119) as u8));
    }

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 29);
    let activation = graph.create_named_value("A", DataType::Float32, static_shape([m, k]));
    let indices = graph.create_named_value("I", DataType::Int64, static_shape([m, selections]));
    graph.add_input(activation);
    graph.add_input(indices);
    let packed = graph.create_named_value("packed", DataType::Uint8, static_shape([encoded.len()]));
    graph.set_initializer(
        packed,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Uint8,
            vec![encoded.len()],
            encoded,
        )),
    );
    let decoded =
        graph.create_named_value("decoded", DataType::Float32, static_shape([groups, n, k]));
    let mut dequantize = Node::new(
        NodeId(0),
        "DequantizePacked",
        vec![Some(packed)],
        vec![decoded],
    );
    dequantize.attributes.insert(
        "format".into(),
        Attribute::String(format.name().as_bytes().to_vec()),
    );
    dequantize.attributes.insert(
        "shape".into(),
        Attribute::Ints(vec![groups as i64, n as i64, k as i64]),
    );
    graph.insert_node(dequantize);
    let output = graph.create_named_value("Y", DataType::Float32, static_shape([m, selections, n]));
    let mut grouped = Node::new(
        NodeId(0),
        "GroupedMatMul",
        vec![Some(activation), Some(decoded), Some(indices)],
        vec![output],
    );
    grouped
        .attributes
        .insert("transB".into(), Attribute::Int(1));
    graph.insert_node(grouped);
    graph.add_output(output);
    if expose_decoded {
        graph.add_output(decoded);
    }
    graph
}

fn compare(build: impl Fn(Graph) -> InferenceSession) {
    for &format in Format::ALL {
        let k = 2 * format.block().0;
        let activation: Vec<_> = (0..2 * k)
            .map(|i| ((i * 7 % 13) as f32 - 6.0) / 7.0)
            .collect();
        let activation = Tensor::from_f32(&[2, k], &activation).unwrap();
        let indices = Tensor::from_i64(&[2, 2], &[1, 0, 0, 1]).unwrap();
        let mut reference = build(graph(format, true));
        let mut fused = build(graph(format, false));
        assert!(
            fused
                .graph()
                .nodes
                .values()
                .any(|node| node.op_type == "PackedGroupedMatMul"),
            "{format:?}"
        );
        assert!(
            !fused
                .graph()
                .nodes
                .values()
                .any(|node| matches!(node.op_type.as_str(), "DequantizePacked" | "GroupedMatMul")),
            "{format:?}"
        );
        let expected = reference
            .run(&[("A", &activation), ("I", &indices)])
            .unwrap()[0]
            .to_vec_f32();
        let actual = fused.run(&[("A", &activation), ("I", &indices)]).unwrap()[0].to_vec_f32();
        let scale = expected
            .iter()
            .filter(|value| value.is_finite())
            .map(|value| value.abs())
            .fold(1.0f32, f32::max);
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                actual == expected
                    || actual.is_nan() && expected.is_nan()
                    || (actual - expected).abs() <= 4e-6 * scale,
                "{format:?}: {actual} versus {expected}"
            );
        }
    }
}

#[test]
fn packed_grouped_cpu_parity_all_formats() {
    compare(|graph| InferenceSession::from_graph(graph).unwrap());
}

#[test]
fn packed_grouped_keeps_shared_decode_as_fallback() {
    let session = InferenceSession::from_graph(graph(Format::Q8_0, true)).unwrap();
    assert!(
        session
            .graph()
            .nodes
            .values()
            .any(|node| node.op_type == "DequantizePacked")
    );
    assert!(
        session
            .graph()
            .nodes
            .values()
            .any(|node| node.op_type == "GroupedMatMul")
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device and NVRTC"]
fn packed_grouped_cuda_parity_all_formats() {
    use onnx_runtime_ep_cuda::CudaExecutionProvider;
    use std::{path::Path, sync::Arc};

    let provider =
        Arc::new(CudaExecutionProvider::initialized(0).expect("initialize CUDA device 0"));
    compare(|graph| {
        let session = InferenceSession::from_graph_with_provider(
            graph,
            Arc::new(onnx_runtime_loader::WeightStore::new()),
            Path::new("."),
            provider.clone(),
            None,
        )
        .unwrap();
        assert!(
            session.execution_provider_fallback_report().is_none(),
            "{:?}",
            session.execution_provider_fallback_report()
        );
        session
    });
}
