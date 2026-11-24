use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, affine_gather, static_shape};
use onnx_runtime_session::{InferenceSession, Tensor};

fn graph(expose_decoded: bool) -> Graph {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 29);
    let w = g.create_named_value("W", DataType::Int8, static_shape([3, 4]));
    let idx = g.create_named_value("indices", DataType::Int64, static_shape([2]));
    let s = g.create_named_value("scale", DataType::Float32, vec![]);
    let z = g.create_named_value("zero", DataType::Int8, vec![]);
    let decoded = g.create_named_value("decoded", DataType::Float32, static_shape([3, 4]));
    let y = g.create_named_value("Y", DataType::Float32, static_shape([3, 2]));
    for v in [w, idx, s, z] {
        g.add_input(v);
    }
    g.insert_node(Node::new(
        NodeId(0),
        "DequantizeLinear",
        vec![Some(w), Some(s), Some(z)],
        vec![decoded],
    ));
    let mut gather = Node::new(NodeId(0), "Gather", vec![Some(decoded), Some(idx)], vec![y]);
    gather.attributes.insert("axis".into(), Attribute::Int(-1));
    g.insert_node(gather);
    g.add_output(y);
    if expose_decoded {
        g.add_output(decoded);
    }
    g
}

fn verify(mut fused: InferenceSession, mut reference: InferenceSession) {
    let ops = |s: &InferenceSession, name: &str| {
        s.graph()
            .nodes
            .values()
            .filter(|n| n.op_type == name)
            .count()
    };
    assert_eq!(ops(&fused, "AffineQuantizedGather"), 1);
    assert_eq!(ops(&fused, "DequantizeLinear"), 0);
    assert_eq!(ops(&reference, affine_gather::OP), 0);
    assert_eq!(ops(&reference, "DequantizeLinear"), 1);
    let w = Tensor::from_raw(
        DataType::Int8,
        vec![3, 4],
        &[0, 1, 2, 128, 4, 5, 6, 127, 8, 9, 10, 255],
    )
    .unwrap();
    let s = Tensor::from_f32(&[], &[0.3]).unwrap();
    let z = Tensor::from_raw(DataType::Int8, vec![], &[253]).unwrap();
    let idx = Tensor::from_i64(&[2], &[-1, 0]).unwrap();
    let inputs = [("W", &w), ("indices", &idx), ("scale", &s), ("zero", &z)];
    let got = fused.run(&inputs).unwrap();
    let expected = reference.run(&inputs).unwrap();
    assert_eq!(got[0].to_vec_f32(), expected[0].to_vec_f32());
    // Independent check, not just agreement between two kernels on one EP.
    let codes = [-128, 0, 127, 4, -1, 8];
    assert_eq!(got[0].to_vec_f32(), codes.map(|q| (q + 3) as f32 * 0.3));

    let bad = Tensor::from_i64(&[2], &[0, -5]).unwrap();
    let inputs = [("W", &w), ("indices", &bad), ("scale", &s), ("zero", &z)];
    let expected_err = reference.run(&inputs).unwrap_err().to_string();
    let got_err = fused.run(&inputs).unwrap_err().to_string();
    let validation = ["out of range", "device validation failed (flags=0x40)"];
    assert!(
        validation
            .iter()
            .any(|message| { expected_err.contains(message) && got_err.contains(message) }),
        "unfused: {expected_err}; fused: {got_err}"
    );
}

#[test]
fn affine_gather_cpu_plan_and_parity() {
    verify(
        InferenceSession::from_graph(graph(false)).unwrap(),
        InferenceSession::from_graph(graph(true)).unwrap(),
    );
}

#[test]
fn affine_gather_preserves_cpu_scale_validation() {
    let w = Tensor::from_raw(DataType::Int8, vec![3, 4], &[0; 12]).unwrap();
    let z = Tensor::from_raw(DataType::Int8, vec![], &[0]).unwrap();
    let idx = Tensor::from_i64(&[2], &[0, -1]).unwrap();
    let s = Tensor::from_f32(&[], &[0.0]).unwrap();
    let inputs = [("W", &w), ("indices", &idx), ("scale", &s), ("zero", &z)];
    for shared in [false, true] {
        let err = InferenceSession::from_graph(graph(shared))
            .unwrap()
            .run(&inputs)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("scale values must be finite and positive"),
            "{err}"
        );
    }
}

#[cfg(feature = "gpu-tests")]
#[test]
fn affine_gather_cuda_plan_and_parity() {
    use onnx_runtime_ep_cuda::CudaExecutionProvider;
    use std::{path::Path, sync::Arc};

    let build = |shared| {
        let provider = Arc::new(CudaExecutionProvider::initialized(0).expect("initialize CUDA"));
        let session = InferenceSession::from_graph_with_provider(
            graph(shared),
            Arc::new(onnx_runtime_loader::WeightStore::new()),
            Path::new("."),
            provider,
            None,
        )
        .unwrap();
        assert!(session.execution_provider_fallback_report().is_none());
        session
    };
    verify(build(false), build(true));
}
