//! Standalone nxrt execution on serialized ONNX IR 13 / opset 25 models.

use onnx_runtime_ir::{
    Attribute, DataType, DeviceType, Dim, Graph, Node, NodeId, TensorData, static_shape,
};
use onnx_runtime_loader::{Model, encode_model};
use onnx_runtime_session::{DevicePreference, InferenceSession, Tensor};

fn graph() -> Graph {
    let mut graph = Graph::new();
    graph.ir_version = Some(13);
    graph.opset_imports.insert(String::new(), 25);
    graph
}

fn filled_graph(dtype: DataType, scalar: u8) -> Graph {
    let mut graph = graph();
    let shape = graph.create_named_value("shape", DataType::Int64, static_shape([1]));
    let n = graph.intern_symbol("n");
    let result = graph.create_named_value("result", dtype, vec![Dim::Symbolic(n)]);
    graph.add_input(shape);
    graph.add_output(result);
    let mut fill = Node::new(
        NodeId(0),
        "ConstantOfShape",
        vec![Some(shape)],
        vec![result],
    );
    fill.attributes.insert(
        "value".into(),
        Attribute::Tensor(TensorData::from_raw(dtype, vec![1], vec![scalar])),
    );
    graph.insert_node(fill);
    graph
}

fn session(graph: &Graph, device: DevicePreference) -> InferenceSession {
    let bytes = encode_model(&Model::new(graph)).expect("encode opset 25 model");
    InferenceSession::builder()
        .model_bytes(&bytes)
        .device(device)
        .build()
        .expect("build native opset 25 session")
}

fn assert_placement(session: &InferenceSession, device_type: DeviceType) {
    assert_eq!(session.device_id().device_type, device_type);
    assert!(session.execution_provider_fallback_report().is_none());
    assert!(session.heterogeneous_placement_report().is_none());
}

fn check_fill(device: DevicePreference, device_type: DeviceType) {
    for (dtype, scalar, expected) in [
        (
            DataType::Int2,
            0x02,
            [&[][..], &[2], &[0x0a], &[0x2a], &[0xaa], &[0xaa, 2]],
        ),
        (
            DataType::Uint2,
            0x03,
            [&[][..], &[3], &[0x0f], &[0x3f], &[0xff], &[0xff, 3]],
        ),
        (
            DataType::Float8E8M0,
            0x81,
            [
                &[][..],
                &[0x81; 1],
                &[0x81; 2],
                &[0x81; 3],
                &[0x81; 4],
                &[0x81; 5],
            ],
        ),
    ] {
        let mut session = session(&filled_graph(dtype, scalar), device.clone());
        assert_placement(&session, device_type);
        for (n, expected) in expected.into_iter().enumerate() {
            let shape = Tensor::from_raw(DataType::Int64, vec![1], &(n as i64).to_le_bytes())
                .expect("shape feed");
            let outputs = session
                .run(&[("shape", &shape)])
                .expect("run ConstantOfShape");
            assert_eq!(outputs[0].dtype, dtype);
            assert_eq!(outputs[0].shape, [n]);
            assert_eq!(outputs[0].as_bytes(), expected, "{dtype:?} fill of {n}");
        }
    }
}

#[test]
fn constant_of_shape_exact_bytes_on_cpu() {
    check_fill(DevicePreference::Cpu, DeviceType::Cpu);
}

#[test]
#[cfg(feature = "cuda")]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires a CUDA device and NVRTC"
)]
fn constant_of_shape_exact_bytes_on_strict_cuda() {
    check_fill(DevicePreference::Gpu { index: Some(0) }, DeviceType::Cuda);
}

#[test]
fn packed_identity_shape_and_size_on_cpu() {
    check_identity_shape_and_size(DevicePreference::Cpu, DeviceType::Cpu);
}

#[test]
#[cfg(feature = "cuda")]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires a CUDA device")]
fn packed_identity_shape_and_size_on_strict_cuda() {
    check_identity_shape_and_size(DevicePreference::Gpu { index: Some(0) }, DeviceType::Cuda);
}

fn check_identity_shape_and_size(device: DevicePreference, device_type: DeviceType) {
    for (dtype, bytes) in [
        (DataType::Int2, [0xe4, 0x02]),
        (DataType::Uint2, [0x1b, 0x03]),
    ] {
        let mut graph = graph();
        let n = graph.intern_symbol("n");
        let shape = vec![Dim::Static(1), Dim::Symbolic(n)];
        let x = graph.create_named_value("x", dtype, shape.clone());
        let copy = graph.create_named_value("copy", dtype, shape);
        let dims = graph.create_named_value("dims", DataType::Int64, static_shape([2]));
        let count = graph.create_named_value("count", DataType::Int64, static_shape([]));
        graph.add_input(x);
        for (id, (op, output)) in [("Identity", copy), ("Shape", dims), ("Size", count)]
            .into_iter()
            .enumerate()
        {
            graph.insert_node(Node::new(
                NodeId(id as u32),
                op,
                vec![Some(x)],
                vec![output],
            ));
            graph.add_output(output);
        }

        let mut session = session(&graph, device.clone());
        assert_placement(&session, device_type);
        let input = Tensor::from_raw(dtype, vec![1, 5], &bytes).expect("packed input");
        let outputs = session.run(&[("x", &input)]).expect("run packed model");
        assert_eq!(outputs[0].dtype, dtype);
        assert_eq!(outputs[0].shape, [1, 5]);
        assert_eq!(outputs[0].as_bytes(), bytes);
        assert_eq!(outputs[1].dtype, DataType::Int64);
        assert_eq!(outputs[1].shape, [2]);
        assert_eq!(
            outputs[1].as_bytes(),
            [1i64.to_le_bytes(), 5i64.to_le_bytes()].concat()
        );
        assert_eq!(outputs[2].dtype, DataType::Int64);
        assert!(outputs[2].shape.is_empty());
        assert_eq!(outputs[2].as_bytes(), 5i64.to_le_bytes());
    }
}
