//! Native shape-only transport at the ONNX low-bit dtype version boundaries.

use onnx_runtime_ir::{
    DataType, DeviceId, Dim, Graph, Node, NodeId, TensorData, WeightRef, static_shape,
};
use onnx_runtime_session::{InferenceSession, Tensor};

pub(super) const IR9_FP8_TYPES: [DataType; 4] = [
    DataType::Float8E4M3FN,
    DataType::Float8E4M3FNUZ,
    DataType::Float8E5M2,
    DataType::Float8E5M2FNUZ,
];

pub(super) fn dynamic_reshape_graph(opset: u64, dtype: DataType) -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), opset);
    let data = graph.create_named_value("data", dtype, static_shape([2, 3]));
    let target = graph.create_named_value("target", DataType::Int64, static_shape([2]));
    graph.add_input(data);
    graph.add_input(target);
    let rows = graph.intern_symbol("rows");
    let cols = graph.intern_symbol("cols");
    let output = graph.create_named_value(
        "output",
        dtype,
        vec![Dim::Symbolic(rows), Dim::Symbolic(cols)],
    );
    graph.insert_node(Node::new(
        NodeId(0),
        "Reshape",
        vec![Some(data), Some(target)],
        vec![output],
    ));
    graph.add_output(output);
    graph
}

pub(super) fn assert_ir9_fp8_reshape(
    mut build: impl FnMut(Graph) -> InferenceSession,
    device: DeviceId,
) {
    // Include sign and special-value encodings: no numeric conversion is allowed.
    let bytes = [0x00, 0x80, 0x7f, 0xff, 0x01, 0x7c];
    for dtype in IR9_FP8_TYPES {
        let mut session = build(dynamic_reshape_graph(19, dtype));
        assert_eq!(session.device_id(), device, "Reshape-19 {dtype:?}");
        assert!(session.execution_provider_fallback_report().is_none());
        assert_eq!(
            session.graph().nodes.values().next().unwrap().op_type,
            "Reshape"
        );
        let data = Tensor::from_raw(dtype, vec![2, 3], &bytes).unwrap();
        // Change target contents while keeping both input shapes unchanged.
        for (target, shape) in [([-1, 2], [3, 2]), ([1, -1], [1, 6]), ([-1, 2], [3, 2])] {
            let target = Tensor::from_i64(&[2], &target).unwrap();
            let outputs = session
                .run(&[("data", &data), ("target", &target)])
                .unwrap_or_else(|err| panic!("Reshape-19 {dtype:?} to {shape:?}: {err}"));
            assert_eq!(outputs[0].dtype, dtype);
            assert_eq!(outputs[0].shape, shape, "Reshape-19 {dtype:?}");
            assert_eq!(
                outputs[0].as_bytes(),
                bytes,
                "Reshape-19 {dtype:?} {shape:?}"
            );
        }
    }
}

pub(super) const TYPES: [(DataType, u64, &[u8]); 6] = [
    (DataType::Int4, 21, &[0x81, 0x7f, 0xb2]),
    (DataType::Uint4, 21, &[0x91, 0x6f, 0xa2]),
    (DataType::Float4E2M1, 23, &[0x12, 0xf8, 0xe7]),
    (DataType::Float8E8M0, 24, &[0x00, 0x01, 0x7f, 0xfe, 0xff]),
    (DataType::Int2, 25, &[0x9c, 0xfa]),
    (DataType::Uint2, 25, &[0xe4, 0xab]),
];

pub(super) type ShapeCase = (
    &'static str,
    &'static [usize],
    &'static [usize],
    &'static [i64],
);
pub(super) const CASES: [ShapeCase; 4] = [
    ("Reshape", &[1, 5], &[5, 1], &[-1, 1]),
    ("Flatten", &[5, 1, 1], &[5, 1], &[]),
    ("Squeeze", &[1, 5, 1], &[5], &[0, -1]),
    ("Unsqueeze", &[5], &[1, 5, 1], &[0, -1]),
];
const EMPTY_CASES: [ShapeCase; 4] = [
    ("Reshape", &[0, 5], &[0], &[0]),
    ("Flatten", &[5, 0, 1], &[5, 0], &[]),
    ("Squeeze", &[1, 0, 1], &[0], &[0, -1]),
    ("Unsqueeze", &[0], &[1, 0, 1], &[0, -1]),
];

pub(super) fn graph(
    opset: u64,
    dtype: DataType,
    (op, in_shape, out_shape, meta): ShapeCase,
) -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), opset);
    let data = graph.create_named_value("data", dtype, static_shape(in_shape.iter().copied()));
    graph.add_input(data);
    let mut inputs = vec![Some(data)];
    if !meta.is_empty() {
        let value = graph.create_named_value("meta", DataType::Int64, static_shape([meta.len()]));
        graph.set_initializer(
            value,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Int64,
                vec![meta.len()],
                meta.iter().flat_map(|x| x.to_le_bytes()).collect(),
            )),
        );
        inputs.push(Some(value));
    }
    let output = graph.create_named_value("output", dtype, static_shape(out_shape.iter().copied()));
    graph.insert_node(Node::new(NodeId(0), op, inputs, vec![output]));
    graph.add_output(output);
    graph
}

pub(super) fn assert_transport(mut build: impl FnMut(Graph) -> InferenceSession, device: DeviceId) {
    // Five logical elements leave unused high bits for both packed widths.
    // Keep these bits nonzero to distinguish storage copies from repacking.
    let packed = TYPES.into_iter().flat_map(|(dtype, since, bytes)| {
        CASES
            .into_iter()
            .chain(EMPTY_CASES)
            .map(move |case| (dtype, since, bytes, case))
    });
    let peers = IR9_FP8_TYPES.into_iter().flat_map(|dtype| {
        CASES
            .into_iter()
            .filter(|case| case.0 != "Reshape")
            .map(move |case| (dtype, 21, &[0x00, 0x80, 0x7f, 0xff, 0x01][..], case))
    });
    for (dtype, since, bytes, case @ (op, in_shape, out_shape, _)) in packed.chain(peers) {
        let label = format!("{op} {dtype:?} at opset {since}: {in_shape:?} -> {out_shape:?}");
        let mut session = build(graph(since, dtype, case));
        assert_eq!(session.device_id(), device, "{label}");
        assert!(
            session.execution_provider_fallback_report().is_none(),
            "{label}"
        );
        assert!(
            session
                .graph()
                .nodes
                .values()
                .any(|node| node.op_type == op),
            "{label}: test must execute the named operator"
        );
        let bytes = if in_shape.contains(&0) { &[] } else { bytes };
        let input = Tensor::from_raw(dtype, in_shape.to_vec(), bytes).unwrap();
        let output = session
            .run(&[("data", &input)])
            .unwrap_or_else(|err| panic!("{label}: {err}"));
        assert_eq!(output[0].dtype, dtype, "{label}");
        assert_eq!(output[0].shape, out_shape, "{label}");
        assert_eq!(output[0].as_bytes(), bytes, "{label}");
    }
}
