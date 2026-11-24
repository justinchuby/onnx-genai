//! Native shape-only transport at the ONNX low-bit dtype version boundaries.

use onnx_runtime_ir::{
    DataType, DeviceId, Graph, Node, NodeId, TensorData, WeightRef, static_shape,
};
use onnx_runtime_session::{InferenceSession, Tensor};

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
    for (dtype, since, bytes) in TYPES {
        for case @ (op, in_shape, out_shape, _) in CASES.into_iter().chain(EMPTY_CASES) {
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
}
