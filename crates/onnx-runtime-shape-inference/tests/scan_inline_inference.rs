//! Cross-crate check for the Inc-1b PR-1 graph transform
//! [`onnx_runtime_ir::inline_single_trip_scan_bodies`].
//!
//! The transform itself, and its structural unit tests, live in
//! `onnx-runtime-ir`. That crate is the dependency-free base contract and cannot
//! depend on this shape-inference crate (which depends on it), so the one check
//! the design (`cohaagen-27b-inc1b-design.md` §4) states in
//! shape-inference terms — *Permissive whole-graph inference re-converges over
//! the inlined body, re-resolving the interior shapes exactly as
//! `ChildExecutor::compile` does at runtime* — is placed here, on the side of
//! the crate edge that can see both the transform and the inference registry.
//!
//! This test is CPU-only, fast, and non-ignored. It does not wire the transform
//! into any run path; it only feeds a hand-built graph through it.

use onnx_runtime_ir::{
    Attribute, DataType, Dim, Graph, Node, NodeId, TensorData, ValueId, WeightRef,
    inline_single_trip_scan_bodies, static_shape,
};
use onnx_runtime_shape_inference::{InferenceRegistry, MergePolicy};

fn zero_weight(dims: &[usize]) -> WeightRef {
    let numel: usize = dims.iter().product();
    WeightRef::Inline(TensorData::from_raw(
        DataType::Float32,
        dims.to_vec(),
        vec![0u8; numel * 4],
    ))
}

/// Straight-line LinearAttention-like body with a lexical capture (`w`), a
/// body-local initializer (`bias`), a recurrent state pair, and a scan output.
fn build_body() -> Graph {
    let mut b = Graph::new();
    let state_in = b.create_named_value("state_in", DataType::Float32, static_shape([2, 4]));
    let scan_in = b.create_named_value("scan_in", DataType::Float32, static_shape([2, 4]));
    b.add_input(state_in);
    b.add_input(scan_in);

    let w = b.create_named_value("w", DataType::Float32, static_shape([4, 4]));
    let bias = b.create_named_value("bias", DataType::Float32, static_shape([4]));
    b.set_initializer(bias, zero_weight(&[4]));

    let t0 = b.create_value(DataType::Float32, static_shape([2, 4]));
    b.insert_node(Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(scan_in), Some(w)],
        vec![t0],
    ));
    let t1 = b.create_value(DataType::Float32, static_shape([2, 4]));
    b.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(t0), Some(state_in)],
        vec![t1],
    ));
    let present = b.create_named_value("present", DataType::Float32, static_shape([2, 4]));
    b.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(t1), Some(bias)],
        vec![present],
    ));
    let y = b.create_named_value("y", DataType::Float32, static_shape([2, 4]));
    b.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(t1), Some(scan_in)],
        vec![y],
    ));
    b.add_output(present);
    b.add_output(y);
    b
}

/// Single-trip-eligible hybrid parent graph; returns the graph plus the
/// present-state and scan-output value ids (stable across the transform).
fn build_hybrid() -> (Graph, ValueId, ValueId) {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 18);

    let x = g.create_named_value("x", DataType::Float32, static_shape([1, 2, 4]));
    g.add_input(x);
    let past_state = g.create_named_value("past_state", DataType::Float32, static_shape([2, 4]));
    g.add_input(past_state);

    let weight = g.create_named_value("w", DataType::Float32, static_shape([4, 4]));
    g.set_initializer(weight, zero_weight(&[4, 4]));

    let present_state =
        g.create_named_value("present_state", DataType::Float32, static_shape([2, 4]));
    let scan_out = g.create_named_value("scan_out", DataType::Float32, static_shape([1, 2, 4]));

    let mut scan = Node::new(
        NodeId(0),
        "Scan",
        vec![Some(past_state), Some(x)],
        vec![present_state, scan_out],
    );
    scan.attributes
        .insert("num_scan_inputs".to_string(), Attribute::Int(1));
    scan.attributes
        .insert("scan_input_axes".to_string(), Attribute::Ints(vec![0]));
    scan.attributes
        .insert("scan_output_axes".to_string(), Attribute::Ints(vec![0]));
    let scan = g.insert_node(scan);
    g.subgraphs.insert((scan, "body".to_string()), build_body());

    g.add_output(present_state);
    g.add_output(scan_out);
    (g, present_state, scan_out)
}

fn static_dims(graph: &Graph, value: ValueId) -> Option<Vec<usize>> {
    graph
        .try_value(value)?
        .shape
        .iter()
        .map(|d| match d {
            Dim::Static(n) => Some(*n),
            Dim::Symbolic(_) => None,
        })
        .collect()
}

#[test]
fn permissive_inference_reconverges_over_inlined_scan_body() {
    let (graph, present_state, scan_out) = build_hybrid();
    let mut inlined = inline_single_trip_scan_bodies(&graph);

    // Sanity: the transform actually fired (Scan lowered to straight-line ops).
    assert!(
        inlined.nodes.iter().all(|(_, n)| n.op_type != "Scan"),
        "the eligible Scan should have been inlined away"
    );
    inlined
        .validate()
        .expect("inlined graph must be structurally valid");

    // Force a genuine re-derivation: blank every produced value's shape so the
    // registry must reconstruct the interior + boundary shapes from the graph
    // inputs and initializers through the inlined nodes. Only sources (graph
    // inputs / initializers) keep their seed shapes.
    let produced: Vec<ValueId> = inlined
        .values
        .iter()
        .filter(|(_, v)| v.producer.is_some())
        .map(|(vid, _)| vid)
        .collect();
    for vid in produced {
        inlined.mark_value_shape_unknown(vid);
    }

    let registry = InferenceRegistry::default_registry();
    let opsets = inlined.opset_imports.clone();
    registry
        .infer_graph(&mut inlined, &opsets, MergePolicy::Permissive)
        .expect("Permissive shape inference must converge over the inlined graph");

    // Interior shapes re-resolved to the single-trip body ranks, and the Scan's
    // declared boundary shapes were reconstructed: present state at body rank,
    // scan output with the size-1 scan axis re-added by the Unsqueeze.
    assert_eq!(static_dims(&inlined, present_state), Some(vec![2, 4]));
    assert_eq!(static_dims(&inlined, scan_out), Some(vec![1, 2, 4]));
}
