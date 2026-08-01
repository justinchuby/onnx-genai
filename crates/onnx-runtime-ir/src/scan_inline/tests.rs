//! Unit tests for [`inline_single_trip_scan_bodies`](super::inline_single_trip_scan_bodies).
//!
//! These are fast, CPU-only, non-ignored tests. They exercise the transform on
//! tiny hand-built graphs and prove three things (Inc-1b design §4 / PR-1 test
//! plan):
//!
//! 1. **Positive:** a single-trip-eligible hybrid `Scan` is lowered to a
//!    straight-line body — Scan removed, body nodes present with remapped fresh
//!    parent ids, state + scan in/out rewired, captures resolved by name, body
//!    initializers promoted, the graph stays structurally valid, and every
//!    rewired boundary/interior value carries the expected static shape.
//! 2. **Negative / no-op:** a non-recurrent `Scan`, a genuinely multi-trip
//!    `Scan`, and a dense `Scan`-free graph are all returned unchanged.
//! 3. **Remap correctness:** no body `ValueId` leaks into the parent (every
//!    interior value is a fresh parent id) and there are no id collisions.
//!
//! The companion assertion that *Permissive shape inference re-converges* over
//! the inlined graph lives in `onnx-runtime-shape-inference`'s integration tests
//! (`tests/scan_inline_inference.rs`): `onnx-runtime-ir` is the dependency-free
//! base contract and cannot depend on the shape-inference crate (which depends
//! on it) even as a dev-dependency without pulling in a second, incompatible
//! copy of the IR types, so that one cross-crate check is placed on the side of
//! the edge that can see both.

use std::collections::HashSet;

use super::inline_single_trip_scan_bodies;
use crate::{
    Attribute, DataType, Dim, Graph, Node, NodeId, TensorData, ValueId, WeightRef, static_shape,
};

// ---------------------------------------------------------------------------
// Tiny-graph builders (DRY: one hybrid builder, parameterized by scan-axis
// extent, plus small helpers for the negative cases).
// ---------------------------------------------------------------------------

/// The named handles a test needs from a built hybrid graph.
struct Hybrid {
    graph: Graph,
    scan: NodeId,
    x: ValueId,
    past_state: ValueId,
    present_state: ValueId,
    scan_out: ValueId,
    weight: ValueId,
}

/// A zero-filled inline f32 initializer of the given dims.
fn zero_weight(dims: &[usize]) -> WeightRef {
    let numel: usize = dims.iter().product();
    WeightRef::Inline(TensorData::from_raw(
        DataType::Float32,
        dims.to_vec(),
        vec![0u8; numel * 4],
    ))
}

/// Build the straight-line LinearAttention-like body:
///
/// ```text
///   state_in:[2,4]  scan_in:[2,4]     (formals)
///   w:[4,4]                           (lexical capture, by name)
///   bias:[4]                          (body-local initializer)
///
///   t0      = MatMul(scan_in, w)      -> [2,4]
///   t1      = Add(t0, state_in)       -> [2,4]     (recurrent accumulate)
///   present = Add(t1, bias)           -> [2,4]     (state output)
///   y       = Add(t1, scan_in)        -> [2,4]     (scan output, per iteration)
/// ```
///
/// Body outputs are `[present (state), y (scan)]`.
fn build_body() -> Graph {
    let mut b = Graph::new();

    let state_in = b.create_named_value("state_in", DataType::Float32, static_shape([2, 4]));
    let scan_in = b.create_named_value("scan_in", DataType::Float32, static_shape([2, 4]));
    b.add_input(state_in);
    b.add_input(scan_in);

    // Lexical capture: producer-less, named, not a formal, not an initializer.
    let w = b.create_named_value("w", DataType::Float32, static_shape([4, 4]));

    // Body-local initializer.
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

/// Build a hybrid parent graph whose recurrent `Scan` has scan-axis extent
/// `scan_extent` (1 = decode/single-trip-eligible, >1 = multi-trip).
fn build_hybrid(scan_extent: usize) -> Hybrid {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 18);

    // Parent scan input carries the scan axis (axis 0) with the given extent.
    let x = g.create_named_value("x", DataType::Float32, static_shape([scan_extent, 2, 4]));
    g.add_input(x);

    // Recurrent state input.
    let past_state = g.create_named_value("past_state", DataType::Float32, static_shape([2, 4]));
    g.add_input(past_state);

    // The weight the body captures by name — a parent initializer.
    let weight = g.create_named_value("w", DataType::Float32, static_shape([4, 4]));
    g.set_initializer(weight, zero_weight(&[4, 4]));

    // Scan outputs, declared with the scan axis re-added on the scan output.
    let present_state =
        g.create_named_value("present_state", DataType::Float32, static_shape([2, 4]));
    let scan_out = g.create_named_value(
        "scan_out",
        DataType::Float32,
        static_shape([scan_extent, 2, 4]),
    );

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

    Hybrid {
        graph: g,
        scan,
        x,
        past_state,
        present_state,
        scan_out,
        weight,
    }
}

fn op_counts(graph: &Graph) -> std::collections::HashMap<String, usize> {
    let mut counts = std::collections::HashMap::new();
    for (_, node) in graph.nodes.iter() {
        *counts.entry(node.op_type.clone()).or_insert(0) += 1;
    }
    counts
}

fn max_value_raw(graph: &Graph) -> u32 {
    graph.values.keys().map(|v| v.0).max().unwrap_or(0)
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

// ---------------------------------------------------------------------------
// 1. Positive: the eligible single-trip Scan is lowered correctly.
// ---------------------------------------------------------------------------

#[test]
fn lowers_single_trip_scan_to_straight_line_body() {
    let h = build_hybrid(1);
    let before = op_counts(&h.graph);
    assert_eq!(before.get("Scan"), Some(&1));
    assert_eq!(
        before.get("MatMul"),
        None,
        "body op must not be in parent yet"
    );

    let out = inline_single_trip_scan_bodies(&h.graph);

    // Input graph is never mutated (pure Graph -> Graph).
    assert_eq!(op_counts(&h.graph).get("Scan"), Some(&1));

    // Scan node removed and its body subgraph dropped.
    assert!(!out.nodes.contains(h.scan), "Scan node must be removed");
    assert!(
        !out.subgraphs.contains_key(&(h.scan, "body".to_string())),
        "Scan body subgraph must be dropped"
    );
    let after = op_counts(&out);
    assert_eq!(after.get("Scan"), None);

    // Body nodes are now present in the parent: MatMul + 3 Adds.
    assert_eq!(after.get("MatMul"), Some(&1));
    assert_eq!(after.get("Add"), Some(&3));
    // Scan-axis marshaling: one Squeeze (scan input) + one Unsqueeze (scan out).
    assert_eq!(after.get("Squeeze"), Some(&1));
    assert_eq!(after.get("Unsqueeze"), Some(&1));

    // State output is written directly by a body node (no Identity copy needed).
    assert_eq!(after.get("Identity"), None);

    // State/scan in/out are rewired: the Squeeze consumes the parent scan input,
    // the Unsqueeze produces the parent scan output, and the parent present
    // state is produced by an inlined Add.
    let squeeze = out
        .nodes
        .iter()
        .find(|(_, n)| n.op_type == "Squeeze")
        .map(|(_, n)| n.clone())
        .expect("Squeeze present");
    assert_eq!(squeeze.inputs, vec![Some(h.x)]);

    let present_producer = out
        .value(h.present_state)
        .producer
        .expect("present state must be produced after inlining");
    assert_eq!(out.node(present_producer).op_type, "Add");

    let scan_out_producer = out
        .value(h.scan_out)
        .producer
        .expect("scan output must be produced after inlining");
    assert_eq!(out.node(scan_out_producer).op_type, "Unsqueeze");

    // The lexical capture `w` resolved by name to the pre-existing parent
    // initializer — the inlined MatMul reads that very value.
    let matmul = out
        .nodes
        .iter()
        .find(|(_, n)| n.op_type == "MatMul")
        .map(|(_, n)| n.clone())
        .expect("MatMul present");
    assert!(
        matmul.input_values().any(|v| v == h.weight),
        "inlined MatMul must consume the by-name-resolved parent weight value"
    );
    assert!(
        out.initializers.contains_key(&h.weight),
        "the captured weight stays a parent initializer"
    );

    // Body-local initializer `bias` was promoted into the parent.
    let promoted_bias = out
        .initializers
        .keys()
        .copied()
        .filter(|&v| v != h.weight)
        .find(|&v| out.try_value(v).and_then(|val| val.name.clone()).as_deref() == Some("bias"));
    assert!(
        promoted_bias.is_some(),
        "body-local `bias` initializer must be promoted to a parent initializer"
    );

    // Structural validity of the inlined graph.
    out.validate()
        .expect("inlined graph must be structurally valid");

    // The rewired boundary and interior values carry the expected static shapes
    // (the transform preserves body-declared shapes and computes the
    // squeezed/unsqueezed ranks itself). Permissive shape inference re-deriving
    // these from scratch is asserted in the shape-inference integration test.
    assert_eq!(static_dims(&out, h.present_state), Some(vec![2, 4]));
    assert_eq!(static_dims(&out, h.scan_out), Some(vec![1, 2, 4]));
    // The squeezed slice drops the size-1 scan axis to the body rank.
    assert_eq!(static_dims(&out, squeeze.outputs[0]), Some(vec![2, 4]));
}

// ---------------------------------------------------------------------------
// 2. Negative / no-op cases.
// ---------------------------------------------------------------------------

/// Assert two graphs are structurally identical for the properties the
/// transform is supposed to preserve on a no-op.
fn assert_unchanged(before: &Graph, after: &Graph) {
    assert_eq!(before.nodes.len(), after.nodes.len(), "node count changed");
    assert_eq!(
        before.values.len(),
        after.values.len(),
        "value count changed"
    );
    assert_eq!(op_counts(before), op_counts(after), "op mix changed");
    let before_keys: HashSet<_> = before.subgraphs.keys().cloned().collect();
    let after_keys: HashSet<_> = after.subgraphs.keys().cloned().collect();
    assert_eq!(before_keys, after_keys, "subgraphs changed");
}

#[test]
fn leaves_multi_trip_scan_untouched() {
    // Recurrent, but the scan axis is a concrete extent 3 (a real multi-trip /
    // prefill Scan) — must NOT be statically collapsed to one iteration.
    let h = build_hybrid(3);
    let out = inline_single_trip_scan_bodies(&h.graph);
    assert!(out.nodes.contains(h.scan), "multi-trip Scan must survive");
    assert!(out.subgraphs.contains_key(&(h.scan, "body".to_string())));
    assert_unchanged(&h.graph, &out);
}

#[test]
fn leaves_non_recurrent_scan_untouched() {
    // A pure element-wise map Scan: no state (num_state == 0). Not eligible.
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 18);
    let x = g.create_named_value("x", DataType::Float32, static_shape([1, 4]));
    g.add_input(x);
    let y = g.create_named_value("y", DataType::Float32, static_shape([1, 4]));

    let mut scan = Node::new(NodeId(0), "Scan", vec![Some(x)], vec![y]);
    scan.attributes
        .insert("num_scan_inputs".to_string(), Attribute::Int(1));
    let scan = g.insert_node(scan);

    let mut body = Graph::new();
    let scan_in = body.create_named_value("scan_in", DataType::Float32, static_shape([4]));
    body.add_input(scan_in);
    let out_v = body.create_named_value("out", DataType::Float32, static_shape([4]));
    body.insert_node(Node::new(
        NodeId(0),
        "Relu",
        vec![Some(scan_in)],
        vec![out_v],
    ));
    body.add_output(out_v);
    g.subgraphs.insert((scan, "body".to_string()), body);
    g.add_output(y);

    let out = inline_single_trip_scan_bodies(&g);
    assert!(out.nodes.contains(scan), "non-recurrent Scan must survive");
    assert_unchanged(&g, &out);
}

#[test]
fn leaves_dense_scanless_graph_untouched() {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 18);
    let x = g.create_named_value("x", DataType::Float32, static_shape([2, 4]));
    g.add_input(x);
    let y = g.create_named_value("y", DataType::Float32, static_shape([2, 4]));
    g.insert_node(Node::new(NodeId(0), "Relu", vec![Some(x)], vec![y]));
    let z = g.create_named_value("z", DataType::Float32, static_shape([2, 4]));
    g.insert_node(Node::new(NodeId(0), "Sigmoid", vec![Some(y)], vec![z]));
    g.add_output(z);

    let out = inline_single_trip_scan_bodies(&g);
    assert_unchanged(&g, &out);
    // And the no-op result is still valid.
    out.validate().expect("dense graph stays valid");
}

// ---------------------------------------------------------------------------
// 3. Remap correctness: no body ValueId leaks; no id collisions.
// ---------------------------------------------------------------------------

#[test]
fn remap_produces_fresh_non_colliding_interior_ids() {
    let h = build_hybrid(1);

    // The body's own ValueId namespace overlaps numerically with the parent's
    // (both arenas start near 0), so a correct remap MUST allocate brand-new
    // parent ids for every interior body value rather than reuse a body id.
    let body = h
        .graph
        .subgraphs
        .get(&(h.scan, "body".to_string()))
        .unwrap();
    let body_ids: HashSet<u32> = body.values.keys().map(|v| v.0).collect();
    let parent_ids_before: HashSet<u32> = h.graph.values.keys().map(|v| v.0).collect();
    // Precondition: the namespaces genuinely overlap, so this test is not vacuous.
    assert!(
        body_ids.iter().any(|id| parent_ids_before.contains(id)),
        "test setup should have overlapping body/parent id ranges"
    );
    let max_before = max_value_raw(&h.graph);

    let out = inline_single_trip_scan_bodies(&h.graph);

    // Every value referenced by every node is live in the parent arena — this
    // alone rules out any leaked (dangling) body ValueId.
    for (_, node) in out.nodes.iter() {
        for v in node.input_values() {
            assert!(out.values.contains(v), "node input {v:?} must be live");
        }
        for &v in &node.outputs {
            assert!(out.values.contains(v), "node output {v:?} must be live");
        }
    }

    // Interior values created for the inlined body (everything except the
    // pre-existing parent values) are strictly-fresh ids allocated above the
    // pre-transform high-water mark — so no body id was smuggled in and no id
    // aliases a pre-existing parent value.
    let boundary: HashSet<ValueId> = [h.x, h.past_state, h.present_state, h.scan_out, h.weight]
        .into_iter()
        .collect();
    let interior_producers = ["Squeeze", "MatMul", "Add", "Unsqueeze"];
    let mut interior_seen = 0;
    for (_, node) in out.nodes.iter() {
        if !interior_producers.contains(&node.op_type.as_str()) {
            continue;
        }
        for &v in &node.outputs {
            if boundary.contains(&v) {
                continue; // a rewired Scan boundary output, legitimately reused
            }
            assert!(
                v.0 > max_before,
                "interior value {v:?} must be a freshly allocated id (> {max_before})"
            );
            interior_seen += 1;
        }
    }
    assert!(
        interior_seen >= 3,
        "expected several fresh interior values (squeezed slice + MatMul + Adds)"
    );

    // validate() proves there are no id collisions: every value has exactly one
    // producer and consistent producer/consumer links.
    out.validate().expect("no id collisions / consistent edges");
}
