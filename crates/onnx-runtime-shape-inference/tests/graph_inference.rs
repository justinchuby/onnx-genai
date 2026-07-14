//! Whole-graph inference tests: a hand-built symbolic transformer fragment and
//! an end-to-end run over the committed `bert_toy` model.

use onnx_runtime_ir::{
    Attribute, DataType, Dim, Graph, Node, NodeId, Shape, TensorData, ValueId, WeightRef,
};
use onnx_runtime_shape_inference::{InferenceRegistry, MergePolicy};

/// Encode i64 values as little-endian bytes for an inline initializer.
fn i64_bytes(vals: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 8);
    for v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn node(id: u32, op: &str, inputs: Vec<Option<ValueId>>, outputs: Vec<ValueId>) -> Node {
    Node::new(NodeId(id), op, inputs, outputs)
}

/// Build a small graph exercising symbolic-batch propagation through
/// MatMul → Add → Reshape, and assert the named batch dim `N` survives.
#[test]
fn symbolic_batch_survives_matmul_add_reshape() {
    let mut g = Graph::new();
    let n_sym = g.intern_symbol("N");

    // x: [N, 8, 768]
    let x = g.create_named_value(
        "x",
        DataType::Float32,
        vec![Dim::Symbolic(n_sym), Dim::Static(8), Dim::Static(768)],
    );
    g.add_input(x);

    // W: [768, 768] initializer (float; shape only matters).
    let w = g.create_named_value(
        "W",
        DataType::Float32,
        vec![Dim::Static(768), Dim::Static(768)],
    );
    g.set_initializer(
        w,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            vec![768, 768],
            vec![0u8; 768 * 768 * 4],
        )),
    );

    // bias: [768] initializer.
    let bias = g.create_named_value("bias", DataType::Float32, vec![Dim::Static(768)]);
    g.set_initializer(
        bias,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            vec![768],
            vec![0u8; 768 * 4],
        )),
    );

    // reshape target [0, 0, 12, -1] as an int64 initializer -> shape-data source.
    let target = g.create_named_value("target", DataType::Int64, vec![Dim::Static(4)]);
    g.set_initializer(
        target,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            vec![4],
            i64_bytes(&[0, 0, 12, -1]),
        )),
    );

    // Interior values (shapes intentionally left blank; inference fills them).
    let m = g.create_named_value("m", DataType::Float32, Shape::new());
    let a = g.create_named_value("a", DataType::Float32, Shape::new());
    let r = g.create_named_value("r", DataType::Float32, Shape::new());

    g.insert_node(node(1, "MatMul", vec![Some(x), Some(w)], vec![m]));
    g.insert_node(node(2, "Add", vec![Some(m), Some(bias)], vec![a]));
    g.insert_node(node(3, "Reshape", vec![Some(a), Some(target)], vec![r]));
    g.add_output(r);
    g.opset_imports.insert(String::new(), 13);

    let reg = InferenceRegistry::default_registry();
    let opsets = g.opset_imports.clone();
    let report = reg
        .infer_graph(&mut g, &opsets, MergePolicy::Permissive)
        .unwrap();
    assert!(
        report.fully_resolved(),
        "unresolved: {:?}",
        report.unresolved
    );

    // m = [N, 8, 768]; a = [N, 8, 768]; r = [N, 8, 12, 64] with N symbolic.
    let m_shape = g.value(m).shape.clone();
    assert!(
        matches!(m_shape[0], Dim::Symbolic(_)),
        "batch stayed symbolic in MatMul"
    );
    assert_eq!(m_shape[1], Dim::Static(8));
    assert_eq!(m_shape[2], Dim::Static(768));

    let r_shape = g.value(r).shape.clone();
    assert_eq!(r_shape.len(), 4);
    assert!(
        matches!(r_shape[0], Dim::Symbolic(_)),
        "batch stayed symbolic through Reshape"
    );
    assert_eq!(r_shape[1], Dim::Static(8));
    assert_eq!(r_shape[2], Dim::Static(12));
    assert_eq!(
        r_shape[3],
        Dim::Static(64),
        "-1 resolved to 64 by symbol cancellation"
    );

    // The batch symbol must be the SAME one throughout (not a fresh clone).
    let (Dim::Symbolic(mb), Dim::Symbolic(rb)) = (m_shape[0], r_shape[0]) else {
        panic!("expected symbolic batch dims");
    };
    assert_eq!(mb, n_sym);
    assert_eq!(rb, n_sym);
}

/// Shape → Gather → Unsqueeze → Concat → Reshape chain: a reshape target
/// assembled from a `Shape` op must resolve symbolically.
#[test]
fn shape_data_chain_drives_reshape() {
    let mut g = Graph::new();
    let n_sym = g.intern_symbol("N");

    // x: [N, 8, 64]
    let x = g.create_named_value(
        "x",
        DataType::Float32,
        vec![Dim::Symbolic(n_sym), Dim::Static(8), Dim::Static(64)],
    );
    g.add_input(x);

    // idx0 = [0] initializer for Gather.
    let idx = g.create_named_value("idx", DataType::Int64, vec![Dim::Static(1)]);
    g.set_initializer(
        idx,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            vec![1],
            i64_bytes(&[0]),
        )),
    );

    // tail = [512] initializer (8*64 flattened) to concat after the batch dim.
    let tail = g.create_named_value("tail", DataType::Int64, vec![Dim::Static(1)]);
    g.set_initializer(
        tail,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            vec![1],
            i64_bytes(&[512]),
        )),
    );

    let shp = g.create_named_value("shp", DataType::Int64, Shape::new());
    let gathered = g.create_named_value("gathered", DataType::Int64, Shape::new());
    let target = g.create_named_value("target", DataType::Int64, Shape::new());
    let out = g.create_named_value("out", DataType::Float32, Shape::new());

    g.insert_node(node(1, "Shape", vec![Some(x)], vec![shp]));
    let mut gnode = node(2, "Gather", vec![Some(shp), Some(idx)], vec![gathered]);
    gnode.attributes.insert("axis".into(), Attribute::Int(0));
    g.insert_node(gnode);
    let mut cnode = node(3, "Concat", vec![Some(gathered), Some(tail)], vec![target]);
    cnode.attributes.insert("axis".into(), Attribute::Int(0));
    g.insert_node(cnode);
    g.insert_node(node(4, "Reshape", vec![Some(x), Some(target)], vec![out]));
    g.add_output(out);
    g.opset_imports.insert(String::new(), 13);

    let reg = InferenceRegistry::default_registry();
    let opsets = g.opset_imports.clone();
    let report = reg
        .infer_graph(&mut g, &opsets, MergePolicy::Permissive)
        .unwrap();
    assert!(
        report.fully_resolved(),
        "unresolved: {:?}",
        report.unresolved
    );

    // Reshape target = [N, 512] -> output [N, 512] with N symbolic.
    let out_shape = g.value(out).shape.clone();
    assert_eq!(out_shape.len(), 2);
    assert_eq!(out_shape[0], Dim::Symbolic(n_sym));
    assert_eq!(out_shape[1], Dim::Static(512));
}

/// End-to-end: load the committed `bert_toy` model and assert that
/// `infer_graph` resolves EVERY value in the graph — matching the bar the
/// loader already meets.
#[test]
fn bert_toy_fully_resolves() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../onnx-runtime-session/tests/fixtures/bert_toy/model.onnx"
    );
    let mut graph = onnx_runtime_loader::load_model(path).expect("load bert_toy");

    let total = graph.num_values();
    assert!(total > 0, "model has values");

    let reg = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    let report = reg
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer bert_toy");

    assert_eq!(
        report.num_unresolved(),
        0,
        "these values did not resolve: {:?}",
        report.unresolved
    );
    assert!(report.fully_resolved());
    assert_eq!(report.num_resolved(), total);

    // Every value must have a concrete rank (Shape is always Vec<Dim>); assert
    // no value was left as the default-empty placeholder unless it truly is a
    // scalar produced as such. We simply confirm the report counts line up.
    let opset = *opsets.get("").unwrap_or(&0);
    assert!(opset >= 1);
}
