//! Whole-graph inference tests: a hand-built symbolic transformer fragment and
//! an end-to-end run over the committed `bert_toy` model.

use onnx_runtime_ir::{
    Attribute, DataType, Dim, Graph, Node, NodeId, Shape, TensorData, ValueId, WeightRef,
};
use onnx_runtime_shape_inference::{
    DimExpr, InferenceRegistry, InferenceReport, MergePolicy, TensorType, ValueType,
};

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

fn if_graph(then_branch: Graph, else_branch: Graph) -> (Graph, ValueId) {
    let mut graph = Graph::new();
    let condition = graph.create_named_value("condition", DataType::Bool, Shape::new());
    graph.add_input(condition);
    let output = graph.create_named_value("output", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(output);
    graph.mark_value_shape_unknown(output);
    let if_node = graph.insert_node(node(0, "If", vec![Some(condition)], vec![output]));
    graph
        .subgraphs
        .insert((if_node, "then_branch".into()), then_branch);
    graph
        .subgraphs
        .insert((if_node, "else_branch".into()), else_branch);
    graph.add_output(output);
    graph.opset_imports.insert(String::new(), 21);
    (graph, output)
}

fn if_graph_with_output_count(
    then_branch: Graph,
    else_branch: Graph,
    output_count: usize,
) -> (Graph, Vec<ValueId>) {
    let mut graph = Graph::new();
    let condition = graph.create_named_value("condition", DataType::Bool, Shape::new());
    graph.add_input(condition);
    let outputs: Vec<_> = (0..output_count)
        .map(|index| {
            let output = graph.create_named_value(
                format!("output_{index}"),
                DataType::Float32,
                Shape::new(),
            );
            graph.mark_value_type_unknown(output);
            graph.mark_value_shape_unknown(output);
            graph.add_output(output);
            output
        })
        .collect();
    let if_node = graph.insert_node(node(0, "If", vec![Some(condition)], outputs.clone()));
    graph
        .subgraphs
        .insert((if_node, "then_branch".into()), then_branch);
    graph
        .subgraphs
        .insert((if_node, "else_branch".into()), else_branch);
    graph.opset_imports.insert(String::new(), 21);
    (graph, outputs)
}

fn captured_identity_branch(name: &str) -> Graph {
    let mut branch = Graph::new();
    let capture = branch.create_named_value(name, DataType::Float32, Shape::new());
    branch.mark_value_type_unknown(capture);
    branch.mark_value_shape_unknown(capture);
    let output = branch.create_named_value("branch_output", DataType::Float32, Shape::new());
    branch.insert_node(node(0, "Identity", vec![Some(capture)], vec![output]));
    branch.add_output(output);
    branch
}

fn nested_captured_if_branch(value_name: &str, condition_name: &str) -> Graph {
    let mut branch = Graph::new();
    let condition = branch.create_named_value(condition_name, DataType::Bool, Shape::new());
    branch.mark_value_type_unknown(condition);
    branch.mark_value_shape_unknown(condition);
    let capture = branch.create_named_value(value_name, DataType::Float32, Shape::new());
    branch.mark_value_type_unknown(capture);
    branch.mark_value_shape_unknown(capture);
    let output = branch.create_named_value("nested_output", DataType::Float32, Shape::new());
    branch.mark_value_type_unknown(output);
    branch.mark_value_shape_unknown(output);
    let if_node = branch.insert_node(node(0, "If", vec![Some(condition)], vec![output]));
    branch.subgraphs.insert(
        (if_node, "then_branch".into()),
        captured_identity_branch(value_name),
    );
    branch.subgraphs.insert(
        (if_node, "else_branch".into()),
        captured_identity_branch(value_name),
    );
    branch.add_output(output);
    branch
}

fn identity_branch(shape: Shape) -> Graph {
    identity_branch_with_element_type(DataType::Float32, shape)
}

fn identity_branch_with_element_type(element_type: DataType, shape: Shape) -> Graph {
    let mut branch = Graph::new();
    let input = branch.create_named_value("local", element_type, shape);
    branch.add_input(input);
    let output = branch.create_named_value("branch_output", element_type, Shape::new());
    branch.insert_node(node(0, "Identity", vec![Some(input)], vec![output]));
    branch.add_output(output);
    branch
}

fn identity_branch_outputs(shapes: Vec<Shape>) -> Graph {
    let mut branch = Graph::new();
    for (index, shape) in shapes.into_iter().enumerate() {
        let input = branch.create_named_value(format!("local_{index}"), DataType::Float32, shape);
        branch.add_input(input);
        let output = branch.create_named_value(
            format!("branch_output_{index}"),
            DataType::Float32,
            Shape::new(),
        );
        branch.insert_node(node(
            index as u32,
            "Identity",
            vec![Some(input)],
            vec![output],
        ));
        branch.add_output(output);
    }
    branch
}

fn nonzero_branch() -> Graph {
    let mut branch = Graph::new();
    let input = branch.create_named_value("local", DataType::Float32, vec![Dim::Static(2)]);
    branch.set_initializer(
        input,
        WeightRef::Inline(TensorData::from_raw(DataType::Float32, vec![2], vec![0; 8])),
    );
    let output = branch.create_named_value("branch_output", DataType::Int64, Shape::new());
    branch.insert_node(node(0, "NonZero", vec![Some(input)], vec![output]));
    branch.add_output(output);
    branch
}

#[test]
fn if_branch_inference_binds_lexically_captured_outer_value() {
    let (mut graph, output) = if_graph(
        captured_identity_branch("captured"),
        captured_identity_branch("captured"),
    );
    let captured = graph.create_named_value(
        "captured",
        DataType::Float16,
        vec![Dim::Static(2), Dim::Static(3)],
    );
    graph.add_input(captured);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer If with lexical capture");

    assert_eq!(graph.value(output).dtype, DataType::Float16);
    assert_eq!(
        graph.value(output).shape,
        vec![Dim::Static(2), Dim::Static(3)]
    );
}

#[test]
fn nested_if_captures_prior_outer_node_output() {
    let mut graph = Graph::new();
    let condition = graph.create_named_value("condition", DataType::Bool, Shape::new());
    graph.add_input(condition);
    let source = graph.create_named_value(
        "source",
        DataType::Float16,
        vec![Dim::Static(2), Dim::Static(3)],
    );
    graph.add_input(source);

    let captured = graph.create_named_value("captured", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(captured);
    graph.mark_value_shape_unknown(captured);
    graph.insert_node(node(0, "Identity", vec![Some(source)], vec![captured]));

    let output = graph.create_named_value("output", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(output);
    graph.mark_value_shape_unknown(output);
    let if_node = graph.insert_node(node(1, "If", vec![Some(condition)], vec![output]));
    graph.subgraphs.insert(
        (if_node, "then_branch".into()),
        nested_captured_if_branch("captured", "condition"),
    );
    graph.subgraphs.insert(
        (if_node, "else_branch".into()),
        nested_captured_if_branch("captured", "condition"),
    );
    graph.add_output(output);
    graph.opset_imports.insert(String::new(), 21);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer doubly nested If with lexical capture");

    let expected_shape = vec![Dim::Static(2), Dim::Static(3)];
    assert_eq!(graph.value(output).dtype, DataType::Float16);
    assert_eq!(graph.value(output).shape, expected_shape);

    for outer_attr in ["then_branch", "else_branch"] {
        let outer_branch = &graph.subgraphs[&(if_node, outer_attr.into())];
        let nested_if = NodeId(0);
        let nested_output = outer_branch.outputs[0];
        assert_eq!(outer_branch.value(nested_output).dtype, DataType::Float16);
        assert_eq!(outer_branch.value(nested_output).shape, expected_shape);

        for inner_attr in ["then_branch", "else_branch"] {
            let inner_branch = &outer_branch.subgraphs[&(nested_if, inner_attr.into())];
            let inner_output = inner_branch.outputs[0];
            assert_eq!(inner_branch.value(inner_output).dtype, DataType::Float16);
            assert_eq!(inner_branch.value(inner_output).shape, expected_shape);
        }
    }
}

#[test]
fn if_branch_local_symbols_merge_to_fresh_parent_symbol() {
    let (mut graph, output) = if_graph(nonzero_branch(), nonzero_branch());

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    let report = registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer If with independent branch symbols");

    let [Dim::Static(1), Dim::Symbolic(merged)] = graph.value(output).shape.as_slice() else {
        panic!("expected [1, fresh_symbol] output shape");
    };
    let if_node = graph.node(NodeId(0));
    let then_branch = &graph.subgraphs[&(if_node.id, "then_branch".into())];
    let else_branch = &graph.subgraphs[&(if_node.id, "else_branch".into())];
    let Dim::Symbolic(then_symbol) = then_branch.value(then_branch.outputs[0]).shape[1] else {
        panic!("expected then-branch local symbol");
    };
    let Dim::Symbolic(else_symbol) = else_branch.value(else_branch.outputs[0]).shape[1] else {
        panic!("expected else-branch local symbol");
    };
    assert_eq!(
        then_symbol, else_symbol,
        "the regression requires colliding numeric branch-local ids"
    );
    assert_eq!(
        report.fresh_symbols, 1,
        "the parent merge must mint its own symbol"
    );
    assert!(
        graph.symbol_constraints.contains_key(merged),
        "merged symbol must belong to the parent graph"
    );
}

#[test]
fn if_captured_symbol_maps_back_to_parent_namespace() {
    let (mut graph, output) = if_graph(
        captured_identity_branch("captured"),
        captured_identity_branch("captured"),
    );
    let batch = graph.intern_symbol("batch");
    let captured = graph.create_named_value(
        "captured",
        DataType::Float32,
        vec![Dim::Symbolic(batch), Dim::Static(3)],
    );
    graph.add_input(captured);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer If with captured parent symbol");

    assert_eq!(
        graph.value(output).shape,
        vec![Dim::Symbolic(batch), Dim::Static(3)]
    );
}

#[test]
fn if_equal_concrete_branch_dims_stay_concrete() {
    let (mut graph, output) = if_graph(
        identity_branch(vec![Dim::Static(7)]),
        identity_branch(vec![Dim::Static(7)]),
    );

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer If with equal concrete dimensions");

    assert_eq!(graph.value(output).shape, vec![Dim::Static(7)]);
}

#[test]
fn if_fewer_declared_outputs_infers_paired_outputs_and_ignores_branch_extras() {
    let (mut graph, outputs) = if_graph_with_output_count(
        identity_branch_outputs(vec![vec![Dim::Static(7)], vec![Dim::Static(2)]]),
        identity_branch_outputs(vec![
            vec![Dim::Static(7)],
            vec![Dim::Static(2), Dim::Static(3)],
        ]),
        1,
    );

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("extra branch outputs must be ignored");

    assert_eq!(graph.value(outputs[0]).shape, vec![Dim::Static(7)]);
}

#[test]
fn if_more_declared_outputs_leaves_unpaired_outputs_unresolved() {
    let (mut graph, outputs) = if_graph_with_output_count(
        identity_branch_outputs(vec![vec![Dim::Static(7)]]),
        identity_branch_outputs(vec![vec![Dim::Static(7)], vec![Dim::Static(9)]]),
        3,
    );

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    let report = registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("missing branch outputs must leave node outputs unresolved");

    assert_eq!(graph.value(outputs[0]).shape, vec![Dim::Static(7)]);
    assert!(report.resolved.contains(&outputs[0]));
    assert!(report.unresolved.contains(&outputs[1]));
    assert!(report.unresolved.contains(&outputs[2]));
}

fn assert_if_rank_mismatch_is_dynamic(then_rank: usize, else_rank: usize) {
    let then_shape = (1..=then_rank).map(Dim::Static).collect();
    let else_shape = (1..=else_rank).map(Dim::Static).collect();
    let (mut graph, output) = if_graph(
        identity_branch_with_element_type(DataType::Float16, then_shape),
        identity_branch_with_element_type(DataType::Float16, else_shape),
    );

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    let report = registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("branch rank mismatch must produce a dynamic-rank output");

    assert_eq!(graph.value(output).dtype, DataType::Float16);
    assert!(graph.value_type_is_known(output));
    assert!(!graph.value_shape_is_known(output));
    assert!(report.unresolved.contains(&output));
}

#[test]
fn if_rank_two_and_rank_three_branches_produce_dynamic_rank() {
    assert_if_rank_mismatch_is_dynamic(2, 3);
}

#[test]
fn if_rank_four_and_rank_five_branches_produce_dynamic_rank() {
    assert_if_rank_mismatch_is_dynamic(4, 5);
}

#[test]
fn if_branch_element_type_mismatch_is_an_error() {
    let (mut graph, _) = if_graph(
        identity_branch_with_element_type(DataType::Float16, vec![Dim::Static(2)]),
        identity_branch_with_element_type(DataType::Float32, vec![Dim::Static(2)]),
    );

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    let error = registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect_err("branch element type mismatch must fail");

    assert!(matches!(
        error,
        onnx_runtime_shape_inference::ShapeInferError::Invalid { op, detail }
            if op == "If" && detail.contains("branch output element types differ")
    ));
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

#[test]
fn dynamic_slice_extent_flows_through_unsqueeze_and_broadcast() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);

    let data = graph.create_named_value(
        "data",
        DataType::Float32,
        vec![Dim::Static(2), Dim::Static(4)],
    );
    graph.add_input(data);
    let ends = graph.create_named_value("ends", DataType::Int64, vec![Dim::Static(1)]);
    graph.add_input(ends);

    let starts = graph.create_named_value("starts", DataType::Int64, vec![Dim::Static(1)]);
    graph.set_initializer(
        starts,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            vec![1],
            i64_bytes(&[0]),
        )),
    );
    let slice_axes = graph.create_named_value("slice_axes", DataType::Int64, vec![Dim::Static(1)]);
    graph.set_initializer(
        slice_axes,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            vec![1],
            i64_bytes(&[1]),
        )),
    );
    let steps = graph.create_named_value("steps", DataType::Int64, vec![Dim::Static(1)]);
    graph.set_initializer(
        steps,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            vec![1],
            i64_bytes(&[1]),
        )),
    );
    let unsqueeze_axes =
        graph.create_named_value("unsqueeze_axes", DataType::Int64, vec![Dim::Static(1)]);
    graph.set_initializer(
        unsqueeze_axes,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            vec![1],
            i64_bytes(&[-1]),
        )),
    );
    let thresholds = graph.create_named_value(
        "thresholds",
        DataType::Float32,
        vec![Dim::Static(1), Dim::Static(1), Dim::Static(2)],
    );
    graph.add_input(thresholds);

    let sliced = graph.create_named_value("sliced", DataType::Float32, Shape::new());
    let unsqueezed = graph.create_named_value("unsqueezed", DataType::Float32, Shape::new());
    let compared = graph.create_named_value("compared", DataType::Bool, Shape::new());
    graph.insert_node(node(
        1,
        "Slice",
        vec![
            Some(data),
            Some(starts),
            Some(ends),
            Some(slice_axes),
            Some(steps),
        ],
        vec![sliced],
    ));
    graph.insert_node(node(
        2,
        "Unsqueeze",
        vec![Some(sliced), Some(unsqueeze_axes)],
        vec![unsqueezed],
    ));
    graph.insert_node(node(
        3,
        "Less",
        vec![Some(unsqueezed), Some(thresholds)],
        vec![compared],
    ));
    graph.add_output(compared);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .unwrap();

    let sliced_shape = graph.value(sliced).shape.clone();
    assert_eq!(sliced_shape.len(), 2);
    assert_eq!(sliced_shape[0], Dim::Static(2));
    let Dim::Symbolic(dynamic_extent) = sliced_shape[1] else {
        panic!("dynamic Slice end must produce a symbolic extent");
    };
    assert_eq!(
        graph.value(unsqueezed).shape,
        vec![
            Dim::Static(2),
            Dim::Symbolic(dynamic_extent),
            Dim::Static(1),
        ]
    );
    assert_eq!(
        graph.value(compared).shape,
        vec![
            Dim::Static(2),
            Dim::Symbolic(dynamic_extent),
            Dim::Static(2),
        ]
    );
}

#[test]
fn unsqueeze_with_runtime_axes_preserves_known_output_rank() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let data = graph.create_named_value(
        "data",
        DataType::Float32,
        vec![Dim::Static(2), Dim::Static(3)],
    );
    graph.add_input(data);
    let axes = graph.create_named_value("axes", DataType::Int64, vec![Dim::Static(2)]);
    graph.add_input(axes);
    let output = graph.create_named_value("output", DataType::Float32, Shape::new());
    graph.insert_node(node(
        1,
        "Unsqueeze",
        vec![Some(data), Some(axes)],
        vec![output],
    ));
    graph.add_output(output);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .unwrap();

    let output_shape = &graph.value(output).shape;
    assert_eq!(output_shape.len(), 4);
    assert!(
        output_shape
            .iter()
            .all(|dim| matches!(dim, Dim::Symbolic(_)))
    );
}

#[test]
fn declared_intermediate_shape_drives_downstream_gather() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 13);

    let opaque_input = graph.create_named_value("opaque_input", DataType::Int64, Shape::new());
    graph.add_input(opaque_input);
    let indices = graph.create_named_value(
        "indices",
        DataType::Int64,
        vec![Dim::Static(1), Dim::Static(8)],
    );
    graph.insert_node(node(
        0,
        "UnsupportedIndicesProducer",
        vec![Some(opaque_input)],
        vec![indices],
    ));

    let data = graph.create_named_value(
        "data",
        DataType::Float32,
        vec![Dim::Static(512), Dim::Static(32)],
    );
    graph.add_input(data);
    let output = graph.create_named_value("output", DataType::Float32, Shape::new());
    graph.mark_value_shape_unknown(output);
    graph.insert_node(node(
        1,
        "Gather",
        vec![Some(data), Some(indices)],
        vec![output],
    ));
    graph.add_output(output);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer Gather from declared intermediate value_info");

    assert_eq!(
        graph.value(output).shape,
        vec![Dim::Static(1), Dim::Static(8), Dim::Static(32)]
    );
}

/// A contrib fused norm with no intermediate value_info must still resolve all
/// requested outputs so session setup can allocate them.
#[test]
fn skip_simplified_layer_norm_resolves_outputs_without_value_info() {
    let mut graph = Graph::new();
    let input_shape = vec![Dim::Static(2), Dim::Static(8), Dim::Static(64)];

    let input = graph.create_named_value("input", DataType::Float16, input_shape.clone());
    let skip = graph.create_named_value("skip", DataType::Float16, input_shape.clone());
    let gamma = graph.create_named_value("gamma", DataType::Float16, vec![Dim::Static(64)]);
    graph.add_input(input);
    graph.add_input(skip);
    graph.add_input(gamma);

    // These empty shapes model omitted intermediate value_info entries.
    let output = graph.create_named_value("output", DataType::Float32, Shape::new());
    let mean = graph.create_named_value("mean", DataType::Float32, Shape::new());
    let inv_std_var = graph.create_named_value("inv_std_var", DataType::Float32, Shape::new());
    let input_skip_bias_sum =
        graph.create_named_value("input_skip_bias_sum", DataType::Float32, Shape::new());

    let mut norm = node(
        1,
        "SkipSimplifiedLayerNormalization",
        vec![Some(input), Some(skip), Some(gamma)],
        vec![output, mean, inv_std_var, input_skip_bias_sum],
    );
    norm.domain = "com.microsoft".into();
    graph.insert_node(norm);
    graph.add_output(output);
    graph.add_output(input_skip_bias_sum);
    graph.opset_imports.insert(String::new(), 21);
    graph.opset_imports.insert("com.microsoft".into(), 1);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    let report = registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer SkipSimplifiedLayerNormalization graph");

    assert!(
        report.fully_resolved(),
        "unresolved: {:?}",
        report.unresolved
    );
    assert_eq!(graph.value(output).shape, input_shape);
    assert_eq!(graph.value(input_skip_bias_sum).shape, input_shape);
    assert_eq!(graph.value(output).dtype, DataType::Float16);
    assert_eq!(graph.value(input_skip_bias_sum).dtype, DataType::Float16);
    assert_eq!(
        graph.value(mean).shape,
        vec![Dim::Static(2), Dim::Static(8), Dim::Static(1)]
    );
    assert_eq!(
        graph.value(inv_std_var).shape,
        vec![Dim::Static(2), Dim::Static(8), Dim::Static(1)]
    );
}

/// End-to-end: load the committed `bert_toy` model and assert that
/// `infer_graph` resolves EVERY value in the graph — matching the bar the
/// loader already meets.
#[test]
fn bert_toy_fully_resolves() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../onnx-runtime-session/tests/fixtures/bert_toy/model.onnx.textproto"
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

// ===========================================================================
// Loop / Scan control-flow inference.
// ===========================================================================

/// A `Loop` body `(iter_num, cond_in, v) -> (cond_out, v_out, scan_out)` that
/// passes the single loop-carried value straight through and emits it as a scan
/// output too. The carried formal input `v` is left shape/type-unknown so the
/// test proves the shape is supplied by *seeding* from the `Loop` operand.
fn loop_passthrough_body() -> Graph {
    let mut body = Graph::new();
    let iter = body.create_named_value("iter", DataType::Int64, Shape::new());
    body.add_input(iter);
    let cond_in = body.create_named_value("cond_in", DataType::Bool, Shape::new());
    body.add_input(cond_in);
    let v = body.create_named_value("v", DataType::Float32, Shape::new());
    body.mark_value_type_unknown(v);
    body.mark_value_shape_unknown(v);
    body.add_input(v);

    let cond_out = body.create_named_value("cond_out", DataType::Bool, Shape::new());
    body.insert_node(node(0, "Identity", vec![Some(cond_in)], vec![cond_out]));
    let v_out = body.create_named_value("v_out", DataType::Float32, Shape::new());
    body.insert_node(node(1, "Identity", vec![Some(v)], vec![v_out]));
    let scan_out = body.create_named_value("scan_out", DataType::Float32, Shape::new());
    body.insert_node(node(2, "Identity", vec![Some(v)], vec![scan_out]));
    body.add_output(cond_out);
    body.add_output(v_out);
    body.add_output(scan_out);
    body
}

/// Build a single-carried `Loop` with the pass-through body. `trip_count`, when
/// `Some`, is supplied as a static scalar `M` initializer; when `None`, `M` is a
/// dynamic input. Returns `(graph, carried_output, scan_output)`.
fn build_loop(trip_count: Option<i64>, carried_shape: Shape) -> (Graph, ValueId, ValueId) {
    let mut graph = Graph::new();
    let m = graph.create_named_value("M", DataType::Int64, Shape::new());
    match trip_count {
        Some(value) => graph.set_initializer(
            m,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Int64,
                vec![],
                i64_bytes(&[value]),
            )),
        ),
        None => graph.add_input(m),
    }
    let cond = graph.create_named_value("cond", DataType::Bool, Shape::new());
    graph.add_input(cond);
    let v = graph.create_named_value("v", DataType::Float32, carried_shape);
    graph.add_input(v);

    let carried_out = graph.create_named_value("carried_out", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(carried_out);
    graph.mark_value_shape_unknown(carried_out);
    let scan = graph.create_named_value("scan", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(scan);
    graph.mark_value_shape_unknown(scan);

    let loop_node = graph.insert_node(node(
        0,
        "Loop",
        vec![Some(m), Some(cond), Some(v)],
        vec![carried_out, scan],
    ));
    graph
        .subgraphs
        .insert((loop_node, "body".into()), loop_passthrough_body());
    graph.add_output(carried_out);
    graph.add_output(scan);
    graph.opset_imports.insert(String::new(), 21);
    (graph, carried_out, scan)
}

#[test]
fn loop_static_trip_count_stacks_scan_output_and_propagates_carried_shape() {
    let (mut graph, carried_out, scan) = build_loop(Some(5), vec![Dim::Static(2), Dim::Static(3)]);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer Loop with static trip count");

    // Loop-carried final keeps the body carried-output shape.
    assert_eq!(
        graph.value(carried_out).shape,
        vec![Dim::Static(2), Dim::Static(3)]
    );
    assert_eq!(graph.value(carried_out).dtype, DataType::Float32);
    // Scan output gains a leading trip-count axis. The trip count is symbolic
    // even for a static `M`, because `cond` can early-exit; execution computes
    // the true extent and eager buffer planning must not over-reserve `M` slots.
    let shape = &graph.value(scan).shape;
    assert!(
        matches!(shape[0], Dim::Symbolic(_)),
        "trip count must produce a symbolic leading dim, got {shape:?}"
    );
    assert_eq!(shape[1..], [Dim::Static(2), Dim::Static(3)]);
}

#[test]
fn loop_dynamic_trip_count_uses_symbolic_leading_scan_dim() {
    let (mut graph, carried_out, scan) = build_loop(None, vec![Dim::Static(2), Dim::Static(3)]);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer Loop with dynamic trip count");

    assert_eq!(
        graph.value(carried_out).shape,
        vec![Dim::Static(2), Dim::Static(3)]
    );
    let shape = &graph.value(scan).shape;
    assert!(
        matches!(shape[0], Dim::Symbolic(_)),
        "unknown trip count must produce a symbolic leading dim, got {shape:?}"
    );
    assert_eq!(shape[1..], [Dim::Static(2), Dim::Static(3)]);
}

#[test]
fn loop_preserves_symbolic_carried_dimension_through_the_body() {
    let mut graph = Graph::new();
    let batch = graph.intern_symbol("batch");
    let m = graph.create_named_value("M", DataType::Int64, Shape::new());
    graph.set_initializer(
        m,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            vec![],
            i64_bytes(&[4]),
        )),
    );
    let cond = graph.create_named_value("cond", DataType::Bool, Shape::new());
    graph.add_input(cond);
    let v = graph.create_named_value(
        "v",
        DataType::Float32,
        vec![Dim::Symbolic(batch), Dim::Static(3)],
    );
    graph.add_input(v);
    let carried_out = graph.create_named_value("carried_out", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(carried_out);
    graph.mark_value_shape_unknown(carried_out);
    let scan = graph.create_named_value("scan", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(scan);
    graph.mark_value_shape_unknown(scan);
    let loop_node = graph.insert_node(node(
        0,
        "Loop",
        vec![Some(m), Some(cond), Some(v)],
        vec![carried_out, scan],
    ));
    graph
        .subgraphs
        .insert((loop_node, "body".into()), loop_passthrough_body());
    graph.add_output(carried_out);
    graph.add_output(scan);
    graph.opset_imports.insert(String::new(), 21);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer Loop with symbolic carried dim");

    assert_eq!(
        graph.value(carried_out).shape,
        vec![Dim::Symbolic(batch), Dim::Static(3)]
    );
    let shape = &graph.value(scan).shape;
    assert!(
        matches!(shape[0], Dim::Symbolic(_)),
        "trip count must produce a symbolic leading dim, got {shape:?}"
    );
    assert_eq!(shape[1..], [Dim::Symbolic(batch), Dim::Static(3)]);
}

/// A `Scan` body `(state, scan_slice) -> (state_out, scan_out)` that passes both
/// through. Both formal inputs are shape/type-unknown, so shapes come only from
/// seeding (state unchanged; scan slice with its scan axis stripped).
fn scan_passthrough_body() -> Graph {
    let mut body = Graph::new();
    let state = body.create_named_value("state", DataType::Float32, Shape::new());
    body.mark_value_type_unknown(state);
    body.mark_value_shape_unknown(state);
    body.add_input(state);
    let slice = body.create_named_value("slice", DataType::Float32, Shape::new());
    body.mark_value_type_unknown(slice);
    body.mark_value_shape_unknown(slice);
    body.add_input(slice);

    let state_out = body.create_named_value("state_out", DataType::Float32, Shape::new());
    body.insert_node(node(0, "Identity", vec![Some(state)], vec![state_out]));
    let scan_out = body.create_named_value("scan_out", DataType::Float32, Shape::new());
    body.insert_node(node(1, "Identity", vec![Some(slice)], vec![scan_out]));
    body.add_output(state_out);
    body.add_output(scan_out);
    body
}

fn build_scan(
    opset: u64,
    state_shape: Shape,
    scan_shape: Shape,
    input_axes: Option<Vec<i64>>,
    output_axes: Option<Vec<i64>>,
) -> (Graph, ValueId, ValueId) {
    let mut graph = Graph::new();
    let state = graph.create_named_value("state_init", DataType::Float32, state_shape);
    graph.add_input(state);
    let scan_in = graph.create_named_value("scan_in", DataType::Float32, scan_shape);
    graph.add_input(scan_in);

    let state_out = graph.create_named_value("state_out", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(state_out);
    graph.mark_value_shape_unknown(state_out);
    let scan_out = graph.create_named_value("scan_out", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(scan_out);
    graph.mark_value_shape_unknown(scan_out);

    let mut scan_node = node(
        0,
        "Scan",
        vec![Some(state), Some(scan_in)],
        vec![state_out, scan_out],
    );
    scan_node
        .attributes
        .insert("num_scan_inputs".into(), Attribute::Int(1));
    if let Some(axes) = input_axes {
        scan_node
            .attributes
            .insert("scan_input_axes".into(), Attribute::Ints(axes));
    }
    if let Some(axes) = output_axes {
        scan_node
            .attributes
            .insert("scan_output_axes".into(), Attribute::Ints(axes));
    }
    let scan_id = graph.insert_node(scan_node);
    graph
        .subgraphs
        .insert((scan_id, "body".into()), scan_passthrough_body());
    graph.add_output(state_out);
    graph.add_output(scan_out);
    graph.opset_imports.insert(String::new(), opset);
    (graph, state_out, scan_out)
}

#[test]
fn scan_strips_input_axis_and_reinserts_output_axis() {
    let (mut graph, state_out, scan_out) = build_scan(
        16,
        vec![Dim::Static(2)],
        vec![Dim::Static(6), Dim::Static(4)],
        None,
        None,
    );

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer Scan");

    // Final state keeps its shape; scan output re-gains the sequence axis (6).
    assert_eq!(graph.value(state_out).shape, vec![Dim::Static(2)]);
    assert_eq!(
        graph.value(scan_out).shape,
        vec![Dim::Static(6), Dim::Static(4)]
    );
}

#[test]
fn scan_honours_non_default_input_and_output_axes() {
    // Scan axis 1 of a [4, 6] input (sequence length 6); output axis 1.
    let (mut graph, state_out, scan_out) = build_scan(
        16,
        vec![Dim::Static(2)],
        vec![Dim::Static(4), Dim::Static(6)],
        Some(vec![1]),
        Some(vec![1]),
    );

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer Scan with custom axes");

    assert_eq!(graph.value(state_out).shape, vec![Dim::Static(2)]);
    // Per-iteration slice is [4]; the sequence axis (6) is inserted at axis 1.
    assert_eq!(
        graph.value(scan_out).shape,
        vec![Dim::Static(4), Dim::Static(6)]
    );
}

#[test]
fn scan_opset_eight_is_not_modelled_and_leaves_outputs_unresolved() {
    let (mut graph, state_out, scan_out) = build_scan(
        8,
        vec![Dim::Static(2)],
        vec![Dim::Static(6), Dim::Static(4)],
        None,
        None,
    );

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    let report = registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer legacy Scan");

    // The opset-8 form (extra sequence_lens input, different body signature) is
    // deliberately not inferred: outputs stay unresolved rather than wrong.
    assert!(report.unresolved.contains(&state_out));
    assert!(report.unresolved.contains(&scan_out));
}

/// A `Scan` body with one state and two scan slices:
/// `(state, slice_a, slice_b) -> (state_out, scan_a_out, scan_b_out)`, all
/// passed straight through so shapes come purely from seeding + axis handling.
fn scan_two_input_body() -> Graph {
    let mut body = Graph::new();
    let names = ["state", "slice_a", "slice_b"];
    let inputs: Vec<ValueId> = names
        .iter()
        .map(|name| {
            let v = body.create_named_value(*name, DataType::Float32, Shape::new());
            body.mark_value_type_unknown(v);
            body.mark_value_shape_unknown(v);
            body.add_input(v);
            v
        })
        .collect();
    for (i, (src, out_name)) in inputs
        .iter()
        .zip(["state_out", "scan_a_out", "scan_b_out"])
        .enumerate()
    {
        let out = body.create_named_value(out_name, DataType::Float32, Shape::new());
        body.insert_node(node(i as u32, "Identity", vec![Some(*src)], vec![out]));
        body.add_output(out);
    }
    body
}

#[test]
fn scan_two_scan_inputs_strip_and_reinsert_per_input_axes() {
    let mut graph = Graph::new();
    let state = graph.create_named_value("state_init", DataType::Float32, vec![Dim::Static(2)]);
    graph.add_input(state);
    // scan_a scanned on axis 0 (sequence length 5, per-iter slice [4]).
    let scan_a = graph.create_named_value(
        "scan_a",
        DataType::Float32,
        vec![Dim::Static(5), Dim::Static(4)],
    );
    graph.add_input(scan_a);
    // scan_b scanned on axis 1 (sequence length 5, per-iter slice [3]).
    let scan_b = graph.create_named_value(
        "scan_b",
        DataType::Float32,
        vec![Dim::Static(3), Dim::Static(5)],
    );
    graph.add_input(scan_b);

    let outs: Vec<ValueId> = ["state_out", "scan_a_out", "scan_b_out"]
        .iter()
        .map(|name| {
            let v = graph.create_named_value(*name, DataType::Float32, Shape::new());
            graph.mark_value_type_unknown(v);
            graph.mark_value_shape_unknown(v);
            v
        })
        .collect();

    let mut scan_node = node(
        0,
        "Scan",
        vec![Some(state), Some(scan_a), Some(scan_b)],
        outs.clone(),
    );
    scan_node
        .attributes
        .insert("num_scan_inputs".into(), Attribute::Int(2));
    scan_node
        .attributes
        .insert("scan_input_axes".into(), Attribute::Ints(vec![0, 1]));
    scan_node
        .attributes
        .insert("scan_output_axes".into(), Attribute::Ints(vec![0, 1]));
    let scan_id = graph.insert_node(scan_node);
    graph
        .subgraphs
        .insert((scan_id, "body".into()), scan_two_input_body());
    for out in &outs {
        graph.add_output(*out);
    }
    graph.opset_imports.insert(String::new(), 16);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer Scan with two scan inputs");

    // Final state keeps its shape.
    assert_eq!(graph.value(outs[0]).shape, vec![Dim::Static(2)]);
    // scan_a_out: per-iter [4], sequence axis (5) re-inserted at output axis 0.
    assert_eq!(
        graph.value(outs[1]).shape,
        vec![Dim::Static(5), Dim::Static(4)]
    );
    // scan_b_out: per-iter [3], sequence axis (5) re-inserted at output axis 1.
    assert_eq!(
        graph.value(outs[2]).shape,
        vec![Dim::Static(3), Dim::Static(5)]
    );
}

/// A `Loop` body with two loop-carried deps, each also emitted as a scan output:
/// `(iter, cond_in, v1, v2) -> (cond_out, v1_out, v2_out, scan1, scan2)`.
fn loop_two_carried_body() -> Graph {
    let mut body = Graph::new();
    let iter = body.create_named_value("iter", DataType::Int64, Shape::new());
    body.add_input(iter);
    let cond_in = body.create_named_value("cond_in", DataType::Bool, Shape::new());
    body.add_input(cond_in);
    let v1 = body.create_named_value("v1", DataType::Float32, Shape::new());
    body.mark_value_type_unknown(v1);
    body.mark_value_shape_unknown(v1);
    body.add_input(v1);
    let v2 = body.create_named_value("v2", DataType::Float32, Shape::new());
    body.mark_value_type_unknown(v2);
    body.mark_value_shape_unknown(v2);
    body.add_input(v2);

    let cond_out = body.create_named_value("cond_out", DataType::Bool, Shape::new());
    body.insert_node(node(0, "Identity", vec![Some(cond_in)], vec![cond_out]));
    body.add_output(cond_out);
    for (id, name, src) in [
        (1, "v1_out", v1),
        (2, "v2_out", v2),
        (3, "scan1", v1),
        (4, "scan2", v2),
    ] {
        let out = body.create_named_value(name, DataType::Float32, Shape::new());
        body.insert_node(node(id, "Identity", vec![Some(src)], vec![out]));
        body.add_output(out);
    }
    body
}

#[test]
fn loop_multiple_carried_and_scan_outputs_share_one_trip_count() {
    let mut graph = Graph::new();
    let m = graph.create_named_value("M", DataType::Int64, Shape::new());
    graph.add_input(m);
    let cond = graph.create_named_value("cond", DataType::Bool, Shape::new());
    graph.add_input(cond);
    let v1 = graph.create_named_value(
        "v1",
        DataType::Float32,
        vec![Dim::Static(2), Dim::Static(3)],
    );
    graph.add_input(v1);
    let v2 = graph.create_named_value("v2", DataType::Float32, vec![Dim::Static(4)]);
    graph.add_input(v2);

    let outs: Vec<ValueId> = ["v1_final", "v2_final", "scan1_out", "scan2_out"]
        .iter()
        .map(|name| {
            let v = graph.create_named_value(*name, DataType::Float32, Shape::new());
            graph.mark_value_type_unknown(v);
            graph.mark_value_shape_unknown(v);
            v
        })
        .collect();

    let loop_node = graph.insert_node(node(
        0,
        "Loop",
        vec![Some(m), Some(cond), Some(v1), Some(v2)],
        outs.clone(),
    ));
    graph
        .subgraphs
        .insert((loop_node, "body".into()), loop_two_carried_body());
    for out in &outs {
        graph.add_output(*out);
    }
    graph.opset_imports.insert(String::new(), 21);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer Loop with two carried deps");

    // Both loop-carried finals keep their seeded operand shapes.
    assert_eq!(
        graph.value(outs[0]).shape,
        vec![Dim::Static(2), Dim::Static(3)]
    );
    assert_eq!(graph.value(outs[1]).shape, vec![Dim::Static(4)]);

    // Each scan output gains a symbolic leading trip-count axis over its
    // per-iteration body shape.
    let scan1 = graph.value(outs[2]).shape.clone();
    let scan2 = graph.value(outs[3]).shape.clone();
    assert!(matches!(scan1[0], Dim::Symbolic(_)), "scan1 {scan1:?}");
    assert!(matches!(scan2[0], Dim::Symbolic(_)), "scan2 {scan2:?}");
    assert_eq!(scan1[1..], [Dim::Static(2), Dim::Static(3)]);
    assert_eq!(scan2[1..], [Dim::Static(4)]);
    // All scan outputs share a single iteration count, so their leading axis is
    // the *same* symbol.
    assert_eq!(scan1[0], scan2[0], "scan outputs must share one trip count");
}

#[test]
fn loop_body_with_nested_if_resolves_through_both_subgraph_levels() {
    let mut body = Graph::new();
    let iter = body.create_named_value("iter", DataType::Int64, Shape::new());
    body.add_input(iter);
    let cond_in = body.create_named_value("cond_in", DataType::Bool, Shape::new());
    body.add_input(cond_in);
    let v = body.create_named_value("v", DataType::Float32, Shape::new());
    body.mark_value_type_unknown(v);
    body.mark_value_shape_unknown(v);
    body.add_input(v);

    // Inner If both branches return the captured carried value `v`.
    let w = body.create_named_value("w", DataType::Float32, Shape::new());
    body.mark_value_type_unknown(w);
    body.mark_value_shape_unknown(w);
    let if_node = body.insert_node(node(0, "If", vec![Some(cond_in)], vec![w]));
    body.subgraphs.insert(
        (if_node, "then_branch".into()),
        captured_identity_branch("v"),
    );
    body.subgraphs.insert(
        (if_node, "else_branch".into()),
        captured_identity_branch("v"),
    );

    let cond_out = body.create_named_value("cond_out", DataType::Bool, Shape::new());
    body.insert_node(node(1, "Identity", vec![Some(cond_in)], vec![cond_out]));
    let v_out = body.create_named_value("v_out", DataType::Float32, Shape::new());
    body.insert_node(node(2, "Identity", vec![Some(w)], vec![v_out]));
    let scan_out = body.create_named_value("scan_out", DataType::Float32, Shape::new());
    body.insert_node(node(3, "Identity", vec![Some(w)], vec![scan_out]));
    body.add_output(cond_out);
    body.add_output(v_out);
    body.add_output(scan_out);

    let mut graph = Graph::new();
    let m = graph.create_named_value("M", DataType::Int64, Shape::new());
    graph.set_initializer(
        m,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            vec![],
            i64_bytes(&[3]),
        )),
    );
    let cond = graph.create_named_value("cond", DataType::Bool, Shape::new());
    graph.add_input(cond);
    let v = graph.create_named_value("v", DataType::Float32, vec![Dim::Static(2), Dim::Static(3)]);
    graph.add_input(v);
    let carried_out = graph.create_named_value("carried_out", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(carried_out);
    graph.mark_value_shape_unknown(carried_out);
    let scan = graph.create_named_value("scan", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(scan);
    graph.mark_value_shape_unknown(scan);
    let loop_node = graph.insert_node(node(
        0,
        "Loop",
        vec![Some(m), Some(cond), Some(v)],
        vec![carried_out, scan],
    ));
    graph.subgraphs.insert((loop_node, "body".into()), body);
    graph.add_output(carried_out);
    graph.add_output(scan);
    graph.opset_imports.insert(String::new(), 21);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer nested Loop/If");

    assert_eq!(
        graph.value(carried_out).shape,
        vec![Dim::Static(2), Dim::Static(3)]
    );
    let shape = &graph.value(scan).shape;
    assert!(
        matches!(shape[0], Dim::Symbolic(_)),
        "trip count must produce a symbolic leading dim, got {shape:?}"
    );
    assert_eq!(shape[1..], [Dim::Static(2), Dim::Static(3)]);
}

/// Regression guard for the additive container-type model (#449): a pure-tensor
/// graph must infer byte-identically after the `ValueType` layer was added. Any
/// corruption of the tensor `TypeInfo` path by the new parallel container map
/// would change one of these resolved dtypes/shapes.
#[test]
fn tensor_only_path_is_byte_identical_after_container_type_model() {
    let mut graph = Graph::new();
    let batch = graph.create_symbol(Some("batch".into()));

    let x = graph.create_named_value(
        "x",
        DataType::Float32,
        vec![Dim::Symbolic(batch), Dim::Static(4)],
    );
    graph.add_input(x);
    let w = graph.create_named_value("w", DataType::Float32, vec![Dim::Static(4), Dim::Static(5)]);
    graph.add_input(w);
    let b = graph.create_named_value("b", DataType::Float32, vec![Dim::Static(5)]);
    graph.add_input(b);

    let mm = graph.create_named_value("mm", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(mm);
    graph.mark_value_shape_unknown(mm);
    let add = graph.create_named_value("add", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(add);
    graph.mark_value_shape_unknown(add);
    let y = graph.create_named_value("y", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(y);
    graph.mark_value_shape_unknown(y);

    graph.insert_node(node(0, "MatMul", vec![Some(x), Some(w)], vec![mm]));
    graph.insert_node(node(1, "Add", vec![Some(mm), Some(b)], vec![add]));
    graph.insert_node(node(2, "Relu", vec![Some(add)], vec![y]));
    graph.add_output(y);
    graph.opset_imports.insert(String::new(), 21);

    let mut opsets = std::collections::HashMap::new();
    opsets.insert(String::new(), 21);
    InferenceRegistry::default_registry()
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .unwrap();

    let expected = vec![Dim::Symbolic(batch), Dim::Static(5)];
    for value in [mm, add, y] {
        assert_eq!(graph.value(value).dtype, DataType::Float32);
        assert_eq!(graph.value(value).shape, expected, "value {value:?}");
    }
}

// ===========================================================================
// Container-aware control flow (#449 inc3a): sequence types threaded through
// `If` branches and `Loop` carried dependencies. Container outputs live in the
// parallel container map, so they are observed via `report.containers` (keyed
// by the outer control-flow node's output value id), never on the IR `Value`.
// ===========================================================================

/// The tensor leaf of a `Sequence<Tensor>` container type.
fn sequence_tensor(value_type: &ValueType) -> &TensorType {
    value_type
        .as_sequence_element()
        .and_then(ValueType::as_tensor)
        .expect("expected a Sequence<Tensor> container type")
}

/// A branch whose single output is a `Sequence<Tensor{element_type, shape}>`,
/// built by `SequenceConstruct` over a concrete tensor input. The branch input
/// is a resolved local (a branch graph input), so the sequence element shape is
/// fully determined inside the branch.
fn sequence_construct_branch(element_type: DataType, shape: Shape) -> Graph {
    let mut branch = Graph::new();
    let input = branch.create_named_value("local", element_type, shape);
    branch.add_input(input);
    let seq = branch.create_named_value("seq", element_type, Shape::new());
    branch.mark_value_type_unknown(seq);
    branch.mark_value_shape_unknown(seq);
    branch.insert_node(node(0, "SequenceConstruct", vec![Some(input)], vec![seq]));
    branch.add_output(seq);
    branch
}

fn run_if(then_branch: Graph, else_branch: Graph) -> (Graph, ValueId, InferenceReport) {
    let (mut graph, output) = if_graph(then_branch, else_branch);
    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    let report = registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer If with container branches");
    (graph, output, report)
}

#[test]
fn if_unifies_matching_sequence_branch_outputs() {
    let (_graph, output, report) = run_if(
        sequence_construct_branch(DataType::Float32, vec![Dim::Static(2), Dim::Static(3)]),
        sequence_construct_branch(DataType::Float32, vec![Dim::Static(2), Dim::Static(3)]),
    );

    let container = report
        .containers
        .get(&output)
        .expect("If output must be a container");
    let tensor = sequence_tensor(container);
    assert_eq!(tensor.dtype, DataType::Float32);
    assert_eq!(
        tensor.shape,
        Some(vec![DimExpr::constant(2), DimExpr::constant(3)]),
        "matching branch element shapes stay concrete"
    );
}

#[test]
fn if_sequence_branch_extent_disagreement_degrades_to_symbol() {
    let (_graph, output, report) = run_if(
        sequence_construct_branch(DataType::Float32, vec![Dim::Static(2), Dim::Static(3)]),
        sequence_construct_branch(DataType::Float32, vec![Dim::Static(2), Dim::Static(5)]),
    );

    let tensor = sequence_tensor(report.containers.get(&output).expect("container output"));
    assert_eq!(tensor.dtype, DataType::Float32);
    let shape = tensor.shape.as_ref().expect("element shape known");
    assert_eq!(
        shape[0],
        DimExpr::constant(2),
        "agreeing dim stays concrete"
    );
    assert!(
        shape[1].as_symbol().is_some(),
        "disagreeing element extent must degrade to a fresh symbol, got {:?}",
        shape[1]
    );
}

#[test]
fn if_sequence_branch_dtype_disagreement_errors() {
    let (mut graph, _output) = if_graph(
        sequence_construct_branch(DataType::Float32, vec![Dim::Static(2)]),
        sequence_construct_branch(DataType::Int64, vec![Dim::Static(2)]),
    );
    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    let error = registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect_err("mismatched sequence element dtypes must error");
    let message = format!("{error}");
    assert!(
        message.contains("If"),
        "error must be attributed to If, got {message}"
    );
}

#[test]
fn if_container_versus_tensor_branch_disagreement_errors() {
    let (mut graph, _output) = if_graph(
        sequence_construct_branch(DataType::Float32, vec![Dim::Static(2)]),
        identity_branch(vec![Dim::Static(2)]),
    );
    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    let error = registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect_err("a container branch and a tensor branch cannot unify");
    let message = format!("{error}");
    assert!(
        message.contains("container"),
        "error must explain the container/tensor mismatch, got {message}"
    );
}

/// A `Loop` body that accumulates into its carried sequence: `SequenceInsert`
/// of a concrete tensor into the seeded carried sequence each iteration.
fn sequence_accumulator_loop_body(element_type: DataType, shape: Vec<Dim>) -> Graph {
    let mut body = Graph::new();
    let iter = body.create_named_value("iter", DataType::Int64, Shape::new());
    body.add_input(iter);
    let cond_in = body.create_named_value("cond_in", DataType::Bool, Shape::new());
    body.add_input(cond_in);
    let seq_in = body.create_named_value("seq_in", element_type, Shape::new());
    body.mark_value_type_unknown(seq_in);
    body.mark_value_shape_unknown(seq_in);
    body.add_input(seq_in);

    let cond_out = body.create_named_value("cond_out", DataType::Bool, Shape::new());
    body.insert_node(node(0, "Identity", vec![Some(cond_in)], vec![cond_out]));

    let dims: Vec<usize> = shape
        .iter()
        .map(|dim| match dim {
            Dim::Static(extent) => *extent,
            _ => 1,
        })
        .collect();
    let byte_len: usize = dims.iter().product::<usize>() * 4;
    let ins = body.create_named_value("ins", element_type, shape);
    body.set_initializer(
        ins,
        WeightRef::Inline(TensorData::from_raw(
            element_type,
            dims,
            vec![0u8; byte_len],
        )),
    );

    let seq_out = body.create_named_value("seq_out", element_type, Shape::new());
    body.mark_value_type_unknown(seq_out);
    body.mark_value_shape_unknown(seq_out);
    body.insert_node(node(
        1,
        "SequenceInsert",
        vec![Some(seq_in), Some(ins)],
        vec![seq_out],
    ));

    body.add_output(cond_out);
    body.add_output(seq_out);
    body
}

/// A `Loop` body that passes its carried sequence straight through (body output
/// is the seeded formal input), proving a seeded container flows to the output.
fn sequence_passthrough_loop_body(element_type: DataType) -> Graph {
    let mut body = Graph::new();
    let iter = body.create_named_value("iter", DataType::Int64, Shape::new());
    body.add_input(iter);
    let cond_in = body.create_named_value("cond_in", DataType::Bool, Shape::new());
    body.add_input(cond_in);
    let seq_in = body.create_named_value("seq_in", element_type, Shape::new());
    body.mark_value_type_unknown(seq_in);
    body.mark_value_shape_unknown(seq_in);
    body.add_input(seq_in);

    let cond_out = body.create_named_value("cond_out", DataType::Bool, Shape::new());
    body.insert_node(node(0, "Identity", vec![Some(cond_in)], vec![cond_out]));
    body.add_output(cond_out);
    body.add_output(seq_in);
    body
}

/// Build a `Loop` carrying one sequence accumulator. The initial carried
/// operand is a sequence produced by an outer `SequenceConstruct` over a
/// concrete tensor of `elem_shape`. Returns `(graph, carried_output, report)`.
fn run_sequence_loop(
    body: Graph,
    element_type: DataType,
    elem_shape: Vec<Dim>,
) -> (ValueId, InferenceReport) {
    let mut graph = Graph::new();
    let m = graph.create_named_value("M", DataType::Int64, Shape::new());
    graph.add_input(m);
    let cond = graph.create_named_value("cond", DataType::Bool, Shape::new());
    graph.add_input(cond);
    let elem = graph.create_named_value("elem", element_type, elem_shape);
    graph.add_input(elem);

    let seq0 = graph.create_named_value("seq0", element_type, Shape::new());
    graph.mark_value_type_unknown(seq0);
    graph.mark_value_shape_unknown(seq0);
    graph.insert_node(node(0, "SequenceConstruct", vec![Some(elem)], vec![seq0]));

    let carried_out = graph.create_named_value("carried_out", element_type, Shape::new());
    graph.mark_value_type_unknown(carried_out);
    graph.mark_value_shape_unknown(carried_out);
    let loop_node = graph.insert_node(node(
        1,
        "Loop",
        vec![Some(m), Some(cond), Some(seq0)],
        vec![carried_out],
    ));
    graph.subgraphs.insert((loop_node, "body".into()), body);
    graph.add_output(carried_out);
    graph.opset_imports.insert(String::new(), 21);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    let report = registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer Loop carrying a sequence");
    (carried_out, report)
}

#[test]
fn loop_carries_sequence_accumulator_preserving_element_type() {
    let (carried_out, report) = run_sequence_loop(
        sequence_accumulator_loop_body(DataType::Float32, vec![Dim::Static(2), Dim::Static(3)]),
        DataType::Float32,
        vec![Dim::Static(2), Dim::Static(3)],
    );

    let tensor = sequence_tensor(
        report
            .containers
            .get(&carried_out)
            .expect("container output"),
    );
    assert_eq!(tensor.dtype, DataType::Float32);
    assert_eq!(
        tensor.shape,
        Some(vec![DimExpr::constant(2), DimExpr::constant(3)]),
        "inserting a matching element preserves the sequence element shape"
    );
}

#[test]
fn loop_passthrough_preserves_seeded_sequence_dtype() {
    let mut graph = Graph::new();
    let batch = graph.intern_symbol("batch");
    let m = graph.create_named_value("M", DataType::Int64, Shape::new());
    graph.add_input(m);
    let cond = graph.create_named_value("cond", DataType::Bool, Shape::new());
    graph.add_input(cond);
    let elem = graph.create_named_value(
        "elem",
        DataType::Float16,
        vec![Dim::Symbolic(batch), Dim::Static(3)],
    );
    graph.add_input(elem);

    let seq0 = graph.create_named_value("seq0", DataType::Float16, Shape::new());
    graph.mark_value_type_unknown(seq0);
    graph.mark_value_shape_unknown(seq0);
    graph.insert_node(node(0, "SequenceConstruct", vec![Some(elem)], vec![seq0]));

    let carried_out = graph.create_named_value("carried_out", DataType::Float16, Shape::new());
    graph.mark_value_type_unknown(carried_out);
    graph.mark_value_shape_unknown(carried_out);
    let loop_node = graph.insert_node(node(
        1,
        "Loop",
        vec![Some(m), Some(cond), Some(seq0)],
        vec![carried_out],
    ));
    graph.subgraphs.insert(
        (loop_node, "body".into()),
        sequence_passthrough_loop_body(DataType::Float16),
    );
    graph.add_output(carried_out);
    graph.opset_imports.insert(String::new(), 21);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    let report = registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer Loop passing a seeded sequence through");

    let tensor = sequence_tensor(
        report
            .containers
            .get(&carried_out)
            .expect("container output"),
    );
    assert_eq!(
        tensor.dtype,
        DataType::Float16,
        "seeded element dtype preserved"
    );
    let shape = tensor.shape.as_ref().expect("element shape known");
    assert_eq!(shape.len(), 2);
    assert!(
        shape[0].as_symbol().is_some(),
        "a seeded symbolic dim degrades to a fresh parent symbol, got {:?}",
        shape[0]
    );
    assert_eq!(
        shape[1],
        DimExpr::constant(3),
        "concrete element extent preserved"
    );
}

// ===========================================================================
// #449 inc4 (closeout): SequenceMap, cross-subgraph container capture, and
// Scan container state. All observed via `report.containers` on the outer node.
// ===========================================================================

/// A `SequenceMap` body with a single element input and one op applied to it.
/// The body input is left type/shape-unknown so it is proven to be *seeded*
/// from the input sequence's element type.
fn unary_map_body(op: &str, out_dtype: DataType) -> Graph {
    let mut body = Graph::new();
    let elem = body.create_named_value("elem", DataType::Float32, Shape::new());
    body.mark_value_type_unknown(elem);
    body.mark_value_shape_unknown(elem);
    body.add_input(elem);
    let mapped = body.create_named_value("mapped", out_dtype, Shape::new());
    body.mark_value_type_unknown(mapped);
    body.mark_value_shape_unknown(mapped);
    body.insert_node(node(0, op, vec![Some(elem)], vec![mapped]));
    body.add_output(mapped);
    body
}

/// Build a single-input-sequence `SequenceMap`: an outer tensor `elem` is wrapped
/// by `SequenceConstruct`, then mapped through `body`. Returns `(map_output,
/// report)`; the output sequence is observed via `report.containers`.
fn run_unary_sequence_map(body: Graph, elem_shape: Vec<Dim>) -> (ValueId, InferenceReport) {
    let mut graph = Graph::new();
    let elem = graph.create_named_value("elem", DataType::Float32, elem_shape);
    graph.add_input(elem);
    let seq0 = graph.create_named_value("seq0", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(seq0);
    graph.mark_value_shape_unknown(seq0);
    graph.insert_node(node(0, "SequenceConstruct", vec![Some(elem)], vec![seq0]));

    let map_out = graph.create_named_value("map_out", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(map_out);
    graph.mark_value_shape_unknown(map_out);
    let map_node = graph.insert_node(node(1, "SequenceMap", vec![Some(seq0)], vec![map_out]));
    graph.subgraphs.insert((map_node, "body".into()), body);
    graph.add_output(map_out);
    graph.opset_imports.insert(String::new(), 21);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    let report = registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer SequenceMap");
    (map_out, report)
}

#[test]
fn sequence_map_identity_preserves_element_type() {
    let (map_out, report) = run_unary_sequence_map(
        unary_map_body("Identity", DataType::Float32),
        vec![Dim::Static(2), Dim::Static(3)],
    );

    let tensor = sequence_tensor(report.containers.get(&map_out).expect("container output"));
    assert_eq!(tensor.dtype, DataType::Float32);
    assert_eq!(
        tensor.shape,
        Some(vec![DimExpr::constant(2), DimExpr::constant(3)]),
        "an identity body maps element type through unchanged"
    );
}

#[test]
fn sequence_map_body_transforms_element_type() {
    // `Shape` turns each [2,3] element into an int64 [2] vector; the OUTPUT
    // sequence element type must reflect that transform, proving the body was
    // seeded and its output wrapped.
    let (map_out, report) = run_unary_sequence_map(
        unary_map_body("Shape", DataType::Int64),
        vec![Dim::Static(2), Dim::Static(3)],
    );

    let tensor = sequence_tensor(report.containers.get(&map_out).expect("container output"));
    assert_eq!(tensor.dtype, DataType::Int64, "Shape output is int64");
    assert_eq!(
        tensor.shape,
        Some(vec![DimExpr::constant(2)]),
        "Shape of a rank-2 element is a length-2 vector"
    );
}

#[test]
fn sequence_map_zips_two_input_sequences() {
    // Two input sequences of [2,3] float elements; the body Adds the two
    // per-element tensors. The output sequence element = [2,3] float32.
    let mut graph = Graph::new();
    let a = graph.create_named_value("a", DataType::Float32, vec![Dim::Static(2), Dim::Static(3)]);
    graph.add_input(a);
    let b = graph.create_named_value("b", DataType::Float32, vec![Dim::Static(2), Dim::Static(3)]);
    graph.add_input(b);
    let seq_a = graph.create_named_value("seq_a", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(seq_a);
    graph.mark_value_shape_unknown(seq_a);
    graph.insert_node(node(0, "SequenceConstruct", vec![Some(a)], vec![seq_a]));
    let seq_b = graph.create_named_value("seq_b", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(seq_b);
    graph.mark_value_shape_unknown(seq_b);
    graph.insert_node(node(1, "SequenceConstruct", vec![Some(b)], vec![seq_b]));

    let mut body = Graph::new();
    let ea = body.create_named_value("ea", DataType::Float32, Shape::new());
    body.mark_value_type_unknown(ea);
    body.mark_value_shape_unknown(ea);
    body.add_input(ea);
    let eb = body.create_named_value("eb", DataType::Float32, Shape::new());
    body.mark_value_type_unknown(eb);
    body.mark_value_shape_unknown(eb);
    body.add_input(eb);
    let sum = body.create_named_value("sum", DataType::Float32, Shape::new());
    body.mark_value_type_unknown(sum);
    body.mark_value_shape_unknown(sum);
    body.insert_node(node(0, "Add", vec![Some(ea), Some(eb)], vec![sum]));
    body.add_output(sum);

    let map_out = graph.create_named_value("map_out", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(map_out);
    graph.mark_value_shape_unknown(map_out);
    let map_node = graph.insert_node(node(
        2,
        "SequenceMap",
        vec![Some(seq_a), Some(seq_b)],
        vec![map_out],
    ));
    graph.subgraphs.insert((map_node, "body".into()), body);
    graph.add_output(map_out);
    graph.opset_imports.insert(String::new(), 21);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    let report = registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer SequenceMap zip");

    let tensor = sequence_tensor(report.containers.get(&map_out).expect("container output"));
    assert_eq!(tensor.dtype, DataType::Float32);
    assert_eq!(
        tensor.shape,
        Some(vec![DimExpr::constant(2), DimExpr::constant(3)]),
        "zipping two sequences maps the element-wise body output type"
    );
}

#[test]
fn sequence_map_additional_tensor_input_is_broadcast() {
    // One input sequence of [2,3] elements + one whole-tensor additional input
    // [2,3]; the body Adds them. The extra input is seeded as a whole tensor
    // (not an element), and the output sequence element = [2,3] float32.
    let mut graph = Graph::new();
    let a = graph.create_named_value("a", DataType::Float32, vec![Dim::Static(2), Dim::Static(3)]);
    graph.add_input(a);
    let bias = graph.create_named_value(
        "bias",
        DataType::Float32,
        vec![Dim::Static(2), Dim::Static(3)],
    );
    graph.add_input(bias);
    let seq_a = graph.create_named_value("seq_a", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(seq_a);
    graph.mark_value_shape_unknown(seq_a);
    graph.insert_node(node(0, "SequenceConstruct", vec![Some(a)], vec![seq_a]));

    let mut body = Graph::new();
    let ea = body.create_named_value("ea", DataType::Float32, Shape::new());
    body.mark_value_type_unknown(ea);
    body.mark_value_shape_unknown(ea);
    body.add_input(ea);
    let eb = body.create_named_value("eb", DataType::Float32, Shape::new());
    body.mark_value_type_unknown(eb);
    body.mark_value_shape_unknown(eb);
    body.add_input(eb);
    let sum = body.create_named_value("sum", DataType::Float32, Shape::new());
    body.mark_value_type_unknown(sum);
    body.mark_value_shape_unknown(sum);
    body.insert_node(node(0, "Add", vec![Some(ea), Some(eb)], vec![sum]));
    body.add_output(sum);

    let map_out = graph.create_named_value("map_out", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(map_out);
    graph.mark_value_shape_unknown(map_out);
    let map_node = graph.insert_node(node(
        1,
        "SequenceMap",
        vec![Some(seq_a), Some(bias)],
        vec![map_out],
    ));
    graph.subgraphs.insert((map_node, "body".into()), body);
    graph.add_output(map_out);
    graph.opset_imports.insert(String::new(), 21);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    let report = registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer SequenceMap with additional tensor input");

    let tensor = sequence_tensor(report.containers.get(&map_out).expect("container output"));
    assert_eq!(tensor.dtype, DataType::Float32);
    assert_eq!(
        tensor.shape,
        Some(vec![DimExpr::constant(2), DimExpr::constant(3)]),
        "a whole-tensor additional input broadcasts against each element"
    );
}

/// A branch that simply returns a container captured by name from the outer
/// scope (no formal inputs, no ops) — proving the capture threaded the
/// container's `ValueType` into the body.
fn captured_sequence_branch(name: &str) -> Graph {
    let mut branch = Graph::new();
    let capture = branch.create_named_value(name, DataType::Float32, Shape::new());
    branch.mark_value_type_unknown(capture);
    branch.mark_value_shape_unknown(capture);
    branch.add_output(capture);
    branch
}

#[test]
fn if_branches_capture_outer_scope_sequence() {
    // seq0 is built in the outer graph, then both If branches return it by
    // lexical capture. Without the remap_node_io container fix, the captured
    // value would lose its type; with it, the If output is a sequence.
    let mut graph = Graph::new();
    let condition = graph.create_named_value("condition", DataType::Bool, Shape::new());
    graph.add_input(condition);
    let elem = graph.create_named_value(
        "elem",
        DataType::Float32,
        vec![Dim::Static(4), Dim::Static(5)],
    );
    graph.add_input(elem);
    let seq0 = graph.create_named_value("seq0", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(seq0);
    graph.mark_value_shape_unknown(seq0);
    graph.insert_node(node(0, "SequenceConstruct", vec![Some(elem)], vec![seq0]));

    let output = graph.create_named_value("output", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(output);
    graph.mark_value_shape_unknown(output);
    let if_node = graph.insert_node(node(1, "If", vec![Some(condition)], vec![output]));
    graph.subgraphs.insert(
        (if_node, "then_branch".into()),
        captured_sequence_branch("seq0"),
    );
    graph.subgraphs.insert(
        (if_node, "else_branch".into()),
        captured_sequence_branch("seq0"),
    );
    graph.add_output(output);
    graph.opset_imports.insert(String::new(), 21);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    let report = registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer If capturing an outer sequence");

    let tensor = sequence_tensor(report.containers.get(&output).expect("container output"));
    assert_eq!(tensor.dtype, DataType::Float32);
    assert_eq!(
        tensor.shape,
        Some(vec![DimExpr::constant(4), DimExpr::constant(5)]),
        "the captured outer sequence's element type crosses the branch boundary"
    );
}

#[test]
fn scan_carries_sequence_state_variable() {
    // A Scan with one container state var and one scan input. The body erases
    // from the carried sequence (state) and passes a scan slice through. The
    // final state output must be a sequence with the element type preserved.
    let mut graph = Graph::new();
    let elem = graph.create_named_value(
        "elem",
        DataType::Float32,
        vec![Dim::Static(2), Dim::Static(3)],
    );
    graph.add_input(elem);
    let seq0 = graph.create_named_value("seq0", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(seq0);
    graph.mark_value_shape_unknown(seq0);
    graph.insert_node(node(0, "SequenceConstruct", vec![Some(elem)], vec![seq0]));
    // Scan input: [T, 4] tensor sliced per-iteration to [4].
    let scan_in = graph.create_named_value(
        "scan_in",
        DataType::Float32,
        vec![Dim::Static(6), Dim::Static(4)],
    );
    graph.add_input(scan_in);

    // Body: (state_seq, scan_slice) -> (state_seq_out, scan_out).
    let mut body = Graph::new();
    let state_in = body.create_named_value("state_in", DataType::Float32, Shape::new());
    body.mark_value_type_unknown(state_in);
    body.mark_value_shape_unknown(state_in);
    body.add_input(state_in);
    let slice_in = body.create_named_value("slice_in", DataType::Float32, Shape::new());
    body.mark_value_type_unknown(slice_in);
    body.mark_value_shape_unknown(slice_in);
    body.add_input(slice_in);
    let state_out = body.create_named_value("state_out", DataType::Float32, Shape::new());
    body.mark_value_type_unknown(state_out);
    body.mark_value_shape_unknown(state_out);
    body.insert_node(node(
        0,
        "SequenceErase",
        vec![Some(state_in)],
        vec![state_out],
    ));
    let scan_out = body.create_named_value("scan_out", DataType::Float32, Shape::new());
    body.insert_node(node(1, "Identity", vec![Some(slice_in)], vec![scan_out]));
    body.add_output(state_out);
    body.add_output(scan_out);

    let final_state = graph.create_named_value("final_state", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(final_state);
    graph.mark_value_shape_unknown(final_state);
    let scan_result = graph.create_named_value("scan_result", DataType::Float32, Shape::new());
    graph.mark_value_type_unknown(scan_result);
    graph.mark_value_shape_unknown(scan_result);
    let mut scan_node = node(
        1,
        "Scan",
        vec![Some(seq0), Some(scan_in)],
        vec![final_state, scan_result],
    );
    scan_node
        .attributes
        .insert("num_scan_inputs".into(), Attribute::Int(1));
    let scan_id = graph.insert_node(scan_node);
    graph.subgraphs.insert((scan_id, "body".into()), body);
    graph.add_output(final_state);
    graph.add_output(scan_result);
    graph.opset_imports.insert(String::new(), 21);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    let report = registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer Scan with a sequence state var");

    let tensor = sequence_tensor(
        report
            .containers
            .get(&final_state)
            .expect("state is a sequence"),
    );
    assert_eq!(tensor.dtype, DataType::Float32);
    assert_eq!(
        tensor.shape,
        Some(vec![DimExpr::constant(2), DimExpr::constant(3)]),
        "the sequence state element type is preserved across the Scan"
    );
    // The scan output stays a plain stacked tensor (not a container).
    assert!(
        !report.containers.contains_key(&scan_result),
        "a scan output stacks tensors and is never a container"
    );
}

/// Whether inference recorded a `(loser, winner)` unification between `x` and
/// `y` (in either order) on the graph's authoritative record.
fn unifies(graph: &Graph, x: onnx_runtime_ir::SymbolId, y: onnx_runtime_ir::SymbolId) -> bool {
    graph
        .symbol_unifications
        .iter()
        .any(|&(a, b)| (a == x && b == y) || (a == y && b == x))
}

fn derives(
    graph: &Graph,
    derived: onnx_runtime_ir::SymbolId,
    source: onnx_runtime_ir::SymbolId,
) -> bool {
    graph
        .symbol_derivations
        .iter()
        .any(|&(d, s)| d == derived && s == source)
}

/// Fresh symbols minted during inference live at or above this id; named graph
/// dim-params stay below it. Mirrors `ANON_SYMBOL_FLOOR` in the crate.
const ANON_FLOOR: u32 = 0x8000_0000;

fn sole_symbol(dim: &Dim) -> onnx_runtime_ir::SymbolId {
    match dim {
        Dim::Symbolic(s) => *s,
        other => panic!("expected a symbolic dim, got {other:?}"),
    }
}

// Path-B completeness: the authoritative symbol-lineage records are populated at
// the single `broadcast_dim` chokepoint, so they capture symbol substitutions
// from EVERY broadcasting handler — not just elementwise. These exercise the two
// non-elementwise handlers the prior executor closure missed (`MatMul` batch
// dims via `linalg.rs`, `Concat` non-concat axes via `movement/concat_slice.rs`);
// `Einsum`/`Expand` funnel through the same chokepoint and are covered by
// construction.
//
// Which record gets written depends on whether collapsing the two dims onto one
// representative is *sound*, which these two pairs of tests pin:
//
//   * anonymous vs named  -> `symbol_unifications`, output keeps the named symbol
//   * named vs named      -> `symbol_derivations`, output is a fresh unknown
#[test]
fn broadcast_unifies_anonymous_symbol_into_named_for_matmul_batch_dims() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let a = graph.intern_symbol("a");
    // An anonymous extent, as minted for a value inference could not track.
    let anon = onnx_runtime_ir::SymbolId(ANON_FLOOR + 7);
    let lhs = graph.create_named_value(
        "lhs",
        DataType::Float32,
        vec![Dim::Symbolic(a), Dim::Static(8), Dim::Static(16)],
    );
    let rhs = graph.create_named_value(
        "rhs",
        DataType::Float32,
        vec![Dim::Symbolic(anon), Dim::Static(16), Dim::Static(32)],
    );
    graph.add_input(lhs);
    graph.add_input(rhs);
    let out = graph.create_named_value("out", DataType::Float32, Shape::new());
    graph.mark_value_shape_unknown(out);
    graph.insert_node(node(0, "MatMul", vec![Some(lhs), Some(rhs)], vec![out]));
    graph.add_output(out);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer MatMul with an anonymous batch dim");

    // The anonymous side carries no independent meaning, so it adopts the named
    // symbol — this is what lets a data-dependent extent re-bind to a real dim.
    assert_eq!(
        graph.value(out).shape,
        vec![Dim::Symbolic(a), Dim::Static(8), Dim::Static(32)],
        "an anonymous batch dim must re-bind to the named graph symbol"
    );
    assert!(
        unifies(&graph, a, anon),
        "the MatMul batch-dim broadcast must be recorded as an authoritative unification, got {:?}",
        graph.symbol_unifications
    );
}

#[test]
fn broadcast_of_two_named_dims_stays_unknown_for_matmul_batch_dims() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let a = graph.intern_symbol("a");
    let b = graph.intern_symbol("b");
    let lhs = graph.create_named_value(
        "lhs",
        DataType::Float32,
        vec![Dim::Symbolic(a), Dim::Static(8), Dim::Static(16)],
    );
    let rhs = graph.create_named_value(
        "rhs",
        DataType::Float32,
        vec![Dim::Symbolic(b), Dim::Static(16), Dim::Static(32)],
    );
    graph.add_input(lhs);
    graph.add_input(rhs);
    let out = graph.create_named_value("out", DataType::Float32, Shape::new());
    graph.mark_value_shape_unknown(out);
    graph.insert_node(node(0, "MatMul", vec![Some(lhs), Some(rhs)], vec![out]));
    graph.add_output(out);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer MatMul with distinct symbolic batch dims");

    // `a` and `b` are separately declared dims: either may be the 1 that
    // broadcasts up, so naming the result after either one would be a guess.
    let shape = &graph.value(out).shape;
    assert_eq!(shape[1..], [Dim::Static(8), Dim::Static(32)]);
    let fresh = sole_symbol(&shape[0]);
    assert!(
        fresh != a && fresh != b && fresh.0 >= ANON_FLOOR,
        "broadcasting two distinct named dims must yield a fresh unknown, got {fresh:?}"
    );
    assert!(
        !unifies(&graph, a, b),
        "two separately declared graph dims must not be asserted equal, got {:?}",
        graph.symbol_unifications
    );
    assert!(
        derives(&graph, fresh, a) && derives(&graph, fresh, b),
        "the unknown extent must record provenance on BOTH operands so a growing \
         dim still disqualifies it from capture, got {:?}",
        graph.symbol_derivations
    );
}

#[test]
fn broadcast_records_symbol_unification_for_concat_non_concat_axes() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let a = graph.intern_symbol("a");
    let anon = onnx_runtime_ir::SymbolId(ANON_FLOOR + 7);
    // Concatenate along axis 1; the non-concat axis 0 broadcasts a and anon.
    let in0 = graph.create_named_value(
        "in0",
        DataType::Float32,
        vec![Dim::Symbolic(a), Dim::Static(3)],
    );
    let in1 = graph.create_named_value(
        "in1",
        DataType::Float32,
        vec![Dim::Symbolic(anon), Dim::Static(5)],
    );
    graph.add_input(in0);
    graph.add_input(in1);
    let out = graph.create_named_value("out", DataType::Float32, Shape::new());
    graph.mark_value_shape_unknown(out);
    let mut concat = node(0, "Concat", vec![Some(in0), Some(in1)], vec![out]);
    concat.attributes.insert("axis".into(), Attribute::Int(1));
    graph.insert_node(concat);
    graph.add_output(out);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer Concat with distinct symbolic non-concat dims");

    assert_eq!(
        graph.value(out).shape,
        vec![Dim::Symbolic(a), Dim::Static(8)],
        "Concat output non-concat axis must re-bind the anonymous dim to the named one"
    );
    assert!(
        unifies(&graph, a, anon),
        "the Concat non-concat-axis broadcast must be recorded as an authoritative unification, \
         got {:?}",
        graph.symbol_unifications
    );
}

// Regression: an export whose rank-3 `MatMul` rhs right-aligns under a rank-4
// lhs makes the rhs *batch* dim broadcast against the lhs *sequence* dim. That
// is only valid because batch is 1 at runtime — so collapsing the two onto the
// lower-id `batch` silently rewrites every downstream sequence extent into a
// batch extent. With batch == 1 and a one-token step the lie is invisible; it
// only surfaces once a step is wider than one token (speculative verify), where
// a kernel then rejects `4 vs 1` sequence dims.
#[test]
fn rank_mismatched_matmul_does_not_collapse_sequence_onto_batch() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let batch = graph.intern_symbol("batch_size");
    let seq = graph.intern_symbol("sequence_length");
    let total = graph.intern_symbol("total_sequence_length");
    // [batch, seq, 2, 8] @ [batch, 8, total]
    let lhs = graph.create_named_value(
        "lhs",
        DataType::Float32,
        vec![
            Dim::Symbolic(batch),
            Dim::Symbolic(seq),
            Dim::Static(2),
            Dim::Static(8),
        ],
    );
    let rhs = graph.create_named_value(
        "rhs",
        DataType::Float32,
        vec![Dim::Symbolic(batch), Dim::Static(8), Dim::Symbolic(total)],
    );
    graph.add_input(lhs);
    graph.add_input(rhs);
    let out = graph.create_named_value("out", DataType::Float32, Shape::new());
    graph.mark_value_shape_unknown(out);
    graph.insert_node(node(0, "MatMul", vec![Some(lhs), Some(rhs)], vec![out]));
    graph.add_output(out);

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect("infer rank-mismatched MatMul");

    let shape = &graph.value(out).shape;
    assert_eq!(
        shape[0],
        Dim::Symbolic(batch),
        "the leading batch dim is unambiguous and must be preserved"
    );
    assert_eq!(shape[2..], [Dim::Static(2), Dim::Symbolic(total)]);
    assert_ne!(
        shape[1],
        Dim::Symbolic(batch),
        "the sequence axis must not be rewritten into the batch symbol"
    );
    assert!(
        !unifies(&graph, batch, seq),
        "batch and sequence_length must never be asserted equal, got {:?}",
        graph.symbol_unifications
    );
}
