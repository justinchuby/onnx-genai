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

fn if_graph(then_branch: Graph, else_branch: Graph) -> (Graph, ValueId) {
    let mut graph = Graph::new();
    let condition = graph.create_named_value("condition", DataType::Bool, Shape::new());
    graph.add_input(condition);
    let output = graph.create_named_value("output", DataType::Float32, Shape::new());
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
    let mut branch = Graph::new();
    let input = branch.create_named_value("local", DataType::Float32, shape);
    branch.add_input(input);
    let output = branch.create_named_value("branch_output", DataType::Float32, Shape::new());
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
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            vec![2],
            vec![0; 8],
        )),
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

#[test]
fn if_branch_rank_mismatch_is_an_error() {
    let (mut graph, _) = if_graph(
        identity_branch(vec![Dim::Static(2)]),
        identity_branch(vec![Dim::Static(2), Dim::Static(3)]),
    );

    let registry = InferenceRegistry::default_registry();
    let opsets = graph.opset_imports.clone();
    let error = registry
        .infer_graph(&mut graph, &opsets, MergePolicy::Permissive)
        .expect_err("branch rank mismatch must fail");

    assert!(matches!(
        error,
        onnx_runtime_shape_inference::ShapeInferError::Invalid { op, detail }
            if op == "If" && detail.contains("branch output ranks differ")
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
