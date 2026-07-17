//! Integration tests for the control-flow operators `If`, `Loop`, and `Scan`.
//!
//! Each test hand-builds a small parent [`Graph`] plus one or more nested body
//! subgraphs via the IR API, registers the bodies in `graph.subgraphs` keyed by
//! `(control_flow_node_id, attribute_name)` exactly as the loader would, then
//! runs the model through the public [`InferenceSession`] surface and checks the
//! numeric result against a reference computed here in the test.

use onnx_runtime_ir::{
    static_shape, Attribute, DataType, Dim, Graph, Node, NodeId, Shape, TensorData, ValueId, WeightRef,
};
use onnx_runtime_session::{InferenceSession, Tensor};

// --- construction helpers ---------------------------------------------------

/// A rank-0 (scalar) static shape.
fn scalar() -> Shape {
    static_shape(std::iter::empty::<usize>())
}

fn f32_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i64_bytes(data: &[i64]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Add a named graph input, returning its value id.
fn input(g: &mut Graph, name: &str, dtype: DataType, dims: &[usize]) -> ValueId {
    let vid = g.create_named_value(name, dtype, static_shape(dims.iter().copied()));
    g.add_input(vid);
    vid
}

/// Add an inline initializer of the given dtype, returning its value id.
fn init(g: &mut Graph, name: &str, dtype: DataType, dims: &[usize], bytes: Vec<u8>) -> ValueId {
    let vid = g.create_named_value(name, dtype, static_shape(dims.iter().copied()));
    g.set_initializer(vid, WeightRef::Inline(TensorData::from_raw(dtype, dims.to_vec(), bytes)));
    vid
}

/// A producer-less named value with no source: a *capture* to be bound from the
/// enclosing scope when the body runs. Must match an outer value by name.
fn capture(g: &mut Graph, name: &str, dtype: DataType, dims: &[usize]) -> ValueId {
    g.create_named_value(name, dtype, static_shape(dims.iter().copied()))
}

/// Insert an op node producing a single output value, returning that value id.
fn op(
    g: &mut Graph,
    op_type: &str,
    inputs: &[ValueId],
    out_name: Option<&str>,
    out_dtype: DataType,
    out_dims: &[usize],
    attrs: &[(&str, Attribute)],
) -> ValueId {
    let out = match out_name {
        Some(n) => g.create_named_value(n, out_dtype, static_shape(out_dims.iter().copied())),
        None => g.create_value(out_dtype, static_shape(out_dims.iter().copied())),
    };
    let mut node = Node::new(
        NodeId(0),
        op_type,
        inputs.iter().map(|&v| Some(v)).collect(),
        vec![out],
    );
    for (k, v) in attrs {
        node.attributes.insert((*k).to_string(), v.clone());
    }
    g.insert_node(node);
    out
}

/// Insert a control-flow node (variadic inputs, some possibly `None`) with the
/// given output value ids, returning its assigned [`NodeId`].
fn control_flow_node(
    g: &mut Graph,
    op_type: &str,
    inputs: Vec<Option<ValueId>>,
    outputs: Vec<ValueId>,
    attrs: &[(&str, Attribute)],
) -> NodeId {
    let mut node = Node::new(NodeId(0), op_type, inputs, outputs);
    for (k, v) in attrs {
        node.attributes.insert((*k).to_string(), v.clone());
    }
    g.insert_node(node)
}

/// Register a body subgraph under `(node_id, attr_key)` — the key the executor
/// looks up when it runs the control-flow node.
fn register(parent: &mut Graph, node_id: NodeId, attr_key: &str, mut body: Graph) {
    body.opset_imports.entry(String::new()).or_insert(17);
    parent.subgraphs.insert((node_id, attr_key.to_string()), body);
}

fn new_parent() -> Graph {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    g
}

// --- If ---------------------------------------------------------------------

/// Build an `If` branch body: `out = X <op> ones`, capturing outer value `X`
/// and using an inline `ones` initializer. Zero formal inputs (per ONNX `If`).
fn if_branch(bin_op: &str) -> Graph {
    let mut b = Graph::new();
    let x = capture(&mut b, "X", DataType::Float32, &[2]);
    let ones = init(&mut b, "ones", DataType::Float32, &[2], f32_bytes(&[1.0, 1.0]));
    let out = op(&mut b, bin_op, &[x, ones], Some("branch_out"), DataType::Float32, &[2], &[]);
    b.add_output(out);
    b
}

#[test]
fn if_executes_selected_branch_with_capture_and_inline_initializer() {
    let mut g = new_parent();
    let cond = input(&mut g, "cond", DataType::Bool, &[1]);
    let _x = input(&mut g, "X", DataType::Float32, &[2]);
    let y = g.create_named_value("Y", DataType::Float32, static_shape([2]));

    let node = control_flow_node(&mut g, "If", vec![Some(cond)], vec![y], &[]);
    // then: X + 1, else: X - 1.
    register(&mut g, node, "then_branch", if_branch("Add"));
    register(&mut g, node, "else_branch", if_branch("Sub"));
    g.add_output(y);

    let mut session = InferenceSession::from_graph(g).expect("build session");
    for (run, (cond_val, expected)) in [
        (true, [3.0f32, 4.0]),
        (false, [1.0f32, 2.0]),
        (true, [3.0f32, 4.0]),
        (false, [1.0f32, 2.0]),
    ]
    .into_iter()
    .enumerate()
    {
        let cond_t = Tensor::from_raw(DataType::Bool, vec![1], &[cond_val as u8]).unwrap();
        let x_t = Tensor::from_f32(&[2], &[2.0, 3.0]).unwrap();
        let outs = session.run(&[("cond", &cond_t), ("X", &x_t)]).expect("run");
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].to_vec_f32(), expected.to_vec(), "cond={cond_val}");
        let stats = session.control_flow_stats();
        assert_eq!(stats.subgraph_builds, ((run + 1).min(2)) as u64);
        assert_eq!(stats.subgraph_runs, (run + 1) as u64);
    }
}

#[test]
fn if_rejects_mismatched_branch_output_counts_before_running_selected_branch() {
    let mut g = new_parent();
    let cond = input(&mut g, "cond", DataType::Bool, &[]);
    let _x = input(&mut g, "X", DataType::Float32, &[2]);
    let y = g.create_named_value("Y", DataType::Float32, static_shape([2]));
    let node = control_flow_node(&mut g, "If", vec![Some(cond)], vec![y], &[]);

    let then_branch = if_branch("Add");
    let mut else_branch = if_branch("Sub");
    let extra = capture(&mut else_branch, "X", DataType::Float32, &[2]);
    else_branch.add_output(extra);
    register(&mut g, node, "then_branch", then_branch);
    register(&mut g, node, "else_branch", else_branch);
    g.add_output(y);

    let mut session = InferenceSession::from_graph(g).expect("build session");
    let cond_t = Tensor::from_raw(DataType::Bool, vec![], &[1]).unwrap();
    let x_t = Tensor::from_f32(&[2], &[2.0, 3.0]).unwrap();
    let err = session
        .run(&[("cond", &cond_t), ("X", &x_t)])
        .expect_err("mismatched branches must fail even when then_branch is selected");
    assert!(
        err.to_string().contains(
            "control-flow op If: branches declare different output counts: then_branch has 1, \
             else_branch has 2"
        ),
        "unexpected error: {err}"
    );
    assert_eq!(session.control_flow_stats().subgraph_runs, 0);
}

#[test]
fn if_rejects_mismatched_branch_output_dtypes() {
    let mut g = new_parent();
    let cond = input(&mut g, "cond", DataType::Bool, &[]);
    let _x = input(&mut g, "X", DataType::Float32, &[2]);
    let y = g.create_named_value("Y", DataType::Float32, static_shape([2]));
    let node = control_flow_node(&mut g, "If", vec![Some(cond)], vec![y], &[]);

    let then_branch = if_branch("Add");
    let mut else_branch = Graph::new();
    let x = capture(&mut else_branch, "X", DataType::Float32, &[2]);
    let out = op(
        &mut else_branch,
        "Cast",
        &[x],
        Some("branch_out"),
        DataType::Int64,
        &[2],
        &[("to", Attribute::Int(DataType::Int64 as i64))],
    );
    else_branch.add_output(out);
    register(&mut g, node, "then_branch", then_branch);
    register(&mut g, node, "else_branch", else_branch);
    g.add_output(y);

    let mut session = InferenceSession::from_graph(g).expect("build session");
    let cond_t = Tensor::from_raw(DataType::Bool, vec![], &[1]).unwrap();
    let x_t = Tensor::from_f32(&[2], &[2.0, 3.0]).unwrap();
    let err = session
        .run(&[("cond", &cond_t), ("X", &x_t)])
        .expect_err("mismatched branch dtypes must fail");
    assert!(
        err.to_string().contains(
            "control-flow op If: branches declare different dtypes for output 0: \
             then_branch is Float32, else_branch is Int64"
        ),
        "unexpected error: {err}"
    );
    assert_eq!(session.control_flow_stats().subgraph_runs, 0);
}

#[test]
fn if_rejects_non_bool_and_multi_element_conditions() {
    let build = |cond_dtype: DataType, cond_dims: &[usize]| {
        let mut g = new_parent();
        let cond = input(&mut g, "cond", cond_dtype, cond_dims);
        let _x = input(&mut g, "X", DataType::Float32, &[2]);
        let y = g.create_named_value("Y", DataType::Float32, static_shape([2]));
        let node = control_flow_node(&mut g, "If", vec![Some(cond)], vec![y], &[]);
        register(&mut g, node, "then_branch", if_branch("Add"));
        register(&mut g, node, "else_branch", if_branch("Sub"));
        g.add_output(y);
        InferenceSession::from_graph(g).expect("build session")
    };

    let x_t = Tensor::from_f32(&[2], &[2.0, 3.0]).unwrap();
    let mut wrong_dtype = build(DataType::Float32, &[]);
    let float_cond = Tensor::from_f32(&[], &[1.0]).unwrap();
    let err = wrong_dtype
        .run(&[("cond", &float_cond), ("X", &x_t)])
        .expect_err("non-bool If cond must fail");
    assert!(
        err.to_string()
            .contains("input If cond: dtype mismatch (expected Bool, got Float32)"),
        "unexpected error: {err}"
    );

    let mut wrong_shape = build(DataType::Bool, &[2]);
    let vector_cond = Tensor::from_raw(DataType::Bool, vec![2], &[1, 0]).unwrap();
    let err = wrong_shape
        .run(&[("cond", &vector_cond), ("X", &x_t)])
        .expect_err("multi-element If cond must fail");
    assert!(
        err.to_string().contains(
            "control-flow op If: 'cond' must be a BOOL scalar or single-element tensor, \
             got shape [2]"
        ),
        "unexpected error: {err}"
    );
}

#[test]
fn if_rebuilds_subgraph_for_shape_varying_capture() {
    // Consecutive runs bind this captured value to distinct concrete shapes.
    let mut g = new_parent();
    let batch = g.intern_symbol("batch");
    let shape = vec![Dim::Symbolic(batch)];
    let cond = input(&mut g, "cond", DataType::Bool, &[]);
    let x = g.create_named_value("X", DataType::Float32, shape.clone());
    g.add_input(x);
    let y = g.create_named_value("Y", DataType::Float32, shape);
    let node = control_flow_node(&mut g, "If", vec![Some(cond)], vec![y], &[]);

    let mut branch = Graph::new();
    let x = capture(&mut branch, "X", DataType::Float32, &[1]);
    let out = op(&mut branch, "Identity", &[x], Some("branch_out"), DataType::Float32, &[1], &[]);
    branch.add_output(out);
    register(&mut g, node, "then_branch", branch);
    register(&mut g, node, "else_branch", if_branch("Identity"));
    g.add_output(y);

    let mut session = InferenceSession::from_graph(g).expect("build session");
    let cond_t = Tensor::from_raw(DataType::Bool, vec![], &[1]).unwrap();
    for (values, expected_shape) in [(&[1.0, 2.0][..], vec![2]), (&[3.0, 4.0, 5.0][..], vec![3])] {
        let x_t = Tensor::from_f32(&expected_shape, values).unwrap();
        let outs = session.run(&[("cond", &cond_t), ("X", &x_t)]).expect("run");
        assert_eq!(outs[0].shape, expected_shape);
        assert_eq!(outs[0].to_vec_f32(), values);
    }

    let stats = session.control_flow_stats();
    assert!(stats.subgraph_builds > 1, "shape changes must rebuild the subgraph executor");
    assert_eq!(stats.subgraph_builds, 2);
    assert_eq!(stats.subgraph_runs, 2);
}

// --- Loop -------------------------------------------------------------------

/// Loop body accumulating `iter_num` into a running f32 sum and emitting the
/// updated sum as a per-iteration scan output.
///
/// `(iter_num i64, cond_in bool, acc f32) -> (cond_out bool, acc_out f32, scan f32)`
fn loop_sum_body() -> Graph {
    let mut b = Graph::new();
    let iter = capture(&mut b, "iter_num", DataType::Int64, &[]);
    let cond_in = capture(&mut b, "cond_in", DataType::Bool, &[]);
    let acc = capture(&mut b, "acc", DataType::Float32, &[]);
    b.add_input(iter);
    b.add_input(cond_in);
    b.add_input(acc);

    let iter_f = op(&mut b, "Cast", &[iter], Some("iter_f"), DataType::Float32, &[], &[(
        "to",
        Attribute::Int(DataType::Float32 as i64),
    )]);
    let acc_out = op(&mut b, "Add", &[acc, iter_f], Some("acc_out"), DataType::Float32, &[], &[]);
    let cond_out = op(&mut b, "Identity", &[cond_in], Some("cond_out"), DataType::Bool, &[], &[]);
    let scan = op(&mut b, "Identity", &[acc_out], Some("scan_out"), DataType::Float32, &[], &[]);
    b.add_output(cond_out);
    b.add_output(acc_out);
    b.add_output(scan);
    b
}

#[test]
fn loop_fixed_trip_count_accumulates_and_stacks_scan_outputs() {
    let n = 4i64;
    let mut g = new_parent();
    let m = init(&mut g, "M", DataType::Int64, &[], i64_bytes(&[n]));
    let cond = init(&mut g, "cond", DataType::Bool, &[], vec![1u8]);
    let acc0 = init(&mut g, "acc0", DataType::Float32, &[], f32_bytes(&[0.0]));
    let acc_final = g.create_named_value("acc_final", DataType::Float32, scalar());
    let scan = g.create_named_value("scan", DataType::Float32, static_shape([n as usize]));

    let node = control_flow_node(
        &mut g,
        "Loop",
        vec![Some(m), Some(cond), Some(acc0)],
        vec![acc_final, scan],
        &[],
    );
    register(&mut g, node, "body", loop_sum_body());
    g.add_output(acc_final);
    g.add_output(scan);

    let mut session = InferenceSession::from_graph(g).expect("build session");
    let outs = session.run(&[]).expect("run");
    assert_eq!(outs.len(), 2);
    // acc after each iter: 0, 1, 3, 6  → final 6.
    assert_eq!(outs[0].to_vec_f32(), vec![6.0]);
    assert_eq!(outs[1].dtype, DataType::Float32);
    assert_eq!(outs[1].shape, vec![n as usize]);
    assert_eq!(outs[1].to_vec_f32(), vec![0.0, 1.0, 3.0, 6.0]);
    assert_eq!(session.control_flow_stats().subgraph_builds, 1);
    assert_eq!(session.control_flow_stats().subgraph_runs, n as u64);
}

/// Loop body decrementing a counter and driving `cond_out` from it via a
/// float→bool Cast (nonzero → keep looping). Tests cond-driven early exit and
/// an omitted (unbounded) `M` input.
///
/// `(iter i64, cond_in bool, rem f32) -> (cond_out bool, rem_out f32)`
fn loop_countdown_body() -> Graph {
    let mut b = Graph::new();
    let iter = capture(&mut b, "iter", DataType::Int64, &[]);
    let cond_in = capture(&mut b, "cond_in", DataType::Bool, &[]);
    let rem = capture(&mut b, "rem", DataType::Float32, &[]);
    b.add_input(iter);
    b.add_input(cond_in);
    b.add_input(rem);

    let one = init(&mut b, "one", DataType::Float32, &[], f32_bytes(&[1.0]));
    let rem_out = op(&mut b, "Sub", &[rem, one], Some("rem_out"), DataType::Float32, &[], &[]);
    let cond_out = op(&mut b, "Cast", &[rem_out], Some("cond_out"), DataType::Bool, &[], &[(
        "to",
        Attribute::Int(DataType::Bool as i64),
    )]);
    b.add_output(cond_out);
    b.add_output(rem_out);
    b
}

#[test]
fn loop_cond_driven_early_exit_with_unbounded_trip_count() {
    let mut g = new_parent();
    let cond = init(&mut g, "cond", DataType::Bool, &[], vec![1u8]);
    let rem0 = init(&mut g, "rem0", DataType::Float32, &[], f32_bytes(&[3.0]));
    let rem_final = g.create_named_value("rem_final", DataType::Float32, scalar());

    // M omitted (None) → unbounded; loop stops when body's cond_out goes false.
    let node = control_flow_node(
        &mut g,
        "Loop",
        vec![None, Some(cond), Some(rem0)],
        vec![rem_final],
        &[],
    );
    register(&mut g, node, "body", loop_countdown_body());
    g.add_output(rem_final);

    let mut session = InferenceSession::from_graph(g).expect("build session");
    let outs = session.run(&[]).expect("run");
    assert_eq!(outs.len(), 1);
    // rem after each iter: 2, 1, 0 → cond false at 0, three iterations, final 0.
    assert_eq!(outs[0].to_vec_f32(), vec![0.0]);
}

fn loop_zero_iteration_body() -> Graph {
    let mut b = Graph::new();
    let iter = capture(&mut b, "iter", DataType::Int64, &[]);
    let cond_in = capture(&mut b, "cond_in", DataType::Bool, &[]);
    let carried = capture(&mut b, "carried", DataType::Float32, &[]);
    b.add_input(iter);
    b.add_input(cond_in);
    b.add_input(carried);

    let cond_out = op(&mut b, "Identity", &[cond_in], Some("cond_out"), DataType::Bool, &[], &[]);
    let carried_out =
        op(&mut b, "Identity", &[carried], Some("carried_out"), DataType::Float32, &[], &[]);
    let pair = init(&mut b, "pair", DataType::Int64, &[2], i64_bytes(&[7, 9]));
    let scan = op(&mut b, "Identity", &[pair], Some("scan_out"), DataType::Int64, &[2], &[]);
    b.add_output(cond_out);
    b.add_output(carried_out);
    b.add_output(scan);
    b
}

#[test]
fn loop_zero_iterations_preserves_carried_and_types_empty_scan_output() {
    let mut g = new_parent();
    let m = init(&mut g, "M", DataType::Int64, &[], i64_bytes(&[0]));
    let carried0 = init(&mut g, "carried0", DataType::Float32, &[], f32_bytes(&[42.0]));
    let carried_final = g.create_named_value("carried_final", DataType::Float32, scalar());
    let scan = g.create_named_value("scan", DataType::Int64, static_shape([0, 2]));

    let node = control_flow_node(
        &mut g,
        "Loop",
        vec![Some(m), None, Some(carried0)],
        vec![carried_final, scan],
        &[],
    );
    register(&mut g, node, "body", loop_zero_iteration_body());
    g.add_output(carried_final);
    g.add_output(scan);

    let mut session = InferenceSession::from_graph(g).expect("build session");
    let outs = session.run(&[]).expect("run");
    assert_eq!(outs[0].to_vec_f32(), vec![42.0]);
    assert_eq!(outs[1].dtype, DataType::Int64);
    assert_eq!(outs[1].shape, vec![0, 2]);
    assert!(outs[1].to_vec_i64().is_empty());
    assert_eq!(session.control_flow_stats().subgraph_builds, 0);
    assert_eq!(session.control_flow_stats().subgraph_runs, 0);
}

fn loop_capture_body() -> Graph {
    let mut b = Graph::new();
    let iter = capture(&mut b, "iter", DataType::Int64, &[]);
    let cond_in = capture(&mut b, "cond_in", DataType::Bool, &[]);
    let acc = capture(&mut b, "acc", DataType::Float32, &[]);
    let step = capture(&mut b, "step", DataType::Float32, &[]);
    b.add_input(iter);
    b.add_input(cond_in);
    b.add_input(acc);

    let acc_out = op(&mut b, "Add", &[acc, step], Some("acc_out"), DataType::Float32, &[], &[]);
    let cond_out = op(&mut b, "Identity", &[cond_in], Some("cond_out"), DataType::Bool, &[], &[]);
    let scan = op(&mut b, "Identity", &[acc_out], Some("scan_out"), DataType::Float32, &[], &[]);
    b.add_output(cond_out);
    b.add_output(acc_out);
    b.add_output(scan);
    b
}

#[test]
fn loop_body_captures_outer_value_and_reuses_child_executor() {
    let n = 3i64;
    let mut g = new_parent();
    let m = init(&mut g, "M", DataType::Int64, &[], i64_bytes(&[n]));
    let acc0 = init(&mut g, "acc0", DataType::Float32, &[], f32_bytes(&[1.0]));
    let _step = input(&mut g, "step", DataType::Float32, &[]);
    let acc_final = g.create_named_value("acc_final", DataType::Float32, scalar());
    let scan = g.create_named_value("scan", DataType::Float32, static_shape([n as usize]));

    let node = control_flow_node(
        &mut g,
        "Loop",
        vec![Some(m), None, Some(acc0)],
        vec![acc_final, scan],
        &[],
    );
    register(&mut g, node, "body", loop_capture_body());
    g.add_output(acc_final);
    g.add_output(scan);

    let mut session = InferenceSession::from_graph(g).expect("build session");
    let step = Tensor::from_f32(&[], &[2.0]).unwrap();
    let outs = session.run(&[("step", &step)]).expect("run");
    assert_eq!(outs[0].to_vec_f32(), vec![7.0]);
    assert_eq!(outs[1].shape, vec![n as usize]);
    assert_eq!(outs[1].to_vec_f32(), vec![3.0, 5.0, 7.0]);
    assert_eq!(session.control_flow_stats().subgraph_builds, 1);
    assert_eq!(session.control_flow_stats().subgraph_runs, n as u64);
}

#[test]
fn loop_many_iterations_accumulates_correctly() {
    // Efficiency-oriented: the child body executor is compiled once and reused
    // across every iteration. A large trip count must still produce the exact
    // arithmetic-series sum.
    let n = 1000i64;
    let mut g = new_parent();
    let m = init(&mut g, "M", DataType::Int64, &[], i64_bytes(&[n]));
    let cond = init(&mut g, "cond", DataType::Bool, &[], vec![1u8]);
    let acc0 = init(&mut g, "acc0", DataType::Float32, &[], f32_bytes(&[0.0]));
    let acc_final = g.create_named_value("acc_final", DataType::Float32, scalar());
    let scan = g.create_named_value("scan", DataType::Float32, static_shape([n as usize]));

    let node = control_flow_node(
        &mut g,
        "Loop",
        vec![Some(m), Some(cond), Some(acc0)],
        vec![acc_final, scan],
        &[],
    );
    register(&mut g, node, "body", loop_sum_body());
    g.add_output(acc_final);
    g.add_output(scan);

    let mut session = InferenceSession::from_graph(g).expect("build session");
    let outs = session.run(&[]).expect("run");
    // sum_{k=0}^{999} k = 999 * 1000 / 2 = 499500.
    assert_eq!(outs[0].to_vec_f32(), vec![499_500.0]);
    let scan_vals = outs[1].to_vec_f32();
    assert_eq!(scan_vals.len(), n as usize);
    assert_eq!(scan_vals[n as usize - 1], 499_500.0);
    assert_eq!(session.control_flow_stats().subgraph_builds, 1);
    assert_eq!(session.control_flow_stats().subgraph_runs, n as u64);
}

// --- Scan -------------------------------------------------------------------

/// Scan body: running-sum state plus per-step emission of the updated state.
///
/// `(state f32[D], x f32[D]) -> (state_out f32[D], scan_out f32[D])`
fn scan_cumsum_body(d: usize) -> Graph {
    let mut b = Graph::new();
    let state = capture(&mut b, "state", DataType::Float32, &[d]);
    let x = capture(&mut b, "x", DataType::Float32, &[d]);
    b.add_input(state);
    b.add_input(x);

    let state_out =
        op(&mut b, "Add", &[state, x], Some("state_out"), DataType::Float32, &[d], &[]);
    let scan_out =
        op(&mut b, "Identity", &[state_out], Some("scan_out"), DataType::Float32, &[d], &[]);
    b.add_output(state_out);
    b.add_output(scan_out);
    b
}

#[test]
fn scan_forward_axis0_threads_state_and_stacks_outputs() {
    let (t, d) = (3usize, 2usize);
    let mut g = new_parent();
    let s0 = init(&mut g, "s0", DataType::Float32, &[d], f32_bytes(&[0.0, 0.0]));
    let x = input(&mut g, "X", DataType::Float32, &[t, d]);
    let s_final = g.create_named_value("s_final", DataType::Float32, static_shape([d]));
    let y = g.create_named_value("Y", DataType::Float32, static_shape([t, d]));

    let node = control_flow_node(
        &mut g,
        "Scan",
        vec![Some(s0), Some(x)],
        vec![s_final, y],
        &[("num_scan_inputs", Attribute::Int(1))],
    );
    register(&mut g, node, "body", scan_cumsum_body(d));
    g.add_output(s_final);
    g.add_output(y);

    let mut session = InferenceSession::from_graph(g).expect("build session");
    let x_t = Tensor::from_f32(&[t, d], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let outs = session.run(&[("X", &x_t)]).expect("run");
    assert_eq!(outs.len(), 2);
    // Running sums: [1,2] -> [4,6] -> [9,12].
    assert_eq!(outs[0].to_vec_f32(), vec![9.0, 12.0]);
    assert_eq!(outs[1].to_vec_f32(), vec![1.0, 2.0, 4.0, 6.0, 9.0, 12.0]);
    assert_eq!(session.control_flow_stats().subgraph_builds, 1);
    assert_eq!(session.control_flow_stats().subgraph_runs, t as u64);
}
