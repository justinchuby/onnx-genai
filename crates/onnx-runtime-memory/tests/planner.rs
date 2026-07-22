//! Integration tests for the liveness-based activation planner.
//!
//! All graphs are built through the public IR API. Every test validates the
//! produced plan with [`validate_static`].

use onnx_runtime_ir::{DataType, Graph, Node, NodeId, Shape, ValueId, static_shape};
use onnx_runtime_memory::{
    PlanOptions, PlanStatus, SlotId, ValidateError, ViewMap, plan_activations,
    plan_activations_static, validate_static,
};

const F32: DataType = DataType::Float32;

/// Insert an op producing one fresh output value from `inputs`.
fn op(g: &mut Graph, op_type: &str, inputs: &[ValueId], out_shape: Shape) -> ValueId {
    let out = g.create_value(F32, out_shape);
    let ins = inputs.iter().map(|v| Some(*v)).collect();
    g.insert_node(Node::new(NodeId(0), op_type, ins, vec![out]));
    out
}

fn plan_of(g: &Graph, vm: &ViewMap) -> onnx_runtime_memory::ActivationPlan {
    let status = plan_activations_static(g, vm, &PlanOptions::default()).unwrap();
    match status {
        PlanStatus::Complete(p) => {
            validate_static(&p, g, vm, &PlanOptions::default()).expect("plan must validate");
            p
        }
        PlanStatus::Deferred { unknown_sizes } => {
            panic!("unexpected deferred plan: {unknown_sizes:?}")
        }
    }
}

/// Linear chain: in -> a -> b -> c -> out. Four activations but only two are
/// ever concurrently live, so the peak is a 2-slot double buffer, not 4.
#[test]
fn linear_chain_double_buffers() {
    let mut g = Graph::new();
    let inp = g.create_named_value("in", F32, static_shape([100]));
    g.add_input(inp);
    let a = op(&mut g, "Relu", &[inp], static_shape([100]));
    let b = op(&mut g, "Relu", &[a], static_shape([100]));
    let c = op(&mut g, "Relu", &[b], static_shape([100]));
    let out = op(&mut g, "Relu", &[c], static_shape([100]));
    g.add_output(out);

    let vm = ViewMap::new();
    let plan = plan_of(&g, &vm);

    // 4 owners (a, b, c, out), input excluded. Each is 100 * 4 = 400 bytes.
    assert_eq!(plan.naive_bytes, 4 * 400);
    // Concurrent peak is a double buffer.
    assert_eq!(plan.num_slots, 2, "linear chain should reuse into 2 slots");
    assert_eq!(plan.peak_bytes, 2 * 400);
    assert!(
        (plan.savings_ratio - 0.5).abs() < 1e-9,
        "expected 50% savings"
    );
    // The input is not part of the arena.
    assert!(!plan.assignments.contains_key(&inp));
}

/// Diamond: a -> {b, c} -> d. b and c are live at the same time (2 concurrent),
/// but their slots are recycled for later values.
#[test]
fn diamond_needs_two_concurrent() {
    let mut g = Graph::new();
    let inp = g.create_named_value("in", F32, static_shape([10]));
    g.add_input(inp);
    let a = op(&mut g, "Relu", &[inp], static_shape([10]));
    let b = op(&mut g, "Relu", &[a], static_shape([10]));
    let c = op(&mut g, "Relu", &[a], static_shape([10]));
    let d = op(&mut g, "Add", &[b, c], static_shape([10]));
    g.add_output(d);

    let vm = ViewMap::new();
    let plan = plan_of(&g, &vm);

    // b and c overlap -> distinct slots.
    assert_ne!(plan.assignments[&b], plan.assignments[&c]);
    // At least 2 slots are needed for the concurrent pair.
    assert!(plan.num_slots >= 2);
    // Savings still positive vs. naive (4 owners).
    assert!(plan.peak_bytes < plan.naive_bytes);
}

/// Residual/skip: x is produced early and consumed late; its slot must not be
/// recycled for the intermediate values in between.
#[test]
fn residual_skip_pins_source_slot() {
    let mut g = Graph::new();
    let inp = g.create_named_value("in", F32, static_shape([8]));
    g.add_input(inp);
    let x = op(&mut g, "Relu", &[inp], static_shape([8]));
    let h1 = op(&mut g, "Relu", &[x], static_shape([8]));
    let h2 = op(&mut g, "Relu", &[h1], static_shape([8]));
    // Residual add consumes the long-lived x again.
    let out = op(&mut g, "Add", &[h2, x], static_shape([8]));
    g.add_output(out);

    let vm = ViewMap::new();
    let plan = plan_of(&g, &vm);

    // x lives across h1 and h2, so its slot differs from both.
    assert_ne!(plan.assignments[&x], plan.assignments[&h1]);
    assert_ne!(plan.assignments[&x], plan.assignments[&h2]);
}

/// Two independent chains with disjoint lifetimes should share slots.
#[test]
fn disjoint_chains_share_slots() {
    let mut g = Graph::new();
    let inp = g.create_named_value("in", F32, static_shape([16]));
    g.add_input(inp);
    // Chain 1
    let a1 = op(&mut g, "Relu", &[inp], static_shape([16]));
    let a2 = op(&mut g, "Relu", &[a1], static_shape([16]));
    // Chain 2 starts from a2 (so chain 1's early values are dead).
    let b1 = op(&mut g, "Relu", &[a2], static_shape([16]));
    let b2 = op(&mut g, "Relu", &[b1], static_shape([16]));
    g.add_output(b2);

    let vm = ViewMap::new();
    let plan = plan_of(&g, &vm);

    // a1 is dead once a2 is produced; b1 should reuse a1's slot.
    assert_eq!(
        plan.assignments[&a1], plan.assignments[&b1],
        "disjoint lifetimes should share a slot"
    );
    assert_eq!(plan.num_slots, 2);
}

/// A Slice view of a source extends the source's lifetime; the view gets no
/// slot and the source slot is not reused while the view is live.
#[test]
fn view_folding_extends_source_and_owns_no_slot() {
    let mut g = Graph::new();
    let inp = g.create_named_value("in", F32, static_shape([32]));
    g.add_input(inp);
    let src = op(&mut g, "Relu", &[inp], static_shape([32]));
    // A zero-copy Slice view over src (owns no buffer).
    let view = op(&mut g, "Slice", &[src], static_shape([16]));
    // An unrelated value produced while the view is still pending use.
    let other = op(&mut g, "Relu", &[src], static_shape([32]));
    // The view is consumed late, after `other`.
    let out = op(&mut g, "Add", &[other, view], static_shape([32]));
    g.add_output(out);

    let vm = ViewMap::from_pairs([(view, src)]);
    let plan = plan_of(&g, &vm);

    // The view owns no slot.
    assert!(!plan.assignments.contains_key(&view));
    // src is pinned by the still-live view -> its slot is not reused by `other`.
    assert_ne!(plan.assignments[&src], plan.assignments[&other]);
}

/// A chain of views (view of a view) folds to the single root buffer owner.
#[test]
fn view_of_view_folds_to_root() {
    let mut g = Graph::new();
    let inp = g.create_named_value("in", F32, static_shape([64]));
    g.add_input(inp);
    let src = op(&mut g, "Relu", &[inp], static_shape([64]));
    let v1 = op(&mut g, "Reshape", &[src], static_shape([8, 8]));
    let v2 = op(&mut g, "Transpose", &[v1], static_shape([8, 8]));
    let other = op(&mut g, "Relu", &[src], static_shape([64]));
    let out = op(&mut g, "Flatten", &[v2], static_shape([64]));
    g.add_output(out);
    // consume `other` too so it is not dead.
    let out2 = op(&mut g, "Add", &[out, other], static_shape([64]));
    g.add_output(out2);

    // v1 -> src, v2 -> v1 (folds to src).
    let vm = ViewMap::from_pairs([(v1, src), (v2, v1)]);
    assert_eq!(vm.root(v2), src);

    let plan = plan_of(&g, &vm);
    assert!(!plan.assignments.contains_key(&v1));
    assert!(!plan.assignments.contains_key(&v2));
    // src outlives the view chain -> not shared with `other`.
    assert_ne!(plan.assignments[&src], plan.assignments[&other]);
}

/// A graph output tensor is never overwritten: its slot is unique to it.
#[test]
fn graph_output_slot_never_reused() {
    let mut g = Graph::new();
    let inp = g.create_named_value("in", F32, static_shape([4]));
    g.add_input(inp);
    let a = op(&mut g, "Relu", &[inp], static_shape([4]));
    let out = op(&mut g, "Relu", &[a], static_shape([4]));
    g.add_output(out);

    let vm = ViewMap::new();
    let plan = plan_of(&g, &vm);

    // No other owner shares the output's slot.
    let out_slot = plan.assignments[&out];
    let shared = plan
        .assignments
        .iter()
        .filter(|(v, s)| **s == out_slot && **v != out)
        .count();
    assert_eq!(shared, 0, "output slot must be exclusive");
}

/// A symbolic-shaped activation makes the plan deferred.
#[test]
fn symbolic_shape_defers_plan() {
    let mut g = Graph::new();
    let seq = g.intern_symbol("seq");
    let inp = g.create_named_value("in", F32, static_shape([8]));
    g.add_input(inp);
    // dynamic-shaped activation.
    let dyn_shape: Shape = vec![seq.into(), 8usize.into()];
    let a = op(&mut g, "Relu", &[inp], dyn_shape);
    let out = op(&mut g, "Relu", &[a], static_shape([8]));
    g.add_output(out);

    let vm = ViewMap::new();
    let status = plan_activations_static(&g, &vm, &PlanOptions::default()).unwrap();
    match status {
        PlanStatus::Deferred { unknown_sizes } => {
            assert!(
                unknown_sizes.contains(&a),
                "symbolic value must be deferred"
            );
        }
        PlanStatus::Complete(_) => panic!("expected a deferred plan for a symbolic shape"),
    }
}

/// A custom run-time size oracle drives planning when static sizing is absent.
#[test]
fn custom_oracle_enables_runtime_planning() {
    let mut g = Graph::new();
    let seq = g.intern_symbol("seq");
    let inp = g.create_named_value("in", F32, vec![seq.into()]);
    g.add_input(inp);
    let a = op(&mut g, "Relu", &[inp], vec![seq.into()]);
    let out = op(&mut g, "Relu", &[a], vec![seq.into()]);
    g.add_output(out);

    let vm = ViewMap::new();
    // Resolve seq = 128 at run time; every value is 128 * 4 bytes.
    let oracle = |_v: ValueId| Some(128 * 4);
    let status = plan_activations(&g, &vm, oracle, &PlanOptions::default()).unwrap();
    let plan = status.unwrap_complete();
    assert_eq!(plan.num_slots, 2);
    validate_static(&plan, &g, &vm, &PlanOptions::default())
        .expect("static validate should skip symbolic sizes");
}

/// validate() catches a deliberately corrupted plan: two overlapping values
/// forced onto the same slot.
#[test]
fn validate_catches_overlap_conflict() {
    let mut g = Graph::new();
    let inp = g.create_named_value("in", F32, static_shape([4]));
    g.add_input(inp);
    let a = op(&mut g, "Relu", &[inp], static_shape([4]));
    let b = op(&mut g, "Relu", &[a], static_shape([4]));
    let out = op(&mut g, "Add", &[a, b], static_shape([4]));
    g.add_output(out);

    let vm = ViewMap::new();
    let mut plan = plan_of(&g, &vm);

    // Force a and b (which overlap at the node producing b) onto slot 0.
    let bad = SlotId(0);
    plan.assignments.insert(a, bad);
    plan.assignments.insert(b, bad);

    let err = validate_static(&plan, &g, &vm, &PlanOptions::default()).unwrap_err();
    assert!(
        matches!(err, ValidateError::SlotConflict { slot, .. } if slot == bad),
        "expected a SlotConflict, got {err:?}"
    );
}

/// A ~10-node chain demonstrates the peak-memory reduction: peak stays at a
/// double buffer regardless of chain length.
#[test]
fn long_chain_savings_ratio() {
    let mut g = Graph::new();
    let inp = g.create_named_value("in", F32, static_shape([1000]));
    g.add_input(inp);
    let mut cur = op(&mut g, "Relu", &[inp], static_shape([1000]));
    let mut count = 1;
    for _ in 0..9 {
        cur = op(&mut g, "Relu", &[cur], static_shape([1000]));
        count += 1;
    }
    g.add_output(cur);

    let vm = ViewMap::new();
    let plan = plan_of(&g, &vm);

    assert_eq!(count, 10);
    assert_eq!(plan.naive_bytes, 10 * 1000 * 4);
    assert_eq!(plan.num_slots, 2, "chain of any length reuses into 2 slots");
    assert_eq!(plan.peak_bytes, 2 * 1000 * 4);
    let expected = 1.0 - (2.0 / 10.0);
    assert!(
        (plan.savings_ratio - expected).abs() < 1e-9,
        "savings_ratio {} != {}",
        plan.savings_ratio,
        expected
    );
    println!(
        "10-node chain: naive={}B peak={}B slots={} savings_ratio={:.3}",
        plan.naive_bytes, plan.peak_bytes, plan.num_slots, plan.savings_ratio
    );
}

/// The include_graph_inputs option pulls graph inputs into the arena.
#[test]
fn include_graph_inputs_option() {
    let mut g = Graph::new();
    let inp = g.create_named_value("in", F32, static_shape([50]));
    g.add_input(inp);
    let a = op(&mut g, "Relu", &[inp], static_shape([50]));
    let out = op(&mut g, "Relu", &[a], static_shape([50]));
    g.add_output(out);

    let vm = ViewMap::new();
    let opts = PlanOptions::default().with_graph_inputs(true);
    let status = plan_activations_static(&g, &vm, &opts).unwrap();
    let plan = status.unwrap_complete();
    validate_static(&plan, &g, &vm, &opts).unwrap();
    // The input is now an arena owner.
    assert!(plan.assignments.contains_key(&inp));
    // naive counts input + a + out = 3 owners.
    assert_eq!(plan.naive_bytes, 3 * 50 * 4);
}

/// A produced value with no consumers and not a graph output is dead-on-arrival
/// and handled gracefully (still validates).
#[test]
fn dead_on_arrival_value_is_graceful() {
    let mut g = Graph::new();
    let inp = g.create_named_value("in", F32, static_shape([4]));
    g.add_input(inp);
    let a = op(&mut g, "Relu", &[inp], static_shape([4]));
    // dead: produced but never consumed and not an output.
    let _dead = op(&mut g, "Relu", &[a], static_shape([4]));
    let out = op(&mut g, "Relu", &[a], static_shape([4]));
    g.add_output(out);

    let vm = ViewMap::new();
    let plan = plan_of(&g, &vm);
    // Everything validates and the dead value got a slot without inflating peak
    // beyond the live set.
    assert!(plan.num_slots >= 1);
}
