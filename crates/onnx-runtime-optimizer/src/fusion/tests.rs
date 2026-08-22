use super::*;
use onnx_runtime_ir::{DataType, Node, NodeId, TensorData, static_shape};

fn val(g: &mut Graph, name: &str) -> ValueId {
    g.create_named_value(name, DataType::Float32, static_shape([4]))
}

/// Build a linear MatMul+Add ending in a graph output.
/// Returns (graph, matmul_out_value).
fn matmul_add_graph() -> Graph {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let a = val(&mut g, "a");
    let w = val(&mut g, "w");
    let bias = val(&mut g, "bias");
    g.add_input(a);
    g.add_input(w);
    g.add_input(bias);

    let m = val(&mut g, "m");
    g.insert_node(Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(a), Some(w)],
        vec![m],
    ));
    let out = val(&mut g, "out");
    g.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(m), Some(bias)],
        vec![out],
    ));
    g.add_output(out);
    g
}

#[test]
fn fuses_matmul_add() {
    let mut g = matmul_add_graph();
    assert_eq!(g.num_nodes(), 2);
    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
    assert_eq!(g.num_nodes(), 1);
    let fused = g.nodes.values().next().unwrap();
    assert_eq!(fused.op_type, "FusedMatMulBias");
    assert_eq!(fused.domain, CONTRIB_DOMAIN);
    // Inputs are [a, w, bias].
    assert_eq!(fused.inputs.len(), 3);
    assert!(g.validate().is_ok());
    // Output still a graph output.
    assert_eq!(g.outputs.len(), 1);
    assert_eq!(fused.outputs, g.outputs);
}

#[test]
fn fuses_matmul_add_relu_before_matmul_add() {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let a = val(&mut g, "a");
    let w = val(&mut g, "w");
    let bias = val(&mut g, "bias");
    g.add_input(a);
    g.add_input(w);
    g.add_input(bias);
    let m = val(&mut g, "m");
    g.insert_node(Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(a), Some(w)],
        vec![m],
    ));
    let s = val(&mut g, "s");
    g.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(m), Some(bias)],
        vec![s],
    ));
    let out = val(&mut g, "out");
    g.insert_node(Node::new(NodeId(0), "Relu", vec![Some(s)], vec![out]));
    g.add_output(out);

    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
    assert_eq!(g.num_nodes(), 1);
    let fused = g.nodes.values().next().unwrap();
    assert_eq!(fused.op_type, "FusedGemm");
    assert_eq!(fused.domain, CONTRIB_DOMAIN);
    assert!(g.validate().is_ok());
}

#[test]
fn does_not_fuse_when_intermediate_has_second_consumer() {
    // MatMul -> m ; Add(m, bias) -> out ; and m also feeds a second Relu.
    let mut g = matmul_add_graph();
    // Find `m` (produced by MatMul, consumed by Add).
    let m = g
        .values
        .iter()
        .find(|(_, v)| v.name.as_deref() == Some("m"))
        .map(|(id, _)| id)
        .unwrap();
    let side = val(&mut g, "side");
    g.insert_node(Node::new(NodeId(0), "Relu", vec![Some(m)], vec![side]));
    g.add_output(side);

    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
    // MatMul's output escapes to the side Relu, so no fusion.
    assert!(
        g.nodes.values().any(|n| n.op_type == "MatMul"),
        "MatMul must remain — its output has a second consumer"
    );
    assert!(g.nodes.values().all(|n| n.op_type != "FusedMatMulBias"));
    assert!(g.validate().is_ok());
}

#[test]
fn no_match_returns_none() {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let a = val(&mut g, "a");
    g.add_input(a);
    let out = val(&mut g, "out");
    g.insert_node(Node::new(NodeId(0), "Relu", vec![Some(a)], vec![out]));
    g.add_output(out);
    let p = FusionPattern::new("MatMul+Bias", &["MatMul", "Add"], "FusedMatMulBias");
    assert!(p.find_match(&g).is_none());
}

/// Build the canonical 9-op LayerNorm decomposition over `x`.
///
/// `eps` is an inline f32 initializer (as it would be after `ConstantFolding`
/// materializes the `var + eps` constant) so the schema-aware rewrite can
/// fold it into the `epsilon` attribute; the `ReduceMean` nodes carry an
/// `axes = [-1]` attribute so `axis` extraction is exercised too.
fn layernorm_graph() -> Graph {
    const EPS: f32 = 1e-12;
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let x = val(&mut g, "x");
    let two = val(&mut g, "two");
    let eps = val(&mut g, "eps");
    let scale = val(&mut g, "scale");
    let bias = val(&mut g, "bias");
    g.add_input(x);
    g.add_input(two);
    g.set_initializer(
        eps,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            vec![],
            EPS.to_le_bytes().to_vec(),
        )),
    );
    g.add_input(scale);
    g.add_input(bias);

    let reduce_mean = |g: &mut Graph, input: ValueId, out: ValueId| {
        let mut n = Node::new(NodeId(0), "ReduceMean", vec![Some(input)], vec![out]);
        n.attributes
            .insert("axes".into(), Attribute::Ints(vec![-1]));
        n.attributes.insert("keepdims".into(), Attribute::Int(1));
        g.insert_node(n);
    };

    let mean = val(&mut g, "mean");
    reduce_mean(&mut g, x, mean);
    let diff = val(&mut g, "diff");
    g.insert_node(Node::new(
        NodeId(0),
        "Sub",
        vec![Some(x), Some(mean)],
        vec![diff],
    ));
    let sq = val(&mut g, "sq");
    g.insert_node(Node::new(
        NodeId(0),
        "Pow",
        vec![Some(diff), Some(two)],
        vec![sq],
    ));
    let var = val(&mut g, "var");
    reduce_mean(&mut g, sq, var);
    let vare = val(&mut g, "vare");
    g.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(var), Some(eps)],
        vec![vare],
    ));
    let std = val(&mut g, "std");
    g.insert_node(Node::new(NodeId(0), "Sqrt", vec![Some(vare)], vec![std]));
    let norm = val(&mut g, "norm");
    g.insert_node(Node::new(
        NodeId(0),
        "Div",
        vec![Some(diff), Some(std)],
        vec![norm],
    ));
    let scaled = val(&mut g, "scaled");
    g.insert_node(Node::new(
        NodeId(0),
        "Mul",
        vec![Some(norm), Some(scale)],
        vec![scaled],
    ));
    let out = val(&mut g, "out");
    g.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(scaled), Some(bias)],
        vec![out],
    ));
    g.add_output(out);
    g
}

#[test]
fn fuses_layernorm_chain() {
    let mut g = layernorm_graph();
    assert_eq!(g.num_nodes(), 9);
    assert!(g.validate().is_ok());

    // Record the value ids the schema-aware rewrite must reference.
    let vid = |name: &str| {
        g.values
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some(name))
            .map(|(id, _)| id)
            .unwrap()
    };
    let x = vid("x");
    let scale = vid("scale");
    let bias = vid("bias");

    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();

    assert_eq!(g.num_nodes(), 1, "9-op chain collapses to one node");
    let fused = g.nodes.values().next().unwrap();
    assert_eq!(fused.op_type, "LayerNormalization");
    assert_eq!(fused.domain, CONTRIB_DOMAIN);
    // Schema-conformant inputs: exactly [X, Scale, B] — NOT the intermediate
    // pow-exponent / epsilon tensors.
    assert_eq!(fused.inputs, vec![Some(x), Some(scale), Some(bias)]);
    // Synthesized attributes read by the kernel.
    assert_eq!(
        fused.attr("axis").and_then(Attribute::as_int),
        Some(-1),
        "axis extracted from ReduceMean axes"
    );
    let eps = fused
        .attr("epsilon")
        .and_then(Attribute::as_float)
        .expect("epsilon attribute present");
    assert!(
        (eps - 1e-12).abs() < 1e-18,
        "epsilon extracted from the var+eps constant, got {eps}"
    );
    assert_eq!(fused.outputs, g.outputs);
    assert!(g.validate().is_ok());
}

#[test]
fn layernorm_count_bookkeeping() {
    let mut g = layernorm_graph();
    let ln_before = g
        .nodes
        .values()
        .filter(|n| n.op_type == "LayerNormalization")
        .count();
    let rm_before = g
        .nodes
        .values()
        .filter(|n| n.op_type == "ReduceMean")
        .count();
    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
    let ln_after = g
        .nodes
        .values()
        .filter(|n| n.op_type == "LayerNormalization")
        .count();
    let rm_after = g
        .nodes
        .values()
        .filter(|n| n.op_type == "ReduceMean")
        .count();
    assert_eq!(ln_before, 0);
    assert_eq!(ln_after, 1);
    assert_eq!(rm_before, 2);
    assert_eq!(rm_after, 0);
}

/// Build the 10-op split-diff LayerNorm decomposition over `x` (the
/// `bert_toy`-style variant): the variance branch and the numerator branch
/// each get their **own** distinct `Sub` node instead of sharing one `diff`.
/// `mean` therefore fans out to two Subs and `x` to two Subs. When
/// `reverse_num_sub` is true the numerator `Sub` is emitted reversed as
/// `Sub(mean, x)` (an adversarial sign-flip) to exercise the operand-order
/// guard.
fn layernorm_split_graph(reverse_num_sub: bool) -> Graph {
    const EPS: f32 = 1e-12;
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let x = val(&mut g, "x");
    let two = val(&mut g, "two");
    let eps = val(&mut g, "eps");
    let scale = val(&mut g, "scale");
    let bias = val(&mut g, "bias");
    g.add_input(x);
    g.add_input(two);
    g.set_initializer(
        eps,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            vec![],
            EPS.to_le_bytes().to_vec(),
        )),
    );
    g.add_input(scale);
    g.add_input(bias);

    let reduce_mean = |g: &mut Graph, input: ValueId, out: ValueId| {
        let mut n = Node::new(NodeId(0), "ReduceMean", vec![Some(input)], vec![out]);
        n.attributes
            .insert("axes".into(), Attribute::Ints(vec![-1]));
        n.attributes.insert("keepdims".into(), Attribute::Int(1));
        g.insert_node(n);
    };

    let mean = val(&mut g, "mean");
    reduce_mean(&mut g, x, mean);
    // Variance-branch Sub: always the canonical `x - mean`.
    let diff_pow = val(&mut g, "diff_pow");
    g.insert_node(Node::new(
        NodeId(0),
        "Sub",
        vec![Some(x), Some(mean)],
        vec![diff_pow],
    ));
    // Numerator-branch Sub: a SECOND, distinct node. Reversed operands when
    // `reverse_num_sub` (adversarial `mean - x`), else canonical `x - mean`.
    let diff_div = val(&mut g, "diff_div");
    let num_inputs = if reverse_num_sub {
        vec![Some(mean), Some(x)]
    } else {
        vec![Some(x), Some(mean)]
    };
    g.insert_node(Node::new(NodeId(0), "Sub", num_inputs, vec![diff_div]));

    let sq = val(&mut g, "sq");
    g.insert_node(Node::new(
        NodeId(0),
        "Pow",
        vec![Some(diff_pow), Some(two)],
        vec![sq],
    ));
    let var = val(&mut g, "var");
    reduce_mean(&mut g, sq, var);
    let vare = val(&mut g, "vare");
    g.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(var), Some(eps)],
        vec![vare],
    ));
    let std = val(&mut g, "std");
    g.insert_node(Node::new(NodeId(0), "Sqrt", vec![Some(vare)], vec![std]));
    let norm = val(&mut g, "norm");
    g.insert_node(Node::new(
        NodeId(0),
        "Div",
        vec![Some(diff_div), Some(std)],
        vec![norm],
    ));
    let scaled = val(&mut g, "scaled");
    g.insert_node(Node::new(
        NodeId(0),
        "Mul",
        vec![Some(norm), Some(scale)],
        vec![scaled],
    ));
    let out = val(&mut g, "out");
    g.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(scaled), Some(bias)],
        vec![out],
    ));
    g.add_output(out);
    g
}

#[test]
fn fuses_layernorm_split_chain() {
    // Isolated optimizer-layer coverage for the 10-op split-diff shape
    // (previously only exercised end-to-end via the bert_toy model).
    let mut g = layernorm_split_graph(false);
    assert_eq!(g.num_nodes(), 10, "split-diff shape has two distinct Subs");
    assert!(g.validate().is_ok());

    let vid = |name: &str| {
        g.values
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some(name))
            .map(|(id, _)| id)
            .unwrap()
    };
    let x = vid("x");
    let scale = vid("scale");
    let bias = vid("bias");

    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();

    assert_eq!(g.num_nodes(), 1, "10-op split chain collapses to one node");
    let fused = g.nodes.values().next().unwrap();
    assert_eq!(fused.op_type, "LayerNormalization");
    assert_eq!(fused.domain, CONTRIB_DOMAIN);
    // Schema-conformant inputs: exactly [X, Scale, B].
    assert_eq!(fused.inputs, vec![Some(x), Some(scale), Some(bias)]);
    assert_eq!(
        fused.attr("axis").and_then(Attribute::as_int),
        Some(-1),
        "axis extracted from ReduceMean axes"
    );
    let eps = fused
        .attr("epsilon")
        .and_then(Attribute::as_float)
        .expect("epsilon attribute present");
    assert!(
        (eps - 1e-12).abs() < 1e-18,
        "epsilon extracted from the var+eps constant, got {eps}"
    );
    assert_eq!(fused.outputs, g.outputs);
    assert!(g.validate().is_ok());
}

#[test]
fn declines_layernorm_when_numerator_sub_reversed() {
    // A-CHEW-1 adversarial: the numerator diamond centers with a REVERSED
    // `Sub(mean, x)` = -(x - mean). Membership of {x, mean} still holds, but
    // the operand-order guard must DECLINE (else the rewrite silently produces
    // a sign-flipped LayerNormalization). Ops must be left untouched.
    let mut g = layernorm_split_graph(true);
    assert_eq!(g.num_nodes(), 10);
    assert!(g.validate().is_ok());

    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();

    assert!(
        g.nodes.values().all(|n| n.op_type != "LayerNormalization"),
        "reversed Sub(mean, x) must NOT fuse — sign-flip over-match"
    );
    assert_eq!(g.num_nodes(), 10, "all 10 ops remain (declined)");
    assert_eq!(
        g.nodes.values().filter(|n| n.op_type == "Sub").count(),
        2,
        "both centering Subs preserved"
    );
    assert!(g.validate().is_ok());
}

#[test]
fn does_not_fuse_partial_layernorm() {
    // A LayerNorm chain missing its final Add must not fuse.
    let mut g = layernorm_graph();
    // Remove the last Add by rebuilding: easier to just check a shorter
    // pattern doesn't accidentally match — assert Sub alone isn't fused.
    let p = FusionPattern::layernorm();
    // Break the chain: give `diff` an external consumer so the safety rule
    // trips (Sub is a non-final matched node).
    let diff = g
        .values
        .iter()
        .find(|(_, v)| v.name.as_deref() == Some("diff"))
        .map(|(id, _)| id)
        .unwrap();
    let side = val(&mut g, "side");
    g.insert_node(Node::new(NodeId(0), "Neg", vec![Some(diff)], vec![side]));
    g.add_output(side);
    assert!(
        p.find_match(&g).is_none(),
        "external consumer on `diff` blocks the fusion"
    );
}

#[test]
fn fuses_two_independent_matmul_adds() {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    for i in 0..2 {
        let a = val(&mut g, &format!("a{i}"));
        let w = val(&mut g, &format!("w{i}"));
        let bias = val(&mut g, &format!("bias{i}"));
        g.add_input(a);
        g.add_input(w);
        g.add_input(bias);
        let m = val(&mut g, &format!("m{i}"));
        g.insert_node(Node::new(
            NodeId(0),
            "MatMul",
            vec![Some(a), Some(w)],
            vec![m],
        ));
        let out = val(&mut g, &format!("out{i}"));
        g.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(m), Some(bias)],
            vec![out],
        ));
        g.add_output(out);
    }
    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
    assert_eq!(g.num_nodes(), 2);
    assert!(g.nodes.values().all(|n| n.op_type == "FusedMatMulBias"));
    assert!(g.validate().is_ok());
}

#[test]
fn find_match_reports_correct_shape() {
    let g = matmul_add_graph();
    let p = FusionPattern::new("MatMul+Bias", &["MatMul", "Add"], "FusedMatMulBias");
    let m = p.find_match(&g).expect("should match");
    assert_eq!(m.nodes.len(), 2);
    assert_eq!(m.external_inputs.len(), 3);
    assert_eq!(p.pattern_name(), "MatMul+Bias");
}

/// Rewrite the `ReduceMean` that produces `out_name` from the attribute form to
/// the opset-24 axes-as-input form, backed by a fresh int64 initializer.
fn convert_reduce_to_axes_input(g: &mut Graph, out_name: &str, init_name: &str, axes: &[i64]) {
    let out = g
        .values
        .iter()
        .find(|(_, v)| v.name.as_deref() == Some(out_name))
        .map(|(id, _)| id)
        .unwrap();
    let rm = g.value(out).producer.unwrap();
    g.node_mut(rm).attributes.remove("axes");
    let axes_in = g.create_named_value(init_name, DataType::Int64, static_shape([axes.len()]));
    let data: Vec<u8> = axes.iter().flat_map(|a| a.to_le_bytes()).collect();
    g.set_initializer(
        axes_in,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            vec![axes.len()],
            data,
        )),
    );
    let input_index = g.node(rm).inputs.len();
    g.node_mut(rm).inputs.push(None);
    g.replace_input(rm, input_index, Some(axes_in));
}

#[test]
fn fuses_layernorm_when_axes_is_input() {
    // Opset-24 style: BOTH `ReduceMean` nodes take `axes` as an INPUT. The fusion
    // resolves the single concrete axis from each int64 initializer, requires the
    // mean and variance reductions to agree, and fuses to one
    // `LayerNormalization` (axis = -1). (Migrating fixtures to opset 24 relies on
    // this.)
    let mut g = layernorm_graph();
    convert_reduce_to_axes_input(&mut g, "mean", "mean_axes", &[-1]);
    convert_reduce_to_axes_input(&mut g, "var", "var_axes", &[-1]);
    assert!(g.validate().is_ok());

    assert_eq!(g.num_nodes(), 9);
    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
    let layernorms: Vec<_> = g
        .nodes
        .values()
        .filter(|n| n.op_type == "LayerNormalization")
        .collect();
    assert_eq!(
        layernorms.len(),
        1,
        "axes-as-input LayerNorm must fuse to one LayerNormalization"
    );
    assert_eq!(
        layernorms[0].attr("axis").and_then(Attribute::as_int),
        Some(-1),
        "axis must be resolved from the int64 axes input"
    );
    assert_eq!(
        g.nodes
            .values()
            .filter(|n| n.op_type == "ReduceMean")
            .count(),
        0,
        "both ReduceMean ops must fuse away"
    );
    assert!(g.validate().is_ok());
}

#[test]
fn declines_layernorm_when_reduce_axes_disagree() {
    // Mean reduces axis -1 but variance reduces axis -2: not a LayerNorm, so the
    // fusion must decline and keep both ReduceMean ops.
    let mut g = layernorm_graph();
    convert_reduce_to_axes_input(&mut g, "mean", "mean_axes", &[-1]);
    convert_reduce_to_axes_input(&mut g, "var", "var_axes", &[-2]);
    assert!(g.validate().is_ok());

    assert_eq!(g.num_nodes(), 9);
    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
    assert!(
        g.nodes.values().all(|n| n.op_type != "LayerNormalization"),
        "mismatched mean/variance reduce axes must not fuse"
    );
    assert_eq!(
        g.nodes
            .values()
            .filter(|n| n.op_type == "ReduceMean")
            .count(),
        2,
        "both ReduceMean ops remain"
    );
}

#[test]
fn declines_layernorm_when_reduce_keepdims_zero() {
    // keepdims = 0 collapses the reduced dim, breaking the LayerNorm broadcast,
    // so the fusion must decline even though the rest of the pattern matches.
    let mut g = layernorm_graph();
    let mean = g
        .values
        .iter()
        .find(|(_, v)| v.name.as_deref() == Some("mean"))
        .map(|(id, _)| id)
        .unwrap();
    let rm1 = g.value(mean).producer.unwrap();
    g.node_mut(rm1)
        .attributes
        .insert("keepdims".into(), Attribute::Int(0));

    assert_eq!(g.num_nodes(), 9);
    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
    assert!(
        g.nodes.values().all(|n| n.op_type != "LayerNormalization"),
        "keepdims = 0 on a reduce must not fuse"
    );
}

/// Rewrite the `ReduceMean` producing `out_name` to take `axes` from a
/// `Constant` node (rather than an initializer), mirroring a graph where
/// `ConstantFolding` has not yet run.
fn convert_reduce_to_axes_constant(g: &mut Graph, out_name: &str, const_name: &str, axes: &[i64]) {
    let out = g
        .values
        .iter()
        .find(|(_, v)| v.name.as_deref() == Some(out_name))
        .map(|(id, _)| id)
        .unwrap();
    let rm = g.value(out).producer.unwrap();
    g.node_mut(rm).attributes.remove("axes");
    let axes_val = g.create_named_value(const_name, DataType::Int64, static_shape([axes.len()]));
    let data: Vec<u8> = axes.iter().flat_map(|a| a.to_le_bytes()).collect();
    let mut constant = Node::new(NodeId(0), "Constant", vec![], vec![axes_val]);
    constant.attributes.insert(
        "value".into(),
        Attribute::Tensor(TensorData::from_raw(
            DataType::Int64,
            vec![axes.len()],
            data,
        )),
    );
    g.insert_node(constant);
    let input_index = g.node(rm).inputs.len();
    g.node_mut(rm).inputs.push(None);
    g.replace_input(rm, input_index, Some(axes_val));
}

#[test]
fn fuses_layernorm_when_axes_is_unfolded_constant_node() {
    // The fusion pass may run before `ConstantFolding` materializes a `Constant`
    // axes node as an initializer. `read_i64_vector` resolves the axes directly
    // from the `Constant` producer, so `OpFusion` alone still fuses.
    let mut g = layernorm_graph();
    convert_reduce_to_axes_constant(&mut g, "mean", "mean_axes_const", &[-1]);
    convert_reduce_to_axes_constant(&mut g, "var", "var_axes_const", &[-1]);
    assert!(g.validate().is_ok());

    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
    let layernorms: Vec<_> = g
        .nodes
        .values()
        .filter(|n| n.op_type == "LayerNormalization")
        .collect();
    assert_eq!(
        layernorms.len(),
        1,
        "Constant-node axes must fuse without prior constant folding"
    );
    assert_eq!(
        layernorms[0].attr("axis").and_then(Attribute::as_int),
        Some(-1),
        "axis must be resolved from the Constant axes node"
    );
    assert_eq!(
        g.nodes
            .values()
            .filter(|n| n.op_type == "ReduceMean")
            .count(),
        0,
        "both ReduceMean ops must fuse away"
    );
}

#[test]
fn declines_layernorm_when_epsilon_not_constant() {
    // If epsilon is a runtime graph INPUT (not a folded f32 initializer) it
    // can't be read as a concrete scalar → DECLINE rather than silently
    // substituting the ONNX default 1e-5.
    let mut g = layernorm_graph();
    let eps = g
        .values
        .iter()
        .find(|(_, v)| v.name.as_deref() == Some("eps"))
        .map(|(id, _)| id)
        .unwrap();
    // Turn the eps initializer into a plain runtime graph input.
    g.initializers.remove(&eps);
    g.add_input(eps);
    assert!(g.validate().is_ok());

    assert_eq!(g.num_nodes(), 9);
    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
    assert_eq!(
        g.num_nodes(),
        9,
        "non-constant epsilon LayerNorm must NOT fuse"
    );
    assert!(g.nodes.values().all(|n| n.op_type != "LayerNormalization"));
    assert!(g.validate().is_ok());
}

#[test]
fn declines_matmul_add_when_bias_expands() {
    // MatMul output is [4]; the Add's bias is [2, 4], whose extra leading dim
    // would broadcast the result UP to [2, 4]. The fused kernel/shape rule
    // assume the output equals the matmul shape and would silently truncate,
    // so the fusion must DECLINE and keep the original MatMul + Add.
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let a = g.create_named_value("a", DataType::Float32, static_shape([4, 4]));
    let w = g.create_named_value("w", DataType::Float32, static_shape([4]));
    let bias = g.create_named_value("bias", DataType::Float32, static_shape([2, 4]));
    g.add_input(a);
    g.add_input(w);
    g.add_input(bias);
    let m = g.create_named_value("m", DataType::Float32, static_shape([4]));
    g.insert_node(Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(a), Some(w)],
        vec![m],
    ));
    let out = g.create_named_value("out", DataType::Float32, static_shape([2, 4]));
    g.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(m), Some(bias)],
        vec![out],
    ));
    g.add_output(out);

    assert_eq!(g.num_nodes(), 2);
    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
    assert_eq!(g.num_nodes(), 2, "expanding bias must NOT fuse");
    assert!(g.nodes.values().any(|n| n.op_type == "MatMul"));
    assert!(g.nodes.values().any(|n| n.op_type == "Add"));
    assert!(g.nodes.values().all(|n| n.op_type != "FusedMatMulBias"));
    assert!(g.validate().is_ok());
}

#[test]
fn fuses_matmul_add_with_trailing_broadcast_bias() {
    // A `[1, 4]` bias broadcasts INTO a `[3, 4]` matmul output without
    // expanding it, so the guard must still allow this common case to fuse.
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let a = g.create_named_value("a", DataType::Float32, static_shape([3, 4]));
    let w = g.create_named_value("w", DataType::Float32, static_shape([4, 4]));
    let bias = g.create_named_value("bias", DataType::Float32, static_shape([1, 4]));
    g.add_input(a);
    g.add_input(w);
    g.add_input(bias);
    let m = g.create_named_value("m", DataType::Float32, static_shape([3, 4]));
    g.insert_node(Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(a), Some(w)],
        vec![m],
    ));
    let out = g.create_named_value("out", DataType::Float32, static_shape([3, 4]));
    g.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(m), Some(bias)],
        vec![out],
    ));
    g.add_output(out);

    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
    assert_eq!(g.num_nodes(), 1, "trailing-broadcast bias must fuse");
    assert_eq!(g.nodes.values().next().unwrap().op_type, "FusedMatMulBias");
    assert!(g.validate().is_ok());
}

#[test]
fn declines_matmul_add_when_shape_unknown() {
    // If the matmul output shape can't be resolved (empty/unknown), the guard
    // can't prove the bias is non-expanding → DECLINE conservatively.
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let a = g.create_named_value("a", DataType::Float32, Vec::new());
    let w = g.create_named_value("w", DataType::Float32, Vec::new());
    let bias = g.create_named_value("bias", DataType::Float32, static_shape([4]));
    g.add_input(a);
    g.add_input(w);
    g.add_input(bias);
    // `m` has an unknown (empty) shape.
    let m = g.create_named_value("m", DataType::Float32, Vec::new());
    g.insert_node(Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(a), Some(w)],
        vec![m],
    ));
    let out = g.create_named_value("out", DataType::Float32, static_shape([4]));
    g.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(m), Some(bias)],
        vec![out],
    ));
    g.add_output(out);

    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
    assert_eq!(g.num_nodes(), 2, "unknown matmul shape must NOT fuse");
    assert!(g.nodes.values().all(|n| n.op_type != "FusedMatMulBias"));
    assert!(g.validate().is_ok());
}

#[test]
fn declines_fused_gemm_when_bias_expands() {
    // Roy's FusedGemm review advisory, locked in: a MatMul+Add+Relu whose
    // bias EXPANDS the matmul output (extra leading/batch dim) must DECLINE
    // to FusedGemm exactly like the FusedMatMulBias case — the trailing Relu
    // is shape-neutral, so the same non-expanding-bias guard applies. MatMul
    // output is [4]; bias [2, 4] would broadcast the result up to [2, 4].
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let a = g.create_named_value("a", DataType::Float32, static_shape([4, 4]));
    let w = g.create_named_value("w", DataType::Float32, static_shape([4]));
    let bias = g.create_named_value("bias", DataType::Float32, static_shape([2, 4]));
    g.add_input(a);
    g.add_input(w);
    g.add_input(bias);
    let m = g.create_named_value("m", DataType::Float32, static_shape([4]));
    g.insert_node(Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(a), Some(w)],
        vec![m],
    ));
    let biased = g.create_named_value("biased", DataType::Float32, static_shape([2, 4]));
    g.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(m), Some(bias)],
        vec![biased],
    ));
    let out = g.create_named_value("out", DataType::Float32, static_shape([2, 4]));
    g.insert_node(Node::new(NodeId(0), "Relu", vec![Some(biased)], vec![out]));
    g.add_output(out);

    assert_eq!(g.num_nodes(), 3);
    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
    assert_eq!(
        g.num_nodes(),
        3,
        "expanding bias must NOT fuse to FusedGemm"
    );
    assert!(g.nodes.values().any(|n| n.op_type == "MatMul"));
    assert!(g.nodes.values().any(|n| n.op_type == "Add"));
    assert!(g.nodes.values().any(|n| n.op_type == "Relu"));
    assert!(g.nodes.values().all(|n| n.op_type != "FusedGemm"));
    assert!(g.validate().is_ok());
}

// --- AttentionFusion (SDPA core) ------------------------------------------

/// Add a strict-scalar f32 initializer, returning its value id.
fn scalar_init(g: &mut Graph, name: &str, v: f32) -> ValueId {
    let vid = g.create_named_value(name, DataType::Float32, Vec::new());
    g.set_initializer(
        vid,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            vec![],
            v.to_le_bytes().to_vec(),
        )),
    );
    vid
}

fn fval(g: &mut Graph, name: &str, dims: &[usize]) -> ValueId {
    g.create_named_value(name, DataType::Float32, static_shape(dims.iter().copied()))
}

/// Look up a value id by name (test convenience).
fn value_id(g: &Graph, name: &str) -> ValueId {
    g.values
        .iter()
        .find(|(_, v)| v.name.as_deref() == Some(name))
        .map(|(id, _)| id)
        .unwrap_or_else(|| panic!("no value named {name}"))
}

/// Build an SDPA core graph `Softmax((Q·Kᵀ)/c [+ mask], axis=-1) · V` with
/// rank-4 `[1, 2, seq, dim]` tensors, K supplied pre-transposed as
/// `[1, 2, d, sk]` (so `k_transposed` should resolve to 1). `masked` adds an
/// additive mask; `axis` is the Softmax reduction axis.
fn sdpa_graph(masked: bool, axis: i64) -> Graph {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 12);
    let q = fval(&mut g, "Q", &[1, 2, 3, 4]);
    let kt = fval(&mut g, "K", &[1, 2, 4, 3]); // pre-transposed [d=4, sk=3]
    let v = fval(&mut g, "V", &[1, 2, 3, 4]);
    g.add_input(q);
    g.add_input(kt);
    g.add_input(v);
    let c = scalar_init(&mut g, "scale_c", 2.0);

    let scores = fval(&mut g, "scores", &[1, 2, 3, 3]);
    g.insert_node(Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(q), Some(kt)],
        vec![scores],
    ));
    let scaled = fval(&mut g, "scaled", &[1, 2, 3, 3]);
    g.insert_node(Node::new(
        NodeId(0),
        "Div",
        vec![Some(scores), Some(c)],
        vec![scaled],
    ));

    let sm_in = if masked {
        let mask = fval(&mut g, "mask", &[1, 1, 3, 3]);
        g.add_input(mask);
        let masked_v = fval(&mut g, "masked", &[1, 2, 3, 3]);
        g.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(scaled), Some(mask)],
            vec![masked_v],
        ));
        masked_v
    } else {
        scaled
    };

    let probs = fval(&mut g, "probs", &[1, 2, 3, 3]);
    let mut sm = Node::new(NodeId(0), "Softmax", vec![Some(sm_in)], vec![probs]);
    sm.attributes.insert("axis".into(), Attribute::Int(axis));
    g.insert_node(sm);
    let out = fval(&mut g, "out", &[1, 2, 3, 4]);
    g.insert_node(Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(probs), Some(v)],
        vec![out],
    ));
    g.add_output(out);
    g
}

fn fused_attention_node(g: &Graph) -> Option<&Node> {
    g.nodes.values().find(|n| n.op_type == "FusedAttention")
}

#[test]
fn fuses_sdpa_unmasked_pretransposed_k() {
    let mut g = sdpa_graph(false, 3);
    let q = value_id(&g, "Q");
    let k = value_id(&g, "K");
    let v = value_id(&g, "V");
    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();

    // Exactly one FusedAttention; no surviving Softmax/MatMul/Div.
    assert_eq!(
        g.nodes
            .values()
            .filter(|n| n.op_type == "FusedAttention")
            .count(),
        1
    );
    assert!(g.nodes.values().all(|n| n.op_type != "Softmax"));
    assert!(g.nodes.values().all(|n| n.op_type != "MatMul"));
    assert!(g.nodes.values().all(|n| n.op_type != "Div"));

    let fa = fused_attention_node(&g).unwrap();
    assert_eq!(fa.domain, CONTRIB_DOMAIN);
    assert_eq!(fa.inputs, vec![Some(q), Some(k), Some(v)]);
    // scale = 1/c = 1/2 = 0.5; K used as-is → k_transposed = 1.
    assert_eq!(fa.attr("scale").and_then(Attribute::as_float), Some(0.5));
    assert_eq!(fa.attr("k_transposed").and_then(Attribute::as_int), Some(1));
    assert!(g.validate().is_ok());
}

#[test]
fn fuses_sdpa_masked() {
    let mut g = sdpa_graph(true, 3);
    let (q, k, v, mask) = (
        value_id(&g, "Q"),
        value_id(&g, "K"),
        value_id(&g, "V"),
        value_id(&g, "mask"),
    );
    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();

    assert_eq!(
        g.nodes
            .values()
            .filter(|n| n.op_type == "FusedAttention")
            .count(),
        1
    );
    assert!(g.nodes.values().all(|n| n.op_type != "Softmax"));
    assert!(g.nodes.values().all(|n| n.op_type != "Add"));
    let fa = fused_attention_node(&g).unwrap();
    // Mask appended as the 4th input.
    assert_eq!(fa.inputs, vec![Some(q), Some(k), Some(v), Some(mask)]);
    assert_eq!(fa.attr("k_transposed").and_then(Attribute::as_int), Some(1));
    assert!(g.validate().is_ok());
}

#[test]
fn fuses_sdpa_absorbing_clean_transpose_sets_k_transposed_0() {
    // K is supplied in natural [1,2,3,4] layout and transposed to Kᵀ by a
    // clean last-two-axis Transpose (perm [0,1,3,2]) consumed only by the
    // score MatMul. The matcher absorbs it: K input becomes the natural K
    // and k_transposed = 0 (kernel transposes internally).
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 12);
    let q = fval(&mut g, "Q", &[1, 2, 3, 4]);
    let k = fval(&mut g, "K", &[1, 2, 3, 4]); // natural [sk=3, d=4]
    let v = fval(&mut g, "V", &[1, 2, 3, 4]);
    g.add_input(q);
    g.add_input(k);
    g.add_input(v);
    let c = scalar_init(&mut g, "scale_c", 4.0);

    let kt = fval(&mut g, "Kt", &[1, 2, 4, 3]);
    let mut tr = Node::new(NodeId(0), "Transpose", vec![Some(k)], vec![kt]);
    tr.attributes
        .insert("perm".into(), Attribute::Ints(vec![0, 1, 3, 2]));
    g.insert_node(tr);
    let scores = fval(&mut g, "scores", &[1, 2, 3, 3]);
    g.insert_node(Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(q), Some(kt)],
        vec![scores],
    ));
    let scaled = fval(&mut g, "scaled", &[1, 2, 3, 3]);
    g.insert_node(Node::new(
        NodeId(0),
        "Div",
        vec![Some(scores), Some(c)],
        vec![scaled],
    ));
    let probs = fval(&mut g, "probs", &[1, 2, 3, 3]);
    let mut sm = Node::new(NodeId(0), "Softmax", vec![Some(scaled)], vec![probs]);
    sm.attributes.insert("axis".into(), Attribute::Int(-1));
    g.insert_node(sm);
    let out = fval(&mut g, "out", &[1, 2, 3, 4]);
    g.insert_node(Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(probs), Some(v)],
        vec![out],
    ));
    g.add_output(out);

    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
    assert!(
        g.nodes.values().all(|n| n.op_type != "Transpose"),
        "clean Kᵀ Transpose absorbed"
    );
    let fa = fused_attention_node(&g).unwrap();
    assert_eq!(
        fa.inputs,
        vec![Some(q), Some(k), Some(v)],
        "K input is the natural (un-transposed) K"
    );
    assert_eq!(fa.attr("k_transposed").and_then(Attribute::as_int), Some(0));
    // scale = 1/4 = 0.25.
    assert_eq!(fa.attr("scale").and_then(Attribute::as_float), Some(0.25));
    assert!(g.validate().is_ok());
}

#[test]
fn declines_sdpa_when_softmax_axis_not_last() {
    // axis 1 on a rank-4 score tensor is not the last axis → decline.
    let mut g = sdpa_graph(false, 1);
    let before = g.num_nodes();
    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
    assert!(g.nodes.values().all(|n| n.op_type != "FusedAttention"));
    assert!(g.nodes.values().any(|n| n.op_type == "Softmax"));
    assert_eq!(g.num_nodes(), before, "no fusion when axis is not last");
}

#[test]
fn declines_sdpa_when_scale_is_not_scalar_constant() {
    // The score-scaling divisor is a runtime graph input (not a constant),
    // so the scale can't be folded to a concrete f32 → decline.
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 12);
    let q = fval(&mut g, "Q", &[1, 2, 3, 4]);
    let kt = fval(&mut g, "K", &[1, 2, 4, 3]);
    let v = fval(&mut g, "V", &[1, 2, 3, 4]);
    let c = fval(&mut g, "scale_c", &[]); // runtime input, NOT an initializer
    g.add_input(q);
    g.add_input(kt);
    g.add_input(v);
    g.add_input(c);
    let scores = fval(&mut g, "scores", &[1, 2, 3, 3]);
    g.insert_node(Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(q), Some(kt)],
        vec![scores],
    ));
    let scaled = fval(&mut g, "scaled", &[1, 2, 3, 3]);
    g.insert_node(Node::new(
        NodeId(0),
        "Div",
        vec![Some(scores), Some(c)],
        vec![scaled],
    ));
    let probs = fval(&mut g, "probs", &[1, 2, 3, 3]);
    let mut sm = Node::new(NodeId(0), "Softmax", vec![Some(scaled)], vec![probs]);
    sm.attributes.insert("axis".into(), Attribute::Int(3));
    g.insert_node(sm);
    let out = fval(&mut g, "out", &[1, 2, 3, 4]);
    g.insert_node(Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(probs), Some(v)],
        vec![out],
    ));
    g.add_output(out);

    let before = g.num_nodes();
    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
    assert!(g.nodes.values().all(|n| n.op_type != "FusedAttention"));
    assert_eq!(g.num_nodes(), before, "non-constant scale must NOT fuse");
}

#[test]
fn declines_sdpa_when_softmax_is_right_operand_of_output_matmul() {
    // out = V · probs (softmax output is the RIGHT operand) is not `probs·V`
    // SDPA — the matcher requires the softmax output be the LEFT operand.
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 12);
    let q = fval(&mut g, "Q", &[1, 2, 3, 4]);
    let kt = fval(&mut g, "K", &[1, 2, 4, 3]);
    let v = fval(&mut g, "V", &[1, 2, 3, 3]);
    g.add_input(q);
    g.add_input(kt);
    g.add_input(v);
    let c = scalar_init(&mut g, "scale_c", 2.0);
    let scores = fval(&mut g, "scores", &[1, 2, 3, 3]);
    g.insert_node(Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(q), Some(kt)],
        vec![scores],
    ));
    let scaled = fval(&mut g, "scaled", &[1, 2, 3, 3]);
    g.insert_node(Node::new(
        NodeId(0),
        "Div",
        vec![Some(scores), Some(c)],
        vec![scaled],
    ));
    let probs = fval(&mut g, "probs", &[1, 2, 3, 3]);
    let mut sm = Node::new(NodeId(0), "Softmax", vec![Some(scaled)], vec![probs]);
    sm.attributes.insert("axis".into(), Attribute::Int(3));
    g.insert_node(sm);
    let out = fval(&mut g, "out", &[1, 2, 3, 3]);
    // Reversed operand order: V · probs.
    g.insert_node(Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(v), Some(probs)],
        vec![out],
    ));
    g.add_output(out);

    let before = g.num_nodes();
    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
    assert!(g.nodes.values().all(|n| n.op_type != "FusedAttention"));
    assert!(g.nodes.values().any(|n| n.op_type == "Softmax"));
    assert_eq!(g.num_nodes(), before);
}

/// Build the exact-GELU `Erf` decomposition `0.5·x·(1 + erf(x / √2))` over a
/// single graph input `x`, with the constants materialized as scalar
/// initializers. `inner`/`half` select the constant encoding to emit so the
/// equivalent forms can be exercised.
fn gelu_graph(inner_div_sqrt2: bool, half_mul: bool) -> Graph {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let x = val(&mut g, "x");
    g.add_input(x);

    // half = 0.5 * x  (via Mul(x, 0.5) or Div(x, 2.0)).
    let half = val(&mut g, "half");
    if half_mul {
        let c = scalar_init(&mut g, "c_half", 0.5);
        g.insert_node(Node::new(
            NodeId(0),
            "Mul",
            vec![Some(x), Some(c)],
            vec![half],
        ));
    } else {
        let c = scalar_init(&mut g, "c_two", 2.0);
        g.insert_node(Node::new(
            NodeId(0),
            "Div",
            vec![Some(x), Some(c)],
            vec![half],
        ));
    }

    // scaled = x / √2  (via Div(x, √2) or Mul(x, 1/√2)).
    let scaled = val(&mut g, "scaled");
    if inner_div_sqrt2 {
        let c = scalar_init(&mut g, "c_sqrt2", std::f32::consts::SQRT_2);
        g.insert_node(Node::new(
            NodeId(0),
            "Div",
            vec![Some(x), Some(c)],
            vec![scaled],
        ));
    } else {
        let c = scalar_init(&mut g, "c_isqrt2", std::f32::consts::FRAC_1_SQRT_2);
        g.insert_node(Node::new(
            NodeId(0),
            "Mul",
            vec![Some(x), Some(c)],
            vec![scaled],
        ));
    }

    let e = val(&mut g, "e");
    g.insert_node(Node::new(NodeId(0), "Erf", vec![Some(scaled)], vec![e]));
    let one = scalar_init(&mut g, "c_one", 1.0);
    let a = val(&mut g, "a");
    g.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(e), Some(one)],
        vec![a],
    ));
    let out = val(&mut g, "out");
    g.insert_node(Node::new(
        NodeId(0),
        "Mul",
        vec![Some(half), Some(a)],
        vec![out],
    ));
    g.add_output(out);
    g
}

#[test]
fn fuses_gelu_div_sqrt2() {
    let mut g = gelu_graph(true, true);
    assert_eq!(g.num_nodes(), 5);
    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
    let gelu: Vec<_> = g.nodes.values().filter(|n| n.op_type == "Gelu").collect();
    assert_eq!(gelu.len(), 1, "the Erf decomposition must fuse to one Gelu");
    let fused = gelu[0];
    assert_eq!(fused.domain, CONTRIB_DOMAIN);
    assert_eq!(fused.inputs.len(), 1, "Gelu takes the single input x");
    assert!(fused.attributes.is_empty(), "exact Gelu has no attributes");
    // Single input is the graph input `x`.
    let x = g
        .values
        .iter()
        .find(|(_, v)| v.name.as_deref() == Some("x"))
        .map(|(id, _)| id)
        .unwrap();
    assert_eq!(fused.inputs[0], Some(x));
    assert_eq!(fused.outputs, g.outputs);
    assert!(g.nodes.values().all(|n| n.op_type != "Erf"));
    assert!(g.validate().is_ok());
}

#[test]
fn fuses_gelu_mul_reciprocal_and_div_two() {
    // Equivalent encodings: inner Mul(x, 1/√2), half Div(x, 2.0).
    let mut g = gelu_graph(false, false);
    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
    assert_eq!(
        g.nodes.values().filter(|n| n.op_type == "Gelu").count(),
        1,
        "the reciprocal/half-divisor encoding must also fuse"
    );
    assert!(g.validate().is_ok());
}

#[test]
fn declines_gelu_wrong_inner_constant() {
    // Div by 2.0 instead of √2 is not x/√2 → decline.
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let x = val(&mut g, "x");
    g.add_input(x);
    let half = val(&mut g, "half");
    let ch = scalar_init(&mut g, "c_half", 0.5);
    g.insert_node(Node::new(
        NodeId(0),
        "Mul",
        vec![Some(x), Some(ch)],
        vec![half],
    ));
    let scaled = val(&mut g, "scaled");
    let cbad = scalar_init(&mut g, "c_bad", 2.0);
    g.insert_node(Node::new(
        NodeId(0),
        "Div",
        vec![Some(x), Some(cbad)],
        vec![scaled],
    ));
    let e = val(&mut g, "e");
    g.insert_node(Node::new(NodeId(0), "Erf", vec![Some(scaled)], vec![e]));
    let one = scalar_init(&mut g, "c_one", 1.0);
    let a = val(&mut g, "a");
    g.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(e), Some(one)],
        vec![a],
    ));
    let out = val(&mut g, "out");
    g.insert_node(Node::new(
        NodeId(0),
        "Mul",
        vec![Some(half), Some(a)],
        vec![out],
    ));
    g.add_output(out);

    let before = g.num_nodes();
    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
    assert!(g.nodes.values().all(|n| n.op_type != "Gelu"));
    assert!(g.nodes.values().any(|n| n.op_type == "Erf"));
    assert_eq!(g.num_nodes(), before);
}

#[test]
fn declines_gelu_wrong_half_constant() {
    // Mul(x, 0.4) instead of 0.5 → decline.
    let mut g = gelu_graph(true, true);
    // Rewrite the half Mul's constant initializer to 0.4.
    let ch = g
        .values
        .iter()
        .find(|(_, v)| v.name.as_deref() == Some("c_half"))
        .map(|(id, _)| id)
        .unwrap();
    g.set_initializer(
        ch,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            vec![],
            0.4f32.to_le_bytes().to_vec(),
        )),
    );
    let before = g.num_nodes();
    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
    assert!(g.nodes.values().all(|n| n.op_type != "Gelu"));
    assert_eq!(g.num_nodes(), before);
}

#[test]
fn declines_gelu_when_half_uses_different_x() {
    // The `0.5··` operand uses a DIFFERENT value than the Erf branch, so the
    // diamond is not closed → decline.
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let x = val(&mut g, "x");
    let y = val(&mut g, "y");
    g.add_input(x);
    g.add_input(y);
    let half = val(&mut g, "half");
    let ch = scalar_init(&mut g, "c_half", 0.5);
    // half = 0.5 * y   (NOT x)
    g.insert_node(Node::new(
        NodeId(0),
        "Mul",
        vec![Some(y), Some(ch)],
        vec![half],
    ));
    let scaled = val(&mut g, "scaled");
    let cs = scalar_init(&mut g, "c_sqrt2", std::f32::consts::SQRT_2);
    g.insert_node(Node::new(
        NodeId(0),
        "Div",
        vec![Some(x), Some(cs)],
        vec![scaled],
    ));
    let e = val(&mut g, "e");
    g.insert_node(Node::new(NodeId(0), "Erf", vec![Some(scaled)], vec![e]));
    let one = scalar_init(&mut g, "c_one", 1.0);
    let a = val(&mut g, "a");
    g.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(e), Some(one)],
        vec![a],
    ));
    let out = val(&mut g, "out");
    g.insert_node(Node::new(
        NodeId(0),
        "Mul",
        vec![Some(half), Some(a)],
        vec![out],
    ));
    g.add_output(out);

    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
    assert!(g.nodes.values().all(|n| n.op_type != "Gelu"));
    assert!(g.nodes.values().any(|n| n.op_type == "Erf"));
}

#[test]
fn declines_gelu_when_interior_escapes() {
    // The Erf output feeds an extra external consumer, so fusing would
    // delete an observed value → decline.
    let mut g = gelu_graph(true, true);
    let e = g
        .values
        .iter()
        .find(|(_, v)| v.name.as_deref() == Some("e"))
        .map(|(id, _)| id)
        .unwrap();
    let side = val(&mut g, "side");
    g.insert_node(Node::new(NodeId(0), "Erf", vec![Some(e)], vec![side]));
    g.add_output(side);

    OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
    assert!(
        g.nodes.values().all(|n| n.op_type != "Gelu"),
        "must not fuse when an interior value escapes"
    );
}

fn run_restart_reference(patterns: &[FusionPattern], graph: &mut Graph) {
    for pattern in patterns {
        while let Some(matched) = pattern.find_match(graph) {
            pattern.apply_fusion(graph, &matched).unwrap();
        }
    }
}

fn serialized_graph_bytes(mut graph: Graph) -> Vec<u8> {
    use std::fmt::Write;

    let mut snapshot = String::new();
    writeln!(&mut snapshot, "inputs={:?}", graph.inputs).unwrap();
    writeln!(&mut snapshot, "outputs={:?}", graph.outputs).unwrap();

    let mut initializers: Vec<_> = graph.initializers.iter().collect();
    initializers.sort_by_key(|(id, _)| id.0);
    writeln!(&mut snapshot, "initializers={initializers:?}").unwrap();
    let mut constraints: Vec<_> = graph.symbol_constraints.iter().collect();
    constraints.sort_by_key(|(id, _)| id.0);
    writeln!(&mut snapshot, "constraints={constraints:?}").unwrap();
    let mut opsets: Vec<_> = graph.opset_imports.iter().collect();
    opsets.sort_by_key(|(domain, _)| *domain);
    writeln!(&mut snapshot, "opsets={opsets:?}").unwrap();
    let mut subgraphs: Vec<_> = graph.subgraphs.iter().collect();
    subgraphs.sort_by_key(|((id, name), _)| (id.0, name.as_str()));
    writeln!(&mut snapshot, "subgraphs={subgraphs:?}").unwrap();

    for (id, node) in graph.nodes.iter() {
        let mut attributes: Vec<_> = node.attributes.iter().collect();
        attributes.sort_by_key(|(name, _)| *name);
        writeln!(
            &mut snapshot,
            "node={id:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{attributes:?}|{:?}|{:?}|{:?}",
            node.id,
            node.name,
            node.op_type,
            node.domain,
            node.inputs,
            node.outputs,
            node.doc_string,
            node.device,
            node.exec_order,
        )
        .unwrap();
    }
    for (id, value) in graph.values.iter() {
        writeln!(&mut snapshot, "value={id:?}|{value:?}").unwrap();
    }
    writeln!(
        &mut snapshot,
        "topological_order={:?}",
        graph.topological_order().unwrap()
    )
    .unwrap();

    // Arena slots/free-list order are private IR details, but their complete
    // observable state is the sequence of IDs returned by future inserts.
    // The generated graphs are far smaller than this probe count.
    for _ in 0..128 {
        let node = graph.insert_node(Node::new(NodeId(0), "ArenaProbe", Vec::new(), Vec::new()));
        let value = graph.create_value(DataType::Float32, static_shape([1]));
        writeln!(&mut snapshot, "probe={node:?}|{value:?}").unwrap();
    }
    snapshot.into_bytes()
}

fn assert_fusion_graphs_byte_identical(actual: Graph, expected: Graph, trial: usize) {
    assert_eq!(
        serialized_graph_bytes(actual),
        serialized_graph_bytes(expected),
        "restart and resumable fixpoints differ byte-for-byte on trial {trial}"
    );
}

struct FusionTestRng(u64);

impl FusionTestRng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn usize(&mut self, upper: usize) -> usize {
        (self.next() as usize) % upper
    }

    fn coin(&mut self) -> bool {
        self.next() & 1 == 0
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            values.swap(i, self.usize(i + 1));
        }
    }
}

struct DifferentialGraphBuilder {
    graph: Graph,
    pending: Vec<Node>,
    next_name: usize,
}

impl DifferentialGraphBuilder {
    fn new() -> Self {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        Self {
            graph,
            pending: Vec::new(),
            next_name: 0,
        }
    }

    fn value(&mut self, prefix: &str, dims: &[usize]) -> ValueId {
        let name = format!("{prefix}_{}", self.next_name);
        self.next_name += 1;
        self.graph
            .create_named_value(name, DataType::Float32, static_shape(dims.iter().copied()))
    }

    fn input(&mut self, prefix: &str, dims: &[usize]) -> ValueId {
        let value = self.value(prefix, dims);
        self.graph.add_input(value);
        value
    }

    fn scalar(&mut self, prefix: &str, value: f32) -> ValueId {
        let id = self.value(prefix, &[]);
        self.graph.set_initializer(
            id,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![],
                value.to_le_bytes().to_vec(),
            )),
        );
        id
    }

    fn node(
        &mut self,
        name: impl Into<String>,
        op: &str,
        inputs: Vec<ValueId>,
        output: ValueId,
    ) -> &mut Node {
        let mut node = Node::new(
            NodeId(0),
            op,
            inputs.into_iter().map(Some).collect(),
            vec![output],
        );
        node.name = name.into();
        self.pending.push(node);
        self.pending.last_mut().unwrap()
    }

    fn output(&mut self, value: ValueId) {
        self.graph.add_output(value);
    }

    fn add_matmul_bias(&mut self, rng: &mut FusionTestRng, relu: bool, overlap: bool) {
        let a = self.input("mm_a", &[4]);
        let w0 = self.input("mm_w0", &[4]);
        let m0 = self.value("mm_m0", &[4]);
        self.node("mm_left", "MatMul", vec![a, w0], m0);

        let bias = if overlap {
            let w1 = self.input("mm_w1", &[4]);
            let m1 = self.value("mm_m1", &[4]);
            self.node("mm_right", "MatMul", vec![a, w1], m1);
            m1
        } else {
            self.input("mm_bias", &[4])
        };

        let add = self.value("mm_add", &[4]);
        let add_inputs = if rng.coin() {
            vec![m0, bias]
        } else {
            vec![bias, m0]
        };
        self.node("mm_add_node", "Add", add_inputs, add);
        if relu {
            let output = self.value("mm_relu", &[4]);
            self.node("mm_relu_node", "Relu", vec![add], output);
            self.output(output);
        } else {
            self.output(add);
        }
    }

    fn add_layernorm(&mut self, split_diff: bool) {
        let x = self.input("ln_x", &[4]);
        let two = self.input("ln_two", &[4]);
        let eps = self.scalar("ln_eps", 1e-12);
        let scale = self.input("ln_scale", &[4]);
        let bias = self.input("ln_bias", &[4]);

        let mean = self.value("ln_mean", &[4]);
        let rm1 = self.node("ln_mean_node", "ReduceMean", vec![x], mean);
        rm1.attributes
            .insert("axes".into(), Attribute::Ints(vec![-1]));
        rm1.attributes.insert("keepdims".into(), Attribute::Int(1));

        let diff_pow = self.value("ln_diff_pow", &[4]);
        self.node("ln_sub_pow", "Sub", vec![x, mean], diff_pow);
        let diff_div = if split_diff {
            let value = self.value("ln_diff_div", &[4]);
            self.node("ln_sub_div", "Sub", vec![x, mean], value);
            value
        } else {
            diff_pow
        };
        let sq = self.value("ln_sq", &[4]);
        self.node("ln_pow", "Pow", vec![diff_pow, two], sq);
        let var = self.value("ln_var", &[4]);
        let rm2 = self.node("ln_var_node", "ReduceMean", vec![sq], var);
        rm2.attributes
            .insert("axes".into(), Attribute::Ints(vec![-1]));
        rm2.attributes.insert("keepdims".into(), Attribute::Int(1));
        let vare = self.value("ln_vare", &[4]);
        self.node("ln_add_eps", "Add", vec![var, eps], vare);
        let std = self.value("ln_std", &[4]);
        self.node("ln_sqrt", "Sqrt", vec![vare], std);
        let norm = self.value("ln_norm", &[4]);
        self.node("ln_div", "Div", vec![diff_div, std], norm);
        let scaled = self.value("ln_scaled", &[4]);
        self.node("ln_mul", "Mul", vec![norm, scale], scaled);
        let output = self.value("ln_output", &[4]);
        self.node("ln_add_bias", "Add", vec![scaled, bias], output);
        self.output(output);
    }

    fn add_gelu(&mut self, rng: &mut FusionTestRng) {
        let x = self.input("gelu_x", &[4]);
        let half = self.value("gelu_half", &[4]);
        if rng.coin() {
            let c = self.scalar("gelu_half_c", 0.5);
            let inputs = if rng.coin() { vec![x, c] } else { vec![c, x] };
            self.node("gelu_half_node", "Mul", inputs, half);
        } else {
            let c = self.scalar("gelu_two_c", 2.0);
            self.node("gelu_half_node", "Div", vec![x, c], half);
        }

        let scaled = self.value("gelu_scaled", &[4]);
        if rng.coin() {
            let c = self.scalar("gelu_sqrt2", std::f32::consts::SQRT_2);
            self.node("gelu_inner", "Div", vec![x, c], scaled);
        } else {
            let c = self.scalar("gelu_isqrt2", std::f32::consts::FRAC_1_SQRT_2);
            let inputs = if rng.coin() { vec![x, c] } else { vec![c, x] };
            self.node("gelu_inner", "Mul", inputs, scaled);
        }
        let erf = self.value("gelu_erf", &[4]);
        self.node("gelu_erf_node", "Erf", vec![scaled], erf);
        let one = self.scalar("gelu_one", 1.0);
        let plus_one = self.value("gelu_plus_one", &[4]);
        let inputs = if rng.coin() {
            vec![erf, one]
        } else {
            vec![one, erf]
        };
        self.node("gelu_add", "Add", inputs, plus_one);
        let output = self.value("gelu_output", &[4]);
        let inputs = if rng.coin() {
            vec![half, plus_one]
        } else {
            vec![plus_one, half]
        };
        self.node("gelu_outer", "Mul", inputs, output);
        self.output(output);
    }

    fn add_attention(&mut self, rng: &mut FusionTestRng) {
        let q = self.input("attn_q", &[1, 2, 3, 4]);
        let k = self.input("attn_k", &[1, 2, 3, 4]);
        let v = self.input("attn_v", &[1, 2, 3, 4]);
        let k_side = if rng.coin() {
            let kt = self.value("attn_kt", &[1, 2, 4, 3]);
            let transpose = self.node("attn_transpose", "Transpose", vec![k], kt);
            transpose
                .attributes
                .insert("perm".into(), Attribute::Ints(vec![0, 1, 3, 2]));
            kt
        } else {
            k
        };
        let scores = self.value("attn_scores", &[1, 2, 3, 3]);
        self.node("attn_score_mm", "MatMul", vec![q, k_side], scores);
        let scale_const = if rng.coin() {
            self.scalar("attn_divisor", 2.0)
        } else {
            self.scalar("attn_multiplier", 0.5)
        };
        let scaled = self.value("attn_scaled", &[1, 2, 3, 3]);
        if self
            .graph
            .value(scale_const)
            .name
            .as_deref()
            .unwrap()
            .contains("divisor")
        {
            self.node("attn_scale", "Div", vec![scores, scale_const], scaled);
        } else {
            let inputs = if rng.coin() {
                vec![scores, scale_const]
            } else {
                vec![scale_const, scores]
            };
            self.node("attn_scale", "Mul", inputs, scaled);
        }
        let softmax_input = if rng.coin() {
            let mask = self.input("attn_mask", &[1, 1, 3, 3]);
            let masked = self.value("attn_masked", &[1, 2, 3, 3]);
            let inputs = if rng.coin() {
                vec![scaled, mask]
            } else {
                vec![mask, scaled]
            };
            self.node("attn_mask_add", "Add", inputs, masked);
            masked
        } else {
            scaled
        };
        let probs = self.value("attn_probs", &[1, 2, 3, 3]);
        let softmax = self.node("attn_softmax", "Softmax", vec![softmax_input], probs);
        softmax.attributes.insert(
            "axis".into(),
            Attribute::Int(if rng.coin() { -1 } else { 3 }),
        );
        let output = self.value("attn_output", &[1, 2, 3, 4]);
        self.node("attn_output_mm", "MatMul", vec![probs, v], output);
        self.output(output);
    }

    fn add_resumable_chain(&mut self, rng: &mut FusionTestRng) {
        let input = self.input("chain_input", &[4]);
        let first = self.value("chain_start", &[4]);
        self.node("chain_0", "ChainStart", vec![input], first);
        let mut value = first;
        for index in 1..(4 + rng.usize(6)) {
            let output = self.value("chain_link", &[4]);
            self.node(format!("chain_{index}"), "ChainLink", vec![value], output);
            value = output;
        }
        self.output(value);
    }

    fn add_noise(&mut self, rng: &mut FusionTestRng) {
        for index in 0..rng.usize(8) {
            let input = self.input("noise_input", &[4]);
            let output = self.value("noise_output", &[4]);
            let op = ["Abs", "Neg", "Identity", "Tanh"][rng.usize(4)];
            self.node(format!("noise_{index}"), op, vec![input], output);
            self.output(output);
        }
    }

    fn finish(mut self, rng: &mut FusionTestRng) -> Graph {
        // Seed and remove one dummy per real node. Random removal order
        // randomizes the arena free-list; independently shuffling real-node
        // insertion then decouples logical/topological order from NodeId.
        let mut slots = Vec::with_capacity(self.pending.len());
        for _ in 0..self.pending.len() {
            slots.push(self.graph.insert_node(Node::new(
                NodeId(0),
                "IdSeed",
                Vec::new(),
                Vec::new(),
            )));
        }
        rng.shuffle(&mut slots);
        for id in slots {
            self.graph.remove_node(id);
        }
        rng.shuffle(&mut self.pending);
        for node in self.pending {
            self.graph.insert_node(node);
        }
        self.graph
    }
}

fn randomized_fusion_graph(rng: &mut FusionTestRng) -> Graph {
    let mut builder = DifferentialGraphBuilder::new();
    // Every trial contains every registered matcher. The two structural
    // motifs deliberately have two MatMul starts sharing the Add (and Relu),
    // so lowest-NodeId overlap resolution affects the exact replacement.
    builder.add_attention(rng);
    builder.add_matmul_bias(rng, true, true);
    builder.add_layernorm(rng.coin());
    builder.add_gelu(rng);
    builder.add_matmul_bias(rng, false, true);

    // Add extra independent registered motifs for structural diversity.
    for _ in 0..rng.usize(4) {
        match rng.usize(5) {
            0 => builder.add_attention(rng),
            1 => {
                let overlap = rng.coin();
                builder.add_matmul_bias(rng, true, overlap);
            }
            2 => builder.add_layernorm(rng.coin()),
            3 => builder.add_gelu(rng),
            _ => {
                let overlap = rng.coin();
                builder.add_matmul_bias(rng, false, overlap);
            }
        }
    }
    builder.add_resumable_chain(rng);
    builder.add_noise(rng);
    builder.finish(rng)
}

fn differential_patterns() -> Vec<FusionPattern> {
    let mut patterns = default_fusion_patterns();
    // Unlike production replacements, this test-only standard-domain op can
    // immediately match the next ChainLink. Its NodeId has already been
    // passed by the ascending cursor, so correctness requires a lower-id
    // revisit after every fusion until the chain reaches its fixpoint.
    patterns.push(
        FusionPattern::new("ResumableChain", &["ChainStart", "ChainLink"], "ChainStart")
            .with_replacement_domain(""),
    );
    patterns
}

struct AffectedRevisitCase {
    graph: Graph,
    lower_start: NodeId,
    later_start: NodeId,
    first_middle: NodeId,
    first_tail: NodeId,
    final_tail: NodeId,
}

fn affected_revisit_case(seed: u64) -> AffectedRevisitCase {
    let mut rng = FusionTestRng(seed ^ 0xbb67_ae85_84ca_a73b);
    let noise_count = 3 + rng.usize(6);
    let slot_count = 5 + noise_count;
    let later_start = NodeId((slot_count - 1) as u32);
    let mut lower_ids: Vec<u32> = (0..later_start.0).collect();
    rng.shuffle(&mut lower_ids);
    let lower_start = NodeId(lower_ids[0]);
    let first_middle = NodeId(lower_ids[1]);
    let first_tail = NodeId(lower_ids[2]);
    let final_tail = NodeId(lower_ids[3]);

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let lower_input = graph.create_named_value(
        format!("lower_input_{seed}"),
        DataType::Float32,
        static_shape([4]),
    );
    let later_input = graph.create_named_value(
        format!("later_input_{seed}"),
        DataType::Float32,
        static_shape([4]),
    );
    graph.add_input(lower_input);
    graph.add_input(later_input);
    let lower_value = graph.create_named_value(
        format!("lower_value_{seed}"),
        DataType::Float32,
        static_shape([4]),
    );
    let middle_value = graph.create_named_value(
        format!("middle_value_{seed}"),
        DataType::Float32,
        static_shape([4]),
    );
    let first_output = graph.create_named_value(
        format!("first_output_{seed}"),
        DataType::Float32,
        static_shape([4]),
    );
    let first_tail_output = graph.create_named_value(
        format!("first_tail_output_{seed}"),
        DataType::Float32,
        static_shape([4]),
    );
    let final_output = graph.create_named_value(
        format!("final_output_{seed}"),
        DataType::Float32,
        static_shape([4]),
    );
    graph.add_output(final_output);

    let mut placements = vec![
        (
            lower_start,
            Node::new(
                NodeId(0),
                "AdversaryStart",
                vec![Some(lower_input)],
                vec![lower_value],
            ),
        ),
        (
            later_start,
            Node::new(
                NodeId(0),
                "AdversaryStart",
                vec![Some(later_input)],
                vec![middle_value],
            ),
        ),
        (
            first_middle,
            Node::new(
                NodeId(0),
                "AdversaryMiddle",
                vec![Some(middle_value)],
                vec![first_output],
            ),
        ),
        (
            first_tail,
            Node::new(
                NodeId(0),
                "AdversaryTail",
                vec![Some(first_output), Some(lower_value)],
                vec![first_tail_output],
            ),
        ),
        (
            final_tail,
            Node::new(
                NodeId(0),
                "AdversaryTail",
                vec![Some(first_tail_output)],
                vec![final_output],
            ),
        ),
    ];
    let core_ids = HashSet::from([
        lower_start,
        later_start,
        first_middle,
        first_tail,
        final_tail,
    ]);
    let mut noise_index = 0;
    for raw_id in 0..slot_count as u32 {
        let id = NodeId(raw_id);
        if core_ids.contains(&id) {
            continue;
        }
        let input = graph.create_named_value(
            format!("noise_input_{seed}_{noise_index}"),
            DataType::Float32,
            static_shape([4]),
        );
        let output = graph.create_named_value(
            format!("noise_output_{seed}_{noise_index}"),
            DataType::Float32,
            static_shape([4]),
        );
        graph.add_input(input);
        graph.add_output(output);
        placements.push((
            id,
            Node::new(
                NodeId(0),
                ["Abs", "Neg", "Identity", "Tanh"][rng.usize(4)],
                vec![Some(input)],
                vec![output],
            ),
        ));
        noise_index += 1;
    }

    rng.shuffle(&mut placements);
    for _ in 0..slot_count {
        graph.insert_node(Node::new(NodeId(0), "IdSeed", Vec::new(), Vec::new()));
    }
    for &(target, _) in placements.iter().rev() {
        graph.remove_node(target);
    }
    for (expected, node) in placements {
        assert_eq!(graph.insert_node(node), expected);
    }

    AffectedRevisitCase {
        graph,
        lower_start,
        later_start,
        first_middle,
        first_tail,
        final_tail,
    }
}

#[test]
fn affected_candidate_starts_revisits_newly_eligible_lower_ids() {
    const TRIALS: usize = 5_000;
    let pattern = FusionPattern::new(
        "AffectedBehindCursor",
        &["AdversaryStart", "AdversaryMiddle", "AdversaryTail"],
        "AdversaryMiddle",
    )
    .with_replacement_domain("");
    let patterns = vec![pattern.clone()];
    let mut reclaimable_low_slot_trials = 0;
    let mut affected_scheduled = 0;
    let mut affected_revisit_hits = 0;

    for trial in 0..TRIALS {
        let case = affected_revisit_case(trial as u64);
        assert!(case.graph.validate().is_ok(), "invalid trial {trial}");
        assert!(case.lower_start.0 < case.later_start.0);
        assert!(case.first_middle.0 < case.later_start.0);
        assert!(case.first_tail.0 < case.later_start.0);
        assert!(
            pattern
                .try_match_at(&case.graph, case.lower_start)
                .is_none()
        );
        let first_match = pattern
            .try_match_at(&case.graph, case.later_start)
            .expect("later start must be the first eligible match");
        assert_eq!(
            first_match.nodes,
            vec![case.later_start, case.first_middle, case.first_tail]
        );

        // Reverse removal followed by LIFO insertion always reuses the
        // match-start slot. The lower interior slots remain reclaimable.
        let mut reclaim_probe = case.graph.clone();
        let first_fused = pattern
            .apply_fusion_returning_id(&mut reclaim_probe, &first_match)
            .unwrap();
        assert_eq!(first_fused, case.later_start);
        let probe_id =
            reclaim_probe.insert_node(Node::new(NodeId(0), "ReclaimProbe", Vec::new(), Vec::new()));
        assert_eq!(probe_id, case.first_middle);
        reclaimable_low_slot_trials += 1;

        let mut reference = case.graph.clone();
        run_restart_reference(&patterns, &mut reference);
        let mut actual = case.graph;
        let mut lower_was_scheduled = false;
        let mut trial_hits = 0;
        OpFusion::with_patterns(patterns.clone())
            .run_with_fusion_observer(
                &mut actual,
                |name, source, start, matched, affected, fused_id| {
                    if name != "AffectedBehindCursor" {
                        return;
                    }
                    assert_eq!(
                        fused_id, matched[0],
                        "replacement must reuse the just-freed match-start slot"
                    );
                    if start == case.later_start {
                        assert_eq!(source, ScanCandidateSource::Initial);
                        assert_eq!(
                            matched,
                            &[case.later_start, case.first_middle, case.first_tail]
                        );
                        assert!(affected.contains(&case.lower_start));
                        assert_ne!(fused_id, case.lower_start);
                        lower_was_scheduled = true;
                        affected_scheduled += 1;
                    } else if start == case.lower_start {
                        assert!(lower_was_scheduled);
                        assert_eq!(source, ScanCandidateSource::Revisit);
                        assert_eq!(
                            matched,
                            &[case.lower_start, case.later_start, case.final_tail]
                        );
                        trial_hits += 1;
                        affected_revisit_hits += 1;
                    }
                },
            )
            .unwrap();

        assert_eq!(trial_hits, 1, "affected revisit not hit on trial {trial}");
        assert!(actual.validate().is_ok(), "invalid result on trial {trial}");
        assert_fusion_graphs_byte_identical(actual, reference, trial);
    }

    assert_eq!(reclaimable_low_slot_trials, TRIALS);
    assert_eq!(affected_scheduled, TRIALS);
    assert_eq!(affected_revisit_hits, TRIALS);
    eprintln!(
        "affected behind-cursor revisit hits: {affected_revisit_hits}/{TRIALS} \
         ({}%); reclaimable lower slots present: {reclaimable_low_slot_trials}/{TRIALS}",
        affected_revisit_hits * 100 / TRIALS
    );
}

fn assert_overlapping_structural_candidates(graph: &Graph, patterns: &[FusionPattern]) {
    let mut saw_gemm_overlap = false;
    let mut saw_bias_overlap = false;
    for (add_id, add) in graph.nodes.iter().filter(|(_, node)| node.op_type == "Add") {
        let starts: Vec<_> = add
            .input_values()
            .filter_map(|value| graph.value(value).producer)
            .filter(|&producer| graph.node(producer).op_type == "MatMul")
            .collect();
        if starts.len() != 2 {
            continue;
        }
        let has_relu = graph
            .successors(add_id)
            .iter()
            .any(|&successor| graph.node(successor).op_type == "Relu");
        let pattern = if has_relu { &patterns[1] } else { &patterns[4] };
        assert!(
            starts
                .iter()
                .all(|&start| pattern.try_match_at(graph, start).is_some()),
            "both MatMul starts must be eligible for the shared structural tail"
        );
        saw_gemm_overlap |= has_relu;
        saw_bias_overlap |= !has_relu;
    }
    assert!(
        saw_gemm_overlap,
        "missing shared MatMul+Add+Relu candidates"
    );
    assert!(saw_bias_overlap, "missing shared MatMul+Add candidates");
}

#[test]
fn resumable_scan_matches_restart_reference_on_randomized_graphs() {
    const TRIALS: usize = 5_000;
    const REGISTERED_PATTERNS: usize = 5;
    let mut rng = FusionTestRng(0x6a09_e667_f3bc_c909);
    let patterns = differential_patterns();
    assert_eq!(default_fusion_patterns().len(), REGISTERED_PATTERNS);

    let mut saw_non_topological_ids = false;
    for trial in 0..TRIALS {
        let mut graph = randomized_fusion_graph(&mut rng);
        assert!(
            graph.validate().is_ok(),
            "invalid input graph on trial {trial}"
        );

        for pattern in &patterns[..REGISTERED_PATTERNS] {
            assert!(
                pattern.find_match(&graph).is_some(),
                "{} was not exercised on trial {trial}",
                pattern.pattern_name()
            );
        }
        assert_overlapping_structural_candidates(&graph, &patterns);
        assert!(
            graph
                .nodes
                .values()
                .filter(|node| node.op_type == "ChainLink")
                .count()
                >= 3,
            "chained replacement adversary must require multiple revisits"
        );
        saw_non_topological_ids |=
            graph.topological_order().unwrap() != graph.nodes.keys().collect::<Vec<_>>();

        let mut reference = graph.clone();
        run_restart_reference(&patterns, &mut reference);
        OpFusion::with_patterns(patterns.clone())
            .run(&mut graph, &PassContext::new())
            .unwrap();

        assert!(
            graph.validate().is_ok(),
            "invalid result graph on trial {trial}"
        );
        assert!(
            graph.nodes.values().all(|node| node.op_type != "ChainLink"),
            "resumable chain did not reach its fixpoint on trial {trial}"
        );
        assert_fusion_graphs_byte_identical(graph, reference, trial);
    }
    assert!(
        saw_non_topological_ids,
        "randomized insertion must decouple NodeId from topological order"
    );
}
