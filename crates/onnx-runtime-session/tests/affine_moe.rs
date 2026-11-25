use onnx_runtime_ir::{
    Attribute as A, DataType as D, Graph, Node, NodeId, TensorData, ValueId, static_shape,
};
use onnx_runtime_optimizer::affine_moe::OP;
use onnx_runtime_session::{InferenceSession, Tensor};

struct Fixture {
    graph: Graph,
    selected: ValueId,
    banks: [ValueId; 2],
    top: NodeId,
}

fn op(
    g: &mut Graph,
    name: &str,
    inputs: &[ValueId],
    dtype: D,
    shape: &[usize],
    attrs: &[(&str, A)],
) -> ValueId {
    let v = g.create_value(dtype, static_shape(shape.iter().copied()));
    let mut node = Node::new(
        NodeId(0),
        name,
        inputs.iter().copied().map(Some).collect(),
        vec![v],
    );
    node.attributes
        .extend(attrs.iter().map(|(k, a)| ((*k).into(), a.clone())));
    g.insert_node(node);
    v
}

fn ints(g: &mut Graph, v: &[i64]) -> ValueId {
    op(
        g,
        "Constant",
        &[],
        D::Int64,
        &[v.len()],
        &[("value_ints", A::Ints(v.to_vec()))],
    )
}

fn f32_const(g: &mut Graph, v: f32) -> ValueId {
    op(
        g,
        "Constant",
        &[],
        D::Float32,
        &[],
        &[("value_float", A::Float(v))],
    )
}

fn input(g: &mut Graph, name: &str, dtype: D, shape: &[usize]) -> ValueId {
    let v = g.create_named_value(name, dtype, static_shape(shape.iter().copied()));
    g.add_input(v);
    v
}

fn unpack(g: &mut Graph, x: ValueId, e: usize, n: usize, b: usize, final_dim: i64) -> ValueId {
    let byte = |g: &mut Graph, v| {
        op(
            g,
            "Constant",
            &[],
            D::Uint8,
            &[],
            &[(
                "value",
                A::Tensor(TensorData::from_raw(D::Uint8, vec![], vec![v])),
            )],
        )
    };
    let mask = byte(g, 15);
    let shift = byte(g, 4);
    let axis = ints(g, &[-1]);
    let flat = ints(g, &[e as i64, n as i64, final_dim]);
    let low = op(g, "BitwiseAnd", &[x, mask], D::Uint8, &[e, n, b], &[]);
    let high = op(
        g,
        "BitShift",
        &[x, shift],
        D::Uint8,
        &[e, n, b],
        &[("direction", A::String("RIGHT".into()))],
    );
    let lo = op(g, "Unsqueeze", &[low, axis], D::Uint8, &[e, n, b, 1], &[]);
    let hi = op(g, "Unsqueeze", &[high, axis], D::Uint8, &[e, n, b, 1], &[]);
    let pairs = op(
        g,
        "Concat",
        &[lo, hi],
        D::Uint8,
        &[e, n, b, 2],
        &[("axis", A::Int(-1))],
    );
    op(g, "Reshape", &[pairs, flat], D::Uint8, &[e, n, 2 * b], &[])
}

fn bank(g: &mut Graph, names: [&str; 3], n: usize, d: usize) -> ValueId {
    let w = input(g, names[0], D::Uint8, &[3, n, d / 2]);
    let s = input(g, names[1], D::Float32, &[3, n, 1]);
    let z = input(g, names[2], D::Uint8, &[3, n, 1]);
    let codes = unpack(g, w, 3, n, d / 2, d as i64);
    let zero = unpack(g, z, 3, n, 1, -1);
    let zero_axis = ints(g, &[2]);
    let zero_start = ints(g, &[0]);
    let zero_end = ints(g, &[1]);
    let zero = op(
        g,
        "Slice",
        &[zero, zero_start, zero_end, zero_axis],
        D::Uint8,
        &[3, n, 1],
        &[],
    );
    let attr = [("to", A::Int(21))];
    let codes = op(g, "Cast", &[codes], D::Uint4, &[3, n, d], &attr);
    let zero = op(g, "Cast", &[zero], D::Uint4, &[3, n, 1], &attr);
    op(
        g,
        "DequantizeLinear",
        &[codes, s, zero],
        D::Float32,
        &[3, n, d],
        &[("axis", A::Int(2)), ("block_size", A::Int(d as i64))],
    )
}

// Pinned Mobius qmoe.py FLOAT, selected-logit Softmax, clipped interleaved SwiGLU, no biases.
fn fixture(t: usize) -> Fixture {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 29);
    const H: usize = 16;
    const I: usize = 32;
    let x = input(&mut g, "x", D::Float32, &[1, t, H]);
    let router = input(&mut g, "router", D::Float32, &[t, 3]);
    let w1 = bank(&mut g, ["w1", "s1", "z1"], 2 * I, H);
    let w2 = bank(&mut g, ["w2", "s2", "z2"], H, I);
    let start = ints(&mut g, &[2]);
    let end = ints(&mut g, &[-4]);
    let axis1 = ints(&mut g, &[1]);
    let step = ints(&mut g, &[-1]);
    let reversed = op(
        &mut g,
        "Slice",
        &[router, start, end, axis1, step],
        D::Float32,
        &[t, 3],
        &[],
    );
    let k = ints(&mut g, &[2]);
    let logits = g.create_value(D::Float32, static_shape([t, 2]));
    let indices = g.create_value(D::Int64, static_shape([t, 2]));
    let mut top = Node::new(
        NodeId(0),
        "TopK",
        vec![Some(reversed), Some(k)],
        vec![logits, indices],
    );
    top.attributes
        .extend([("axis".into(), A::Int(1)), ("sorted".into(), A::Int(1))]);
    let top = g.insert_node(top);
    let selected = op(&mut g, "Sub", &[start, indices], D::Int64, &[t, 2], &[]);
    let routing = op(
        &mut g,
        "Softmax",
        &[logits],
        D::Float32,
        &[t, 2],
        &[("axis", A::Int(1))],
    );
    let threshold = f32_const(&mut g, 1e-8);
    let active = op(
        &mut g,
        "Greater",
        &[routing, threshold],
        D::Bool,
        &[t, 2],
        &[],
    );
    let flat_shape = ints(&mut g, &[-1, H as i64]);
    let flat = op(
        &mut g,
        "Reshape",
        &[x, flat_shape],
        D::Float32,
        &[t, H],
        &[],
    );
    let trans = [("transB", A::Int(1))];
    let proj1 = op(
        &mut g,
        "GroupedMatMul",
        &[flat, w1, selected],
        D::Float32,
        &[t, 2, 2 * I],
        &trans,
    );
    let gate_start = ints(&mut g, &[0]);
    let up_start = ints(&mut g, &[1]);
    let axis2 = ints(&mut g, &[2]);
    let end = ints(&mut g, &[(2 * I) as i64]);
    let gate = op(
        &mut g,
        "Slice",
        &[proj1, gate_start, end, axis2, axis2],
        D::Float32,
        &[t, 2, I],
        &[],
    );
    let up = op(
        &mut g,
        "Slice",
        &[proj1, up_start, end, axis2, axis2],
        D::Float32,
        &[t, 2, I],
        &[],
    );
    let max = f32_const(&mut g, 0.75);
    let min = f32_const(&mut g, -0.75);
    let beta = f32_const(&mut g, 0.125);
    let gate = op(&mut g, "Min", &[gate, max], D::Float32, &[t, 2, I], &[]);
    let up = op(&mut g, "Clip", &[up, min, max], D::Float32, &[t, 2, I], &[]);
    let up = op(&mut g, "Add", &[up, beta], D::Float32, &[t, 2, I], &[]);
    let act = op(
        &mut g,
        "SwiGLU",
        &[gate, up],
        D::Float32,
        &[t, 2, I],
        &[("alpha", A::Float(1.702))],
    );
    let act_shape = ints(&mut g, &[-1, I as i64]);
    let selected_shape = ints(&mut g, &[-1, 1]);
    let act = op(
        &mut g,
        "Reshape",
        &[act, act_shape],
        D::Float32,
        &[t * 2, I],
        &[],
    );
    let selected_flat = op(
        &mut g,
        "Reshape",
        &[selected, selected_shape],
        D::Int64,
        &[t * 2, 1],
        &[],
    );
    let proj2 = op(
        &mut g,
        "GroupedMatMul",
        &[act, w2, selected_flat],
        D::Float32,
        &[t * 2, 1, H],
        &trans,
    );
    let down_shape = ints(&mut g, &[-1, 2, H as i64]);
    let down = op(
        &mut g,
        "Reshape",
        &[proj2, down_shape],
        D::Float32,
        &[t, 2, H],
        &[],
    );
    let routing = op(
        &mut g,
        "Unsqueeze",
        &[routing, axis2],
        D::Float32,
        &[t, 2, 1],
        &[],
    );
    let active = op(
        &mut g,
        "Unsqueeze",
        &[active, axis2],
        D::Bool,
        &[t, 2, 1],
        &[],
    );
    let weighted = op(&mut g, "Mul", &[down, routing], D::Float32, &[t, 2, H], &[]);
    let zero = f32_const(&mut g, 0.0);
    let masked = op(
        &mut g,
        "Where",
        &[active, weighted, zero],
        D::Float32,
        &[t, 2, H],
        &[],
    );
    let sum = op(
        &mut g,
        "ReduceSum",
        &[masked, axis1],
        D::Float32,
        &[t, H],
        &[("keepdims", A::Int(0))],
    );
    let shape = op(&mut g, "Shape", &[x], D::Int64, &[3], &[]);
    let result = op(
        &mut g,
        "Reshape",
        &[sum, shape],
        D::Float32,
        &[1, t, H],
        &[("allowzero", A::Int(1))],
    );
    g.add_output(result);
    Fixture {
        graph: g,
        selected,
        banks: [w1, w2],
        top,
    }
}

#[test]
fn affine_moe_cpu_executes_selected_fusion_against_standard_graph() {
    for t in [1, 3] {
        let Fixture { graph, banks, .. } = fixture(t);
        let mut reference = graph.clone();
        for bank in banks {
            // Exposing the decoded banks prevents both the projection and full-region rewrites.
            reference.add_output(bank);
        }
        let mut reference = InferenceSession::from_graph(reference).unwrap();
        let mut fused = InferenceSession::from_graph(graph).unwrap();
        assert_eq!(
            fused
                .graph()
                .node(
                    fused
                        .graph()
                        .value(fused.graph().outputs[0])
                        .producer
                        .unwrap()
                )
                .op_type,
            OP
        );
        assert!(fused.execution_provider_fallback_report().is_none());
        assert_eq!(
            reference
                .graph()
                .nodes
                .values()
                .filter(|n| n.op_type == "GroupedMatMul")
                .count(),
            2
        );
        let x: Vec<_> = (0..t * 16).map(|i| (i as f32 % 17.0 - 8.0) / 8.0).collect();
        let logits = [2.0f32, 2.0, 7.0, 2.0, -30.0, -31.0, f32::NAN, 1.0, 1.0];
        let packed = |n: usize, d: usize| {
            (0..3 * n * (d / 2))
                .map(|i| (i.wrapping_mul(47) ^ (i / 9)) as u8)
                .collect::<Vec<_>>()
        };
        let w1 = packed(64usize, 16usize);
        let w2 = packed(16usize, 32usize);
        let tensors = [
            ("x", Tensor::from_f32(&[1, t, 16], &x).unwrap()),
            (
                "router",
                Tensor::from_f32(&[t, 3], &logits[..t * 3]).unwrap(),
            ),
            (
                "w1",
                Tensor::from_raw(D::Uint8, vec![3, 64, 8], &w1).unwrap(),
            ),
            (
                "s1",
                Tensor::from_f32(&[3, 64, 1], &[0.125f32; 192]).unwrap(),
            ),
            (
                "z1",
                Tensor::from_raw(D::Uint8, vec![3, 64, 1], &[0xf7u8; 192]).unwrap(),
            ),
            (
                "w2",
                Tensor::from_raw(D::Uint8, vec![3, 16, 16], &w2).unwrap(),
            ),
            (
                "s2",
                Tensor::from_f32(&[3, 16, 1], &[0.0625f32; 48]).unwrap(),
            ),
            (
                "z2",
                Tensor::from_raw(D::Uint8, vec![3, 16, 1], &[0xe9u8; 48]).unwrap(),
            ),
        ];
        let inputs: Vec<_> = tensors
            .iter()
            .map(|(name, tensor)| (*name, tensor))
            .collect();
        let got = fused.run(&inputs).unwrap();
        let want = reference.run(&inputs).unwrap();
        assert_eq!(got[0].shape, [1, t, 16]);
        let got = got[0].to_vec_f32();
        let want = want[0].to_vec_f32();
        assert!(want[..16].iter().any(|v| v.abs() > 0.01));
        for (got, want) in got.into_iter().zip(want) {
            assert!(
                (got - want).abs() <= 1e-6 * (1.0 + want.abs()),
                "{got} != {want}"
            );
        }
    }
}

#[test]
fn affine_moe_declines_observable_routes_and_changed_topk() {
    for expose in [true, false] {
        let Fixture {
            mut graph,
            selected,
            top,
            ..
        } = fixture(2);
        if expose {
            graph.add_output(selected);
        } else {
            graph
                .node_mut(top)
                .attributes
                .insert("largest".into(), A::Int(0));
        }
        let session = InferenceSession::from_graph(graph).unwrap();
        assert!(!session.graph().nodes.values().any(|n| n.op_type == OP));
        assert!(session.graph().nodes.values().any(|n| n.op_type == "TopK"));
    }
}
