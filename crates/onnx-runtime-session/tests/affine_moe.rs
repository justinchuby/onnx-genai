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
    const BLOCK: usize = 16;
    let blocks = d / BLOCK;
    let w = input(g, names[0], D::Uint8, &[3, n, d / 2]);
    let s = input(g, names[1], D::Float32, &[3, n, blocks]);
    let z = input(g, names[2], D::Uint8, &[3, n, 1]);
    let codes = unpack(g, w, 3, n, d / 2, d as i64);
    let zero = unpack(g, z, 3, n, 1, -1);
    let zero_axis = ints(g, &[2]);
    let zero_start = ints(g, &[0]);
    let zero_end = ints(g, &[blocks as i64]);
    let zero = op(
        g,
        "Slice",
        &[zero, zero_start, zero_end, zero_axis],
        D::Uint8,
        &[3, n, blocks],
        &[],
    );
    let attr = [("to", A::Int(21))];
    let codes = op(g, "Cast", &[codes], D::Uint4, &[3, n, d], &attr);
    let zero = op(g, "Cast", &[zero], D::Uint4, &[3, n, blocks], &attr);
    op(
        g,
        "DequantizeLinear",
        &[codes, s, zero],
        D::Float32,
        &[3, n, d],
        &[("axis", A::Int(2)), ("block_size", A::Int(BLOCK as i64))],
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

fn unfused_reference(graph: &Graph, banks: [ValueId; 2]) -> Graph {
    let mut reference = graph.clone();
    for bank in banks {
        let dq = reference.value(bank).producer.unwrap();
        assert_eq!(reference.node(dq).op_type, "DequantizeLinear");
        // Standalone Cast cannot write UINT4. The fixture's nibble unpack
        // bounds both codes and zero points to 0..=15 before the Casts. Keeping
        // both as UINT8 preserves the values and the blocked affine DQ math.
        for index in [0, 2] {
            let packed = reference.node(dq).inputs[index].unwrap();
            let cast = reference.value(packed).producer.unwrap();
            assert_eq!(reference.node(cast).op_type, "Cast");
            let lanes = reference.node(cast).inputs[0].unwrap();
            assert_eq!(reference.value(packed).dtype, D::Uint4);
            assert_eq!(reference.value(lanes).dtype, D::Uint8);
            reference.replace_input(dq, index, Some(lanes));
            reference.remove_node(cast);
        }
        // Keep both separate DQ and standard GroupedMatMul on the oracle path.
        reference.add_output(bank);
    }
    reference
}

#[test]
fn affine_moe_cpu_executes_selected_fusion_against_standard_graph() {
    for t in [1, 4] {
        let Fixture {
            mut graph, banks, ..
        } = fixture(t);
        let mut reference = InferenceSession::from_graph(unfused_reference(&graph, banks)).unwrap();
        // Serialized ONNX uses names as SSA edges; the original in-memory
        // fixture deliberately left its intermediate values anonymous.
        let value_ids: Vec<_> = graph.values.iter().map(|(id, _)| id).collect();
        for id in value_ids {
            let value = graph.value_mut(id);
            if value.name.is_none() {
                value.name = Some(format!("moe_value_{}", id.0));
            }
        }
        let source_bytes =
            onnx_runtime_loader::encode_model(&onnx_runtime_loader::Model::new(&graph)).unwrap();
        let decoded = onnx_runtime_loader::load_model_bytes(&source_bytes).unwrap();
        let mut fused = InferenceSession::from_graph(decoded).unwrap();
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
            fused.graph().opset_imports.get("pkg.nxrt._private"),
            Some(&1)
        );
        let fused_bytes =
            onnx_runtime_loader::encode_model(&onnx_runtime_loader::Model::new(fused.graph()))
                .unwrap();
        let mut reloaded = InferenceSession::from_graph(
            onnx_runtime_loader::load_model_bytes(&fused_bytes).unwrap(),
        )
        .unwrap();
        let target = fused.graph().node(
            fused
                .graph()
                .value(fused.graph().outputs[0])
                .producer
                .unwrap(),
        );
        assert_eq!(target.domain, "pkg.nxrt._private");
        assert_eq!(fused.graph().effective_opset(target), Some(1));
        assert!(
            fused
                .graph()
                .nodes
                .values()
                .all(|node| node.domain != "pkg.nxrt")
        );
        assert!(
            reference
                .graph()
                .nodes
                .values()
                .filter(|n| n.op_type == "GroupedMatMul")
                .all(|n| n.is_default_domain() && reference.graph().effective_opset(n) == Some(29))
        );
        assert_eq!(
            reference
                .graph()
                .nodes
                .values()
                .filter(|n| n.op_type == "GroupedMatMul")
                .count(),
            2
        );
        let mut x: Vec<_> = (0..t * 16).map(|i| (i as f32 % 17.0 - 8.0) / 8.0).collect();
        if t > 1 {
            x[32] = f32::NAN;
        }
        let logits = [
            2.0f32,
            2.0,
            7.0,
            -30.0,
            2.0,
            -31.0,
            -31.0,
            2.0,
            1.0,
            f32::NAN,
            1.0,
            1.0,
        ];
        let packed = |n: usize, d: usize| {
            (0..3 * n * (d / 2))
                .map(|i| (i.wrapping_mul(47) ^ (i / 9)) as u8)
                .collect::<Vec<_>>()
        };
        let w1 = packed(64usize, 16usize);
        let w2 = packed(16usize, 32usize);
        let s2: Vec<_> = (0..96)
            .map(|i| {
                // Expert 0 is selected with a small but nonzero probability on
                // row 1. Its down projection is nonfinite, and must be masked.
                if i < 32 {
                    f32::MAX
                } else if i % 2 == 0 {
                    0.0625
                } else {
                    0.125
                }
            })
            .collect();
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
            ("s2", Tensor::from_f32(&[3, 16, 2], &s2).unwrap()),
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
        let reloaded_output = reloaded.run(&inputs).unwrap();
        let got_bits = got[0]
            .to_vec_f32()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>();
        let reloaded_bits = reloaded_output[0]
            .to_vec_f32()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(
            got_bits, reloaded_bits,
            "private fused model must survive serialization and execute identically"
        );
        let want = reference.run(&inputs).unwrap();
        assert_eq!(got[0].shape, [1, t, 16]);
        let got = got[0].to_vec_f32();
        let want = want[0].to_vec_f32();
        assert!(want[..16].iter().any(|v| v.abs() > 0.01));
        if t > 1 {
            assert!(want[16..32].iter().all(|v| v.is_finite()));
            assert!(want[32..48].iter().all(|v| v.is_nan()));
            assert!(want[48..64].iter().all(|&v| v == 0.0));
        }
        for (got, want) in got.into_iter().zip(want) {
            if want.is_nan() {
                assert!(got.is_nan());
                continue;
            }
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

mod softmax_first {
    use super::*;
    use onnx_runtime_optimizer::affine_moe::softmax_first::{Fusion, OP};
    use onnx_runtime_optimizer::{
        OptimizationPass, PassContext, quantized_grouped::QuantizedGroupedFusion,
    };

    const OUTPUT: &str = "output";

    fn fixture(t: usize) -> Fixture {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 29);
        let x = input(&mut g, "x", D::Float32, &[t, 16]);
        let logits = input(&mut g, "router", D::Float32, &[1, t, 3]);
        let last = [("axis", A::Int(-1))];
        let probability = op(&mut g, "Softmax", &[logits], D::Float32, &[1, t, 3], &last);
        let flat_router = ints(&mut g, &[-1, 3]);
        let flat_probability = op(
            &mut g,
            "Reshape",
            &[probability, flat_router],
            D::Float32,
            &[t, 3],
            &[],
        );
        let cast_float = [("to", A::Int(1))];
        let probability = op(
            &mut g,
            "Cast",
            &[flat_probability],
            D::Float32,
            &[t, 3],
            &cast_float,
        );
        let k = ints(&mut g, &[2]);
        let routing = g.create_value(D::Float32, static_shape([t, 2]));
        let selected = g.create_value(D::Int64, static_shape([t, 2]));
        let mut node = Node::new(
            NodeId(0),
            "TopK",
            vec![Some(probability), Some(k)],
            vec![routing, selected],
        );
        node.attributes.insert("axis".into(), A::Int(-1));
        let top = g.insert_node(node);
        let routing = op(&mut g, "Cast", &[routing], D::Float32, &[t, 2], &cast_float);
        let w1 = bank(&mut g, ["w1", "s1", "z1"], 64, 16);
        let w2 = bank(&mut g, ["w2", "s2", "z2"], 16, 32);
        let trans = [("transB", A::Int(1))];
        let first = op(
            &mut g,
            "GroupedMatMul",
            &[x, w1, selected],
            D::Float32,
            &[t, 2, 64],
            &trans,
        );
        let split = ints(&mut g, &[32, 32]);
        let gate = g.create_value(D::Float32, static_shape([t, 2, 32]));
        let up = g.create_value(D::Float32, static_shape([t, 2, 32]));
        let mut node = Node::new(
            NodeId(0),
            "Split",
            vec![Some(first), Some(split)],
            vec![gate, up],
        );
        node.attributes.insert("axis".into(), A::Int(-1));
        g.insert_node(node);
        let f = |g: &mut Graph, name, inputs: &[ValueId]| {
            op(g, name, inputs, D::Float32, &[t, 2, 32], &[])
        };
        let neg = f(&mut g, "Neg", &[gate]);
        let exp = f(&mut g, "Exp", &[neg]);
        let one = f32_const(&mut g, 1.0);
        let denom = f(&mut g, "Add", &[one, exp]);
        let silu = f(&mut g, "Div", &[gate, denom]);
        let act = f(&mut g, "Mul", &[silu, up]);
        let act_shape = ints(&mut g, &[-1, 32]);
        let id_shape = ints(&mut g, &[-1, 1]);
        let act = op(
            &mut g,
            "Reshape",
            &[act, act_shape],
            D::Float32,
            &[t * 2, 32],
            &[],
        );
        let ids = op(
            &mut g,
            "Reshape",
            &[selected, id_shape],
            D::Int64,
            &[t * 2, 1],
            &[],
        );
        let down = op(
            &mut g,
            "GroupedMatMul",
            &[act, w2, ids],
            D::Float32,
            &[t * 2, 1, 16],
            &trans,
        );
        let down_shape = ints(&mut g, &[-1, 2, 16]);
        let down = op(
            &mut g,
            "Reshape",
            &[down, down_shape],
            D::Float32,
            &[t, 2, 16],
            &[],
        );
        let axis = ints(&mut g, &[2]);
        let routing = op(
            &mut g,
            "Unsqueeze",
            &[routing, axis],
            D::Float32,
            &[t, 2, 1],
            &[],
        );
        let weighted = op(
            &mut g,
            "Mul",
            &[down, routing],
            D::Float32,
            &[t, 2, 16],
            &[],
        );
        let axis = ints(&mut g, &[1]);
        let output = op(
            &mut g,
            "ReduceSum",
            &[weighted, axis],
            D::Float32,
            &[t, 16],
            &[("keepdims", A::Int(0))],
        );
        g.value_mut(output).name = Some(OUTPUT.into());
        g.add_output(output);
        Fixture {
            graph: g,
            selected,
            banks: [w1, w2],
            top,
        }
    }

    fn rewrite(graph: &mut Graph) {
        let ctx = PassContext::new();
        QuantizedGroupedFusion.run(graph, &ctx).unwrap();
        Fusion.run(graph, &ctx).unwrap();
    }

    #[test]
    fn affine_softmax_first_moe_admission() {
        let mut accepted = fixture(2).graph;
        assert_eq!(accepted.opset_imports.len(), 1);
        assert_eq!(accepted.opset_imports.get(""), Some(&29));
        assert!(accepted.nodes.values().all(Node::is_default_domain));
        rewrite(&mut accepted);
        let target = accepted.nodes.values().find(|n| n.op_type == OP).unwrap();
        assert_eq!(target.domain, "pkg.nxrt._private");
        assert_eq!(accepted.opset_imports.get(target.domain.as_str()), Some(&1));
        assert_eq!(accepted.effective_opset(target), Some(1));
        assert!(accepted.nodes.values().all(|n| n.domain != "pkg.nxrt"));
        assert!(accepted.nodes.values().any(|n| n.op_type == "Softmax"));
        assert!(!accepted.nodes.values().any(|n| n.op_type == "TopK"));

        for variant in [
            "smallest",
            "observable_routes",
            "wrong_softmax_axis",
            "wrong_scale_shape",
        ] {
            let Fixture {
                mut graph,
                selected,
                top,
                ..
            } = fixture(2);
            match variant {
                "smallest" => {
                    graph
                        .node_mut(top)
                        .attributes
                        .insert("largest".into(), A::Int(0));
                }
                "observable_routes" => graph.add_output(selected),
                "wrong_softmax_axis" => {
                    let id = graph
                        .nodes
                        .values()
                        .find(|n| n.op_type == "Softmax")
                        .unwrap()
                        .id;
                    graph
                        .node_mut(id)
                        .attributes
                        .insert("axis".into(), A::Int(1));
                }
                "wrong_scale_shape" => {
                    let id = graph
                        .nodes
                        .values()
                        .find(|n| n.op_type == "DequantizeLinear")
                        .unwrap()
                        .id;
                    graph
                        .node_mut(id)
                        .attributes
                        .insert("block_size".into(), A::Int(8));
                }
                _ => unreachable!(),
            }
            rewrite(&mut graph);
            assert!(!graph.nodes.values().any(|n| n.op_type == OP), "{variant}");
            for op in ["TopK", "Split", "ReduceSum"] {
                assert!(
                    graph.nodes.values().any(|n| n.op_type == op),
                    "{variant}: lost {op}"
                );
            }
        }
        let mut reversed_masked = super::fixture(2).graph;
        rewrite(&mut reversed_masked);
        assert!(!reversed_masked.nodes.values().any(|n| n.op_type == OP));
        assert!(reversed_masked.nodes.values().any(|n| n.op_type == "Where"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires a CUDA device and NVRTC"
    )]
    fn affine_softmax_first_moe_native_cuda_matches_decomposed_graph() {
        use onnx_runtime_ep_api::ExecutionProvider;
        use onnx_runtime_ep_cuda::{CudaExecutionProvider, build_cuda_registry_descriptors};
        use std::{path::Path, sync::Arc};

        let ep = Arc::new(CudaExecutionProvider::initialized(0).expect("initialize CUDA device 0"));
        let plugin = CudaExecutionProvider::new_for_ort_plugin()
            .expect("initialize the shipped CUDA provider configuration");
        assert!(
            !plugin
                .registry()
                .supports(OP, onnx_runtime_ir::RUNTIME_DOMAIN, 1)
        );
        assert!(
            !plugin
                .custom_passes()
                .iter()
                .any(|pass| pass.name() == "AffineSoftmaxFirstMoeFusion")
        );
        assert!(
            !build_cuda_registry_descriptors(plugin.registry())
                .iter()
                .any(|d| d.op_type == OP)
        );
        drop(plugin);

        for (t, poison) in [(1, false), (4, false), (1, true)] {
            let Fixture { graph, banks, .. } = fixture(t);
            let mut reference =
                InferenceSession::from_graph(unfused_reference(&graph, banks)).unwrap();
            assert!(
                reference
                    .graph()
                    .nodes
                    .values()
                    .any(|n| n.op_type == "TopK")
            );
            let mut cuda = InferenceSession::from_graph_with_provider(
                graph,
                Arc::new(onnx_runtime_loader::WeightStore::new()),
                Path::new("."),
                ep.clone(),
                None,
            )
            .unwrap();
            let target = cuda
                .graph()
                .nodes
                .values()
                .find(|n| n.op_type == OP)
                .unwrap();
            assert_eq!(target.domain, onnx_runtime_ir::RUNTIME_DOMAIN);
            assert_eq!(cuda.graph().effective_opset(target), Some(1));
            assert!(!cuda.graph().nodes.values().any(|n| n.op_type == "QMoE"));
            assert!(
                cuda.execution_provider_fallback_report().is_none(),
                "{:?}",
                cuda.execution_provider_fallback_report()
            );

            let mut x = vec![0.0; t * 16];
            for row in x.chunks_exact_mut(16) {
                row[0] = 1.0;
            }
            let mut w1 = vec![0; 3 * 64 * 8];
            for expert in w1.chunks_exact_mut(64 * 8) {
                expert[0] = 4;
                expert[32 * 8] = 4;
            }
            let mut w2 = vec![0; 3 * 16 * 16];
            for (expert, code) in w2.chunks_exact_mut(16 * 16).zip([1, 2, 4]) {
                for row in expert.chunks_exact_mut(16) {
                    row[0] = code;
                }
            }
            let mut s2 = vec![1.0; 3 * 16 * 2];
            if poison {
                // Expert 1 has a zero softmax probability, but its nonfinite
                // selected projection must still contribute NaN via 0 * Inf.
                s2[32..64].fill(f32::MAX);
            }
            let mut logits = [
                0.0, 0.0, 0.0, // Lower-ID boundary tie, mass is 2/3.
                1.0, 0.0, -1.0, // Excluded expert still contributes to denominator.
                1e-8, 0.0, 2e-8, // Distinct logits, equal rounded FLOAT probabilities.
                5.0, -1000.0, -1000.0,
            ];
            if poison {
                logits[..3].copy_from_slice(&[0.0, -1000.0, -1000.0]);
            }
            let inputs = [
                ("x", Tensor::from_f32(&[t, 16], &x).unwrap()),
                (
                    "router",
                    Tensor::from_f32(&[1, t, 3], &logits[..t * 3]).unwrap(),
                ),
                (
                    "w1",
                    Tensor::from_raw(D::Uint8, vec![3, 64, 8], &w1).unwrap(),
                ),
                ("s1", Tensor::from_f32(&[3, 64, 1], &[0.25; 192]).unwrap()),
                (
                    "z1",
                    Tensor::from_raw(D::Uint8, vec![3, 64, 1], &[0; 192]).unwrap(),
                ),
                (
                    "w2",
                    Tensor::from_raw(D::Uint8, vec![3, 16, 16], &w2).unwrap(),
                ),
                ("s2", Tensor::from_f32(&[3, 16, 2], &s2).unwrap()),
                (
                    "z2",
                    Tensor::from_raw(D::Uint8, vec![3, 16, 1], &[0; 48]).unwrap(),
                ),
            ];
            let inputs: Vec<_> = inputs
                .iter()
                .map(|(name, tensor)| (*name, tensor))
                .collect();
            let got = if t == 4 {
                let mut bindings = Vec::new();
                for &(name, tensor) in &inputs {
                    let mut binding = cuda
                        .allocate_device_binding(
                            name,
                            None::<String>,
                            tensor.dtype,
                            tensor.shape.clone(),
                            tensor.shape.clone(),
                        )
                        .unwrap_or_else(|e| panic!("allocate CUDA input {name}: {e}"));
                    binding
                        .write_bytes(0, tensor.as_bytes())
                        .unwrap_or_else(|e| panic!("upload CUDA input {name}: {e}"));
                    bindings.push(binding);
                }
                bindings.push(
                    cuda.allocate_device_output_binding(
                        OUTPUT,
                        D::Float32,
                        vec![t, 16],
                        vec![t, 16],
                    )
                    .expect("allocate CUDA MoE output"),
                );
                let device = ep.device_id();
                assert!(!device.is_host_accessible());
                for binding in &bindings {
                    assert_eq!(binding.allocator().device_id(), device);
                }
                let outputs = cuda
                    .run_with_device_bindings(&[], &mut bindings)
                    .expect("run CUDA MoE with device inputs and output");
                assert_eq!(outputs.len(), 1);
                assert!(outputs[0].is_none(), "output must stay device-bound");
                let output = bindings.last_mut().unwrap();
                assert_eq!(output.logical_shape(), &[t, 16]);
                let bytes = output
                    .read_bytes()
                    .expect("explicitly download CUDA MoE output");
                Tensor::from_raw(D::Float32, vec![t, 16], &bytes).unwrap()
            } else {
                cuda.run(&inputs).unwrap().remove(0)
            };
            let want = reference.run(&inputs).unwrap();
            assert_eq!(got.shape, [t, 16]);
            let got = got.to_vec_f32();
            let want = want[0].to_vec_f32();
            let silu = 1.0 / (1.0 + (-1.0f32).exp());
            let near = |a: f32, b: f32| assert!((a - b).abs() < 2e-6, "got {a}, expected {b}");
            for (&a, &b) in got.iter().zip(&want) {
                if poison {
                    assert!(
                        a.is_nan() && b.is_nan(),
                        "unmasked combine returned {a}, reference {b}"
                    );
                } else {
                    near(a, b);
                }
            }
            if !poison {
                near(got[0], silu);
                if t > 1 {
                    near(
                        got[16],
                        silu * (1.0 + 2.0 * (-1.0f32).exp())
                            / (1.0 + (-1.0f32).exp() + (-2.0f32).exp()),
                    );
                    near(got[32], silu);
                    near(got[48], silu);
                }
            }
        }
    }
}
