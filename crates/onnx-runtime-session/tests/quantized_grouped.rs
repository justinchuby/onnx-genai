use onnx_runtime_ir::{
    Attribute, DataType as D, Graph, GraphAttribute, Node, NodeId, TensorData, quantized_grouped,
    static_shape,
};
use onnx_runtime_optimizer::{
    OptimizationPass, PassContext,
    quantized_grouped::{QuantizedGroupedFusion, explain_at},
};
use onnx_runtime_session::{InferenceSession, Tensor};

enum Spelling {
    Unpack,
    Bitwise,
    BitwiseWithZero,
    Dense,
}

fn graph(spelling: Spelling) -> Graph {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 29);
    let mut input = |n, d, s: &[usize]| {
        let v = g.create_named_value(n, d, static_shape(s.iter().copied()));
        g.add_input(v);
        v
    };
    let a = input("A", D::Float32, &[2, 4]);
    let idx = input("I", D::Int64, &[2, 2]);
    let dense = matches!(spelling, Spelling::Dense);
    let w = if dense {
        input("W", D::Float32, &[2, 2, 4])
    } else {
        input("W", D::Uint8, &[2, 2, 2])
    };
    let s = (!dense).then(|| input("S", D::Float32, &[2, 2, 1]));
    let z = matches!(spelling, Spelling::Unpack | Spelling::BitwiseWithZero)
        .then(|| input("Z", D::Uint8, &[2, 2, 1]));
    let mut op = |name, inputs, dtype, shape: &[usize], attrs: &[(&str, Attribute)]| {
        let v = g.create_value(dtype, static_shape(shape.iter().copied()));
        let mut n = Node::new(NodeId(0), name, inputs, vec![v]);
        n.attributes
            .extend(attrs.iter().map(|(k, a)| (k.to_string(), a.clone())));
        g.insert_node(n);
        v
    };
    let mut bitwise_unpack = |value, bytes: usize, infer_width: bool| {
        let u8_const = |n| Attribute::Tensor(TensorData::from_raw(D::Uint8, vec![], vec![n]));
        let mask = op(
            "Constant",
            vec![],
            D::Uint8,
            &[],
            &[("value", u8_const(15))],
        );
        let shift = op("Constant", vec![], D::Uint8, &[], &[("value", u8_const(4))]);
        let axes = op(
            "Constant",
            vec![],
            D::Int64,
            &[1],
            &[("value_ints", Attribute::Ints(vec![-1]))],
        );
        let shape = op(
            "Constant",
            vec![],
            D::Int64,
            &[3],
            &[(
                "value_ints",
                Attribute::Ints(vec![
                    2,
                    2,
                    if infer_width { -1 } else { (bytes * 2) as i64 },
                ]),
            )],
        );
        let low = op(
            "BitwiseAnd",
            vec![Some(value), Some(mask)],
            D::Uint8,
            &[2, 2, bytes],
            &[],
        );
        let high = op(
            "BitShift",
            vec![Some(value), Some(shift)],
            D::Uint8,
            &[2, 2, bytes],
            &[("direction", Attribute::String("RIGHT".into()))],
        );
        let low = op(
            "Unsqueeze",
            vec![Some(low), Some(axes)],
            D::Uint8,
            &[2, 2, bytes, 1],
            &[],
        );
        let high = op(
            "Unsqueeze",
            vec![Some(high), Some(axes)],
            D::Uint8,
            &[2, 2, bytes, 1],
            &[],
        );
        let pairs = op(
            "Concat",
            vec![Some(low), Some(high)],
            D::Uint8,
            &[2, 2, bytes, 2],
            &[("axis", Attribute::Int(-1))],
        );
        op(
            "Reshape",
            vec![Some(pairs), Some(shape)],
            D::Uint8,
            &[2, 2, bytes * 2],
            &[],
        )
    };
    let bitwise = matches!(spelling, Spelling::Bitwise | Spelling::BitwiseWithZero);
    let expanded_weights = bitwise.then(|| bitwise_unpack(w, 2, false));
    let expanded_zero = if matches!(spelling, Spelling::BitwiseWithZero) {
        z.map(|z| bitwise_unpack(z, 1, true))
    } else {
        None
    };
    let weights = match spelling {
        Spelling::Dense => w,
        Spelling::Unpack => {
            let count = op(
                "Constant",
                vec![],
                D::Int64,
                &[],
                &[("value_int", Attribute::Int(4))],
            );
            op(
                "Unpack",
                vec![Some(w), Some(count)],
                D::Uint8,
                &[2, 2, 4],
                &[("bits", Attribute::Int(4))],
            )
        }
        Spelling::Bitwise | Spelling::BitwiseWithZero => op(
            "Cast",
            vec![expanded_weights],
            D::Uint4,
            &[2, 2, 4],
            &[("to", Attribute::Int(i64::from(D::Uint4.to_onnx())))],
        ),
    };
    let weights = if let Some(s) = s {
        let zero = if let Some(expanded_zero) = expanded_zero {
            let mut constant = |v| {
                op(
                    "Constant",
                    vec![],
                    D::Int64,
                    &[1],
                    &[("value_ints", Attribute::Ints(vec![v]))],
                )
            };
            let start = constant(0);
            let end = constant(1);
            let axis = constant(2);
            let sliced = op(
                "Slice",
                vec![Some(expanded_zero), Some(start), Some(end), Some(axis)],
                D::Uint8,
                &[2, 2, 1],
                &[],
            );
            op(
                "Cast",
                vec![Some(sliced)],
                D::Uint4,
                &[2, 2, 1],
                &[("to", Attribute::Int(i64::from(D::Uint4.to_onnx())))],
            )
        } else if let Some(z) = z {
            let count = op(
                "Constant",
                vec![],
                D::Int64,
                &[],
                &[("value_int", Attribute::Int(1))],
            );
            op(
                "Unpack",
                vec![Some(z), Some(count)],
                D::Uint8,
                &[2, 2, 1],
                &[("bits", Attribute::Int(4))],
            )
        } else {
            let shape = op("Shape", vec![Some(s)], D::Int64, &[3], &[]);
            let fill = op(
                "ConstantOfShape",
                vec![Some(shape)],
                D::Uint8,
                &[2, 2, 1],
                &[(
                    "value",
                    Attribute::Tensor(TensorData::from_raw(D::Uint8, vec![1], vec![8])),
                )],
            );
            op(
                "Cast",
                vec![Some(fill)],
                D::Uint4,
                &[2, 2, 1],
                &[("to", Attribute::Int(i64::from(D::Uint4.to_onnx())))],
            )
        };
        op(
            "DequantizeLinear",
            vec![Some(weights), Some(s), Some(zero)],
            D::Float32,
            &[2, 2, 4],
            &[
                ("axis", Attribute::Int(2)),
                ("block_size", Attribute::Int(4)),
            ],
        )
    } else {
        weights
    };
    let y = g.create_named_value("Y", D::Float32, static_shape([2, 2, 2]));
    let mut root = Node::new(
        NodeId(0),
        "GroupedMatMul",
        vec![Some(a), Some(weights), Some(idx)],
        vec![y],
    );
    root.attributes.insert("transB".into(), Attribute::Int(1));
    g.insert_node(root);
    g.add_output(y);
    g
}

fn verify(build: impl Fn(Graph) -> InferenceSession) {
    let a = Tensor::from_f32(&[2, 4], &[1., -2., 3., -4., -1., 2., -3., 4.]).unwrap();
    let idx = Tensor::from_i64(&[2, 2], &[1, 0, 0, 1]).unwrap();
    let codes = [0x10, 0x32, 0xf4, 0x56, 0x78, 0x90, 0xab, 0xcd];
    let scales = [0.125, 0.0625, 0.25, 0.5];
    let zeros = [0x93, 0x47, 0x60, 0xf2];
    let packed = Tensor::from_raw(D::Uint8, vec![2, 2, 2], &codes).unwrap();
    let z = Tensor::from_raw(D::Uint8, vec![2, 2, 1], &zeros).unwrap();
    let s = Tensor::from_f32(&[2, 2, 1], &scales).unwrap();
    let mut oracle = build(graph(Spelling::Dense));
    for spelling in [
        Spelling::Unpack,
        Spelling::Bitwise,
        Spelling::BitwiseWithZero,
    ] {
        let use_z = matches!(spelling, Spelling::Unpack | Spelling::BitwiseWithZero);
        let decoded: Vec<_> = codes
            .iter()
            .enumerate()
            .flat_map(|(i, &byte)| {
                let zero = if use_z { zeros[i / 2] & 15 } else { 8 };
                [byte & 15, byte >> 4]
                    .map(|q| f32::from(i16::from(q) - i16::from(zero)) * scales[i / 2])
            })
            .collect();
        let dense = Tensor::from_f32(&[2, 2, 4], &decoded).unwrap();
        let expected = oracle
            .run(&[("A", &a), ("I", &idx), ("W", &dense)])
            .unwrap()[0]
            .to_vec_f32();
        let mut fused = build(graph(spelling));
        assert!(
            fused
                .graph()
                .nodes
                .values()
                .any(|n| n.op_type == "AffineQuantizedGroupedMatMul")
        );
        assert!(!fused.graph().nodes.values().any(|n| {
            matches!(
                n.op_type.as_str(),
                "GroupedMatMul" | "Unpack" | "DequantizeLinear" | "BitwiseAnd" | "Cast"
            )
        }));
        let mut inputs = vec![("A", &a), ("I", &idx), ("W", &packed), ("S", &s)];
        if use_z {
            inputs.push(("Z", &z));
        }
        let got = fused.run(&inputs).unwrap();
        assert_eq!(got[0].to_vec_f32(), expected);
        let only_first = Tensor::from_i64(&[2, 2], &[0; 4]).unwrap();
        let bad_scale = Tensor::from_f32(&[2, 2, 1], &[0.125, 0.0625, f32::NAN, 0.5]).unwrap();
        inputs[1].1 = &only_first;
        inputs[3].1 = &bad_scale;
        assert!(
            fused
                .run(&inputs)
                .unwrap_err()
                .to_string()
                .contains("scales must be positive and finite")
        );
        inputs[3].1 = &s;
        let bad = Tensor::from_i64(&[2, 2], &[1, 0, -1, 1]).unwrap();
        inputs[1].1 = &bad;
        assert!(
            fused
                .run(&inputs)
                .unwrap_err()
                .to_string()
                .contains("group_indices")
        );
    }
}

#[test]
fn matcher_keeps_unfused_graph_on_unsupported_version_or_shared_output() {
    for shared in [false, true] {
        let mut g = graph(Spelling::Bitwise);
        let op = if shared {
            "DequantizeLinear"
        } else {
            "Unsqueeze"
        };
        let n = g.nodes.values().find(|n| n.op_type == op).unwrap();
        if shared {
            g.add_output(n.outputs[0].unwrap());
        } else {
            let id = n.id;
            g.node_mut(id).version = Some(11);
        }
        QuantizedGroupedFusion
            .run(&mut g, &PassContext::new())
            .unwrap();
        assert!(g.nodes.values().any(|n| n.op_type == "GroupedMatMul"));
        assert!(g.nodes.values().any(|n| n.op_type == "DequantizeLinear"));
    }
}

#[test]
fn affine_explanation_distinguishes_topology_eligibility_and_escape() {
    let ctx = PassContext::new();
    for (case, expected) in [
        ("domain", "structural nonmatch"),
        ("packed input", "structural nonmatch"),
        ("shape", "eligibility rejection"),
        ("outside consumer", "has outside consumer"),
        ("graph output", "is a graph output"),
        ("Cast", "is a graph output"),
        ("Slice", "is a graph output"),
        ("ConstantOfShape", "is a graph output"),
        ("capture", "is lexically captured"),
    ] {
        let mut g = graph(if case == "Slice" {
            Spelling::BitwiseWithZero
        } else {
            Spelling::Bitwise
        });
        let find = |op| g.nodes.values().find(|n| n.op_type == op).unwrap().id;
        let anchor = find("GroupedMatMul");
        let low = find("BitwiseAnd");
        let high = find("BitShift");
        let dq = find("DequantizeLinear");
        let wrapper = match case {
            "Cast" | "Slice" | "ConstantOfShape" => g.node(find(case)).outputs[0],
            _ => None,
        };
        let low_output = g.node(low).outputs[0].unwrap();
        g.value_mut(low_output).name = Some("low_nibbles".into());
        match case {
            "domain" => g.node_mut(low).domain = "custom".into(),
            "packed input" => {
                let other = g.create_value(D::Uint8, static_shape([2, 2, 2]));
                g.add_input(other);
                g.replace_input(high, 0, Some(other));
            }
            "shape" => {
                let wrong_scale = g.create_value(D::Float32, static_shape([2, 2, 2]));
                g.add_input(wrong_scale);
                g.replace_input(dq, 1, Some(wrong_scale));
            }
            "outside consumer" => {
                let copy = g.create_value(D::Uint8, static_shape([2, 2, 2]));
                g.insert_node(Node::new(
                    NodeId(0),
                    "Identity",
                    vec![Some(low_output)],
                    vec![copy],
                ));
                g.add_output(copy);
            }
            "graph output" => g.add_output(low_output),
            "Cast" | "Slice" | "ConstantOfShape" => {
                let wrapper = wrapper.unwrap();
                g.value_mut(wrapper).name = Some("wrapper_output".into());
                g.add_output(wrapper);
            }
            "capture" => {
                let cond = g.create_value(D::Bool, vec![]);
                g.add_input(cond);
                let out = g.create_value(D::Uint8, static_shape([2, 2, 2]));
                let owner = g.insert_node(Node::new(NodeId(0), "If", vec![Some(cond)], vec![out]));
                g.add_output(out);
                for branch in ["then_branch", "else_branch"] {
                    let mut body = Graph::new();
                    let capture =
                        body.create_named_value("low_nibbles", D::Uint8, static_shape([2, 2, 2]));
                    body.add_output(capture);
                    g.subgraphs
                        .insert((owner, GraphAttribute::single(branch)), body);
                }
            }
            _ => unreachable!(),
        }
        let message = explain_at(&g, &ctx, anchor).unwrap();
        assert!(message.contains(expected), "{case}: {message}");
        if case == "shape" {
            assert!(
                message.contains("row-wise blocked affine weights"),
                "{message}"
            );
        }
        if matches!(case, "outside consumer" | "graph output" | "capture") {
            assert!(message.contains("low_nibbles"), "{message}");
        }
        if wrapper.is_some() {
            assert!(message.contains("wrapper_output"), "{message}");
        }
        let order = g.topological_order().unwrap();
        QuantizedGroupedFusion.run(&mut g, &ctx).unwrap();
        assert_eq!(g.node(anchor).op_type, "GroupedMatMul", "{case}");
        assert_eq!(g.topological_order().unwrap(), order, "{case}");
    }
}

#[test]
fn affine_rewrite_erases_an_internally_shared_decode_but_keeps_projection_output() {
    let mut g = graph(Spelling::Unpack);
    let dq = g
        .nodes
        .values()
        .find(|n| n.op_type == "DequantizeLinear")
        .unwrap()
        .id;
    let codes = g.node(dq).inputs[0].unwrap();
    let decode = g.value(codes).producer.unwrap();
    let bank = g.node(dq).outputs[0].unwrap();
    let scale = g.create_value(D::Float32, static_shape([2, 2, 4]));
    g.add_input(scale);
    g.replace_input(dq, 1, Some(scale));
    g.replace_input(dq, 2, Some(codes));
    g.node_mut(dq)
        .attributes
        .insert("block_size".into(), Attribute::Int(1));
    let output = g.outputs[0];
    let anchor = g.value(output).producer.unwrap();
    let ctx = PassContext::new();
    assert!(explain_at(&g, &ctx, anchor).unwrap().contains("eligible"));
    QuantizedGroupedFusion.run(&mut g, &ctx).unwrap();
    assert_eq!(g.node(anchor).op_type, quantized_grouped::OP);
    assert_eq!(g.node(anchor).inputs[1], g.node(anchor).inputs[4]);
    assert_eq!(g.outputs, [output]);
    assert_eq!(g.value(output).producer, Some(anchor));
    assert!(g.try_node(dq).is_none());
    assert!(g.try_node(decode).is_none());
    assert!(g.try_value(bank).is_none());
    assert!(g.try_value(codes).is_none());
    g.validate().unwrap();
}

#[test]
fn affine_rewrite_preserves_order_and_outputs_for_independent_projections() {
    let mut g = graph(Spelling::Unpack);
    let first_output = g.outputs[0];
    let first = g.value(first_output).producer.unwrap();
    let bank = g.node(first).inputs[1].unwrap();
    let mut dq = g.node(g.value(bank).producer.unwrap()).clone();
    let codes = dq.inputs[0].unwrap();
    let mut unpack = g.node(g.value(codes).producer.unwrap()).clone();
    let packed = g.create_value(D::Uint8, static_shape([2, 2, 2]));
    let decoded = g.create_value(D::Uint8, static_shape([2, 2, 4]));
    g.add_input(packed);
    unpack.inputs[0] = Some(packed);
    unpack.outputs[0] = Some(decoded);
    let unpack = g.insert_node(unpack);
    let new_bank = g.create_value(D::Float32, static_shape([2, 2, 4]));
    dq.inputs[0] = Some(decoded);
    dq.inputs.truncate(2);
    dq.outputs[0] = Some(new_bank);
    let dq = g.insert_node(dq);
    let output = g.create_value(D::Float32, static_shape([2, 2, 2]));
    let mut second = g.node(first).clone();
    second.inputs[1] = Some(new_bank);
    second.outputs[0] = Some(output);
    let second = g.insert_node(second);
    g.add_output(output);

    QuantizedGroupedFusion
        .run(&mut g, &PassContext::new())
        .unwrap();

    assert_eq!(g.outputs, [first_output, output]);
    let fused: Vec<_> = g
        .topological_order()
        .unwrap()
        .into_iter()
        .filter(|&id| g.node(id).op_type == quantized_grouped::OP)
        .collect();
    assert_eq!(fused, [first, second]);
    assert_eq!(g.node(second).inputs[1], Some(packed));
    assert!(g.try_node(unpack).is_none());
    assert!(g.try_node(dq).is_none());
    g.validate().unwrap();
}

#[test]
fn packed_projection_cpu_spelling_and_parity() {
    verify(|g| InferenceSession::from_graph(g).unwrap());
}

#[cfg(feature = "gpu-tests")]
#[test]
fn packed_projection_cuda_spelling_and_parity() {
    use onnx_runtime_ep_cuda::CudaExecutionProvider;
    use std::{path::Path, sync::Arc};

    verify(|graph| {
        let session = InferenceSession::from_graph_with_provider(
            graph,
            Arc::new(onnx_runtime_loader::WeightStore::new()),
            Path::new("."),
            Arc::new(CudaExecutionProvider::initialized(0).expect("initialize CUDA")),
            None,
        )
        .unwrap();
        assert!(session.execution_provider_fallback_report().is_none());
        session
    });
}
