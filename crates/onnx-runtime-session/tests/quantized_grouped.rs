use onnx_runtime_ir::{
    Attribute, DataType as D, Graph, Node, NodeId, TensorData, quantized_grouped, static_shape,
};
use onnx_runtime_session::{InferenceSession, Tensor};

enum Spelling {
    Unpack,
    Bitwise,
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
    let z = matches!(spelling, Spelling::Unpack).then(|| input("Z", D::Uint8, &[2, 2, 1]));
    let mut op = |name, inputs, dtype, shape: &[usize], attrs: &[(&str, Attribute)]| {
        let v = g.create_value(dtype, static_shape(shape.iter().copied()));
        let mut n = Node::new(NodeId(0), name, inputs, vec![v]);
        n.attributes
            .extend(attrs.iter().map(|(k, a)| (k.to_string(), a.clone())));
        g.insert_node(n);
        v
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
        Spelling::Bitwise => {
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
                &[("value_ints", Attribute::Ints(vec![2, 2, 4]))],
            );
            let low = op(
                "BitwiseAnd",
                vec![Some(w), Some(mask)],
                D::Uint8,
                &[2, 2, 2],
                &[],
            );
            let high = op(
                "BitShift",
                vec![Some(w), Some(shift)],
                D::Uint8,
                &[2, 2, 2],
                &[("direction", Attribute::String("RIGHT".into()))],
            );
            let low = op(
                "Unsqueeze",
                vec![Some(low), Some(axes)],
                D::Uint8,
                &[2, 2, 2, 1],
                &[],
            );
            let high = op(
                "Unsqueeze",
                vec![Some(high), Some(axes)],
                D::Uint8,
                &[2, 2, 2, 1],
                &[],
            );
            let pairs = op(
                "Concat",
                vec![Some(low), Some(high)],
                D::Uint8,
                &[2, 2, 2, 2],
                &[("axis", Attribute::Int(-1))],
            );
            let bytes = op(
                "Reshape",
                vec![Some(pairs), Some(shape)],
                D::Uint8,
                &[2, 2, 4],
                &[],
            );
            op(
                "Cast",
                vec![Some(bytes)],
                D::Uint4,
                &[2, 2, 4],
                &[("to", Attribute::Int(i64::from(D::Uint4.to_onnx())))],
            )
        }
    };
    let weights = if let Some(s) = s {
        let zero = if let Some(z) = z {
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
    for spelling in [Spelling::Unpack, Spelling::Bitwise] {
        let use_z = matches!(spelling, Spelling::Unpack);
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
                .any(|n| n.op_type == quantized_grouped::OP)
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
