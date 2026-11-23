use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, TensorData, ValueId, WeightRef, static_shape,
};
use onnx_runtime_session::{InferenceSession, Tensor};

#[derive(Clone, Copy, Debug)]
enum Storage {
    Int4,
    Uint4,
    Unpack,
}

fn initializer(
    graph: &mut Graph,
    name: &str,
    dtype: DataType,
    dims: &[usize],
    bytes: Vec<u8>,
) -> ValueId {
    let id = graph.create_named_value(name, dtype, static_shape(dims.iter().copied()));
    graph.set_initializer(
        id,
        WeightRef::Inline(TensorData::from_raw(dtype, dims.to_vec(), bytes)),
    );
    id
}

fn pack(codes: &[u8]) -> Vec<u8> {
    codes
        .chunks(2)
        .map(|p| p[0] | p.get(1).copied().unwrap_or(0) << 4)
        .collect()
}

fn graph(
    storage: Storage,
    transposed: bool,
    zp: bool,
    expose_dq: bool,
    m: usize,
    k: usize,
    n: usize,
) -> Graph {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 29);
    let block = 16;
    let blocks = k.div_ceil(block);
    let (shape, scales, axis) = if transposed {
        ([n, k], [n, blocks], -1)
    } else {
        ([k, n], [blocks, n], -2)
    };
    let x = g.create_named_value("A", DataType::Float32, static_shape([m, k]));
    g.add_input(x);
    let dtype = match storage {
        Storage::Int4 => DataType::Int4,
        Storage::Uint4 => DataType::Uint4,
        Storage::Unpack => DataType::Uint8,
    };
    let q = |i| ((i * 13 + i / 7) % 16) as u8;
    let weight = if matches!(storage, Storage::Unpack) {
        let packed_shape = if transposed {
            [n, k.div_ceil(2)]
        } else {
            [k.div_ceil(2), n]
        };
        let mut bytes = vec![0; n * k.div_ceil(2)];
        for i in 0..n {
            for d in 0..k {
                let (index, code) = if transposed {
                    (i * k.div_ceil(2) + d / 2, q(i * k + d))
                } else {
                    (d / 2 * n + i, q(d * n + i))
                };
                bytes[index] |= code << (d % 2 * 4);
            }
        }
        let packed = initializer(&mut g, "packed", dtype, &packed_shape, bytes);
        let count = initializer(
            &mut g,
            "count",
            DataType::Int64,
            &[],
            (k as i64).to_le_bytes().to_vec(),
        );
        let codes = g.create_named_value("codes", dtype, static_shape(shape));
        let mut unpack = Node::new(
            NodeId(0),
            "Unpack",
            vec![Some(packed), Some(count)],
            vec![codes],
        );
        unpack
            .attributes
            .insert("axis".into(), Attribute::Int(axis));
        unpack.attributes.insert("bits".into(), Attribute::Int(4));
        g.insert_node(unpack);
        codes
    } else {
        initializer(
            &mut g,
            "weight",
            dtype,
            &shape,
            pack(&(0..n * k).map(q).collect::<Vec<_>>()),
        )
    };
    let scale_count = n * blocks;
    let scale_bytes = (0..scale_count)
        .flat_map(|i| (0.00813f32 + (i % 23) as f32 * 0.00019).to_le_bytes())
        .collect();
    let scale = initializer(&mut g, "scale", DataType::Float32, &scales, scale_bytes);
    let zero_point = zp.then(|| {
        let values = (0..scale_count).map(|i| q(i + 11)).collect::<Vec<_>>();
        let bytes = if matches!(storage, Storage::Unpack) {
            values
        } else {
            pack(&values)
        };
        initializer(&mut g, "zp", dtype, &scales, bytes)
    });
    let decoded = g.create_named_value("decoded", DataType::Float32, static_shape(shape));
    let mut dq = Node::new(
        NodeId(0),
        "DequantizeLinear",
        vec![Some(weight), Some(scale), zero_point],
        vec![decoded],
    );
    dq.attributes.insert("axis".into(), Attribute::Int(axis));
    dq.attributes
        .insert("block_size".into(), Attribute::Int(block as i64));
    g.insert_node(dq);
    let rhs = if transposed {
        let rhs = g.create_named_value("rhs", DataType::Float32, static_shape([k, n]));
        g.insert_node(Node::new(
            NodeId(0),
            "Transpose",
            vec![Some(decoded)],
            vec![rhs],
        ));
        rhs
    } else {
        decoded
    };
    let y = g.create_named_value("Y", DataType::Float32, static_shape([m, n]));
    g.insert_node(Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(x), Some(rhs)],
        vec![y],
    ));
    g.add_output(y);
    if expose_dq {
        g.add_output(decoded);
    }
    g
}

fn assert_selection(session: &InferenceSession, fused: bool) {
    let graph = session.graph();
    assert_eq!(
        graph
            .nodes
            .values()
            .filter(|n| n.op_type == "MatMulNBits")
            .count(),
        usize::from(fused)
    );
    assert_eq!(
        graph
            .nodes
            .values()
            .filter(|n| n.op_type == "DequantizeLinear")
            .count(),
        usize::from(!fused)
    );
}

fn parity(build: impl Fn(Graph) -> InferenceSession, m: usize) {
    let (k, n) = (35, 7); // Odd strides and three blocks exercise both nibble padding rules.
    let a: Vec<_> = (0..m * k)
        .map(|i| ((i * 17 % 31) as f32 - 15.0) / 7.0)
        .collect();
    let a = Tensor::from_f32(&[m, k], &a).unwrap();
    for storage in [Storage::Int4, Storage::Uint4, Storage::Unpack] {
        for transposed in [false, true] {
            for zp in [false, true] {
                let mut reference = build(graph(storage, transposed, zp, true, m, k, n));
                assert_selection(&reference, false);
                let mut fused = build(graph(storage, transposed, zp, false, m, k, n));
                assert_selection(&fused, true);
                let expected = reference.run(&[("A", &a)]).unwrap();
                let actual = fused.run(&[("A", &a)]).unwrap();
                let expected = expected[0].to_vec_f32();
                let actual = actual[0].to_vec_f32();
                assert_eq!(actual.len(), m * n);
                for (idx, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        (a - b).abs() <= 2e-6 * b.abs().max(1.0),
                        "{storage:?} transposed={transposed} zp={zp} output[{idx}]: {a} versus {b}"
                    );
                }
            }
        }
    }
}

#[test]
fn packed_rhs_matches_native_cpu() {
    let build = |g| InferenceSession::from_graph(g).unwrap();
    parity(build, 1);
    parity(build, 3);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device and NVRTC"]
fn packed_rhs_matches_native_cuda() {
    use onnx_runtime_ep_cuda::CudaExecutionProvider;
    use std::{path::Path, sync::Arc};
    let ep = Arc::new(CudaExecutionProvider::initialized(0).expect("initialize CUDA device 0"));
    let build = |g| {
        let session = InferenceSession::from_graph_with_provider(
            g,
            Arc::new(onnx_runtime_loader::WeightStore::new()),
            Path::new("."),
            ep.clone(),
            None,
        )
        .unwrap();
        assert!(
            session.execution_provider_fallback_report().is_none(),
            "{:?}",
            session.execution_provider_fallback_report()
        );
        session
    };
    parity(build, 1);
    let m = 3;
    let mut prefill = build(graph(Storage::Uint4, false, true, false, m, 35, 7));
    assert_selection(&prefill, false);
    assert_eq!(
        prefill
            .run(&[("A", &Tensor::from_f32(&[m, 35], &[1.0; 105]).unwrap())])
            .unwrap()[0]
            .shape(),
        &[m, 7]
    );
}
