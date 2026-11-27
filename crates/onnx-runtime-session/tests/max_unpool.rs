use onnx_runtime_ir::{Attribute, DataType, Dim, Graph, Node, NodeId, SymbolId, static_shape};
use onnx_runtime_session::{InferenceSession, Tensor};

fn graph(opset: u64, dtype: DataType, explicit: bool) -> Graph {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), opset);
    let x = g.create_named_value("X", dtype, static_shape([1, 1, 2, 2]));
    let i = g.create_named_value("I", DataType::Int64, static_shape([1, 1, 2, 2]));
    let mut inputs = vec![Some(x), Some(i)];
    for v in [x, i] {
        g.add_input(v);
    }
    if explicit {
        let s = g.create_named_value("output_shape", DataType::Int64, static_shape([4]));
        g.add_input(s);
        inputs.push(Some(s));
    } else {
        inputs.push(None);
    }
    let y_shape = if explicit {
        (0..4).map(|id| Dim::Symbolic(SymbolId(id))).collect()
    } else {
        static_shape([1, 1, 4, 4])
    };
    let y = g.create_named_value("Y", dtype, y_shape);
    let mut node = Node::new(NodeId(0), "MaxUnpool", inputs, vec![y]);
    node.attributes
        .insert("kernel_shape".into(), Attribute::Ints(vec![2, 2]));
    node.attributes
        .insert("strides".into(), Attribute::Ints(vec![2, 2]));
    if explicit {
        node.attributes
            .insert("pads".into(), Attribute::Ints(vec![1, 0, 0, 1]));
    }
    g.insert_node(node);
    g.add_output(y);
    g
}

#[test]
fn max_unpool_cpu_pinned_default_shape_and_opsets() {
    // ONNX 622adf94d4e720142950aa8178f7357d52e63db0:
    // onnx/backend/test/case/node/maxunpool.py, export_without_output_shape.
    let x = Tensor::from_f32(&[1, 1, 2, 2], &[1., 2., 3., 4.]).unwrap();
    let i = Tensor::from_i64(&[1, 1, 2, 2], &[5, 7, 13, 15]).unwrap();
    for opset in [9, 11, 22] {
        let mut session =
            InferenceSession::from_graph(graph(opset, DataType::Float32, false)).unwrap();
        let actual = session.run(&[("X", &x), ("I", &i)]).unwrap();
        assert_eq!(actual[0].shape, [1, 1, 4, 4]);
        assert_eq!(
            actual[0].to_vec_f32(),
            [
                0., 0., 0., 0., 0., 1., 0., 2., 0., 0., 0., 0., 0., 3., 0., 4.,
            ]
        );
    }
}

#[test]
fn max_unpool_cpu_live_output_shape_and_invalid_indices() {
    let mut session = InferenceSession::from_graph(graph(22, DataType::Float32, true)).unwrap();
    check_live_output_shape_and_invalid_indices(&mut session);
}

fn check_live_output_shape_and_invalid_indices(session: &mut InferenceSession) {
    let x = Tensor::from_f32(&[1, 1, 2, 2], &[5., 6., 7., 8.]).unwrap();
    let i = Tensor::from_i64(&[1, 1, 2, 2], &[6, 8, 16, 18]).unwrap();
    // From ONNX export_with_output_shape, then change all four dimensions in
    // the same session: the flat positions are relative to the full output.
    for dims in [[1, 1, 5, 5], [2, 2, 16, 16], [2, 2, 2, 3]] {
        let s = Tensor::from_i64(&[4], &dims).unwrap();
        let actual = session
            .run(&[("X", &x), ("I", &i), ("output_shape", &s)])
            .unwrap();
        assert_eq!(actual[0].shape, dims.map(|d| d as usize));
        let mut expected = vec![0.; dims.iter().product::<i64>() as usize];
        for (index, value) in [(6, 5.), (8, 6.), (16, 7.), (18, 8.)] {
            expected[index] = value;
        }
        assert_eq!(actual[0].to_vec_f32(), expected);
    }
    let s = Tensor::from_i64(&[4], &[1, 1, 5, 5]).unwrap();
    for invalid in [-1, 25] {
        let i = Tensor::from_i64(&[1, 1, 2, 2], &[6, invalid, 16, 18]).unwrap();
        let error = session
            .run(&[("X", &x), ("I", &i), ("output_shape", &s)])
            .unwrap_err();
        assert!(
            error.to_string().contains(&format!("I[1]={invalid}")),
            "{error}"
        );
    }
    let bad_shape = Tensor::from_i64(&[4], &[1, 1, 5, -1]).unwrap();
    let error = session
        .run(&[("X", &x), ("I", &i), ("output_shape", &bad_shape)])
        .unwrap_err();
    assert!(error.to_string().contains("output_shape[3]=-1"), "{error}");
}

#[cfg(feature = "gpu-tests")]
#[test]
fn max_unpool_cuda_session_live_output_shape_without_fallback() {
    use std::{path::Path, sync::Arc};

    use onnx_runtime_ep_cuda::CudaExecutionProvider;
    use onnx_runtime_loader::WeightStore;

    let provider = Arc::new(CudaExecutionProvider::initialized(0).expect("initialize CUDA"));
    let mut session = InferenceSession::from_graph_with_provider(
        graph(22, DataType::Float32, true),
        Arc::new(WeightStore::new()),
        Path::new("."),
        provider,
        None,
    )
    .expect("MaxUnpool must be placed on CUDA with no CPU fallback");
    check_live_output_shape_and_invalid_indices(&mut session);
}

#[test]
fn max_unpool_cpu_preserves_f64_bits_and_bfloat16_schema_gate() {
    let shape = [1, 1, 2, 2];
    let i = Tensor::from_i64(&shape, &[2, 5, 5, 15]).unwrap();
    let precise = 0x3ff0000000000001u64;
    let payload: Vec<u8> = [0x8000000000000000u64, 1, precise, 0x7ff8000000000004]
        .into_iter()
        .flat_map(u64::to_ne_bytes)
        .collect();
    let x = Tensor::from_raw(DataType::Float64, shape.to_vec(), &payload).unwrap();
    let mut session = InferenceSession::from_graph(graph(11, DataType::Float64, false)).unwrap();
    let output = session.run(&[("X", &x), ("I", &i)]).unwrap();
    let mut expected = [0u64; 16];
    expected[2] = 0x8000000000000000;
    expected[5] = precise; // last source is the deterministic reference behavior
    expected[15] = 0x7ff8000000000004;
    assert_eq!(
        output[0].as_bytes(),
        expected
            .into_iter()
            .flat_map(u64::to_ne_bytes)
            .collect::<Vec<_>>(),
    );

    let dtype = DataType::BFloat16;
    let rejected = InferenceSession::from_graph(graph(21, dtype, false));
    assert!(matches!(rejected, Err(ref error) if error.to_string().contains("BFloat16")));
    let mut session = InferenceSession::from_graph(graph(22, dtype, false)).unwrap();
    let payload: Vec<u8> = [0x3f80u16, 0x4000, 0x4040, 0x4080]
        .into_iter()
        .flat_map(u16::to_ne_bytes)
        .collect();
    let x = Tensor::from_raw(dtype, shape.to_vec(), &payload).unwrap();
    let output = session.run(&[("X", &x), ("I", &i)]).unwrap();
    let mut expected = [0u16; 16];
    expected[2] = 0x3f80;
    expected[5] = 0x4040;
    expected[15] = 0x4080;
    assert_eq!(
        output[0].as_bytes(),
        expected
            .into_iter()
            .flat_map(u16::to_ne_bytes)
            .collect::<Vec<_>>(),
    );
}

#[test]
fn max_unpool_cpu_max_pool_indices_are_global_row_major() {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 22);
    let x = g.create_named_value("X", DataType::Float32, static_shape([2, 2, 4]));
    let pooled = g.create_named_value("pooled", DataType::Float32, static_shape([2, 2, 2]));
    let indices = g.create_named_value("I", DataType::Int64, static_shape([2, 2, 2]));
    let y = g.create_named_value("Y", DataType::Float32, static_shape([2, 2, 4]));
    g.add_input(x);
    for (name, inputs, outputs) in [
        ("MaxPool", vec![Some(x)], vec![pooled, indices]),
        ("MaxUnpool", vec![Some(pooled), Some(indices)], vec![y]),
    ] {
        let mut node = Node::new(NodeId(0), name, inputs, outputs);
        node.attributes
            .insert("kernel_shape".into(), Attribute::Ints(vec![2]));
        node.attributes
            .insert("strides".into(), Attribute::Ints(vec![2]));
        g.insert_node(node);
    }
    g.add_output(y);
    let mut session = InferenceSession::from_graph(g).unwrap();
    let x = Tensor::from_f32(
        &[2, 2, 4],
        &[
            1., 2., 3., 4., 8., 7., 6., 5., 9., 10., 11., 12., 16., 15., 14., 13.,
        ],
    )
    .unwrap();
    let result = session.run(&[("X", &x)]).unwrap();
    assert_eq!(
        result[0].to_vec_f32(),
        [
            0., 2., 0., 4., 8., 0., 6., 0., 0., 10., 0., 12., 16., 0., 14., 0.
        ]
    );
}
