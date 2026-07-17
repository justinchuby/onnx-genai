//! GPU numeric parity tests for cuBLASLt bias/activation GEMM epilogues.

use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, ExecutionProvider, KernelMatch, TensorMut, TensorView,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, TensorLayout, as_static_shape,
    compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;

fn build_model(
    op_type: &str,
    a_shape: &[usize],
    b_shape: &[usize],
    bias_shape: &[usize],
    out_shape: &[usize],
    attributes: &[(&str, Attribute)],
) -> (Graph, NodeId) {
    let mut graph = Graph::new();
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let a = graph.create_named_value(
        "A",
        DataType::Float32,
        static_shape(a_shape.iter().copied()),
    );
    let b = graph.create_named_value(
        "B",
        DataType::Float32,
        static_shape(b_shape.iter().copied()),
    );
    let bias = graph.create_named_value(
        "bias",
        DataType::Float32,
        static_shape(bias_shape.iter().copied()),
    );
    for value in [a, b, bias] {
        graph.add_input(value);
    }
    let y = graph.create_named_value(
        "Y",
        DataType::Float32,
        static_shape(out_shape.iter().copied()),
    );
    let mut node = Node::new(
        NodeId(0),
        op_type,
        vec![Some(a), Some(b), Some(bias)],
        vec![y],
    );
    node.domain = "com.microsoft".into();
    for (name, value) in attributes {
        node.attributes.insert((*name).into(), value.clone());
    }
    let node_id = graph.insert_node(node);
    graph.add_output(y);
    (graph, node_id)
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: f32 is plain data and the byte slice retains the input lifetime.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn run_model(
    ep: &CudaExecutionProvider,
    model: &Model<'_>,
    node_id: NodeId,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let graph = model.graph;
    let node = graph.node(node_id);
    let ids = [
        node.inputs[0].unwrap(),
        node.inputs[1].unwrap(),
        node.inputs[2].unwrap(),
    ];
    let shapes: Vec<Vec<usize>> = ids
        .iter()
        .map(|&id| as_static_shape(&graph.value(id).shape).unwrap())
        .collect();
    let out_shape = as_static_shape(&graph.value(node.outputs[0]).shape).unwrap();
    let host_inputs = [a, b, bias];
    let mut buffers = Vec::new();
    for values in host_inputs {
        let buffer = ep.allocate(std::mem::size_of_val(values), 256).unwrap();
        // SAFETY: the allocation is exactly the size of the copied slice.
        unsafe {
            ep.runtime()
                .htod(f32_bytes(values), cuptr(buffer.as_ptr()))
                .unwrap();
        }
        buffers.push(buffer);
    }
    let mut out_buffer = ep
        .allocate(out_shape.iter().product::<usize>() * 4, 256)
        .unwrap();
    let strides: Vec<Vec<i64>> = shapes
        .iter()
        .map(|shape| compute_contiguous_strides(shape))
        .collect();
    let out_strides = compute_contiguous_strides(&out_shape);
    let device = ep.device_id();
    let inputs = [
        TensorView::new(
            DevicePtr(buffers[0].as_ptr()),
            DataType::Float32,
            &shapes[0],
            &strides[0],
            device,
        ),
        TensorView::new(
            DevicePtr(buffers[1].as_ptr()),
            DataType::Float32,
            &shapes[1],
            &strides[1],
            device,
        ),
        TensorView::new(
            DevicePtr(buffers[2].as_ptr()),
            DataType::Float32,
            &shapes[2],
            &strides[2],
            device,
        ),
    ];
    let output = TensorMut::new(
        DevicePtrMut(out_buffer.as_mut_ptr()),
        DataType::Float32,
        &out_shape,
        &out_strides,
        device,
    );
    ep.get_kernel(node, &shapes, 1)
        .unwrap()
        .execute(&inputs, &mut [output])
        .unwrap();
    let mut bytes = vec![0; out_shape.iter().product::<usize>() * 4];
    // SAFETY: the destination exactly matches the output allocation.
    unsafe {
        ep.runtime()
            .dtoh(&mut bytes, cuptr(out_buffer.as_ptr()))
            .unwrap();
    }
    for buffer in buffers {
        ep.deallocate(buffer).unwrap();
    }
    ep.deallocate(out_buffer).unwrap();
    bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn reference_gemm(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    a_shape: [usize; 2],
    b_shape: [usize; 2],
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
) -> Vec<f32> {
    let (m, k) = if trans_a {
        (a_shape[1], a_shape[0])
    } else {
        (a_shape[0], a_shape[1])
    };
    let n = if trans_b { b_shape[0] } else { b_shape[1] };
    let mut out = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0;
            for inner in 0..k {
                let ai = if trans_a {
                    inner * a_shape[1] + row
                } else {
                    row * a_shape[1] + inner
                };
                let bi = if trans_b {
                    col * b_shape[1] + inner
                } else {
                    inner * b_shape[1] + col
                };
                sum += a[ai] * b[bi];
            }
            out[row * n + col] = alpha * sum + bias[col];
        }
    }
    out
}

fn max_abs(got: &[f32], expected: &[f32]) -> f32 {
    got.iter()
        .zip(expected)
        .map(|(got, expected)| (got - expected).abs())
        .fold(0.0, f32::max)
}

fn assert_close(label: &str, got: &[f32], expected: &[f32], tolerance: f32) {
    let error = max_abs(got, expected);
    println!("{label} max_abs_error={error:.9e}");
    assert!(error <= tolerance, "{label}: {got:?} vs {expected:?}");
}

fn cuda_ep() -> Option<CudaExecutionProvider> {
    match std::panic::catch_unwind(CudaExecutionProvider::new_default) {
        Ok(Ok(ep)) => Some(ep),
        Ok(Err(error)) => {
            eprintln!("skip: no CUDA GPU/runtime available ({error})");
            None
        }
        Err(_) => {
            eprintln!("skip: CUDA runtime library loading panicked");
            None
        }
    }
}

#[test]
fn fused_matmul_bias_matches_matmul_then_bias() {
    let Some(ep) = cuda_ep() else {
        return;
    };
    let a = [0.5, -1.0, 2.0, 1.5, 0.25, -0.75];
    let b = [1.0, 0.5, -2.0, -1.0, 3.0, 0.25, 2.0, -0.5, 1.25];
    let bias = [0.25, -1.5, 2.0];
    let (graph, node) = build_model("FusedMatMulBias", &[2, 3], &[3, 3], &[3], &[2, 3], &[]);
    let model = Model::new(&graph);
    let got = run_model(&ep, &model, node, &a, &b, &bias);
    let expected = reference_gemm(&a, &b, &bias, [2, 3], [3, 3], false, false, 1.0);
    assert_close("FusedMatMulBias", &got, &expected, 1e-5);
}

#[test]
fn fused_gemm_relu_bias_matches_reference_with_transpose_and_alpha() {
    let Some(ep) = cuda_ep() else {
        return;
    };
    let a = [0.5, -1.0, 2.0, 1.5, -0.25, 0.75];
    let b = [
        1.0, -2.0, 0.5, -1.0, 0.25, 2.0, 0.75, -0.5, 1.5, -2.0, 1.0, 0.25,
    ];
    let bias = [-0.5, 0.25, -1.0, 1.5];
    let attrs = [
        ("activation", Attribute::String(b"Relu".to_vec())),
        ("alpha", Attribute::Float(0.75)),
        ("transA", Attribute::Int(1)),
        ("transB", Attribute::Int(1)),
    ];
    let (graph, node) = build_model("FusedGemm", &[3, 2], &[4, 3], &[4], &[2, 4], &attrs);
    let model = Model::new(&graph);
    let got = run_model(&ep, &model, node, &a, &b, &bias);
    let mut expected = reference_gemm(&a, &b, &bias, [3, 2], [4, 3], true, true, 0.75);
    expected
        .iter_mut()
        .for_each(|value| *value = value.max(0.0));
    assert_close("FusedGemm RELU_BIAS", &got, &expected, 1e-5);
}

#[test]
fn fused_gemm_gelu_bias_matches_tanh_reference() {
    let Some(ep) = cuda_ep() else {
        return;
    };
    let a = [0.5, -1.0, 2.0, -0.75, 1.25, 0.25];
    let b = [1.0, -0.5, 0.25, -1.5, 0.75, 2.0];
    let bias = [-0.25, 0.5];
    let attrs = [("activation", Attribute::String(b"Gelu".to_vec()))];
    let (graph, node) = build_model("FusedGemm", &[2, 3], &[3, 2], &[2], &[2, 2], &attrs);
    let model = Model::new(&graph);
    let got = run_model(&ep, &model, node, &a, &b, &bias);
    let mut expected = reference_gemm(&a, &b, &bias, [2, 3], [3, 2], false, false, 1.0);
    expected.iter_mut().for_each(|value| {
        let x = *value as f64;
        *value = (0.5
            * x
            * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh()))
            as f32;
    });
    assert_close("FusedGemm GELU_BIAS (tanh)", &got, &expected, 2e-6);
}

#[test]
fn placement_declines_broadcast_bias_and_batched_matmul() {
    let Some(ep) = cuda_ep() else {
        return;
    };

    for op_type in ["FusedMatMulBias", "FusedGemm"] {
        for (a_shape, b_shape, bias_shape, out_shape, expected) in [
            (&[2, 3][..], &[3, 4][..], &[4][..], &[2, 4][..], true),
            (&[2, 3][..], &[3, 4][..], &[][..], &[2, 4][..], false),
            (&[2, 3][..], &[3, 4][..], &[1, 4][..], &[2, 4][..], false),
            (&[2, 3][..], &[3, 4][..], &[2, 4][..], &[2, 4][..], false),
            (&[2, 2, 3][..], &[3, 4][..], &[4][..], &[2, 2, 4][..], false),
        ] {
            let (graph, node_id) =
                build_model(op_type, a_shape, b_shape, bias_shape, out_shape, &[]);
            let node = graph.node(node_id);
            let shapes: Vec<_> = node
                .input_values()
                .map(|value| graph.value(value).shape.clone())
                .collect();
            let layouts = vec![TensorLayout::contiguous(); shapes.len()];
            assert_eq!(
                matches!(
                    ep.supports_op(node, 1, &shapes, &[], &layouts),
                    KernelMatch::Supported { .. }
                ),
                expected,
                "unexpected {op_type} placement for A={a_shape:?}, B={b_shape:?}, bias={bias_shape:?}"
            );
        }
    }
}
