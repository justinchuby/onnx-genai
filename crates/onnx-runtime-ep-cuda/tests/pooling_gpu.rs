//! GPU parity tests for the cuDNN-backed ONNX pooling kernels.

use half::f16;
use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, as_static_shape, compute_contiguous_strides,
    static_shape,
};
use onnx_runtime_loader::Model;

fn tensor_bytes(dtype: DataType, values: &[f32]) -> Vec<u8> {
    match dtype {
        DataType::Float32 => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        DataType::Float16 => values
            .iter()
            .flat_map(|&value| f16::from_f32(value).to_bits().to_le_bytes())
            .collect(),
        other => panic!("unsupported test dtype {other:?}"),
    }
}

fn build_pool_model(
    op_type: &str,
    dtype: DataType,
    input_shape: &[usize],
    output_shape: &[usize],
    kernel_shape: [i64; 2],
    strides: [i64; 2],
    pads: [i64; 4],
    count_include_pad: Option<i64>,
) -> (Graph, NodeId) {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let input = graph.create_named_value("X", dtype, static_shape(input_shape.iter().copied()));
    graph.add_input(input);
    let output = graph.create_named_value("Y", dtype, static_shape(output_shape.iter().copied()));
    let mut pool = Node::new(NodeId(0), op_type, vec![Some(input)], vec![output]);
    pool.attributes.insert(
        "kernel_shape".into(),
        Attribute::Ints(kernel_shape.to_vec()),
    );
    pool.attributes
        .insert("strides".into(), Attribute::Ints(strides.to_vec()));
    pool.attributes
        .insert("pads".into(), Attribute::Ints(pads.to_vec()));
    if let Some(value) = count_include_pad {
        pool.attributes
            .insert("count_include_pad".into(), Attribute::Int(value));
    }
    let node = graph.insert_node(pool);
    graph.add_output(output);
    (graph, node)
}

fn run_model(
    ep: &CudaExecutionProvider,
    model: &Model<'_>,
    node_id: NodeId,
    values: &[f32],
) -> Vec<f32> {
    let graph = model.graph;
    let node = graph.node(node_id);
    let input_id = node.inputs[0].unwrap();
    let output_id = node.outputs[0];
    let input_shape = as_static_shape(&graph.value(input_id).shape).unwrap();
    let output_shape = as_static_shape(&graph.value(output_id).shape).unwrap();
    let dtype = graph.value(input_id).dtype;
    let input_bytes = tensor_bytes(dtype, values);
    let input_buf = ep.allocate(input_bytes.len(), 256).unwrap();
    let mut output_buf = ep
        .allocate(
            output_shape.iter().product::<usize>() * dtype.byte_size(),
            256,
        )
        .unwrap();
    unsafe {
        ep.runtime()
            .htod(&input_bytes, cuptr(input_buf.as_ptr()))
            .unwrap();
    }
    let input_strides = compute_contiguous_strides(&input_shape);
    let output_strides = compute_contiguous_strides(&output_shape);
    let device = ep.device_id();
    let input = TensorView::new(
        DevicePtr(input_buf.as_ptr()),
        dtype,
        &input_shape,
        &input_strides,
        device,
    );
    let output = TensorMut::new(
        DevicePtrMut(output_buf.as_mut_ptr()),
        dtype,
        &output_shape,
        &output_strides,
        device,
    );
    ep.get_kernel(node, &[input_shape.clone()], 17)
        .unwrap()
        .execute(&[input], &mut [output])
        .unwrap();

    let mut bytes = vec![0; output_shape.iter().product::<usize>() * dtype.byte_size()];
    unsafe {
        ep.runtime()
            .dtoh(&mut bytes, cuptr(output_buf.as_ptr()))
            .unwrap();
    }
    ep.deallocate(input_buf).unwrap();
    ep.deallocate(output_buf).unwrap();
    match dtype {
        DataType::Float32 => bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect(),
        DataType::Float16 => bytes
            .chunks_exact(2)
            .map(|bytes| f16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32())
            .collect(),
        other => panic!("unsupported test dtype {other:?}"),
    }
}

fn assert_close(got: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(got.len(), expected.len());
    for (index, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        assert!(
            (got - expected).abs() <= tolerance,
            "index {index}: got {got}, expected {expected}"
        );
    }
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
fn cudnn_maxpool_matches_cpu_for_f32_and_f16() {
    let Some(ep) = cuda_ep() else {
        return;
    };
    let input = [
        1.0, 3.0, 2.0, 0.0, 4.0, 6.0, 5.0, 1.0, 7.0, 8.0, 9.0, 2.0, 3.0, 4.0, 1.0, 0.0,
    ];
    let expected = [6.0, 5.0, 8.0, 9.0];
    for (dtype, tolerance) in [(DataType::Float32, 1e-6), (DataType::Float16, 1e-3)] {
        let (graph, node) = build_pool_model(
            "MaxPool",
            dtype,
            &[1, 1, 4, 4],
            &[1, 1, 2, 2],
            [2, 2],
            [2, 2],
            [0, 0, 0, 0],
            None,
        );
        let model = Model::new(&graph);
        assert_close(&run_model(&ep, &model, node, &input), &expected, tolerance);
    }
    println!("cuDNN MaxPool 2x2 stride-2 f32/f16 cases passed");
}

#[test]
fn cudnn_averagepool_padding_count_modes_match_cpu() {
    let Some(ep) = cuda_ep() else {
        return;
    };
    let input = [1.0, 2.0, 3.0, 4.0];
    for (count_include_pad, expected) in [
        (0, [2.5, 2.5, 2.5, 2.5]),
        (1, [10.0 / 9.0, 10.0 / 9.0, 10.0 / 9.0, 10.0 / 9.0]),
    ] {
        let (graph, node) = build_pool_model(
            "AveragePool",
            DataType::Float32,
            &[1, 1, 2, 2],
            &[1, 1, 2, 2],
            [3, 3],
            [1, 1],
            [1, 1, 1, 1],
            Some(count_include_pad),
        );
        let model = Model::new(&graph);
        assert_close(&run_model(&ep, &model, node, &input), &expected, 1e-5);
    }
    println!("cuDNN AveragePool padded include/exclude-padding cases passed");
}

#[test]
fn cudnn_averagepool_rejects_dilations() {
    let Some(ep) = cuda_ep() else {
        return;
    };
    let (mut graph, node) = build_pool_model(
        "AveragePool",
        DataType::Float32,
        &[1, 1, 4, 4],
        &[1, 1, 2, 2],
        [2, 2],
        [2, 2],
        [0, 0, 0, 0],
        None,
    );
    graph
        .node_mut(node)
        .attributes
        .insert("dilations".into(), Attribute::Ints(vec![2, 2]));
    let model = Model::new(&graph);

    let error = match ep.get_kernel(model.graph.node(node), &[vec![1, 1, 4, 4]], 17) {
        Ok(_) => panic!("AveragePool with dilations must be rejected"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("AveragePool dilations=[2, 2] (cuDNN pooling descriptor has no dilation)")
    );
}
