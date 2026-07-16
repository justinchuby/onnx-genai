//! GPU numeric parity for standard and contrib `SimplifiedLayerNormalization`.

use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;

fn f32_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: f32 is plain data and the byte slice retains the source lifetime.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn cuda_ep() -> Option<CudaExecutionProvider> {
    match CudaExecutionProvider::new_default() {
        Ok(ep) => Some(ep),
        Err(error) => {
            eprintln!("skip: no CUDA GPU/runtime available ({error})");
            None
        }
    }
}

fn run(
    ep: &CudaExecutionProvider,
    registration: (&str, u64),
    input: &[f32],
    scale: &[f32],
    shape: &[usize],
    axis: i64,
    epsilon: f32,
) -> onnx_runtime_ep_api::Result<(Vec<f32>, Vec<f32>)> {
    let (domain, opset) = registration;
    let mut graph = Graph::new();
    graph.opset_imports.insert(domain.into(), opset);
    let x = graph.create_named_value("x", DataType::Float32, static_shape(shape.iter().copied()));
    let scale_id =
        graph.create_named_value("scale", DataType::Float32, static_shape([scale.len()]));
    graph.add_input(x);
    graph.add_input(scale_id);
    let y = graph.create_named_value("y", DataType::Float32, static_shape(shape.iter().copied()));
    let stats_shape = shape[..axis.rem_euclid(shape.len() as i64) as usize].to_vec();
    let invstd = graph.create_named_value(
        "invstd",
        DataType::Float32,
        static_shape(stats_shape.iter().copied()),
    );
    let mut node = Node::new(
        NodeId(0),
        "SimplifiedLayerNormalization",
        vec![Some(x), Some(scale_id)],
        vec![y, invstd],
    );
    node.domain = domain.into();
    node.attributes.insert("axis".into(), Attribute::Int(axis));
    node.attributes
        .insert("epsilon".into(), Attribute::Float(epsilon));
    let node_id = graph.insert_node(node);
    graph.add_output(y);
    graph.add_output(invstd);
    let model = Model::new(&graph);
    let kernel = ep.get_kernel(model.graph.node(node_id), &[], opset)?;

    let runtime = ep.runtime();
    let device = ep.device_id();
    let input_buffer = ep.allocate(std::mem::size_of_val(input), 256)?;
    let scale_buffer = ep.allocate(std::mem::size_of_val(scale), 256)?;
    let mut output_buffer = ep.allocate(std::mem::size_of_val(input), 256)?;
    let mut invstd_buffer = ep.allocate(stats_shape.iter().product::<usize>() * 4, 256)?;
    unsafe {
        runtime.htod(f32_bytes(input), cuptr(input_buffer.as_ptr()))?;
        runtime.htod(f32_bytes(scale), cuptr(scale_buffer.as_ptr()))?;
    }

    let input_strides = compute_contiguous_strides(shape);
    let scale_shape = [scale.len()];
    let scale_strides = compute_contiguous_strides(&scale_shape);
    let stats_strides = compute_contiguous_strides(&stats_shape);
    let inputs = [
        TensorView::new(
            DevicePtr(input_buffer.as_ptr()),
            DataType::Float32,
            shape,
            &input_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(scale_buffer.as_ptr()),
            DataType::Float32,
            &scale_shape,
            &scale_strides,
            device,
        ),
    ];
    let mut outputs = [
        TensorMut::new(
            DevicePtrMut(output_buffer.as_mut_ptr()),
            DataType::Float32,
            shape,
            &input_strides,
            device,
        ),
        TensorMut::new(
            DevicePtrMut(invstd_buffer.as_mut_ptr()),
            DataType::Float32,
            &stats_shape,
            &stats_strides,
            device,
        ),
    ];
    kernel.execute(&inputs, &mut outputs)?;

    let mut output_bytes = vec![0; std::mem::size_of_val(input)];
    let mut invstd_bytes = vec![0; stats_shape.iter().product::<usize>() * 4];
    unsafe {
        runtime.dtoh(&mut output_bytes, cuptr(output_buffer.as_ptr()))?;
        runtime.dtoh(&mut invstd_bytes, cuptr(invstd_buffer.as_ptr()))?;
    }
    ep.deallocate(input_buffer)?;
    ep.deallocate(scale_buffer)?;
    ep.deallocate(output_buffer)?;
    ep.deallocate(invstd_buffer)?;
    Ok((bytes_to_f32(&output_bytes), bytes_to_f32(&invstd_bytes)))
}

fn reference(input: &[f32], scale: &[f32], norm_size: usize, epsilon: f32) -> (Vec<f32>, Vec<f32>) {
    let mut output = Vec::with_capacity(input.len());
    let mut invstd = Vec::with_capacity(input.len() / norm_size);
    for group in input.chunks_exact(norm_size) {
        let inv = 1.0
            / (group.iter().map(|value| value * value).sum::<f32>() / norm_size as f32 + epsilon)
                .sqrt();
        invstd.push(inv);
        output.extend(
            group
                .iter()
                .zip(scale)
                .map(|(&value, &weight)| value * inv * weight),
        );
    }
    (output, invstd)
}

fn assert_close(label: &str, got: &[f32], expected: &[f32]) {
    let error = got
        .iter()
        .zip(expected)
        .map(|(got, expected)| (got - expected).abs())
        .fold(0.0f32, f32::max);
    println!("{label} max_abs_error={error:.9e}");
    assert!(error <= 1e-5, "{label}: {got:?} vs {expected:?}");
}

#[test]
fn standard_simplified_layer_norm_matches_contrib_and_reference() {
    let Some(ep) = cuda_ep() else { return };
    let shape = [2, 4];
    let input = [1.0, 2.0, 3.0, 4.0, -2.0, 0.0, 2.0, 4.0];
    let scale = [1.0, 2.0, 0.5, 1.5];
    let epsilon = 1e-5;
    let standard = run(&ep, ("", 21), &input, &scale, &shape, -1, epsilon).unwrap();
    let contrib = run(
        &ep,
        ("com.microsoft", 1),
        &input,
        &scale,
        &shape,
        -1,
        epsilon,
    )
    .unwrap();
    let expected = reference(&input, &scale, 4, epsilon);

    assert_close("standard output", &standard.0, &expected.0);
    assert_close("standard invstd", &standard.1, &expected.1);
    assert_close("domain output parity", &standard.0, &contrib.0);
    assert_close("domain invstd parity", &standard.1, &contrib.1);
}

#[test]
fn simplified_layer_norm_does_not_contract_square_accumulation() {
    let Some(ep) = cuda_ep() else { return };
    let input = [-0.09129826, -1.0101787, 3.0318594, 5.774467];
    let scale = [1.0; 4];
    let expected = reference(&input, &scale, 4, 1e-5);
    let got = run(&ep, ("", 21), &input, &scale, &[1, 4], -1, 1e-5).unwrap();

    assert_eq!(got.0, expected.0);
    assert_eq!(got.1, expected.1);
}
