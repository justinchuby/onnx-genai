//! GPU numeric parity tests for `com.microsoft::SkipSimplifiedLayerNormalization`.

use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;

static GPU_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn f32_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: f32 is plain data and the byte slice retains the input lifetime.
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
    input_shape: &[usize],
    skip_shape: &[usize],
    input: &[f32],
    skip: &[f32],
    gamma: &[f32],
    bias: &[f32],
    epsilon: f32,
) -> onnx_runtime_ep_api::Result<Vec<Vec<f32>>> {
    let mut graph = Graph::new();
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let input_id = graph.create_named_value(
        "input",
        DataType::Float32,
        static_shape(input_shape.iter().copied()),
    );
    let skip_id = graph.create_named_value(
        "skip",
        DataType::Float32,
        static_shape(skip_shape.iter().copied()),
    );
    let gamma_id =
        graph.create_named_value("gamma", DataType::Float32, static_shape([gamma.len()]));
    let bias_id = graph.create_named_value("bias", DataType::Float32, static_shape([bias.len()]));
    for value in [input_id, skip_id, gamma_id, bias_id] {
        graph.add_input(value);
    }
    let stats_shape = input_shape[..input_shape.len() - 1]
        .iter()
        .copied()
        .chain(std::iter::once(1))
        .collect::<Vec<_>>();
    let y = graph.create_named_value(
        "y",
        DataType::Float32,
        static_shape(input_shape.iter().copied()),
    );
    let mean = graph.create_named_value(
        "mean",
        DataType::Float32,
        static_shape(stats_shape.iter().copied()),
    );
    let invstd = graph.create_named_value(
        "invstd",
        DataType::Float32,
        static_shape(stats_shape.iter().copied()),
    );
    let sum = graph.create_named_value(
        "sum",
        DataType::Float32,
        static_shape(input_shape.iter().copied()),
    );
    let mut node = Node::new(
        NodeId(0),
        "SkipSimplifiedLayerNormalization",
        vec![Some(input_id), Some(skip_id), Some(gamma_id), Some(bias_id)],
        vec![y, mean, invstd, sum],
    );
    node.domain = "com.microsoft".into();
    node.attributes
        .insert("epsilon".into(), Attribute::Float(epsilon));
    let node_id = graph.insert_node(node);
    for output in [y, mean, invstd, sum] {
        graph.add_output(output);
    }
    let model = Model::new(&graph);
    let kernel = ep.get_kernel(model.graph.node(node_id), &[], 1)?;

    let shapes = [input_shape, skip_shape, &[gamma.len()], &[bias.len()]];
    let host_inputs = [input, skip, gamma, bias];
    let runtime = ep.runtime();
    let device = ep.device_id();
    let mut input_buffers: Vec<DeviceBuffer> = Vec::new();
    for values in host_inputs {
        let buffer = ep.allocate(std::mem::size_of_val(values), 256)?;
        // SAFETY: allocation exactly covers the source byte slice.
        unsafe { runtime.htod(f32_bytes(values), cuptr(buffer.as_ptr()))? };
        input_buffers.push(buffer);
    }
    let input_strides: Vec<_> = shapes
        .iter()
        .map(|shape| compute_contiguous_strides(shape))
        .collect();
    let input_views: Vec<_> = input_buffers
        .iter()
        .zip(shapes)
        .zip(&input_strides)
        .map(|((buffer, shape), strides)| {
            TensorView::new(
                DevicePtr(buffer.as_ptr()),
                DataType::Float32,
                shape,
                strides,
                device,
            )
        })
        .collect();
    let output_shapes = [input_shape, &stats_shape, &stats_shape, input_shape];
    let output_strides: Vec<_> = output_shapes
        .iter()
        .map(|shape| compute_contiguous_strides(shape))
        .collect();
    let mut output_buffers = output_shapes
        .iter()
        .map(|shape| ep.allocate(shape.iter().product::<usize>() * 4, 256))
        .collect::<onnx_runtime_ep_api::Result<Vec<_>>>()?;
    {
        let mut output_views: Vec<_> = output_buffers
            .iter_mut()
            .zip(output_shapes)
            .zip(&output_strides)
            .map(|((buffer, shape), strides)| {
                TensorMut::new(
                    DevicePtrMut(buffer.as_mut_ptr()),
                    DataType::Float32,
                    shape,
                    strides,
                    device,
                )
            })
            .collect();
        kernel.execute(&input_views, &mut output_views)?;
    }
    let result = output_buffers
        .iter()
        .zip(output_shapes)
        .map(|(buffer, shape)| {
            let mut bytes = vec![0; shape.iter().product::<usize>() * 4];
            // SAFETY: destination exactly covers the device output allocation.
            unsafe { runtime.dtoh(&mut bytes, cuptr(buffer.as_ptr()))? };
            Ok(bytes_to_f32(&bytes))
        })
        .collect();
    drop(input_views);
    for buffer in input_buffers {
        ep.deallocate(buffer)?;
    }
    for buffer in output_buffers {
        ep.deallocate(buffer)?;
    }
    result
}

fn run_available(
    ep: &CudaExecutionProvider,
    input_shape: &[usize],
    skip_shape: &[usize],
    input: &[f32],
    skip: &[f32],
    gamma: &[f32],
    bias: &[f32],
    epsilon: f32,
) -> onnx_runtime_ep_api::Result<Vec<Vec<f32>>> {
    match run(
        ep,
        input_shape,
        skip_shape,
        input,
        skip,
        gamma,
        bias,
        epsilon,
    ) {
        Err(error) if format!("{error}").contains("CUDA_ERROR_UNSUPPORTED_PTX_VERSION") => {
            eprintln!("skip: NVRTC PTX is newer than the installed CUDA driver ({error})");
            Ok(Vec::new())
        }
        result => result,
    }
}

fn reference(
    input_shape: &[usize],
    skip_shape: &[usize],
    input: &[f32],
    skip: &[f32],
    gamma: &[f32],
    bias: &[f32],
    epsilon: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hidden = *input_shape.last().unwrap();
    let skip_strides = compute_contiguous_strides(skip_shape);
    let leading = input_shape.len() - skip_shape.len();
    let mut sum = vec![0.0; input.len()];
    for (flat, value) in sum.iter_mut().enumerate() {
        let mut linear = flat;
        let mut skip_index = 0;
        for axis in (0..input_shape.len()).rev() {
            let coord = linear % input_shape[axis];
            linear /= input_shape[axis];
            if axis >= leading {
                let skip_axis = axis - leading;
                if skip_shape[skip_axis] != 1 {
                    skip_index += coord * skip_strides[skip_axis] as usize;
                }
            }
        }
        *value = input[flat] + skip[skip_index] + bias[flat % hidden];
    }
    let mut output = vec![0.0; input.len()];
    let mut invstd = Vec::with_capacity(input.len() / hidden);
    for (row, normalized) in sum
        .chunks_exact(hidden)
        .zip(output.chunks_exact_mut(hidden))
    {
        let mean_square = row.iter().map(|value| value * value).sum::<f32>() / hidden as f32;
        let inv = 1.0 / (mean_square + epsilon).sqrt();
        invstd.push(inv);
        for (index, value) in row.iter().enumerate() {
            normalized[index] = value * inv * gamma[index];
        }
    }
    (output, sum, invstd)
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
fn skip_simplified_layer_norm_matches_independent_residual_rms_reference() {
    let _guard = GPU_SERIAL.lock().unwrap();
    let Some(ep) = cuda_ep() else {
        return;
    };
    let input_shape = [2, 3, 4];
    let skip_shape = [3, 4];
    let input = [
        1.0, -2.0, 3.0, -4.0, 0.5, 1.5, -0.5, 2.5, -1.0, 0.0, 1.0, 2.0, 3.0, -3.0, 2.0, -2.0, 1.25,
        -0.75, 0.25, -1.25, 2.0, 1.0, -2.0, -1.0,
    ];
    let skip = [
        0.25, -0.5, 0.75, -1.0, 1.0, 0.5, -1.5, -0.25, -0.75, 1.25, 0.5, -0.5,
    ];
    let gamma = [1.0, 0.5, 1.5, 2.0];
    let bias = [0.125, -0.25, 0.375, -0.5];
    let epsilon = 1e-4;
    let got = run_available(
        &ep,
        &input_shape,
        &skip_shape,
        &input,
        &skip,
        &gamma,
        &bias,
        epsilon,
    )
    .unwrap();
    if got.is_empty() {
        return;
    }
    let (expected_y, expected_sum, expected_invstd) = reference(
        &input_shape,
        &skip_shape,
        &input,
        &skip,
        &gamma,
        &bias,
        epsilon,
    );
    assert_close("normalized output", &got[0], &expected_y);
    assert_close("residual sum", &got[3], &expected_sum);
    assert_close("mean", &got[1], &vec![0.0; expected_invstd.len()]);
    assert_close("inverse RMS", &got[2], &expected_invstd);
}

#[test]
fn skip_simplified_layer_norm_does_not_contract_square_accumulation() {
    let _guard = GPU_SERIAL.lock().unwrap();
    let Some(ep) = cuda_ep() else {
        return;
    };
    let input = [-0.09129826, -1.0101787, 3.0318594, 5.774467];
    let skip = [0.0; 4];
    let gamma = [1.0; 4];
    let bias = [0.0; 4];
    let got = run_available(&ep, &[1, 4], &[1, 4], &input, &skip, &gamma, &bias, 1e-5).unwrap();
    if got.is_empty() {
        return;
    }
    let (expected_y, expected_sum, expected_invstd) =
        reference(&[1, 4], &[1, 4], &input, &skip, &gamma, &bias, 1e-5);

    assert_eq!(got[0], expected_y);
    assert_eq!(got[2], expected_invstd);
    assert_eq!(got[3], expected_sum);
}

#[test]
fn skip_simplified_layer_norm_fixed_decode_capture_replays_bit_identically() {
    let _guard = GPU_SERIAL.lock().unwrap();
    let Some(ep) = cuda_ep() else {
        return;
    };
    let input_shape = [1, 1, 4];
    let skip_shape = [1, 4];
    let output_shape = input_shape;
    let initial = [1.0f32, -2.0, 3.0, -4.0];
    let mutated = [-0.5f32, 1.5, -2.5, 3.5];
    let skip = [0.25f32, -0.5, 0.75, -1.0];
    let gamma = [1.0f32, 0.5, 1.5, 2.0];
    let bias = [0.125f32, -0.25, 0.375, -0.5];

    let mut graph = Graph::new();
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let input_id = graph.create_named_value("input", DataType::Float32, static_shape(input_shape));
    let skip_id = graph.create_named_value("skip", DataType::Float32, static_shape(skip_shape));
    let gamma_id = graph.create_named_value("gamma", DataType::Float32, static_shape([4]));
    let bias_id = graph.create_named_value("bias", DataType::Float32, static_shape([4]));
    for value in [input_id, skip_id, gamma_id, bias_id] {
        graph.add_input(value);
    }
    let output_id =
        graph.create_named_value("output", DataType::Float32, static_shape(output_shape));
    let mut node = Node::new(
        NodeId(0),
        "SkipSimplifiedLayerNormalization",
        vec![Some(input_id), Some(skip_id), Some(gamma_id), Some(bias_id)],
        vec![output_id],
    );
    node.domain = "com.microsoft".into();
    let node_id = graph.insert_node(node);
    graph.add_output(output_id);
    let model = Model::new(&graph);
    let kernel = ep.get_kernel(model.graph.node(node_id), &[], 1).unwrap();
    assert!(
        !kernel.cuda_graph_compatible(),
        "an unwarmed normalization kernel must not be capture eligible"
    );

    let runtime = ep.runtime();
    let device = ep.device_id();
    let input_values: [&[f32]; 4] = [&initial, &skip, &gamma, &bias];
    let input_shapes: [&[usize]; 4] = [&input_shape, &skip_shape, &[4], &[4]];
    let mut input_buffers = Vec::<DeviceBuffer>::new();
    for values in input_values {
        let buffer = ep.allocate(std::mem::size_of_val(values), 256).unwrap();
        // SAFETY: allocation exactly covers the source byte slice.
        unsafe {
            runtime
                .htod(f32_bytes(values), cuptr(buffer.as_ptr()))
                .unwrap()
        };
        input_buffers.push(buffer);
    }
    let input_strides = input_shapes
        .iter()
        .map(|shape| compute_contiguous_strides(shape))
        .collect::<Vec<_>>();
    let input_views = input_buffers
        .iter()
        .zip(input_shapes)
        .zip(&input_strides)
        .map(|((buffer, shape), strides)| {
            TensorView::new(
                DevicePtr(buffer.as_ptr()),
                DataType::Float32,
                shape,
                strides,
                device,
            )
        })
        .collect::<Vec<_>>();
    let mut output_buffer = ep.allocate(std::mem::size_of_val(&initial), 256).unwrap();
    let output_strides = compute_contiguous_strides(&output_shape);
    macro_rules! execute {
        () => {{
            let output = TensorMut::new(
                DevicePtrMut(output_buffer.as_mut_ptr()),
                DataType::Float32,
                &output_shape,
                &output_strides,
                device,
            );
            kernel.execute(&input_views, &mut [output]).unwrap();
        }};
    }

    execute!();
    assert!(
        kernel.cuda_graph_compatible(),
        "warmed single-group decode must be capture eligible"
    );
    // SAFETY: the input allocation exactly covers the replacement activation.
    unsafe {
        runtime
            .htod(f32_bytes(&mutated), cuptr(input_buffers[0].as_ptr()))
            .unwrap()
    };
    execute!();
    let mut eager = vec![0u8; std::mem::size_of_val(&mutated)];
    // SAFETY: destination exactly covers the output allocation.
    unsafe {
        runtime
            .dtoh(&mut eager, cuptr(output_buffer.as_ptr()))
            .unwrap()
    };

    // SAFETY: the input allocation exactly covers the replacement activation.
    unsafe {
        runtime
            .htod(f32_bytes(&initial), cuptr(input_buffers[0].as_ptr()))
            .unwrap()
    };
    let allocation_counts = runtime.allocation_counts();
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    execute!();
    runtime.end_graph_capture().unwrap();
    // SAFETY: the input allocation exactly covers the replacement activation.
    unsafe {
        runtime
            .htod(f32_bytes(&mutated), cuptr(input_buffers[0].as_ptr()))
            .unwrap()
    };
    runtime.replay_graph().unwrap();
    let mut replayed = vec![0u8; eager.len()];
    // SAFETY: destination exactly covers the output allocation.
    unsafe {
        runtime
            .dtoh(&mut replayed, cuptr(output_buffer.as_ptr()))
            .unwrap()
    };
    assert_eq!(replayed, eager);
    assert_eq!(runtime.allocation_counts(), allocation_counts);
    assert!(runtime.reset_graph().unwrap());

    drop(input_views);
    for buffer in input_buffers {
        ep.deallocate(buffer).unwrap();
    }
    ep.deallocate(output_buffer).unwrap();

    let prefill_input = [0.5f32; 8];
    let prefill_shape = [1, 2, 4];
    let prefill_values: [&[f32]; 4] = [&prefill_input, &skip, &gamma, &bias];
    let prefill_shapes: [&[usize]; 4] = [&prefill_shape, &skip_shape, &[4], &[4]];
    let mut prefill_buffers = Vec::<DeviceBuffer>::new();
    for values in prefill_values {
        let buffer = ep.allocate(std::mem::size_of_val(values), 256).unwrap();
        // SAFETY: allocation exactly covers the source byte slice.
        unsafe {
            runtime
                .htod(f32_bytes(values), cuptr(buffer.as_ptr()))
                .unwrap()
        };
        prefill_buffers.push(buffer);
    }
    let prefill_strides = prefill_shapes
        .iter()
        .map(|shape| compute_contiguous_strides(shape))
        .collect::<Vec<_>>();
    let prefill_views = prefill_buffers
        .iter()
        .zip(prefill_shapes)
        .zip(&prefill_strides)
        .map(|((buffer, shape), strides)| {
            TensorView::new(
                DevicePtr(buffer.as_ptr()),
                DataType::Float32,
                shape,
                strides,
                device,
            )
        })
        .collect::<Vec<_>>();
    let mut prefill_output = ep
        .allocate(std::mem::size_of_val(&prefill_input), 256)
        .unwrap();
    let prefill_output_strides = compute_contiguous_strides(&prefill_shape);
    let output = TensorMut::new(
        DevicePtrMut(prefill_output.as_mut_ptr()),
        DataType::Float32,
        &prefill_shape,
        &prefill_output_strides,
        device,
    );
    kernel.execute(&prefill_views, &mut [output]).unwrap();
    assert!(
        !kernel.cuda_graph_compatible(),
        "multi-group prefill must not be capture eligible"
    );
    drop(prefill_views);
    for buffer in prefill_buffers {
        ep.deallocate(buffer).unwrap();
    }
    ep.deallocate(prefill_output).unwrap();
}
