use half::f16;
use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, DeviceId, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;

#[derive(Clone)]
struct HostTensor {
    dtype: DataType,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn typed_bytes<T: Copy>(values: &[T]) -> Vec<u8> {
    // SAFETY: test inputs are primitive POD values.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
            .to_vec()
    }
}

fn tensor<T: Copy>(dtype: DataType, shape: &[usize], values: &[T]) -> HostTensor {
    HostTensor {
        dtype,
        shape: shape.to_vec(),
        bytes: typed_bytes(values),
    }
}

fn gpu() -> Option<CudaExecutionProvider> {
    match CudaExecutionProvider::new_default() {
        Ok(ep) => Some(ep),
        Err(error) => {
            eprintln!("skip: no CUDA GPU available ({error})");
            None
        }
    }
}

fn quantize(
    weights_nk: &[f32],
    n: usize,
    k: usize,
    block_size: usize,
    asymmetric: bool,
) -> (Vec<u8>, Vec<f32>, Option<Vec<u8>>) {
    let blocks = k.div_ceil(block_size);
    let blob_size = block_size / 2;
    let mut packed = vec![0u8; n * blocks * blob_size];
    let mut scales = vec![0.0f32; n * blocks];
    let mut zero_points = vec![0u8; n * blocks.div_ceil(2)];

    for output in 0..n {
        for block in 0..blocks {
            let start = block * block_size;
            let end = (start + block_size).min(k);
            let values = &weights_nk[output * k + start..output * k + end];
            let (scale, zero_point) = if asymmetric {
                let min = values.iter().copied().fold(f32::INFINITY, f32::min);
                let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let scale = ((max - min) / 15.0).max(1e-6);
                (scale, (-min / scale).round().clamp(0.0, 15.0) as u8)
            } else {
                let max_abs = values.iter().map(|value| value.abs()).fold(0.0, f32::max);
                ((max_abs / 7.0).max(1e-6), 8)
            };
            scales[output * blocks + block] = scale;
            if asymmetric {
                zero_points[output * blocks.div_ceil(2) + block / 2] |=
                    zero_point << (4 * (block % 2));
            }
            for (offset, &value) in values.iter().enumerate() {
                let quantized = (value / scale + zero_point as f32).round().clamp(0.0, 15.0) as u8;
                packed[(output * blocks + block) * blob_size + offset / 2] |=
                    quantized << (4 * (offset % 2));
            }
        }
    }
    (packed, scales, asymmetric.then_some(zero_points))
}

fn independent_reference(
    activations: &[f32],
    packed: &[u8],
    scales: &[f32],
    zero_points: Option<&[u8]>,
    m: usize,
    k: usize,
    n: usize,
    block_size: usize,
) -> Vec<f32> {
    let blocks = k.div_ceil(block_size);
    let blob_size = block_size / 2;
    let zp_row_bytes = blocks.div_ceil(2);
    let mut output = vec![0.0f32; m * n];
    for row in 0..m {
        for column in 0..n {
            let mut sum = 0.0f32;
            for depth in 0..k {
                let block = depth / block_size;
                let within = depth % block_size;
                let byte = packed[(column * blocks + block) * blob_size + within / 2];
                let quantized = if within % 2 == 0 {
                    byte & 0x0f
                } else {
                    byte >> 4
                };
                let zero_point = zero_points.map_or(8, |points| {
                    let byte = points[column * zp_row_bytes + block / 2];
                    if block % 2 == 0 {
                        byte & 0x0f
                    } else {
                        byte >> 4
                    }
                });
                let weight =
                    (quantized as f32 - zero_point as f32) * scales[column * blocks + block];
                sum += activations[row * k + depth] * weight;
            }
            output[row * n + column] = sum;
        }
    }
    output
}

fn accuracy4_reference(
    activations: &[f32],
    packed: &[u8],
    scales: &[f32],
    k: usize,
    n: usize,
    block_size: usize,
) -> Vec<f32> {
    let blocks = k.div_ceil(block_size);
    let blob_size = block_size / 2;
    let max_abs = activations
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f32::max);
    if max_abs == 0.0 {
        return vec![0.0; n];
    }
    let activation_scale = max_abs / 127.0;
    let inverse_scale = activation_scale.recip();
    let quantized_activations: Vec<i32> = activations
        .iter()
        .map(|value| (value * inverse_scale).round().clamp(-127.0, 127.0) as i32)
        .collect();
    let mut output = vec![0.0; n];
    for column in 0..n {
        let mut value = 0.0;
        for block in 0..blocks {
            let begin = block * block_size;
            let end = (begin + block_size).min(k);
            let packed_start = (column * blocks + block) * blob_size;
            let mut dot = 0i32;
            for depth in begin..end {
                let within = depth - begin;
                let byte = packed[packed_start + within / 2];
                let quantized_weight = if within % 2 == 0 {
                    byte & 0x0f
                } else {
                    byte >> 4
                };
                dot += quantized_activations[depth] * (i32::from(quantized_weight) - 8);
            }
            value += dot as f32 * scales[column * blocks + block];
        }
        output[column] = value * activation_scale;
    }
    output
}

fn accuracy4_reference_columns(
    activations: &[f32],
    packed: &[u8],
    scales: &[f32],
    k: usize,
    n: usize,
    block_size: usize,
    columns: &[usize],
) -> Vec<f32> {
    let blocks = k.div_ceil(block_size);
    let blob_size = block_size / 2;
    let max_abs = activations
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f32::max);
    if max_abs == 0.0 {
        return vec![0.0; columns.len()];
    }
    let activation_scale = max_abs / 127.0;
    let inverse_scale = activation_scale.recip();
    let quantized_activations: Vec<i32> = activations
        .iter()
        .map(|value| (value * inverse_scale).round().clamp(-127.0, 127.0) as i32)
        .collect();
    columns
        .iter()
        .map(|&column| {
            assert!(column < n);
            let mut value = 0.0;
            for block in 0..blocks {
                let begin = block * block_size;
                let end = (begin + block_size).min(k);
                let packed_start = (column * blocks + block) * blob_size;
                let mut dot = 0i32;
                for depth in begin..end {
                    let within = depth - begin;
                    let byte = packed[packed_start + within / 2];
                    let quantized_weight = if within % 2 == 0 {
                        byte & 0x0f
                    } else {
                        byte >> 4
                    };
                    dot += quantized_activations[depth] * (i32::from(quantized_weight) - 8);
                }
                value += dot as f32 * scales[column * blocks + block];
            }
            value * activation_scale
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_case_with_bits(
    ep: &CudaExecutionProvider,
    a_shape: &[usize],
    activations: &[f32],
    packed: &[u8],
    scales: &[f32],
    zero_points: Option<&[u8]>,
    k: usize,
    n: usize,
    block_size: usize,
    bits: usize,
    accuracy_level: i64,
) -> onnx_runtime_ep_api::Result<Vec<f32>> {
    let blocks = k.div_ceil(block_size);
    let blob_size = block_size * bits / 8;
    let zp_row_bytes = (blocks * bits).div_ceil(8);
    let mut graph = Graph::new();
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let a = graph.create_named_value(
        "A",
        DataType::Float32,
        static_shape(a_shape.iter().copied()),
    );
    let b = graph.create_named_value("B", DataType::Uint8, static_shape([n, blocks, blob_size]));
    let scales_value =
        graph.create_named_value("scales", DataType::Float32, static_shape([n, blocks]));
    for value in [a, b, scales_value] {
        graph.add_input(value);
    }
    let mut node_inputs = vec![Some(a), Some(b), Some(scales_value)];
    if zero_points.is_some() {
        let zp = graph.create_named_value(
            "zero_points",
            DataType::Uint8,
            static_shape([n, zp_row_bytes]),
        );
        graph.add_input(zp);
        node_inputs.push(Some(zp));
    }
    let mut output_shape = a_shape[..a_shape.len() - 1].to_vec();
    output_shape.push(n);
    let output = graph.create_named_value(
        "Y",
        DataType::Float32,
        static_shape(output_shape.iter().copied()),
    );
    let mut node = Node::new(NodeId(0), "MatMulNBits", node_inputs, vec![output]);
    node.domain = "com.microsoft".into();
    node.attributes.insert("K".into(), Attribute::Int(k as i64));
    node.attributes.insert("N".into(), Attribute::Int(n as i64));
    node.attributes
        .insert("bits".into(), Attribute::Int(bits as i64));
    node.attributes
        .insert("block_size".into(), Attribute::Int(block_size as i64));
    if accuracy_level != 0 {
        node.attributes
            .insert("accuracy_level".into(), Attribute::Int(accuracy_level));
    }
    let node = graph.insert_node(node);
    graph.add_output(output);
    let model = Model::new(&graph);
    let kernel = ep.get_kernel(model.graph.node(node), &[], 1)?;
    assert!(!kernel.cuda_graph_compatible());

    let mut inputs = vec![
        tensor(DataType::Float32, a_shape, activations),
        tensor(DataType::Uint8, &[n, blocks, blob_size], packed),
        tensor(DataType::Float32, &[n, blocks], scales),
    ];
    if let Some(zero_points) = zero_points {
        inputs.push(tensor(DataType::Uint8, &[n, zp_row_bytes], zero_points));
    }

    let runtime = ep.runtime();
    let device = ep.device_id();
    let mut input_buffers = Vec::<DeviceBuffer>::new();
    for input in &inputs {
        let buffer = ep.allocate(input.bytes.len(), 256)?;
        // SAFETY: allocation size equals the source byte length.
        unsafe { runtime.htod(&input.bytes, cuptr(buffer.as_ptr()))? };
        input_buffers.push(buffer);
    }
    let input_strides: Vec<_> = inputs
        .iter()
        .map(|input| compute_contiguous_strides(&input.shape))
        .collect();
    let input_views: Vec<_> = inputs
        .iter()
        .zip(&input_buffers)
        .zip(&input_strides)
        .map(|((input, buffer), strides)| {
            TensorView::new(
                DevicePtr(buffer.as_ptr()),
                input.dtype,
                &input.shape,
                strides,
                device,
            )
        })
        .collect();

    let output_len = output_shape.iter().product::<usize>();
    let mut output_buffer = ep.allocate(output_len * 4, 256)?;
    let output_strides = compute_contiguous_strides(&output_shape);
    let output_view = TensorMut::new(
        DevicePtrMut(output_buffer.as_mut_ptr()),
        DataType::Float32,
        &output_shape,
        &output_strides,
        device,
    );
    kernel.execute(&input_views, &mut [output_view])?;
    assert_eq!(
        kernel.cuda_graph_compatible(),
        a_shape[..a_shape.len() - 1].iter().product::<usize>() == 1
    );

    let mut bytes = vec![0u8; output_len * 4];
    // SAFETY: output allocation contains `output_len` f32 values.
    unsafe { runtime.dtoh(&mut bytes, cuptr(output_buffer.as_ptr()))? };
    drop(input_views);
    for buffer in input_buffers {
        ep.deallocate(buffer)?;
    }
    ep.deallocate(output_buffer)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|value| f32::from_ne_bytes(value.try_into().unwrap()))
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    ep: &CudaExecutionProvider,
    a_shape: &[usize],
    activations: &[f32],
    packed: &[u8],
    scales: &[f32],
    zero_points: Option<&[u8]>,
    k: usize,
    n: usize,
    block_size: usize,
    accuracy_level: i64,
) -> onnx_runtime_ep_api::Result<Vec<f32>> {
    run_case_with_bits(
        ep,
        a_shape,
        activations,
        packed,
        scales,
        zero_points,
        k,
        n,
        block_size,
        4,
        accuracy_level,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_f16_case_with_bits(
    ep: &CudaExecutionProvider,
    activations: &[f16],
    packed: &[u8],
    scales: &[f16],
    zero_points: Option<&[u8]>,
    k: usize,
    n: usize,
    bits: usize,
) -> onnx_runtime_ep_api::Result<Vec<f16>> {
    let block_size = 32usize;
    let blocks = k.div_ceil(block_size);
    let blob_size = block_size * bits / 8;
    let mut graph = Graph::new();
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let a = graph.create_named_value("A", DataType::Float16, static_shape([1, k]));
    let b = graph.create_named_value("B", DataType::Uint8, static_shape([n, blocks, blob_size]));
    let scales_value =
        graph.create_named_value("scales", DataType::Float16, static_shape([n, blocks]));
    for value in [a, b, scales_value] {
        graph.add_input(value);
    }
    let mut node_inputs = vec![Some(a), Some(b), Some(scales_value)];
    if zero_points.is_some() {
        let zp =
            graph.create_named_value("zero_points", DataType::Uint8, static_shape([n, blocks]));
        graph.add_input(zp);
        node_inputs.push(Some(zp));
    }
    let output = graph.create_named_value("Y", DataType::Float16, static_shape([1, n]));
    let mut node = Node::new(NodeId(0), "MatMulNBits", node_inputs, vec![output]);
    node.domain = "com.microsoft".into();
    node.attributes.insert("K".into(), Attribute::Int(k as i64));
    node.attributes.insert("N".into(), Attribute::Int(n as i64));
    node.attributes
        .insert("bits".into(), Attribute::Int(bits as i64));
    node.attributes
        .insert("block_size".into(), Attribute::Int(block_size as i64));
    node.attributes
        .insert("accuracy_level".into(), Attribute::Int(4));
    let node = graph.insert_node(node);
    graph.add_output(output);
    let model = Model::new(&graph);
    let kernel = ep.get_kernel(model.graph.node(node), &[], 1)?;
    assert!(!kernel.cuda_graph_compatible());

    let mut inputs = vec![
        tensor(DataType::Float16, &[1, k], activations),
        tensor(DataType::Uint8, &[n, blocks, blob_size], packed),
        tensor(DataType::Float16, &[n, blocks], scales),
    ];
    if let Some(zero_points) = zero_points {
        inputs.push(tensor(DataType::Uint8, &[n, blocks], zero_points));
    }
    let runtime = ep.runtime();
    let device = ep.device_id();
    let mut input_buffers = Vec::<DeviceBuffer>::new();
    for input in &inputs {
        let buffer = ep.allocate(input.bytes.len(), 256)?;
        // SAFETY: allocation size equals the source byte length.
        unsafe { runtime.htod(&input.bytes, cuptr(buffer.as_ptr()))? };
        input_buffers.push(buffer);
    }
    let input_strides: Vec<_> = inputs
        .iter()
        .map(|input| compute_contiguous_strides(&input.shape))
        .collect();
    let input_views: Vec<_> = inputs
        .iter()
        .zip(&input_buffers)
        .zip(&input_strides)
        .map(|((input, buffer), strides)| {
            TensorView::new(
                DevicePtr(buffer.as_ptr()),
                input.dtype,
                &input.shape,
                strides,
                device,
            )
        })
        .collect();
    let mut output_buffer = ep.allocate(n * 2, 256)?;
    let output_shape = [1, n];
    let output_strides = compute_contiguous_strides(&output_shape);
    kernel.execute(
        &input_views,
        &mut [TensorMut::new(
            DevicePtrMut(output_buffer.as_mut_ptr()),
            DataType::Float16,
            &output_shape,
            &output_strides,
            device,
        )],
    )?;
    assert!(kernel.cuda_graph_compatible());
    if bits == 8 {
        let mut eager = vec![0u8; n * 2];
        // SAFETY: output allocation contains `n` fp16 values.
        unsafe { runtime.dtoh(&mut eager, cuptr(output_buffer.as_ptr()))? };
        let allocation_counts = runtime.allocation_counts();
        runtime.begin_graph_capture(&[kernel.as_ref()])?;
        kernel.execute(
            &input_views,
            &mut [TensorMut::new(
                DevicePtrMut(output_buffer.as_mut_ptr()),
                DataType::Float16,
                &output_shape,
                &output_strides,
                device,
            )],
        )?;
        runtime.end_graph_capture()?;
        runtime.replay_graph()?;
        let mut replayed = vec![0u8; n * 2];
        // SAFETY: output allocation contains `n` fp16 values.
        unsafe { runtime.dtoh(&mut replayed, cuptr(output_buffer.as_ptr()))? };
        assert_eq!(replayed, eager);
        assert_eq!(runtime.allocation_counts(), allocation_counts);
        assert!(runtime.reset_graph()?);
    }

    let mut bytes = vec![0u8; n * 2];
    // SAFETY: output allocation contains `n` fp16 values.
    unsafe { runtime.dtoh(&mut bytes, cuptr(output_buffer.as_ptr()))? };
    drop(input_views);
    for buffer in input_buffers {
        ep.deallocate(buffer)?;
    }
    ep.deallocate(output_buffer)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|value| f16::from_bits(u16::from_ne_bytes(value.try_into().unwrap())))
        .collect())
}

fn run_f16_case(
    ep: &CudaExecutionProvider,
    activations: &[f16],
    packed: &[u8],
    scales: &[f16],
    k: usize,
    n: usize,
) -> onnx_runtime_ep_api::Result<Vec<f16>> {
    run_f16_case_with_bits(ep, activations, packed, scales, None, k, n, 4)
}

#[allow(clippy::too_many_arguments)]
fn run_cpu_case_with_bits(
    a_shape: &[usize],
    activations: &[f32],
    packed: &[u8],
    scales: &[f32],
    zero_points: Option<&[u8]>,
    k: usize,
    n: usize,
    block_size: usize,
    bits: usize,
    accuracy_level: i64,
) -> Vec<f32> {
    let blocks = k.div_ceil(block_size);
    let blob_size = block_size * bits / 8;
    let zp_row_bytes = (blocks * bits).div_ceil(8);
    let mut graph = Graph::new();
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let a = graph.create_named_value(
        "A",
        DataType::Float32,
        static_shape(a_shape.iter().copied()),
    );
    let b = graph.create_named_value("B", DataType::Uint8, static_shape([n, blocks, blob_size]));
    let scales_value =
        graph.create_named_value("scales", DataType::Float32, static_shape([n, blocks]));
    for value in [a, b, scales_value] {
        graph.add_input(value);
    }
    let mut node_inputs = vec![Some(a), Some(b), Some(scales_value)];
    if zero_points.is_some() {
        let zp = graph.create_named_value(
            "zero_points",
            DataType::Uint8,
            static_shape([n, zp_row_bytes]),
        );
        graph.add_input(zp);
        node_inputs.push(Some(zp));
    }
    let mut output_shape = a_shape[..a_shape.len() - 1].to_vec();
    output_shape.push(n);
    let output = graph.create_named_value(
        "Y",
        DataType::Float32,
        static_shape(output_shape.iter().copied()),
    );
    let mut node = Node::new(NodeId(0), "MatMulNBits", node_inputs, vec![output]);
    node.domain = "com.microsoft".into();
    node.attributes.insert("K".into(), Attribute::Int(k as i64));
    node.attributes.insert("N".into(), Attribute::Int(n as i64));
    node.attributes
        .insert("bits".into(), Attribute::Int(bits as i64));
    node.attributes
        .insert("block_size".into(), Attribute::Int(block_size as i64));
    node.attributes
        .insert("accuracy_level".into(), Attribute::Int(accuracy_level));
    let node = graph.insert_node(node);
    graph.add_output(output);
    let model = Model::new(&graph);
    let kernel = CpuExecutionProvider::new()
        .get_kernel(model.graph.node(node), &[], 1)
        .unwrap();

    let mut inputs = vec![
        tensor(DataType::Float32, a_shape, activations),
        tensor(DataType::Uint8, &[n, blocks, blob_size], packed),
        tensor(DataType::Float32, &[n, blocks], scales),
    ];
    if let Some(zero_points) = zero_points {
        inputs.push(tensor(DataType::Uint8, &[n, zp_row_bytes], zero_points));
    }
    let input_strides: Vec<_> = inputs
        .iter()
        .map(|input| compute_contiguous_strides(&input.shape))
        .collect();
    let input_views: Vec<_> = inputs
        .iter()
        .zip(&input_strides)
        .map(|(input, strides)| {
            TensorView::new(
                DevicePtr(input.bytes.as_ptr().cast()),
                input.dtype,
                &input.shape,
                strides,
                DeviceId::cpu(),
            )
        })
        .collect();
    let mut output_bytes = vec![0u8; output_shape.iter().product::<usize>() * 4];
    let output_strides = compute_contiguous_strides(&output_shape);
    let output_view = TensorMut::new(
        DevicePtrMut(output_bytes.as_mut_ptr().cast()),
        DataType::Float32,
        &output_shape,
        &output_strides,
        DeviceId::cpu(),
    );
    kernel.execute(&input_views, &mut [output_view]).unwrap();
    output_bytes
        .chunks_exact(4)
        .map(|value| f32::from_ne_bytes(value.try_into().unwrap()))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_cpu_case(
    a_shape: &[usize],
    activations: &[f32],
    packed: &[u8],
    scales: &[f32],
    k: usize,
    n: usize,
    block_size: usize,
    accuracy_level: i64,
) -> Vec<f32> {
    run_cpu_case_with_bits(
        a_shape,
        activations,
        packed,
        scales,
        None,
        k,
        n,
        block_size,
        4,
        accuracy_level,
    )
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 2e-4_f32.max(expected.abs() * 2e-5);
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

fn error_metrics(actual: &[f32], expected: &[f32]) -> (f32, u32) {
    actual.iter().zip(expected).fold(
        (0.0f32, 0u32),
        |(max_abs, max_ulp), (&actual, &expected)| {
            let actual_key = if actual.is_sign_negative() {
                !actual.to_bits()
            } else {
                actual.to_bits() | 0x8000_0000
            };
            let expected_key = if expected.is_sign_negative() {
                !expected.to_bits()
            } else {
                expected.to_bits() | 0x8000_0000
            };
            (
                max_abs.max((actual - expected).abs()),
                max_ulp.max(actual_key.abs_diff(expected_key)),
            )
        },
    )
}

fn random_u32(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 32) as u32
}

fn run_int8_cpu_gpu_parity(explicit_zero_points: bool) {
    let Some(ep) = gpu() else { return };
    let (m, k, n, block_size, bits) = (1usize, 77usize, 19usize, 32usize, 8usize);
    let blocks = k.div_ceil(block_size);
    let mut state = if explicit_zero_points {
        0x8f0b_7a51_d264_c39eu64
    } else {
        0x1ce4_a902_76bd_53f8u64
    };
    let activations: Vec<f32> = (0..m * k)
        .map(|_| (random_u32(&mut state) as f32 / u32::MAX as f32 - 0.5) * 4.0)
        .collect();
    let packed: Vec<u8> = (0..n * blocks * block_size)
        .map(|_| random_u32(&mut state) as u8)
        .collect();
    let scales: Vec<f32> = (0..n * blocks)
        .map(|_| 0.001 + (random_u32(&mut state) as f32 / u32::MAX as f32) * 0.02)
        .collect();
    let zero_points = explicit_zero_points.then(|| {
        (0..n * blocks)
            .map(|_| random_u32(&mut state) as u8)
            .collect::<Vec<_>>()
    });

    let expected = run_cpu_case_with_bits(
        &[m, k],
        &activations,
        &packed,
        &scales,
        zero_points.as_deref(),
        k,
        n,
        block_size,
        bits,
        4,
    );
    let actual = run_case_with_bits(
        &ep,
        &[m, k],
        &activations,
        &packed,
        &scales,
        zero_points.as_deref(),
        k,
        n,
        block_size,
        bits,
        4,
    )
    .unwrap();
    assert_close(&actual, &expected);
    let (max_abs, max_ulp) = error_metrics(&actual, &expected);
    eprintln!(
        "MatMulNBits int8 block32 CPU/CUDA parity explicit_zp={explicit_zero_points} \
         max_abs_diff={max_abs:e} max_ulp_diff={max_ulp}"
    );
}

#[test]
fn matmul_nbits_gpu_int8_block32_default_zero_point_matches_cpu() {
    run_int8_cpu_gpu_parity(false);
}

#[test]
fn matmul_nbits_gpu_int8_block32_explicit_zero_points_match_cpu() {
    run_int8_cpu_gpu_parity(true);
}

#[test]
fn matmul_nbits_gpu_int8_block32_batched_fallback_matches_cpu() {
    let Some(ep) = gpu() else { return };
    let (m, k, n, block_size, bits) = (3usize, 77usize, 19usize, 32usize, 8usize);
    let blocks = k.div_ceil(block_size);
    let mut state = 0xd2c7_406e_1ab9_f853u64;
    let activations: Vec<f32> = (0..m * k)
        .map(|_| (random_u32(&mut state) as f32 / u32::MAX as f32 - 0.5) * 4.0)
        .collect();
    let packed: Vec<u8> = (0..n * blocks * block_size)
        .map(|_| random_u32(&mut state) as u8)
        .collect();
    let scales: Vec<f32> = (0..n * blocks)
        .map(|_| 0.001 + (random_u32(&mut state) as f32 / u32::MAX as f32) * 0.02)
        .collect();
    let zero_points: Vec<u8> = (0..n * blocks)
        .map(|_| random_u32(&mut state) as u8)
        .collect();
    let expected = run_cpu_case_with_bits(
        &[m, k],
        &activations,
        &packed,
        &scales,
        Some(&zero_points),
        k,
        n,
        block_size,
        bits,
        4,
    );
    let actual = run_case_with_bits(
        &ep,
        &[m, k],
        &activations,
        &packed,
        &scales,
        Some(&zero_points),
        k,
        n,
        block_size,
        bits,
        4,
    )
    .unwrap();
    assert_close(&actual, &expected);
}

#[test]
fn matmul_nbits_gpu_int8_block32_capture_replay_is_bit_exact() {
    let Some(ep) = gpu() else { return };
    let (k, n, block_size) = (77usize, 19usize, 32usize);
    let blocks = k.div_ceil(block_size);
    let mut state = 0x9da3_51e7_24bc_08f6u64;
    let initial: Vec<f32> = (0..k)
        .map(|_| (random_u32(&mut state) as f32 / u32::MAX as f32 - 0.5) * 4.0)
        .collect();
    let mutated: Vec<f32> = (0..k)
        .map(|_| (random_u32(&mut state) as f32 / u32::MAX as f32 - 0.5) * 6.0)
        .collect();
    let packed: Vec<u8> = (0..n * blocks * block_size)
        .map(|_| random_u32(&mut state) as u8)
        .collect();
    let scales: Vec<f32> = (0..n * blocks)
        .map(|_| 0.001 + (random_u32(&mut state) as f32 / u32::MAX as f32) * 0.02)
        .collect();
    let zero_points: Vec<u8> = (0..n * blocks)
        .map(|_| random_u32(&mut state) as u8)
        .collect();

    let mut graph = Graph::new();
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let a = graph.create_named_value("A", DataType::Float32, static_shape([1, k]));
    let b = graph.create_named_value("B", DataType::Uint8, static_shape([n, blocks, block_size]));
    let scales_value =
        graph.create_named_value("scales", DataType::Float32, static_shape([n, blocks]));
    let zp = graph.create_named_value("zero_points", DataType::Uint8, static_shape([n, blocks]));
    for value in [a, b, scales_value, zp] {
        graph.add_input(value);
    }
    let output = graph.create_named_value("Y", DataType::Float32, static_shape([1, n]));
    let mut node = Node::new(
        NodeId(0),
        "MatMulNBits",
        vec![Some(a), Some(b), Some(scales_value), Some(zp)],
        vec![output],
    );
    node.domain = "com.microsoft".into();
    node.attributes.insert("K".into(), Attribute::Int(k as i64));
    node.attributes.insert("N".into(), Attribute::Int(n as i64));
    node.attributes.insert("bits".into(), Attribute::Int(8));
    node.attributes
        .insert("block_size".into(), Attribute::Int(block_size as i64));
    node.attributes
        .insert("accuracy_level".into(), Attribute::Int(4));
    let node = graph.insert_node(node);
    graph.add_output(output);
    let model = Model::new(&graph);
    let kernel = ep.get_kernel(model.graph.node(node), &[], 1).unwrap();

    let runtime = ep.runtime();
    let device = ep.device_id();
    let inputs = [
        tensor(DataType::Float32, &[1, k], &initial),
        tensor(DataType::Uint8, &[n, blocks, block_size], &packed),
        tensor(DataType::Float32, &[n, blocks], &scales),
        tensor(DataType::Uint8, &[n, blocks], &zero_points),
    ];
    let mut input_buffers = Vec::<DeviceBuffer>::new();
    for input in &inputs {
        let buffer = ep.allocate(input.bytes.len(), 256).unwrap();
        // SAFETY: allocation size equals the source byte length.
        unsafe { runtime.htod(&input.bytes, cuptr(buffer.as_ptr())).unwrap() };
        input_buffers.push(buffer);
    }
    let input_strides: Vec<_> = inputs
        .iter()
        .map(|input| compute_contiguous_strides(&input.shape))
        .collect();
    let input_views: Vec<_> = inputs
        .iter()
        .zip(&input_buffers)
        .zip(&input_strides)
        .map(|((input, buffer), strides)| {
            TensorView::new(
                DevicePtr(buffer.as_ptr()),
                input.dtype,
                &input.shape,
                strides,
                device,
            )
        })
        .collect();
    let mut output_buffer = ep.allocate(n * 4, 256).unwrap();
    let output_shape = [1, n];
    let output_strides = compute_contiguous_strides(&output_shape);
    macro_rules! execute {
        () => {
            kernel
                .execute(
                    &input_views,
                    &mut [TensorMut::new(
                        DevicePtrMut(output_buffer.as_mut_ptr()),
                        DataType::Float32,
                        &output_shape,
                        &output_strides,
                        device,
                    )],
                )
                .unwrap()
        };
    }

    execute!();
    assert!(kernel.cuda_graph_compatible());
    let mutated_bytes = typed_bytes(&mutated);
    // SAFETY: the activation allocation exactly covers the source bytes.
    unsafe {
        runtime
            .htod(&mutated_bytes, cuptr(input_buffers[0].as_ptr()))
            .unwrap()
    };
    execute!();
    let mut eager = vec![0u8; n * 4];
    // SAFETY: output allocation exactly covers `eager`.
    unsafe {
        runtime
            .dtoh(&mut eager, cuptr(output_buffer.as_ptr()))
            .unwrap()
    };

    let initial_bytes = typed_bytes(&initial);
    // SAFETY: the activation allocation exactly covers the source bytes.
    unsafe {
        runtime
            .htod(&initial_bytes, cuptr(input_buffers[0].as_ptr()))
            .unwrap()
    };
    let allocation_counts = runtime.allocation_counts();
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    execute!();
    runtime.end_graph_capture().unwrap();
    // SAFETY: the activation allocation exactly covers the source bytes.
    unsafe {
        runtime
            .htod(&mutated_bytes, cuptr(input_buffers[0].as_ptr()))
            .unwrap()
    };
    runtime.replay_graph().unwrap();
    let mut replayed = vec![0u8; n * 4];
    // SAFETY: output allocation exactly covers `replayed`.
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
}

#[test]
fn matmul_nbits_gpu_gemv_streams_random_packed_int4() {
    let Some(ep) = gpu() else { return };
    let (m, k, n, block_size) = (1usize, 1003usize, 37usize, 32usize);
    let blocks = k.div_ceil(block_size);
    let blob_size = block_size / 2;
    let mut state = 0x7d54_3a91_c2e8_6b0fu64;
    let activations: Vec<f32> = (0..k)
        .map(|_| (random_u32(&mut state) as f32 / u32::MAX as f32 - 0.5) * 4.0)
        .collect();
    let packed: Vec<u8> = (0..n * blocks * blob_size)
        .map(|_| random_u32(&mut state) as u8)
        .collect();
    let scales: Vec<f32> = (0..n * blocks)
        .map(|_| 0.002 + (random_u32(&mut state) as f32 / u32::MAX as f32) * 0.08)
        .collect();
    let expected = independent_reference(&activations, &packed, &scales, None, m, k, n, block_size);
    let actual = run_case(
        &ep,
        &[m, k],
        &activations,
        &packed,
        &scales,
        None,
        k,
        n,
        block_size,
        0,
    )
    .unwrap();
    assert_close(&actual, &expected);
    let (max_abs, max_ulp) = error_metrics(&actual, &expected);
    eprintln!(
        "MatMulNBits default GEMV CPU-reference max_abs_diff={max_abs:e} max_ulp_diff={max_ulp}"
    );
}

#[test]
fn matmul_nbits_gpu_accuracy4_m1_capture_replay_is_bit_exact() {
    let Some(ep) = gpu() else { return };
    let (k, n, block_size) = (1003usize, 37usize, 32usize);
    let blocks = k.div_ceil(block_size);
    let blob_size = block_size / 2;
    let mut state = 0xb63a_0975_4c21_d8efu64;
    let initial: Vec<f32> = (0..k)
        .map(|_| (random_u32(&mut state) as f32 / u32::MAX as f32 - 0.5) * 4.0)
        .collect();
    let mutated: Vec<f32> = (0..k)
        .map(|_| (random_u32(&mut state) as f32 / u32::MAX as f32 - 0.5) * 6.0)
        .collect();
    let packed: Vec<u8> = (0..n * blocks * blob_size)
        .map(|_| random_u32(&mut state) as u8)
        .collect();
    let scales: Vec<f32> = (0..n * blocks)
        .map(|_| 0.002 + (random_u32(&mut state) as f32 / u32::MAX as f32) * 0.08)
        .collect();

    let mut graph = Graph::new();
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let a = graph.create_named_value("A", DataType::Float32, static_shape([1, k]));
    let b = graph.create_named_value("B", DataType::Uint8, static_shape([n, blocks, blob_size]));
    let scales_value =
        graph.create_named_value("scales", DataType::Float32, static_shape([n, blocks]));
    for value in [a, b, scales_value] {
        graph.add_input(value);
    }
    let output = graph.create_named_value("Y", DataType::Float32, static_shape([1, n]));
    let mut node = Node::new(
        NodeId(0),
        "MatMulNBits",
        vec![Some(a), Some(b), Some(scales_value)],
        vec![output],
    );
    node.domain = "com.microsoft".into();
    node.attributes.insert("K".into(), Attribute::Int(k as i64));
    node.attributes.insert("N".into(), Attribute::Int(n as i64));
    node.attributes.insert("bits".into(), Attribute::Int(4));
    node.attributes
        .insert("block_size".into(), Attribute::Int(block_size as i64));
    node.attributes
        .insert("accuracy_level".into(), Attribute::Int(4));
    let node = graph.insert_node(node);
    graph.add_output(output);
    let model = Model::new(&graph);
    let kernel = ep.get_kernel(model.graph.node(node), &[], 1).unwrap();
    assert!(!kernel.cuda_graph_compatible());

    let runtime = ep.runtime();
    let device = ep.device_id();
    let inputs = [
        tensor(DataType::Float32, &[1, k], &initial),
        tensor(DataType::Uint8, &[n, blocks, blob_size], &packed),
        tensor(DataType::Float32, &[n, blocks], &scales),
    ];
    let mut input_buffers = Vec::<DeviceBuffer>::new();
    for input in &inputs {
        let buffer = ep.allocate(input.bytes.len(), 256).unwrap();
        // SAFETY: allocation size equals the source byte length.
        unsafe { runtime.htod(&input.bytes, cuptr(buffer.as_ptr())).unwrap() };
        input_buffers.push(buffer);
    }
    let input_strides: Vec<_> = inputs
        .iter()
        .map(|input| compute_contiguous_strides(&input.shape))
        .collect();
    let input_views: Vec<_> = inputs
        .iter()
        .zip(&input_buffers)
        .zip(&input_strides)
        .map(|((input, buffer), strides)| {
            TensorView::new(
                DevicePtr(buffer.as_ptr()),
                input.dtype,
                &input.shape,
                strides,
                device,
            )
        })
        .collect();
    let mut output_buffer = ep.allocate(n * 4, 256).unwrap();
    let output_shape = [1, n];
    let output_strides = compute_contiguous_strides(&output_shape);
    macro_rules! execute {
        () => {{
            let output_view = TensorMut::new(
                DevicePtrMut(output_buffer.as_mut_ptr()),
                DataType::Float32,
                &output_shape,
                &output_strides,
                device,
            );
            kernel.execute(&input_views, &mut [output_view]).unwrap();
        }};
    }

    execute!();
    assert!(kernel.cuda_graph_compatible());
    let mutated_bytes = typed_bytes(&mutated);
    // SAFETY: the first input allocation exactly covers the activation bytes.
    unsafe {
        runtime
            .htod(&mutated_bytes, cuptr(input_buffers[0].as_ptr()))
            .unwrap()
    };
    execute!();
    let mut eager = vec![0u8; n * 4];
    // SAFETY: the output allocation exactly covers `eager`.
    unsafe {
        runtime
            .dtoh(&mut eager, cuptr(output_buffer.as_ptr()))
            .unwrap()
    };

    let initial_bytes = typed_bytes(&initial);
    // SAFETY: the first input allocation exactly covers the activation bytes.
    unsafe {
        runtime
            .htod(&initial_bytes, cuptr(input_buffers[0].as_ptr()))
            .unwrap()
    };
    let allocation_counts = runtime.allocation_counts();
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    execute!();
    runtime.end_graph_capture().unwrap();
    assert!(runtime.has_graph_executable().unwrap());

    // SAFETY: the first input allocation exactly covers the activation bytes.
    unsafe {
        runtime
            .htod(&mutated_bytes, cuptr(input_buffers[0].as_ptr()))
            .unwrap()
    };
    runtime.replay_graph().unwrap();
    let mut replayed = vec![0u8; n * 4];
    // SAFETY: the output allocation exactly covers `replayed`.
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
}

#[test]
fn matmul_nbits_gpu_block32_decode_symmetric_non_multiple_k() {
    let Some(ep) = gpu() else { return };
    let (m, k, n, block_size) = (1, 45, 7, 32);
    let activations: Vec<f32> = (0..m * k)
        .map(|index| ((index * 17 % 29) as f32 - 14.0) / 11.0)
        .collect();
    let weights: Vec<f32> = (0..n * k)
        .map(|index| ((index * 13 % 31) as f32 - 15.0) / 9.0)
        .collect();
    let (packed, scales, zero_points) = quantize(&weights, n, k, block_size, false);
    let expected = independent_reference(&activations, &packed, &scales, None, m, k, n, block_size);
    let actual = match run_case(
        &ep,
        &[m, k],
        &activations,
        &packed,
        &scales,
        zero_points.as_deref(),
        k,
        n,
        block_size,
        0,
    ) {
        Err(error) if format!("{error}").contains("CUDA_ERROR_UNSUPPORTED_PTX_VERSION") => {
            eprintln!("skip: NVRTC PTX is newer than the installed CUDA driver ({error})");
            return;
        }
        result => result.unwrap(),
    };
    assert_close(&actual, &expected);
    eprintln!("verified real GPU block32 M=1 numerics against independent reference");
}

#[test]
fn matmul_nbits_gpu_block128_batched_asymmetric_non_multiple_k() {
    let Some(ep) = gpu() else { return };
    let (m, k, n, block_size) = (6, 173, 5, 128);
    let activations: Vec<f32> = (0..m * k)
        .map(|index| ((index * 7 % 37) as f32 - 11.0) / 13.0)
        .collect();
    let weights: Vec<f32> = (0..n * k)
        .map(|index| ((index * 19 % 43) as f32 - 9.0) / 10.0)
        .collect();
    let (packed, scales, zero_points) = quantize(&weights, n, k, block_size, true);
    let expected = independent_reference(
        &activations,
        &packed,
        &scales,
        zero_points.as_deref(),
        m,
        k,
        n,
        block_size,
    );
    let actual = match run_case(
        &ep,
        &[2, 3, k],
        &activations,
        &packed,
        &scales,
        zero_points.as_deref(),
        k,
        n,
        block_size,
        0,
    ) {
        Err(error) if format!("{error}").contains("CUDA_ERROR_UNSUPPORTED_PTX_VERSION") => {
            eprintln!("skip: NVRTC PTX is newer than the installed CUDA driver ({error})");
            return;
        }
        result => result.unwrap(),
    };
    assert_close(&actual, &expected);
    eprintln!("verified real GPU block128 M>1 numerics against independent reference");
}

#[test]
fn matmul_nbits_gpu_accuracy4_block32_decode_matches_quantized_reference() {
    let Some(ep) = gpu() else { return };
    let (m, k, n, block_size) = (1, 77, 19, 32);
    let activations: Vec<f32> = (0..k)
        .map(|index| ((index * 23 % 53) as f32 - 26.0) / 17.0)
        .collect();
    let weights: Vec<f32> = (0..n * k)
        .map(|index| ((index * 19 % 47) as f32 - 23.0) / 12.0)
        .collect();
    let (packed, scales, zero_points) = quantize(&weights, n, k, block_size, false);
    let expected = accuracy4_reference(&activations, &packed, &scales, k, n, block_size);
    let actual = run_case(
        &ep,
        &[m, k],
        &activations,
        &packed,
        &scales,
        zero_points.as_deref(),
        k,
        n,
        block_size,
        4,
    )
    .unwrap();
    assert_close(&actual, &expected);
    eprintln!("verified accuracy_level=4 packed-int4 CUDA GEMV semantics");
}

#[test]
fn matmul_nbits_gpu_accuracy4_offending_decode_shape_stays_within_cpu_vnni_tolerance() {
    let Some(ep) = gpu() else { return };
    let (m, k, n, block_size) = (1usize, 4864usize, 896usize, 32usize);
    let blocks = k.div_ceil(block_size);
    let blob_size = block_size / 2;
    let mut state = 0xd18a_46ce_3b79_205fu64;
    let activations: Vec<f32> = (0..k)
        .map(|_| (random_u32(&mut state) as f32 / u32::MAX as f32 - 0.5) * 6.0)
        .collect();
    let packed: Vec<u8> = (0..n * blocks * blob_size)
        .map(|_| random_u32(&mut state) as u8)
        .collect();
    let scales: Vec<f32> = (0..n * blocks)
        .map(|_| 0.001 + (random_u32(&mut state) as f32 / u32::MAX as f32) * 0.09)
        .collect();

    let expected = run_cpu_case(&[m, k], &activations, &packed, &scales, k, n, block_size, 4);
    let actual = run_case(
        &ep,
        &[m, k],
        &activations,
        &packed,
        &scales,
        None,
        k,
        n,
        block_size,
        4,
    )
    .unwrap();

    assert_close(&actual, &expected);
    let (max_abs_diff, max_ulp_diff) = error_metrics(&actual, &expected);
    assert!(
        max_abs_diff <= 3e-5,
        "offending K=4864,N=896 decode shape exceeded its numeric bound: {max_abs_diff:e}"
    );
    eprintln!(
        "accuracy_level=4 K=4864,N=896 CPU/CUDA max_abs_diff={max_abs_diff:e} max_ulp_diff={max_ulp_diff}"
    );
}

#[test]
fn matmul_nbits_gpu_accuracy4_real_decode_widths_match_current_algorithm() {
    let Some(ep) = gpu() else { return };
    let mut state = 0x2c91_a560_18f4_7b3du64;
    for (k, n) in [
        (4864usize, 896usize),
        (896, 1152),
        (896, 4864),
        (896, 151_936),
    ] {
        let blocks = k.div_ceil(32);
        let blob_size = 16;
        let activations: Vec<f32> = (0..k)
            .map(|_| (random_u32(&mut state) as f32 / u32::MAX as f32 - 0.5) * 6.0)
            .collect();
        let packed: Vec<u8> = (0..n * blocks * blob_size)
            .map(|_| random_u32(&mut state) as u8)
            .collect();
        let scales: Vec<f32> = (0..n * blocks)
            .map(|_| 0.001 + (random_u32(&mut state) as f32 / u32::MAX as f32) * 0.09)
            .collect();
        let columns = [0, n / 7, n / 3, n / 2, n - 1];
        let expected =
            accuracy4_reference_columns(&activations, &packed, &scales, k, n, 32, &columns);
        let actual = run_case(
            &ep,
            &[1, k],
            &activations,
            &packed,
            &scales,
            None,
            k,
            n,
            32,
            4,
        )
        .unwrap();
        let sampled: Vec<f32> = columns.iter().map(|&column| actual[column]).collect();
        assert_close(&sampled, &expected);
        let max_abs = sampled
            .iter()
            .zip(&expected)
            .map(|(&actual, &expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        let max_rel = sampled
            .iter()
            .zip(&expected)
            .map(|(&actual, &expected)| (actual - expected).abs() / expected.abs().max(1e-3))
            .fold(0.0f32, f32::max);
        eprintln!(
            "accuracy_level=4 K={k},N={n} sampled current-algorithm parity \
             max_abs_diff={max_abs:e} max_rel_diff={max_rel:e}"
        );
    }
}

fn f16_reference_column(
    activations: &[f16],
    packed: &[u8],
    scales: &[f16],
    k: usize,
    column: usize,
) -> f32 {
    let blocks = k.div_ceil(32);
    let mut value = 0.0f32;
    for block in 0..blocks {
        let mut block_dot = 0.0f32;
        for within in 0..32 {
            let depth = block * 32 + within;
            if depth >= k {
                break;
            }
            let byte = packed[(column * blocks + block) * 16 + within / 2];
            let quantized = if within & 1 == 0 {
                byte & 15
            } else {
                byte >> 4
            };
            block_dot += (i32::from(quantized) - 8) as f32 * activations[depth].to_f32();
        }
        value += block_dot * scales[column * blocks + block].to_f32();
    }
    value
}

fn f16_int8_reference_column(
    activations: &[f16],
    packed: &[u8],
    scales: &[f16],
    zero_points: &[u8],
    k: usize,
    column: usize,
) -> f32 {
    let blocks = k.div_ceil(32);
    let mut value = 0.0f32;
    for block in 0..blocks {
        for within in 0..32 {
            let depth = block * 32 + within;
            if depth >= k {
                break;
            }
            let quantized = packed[(column * blocks + block) * 32 + within];
            let zero_point = zero_points[column * blocks + block];
            value += activations[depth].to_f32()
                * (f32::from(quantized) - f32::from(zero_point))
                * scales[column * blocks + block].to_f32();
        }
    }
    value
}

#[test]
fn matmul_nbits_gpu_int8_fp16_block32_explicit_zero_points_match_reference() {
    let Some(ep) = gpu() else { return };
    let (k, n) = (77usize, 73usize);
    let blocks = k.div_ceil(32);
    let activations: Vec<f16> = (0..k)
        .map(|index| f16::from_f32(((index * 29 % 257) as f32 - 128.0) / 97.0))
        .collect();
    let packed: Vec<u8> = (0..n * blocks * 32)
        .map(|index| ((index * 37 + index / 11 + 19) & 255) as u8)
        .collect();
    let scales: Vec<f16> = (0..n * blocks)
        .map(|index| f16::from_f32(0.001 + (index * 17 % 31) as f32 * 0.0002))
        .collect();
    let zero_points: Vec<u8> = (0..n * blocks)
        .map(|index| ((index * 13 + 97) & 255) as u8)
        .collect();
    let actual = run_f16_case_with_bits(
        &ep,
        &activations,
        &packed,
        &scales,
        Some(&zero_points),
        k,
        n,
        8,
    )
    .unwrap();
    for (column, got) in actual.iter().enumerate() {
        let expected =
            f16_int8_reference_column(&activations, &packed, &scales, &zero_points, k, column);
        let expected = f16::from_f32(expected).to_f32();
        let tolerance = 0.02f32.max(expected.abs() * 2e-3);
        assert!(
            (got.to_f32() - expected).abs() <= tolerance,
            "column {column}: got={} expected={expected} tolerance={tolerance}",
            got.to_f32()
        );
    }
}

#[test]
fn matmul_nbits_gpu_int8_fp16_block32_default_zero_point_matches_reference() {
    let Some(ep) = gpu() else { return };
    let (k, n) = (77usize, 73usize);
    let blocks = k.div_ceil(32);
    let activations: Vec<f16> = (0..k)
        .map(|index| f16::from_f32(((index * 29 % 257) as f32 - 128.0) / 97.0))
        .collect();
    let packed: Vec<u8> = (0..n * blocks * 32)
        .map(|index| ((index * 37 + index / 11 + 19) & 255) as u8)
        .collect();
    let scales: Vec<f16> = (0..n * blocks)
        .map(|index| f16::from_f32(0.001 + (index * 17 % 31) as f32 * 0.0002))
        .collect();
    let zero_points = vec![128u8; n * blocks];
    let actual =
        run_f16_case_with_bits(&ep, &activations, &packed, &scales, None, k, n, 8).unwrap();
    for (column, got) in actual.iter().enumerate() {
        let expected =
            f16_int8_reference_column(&activations, &packed, &scales, &zero_points, k, column);
        let expected = f16::from_f32(expected).to_f32();
        let tolerance = 0.02f32.max(expected.abs() * 2e-3);
        assert!(
            (got.to_f32() - expected).abs() <= tolerance,
            "column {column}: got={} expected={expected} tolerance={tolerance}",
            got.to_f32()
        );
    }
}

#[test]
fn matmul_nbits_gpu_fp16_vectorized_block32_matches_reference() {
    let Some(ep) = gpu() else { return };
    let (k, n) = (4096usize, 73usize);
    let blocks = k / 32;
    let activations: Vec<f16> = (0..k)
        .map(|index| f16::from_f32(((index * 29 % 257) as f32 - 128.0) / 97.0))
        .collect();
    let packed: Vec<u8> = (0..n * blocks * 16)
        .map(|index| ((index * 37 + index / 11 + 19) & 255) as u8)
        .collect();
    let scales: Vec<f16> = (0..n * blocks)
        .map(|index| f16::from_f32(0.008 + (index * 17 % 31) as f32 * 0.0007))
        .collect();
    let actual = run_f16_case(&ep, &activations, &packed, &scales, k, n).unwrap();
    for (column, got) in actual.iter().enumerate() {
        let expected = f16_reference_column(&activations, &packed, &scales, k, column);
        let tolerance = 0.2f32.max(expected.abs() * 2e-3);
        assert!(
            (got.to_f32() - expected).abs() <= tolerance,
            "column {column}: got={} expected={expected} tolerance={tolerance}",
            got.to_f32()
        );
    }
}

#[test]
fn matmul_nbits_gpu_fp16_lm_head_width_is_deterministic() {
    let Some(ep) = gpu() else { return };
    let (k, n) = (64usize, 151_936usize);
    let blocks = k / 32;
    let activations: Vec<f16> = (0..k)
        .map(|index| f16::from_f32(((index * 13 % 67) as f32 - 33.0) / 29.0))
        .collect();
    let packed: Vec<u8> = (0..n * blocks * 16)
        .map(|index| ((index * 43 + index / 7 + 5) & 255) as u8)
        .collect();
    let scales: Vec<f16> = (0..n * blocks)
        .map(|index| f16::from_f32(0.01 + (index * 11 % 23) as f32 * 0.0009))
        .collect();
    let first = run_f16_case(&ep, &activations, &packed, &scales, k, n).unwrap();
    let second = run_f16_case(&ep, &activations, &packed, &scales, k, n).unwrap();
    assert_eq!(
        first
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        second
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    for column in [0, 1, 63, 895, 4863, n - 1] {
        let expected = f16_reference_column(&activations, &packed, &scales, k, column);
        let got = first[column].to_f32();
        let tolerance = 0.03f32.max(expected.abs() * 2e-3);
        assert!(
            (got - expected).abs() <= tolerance,
            "column {column}: got={got} expected={expected} tolerance={tolerance}"
        );
    }
}
