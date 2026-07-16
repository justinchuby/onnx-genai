use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
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
    let blocks = k.div_ceil(block_size);
    let blob_size = block_size / 2;
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
            static_shape([n, blocks.div_ceil(2)]),
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
    node.attributes.insert("bits".into(), Attribute::Int(4));
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

    let mut inputs = vec![
        tensor(DataType::Float32, a_shape, activations),
        tensor(DataType::Uint8, &[n, blocks, blob_size], packed),
        tensor(DataType::Float32, &[n, blocks], scales),
    ];
    if let Some(zero_points) = zero_points {
        inputs.push(tensor(
            DataType::Uint8,
            &[n, blocks.div_ceil(2)],
            zero_points,
        ));
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

fn random_u32(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 32) as u32
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
    eprintln!("verified packed-int4 CUDA GEMV against independent f32 dequant reference");
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
