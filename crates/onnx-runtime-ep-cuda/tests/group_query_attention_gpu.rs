use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, KernelMatch, TensorMut, TensorView,
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
    // SAFETY: test inputs use primitive POD values with no padding.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
            .to_vec()
    }
}

fn f32_tensor(shape: &[usize], values: &[f32]) -> HostTensor {
    HostTensor {
        dtype: DataType::Float32,
        shape: shape.to_vec(),
        bytes: typed_bytes(values),
    }
}

fn i32_tensor(shape: &[usize], values: &[i32]) -> HostTensor {
    HostTensor {
        dtype: DataType::Int32,
        shape: shape.to_vec(),
        bytes: typed_bytes(values),
    }
}

fn i64_tensor(shape: &[usize], values: &[i64]) -> HostTensor {
    HostTensor {
        dtype: DataType::Int64,
        shape: shape.to_vec(),
        bytes: typed_bytes(values),
    }
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|x| f32::from_ne_bytes(x.try_into().unwrap()))
        .collect()
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

fn run(
    ep: &CudaExecutionProvider,
    attrs: &[(&str, Attribute)],
    inputs: &[Option<HostTensor>],
    output_shapes: &[Vec<usize>],
) -> onnx_runtime_ep_api::Result<Vec<Vec<f32>>> {
    let mut graph = Graph::new();
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let node_inputs = inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            input.as_ref().map(|tensor| {
                let value = graph.create_named_value(
                    format!("input_{index}"),
                    tensor.dtype,
                    static_shape(tensor.shape.clone()),
                );
                graph.add_input(value);
                value
            })
        })
        .collect();
    let node_outputs: Vec<_> = output_shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| {
            graph.create_named_value(
                format!("output_{index}"),
                DataType::Float32,
                static_shape(shape.clone()),
            )
        })
        .collect();
    let mut node = Node::new(
        NodeId(0),
        "GroupQueryAttention",
        node_inputs,
        node_outputs.clone(),
    );
    node.domain = "com.microsoft".into();
    for (name, value) in attrs {
        node.attributes.insert((*name).into(), value.clone());
    }
    let node_id = graph.insert_node(node);
    for output in node_outputs {
        graph.add_output(output);
    }
    let model = Model::new(&graph);
    let kernel = ep.get_kernel(model.graph.node(node_id), &[], 1)?;

    let runtime = ep.runtime();
    let device = ep.device_id();
    let mut input_buffers: Vec<Option<DeviceBuffer>> = Vec::with_capacity(inputs.len());
    for input in inputs {
        let buffer = if let Some(tensor) = input {
            let buffer = ep.allocate(tensor.bytes.len(), 256)?;
            // SAFETY: the allocation exactly covers the source byte slice.
            unsafe {
                runtime.htod(&tensor.bytes, cuptr(buffer.as_ptr()))?;
            }
            Some(buffer)
        } else {
            None
        };
        input_buffers.push(buffer);
    }
    let input_strides: Vec<_> = inputs
        .iter()
        .map(|input| {
            input
                .as_ref()
                .map(|tensor| compute_contiguous_strides(&tensor.shape))
        })
        .collect();
    let input_views: Vec<_> = inputs
        .iter()
        .zip(&input_buffers)
        .zip(&input_strides)
        .map(
            |((input, buffer), strides)| match (input, buffer, strides) {
                (Some(tensor), Some(buffer), Some(strides)) => TensorView::new(
                    DevicePtr(buffer.as_ptr()),
                    tensor.dtype,
                    &tensor.shape,
                    strides,
                    device,
                ),
                _ => TensorView::absent(DataType::Float32),
            },
        )
        .collect();

    let mut output_buffers = output_shapes
        .iter()
        .map(|shape| ep.allocate(shape.iter().product::<usize>() * 4, 256))
        .collect::<onnx_runtime_ep_api::Result<Vec<_>>>()?;
    let output_strides: Vec<_> = output_shapes
        .iter()
        .map(|shape| compute_contiguous_strides(shape))
        .collect();
    {
        let output_views: Vec<_> = output_buffers
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
        kernel.execute(
            &input_views,
            &mut output_views.into_iter().collect::<Vec<_>>(),
        )?;
    }

    let mut results = Vec::with_capacity(output_buffers.len());
    for (buffer, shape) in output_buffers.iter().zip(output_shapes) {
        let mut bytes = vec![0u8; shape.iter().product::<usize>() * 4];
        // SAFETY: the output allocation contains exactly this many bytes.
        unsafe {
            runtime.dtoh(&mut bytes, cuptr(buffer.as_ptr()))?;
        }
        results.push(bytes_to_f32(&bytes));
    }
    drop(input_views);
    for buffer in input_buffers.into_iter().flatten() {
        ep.deallocate(buffer)?;
    }
    for buffer in output_buffers {
        ep.deallocate(buffer)?;
    }
    Ok(results)
}

fn run_available(
    ep: &CudaExecutionProvider,
    attrs: &[(&str, Attribute)],
    inputs: &[Option<HostTensor>],
    output_shapes: &[Vec<usize>],
) -> onnx_runtime_ep_api::Result<Vec<Vec<f32>>> {
    match run(ep, attrs, inputs, output_shapes) {
        Err(error) if format!("{error}").contains("CUDA_ERROR_UNSUPPORTED_PTX_VERSION") => {
            eprintln!("skip: NVRTC PTX is newer than the installed CUDA driver ({error})");
            Ok(Vec::new())
        }
        result => result,
    }
}

fn upload(
    ep: &CudaExecutionProvider,
    tensor: &HostTensor,
) -> onnx_runtime_ep_api::Result<DeviceBuffer> {
    let buffer = ep.allocate(tensor.bytes.len(), 256)?;
    // SAFETY: the allocation exactly covers the source byte slice.
    unsafe {
        ep.runtime().htod(&tensor.bytes, cuptr(buffer.as_ptr()))?;
    }
    Ok(buffer)
}

fn run_packed_decode_in_place(
    ep: &CudaExecutionProvider,
    packed: &[f32],
    cache_k: &mut DeviceBuffer,
    cache_v: &mut DeviceBuffer,
    seqlen: i32,
    total: i32,
    cos: &[f32],
    sin: &[f32],
    position: i64,
) -> onnx_runtime_ep_api::Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    const NUM_HEADS: usize = 14;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 64;
    const CAPACITY: usize = 5;
    const PACKED_WIDTH: usize = (NUM_HEADS + 2 * KV_HEADS) * HEAD_DIM;

    let packed = f32_tensor(&[1, 1, PACKED_WIDTH], packed);
    let seqlens = i32_tensor(&[1], &[seqlen]);
    let total = i32_tensor(&[], &[total]);
    let cos = f32_tensor(&[CAPACITY, HEAD_DIM / 2], cos);
    let sin = f32_tensor(&[CAPACITY, HEAD_DIM / 2], sin);
    let position = i64_tensor(&[1, 1], &[position]);
    let transient = [&packed, &seqlens, &total, &cos, &sin, &position]
        .into_iter()
        .map(|tensor| upload(ep, tensor))
        .collect::<onnx_runtime_ep_api::Result<Vec<_>>>()?;

    let cache_shape = vec![1, KV_HEADS, CAPACITY, HEAD_DIM];
    let input_specs = [
        Some((DataType::Float32, vec![1, 1, PACKED_WIDTH])),
        None,
        None,
        Some((DataType::Float32, cache_shape.clone())),
        Some((DataType::Float32, cache_shape.clone())),
        Some((DataType::Int32, vec![1])),
        Some((DataType::Int32, vec![])),
        Some((DataType::Float32, vec![CAPACITY, HEAD_DIM / 2])),
        Some((DataType::Float32, vec![CAPACITY, HEAD_DIM / 2])),
        Some((DataType::Int64, vec![1, 1])),
    ];
    let mut graph = Graph::new();
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let node_inputs = input_specs
        .iter()
        .enumerate()
        .map(|(index, spec)| {
            spec.as_ref().map(|(dtype, shape)| {
                let value = graph.create_named_value(
                    format!("input_{index}"),
                    *dtype,
                    static_shape(shape.clone()),
                );
                graph.add_input(value);
                value
            })
        })
        .collect();
    let output_shapes = [
        vec![1, 1, NUM_HEADS * HEAD_DIM],
        cache_shape.clone(),
        cache_shape.clone(),
    ];
    let node_outputs: Vec<_> = output_shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| {
            graph.create_named_value(
                format!("output_{index}"),
                DataType::Float32,
                static_shape(shape.clone()),
            )
        })
        .collect();
    let mut node = Node::new(
        NodeId(0),
        "GroupQueryAttention",
        node_inputs,
        node_outputs.clone(),
    );
    node.domain = "com.microsoft".into();
    node.attributes
        .insert("num_heads".into(), Attribute::Int(NUM_HEADS as i64));
    node.attributes
        .insert("kv_num_heads".into(), Attribute::Int(KV_HEADS as i64));
    node.attributes
        .insert("do_rotary".into(), Attribute::Int(1));
    let node_id = graph.insert_node(node);
    for output in node_outputs {
        graph.add_output(output);
    }
    let model = Model::new(&graph);
    let kernel = ep.get_kernel(model.graph.node(node_id), &[], 1)?;

    let device = ep.device_id();
    let cache_strides = compute_contiguous_strides(&cache_shape);
    let transient_shapes = [
        &[1, 1, PACKED_WIDTH][..],
        &[1][..],
        &[][..],
        &[CAPACITY, HEAD_DIM / 2][..],
        &[CAPACITY, HEAD_DIM / 2][..],
        &[1, 1][..],
    ];
    let transient_dtypes = [
        DataType::Float32,
        DataType::Int32,
        DataType::Int32,
        DataType::Float32,
        DataType::Float32,
        DataType::Int64,
    ];
    let transient_strides: Vec<_> = transient_shapes
        .iter()
        .map(|shape| compute_contiguous_strides(shape))
        .collect();
    let transient_views: Vec<_> = transient
        .iter()
        .zip(transient_shapes)
        .zip(transient_dtypes)
        .zip(&transient_strides)
        .map(|(((buffer, shape), dtype), strides)| {
            TensorView::new(DevicePtr(buffer.as_ptr()), dtype, shape, strides, device)
        })
        .collect();
    let mut transient_views = transient_views.into_iter();
    let inputs = vec![
        transient_views.next().unwrap(),
        TensorView::absent(DataType::Float32),
        TensorView::absent(DataType::Float32),
        TensorView::new(
            DevicePtr(cache_k.as_ptr()),
            DataType::Float32,
            &cache_shape,
            &cache_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(cache_v.as_ptr()),
            DataType::Float32,
            &cache_shape,
            &cache_strides,
            device,
        ),
        transient_views.next().unwrap(),
        transient_views.next().unwrap(),
        transient_views.next().unwrap(),
        transient_views.next().unwrap(),
        transient_views.next().unwrap(),
    ];

    let output_shape = &output_shapes[0];
    let output_strides = compute_contiguous_strides(output_shape);
    let mut output = ep.allocate(output_shape.iter().product::<usize>() * 4, 256)?;
    kernel.execute(
        &inputs,
        &mut [
            TensorMut::new(
                DevicePtrMut(output.as_mut_ptr()),
                DataType::Float32,
                output_shape,
                &output_strides,
                device,
            ),
            TensorMut::new(
                DevicePtrMut(cache_k.as_mut_ptr()),
                DataType::Float32,
                &cache_shape,
                &cache_strides,
                device,
            ),
            TensorMut::new(
                DevicePtrMut(cache_v.as_mut_ptr()),
                DataType::Float32,
                &cache_shape,
                &cache_strides,
                device,
            ),
        ],
    )?;

    let mut output_bytes = vec![0u8; output_shape.iter().product::<usize>() * 4];
    let mut key_bytes = vec![0u8; cache_shape.iter().product::<usize>() * 4];
    let mut value_bytes = vec![0u8; key_bytes.len()];
    // SAFETY: destination slices exactly match their source allocations.
    unsafe {
        ep.runtime()
            .dtoh(&mut output_bytes, cuptr(output.as_ptr()))?;
        ep.runtime().dtoh(&mut key_bytes, cuptr(cache_k.as_ptr()))?;
        ep.runtime()
            .dtoh(&mut value_bytes, cuptr(cache_v.as_ptr()))?;
    }
    ep.deallocate(output)?;
    for buffer in transient {
        ep.deallocate(buffer)?;
    }
    Ok((
        bytes_to_f32(&output_bytes),
        bytes_to_f32(&key_bytes),
        bytes_to_f32(&value_bytes),
    ))
}

fn base_inputs(
    q_shape: &[usize],
    q: &[f32],
    k_shape: &[usize],
    k: &[f32],
    v: &[f32],
    past_k: Option<(&[usize], &[f32])>,
    past_v: Option<(&[usize], &[f32])>,
    seqlens: &[i32],
    total: i32,
) -> Vec<Option<HostTensor>> {
    vec![
        Some(f32_tensor(q_shape, q)),
        Some(f32_tensor(k_shape, k)),
        Some(f32_tensor(k_shape, v)),
        past_k.map(|(shape, data)| f32_tensor(shape, data)),
        past_v.map(|(shape, data)| f32_tensor(shape, data)),
        Some(i32_tensor(&[seqlens.len()], seqlens)),
        Some(i32_tensor(&[], &[total])),
    ]
}

fn reference(
    q: &[f32],
    k_bnsh: &[f32],
    v_bnsh: &[f32],
    q_seq: usize,
    capacity: usize,
    past: usize,
    scale: f32,
    softcap: f32,
    window: Option<usize>,
) -> Vec<f32> {
    let (heads, kv_heads, dim) = (4, 2, 2);
    let mut output = vec![0.0; q_seq * heads * dim];
    for s in 0..q_seq {
        for h in 0..heads {
            let kv = h / (heads / kv_heads);
            let end = past + s;
            let start = window.map_or(0, |w| (end + 1).saturating_sub(w));
            let mut scores = Vec::new();
            for key_s in start..=end {
                let mut score = (0..dim)
                    .map(|d| {
                        q[(s * heads + h) * dim + d] * k_bnsh[(kv * capacity + key_s) * dim + d]
                    })
                    .sum::<f32>()
                    * scale;
                if softcap != 0.0 {
                    score = softcap * (score / softcap).tanh();
                }
                scores.push(score);
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = scores
                .iter_mut()
                .map(|x| {
                    *x = (*x - max).exp();
                    *x
                })
                .sum();
            for d in 0..dim {
                output[(s * heads + h) * dim + d] = scores
                    .iter()
                    .enumerate()
                    .map(|(offset, probability)| {
                        probability / sum * v_bnsh[(kv * capacity + start + offset) * dim + d]
                    })
                    .sum();
            }
        }
    }
    output
}

fn close(got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len());
    for (index, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        assert!(
            (got - expected).abs() < 1e-3,
            "{index}: {got} != {expected}"
        );
    }
}

fn rotate_target(
    data: &[f32],
    heads: usize,
    position: usize,
    cos: &[f32],
    sin: &[f32],
) -> Vec<f32> {
    const HEAD_DIM: usize = 64;
    let half = HEAD_DIM / 2;
    let mut output = data.to_vec();
    for head in 0..heads {
        let base = head * HEAD_DIM;
        for k in 0..half {
            let x0 = data[base + k];
            let x1 = data[base + k + half];
            let cache = position * half + k;
            output[base + k] = cos[cache] * x0 - sin[cache] * x1;
            output[base + k + half] = sin[cache] * x0 + cos[cache] * x1;
        }
    }
    output
}

fn target_decode_reference(
    query: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    valid_length: usize,
    capacity: usize,
) -> Vec<f32> {
    const NUM_HEADS: usize = 14;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 64;
    let group = NUM_HEADS / KV_HEADS;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut output = vec![0.0; NUM_HEADS * HEAD_DIM];
    for head in 0..NUM_HEADS {
        let kv_head = head / group;
        let mut scores = Vec::with_capacity(valid_length);
        for position in 0..valid_length {
            let score = (0..HEAD_DIM)
                .map(|dim| {
                    query[head * HEAD_DIM + dim]
                        * key_cache[(kv_head * capacity + position) * HEAD_DIM + dim]
                })
                .sum::<f32>()
                * scale;
            scores.push(score);
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = scores
            .iter_mut()
            .map(|score| {
                *score = (*score - max).exp();
                *score
            })
            .sum();
        for dim in 0..HEAD_DIM {
            output[head * HEAD_DIM + dim] = scores
                .iter()
                .enumerate()
                .map(|(position, probability)| {
                    probability / sum
                        * value_cache[(kv_head * capacity + position) * HEAD_DIM + dim]
                })
                .sum();
        }
    }
    output
}

fn attrs<'a>(extra: &'a [(&'a str, Attribute)]) -> Vec<(&'a str, Attribute)> {
    let mut attrs = vec![
        ("num_heads", Attribute::Int(4)),
        ("kv_num_heads", Attribute::Int(2)),
    ];
    attrs.extend_from_slice(extra);
    attrs
}

#[test]
fn gqa_gpu_head_sharing_matches_manual_repeat_kv_reference() {
    let Some(ep) = gpu() else { return };
    let q = [
        1., 0., 1., 0., 0., 1., 0., 1., 0., 1., 0., 1., 1., 0., 1., 0.,
    ];
    let k_bsh = [1., 0., 0., 1., 0., 1., 1., 0.];
    let v_bsh = [1., 2., 10., 20., 3., 4., 30., 40.];
    let k_bnsh = [1., 0., 0., 1., 0., 1., 1., 0.];
    let v_bnsh = [1., 2., 3., 4., 10., 20., 30., 40.];
    let outputs = run_available(
        &ep,
        &attrs(&[]),
        &base_inputs(
            &[1, 2, 8],
            &q,
            &[1, 2, 4],
            &k_bsh,
            &v_bsh,
            None,
            None,
            &[1],
            2,
        ),
        &[vec![1, 2, 8], vec![1, 2, 2, 2], vec![1, 2, 2, 2]],
    )
    .unwrap();
    if outputs.is_empty() {
        return;
    }
    close(
        &outputs[0],
        &reference(
            &q,
            &k_bnsh,
            &v_bnsh,
            2,
            2,
            0,
            1.0 / 2.0_f32.sqrt(),
            0.0,
            None,
        ),
    );
    close(&outputs[1], &k_bnsh);
    close(&outputs[2], &v_bnsh);
}

#[test]
fn gqa_gpu_decode_preserves_fixed_cache_capacity_and_write_offset() {
    let Some(ep) = gpu() else { return };
    let q = [1., 0., 1., 0., 0., 1., 0., 1.];
    let past_k = [
        1., 0., 0., 1., 91., 92., 93., 94., 95., 96., 10., 0., 0., 10., 81., 82., 83., 84., 85.,
        86.,
    ];
    let past_v = [
        1., 2., 3., 4., 71., 72., 73., 74., 75., 76., 10., 20., 30., 40., 61., 62., 63., 64., 65.,
        66.,
    ];
    let current_k = [1., 1., 10., 10.];
    let current_v = [5., 6., 50., 60.];
    let expected_k = [
        1., 0., 0., 1., 1., 1., 0., 0., 0., 0., 10., 0., 0., 10., 10., 10., 0., 0., 0., 0.,
    ];
    let expected_v = [
        1., 2., 3., 4., 5., 6., 0., 0., 0., 0., 10., 20., 30., 40., 50., 60., 0., 0., 0., 0.,
    ];
    let outputs = run_available(
        &ep,
        &attrs(&[]),
        &base_inputs(
            &[1, 1, 8],
            &q,
            &[1, 1, 4],
            &current_k,
            &current_v,
            Some((&[1, 2, 5, 2], &past_k)),
            Some((&[1, 2, 5, 2], &past_v)),
            &[2],
            3,
        ),
        &[vec![1, 1, 8], vec![1, 2, 5, 2], vec![1, 2, 5, 2]],
    )
    .unwrap();
    if outputs.is_empty() {
        return;
    }
    close(&outputs[1], &expected_k);
    close(&outputs[2], &expected_v);
    close(
        &outputs[0],
        &reference(
            &q,
            &expected_k,
            &expected_v,
            1,
            5,
            2,
            1.0 / 2.0_f32.sqrt(),
            0.0,
            None,
        ),
    );
}

#[test]
fn gqa_gpu_rope_explicit_positions_rotate_query_and_key() {
    let Some(ep) = gpu() else { return };
    let q = [
        1., 2., 2., -1., -1., 3., 4., 2., 3., -2., 1., 4., -3., 1., 2., 5.,
    ];
    let k = [2., 1., -1., 3., 4., -2., 2., 5.];
    let v = [1., 2., 10., 20., 3., 4., 30., 40.];
    let angles = [0.0_f32, 0.2, 0.7, 1.1, 1.6];
    let cos: Vec<f32> = angles.iter().map(|x| x.cos()).collect();
    let sin: Vec<f32> = angles.iter().map(|x| x.sin()).collect();
    let positions = [2_i64, 4];
    let mut inputs = base_inputs(&[1, 2, 8], &q, &[1, 2, 4], &k, &v, None, None, &[1], 2);
    inputs.push(Some(f32_tensor(&[5, 1], &cos)));
    inputs.push(Some(f32_tensor(&[5, 1], &sin)));
    inputs.push(Some(i64_tensor(&[1, 2], &positions)));
    let outputs = run_available(
        &ep,
        &attrs(&[("do_rotary", Attribute::Int(1))]),
        &inputs,
        &[vec![1, 2, 8], vec![1, 2, 2, 2]],
    )
    .unwrap();
    if outputs.is_empty() {
        return;
    }
    let rotate = |data: &[f32], heads: usize| {
        let mut out = data.to_vec();
        for s in 0..2 {
            for h in 0..heads {
                let base = (s * heads + h) * 2;
                let (x0, x1) = (data[base], data[base + 1]);
                let p = positions[s] as usize;
                out[base] = cos[p] * x0 - sin[p] * x1;
                out[base + 1] = sin[p] * x0 + cos[p] * x1;
            }
        }
        out
    };
    let q_rot = rotate(&q, 4);
    let k_rot_bsh = rotate(&k, 2);
    let k_rot_bnsh = [
        k_rot_bsh[0],
        k_rot_bsh[1],
        k_rot_bsh[4],
        k_rot_bsh[5],
        k_rot_bsh[2],
        k_rot_bsh[3],
        k_rot_bsh[6],
        k_rot_bsh[7],
    ];
    let v_bnsh = [1., 2., 3., 4., 10., 20., 30., 40.];
    close(&outputs[1], &k_rot_bnsh);
    close(
        &outputs[0],
        &reference(
            &q_rot,
            &k_rot_bnsh,
            &v_bnsh,
            2,
            2,
            0,
            1.0 / 2.0_f32.sqrt(),
            0.0,
            None,
        ),
    );
}

#[test]
fn gqa_gpu_zero_scale_softcap_and_sliding_window_match_reference() {
    let Some(ep) = gpu() else { return };
    let q = [2., 0., 2., 0., 2., 0., 2., 0.];
    let past_k = [1., 0., 4., 0., 10., 0., 40., 0.];
    let past_v = [1., 0., 3., 0., 10., 0., 30., 0.];
    let current_k = [8., 0., 80., 0.];
    let current_v = [9., 0., 90., 0.];
    let expected_k = [1., 0., 4., 0., 8., 0., 10., 0., 40., 0., 80., 0.];
    let expected_v = [1., 0., 3., 0., 9., 0., 10., 0., 30., 0., 90., 0.];
    let outputs = run_available(
        &ep,
        &attrs(&[
            ("scale", Attribute::Float(0.0)),
            ("softcap", Attribute::Float(1.5)),
            ("local_window_size", Attribute::Int(2)),
        ]),
        &base_inputs(
            &[1, 1, 8],
            &q,
            &[1, 1, 4],
            &current_k,
            &current_v,
            Some((&[1, 2, 2, 2], &past_k)),
            Some((&[1, 2, 2, 2], &past_v)),
            &[2],
            3,
        ),
        &[vec![1, 1, 8]],
    )
    .unwrap();
    if outputs.is_empty() {
        return;
    }
    close(
        &outputs[0],
        &reference(
            &q,
            &expected_k,
            &expected_v,
            1,
            3,
            2,
            1.0 / 2.0_f32.sqrt(),
            1.5,
            Some(2),
        ),
    );
}

#[test]
fn gqa_gpu_packed_qkv_rope_decode_appends_in_place_across_steps() {
    const NUM_HEADS: usize = 14;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 64;
    const CAPACITY: usize = 5;
    let Some(ep) = gpu() else { return };

    let mut cos = Vec::with_capacity(CAPACITY * HEAD_DIM / 2);
    let mut sin = Vec::with_capacity(cos.capacity());
    for position in 0..CAPACITY {
        for dim in 0..HEAD_DIM / 2 {
            let angle = position as f32 * (dim + 1) as f32 * 0.01;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }

    let cache_len = KV_HEADS * CAPACITY * HEAD_DIM;
    let mut expected_k = vec![91.0; cache_len];
    let mut expected_v = vec![73.0; cache_len];
    for head in 0..KV_HEADS {
        for dim in 0..HEAD_DIM {
            expected_k[(head * CAPACITY) * HEAD_DIM + dim] =
                (head * HEAD_DIM + dim + 1) as f32 * 0.002;
            expected_v[(head * CAPACITY) * HEAD_DIM + dim] =
                (head * HEAD_DIM + dim + 1) as f32 * 0.003 - 0.2;
        }
    }
    let mut cache_k = ep.allocate(cache_len * 4, 256).unwrap();
    let mut cache_v = ep.allocate(cache_len * 4, 256).unwrap();
    // SAFETY: cache allocations exactly cover the source slices.
    unsafe {
        ep.runtime()
            .htod(&typed_bytes(&expected_k), cuptr(cache_k.as_ptr()))
            .unwrap();
        ep.runtime()
            .htod(&typed_bytes(&expected_v), cuptr(cache_v.as_ptr()))
            .unwrap();
    }

    for step in 1..=2 {
        let query: Vec<f32> = (0..NUM_HEADS * HEAD_DIM)
            .map(|index| ((index * 13 + step * 7) % 101) as f32 * 0.002 - 0.1)
            .collect();
        let key: Vec<f32> = (0..KV_HEADS * HEAD_DIM)
            .map(|index| ((index * 11 + step * 5) % 67) as f32 * 0.003 - 0.08)
            .collect();
        let value: Vec<f32> = (0..KV_HEADS * HEAD_DIM)
            .map(|index| ((index * 17 + step * 3) % 79) as f32 * 0.004 - 0.12)
            .collect();
        let mut packed = Vec::with_capacity(query.len() + key.len() + value.len());
        packed.extend_from_slice(&query);
        packed.extend_from_slice(&key);
        packed.extend_from_slice(&value);

        let rotated_query = rotate_target(&query, NUM_HEADS, step, &cos, &sin);
        let rotated_key = rotate_target(&key, KV_HEADS, step, &cos, &sin);
        for head in 0..KV_HEADS {
            for dim in 0..HEAD_DIM {
                let cache_index = (head * CAPACITY + step) * HEAD_DIM + dim;
                expected_k[cache_index] = rotated_key[head * HEAD_DIM + dim];
                expected_v[cache_index] = value[head * HEAD_DIM + dim];
            }
        }
        let expected_output =
            target_decode_reference(&rotated_query, &expected_k, &expected_v, step + 1, CAPACITY);
        let (output, got_k, got_v) = run_packed_decode_in_place(
            &ep,
            &packed,
            &mut cache_k,
            &mut cache_v,
            step as i32,
            (step + 1) as i32,
            &cos,
            &sin,
            step as i64,
        )
        .unwrap();
        close(&output, &expected_output);
        close(&got_k, &expected_k);
        close(&got_v, &expected_v);
    }

    ep.deallocate(cache_k).unwrap();
    ep.deallocate(cache_v).unwrap();
}

#[test]
fn gqa_gpu_rejected_features_return_clear_errors() {
    let Some(ep) = gpu() else { return };
    let mut registered = Node::new(NodeId(0), "GroupQueryAttention", vec![], vec![]);
    registered.domain = "com.microsoft".into();
    assert!(matches!(
        ep.supports_op(&registered, &[], &[]),
        KernelMatch::Supported { .. }
    ));

    let q = [0.0; 8];
    let kv = [0.0; 4];
    let base = base_inputs(&[1, 1, 8], &q, &[1, 1, 4], &kv, &kv, None, None, &[0], 1);
    for (name, attr) in [
        ("smooth_softmax", Attribute::Int(1)),
        ("kv_cache_bit_width", Attribute::Int(8)),
        ("qk_output", Attribute::Int(1)),
        ("k_quant_type", Attribute::String(b"INT8".to_vec())),
    ] {
        let error = run(&ep, &attrs(&[(name, attr)]), &base, &[vec![1, 1, 8]])
            .expect_err("feature must be rejected");
        assert!(format!("{error}").contains("not supported"));
    }
    for (index, feature) in [
        (10, "attention_bias"),
        (11, "head_sink"),
        (12, "quantized-cache k_scale"),
        (13, "quantized-cache v_scale"),
    ] {
        let mut feature_inputs = base.clone();
        while feature_inputs.len() <= index {
            feature_inputs.push(None);
        }
        feature_inputs[index] = Some(f32_tensor(&[1], &[0.0]));
        let error = run(&ep, &attrs(&[]), &feature_inputs, &[vec![1, 1, 8]])
            .expect_err("feature input must be rejected");
        assert!(format!("{error}").contains(feature));
    }

    let mut incomplete_inputs = base;
    incomplete_inputs[1] = None;
    let error = run(&ep, &attrs(&[]), &incomplete_inputs, &[vec![1, 1, 8]])
        .expect_err("partially packed QKV must be rejected");
    assert!(format!("{error}").contains("must both be present"));
}
