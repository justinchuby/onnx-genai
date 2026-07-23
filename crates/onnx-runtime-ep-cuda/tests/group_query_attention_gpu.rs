use half::{bf16, f16};
use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, Kernel, KernelMatch, TensorMut,
    TensorView,
};
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ep_cuda::{
    gqa_capture_error_description, CudaExecutionProvider, GroupQueryAttentionBackend,
    GroupQueryAttentionKernel, GQA_CAPTURE_ERROR_PAST_CAPACITY, GQA_CAPTURE_ERROR_PAST_NEGATIVE,
    GQA_CAPTURE_ERROR_POSITION, GQA_CAPTURE_ERROR_QUERY_NEGATIVE, GQA_CAPTURE_ERROR_TOTAL_OVERFLOW,
};
use onnx_runtime_ir::{
    compute_contiguous_strides, static_shape, Attribute, DataType, Graph, Node, NodeId,
};
use onnx_runtime_loader::Model;
use std::time::Instant;

#[derive(Clone)]
struct HostTensor {
    dtype: DataType,
    shape: Vec<usize>,
    strides: Option<Vec<i64>>,
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
    float_tensor(DataType::Float32, shape, values)
}

fn float_tensor(dtype: DataType, shape: &[usize], values: &[f32]) -> HostTensor {
    let bytes = match dtype {
        DataType::Float32 => typed_bytes(values),
        DataType::Float16 => values
            .iter()
            .flat_map(|&value| f16::from_f32(value).to_bits().to_ne_bytes())
            .collect(),
        DataType::BFloat16 => values
            .iter()
            .flat_map(|&value| bf16::from_f32(value).to_bits().to_ne_bytes())
            .collect(),
        _ => unreachable!("floating GQA test tensor dtype"),
    };
    HostTensor {
        dtype,
        shape: shape.to_vec(),
        strides: None,
        bytes,
    }
}

fn i32_tensor(shape: &[usize], values: &[i32]) -> HostTensor {
    HostTensor {
        dtype: DataType::Int32,
        shape: shape.to_vec(),
        strides: None,
        bytes: typed_bytes(values),
    }
}

fn i64_tensor(shape: &[usize], values: &[i64]) -> HostTensor {
    HostTensor {
        dtype: DataType::Int64,
        shape: shape.to_vec(),
        strides: None,
        bytes: typed_bytes(values),
    }
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|x| f32::from_ne_bytes(x.try_into().unwrap()))
        .collect()
}

fn decode_float(bytes: &[u8], dtype: DataType) -> Vec<f32> {
    match dtype {
        DataType::Float32 => bytes_to_f32(bytes),
        DataType::Float16 | DataType::BFloat16 => bytes
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_ne_bytes([chunk[0], chunk[1]]);
                match dtype {
                    DataType::Float16 => f16::from_bits(bits).to_f32(),
                    DataType::BFloat16 => bf16::from_bits(bits).to_f32(),
                    _ => unreachable!(),
                }
            })
            .collect(),
        _ => unreachable!("floating GQA test output dtype"),
    }
}

fn quantize(values: &[f32], dtype: DataType) -> Vec<f32> {
    decode_float(&float_tensor(dtype, &[], values).bytes, dtype)
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
    run_with_backend(ep, attrs, inputs, output_shapes, None)
}

fn run_with_backend(
    ep: &CudaExecutionProvider,
    attrs: &[(&str, Attribute)],
    inputs: &[Option<HostTensor>],
    output_shapes: &[Vec<usize>],
    backend: Option<GroupQueryAttentionBackend>,
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
    let output_dtype = inputs[0]
        .as_ref()
        .map(|input| input.dtype)
        .unwrap_or(DataType::Float32);
    let element_size = match output_dtype {
        DataType::Float32 => 4,
        DataType::Float16 | DataType::BFloat16 => 2,
        _ => unreachable!("floating GQA output dtype"),
    };
    let node_outputs: Vec<_> = output_shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| {
            graph.create_named_value(
                format!("output_{index}"),
                output_dtype,
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
    let kernel: Box<dyn Kernel> = if let Some(backend) = backend {
        let int_attr = |name: &str, default: i64| {
            attrs
                .iter()
                .find(|(attr_name, _)| *attr_name == name)
                .and_then(|(_, value)| value.as_int())
                .unwrap_or(default)
        };
        let float_attr = |name: &str| {
            attrs
                .iter()
                .find(|(attr_name, _)| *attr_name == name)
                .and_then(|(_, value)| value.as_float())
        };
        Box::new(
            GroupQueryAttentionKernel::new(
                ep.runtime().clone(),
                usize::try_from(int_attr("num_heads", 0)).unwrap(),
                usize::try_from(int_attr("kv_num_heads", 0)).unwrap(),
                float_attr("scale"),
                int_attr("do_rotary", 0) != 0,
                int_attr("rotary_interleaved", 0) != 0,
                int_attr("local_window_size", -1),
                float_attr("softcap").unwrap_or(0.0),
            )?
            .with_backend(backend),
        )
    } else {
        ep.get_kernel(model.graph.node(node_id), &[], 1)?
    };

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
            input.as_ref().map(|tensor| {
                tensor
                    .strides
                    .clone()
                    .unwrap_or_else(|| compute_contiguous_strides(&tensor.shape))
            })
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
        .map(|shape| ep.allocate(shape.iter().product::<usize>() * element_size, 256))
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
                    output_dtype,
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
        let mut bytes = vec![0u8; shape.iter().product::<usize>() * element_size];
        // SAFETY: the output allocation contains exactly this many bytes.
        unsafe {
            runtime.dtoh(&mut bytes, cuptr(buffer.as_ptr()))?;
        }
        results.push(decode_float(&bytes, output_dtype));
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

fn run_available<T>(result: onnx_runtime_ep_api::Result<T>) -> onnx_runtime_ep_api::Result<T> {
    match result {
        Err(error) if format!("{error}").contains("CUDA_ERROR_UNSUPPORTED_PTX_VERSION") => {
            panic!("CUDA GPU tests must execute; unsupported PTX cannot be skipped: {error}")
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

fn overwrite(
    ep: &CudaExecutionProvider,
    buffer: &DeviceBuffer,
    tensor: &HostTensor,
) -> onnx_runtime_ep_api::Result<()> {
    // SAFETY: callers provide a persistent allocation large enough for `tensor`.
    unsafe { ep.runtime().htod(&tensor.bytes, cuptr(buffer.as_ptr())) }
}

fn read_bytes(
    ep: &CudaExecutionProvider,
    buffer: &DeviceBuffer,
    bytes: usize,
) -> onnx_runtime_ep_api::Result<Vec<u8>> {
    let mut host = vec![0_u8; bytes];
    // SAFETY: callers request no more bytes than the live allocation contains.
    unsafe { ep.runtime().dtoh(&mut host, cuptr(buffer.as_ptr())) }?;
    Ok(host)
}

#[allow(clippy::too_many_arguments)]
fn execute_in_place_packed_f32_gqa(
    kernel: &GroupQueryAttentionKernel,
    ep: &CudaExecutionProvider,
    packed_qkv: &DeviceBuffer,
    cache_k: &mut DeviceBuffer,
    cache_v: &mut DeviceBuffer,
    seqlens: &DeviceBuffer,
    total: &DeviceBuffer,
    cos: &DeviceBuffer,
    sin: &DeviceBuffer,
    positions: &DeviceBuffer,
    output: &mut DeviceBuffer,
    query_sequence_length: usize,
) -> onnx_runtime_ep_api::Result<()> {
    execute_in_place_packed_f32_gqa_with_seqlens_shape(
        kernel,
        ep,
        packed_qkv,
        cache_k,
        cache_v,
        seqlens,
        total,
        cos,
        sin,
        positions,
        output,
        query_sequence_length,
        &[1],
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_in_place_packed_f32_gqa_with_seqlens_shape(
    kernel: &GroupQueryAttentionKernel,
    ep: &CudaExecutionProvider,
    packed_qkv: &DeviceBuffer,
    cache_k: &mut DeviceBuffer,
    cache_v: &mut DeviceBuffer,
    seqlens: &DeviceBuffer,
    total: &DeviceBuffer,
    cos: &DeviceBuffer,
    sin: &DeviceBuffer,
    positions: &DeviceBuffer,
    output: &mut DeviceBuffer,
    query_sequence_length: usize,
    seqlens_shape: &[usize],
) -> onnx_runtime_ep_api::Result<()> {
    let device = ep.device_id();
    let packed_shape = [1, query_sequence_length, 16];
    let output_shape = [1, query_sequence_length, 8];
    let cache_shape = [1, 2, 5, 2];
    let total_shape = [1];
    let rotary_cache_shape = [5, 1];
    let positions_shape = [1, query_sequence_length];
    let packed_strides = compute_contiguous_strides(&packed_shape);
    let output_strides = compute_contiguous_strides(&output_shape);
    let cache_strides = compute_contiguous_strides(&cache_shape);
    let seqlens_strides = compute_contiguous_strides(seqlens_shape);
    let total_strides = compute_contiguous_strides(&total_shape);
    let rotary_cache_strides = compute_contiguous_strides(&rotary_cache_shape);
    let positions_strides = compute_contiguous_strides(&positions_shape);
    let cache_k_input = DevicePtr(cache_k.as_ptr());
    let cache_v_input = DevicePtr(cache_v.as_ptr());
    let cache_k_output = DevicePtrMut(cache_k.as_mut_ptr());
    let cache_v_output = DevicePtrMut(cache_v.as_mut_ptr());
    let inputs = [
        TensorView::new(
            DevicePtr(packed_qkv.as_ptr()),
            DataType::Float32,
            &packed_shape,
            &packed_strides,
            device,
        ),
        TensorView::absent(DataType::Float32),
        TensorView::absent(DataType::Float32),
        TensorView::new(
            cache_k_input,
            DataType::Float32,
            &cache_shape,
            &cache_strides,
            device,
        ),
        TensorView::new(
            cache_v_input,
            DataType::Float32,
            &cache_shape,
            &cache_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(seqlens.as_ptr()),
            DataType::Int32,
            seqlens_shape,
            &seqlens_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(total.as_ptr()),
            DataType::Int32,
            &total_shape,
            &total_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(cos.as_ptr()),
            DataType::Float32,
            &rotary_cache_shape,
            &rotary_cache_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(sin.as_ptr()),
            DataType::Float32,
            &rotary_cache_shape,
            &rotary_cache_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(positions.as_ptr()),
            DataType::Int64,
            &positions_shape,
            &positions_strides,
            device,
        ),
    ];
    let mut outputs = [
        TensorMut::new(
            DevicePtrMut(output.as_mut_ptr()),
            DataType::Float32,
            &output_shape,
            &output_strides,
            device,
        ),
        TensorMut::new(
            cache_k_output,
            DataType::Float32,
            &cache_shape,
            &cache_strides,
            device,
        ),
        TensorMut::new(
            cache_v_output,
            DataType::Float32,
            &cache_shape,
            &cache_strides,
            device,
        ),
    ];
    kernel.execute(&inputs, &mut outputs)
}

struct PackedStep {
    output: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    cache_k: DeviceBuffer,
    cache_v: DeviceBuffer,
}

#[allow(clippy::too_many_arguments)]
fn run_packed_step(
    ep: &CudaExecutionProvider,
    packed: &[f32],
    seq: usize,
    cache_k: Option<DeviceBuffer>,
    cache_v: Option<DeviceBuffer>,
    past_len: usize,
    total: usize,
    capacity: usize,
    cos: &[f32],
    sin: &[f32],
    positions: &[i64],
) -> onnx_runtime_ep_api::Result<PackedStep> {
    const NUM_HEADS: usize = 14;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 64;
    const PACKED_WIDTH: usize = (NUM_HEADS + 2 * KV_HEADS) * HEAD_DIM;
    let rotary_half = cos.len().checked_div(capacity).unwrap_or(0);

    if packed.len() != seq * PACKED_WIDTH
        || positions.len() != seq
        || total != past_len + seq
        || rotary_half == 0
        || rotary_half > HEAD_DIM / 2
        || cos.len() != capacity * rotary_half
        || sin.len() != cos.len()
        || cache_k.is_some() != cache_v.is_some()
        || (past_len > 0 && cache_k.is_none())
    {
        return Err(onnx_runtime_ep_api::EpError::KernelFailed(
            "invalid packed GQA test step".into(),
        ));
    }
    let total_i32 = i32::try_from(total).unwrap();
    let packed = f32_tensor(&[1, seq, PACKED_WIDTH], packed);
    let seqlens = i32_tensor(&[1], &[total_i32 - 1]);
    let total = i32_tensor(&[], &[i32::try_from(capacity).unwrap()]);
    let cos = f32_tensor(&[capacity, rotary_half], cos);
    let sin = f32_tensor(&[capacity, rotary_half], sin);
    let position = i64_tensor(&[1, seq], positions);
    let transient = [&packed, &seqlens, &total, &cos, &sin, &position]
        .into_iter()
        .map(|tensor| upload(ep, tensor))
        .collect::<onnx_runtime_ep_api::Result<Vec<_>>>()?;

    let has_past = cache_k.is_some();
    let cache_shape = vec![1, KV_HEADS, capacity, HEAD_DIM];
    let input_specs = [
        Some((DataType::Float32, vec![1, seq, PACKED_WIDTH])),
        None,
        None,
        has_past.then(|| (DataType::Float32, cache_shape.clone())),
        has_past.then(|| (DataType::Float32, cache_shape.clone())),
        Some((DataType::Int32, vec![1])),
        Some((DataType::Int32, vec![])),
        Some((DataType::Float32, vec![capacity, rotary_half])),
        Some((DataType::Float32, vec![capacity, rotary_half])),
        Some((DataType::Int64, vec![1, seq])),
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
        vec![1, seq, NUM_HEADS * HEAD_DIM],
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
        &[1, seq, PACKED_WIDTH][..],
        &[1][..],
        &[][..],
        &[capacity, rotary_half][..],
        &[capacity, rotary_half][..],
        &[1, seq][..],
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
    let (mut cache_k, mut cache_v) = match (cache_k, cache_v) {
        (Some(cache_k), Some(cache_v)) => (cache_k, cache_v),
        (None, None) => (
            ep.allocate(cache_shape.iter().product::<usize>() * 4, 256)?,
            ep.allocate(cache_shape.iter().product::<usize>() * 4, 256)?,
        ),
        _ => unreachable!(),
    };
    let inputs = vec![
        transient_views.next().unwrap(),
        TensorView::absent(DataType::Float32),
        TensorView::absent(DataType::Float32),
        if has_past {
            TensorView::new(
                DevicePtr(cache_k.as_ptr()),
                DataType::Float32,
                &cache_shape,
                &cache_strides,
                device,
            )
        } else {
            TensorView::absent(DataType::Float32)
        },
        if has_past {
            TensorView::new(
                DevicePtr(cache_v.as_ptr()),
                DataType::Float32,
                &cache_shape,
                &cache_strides,
                device,
            )
        } else {
            TensorView::absent(DataType::Float32)
        },
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
    Ok(PackedStep {
        output: bytes_to_f32(&output_bytes),
        key: bytes_to_f32(&key_bytes),
        value: bytes_to_f32(&value_bytes),
        cache_k,
        cache_v,
    })
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
    base_inputs_dtype(
        DataType::Float32,
        q_shape,
        q,
        k_shape,
        k,
        v,
        past_k,
        past_v,
        seqlens,
        total,
    )
}

#[allow(clippy::too_many_arguments)]
fn base_inputs_dtype(
    dtype: DataType,
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
        Some(float_tensor(dtype, q_shape, q)),
        Some(float_tensor(dtype, k_shape, k)),
        Some(float_tensor(dtype, k_shape, v)),
        past_k.map(|(shape, data)| float_tensor(dtype, shape, data)),
        past_v.map(|(shape, data)| float_tensor(dtype, shape, data)),
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

fn error_metrics(got: &[f32], expected: &[f32]) -> (f32, u32) {
    got.iter()
        .zip(expected)
        .fold((0.0f32, 0u32), |(max_abs, max_ulp), (&got, &expected)| {
            let got_key = if got.is_sign_negative() {
                !got.to_bits()
            } else {
                got.to_bits() | 0x8000_0000
            };
            let expected_key = if expected.is_sign_negative() {
                !expected.to_bits()
            } else {
                expected.to_bits() | 0x8000_0000
            };
            (
                max_abs.max((got - expected).abs()),
                max_ulp.max(got_key.abs_diff(expected_key)),
            )
        })
}

fn rotate_target(
    data: &[f32],
    seq: usize,
    heads: usize,
    dimensions: (usize, usize),
    interleaved: bool,
    positions: &[usize],
    caches: (&[f32], &[f32]),
) -> Vec<f32> {
    let (head_dim, rotary_dim) = dimensions;
    let (cos, sin) = caches;
    let half = rotary_dim / 2;
    let mut output = data.to_vec();
    for (token, &position) in positions.iter().enumerate().take(seq) {
        for head in 0..heads {
            let base = (token * heads + head) * head_dim;
            for k in 0..half {
                let d0 = if interleaved { 2 * k } else { k };
                let d1 = if interleaved { 2 * k + 1 } else { k + half };
                let x0 = data[base + d0];
                let x1 = data[base + d1];
                let cache = position * half + k;
                output[base + d0] = cos[cache] * x0 - sin[cache] * x1;
                output[base + d1] = sin[cache] * x0 + cos[cache] * x1;
            }
        }
    }
    output
}

fn target_attention_reference(
    query: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    q_seq: usize,
    past_len: usize,
    capacity: usize,
) -> Vec<f32> {
    const NUM_HEADS: usize = 14;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 64;
    let group = NUM_HEADS / KV_HEADS;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut output = vec![0.0; q_seq * NUM_HEADS * HEAD_DIM];
    for token in 0..q_seq {
        let valid_length = past_len + token + 1;
        for head in 0..NUM_HEADS {
            let kv_head = head / group;
            let mut scores = Vec::with_capacity(valid_length);
            for position in 0..valid_length {
                let score = (0..HEAD_DIM)
                    .map(|dim| {
                        query[(token * NUM_HEADS + head) * HEAD_DIM + dim]
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
                output[(token * NUM_HEADS + head) * HEAD_DIM + dim] = scores
                    .iter()
                    .enumerate()
                    .map(|(position, probability)| {
                        probability / sum
                            * value_cache[(kv_head * capacity + position) * HEAD_DIM + dim]
                    })
                    .sum();
            }
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

fn fill(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn assert_close(got: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), expected.len());
    let mut max_abs = 0.0f32;
    for (index, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        let error = (got - expected).abs();
        max_abs = max_abs.max(error);
        assert!(
            error <= atol + rtol * expected.abs(),
            "{index}: {got} != {expected}, error={error}, atol={atol}, rtol={rtol}"
        );
    }
    eprintln!("GQA fused-vs-baseline max_abs={max_abs:e}");
}

#[derive(Clone)]
struct ParityCase {
    name: String,
    dtype: DataType,
    batch: usize,
    heads: usize,
    kv_heads: usize,
    q_seq: usize,
    k_seq: usize,
    dim: usize,
    past_capacity: usize,
    capacity: usize,
    totals: Vec<usize>,
    rope: bool,
    local_window: i64,
    softcap: f32,
    magnitude: f32,
    seed: u64,
}

fn parity_tolerances(dtype: DataType) -> (f32, f32) {
    match dtype {
        DataType::Float32 => (3e-6, 2e-6),
        DataType::Float16 => (7e-4, 7e-4),
        DataType::BFloat16 => (5e-3, 5e-3),
        _ => unreachable!("GQA parity dtype"),
    }
}

fn parity_fixture(
    case: &ParityCase,
) -> (
    Vec<(&'static str, Attribute)>,
    Vec<Option<HostTensor>>,
    Vec<Vec<usize>>,
) {
    assert_eq!(case.totals.len(), case.batch);
    assert!(case.totals.iter().all(|&total| total >= case.k_seq));
    assert!(case.totals.iter().all(|&total| total >= case.q_seq));
    assert!(case
        .totals
        .iter()
        .all(|&total| total - case.k_seq <= case.past_capacity));
    let scale_values = |mut values: Vec<f32>| {
        for value in &mut values {
            *value *= case.magnitude;
        }
        quantize(&values, case.dtype)
    };
    let q = scale_values(fill(
        case.batch * case.q_seq * case.heads * case.dim,
        case.seed,
    ));
    let k = scale_values(fill(
        case.batch * case.k_seq * case.kv_heads * case.dim,
        case.seed + 1,
    ));
    let v = quantize(
        &fill(
            case.batch * case.k_seq * case.kv_heads * case.dim,
            case.seed + 2,
        ),
        case.dtype,
    );
    let past_k = (case.past_capacity > 0).then(|| {
        quantize(
            &fill(
                case.batch * case.kv_heads * case.past_capacity * case.dim,
                case.seed + 3,
            ),
            case.dtype,
        )
    });
    let past_v = (case.past_capacity > 0).then(|| {
        quantize(
            &fill(
                case.batch * case.kv_heads * case.past_capacity * case.dim,
                case.seed + 4,
            ),
            case.dtype,
        )
    });
    let past_shape = [case.batch, case.kv_heads, case.past_capacity, case.dim];
    let seqlens = case
        .totals
        .iter()
        .map(|&total| (total - 1) as i32)
        .collect::<Vec<_>>();
    let mut inputs = base_inputs_dtype(
        case.dtype,
        &[case.batch, case.q_seq, case.heads * case.dim],
        &q,
        &[case.batch, case.k_seq, case.kv_heads * case.dim],
        &k,
        &v,
        past_k
            .as_ref()
            .map(|data| (&past_shape[..], data.as_slice())),
        past_v
            .as_ref()
            .map(|data| (&past_shape[..], data.as_slice())),
        &seqlens,
        case.capacity as i32,
    );
    let mut attrs = vec![
        ("num_heads", Attribute::Int(case.heads as i64)),
        ("kv_num_heads", Attribute::Int(case.kv_heads as i64)),
    ];
    if case.rope {
        attrs.push(("do_rotary", Attribute::Int(1)));
        let mut cos = Vec::with_capacity(case.capacity * case.dim / 2);
        let mut sin = Vec::with_capacity(cos.capacity());
        for position in 0..case.capacity {
            for index in 0..case.dim / 2 {
                let angle = position as f32 * (index + 1) as f32 * 0.013;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        inputs.push(Some(f32_tensor(&[case.capacity, case.dim / 2], &cos)));
        inputs.push(Some(f32_tensor(&[case.capacity, case.dim / 2], &sin)));
    }
    if case.local_window > 0 {
        attrs.push(("local_window_size", Attribute::Int(case.local_window)));
    }
    if case.softcap > 0.0 {
        attrs.push(("softcap", Attribute::Float(case.softcap)));
    }
    let output_shapes = vec![
        vec![case.batch, case.q_seq, case.heads * case.dim],
        vec![case.batch, case.kv_heads, case.capacity, case.dim],
        vec![case.batch, case.kv_heads, case.capacity, case.dim],
    ];
    (attrs, inputs, output_shapes)
}

fn run_forced_parity_case(ep: &CudaExecutionProvider, case: &ParityCase) {
    let (attrs, inputs, output_shapes) = parity_fixture(case);
    let fused = run_available(run_with_backend(
        ep,
        &attrs,
        &inputs,
        &output_shapes,
        Some(GroupQueryAttentionBackend::Fused),
    ))
    .unwrap();
    let baseline = run_available(run_with_backend(
        ep,
        &attrs,
        &inputs,
        &output_shapes,
        Some(GroupQueryAttentionBackend::Phase2a),
    ))
    .unwrap();
    let (atol, rtol) = parity_tolerances(case.dtype);
    eprintln!("GQA forced parity {}", case.name);
    assert!(fused[0].iter().all(|value| value.is_finite()));
    assert_close(&fused[0], &baseline[0], atol, rtol);
    assert_eq!(fused[1], baseline[1]);
    assert_eq!(fused[2], baseline[2]);
}

#[test]
fn gqa_gpu_fused_causal_origin_matches_baseline_when_query_and_key_lengths_differ() {
    let Some(ep) = gpu() else { return };
    for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
        for (name, batch, totals, past_capacity, capacity, seed) in [
            ("fresh", 1usize, vec![4usize], 0usize, 4usize, 681u64),
            (
                "cached-ragged",
                2usize,
                vec![7usize, 9usize],
                6usize,
                10usize,
                691u64,
            ),
        ] {
            run_forced_parity_case(
                &ep,
                &ParityCase {
                    name: format!(
                        "causal-origin-{name}-q2-k4-{}",
                        format!("{dtype:?}").to_lowercase()
                    ),
                    dtype,
                    batch,
                    heads: 4,
                    kv_heads: 2,
                    q_seq: 2,
                    k_seq: 4,
                    dim: 64,
                    past_capacity,
                    capacity,
                    totals,
                    rope: false,
                    local_window: -1,
                    softcap: 0.0,
                    magnitude: 1.0,
                    seed,
                },
            );
        }
    }
}

#[test]
fn gqa_gpu_forced_fused_matches_baseline_parity_matrix() {
    let Some(ep) = gpu() else { return };
    let mut cases = Vec::new();
    let mut seed = 701u64;
    for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
        for (sharing, heads, kv_heads) in [
            ("mha", 4usize, 4usize),
            ("gqa", 8usize, 2usize),
            ("mqa", 4usize, 1usize),
        ] {
            for (mode, batch, past_capacity, capacity, totals, local_window, softcap) in [
                (
                    "fresh-uniform",
                    1usize,
                    0usize,
                    7usize,
                    vec![7usize],
                    -1,
                    0.0,
                ),
                ("cached-uniform-window-softcap", 1, 6, 12, vec![10], 4, 2.0),
                ("cached-ragged", 2, 7, 12, vec![8, 11], -1, 0.0),
            ] {
                cases.push(ParityCase {
                    name: format!("{}-{sharing}-{mode}", format!("{dtype:?}").to_lowercase()),
                    dtype,
                    batch,
                    heads,
                    kv_heads,
                    q_seq: if mode == "fresh-uniform" { 7 } else { 5 },
                    k_seq: if mode == "fresh-uniform" { 7 } else { 5 },
                    dim: 64,
                    past_capacity,
                    capacity,
                    totals,
                    rope: false,
                    local_window,
                    softcap,
                    magnitude: 1.0,
                    seed,
                });
                seed += 10;
            }
        }
        cases.push(ParityCase {
            name: format!("{}-mqa-rope", format!("{dtype:?}").to_lowercase()),
            dtype,
            batch: 1,
            heads: 4,
            kv_heads: 1,
            q_seq: 5,
            k_seq: 5,
            dim: 64,
            past_capacity: 0,
            capacity: 5,
            totals: vec![5],
            rope: true,
            local_window: -1,
            softcap: 0.0,
            magnitude: 1.0,
            seed,
        });
        seed += 10;
    }
    cases.extend([
        ParityCase {
            name: "float16-gqa-generic-non-wmma-d72".into(),
            dtype: DataType::Float16,
            batch: 1,
            heads: 8,
            kv_heads: 2,
            q_seq: 6,
            k_seq: 6,
            dim: 72,
            past_capacity: 0,
            capacity: 6,
            totals: vec![6],
            rope: false,
            local_window: -1,
            softcap: 0.0,
            magnitude: 1.0,
            seed,
        },
        ParityCase {
            name: "float32-gqa-ragged-large-magnitude".into(),
            dtype: DataType::Float32,
            batch: 2,
            heads: 8,
            kv_heads: 2,
            q_seq: 5,
            k_seq: 5,
            dim: 64,
            past_capacity: 8,
            capacity: 16,
            totals: vec![8, 11],
            rope: false,
            local_window: -1,
            softcap: 0.0,
            magnitude: 40.0,
            seed: seed + 10,
        },
    ]);

    for case in &cases {
        run_forced_parity_case(&ep, case);
    }
}

#[test]
fn gqa_gpu_fp16_decode_split_k_long_context_is_deterministic_and_matches_baseline() {
    let Some(ep) = gpu() else { return };
    let case = ParityCase {
        name: "float16-gqa-decode-split-k-long-rope".into(),
        dtype: DataType::Float16,
        batch: 1,
        heads: 14,
        kv_heads: 2,
        q_seq: 1,
        k_seq: 1,
        dim: 64,
        past_capacity: 1024,
        capacity: 1024,
        totals: vec![1024],
        rope: true,
        local_window: -1,
        softcap: 0.0,
        magnitude: 1.0,
        seed: 971,
    };
    let (attrs, inputs, output_shapes) = parity_fixture(&case);
    let first = run_available(run_with_backend(
        &ep,
        &attrs,
        &inputs,
        &output_shapes,
        Some(GroupQueryAttentionBackend::Fused),
    ))
    .unwrap();
    let second = run_available(run_with_backend(
        &ep,
        &attrs,
        &inputs,
        &output_shapes,
        Some(GroupQueryAttentionBackend::Fused),
    ))
    .unwrap();
    let baseline = run_available(run_with_backend(
        &ep,
        &attrs,
        &inputs,
        &output_shapes,
        Some(GroupQueryAttentionBackend::Phase2a),
    ))
    .unwrap();
    assert_eq!(first, second, "fp16 split-K decode must be deterministic");
    let (atol, rtol) = parity_tolerances(case.dtype);
    assert_close(&first[0], &baseline[0], atol, rtol);
    assert_eq!(first[1], baseline[1]);
    assert_eq!(first[2], baseline[2]);
}

#[test]
fn gqa_gpu_auto_fallback_matches_baseline_and_reports_selected_backend() {
    let Some(ep) = gpu() else { return };
    for case in [
        ParityCase {
            name: "auto-decode-fallback".into(),
            dtype: DataType::Float16,
            batch: 1,
            heads: 4,
            kv_heads: 1,
            q_seq: 1,
            k_seq: 1,
            dim: 64,
            past_capacity: 8,
            capacity: 9,
            totals: vec![9],
            rope: false,
            local_window: -1,
            softcap: 0.0,
            magnitude: 1.0,
            seed: 991,
        },
        ParityCase {
            name: "auto-cached-large-slow-fallback".into(),
            dtype: DataType::Float16,
            batch: 1,
            heads: 4,
            kv_heads: 1,
            q_seq: 512,
            k_seq: 512,
            dim: 64,
            past_capacity: 512,
            capacity: 1024,
            totals: vec![1024],
            rope: false,
            local_window: -1,
            softcap: 0.0,
            magnitude: 1.0,
            seed: 1001,
        },
    ] {
        let (attrs, inputs, output_shapes) = parity_fixture(&case);
        let auto_kernel = GroupQueryAttentionKernel::new(
            ep.runtime().clone(),
            case.heads,
            case.kv_heads,
            None,
            false,
            false,
            case.local_window,
            case.softcap,
        )
        .unwrap();
        assert_eq!(
            auto_kernel.selected_backend_for_shape(
                case.dtype,
                case.q_seq,
                *case.totals.iter().max().unwrap(),
                case.dim,
            ),
            GroupQueryAttentionBackend::Phase2a,
            "{} must select the baseline",
            case.name
        );
        let auto = run_available(run_with_backend(
            &ep,
            &attrs,
            &inputs,
            &output_shapes,
            Some(GroupQueryAttentionBackend::Auto),
        ))
        .unwrap();
        let baseline = run_available(run_with_backend(
            &ep,
            &attrs,
            &inputs,
            &output_shapes,
            Some(GroupQueryAttentionBackend::Phase2a),
        ))
        .unwrap();
        let (atol, rtol) = parity_tolerances(case.dtype);
        eprintln!("GQA {}", case.name);
        assert_close(&auto[0], &baseline[0], atol, rtol);
        assert_eq!(auto[1], baseline[1]);
        assert_eq!(auto[2], baseline[2]);
    }
}

fn benchmark_gqa_case(
    ep: &CudaExecutionProvider,
    q_seq: usize,
    past_len: usize,
    backend: GroupQueryAttentionBackend,
    iterations: usize,
) -> f64 {
    let (batch, heads, kv_heads, dim) = (1usize, 32usize, 8usize, 128usize);
    let total = past_len + q_seq;
    let capacity = total;
    let dtype = DataType::Float16;
    let runtime = ep.runtime();
    let device = ep.device_id();
    let upload = |values: &[f32]| {
        let bytes = float_tensor(dtype, &[], values).bytes;
        let buffer = ep.allocate(bytes.len(), 256).unwrap();
        // SAFETY: the allocation exactly covers the encoded values.
        unsafe {
            runtime.htod(&bytes, cuptr(buffer.as_ptr())).unwrap();
        }
        buffer
    };
    let q = upload(&fill(batch * q_seq * heads * dim, 801 + q_seq as u64));
    let k = upload(&fill(batch * q_seq * kv_heads * dim, 802 + q_seq as u64));
    let v = upload(&fill(batch * q_seq * kv_heads * dim, 803 + q_seq as u64));
    let mut cache_k = upload(&fill(batch * kv_heads * capacity * dim, 804 + q_seq as u64));
    let mut cache_v = upload(&fill(batch * kv_heads * capacity * dim, 805 + q_seq as u64));
    let seqlens_host = i32_tensor(&[1], &[(total - 1) as i32]);
    let total_host = i32_tensor(&[], &[capacity as i32]);
    let seqlens = upload_bytes(ep, &seqlens_host);
    let total_length = upload_bytes(ep, &total_host);
    let mut output = ep.allocate(batch * q_seq * heads * dim * 2, 256).unwrap();

    let q_shape = [batch, q_seq, heads * dim];
    let kv_shape = [batch, q_seq, kv_heads * dim];
    let cache_shape = [batch, kv_heads, capacity, dim];
    let q_strides = compute_contiguous_strides(&q_shape);
    let kv_strides = compute_contiguous_strides(&kv_shape);
    let cache_strides = compute_contiguous_strides(&cache_shape);
    let scalar_strides = compute_contiguous_strides(&[]);
    let seqlens_strides = compute_contiguous_strides(&[1]);
    let output_strides = compute_contiguous_strides(&q_shape);
    let inputs = vec![
        TensorView::new(DevicePtr(q.as_ptr()), dtype, &q_shape, &q_strides, device),
        TensorView::new(DevicePtr(k.as_ptr()), dtype, &kv_shape, &kv_strides, device),
        TensorView::new(DevicePtr(v.as_ptr()), dtype, &kv_shape, &kv_strides, device),
        if past_len > 0 {
            TensorView::new(
                DevicePtr(cache_k.as_ptr()),
                dtype,
                &cache_shape,
                &cache_strides,
                device,
            )
        } else {
            TensorView::absent(dtype)
        },
        if past_len > 0 {
            TensorView::new(
                DevicePtr(cache_v.as_ptr()),
                dtype,
                &cache_shape,
                &cache_strides,
                device,
            )
        } else {
            TensorView::absent(dtype)
        },
        TensorView::new(
            DevicePtr(seqlens.as_ptr()),
            DataType::Int32,
            &[1],
            &seqlens_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(total_length.as_ptr()),
            DataType::Int32,
            &[],
            &scalar_strides,
            device,
        ),
    ];
    let kernel = GroupQueryAttentionKernel::new(
        runtime.clone(),
        heads,
        kv_heads,
        Some(1.0 / (dim as f32).sqrt()),
        false,
        false,
        -1,
        0.0,
    )
    .unwrap()
    .with_backend(backend);
    let mut run = || {
        kernel
            .execute(
                &inputs,
                &mut [
                    TensorMut::new(
                        DevicePtrMut(output.as_mut_ptr()),
                        dtype,
                        &q_shape,
                        &output_strides,
                        device,
                    ),
                    TensorMut::new(
                        DevicePtrMut(cache_k.as_mut_ptr()),
                        dtype,
                        &cache_shape,
                        &cache_strides,
                        device,
                    ),
                    TensorMut::new(
                        DevicePtrMut(cache_v.as_mut_ptr()),
                        dtype,
                        &cache_shape,
                        &cache_strides,
                        device,
                    ),
                ],
            )
            .unwrap();
    };
    run();
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        run();
        samples.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    samples.sort_by(f64::total_cmp);
    let median = samples[samples.len() / 2];
    drop(run);
    drop(inputs);
    for buffer in [q, k, v, cache_k, cache_v, seqlens, total_length, output] {
        ep.deallocate(buffer).unwrap();
    }
    median
}

fn upload_bytes(ep: &CudaExecutionProvider, tensor: &HostTensor) -> DeviceBuffer {
    let buffer = ep.allocate(tensor.bytes.len(), 256).unwrap();
    // SAFETY: the allocation exactly covers the source bytes.
    unsafe {
        ep.runtime()
            .htod(&tensor.bytes, cuptr(buffer.as_ptr()))
            .unwrap();
    }
    buffer
}

#[test]
#[ignore = "H200 performance benchmark; run explicitly with --ignored --nocapture"]
fn gqa_prefill_h200_benchmark() {
    let ep = CudaExecutionProvider::new_default().expect("benchmark requires a CUDA GPU");
    for (q_seq, iterations) in [(512usize, 10usize), (2048usize, 3usize)] {
        for past_len in [0usize, q_seq] {
            let fused = benchmark_gqa_case(
                &ep,
                q_seq,
                past_len,
                GroupQueryAttentionBackend::Fused,
                iterations,
            );
            let baseline = benchmark_gqa_case(
                &ep,
                q_seq,
                past_len,
                GroupQueryAttentionBackend::Phase2a,
                iterations,
            );
            let total = q_seq + past_len;
            let score_bytes = 32usize * q_seq * total * 2;
            let baseline_scratch = score_bytes + 32 * 1024 * 1024;
            println!(
                "H200 GQA f16 causal B=1 H=32 KVH=8 Q={q_seq} past={past_len} D=128: \
                 fused={fused:.3} ms baseline={baseline:.3} ms speedup={:.2}x; \
                 attention scratch fused=0 MiB baseline={:.1} MiB",
                baseline / fused,
                baseline_scratch as f64 / (1024.0 * 1024.0),
            );
        }
    }
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
    let outputs = run_available(run(
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
    ))
    .unwrap();
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
fn gqa_gpu_scalar_seqlens_requires_unit_batch_and_valid_value() {
    let Some(ep) = gpu() else { return };
    let mut inputs = base_inputs(
        &[1, 1, 8],
        &[1., 0., 1., 0., 0., 1., 0., 1.],
        &[1, 1, 4],
        &[1., 0., 0., 1.],
        &[1., 2., 10., 20.],
        None,
        None,
        &[0],
        1,
    );
    inputs[5].as_mut().unwrap().shape.clear();
    let scalar = run_available(run(
        &ep,
        &attrs(&[]),
        &inputs,
        &[vec![1, 1, 8], vec![1, 2, 1, 2], vec![1, 2, 1, 2]],
    ))
    .expect("unit-batch scalar seqlens_k must be promoted");
    assert_eq!(scalar[0], vec![1., 2., 1., 2., 10., 20., 10., 20.]);

    let mut negative = inputs.clone();
    negative[5].as_mut().unwrap().bytes = typed_bytes(&[-1_i32]);
    let error = run_available(run(
        &ep,
        &attrs(&[]),
        &negative,
        &[vec![1, 1, 8], vec![1, 2, 1, 2], vec![1, 2, 1, 2]],
    ))
    .expect_err("negative scalar must fail");
    assert!(format!("{error}").contains("non-negative"));

    let mut out_of_range = inputs;
    out_of_range[5].as_mut().unwrap().bytes = typed_bytes(&[1_i32]);
    let error = run_available(run(
        &ep,
        &attrs(&[]),
        &out_of_range,
        &[vec![1, 1, 8], vec![1, 2, 1, 2], vec![1, 2, 1, 2]],
    ))
    .expect_err("scalar exceeding total sequence length must fail");
    assert!(format!("{error}").contains("exceeds physical total_sequence_length"));
}

#[test]
fn gqa_gpu_unit_batch_scalar_matches_cpu_oracle_and_cuda_vector() {
    let Some(ep) = gpu() else {
        return;
    };
    let q = [1., 0., 1., 0., 0., 1., 0., 1.];
    let k = [1., 0., 0., 1.];
    let v = [1., 2., 10., 20.];
    let output_shapes = [vec![1, 1, 8], vec![1, 2, 1, 2], vec![1, 2, 1, 2]];
    let vector_inputs = base_inputs(&[1, 1, 8], &q, &[1, 1, 4], &k, &v, None, None, &[0], 1);
    let mut scalar_inputs = vector_inputs.clone();
    scalar_inputs[5].as_mut().unwrap().shape.clear();

    let vector = run_available(run(&ep, &attrs(&[]), &vector_inputs, &output_shapes)).unwrap();
    let scalar = run_available(run(&ep, &attrs(&[]), &scalar_inputs, &output_shapes)).unwrap();
    assert_eq!(scalar, vector);

    let cpu_oracle = [
        vec![1., 2., 1., 2., 10., 20., 10., 20.],
        vec![1., 0., 0., 1.],
        vec![1., 2., 10., 20.],
    ];
    for (got, expected) in scalar.iter().zip(cpu_oracle.iter()) {
        close(got, expected);
    }
}

#[test]
fn gqa_gpu_batch_trailing_singleton_seqlens_matches_vector_for_any_batch() {
    let Some(ep) = gpu() else { return };
    for case in [
        ParityCase {
            name: "seqlens-batch1".into(),
            dtype: DataType::Float32,
            batch: 1,
            heads: 4,
            kv_heads: 2,
            q_seq: 2,
            k_seq: 2,
            dim: 64,
            past_capacity: 0,
            capacity: 2,
            totals: vec![2],
            rope: false,
            local_window: -1,
            softcap: 0.0,
            magnitude: 1.0,
            seed: 1201,
        },
        ParityCase {
            name: "seqlens-batch2-ragged".into(),
            dtype: DataType::Float32,
            batch: 2,
            heads: 4,
            kv_heads: 2,
            q_seq: 1,
            k_seq: 1,
            dim: 64,
            past_capacity: 2,
            capacity: 3,
            totals: vec![2, 3],
            rope: false,
            local_window: -1,
            softcap: 0.0,
            magnitude: 1.0,
            seed: 1211,
        },
    ] {
        let (attrs, vector_inputs, output_shapes) = parity_fixture(&case);
        let mut trailing_singleton_inputs = vector_inputs.clone();
        trailing_singleton_inputs[5].as_mut().unwrap().shape = vec![case.batch, 1];

        let vector = run_available(run(&ep, &attrs, &vector_inputs, &output_shapes)).unwrap();
        let trailing_singleton =
            run_available(run(&ep, &attrs, &trailing_singleton_inputs, &output_shapes)).unwrap();
        assert_eq!(
            trailing_singleton, vector,
            "{} must treat [batch_size, 1] seqlens_k identically to [batch_size]",
            case.name
        );
    }
}

#[test]
fn gqa_gpu_rejects_noncanonical_seqlens_singleton_shapes_actionably() {
    let Some(ep) = gpu() else { return };
    let case = ParityCase {
        name: "seqlens-invalid-layouts".into(),
        dtype: DataType::Float32,
        batch: 2,
        heads: 4,
        kv_heads: 2,
        q_seq: 1,
        k_seq: 1,
        dim: 64,
        past_capacity: 0,
        capacity: 1,
        totals: vec![1, 1],
        rope: false,
        local_window: -1,
        softcap: 0.0,
        magnitude: 1.0,
        seed: 1221,
    };
    let (attrs, inputs, output_shapes) = parity_fixture(&case);

    for shape in [vec![1, case.batch], vec![case.batch, 1, 1]] {
        let mut invalid_inputs = inputs.clone();
        invalid_inputs[5].as_mut().unwrap().shape = shape.clone();
        let error = run_available(run(&ep, &attrs, &invalid_inputs, &output_shapes))
            .expect_err("noncanonical seqlens_k singleton shape must fail");
        let message = format!("{error}");
        assert!(message.contains("[batch_size], [batch_size, 1]"));
        assert!(message.contains(&format!("got shape {shape:?}")));
    }
}

#[test]
fn gqa_gpu_scalar_rejects_multi_batch() {
    let Some(ep) = gpu() else {
        return;
    };
    let mut inputs = base_inputs(
        &[2, 1, 8],
        &[
            1., 0., 1., 0., 0., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 1.,
        ],
        &[2, 1, 4],
        &[1., 0., 0., 1., 1., 0., 0., 1.],
        &[1., 2., 10., 20., 3., 4., 30., 40.],
        None,
        None,
        &[0],
        1,
    );
    inputs[5].as_mut().unwrap().shape.clear();
    let error = run_available(run(
        &ep,
        &attrs(&[]),
        &inputs,
        &[vec![2, 1, 8], vec![2, 2, 1, 2], vec![2, 2, 1, 2]],
    ))
    .expect_err("scalar cannot encode multiple batch rows");
    let message = format!("{error}");
    assert!(message.contains("only be promoted to [1] when batch_size is 1"));
    assert!(message.contains("contiguous int32 [batch_size] or [batch_size, 1]"));
}

#[test]
fn gqa_gpu_scalar_preserves_dtype_layout_value_and_shape_validation() {
    let Some(ep) = gpu() else {
        return;
    };
    let base = || {
        base_inputs(
            &[1, 1, 8],
            &[1., 0., 1., 0., 0., 1., 0., 1.],
            &[1, 1, 4],
            &[1., 0., 0., 1.],
            &[1., 2., 10., 20.],
            None,
            None,
            &[0],
            1,
        )
    };
    let output_shapes = [vec![1, 1, 8], vec![1, 2, 1, 2], vec![1, 2, 1, 2]];

    let mut wrong_dtype = base();
    wrong_dtype[5] = Some(i64_tensor(&[], &[0]));
    let error = run_available(run(&ep, &attrs(&[]), &wrong_dtype, &output_shapes))
        .expect_err("wrong dtype must fail");
    assert!(format!("{error}").contains("seqlens_k must have dtype Int32"));

    let mut non_contiguous = base();
    non_contiguous[5].as_mut().unwrap().strides = Some(vec![2]);
    let error = run_available(run(&ep, &attrs(&[]), &non_contiguous, &output_shapes))
        .expect_err("non-contiguous seqlens_k must fail");
    let message = format!("{error}");
    assert!(message.contains("non-contiguous seqlens_k"));
    assert!(message.contains("[batch_size]"));
    assert!(message.contains("[batch_size, 1]"));

    let mut negative_scalar = base();
    negative_scalar[5] = Some(i32_tensor(&[], &[-1]));
    let error = run_available(run(&ep, &attrs(&[]), &negative_scalar, &output_shapes))
        .expect_err("negative scalar must fail");
    assert!(format!("{error}").contains("seqlens_k must be non-negative"));

    let mut non_one_element = base();
    non_one_element[5] = Some(i32_tensor(&[2], &[0, 0]));
    let error = run_available(run(&ep, &attrs(&[]), &non_one_element, &output_shapes))
        .expect_err("non-one-element alternate shape must fail");
    let message = format!("{error}");
    assert!(message.contains("[batch_size], [batch_size, 1]"));
    assert!(message.contains("got shape [2]"));
}

#[test]
fn gqa_gpu_fixed_decode_capture_replays_bit_identically() {
    let Some(ep) = gpu() else { return };
    let runtime = ep.runtime();
    let kernel =
        GroupQueryAttentionKernel::new(runtime.clone(), 4, 2, None, true, false, -1, 0.0).unwrap();
    let eager_kernel =
        GroupQueryAttentionKernel::new(runtime.clone(), 4, 2, None, true, false, -1, 0.0).unwrap();
    assert!(!kernel.cuda_graph_compatible());

    let packed_host = f32_tensor(
        &[1, 1, 16],
        &[
            1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 10., 10., 5., 6., 50., 60.,
        ],
    );
    let cache_k_host = f32_tensor(
        &[1, 2, 5, 2],
        &[
            1., 0., 0., 1., 91., 92., 93., 94., 95., 96., 10., 0., 0., 10., 81., 82., 83., 84.,
            85., 86.,
        ],
    );
    let cache_v_host = f32_tensor(
        &[1, 2, 5, 2],
        &[
            1., 2., 3., 4., 71., 72., 73., 74., 75., 76., 10., 20., 30., 40., 61., 62., 63., 64.,
            65., 66.,
        ],
    );
    let seqlens_host = i32_tensor(&[], &[1]);
    let total_host = i32_tensor(&[1], &[5]);
    let cos_host = f32_tensor(&[5, 1], &[1.0, 0.0, -1.0, 0.5, -0.5]);
    let sin_host = f32_tensor(&[5, 1], &[0.0, 1.0, 0.0, 0.866_025_4, 0.866_025_4]);
    let positions_host = i64_tensor(&[1, 2], &[1, 0]);
    let packed = upload(&ep, &packed_host).unwrap();
    let mut cache_k = upload(&ep, &cache_k_host).unwrap();
    let mut cache_v = upload(&ep, &cache_v_host).unwrap();
    let seqlens = upload(&ep, &seqlens_host).unwrap();
    let total = upload(&ep, &total_host).unwrap();
    let cos = upload(&ep, &cos_host).unwrap();
    let sin = upload(&ep, &sin_host).unwrap();
    let positions = upload(&ep, &positions_host).unwrap();
    let output_bytes = 8 * std::mem::size_of::<f32>();
    let mut output = ep.allocate(output_bytes, 256).unwrap();
    let mut eager_cache_k = upload(&ep, &cache_k_host).unwrap();
    let mut eager_cache_v = upload(&ep, &cache_v_host).unwrap();
    let eager_seqlens = upload(&ep, &seqlens_host).unwrap();
    let eager_positions = upload(&ep, &positions_host).unwrap();
    let mut eager_output = ep.allocate(output_bytes, 256).unwrap();

    execute_in_place_packed_f32_gqa_with_seqlens_shape(
        &kernel,
        &ep,
        &packed,
        &mut cache_k,
        &mut cache_v,
        &seqlens,
        &total,
        &cos,
        &sin,
        &positions,
        &mut output,
        1,
        &[],
    )
    .unwrap();
    execute_in_place_packed_f32_gqa_with_seqlens_shape(
        &eager_kernel,
        &ep,
        &packed,
        &mut eager_cache_k,
        &mut eager_cache_v,
        &eager_seqlens,
        &total,
        &cos,
        &sin,
        &eager_positions,
        &mut eager_output,
        1,
        &[],
    )
    .unwrap();
    assert_eq!(
        read_bytes(&ep, &output, output_bytes).unwrap(),
        read_bytes(&ep, &eager_output, output_bytes).unwrap()
    );
    assert!(kernel.cuda_graph_compatible());

    let allocation_counts = runtime.allocation_counts();
    let kernels: [&dyn Kernel; 1] = [&kernel];
    runtime.begin_graph_capture(&kernels).unwrap();
    execute_in_place_packed_f32_gqa_with_seqlens_shape(
        &kernel,
        &ep,
        &packed,
        &mut cache_k,
        &mut cache_v,
        &seqlens,
        &total,
        &cos,
        &sin,
        &positions,
        &mut output,
        1,
        &[],
    )
    .unwrap();
    runtime.end_graph_capture().unwrap();

    let mut previous = read_bytes(&ep, &output, output_bytes).unwrap();
    for step in 2_i32..=4 {
        let seqlens_step = i32_tensor(&[1], &[step]);
        let positions_step = i64_tensor(&[1], &[i64::from(step)]);
        overwrite(&ep, &seqlens, &seqlens_step).unwrap();
        overwrite(&ep, &positions, &positions_step).unwrap();
        overwrite(&ep, &eager_seqlens, &seqlens_step).unwrap();
        overwrite(&ep, &eager_positions, &positions_step).unwrap();

        execute_in_place_packed_f32_gqa_with_seqlens_shape(
            &eager_kernel,
            &ep,
            &packed,
            &mut eager_cache_k,
            &mut eager_cache_v,
            &eager_seqlens,
            &total,
            &cos,
            &sin,
            &eager_positions,
            &mut eager_output,
            1,
            &[],
        )
        .unwrap();
        runtime.replay_graph().unwrap();
        let replayed = read_bytes(&ep, &output, output_bytes).unwrap();
        let eager = read_bytes(&ep, &eager_output, output_bytes).unwrap();
        assert_eq!(
            replayed, eager,
            "capture replay diverged at decode step {step}"
        );
        assert_eq!(
            read_bytes(&ep, &cache_k, cache_k.len()).unwrap(),
            read_bytes(&ep, &eager_cache_k, eager_cache_k.len()).unwrap(),
            "captured present key diverged at decode step {step}"
        );
        assert_eq!(
            read_bytes(&ep, &cache_v, cache_v.len()).unwrap(),
            read_bytes(&ep, &eager_cache_v, eager_cache_v.len()).unwrap(),
            "captured present value diverged at decode step {step}"
        );
        assert_eq!(runtime.check_capture_error().unwrap(), 0);
        assert_ne!(
            replayed, previous,
            "attention output did not change when the valid window grew at step {step}"
        );
        previous = replayed;
    }
    assert_eq!(runtime.allocation_counts(), allocation_counts);
    assert!(runtime.reset_graph().unwrap());

    let prefill_packed_host = f32_tensor(&[1, 2, 16], &[0.25; 32]);
    let prefill_seqlens_host = i32_tensor(&[1], &[1]);
    let prefill_total_host = i32_tensor(&[1], &[2]);
    let prefill_positions_host = i64_tensor(&[1, 2], &[0, 1]);
    let prefill_packed = upload(&ep, &prefill_packed_host).unwrap();
    overwrite(&ep, &seqlens, &prefill_seqlens_host).unwrap();
    overwrite(&ep, &total, &prefill_total_host).unwrap();
    overwrite(&ep, &positions, &prefill_positions_host).unwrap();
    let mut prefill_output = ep.allocate(2 * output_bytes, 256).unwrap();
    let kernels: [&dyn Kernel; 1] = [&kernel];
    runtime.begin_graph_capture(&kernels).unwrap();
    let error = execute_in_place_packed_f32_gqa(
        &kernel,
        &ep,
        &prefill_packed,
        &mut cache_k,
        &mut cache_v,
        &seqlens,
        &total,
        &cos,
        &sin,
        &positions,
        &mut prefill_output,
        2,
    )
    .unwrap_err();
    assert!(format!("{error}").contains("dtype, decode mode, or shape changed"));
    let _ = runtime.end_graph_capture();
    assert!(!kernel.cuda_graph_compatible());

    for buffer in [
        prefill_output,
        prefill_packed,
        eager_output,
        eager_positions,
        eager_seqlens,
        eager_cache_v,
        eager_cache_k,
        output,
        positions,
        sin,
        cos,
        total,
        seqlens,
        cache_v,
        cache_k,
        packed,
    ] {
        ep.deallocate(buffer).unwrap();
    }
}

#[test]
fn gqa_gpu_capture_detects_invalid_decode_metadata() {
    let Some(ep) = gpu() else { return };
    let runtime = ep.runtime();
    let kernel =
        GroupQueryAttentionKernel::new(runtime.clone(), 4, 2, None, true, false, -1, 0.0).unwrap();
    let packed_host = f32_tensor(
        &[1, 1, 16],
        &[
            1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 10., 10., 5., 6., 50., 60.,
        ],
    );
    let cache_k_host = f32_tensor(&[1, 2, 5, 2], &[0.0; 20]);
    let cache_v_host = f32_tensor(&[1, 2, 5, 2], &[0.0; 20]);
    let valid_seqlens = i32_tensor(&[1], &[2]);
    let total_host = i32_tensor(&[1], &[5]);
    let cos_host = f32_tensor(&[5, 1], &[1.0; 5]);
    let sin_host = f32_tensor(&[5, 1], &[0.0; 5]);
    let valid_positions = i64_tensor(&[1, 2], &[2, 0]);
    let packed = upload(&ep, &packed_host).unwrap();
    let mut cache_k = upload(&ep, &cache_k_host).unwrap();
    let mut cache_v = upload(&ep, &cache_v_host).unwrap();
    let seqlens = upload(&ep, &valid_seqlens).unwrap();
    let total = upload(&ep, &total_host).unwrap();
    let cos = upload(&ep, &cos_host).unwrap();
    let sin = upload(&ep, &sin_host).unwrap();
    let positions = upload(&ep, &valid_positions).unwrap();
    let mut output = ep.allocate(8 * std::mem::size_of::<f32>(), 256).unwrap();

    let mut capture_invalid = |bad_seqlens: Option<i32>, bad_position: Option<i64>| -> u32 {
        // Explicit host reset returns the latch to a clean slate for each case.
        runtime.reset_capture_error().unwrap();
        overwrite(&ep, &seqlens, &valid_seqlens).unwrap();
        overwrite(&ep, &positions, &valid_positions).unwrap();
        execute_in_place_packed_f32_gqa(
            &kernel,
            &ep,
            &packed,
            &mut cache_k,
            &mut cache_v,
            &seqlens,
            &total,
            &cos,
            &sin,
            &positions,
            &mut output,
            1,
        )
        .unwrap();
        let kernels: [&dyn Kernel; 1] = [&kernel];
        runtime.begin_graph_capture(&kernels).unwrap();
        execute_in_place_packed_f32_gqa(
            &kernel,
            &ep,
            &packed,
            &mut cache_k,
            &mut cache_v,
            &seqlens,
            &total,
            &cos,
            &sin,
            &positions,
            &mut output,
            1,
        )
        .unwrap();
        runtime.end_graph_capture().unwrap();
        let cache_k_before = read_bytes(&ep, &cache_k, 20 * std::mem::size_of::<f32>()).unwrap();
        let cache_v_before = read_bytes(&ep, &cache_v, 20 * std::mem::size_of::<f32>()).unwrap();

        if let Some(value) = bad_seqlens {
            overwrite(&ep, &seqlens, &i32_tensor(&[1], &[value])).unwrap();
        }
        if let Some(value) = bad_position {
            overwrite(&ep, &positions, &i64_tensor(&[1], &[value])).unwrap();
        }
        runtime.replay_graph().unwrap();
        runtime.synchronize().unwrap();
        // Sentinel skip is preserved: the out-of-range step writes no KV row and
        // can never alias another sequence's cache.
        assert_eq!(
            read_bytes(&ep, &cache_k, cache_k_before.len()).unwrap(),
            cache_k_before
        );
        assert_eq!(
            read_bytes(&ep, &cache_v, cache_v_before.len()).unwrap(),
            cache_v_before
        );
        // Detection-before-consumption: the violation is visible at the replay
        // sync (the same point the host reads logits), so the plausible-looking
        // zero token would be rejected before it is ever consumed.
        let bits = runtime.check_capture_error().unwrap();
        assert!(
            bits != 0,
            "captured out-of-range replay did not latch an error"
        );
        assert!(runtime.reset_graph().unwrap());
        bits
    };

    let overflow = capture_invalid(Some(i32::MAX), None);
    assert_ne!(overflow & GQA_CAPTURE_ERROR_TOTAL_OVERFLOW, 0);
    assert!(gqa_capture_error_description(overflow)
        .unwrap()
        .contains("seqlens_k + 1 overflows int32"));

    let negative = capture_invalid(Some(-1), None);
    assert_ne!(negative & GQA_CAPTURE_ERROR_PAST_NEGATIVE, 0);
    assert_ne!(negative & GQA_CAPTURE_ERROR_QUERY_NEGATIVE, 0);
    let negative = gqa_capture_error_description(negative).unwrap();
    assert!(negative.contains("shorter than current key sequence"));
    assert!(negative.contains("shorter than current query sequence"));

    let capacity = capture_invalid(Some(6), None);
    assert_ne!(capacity & GQA_CAPTURE_ERROR_PAST_CAPACITY, 0);
    assert!(gqa_capture_error_description(capacity)
        .unwrap()
        .contains("effective past length exceeds past cache extent"));

    let position = capture_invalid(None, Some(i64::MAX));
    assert_ne!(position & GQA_CAPTURE_ERROR_POSITION, 0);
    assert!(gqa_capture_error_description(position)
        .unwrap()
        .contains("rotary position exceeds cache rows"));

    // Leave the latch clean for any later kernel sharing this runtime.
    runtime.reset_capture_error().unwrap();

    for buffer in [
        output, positions, sin, cos, total, seqlens, cache_v, cache_k, packed,
    ] {
        ep.deallocate(buffer).unwrap();
    }
}

#[test]
fn gqa_gpu_capture_error_latches_until_reset_without_resuming_over_hole() {
    let Some(ep) = gpu() else { return };
    let runtime = ep.runtime();
    let kernel =
        GroupQueryAttentionKernel::new(runtime.clone(), 4, 2, None, true, false, -1, 0.0).unwrap();
    let eager_kernel =
        GroupQueryAttentionKernel::new(runtime.clone(), 4, 2, None, true, false, -1, 0.0).unwrap();

    let packed_host = f32_tensor(
        &[1, 1, 16],
        &[
            1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 10., 10., 5., 6., 50., 60.,
        ],
    );
    let cache_k_host = f32_tensor(
        &[1, 2, 5, 2],
        &[
            1., 0., 0., 1., 91., 92., 93., 94., 95., 96., 10., 0., 0., 10., 81., 82., 83., 84.,
            85., 86.,
        ],
    );
    let cache_v_host = f32_tensor(
        &[1, 2, 5, 2],
        &[
            1., 2., 3., 4., 71., 72., 73., 74., 75., 76., 10., 20., 30., 40., 61., 62., 63., 64.,
            65., 66.,
        ],
    );
    let seqlens_host = i32_tensor(&[1], &[1]);
    let total_host = i32_tensor(&[1], &[5]);
    let cos_host = f32_tensor(&[5, 1], &[1.0, 0.0, -1.0, 0.5, -0.5]);
    let sin_host = f32_tensor(&[5, 1], &[0.0, 1.0, 0.0, 0.866_025_4, 0.866_025_4]);
    let positions_host = i64_tensor(&[1, 2], &[1, 0]);

    let packed = upload(&ep, &packed_host).unwrap();
    let mut cache_k = upload(&ep, &cache_k_host).unwrap();
    let mut cache_v = upload(&ep, &cache_v_host).unwrap();
    let seqlens = upload(&ep, &seqlens_host).unwrap();
    let total = upload(&ep, &total_host).unwrap();
    let cos = upload(&ep, &cos_host).unwrap();
    let sin = upload(&ep, &sin_host).unwrap();
    let positions = upload(&ep, &positions_host).unwrap();
    let output_bytes = 8 * std::mem::size_of::<f32>();
    let mut output = ep.allocate(output_bytes, 256).unwrap();

    let mut eager_cache_k = upload(&ep, &cache_k_host).unwrap();
    let mut eager_cache_v = upload(&ep, &cache_v_host).unwrap();
    let eager_seqlens = upload(&ep, &seqlens_host).unwrap();
    let eager_positions = upload(&ep, &positions_host).unwrap();
    let mut eager_output = ep.allocate(output_bytes, 256).unwrap();

    let cache_bytes = 20 * std::mem::size_of::<f32>();
    runtime.reset_capture_error().unwrap();

    // Warm the fixed decode signature, then capture it.
    execute_in_place_packed_f32_gqa(
        &kernel,
        &ep,
        &packed,
        &mut cache_k,
        &mut cache_v,
        &seqlens,
        &total,
        &cos,
        &sin,
        &positions,
        &mut output,
        1,
    )
    .unwrap();
    execute_in_place_packed_f32_gqa(
        &eager_kernel,
        &ep,
        &packed,
        &mut eager_cache_k,
        &mut eager_cache_v,
        &eager_seqlens,
        &total,
        &cos,
        &sin,
        &eager_positions,
        &mut eager_output,
        1,
    )
    .unwrap();
    assert!(kernel.cuda_graph_compatible());

    let kernels: [&dyn Kernel; 1] = [&kernel];
    runtime.begin_graph_capture(&kernels).unwrap();
    execute_in_place_packed_f32_gqa(
        &kernel,
        &ep,
        &packed,
        &mut cache_k,
        &mut cache_v,
        &seqlens,
        &total,
        &cos,
        &sin,
        &positions,
        &mut output,
        1,
    )
    .unwrap();
    runtime.end_graph_capture().unwrap();

    // A normal in-range advancing replay stays bit-identical to the eager kernel
    // and never latches an error.
    for step in 2_i32..=3 {
        let seqlens_step = i32_tensor(&[1], &[step]);
        let positions_step = i64_tensor(&[1], &[i64::from(step)]);
        overwrite(&ep, &seqlens, &seqlens_step).unwrap();
        overwrite(&ep, &positions, &positions_step).unwrap();
        overwrite(&ep, &eager_seqlens, &seqlens_step).unwrap();
        overwrite(&ep, &eager_positions, &positions_step).unwrap();
        execute_in_place_packed_f32_gqa(
            &eager_kernel,
            &ep,
            &packed,
            &mut eager_cache_k,
            &mut eager_cache_v,
            &eager_seqlens,
            &total,
            &cos,
            &sin,
            &eager_positions,
            &mut eager_output,
            1,
        )
        .unwrap();
        runtime.replay_graph().unwrap();
        assert_eq!(runtime.check_capture_error().unwrap(), 0);
        assert_eq!(
            read_bytes(&ep, &output, output_bytes).unwrap(),
            read_bytes(&ep, &eager_output, output_bytes).unwrap(),
            "in-range replay diverged from eager at step {step}"
        );
        assert_eq!(
            read_bytes(&ep, &cache_k, cache_bytes).unwrap(),
            read_bytes(&ep, &eager_cache_k, cache_bytes).unwrap(),
            "in-range replay KV diverged from eager at step {step}"
        );
    }

    // Freeze the current captured KV state, then inject an out-of-range length.
    let frozen_k = read_bytes(&ep, &cache_k, cache_bytes).unwrap();
    let frozen_v = read_bytes(&ep, &cache_v, cache_bytes).unwrap();
    overwrite(&ep, &seqlens, &i32_tensor(&[1], &[i32::MAX])).unwrap();
    runtime.replay_graph().unwrap();
    runtime.synchronize().unwrap();
    // Detection-before-consumption: the latch is set at the replay sync.
    let tripped = runtime.check_capture_error().unwrap();
    assert_ne!(tripped & GQA_CAPTURE_ERROR_TOTAL_OVERFLOW, 0);
    // Sentinel skip: no KV row was written.
    assert_eq!(read_bytes(&ep, &cache_k, cache_bytes).unwrap(), frozen_k);
    assert_eq!(read_bytes(&ep, &cache_v, cache_bytes).unwrap(), frozen_v);

    // The crux: restore an otherwise-valid length and replay again. Because the
    // latch is sticky, the poison propagates — the step still skips its KV write
    // rather than resuming over the hole, and the error remains detectable.
    overwrite(&ep, &seqlens, &i32_tensor(&[1], &[4])).unwrap();
    overwrite(&ep, &positions, &i64_tensor(&[1], &[4])).unwrap();
    runtime.replay_graph().unwrap();
    runtime.synchronize().unwrap();
    assert_ne!(
        runtime.check_capture_error().unwrap(),
        0,
        "latch cleared itself; a later step could resume over a skipped KV row"
    );
    assert_eq!(
        read_bytes(&ep, &cache_k, cache_bytes).unwrap(),
        frozen_k,
        "poisoned replay resumed a KV write over a hole"
    );
    assert_eq!(read_bytes(&ep, &cache_v, cache_bytes).unwrap(), frozen_v);

    // Explicit host reset clears the latch; a fresh in-range replay resumes
    // normal operation and advances the KV cache again.
    runtime.reset_capture_error().unwrap();
    assert_eq!(runtime.check_capture_error().unwrap(), 0);
    runtime.replay_graph().unwrap();
    runtime.synchronize().unwrap();
    assert_eq!(runtime.check_capture_error().unwrap(), 0);
    assert_ne!(
        read_bytes(&ep, &cache_k, cache_bytes).unwrap(),
        frozen_k,
        "cache did not advance after the latch was reset"
    );

    let _ = runtime.reset_graph();
    runtime.reset_capture_error().unwrap();

    for buffer in [
        eager_output,
        eager_positions,
        eager_seqlens,
        eager_cache_v,
        eager_cache_k,
        output,
        positions,
        sin,
        cos,
        total,
        seqlens,
        cache_v,
        cache_k,
        packed,
    ] {
        ep.deallocate(buffer).unwrap();
    }
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
    let outputs = run_available(run(
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
    ))
    .unwrap();
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

// Feasibility probe for shared-buffer continuous batching: two sequences with
// DIFFERENT valid KV lengths must share one batched GQA run without cross-row
// contamination. Each row is written into its own fixed-capacity cache slice at
// its own per-row write offset (seqlens_k[row]), and a single shared
// total_sequence_length scalar (= max valid length) is used for the whole batch.
// If the CUDA GQA kernel honors per-row seqlens_k for both attention masking and
// the present-cache write position, the batched output for each row must equal
// that row run alone. This validates that BatchedSharedBufferDecodeSession is
// possible.
#[test]
fn gqa_gpu_shared_buffer_batches_rows_of_different_lengths() {
    let Some(ep) = gpu() else { return };

    // Row 0: past_len = 2, capacity = 5.
    let q0 = [1., 0., 1., 0., 0., 1., 0., 1.];
    let past_k0 = [
        1., 0., 0., 1., 91., 92., 93., 94., 95., 96., 10., 0., 0., 10., 81., 82., 83., 84., 85.,
        86.,
    ];
    let past_v0 = [
        1., 2., 3., 4., 71., 72., 73., 74., 75., 76., 10., 20., 30., 40., 61., 62., 63., 64., 65.,
        66.,
    ];
    let cur_k0 = [1., 1., 10., 10.];
    let cur_v0 = [5., 6., 50., 60.];

    // Row 1: past_len = 3 (different length), capacity = 5, different data.
    let q1 = [0., 1., 1., 1., 1., 0., 1., 1.];
    let past_k1 = [
        2., 0., 0., 2., 1., 1., 44., 44., 55., 55., 3., 0., 0., 3., 2., 2., 66., 66., 77., 77.,
    ];
    let past_v1 = [
        1., 1., 2., 2., 3., 3., 88., 88., 99., 99., 4., 4., 5., 5., 6., 6., 11., 11., 22., 22.,
    ];
    let cur_k1 = [1., 0., 0., 1.];
    let cur_v1 = [7., 7., 70., 70.];

    // Standalone runs (ground truth from the real kernel), one row at a time.
    let out_row0 = run_available(run(
        &ep,
        &attrs(&[]),
        &base_inputs(
            &[1, 1, 8],
            &q0,
            &[1, 1, 4],
            &cur_k0,
            &cur_v0,
            Some((&[1, 2, 5, 2], &past_k0)),
            Some((&[1, 2, 5, 2], &past_v0)),
            &[2],
            3,
        ),
        &[vec![1, 1, 8], vec![1, 2, 5, 2], vec![1, 2, 5, 2]],
    ))
    .unwrap();
    let out_row1 = run(
        &ep,
        &attrs(&[]),
        &base_inputs(
            &[1, 1, 8],
            &q1,
            &[1, 1, 4],
            &cur_k1,
            &cur_v1,
            Some((&[1, 2, 5, 2], &past_k1)),
            Some((&[1, 2, 5, 2], &past_v1)),
            &[3],
            4,
        ),
        &[vec![1, 1, 8], vec![1, 2, 5, 2], vec![1, 2, 5, 2]],
    )
    .unwrap();

    // Batched run: stack both rows, per-row seqlens_k = [2, 3], shared
    // total_sequence_length = max(seqlens_k) + 1 = 4.
    let cat = |a: &[f32], b: &[f32]| -> Vec<f32> { a.iter().chain(b).copied().collect() };
    let q = cat(&q0, &q1);
    let cur_k = cat(&cur_k0, &cur_k1);
    let cur_v = cat(&cur_v0, &cur_v1);
    let past_k = cat(&past_k0, &past_k1);
    let past_v = cat(&past_v0, &past_v1);
    let batched = run(
        &ep,
        &attrs(&[]),
        &base_inputs(
            &[2, 1, 8],
            &q,
            &[2, 1, 4],
            &cur_k,
            &cur_v,
            Some((&[2, 2, 5, 2], &past_k)),
            Some((&[2, 2, 5, 2], &past_v)),
            &[2, 3],
            4,
        ),
        &[vec![2, 1, 8], vec![2, 2, 5, 2], vec![2, 2, 5, 2]],
    )
    .unwrap();

    // Each batched row's attention output must equal that row run alone.
    close(&batched[0][0..8], &out_row0[0]);
    close(&batched[0][8..16], &out_row1[0]);
    // And each row's present KV cache (written at its own offset) must match too.
    close(&batched[1][0..20], &out_row0[1]);
    close(&batched[1][20..40], &out_row1[1]);
    close(&batched[2][0..20], &out_row0[2]);
    close(&batched[2][20..40], &out_row1[2]);
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
    let outputs = run_available(run(
        &ep,
        &attrs(&[("do_rotary", Attribute::Int(1))]),
        &inputs,
        &[vec![1, 2, 8], vec![1, 2, 2, 2]],
    ))
    .unwrap();
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
fn gqa_gpu_partial_rope_supports_all_query_dtypes() {
    const HEAD_DIM: usize = 6;
    const ROTARY_DIM: usize = 4;
    let Some(ep) = gpu() else { return };
    let positions = [1_i64, 2];
    let mut cos = Vec::with_capacity(3 * ROTARY_DIM / 2);
    let mut sin = Vec::with_capacity(cos.capacity());
    for position in 0..3 {
        for lane in 0..ROTARY_DIM / 2 {
            let angle = (position + 1) as f32 * (lane + 1) as f32 * 0.2;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    let raw_q = fill(2 * 4 * HEAD_DIM, 0x401);
    let raw_k = fill(2 * 2 * HEAD_DIM, 0x402);
    let raw_v = fill(2 * 2 * HEAD_DIM, 0x403);

    for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
        for interleaved in [false, true] {
            let q = quantize(&raw_q, dtype);
            let k = quantize(&raw_k, dtype);
            let v = quantize(&raw_v, dtype);
            let mut inputs = base_inputs_dtype(
                dtype,
                &[1, 2, 4 * HEAD_DIM],
                &q,
                &[1, 2, 2 * HEAD_DIM],
                &k,
                &v,
                None,
                None,
                &[1],
                2,
            );
            inputs.push(Some(f32_tensor(&[3, ROTARY_DIM / 2], &cos)));
            inputs.push(Some(f32_tensor(&[3, ROTARY_DIM / 2], &sin)));
            inputs.push(Some(i64_tensor(&[1, 2], &positions)));
            let outputs = run_available(run(
                &ep,
                &attrs(&[
                    ("do_rotary", Attribute::Int(1)),
                    ("rotary_interleaved", Attribute::Int(interleaved.into())),
                ]),
                &inputs,
                &[
                    vec![1, 2, 4 * HEAD_DIM],
                    vec![1, 2, 2, HEAD_DIM],
                    vec![1, 2, 2, HEAD_DIM],
                ],
            ))
            .unwrap();

            let expected_bsh = quantize(
                &rotate_target(
                    &k,
                    2,
                    2,
                    (HEAD_DIM, ROTARY_DIM),
                    interleaved,
                    &[1, 2],
                    (&cos, &sin),
                ),
                dtype,
            );
            let mut expected_bnsh = vec![0.0; expected_bsh.len()];
            for sequence in 0..2 {
                for head in 0..2 {
                    let src = (sequence * 2 + head) * HEAD_DIM;
                    let dst = (head * 2 + sequence) * HEAD_DIM;
                    expected_bnsh[dst..dst + HEAD_DIM]
                        .copy_from_slice(&expected_bsh[src..src + HEAD_DIM]);
                }
            }
            let (atol, rtol) = parity_tolerances(dtype);
            assert_close(&outputs[1], &expected_bnsh, atol, rtol);
            for sequence in 0..2 {
                for head in 0..2 {
                    let got = (head * 2 + sequence) * HEAD_DIM;
                    let source = (sequence * 2 + head) * HEAD_DIM;
                    assert_eq!(
                        &outputs[1][got + ROTARY_DIM..got + HEAD_DIM],
                        &k[source + ROTARY_DIM..source + HEAD_DIM],
                        "{dtype:?} partial RoPE tail must pass through unchanged"
                    );
                }
            }
        }
    }
}

#[test]
fn gqa_gpu_partial_rope_rejects_invalid_cache_shapes_actionably() {
    let Some(ep) = gpu() else { return };
    let q = [0.0; 16];
    let kv = [0.0; 8];
    let base = base_inputs(&[1, 1, 16], &q, &[1, 1, 8], &kv, &kv, None, None, &[0], 1);
    let attrs = attrs(&[("do_rotary", Attribute::Int(1))]);
    let output_shapes = [vec![1, 1, 16], vec![1, 2, 1, 4], vec![1, 2, 1, 4]];

    let mut too_wide = base.clone();
    too_wide.push(Some(f32_tensor(&[2, 3], &[1.0; 6])));
    too_wide.push(Some(f32_tensor(&[2, 3], &[0.0; 6])));
    let error = run(&ep, &attrs, &too_wide, &output_shapes)
        .expect_err("rotary_dim wider than head_size must be rejected");
    let message = format!("{error}");
    assert!(message.contains("rotary_dim derived from cos_cache width 3 is 6"));
    assert!(message.contains("2 <= rotary_dim <= head_size=4"));

    let mut mismatched = base;
    mismatched.push(Some(f32_tensor(&[2, 1], &[1.0; 2])));
    mismatched.push(Some(f32_tensor(&[2, 2], &[0.0; 4])));
    let error = run(&ep, &attrs, &mismatched, &output_shapes)
        .expect_err("different sin/cos cache shapes must be rejected");
    let message = format!("{error}");
    assert!(message.contains("sin_cache shape [2, 2] must exactly match cos_cache shape [2, 1]"));
    assert!(message.contains("same rotary_dim"));
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
    let outputs = run_available(run(
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
    ))
    .unwrap();
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
fn gqa_gpu_physical_capacity_can_exceed_valid_prefix() {
    let Some(ep) = gpu() else { return };
    const VALID: usize = 5;
    const PAST: usize = VALID - 1;
    const CAPACITY: usize = 128;
    let q = [0.2, -0.1, 0.3, 0.4, -0.2, 0.5, 0.1, -0.3];
    let current_k = [0.7, -0.4, 0.2, 0.6];
    let current_v = [1.5, -0.5, 0.25, 2.0];
    let compact_k = (0..2 * PAST * 2)
        .map(|index| index as f32 * 0.03 - 0.2)
        .collect::<Vec<_>>();
    let compact_v = (0..2 * PAST * 2)
        .map(|index| index as f32 * -0.02 + 0.4)
        .collect::<Vec<_>>();
    let mut capacity_k = vec![0.0; 2 * CAPACITY * 2];
    let mut capacity_v = vec![0.0; 2 * CAPACITY * 2];
    for head in 0..2 {
        for position in 0..PAST {
            for dim in 0..2 {
                let compact = (head * PAST + position) * 2 + dim;
                let capacity = (head * CAPACITY + position) * 2 + dim;
                capacity_k[capacity] = compact_k[compact];
                capacity_v[capacity] = compact_v[compact];
            }
        }
    }

    let exact = run_available(run(
        &ep,
        &attrs(&[]),
        &base_inputs(
            &[1, 1, 8],
            &q,
            &[1, 1, 4],
            &current_k,
            &current_v,
            Some((&[1, 2, PAST, 2], &compact_k)),
            Some((&[1, 2, PAST, 2], &compact_v)),
            &[(VALID - 1) as i32],
            VALID as i32,
        ),
        &[vec![1, 1, 8]],
    ))
    .unwrap();
    let fixed = run_available(run(
        &ep,
        &attrs(&[]),
        &base_inputs(
            &[1, 1, 8],
            &q,
            &[1, 1, 4],
            &current_k,
            &current_v,
            Some((&[1, 2, CAPACITY, 2], &capacity_k)),
            Some((&[1, 2, CAPACITY, 2], &capacity_v)),
            &[(VALID - 1) as i32],
            CAPACITY as i32,
        ),
        &[vec![1, 1, 8]],
    ))
    .unwrap();
    close(&fixed[0], &exact[0]);
}

#[test]
fn gqa_gpu_packed_qkv_rope_decode_appends_in_place_across_steps() {
    const NUM_HEADS: usize = 14;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 64;
    const CAPACITY: usize = 64;
    const PREFILL: usize = 3;
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

    let make_token = |position: usize| {
        let query: Vec<f32> = (0..NUM_HEADS * HEAD_DIM)
            .map(|index| ((index * 13 + position * 7) % 101) as f32 * 0.002 - 0.1)
            .collect();
        let key: Vec<f32> = (0..KV_HEADS * HEAD_DIM)
            .map(|index| ((index * 11 + position * 5) % 67) as f32 * 0.003 - 0.08)
            .collect();
        let value: Vec<f32> = (0..KV_HEADS * HEAD_DIM)
            .map(|index| ((index * 17 + position * 3) % 79) as f32 * 0.004 - 0.12)
            .collect();
        (query, key, value)
    };

    let mut prefill_query = Vec::with_capacity(PREFILL * NUM_HEADS * HEAD_DIM);
    let mut prefill_key = Vec::with_capacity(PREFILL * KV_HEADS * HEAD_DIM);
    let mut prefill_value = Vec::with_capacity(PREFILL * KV_HEADS * HEAD_DIM);
    let mut packed = Vec::with_capacity(PREFILL * (NUM_HEADS + 2 * KV_HEADS) * HEAD_DIM);
    for position in 0..PREFILL {
        let (query, key, value) = make_token(position);
        packed.extend_from_slice(&query);
        packed.extend_from_slice(&key);
        packed.extend_from_slice(&value);
        prefill_query.extend_from_slice(&query);
        prefill_key.extend_from_slice(&key);
        prefill_value.extend_from_slice(&value);
    }

    let prefill_positions: Vec<_> = (0..PREFILL).collect();
    let rotated_query = rotate_target(
        &prefill_query,
        PREFILL,
        NUM_HEADS,
        (HEAD_DIM, HEAD_DIM),
        false,
        &prefill_positions,
        (&cos, &sin),
    );
    let rotated_key = rotate_target(
        &prefill_key,
        PREFILL,
        KV_HEADS,
        (HEAD_DIM, HEAD_DIM),
        false,
        &prefill_positions,
        (&cos, &sin),
    );
    let cache_len = KV_HEADS * CAPACITY * HEAD_DIM;
    let mut expected_k = vec![0.0; cache_len];
    let mut expected_v = vec![0.0; cache_len];
    for position in 0..PREFILL {
        for head in 0..KV_HEADS {
            for dim in 0..HEAD_DIM {
                let cache_index = (head * CAPACITY + position) * HEAD_DIM + dim;
                let source_index = (position * KV_HEADS + head) * HEAD_DIM + dim;
                expected_k[cache_index] = rotated_key[source_index];
                expected_v[cache_index] = prefill_value[source_index];
            }
        }
    }
    let expected_output = target_attention_reference(
        &rotated_query,
        &expected_k,
        &expected_v,
        PREFILL,
        0,
        CAPACITY,
    );
    let prefill_position_ids: Vec<_> = prefill_positions.iter().map(|&x| x as i64).collect();
    let mut step = run_available(run_packed_step(
        &ep,
        &packed,
        PREFILL,
        None,
        None,
        0,
        PREFILL,
        CAPACITY,
        &cos,
        &sin,
        &prefill_position_ids,
    ))
    .unwrap();
    close(&step.output, &expected_output);
    close(&step.key, &expected_k);
    close(&step.value, &expected_v);
    let (mut max_attention_abs, mut max_attention_ulp) =
        error_metrics(&step.output, &expected_output);
    let (mut max_rope_abs, mut max_rope_ulp) = error_metrics(&step.key, &expected_k);

    for position in PREFILL..CAPACITY {
        let (query, key, value) = make_token(position);
        let mut packed = Vec::with_capacity((NUM_HEADS + 2 * KV_HEADS) * HEAD_DIM);
        packed.extend_from_slice(&query);
        packed.extend_from_slice(&key);
        packed.extend_from_slice(&value);
        let rotated_query = rotate_target(
            &query,
            1,
            NUM_HEADS,
            (HEAD_DIM, HEAD_DIM),
            false,
            &[position],
            (&cos, &sin),
        );
        let rotated_key = rotate_target(
            &key,
            1,
            KV_HEADS,
            (HEAD_DIM, HEAD_DIM),
            false,
            &[position],
            (&cos, &sin),
        );
        for head in 0..KV_HEADS {
            for dim in 0..HEAD_DIM {
                let cache_index = (head * CAPACITY + position) * HEAD_DIM + dim;
                expected_k[cache_index] = rotated_key[head * HEAD_DIM + dim];
                expected_v[cache_index] = value[head * HEAD_DIM + dim];
            }
        }
        let expected_output = target_attention_reference(
            &rotated_query,
            &expected_k,
            &expected_v,
            1,
            position,
            CAPACITY,
        );
        let key_ptr = step.cache_k.as_ptr();
        let value_ptr = step.cache_v.as_ptr();
        step = run_available(run_packed_step(
            &ep,
            &packed,
            1,
            Some(step.cache_k),
            Some(step.cache_v),
            position,
            position + 1,
            CAPACITY,
            &cos,
            &sin,
            &[position as i64],
        ))
        .unwrap();
        assert_eq!(step.cache_k.as_ptr(), key_ptr);
        assert_eq!(step.cache_v.as_ptr(), value_ptr);
        close(&step.output, &expected_output);
        close(&step.key, &expected_k);
        close(&step.value, &expected_v);
        let (attention_abs, attention_ulp) = error_metrics(&step.output, &expected_output);
        let (rope_abs, rope_ulp) = error_metrics(&step.key, &expected_k);
        max_attention_abs = max_attention_abs.max(attention_abs);
        max_attention_ulp = max_attention_ulp.max(attention_ulp);
        max_rope_abs = max_rope_abs.max(rope_abs);
        max_rope_ulp = max_rope_ulp.max(rope_ulp);
    }

    eprintln!(
        "64-token GQA/RoPE CPU-reference: rope max_abs_diff={max_rope_abs:e} max_ulp_diff={max_rope_ulp}; attention max_abs_diff={max_attention_abs:e} max_ulp_diff={max_attention_ulp}"
    );
    assert_eq!(
        max_rope_abs, 0.0,
        "RoPE must match CPU operation order exactly across decode"
    );
    assert!(
        max_attention_abs <= 1e-6,
        "GQA error accumulated across decode: {max_attention_abs:e}"
    );
    ep.deallocate(step.cache_k).unwrap();
    ep.deallocate(step.cache_v).unwrap();
}

#[test]
fn gqa_gpu_partial_rope_rotates_prefix_and_preserves_tail() {
    const NUM_HEADS: usize = 14;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 64;
    const ROTARY_DIM: usize = 32;
    const CAPACITY: usize = 4;
    let Some(ep) = gpu() else { return };

    let mut cos = Vec::with_capacity(CAPACITY * ROTARY_DIM / 2);
    let mut sin = Vec::with_capacity(cos.capacity());
    for position in 0..CAPACITY {
        for dim in 0..ROTARY_DIM / 2 {
            let angle = (position + 1) as f32 * (dim + 1) as f32 * 0.01;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    let make_token = |position: usize| {
        let query = fill(NUM_HEADS * HEAD_DIM, 0x100 + position as u64);
        let key = fill(KV_HEADS * HEAD_DIM, 0x200 + position as u64);
        let value = fill(KV_HEADS * HEAD_DIM, 0x300 + position as u64);
        let mut packed = Vec::with_capacity((NUM_HEADS + 2 * KV_HEADS) * HEAD_DIM);
        packed.extend_from_slice(&query);
        packed.extend_from_slice(&key);
        packed.extend_from_slice(&value);
        (query, key, value, packed)
    };

    let cache_len = KV_HEADS * CAPACITY * HEAD_DIM;
    let mut expected_k = vec![0.0; cache_len];
    let mut expected_v = vec![0.0; cache_len];

    let (query0, key0, value0, packed0) = make_token(0);
    let rotated_query0 = rotate_target(
        &query0,
        1,
        NUM_HEADS,
        (HEAD_DIM, ROTARY_DIM),
        false,
        &[1],
        (&cos, &sin),
    );
    let rotated_key0 = rotate_target(
        &key0,
        1,
        KV_HEADS,
        (HEAD_DIM, ROTARY_DIM),
        false,
        &[1],
        (&cos, &sin),
    );
    for head in 0..KV_HEADS {
        let cache_base = head * CAPACITY * HEAD_DIM;
        let source_base = head * HEAD_DIM;
        expected_k[cache_base..cache_base + HEAD_DIM]
            .copy_from_slice(&rotated_key0[source_base..source_base + HEAD_DIM]);
        expected_v[cache_base..cache_base + HEAD_DIM]
            .copy_from_slice(&value0[source_base..source_base + HEAD_DIM]);
    }
    let expected_output0 =
        target_attention_reference(&rotated_query0, &expected_k, &expected_v, 1, 0, CAPACITY);
    let mut step = run_available(run_packed_step(
        &ep,
        &packed0,
        1,
        None,
        None,
        0,
        1,
        CAPACITY,
        &cos,
        &sin,
        &[1],
    ))
    .unwrap();
    assert_eq!(
        step.key, expected_k,
        "partial prefill RoPE must be bit-exact"
    );
    close(&step.output, &expected_output0);

    let (query1, key1, value1, packed1) = make_token(1);
    let rotated_query1 = rotate_target(
        &query1,
        1,
        NUM_HEADS,
        (HEAD_DIM, ROTARY_DIM),
        false,
        &[2],
        (&cos, &sin),
    );
    let rotated_key1 = rotate_target(
        &key1,
        1,
        KV_HEADS,
        (HEAD_DIM, ROTARY_DIM),
        false,
        &[2],
        (&cos, &sin),
    );
    for head in 0..KV_HEADS {
        let cache_base = (head * CAPACITY + 1) * HEAD_DIM;
        let source_base = head * HEAD_DIM;
        expected_k[cache_base..cache_base + HEAD_DIM]
            .copy_from_slice(&rotated_key1[source_base..source_base + HEAD_DIM]);
        expected_v[cache_base..cache_base + HEAD_DIM]
            .copy_from_slice(&value1[source_base..source_base + HEAD_DIM]);
    }
    let expected_output1 =
        target_attention_reference(&rotated_query1, &expected_k, &expected_v, 1, 1, CAPACITY);
    let key_ptr = step.cache_k.as_ptr();
    let value_ptr = step.cache_v.as_ptr();
    step = run_available(run_packed_step(
        &ep,
        &packed1,
        1,
        Some(step.cache_k),
        Some(step.cache_v),
        1,
        2,
        CAPACITY,
        &cos,
        &sin,
        &[2],
    ))
    .unwrap();
    assert_eq!(step.cache_k.as_ptr(), key_ptr);
    assert_eq!(step.cache_v.as_ptr(), value_ptr);
    assert_eq!(
        step.key, expected_k,
        "fused partial RoPE must rotate only the prefix"
    );
    close(&step.output, &expected_output1);
    for head in 0..KV_HEADS {
        let cache_base = (head * CAPACITY + 1) * HEAD_DIM;
        let source_base = head * HEAD_DIM;
        assert_eq!(
            &step.key[cache_base + ROTARY_DIM..cache_base + HEAD_DIM],
            &key1[source_base + ROTARY_DIM..source_base + HEAD_DIM],
            "partial RoPE tail must pass through unchanged"
        );
    }

    ep.deallocate(step.cache_k).unwrap();
    ep.deallocate(step.cache_v).unwrap();
}

#[test]
fn gqa_gpu_fused_decode_prep_matches_unfused_bit_exactly() {
    // The fused single-token decode-prep launch must be byte-for-byte identical
    // to the unfused split+transpose+append+RoPE chain it replaces. Runs the
    // same decode step through both paths (fusion on vs. forced off) on cloned
    // device state and asserts the output and the appended KV cache match
    // exactly. Uses the 4-head/2-kv/dim-2/capacity-5 fixed decode signature.
    let Some(ep) = gpu() else { return };
    let runtime = ep.runtime();
    let fused =
        GroupQueryAttentionKernel::new(runtime.clone(), 4, 2, None, true, false, -1, 0.0).unwrap();
    let unfused = GroupQueryAttentionKernel::new(runtime.clone(), 4, 2, None, true, false, -1, 0.0)
        .unwrap()
        .with_prep_fusion_disabled(true);

    let packed_host = f32_tensor(&[1, 1, 16], &fill(16, 0x51));
    let cache_k_host = f32_tensor(&[1, 2, 5, 2], &fill(20, 0x71));
    let cache_v_host = f32_tensor(&[1, 2, 5, 2], &fill(20, 0x91));
    // seqlens_k = 1 -> valid length 2, so past_len = 1 and the new token lands
    // at cache row 1; total_sequence_length carries the physical capacity.
    let seqlens_host = i32_tensor(&[1], &[1]);
    let total_host = i32_tensor(&[1], &[5]);
    let cos_host = f32_tensor(&[5, 1], &[1.0, 0.5, -1.0, 0.25, -0.5]);
    let sin_host = f32_tensor(&[5, 1], &[0.0, 0.866_025_4, 0.0, 0.968_245_8, 0.866_025_4]);
    let positions_host = i64_tensor(&[1, 1], &[1]);

    let packed = upload(&ep, &packed_host).unwrap();
    let seqlens = upload(&ep, &seqlens_host).unwrap();
    let total = upload(&ep, &total_host).unwrap();
    let cos = upload(&ep, &cos_host).unwrap();
    let sin = upload(&ep, &sin_host).unwrap();
    let positions = upload(&ep, &positions_host).unwrap();

    let output_bytes = 8 * std::mem::size_of::<f32>();
    let cache_bytes = 20 * std::mem::size_of::<f32>();

    let run_once =
        |kernel: &GroupQueryAttentionKernel| -> (Vec<u8>, Vec<u8>, Vec<u8>, (Vec<i32>, Vec<i32>, Vec<i32>)) {
        let mut cache_k = upload(&ep, &cache_k_host).unwrap();
        let mut cache_v = upload(&ep, &cache_v_host).unwrap();
        let mut output = ep.allocate(output_bytes, 256).unwrap();
        execute_in_place_packed_f32_gqa(
            kernel,
            &ep,
            &packed,
            &mut cache_k,
            &mut cache_v,
            &seqlens,
            &total,
            &cos,
            &sin,
            &positions,
            &mut output,
            1,
        )
        .unwrap();
        runtime.synchronize().unwrap();
        let metadata = kernel.read_prepared_metadata_for_test(1).unwrap();
        let out = read_bytes(&ep, &output, output_bytes).unwrap();
        let ck = read_bytes(&ep, &cache_k, cache_bytes).unwrap();
        let cv = read_bytes(&ep, &cache_v, cache_bytes).unwrap();
        ep.deallocate(output).unwrap();
        ep.deallocate(cache_k).unwrap();
        ep.deallocate(cache_v).unwrap();
        (out, ck, cv, metadata)
    };

    let (fused_out, fused_k, fused_v, fused_metadata) = run_once(&fused);
    let (unfused_out, unfused_k, unfused_v, separate_metadata) = run_once(&unfused);

    assert_eq!(
        fused_metadata, separate_metadata,
        "metadata derived inside fused prep diverged from gqa_prepare_metadata"
    );
    assert_eq!(fused_metadata, (vec![2], vec![1], vec![1]));
    assert_eq!(
        fused_out, unfused_out,
        "fused decode output diverged from the unfused prep chain"
    );
    assert_eq!(
        fused_k, unfused_k,
        "fused decode present-K diverged from the unfused append+RoPE chain"
    );
    assert_eq!(
        fused_v, unfused_v,
        "fused decode present-V diverged from the unfused append chain"
    );

    for buffer in [positions, sin, cos, total, seqlens, packed] {
        ep.deallocate(buffer).unwrap();
    }
}

#[test]
fn gqa_gpu_rejected_features_return_clear_errors() {
    let Some(ep) = gpu() else { return };
    let mut registered = Node::new(NodeId(0), "GroupQueryAttention", vec![], vec![]);
    registered.domain = "com.microsoft".into();
    assert!(matches!(
        ep.supports_op(&registered, 1, &[], &[], &[]),
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
