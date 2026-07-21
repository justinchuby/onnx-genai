use half::f16;
use onnx_runtime_ep_api::{
    CaptureSupport, DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, Kernel, KernelMatch,
    TensorMut, TensorView,
};
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ep_cuda::{
    CudaExecutionProvider, GroupQueryAttentionBackend, GroupQueryAttentionKernel,
};
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
};

const BATCH: usize = 1;
const QUERY_HEADS: usize = 8;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 256;
const CACHE_CAPACITY: usize = 4096;
const ROTARY_ROWS: usize = CACHE_CAPACITY + 1;
const LOCAL_WINDOW: usize = 512;

fn typed_bytes<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: test data contains plain-old-data values with no padding.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
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

fn upload(ep: &CudaExecutionProvider, bytes: &[u8]) -> onnx_runtime_ep_api::Result<DeviceBuffer> {
    let buffer = ep.allocate(bytes.len().max(1), 256)?;
    if !bytes.is_empty() {
        // SAFETY: the allocation is at least `bytes.len()` bytes.
        unsafe {
            ep.runtime().htod(bytes, cuptr(buffer.as_ptr()))?;
        }
    }
    Ok(buffer)
}

fn read(
    ep: &CudaExecutionProvider,
    buffer: &DeviceBuffer,
    bytes: usize,
) -> onnx_runtime_ep_api::Result<Vec<u8>> {
    let mut host = vec![0_u8; bytes];
    // SAFETY: callers request exactly the initialized tensor extent.
    unsafe {
        ep.runtime().dtoh(&mut host, cuptr(buffer.as_ptr()))?;
    }
    Ok(host)
}

fn fp16_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| f16::from_bits(u16::from_ne_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
}

fn gqa_node() -> (Graph, NodeId, Vec<onnx_runtime_ir::Shape>, Vec<DataType>) {
    let specs = [
        (DataType::Float16, vec![1, 1, QUERY_HEADS * HEAD_DIM]),
        (DataType::Float16, vec![1, 0, KV_HEADS * HEAD_DIM]),
        (DataType::Float16, vec![1, 0, KV_HEADS * HEAD_DIM]),
        (
            DataType::Float16,
            vec![BATCH, KV_HEADS, CACHE_CAPACITY, HEAD_DIM],
        ),
        (
            DataType::Float16,
            vec![BATCH, KV_HEADS, CACHE_CAPACITY, HEAD_DIM],
        ),
        (DataType::Int32, vec![BATCH]),
        (DataType::Int32, vec![]),
        (DataType::Float16, vec![ROTARY_ROWS, HEAD_DIM / 2]),
        (DataType::Float16, vec![ROTARY_ROWS, HEAD_DIM / 2]),
    ];
    let mut graph = Graph::new();
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let inputs = specs
        .iter()
        .enumerate()
        .map(|(index, (dtype, shape))| {
            let value = graph.create_named_value(
                format!("input_{index}"),
                *dtype,
                static_shape(shape.iter().copied()),
            );
            graph.add_input(value);
            Some(value)
        })
        .collect();
    let output_specs = [
        vec![1, 1, QUERY_HEADS * HEAD_DIM],
        vec![BATCH, KV_HEADS, CACHE_CAPACITY, HEAD_DIM],
        vec![BATCH, KV_HEADS, CACHE_CAPACITY, HEAD_DIM],
    ];
    let outputs = output_specs
        .iter()
        .enumerate()
        .map(|(index, shape)| {
            graph.create_named_value(
                format!("output_{index}"),
                DataType::Float16,
                static_shape(shape.iter().copied()),
            )
        })
        .collect();
    let mut node = Node::new(NodeId(0), "GroupQueryAttention", inputs, outputs);
    node.domain = "com.microsoft".into();
    node.attributes
        .insert("num_heads".into(), Attribute::Int(QUERY_HEADS as i64));
    node.attributes
        .insert("kv_num_heads".into(), Attribute::Int(KV_HEADS as i64));
    node.attributes
        .insert("scale".into(), Attribute::Float(1.0));
    node.attributes
        .insert("do_rotary".into(), Attribute::Int(1));
    node.attributes.insert(
        "local_window_size".into(),
        Attribute::Int(LOCAL_WINDOW as i64),
    );
    let id = graph.insert_node(node);
    let shapes = specs
        .iter()
        .map(|(_, shape)| static_shape(shape.iter().copied()))
        .collect();
    let dtypes = specs.iter().map(|(dtype, _)| *dtype).collect();
    (graph, id, shapes, dtypes)
}

fn rotate_query(query: &[f16], cos: &[f16], sin: &[f16], position: usize) -> Vec<f32> {
    let half = HEAD_DIM / 2;
    let mut rotated = vec![0.0_f32; query.len()];
    for head in 0..QUERY_HEADS {
        let base = head * HEAD_DIM;
        for k in 0..half {
            let x0 = query[base + k].to_f32();
            let x1 = query[base + k + half].to_f32();
            let c = cos[position * half + k].to_f32();
            let s = sin[position * half + k].to_f32();
            rotated[base + k] = f16::from_f32(c * x0 - s * x1).to_f32();
            rotated[base + k + half] = f16::from_f32(s * x0 + c * x1).to_f32();
        }
    }
    rotated
}

fn cpu_reference(query: &[f32], key: &[f16], value: &[f16]) -> Vec<f32> {
    let first_key = CACHE_CAPACITY - LOCAL_WINDOW;
    let mut output = vec![0.0_f32; QUERY_HEADS * HEAD_DIM];
    for head in 0..QUERY_HEADS {
        let q_base = head * HEAD_DIM;
        let mut scores = vec![0.0_f64; LOCAL_WINDOW];
        let mut maximum = f64::NEG_INFINITY;
        for (offset, score) in scores.iter_mut().enumerate() {
            let key_position = first_key + offset;
            let k_base = key_position * HEAD_DIM;
            let mut dot = 0.0_f64;
            for d in 0..HEAD_DIM {
                dot += query[q_base + d] as f64 * key[k_base + d].to_f32() as f64;
            }
            *score = dot;
            maximum = maximum.max(dot);
        }
        let denominator = scores
            .iter_mut()
            .map(|score| {
                *score = (*score - maximum).exp();
                *score
            })
            .sum::<f64>();
        for d in 0..HEAD_DIM {
            let mut accumulator = 0.0_f64;
            for (offset, probability) in scores.iter().enumerate() {
                let key_position = first_key + offset;
                accumulator +=
                    probability / denominator * value[key_position * HEAD_DIM + d].to_f32() as f64;
            }
            output[q_base + d] = accumulator as f32;
        }
    }
    output
}

#[test]
fn claim_accepts_fp16_zero_append_shared_cache_decode() {
    let Some(ep) = gpu() else { return };
    let (graph, node, shapes, dtypes) = gqa_node();
    assert!(
        matches!(
            ep.supports_op(graph.node(node), 1, &shapes, &dtypes, &[]),
            KernelMatch::Supported { .. }
        ),
        "CUDA must claim structural fp16 GQA decode with empty current K/V and shared past"
    );
}

#[test]
fn fp16_zero_append_shared_cache_matches_cpu_and_replays() {
    let Some(ep) = gpu() else { return };
    let runtime = ep.runtime();
    let kernel = GroupQueryAttentionKernel::new(
        runtime.clone(),
        QUERY_HEADS,
        KV_HEADS,
        Some(1.0),
        true,
        false,
        LOCAL_WINDOW as i64,
        0.0,
    )
    .unwrap()
    .with_backend(GroupQueryAttentionBackend::Phase2a);

    let query: Vec<f16> = (0..QUERY_HEADS * HEAD_DIM)
        .map(|index| f16::from_f32(((index * 17 % 97) as f32 - 48.0) / 1024.0))
        .collect();
    let cache_elements = KV_HEADS * CACHE_CAPACITY * HEAD_DIM;
    let key: Vec<f16> = (0..cache_elements)
        .map(|index| f16::from_f32(((index * 13 + 7) % 101) as f32 / 1024.0 - 0.05))
        .collect();
    let value: Vec<f16> = (0..cache_elements)
        .map(|index| f16::from_f32(((index * 29 + 3) % 113) as f32 / 256.0 - 0.22))
        .collect();
    let mut cos = vec![f16::ONE; ROTARY_ROWS * (HEAD_DIM / 2)];
    let mut sin = vec![f16::ZERO; cos.len()];
    for position in [CACHE_CAPACITY - 1, CACHE_CAPACITY] {
        for k in 0..HEAD_DIM / 2 {
            let offset = if position == CACHE_CAPACITY - 1 {
                0.2
            } else {
                0.9
            };
            let angle = offset + (k % 17) as f32 * 0.03;
            cos[position * (HEAD_DIM / 2) + k] = f16::from_f32(angle.cos());
            sin[position * (HEAD_DIM / 2) + k] = f16::from_f32(angle.sin());
        }
    }
    let seqlens = [CACHE_CAPACITY as i32 - 1];
    let total = [CACHE_CAPACITY as i32];

    let query_buffer = upload(&ep, typed_bytes(&query)).unwrap();
    let empty_key_buffer = upload(&ep, &[]).unwrap();
    let empty_value_buffer = upload(&ep, &[]).unwrap();
    let mut key_buffer = upload(&ep, typed_bytes(&key)).unwrap();
    let mut value_buffer = upload(&ep, typed_bytes(&value)).unwrap();
    let seqlens_buffer = upload(&ep, typed_bytes(&seqlens)).unwrap();
    let total_buffer = upload(&ep, typed_bytes(&total)).unwrap();
    let cos_buffer = upload(&ep, typed_bytes(&cos)).unwrap();
    let sin_buffer = upload(&ep, typed_bytes(&sin)).unwrap();
    let mut output_buffer = ep
        .allocate(QUERY_HEADS * HEAD_DIM * std::mem::size_of::<f16>(), 256)
        .unwrap();

    let query_shape = [BATCH, 1, QUERY_HEADS * HEAD_DIM];
    let current_shape = [BATCH, 0, KV_HEADS * HEAD_DIM];
    let cache_shape = [BATCH, KV_HEADS, CACHE_CAPACITY, HEAD_DIM];
    let seqlens_shape = [BATCH];
    let scalar_shape: [usize; 0] = [];
    let rotary_shape = [ROTARY_ROWS, HEAD_DIM / 2];
    let output_shape = query_shape;
    let query_strides = compute_contiguous_strides(&query_shape);
    let current_strides = compute_contiguous_strides(&current_shape);
    let cache_strides = compute_contiguous_strides(&cache_shape);
    let seqlens_strides = compute_contiguous_strides(&seqlens_shape);
    let scalar_strides = compute_contiguous_strides(&scalar_shape);
    let rotary_strides = compute_contiguous_strides(&rotary_shape);
    let output_strides = compute_contiguous_strides(&output_shape);
    let device = ep.device_id();

    let execute = |key_buffer: &mut DeviceBuffer,
                   value_buffer: &mut DeviceBuffer,
                   output_buffer: &mut DeviceBuffer| {
        let key_input = DevicePtr(key_buffer.as_ptr());
        let value_input = DevicePtr(value_buffer.as_ptr());
        let key_output = DevicePtrMut(key_buffer.as_mut_ptr());
        let value_output = DevicePtrMut(value_buffer.as_mut_ptr());
        let inputs = [
            TensorView::new(
                DevicePtr(query_buffer.as_ptr()),
                DataType::Float16,
                &query_shape,
                &query_strides,
                device,
            ),
            TensorView::new(
                DevicePtr(empty_key_buffer.as_ptr()),
                DataType::Float16,
                &current_shape,
                &current_strides,
                device,
            ),
            TensorView::new(
                DevicePtr(empty_value_buffer.as_ptr()),
                DataType::Float16,
                &current_shape,
                &current_strides,
                device,
            ),
            TensorView::new(
                key_input,
                DataType::Float16,
                &cache_shape,
                &cache_strides,
                device,
            ),
            TensorView::new(
                value_input,
                DataType::Float16,
                &cache_shape,
                &cache_strides,
                device,
            ),
            TensorView::new(
                DevicePtr(seqlens_buffer.as_ptr()),
                DataType::Int32,
                &seqlens_shape,
                &seqlens_strides,
                device,
            ),
            TensorView::new(
                DevicePtr(total_buffer.as_ptr()),
                DataType::Int32,
                &scalar_shape,
                &scalar_strides,
                device,
            ),
            TensorView::new(
                DevicePtr(cos_buffer.as_ptr()),
                DataType::Float16,
                &rotary_shape,
                &rotary_strides,
                device,
            ),
            TensorView::new(
                DevicePtr(sin_buffer.as_ptr()),
                DataType::Float16,
                &rotary_shape,
                &rotary_strides,
                device,
            ),
        ];
        let mut outputs = [
            TensorMut::new(
                DevicePtrMut(output_buffer.as_mut_ptr()),
                DataType::Float16,
                &output_shape,
                &output_strides,
                device,
            ),
            TensorMut::new(
                key_output,
                DataType::Float16,
                &cache_shape,
                &cache_strides,
                device,
            ),
            TensorMut::new(
                value_output,
                DataType::Float16,
                &cache_shape,
                &cache_strides,
                device,
            ),
        ];
        kernel.execute(&inputs, &mut outputs)
    };

    execute(&mut key_buffer, &mut value_buffer, &mut output_buffer).unwrap();
    assert_eq!(
        kernel.read_prepared_metadata_for_test(BATCH).unwrap(),
        (vec![4096], vec![4096], vec![4095])
    );
    assert!(matches!(
        kernel.capture_support(),
        CaptureSupport::Supported
    ));

    let allocations_after_warmup = runtime.allocation_counts();
    runtime.begin_graph_capture(&[]).unwrap();
    execute(&mut key_buffer, &mut value_buffer, &mut output_buffer).unwrap();
    runtime.end_graph_capture().unwrap();
    assert_eq!(
        runtime.allocation_counts(),
        allocations_after_warmup,
        "zero-append fp16 GQA capture path must not allocate or free"
    );
    runtime.replay_graph().unwrap();
    runtime.replay_graph().unwrap();

    let output_bytes = read(
        &ep,
        &output_buffer,
        QUERY_HEADS * HEAD_DIM * std::mem::size_of::<f16>(),
    )
    .unwrap();
    let got = fp16_values(&output_bytes);
    let rotated = rotate_query(&query, &cos, &sin, CACHE_CAPACITY - 1);
    let expected = cpu_reference(&rotated, &key, &value);
    let maximum_error = got
        .iter()
        .zip(&expected)
        .map(|(got, expected)| (got - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        maximum_error < 2e-3,
        "fp16 zero-append GQA diverged from CPU/f32 oracle: max_abs={maximum_error:e}"
    );

    assert_eq!(
        read(&ep, &key_buffer, typed_bytes(&key).len()).unwrap(),
        typed_bytes(&key),
        "zero-append present key must preserve the shared past cache"
    );
    assert_eq!(
        read(&ep, &value_buffer, typed_bytes(&value).len()).unwrap(),
        typed_bytes(&value),
        "zero-append present value must preserve the shared past cache"
    );
    runtime.reset_graph().unwrap();

    for buffer in [
        output_buffer,
        sin_buffer,
        cos_buffer,
        total_buffer,
        seqlens_buffer,
        value_buffer,
        key_buffer,
        empty_value_buffer,
        empty_key_buffer,
        query_buffer,
    ] {
        ep.deallocate(buffer).unwrap();
    }
}
