//! CUDA conformance tests for router/mask indexing and scan operators.

use half::f16;
use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, Result, TensorMut, TensorView,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, DeviceId, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;

struct Tensor {
    dtype: DataType,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn raw<T: Copy>(values: &[T]) -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)).to_vec()
    }
}

fn tensor<T: Copy>(dtype: DataType, shape: &[usize], values: &[T]) -> Tensor {
    Tensor {
        dtype,
        shape: shape.to_vec(),
        bytes: raw(values),
    }
}

/// Like the standard Attention GPU test helper, but individual input slots may be omitted., but individual input slots may be `None` to model an omitted
/// optional ONNX input (an empty-string input name → an absent [`TensorView`]).
fn run_opt(
    op: &str,
    opset: u64,
    inputs: &[Option<Tensor>],
    outputs: &[(DataType, Vec<usize>)],
    attrs: &[(&str, Attribute)],
) -> Vec<Vec<u8>> {
    let optional = inputs.iter().map(|o| o.as_ref()).collect::<Vec<_>>();
    run_result_core(op, opset, &optional, outputs, attrs).unwrap()
}

fn run_result_core(
    op: &str,
    opset: u64,
    inputs: &[Option<&Tensor>],
    outputs: &[(DataType, Vec<usize>)],
    attrs: &[(&str, Attribute)],
) -> Result<Vec<Vec<u8>>> {
    let ep = CudaExecutionProvider::new_default().expect("CUDA runtime must be available");
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), opset);
    let input_values = inputs
        .iter()
        .enumerate()
        .map(|(i, input)| {
            input.map(|input| {
                let value = graph.create_named_value(
                    format!("input_{i}"),
                    input.dtype,
                    static_shape(input.shape.iter().copied()),
                );
                graph.add_input(value);
                value
            })
        })
        .collect::<Vec<_>>();
    let output_values = outputs
        .iter()
        .enumerate()
        .map(|(i, (dtype, shape))| {
            graph.create_named_value(
                format!("output_{i}"),
                *dtype,
                static_shape(shape.iter().copied()),
            )
        })
        .collect::<Vec<_>>();
    let mut node = Node::new(
        NodeId(0),
        op,
        input_values.into_iter().collect(),
        output_values.clone(),
    );
    for (name, value) in attrs {
        node.attributes.insert((*name).into(), value.clone());
    }
    let node_id = graph.insert_node(node);
    for output in output_values {
        graph.add_output(output);
    }
    let model = Model::new(&graph);
    let kernel = ep.get_kernel(model.graph.node(node_id), &[], opset)?;

    let input_buffers = inputs
        .iter()
        .map(|input| -> Result<Option<DeviceBuffer>> {
            let Some(input) = input else {
                return Ok(None);
            };
            let buffer = ep.allocate(input.bytes.len(), 256)?;
            if !input.bytes.is_empty() {
                unsafe { ep.runtime().htod(&input.bytes, cuptr(buffer.as_ptr()))? };
            }
            Ok(Some(buffer))
        })
        .collect::<Result<Vec<_>>>()?;
    let input_strides = inputs
        .iter()
        .map(|input| {
            input
                .map(|input| compute_contiguous_strides(&input.shape))
                .unwrap_or_default()
        })
        .collect::<Vec<_>>();
    let input_views = inputs
        .iter()
        .zip(&input_buffers)
        .zip(&input_strides)
        .map(|((input, buffer), strides)| match (input, buffer) {
            (Some(input), Some(buffer)) => TensorView::new(
                DevicePtr(buffer.as_ptr()),
                input.dtype,
                &input.shape,
                strides,
                ep.device_id(),
            ),
            _ => TensorView::absent(DataType::Float32),
        })
        .collect::<Vec<_>>();
    let mut output_buffers = outputs
        .iter()
        .map(|(dtype, shape)| -> Result<DeviceBuffer> {
            ep.allocate(dtype.storage_bytes(shape.iter().product()), 256)
        })
        .collect::<Result<Vec<DeviceBuffer>>>()?;
    let output_strides = outputs
        .iter()
        .map(|(_, shape)| compute_contiguous_strides(shape))
        .collect::<Vec<_>>();
    let mut output_views = outputs
        .iter()
        .zip(output_buffers.iter_mut())
        .zip(&output_strides)
        .map(|(((dtype, shape), buffer), strides)| {
            TensorMut::new(
                DevicePtrMut(buffer.as_mut_ptr()),
                *dtype,
                shape,
                strides,
                ep.device_id(),
            )
        })
        .collect::<Vec<_>>();
    let allocations_before = ep.runtime().allocation_counts();
    if let Err(error) = kernel.execute(&input_views, &mut output_views) {
        for buffer in input_buffers.into_iter().flatten() {
            ep.deallocate(buffer)?;
        }
        for buffer in output_buffers {
            ep.deallocate(buffer)?;
        }
        return Err(error);
    }
    if op == "RotaryEmbedding" {
        assert_eq!(
            ep.runtime().allocation_counts(),
            allocations_before,
            "RotaryEmbedding launch path must not allocate or free CUDA memory"
        );
        assert!(
            kernel.cuda_graph_compatible(),
            "warmed RotaryEmbedding signature must be capture-supported"
        );
    }

    let result = outputs
        .iter()
        .zip(&output_buffers)
        .map(|((dtype, shape), buffer)| -> Result<Vec<u8>> {
            let mut bytes = vec![0; dtype.storage_bytes(shape.iter().product())];
            if !bytes.is_empty() {
                unsafe { ep.runtime().dtoh(&mut bytes, cuptr(buffer.as_ptr()))? };
            }
            Ok(bytes)
        })
        .collect::<Result<Vec<_>>>()?;
    for buffer in input_buffers.into_iter().flatten() {
        ep.deallocate(buffer)?;
    }
    for buffer in output_buffers {
        ep.deallocate(buffer)?;
    }
    Ok(result)
}

fn run_cpu_opt(
    op: &str,
    opset: u64,
    inputs: &[Option<Tensor>],
    outputs: &[(DataType, Vec<usize>)],
    attrs: &[(&str, Attribute)],
) -> Vec<Vec<u8>> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), opset);
    let input_values = inputs
        .iter()
        .enumerate()
        .map(|(i, input)| {
            input.as_ref().map(|input| {
                let value = graph.create_named_value(
                    format!("input_{i}"),
                    input.dtype,
                    static_shape(input.shape.iter().copied()),
                );
                graph.add_input(value);
                value
            })
        })
        .collect::<Vec<_>>();
    let output_values = outputs
        .iter()
        .enumerate()
        .map(|(i, (dtype, shape))| {
            graph.create_named_value(
                format!("output_{i}"),
                *dtype,
                static_shape(shape.iter().copied()),
            )
        })
        .collect::<Vec<_>>();
    let mut node = Node::new(
        NodeId(0),
        op,
        input_values.into_iter().collect(),
        output_values.clone(),
    );
    for (name, value) in attrs {
        node.attributes.insert((*name).into(), value.clone());
    }
    let node_id = graph.insert_node(node);
    for output in output_values {
        graph.add_output(output);
    }
    let model = Model::new(&graph);
    let kernel = CpuExecutionProvider::new()
        .get_kernel(model.graph.node(node_id), &[], opset)
        .unwrap();

    let input_strides = inputs
        .iter()
        .map(|input| {
            input
                .as_ref()
                .map(|input| compute_contiguous_strides(&input.shape))
                .unwrap_or_default()
        })
        .collect::<Vec<_>>();
    let input_views = inputs
        .iter()
        .zip(&input_strides)
        .map(|(input, strides)| match input {
            Some(input) => TensorView::new(
                DevicePtr(input.bytes.as_ptr().cast()),
                input.dtype,
                &input.shape,
                strides,
                DeviceId::cpu(),
            ),
            None => TensorView::absent(DataType::Float32),
        })
        .collect::<Vec<_>>();
    let mut output_buffers = outputs
        .iter()
        .map(|(dtype, shape)| vec![0; dtype.storage_bytes(shape.iter().product())])
        .collect::<Vec<_>>();
    let output_strides = outputs
        .iter()
        .map(|(_, shape)| compute_contiguous_strides(shape))
        .collect::<Vec<_>>();
    let mut output_views = outputs
        .iter()
        .zip(output_buffers.iter_mut())
        .zip(&output_strides)
        .map(|(((dtype, shape), bytes), strides)| {
            TensorMut::new(
                DevicePtrMut(bytes.as_mut_ptr().cast()),
                *dtype,
                shape,
                strides,
                DeviceId::cpu(),
            )
        })
        .collect::<Vec<_>>();
    kernel.execute(&input_views, &mut output_views).unwrap();
    drop(output_views);
    output_buffers
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
        .collect()
}

fn f16s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|v| f16::from_bits(u16::from_ne_bytes(v.try_into().unwrap())).to_f32())
        .collect()
}
fn fp16_tensor(shape: &[usize], values: &[f32]) -> Tensor {
    let values = values
        .iter()
        .copied()
        .map(f16::from_f32)
        .collect::<Vec<_>>();
    tensor(DataType::Float16, shape, &values)
}

fn max_abs_error(got: &[f32], expected: &[f32]) -> f32 {
    got.iter()
        .zip(expected)
        .map(|(got, expected)| (got - expected).abs())
        .fold(0.0_f32, f32::max)
}

#[test]
fn standard_attention_fp16_matches_f32_cpu_oracle_for_prefill_mask_and_cached_decode() {
    let Ok(_ep) = CudaExecutionProvider::new_default() else {
        eprintln!("skip: no CUDA GPU available");
        return;
    };

    let query = (0..24)
        .map(|index| ((index * 17 % 29) as f32 - 14.0) / 16.0)
        .collect::<Vec<_>>();
    let key = (0..12)
        .map(|index| ((index * 11 % 23) as f32 - 11.0) / 13.0)
        .collect::<Vec<_>>();
    let value = (0..12)
        .map(|index| ((index * 7 % 19) as f32 - 9.0) / 11.0)
        .collect::<Vec<_>>();
    let additive_mask = [0.0_f32, -0.25, -1.5, 0.0, 0.0, -0.75, -0.5, 0.0, 0.0];
    let attrs = [
        ("q_num_heads", Attribute::Int(2)),
        ("kv_num_heads", Attribute::Int(1)),
        ("is_causal", Attribute::Int(1)),
    ];

    let cpu_prefill = run_cpu_opt(
        "Attention",
        23,
        &[
            Some(tensor(DataType::Float32, &[1, 2, 3, 4], &query)),
            Some(tensor(DataType::Float32, &[1, 1, 3, 4], &key)),
            Some(tensor(DataType::Float32, &[1, 1, 3, 4], &value)),
            None,
        ],
        &[(DataType::Float32, vec![1, 2, 3, 4])],
        &attrs,
    );
    let gpu_prefill = run_opt(
        "Attention",
        23,
        &[
            Some(fp16_tensor(&[1, 2, 3, 4], &query)),
            Some(fp16_tensor(&[1, 1, 3, 4], &key)),
            Some(fp16_tensor(&[1, 1, 3, 4], &value)),
            None,
        ],
        &[(DataType::Float16, vec![1, 2, 3, 4])],
        &attrs,
    );
    let prefill_error = max_abs_error(&f16s(&gpu_prefill[0]), &f32s(&cpu_prefill[0]));
    assert!(
        prefill_error < 3e-3,
        "fp16 prefill diverged from f32 CPU oracle: max_abs={prefill_error:e}"
    );

    let cpu_masked = run_cpu_opt(
        "Attention",
        23,
        &[
            Some(tensor(DataType::Float32, &[1, 2, 3, 4], &query)),
            Some(tensor(DataType::Float32, &[1, 1, 3, 4], &key)),
            Some(tensor(DataType::Float32, &[1, 1, 3, 4], &value)),
            Some(tensor(DataType::Float32, &[1, 1, 3, 3], &additive_mask)),
        ],
        &[(DataType::Float32, vec![1, 2, 3, 4])],
        &attrs,
    );
    let gpu_masked = run_opt(
        "Attention",
        23,
        &[
            Some(fp16_tensor(&[1, 2, 3, 4], &query)),
            Some(fp16_tensor(&[1, 1, 3, 4], &key)),
            Some(fp16_tensor(&[1, 1, 3, 4], &value)),
            Some(fp16_tensor(&[1, 1, 3, 3], &additive_mask)),
        ],
        &[(DataType::Float16, vec![1, 2, 3, 4])],
        &attrs,
    );
    let masked_error = max_abs_error(&f16s(&gpu_masked[0]), &f32s(&cpu_masked[0]));
    assert!(
        masked_error < 3e-3,
        "fp16 additive-mask prefill diverged from f32 CPU oracle: max_abs={masked_error:e}"
    );

    let decode_query = &query[..8];
    let decode_key = &key[..4];
    let decode_value = &value[..4];
    let cpu_decode = run_cpu_opt(
        "Attention",
        23,
        &[
            Some(tensor(DataType::Float32, &[1, 2, 1, 4], decode_query)),
            Some(tensor(DataType::Float32, &[1, 1, 1, 4], decode_key)),
            Some(tensor(DataType::Float32, &[1, 1, 1, 4], decode_value)),
            None,
            Some(tensor(DataType::Float32, &[1, 1, 2, 4], &key[..8])),
            Some(tensor(DataType::Float32, &[1, 1, 2, 4], &value[..8])),
        ],
        &[(DataType::Float32, vec![1, 2, 1, 4])],
        &attrs,
    );
    let gpu_decode = run_opt(
        "Attention",
        23,
        &[
            Some(fp16_tensor(&[1, 2, 1, 4], decode_query)),
            Some(fp16_tensor(&[1, 1, 1, 4], decode_key)),
            Some(fp16_tensor(&[1, 1, 1, 4], decode_value)),
            None,
            Some(fp16_tensor(&[1, 1, 2, 4], &key[..8])),
            Some(fp16_tensor(&[1, 1, 2, 4], &value[..8])),
        ],
        &[(DataType::Float16, vec![1, 2, 1, 4])],
        &attrs,
    );
    let decode_error = max_abs_error(&f16s(&gpu_decode[0]), &f32s(&cpu_decode[0]));
    assert!(
        decode_error < 3e-3,
        "fp16 cached decode diverged from f32 CPU oracle: max_abs={decode_error:e}"
    );

    eprintln!(
        "fp16 Attention max_abs: prefill={prefill_error:e}, additive_mask={masked_error:e}, cached_decode={decode_error:e}"
    );
}
