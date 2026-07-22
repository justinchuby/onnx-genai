//! CUDA conformance tests for router/mask indexing and scan operators.

use half::f16;
use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, KernelMatch, Result, TensorMut,
    TensorView,
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

#[test]
fn standard_attention_and_rope_claim_supported_dtypes_and_require_contiguous_inputs() {
    let ep = CudaExecutionProvider::new_default().expect("CUDA runtime must be available");

    for (op_type, opset, dtype, expected_reason) in [
        ("Attention", 23, DataType::Float16, "Attention: dtype f16"),
        ("Attention", 23, DataType::BFloat16, "Attention: dtype bf16"),
        (
            "RotaryEmbedding",
            23,
            DataType::BFloat16,
            "RotaryEmbedding: dtype bf16",
        ),
    ] {
        let mut graph = Graph::new();
        let inputs = (0..3)
            .map(|i| {
                graph.create_named_value(format!("input_{i}"), dtype, static_shape([1, 1, 1, 2]))
            })
            .collect::<Vec<_>>();
        let output = graph.create_named_value("output", dtype, static_shape([1, 1, 1, 2]));
        let node = Node::new(
            NodeId(0),
            op_type,
            inputs.into_iter().map(Some).collect(),
            vec![output],
        );
        let input_dtypes = [dtype; 3];
        assert!(matches!(
            ep.supports_op(&node, opset, &[], &input_dtypes, &[]),
            KernelMatch::Unsupported { ref reason } if reason.contains(expected_reason)
        ));

        let f32_dtypes = [DataType::Float32; 3];
        assert!(
            ep.supports_op(&node, opset, &[], &f32_dtypes, &[])
                .is_supported()
        );
        let kernel = ep.get_kernel(&node, &[], opset).unwrap();
        assert!(
            !kernel.supports_strided_input(0),
            "{op_type} must request contiguous inputs"
        );
        if op_type == "RotaryEmbedding" {
            assert!(
                !kernel.cuda_graph_compatible(),
                "RotaryEmbedding requires an exact warmed signature"
            );
        }
    }

    let mut graph = Graph::new();
    let inputs = (0..3)
        .map(|i| {
            graph.create_named_value(
                format!("rope_input_{i}"),
                DataType::Float16,
                static_shape([1, 1, 1, 2]),
            )
        })
        .collect::<Vec<_>>();
    let output =
        graph.create_named_value("rope_output", DataType::Float16, static_shape([1, 1, 1, 2]));
    let node = Node::new(
        NodeId(0),
        "RotaryEmbedding",
        inputs.into_iter().map(Some).collect(),
        vec![output],
    );
    assert!(
        ep.supports_op(&node, 23, &[], &[DataType::Float16; 3], &[])
            .is_supported(),
        "RotaryEmbedding must claim its fp16 activation path"
    );
}

#[test]
fn standard_attention_claim_distinguishes_omitted_and_wrong_typed_past_cache() {
    let ep = CudaExecutionProvider::new_default().expect("CUDA runtime must be available");
    let mut graph = Graph::new();
    let q = graph.create_named_value("q", DataType::Float32, static_shape([1, 1, 1, 2]));
    let k = graph.create_named_value("k", DataType::Float32, static_shape([1, 1, 1, 2]));
    let v = graph.create_named_value("v", DataType::Float32, static_shape([1, 1, 1, 2]));
    let y = graph.create_named_value("y", DataType::Float32, static_shape([1, 1, 1, 2]));
    let node_id = graph.insert_node(Node::new(
        NodeId(0),
        "Attention",
        vec![Some(q), Some(k), Some(v), None, None, None],
        vec![y],
    ));
    let node = graph.node(node_id);

    let omitted_past_dtypes = [
        DataType::Float32,
        DataType::Float32,
        DataType::Float32,
        DataType::Undefined,
        DataType::Undefined,
        DataType::Undefined,
    ];
    assert!(
        ep.supports_op(node, 23, &[], &omitted_past_dtypes, &[])
            .is_supported(),
        "omitted optional Attention inputs must not prevent the CUDA EP claim"
    );

    let wrong_past_key_dtypes = [
        DataType::Float32,
        DataType::Float32,
        DataType::Float32,
        DataType::Undefined,
        DataType::Int64,
        DataType::Float32,
    ];
    let past_key =
        graph.create_named_value("past_key", DataType::Int64, static_shape([1, 1, 1, 2]));
    let past_value =
        graph.create_named_value("past_value", DataType::Float32, static_shape([1, 1, 1, 2]));
    let wrong_y =
        graph.create_named_value("wrong_y", DataType::Float32, static_shape([1, 1, 1, 2]));
    let wrong_node_id = graph.insert_node(Node::new(
        NodeId(1),
        "Attention",
        vec![
            Some(q),
            Some(k),
            Some(v),
            None,
            Some(past_key),
            Some(past_value),
        ],
        vec![wrong_y],
    ));
    assert!(matches!(
        ep.supports_op(
            graph.node(wrong_node_id),
            23,
            &[],
            &wrong_past_key_dtypes,
            &[]
        ),
        KernelMatch::Unsupported { ref reason }
            if reason.contains("Attention: dtype Int64")
    ));
}

fn run(
    op: &str,
    opset: u64,
    inputs: &[Tensor],
    outputs: &[(DataType, Vec<usize>)],
    attrs: &[(&str, Attribute)],
) -> Vec<Vec<u8>> {
    run_result(op, opset, inputs, outputs, attrs).unwrap()
}

fn run_result(
    op: &str,
    opset: u64,
    inputs: &[Tensor],
    outputs: &[(DataType, Vec<usize>)],
    attrs: &[(&str, Attribute)],
) -> Result<Vec<Vec<u8>>> {
    let optional = inputs.iter().map(Some).collect::<Vec<_>>();
    run_result_core(op, opset, &optional, outputs, attrs)
}

/// Like [`run`], but individual input slots may be `None` to model an omitted
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

fn attention_reference(q: &[f32], k: &[f32], v: &[f32], mask: &[u8]) -> Vec<f32> {
    // [B=1,Hq=2,S=2,D=2], [B=1,Hkv=1,S=2,D=2], causal; mask is [1,1,S,S].
    let mut result = vec![0.0; 8];
    let scale = 1.0 / 2.0_f32.sqrt();
    for h in 0..2 {
        for i in 0..2 {
            let mut scores = [f32::NEG_INFINITY; 2];
            for j in 0..2 {
                if j <= i && mask[i * 2 + j] != 0 {
                    scores[j] = (0..2)
                        .map(|d| q[(h * 2 + i) * 2 + d] * k[j * 2 + d])
                        .sum::<f32>()
                        * scale;
                }
            }
            let max = scores.into_iter().fold(f32::NEG_INFINITY, f32::max);
            let exp = scores.map(|x| if x.is_finite() { (x - max).exp() } else { 0.0 });
            let sum: f32 = exp.iter().sum();
            for d in 0..2 {
                result[(h * 2 + i) * 2 + d] = (0..2).map(|j| exp[j] / sum * v[j * 2 + d]).sum();
            }
        }
    }
    result
}

fn assert_close(got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len());
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        assert!((g - e).abs() <= 1e-4, "element {i}: {g} vs {e}");
    }
}

#[allow(clippy::too_many_arguments)]
fn rotary_embedding_reference_4d(
    x: &[f32],
    batch: usize,
    heads: usize,
    seq: usize,
    head_size: usize,
    cos_cache: &[f32],
    sin_cache: &[f32],
    position_ids: &[i64],
) -> Vec<f32> {
    let half = head_size / 2;
    let mut y = vec![0.0; x.len()];
    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq {
                let row = position_ids[b * seq + s] as usize;
                for d in 0..half {
                    let offset = ((b * heads + h) * seq + s) * head_size;
                    let cos = cos_cache[row * half + d];
                    let sin = sin_cache[row * half + d];
                    y[offset + d] = cos * x[offset + d] - sin * x[offset + d + half];
                    y[offset + d + half] = sin * x[offset + d] + cos * x[offset + d + half];
                }
            }
        }
    }
    y
}

#[allow(clippy::too_many_arguments)]
fn rotary_embedding_reference_fp16(
    x: &[f32],
    batch: usize,
    heads: usize,
    seq: usize,
    head_size: usize,
    cos_cache: &[f32],
    sin_cache: &[f32],
    position_ids: &[i64],
    interleaved: bool,
) -> Vec<f32> {
    let half = head_size / 2;
    let mut y = vec![0.0; x.len()];
    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq {
                let row = position_ids[b * seq + s] as usize;
                let offset = ((b * heads + h) * seq + s) * head_size;
                for d in 0..head_size {
                    let (k, partner) = if interleaved {
                        (d / 2, d ^ 1)
                    } else if d < half {
                        (d, d + half)
                    } else {
                        (d - half, d - half)
                    };
                    let cos = cos_cache[row * half + k];
                    let sin = sin_cache[row * half + k];
                    y[offset + d] = if interleaved {
                        if d.is_multiple_of(2) {
                            cos * x[offset + d] - sin * x[offset + partner]
                        } else {
                            sin * x[offset + partner] + cos * x[offset + d]
                        }
                    } else if d < half {
                        cos * x[offset + d] - sin * x[offset + partner]
                    } else {
                        sin * x[offset + partner] + cos * x[offset + d]
                    };
                }
            }
        }
    }
    y
}

#[test]
fn standard_attention_prefill_gqa_bool_mask_is_deterministic() {
    let inputs = [
        tensor(
            DataType::Float32,
            &[1, 2, 2, 2],
            &[1_f32, 0., 0., 1., 1., 1., 2., -1.],
        ),
        tensor(DataType::Float32, &[1, 1, 2, 2], &[1_f32, 2., 3., 4.]),
        tensor(DataType::Float32, &[1, 1, 2, 2], &[10_f32, 20., 30., 40.]),
        tensor(DataType::Bool, &[1, 1, 2, 2], &[1_u8, 1, 1, 0]),
    ];
    let attrs = [
        ("is_causal", Attribute::Int(1)),
        ("q_num_heads", Attribute::Int(2)),
        ("kv_num_heads", Attribute::Int(1)),
    ];
    let once = run(
        "Attention",
        23,
        &inputs,
        &[(DataType::Float32, vec![1, 2, 2, 2])],
        &attrs,
    );
    let twice = run(
        "Attention",
        23,
        &inputs,
        &[(DataType::Float32, vec![1, 2, 2, 2])],
        &attrs,
    );
    assert_eq!(
        once, twice,
        "standard Attention must be byte-identical across runs"
    );
    assert_close(
        &f32s(&once[0]),
        &attention_reference(
            &[1_f32, 0., 0., 1., 1., 1., 2., -1.],
            &[1_f32, 2., 3., 4.],
            &[10_f32, 20., 30., 40.],
            &[1, 1, 1, 0],
        ),
    );
}

#[test]
fn rotary_embedding_interleaved_partial_position_ids_is_deterministic() {
    let inputs = [
        tensor(
            DataType::Float32,
            &[1, 1, 2, 4],
            &[1_f32, 2., 3., 4., 5., 6., 7., 8.],
        ),
        tensor(DataType::Float32, &[2, 1], &[1_f32, 0.]),
        tensor(DataType::Float32, &[2, 1], &[0_f32, 1.]),
        tensor(DataType::Int64, &[1, 2], &[0_i64, 1]),
    ];
    let attrs = [
        ("interleaved", Attribute::Int(1)),
        ("rotary_embedding_dim", Attribute::Int(2)),
    ];
    let once = run(
        "RotaryEmbedding",
        23,
        &inputs,
        &[(DataType::Float32, vec![1, 1, 2, 4])],
        &attrs,
    );
    let twice = run(
        "RotaryEmbedding",
        23,
        &inputs,
        &[(DataType::Float32, vec![1, 1, 2, 4])],
        &attrs,
    );
    assert_eq!(
        once, twice,
        "RotaryEmbedding must be byte-identical across runs"
    );
    assert_close(&f32s(&once[0]), &[1_f32, 2., 3., 4., -6., 5., 7., 8.]);
}

#[test]
fn rotary_embedding_3d_rotate_half_direct_cache_broadcasts_across_heads() {
    // [B, S, H*D] with two heads. Each cache row is shared by both heads.
    let inputs = [
        tensor(
            DataType::Float32,
            &[1, 2, 8],
            &[
                1_f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
        ),
        tensor(DataType::Float32, &[1, 2, 2], &[1_f32, 0.5, 0., 1.]),
        tensor(DataType::Float32, &[1, 2, 2], &[0_f32, 0.5, 1., 0.]),
    ];
    let attrs = [("num_heads", Attribute::Int(2))];
    let once = run(
        "RotaryEmbedding",
        23,
        &inputs,
        &[(DataType::Float32, vec![1, 2, 8])],
        &attrs,
    );
    let twice = run(
        "RotaryEmbedding",
        23,
        &inputs,
        &[(DataType::Float32, vec![1, 2, 8])],
        &attrs,
    );
    assert_eq!(
        once, twice,
        "RotaryEmbedding direct-cache path must be byte-identical across runs"
    );
    assert_eq!(
        f32s(&once[0]),
        vec![
            1., -1., 3., 3., 5., -1., 7., 7., -11., 10., 9., 12., -15., 14., 13., 16.,
        ]
    );
}

#[test]
fn rotary_embedding_4d_multi_batch_multi_head_position_ids_matches_reference() {
    // Each (batch, sequence) position must select one cache row, then broadcast
    // it across heads in X's [B, H, S, D] layout.
    let x = (1..=16).map(|value| value as f32).collect::<Vec<_>>();
    let position_ids = [0_i64, 1, 2, 0];
    let cos_cache = [1_f32, 0., -1.];
    let sin_cache = [0_f32, 1., 0.];
    let inputs = [
        tensor(DataType::Float32, &[2, 2, 2, 2], &x),
        tensor(DataType::Float32, &[3, 1], &cos_cache),
        tensor(DataType::Float32, &[3, 1], &sin_cache),
        tensor(DataType::Int64, &[2, 2], &position_ids),
    ];
    let output = run(
        "RotaryEmbedding",
        23,
        &inputs,
        &[(DataType::Float32, vec![2, 2, 2, 2])],
        &[],
    );
    assert_close(
        &f32s(&output[0]),
        &rotary_embedding_reference_4d(&x, 2, 2, 2, 2, &cos_cache, &sin_cache, &position_ids),
    );
}

#[test]
fn rotary_embedding_fp16_matches_fp32_reference_for_both_rotation_layouts() {
    let (batch, heads, seq, head_size) = (2, 2, 3, 8);
    let x = (0..batch * heads * seq * head_size)
        .map(|index| f16::from_f32(((index * 19 + 7) % 97) as f32 / 32.0 - 1.5))
        .collect::<Vec<_>>();
    let position_ids = [0_i64, 2, 4, 1, 3, 0];
    let cache_rows = 5;
    let half = head_size / 2;
    let cos_cache = (0..cache_rows * half)
        .map(|index| f16::from_f32((index as f32 * 0.17).cos()))
        .collect::<Vec<_>>();
    let sin_cache = (0..cache_rows * half)
        .map(|index| f16::from_f32((index as f32 * 0.17).sin()))
        .collect::<Vec<_>>();
    let x_reference = x.iter().map(|value| value.to_f32()).collect::<Vec<_>>();
    let cos_reference = cos_cache
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let sin_reference = sin_cache
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();

    for interleaved in [false, true] {
        let inputs = [
            tensor(DataType::Float16, &[batch, heads, seq, head_size], &x),
            tensor(DataType::Float16, &[cache_rows, half], &cos_cache),
            tensor(DataType::Float16, &[cache_rows, half], &sin_cache),
            tensor(DataType::Int64, &[batch, seq], &position_ids),
        ];
        let output = run(
            "RotaryEmbedding",
            23,
            &inputs,
            &[(DataType::Float16, vec![batch, heads, seq, head_size])],
            &[("interleaved", Attribute::Int(i64::from(interleaved)))],
        );
        let got = f16s(&output[0]);
        let expected = rotary_embedding_reference_fp16(
            &x_reference,
            batch,
            heads,
            seq,
            head_size,
            &cos_reference,
            &sin_reference,
            &position_ids,
            interleaved,
        );
        let error = got
            .iter()
            .zip(&expected)
            .map(|(&got, &expected)| {
                assert!(got.is_finite(), "fp16 RoPE output contains NaN/Inf");
                (got - expected).abs()
            })
            .fold(0.0f32, f32::max);
        println!("fp16 RoPE interleaved={interleaved} max_abs_error={error:.9e}");
        assert!(
            error <= 1e-3,
            "single fp32 rotation followed by fp16 output rounding exceeded tolerance"
        );
    }
}

#[test]
fn rotary_embedding_negative_position_ids_return_error() {
    let inputs = [
        tensor(DataType::Float32, &[1, 1, 1, 2], &[1_f32, 2.]),
        tensor(DataType::Float32, &[1, 1], &[1_f32]),
        tensor(DataType::Float32, &[1, 1], &[0_f32]),
        tensor(DataType::Int64, &[1, 1], &[-1_i64]),
    ];
    assert!(
        run_result(
            "RotaryEmbedding",
            23,
            &inputs,
            &[(DataType::Float32, vec![1, 1, 1, 2])],
            &[],
        )
        .is_err(),
        "negative position_ids must return an error"
    );
}

#[test]
fn rotary_embedding_out_of_range_position_ids_return_error() {
    let inputs = [
        tensor(DataType::Float32, &[1, 1, 1, 2], &[1_f32, 2.]),
        tensor(DataType::Float32, &[1, 1], &[1_f32]),
        tensor(DataType::Float32, &[1, 1], &[0_f32]),
        tensor(DataType::Int64, &[1, 1], &[1_i64]),
    ];
    assert!(
        run_result(
            "RotaryEmbedding",
            23,
            &inputs,
            &[(DataType::Float32, vec![1, 1, 1, 2])],
            &[],
        )
        .is_err(),
        "position_ids beyond the cache rows must return an error"
    );
}

// ---------------------------------------------------------------------------
// GPU-vs-reference parity coverage for the GPU-native standard Attention kernel.
// ---------------------------------------------------------------------------

/// A resolved reference mask mirroring the kernel's broadcast + short-last-dim
/// semantics exactly.
enum RefMask {
    None,
    Float(Vec<f32>, Vec<usize>),
    Bool(Vec<u8>, Vec<usize>),
}

impl RefMask {
    fn offset(shape: &[usize], b: usize, h: usize, i: usize, j: usize) -> usize {
        let full = [b, h, i, j];
        let rank = shape.len();
        let mut off = 0usize;
        for (k, &dim) in shape.iter().enumerate() {
            let logical = full[4 - rank + k];
            let idx = if dim == 1 { 0 } else { logical };
            off = off * dim + idx;
        }
        off
    }

    fn bias(&self, b: usize, h: usize, i: usize, j: usize, total: usize) -> f32 {
        match self {
            RefMask::None => 0.0,
            RefMask::Float(data, shape) => {
                if !shape.is_empty() {
                    let last = shape[shape.len() - 1];
                    if j >= last && last < total {
                        return f32::NEG_INFINITY;
                    }
                }
                data[Self::offset(shape, b, h, i, j)]
            }
            RefMask::Bool(data, shape) => {
                if !shape.is_empty() {
                    let last = shape[shape.len() - 1];
                    if j >= last && last < total {
                        return f32::NEG_INFINITY;
                    }
                }
                if data[Self::offset(shape, b, h, i, j)] != 0 {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
struct RefCase<'a> {
    q: &'a [f32],
    key: &'a [f32],
    value: &'a [f32],
    batch: usize,
    q_heads: usize,
    q_seq: usize,
    kv_heads: usize,
    total_seq: usize,
    head_size: usize,
    v_head_size: usize,
    mask: RefMask,
    is_causal: bool,
    offset: i64,
    scale: Option<f32>,
    softcap: f32,
}

/// Full-precision reference SDPA over already-concatenated present K/V, matching
/// the kernel's stage ordering. `qk_mode` selects the captured qk stage (0 raw,
/// 1 after softcap, 2 after mask, 3 after softmax), returned alongside Y.
fn sdpa_ref(case: &RefCase, qk_mode: i64) -> (Vec<f32>, Vec<f32>) {
    sdpa_ref_with_pad_limit(case, qk_mode, None)
}

fn sdpa_ref_with_pad_limit(
    case: &RefCase,
    qk_mode: i64,
    pad_limit: Option<i64>,
) -> (Vec<f32>, Vec<f32>) {
    let RefCase {
        q,
        key,
        value,
        batch,
        q_heads,
        q_seq,
        kv_heads,
        total_seq,
        head_size,
        v_head_size,
        mask,
        is_causal,
        offset,
        scale,
        softcap,
    } = case;
    let (batch, q_heads, q_seq, kv_heads, total_seq, head_size, v_head_size) = (
        *batch,
        *q_heads,
        *q_seq,
        *kv_heads,
        *total_seq,
        *head_size,
        *v_head_size,
    );
    let scale = scale.unwrap_or(1.0 / (head_size as f32).sqrt());
    let ss = scale.sqrt();
    let group = q_heads / kv_heads;
    let mut y = vec![0.0f32; batch * q_heads * q_seq * v_head_size];
    let mut qk = vec![0.0f32; batch * q_heads * q_seq * total_seq];
    for b in 0..batch {
        for qh in 0..q_heads {
            let kvh = qh / group;
            for i in 0..q_seq {
                let mut scores = vec![0.0f32; total_seq];
                for (j, sc) in scores.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for p in 0..head_size {
                        let qv = q[((b * q_heads + qh) * q_seq + i) * head_size + p] * ss;
                        let kv = key[((b * kv_heads + kvh) * total_seq + j) * head_size + p] * ss;
                        acc += qv * kv;
                    }
                    *sc = acc;
                }
                let base = ((b * q_heads + qh) * q_seq + i) * total_seq;
                if qk_mode == 0 {
                    qk[base..base + total_seq].copy_from_slice(&scores);
                }
                if *softcap != 0.0 {
                    for sc in scores.iter_mut() {
                        *sc = *softcap * (*sc / *softcap).tanh();
                    }
                }
                if qk_mode == 1 {
                    qk[base..base + total_seq].copy_from_slice(&scores);
                }
                let causal_limit = i as i64 + *offset;
                for (j, sc) in scores.iter_mut().enumerate() {
                    if pad_limit.is_some_and(|limit| j as i64 >= limit) {
                        *sc = f32::NEG_INFINITY;
                        continue;
                    }
                    if *is_causal && (j as i64) > causal_limit {
                        *sc = f32::NEG_INFINITY;
                        continue;
                    }
                    *sc += mask.bias(b, qh, i, j, total_seq);
                }
                if qk_mode == 2 {
                    qk[base..base + total_seq].copy_from_slice(&scores);
                }
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                if max == f32::NEG_INFINITY {
                    for sc in scores.iter_mut() {
                        *sc = 0.0;
                    }
                } else {
                    let mut sum = 0.0f32;
                    for sc in scores.iter_mut() {
                        let e = (*sc - max).exp();
                        *sc = e;
                        sum += e;
                    }
                    let inv = 1.0 / sum;
                    for sc in scores.iter_mut() {
                        *sc *= inv;
                    }
                }
                if qk_mode == 3 {
                    qk[base..base + total_seq].copy_from_slice(&scores);
                }
                let y_base = ((b * q_heads + qh) * q_seq + i) * v_head_size;
                for c in 0..v_head_size {
                    let mut acc = 0.0f32;
                    for (j, &p) in scores.iter().enumerate() {
                        acc += p * value[((b * kv_heads + kvh) * total_seq + j) * v_head_size + c];
                    }
                    y[y_base + c] = acc;
                }
            }
        }
    }
    (y, qk)
}

fn seq_f32(n: usize) -> Vec<f32> {
    (0..n).map(|v| ((v % 13) as f32) * 0.25 - 1.0).collect()
}

fn conformance_values(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 17 + salt * 29) % 97) as f32 - 48.0) / 128.0)
        .collect()
}

fn concat_bsh(
    first: &[f32],
    second: &[f32],
    batch: usize,
    first_seq: usize,
    second_seq: usize,
    hidden: usize,
) -> Vec<f32> {
    let total_seq = first_seq + second_seq;
    let mut dst = vec![0.0; batch * total_seq * hidden];
    for b in 0..batch {
        let first_src = b * first_seq * hidden;
        let second_src = b * second_seq * hidden;
        let dst_base = b * total_seq * hidden;
        dst[dst_base..dst_base + first_seq * hidden]
            .copy_from_slice(&first[first_src..first_src + first_seq * hidden]);
        dst[dst_base + first_seq * hidden..dst_base + total_seq * hidden]
            .copy_from_slice(&second[second_src..second_src + second_seq * hidden]);
    }
    dst
}

fn asymmetric_3d_prefill_decode_cpu_cuda_case(kv_heads: usize) {
    let (batch, q_heads, prefill_seq, decode_seq) = (1usize, 4usize, 3usize, 1usize);
    let (qk_head_dim, v_head_dim) = (192usize, 128usize);
    assert_ne!(qk_head_dim, v_head_dim);
    let attrs = [
        ("is_causal", Attribute::Int(1)),
        ("q_num_heads", Attribute::Int(q_heads as i64)),
        ("kv_num_heads", Attribute::Int(kv_heads as i64)),
    ];

    let q_prefill = conformance_values(batch * prefill_seq * q_heads * qk_head_dim, 1 + kv_heads);
    let k_prefill = conformance_values(batch * prefill_seq * kv_heads * qk_head_dim, 3 + kv_heads);
    let v_prefill = conformance_values(batch * prefill_seq * kv_heads * v_head_dim, 5 + kv_heads);
    let prefill_inputs = vec![
        Some(tensor(
            DataType::Float32,
            &[batch, prefill_seq, q_heads * qk_head_dim],
            &q_prefill,
        )),
        Some(tensor(
            DataType::Float32,
            &[batch, prefill_seq, kv_heads * qk_head_dim],
            &k_prefill,
        )),
        Some(tensor(
            DataType::Float32,
            &[batch, prefill_seq, kv_heads * v_head_dim],
            &v_prefill,
        )),
    ];
    let prefill_outputs = vec![
        (
            DataType::Float32,
            vec![batch, prefill_seq, q_heads * v_head_dim],
        ),
        (
            DataType::Float32,
            vec![batch, kv_heads, prefill_seq, qk_head_dim],
        ),
        (
            DataType::Float32,
            vec![batch, kv_heads, prefill_seq, v_head_dim],
        ),
    ];
    assert_eq!(
        prefill_outputs[0].1,
        [batch, prefill_seq, q_heads * v_head_dim],
        "3D Attention output hidden width must use V head width"
    );
    assert_eq!(
        prefill_outputs[1].1,
        [batch, kv_heads, prefill_seq, qk_head_dim],
        "present_key must preserve Q/K head width"
    );
    assert_eq!(
        prefill_outputs[2].1,
        [batch, kv_heads, prefill_seq, v_head_dim],
        "present_value must preserve V head width"
    );
    let prefill_cpu = run_cpu_opt("Attention", 23, &prefill_inputs, &prefill_outputs, &attrs);
    let prefill_gpu = run_opt("Attention", 23, &prefill_inputs, &prefill_outputs, &attrs);
    for (name, gpu, cpu) in [
        ("prefill Y", &prefill_gpu[0], &prefill_cpu[0]),
        ("prefill present_key", &prefill_gpu[1], &prefill_cpu[1]),
        ("prefill present_value", &prefill_gpu[2], &prefill_cpu[2]),
    ] {
        let gpu = f32s(gpu);
        let cpu = f32s(cpu);
        assert_close(&gpu, &cpu);
        assert!(gpu.iter().all(|value| value.is_finite()), "{name}");
    }

    let q_decode = conformance_values(batch * decode_seq * q_heads * qk_head_dim, 7 + kv_heads);
    let k_decode = conformance_values(batch * decode_seq * kv_heads * qk_head_dim, 9 + kv_heads);
    let v_decode = conformance_values(batch * decode_seq * kv_heads * v_head_dim, 11 + kv_heads);
    let total_seq = prefill_seq + decode_seq;
    let decode_outputs = vec![
        (
            DataType::Float32,
            vec![batch, decode_seq, q_heads * v_head_dim],
        ),
        (
            DataType::Float32,
            vec![batch, kv_heads, total_seq, qk_head_dim],
        ),
        (
            DataType::Float32,
            vec![batch, kv_heads, total_seq, v_head_dim],
        ),
    ];
    assert_eq!(
        decode_outputs[0].1,
        [batch, decode_seq, q_heads * v_head_dim]
    );
    assert_eq!(
        decode_outputs[1].1,
        [batch, kv_heads, total_seq, qk_head_dim]
    );
    assert_eq!(
        decode_outputs[2].1,
        [batch, kv_heads, total_seq, v_head_dim]
    );

    let decode_inputs = |present_key: &[u8], present_value: &[u8]| {
        vec![
            Some(tensor(
                DataType::Float32,
                &[batch, decode_seq, q_heads * qk_head_dim],
                &q_decode,
            )),
            Some(tensor(
                DataType::Float32,
                &[batch, decode_seq, kv_heads * qk_head_dim],
                &k_decode,
            )),
            Some(tensor(
                DataType::Float32,
                &[batch, decode_seq, kv_heads * v_head_dim],
                &v_decode,
            )),
            None,
            Some(Tensor {
                dtype: DataType::Float32,
                shape: vec![batch, kv_heads, prefill_seq, qk_head_dim],
                bytes: present_key.to_vec(),
            }),
            Some(Tensor {
                dtype: DataType::Float32,
                shape: vec![batch, kv_heads, prefill_seq, v_head_dim],
                bytes: present_value.to_vec(),
            }),
        ]
    };
    let decode_cpu_inputs = decode_inputs(&prefill_cpu[1], &prefill_cpu[2]);
    let decode_gpu_inputs = decode_inputs(&prefill_gpu[1], &prefill_gpu[2]);
    let decode_cpu = run_cpu_opt("Attention", 23, &decode_cpu_inputs, &decode_outputs, &attrs);
    let decode_gpu = run_opt("Attention", 23, &decode_gpu_inputs, &decode_outputs, &attrs);
    for (name, gpu, cpu) in [
        ("decode Y", &decode_gpu[0], &decode_cpu[0]),
        ("decode present_key", &decode_gpu[1], &decode_cpu[1]),
        ("decode present_value", &decode_gpu[2], &decode_cpu[2]),
    ] {
        let gpu = f32s(gpu);
        let cpu = f32s(cpu);
        assert_close(&gpu, &cpu);
        assert!(gpu.iter().all(|value| value.is_finite()), "{name}");
    }

    let full_inputs = vec![
        Some(tensor(
            DataType::Float32,
            &[batch, total_seq, q_heads * qk_head_dim],
            &concat_bsh(
                &q_prefill,
                &q_decode,
                batch,
                prefill_seq,
                decode_seq,
                q_heads * qk_head_dim,
            ),
        )),
        Some(tensor(
            DataType::Float32,
            &[batch, total_seq, kv_heads * qk_head_dim],
            &concat_bsh(
                &k_prefill,
                &k_decode,
                batch,
                prefill_seq,
                decode_seq,
                kv_heads * qk_head_dim,
            ),
        )),
        Some(tensor(
            DataType::Float32,
            &[batch, total_seq, kv_heads * v_head_dim],
            &concat_bsh(
                &v_prefill,
                &v_decode,
                batch,
                prefill_seq,
                decode_seq,
                kv_heads * v_head_dim,
            ),
        )),
    ];
    let full_outputs = vec![
        (
            DataType::Float32,
            vec![batch, total_seq, q_heads * v_head_dim],
        ),
        (
            DataType::Float32,
            vec![batch, kv_heads, total_seq, qk_head_dim],
        ),
        (
            DataType::Float32,
            vec![batch, kv_heads, total_seq, v_head_dim],
        ),
    ];
    let full_cpu = run_cpu_opt("Attention", 23, &full_inputs, &full_outputs, &attrs);
    let full_gpu = run_opt("Attention", 23, &full_inputs, &full_outputs, &attrs);
    for (name, gpu, cpu) in [
        ("full prefill Y", &full_gpu[0], &full_cpu[0]),
        ("full prefill present_key", &full_gpu[1], &full_cpu[1]),
        ("full prefill present_value", &full_gpu[2], &full_cpu[2]),
    ] {
        let gpu = f32s(gpu);
        let cpu = f32s(cpu);
        assert_close(&gpu, &cpu);
        assert!(gpu.iter().all(|value| value.is_finite()), "{name}");
    }

    let expected_cpu_y = concat_bsh(
        &f32s(&prefill_cpu[0]),
        &f32s(&decode_cpu[0]),
        batch,
        prefill_seq,
        decode_seq,
        q_heads * v_head_dim,
    );
    let expected_gpu_y = concat_bsh(
        &f32s(&prefill_gpu[0]),
        &f32s(&decode_gpu[0]),
        batch,
        prefill_seq,
        decode_seq,
        q_heads * v_head_dim,
    );
    assert_close(&f32s(&full_cpu[0]), &expected_cpu_y);
    assert_close(&f32s(&full_gpu[0]), &expected_gpu_y);
    assert_close(&f32s(&full_cpu[1]), &f32s(&decode_cpu[1]));
    assert_close(&f32s(&full_cpu[2]), &f32s(&decode_cpu[2]));
    assert_close(&f32s(&full_gpu[1]), &f32s(&decode_gpu[1]));
    assert_close(&f32s(&full_gpu[2]), &f32s(&decode_gpu[2]));
}

#[test]
fn standard_attention_asymmetric_3d_prefill_decode_gqa_matches_cpu() {
    asymmetric_3d_prefill_decode_cpu_cuda_case(2);
}

#[test]
fn standard_attention_asymmetric_3d_prefill_decode_mqa_matches_cpu() {
    asymmetric_3d_prefill_decode_cpu_cuda_case(1);
}

#[test]
fn standard_attention_basic_mha_matches_reference() {
    let (b, h, sq, d) = (1usize, 2usize, 3usize, 4usize);
    let q = seq_f32(b * h * sq * d);
    let k = seq_f32(b * h * sq * d);
    let v = seq_f32(b * h * sq * d);
    let (y_ref, _) = sdpa_ref(
        &RefCase {
            q: &q,
            key: &k,
            value: &v,
            batch: b,
            q_heads: h,
            q_seq: sq,
            kv_heads: h,
            total_seq: sq,
            head_size: d,
            v_head_size: d,
            mask: RefMask::None,
            is_causal: false,
            offset: 0,
            scale: None,
            softcap: 0.0,
        },
        -1,
    );
    let inputs = [
        tensor(DataType::Float32, &[b, h, sq, d], &q),
        tensor(DataType::Float32, &[b, h, sq, d], &k),
        tensor(DataType::Float32, &[b, h, sq, d], &v),
    ];
    let out = run(
        "Attention",
        23,
        &inputs,
        &[(DataType::Float32, vec![b, h, sq, d])],
        &[],
    );
    assert_close(&f32s(&out[0]), &y_ref);
}

#[test]
fn standard_attention_gqa_multi_batch_multi_head_matches_reference() {
    let (b, hq, hkv, sq, d) = (2usize, 4usize, 2usize, 3usize, 4usize);
    let q = seq_f32(b * hq * sq * d);
    let k = seq_f32(b * hkv * sq * d);
    let v = seq_f32(b * hkv * sq * d);
    let (y_ref, _) = sdpa_ref(
        &RefCase {
            q: &q,
            key: &k,
            value: &v,
            batch: b,
            q_heads: hq,
            q_seq: sq,
            kv_heads: hkv,
            total_seq: sq,
            head_size: d,
            v_head_size: d,
            mask: RefMask::None,
            is_causal: true,
            offset: 0,
            scale: None,
            softcap: 0.0,
        },
        -1,
    );
    let inputs = [
        tensor(DataType::Float32, &[b, hq, sq, d], &q),
        tensor(DataType::Float32, &[b, hkv, sq, d], &k),
        tensor(DataType::Float32, &[b, hkv, sq, d], &v),
    ];
    let attrs = [("is_causal", Attribute::Int(1))];
    let out = run(
        "Attention",
        23,
        &inputs,
        &[(DataType::Float32, vec![b, hq, sq, d])],
        &attrs,
    );
    assert_close(&f32s(&out[0]), &y_ref);
}

#[test]
fn standard_attention_3d_input_reshape_matches_reference() {
    // 3D (batch, seq, heads*dim) input; reference consumes the equivalent 4D
    // (batch, heads, seq, dim) transpose.
    let (b, h, sq, d) = (1usize, 2usize, 2usize, 2usize);
    let q3 = seq_f32(b * sq * h * d);
    let k3 = seq_f32(b * sq * h * d);
    let v3 = seq_f32(b * sq * h * d);
    // Transpose (b, s, h, d) -> (b, h, s, d) for the reference.
    let to_bhsd = |src: &[f32]| -> Vec<f32> {
        let mut dst = vec![0.0f32; src.len()];
        for bi in 0..b {
            for s in 0..sq {
                for hi in 0..h {
                    for di in 0..d {
                        let si = ((bi * sq + s) * h + hi) * d + di;
                        let dj = ((bi * h + hi) * sq + s) * d + di;
                        dst[dj] = src[si];
                    }
                }
            }
        }
        dst
    };
    let q4 = to_bhsd(&q3);
    let k4 = to_bhsd(&k3);
    let v4 = to_bhsd(&v3);
    let (y_ref_bhsd, _) = sdpa_ref(
        &RefCase {
            q: &q4,
            key: &k4,
            value: &v4,
            batch: b,
            q_heads: h,
            q_seq: sq,
            kv_heads: h,
            total_seq: sq,
            head_size: d,
            v_head_size: d,
            mask: RefMask::None,
            is_causal: false,
            offset: 0,
            scale: None,
            softcap: 0.0,
        },
        -1,
    );
    // 3D output is (batch, seq, heads*dim); transpose the reference back.
    let mut y_ref3 = vec![0.0f32; y_ref_bhsd.len()];
    for bi in 0..b {
        for hi in 0..h {
            for s in 0..sq {
                for di in 0..d {
                    let src = ((bi * h + hi) * sq + s) * d + di;
                    let dst = (bi * sq + s) * (h * d) + hi * d + di;
                    y_ref3[dst] = y_ref_bhsd[src];
                }
            }
        }
    }
    let inputs = [
        tensor(DataType::Float32, &[b, sq, h * d], &q3),
        tensor(DataType::Float32, &[b, sq, h * d], &k3),
        tensor(DataType::Float32, &[b, sq, h * d], &v3),
    ];
    let attrs = [
        ("q_num_heads", Attribute::Int(h as i64)),
        ("kv_num_heads", Attribute::Int(h as i64)),
    ];
    let out = run(
        "Attention",
        23,
        &inputs,
        &[(DataType::Float32, vec![b, sq, h * d])],
        &attrs,
    );
    assert_close(&f32s(&out[0]), &y_ref3);
}

#[test]
fn standard_attention_in_op_past_cache_matches_reference_and_present() {
    // Causal decode step: past cache of length 2, current step of length 1.
    let (b, h, d) = (1usize, 2usize, 2usize);
    let (past_seq, cur_seq) = (2usize, 1usize);
    let total = past_seq + cur_seq;
    let q = seq_f32(b * h * cur_seq * d);
    let past_k = seq_f32(b * h * past_seq * d);
    let past_v = seq_f32(b * h * past_seq * d);
    let cur_k = seq_f32(b * h * cur_seq * d);
    let cur_v = seq_f32(b * h * cur_seq * d);
    // Build present = concat(past, current) along seq for the reference.
    let concat = |past: &[f32], cur: &[f32], dim: usize| -> Vec<f32> {
        let mut out = vec![0.0f32; b * h * total * dim];
        for bi in 0..b {
            for hi in 0..h {
                for di in 0..dim {
                    for t in 0..past_seq {
                        out[((bi * h + hi) * total + t) * dim + di] =
                            past[((bi * h + hi) * past_seq + t) * dim + di];
                    }
                    for t in 0..cur_seq {
                        out[((bi * h + hi) * total + past_seq + t) * dim + di] =
                            cur[((bi * h + hi) * cur_seq + t) * dim + di];
                    }
                }
            }
        }
        out
    };
    let present_k = concat(&past_k, &cur_k, d);
    let present_v = concat(&past_v, &cur_v, d);
    let (y_ref, _) = sdpa_ref(
        &RefCase {
            q: &q,
            key: &present_k,
            value: &present_v,
            batch: b,
            q_heads: h,
            q_seq: cur_seq,
            kv_heads: h,
            total_seq: total,
            head_size: d,
            v_head_size: d,
            mask: RefMask::None,
            is_causal: true,
            offset: past_seq as i64,
            scale: None,
            softcap: 0.0,
        },
        -1,
    );
    // Inputs: Q, K, V, (mask omitted), past_key, past_value.
    let inputs = [
        Some(tensor(DataType::Float32, &[b, h, cur_seq, d], &q)),
        Some(tensor(DataType::Float32, &[b, h, cur_seq, d], &cur_k)),
        Some(tensor(DataType::Float32, &[b, h, cur_seq, d], &cur_v)),
        None,
        Some(tensor(DataType::Float32, &[b, h, past_seq, d], &past_k)),
        Some(tensor(DataType::Float32, &[b, h, past_seq, d], &past_v)),
    ];
    let attrs = [("is_causal", Attribute::Int(1))];
    let out = run_opt(
        "Attention",
        23,
        &inputs,
        &[
            (DataType::Float32, vec![b, h, cur_seq, d]),
            (DataType::Float32, vec![b, h, total, d]),
            (DataType::Float32, vec![b, h, total, d]),
        ],
        &attrs,
    );
    assert_close(&f32s(&out[0]), &y_ref);
    assert_close(&f32s(&out[1]), &present_k);
    assert_close(&f32s(&out[2]), &present_v);
}

#[test]
fn standard_attention_present_value_uses_its_own_concat_geometry() {
    // Key and value use different past/current splits but the same present
    // length. The CPU reference concatenates each cache using its own sequence
    // geometry before checking that the resulting present lengths agree.
    let (b, h, q_seq, d) = (1usize, 1usize, 2usize, 1usize);
    let (key_past_seq, key_cur_seq) = (3usize, 2usize);
    let (value_past_seq, value_cur_seq) = (2usize, 3usize);
    let total = key_past_seq + key_cur_seq;
    assert_eq!(total, value_past_seq + value_cur_seq);

    let q = [1.0f32, 1.0];
    let past_k = [1.0f32, 2.0, 3.0];
    let cur_k = [4.0f32, 5.0];
    let past_v = [100.0f32, 200.0];
    let cur_v = [300.0f32, 400.0, 500.0];
    let present_k = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let present_v = [100.0f32, 200.0, 300.0, 400.0, 500.0];
    let (y_ref, _) = sdpa_ref(
        &RefCase {
            q: &q,
            key: &present_k,
            value: &present_v,
            batch: b,
            q_heads: h,
            q_seq,
            kv_heads: h,
            total_seq: total,
            head_size: d,
            v_head_size: d,
            mask: RefMask::None,
            is_causal: true,
            offset: key_past_seq as i64,
            scale: None,
            softcap: 0.0,
        },
        -1,
    );
    let inputs = [
        Some(tensor(DataType::Float32, &[b, h, q_seq, d], &q)),
        Some(tensor(DataType::Float32, &[b, h, key_cur_seq, d], &cur_k)),
        Some(tensor(DataType::Float32, &[b, h, value_cur_seq, d], &cur_v)),
        None,
        Some(tensor(DataType::Float32, &[b, h, key_past_seq, d], &past_k)),
        Some(tensor(
            DataType::Float32,
            &[b, h, value_past_seq, d],
            &past_v,
        )),
    ];
    let out = run_opt(
        "Attention",
        23,
        &inputs,
        &[
            (DataType::Float32, vec![b, h, q_seq, d]),
            (DataType::Float32, vec![b, h, total, d]),
            (DataType::Float32, vec![b, h, total, d]),
        ],
        &[("is_causal", Attribute::Int(1))],
    );
    assert_eq!(f32s(&out[1]), present_k);
    assert_eq!(
        f32s(&out[2]),
        present_v,
        "present_value must concatenate at value_past_seq, not key_past_seq"
    );
    assert_close(&f32s(&out[0]), &y_ref);
}

#[test]
fn standard_attention_opset24_nonpad_kv_seqlen_masks_padding_and_offsets_causal() {
    let (b, h, q_seq, kv_seq, d) = (1usize, 1usize, 2usize, 4usize, 2usize);
    let nonpad = 3i64;
    let offset = nonpad - q_seq as i64;
    let q = seq_f32(b * h * q_seq * d);
    let k = seq_f32(b * h * kv_seq * d);
    let v = [10.0f32, 11.0, 20.0, 21.0, 30.0, 31.0, 900.0, 901.0];
    let inputs = [
        Some(tensor(DataType::Float32, &[b, h, q_seq, d], &q)),
        Some(tensor(DataType::Float32, &[b, h, kv_seq, d], &k)),
        Some(tensor(DataType::Float32, &[b, h, kv_seq, d], &v)),
        None,
        None,
        None,
        Some(tensor(DataType::Int64, &[b], &[nonpad])),
    ];

    for is_causal in [false, true] {
        let (y_ref, qk_ref) = sdpa_ref_with_pad_limit(
            &RefCase {
                q: &q,
                key: &k,
                value: &v,
                batch: b,
                q_heads: h,
                q_seq,
                kv_heads: h,
                total_seq: kv_seq,
                head_size: d,
                v_head_size: d,
                mask: RefMask::None,
                is_causal,
                offset,
                scale: None,
                softcap: 0.0,
            },
            2,
            Some(nonpad),
        );
        let attrs = [
            ("is_causal", Attribute::Int(i64::from(is_causal))),
            ("qk_matmul_output_mode", Attribute::Int(2)),
        ];
        let out = run_opt(
            "Attention",
            24,
            &inputs,
            &[
                (DataType::Float32, vec![b, h, q_seq, d]),
                (DataType::Float32, vec![b, h, kv_seq, d]),
                (DataType::Float32, vec![b, h, kv_seq, d]),
                (DataType::Float32, vec![b, h, q_seq, kv_seq]),
            ],
            &attrs,
        );
        assert_close(&f32s(&out[0]), &y_ref);
        let got_qk = f32s(&out[3]);
        for (idx, (&got, &expected)) in got_qk.iter().zip(&qk_ref).enumerate() {
            if expected.is_finite() {
                assert!(
                    (got - expected).abs() <= 1e-4,
                    "qk[{idx}]: {got} vs {expected}"
                );
            } else {
                assert_eq!(got, expected, "qk[{idx}] infinity mismatch");
            }
        }
        for row in got_qk.chunks_exact(kv_seq) {
            assert_eq!(
                row[3],
                f32::NEG_INFINITY,
                "padding key must always be masked"
            );
        }
        if is_causal {
            assert!(got_qk[0].is_finite() && got_qk[1].is_finite());
            assert_eq!(got_qk[2], f32::NEG_INFINITY);
            assert!(got_qk[kv_seq + 2].is_finite());
        } else {
            assert!(got_qk[..3].iter().all(|score| score.is_finite()));
        }
    }

    let error = run_result_core(
        "Attention",
        23,
        &inputs
            .iter()
            .map(|input| input.as_ref())
            .collect::<Vec<_>>(),
        &[(DataType::Float32, vec![b, h, q_seq, d])],
        &[],
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("added in opset 24"),
        "unexpected v23 error: {error}"
    );
}

#[test]
fn standard_attention_float_mask_add_matches_reference() {
    let (b, h, sq, d) = (1usize, 2usize, 2usize, 2usize);
    let q = seq_f32(b * h * sq * d);
    let k = seq_f32(b * h * sq * d);
    let v = seq_f32(b * h * sq * d);
    let mask_data = vec![0.0f32, -2.0, 1.5, 0.0];
    let (y_ref, _) = sdpa_ref(
        &RefCase {
            q: &q,
            key: &k,
            value: &v,
            batch: b,
            q_heads: h,
            q_seq: sq,
            kv_heads: h,
            total_seq: sq,
            head_size: d,
            v_head_size: d,
            mask: RefMask::Float(mask_data.clone(), vec![1, 1, sq, sq]),
            is_causal: false,
            offset: 0,
            scale: None,
            softcap: 0.0,
        },
        -1,
    );
    let inputs = [
        tensor(DataType::Float32, &[b, h, sq, d], &q),
        tensor(DataType::Float32, &[b, h, sq, d], &k),
        tensor(DataType::Float32, &[b, h, sq, d], &v),
        tensor(DataType::Float32, &[1, 1, sq, sq], &mask_data),
    ];
    let out = run(
        "Attention",
        23,
        &inputs,
        &[(DataType::Float32, vec![b, h, sq, d])],
        &[],
    );
    assert_close(&f32s(&out[0]), &y_ref);
}

#[test]
fn standard_attention_bool_mask_matches_reference() {
    let (b, h, sq, d) = (1usize, 2usize, 2usize, 2usize);
    let q = seq_f32(b * h * sq * d);
    let k = seq_f32(b * h * sq * d);
    let v = seq_f32(b * h * sq * d);
    let mask_bytes = vec![1u8, 0, 1, 1];
    let (y_ref, _) = sdpa_ref(
        &RefCase {
            q: &q,
            key: &k,
            value: &v,
            batch: b,
            q_heads: h,
            q_seq: sq,
            kv_heads: h,
            total_seq: sq,
            head_size: d,
            v_head_size: d,
            mask: RefMask::Bool(mask_bytes.clone(), vec![1, 1, sq, sq]),
            is_causal: false,
            offset: 0,
            scale: None,
            softcap: 0.0,
        },
        -1,
    );
    let inputs = [
        tensor(DataType::Float32, &[b, h, sq, d], &q),
        tensor(DataType::Float32, &[b, h, sq, d], &k),
        tensor(DataType::Float32, &[b, h, sq, d], &v),
        tensor(DataType::Bool, &[1, 1, sq, sq], &mask_bytes),
    ];
    let out = run(
        "Attention",
        23,
        &inputs,
        &[(DataType::Float32, vec![b, h, sq, d])],
        &[],
    );
    assert_close(&f32s(&out[0]), &y_ref);
}

#[test]
fn standard_attention_softcap_matches_reference() {
    let (b, h, sq, d) = (1usize, 2usize, 3usize, 4usize);
    let q = seq_f32(b * h * sq * d);
    let k = seq_f32(b * h * sq * d);
    let v = seq_f32(b * h * sq * d);
    let softcap = 2.5f32;
    let (y_ref, _) = sdpa_ref(
        &RefCase {
            q: &q,
            key: &k,
            value: &v,
            batch: b,
            q_heads: h,
            q_seq: sq,
            kv_heads: h,
            total_seq: sq,
            head_size: d,
            v_head_size: d,
            mask: RefMask::None,
            is_causal: true,
            offset: 0,
            scale: None,
            softcap,
        },
        -1,
    );
    let inputs = [
        tensor(DataType::Float32, &[b, h, sq, d], &q),
        tensor(DataType::Float32, &[b, h, sq, d], &k),
        tensor(DataType::Float32, &[b, h, sq, d], &v),
    ];
    let attrs = [
        ("is_causal", Attribute::Int(1)),
        ("softcap", Attribute::Float(softcap)),
    ];
    let out = run(
        "Attention",
        23,
        &inputs,
        &[(DataType::Float32, vec![b, h, sq, d])],
        &attrs,
    );
    assert_close(&f32s(&out[0]), &y_ref);
}

#[test]
fn standard_attention_explicit_scale_matches_reference() {
    let (b, h, sq, d) = (1usize, 1usize, 3usize, 4usize);
    let q = seq_f32(b * h * sq * d);
    let k = seq_f32(b * h * sq * d)
        .into_iter()
        .map(|value| value + 0.75)
        .collect::<Vec<_>>();
    let v = seq_f32(b * h * sq * d)
        .into_iter()
        .map(|value| value * 3.0 + 2.0)
        .collect::<Vec<_>>();
    let scale = 0.125f32;
    let (y_ref, _) = sdpa_ref(
        &RefCase {
            q: &q,
            key: &k,
            value: &v,
            batch: b,
            q_heads: h,
            q_seq: sq,
            kv_heads: h,
            total_seq: sq,
            head_size: d,
            v_head_size: d,
            mask: RefMask::None,
            is_causal: false,
            offset: 0,
            scale: Some(scale),
            softcap: 0.0,
        },
        -1,
    );
    let inputs = [
        tensor(DataType::Float32, &[b, h, sq, d], &q),
        tensor(DataType::Float32, &[b, h, sq, d], &k),
        tensor(DataType::Float32, &[b, h, sq, d], &v),
    ];
    let explicit = run(
        "Attention",
        23,
        &inputs,
        &[(DataType::Float32, vec![b, h, sq, d])],
        &[("scale", Attribute::Float(scale))],
    );
    let default = run(
        "Attention",
        23,
        &inputs,
        &[(DataType::Float32, vec![b, h, sq, d])],
        &[],
    );
    let explicit = f32s(&explicit[0]);
    assert_close(&explicit, &y_ref);
    assert!(
        explicit
            .iter()
            .zip(f32s(&default[0]))
            .any(|(explicit, default)| (explicit - default).abs() > 1e-4),
        "explicit non-default scale must affect the result"
    );
}

#[test]
fn standard_attention_fully_masked_row_is_zero() {
    // A bool mask row that is entirely `false` must yield an all-zero output row
    // (numerically-stable softmax guard), not NaN.
    let (b, h, sq, d) = (1usize, 1usize, 2usize, 2usize);
    let q = seq_f32(b * h * sq * d);
    let k = seq_f32(b * h * sq * d);
    let v = seq_f32(b * h * sq * d);
    // Row 0 fully masked; row 1 keeps key 0.
    let mask_bytes = vec![0u8, 0, 1, 0];
    let inputs = [
        tensor(DataType::Float32, &[b, h, sq, d], &q),
        tensor(DataType::Float32, &[b, h, sq, d], &k),
        tensor(DataType::Float32, &[b, h, sq, d], &v),
        tensor(DataType::Bool, &[1, 1, sq, sq], &mask_bytes),
    ];
    let out = run(
        "Attention",
        23,
        &inputs,
        &[(DataType::Float32, vec![b, h, sq, d])],
        &[],
    );
    let y = f32s(&out[0]);
    assert_eq!(&y[0..d], &[0.0, 0.0], "fully-masked row 0 must be zero");
    assert!(
        y[d..].iter().any(|&x| x != 0.0),
        "row 1 must be non-zero (attends key 0)"
    );
    let (y_ref, _) = sdpa_ref(
        &RefCase {
            q: &q,
            key: &k,
            value: &v,
            batch: b,
            q_heads: h,
            q_seq: sq,
            kv_heads: h,
            total_seq: sq,
            head_size: d,
            v_head_size: d,
            mask: RefMask::Bool(mask_bytes, vec![1, 1, sq, sq]),
            is_causal: false,
            offset: 0,
            scale: None,
            softcap: 0.0,
        },
        -1,
    );
    assert_close(&y, &y_ref);
}

#[test]
fn standard_attention_qk_matmul_output_modes_match_reference() {
    let (b, h, sq, d) = (1usize, 2usize, 2usize, 2usize);
    let q = seq_f32(b * h * sq * d);
    let k = seq_f32(b * h * sq * d);
    let v = seq_f32(b * h * sq * d);
    let mask_data = vec![0.0f32, -1.0, 0.5, 0.0];
    let softcap = 3.0f32;
    for mode in 0..=3i64 {
        let (y_ref, qk_ref) = sdpa_ref(
            &RefCase {
                q: &q,
                key: &k,
                value: &v,
                batch: b,
                q_heads: h,
                q_seq: sq,
                kv_heads: h,
                total_seq: sq,
                head_size: d,
                v_head_size: d,
                mask: RefMask::Float(mask_data.clone(), vec![1, 1, sq, sq]),
                is_causal: false,
                offset: 0,
                scale: None,
                softcap,
            },
            mode,
        );
        let inputs = [
            tensor(DataType::Float32, &[b, h, sq, d], &q),
            tensor(DataType::Float32, &[b, h, sq, d], &k),
            tensor(DataType::Float32, &[b, h, sq, d], &v),
            tensor(DataType::Float32, &[1, 1, sq, sq], &mask_data),
        ];
        let attrs = [
            ("softcap", Attribute::Float(softcap)),
            ("qk_matmul_output_mode", Attribute::Int(mode)),
        ];
        let out = run(
            "Attention",
            23,
            &inputs,
            &[
                (DataType::Float32, vec![b, h, sq, d]),
                (DataType::Float32, vec![b, h, sq, d]),
                (DataType::Float32, vec![b, h, sq, d]),
                (DataType::Float32, vec![b, h, sq, sq]),
            ],
            &attrs,
        );
        assert_close(&f32s(&out[0]), &y_ref);
        // qk_matmul_output may legitimately contain -inf (masked positions in
        // mode 2). Compare finite entries and require matching infinities.
        let got_qk = f32s(&out[3]);
        assert_eq!(got_qk.len(), qk_ref.len());
        for (idx, (&g, &e)) in got_qk.iter().zip(&qk_ref).enumerate() {
            if e.is_finite() {
                assert!((g - e).abs() <= 1e-4, "mode {mode} qk[{idx}]: {g} vs {e}");
            } else {
                assert_eq!(g, e, "mode {mode} qk[{idx}] infinity mismatch: {g} vs {e}");
            }
        }
    }
}
