//! CUDA conformance tests for router/mask indexing and scan operators.

use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, KernelMatch, Result, TensorMut,
    TensorView,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
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
fn standard_attention_and_rope_claim_only_f32_and_require_contiguous_inputs() {
    let ep = CudaExecutionProvider::new_default().expect("CUDA runtime must be available");

    for (op_type, opset, dtype, expected_reason) in [
        ("Attention", 23, DataType::Float16, "Attention: dtype f16"),
        ("Attention", 23, DataType::BFloat16, "Attention: dtype bf16"),
        (
            "RotaryEmbedding",
            23,
            DataType::Float16,
            "RotaryEmbedding: dtype f16",
        ),
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
                "RotaryEmbedding validates device position_ids with a host synchronization"
            );
        }
    }
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
    let ep = CudaExecutionProvider::new_default().expect("CUDA runtime must be available");
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), opset);
    let input_values = inputs
        .iter()
        .enumerate()
        .map(|(i, input)| {
            let value = graph.create_named_value(
                &format!("input_{i}"),
                input.dtype,
                static_shape(input.shape.iter().copied()),
            );
            graph.add_input(value);
            value
        })
        .collect::<Vec<_>>();
    let output_values = outputs
        .iter()
        .enumerate()
        .map(|(i, (dtype, shape))| {
            graph.create_named_value(
                &format!("output_{i}"),
                *dtype,
                static_shape(shape.iter().copied()),
            )
        })
        .collect::<Vec<_>>();
    let mut node = Node::new(
        NodeId(0),
        op,
        input_values.into_iter().map(Some).collect(),
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
        .map(|input| -> Result<DeviceBuffer> {
            let buffer = ep.allocate(input.bytes.len(), 256)?;
            if !input.bytes.is_empty() {
                unsafe { ep.runtime().htod(&input.bytes, cuptr(buffer.as_ptr()))? };
            }
            Ok(buffer)
        })
        .collect::<Result<Vec<_>>>()?;
    let input_strides = inputs
        .iter()
        .map(|input| compute_contiguous_strides(&input.shape))
        .collect::<Vec<_>>();
    let input_views = inputs
        .iter()
        .zip(&input_buffers)
        .zip(&input_strides)
        .map(|((input, buffer), strides)| {
            TensorView::new(
                DevicePtr(buffer.as_ptr()),
                input.dtype,
                &input.shape,
                strides,
                ep.device_id(),
            )
        })
        .collect::<Vec<_>>();
    let mut output_buffers = outputs
        .iter()
        .map(|(dtype, shape)| -> Result<DeviceBuffer> {
            Ok(ep.allocate(dtype.storage_bytes(shape.iter().product()), 256)?)
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
    if let Err(error) = kernel.execute(&input_views, &mut output_views) {
        for buffer in input_buffers {
            ep.deallocate(buffer)?;
        }
        for buffer in output_buffers {
            ep.deallocate(buffer)?;
        }
        return Err(error);
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
    for buffer in input_buffers {
        ep.deallocate(buffer)?;
    }
    for buffer in output_buffers {
        ep.deallocate(buffer)?;
    }
    Ok(result)
}
fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
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
