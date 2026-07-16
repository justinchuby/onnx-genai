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

    let mut packed_inputs = base;
    packed_inputs[1] = None;
    let error = run(&ep, &attrs(&[]), &packed_inputs, &[vec![1, 1, 8]])
        .expect_err("packed QKV must be rejected");
    assert!(format!("{error}").contains("packed QKV"));
}
