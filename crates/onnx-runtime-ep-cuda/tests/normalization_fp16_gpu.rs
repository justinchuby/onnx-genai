//! fp16 activation parity for CUDA LayerNorm and RMSNorm.

use half::f16;
use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, Result, TensorMut, TensorView,
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

fn bytes<T: Copy>(values: &[T]) -> Vec<u8> {
    // SAFETY: test inputs are fixed-width plain data.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)).to_vec()
    }
}

fn half_tensor(shape: &[usize], values: &[f16]) -> HostTensor {
    HostTensor {
        dtype: DataType::Float16,
        shape: shape.to_vec(),
        bytes: bytes(values),
    }
}

fn float_tensor(shape: &[usize], values: &[f32]) -> HostTensor {
    HostTensor {
        dtype: DataType::Float32,
        shape: shape.to_vec(),
        bytes: bytes(values),
    }
}

fn run_norm(
    op: &str,
    opset: u64,
    inputs: &[HostTensor],
    output_specs: &[(DataType, Vec<usize>)],
    attrs: &[(&str, Attribute)],
) -> Result<Vec<Vec<u8>>> {
    let ep = CudaExecutionProvider::new_default()?;
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), opset);
    let input_ids = inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            let id = graph.create_named_value(
                format!("input_{index}"),
                input.dtype,
                static_shape(input.shape.iter().copied()),
            );
            graph.add_input(id);
            Some(id)
        })
        .collect();
    let output_ids = output_specs
        .iter()
        .enumerate()
        .map(|(index, (dtype, shape))| {
            graph.create_named_value(
                format!("output_{index}"),
                *dtype,
                static_shape(shape.iter().copied()),
            )
        })
        .collect::<Vec<_>>();
    let mut node = Node::new(NodeId(0), op, input_ids, output_ids.clone());
    for (name, value) in attrs {
        node.attributes.insert((*name).into(), value.clone());
    }
    let node_id = graph.insert_node(node);
    for &output in &output_ids {
        graph.add_output(output);
    }
    let model = Model::new(&graph);
    let kernel = ep.get_kernel(model.graph.node(node_id), &[], opset)?;

    let input_buffers = inputs
        .iter()
        .map(|input| -> Result<DeviceBuffer> {
            let buffer = ep.allocate(input.bytes.len(), 256)?;
            unsafe { ep.runtime().htod(&input.bytes, cuptr(buffer.as_ptr()))? };
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

    let mut output_buffers = output_specs
        .iter()
        .map(|(dtype, shape)| ep.allocate(dtype.storage_bytes(shape.iter().product()), 256))
        .collect::<Result<Vec<_>>>()?;
    let output_strides = output_specs
        .iter()
        .map(|(_, shape)| compute_contiguous_strides(shape))
        .collect::<Vec<_>>();
    let mut output_views = output_specs
        .iter()
        .zip(&mut output_buffers)
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
    kernel.execute(&input_views, &mut output_views)?;
    let allocations_after = ep.runtime().allocation_counts();
    assert_eq!(
        allocations_after, allocations_before,
        "{op} launch path must not allocate or free CUDA memory"
    );
    assert!(
        kernel.cuda_graph_compatible(),
        "{op} warmed fp16 path must remain capture-supported"
    );

    let result = output_specs
        .iter()
        .zip(&output_buffers)
        .map(|((dtype, shape), buffer)| -> Result<Vec<u8>> {
            let mut output = vec![0; dtype.storage_bytes(shape.iter().product())];
            unsafe { ep.runtime().dtoh(&mut output, cuptr(buffer.as_ptr()))? };
            Ok(output)
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

fn decode_half(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|raw| f16::from_bits(u16::from_ne_bytes(raw.try_into().unwrap())).to_f32())
        .collect()
}

fn decode_float(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|raw| f32::from_ne_bytes(raw.try_into().unwrap()))
        .collect()
}

fn max_abs_error(got: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(got.len(), expected.len());
    got.iter()
        .zip(expected)
        .map(|(&got, &expected)| {
            assert!(got.is_finite(), "CUDA output contains non-finite value");
            (got - expected).abs()
        })
        .fold(0.0, f32::max)
}

fn rms_reference(input: &[f32], scale: &[f32], norm_size: usize, epsilon: f32) -> Vec<f32> {
    input
        .chunks_exact(norm_size)
        .flat_map(|group| {
            let sum_squares = group.iter().fold(0.0f32, |sum, &value| sum + value * value);
            let inv_std = 1.0 / (sum_squares / norm_size as f32 + epsilon).sqrt();
            group
                .iter()
                .zip(scale)
                .map(move |(&value, &weight)| value * inv_std * weight)
        })
        .collect()
}

fn layer_reference(
    input: &[f32],
    scale: &[f32],
    bias: &[f32],
    norm_size: usize,
    epsilon: f32,
) -> Vec<f32> {
    input
        .chunks_exact(norm_size)
        .flat_map(|group| {
            let mean = group.iter().fold(0.0f32, |sum, &value| sum + value) / norm_size as f32;
            let variance = group.iter().fold(0.0f32, |sum, &value| {
                let delta = value - mean;
                sum + delta * delta
            }) / norm_size as f32;
            let inv_std = 1.0 / (variance + epsilon).sqrt();
            group
                .iter()
                .zip(scale)
                .zip(bias)
                .map(move |((&value, &weight), &offset)| (value - mean) * inv_std * weight + offset)
        })
        .collect()
}

fn test_values(norm_size: usize) -> (Vec<f16>, Vec<f16>, Vec<f32>, Vec<f16>, Vec<f32>) {
    let input = (0..norm_size)
        .map(|index| {
            let value = ((index * 37 + 11) % 257) as f32 / 64.0 - 2.0;
            f16::from_f32(value)
        })
        .collect();
    let scale_f32 = (0..norm_size)
        .map(|index| 0.75 + (index % 29) as f32 / 64.0)
        .collect::<Vec<_>>();
    let scale_half = scale_f32
        .iter()
        .copied()
        .map(f16::from_f32)
        .collect::<Vec<_>>();
    let bias_f32 = (0..norm_size)
        .map(|index| (index % 17) as f32 / 128.0 - 0.0625)
        .collect::<Vec<_>>();
    let bias_half = bias_f32
        .iter()
        .copied()
        .map(f16::from_f32)
        .collect::<Vec<_>>();
    (input, scale_half, scale_f32, bias_half, bias_f32)
}

#[test]
fn fp16_rmsnorm_parallel_reduction_matches_serial_f32_reference() {
    let norm_size = 896;
    let epsilon = 1e-5;
    let shape = [1, norm_size];
    let (input, scale_half, scale_f32, _, _) = test_values(norm_size);
    let input_f32 = input.iter().map(|value| value.to_f32()).collect::<Vec<_>>();

    for (label, scale) in [
        ("fp16-scale", half_tensor(&[norm_size], &scale_half)),
        ("fp32-scale", float_tensor(&[norm_size], &scale_f32)),
    ] {
        let scale_reference = match scale.dtype {
            DataType::Float16 => scale_half.iter().map(|value| value.to_f32()).collect(),
            DataType::Float32 => scale_f32.clone(),
            _ => unreachable!(),
        };
        let expected = rms_reference(&input_f32, &scale_reference, norm_size, epsilon);
        let outputs = run_norm(
            "RMSNormalization",
            23,
            &[half_tensor(&shape, &input), scale],
            &[
                (DataType::Float16, shape.to_vec()),
                (DataType::Float32, vec![1]),
            ],
            &[
                ("axis", Attribute::Int(-1)),
                ("epsilon", Attribute::Float(epsilon)),
            ],
        )
        .unwrap();
        let error = max_abs_error(&decode_half(&outputs[0]), &expected);
        let invstd = decode_float(&outputs[1]);
        assert!(invstd.iter().all(|value| value.is_finite()));
        println!("fp16 RMSNorm {label} max_abs_error={error:.9e}");
        assert!(
            error <= 2e-3,
            "fp16 output rounding plus parallel f32 reduction exceeded tolerance"
        );
    }
}

#[test]
fn fp16_layernorm_matches_serial_f32_reference_for_half_and_float_affine() {
    let norm_size = 1024;
    let epsilon = 1e-5;
    let shape = [1, norm_size];
    let (input, scale_half, scale_f32, bias_half, bias_f32) = test_values(norm_size);
    let input_f32 = input.iter().map(|value| value.to_f32()).collect::<Vec<_>>();

    for (label, scale, bias, scale_reference, bias_reference) in [
        (
            "fp16-affine",
            half_tensor(&[norm_size], &scale_half),
            half_tensor(&[norm_size], &bias_half),
            scale_half.iter().map(|value| value.to_f32()).collect(),
            bias_half.iter().map(|value| value.to_f32()).collect(),
        ),
        (
            "fp32-affine",
            float_tensor(&[norm_size], &scale_f32),
            float_tensor(&[norm_size], &bias_f32),
            scale_f32.clone(),
            bias_f32.clone(),
        ),
    ] {
        let expected = layer_reference(
            &input_f32,
            &scale_reference,
            &bias_reference,
            norm_size,
            epsilon,
        );
        for iteration in 0..4 {
            let outputs = run_norm(
                "LayerNormalization",
                17,
                &[half_tensor(&shape, &input), scale.clone(), bias.clone()],
                &[
                    (DataType::Float16, shape.to_vec()),
                    (DataType::Float32, vec![1]),
                    (DataType::Float32, vec![1]),
                ],
                &[
                    ("axis", Attribute::Int(-1)),
                    ("epsilon", Attribute::Float(epsilon)),
                ],
            )
            .unwrap();
            let error = max_abs_error(&decode_half(&outputs[0]), &expected);
            assert!(
                decode_float(&outputs[1])
                    .into_iter()
                    .all(|value| value.is_finite())
            );
            assert!(
                decode_float(&outputs[2])
                    .into_iter()
                    .all(|value| value.is_finite())
            );
            println!("fp16 LayerNorm {label} iteration {iteration} max_abs_error={error:.9e}");
            assert!(
                error <= 2e-3,
                "fp16 output rounding plus parallel f32 reductions exceeded tolerance"
            );
        }
    }
}
