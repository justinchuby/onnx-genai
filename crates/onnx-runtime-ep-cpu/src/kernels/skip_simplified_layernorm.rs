//! `com.microsoft::SkipSimplifiedLayerNormalization`: fused residual add and
//! last-axis RMS normalization.
//!
//! Floating-point inputs are widened to f32 for the calculation and narrowed
//! back to the requested output dtype.
//!
//! ```text
//! sum = input + skip + bias
//! y   = sum / sqrt(mean(sum²) + epsilon) * gamma
//! ```
//!
//! `bias` is optional and broadcasts over the last dimension. `skip` uses
//! right-aligned NumPy broadcasting, including the common `[seq, hidden]` to
//! `[batch, seq, hidden]` case.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::check_arity;
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};

pub struct SkipSimplifiedLayerNormKernel {
    epsilon: f32,
}

pub struct SkipSimplifiedLayerNormFactory;

impl KernelFactory for SkipSimplifiedLayerNormFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(SkipSimplifiedLayerNormKernel {
            epsilon: node
                .attr("epsilon")
                .and_then(|a| a.as_float())
                .unwrap_or(1e-5),
        }))
    }
}

impl Kernel for SkipSimplifiedLayerNormKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        const OP: &str = "SkipSimplifiedLayerNormalization";
        check_arity(OP, inputs, outputs, 3, 4, 1)?;
        if outputs.len() > 4 {
            return Err(EpError::KernelFailed(format!(
                "{OP}: expected at most 4 outputs, got {}",
                outputs.len()
            )));
        }
        let input = to_dense_f32_widen(OP, &inputs[0])?;
        let skip = to_dense_f32_widen(OP, &inputs[1])?;
        let gamma = to_dense_f32_widen(OP, &inputs[2])?;
        let bias = if inputs.len() == 4 && !inputs[3].is_absent() {
            Some(to_dense_f32_widen(OP, &inputs[3])?)
        } else {
            None
        };

        let shape = inputs[0].shape;
        let Some(&hidden) = shape.last() else {
            return Err(EpError::KernelFailed(format!(
                "{OP}: input must have rank at least 1"
            )));
        };
        if hidden == 0 {
            return Err(EpError::KernelFailed(format!(
                "{OP}: hidden (last) dimension must be non-empty"
            )));
        }
        if gamma.len() != hidden || inputs[2].shape != [hidden] {
            return Err(EpError::KernelFailed(format!(
                "{OP}: gamma must have shape [{hidden}], got {:?}",
                inputs[2].shape
            )));
        }
        if let Some(bias) = bias.as_deref()
            && (bias.len() != hidden || inputs[3].shape != [hidden])
        {
            return Err(EpError::KernelFailed(format!(
                "{OP}: bias must have shape [{hidden}], got {:?}",
                inputs[3].shape
            )));
        }

        let groups = input.len() / hidden;
        let writes_mean = outputs
            .get(1)
            .is_some_and(|output| is_stats_shape(output.shape, shape));
        let writes_inv_std = outputs
            .get(2)
            .is_some_and(|output| is_stats_shape(output.shape, shape));
        if inputs[1].shape == shape
            && inputs[0].is_contiguous()
            && inputs[1].is_contiguous()
            && inputs[2].is_contiguous()
            && inputs
                .get(3)
                .is_none_or(|input| input.is_absent() || input.is_contiguous())
            && outputs[0].shape == shape
            && outputs[0].dtype == DataType::Float32
            && outputs[0].is_contiguous()
            && outputs.get(3).is_some_and(|output| {
                output.shape == shape && output.dtype == DataType::Float32 && output.is_contiguous()
            })
        {
            let (output, remaining) = outputs.split_at_mut(1);
            let output = &mut output[0];
            let (stats_outputs, sum_output) = remaining.split_at_mut(2);
            let sum_output = &mut sum_output[0];
            output.validate()?;
            sum_output.validate()?;

            // SAFETY: validated contiguous f32 output views each describe exactly
            // `input.len()` writable elements. Kernel output views are exclusive
            // and disjoint by the EP API contract.
            let output = unsafe {
                std::slice::from_raw_parts_mut(output.data_ptr_mut::<f32>(), input.len())
            };
            let sum_output = unsafe {
                std::slice::from_raw_parts_mut(sum_output.data_ptr_mut::<f32>(), input.len())
            };

            let mut inv_std_vars = writes_inv_std.then(|| vec![0.0f32; groups]);
            for (group, (((input_row, skip_row), sum_row), normalized)) in input
                .chunks_exact(hidden)
                .zip(skip.chunks_exact(hidden))
                .zip(sum_output.chunks_exact_mut(hidden))
                .zip(output.chunks_exact_mut(hidden))
                .enumerate()
            {
                let square_sum =
                    assemble_sum_and_sum_squares(input_row, skip_row, bias.as_deref(), sum_row);
                let variance = square_sum / hidden as f32;
                let inv_std_var = 1.0 / (variance + self.epsilon).sqrt();
                if let Some(values) = inv_std_vars.as_mut() {
                    values[group] = inv_std_var;
                }
                normalize_and_scale(sum_row, normalized, inv_std_var, &gamma);
            }
            if writes_mean {
                write_dense_f32_narrow(OP, &mut stats_outputs[0], &vec![0.0f32; groups])?;
            }
            if let Some(inv_std_vars) = inv_std_vars {
                write_dense_f32_narrow(OP, &mut stats_outputs[1], &inv_std_vars)?;
            }
            return Ok(());
        }

        let skip_strides = broadcast_strides(inputs[1].shape, shape, OP)?;
        let input_strides = row_major_strides(shape);
        let mut sum = vec![0.0f32; input.len()];
        for (flat, value) in sum.iter_mut().enumerate() {
            let mut rem = flat;
            let mut skip_index = 0;
            for axis in 0..shape.len() {
                let coord = rem / input_strides[axis];
                rem %= input_strides[axis];
                skip_index += coord * skip_strides[axis];
            }
            *value = input[flat]
                + skip[skip_index]
                + bias.as_ref().map_or(0.0, |values| values[flat % hidden]);
        }

        let mut output = vec![0.0f32; input.len()];
        let mut inv_std_vars = writes_inv_std.then(|| vec![0.0f32; groups]);
        for group in 0..groups {
            let base = group * hidden;
            let row = &sum[base..base + hidden];
            let variance = sum_squares(row) / hidden as f32;
            let inv_std_var = 1.0 / (variance + self.epsilon).sqrt();
            if let Some(values) = inv_std_vars.as_mut() {
                values[group] = inv_std_var;
            }
            normalize_and_scale(row, &mut output[base..base + hidden], inv_std_var, &gamma);
        }

        write_dense_f32_narrow(OP, &mut outputs[0], &output)?;
        if writes_mean {
            write_dense_f32_narrow(OP, &mut outputs[1], &vec![0.0f32; groups])?;
        }
        if let Some(inv_std_vars) = inv_std_vars {
            write_dense_f32_narrow(OP, &mut outputs[2], &inv_std_vars)?;
        }
        if outputs.get(3).is_some_and(|output| output.shape == shape) {
            write_dense_f32_narrow(OP, &mut outputs[3], &sum)?;
        }
        Ok(())
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

const SIMD_LANES: usize = 8;

fn assemble_sum_and_sum_squares(
    input: &[f32],
    skip: &[f32],
    bias: Option<&[f32]>,
    sum: &mut [f32],
) -> f32 {
    debug_assert_eq!(input.len(), skip.len());
    debug_assert_eq!(input.len(), sum.len());
    debug_assert!(bias.is_none_or(|bias| bias.len() == input.len()));

    let mut lane_sums = [0.0f32; SIMD_LANES];
    let bulk_len = input.len() / SIMD_LANES * SIMD_LANES;
    let mut base = 0;
    if let Some(bias) = bias {
        while base < bulk_len {
            for (lane, lane_sum) in lane_sums.iter_mut().enumerate() {
                let index = base + lane;
                let value = input[index] + skip[index] + bias[index];
                sum[index] = value;
                *lane_sum += value * value;
            }
            base += SIMD_LANES;
        }
    } else {
        while base < bulk_len {
            for (lane, lane_sum) in lane_sums.iter_mut().enumerate() {
                let index = base + lane;
                let value = input[index] + skip[index];
                sum[index] = value;
                *lane_sum += value * value;
            }
            base += SIMD_LANES;
        }
    }

    let mut square_sum = lane_sums.into_iter().sum::<f32>();
    for index in bulk_len..input.len() {
        let value = input[index] + skip[index] + bias.map_or(0.0, |bias| bias[index]);
        sum[index] = value;
        square_sum += value * value;
    }
    square_sum
}

fn sum_squares(values: &[f32]) -> f32 {
    let mut lane_sums = [0.0f32; SIMD_LANES];
    let bulk_len = values.len() / SIMD_LANES * SIMD_LANES;
    let mut base = 0;
    while base < bulk_len {
        for (lane, lane_sum) in lane_sums.iter_mut().enumerate() {
            let value = values[base + lane];
            *lane_sum += value * value;
        }
        base += SIMD_LANES;
    }

    let mut square_sum = lane_sums.into_iter().sum::<f32>();
    for &value in &values[bulk_len..] {
        square_sum += value * value;
    }
    square_sum
}

fn normalize_and_scale(sum: &[f32], output: &mut [f32], inv_std: f32, gamma: &[f32]) {
    debug_assert_eq!(sum.len(), output.len());
    debug_assert_eq!(sum.len(), gamma.len());

    let bulk_len = sum.len() / SIMD_LANES * SIMD_LANES;
    let mut base = 0;
    while base < bulk_len {
        for lane in 0..SIMD_LANES {
            let index = base + lane;
            output[index] = sum[index] * inv_std * gamma[index];
        }
        base += SIMD_LANES;
    }
    for index in bulk_len..sum.len() {
        output[index] = sum[index] * inv_std * gamma[index];
    }
}

fn is_stats_shape(candidate: &[usize], input: &[usize]) -> bool {
    candidate.len() == input.len()
        && candidate.last() == Some(&1)
        && candidate[..candidate.len() - 1] == input[..input.len() - 1]
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

fn broadcast_strides(source: &[usize], target: &[usize], op: &str) -> Result<Vec<usize>> {
    if source.len() > target.len() {
        return Err(EpError::KernelFailed(format!(
            "{op}: skip shape {source:?} is not broadcastable to input shape {target:?}"
        )));
    }
    let source_contiguous = row_major_strides(source);
    let offset = target.len() - source.len();
    let mut strides = vec![0; target.len()];
    for axis in 0..source.len() {
        let source_dim = source[axis];
        let target_dim = target[offset + axis];
        if source_dim != 1 && source_dim != target_dim {
            return Err(EpError::KernelFailed(format!(
                "{op}: skip shape {source:?} is not broadcastable to input shape {target:?}"
            )));
        }
        if source_dim != 1 {
            strides[offset + axis] = source_contiguous[axis];
        }
    }
    Ok(strides)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuExecutionProvider;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, static_shape};
    use onnx_runtime_loader::{Model, encode_model_proto};

    fn kernel(epsilon: Option<f32>, with_bias: bool, with_sum: bool) -> Box<dyn Kernel> {
        let mut graph = Graph::new();
        graph.opset_imports.insert("com.microsoft".into(), 1);
        let mut inputs = Vec::new();
        for (name, shape) in [
            ("input", vec![1, 2, 4]),
            ("skip", vec![1, 2, 4]),
            ("gamma", vec![4]),
        ] {
            let value = graph.create_named_value(name, DataType::Float32, static_shape(shape));
            graph.add_input(value);
            inputs.push(Some(value));
        }
        if with_bias {
            let value = graph.create_named_value("bias", DataType::Float32, static_shape([4]));
            graph.add_input(value);
            inputs.push(Some(value));
        }

        let output = graph.create_named_value("output", DataType::Float32, static_shape([1, 2, 4]));
        let outputs = if with_sum {
            let mean = graph.create_value(DataType::Float32, Vec::new());
            let inv_std = graph.create_value(DataType::Float32, Vec::new());
            let sum = graph.create_named_value(
                "input_skip_bias_sum",
                DataType::Float32,
                static_shape([1, 2, 4]),
            );
            graph.add_output(sum);
            vec![output, mean, inv_std, sum]
        } else {
            vec![output]
        };
        let mut node = Node::new(
            NodeId(0),
            "SkipSimplifiedLayerNormalization",
            inputs,
            outputs,
        );
        node.domain = "com.microsoft".into();
        if let Some(epsilon) = epsilon {
            node.attributes
                .insert("epsilon".into(), Attribute::Float(epsilon));
        }
        let node_id = graph.insert_node(node);
        graph.add_output(output);

        let model = Model::new(&graph);
        let proto = encode_model_proto(&model).unwrap();
        assert_eq!(
            proto.graph.as_ref().unwrap().node[0].op_type,
            "SkipSimplifiedLayerNormalization"
        );
        CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node_id), &[], 1)
            .unwrap()
    }

    fn reference(
        input: &[f32],
        skip: &[f32],
        gamma: &[f32],
        bias: Option<&[f32]>,
        epsilon: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let hidden = gamma.len();
        let sum = input
            .iter()
            .zip(skip)
            .enumerate()
            .map(|(index, (&input, &skip))| {
                input + skip + bias.map_or(0.0, |bias| bias[index % hidden])
            })
            .collect::<Vec<_>>();
        let mut output = Vec::with_capacity(sum.len());
        for row in sum.chunks_exact(hidden) {
            let variance = row.iter().map(|value| value * value).sum::<f32>() / hidden as f32;
            let inv = 1.0 / (variance + epsilon).sqrt();
            output.extend(
                row.iter()
                    .zip(gamma)
                    .map(|(&value, &scale)| value * inv * scale),
            );
        }
        (output, sum)
    }

    fn assert_close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (index, (&got, &want)) in got.iter().zip(want).enumerate() {
            assert!((got - want).abs() < 1e-5, "{index}: {got} != {want}");
        }
    }

    #[test]
    fn skip_simplified_layer_norm_basic_writes_residual_sum() {
        let input_data = [1., 2., 3., 4., -1., 0., 1., 2.];
        let skip_data = [0.5, -1., 1., 0., 1., 2., -1., 0.5];
        let gamma_data = [1., 2., 0.5, 1.5];
        let (want, want_sum) = reference(&input_data, &skip_data, &gamma_data, None, 1e-4);
        let input = Owned::f32(&[1, 2, 4], &input_data);
        let skip = Owned::f32(&[1, 2, 4], &skip_data);
        let gamma = Owned::f32(&[4], &gamma_data);
        let mut output = Owned::zeros_f32(&[1, 2, 4]);
        let mut mean = Owned::zeros_f32(&[]);
        let mut inv_std = Owned::zeros_f32(&[]);
        let mut sum = Owned::zeros_f32(&[1, 2, 4]);
        kernel(Some(1e-4), false, true)
            .execute(
                &[input.view(), skip.view(), gamma.view()],
                &mut [
                    output.view_mut(),
                    mean.view_mut(),
                    inv_std.view_mut(),
                    sum.view_mut(),
                ],
            )
            .unwrap();
        assert_close(&output.to_f32(), &want);
        assert_close(&sum.to_f32(), &want_sum);
    }

    #[test]
    fn skip_simplified_layer_norm_bias_precedes_norm_and_sum_output() {
        let input_data = [1., 2., 3., 4., -1., 0., 1., 2.];
        let skip_data = [0.5, -1., 1., 0., 1., 2., -1., 0.5];
        let gamma_data = [1., 2., 0.5, 1.5];
        let bias_data = [0.25, -0.5, 1., 2.];
        let (want, want_sum) =
            reference(&input_data, &skip_data, &gamma_data, Some(&bias_data), 1e-4);
        let input = Owned::f32(&[1, 2, 4], &input_data);
        let skip = Owned::f32(&[1, 2, 4], &skip_data);
        let gamma = Owned::f32(&[4], &gamma_data);
        let bias = Owned::f32(&[4], &bias_data);
        let mut output = Owned::zeros_f32(&[1, 2, 4]);
        let mut mean = Owned::zeros_f32(&[]);
        let mut inv_std = Owned::zeros_f32(&[]);
        let mut sum = Owned::zeros_f32(&[1, 2, 4]);
        kernel(Some(1e-4), true, true)
            .execute(
                &[input.view(), skip.view(), gamma.view(), bias.view()],
                &mut [
                    output.view_mut(),
                    mean.view_mut(),
                    inv_std.view_mut(),
                    sum.view_mut(),
                ],
            )
            .unwrap();
        assert_close(&output.to_f32(), &want);
        assert_close(&sum.to_f32(), &want_sum);
    }

    #[test]
    fn skip_simplified_layer_norm_uses_default_epsilon() {
        let input_data = [1., 2., 3., 4., -1., 0., 1., 2.];
        let skip_data = [0.5, -1., 1., 0., 1., 2., -1., 0.5];
        let gamma_data = [1., 2., 0.5, 1.5];
        let (want, _) = reference(&input_data, &skip_data, &gamma_data, None, 1e-5);
        let input = Owned::f32(&[1, 2, 4], &input_data);
        let skip = Owned::f32(&[1, 2, 4], &skip_data);
        let gamma = Owned::f32(&[4], &gamma_data);
        let mut output = Owned::zeros_f32(&[1, 2, 4]);
        kernel(None, false, false)
            .execute(
                &[input.view(), skip.view(), gamma.view()],
                &mut [output.view_mut()],
            )
            .unwrap();
        assert_close(&output.to_f32(), &want);
    }

    #[test]
    fn skip_simplified_layer_norm_output_only_succeeds() {
        let input_data = [1., 2., 3., 4., -1., 0., 1., 2.];
        let skip_data = [0.5, -1., 1., 0., 1., 2., -1., 0.5];
        let gamma_data = [1., 2., 0.5, 1.5];
        let (want, _) = reference(&input_data, &skip_data, &gamma_data, None, 1e-4);
        let input = Owned::f32(&[1, 2, 4], &input_data);
        let skip = Owned::f32(&[1, 2, 4], &skip_data);
        let gamma = Owned::f32(&[4], &gamma_data);
        let mut output = Owned::zeros_f32(&[1, 2, 4]);
        kernel(Some(1e-4), false, false)
            .execute(
                &[input.view(), skip.view(), gamma.view()],
                &mut [output.view_mut()],
            )
            .unwrap();
        assert_close(&output.to_f32(), &want);
    }

    #[test]
    fn skip_simplified_layer_norm_broadcasts_seq_hidden_skip() {
        let input_data = (1..=16).map(|value| value as f32).collect::<Vec<_>>();
        let skip_data = [1., 0., -1., 2., 0.5, -0.5, 1.5, -1.5];
        let gamma_data = [1., 1., 1., 1.];
        let expanded_skip = skip_data.repeat(2);
        let (want, _) = reference(&input_data, &expanded_skip, &gamma_data, None, 1e-5);
        let input = Owned::f32(&[2, 2, 4], &input_data);
        let skip = Owned::f32(&[2, 4], &skip_data);
        let gamma = Owned::f32(&[4], &gamma_data);
        let mut output = Owned::zeros_f32(&[2, 2, 4]);
        SkipSimplifiedLayerNormKernel { epsilon: 1e-5 }
            .execute(
                &[input.view(), skip.view(), gamma.view()],
                &mut [output.view_mut()],
            )
            .unwrap();
        assert_close(&output.to_f32(), &want);
    }

    #[test]
    fn skip_simplified_layer_norm_vector_bulk_and_remainder_match_reference() {
        for hidden in [13, 4096] {
            let shape = [2, hidden];
            let input_data = (0..2 * hidden)
                .map(|index| (index % 31) as f32 * 0.03125 - 0.5)
                .collect::<Vec<_>>();
            let skip_data = (0..2 * hidden)
                .map(|index| (index % 17) as f32 * -0.015625 + 0.125)
                .collect::<Vec<_>>();
            let gamma_data = (0..hidden)
                .map(|index| 0.75 + (index % 11) as f32 * 0.03125)
                .collect::<Vec<_>>();
            let bias_data = (0..hidden)
                .map(|index| (index % 7) as f32 * 0.0078125 - 0.015625)
                .collect::<Vec<_>>();
            let (want, want_sum) =
                reference(&input_data, &skip_data, &gamma_data, Some(&bias_data), 1e-5);
            let input = Owned::f32(&shape, &input_data);
            let skip = Owned::f32(&shape, &skip_data);
            let gamma = Owned::f32(&[hidden], &gamma_data);
            let bias = Owned::f32(&[hidden], &bias_data);
            let mut output = Owned::zeros_f32(&shape);
            let mut mean = Owned::zeros_f32(&[2, 1]);
            let mut inv_std = Owned::zeros_f32(&[2, 1]);
            let mut sum = Owned::zeros_f32(&shape);

            SkipSimplifiedLayerNormKernel { epsilon: 1e-5 }
                .execute(
                    &[input.view(), skip.view(), gamma.view(), bias.view()],
                    &mut [
                        output.view_mut(),
                        mean.view_mut(),
                        inv_std.view_mut(),
                        sum.view_mut(),
                    ],
                )
                .unwrap();

            assert_close(&output.to_f32(), &want);
            assert_eq!(sum.to_f32(), want_sum);
            assert_eq!(mean.to_f32(), vec![0.0; 2]);
            let want_inv_std = want_sum
                .chunks_exact(hidden)
                .map(|row| {
                    1.0 / (row.iter().map(|value| value * value).sum::<f32>() / hidden as f32
                        + 1e-5)
                        .sqrt()
                })
                .collect::<Vec<_>>();
            assert_close(&inv_std.to_f32(), &want_inv_std);
        }
    }

    #[test]
    fn skip_simplified_layer_norm_f16_widens_and_narrows() {
        let input = Owned::f16(&[1, 1, 4], &[1., 2., 3., 4.]);
        let skip = Owned::f16(&[1, 1, 4], &[0.5, -1., 1., 0.]);
        let gamma = Owned::f16(&[4], &[1., 2., 0.5, 1.5]);
        let mut output = Owned::zeros(DataType::Float16, &[1, 1, 4]);
        let mut mean = Owned::zeros(DataType::Float16, &[1, 1, 1]);
        let mut inv_std = Owned::zeros(DataType::Float16, &[1, 1, 1]);
        let mut sum = Owned::zeros(DataType::Float16, &[1, 1, 4]);
        SkipSimplifiedLayerNormKernel { epsilon: 1e-4 }
            .execute(
                &[input.view(), skip.view(), gamma.view()],
                &mut [
                    output.view_mut(),
                    mean.view_mut(),
                    inv_std.view_mut(),
                    sum.view_mut(),
                ],
            )
            .unwrap();
        let (want, want_sum) = reference(
            &[1., 2., 3., 4.],
            &[0.5, -1., 1., 0.],
            &[1., 2., 0.5, 1.5],
            None,
            1e-4,
        );
        for (got, expected) in output.to_f16_as_f32().iter().zip(&want) {
            assert!((got - expected).abs() < 1e-3);
        }
        for (got, expected) in sum.to_f16_as_f32().iter().zip(&want_sum) {
            assert!((got - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn skip_simplified_layer_norm_bf16_widens_and_narrows() {
        let input = Owned::bf16(&[1, 1, 4], &[1., 2., 3., 4.]);
        let skip = Owned::bf16(&[1, 1, 4], &[0.5, -1., 1., 0.]);
        let gamma = Owned::bf16(&[4], &[1., 2., 0.5, 1.5]);
        let mut output = Owned::zeros(DataType::BFloat16, &[1, 1, 4]);
        SkipSimplifiedLayerNormKernel { epsilon: 1e-4 }
            .execute(
                &[input.view(), skip.view(), gamma.view()],
                &mut [output.view_mut()],
            )
            .unwrap();
        let (want, _) = reference(
            &[1., 2., 3., 4.],
            &[0.5, -1., 1., 0.],
            &[1., 2., 0.5, 1.5],
            None,
            1e-4,
        );
        for (got, expected) in output.to_bf16_as_f32().iter().zip(&want) {
            assert!((got - expected).abs() < 1e-2);
        }
    }
}
