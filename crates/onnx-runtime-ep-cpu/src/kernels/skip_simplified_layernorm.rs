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
        let gamma_is_exact_identity = crate::kernels::simd_normalize::scale_shape_is_exact_identity(
            shape,
            shape.len() - 1,
            inputs[2].shape,
        );
        if gamma.len() != hidden || !gamma_is_exact_identity {
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
        // `input_skip_bias_sum` is optional. Materialize the whole X-shaped sum
        // only when the graph actually consumes it; otherwise one reusable
        // `hidden`-sized row scratch stays resident in L1 across every group.
        let writes_sum = outputs.get(3).is_some_and(|output| output.shape == shape);
        let sum_output_is_direct =
            writes_sum && outputs[3].dtype == DataType::Float32 && outputs[3].is_contiguous();
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
            && (sum_output_is_direct || !writes_sum)
        {
            let (output, remaining) = outputs.split_at_mut(1);
            let output = &mut output[0];
            output.validate()?;
            // `remaining` is `outputs[1..]`: at most `mean`, `inv_std_var` and
            // `input_skip_bias_sum`. A node may bind as few as one output, so
            // split defensively instead of assuming all four are present.
            let (stats_outputs, sum_outputs) = if remaining.len() > 2 {
                remaining.split_at_mut(2)
            } else {
                let len = remaining.len();
                remaining.split_at_mut(len)
            };

            // SAFETY: a validated contiguous f32 output view describes exactly
            // `input.len()` writable elements. Kernel output views are exclusive
            // and disjoint by the EP API contract.
            let output = unsafe {
                std::slice::from_raw_parts_mut(output.data_ptr_mut::<f32>(), input.len())
            };
            let mut sum_scratch;
            // `sum_stride` is `hidden` when each group owns a distinct row of
            // the X-shaped output tensor, and 0 when every group reuses the
            // single scratch row.
            let (sum_buffer, sum_stride): (&mut [f32], usize) = if sum_output_is_direct {
                let sum_output = &mut sum_outputs[0];
                sum_output.validate()?;
                // SAFETY: as above for the primary output.
                let sum_output = unsafe {
                    std::slice::from_raw_parts_mut(sum_output.data_ptr_mut::<f32>(), input.len())
                };
                (sum_output, hidden)
            } else {
                sum_scratch = vec![0.0f32; hidden];
                (&mut sum_scratch, 0)
            };

            let mut inv_std_vars = writes_inv_std.then(|| vec![0.0f32; groups]);
            for (group, (input_row, normalized)) in input
                .chunks_exact(hidden)
                .zip(output.chunks_exact_mut(hidden))
                .enumerate()
            {
                let skip_row = &skip[group * hidden..group * hidden + hidden];
                let sum_base = group * sum_stride;
                let sum_row = &mut sum_buffer[sum_base..sum_base + hidden];
                let square_sum = crate::kernels::simd_sumsq::assemble_and_sum_of_squares(
                    input_row,
                    skip_row,
                    bias.as_deref(),
                    sum_row,
                );
                let variance = square_sum / hidden as f32;
                let inv_std_var = 1.0 / (variance + self.epsilon).sqrt();
                if let Some(values) = inv_std_vars.as_mut() {
                    values[group] = inv_std_var;
                }
                crate::kernels::simd_normalize::normalize_and_scale(
                    sum_row,
                    normalized,
                    inv_std_var,
                    &gamma,
                );
            }
            if writes_mean {
                write_dense_f32_narrow(OP, &mut stats_outputs[0], &vec![0.0f32; groups])?;
            }
            if let Some(inv_std_vars) = inv_std_vars {
                write_dense_f32_narrow(OP, &mut stats_outputs[1], &inv_std_vars)?;
            }
            return Ok(());
        }

        // General path: a narrowing/strided output, or a broadcasting `skip`.
        // `skip` is resolved one row at a time — the previous implementation
        // unravelled every *element* through `rank` integer divisions, which
        // cost more than the normalization itself.
        let skip_strides = broadcast_strides(inputs[1].shape, shape, OP)?;
        let rank = shape.len();
        let skip_last_stride = skip_strides[rank - 1];
        let group_shape = &shape[..rank - 1];
        let group_skip_strides = &skip_strides[..rank - 1];

        let mut output = vec![0.0f32; input.len()];
        let mut sum_scratch = vec![0.0f32; if writes_sum { input.len() } else { hidden }];
        let sum_stride = if writes_sum { hidden } else { 0 };
        let mut inv_std_vars = writes_inv_std.then(|| vec![0.0f32; groups]);
        // Coordinates over the leading (group) axes, advanced odometer-style so
        // the per-row `skip` base is maintained without any division.
        let mut group_coords = vec![0usize; group_shape.len()];
        let mut skip_base = 0usize;
        // When `skip` broadcasts along the normalized axis there is one scalar
        // per row. Materializing it into a row lets every path in this kernel
        // share one reduction, so a broadcast `skip` and its expansion agree
        // bit-for-bit on every ISA.
        let mut broadcast_skip_row = if skip_last_stride == 0 {
            vec![0.0f32; hidden]
        } else {
            Vec::new()
        };
        for group in 0..groups {
            let input_row = &input[group * hidden..group * hidden + hidden];
            let sum_base = group * sum_stride;
            let sum_row = &mut sum_scratch[sum_base..sum_base + hidden];
            let skip_row = if skip_last_stride == 1 {
                &skip[skip_base..skip_base + hidden]
            } else {
                broadcast_skip_row.fill(skip[skip_base]);
                &broadcast_skip_row
            };
            let square_sum = crate::kernels::simd_sumsq::assemble_and_sum_of_squares(
                input_row,
                skip_row,
                bias.as_deref(),
                sum_row,
            );
            let variance = square_sum / hidden as f32;
            let inv_std_var = 1.0 / (variance + self.epsilon).sqrt();
            if let Some(values) = inv_std_vars.as_mut() {
                values[group] = inv_std_var;
            }
            crate::kernels::simd_normalize::normalize_and_scale(
                sum_row,
                &mut output[group * hidden..group * hidden + hidden],
                inv_std_var,
                &gamma,
            );
            // Advance the odometer to the next group's `skip` row.
            for axis in (0..group_shape.len()).rev() {
                group_coords[axis] += 1;
                skip_base += group_skip_strides[axis];
                if group_coords[axis] < group_shape[axis] {
                    break;
                }
                skip_base -= group_coords[axis] * group_skip_strides[axis];
                group_coords[axis] = 0;
            }
        }

        write_dense_f32_narrow(OP, &mut outputs[0], &output)?;
        if writes_mean {
            write_dense_f32_narrow(OP, &mut outputs[1], &vec![0.0f32; groups])?;
        }
        if let Some(inv_std_vars) = inv_std_vars {
            write_dense_f32_narrow(OP, &mut outputs[2], &inv_std_vars)?;
        }
        if writes_sum {
            write_dense_f32_narrow(OP, &mut outputs[3], &sum_scratch)?;
        }
        Ok(())
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
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
    fn skip_simplified_layer_norm_simd_matches_scalar_across_dtypes() {
        // The f32 four-output form takes the SIMD sum-square reduction. Compare
        // it with the scalar reference for both its remainder and bulk paths.
        for (hidden, with_bias) in [(13, false), (257, true)] {
            let shape = [2, hidden];
            let input_data = (0..2 * hidden)
                .map(|i| (i % 37) as f32 * 0.046875 - 0.75)
                .collect::<Vec<_>>();
            let skip_data = (0..2 * hidden)
                .map(|i| (i % 23) as f32 * -0.03125 + 0.3125)
                .collect::<Vec<_>>();
            let gamma_data = (0..hidden)
                .map(|i| 0.625 + (i % 13) as f32 * 0.0390625)
                .collect::<Vec<_>>();
            let bias_data = (0..hidden)
                .map(|i| (i % 9) as f32 * 0.015625 - 0.0625)
                .collect::<Vec<_>>();
            let bias = with_bias.then_some(bias_data.as_slice());
            let (want, want_sum) = reference(&input_data, &skip_data, &gamma_data, bias, 1e-5);

            let input = Owned::f32(&shape, &input_data);
            let skip = Owned::f32(&shape, &skip_data);
            let gamma = Owned::f32(&[hidden], &gamma_data);
            let bias = Owned::f32(&[hidden], &bias_data);
            let mut output = Owned::zeros_f32(&shape);
            let mut mean = Owned::zeros_f32(&[2, 1]);
            let mut inv_std = Owned::zeros_f32(&[2, 1]);
            let mut sum = Owned::zeros_f32(&shape);
            let inputs = if with_bias {
                vec![input.view(), skip.view(), gamma.view(), bias.view()]
            } else {
                vec![input.view(), skip.view(), gamma.view()]
            };
            SkipSimplifiedLayerNormKernel { epsilon: 1e-5 }
                .execute(
                    &inputs,
                    &mut [
                        output.view_mut(),
                        mean.view_mut(),
                        inv_std.view_mut(),
                        sum.view_mut(),
                    ],
                )
                .unwrap();

            assert_eq!(sum.to_f32(), want_sum, "hidden={hidden}");
            assert_eq!(mean.to_f32(), vec![0.0; 2], "hidden={hidden}");
            assert_close(&output.to_f32(), &want);
            let want_inv_std = want_sum
                .chunks_exact(hidden)
                .map(|row| {
                    1.0 / (row.iter().map(|x| x * x).sum::<f32>() / hidden as f32 + 1e-5).sqrt()
                })
                .collect::<Vec<_>>();
            assert_close(&inv_std.to_f32(), &want_inv_std);
        }

        // Reduced-precision tensors use the scalar widened path. Verify that
        // its narrowing agrees with the same scalar oracle for both dtypes.
        for (dtype, hidden, with_bias) in [
            (DataType::Float16, 13, false),
            (DataType::BFloat16, 257, true),
        ] {
            let shape = [2, hidden];
            let values = (0..2 * hidden)
                .map(|i| (i % 29) as f32 * 0.0625 - 0.875)
                .collect::<Vec<_>>();
            let skip_values = (0..2 * hidden)
                .map(|i| (i % 19) as f32 * -0.03125 + 0.25)
                .collect::<Vec<_>>();
            let gamma_values = (0..hidden)
                .map(|i| 0.75 + (i % 7) as f32 * 0.0625)
                .collect::<Vec<_>>();
            let bias_values = (0..hidden)
                .map(|i| (i % 5) as f32 * 0.03125 - 0.0625)
                .collect::<Vec<_>>();
            let (input, skip, gamma, bias) = match dtype {
                DataType::Float16 => (
                    Owned::f16(&shape, &values),
                    Owned::f16(&shape, &skip_values),
                    Owned::f16(&[hidden], &gamma_values),
                    Owned::f16(&[hidden], &bias_values),
                ),
                DataType::BFloat16 => (
                    Owned::bf16(&shape, &values),
                    Owned::bf16(&shape, &skip_values),
                    Owned::bf16(&[hidden], &gamma_values),
                    Owned::bf16(&[hidden], &bias_values),
                ),
                _ => unreachable!(),
            };
            let widen = |tensor: &Owned| match dtype {
                DataType::Float16 => tensor.to_f16_as_f32(),
                DataType::BFloat16 => tensor.to_bf16_as_f32(),
                _ => unreachable!(),
            };
            let input_values = widen(&input);
            let skip_values = widen(&skip);
            let gamma_values = widen(&gamma);
            let bias_values = widen(&bias);
            let (want, want_sum) = reference(
                &input_values,
                &skip_values,
                &gamma_values,
                with_bias.then_some(bias_values.as_slice()),
                1e-5,
            );
            let mut output = Owned::zeros(dtype, &shape);
            let mut mean = Owned::zeros(dtype, &[2, 1]);
            let mut inv_std = Owned::zeros(dtype, &[2, 1]);
            let mut sum = Owned::zeros(dtype, &shape);
            let inputs = if with_bias {
                vec![input.view(), skip.view(), gamma.view(), bias.view()]
            } else {
                vec![input.view(), skip.view(), gamma.view()]
            };
            SkipSimplifiedLayerNormKernel { epsilon: 1e-5 }
                .execute(
                    &inputs,
                    &mut [
                        output.view_mut(),
                        mean.view_mut(),
                        inv_std.view_mut(),
                        sum.view_mut(),
                    ],
                )
                .unwrap();
            let got = widen(&output);
            let got_sum = widen(&sum);
            let got_inv_std = widen(&inv_std);
            let narrow = |values: &[f32]| match dtype {
                DataType::Float16 => Owned::f16(&[values.len()], values).to_f16_as_f32(),
                DataType::BFloat16 => Owned::bf16(&[values.len()], values).to_bf16_as_f32(),
                _ => unreachable!(),
            };
            assert_eq!(got_sum, narrow(&want_sum), "{dtype:?} sum");
            assert_eq!(widen(&mean), vec![0.0; 2], "{dtype:?} mean");
            let tolerance = if dtype == DataType::Float16 {
                1e-3
            } else {
                1e-2
            };
            for (got, want) in got.iter().zip(narrow(&want)) {
                assert!(
                    (got - want).abs() <= tolerance,
                    "{dtype:?}: {got} != {want}"
                );
            }
            let want_inv_std = want_sum
                .chunks_exact(hidden)
                .map(|row| {
                    1.0 / (row.iter().map(|x| x * x).sum::<f32>() / hidden as f32 + 1e-5).sqrt()
                })
                .collect::<Vec<_>>();
            for (got, want) in got_inv_std.iter().zip(narrow(&want_inv_std)) {
                assert!(
                    (got - want).abs() <= tolerance,
                    "{dtype:?}: {got} != {want}"
                );
            }
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

    /// Deterministic pseudo-random values covering positive/negative magnitudes.
    fn values(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as i32 - (1 << 23)) as f32 * (1.0 / (1 << 20) as f32)
            })
            .collect()
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    /// Binding `input_skip_bias_sum` selects a different internal buffer for the
    /// assembled row; it must not perturb the primary output by even one bit.
    #[test]
    fn skip_simplified_layer_norm_sum_binding_is_output_bit_identical() {
        for shape in [
            vec![1usize, 1, 896],
            vec![1, 1, 4096],
            vec![1, 7, 3],
            vec![2, 3, 1537],
            vec![5],
            vec![2, 2, 3, 17],
        ] {
            let hidden = *shape.last().unwrap();
            let len: usize = shape.iter().product();
            let groups = len / hidden;
            let mut stats_shape = shape.clone();
            *stats_shape.last_mut().unwrap() = 1;
            for with_bias in [false, true] {
                let input = Owned::f32(&shape, &values(len, 11));
                let skip = Owned::f32(&shape, &values(len, 29));
                let gamma = Owned::f32(&[hidden], &values(hidden, 43));
                let bias = Owned::f32(&[hidden], &values(hidden, 71));
                let mut inputs = vec![input.view(), skip.view(), gamma.view()];
                if with_bias {
                    inputs.push(bias.view());
                }

                let mut only_output = Owned::zeros_f32(&shape);
                SkipSimplifiedLayerNormKernel { epsilon: 1e-5 }
                    .execute(&inputs, &mut [only_output.view_mut()])
                    .unwrap();

                let mut full_output = Owned::zeros_f32(&shape);
                let mut mean = Owned::zeros_f32(&stats_shape);
                let mut inv_std = Owned::zeros_f32(&stats_shape);
                let mut sum = Owned::zeros_f32(&shape);
                SkipSimplifiedLayerNormKernel { epsilon: 1e-5 }
                    .execute(
                        &inputs,
                        &mut [
                            full_output.view_mut(),
                            mean.view_mut(),
                            inv_std.view_mut(),
                            sum.view_mut(),
                        ],
                    )
                    .unwrap();

                assert_eq!(
                    bits(&only_output.to_f32()),
                    bits(&full_output.to_f32()),
                    "shape {shape:?} bias={with_bias}: output differs when the \
                     residual-sum slot is bound"
                );
                assert_eq!(mean.to_f32(), vec![0.0f32; groups]);
                // The residual sum must equal input + skip (+ bias) exactly.
                let expected_sum: Vec<f32> = input
                    .to_f32()
                    .iter()
                    .zip(skip.to_f32().iter())
                    .enumerate()
                    .map(|(index, (&x, &s))| {
                        x + s
                            + if with_bias {
                                gamma_free_bias(&bias.to_f32(), index, hidden)
                            } else {
                                0.0
                            }
                    })
                    .collect();
                assert_eq!(bits(&sum.to_f32()), bits(&expected_sum), "shape {shape:?}");
            }
        }
    }

    fn gamma_free_bias(bias: &[f32], index: usize, hidden: usize) -> f32 {
        bias[index % hidden]
    }

    /// A broadcasting `skip` must produce exactly what the materially expanded
    /// `skip` produces — the row-resolution fast path replaced a per-element
    /// index unravel, so every broadcast form needs a bit-exact guard.
    #[test]
    fn skip_simplified_layer_norm_broadcast_matches_expanded_skip() {
        // `hidden` must cross the 8-lane accumulator width used by
        // `simd_sumsq`: below it the lane-parallel reduction degenerates to a
        // serial remainder loop and cannot distinguish accumulation orders.
        for hidden in [5usize, 8, 33, 128] {
            let shape = [2usize, 3, hidden];
            let len: usize = shape.iter().product();
            let input_data = values(len, 101);
            let gamma_data = values(hidden, 103);

            for skip_shape in [
                vec![hidden],
                vec![1, hidden],
                vec![3, hidden],
                vec![1, 3, hidden],
                vec![2, 1, hidden],
                vec![2, 3, hidden],
                vec![1],
                vec![3, 1],
                vec![2, 3, 1],
            ] {
                let skip_len: usize = skip_shape.iter().product();
                let skip_data = values(skip_len, 107);

                // Materialize the broadcast by hand, right-aligned NumPy rules.
                let offset = shape.len() - skip_shape.len();
                let mut expanded = vec![0.0f32; len];
                for (flat, slot) in expanded.iter_mut().enumerate() {
                    let mut remainder = flat;
                    let mut coords = [0usize; 3];
                    for axis in (0..shape.len()).rev() {
                        coords[axis] = remainder % shape[axis];
                        remainder /= shape[axis];
                    }
                    let mut source = 0usize;
                    let mut stride = 1usize;
                    for axis in (0..skip_shape.len()).rev() {
                        let coord = if skip_shape[axis] == 1 {
                            0
                        } else {
                            coords[offset + axis]
                        };
                        source += coord * stride;
                        stride *= skip_shape[axis];
                    }
                    *slot = skip_data[source];
                }

                let input = Owned::f32(&shape, &input_data);
                let gamma = Owned::f32(&[hidden], &gamma_data);
                let broadcast_skip = Owned::f32(&skip_shape, &skip_data);
                let expanded_skip = Owned::f32(&shape, &expanded);

                let mut broadcast_output = Owned::zeros_f32(&shape);
                SkipSimplifiedLayerNormKernel { epsilon: 1e-5 }
                    .execute(
                        &[input.view(), broadcast_skip.view(), gamma.view()],
                        &mut [broadcast_output.view_mut()],
                    )
                    .unwrap();

                let mut expanded_output = Owned::zeros_f32(&shape);
                SkipSimplifiedLayerNormKernel { epsilon: 1e-5 }
                    .execute(
                        &[input.view(), expanded_skip.view(), gamma.view()],
                        &mut [expanded_output.view_mut()],
                    )
                    .unwrap();

                assert_eq!(
                    bits(&broadcast_output.to_f32()),
                    bits(&expanded_output.to_f32()),
                    "hidden {hidden}, skip shape {skip_shape:?} does not match its expansion"
                );
            }
        }
    }

    /// f16/bf16 outputs take the narrowing path, which previously reassembled
    /// every element through an index unravel. Pin it against the f32 kernel.
    #[test]
    fn skip_simplified_layer_norm_narrowed_output_matches_narrowed_f32() {
        let shape = [2usize, 129];
        let len: usize = shape.iter().product();
        let hidden = shape[1];
        let input_data = values(len, 211);
        let skip_data = values(len, 223);
        let gamma_data = values(hidden, 227);

        let mut reference_output = Owned::zeros_f32(&shape);
        SkipSimplifiedLayerNormKernel { epsilon: 1e-5 }
            .execute(
                &[
                    Owned::f32(&shape, &input_data).view(),
                    Owned::f32(&shape, &skip_data).view(),
                    Owned::f32(&[hidden], &gamma_data).view(),
                ],
                &mut [reference_output.view_mut()],
            )
            .unwrap();
        let reference_output = reference_output.to_f32();

        let mut narrowed = Owned::zeros(DataType::Float16, &shape);
        SkipSimplifiedLayerNormKernel { epsilon: 1e-5 }
            .execute(
                &[
                    Owned::f32(&shape, &input_data).view(),
                    Owned::f32(&shape, &skip_data).view(),
                    Owned::f32(&[hidden], &gamma_data).view(),
                ],
                &mut [narrowed.view_mut()],
            )
            .unwrap();
        let expected: Vec<f32> = reference_output
            .iter()
            .map(|&value| half::f16::from_f32(value).to_f32())
            .collect();
        assert_eq!(narrowed.to_f16_as_f32(), expected);
    }

    /// NaN / +-Inf / denormal inputs must propagate identically whether or not
    /// the residual-sum output is bound.
    #[test]
    fn skip_simplified_layer_norm_propagates_non_finite_and_denormals() {
        let hidden = 8;
        let shape = [2usize, hidden];
        let input_data = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MIN_POSITIVE / 4.0,
            -f32::MIN_POSITIVE / 4.0,
            0.0,
            -0.0,
            1.0,
            1e30,
            -1e30,
            1e-30,
            -1e-30,
            f32::MAX,
            f32::MIN,
            2.0,
            -2.0,
        ];
        let skip_data = [0.5f32; 16];
        let gamma_data = [1.0f32; 8];

        let mut short = Owned::zeros_f32(&shape);
        SkipSimplifiedLayerNormKernel { epsilon: 1e-5 }
            .execute(
                &[
                    Owned::f32(&shape, &input_data).view(),
                    Owned::f32(&shape, &skip_data).view(),
                    Owned::f32(&[hidden], &gamma_data).view(),
                ],
                &mut [short.view_mut()],
            )
            .unwrap();

        let mut long = Owned::zeros_f32(&shape);
        let mut mean = Owned::zeros_f32(&[2, 1]);
        let mut inv_std = Owned::zeros_f32(&[2, 1]);
        let mut sum = Owned::zeros_f32(&shape);
        SkipSimplifiedLayerNormKernel { epsilon: 1e-5 }
            .execute(
                &[
                    Owned::f32(&shape, &input_data).view(),
                    Owned::f32(&shape, &skip_data).view(),
                    Owned::f32(&[hidden], &gamma_data).view(),
                ],
                &mut [
                    long.view_mut(),
                    mean.view_mut(),
                    inv_std.view_mut(),
                    sum.view_mut(),
                ],
            )
            .unwrap();

        assert_eq!(bits(&short.to_f32()), bits(&long.to_f32()));
        // Row 0 contains NaN/Inf, so every normalized element there is non-finite.
        assert!(short.to_f32()[..hidden].iter().all(|v| !v.is_finite()));
        for (index, (&got, &x)) in sum.to_f32().iter().zip(&input_data).enumerate() {
            let want = x + 0.5;
            assert_eq!(got.to_bits(), want.to_bits(), "residual sum at {index}");
        }
    }
}
