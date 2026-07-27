//! Pure-Rust ONNX `Conv` reference kernel used when the optional MLAS backend is disabled.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use super::check_arity;
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};
use crate::strided::numel;

const OP: &str = "Conv";

#[derive(Clone, Copy)]
enum AutoPad {
    NotSet,
    SameUpper,
    SameLower,
    Valid,
}

pub struct ConvFactory;

pub struct ConvKernel {
    x_shape: Vec<usize>,
    w_shape: Vec<usize>,
    output_shape: Vec<usize>,
    group: usize,
    strides: Vec<usize>,
    dilations: Vec<usize>,
    pads: Vec<usize>,
    relu: bool,
}

fn positive_attribute(node: &Node, name: &str, rank: usize, default: usize) -> Result<Vec<usize>> {
    let values = node
        .attr(name)
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![default as i64; rank]);
    if values.len() != rank || values.iter().any(|&value| value <= 0) {
        return Err(EpError::KernelFailed(format!(
            "{OP}: {name} must contain {rank} positive values, got {values:?}"
        )));
    }
    Ok(values.into_iter().map(|value| value as usize).collect())
}

fn explicit_pads(node: &Node, rank: usize) -> Result<Vec<usize>> {
    let values = node
        .attr("pads")
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![0; rank * 2]);
    if values.len() != rank * 2 || values.iter().any(|&value| value < 0) {
        return Err(EpError::KernelFailed(format!(
            "{OP}: pads must contain {} non-negative values, got {values:?}",
            rank * 2
        )));
    }
    Ok(values.into_iter().map(|value| value as usize).collect())
}

fn auto_pad(node: &Node) -> Result<AutoPad> {
    match node.attr("auto_pad").and_then(Attribute::as_str) {
        None | Some("NOTSET") => Ok(AutoPad::NotSet),
        Some("SAME_UPPER") => Ok(AutoPad::SameUpper),
        Some("SAME_LOWER") => Ok(AutoPad::SameLower),
        Some("VALID") => Ok(AutoPad::Valid),
        Some(value) => Err(EpError::KernelFailed(format!(
            "{OP}: unsupported auto_pad {value:?}"
        ))),
    }
}

fn output_geometry(
    input: &[usize],
    kernel: &[usize],
    dilations: &[usize],
    strides: &[usize],
    mut pads: Vec<usize>,
    auto_pad: AutoPad,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let rank = input.len();
    let mut output = vec![0; rank];
    for axis in 0..rank {
        let effective = dilations[axis]
            .checked_mul(kernel[axis].saturating_sub(1))
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| EpError::KernelFailed(format!("{OP}: kernel size overflow")))?;
        match auto_pad {
            AutoPad::SameUpper | AutoPad::SameLower => {
                output[axis] = input[axis].div_ceil(strides[axis]);
                let total = output[axis]
                    .saturating_sub(1)
                    .checked_mul(strides[axis])
                    .and_then(|value| value.checked_add(effective))
                    .map(|value| value.saturating_sub(input[axis]))
                    .ok_or_else(|| EpError::KernelFailed(format!("{OP}: padding size overflow")))?;
                let begin = if matches!(auto_pad, AutoPad::SameUpper) {
                    total / 2
                } else {
                    total - total / 2
                };
                pads[axis] = begin;
                pads[axis + rank] = total - begin;
            }
            AutoPad::Valid => {
                pads[axis] = 0;
                pads[axis + rank] = 0;
                output[axis] = input[axis]
                    .checked_sub(effective)
                    .map_or(0, |value| value / strides[axis] + 1);
            }
            AutoPad::NotSet => {
                let padded = input[axis]
                    .checked_add(pads[axis])
                    .and_then(|value| value.checked_add(pads[axis + rank]))
                    .ok_or_else(|| EpError::KernelFailed(format!("{OP}: padded size overflow")))?;
                output[axis] = padded
                    .checked_sub(effective)
                    .map_or(0, |value| value / strides[axis] + 1);
            }
        }
    }
    Ok((output, pads))
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

impl KernelFactory for ConvFactory {
    fn create(&self, node: &Node, shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let x_shape = shapes
            .first()
            .ok_or_else(|| EpError::KernelFailed(format!("{OP}: missing X shape")))?;
        let w_shape = shapes
            .get(1)
            .ok_or_else(|| EpError::KernelFailed(format!("{OP}: missing W shape")))?;
        if x_shape.len() != w_shape.len() || !matches!(x_shape.len(), 3 | 4) {
            return Err(EpError::KernelFailed(format!(
                "{OP}: requires matching rank-3 NCL or rank-4 NCHW tensors, got X={x_shape:?}, W={w_shape:?}"
            )));
        }
        let spatial_rank = x_shape.len() - 2;
        let group = node.attr("group").and_then(Attribute::as_int).unwrap_or(1);
        if group <= 0 {
            return Err(EpError::KernelFailed(format!(
                "{OP}: group must be positive, got {group}"
            )));
        }
        let group = group as usize;
        let input_channels = x_shape[1];
        let output_channels = w_shape[0];
        if !input_channels.is_multiple_of(group)
            || !output_channels.is_multiple_of(group)
            || w_shape[1] != input_channels / group
        {
            return Err(EpError::KernelFailed(format!(
                "{OP}: incompatible channels/group: X channels={input_channels}, W={w_shape:?}, group={group}"
            )));
        }
        let kernel = w_shape[2..].to_vec();
        if kernel.contains(&0) {
            return Err(EpError::KernelFailed(format!(
                "{OP}: kernel dimensions must be positive, got {kernel:?}"
            )));
        }
        if let Some(declared) = node.attr("kernel_shape").and_then(Attribute::as_ints)
            && (declared.len() != spatial_rank
                || declared
                    .iter()
                    .zip(&kernel)
                    .any(|(&value, &actual)| value <= 0 || value as usize != actual))
        {
            return Err(EpError::KernelFailed(format!(
                "{OP}: kernel_shape must match W spatial shape {kernel:?}, got {declared:?}"
            )));
        }
        let strides = positive_attribute(node, "strides", spatial_rank, 1)?;
        let dilations = positive_attribute(node, "dilations", spatial_rank, 1)?;
        let (output_spatial, pads) = output_geometry(
            &x_shape[2..],
            &kernel,
            &dilations,
            &strides,
            explicit_pads(node, spatial_rank)?,
            auto_pad(node)?,
        )?;
        let mut output_shape = vec![x_shape[0], output_channels];
        output_shape.extend(output_spatial);
        let relu = matches!(
            node.attr("activation").and_then(Attribute::as_str),
            Some("Relu")
        );
        Ok(Box::new(ConvKernel {
            x_shape: x_shape.clone(),
            w_shape: w_shape.clone(),
            output_shape,
            group,
            strides,
            dilations,
            pads,
            relu,
        }))
    }
}

impl Kernel for ConvKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(OP, inputs, outputs, 2, 3, 1)?;
        let dtype = outputs[0].dtype;
        if !matches!(
            dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) || inputs[0].dtype != dtype
            || inputs[1].dtype != dtype
            || inputs.get(2).is_some_and(|bias| bias.dtype != dtype)
        {
            return Err(EpError::KernelFailed(format!(
                "{OP}: X, W, optional B, and Y must share f32, f16, or bf16 dtype"
            )));
        }
        if inputs[0].shape != self.x_shape
            || inputs[1].shape != self.w_shape
            || outputs[0].shape != self.output_shape
        {
            return Err(EpError::KernelFailed(format!(
                "{OP}: runtime shapes X={:?}, W={:?}, Y={:?}; expected X={:?}, W={:?}, Y={:?}",
                inputs[0].shape,
                inputs[1].shape,
                outputs[0].shape,
                self.x_shape,
                self.w_shape,
                self.output_shape
            )));
        }
        let output_channels = self.w_shape[0];
        if let Some(bias) = inputs.get(2)
            && bias.shape != [output_channels]
        {
            return Err(EpError::KernelFailed(format!(
                "{OP}: bias must have shape [{output_channels}], got {:?}",
                bias.shape
            )));
        }

        let x = to_dense_f32_widen(OP, &inputs[0])?;
        let weights = to_dense_f32_widen(OP, &inputs[1])?;
        let bias = inputs
            .get(2)
            .map(|value| to_dense_f32_widen(OP, value))
            .transpose()?;
        let spatial_rank = self.x_shape.len() - 2;
        let input_spatial = &self.x_shape[2..];
        let kernel_shape = &self.w_shape[2..];
        let output_spatial = &self.output_shape[2..];
        let input_spatial_size = numel(input_spatial);
        let kernel_size = numel(kernel_shape);
        let output_spatial_size = numel(output_spatial);
        let input_spatial_strides = contiguous_strides(input_spatial);
        let kernel_strides = contiguous_strides(kernel_shape);
        let output_strides = contiguous_strides(output_spatial);
        let input_channels = self.x_shape[1];
        let channels_per_group = input_channels / self.group;
        let outputs_per_group = output_channels / self.group;
        let mut output = vec![0.0f32; numel(&self.output_shape)];

        for batch in 0..self.x_shape[0] {
            for output_channel in 0..output_channels {
                let group = output_channel / outputs_per_group;
                for output_linear in 0..output_spatial_size {
                    let mut output_remainder = output_linear;
                    let mut output_coordinates = vec![0; spatial_rank];
                    for axis in 0..spatial_rank {
                        output_coordinates[axis] = output_remainder / output_strides[axis];
                        output_remainder %= output_strides[axis];
                    }
                    let mut sum = bias.as_ref().map_or(0.0, |values| values[output_channel]);
                    for input_channel in 0..channels_per_group {
                        let absolute_channel = group * channels_per_group + input_channel;
                        for kernel_linear in 0..kernel_size {
                            let mut kernel_remainder = kernel_linear;
                            let mut input_offset = 0usize;
                            let mut in_bounds = true;
                            for axis in 0..spatial_rank {
                                let kernel_coordinate = kernel_remainder / kernel_strides[axis];
                                kernel_remainder %= kernel_strides[axis];
                                let coordinate = output_coordinates[axis]
                                    .saturating_mul(self.strides[axis])
                                    .saturating_add(
                                        kernel_coordinate.saturating_mul(self.dilations[axis]),
                                    );
                                let Some(coordinate) = coordinate.checked_sub(self.pads[axis])
                                else {
                                    in_bounds = false;
                                    break;
                                };
                                if coordinate >= input_spatial[axis] {
                                    in_bounds = false;
                                    break;
                                }
                                input_offset += coordinate * input_spatial_strides[axis];
                            }
                            if in_bounds {
                                let x_index = (batch * input_channels + absolute_channel)
                                    * input_spatial_size
                                    + input_offset;
                                let w_index = (output_channel * channels_per_group + input_channel)
                                    * kernel_size
                                    + kernel_linear;
                                sum += x[x_index] * weights[w_index];
                            }
                        }
                    }
                    if self.relu {
                        sum = sum.max(0.0);
                    }
                    output[(batch * output_channels + output_channel) * output_spatial_size
                        + output_linear] = sum;
                }
            }
        }
        write_dense_f32_narrow(OP, &mut outputs[0], &output)?;
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            (self.x_shape[0] as u64)
                .saturating_mul(output_channels as u64)
                .saturating_mul(output_spatial_size as u64)
                .saturating_mul(channels_per_group as u64)
                .saturating_mul(kernel_size as u64)
                .saturating_mul(2)
        });
        Ok(())
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::NodeId;

    fn run(
        x_shape: &[usize],
        x: &[f32],
        w_shape: &[usize],
        w: &[f32],
        bias: Option<&[f32]>,
        output_shape: &[usize],
        attributes: &[(&str, Attribute)],
    ) -> Vec<f32> {
        let mut node = Node::new(NodeId(0), OP, vec![], vec![]);
        for (name, value) in attributes {
            node.attributes.insert((*name).into(), value.clone());
        }
        let kernel = ConvFactory
            .create(&node, &[x_shape.to_vec(), w_shape.to_vec()])
            .unwrap();
        let x = Owned::f32(x_shape, x);
        let w = Owned::f32(w_shape, w);
        let bias = bias.map(|values| Owned::f32(&[values.len()], values));
        let mut output = Owned::zeros_f32(output_shape);
        let mut inputs = vec![x.view(), w.view()];
        if let Some(bias) = &bias {
            inputs.push(bias.view());
        }
        kernel.execute(&inputs, &mut [output.view_mut()]).unwrap();
        output.to_f32()
    }

    #[test]
    fn conv_2d_bias_stride_and_explicit_padding() {
        assert_eq!(
            run(
                &[1, 1, 3, 3],
                &[1., 2., 3., 4., 5., 6., 7., 8., 9.],
                &[1, 1, 2, 2],
                &[1., 0., 0., 1.],
                Some(&[1.]),
                &[1, 1, 2, 2],
                &[
                    ("strides", Attribute::Ints(vec![2, 2])),
                    ("pads", Attribute::Ints(vec![1, 1, 0, 0])),
                ],
            ),
            vec![2., 4., 8., 15.]
        );
    }

    #[test]
    fn conv_2d_dilation_and_non_square_kernel() {
        assert_eq!(
            run(
                &[1, 1, 3, 5],
                &(1..=15).map(|value| value as f32).collect::<Vec<_>>(),
                &[1, 1, 2, 3],
                &[1.; 6],
                None,
                &[1, 1, 1, 3],
                &[("dilations", Attribute::Ints(vec![2, 1]))],
            ),
            vec![42., 48., 54.]
        );
    }

    #[test]
    fn conv_2d_groups_and_depthwise_multiplier() {
        assert_eq!(
            run(
                &[1, 2, 2, 2],
                &[1., 2., 3., 4., 10., 20., 30., 40.],
                &[4, 1, 1, 1],
                &[1., 2., 3., 4.],
                Some(&[0., 1., 2., 3.]),
                &[1, 4, 2, 2],
                &[("group", Attribute::Int(2))],
            ),
            vec![
                1., 2., 3., 4., 3., 5., 7., 9., 32., 62., 92., 122., 43., 83., 123., 163.
            ]
        );
    }

    #[test]
    fn conv_1d_matches_onnxruntime_reference() {
        let x = (1..=16).map(|value| value as f32).collect::<Vec<_>>();
        let w = (1..=18).map(|value| value as f32 * 0.1).collect::<Vec<_>>();
        let actual = run(
            &[1, 2, 8],
            &x,
            &[3, 2, 3],
            &w,
            Some(&[0.5, -0.5, 1.0]),
            &[1, 3, 4],
            &[
                ("strides", Attribute::Ints(vec![2])),
                ("pads", Attribute::Ints(vec![1, 1])),
            ],
        );
        let expected = [
            11.8, 19.2, 23.4, 27.6, 24.0, 43.4, 54.8, 66.2, 38.7, 70.1, 88.7, 107.3,
        ];
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-5 * expected.abs().max(1.0));
        }
    }

    #[test]
    fn conv_same_upper_and_same_lower_split_odd_padding() {
        let x = [1., 2., 3., 4.];
        let upper = run(
            &[1, 1, 4],
            &x,
            &[1, 1, 3],
            &[1.; 3],
            None,
            &[1, 1, 2],
            &[
                ("strides", Attribute::Ints(vec![2])),
                ("auto_pad", Attribute::String(b"SAME_UPPER".to_vec())),
            ],
        );
        let lower = run(
            &[1, 1, 4],
            &x,
            &[1, 1, 3],
            &[1.; 3],
            None,
            &[1, 1, 2],
            &[
                ("strides", Attribute::Ints(vec![2])),
                ("auto_pad", Attribute::String(b"SAME_LOWER".to_vec())),
            ],
        );
        assert_eq!(upper, vec![6., 7.]);
        assert_eq!(lower, vec![3., 9.]);
    }

    #[test]
    fn conv_empty_output_when_kernel_exceeds_unpadded_input() {
        assert!(
            run(
                &[1, 1, 2],
                &[1., 2.],
                &[1, 1, 3],
                &[1.; 3],
                None,
                &[1, 1, 0],
                &[],
            )
            .is_empty()
        );
    }

    #[test]
    fn conv_bfloat16_widens_and_narrows() {
        let node = Node::new(NodeId(0), OP, vec![], vec![]);
        let kernel = ConvFactory
            .create(&node, &[vec![1, 1, 2, 2], vec![1, 1, 1, 1]])
            .unwrap();
        let x = Owned::bf16(&[1, 1, 2, 2], &[1., 2., 3., 4.]);
        let w = Owned::bf16(&[1, 1, 1, 1], &[2.]);
        let bias = Owned::bf16(&[1], &[1.]);
        let mut output = Owned::zeros(DataType::BFloat16, &[1, 1, 2, 2]);
        kernel
            .execute(&[x.view(), w.view(), bias.view()], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(output.to_bf16_as_f32(), vec![3., 5., 7., 9.]);
    }
}
