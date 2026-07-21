//! N-dimensional `ConvTranspose` for the CPU execution provider.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, Node};

use crate::dispatch_float;
use crate::dtype::{ComputeDomain, NumericElem, to_dense, write_dense};
use crate::strided::numel;

use super::check_arity;

#[derive(Clone, Copy)]
enum AutoPad {
    NotSet,
    SameUpper,
    SameLower,
    Valid,
}

pub struct ConvTransposeFactory;

pub struct ConvTransposeKernel {
    auto_pad: AutoPad,
    dilations: Vec<usize>,
    group: usize,
    kernel_shape: Vec<usize>,
    output_padding: Vec<usize>,
    output_shape: Option<Vec<usize>>,
    pads: Vec<i64>,
    strides: Vec<usize>,
}

fn positive_values(node: &Node, name: &str, rank: usize, default: usize) -> Result<Vec<usize>> {
    let values = node
        .attr(name)
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![default as i64; rank]);
    if values.len() != rank || values.iter().any(|&value| value <= 0) {
        return Err(EpError::KernelFailed(format!(
            "ConvTranspose: {name} must contain {rank} positive values"
        )));
    }
    Ok(values.into_iter().map(|value| value as usize).collect())
}

fn nonnegative_values(node: &Node, name: &str, rank: usize, count: usize) -> Result<Vec<usize>> {
    let values = node
        .attr(name)
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![0; count]);
    if values.len() != count || values.iter().any(|&value| value < 0) {
        return Err(EpError::KernelFailed(format!(
            "ConvTranspose: {name} must contain {count} non-negative values for spatial rank {rank}"
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
            "ConvTranspose: unsupported auto_pad {value:?}"
        ))),
    }
}

impl KernelFactory for ConvTransposeFactory {
    fn create(&self, node: &Node, shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let x_shape = shapes
            .first()
            .ok_or_else(|| EpError::KernelFailed("ConvTranspose: missing X shape".into()))?;
        let w_shape = shapes
            .get(1)
            .ok_or_else(|| EpError::KernelFailed("ConvTranspose: missing W shape".into()))?;
        if !(3..=5).contains(&x_shape.len()) || w_shape.len() != x_shape.len() {
            return Err(EpError::KernelFailed(
                "ConvTranspose: X and W must have equal rank between 3 and 5".into(),
            ));
        }
        let rank = x_shape.len() - 2;
        let inferred_kernel = w_shape[2..].to_vec();
        let kernel_shape = node
            .attr("kernel_shape")
            .and_then(Attribute::as_ints)
            .map(<[i64]>::to_vec)
            .unwrap_or_else(|| inferred_kernel.iter().map(|&value| value as i64).collect());
        if kernel_shape.len() != rank
            || kernel_shape.iter().any(|&value| value <= 0)
            || kernel_shape
                .iter()
                .zip(&inferred_kernel)
                .any(|(&attribute, &weight)| attribute as usize != weight)
        {
            return Err(EpError::KernelFailed(
                "ConvTranspose: kernel_shape must match the spatial dimensions of W".into(),
            ));
        }

        let group = node.attr("group").and_then(Attribute::as_int).unwrap_or(1);
        if group <= 0 {
            return Err(EpError::KernelFailed(
                "ConvTranspose: group must be positive".into(),
            ));
        }
        let output_shape = node
            .attr("output_shape")
            .and_then(Attribute::as_ints)
            .map(|values| {
                if values.len() != rank || values.iter().any(|&value| value < 0) {
                    return Err(EpError::KernelFailed(format!(
                        "ConvTranspose: output_shape must contain {rank} non-negative values"
                    )));
                }
                Ok(values.iter().map(|&value| value as usize).collect())
            })
            .transpose()?;
        let pads = nonnegative_values(node, "pads", rank, rank * 2)?
            .into_iter()
            .map(|value| value as i64)
            .collect();

        let dilations = positive_values(node, "dilations", rank, 1)?;
        let strides = positive_values(node, "strides", rank, 1)?;
        let output_padding = nonnegative_values(node, "output_padding", rank, rank)?;
        if output_padding
            .iter()
            .enumerate()
            .any(|(axis, &value)| value >= strides[axis] && value >= dilations[axis])
        {
            return Err(EpError::KernelFailed(
                "ConvTranspose: each output_padding value must be smaller than the corresponding stride or dilation"
                    .into(),
            ));
        }

        Ok(Box::new(ConvTransposeKernel {
            auto_pad: auto_pad(node)?,
            dilations,
            group: group as usize,
            kernel_shape: kernel_shape
                .into_iter()
                .map(|value| value as usize)
                .collect(),
            output_padding,
            output_shape,
            pads,
            strides,
        }))
    }
}

fn coordinates(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut coordinates = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        coordinates[axis] = flat % shape[axis];
        flat /= shape[axis];
    }
    coordinates
}

fn offset(coordinates: &[usize], shape: &[usize]) -> usize {
    coordinates
        .iter()
        .zip(shape)
        .fold(0, |offset, (&coordinate, &extent)| {
            offset * extent + coordinate
        })
}

impl ConvTransposeKernel {
    fn output_geometry(&self, input: &[usize]) -> Result<(Vec<usize>, Vec<i64>)> {
        let rank = input.len();
        let mut output = Vec::with_capacity(rank);
        let mut pads = self.pads.clone();
        for axis in 0..rank {
            let effective_kernel = self.dilations[axis]
                .checked_mul(self.kernel_shape[axis] - 1)
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| {
                    EpError::KernelFailed("ConvTranspose: effective kernel overflow".into())
                })?;
            let base = self.strides[axis]
                .checked_mul(input[axis].saturating_sub(1))
                .and_then(|value| value.checked_add(self.output_padding[axis]))
                .and_then(|value| value.checked_add(effective_kernel))
                .ok_or_else(|| {
                    EpError::KernelFailed("ConvTranspose: output size overflow".into())
                })?;

            let desired = self.output_shape.as_ref().map(|shape| shape[axis]);
            match (desired, self.auto_pad) {
                (Some(desired), _) => {
                    let total = base as i64 - desired as i64;
                    let lower_half = total.div_euclid(2);
                    let begin = if matches!(self.auto_pad, AutoPad::SameUpper) {
                        lower_half
                    } else {
                        total - lower_half
                    };
                    pads[axis] = begin;
                    pads[axis + rank] = total - begin;
                    output.push(desired);
                }
                (None, AutoPad::SameUpper | AutoPad::SameLower) => {
                    let desired = input[axis].checked_mul(self.strides[axis]).ok_or_else(|| {
                        EpError::KernelFailed("ConvTranspose: output size overflow".into())
                    })?;
                    let total = base as i64 - desired as i64;
                    let lower_half = total.div_euclid(2);
                    let begin = if matches!(self.auto_pad, AutoPad::SameUpper) {
                        lower_half
                    } else {
                        total - lower_half
                    };
                    pads[axis] = begin;
                    pads[axis + rank] = total - begin;
                    output.push(desired);
                }
                (None, AutoPad::Valid) => {
                    pads[axis] = 0;
                    pads[axis + rank] = 0;
                    output.push(base);
                }
                (None, AutoPad::NotSet) => {
                    let size = base as i64 - pads[axis] - pads[axis + rank];
                    if size < 0 {
                        return Err(EpError::KernelFailed(
                            "ConvTranspose: pads produce a negative output dimension".into(),
                        ));
                    }
                    output.push(size as usize);
                }
            }
        }
        Ok((output, pads))
    }

    fn execute_typed<T: NumericElem>(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
    ) -> Result<()> {
        let x_shape = inputs[0].shape;
        let w_shape = inputs[1].shape;
        let rank = x_shape.len() - 2;
        let input_spatial = &x_shape[2..];
        let (output_spatial, pads) = self.output_geometry(input_spatial)?;
        let output_channels = w_shape[1]
            .checked_mul(self.group)
            .ok_or_else(|| EpError::KernelFailed("ConvTranspose: channel count overflow".into()))?;
        let expected = [x_shape[0], output_channels]
            .into_iter()
            .chain(output_spatial.iter().copied())
            .collect::<Vec<_>>();
        if outputs[0].shape != expected {
            return Err(EpError::KernelFailed(format!(
                "ConvTranspose: output shape {:?}, expected {expected:?}",
                outputs[0].shape
            )));
        }
        if inputs[0].dtype != inputs[1].dtype || outputs[0].dtype != inputs[0].dtype {
            return Err(EpError::KernelFailed(
                "ConvTranspose: X, W, and Y must have the same dtype".into(),
            ));
        }
        if inputs.len() == 3
            && (inputs[2].dtype != inputs[0].dtype || inputs[2].shape != [output_channels])
        {
            return Err(EpError::KernelFailed(format!(
                "ConvTranspose: B must have shape [{output_channels}] and match X dtype"
            )));
        }

        let x = to_dense::<T>(&inputs[0])?;
        let w = to_dense::<T>(&inputs[1])?;
        let bias = if inputs.len() == 3 {
            Some(to_dense::<T>(&inputs[2])?)
        } else {
            None
        };
        let output_spatial_size = numel(&output_spatial);
        let input_spatial_size = numel(input_spatial);
        let kernel_size = numel(&self.kernel_shape);
        let input_channels_per_group = x_shape[1] / self.group;
        let output_channels_per_group = w_shape[1];
        let mut output = vec![T::Acc::default(); numel(&expected)];

        if let Some(bias) = bias {
            for n in 0..x_shape[0] {
                for (output_channel, bias_value) in bias.iter().enumerate() {
                    let value = bias_value.to_acc();
                    let start = (n * output_channels + output_channel) * output_spatial_size;
                    output[start..start + output_spatial_size].fill(value);
                }
            }
        }

        for n in 0..x_shape[0] {
            for input_channel in 0..x_shape[1] {
                let group = input_channel / input_channels_per_group;
                for input_flat in 0..input_spatial_size {
                    let input_coordinates = coordinates(input_flat, input_spatial);
                    let x_value = x
                        [(n * x_shape[1] + input_channel) * input_spatial_size + input_flat]
                        .to_acc();
                    for kernel_flat in 0..kernel_size {
                        let kernel_coordinates = coordinates(kernel_flat, &self.kernel_shape);
                        let mut output_coordinates = Vec::with_capacity(rank);
                        let mut in_bounds = true;
                        for axis in 0..rank {
                            let coordinate = input_coordinates[axis] as i64
                                * self.strides[axis] as i64
                                + kernel_coordinates[axis] as i64 * self.dilations[axis] as i64
                                - pads[axis];
                            if coordinate < 0 || coordinate >= output_spatial[axis] as i64 {
                                in_bounds = false;
                                break;
                            }
                            output_coordinates.push(coordinate as usize);
                        }
                        if !in_bounds {
                            continue;
                        }
                        let output_spatial_offset = offset(&output_coordinates, &output_spatial);
                        for output_in_group in 0..output_channels_per_group {
                            let output_channel =
                                group * output_channels_per_group + output_in_group;
                            let weight = w[(input_channel * output_channels_per_group
                                + output_in_group)
                                * kernel_size
                                + kernel_flat]
                                .to_acc();
                            let output_offset = (n * output_channels + output_channel)
                                * output_spatial_size
                                + output_spatial_offset;
                            output[output_offset] =
                                output[output_offset].c_add(x_value.c_mul(weight));
                        }
                    }
                }
            }
        }

        let output = output.into_iter().map(T::from_acc).collect::<Vec<_>>();
        write_dense::<T>(&mut outputs[0], &output)
    }
}

impl Kernel for ConvTransposeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("ConvTranspose", inputs, outputs, 2, 3, 1)?;
        let x_shape = inputs[0].shape;
        let w_shape = inputs[1].shape;
        if !(3..=5).contains(&x_shape.len()) || w_shape.len() != x_shape.len() {
            return Err(EpError::KernelFailed(
                "ConvTranspose: X and W must have equal rank between 3 and 5".into(),
            ));
        }
        if w_shape[0] != x_shape[1] || !x_shape[1].is_multiple_of(self.group) || w_shape[1] == 0 {
            return Err(EpError::KernelFailed(
                "ConvTranspose: W must be [C_in, C_out/group, kernel...] and C_in must be divisible by group"
                    .into(),
            ));
        }
        dispatch_float!(inputs[0].dtype, "ConvTranspose", T => {
            self.execute_typed::<T>(inputs, outputs)
        })
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    fn kernel(rank: usize, kernel_shape: Vec<usize>, group: usize) -> ConvTransposeKernel {
        ConvTransposeKernel {
            auto_pad: AutoPad::NotSet,
            dilations: vec![1; rank],
            group,
            kernel_shape,
            output_padding: vec![0; rank],
            output_shape: None,
            pads: vec![0; rank * 2],
            strides: vec![1; rank],
        }
    }

    #[test]
    fn conv_transpose_1d_spreads_each_input_over_the_kernel() {
        let x = Owned::f32(&[1, 1, 3], &[1., 2., 3.]);
        let w = Owned::f32(&[1, 1, 3], &[1., 1., 1.]);
        let mut y = Owned::zeros_f32(&[1, 1, 5]);
        kernel(1, vec![3], 1)
            .execute(&[x.view(), w.view()], &mut [y.view_mut()])
            .unwrap();
        assert_eq!(y.to_f32(), vec![1., 3., 6., 5., 3.]);
    }

    #[test]
    fn conv_transpose_keeps_groups_separate_and_adds_bias() {
        let x = Owned::f32(&[1, 2, 1], &[2., 3.]);
        let w = Owned::f32(&[2, 1, 1], &[4., 5.]);
        let b = Owned::f32(&[2], &[1., 2.]);
        let mut y = Owned::zeros_f32(&[1, 2, 1]);
        kernel(1, vec![1], 2)
            .execute(&[x.view(), w.view(), b.view()], &mut [y.view_mut()])
            .unwrap();
        assert_eq!(y.to_f32(), vec![9., 17.]);
    }

    #[test]
    fn explicit_output_shape_generates_asymmetric_negative_end_padding() {
        let x = Owned::f32(&[1, 1, 1], &[2.]);
        let w = Owned::f32(&[1, 1, 1], &[3.]);
        let mut y = Owned::zeros_f32(&[1, 1, 2]);
        let mut op = kernel(1, vec![1], 1);
        op.output_shape = Some(vec![2]);
        op.execute(&[x.view(), w.view()], &mut [y.view_mut()])
            .unwrap();
        assert_eq!(y.to_f32(), vec![6., 0.]);
    }

    #[test]
    fn conv_transpose_computes_f64_without_narrowing() {
        let x = Owned::f64(&[1, 1, 2], &[0.1, 0.2]);
        let w = Owned::f64(&[1, 1, 2], &[0.3, 0.4]);
        let b = Owned::f64(&[1], &[0.5]);
        let mut y = Owned::zeros(onnx_runtime_ir::DataType::Float64, &[1, 1, 3]);
        kernel(1, vec![2], 1)
            .execute(&[x.view(), w.view(), b.view()], &mut [y.view_mut()])
            .unwrap();
        assert_eq!(
            y.to_f64(),
            vec![
                0.5 + 0.1 * 0.3,
                0.5 + 0.1 * 0.4 + 0.2 * 0.3,
                0.5 + 0.2 * 0.4
            ]
        );
    }
}
