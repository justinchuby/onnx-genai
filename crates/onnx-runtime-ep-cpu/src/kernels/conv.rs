//! MLAS-backed 2-D NCHW `Conv` for Float32 tensors.

use std::sync::Mutex;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use super::check_arity;
use crate::dtype::to_dense_f32_widen;
use crate::strided::numel;

#[derive(Clone, Copy)]
enum AutoPad {
    NotSet,
    SameUpper,
    SameLower,
    Valid,
}

pub struct ConvFactory;

pub struct ConvKernel {
    plan: mlas_sys::ConvPlan,
    expected_input_shape: Vec<usize>,
    expected_weight_shape: Vec<usize>,
    expected_output_shape: Vec<usize>,
    output_channels: usize,
    scratch: Mutex<Vec<f32>>,
}

fn auto_pad(node: &Node) -> Result<AutoPad> {
    match node.attr("auto_pad").and_then(Attribute::as_str) {
        None | Some("NOTSET") => Ok(AutoPad::NotSet),
        Some("SAME_UPPER") => Ok(AutoPad::SameUpper),
        Some("SAME_LOWER") => Ok(AutoPad::SameLower),
        Some("VALID") => Ok(AutoPad::Valid),
        Some(value) => Err(EpError::KernelFailed(format!(
            "Conv: unsupported auto_pad {value:?}; expected NOTSET, SAME_UPPER, SAME_LOWER, or VALID"
        ))),
    }
}

fn positive_values(node: &Node, name: &str, default: usize) -> Result<[usize; 2]> {
    let values = node
        .attr(name)
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![default as i64; 2]);
    if values.len() != 2 || values.iter().any(|&value| value <= 0) {
        return Err(EpError::KernelFailed(format!(
            "Conv: {name} must contain two positive values, got {values:?}"
        )));
    }
    Ok([values[0] as usize, values[1] as usize])
}

fn explicit_pads(node: &Node) -> Result<[usize; 4]> {
    let values = node
        .attr("pads")
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![0; 4]);
    if values.len() != 4 || values.iter().any(|&value| value < 0) {
        return Err(EpError::KernelFailed(format!(
            "Conv: pads must contain four non-negative values, got {values:?}"
        )));
    }
    Ok([
        values[0] as usize,
        values[1] as usize,
        values[2] as usize,
        values[3] as usize,
    ])
}

fn output_geometry(
    input: [usize; 2],
    kernel: [usize; 2],
    dilations: [usize; 2],
    strides: [usize; 2],
    mut pads: [usize; 4],
    auto_pad: AutoPad,
) -> Result<([usize; 2], [usize; 4])> {
    let mut output = [0; 2];
    for axis in 0..2 {
        let effective = dilations[axis]
            .checked_mul(kernel[axis] - 1)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| EpError::KernelFailed("Conv: effective kernel size overflow".into()))?;
        match auto_pad {
            AutoPad::SameUpper | AutoPad::SameLower => {
                output[axis] = input[axis].div_ceil(strides[axis]);
                let total = output[axis]
                    .saturating_sub(1)
                    .checked_mul(strides[axis])
                    .and_then(|value| value.checked_add(effective))
                    .map(|value| value.saturating_sub(input[axis]))
                    .ok_or_else(|| EpError::KernelFailed("Conv: padding size overflow".into()))?;
                let begin = if matches!(auto_pad, AutoPad::SameUpper) {
                    total / 2
                } else {
                    total - total / 2
                };
                pads[axis] = begin;
                pads[axis + 2] = total - begin;
            }
            AutoPad::Valid => {
                pads[axis] = 0;
                pads[axis + 2] = 0;
                output[axis] = if input[axis] < effective {
                    0
                } else {
                    (input[axis] - effective) / strides[axis] + 1
                };
            }
            AutoPad::NotSet => {
                let padded = input[axis]
                    .checked_add(pads[axis])
                    .and_then(|value| value.checked_add(pads[axis + 2]))
                    .ok_or_else(|| {
                        EpError::KernelFailed("Conv: padded input size overflow".into())
                    })?;
                output[axis] = if padded < effective {
                    0
                } else {
                    (padded - effective) / strides[axis] + 1
                };
            }
        }
    }
    Ok((output, pads))
}

fn to_i64(values: impl IntoIterator<Item = usize>, what: &str) -> Result<Vec<i64>> {
    values
        .into_iter()
        .map(|value| {
            i64::try_from(value)
                .map_err(|_| EpError::KernelFailed(format!("Conv: {what} exceeds i64")))
        })
        .collect()
}

impl KernelFactory for ConvFactory {
    fn create(&self, node: &Node, shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let x_shape = shapes
            .first()
            .ok_or_else(|| EpError::KernelFailed("Conv: missing X shape".into()))?;
        let w_shape = shapes
            .get(1)
            .ok_or_else(|| EpError::KernelFailed("Conv: missing W shape".into()))?;
        if x_shape.len() != 4 || w_shape.len() != 4 {
            return Err(EpError::KernelFailed(format!(
                "Conv: MLAS kernel currently supports 2-D NCHW tensors; got X={x_shape:?}, W={w_shape:?}"
            )));
        }

        let group = node.attr("group").and_then(Attribute::as_int).unwrap_or(1);
        if group <= 0 {
            return Err(EpError::KernelFailed(format!(
                "Conv: group must be positive, got {group}"
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
                "Conv: incompatible channels/group: X channels={input_channels}, W={w_shape:?}, group={group}"
            )));
        }

        let inferred_kernel = [w_shape[2], w_shape[3]];
        let kernel = match node.attr("kernel_shape").and_then(Attribute::as_ints) {
            None => inferred_kernel,
            Some(values)
                if values.len() == 2
                    && values.iter().all(|&value| value > 0)
                    && values[0] as usize == inferred_kernel[0]
                    && values[1] as usize == inferred_kernel[1] =>
            {
                inferred_kernel
            }
            Some(values) => {
                return Err(EpError::KernelFailed(format!(
                    "Conv: kernel_shape must match W spatial shape {inferred_kernel:?}, got {values:?}"
                )));
            }
        };
        let dilations = positive_values(node, "dilations", 1)?;
        let strides = positive_values(node, "strides", 1)?;
        let (output_spatial, pads) = output_geometry(
            [x_shape[2], x_shape[3]],
            kernel,
            dilations,
            strides,
            explicit_pads(node)?,
            auto_pad(node)?,
        )?;
        let expected_output_shape = vec![
            x_shape[0],
            output_channels,
            output_spatial[0],
            output_spatial[1],
        ];

        let plan = mlas_sys::ConvPlan::new(
            x_shape[0],
            group,
            input_channels / group,
            &to_i64([x_shape[2], x_shape[3]], "input shape")?,
            &to_i64(kernel, "kernel shape")?,
            &to_i64(dilations, "dilations")?,
            &to_i64(pads, "pads")?,
            &to_i64(strides, "strides")?,
            &to_i64(output_spatial, "output shape")?,
            output_channels / group,
        )
        .ok_or_else(|| EpError::KernelFailed("Conv: MLAS failed to prepare convolution".into()))?;
        let scratch = vec![0.0; plan.working_buffer_elements()];

        Ok(Box::new(ConvKernel {
            plan,
            expected_input_shape: x_shape.clone(),
            expected_weight_shape: w_shape.clone(),
            expected_output_shape,
            output_channels,
            scratch: Mutex::new(scratch),
        }))
    }
}

fn byte_ranges_overlap(input: &TensorView<'_>, output: &mut TensorMut<'_>) -> bool {
    let input_start = input.data_ptr::<u8>() as usize;
    let input_end = input_start.saturating_add(input.byte_size());
    let output_start = output.data_ptr_mut::<u8>() as usize;
    let output_end = output_start.saturating_add(output.byte_size());
    output_start < input_end && input_start < output_end
}

impl Kernel for ConvKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Conv", inputs, outputs, 2, 3, 1)?;
        if inputs[0].dtype != DataType::Float32
            || inputs[1].dtype != DataType::Float32
            || inputs
                .get(2)
                .is_some_and(|bias| bias.dtype != DataType::Float32)
            || outputs[0].dtype != DataType::Float32
        {
            return Err(EpError::KernelFailed(
                "Conv: MLAS kernel requires Float32 X, W, optional B, and Y".into(),
            ));
        }
        if inputs[0].shape != self.expected_input_shape
            || inputs[1].shape != self.expected_weight_shape
            || outputs[0].shape != self.expected_output_shape
        {
            return Err(EpError::KernelFailed(format!(
                "Conv: runtime shapes X={:?}, W={:?}, Y={:?}; expected X={:?}, W={:?}, Y={:?}",
                inputs[0].shape,
                inputs[1].shape,
                outputs[0].shape,
                self.expected_input_shape,
                self.expected_weight_shape,
                self.expected_output_shape
            )));
        }
        if let Some(bias) = inputs.get(2)
            && bias.shape != [self.output_channels]
        {
            return Err(EpError::KernelFailed(format!(
                "Conv: bias must have shape [{}], got {:?}",
                self.output_channels, bias.shape
            )));
        }
        if !outputs[0].is_contiguous()
            || inputs
                .iter()
                .any(|input| byte_ranges_overlap(input, &mut outputs[0]))
        {
            return Err(EpError::KernelFailed(
                "Conv: output must be contiguous and must not alias an input".into(),
            ));
        }

        let x = to_dense_f32_widen("Conv", &inputs[0])?;
        let weights = to_dense_f32_widen("Conv", &inputs[1])?;
        let bias = inputs
            .get(2)
            .map(|value| to_dense_f32_widen("Conv", value))
            .transpose()?;
        let output_elements = numel(&self.expected_output_shape);
        // SAFETY: the executor validated this contiguous Float32 output view,
        // and `output_elements` is exactly the product of its checked shape.
        let output = unsafe {
            std::slice::from_raw_parts_mut(outputs[0].data_ptr_mut::<f32>(), output_elements)
        };
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| EpError::KernelFailed("Conv: scratch lock poisoned".into()))?;
        self.plan
            .run(&x, &weights, bias.as_deref(), &mut scratch, output);

        crate::trace::record_kernel_metrics(inputs, outputs, || {
            let output_spatial =
                self.expected_output_shape[2].saturating_mul(self.expected_output_shape[3]);
            let kernel_elements = self.expected_weight_shape[1]
                .saturating_mul(self.expected_weight_shape[2])
                .saturating_mul(self.expected_weight_shape[3]);
            (self.expected_output_shape[0] as u64)
                .saturating_mul(self.output_channels as u64)
                .saturating_mul(output_spatial as u64)
                .saturating_mul(kernel_elements as u64)
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
    use onnx_runtime_ir::{Attribute, NodeId};

    fn run_conv(
        x_shape: &[usize],
        x: &[f32],
        w_shape: &[usize],
        w: &[f32],
        bias: Option<&[f32]>,
        output_shape: &[usize],
        attributes: &[(&str, Attribute)],
    ) -> Vec<f32> {
        let mut node = Node::new(NodeId(0), "Conv", vec![], vec![]);
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
    fn conv_bias_stride_and_explicit_padding() {
        let output = run_conv(
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
        );
        assert_eq!(output, vec![2., 4., 8., 15.]);
    }

    #[test]
    fn conv_dilation() {
        let output = run_conv(
            &[1, 1, 3, 3],
            &[1., 2., 3., 4., 5., 6., 7., 8., 9.],
            &[1, 1, 2, 2],
            &[1., 1., 1., 1.],
            None,
            &[1, 1, 1, 1],
            &[("dilations", Attribute::Ints(vec![2, 2]))],
        );
        assert_eq!(output, vec![20.]);
    }

    #[test]
    fn conv_grouped_and_depthwise() {
        let grouped = run_conv(
            &[1, 2, 2, 2],
            &[1., 2., 3., 4., 10., 20., 30., 40.],
            &[2, 1, 1, 1],
            &[2., 3.],
            None,
            &[1, 2, 2, 2],
            &[("group", Attribute::Int(2))],
        );
        assert_eq!(grouped, vec![2., 4., 6., 8., 30., 60., 90., 120.]);

        let depthwise = run_conv(
            &[1, 2, 2, 2],
            &[1., 2., 3., 4., 10., 20., 30., 40.],
            &[4, 1, 1, 1],
            &[1., 2., 3., 4.],
            Some(&[0., 1., 2., 3.]),
            &[1, 4, 2, 2],
            &[("group", Attribute::Int(2))],
        );
        assert_eq!(
            depthwise,
            vec![
                1., 2., 3., 4., 3., 5., 7., 9., 32., 62., 92., 122., 43., 83., 123., 163.
            ]
        );
    }

    #[test]
    fn same_upper_same_lower_and_valid_geometry() {
        let input = [4, 4];
        let kernel = [3, 3];
        let dilation = [1, 1];
        let stride = [2, 2];
        let (upper_output, upper_pads) =
            output_geometry(input, kernel, dilation, stride, [0; 4], AutoPad::SameUpper).unwrap();
        let (lower_output, lower_pads) =
            output_geometry(input, kernel, dilation, stride, [0; 4], AutoPad::SameLower).unwrap();
        let (valid_output, valid_pads) =
            output_geometry(input, kernel, dilation, stride, [9; 4], AutoPad::Valid).unwrap();
        assert_eq!(upper_output, [2, 2]);
        assert_eq!(lower_output, [2, 2]);
        assert_eq!(upper_pads, [0, 0, 1, 1]);
        assert_eq!(lower_pads, [1, 1, 0, 0]);
        assert_eq!(valid_output, [1, 1]);
        assert_eq!(valid_pads, [0; 4]);
    }
}
