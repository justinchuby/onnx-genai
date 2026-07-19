//! N-D spatial pooling kernels.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use super::{check_arity, write_dense_bytes};
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};
use crate::strided::numel;

#[derive(Clone, Copy)]
enum AutoPad {
    NotSet,
    SameUpper,
    SameLower,
    Valid,
}

#[derive(Clone)]
struct PoolParams {
    kernel: Vec<usize>,
    strides: Vec<usize>,
    dilations: Vec<usize>,
    pads: Vec<usize>,
}

impl PoolParams {
    fn output_shape(
        &self,
        spatial: &[usize],
        auto_pad: AutoPad,
        ceil_mode: bool,
    ) -> (Vec<usize>, Vec<usize>) {
        let dims = spatial.len();
        let mut output = Vec::with_capacity(dims);
        let mut pads = self.pads.clone();
        for axis in 0..dims {
            let effective = self.dilations[axis] * (self.kernel[axis] - 1) + 1;
            let (begin, end, out) = match auto_pad {
                AutoPad::SameUpper | AutoPad::SameLower => {
                    let out = spatial[axis].div_ceil(self.strides[axis]);
                    let total = ((out.saturating_sub(1)) * self.strides[axis] + effective)
                        .saturating_sub(spatial[axis]);
                    let begin = if matches!(auto_pad, AutoPad::SameUpper) {
                        total / 2
                    } else {
                        total - total / 2
                    };
                    (begin, total - begin, out)
                }
                AutoPad::Valid => (
                    0,
                    0,
                    (((spatial[axis] as i64 - effective as i64) / self.strides[axis] as i64) + 1)
                        .max(0) as usize,
                ),
                AutoPad::NotSet => {
                    let numerator =
                        spatial[axis] as i64 + pads[axis] as i64 + pads[axis + dims] as i64
                            - effective as i64;
                    let out = if ceil_mode {
                        ((numerator + self.strides[axis] as i64 - 1) / self.strides[axis] as i64
                            + 1)
                        .max(0) as usize
                    } else {
                        (numerator / self.strides[axis] as i64 + 1).max(0) as usize
                    };
                    (pads[axis], pads[axis + dims], out)
                }
            };
            pads[axis] = begin;
            pads[axis + dims] = end;
            output.push(out);
        }
        (output, pads)
    }
}

fn auto_pad(node: &Node) -> Result<AutoPad> {
    match node.attr("auto_pad").and_then(Attribute::as_str) {
        None | Some("NOTSET") => Ok(AutoPad::NotSet),
        Some("SAME_UPPER") => Ok(AutoPad::SameUpper),
        Some("SAME_LOWER") => Ok(AutoPad::SameLower),
        Some("VALID") => Ok(AutoPad::Valid),
        Some(value) => Err(EpError::KernelFailed(format!(
            "Pool: unsupported auto_pad {value:?}"
        ))),
    }
}

fn positive_attrs(node: &Node, name: &str, dims: usize, default: usize) -> Result<Vec<usize>> {
    let values = node
        .attr(name)
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![default as i64; dims]);
    if values.len() != dims || values.iter().any(|&v| v <= 0) {
        return Err(EpError::KernelFailed(format!(
            "Pool: {name} must contain {dims} positive values"
        )));
    }
    Ok(values.into_iter().map(|v| v as usize).collect())
}

fn pool_params(node: &Node, dims: usize) -> Result<(PoolParams, AutoPad, bool)> {
    let kernel = node
        .attr("kernel_shape")
        .and_then(Attribute::as_ints)
        .ok_or_else(|| EpError::KernelFailed("Pool: kernel_shape is required".into()))?;
    if kernel.len() != dims || kernel.iter().any(|&v| v <= 0) {
        return Err(EpError::KernelFailed(format!(
            "Pool: kernel_shape must contain {dims} positive values"
        )));
    }
    let pads = node
        .attr("pads")
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![0; dims * 2]);
    if pads.len() != dims * 2 || pads.iter().any(|&v| v < 0) {
        return Err(EpError::KernelFailed(format!(
            "Pool: pads must contain {dims} * 2 non-negative values"
        )));
    }
    Ok((
        PoolParams {
            kernel: kernel.iter().map(|&v| v as usize).collect(),
            strides: positive_attrs(node, "strides", dims, 1)?,
            dilations: positive_attrs(node, "dilations", dims, 1)?,
            pads: pads.into_iter().map(|v| v as usize).collect(),
        },
        auto_pad(node)?,
        node.attr("ceil_mode")
            .and_then(Attribute::as_int)
            .unwrap_or(0)
            != 0,
    ))
}

fn coordinates(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        coords[axis] = index % shape[axis];
        index /= shape[axis];
    }
    coords
}

fn spatial_index(coords: &[usize], shape: &[usize], storage_order: bool) -> usize {
    if storage_order {
        let mut stride = 1;
        let mut index = 0;
        for (&coord, &dim) in coords.iter().zip(shape) {
            index += coord * stride;
            stride *= dim;
        }
        index
    } else {
        coords
            .iter()
            .zip(shape)
            .fold(0, |index, (&coord, &dim)| index * dim + coord)
    }
}

enum PoolKind {
    Average { include_pad: bool },
    Lp { p: i32 },
    Max { storage_order: bool },
}

pub struct PoolKernel {
    params: PoolParams,
    auto_pad: AutoPad,
    ceil_mode: bool,
    kind: PoolKind,
}

pub struct AveragePoolFactory;
pub struct LpPoolFactory;
pub struct MaxPoolFactory;

impl KernelFactory for AveragePoolFactory {
    fn create(&self, node: &Node, shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let dims = shapes
            .first()
            .ok_or_else(|| EpError::KernelFailed("AveragePool: missing input shape".into()))?
            .len()
            .checked_sub(2)
            .ok_or_else(|| {
                EpError::KernelFailed("AveragePool: input must have rank >= 3".into())
            })?;
        let (params, auto_pad, ceil_mode) = pool_params(node, dims)?;
        Ok(Box::new(PoolKernel {
            params,
            auto_pad,
            ceil_mode,
            kind: PoolKind::Average {
                include_pad: node
                    .attr("count_include_pad")
                    .and_then(Attribute::as_int)
                    .unwrap_or(0)
                    != 0,
            },
        }))
    }
}

impl KernelFactory for LpPoolFactory {
    fn create(&self, node: &Node, shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let dims = shapes
            .first()
            .ok_or_else(|| EpError::KernelFailed("LpPool: missing input shape".into()))?
            .len()
            .checked_sub(2)
            .ok_or_else(|| EpError::KernelFailed("LpPool: input must have rank >= 3".into()))?;
        let p = node.attr("p").and_then(Attribute::as_int).unwrap_or(2);
        if p <= 0 || p > i32::MAX as i64 {
            return Err(EpError::KernelFailed(
                "LpPool: p must be a positive 32-bit integer".into(),
            ));
        }
        let (params, auto_pad, ceil_mode) = pool_params(node, dims)?;
        Ok(Box::new(PoolKernel {
            params,
            auto_pad,
            ceil_mode,
            kind: PoolKind::Lp { p: p as i32 },
        }))
    }
}

impl KernelFactory for MaxPoolFactory {
    fn create(&self, node: &Node, shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let dims = shapes
            .first()
            .ok_or_else(|| EpError::KernelFailed("MaxPool: missing input shape".into()))?
            .len()
            .checked_sub(2)
            .ok_or_else(|| EpError::KernelFailed("MaxPool: input must have rank >= 3".into()))?;
        let (params, auto_pad, ceil_mode) = pool_params(node, dims)?;
        Ok(Box::new(PoolKernel {
            params,
            auto_pad,
            ceil_mode,
            kind: PoolKind::Max {
                storage_order: node
                    .attr("storage_order")
                    .and_then(Attribute::as_int)
                    .unwrap_or(0)
                    != 0,
            },
        }))
    }
}

impl Kernel for PoolKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Pool", inputs, outputs, 1, 1, 1)?;
        let x_shape = inputs[0].shape;
        if x_shape.len() < 3 {
            return Err(EpError::KernelFailed(
                "Pool: input must have rank >= 3".into(),
            ));
        }
        if outputs.len() > 2 {
            return Err(EpError::KernelFailed(
                "MaxPool: supports at most two outputs".into(),
            ));
        }
        if matches!(self.kind, PoolKind::Average { .. } | PoolKind::Lp { .. }) && outputs.len() != 1
        {
            return Err(EpError::KernelFailed(
                "AveragePool and LpPool have exactly one output".into(),
            ));
        }
        let spatial = &x_shape[2..];
        let (output_spatial, pads) =
            self.params
                .output_shape(spatial, self.auto_pad, self.ceil_mode);
        let expected = [x_shape[0], x_shape[1]]
            .into_iter()
            .chain(output_spatial.iter().copied())
            .collect::<Vec<_>>();
        if outputs[0].shape != expected {
            return Err(EpError::KernelFailed(format!(
                "Pool: output shape {:?} does not match expected {expected:?}",
                outputs[0].shape
            )));
        }
        if outputs.len() == 2
            && (outputs[1].dtype != DataType::Int64 || outputs[1].shape != expected)
        {
            return Err(EpError::KernelFailed(
                "MaxPool: Indices must be an Int64 tensor with the output shape".into(),
            ));
        }

        let x = to_dense_f32_widen("Pool", &inputs[0])?;
        let spatial_size = numel(spatial);
        let kernel_size = numel(&self.params.kernel);
        let mut values = Vec::with_capacity(numel(&expected));
        let mut indices = Vec::with_capacity(numel(&expected));
        for nc in 0..x_shape[0] * x_shape[1] {
            for out_flat in 0..numel(&output_spatial) {
                let out_coords = coordinates(out_flat, &output_spatial);
                let starts: Vec<isize> = out_coords
                    .iter()
                    .enumerate()
                    .map(|(axis, &coord)| {
                        (coord * self.params.strides[axis]) as isize - pads[axis] as isize
                    })
                    .collect();
                let mut sum = 0.0;
                let mut count = 0usize;
                let mut maximum = f32::NEG_INFINITY;
                let mut maximum_index = 0usize;
                for kernel_flat in 0..kernel_size {
                    let kernel_coords = coordinates(kernel_flat, &self.params.kernel);
                    let mut input_coords = Vec::with_capacity(spatial.len());
                    let mut in_bounds = true;
                    for axis in 0..spatial.len() {
                        let coordinate = starts[axis]
                            + (kernel_coords[axis] * self.params.dilations[axis]) as isize;
                        if coordinate < 0 || coordinate >= spatial[axis] as isize {
                            in_bounds = false;
                            break;
                        }
                        input_coords.push(coordinate as usize);
                    }
                    if !in_bounds {
                        continue;
                    }
                    let index = nc * spatial_size + spatial_index(&input_coords, spatial, false);
                    let value = x[index];
                    match self.kind {
                        PoolKind::Average { .. } => {
                            sum += value;
                            count += 1;
                        }
                        PoolKind::Lp { p } => {
                            sum += value.abs().powi(p);
                            count += 1;
                        }
                        PoolKind::Max { storage_order } => {
                            if value > maximum {
                                maximum = value;
                                maximum_index = nc * spatial_size
                                    + spatial_index(&input_coords, spatial, storage_order);
                            }
                        }
                    }
                }
                match self.kind {
                    PoolKind::Average { include_pad } => {
                        let divisor = if include_pad { kernel_size } else { count };
                        values.push(if divisor == 0 {
                            0.0
                        } else {
                            sum / divisor as f32
                        });
                    }
                    PoolKind::Lp { p } => {
                        values.push(if count == 0 {
                            0.0
                        } else {
                            sum.powf(1.0 / p as f32)
                        });
                    }
                    PoolKind::Max { .. } => {
                        values.push(maximum);
                        indices.push(maximum_index as i64);
                    }
                }
            }
        }
        write_dense_f32_narrow("Pool", &mut outputs[0], &values)?;
        if outputs.len() == 2 {
            write_dense_bytes(
                &mut outputs[1],
                &indices
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
        }
        Ok(())
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

pub struct GlobalPoolKernel {
    kind: GlobalPoolKind,
}

pub struct GlobalAveragePoolFactory;
pub struct GlobalLpPoolFactory;
pub struct GlobalMaxPoolFactory;

enum GlobalPoolKind {
    Average,
    Lp(i32),
    Max,
}

impl KernelFactory for GlobalAveragePoolFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(GlobalPoolKernel {
            kind: GlobalPoolKind::Average,
        }))
    }
}

impl KernelFactory for GlobalLpPoolFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let p = node.attr("p").and_then(Attribute::as_int).unwrap_or(2);
        if p <= 0 || p > i32::MAX as i64 {
            return Err(EpError::KernelFailed(
                "GlobalLpPool: p must be a positive 32-bit integer".into(),
            ));
        }
        Ok(Box::new(GlobalPoolKernel {
            kind: GlobalPoolKind::Lp(p as i32),
        }))
    }
}

impl KernelFactory for GlobalMaxPoolFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(GlobalPoolKernel {
            kind: GlobalPoolKind::Max,
        }))
    }
}

impl Kernel for GlobalPoolKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("GlobalPool", inputs, outputs, 1, 1, 1)?;
        let shape = inputs[0].shape;
        if shape.len() < 3 {
            return Err(EpError::KernelFailed(
                "GlobalPool: input must have rank >= 3".into(),
            ));
        }
        let expected = [shape[0], shape[1]]
            .into_iter()
            .chain(std::iter::repeat_n(1, shape.len() - 2))
            .collect::<Vec<_>>();
        if outputs[0].shape != expected {
            return Err(EpError::KernelFailed(format!(
                "GlobalPool: output shape {:?} does not match expected {expected:?}",
                outputs[0].shape
            )));
        }
        let x = to_dense_f32_widen("GlobalPool", &inputs[0])?;
        let spatial_size = numel(&shape[2..]);
        let mut values = Vec::with_capacity(shape[0] * shape[1]);
        for nc in 0..shape[0] * shape[1] {
            let input = &x[nc * spatial_size..(nc + 1) * spatial_size];
            let value = match self.kind {
                GlobalPoolKind::Max => input.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                GlobalPoolKind::Average if spatial_size != 0 => {
                    input.iter().sum::<f32>() / spatial_size as f32
                }
                GlobalPoolKind::Lp(p) if spatial_size != 0 => input
                    .iter()
                    .map(|value| value.abs().powi(p))
                    .sum::<f32>()
                    .powf(1.0 / p as f32),
                GlobalPoolKind::Average | GlobalPoolKind::Lp(_) => 0.0,
            };
            values.push(value);
        }
        write_dense_f32_narrow("GlobalPool", &mut outputs[0], &values)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    fn average(kernel: &PoolKernel, x: &Owned, shape: &[usize]) -> Vec<f32> {
        let mut out = Owned::zeros_f32(shape);
        kernel.execute(&[x.view()], &mut [out.view_mut()]).unwrap();
        out.to_f32()
    }

    fn params(
        kernel: &[usize],
        strides: &[usize],
        pads: &[usize],
        dilations: &[usize],
    ) -> PoolParams {
        PoolParams {
            kernel: kernel.to_vec(),
            strides: strides.to_vec(),
            pads: pads.to_vec(),
            dilations: dilations.to_vec(),
        }
    }

    #[test]
    fn average_pool_stride_padding_and_ceil_mode() {
        let x = Owned::f32(
            &[1, 1, 4, 4],
            &(1..=16).map(|v| v as f32).collect::<Vec<_>>(),
        );
        let base = PoolKernel {
            params: params(&[2, 2], &[2, 2], &[0, 0, 0, 0], &[1, 1]),
            auto_pad: AutoPad::NotSet,
            ceil_mode: false,
            kind: PoolKind::Average { include_pad: false },
        };
        assert_eq!(
            average(&base, &x, &[1, 1, 2, 2]),
            vec![3.5, 5.5, 11.5, 13.5]
        );

        let padded = PoolKernel {
            params: params(&[2, 2], &[2, 2], &[1, 1, 1, 1], &[1, 1]),
            auto_pad: AutoPad::NotSet,
            ceil_mode: false,
            kind: PoolKind::Average { include_pad: false },
        };
        assert_eq!(average(&padded, &x, &[1, 1, 3, 3])[0], 1.0);
        let include_pad = PoolKernel {
            kind: PoolKind::Average { include_pad: true },
            ..padded
        };
        assert_eq!(average(&include_pad, &x, &[1, 1, 3, 3])[0], 0.25);

        let ceil = PoolKernel {
            params: params(&[2, 2], &[3, 3], &[0, 0, 0, 0], &[1, 1]),
            auto_pad: AutoPad::NotSet,
            ceil_mode: true,
            kind: PoolKind::Average { include_pad: false },
        };
        assert_eq!(
            average(&ceil, &x, &[1, 1, 2, 2]),
            vec![3.5, 6.0, 13.5, 16.0]
        );
    }

    #[test]
    fn max_pool_indices_and_dilations() {
        let x = Owned::f32(
            &[1, 1, 4, 4],
            &(1..=16).map(|v| v as f32).collect::<Vec<_>>(),
        );
        let kernel = PoolKernel {
            params: params(&[2, 2], &[2, 2], &[0, 0, 0, 0], &[1, 1]),
            auto_pad: AutoPad::NotSet,
            ceil_mode: false,
            kind: PoolKind::Max {
                storage_order: false,
            },
        };
        let mut out = Owned::zeros_f32(&[1, 1, 2, 2]);
        let mut indices = Owned::i64(&[1, 1, 2, 2], &[0; 4]);
        kernel
            .execute(&[x.view()], &mut [out.view_mut(), indices.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![6., 8., 14., 16.]);
        assert_eq!(indices.to_i64(), vec![5, 7, 13, 15]);

        let dilated = PoolKernel {
            params: params(&[2, 2], &[1, 1], &[0, 0, 0, 0], &[2, 2]),
            auto_pad: AutoPad::NotSet,
            ceil_mode: false,
            kind: PoolKind::Max {
                storage_order: false,
            },
        };
        let mut dilated_out = Owned::zeros_f32(&[1, 1, 2, 2]);
        dilated
            .execute(&[x.view()], &mut [dilated_out.view_mut()])
            .unwrap();
        assert_eq!(dilated_out.to_f32(), vec![11., 12., 15., 16.]);
    }

    #[test]
    fn max_pool_indices_include_nc_base_for_both_storage_orders() {
        let x = Owned::f32(
            &[1, 2, 4, 4],
            &(1..=32).map(|v| v as f32).collect::<Vec<_>>(),
        );
        for (storage_order, expected_indices) in [
            (false, vec![5, 7, 13, 15, 21, 23, 29, 31]),
            (true, vec![5, 13, 7, 15, 21, 29, 23, 31]),
        ] {
            let kernel = PoolKernel {
                params: params(&[2, 2], &[2, 2], &[0, 0, 0, 0], &[1, 1]),
                auto_pad: AutoPad::NotSet,
                ceil_mode: false,
                kind: PoolKind::Max { storage_order },
            };
            let mut out = Owned::zeros_f32(&[1, 2, 2, 2]);
            let mut indices = Owned::i64(&[1, 2, 2, 2], &[0; 8]);
            kernel
                .execute(&[x.view()], &mut [out.view_mut(), indices.view_mut()])
                .unwrap();

            assert_eq!(indices.to_i64(), expected_indices);
        }
    }

    #[test]
    fn global_pool_and_same_upper_match_explicit_padding() {
        let x = Owned::f32(
            &[1, 2, 3, 3],
            &[
                1., 2., 3., 4., 5., 6., 7., 8., 9., -1., -2., -3., -4., -5., -6., -7., -8., -9.,
            ],
        );
        let mut avg = Owned::zeros_f32(&[1, 2, 1, 1]);
        GlobalPoolKernel {
            kind: GlobalPoolKind::Average,
        }
        .execute(&[x.view()], &mut [avg.view_mut()])
        .unwrap();
        assert_eq!(avg.to_f32(), vec![5., -5.]);
        let mut max = Owned::zeros_f32(&[1, 2, 1, 1]);
        GlobalPoolKernel {
            kind: GlobalPoolKind::Max,
        }
        .execute(&[x.view()], &mut [max.view_mut()])
        .unwrap();
        assert_eq!(max.to_f32(), vec![9., -1.]);

        let same = PoolKernel {
            params: params(&[2, 2], &[2, 2], &[0, 0, 0, 0], &[1, 1]),
            auto_pad: AutoPad::SameUpper,
            ceil_mode: false,
            kind: PoolKind::Average { include_pad: false },
        };
        let explicit = PoolKernel {
            params: params(&[2, 2], &[2, 2], &[0, 0, 1, 1], &[1, 1]),
            auto_pad: AutoPad::NotSet,
            ceil_mode: false,
            kind: PoolKind::Average { include_pad: false },
        };
        assert_eq!(
            average(&same, &x, &[1, 2, 2, 2]),
            average(&explicit, &x, &[1, 2, 2, 2])
        );
    }

    fn lp(kernel: PoolKernel, x: &Owned, shape: &[usize]) -> Vec<f32> {
        let mut out = Owned::zeros_f32(shape);
        kernel.execute(&[x.view()], &mut [out.view_mut()]).unwrap();
        out.to_f32()
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn lp_pool_1d_default_p() {
        let x = Owned::f32(&[1, 1, 3], &[1., -2., 3.]);
        let mut node = Node::new(onnx_runtime_ir::NodeId(0), "LpPool", vec![], vec![]);
        node.attributes
            .insert("kernel_shape".into(), Attribute::Ints(vec![2]));
        let kernel = LpPoolFactory.create(&node, &[vec![1, 1, 3]]).unwrap();
        let mut output = Owned::zeros_f32(&[1, 1, 2]);
        kernel
            .execute(&[x.view()], &mut [output.view_mut()])
            .unwrap();
        assert_close(&output.to_f32(), &[5.0_f32.sqrt(), 13.0_f32.sqrt()]);
    }

    #[test]
    fn lp_pool_2d_pads_strides_and_p_one() {
        let x = Owned::f32(&[1, 1, 2, 2], &[1., 2., 3., 4.]);
        let values = lp(
            PoolKernel {
                params: params(&[2, 2], &[1, 1], &[1, 1, 1, 1], &[1, 1]),
                auto_pad: AutoPad::NotSet,
                ceil_mode: false,
                kind: PoolKind::Lp { p: 1 },
            },
            &x,
            &[1, 1, 3, 3],
        );
        assert_eq!(values, vec![1., 3., 2., 4., 10., 6., 3., 7., 4.]);

        let strided = lp(
            PoolKernel {
                params: params(&[2, 2], &[2, 2], &[0, 0, 0, 0], &[1, 1]),
                auto_pad: AutoPad::NotSet,
                ceil_mode: false,
                kind: PoolKind::Lp { p: 2 },
            },
            &Owned::f32(
                &[1, 1, 4, 4],
                &(1..=16).map(|value| value as f32).collect::<Vec<_>>(),
            ),
            &[1, 1, 2, 2],
        );
        assert_close(
            &strided,
            &[
                66.0_f32.sqrt(),
                138.0_f32.sqrt(),
                546.0_f32.sqrt(),
                746.0_f32.sqrt(),
            ],
        );
    }

    #[test]
    fn lp_pool_2d_dilations() {
        let x = Owned::f32(
            &[1, 1, 3, 3],
            &(1..=9).map(|value| value as f32).collect::<Vec<_>>(),
        );
        let values = lp(
            PoolKernel {
                params: params(&[2, 2], &[1, 1], &[0, 0, 0, 0], &[2, 2]),
                auto_pad: AutoPad::NotSet,
                ceil_mode: false,
                kind: PoolKind::Lp { p: 2 },
            },
            &x,
            &[1, 1, 1, 1],
        );
        assert_close(&values, &[140.0_f32.sqrt()]);
    }

    #[test]
    fn lp_pool_2d_same_upper_and_lower() {
        let x = Owned::f32(
            &[1, 1, 3, 3],
            &(1..=9).map(|value| value as f32).collect::<Vec<_>>(),
        );
        let make_kernel = |auto_pad| PoolKernel {
            params: params(&[2, 2], &[2, 2], &[0, 0, 0, 0], &[1, 1]),
            auto_pad,
            ceil_mode: false,
            kind: PoolKind::Lp { p: 2 },
        };
        assert_close(
            &lp(make_kernel(AutoPad::SameUpper), &x, &[1, 1, 2, 2]),
            &[46.0_f32.sqrt(), 45.0_f32.sqrt(), 113.0_f32.sqrt(), 9.],
        );
        assert_close(
            &lp(make_kernel(AutoPad::SameLower), &x, &[1, 1, 2, 2]),
            &[1., 13.0_f32.sqrt(), 65.0_f32.sqrt(), 206.0_f32.sqrt()],
        );
    }

    #[test]
    fn lp_pool_3d_and_global_lp_pool() {
        let x = Owned::f32(
            &[1, 1, 2, 2, 2],
            &(1..=8).map(|value| value as f32).collect::<Vec<_>>(),
        );
        let values = lp(
            PoolKernel {
                params: params(&[2, 2, 2], &[1, 1, 1], &[0, 0, 0, 0, 0, 0], &[1, 1, 1]),
                auto_pad: AutoPad::NotSet,
                ceil_mode: false,
                kind: PoolKind::Lp { p: 2 },
            },
            &x,
            &[1, 1, 1, 1, 1],
        );
        assert_close(&values, &[204.0_f32.sqrt()]);

        let global_input = Owned::f32(&[1, 1, 3], &[-1., 2., -2.]);
        let mut global_output = Owned::zeros_f32(&[1, 1, 1]);
        GlobalPoolKernel {
            kind: GlobalPoolKind::Lp(3),
        }
        .execute(&[global_input.view()], &mut [global_output.view_mut()])
        .unwrap();
        assert_close(&global_output.to_f32(), &[17.0_f32.cbrt()]);
    }
}
