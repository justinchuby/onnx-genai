//! `CenterCropPad`: centered crop or zero-pad selected tensor axes.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, Node, compute_contiguous_strides};

use super::{check_arity, elem_size, to_dense_bytes, to_dense_i64, write_dense_bytes};
use crate::strided::{next_index, numel};

pub struct CenterCropPadKernel {
    axes: Option<Vec<i64>>,
}

pub struct CenterCropPadFactory;

impl KernelFactory for CenterCropPadFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(CenterCropPadKernel {
            axes: node
                .attr("axes")
                .and_then(Attribute::as_ints)
                .map(<[i64]>::to_vec),
        }))
    }
}

impl Kernel for CenterCropPadKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("CenterCropPad", inputs, outputs, 2, 2, 1)?;
        let rank = inputs[0].shape.len();
        let shape = to_dense_i64(&inputs[1])?;
        let raw_axes: Vec<i64> = self
            .axes
            .clone()
            .unwrap_or_else(|| (0..rank as i64).collect());
        if shape.len() != raw_axes.len() {
            return Err(EpError::KernelFailed(format!(
                "CenterCropPad: shape has {} dimensions for {} axes",
                shape.len(),
                raw_axes.len()
            )));
        }
        let mut target = inputs[0].shape.to_vec();
        let mut axes = Vec::with_capacity(raw_axes.len());
        for (&raw_axis, &dimension) in raw_axes.iter().zip(&shape) {
            let axis = if raw_axis < 0 {
                raw_axis + rank as i64
            } else {
                raw_axis
            };
            if axis < 0 || axis as usize >= rank || dimension < 0 {
                return Err(EpError::KernelFailed(
                    "CenterCropPad: axes must be in range and shape dimensions non-negative".into(),
                ));
            }
            target[axis as usize] = dimension as usize;
            axes.push(axis as usize);
        }
        if outputs[0].shape != target {
            return Err(EpError::KernelFailed(format!(
                "CenterCropPad: output shape {:?}, expected {target:?}",
                outputs[0].shape
            )));
        }

        let esize = elem_size(inputs[0].dtype)?;
        let source = to_dense_bytes(&inputs[0])?;
        let source_strides = compute_contiguous_strides(inputs[0].shape);
        let mut output = vec![0; numel(&target) * esize];
        if !output.is_empty() {
            let mut index = vec![0; rank];
            let mut write = 0;
            loop {
                let mut source_offset = 0usize;
                let mut in_range = true;
                for axis in 0..rank {
                    let source_dimension = inputs[0].shape[axis];
                    let target_dimension = target[axis];
                    let source_index = if axes.contains(&axis) {
                        let crop_before = source_dimension.saturating_sub(target_dimension) / 2;
                        let pad_before = target_dimension.saturating_sub(source_dimension) / 2;
                        index[axis] as isize + crop_before as isize - pad_before as isize
                    } else {
                        index[axis] as isize
                    };
                    if source_index < 0 || source_index >= source_dimension as isize {
                        in_range = false;
                        break;
                    }
                    source_offset += source_index as usize * source_strides[axis] as usize;
                }
                if in_range {
                    let read = source_offset * esize;
                    output[write..write + esize].copy_from_slice(&source[read..read + esize]);
                }
                write += esize;
                if !next_index(&target, &mut index) {
                    break;
                }
            }
        }
        write_dense_bytes(&mut outputs[0], &output)
    }

    fn supports_strided_input(&self, input_idx: usize) -> bool {
        input_idx == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    fn kernel(axes: Option<Vec<i64>>) -> CenterCropPadKernel {
        CenterCropPadKernel { axes }
    }

    #[test]
    fn center_crop_picks_lower_start_for_odd_difference() {
        let input = Owned::f32(&[5], &[0., 1., 2., 3., 4.]);
        let shape = Owned::i64(&[1], &[2]);
        let mut out = Owned::zeros_f32(&[2]);
        kernel(None)
            .execute(&[input.view(), shape.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![1., 2.]);
    }

    #[test]
    fn center_pad_adds_extra_cell_at_end() {
        let input = Owned::f32(&[2], &[4., 5.]);
        let shape = Owned::i64(&[1], &[5]);
        let mut out = Owned::zeros_f32(&[5]);
        kernel(None)
            .execute(&[input.view(), shape.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![0., 4., 5., 0., 0.]);
    }

    #[test]
    fn center_crop_and_pad_across_axes() {
        let input = Owned::f32(&[3, 2], &[1., 2., 3., 4., 5., 6.]);
        let shape = Owned::i64(&[2], &[2, 4]);
        let mut out = Owned::zeros_f32(&[2, 4]);
        kernel(None)
            .execute(&[input.view(), shape.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![0., 1., 2., 0., 0., 3., 4., 0.]);
    }

    #[test]
    fn center_crop_negative_axes_preserves_other_dimensions() {
        let input = Owned::f32(&[2, 3, 4], &(0..24).map(|v| v as f32).collect::<Vec<_>>());
        let shape = Owned::i64(&[2], &[2, 2]);
        let mut out = Owned::zeros_f32(&[2, 2, 2]);
        kernel(Some(vec![-2, -1]))
            .execute(&[input.view(), shape.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![1., 2., 5., 6., 13., 14., 17., 18.]);
    }
}
