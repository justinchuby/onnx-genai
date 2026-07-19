//! `Compress` selection along an optional axis.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use super::{check_arity, elem_size, to_dense_bytes, write_dense_bytes};
use crate::strided::numel;

pub struct CompressKernel {
    axis: Option<i64>,
}

pub struct CompressFactory;

impl KernelFactory for CompressFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(CompressKernel {
            axis: node.attr("axis").and_then(Attribute::as_int),
        }))
    }
}

impl Kernel for CompressKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Compress", inputs, outputs, 2, 2, 1)?;
        let input = &inputs[0];
        let condition = &inputs[1];
        if condition.dtype != DataType::Bool || condition.shape.len() != 1 {
            return Err(EpError::KernelFailed(
                "Compress: condition must be a one-dimensional Bool tensor".into(),
            ));
        }
        if outputs[0].dtype != input.dtype {
            return Err(EpError::KernelFailed(
                "Compress: output dtype must match input dtype".into(),
            ));
        }

        let (input_shape, axis) = match self.axis {
            Some(axis) => {
                let rank = input.shape.len();
                let axis = if axis < 0 { axis + rank as i64 } else { axis };
                if axis < 0 || axis as usize >= rank {
                    return Err(EpError::KernelFailed("Compress: axis out of range".into()));
                }
                (input.shape.to_vec(), axis as usize)
            }
            None => (vec![numel(input.shape)], 0),
        };
        let condition = to_dense_bytes(condition)?;
        let axis_len = input_shape[axis];
        let selected: Vec<usize> = (0..axis_len)
            .filter(|&i| i < condition.len() && condition[i] != 0)
            .collect();

        let mut expected_shape = input_shape.clone();
        expected_shape[axis] = selected.len();
        if outputs[0].shape != expected_shape {
            return Err(EpError::KernelFailed(
                "Compress: output shape does not match the selected condition elements".into(),
            ));
        }

        let element_size = elem_size(input.dtype)?;
        let src = to_dense_bytes(input)?;
        let outer = numel(&input_shape[..axis]);
        let inner = numel(&input_shape[axis + 1..]);
        let mut out = vec![0; numel(&expected_shape) * element_size];
        for outer_index in 0..outer {
            for (output_axis, &input_axis) in selected.iter().enumerate() {
                let src_offset = (outer_index * axis_len + input_axis) * inner * element_size;
                let dst_offset =
                    (outer_index * selected.len() + output_axis) * inner * element_size;
                let bytes = inner * element_size;
                out[dst_offset..dst_offset + bytes]
                    .copy_from_slice(&src[src_offset..src_offset + bytes]);
            }
        }
        write_dense_bytes(&mut outputs[0], &out)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    #[test]
    fn selects_axis_zero() {
        let input = Owned::i32(&[3, 2], &[1, 2, 3, 4, 5, 6]);
        let condition = Owned::bool_(&[3], &[true, false, true]);
        let mut out = Owned::zeros(DataType::Int32, &[2, 2]);
        CompressKernel { axis: Some(0) }
            .execute(&[input.view(), condition.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i32(), vec![1, 2, 5, 6]);
    }

    #[test]
    fn selects_negative_axis_with_short_condition() {
        let input = Owned::i32(&[2, 3], &[1, 2, 3, 4, 5, 6]);
        let condition = Owned::bool_(&[2], &[false, true]);
        let mut out = Owned::zeros(DataType::Int32, &[2, 1]);
        CompressKernel { axis: Some(-1) }
            .execute(&[input.view(), condition.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i32(), vec![2, 5]);
    }

    #[test]
    fn omitting_axis_flattens_before_selection() {
        let input = Owned::i32(&[2, 2], &[1, 2, 3, 4]);
        let condition = Owned::bool_(&[4], &[false, true, false, true]);
        let mut out = Owned::zeros(DataType::Int32, &[2]);
        CompressKernel { axis: None }
            .execute(&[input.view(), condition.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i32(), vec![2, 4]);
    }
}
