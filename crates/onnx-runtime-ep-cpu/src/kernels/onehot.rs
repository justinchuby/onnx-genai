//! Opset-11 `OneHot`, including valid negative index wrapping.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use super::{check_arity, elem_size, to_dense_bytes, to_dense_i64, write_dense_bytes};
use crate::strided::numel;

pub struct OneHotKernel {
    axis: i64,
}

pub struct OneHotFactory;

impl KernelFactory for OneHotFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(OneHotKernel {
            axis: node.attr("axis").and_then(Attribute::as_int).unwrap_or(-1),
        }))
    }
}

impl Kernel for OneHotKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("OneHot", inputs, outputs, 3, 3, 1)?;
        if inputs[0].dtype != DataType::Int64 || inputs[1].dtype != DataType::Int64 {
            return Err(EpError::KernelFailed(
                "OneHot: indices and depth must be Int64".into(),
            ));
        }
        let depth_input = to_dense_i64(&inputs[1])?;
        if depth_input.len() != 1 || depth_input[0] < 0 {
            return Err(EpError::KernelFailed(
                "OneHot: depth must be a non-negative scalar".into(),
            ));
        }
        let depth = usize::try_from(depth_input[0]).map_err(|_| {
            EpError::KernelFailed("OneHot: depth exceeds addressable memory".into())
        })?;
        let values = &inputs[2];
        if values.shape != [2] || outputs[0].dtype != values.dtype {
            return Err(EpError::KernelFailed(
                "OneHot: values must have shape [2] and output must match its dtype".into(),
            ));
        }

        let output_rank = inputs[0].shape.len() + 1;
        let axis = if self.axis < 0 {
            self.axis + output_rank as i64
        } else {
            self.axis
        };
        if axis < 0 || axis as usize >= output_rank {
            return Err(EpError::KernelFailed("OneHot: axis out of range".into()));
        }
        let axis = axis as usize;
        let mut expected_shape = inputs[0].shape.to_vec();
        expected_shape.insert(axis, depth);
        if outputs[0].shape != expected_shape {
            return Err(EpError::KernelFailed(
                "OneHot: output shape does not match indices, depth, and axis".into(),
            ));
        }

        let element_size = elem_size(values.dtype)?;
        let values = to_dense_bytes(values)?;
        let off_value = &values[..element_size];
        let on_value = &values[element_size..];
        let mut out = vec![0; numel(&expected_shape) * element_size];
        for chunk in out.chunks_exact_mut(element_size) {
            chunk.copy_from_slice(off_value);
        }
        if depth == 0 {
            return write_dense_bytes(&mut outputs[0], &out);
        }

        let output_strides = contiguous(&expected_shape);
        let index_strides = contiguous(inputs[0].shape);
        for (index_linear, index) in to_dense_i64(&inputs[0])?.into_iter().enumerate() {
            let index = match index {
                index if index >= 0 && index < depth as i64 => index as usize,
                index if index < 0 && index >= -(depth as i64) => (index + depth as i64) as usize,
                _ => continue,
            };
            let mut rem = index_linear;
            let mut output_linear = index * output_strides[axis];
            for (d, &output_stride) in output_strides.iter().enumerate() {
                if d != axis {
                    let index_axis = if d < axis { d } else { d - 1 };
                    let coordinate = rem / index_strides[index_axis];
                    rem %= index_strides[index_axis];
                    output_linear += coordinate * output_stride;
                }
            }
            out[output_linear * element_size..(output_linear + 1) * element_size]
                .copy_from_slice(on_value);
        }
        write_dense_bytes(&mut outputs[0], &out)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn contiguous(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len()).rev().skip(1) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    #[test]
    fn inserts_nonfinal_axis() {
        let indices = Owned::i64(&[2, 2], &[0, 2, 1, 0]);
        let depth = Owned::i64(&[], &[3]);
        let values = Owned::i32(&[2], &[4, 9]);
        let mut out = Owned::zeros(DataType::Int32, &[2, 3, 2]);
        OneHotKernel { axis: 1 }
            .execute(
                &[indices.view(), depth.view(), values.view()],
                &mut [out.view_mut()],
            )
            .unwrap();
        assert_eq!(out.to_i32(), vec![9, 4, 4, 4, 4, 9, 4, 9, 9, 4, 4, 4]);
    }

    #[test]
    fn valid_negative_indices_wrap_and_out_of_range_indices_are_off() {
        let indices = Owned::i64(&[3], &[-1, 3, -4]);
        let depth = Owned::i64(&[], &[3]);
        let values = Owned::f32(&[2], &[0., 1.]);
        let mut out = Owned::zeros_f32(&[3, 3]);
        OneHotKernel { axis: -1 }
            .execute(
                &[indices.view(), depth.view(), values.view()],
                &mut [out.view_mut()],
            )
            .unwrap();
        assert_eq!(out.to_f32(), vec![0., 0., 1., 0., 0., 0., 0., 0., 0.]);
    }
}
