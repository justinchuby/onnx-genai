//! `Reshape`: a metadata-only op for contiguous inputs, with a logical-order
//! copy fallback for strided inputs.

use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, Result, TensorMut, TensorView, ViewOutput,
};
use onnx_runtime_ir::{Node, compute_contiguous_strides, is_contiguous};

use super::{check_arity, to_dense_bytes, to_dense_i64, write_dense_bytes};

pub struct ReshapeKernel {
    allowzero: bool,
}

pub struct ReshapeFactory;

impl KernelFactory for ReshapeFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let allowzero = node
            .attr("allowzero")
            .and_then(|value| value.as_int())
            .unwrap_or(0)
            != 0;
        Ok(Box::new(ReshapeKernel { allowzero }))
    }
}

impl Kernel for ReshapeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Reshape", inputs, outputs, 1, 2, 1)?;
        if outputs[0].dtype != inputs[0].dtype {
            return Err(EpError::KernelFailed(format!(
                "Reshape: output dtype {:?} must match input dtype {:?}",
                outputs[0].dtype, inputs[0].dtype
            )));
        }
        let data = to_dense_bytes(&inputs[0])?;
        write_dense_bytes(&mut outputs[0], &data)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }

    fn view_outputs(&self, inputs: &[TensorView], num_outputs: usize) -> Option<Vec<ViewOutput>> {
        if num_outputs != 1 || inputs.len() != 2 {
            return None;
        }
        let data = &inputs[0];
        if data.dtype.byte_size() == 0 || !is_contiguous(data.shape, data.strides) {
            return None;
        }
        let requested = to_dense_i64(&inputs[1]).ok()?;
        let shape = resolve_shape(data.shape, &requested, self.allowzero)?;
        Some(vec![ViewOutput {
            input_index: 0,
            strides: compute_contiguous_strides(&shape),
            shape,
            byte_offset: data.byte_offset,
        }])
    }
}

fn resolve_shape(input: &[usize], requested: &[i64], allowzero: bool) -> Option<Vec<usize>> {
    let input_len = input
        .iter()
        .try_fold(1usize, |n, &dim| n.checked_mul(dim))?;
    let mut inferred = None;
    let mut known_len = 1usize;
    let mut output = Vec::with_capacity(requested.len());
    for (axis, &dim) in requested.iter().enumerate() {
        let resolved = match dim {
            -1 if inferred.replace(axis).is_none() => {
                output.push(1);
                continue;
            }
            -1 => return None,
            0 if !allowzero => *input.get(axis)?,
            0 => 0,
            positive if positive > 0 => usize::try_from(positive).ok()?,
            _ => return None,
        };
        known_len = known_len.checked_mul(resolved)?;
        output.push(resolved);
    }
    if let Some(axis) = inferred {
        if known_len == 0 || !input_len.is_multiple_of(known_len) {
            return None;
        }
        output[axis] = input_len / known_len;
    } else if known_len != input_len {
        return None;
    }
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    #[test]
    fn reshape_preserves_row_major_order() {
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let mut out = Owned::zeros_f32(&[3, 2]);
        ReshapeKernel { allowzero: false }
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![1., 2., 3., 4., 5., 6.]);
    }

    #[test]
    fn reshape_preserves_float16_bits() {
        let bits = [0x0001, 0x3c00, 0x7c00, 0x7e01, 0x8000, 0xfc00];
        let a = Owned::f16_bits(&[2, 3], &bits);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::Float16, &[3, 2]);
        ReshapeKernel { allowzero: false }
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_u16_bits(), bits);
    }

    #[test]
    fn reshape_bf16_preserves_element_bits() {
        let x = Owned::bf16(&[2, 2], &[1., -2., 3., 4.]);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[4]);
        ReshapeKernel { allowzero: false }
            .execute(&[x.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_u16_bits(), x.to_u16_bits());
    }

    #[test]
    fn reshape_contiguous_input_is_a_zero_copy_view() {
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let shape = Owned::i64(&[2], &[3, 2]);
        let view = ReshapeKernel { allowzero: false }
            .view_outputs(&[a.view(), shape.view()], 1)
            .expect("contiguous reshape should be a view")
            .pop()
            .unwrap();
        assert_eq!(view.input_index, 0);
        assert_eq!(view.shape, [3, 2]);
        assert_eq!(view.strides, [2, 1]);
        assert_eq!(view.byte_offset, 0);
    }

    #[test]
    fn reshape_shape_resolution_matches_onnx_rules() {
        assert_eq!(
            resolve_shape(&[2, 3, 4], &[0, -1], false),
            Some(vec![2, 12])
        );
        assert_eq!(resolve_shape(&[0, 3], &[0, 3], true), Some(vec![0, 3]));
        assert_eq!(resolve_shape(&[2, 3], &[-1, -1], false), None);
        assert_eq!(resolve_shape(&[2, 3], &[4, 2], false), None);
    }
}
