//! `Expand`: broadcast the input to a target shape (`docs/ORT2.md` §4.4).
//!
//! ONNX `Expand` uses **bidirectional** (numpy) broadcasting between the input
//! shape and the `shape` input; the resulting shape is the broadcast of the
//! two. The session materializes that broadcast shape as the pre-allocated
//! output view, so this kernel only has to replicate the input's elements into
//! it. It moves raw element bytes and is therefore dtype-agnostic.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Node, compute_contiguous_strides};

use super::{check_arity, elem_size, to_dense_bytes, write_dense_bytes};
use crate::strided::{next_index, numel};

/// Stateless Expand kernel.
pub struct ExpandKernel;

/// Factory for [`ExpandKernel`] (no attributes).
pub struct ExpandFactory;

impl KernelFactory for ExpandFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ExpandKernel))
    }
}

impl Kernel for ExpandKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        // data + shape tensor; the shape tensor is consumed to build the output
        // view upstream, so only the data input is read here.
        check_arity("Expand", inputs, outputs, 2, 2, 1)?;
        let esize = elem_size(inputs[0].dtype)?;
        let src = to_dense_bytes(&inputs[0])?;
        let in_shape = inputs[0].shape;
        let out_shape = outputs[0].shape.to_vec();
        let out_rank = out_shape.len();

        // Effective element stride of each output axis into the source (0 where
        // the source axis is broadcast / absent). Right-aligned numpy rule.
        let src_strides = compute_contiguous_strides(in_shape);
        let mut eff = vec![0i64; out_rank];
        for axis in 0..out_rank {
            let src_axis = axis as isize - (out_rank as isize - in_shape.len() as isize);
            if src_axis < 0 {
                continue;
            }
            let src_axis = src_axis as usize;
            let src_dim = in_shape[src_axis];
            if src_dim == out_shape[axis] {
                eff[axis] = src_strides[src_axis];
            } else if src_dim == 1 {
                eff[axis] = 0;
            } else {
                return Err(EpError::KernelFailed(format!(
                    "Expand: input shape {in_shape:?} not broadcastable to {out_shape:?}"
                )));
            }
        }

        let n = numel(&out_shape);
        let mut out = vec![0u8; n * esize];
        if n > 0 {
            let mut idx = vec![0usize; out_rank];
            let mut w = 0usize;
            loop {
                let mut src_off = 0i64;
                for (e, &i) in eff.iter().zip(&idx) {
                    src_off += e * i as i64;
                }
                let s = src_off as usize * esize;
                out[w..w + esize].copy_from_slice(&src[s..s + esize]);
                w += esize;
                if !next_index(&out_shape, &mut idx) {
                    break;
                }
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
    fn expand_bf16_preserves_element_bits() {
        // [1,3] -> [2,3]; pure byte movement must preserve bf16 patterns exactly.
        let x = Owned::bf16(&[1, 3], &[1.5, -2.25, 3.75]);
        let shape = Owned::i64(&[2], &[2, 3]);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[2, 3]);
        ExpandKernel
            .execute(&[x.view(), shape.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(
            out.to_bf16_as_f32(),
            vec![1.5, -2.25, 3.75, 1.5, -2.25, 3.75]
        );
    }

    #[test]
    fn expand_row_vector_to_matrix() {
        // [1,3] -> [2,3]
        let x = Owned::f32(&[1, 3], &[1., 2., 3.]);
        let shape = Owned::i64(&[2], &[2, 3]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        ExpandKernel
            .execute(&[x.view(), shape.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![1., 2., 3., 1., 2., 3.]);
    }

    #[test]
    fn expand_column_vector_to_matrix() {
        // [3,1] -> [3,4]
        let x = Owned::f32(&[3, 1], &[1., 2., 3.]);
        let shape = Owned::i64(&[2], &[3, 4]);
        let mut out = Owned::zeros_f32(&[3, 4]);
        ExpandKernel
            .execute(&[x.view(), shape.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(
            out.to_f32(),
            vec![1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.]
        );
    }

    #[test]
    fn expand_bidirectional_new_leading_axis_int64() {
        // input [3] with target [2,1] broadcasts bidirectionally to [2,3].
        let x = Owned::i64(&[3], &[7, 8, 9]);
        let shape = Owned::i64(&[2], &[2, 1]);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::Int64, &[2, 3]);
        ExpandKernel
            .execute(&[x.view(), shape.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i64(), vec![7, 8, 9, 7, 8, 9]);
    }
}
