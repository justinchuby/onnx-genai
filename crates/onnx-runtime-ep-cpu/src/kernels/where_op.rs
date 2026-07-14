//! `Where(cond, x, y)`: elementwise select with numpy-style broadcasting
//! (`docs/ORT2.md` §4.4).
//!
//! `cond` is a `Bool` tensor; `x` and `y` share the output dtype. The three
//! operands broadcast together to the output shape. Selection copies raw
//! element bytes (`x` where `cond` is true, else `y`), so the kernel is
//! **dtype-agnostic** across every fixed-width `x`/`y` dtype.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{compute_contiguous_strides, DataType, Node};

use super::{check_arity, elem_size, to_dense_bytes, write_dense_bytes};
use crate::strided::{next_index, numel};

/// Stateless `Where` kernel (broadcasting select, dtype-agnostic on x/y).
pub struct WhereKernel;

/// Factory for [`WhereKernel`] (no attributes).
pub struct WhereFactory;

impl KernelFactory for WhereFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(WhereKernel))
    }
}

/// Effective per-output-axis element strides of `src_shape` right-aligned to
/// `out_shape` (0 where the axis is broadcast). Errors on an incompatible axis.
fn effective_strides(src_shape: &[usize], out_shape: &[usize]) -> Result<Vec<i64>> {
    let out_rank = out_shape.len();
    let src_strides = compute_contiguous_strides(src_shape);
    let mut eff = vec![0i64; out_rank];
    for (axis, e) in eff.iter_mut().enumerate() {
        let src_axis = axis as isize - (out_rank as isize - src_shape.len() as isize);
        if src_axis < 0 {
            continue;
        }
        let src_axis = src_axis as usize;
        let src_dim = src_shape[src_axis];
        if src_dim == out_shape[axis] {
            *e = src_strides[src_axis];
        } else if src_dim == 1 {
            *e = 0;
        } else {
            return Err(EpError::KernelFailed(format!(
                "Where: operand shape {src_shape:?} is not broadcast-compatible with output \
                 shape {out_shape:?}. WHY: axis {axis} has extent {src_dim}, expected 1 or {}. \
                 HOW: reshape the operand to broadcast against the output.",
                out_shape[axis]
            )));
        }
    }
    Ok(eff)
}

impl Kernel for WhereKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Where", inputs, outputs, 3, 3, 1)?;
        let (cond, x, y) = (&inputs[0], &inputs[1], &inputs[2]);
        if cond.dtype != DataType::Bool {
            return Err(EpError::KernelFailed(format!(
                "Where: condition must be Bool, got {:?}. WHY: `Where` selects on a boolean \
                 mask. HOW: pass a Bool condition tensor.",
                cond.dtype
            )));
        }
        let dtype = outputs[0].dtype;
        if x.dtype != dtype || y.dtype != dtype {
            return Err(EpError::KernelFailed(format!(
                "Where: branch dtypes {:?}/{:?} must match output dtype {:?}. WHY: both branches \
                 feed one typed output. HOW: cast x and y to the output dtype.",
                x.dtype, y.dtype, dtype
            )));
        }
        let esize = elem_size(dtype)?;

        let out_shape = outputs[0].shape.to_vec();
        let out_rank = out_shape.len();
        let cond_eff = effective_strides(cond.shape, &out_shape)?;
        let x_eff = effective_strides(x.shape, &out_shape)?;
        let y_eff = effective_strides(y.shape, &out_shape)?;

        let cond_bytes = to_dense_bytes(cond)?;
        let x_bytes = to_dense_bytes(x)?;
        let y_bytes = to_dense_bytes(y)?;

        let n = numel(&out_shape);
        let mut out = vec![0u8; n * esize];
        if n == 0 {
            return write_dense_bytes(&mut outputs[0], &out);
        }

        let mut idx = vec![0usize; out_rank];
        let mut w = 0usize;
        loop {
            let (mut c_off, mut x_off, mut y_off) = (0i64, 0i64, 0i64);
            for d in 0..out_rank {
                let i = idx[d] as i64;
                c_off += cond_eff[d] * i;
                x_off += x_eff[d] * i;
                y_off += y_eff[d] * i;
            }
            let take_x = cond_bytes[c_off as usize] != 0;
            let (src, off) = if take_x {
                (&x_bytes, x_off as usize)
            } else {
                (&y_bytes, y_off as usize)
            };
            let start = off * esize;
            out[w..w + esize].copy_from_slice(&src[start..start + esize]);
            w += esize;
            if !next_index(&out_shape, &mut idx) {
                break;
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

    fn run(cond: &Owned, x: &Owned, y: &Owned, out: &mut Owned) {
        WhereKernel
            .execute(
                &[cond.view(), x.view(), y.view()],
                &mut [out.view_mut()],
            )
            .unwrap();
    }

    #[test]
    fn where_same_shape() {
        let c = Owned::bool_(&[4], &[true, false, true, false]);
        let x = Owned::f32(&[4], &[1., 2., 3., 4.]);
        let y = Owned::f32(&[4], &[10., 20., 30., 40.]);
        let mut out = Owned::zeros_f32(&[4]);
        run(&c, &x, &y, &mut out);
        assert_eq!(out.to_f32(), vec![1., 20., 3., 40.]);
    }

    #[test]
    fn where_broadcasts_condition() {
        // cond [2,1], x [2,2], y scalar -> [2,2]
        let c = Owned::bool_(&[2, 1], &[true, false]);
        let x = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let y = Owned::f32(&[], &[0.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        run(&c, &x, &y, &mut out);
        // row0 cond true -> x, row1 cond false -> y(0)
        assert_eq!(out.to_f32(), vec![1., 2., 0., 0.]);
    }

    #[test]
    fn where_int64_dtype_agnostic() {
        let c = Owned::bool_(&[3], &[false, true, false]);
        let x = Owned::i64(&[3], &[1, 2, 3]);
        let y = Owned::i64(&[3], &[7, 8, 9]);
        let mut out = Owned::zeros(DataType::Int64, &[3]);
        run(&c, &x, &y, &mut out);
        assert_eq!(out.to_i64(), vec![7, 2, 9]);
    }

    #[test]
    fn where_broadcasts_branches() {
        // cond [2,2], x [1,2], y [2,1] -> [2,2]
        let c = Owned::bool_(&[2, 2], &[true, false, false, true]);
        let x = Owned::f32(&[1, 2], &[1., 2.]);
        let y = Owned::f32(&[2, 1], &[10., 20.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        run(&c, &x, &y, &mut out);
        // (0,0) cond T -> x[0,0]=1; (0,1) F -> y[0,0]=10;
        // (1,0) F -> y[1,0]=20; (1,1) T -> x[0,1]=2
        assert_eq!(out.to_f32(), vec![1., 10., 20., 2.]);
    }
}
