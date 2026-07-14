//! `Add`: elementwise addition with numpy-style broadcasting for f32
//! (`docs/ORT2.md` §4.4).

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{compute_contiguous_strides, Node};

use super::{check_arity, to_dense_f32, write_dense_f32};
use crate::strided::{next_index, numel};

/// Stateless f32 broadcasting Add kernel.
pub struct AddKernel;

/// Factory for [`AddKernel`] (no attributes).
pub struct AddFactory;

impl KernelFactory for AddFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(AddKernel))
    }
}

/// Broadcast a dense row-major `src` of `src_shape` onto `out_shape`, calling
/// `f` with `(flat_out_index, src_value)` for every output element.
///
/// Implements numpy broadcasting: `src_shape` is right-aligned to `out_shape`
/// and any axis of extent 1 (or missing) contributes stride 0.
pub fn broadcast_apply(
    src: &[f32],
    src_shape: &[usize],
    out_shape: &[usize],
    mut f: impl FnMut(usize, f32),
) -> Result<()> {
    let out_rank = out_shape.len();
    let src_strides = compute_contiguous_strides(src_shape);
    // Effective stride of each output axis into `src` (0 where broadcast).
    let mut eff = vec![0i64; out_rank];
    for axis in 0..out_rank {
        // Corresponding axis in src (right-aligned); absent => broadcast.
        let src_axis = axis as isize - (out_rank as isize - src_shape.len() as isize);
        if src_axis < 0 {
            continue;
        }
        let src_axis = src_axis as usize;
        let src_dim = src_shape[src_axis];
        if src_dim == out_shape[axis] {
            eff[axis] = src_strides[src_axis];
        } else if src_dim == 1 {
            eff[axis] = 0;
        } else {
            return Err(EpError::Ir(
                onnx_runtime_ir::IrError::BroadcastIncompatible {
                    a: src_shape.to_vec(),
                    b: out_shape.to_vec(),
                },
            ));
        }
    }
    let n = numel(out_shape);
    if n == 0 {
        return Ok(());
    }
    let mut idx = vec![0usize; out_rank];
    let mut flat = 0usize;
    loop {
        let mut src_off = 0i64;
        for (e, &i) in eff.iter().zip(&idx) {
            src_off += e * i as i64;
        }
        f(flat, src[src_off as usize]);
        flat += 1;
        if !next_index(out_shape, &mut idx) {
            break;
        }
    }
    Ok(())
}

impl Kernel for AddKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Add", inputs, outputs, 2, 2, 1)?;
        let a = to_dense_f32(&inputs[0])?;
        let b = to_dense_f32(&inputs[1])?;
        let a_shape = inputs[0].shape;
        let b_shape = inputs[1].shape;
        let out_shape = outputs[0].shape.to_vec();
        let mut out = vec![0.0f32; numel(&out_shape)];
        broadcast_apply(&a, a_shape, &out_shape, |i, v| out[i] = v)?;
        broadcast_apply(&b, b_shape, &out_shape, |i, v| out[i] += v)?;
        write_dense_f32(&mut outputs[0], &out)
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
    fn add_same_shape() {
        let a = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let b = Owned::f32(&[2, 2], &[10., 20., 30., 40.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        AddKernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![11., 22., 33., 44.]);
    }

    #[test]
    fn add_broadcasts_row_vector() {
        // [2,3] + [3] -> [2,3]
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3], &[10., 20., 30.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        AddKernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![11., 22., 33., 14., 25., 36.]);
    }

    #[test]
    fn add_broadcasts_column_vector() {
        // [2,3] + [2,1] -> [2,3]
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[2, 1], &[10., 20.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        AddKernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![11., 12., 13., 24., 25., 26.]);
    }
}
