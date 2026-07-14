//! `com.microsoft::FusedMatMulBias`: the optimizer's fusion of `MatMul(A, B)`
//! followed by a broadcasting `Add(_, bias)` into a single node
//! (`docs/ORT2.md` §18.2).
//!
//! `Y = MatMul(A, B) + bias`, where the matmul follows full numpy semantics
//! (batched, broadcast leading dims, 1-D operand promotion) and `bias` is
//! numpy-broadcast onto the matmul result. This is a pure convenience fusion:
//! it produces exactly the same values as the two ops it replaces, reusing the
//! shared [`matmul_dense`](super::matmul::matmul_dense) GEMM and the shared
//! [`broadcast_apply`](super::add::broadcast_apply) so there is a single source
//! of truth for both halves of the computation.

use onnx_runtime_ep_api::{Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::add::broadcast_apply;
use super::matmul::matmul_dense;
use super::{check_arity, to_dense_f32, write_dense_f32};

/// Stateless f32 `MatMul(A, B) + bias` kernel.
pub struct FusedMatMulBiasKernel;

/// Factory for [`FusedMatMulBiasKernel`] (no attributes).
pub struct FusedMatMulBiasFactory;

impl KernelFactory for FusedMatMulBiasFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(FusedMatMulBiasKernel))
    }
}

impl Kernel for FusedMatMulBiasKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("FusedMatMulBias", inputs, outputs, 3, 3, 1)?;
        // MatMul(A, B) into a dense buffer laid out over the output shape.
        let mut out = matmul_dense(&inputs[0], &inputs[1])?;
        // Broadcast-add the bias in place, matching a standalone `Add`.
        let bias = to_dense_f32(&inputs[2])?;
        let bias_shape = inputs[2].shape;
        let out_shape = outputs[0].shape.to_vec();
        broadcast_apply(&bias, bias_shape, &out_shape, |i, v| out[i] += v)?;
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
    fn matmul_plus_row_bias() {
        // A[2,3] @ B[3,2] = [[58,64],[139,154]]; + bias[2] = [10, 20].
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
        let bias = Owned::f32(&[2], &[10., 20.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        FusedMatMulBiasKernel
            .execute(&[a.view(), b.view(), bias.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![68., 84., 149., 174.]);
    }

    #[test]
    fn matches_matmul_then_add() {
        // Cross-check against running MatMul then Add separately.
        use crate::kernels::matmul::MatMulKernel;
        let a = Owned::f32(&[2, 4], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let b = Owned::f32(&[4, 3], &(1..=12).map(|x| x as f32).collect::<Vec<_>>());
        let bias = Owned::f32(&[3], &[0.5, -1.0, 2.0]);

        let mut mm = Owned::zeros_f32(&[2, 3]);
        MatMulKernel
            .execute(&[a.view(), b.view()], &mut [mm.view_mut()])
            .unwrap();
        let mut expect = mm.to_f32();
        for row in 0..2 {
            for col in 0..3 {
                expect[row * 3 + col] += [0.5, -1.0, 2.0][col];
            }
        }

        let mut out = Owned::zeros_f32(&[2, 3]);
        FusedMatMulBiasKernel
            .execute(&[a.view(), b.view(), bias.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), expect);
    }

    #[test]
    fn batched_matmul_with_bias() {
        // A[2,2,2] @ B[2,2] (broadcast B) + scalar-ish bias[2].
        let a = Owned::f32(&[2, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let b = Owned::f32(&[2, 2], &[1., 0., 0., 1.]); // identity
        let bias = Owned::f32(&[2], &[100., 200.]);
        let mut out = Owned::zeros_f32(&[2, 2, 2]);
        FusedMatMulBiasKernel
            .execute(&[a.view(), b.view(), bias.view()], &mut [out.view_mut()])
            .unwrap();
        // Identity matmul leaves A; add [100,200] across the last axis.
        assert_eq!(
            out.to_f32(),
            vec![101., 202., 103., 204., 105., 206., 107., 208.]
        );
    }
}
