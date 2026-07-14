//! `com.microsoft::FusedGemm`: the optimizer's fusion of `MatMul(A, B)`
//! followed by a broadcasting `Add(_, bias)` and an elementwise `Relu` into a
//! single node (`docs/ORT2.md` §18.2).
//!
//! `Y = Relu(MatMul(A, B) + bias)`, where the matmul follows full numpy
//! semantics (batched, broadcast leading dims, 1-D operand promotion), `bias`
//! is numpy-broadcast onto the matmul result, and `Relu` is elementwise
//! `max(0, x)` which does not change the shape. Like [`FusedMatMulBias`], this
//! is a pure convenience fusion that produces exactly the same values as the
//! three ops it replaces, reusing the shared
//! [`matmul_dense`](super::matmul::matmul_dense) GEMM, the shared
//! [`broadcast_apply`](super::add::broadcast_apply) bias add, and the shared
//! [`relu_in_place`](super::relu::relu_in_place) activation so there is a single
//! source of truth for every stage of the computation.
//!
//! [`FusedMatMulBias`]: super::fused_matmul_bias::FusedMatMulBiasKernel

use onnx_runtime_ep_api::{Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::add::broadcast_apply;
use super::matmul::matmul_dense;
use super::relu::relu_in_place;
use super::{check_arity, to_dense_f32, write_dense_f32};

/// Stateless f32 `Relu(MatMul(A, B) + bias)` kernel.
pub struct FusedGemmKernel;

/// Factory for [`FusedGemmKernel`] (no attributes).
pub struct FusedGemmFactory;

impl KernelFactory for FusedGemmFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(FusedGemmKernel))
    }
}

impl Kernel for FusedGemmKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("FusedGemm", inputs, outputs, 3, 3, 1)?;
        // MatMul(A, B) into a dense buffer laid out over the output shape.
        let mut out = matmul_dense(&inputs[0], &inputs[1])?;
        // Broadcast-add the bias in place, matching a standalone `Add`.
        let bias = to_dense_f32(&inputs[2])?;
        let bias_shape = inputs[2].shape;
        let out_shape = outputs[0].shape.to_vec();
        broadcast_apply(&bias, bias_shape, &out_shape, |i, v| out[i] += v)?;
        // Elementwise Relu, matching a standalone `Relu` on the biased result.
        relu_in_place(&mut out);
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
    fn relu_clamps_negative_prebias_sums() {
        // A[2,3] @ B[3,2] = [[58,64],[139,154]]; + bias[2] = [-60, 20]
        // -> [[-2, 84],[79, 174]]; Relu -> [[0, 84],[79, 174]].
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
        let bias = Owned::f32(&[2], &[-60., 20.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        FusedGemmKernel
            .execute(&[a.view(), b.view(), bias.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![0., 84., 79., 174.]);
    }

    #[test]
    fn matches_matmul_then_add_then_relu() {
        // Cross-check against running MatMul, then Add, then Relu separately.
        use crate::kernels::matmul::MatMulKernel;
        use crate::kernels::relu::ReluKernel;
        let a = Owned::f32(&[2, 4], &[1., -2., 3., -4., 5., -6., 7., -8.]);
        let b = Owned::f32(&[4, 3], &[1., -2., 3., -4., 5., -6., 7., -8., 9., -10., 11., -12.]);
        let bias = Owned::f32(&[3], &[0.5, -100.0, 2.0]);

        // Reference: MatMul -> Add -> Relu.
        let mut mm = Owned::zeros_f32(&[2, 3]);
        MatMulKernel
            .execute(&[a.view(), b.view()], &mut [mm.view_mut()])
            .unwrap();
        let mut biased = mm.to_f32();
        for row in 0..2 {
            for col in 0..3 {
                biased[row * 3 + col] += [0.5, -100.0, 2.0][col];
            }
        }
        let biased_owned = Owned::f32(&[2, 3], &biased);
        let mut expect = Owned::zeros_f32(&[2, 3]);
        ReluKernel
            .execute(&[biased_owned.view()], &mut [expect.view_mut()])
            .unwrap();

        let mut out = Owned::zeros_f32(&[2, 3]);
        FusedGemmKernel
            .execute(&[a.view(), b.view(), bias.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), expect.to_f32());
        // Sanity: the reference actually clamped at least one negative to 0.
        assert!(expect.to_f32().contains(&0.0));
    }

    #[test]
    fn batched_matmul_with_bias_and_relu() {
        // A[2,2,2] @ B[2,2] (identity, broadcast) + bias[2] = [-5, 3]; Relu.
        let a = Owned::f32(&[2, 2, 2], &[1., -2., -3., 4., 5., -6., -7., 8.]);
        let b = Owned::f32(&[2, 2], &[1., 0., 0., 1.]); // identity
        let bias = Owned::f32(&[2], &[-5., 3.]);
        let mut out = Owned::zeros_f32(&[2, 2, 2]);
        FusedGemmKernel
            .execute(&[a.view(), b.view(), bias.view()], &mut [out.view_mut()])
            .unwrap();
        // A + [-5, 3] across last axis, then Relu:
        // [1-5, -2+3, -3-5, 4+3, 5-5, -6+3, -7-5, 8+3]
        //   = [-4, 1, -8, 7, 0, -3, -12, 11] -> Relu
        //   = [0, 1, 0, 7, 0, 0, 0, 11].
        assert_eq!(out.to_f32(), vec![0., 1., 0., 7., 0., 0., 0., 11.]);
    }
}
