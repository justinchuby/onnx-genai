//! `com.microsoft::FusedMatMulBias`: the optimizer's fusion of `MatMul(A, B)`
//! followed by a broadcasting `Add(_, bias)` into a single node
//! (`docs/architecture/ORT2.md` §18.2).
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
use super::check_arity;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use super::matmul::neon_gemv_f16_col_parallel;
use super::matmul::{
    MatMulPrepack, matmul_dense_prepacked, matmul_dense_prepacked_into,
    output_is_direct_f32_eligible,
};
#[cfg(any(target_os = "macos", target_os = "ios"))]
use crate::backend::CpuBackend;
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};

/// f32 `MatMul(A, B) + bias` kernel with initializer-only MatMul prepacking.
#[derive(Default)]
pub struct FusedMatMulBiasKernel {
    prepack: MatMulPrepack,
}

/// Factory for [`FusedMatMulBiasKernel`] (no attributes).
pub struct FusedMatMulBiasFactory;

impl KernelFactory for FusedMatMulBiasFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(FusedMatMulBiasKernel::default()))
    }
}

impl Kernel for FusedMatMulBiasKernel {
    fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        self.prepack.set_constant_inputs(constant_inputs);
    }

    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("FusedMatMulBias", inputs, outputs, 3, 3, 1)?;

        // FP16 storage GEMV fast path for FusedMatMulBias:
        // When B is a constant Float16 weight and M=1 (decode), GEMV directly
        // from the f16 mmap'd data, add bias, then narrow to the output dtype.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if CpuBackend::auto_detect() == CpuBackend::Accelerate
            && inputs[1].dtype == onnx_runtime_ir::DataType::Float16
        {
            let a_shape = inputs[0].shape;
            let b_shape = inputs[1].shape;
            if a_shape.len() >= 2 && b_shape.len() == 2 {
                let m = a_shape[a_shape.len() - 2];
                let k = a_shape[a_shape.len() - 1];
                let n = b_shape[1];
                let batch_numel: usize = if a_shape.len() > 2 {
                    a_shape[..a_shape.len() - 2].iter().product()
                } else {
                    1
                };
                if m == 1
                    && batch_numel <= 1
                    && let Some(bt_f16) = self.prepack.transposed_b_f16(&inputs[1], k, n)
                {
                    let a_dense = self.prepack.dense(0, &inputs[0])?;
                    let bias = to_dense_f32_widen("FusedMatMulBias", &inputs[2])?;
                    let mut result = vec![0.0f32; n];
                    neon_gemv_f16_col_parallel(&a_dense, bt_f16, &mut result, k, n);
                    // Add 1-D bias in-place.
                    let bias_shape = inputs[2].shape;
                    if bias_shape.len() == 1 && bias_shape[0] == n {
                        for (o, &b) in result.iter_mut().zip(bias.iter()) {
                            *o += b;
                        }
                    } else {
                        let out_shape = outputs[0].shape.to_vec();
                        broadcast_apply(&bias, bias_shape, &out_shape, |i, v| {
                            result[i] += v;
                        })?;
                    }
                    return write_dense_f32_narrow("FusedMatMulBias", &mut outputs[0], &result);
                }
            }
        }

        // Direct f32 output fast path: when the output is a contiguous Float32
        // CPU tensor that does not alias either matmul input, GEMV writes
        // straight into its backing buffer and bias is added in-place — skipping
        // the intermediate Vec<f32> allocation and write_dense_f32_narrow copy.
        let bias = to_dense_f32_widen("FusedMatMulBias", &inputs[2])?;
        let bias_shape = inputs[2].shape;
        let out_shape = &outputs[0].shape;
        let is_1d_bias = bias_shape.len() == 1
            && !out_shape.is_empty()
            && bias_shape[0] == out_shape[out_shape.len() - 1];

        if is_1d_bias && output_is_direct_f32_eligible(&inputs[0], &inputs[1], &outputs[0]) {
            let out_tensor = &mut outputs[0];
            out_tensor.validate()?;
            let numel = out_tensor.numel();
            let ptr = out_tensor.data_ptr_mut::<f32>();
            // SAFETY: same as MatMulKernel's direct-output path — eligibility
            // check proved contiguous f32 CPU tensor, no alias, and the executor's
            // bounds contract ensures numel slots exist.
            let out_slice = unsafe { std::slice::from_raw_parts_mut(ptr, numel) };
            let written =
                matmul_dense_prepacked_into(&inputs[0], &inputs[1], &self.prepack, out_slice)?;
            // Add 1-D bias in-place.
            let n = bias_shape[0];
            for chunk in out_slice[..written].chunks_exact_mut(n) {
                for (o, &b) in chunk.iter_mut().zip(bias.iter()) {
                    *o += b;
                }
            }
            return Ok(());
        }

        // Fallback: allocate intermediate buffer (non-f32 output, strided, or
        // broadcast bias that needs the generic broadcast_apply path).
        let mut out = matmul_dense_prepacked(&inputs[0], &inputs[1], &self.prepack)?;
        if is_1d_bias {
            let n = bias_shape[0];
            for chunk in out.chunks_exact_mut(n) {
                for (o, &b) in chunk.iter_mut().zip(bias.iter()) {
                    *o += b;
                }
            }
        } else {
            let out_shape_vec = out_shape.to_vec();
            broadcast_apply(&bias, bias_shape, &out_shape_vec, |i, v| out[i] += v)?;
        }
        write_dense_f32_narrow("FusedMatMulBias", &mut outputs[0], &out)
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
    fn fused_matmul_bias_bf16_matches_widened_f32_reference() {
        let a_vals = [1.0f32, 2., 3., 4., 5., 6.];
        let b_vals = [7.0f32, 8., 9., 10., 11., 12.];
        let bias_vals = [10.0f32, 20.];
        let a = Owned::f32(&[2, 3], &a_vals);
        let b = Owned::f32(&[3, 2], &b_vals);
        let bias = Owned::f32(&[2], &bias_vals);
        let mut ref_out = Owned::zeros_f32(&[2, 2]);
        FusedMatMulBiasKernel::default()
            .execute(
                &[a.view(), b.view(), bias.view()],
                &mut [ref_out.view_mut()],
            )
            .unwrap();

        let a = Owned::bf16(&[2, 3], &a_vals);
        let b = Owned::bf16(&[3, 2], &b_vals);
        let bias = Owned::bf16(&[2], &bias_vals);
        let mut bf16_out = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[2, 2]);
        FusedMatMulBiasKernel::default()
            .execute(
                &[a.view(), b.view(), bias.view()],
                &mut [bf16_out.view_mut()],
            )
            .unwrap();

        for (&r, &g) in ref_out
            .to_f32()
            .iter()
            .zip(bf16_out.to_bf16_as_f32().iter())
        {
            assert!(
                (r - g).abs() <= 0.03 * r.abs().max(1.0),
                "fused_matmul_bias bf16 {g} vs f32 {r}"
            );
        }
    }

    #[test]
    fn matmul_plus_row_bias() {
        // A[2,3] @ B[3,2] = [[58,64],[139,154]]; + bias[2] = [10, 20].
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
        let bias = Owned::f32(&[2], &[10., 20.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        FusedMatMulBiasKernel::default()
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
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [mm.view_mut()])
            .unwrap();
        let mut expect = mm.to_f32();
        for row in 0..2 {
            for col in 0..3 {
                expect[row * 3 + col] += [0.5, -1.0, 2.0][col];
            }
        }

        let mut out = Owned::zeros_f32(&[2, 3]);
        FusedMatMulBiasKernel::default()
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
        FusedMatMulBiasKernel::default()
            .execute(&[a.view(), b.view(), bias.view()], &mut [out.view_mut()])
            .unwrap();
        // Identity matmul leaves A; add [100,200] across the last axis.
        assert_eq!(
            out.to_f32(),
            vec![101., 202., 103., 204., 105., 206., 107., 208.]
        );
    }
}
