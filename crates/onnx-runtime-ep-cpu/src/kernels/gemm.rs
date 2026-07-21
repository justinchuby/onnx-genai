//! `Gemm`: general matrix multiply `Y = alpha * A' * B' + beta * C` for f32
//! (`docs/ORT2.md` §4.4).
//!
//! `A'`/`B'` are `A`/`B` optionally transposed per `transA`/`transB`. `A` is
//! 2-D `[M,K]` (or `[K,M]` when transposed), `B` is `[K,N]` (or `[N,K]`). The
//! optional bias `C` is unidirectionally broadcast to `[M,N]`.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::add::broadcast_apply;
use super::check_arity;
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};

/// f32 Gemm kernel carrying its scalar/transpose attributes.
pub struct GemmKernel {
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
}

/// Factory reading `alpha`/`beta` (default 1.0) and `transA`/`transB`
/// (default 0).
pub struct GemmFactory;

impl KernelFactory for GemmFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let alpha = node.attr("alpha").and_then(|a| a.as_float()).unwrap_or(1.0);
        let beta = node.attr("beta").and_then(|a| a.as_float()).unwrap_or(1.0);
        let trans_a = node.attr("transA").and_then(|a| a.as_int()).unwrap_or(0) != 0;
        let trans_b = node.attr("transB").and_then(|a| a.as_int()).unwrap_or(0) != 0;
        Ok(Box::new(GemmKernel {
            alpha,
            beta,
            trans_a,
            trans_b,
        }))
    }
}

impl Kernel for GemmKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Gemm", inputs, outputs, 2, 3, 1)?;
        let a = to_dense_f32_widen("Gemm", &inputs[0])?;
        let b = to_dense_f32_widen("Gemm", &inputs[1])?;
        let a_shape = inputs[0].shape;
        let b_shape = inputs[1].shape;
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(EpError::KernelFailed(format!(
                "Gemm: A and B must be 2-D, got {a_shape:?} and {b_shape:?}"
            )));
        }

        // Logical M,K,N after honoring the transpose flags.
        let (m, ka) = if self.trans_a {
            (a_shape[1], a_shape[0])
        } else {
            (a_shape[0], a_shape[1])
        };
        let (kb, n) = if self.trans_b {
            (b_shape[1], b_shape[0])
        } else {
            (b_shape[0], b_shape[1])
        };
        if ka != kb {
            return Err(EpError::KernelFailed(format!(
                "Gemm: inner dims disagree ({ka} vs {kb})"
            )));
        }
        let k = ka;
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            // Conventional roofline estimate: one multiply and one add per GEMM
            // contraction, plus scale-and-add for each optional bias output.
            let mut flops = (m as u64)
                .saturating_mul(n as u64)
                .saturating_mul(k as u64)
                .saturating_mul(2);
            if inputs.len() == 3 && self.beta != 0.0 {
                flops = flops.saturating_add((m as u64).saturating_mul(n as u64).saturating_mul(2));
            }
            flops
        });

        // Accessors into the row-major dense buffers, applying transposition.
        let a_at = |i: usize, p: usize| -> f32 {
            if self.trans_a {
                a[p * m + i] // A stored [K,M]
            } else {
                a[i * k + p] // A stored [M,K]
            }
        };
        let b_at = |p: usize, j: usize| -> f32 {
            if self.trans_b {
                b[j * k + p] // B stored [N,K]
            } else {
                b[p * n + j] // B stored [K,N]
            }
        };

        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                let aip = self.alpha * a_at(i, p);
                if aip == 0.0 {
                    continue;
                }
                let row = &mut out[i * n..i * n + n];
                for (j, cell) in row.iter_mut().enumerate() {
                    *cell += aip * b_at(p, j);
                }
            }
        }

        // Optional bias: Y += beta * C, with C broadcast to [M,N].
        if inputs.len() == 3 && self.beta != 0.0 {
            let c = to_dense_f32_widen("Gemm", &inputs[2])?;
            let beta = self.beta;
            broadcast_apply(&c, inputs[2].shape, &[m, n], |idx, v| out[idx] += beta * v)?;
        }

        write_dense_f32_narrow("Gemm", &mut outputs[0], &out)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    #[allow(clippy::too_many_arguments)]
    fn gemm(
        alpha: f32,
        beta: f32,
        ta: bool,
        tb: bool,
        a: &Owned,
        b: &Owned,
        c: Option<&Owned>,
        out: &mut Owned,
    ) {
        let k = GemmKernel {
            alpha,
            beta,
            trans_a: ta,
            trans_b: tb,
        };
        let mut ins = vec![a.view(), b.view()];
        if let Some(c) = c {
            ins.push(c.view());
        }
        k.execute(&ins, &mut [out.view_mut()]).unwrap();
    }

    #[test]
    fn plain_gemm_no_transpose_no_c() {
        // A[2,3] @ B[3,2] = [[58,64],[139,154]]
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        gemm(1.0, 1.0, false, false, &a, &b, None, &mut out);
        assert_eq!(out.to_f32(), vec![58., 64., 139., 154.]);
    }

    #[test]
    fn gemm_with_bias_and_alpha_beta() {
        // alpha=2, beta=3, C broadcast row [10,20].
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
        let c = Owned::f32(&[2], &[10., 20.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        gemm(2.0, 3.0, false, false, &a, &b, Some(&c), &mut out);
        // 2*[[58,64],[139,154]] + 3*[[10,20],[10,20]]
        // = [[116+30, 128+60],[278+30, 308+60]] = [[146,188],[308,368]]
        assert_eq!(out.to_f32(), vec![146., 188., 308., 368.]);
    }

    #[test]
    fn gemm_trans_a() {
        // A stored [3,2] = A_logical^T, so transA gives A_logical [2,3].
        // A_logical = [[1,2,3],[4,5,6]] means stored [K,M]=[3,2] = [[1,4],[2,5],[3,6]]
        let a = Owned::f32(&[3, 2], &[1., 4., 2., 5., 3., 6.]);
        let b = Owned::f32(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        gemm(1.0, 1.0, true, false, &a, &b, None, &mut out);
        assert_eq!(out.to_f32(), vec![58., 64., 139., 154.]);
    }

    #[test]
    fn gemm_trans_b() {
        // B stored [2,3] = B_logical^T; transB gives B_logical [3,2].
        // B_logical = [[7,8],[9,10],[11,12]] -> stored [N,K]=[2,3]=[[7,9,11],[8,10,12]]
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[2, 3], &[7., 9., 11., 8., 10., 12.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        gemm(1.0, 1.0, false, true, &a, &b, None, &mut out);
        assert_eq!(out.to_f32(), vec![58., 64., 139., 154.]);
    }

    #[test]
    fn gemm_trans_a_and_b() {
        let a = Owned::f32(&[3, 2], &[1., 4., 2., 5., 3., 6.]); // A^T
        let b = Owned::f32(&[2, 3], &[7., 9., 11., 8., 10., 12.]); // B^T
        let mut out = Owned::zeros_f32(&[2, 2]);
        gemm(1.0, 1.0, true, true, &a, &b, None, &mut out);
        assert_eq!(out.to_f32(), vec![58., 64., 139., 154.]);
    }

    #[test]
    fn gemm_scalar_bias_broadcast() {
        let a = Owned::f32(&[1, 2], &[1., 1.]);
        let b = Owned::f32(&[2, 1], &[2., 3.]);
        let c = Owned::f32(&[], &[100.]); // scalar broadcast
        let mut out = Owned::zeros_f32(&[1, 1]);
        gemm(1.0, 1.0, false, false, &a, &b, Some(&c), &mut out);
        // 1*2 + 1*3 = 5, + 100 = 105
        assert_eq!(out.to_f32(), vec![105.]);
    }

    #[test]
    fn gemm_f16_with_bias() {
        use onnx_runtime_ir::DataType;
        let a = Owned::f16(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f16(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
        let c = Owned::f16(&[2], &[10., 20.]);
        let mut out = Owned::zeros(DataType::Float16, &[2, 2]);
        GemmKernel {
            alpha: 1.0,
            beta: 1.0,
            trans_a: false,
            trans_b: false,
        }
        .execute(&[a.view(), b.view(), c.view()], &mut [out.view_mut()])
        .unwrap();
        // [[58,64],[139,154]] + [[10,20],[10,20]]
        assert_eq!(out.to_f16_as_f32(), vec![68., 84., 149., 174.]);
    }

    #[test]
    fn gemm_bf16_plain() {
        use onnx_runtime_ir::DataType;
        let a = Owned::bf16(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::bf16(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
        let mut out = Owned::zeros(DataType::BFloat16, &[2, 2]);
        GemmKernel {
            alpha: 1.0,
            beta: 1.0,
            trans_a: false,
            trans_b: false,
        }
        .execute(&[a.view(), b.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_bf16_as_f32(), vec![58., 64., 139., 154.]);
    }
}
