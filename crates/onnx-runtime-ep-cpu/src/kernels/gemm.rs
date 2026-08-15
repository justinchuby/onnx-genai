//! `Gemm`: general matrix multiply `Y = alpha * A' * B' + beta * C` for floating
//! point tensors (`docs/architecture/ORT2.md` §4.4).
//!
//! `A'`/`B'` are `A`/`B` optionally transposed per `transA`/`transB`. `A` is
//! 2-D `[M,K]` (or `[K,M]` when transposed), `B` is `[K,N]` (or `[N,K]`). The
//! optional bias `C` is unidirectionally broadcast to `[M,N]`.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::add::broadcast_apply;
use super::check_arity;
use super::half_gemm::{self, HalfFormat, MatrixLayout};
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

        let mut out = if let Some(mut half_output) =
            try_half_gemm(&inputs[0], &inputs[1], m, k, n, self.trans_a, self.trans_b)?
        {
            if self.alpha != 1.0 {
                for value in &mut half_output {
                    *value *= self.alpha;
                }
            }
            half_output
        } else {
            let a = to_dense_f32_widen("Gemm", &inputs[0])?;
            let b = to_dense_f32_widen("Gemm", &inputs[1])?;

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

            let mut output = vec![0.0f32; m * n];
            for i in 0..m {
                for p in 0..k {
                    let aip = self.alpha * a_at(i, p);
                    if aip == 0.0 {
                        continue;
                    }
                    let row = &mut output[i * n..i * n + n];
                    for (j, cell) in row.iter_mut().enumerate() {
                        *cell += aip * b_at(p, j);
                    }
                }
            }
            output
        };

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

#[allow(clippy::too_many_arguments)]
fn try_half_gemm(
    a: &TensorView,
    b: &TensorView,
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) -> Result<Option<Vec<f32>>> {
    let format = match (a.dtype, b.dtype) {
        (DataType::Float16, DataType::Float16) => HalfFormat::F16,
        (DataType::BFloat16, DataType::BFloat16) => HalfFormat::Bf16,
        _ => return Ok(None),
    };
    if !a.is_contiguous() || !b.is_contiguous() {
        return Ok(None);
    }
    a.validate()?;
    b.validate()?;
    let mut output = vec![0.0; m * n];
    if output.is_empty() || k == 0 {
        return Ok(Some(output));
    }

    // SAFETY: validated contiguous Float16/BFloat16 views contain `numel`
    // two-byte elements, and both half types are transparent over `u16`.
    let a_bits = unsafe { std::slice::from_raw_parts(a.data_ptr::<u16>(), a.numel()) };
    let b_bits = unsafe { std::slice::from_raw_parts(b.data_ptr::<u16>(), b.numel()) };
    let a_layout = if trans_a {
        MatrixLayout::transposed(m)
    } else {
        MatrixLayout::row_major(k)
    };
    let b_layout = if trans_b {
        MatrixLayout::transposed(k)
    } else {
        MatrixLayout::row_major(n)
    };
    half_gemm::gemm(
        format,
        a_bits,
        a_layout,
        b_bits,
        b_layout,
        &mut output,
        m,
        k,
        n,
    );
    Ok(Some(output))
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

    #[test]
    fn gemm_half_dispatch_supports_transpose_bias_and_determinism() {
        let (m, k, n) = (5usize, 7usize, 3usize);
        let logical_a: Vec<f32> = (0..m * k)
            .map(|index| ((index as f32 * 0.17).sin()) * 0.5)
            .collect();
        let logical_b: Vec<f32> = (0..k * n)
            .map(|index| ((index as f32 * 0.11 + 0.3).cos()) * 0.5)
            .collect();
        let mut stored_a = vec![0.0; k * m];
        let mut stored_b = vec![0.0; n * k];
        for row in 0..m {
            for depth in 0..k {
                stored_a[depth * m + row] = logical_a[row * k + depth];
            }
        }
        for depth in 0..k {
            for column in 0..n {
                stored_b[column * k + depth] = logical_b[depth * n + column];
            }
        }
        let bias_values = [0.25f32, -0.5, 0.75];
        let kernel = GemmKernel {
            alpha: 0.75,
            beta: -0.5,
            trans_a: true,
            trans_b: true,
        };

        for dtype in [DataType::Float16, DataType::BFloat16] {
            let round = |value: f32| match dtype {
                DataType::Float16 => half::f16::from_f32(value).to_f32(),
                DataType::BFloat16 => half::bf16::from_f32(value).to_f32(),
                _ => unreachable!(),
            };
            let a = match dtype {
                DataType::Float16 => Owned::f16(&[k, m], &stored_a),
                DataType::BFloat16 => Owned::bf16(&[k, m], &stored_a),
                _ => unreachable!(),
            };
            let b = match dtype {
                DataType::Float16 => Owned::f16(&[n, k], &stored_b),
                DataType::BFloat16 => Owned::bf16(&[n, k], &stored_b),
                _ => unreachable!(),
            };
            let bias = match dtype {
                DataType::Float16 => Owned::f16(&[n], &bias_values),
                DataType::BFloat16 => Owned::bf16(&[n], &bias_values),
                _ => unreachable!(),
            };
            assert!(
                try_half_gemm(&a.view(), &b.view(), m, k, n, true, true)
                    .unwrap()
                    .is_some(),
                "{dtype:?} should select the dedicated half GEMM"
            );

            let logical_a: Vec<f32> = logical_a.iter().copied().map(round).collect();
            let logical_b: Vec<f32> = logical_b.iter().copied().map(round).collect();
            let bias_wide: Vec<f32> = bias_values.iter().copied().map(round).collect();
            let mut expected = vec![0.0; m * n];
            for row in 0..m {
                for column in 0..n {
                    for depth in 0..k {
                        expected[row * n + column] +=
                            logical_a[row * k + depth] * logical_b[depth * n + column];
                    }
                    expected[row * n + column] =
                        round(0.75 * expected[row * n + column] - 0.5 * bias_wide[column]);
                }
            }

            let mut first = Owned::zeros(dtype, &[m, n]);
            let mut second = Owned::zeros(dtype, &[m, n]);
            kernel
                .execute(&[a.view(), b.view(), bias.view()], &mut [first.view_mut()])
                .unwrap();
            kernel
                .execute(&[a.view(), b.view(), bias.view()], &mut [second.view_mut()])
                .unwrap();
            let first = match dtype {
                DataType::Float16 => first.to_f16_as_f32(),
                DataType::BFloat16 => first.to_bf16_as_f32(),
                _ => unreachable!(),
            };
            let second = match dtype {
                DataType::Float16 => second.to_f16_as_f32(),
                DataType::BFloat16 => second.to_bf16_as_f32(),
                _ => unreachable!(),
            };
            assert_eq!(first, second, "{dtype:?} Gemm was not deterministic");
            let tolerance = match dtype {
                DataType::Float16 => 2e-3,
                DataType::BFloat16 => 2e-2,
                _ => unreachable!(),
            };
            let max_error = first
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_error <= tolerance,
                "{dtype:?} transposed Gemm max error {max_error} exceeds {tolerance}"
            );
        }
    }
}
