//! `Gemm`: general matrix multiply `Y = alpha * A' * B' + beta * C` for floating
//! point tensors (`docs/architecture/ORT2.md` §4.4).
//!
//! `A'`/`B'` are `A`/`B` optionally transposed per `transA`/`transB`. `A` is
//! 2-D `[M,K]` (or `[K,M]` when transposed), `B` is `[K,N]` (or `[N,K]`). The
//! optional bias `C` is unidirectionally broadcast to `[M,N]`.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use std::borrow::Cow;

use super::add::broadcast_apply;
use super::check_arity;
use super::half_gemm::{self, HalfFormat, MatrixLayout};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::half_gemv;
use super::matmul::{self, MatMulPrepack};
use super::weight_transpose::transpose_row_major;
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};

/// f32 Gemm kernel carrying its scalar/transpose attributes.
pub struct GemmKernel {
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
    /// Densification and weight-transpose memos for constant (initializer)
    /// operands, shared with `MatMul` so both ops cache a given weight once.
    prepack: MatMulPrepack,
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
            prepack: MatMulPrepack::default(),
        }))
    }
}

impl GemmKernel {
    /// The f16 decode GEMV and prefill packed-SGEMM routes, in that order.
    ///
    /// Returns `None` whenever neither applies, leaving the portable blocked
    /// half GEMM to serve the call exactly as before.
    fn try_half_fast_path(
        &self,
        a: &TensorView,
        b: &TensorView,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Option<Vec<f32>>> {
        if a.dtype != DataType::Float16 || b.dtype != DataType::Float16 {
            return Ok(None);
        }

        // Decode: read B in place as f16 rather than widening K*N floats for a
        // single row. Memory-bound, and measured at parity with ORT in MatMul.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if m == 1
            && b.is_contiguous()
            && b.numel() == k.saturating_mul(n)
            && half_gemv::simd_available()
        {
            b.validate()?;
            let a_dense = self.prepack.dense(0, a)?;
            if a_dense.len() == k {
                // SAFETY: `b` was just validated as a contiguous Float16 view
                // whose element count equals `k * n`. `f16` is transparent over
                // `u16`, so reading its storage as raw bit patterns is sound,
                // and the view outlives this call.
                let b_bits = unsafe { std::slice::from_raw_parts(b.data_ptr::<u16>(), k * n) };
                let mut result = vec![0.0f32; n];
                half_gemv::gemv_f16_kn(&a_dense, b_bits, &mut result, k, n);
                return Ok(Some(result));
            }
        }

        #[cfg(feature = "mlas")]
        {
            matmul::try_packed_half_prefill(
                &self.prepack,
                crate::backend::CpuBackend::auto_detect(),
                a,
                b,
                m,
                k,
                n,
            )
        }
        #[cfg(not(feature = "mlas"))]
        {
            // Off x86 the decode GEMV above is compiled out too, so without
            // MLAS neither fast path exists and the dimensions go unread.
            let _ = (m, k, n);
            Ok(None)
        }
    }
}

impl Kernel for GemmKernel {
    fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        self.prepack.set_constant_inputs(constant_inputs);
    }

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

        // f16 fast paths, shared with `MatMul`. Only the untransposed case
        // qualifies: both read B in its stored [K, N] order, and materialising
        // a transpose first would give back what they save.
        //
        // Without these, an f16 `Gemm` falls into the portable blocked half
        // GEMM, which is the worst dense region measured anywhere in this EP:
        // at K=N=3584 it was 6.57x slower than ORT at M=1/1 thread and 46.67x
        // at M=1/8 threads, because that path never scales with thread count on
        // a single-row problem (10.07 ms at 1 thread, 10.26 ms at 8).
        let half_fast_path: Option<Vec<f32>> = if self.trans_a || self.trans_b {
            None
        } else {
            self.try_half_fast_path(&inputs[0], &inputs[1], m, k, n)?
        };

        let mut out = if let Some(mut half_output) = half_fast_path {
            if self.alpha != 1.0 {
                for value in &mut half_output {
                    *value *= self.alpha;
                }
            }
            half_output
        } else if let Some(mut half_output) =
            try_half_gemm(&inputs[0], &inputs[1], m, k, n, self.trans_a, self.trans_b)?
        {
            if self.alpha != 1.0 {
                for value in &mut half_output {
                    *value *= self.alpha;
                }
            }
            half_output
        } else {
            let a = self.prepack.dense(0, &inputs[0])?;
            let b = self.prepack.dense(1, &inputs[1])?;

            // The shared GEMM consumes row-major `A[M,K]` and `B[K,N]`, so an
            // operand stored transposed is materialized in that layout first.
            // The transposes are O(M*K) and O(K*N) against an O(M*N*K) product,
            // and a constant `B` is transposed once per session and memoized.
            let a_rm: Cow<'_, [f32]> = if self.trans_a {
                Cow::Owned(transpose_row_major(&a, k, m))
            } else {
                a
            };
            let b_rm: Cow<'_, [f32]> = if self.trans_b {
                match self.prepack.transposed_b(&b, n, k) {
                    Some(cached) => Cow::Borrowed(cached),
                    None => Cow::Owned(transpose_row_major(&b, n, k)),
                }
            } else {
                b
            };

            let mut output = vec![0.0f32; m * n];
            if m != 0 && n != 0 && k != 0 {
                matmul::gemm(&a_rm, &b_rm, &mut output, m, k, n)?;
            }
            if self.alpha != 1.0 {
                for value in &mut output {
                    *value *= self.alpha;
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
            prepack: MatMulPrepack::default(),
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
            prepack: MatMulPrepack::default(),
        }
        .execute(&[a.view(), b.view(), c.view()], &mut [out.view_mut()])
        .unwrap();
        // [[58,64],[139,154]] + [[10,20],[10,20]]
        assert_eq!(out.to_f16_as_f32(), vec![68., 84., 149., 174.]);
    }

    /// Builds a `Gemm` whose B is a constant weight, runs it, and returns the
    /// output together with whether the packed f16 route served the call.
    fn run_half_gemm(
        m: usize,
        k: usize,
        n: usize,
        a_data: &[f32],
        b_data: &[f32],
        trans_b: bool,
    ) -> (Vec<f32>, bool) {
        let a = Owned::f16(&[m, k], a_data);
        let b = if trans_b {
            Owned::f16(&[n, k], b_data)
        } else {
            Owned::f16(&[k, n], b_data)
        };
        let mut out = Owned::zeros_f32(&[m, n]);
        let mut kernel = GemmKernel {
            alpha: 1.0,
            beta: 1.0,
            trans_a: false,
            trans_b,
            prepack: MatMulPrepack::default(),
        };
        kernel.set_constant_inputs(&[false, true]);
        kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        let packed = kernel.prepack.half_pack_is_built();
        (out.to_f32(), packed)
    }

    fn reference_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                for j in 0..n {
                    c[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }
        c
    }

    /// An f16 `Gemm` prefill with a constant weight takes the packed SGEMM
    /// route and agrees with the reference. `K` and `N` are odd and `N` is
    /// under a SIMD step, so the packed kernel's tails carry the result.
    ///
    /// The `packed` assertion is what makes this a performance test as well as
    /// a numerical one: without it the test would keep passing if the call
    /// silently fell back to the blocked path that measured 2.46x slower than
    /// ORT at one thread and 7.16x at eight.
    #[cfg(feature = "mlas")]
    #[test]
    fn gemm_f16_prefill_takes_the_packed_route() {
        let (m, k, n) = (6usize, 37usize, 5usize);
        // Multiples of 1/16 are exact in f16, so a mismatch is arithmetic.
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 % 23) as f32 - 11.0) / 16.0)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i * 13 % 31) as f32 - 15.0) / 16.0)
            .collect();
        let (out, packed) = run_half_gemm(m, k, n, &a, &b, false);
        // `execute` resolves its backend by auto-detection, and only MLAS has a
        // pack to build; elsewhere the blocked path serves the call correctly.
        assert_eq!(
            packed,
            crate::backend::CpuBackend::auto_detect() == crate::backend::CpuBackend::Mlas,
            "a constant f16 B at M>1 must be widened and packed once on MLAS"
        );
        for (actual, want) in out.iter().zip(reference_matmul(&a, &b, m, k, n).iter()) {
            assert!(
                (actual - want).abs() <= 1e-3,
                "f16 Gemm prefill disagreed: got {actual}, want {want}"
            );
        }
    }

    /// `M = 1` takes the in-place f16 GEMV instead: it reads B as 16-bit rather
    /// than widening `K*N` floats for a single row, and must not build a pack.
    #[test]
    fn gemm_f16_decode_takes_the_gemv_not_the_pack() {
        let (k, n) = (37usize, 5usize);
        let a: Vec<f32> = (0..k)
            .map(|i| ((i * 7 % 23) as f32 - 11.0) / 16.0)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i * 13 % 31) as f32 - 15.0) / 16.0)
            .collect();
        let (out, packed) = run_half_gemm(1, k, n, &a, &b, false);
        assert!(
            !packed,
            "M=1 must stay on the GEMV rather than building a pack"
        );
        for (actual, want) in out.iter().zip(reference_matmul(&a, &b, 1, k, n).iter()) {
            assert!(
                (actual - want).abs() <= 1e-3,
                "f16 Gemm decode disagreed: got {actual}, want {want}"
            );
        }
    }

    /// A transposed B keeps the blocked half GEMM. Both fast paths read B in
    /// its stored [K, N] order, so materialising a transpose first would give
    /// back exactly what they save.
    #[test]
    fn gemm_f16_transposed_b_keeps_the_blocked_path() {
        let (m, k, n) = (4usize, 6usize, 3usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 9) as f32 - 4.0) / 8.0).collect();
        // B^T is [n, k]; build the [k, n] logical view for the reference.
        let bt: Vec<f32> = (0..n * k).map(|i| ((i % 7) as f32 - 3.0) / 8.0).collect();
        let mut b = vec![0.0f32; k * n];
        for row in 0..n {
            for col in 0..k {
                b[col * n + row] = bt[row * k + col];
            }
        }
        let (out, packed) = run_half_gemm(m, k, n, &a, &bt, true);
        assert!(!packed, "a transposed B must not enter the packed route");
        for (actual, want) in out.iter().zip(reference_matmul(&a, &b, m, k, n).iter()) {
            assert!(
                (actual - want).abs() <= 1e-3,
                "transposed f16 Gemm disagreed: got {actual}, want {want}"
            );
        }
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
            prepack: MatMulPrepack::default(),
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
            prepack: MatMulPrepack::default(),
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

    /// Reference `Y = alpha * A' * B' + beta * C`, written the obvious way. The
    /// production path now reshapes both operands and calls the shared GEMM
    /// backend, so this is what proves the reshaping is right for every
    /// transpose combination.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        alpha: f32,
        beta: f32,
        ta: bool,
        tb: bool,
        a: &[f32],
        b: &[f32],
        bias: Option<&[f32]>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    let av = if ta { a[p * m + i] } else { a[i * k + p] };
                    let bv = if tb { b[j * k + p] } else { b[p * n + j] };
                    acc += av * bv;
                }
                out[i * n + j] = alpha * acc;
                if let Some(bias) = bias {
                    out[i * n + j] += beta * bias[j];
                }
            }
        }
        out
    }

    fn pseudo(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|i| (i as f32 * 0.7 + seed).sin() * 1.5)
            .collect()
    }

    /// The reshape-and-dispatch rewrite has to hold for every `transA`/`transB`
    /// combination, for `alpha != 1` (which used to be folded into `A` and is
    /// now applied to the product), for a broadcast bias, and for shapes that
    /// are not multiples of any register tile -- `K = 67` and `N = 33` land in
    /// the microkernel's edge handling, which is exactly where a layout mistake
    /// hides.
    #[test]
    fn gemm_matches_reference_across_transposes_and_odd_shapes() {
        for &(m, k, n) in &[
            (1usize, 67usize, 33usize),
            (5, 64, 32),
            (3, 1, 7),
            (17, 129, 65),
        ] {
            for &ta in &[false, true] {
                for &tb in &[false, true] {
                    for &alpha in &[1.0f32, 0.5] {
                        let a_values = pseudo(m * k, 0.1);
                        let b_values = pseudo(k * n, 0.3);
                        let bias = pseudo(n, 0.9);
                        let a_shape = if ta { [k, m] } else { [m, k] };
                        let b_shape = if tb { [n, k] } else { [k, n] };
                        let a = Owned::f32(&a_shape, &a_values);
                        let b = Owned::f32(&b_shape, &b_values);
                        let c = Owned::f32(&[n], &bias);
                        let mut out = Owned::zeros_f32(&[m, n]);
                        gemm(alpha, 1.0, ta, tb, &a, &b, Some(&c), &mut out);

                        let expected = reference(
                            alpha,
                            1.0,
                            ta,
                            tb,
                            &a_values,
                            &b_values,
                            Some(&bias),
                            m,
                            k,
                            n,
                        );
                        let got = out.to_f32();
                        let max_error = got
                            .iter()
                            .zip(&expected)
                            .map(|(actual, want)| (actual - want).abs())
                            .fold(0.0f32, f32::max);
                        assert!(
                            max_error <= 1e-4,
                            "m={m} k={k} n={n} transA={ta} transB={tb} alpha={alpha}: \
                             max error {max_error} against the reference"
                        );
                    }
                }
            }
        }
    }

    /// Repeated execution of one kernel must keep returning the same answer.
    ///
    /// A constant `B` with `transB = 1` is transposed once and memoized in the
    /// process-global weight-transpose cache, keyed by source address and shape.
    /// A stale or mis-keyed entry would show up here as a second call that
    /// disagrees with the first -- the failure mode #845 was filed for.
    #[test]
    fn gemm_constant_transposed_weight_is_stable_across_calls() {
        let (m, k, n) = (2usize, 48usize, 24usize);
        let a_values = pseudo(m * k, 0.2);
        let b_values = pseudo(n * k, 0.6);
        let a = Owned::f32(&[m, k], &a_values);
        let b = Owned::f32(&[n, k], &b_values);

        let mut kernel = GemmKernel {
            alpha: 1.0,
            beta: 1.0,
            trans_a: false,
            trans_b: true,
            prepack: MatMulPrepack::default(),
        };
        kernel.set_constant_inputs(&[false, true]);

        let expected = reference(1.0, 1.0, false, true, &a_values, &b_values, None, m, k, n);
        for round in 0..3 {
            let mut out = Owned::zeros_f32(&[m, n]);
            kernel
                .execute(&[a.view(), b.view()], &mut [out.view_mut()])
                .unwrap();
            let got = out.to_f32();
            let max_error = got
                .iter()
                .zip(&expected)
                .map(|(actual, want)| (actual - want).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_error <= 1e-4,
                "round {round}: cached transposed weight gave max error {max_error}"
            );
        }
    }

    /// Structural proof that `Gemm` delegates to the shared backend, with no
    /// timing involved.
    ///
    /// With `alpha = 1`, no bias and no transposes, `Gemm` is exactly
    /// `matmul::gemm` plus buffer plumbing, so its output must be **bit**
    /// identical to calling that backend directly. A scalar re-implementation
    /// would have to reproduce the blocked, register-tiled, rayon-partitioned
    /// summation order of `gemm_generic`/`x86_sgemm` float-for-float to pass
    /// this, which is precisely the property the removed triple loop lacked:
    /// it accumulated row-at-a-time and rounded differently.
    ///
    /// This runs in CI on every platform, unlike the timing guard below.
    #[test]
    fn gemm_output_is_bit_identical_to_the_shared_backend() {
        let (m, k, n) = (7usize, 96usize, 48usize);
        let a_values = pseudo(m * k, 0.4);
        let b_values = pseudo(k * n, 0.8);

        let mut expected = vec![0.0f32; m * n];
        matmul::gemm(&a_values, &b_values, &mut expected, m, k, n).unwrap();

        let a = Owned::f32(&[m, k], &a_values);
        let b = Owned::f32(&[k, n], &b_values);
        let mut out = Owned::zeros_f32(&[m, n]);
        gemm(1.0, 1.0, false, false, &a, &b, None, &mut out);

        assert_eq!(
            out.to_f32(),
            expected,
            "Gemm is no longer computing its f32 product with matmul::gemm"
        );
    }

    /// Performance regression guard, not a benchmark.
    ///
    /// The defect this file's rewrite fixes was structural: `Gemm` computed its
    /// product with a scalar triple loop while every other f32 matrix multiply
    /// in the EP used the shared, SIMD, multi-threaded backend, which measured
    /// 30x-1053x slower than ORT depending on shape. Reintroducing a scalar loop
    /// would not fail any correctness test, so the guard is a wall-clock ratio
    /// against `matmul::gemm` on the same data: `Gemm` may pay for reshaping its
    /// operands, but it may not be in a different performance class.
    ///
    /// Ignored by default -- timing on a shared CI runner is not a gate -- and
    /// run explicitly with `--ignored`.
    #[test]
    #[ignore = "timing-sensitive; run explicitly with --ignored"]
    fn gemm_stays_within_reach_of_the_shared_backend() {
        use std::time::Instant;

        let (m, k, n) = (64usize, 512usize, 512usize);
        let a_values = pseudo(m * k, 0.2);
        let b_values = pseudo(k * n, 0.6);
        let a = Owned::f32(&[m, k], &a_values);
        let b = Owned::f32(&[k, n], &b_values);

        let mut direct = vec![0.0f32; m * n];
        for _ in 0..3 {
            matmul::gemm(&a_values, &b_values, &mut direct, m, k, n).unwrap();
        }
        let started = Instant::now();
        for _ in 0..10 {
            matmul::gemm(&a_values, &b_values, &mut direct, m, k, n).unwrap();
        }
        let backend = started.elapsed();

        let mut out = Owned::zeros_f32(&[m, n]);
        for _ in 0..3 {
            gemm(1.0, 1.0, false, false, &a, &b, None, &mut out);
        }
        let started = Instant::now();
        for _ in 0..10 {
            gemm(1.0, 1.0, false, false, &a, &b, None, &mut out);
        }
        let kernel = started.elapsed();

        // 8x is deliberately loose: this catches "fell back to a scalar loop"
        // (two to three orders of magnitude) without failing on a noisy runner.
        assert!(
            kernel.as_secs_f64() <= backend.as_secs_f64() * 8.0 + 1e-3,
            "Gemm took {kernel:?} against {backend:?} for the same product; \
             it is no longer using the shared GEMM backend"
        );
    }
}
