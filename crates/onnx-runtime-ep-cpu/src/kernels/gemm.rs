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
    ///
    /// `#[inline(never)]` is load-bearing and was **measured**, not assumed.
    /// Carrying a second kernel made this body large enough that inlining it
    /// into [`GemmKernel::execute`] cost the `M = 128` `transB` prefill 12%
    /// (69.7 -> 78.9 ms, median of 10 per-run minima) -- a path this function
    /// declines outright and never touches. It is a once-per-execute dispatch
    /// probe returning an `Option`, so there is nothing to inline it for.
    #[inline(never)]
    fn try_half_fast_path(
        &self,
        a: &TensorView,
        b: &TensorView,
        m: usize,
        k: usize,
        n: usize,
        trans_b: bool,
    ) -> Result<Option<Vec<f32>>> {
        // `bf16` is admitted here as well as `f16`. It was not, and the
        // asymmetry cost the same decode a different kernel depending on op:
        // `MatMul` serves a `bf16` `m == 1` from the same GEMV, while `Gemm`
        // fell into the portable blocked half GEMM.
        let Some(format) = matmul::half_storage_format(a.dtype, b.dtype) else {
            return Ok(None);
        };
        // The only reader of `format` is the decode arm below, which is x86
        // only, so off x86 it is bound and never used -- the same reason
        // `m`, `k` and `n` are discarded on the fallthrough paths.
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let _ = format;

        // Decode: read B in place as f16 rather than widening K*N floats for a
        // single row. Memory-bound, and measured at parity with ORT in MatMul.
        // Both stored orders have a kernel, so `trans_b` picks one rather than
        // disqualifying the call — see the dispatch comment in `execute`.
        //
        // No weight diverts this to the fused GEBP. `MatMul` used to divert at
        // `k * n >= 1M` and `Gemm` never did -- the divergence #1381 recorded.
        // Re-measured through both production kernels the divert is a loss
        // (1.3x-5.0x at 8 threads), so it was retired from `MatMul` rather
        // than copied here, and the two ops now take the same **f16** route at
        // every weight -- `half_decode_route_tests` pins that. `bf16` is a
        // f16 route at every weight -- `half_decode_route_tests` pins that,
        // and pins the `bf16` pair too.
        //
        // The transposed `bf16` asymmetry §15 left open is closed: the `[N, K]`
        // GEMV is instantiated per format from one macro, the way the `[K, N]`
        // stripe already was, so `trans_b` now picks a layout rather than a
        // dtype. Both operators, both stored orders and both 16-bit formats
        // reach the same GEMV backend.
        //
        // Reaching the same *backend* is not the same as reaching the same
        // *kernel*, which is what was still divergent: `trans_b` chose between
        // two kernels differing by up to 2.98x in speed and 9.3x in accuracy,
        // so equivalent math was still priced -- and rounded -- by how the
        // exporter happened to store the weight. `half_decode_gemv_dispatch`
        // closes that: a constant `[K, N]` weight is transposed once into the
        // prepack cache so both stored orders run the same `[N, K]` kernel,
        // bit for bit. Its doc comment carries the measurements.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if m == 1
            && b.is_contiguous()
            && b.numel() == k.saturating_mul(n)
            && half_gemv::simd_available(format)
            && matmul::half_decode_gemv_enabled()
        {
            b.validate()?;
            let a_dense = self.prepack.dense(0, a)?;
            if a_dense.len() == k {
                // SAFETY: `b` was just validated as a contiguous view of a
                // 16-bit float dtype (`half_storage_format` admits only
                // `Float16`/`BFloat16`) whose element count equals `k * n`.
                // Both are transparent over `u16`, so reading their storage as
                // raw bit patterns is sound, and the view outlives this call.
                let b_bits = unsafe { std::slice::from_raw_parts(b.data_ptr::<u16>(), k * n) };
                let mut result = vec![0.0f32; n];
                count_half_decode_gemv();
                matmul::half_decode_gemv_dispatch(
                    format,
                    &self.prepack,
                    b,
                    &a_dense,
                    b_bits,
                    &mut result,
                    k,
                    n,
                    trans_b,
                );
                return Ok(Some(result));
            }
        }

        // Everything past here reads B as [K, N], so a transposed weight would
        // have to be materialised first -- which is the trade the module header
        // of `half_gemv` explains is not worth making.
        if trans_b {
            let _ = (m, k, n);
            return Ok(None);
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

        // f16 fast paths, shared with `MatMul`. `trans_a` still disqualifies —
        // A is a single row at decode, so transposing it is meaningless, and
        // the packed prefill path wants it untransposed. `trans_b` no longer
        // does: at M = 1 a `[N, K]` weight is the *better* GEMV layout, because
        // every output element is one contiguous dot product.
        //
        // Without these, an f16 `Gemm` falls into the portable blocked half
        // GEMM, which is the worst dense region measured anywhere in this EP:
        // at K=N=3584 it was 6.57x slower than ORT at M=1/1 thread and 46.67x
        // at M=1/8 threads, because that path never scales with thread count on
        // a single-row problem (10.07 ms at 1 thread, 10.26 ms at 8).
        let half_fast_path: Option<Vec<f32>> = if self.trans_a {
            None
        } else {
            self.try_half_fast_path(&inputs[0], &inputs[1], m, k, n, self.trans_b)?
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
thread_local! {
    /// Test-only count of `m == 1` f16 decodes served by the GEMV, so a test
    /// can assert *which route* ran rather than only that the numbers are
    /// plausible -- every route here agrees to half-precision rounding, so the
    /// numbers alone cannot tell them apart.
    static HALF_DECODE_GEMV_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn count_half_decode_gemv() {
    #[cfg(test)]
    HALF_DECODE_GEMV_CALLS.with(|c| c.set(c.get() + 1));
}

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
fn half_decode_gemv_calls() -> u64 {
    HALF_DECODE_GEMV_CALLS.with(std::cell::Cell::get)
}

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
fn reset_half_decode_gemv_calls() {
    HALF_DECODE_GEMV_CALLS.with(|c| c.set(0));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::{Attribute, Graph, NodeId, TensorData, WeightRef};

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

    /// An f16 `Gemm` decode with `transB = 1` -- the layout every `nn.Linear`
    /// export produces -- takes the `[N, K]` GEMV and agrees with the same
    /// logical matrix stored untransposed.
    ///
    /// Before this route existed, `trans_b` disqualified the f16 fast path
    /// outright and the call fell into the portable blocked half GEMM. That is
    /// not a small difference: at K=N=3584 it measured 32-36 ms against ORT's
    /// 0.24-1.5 ms, and it did not improve with thread count at all.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn half_decode_gemm_takes_the_nk_gemv_when_b_is_transposed() {
        let (m, k, n) = (1usize, 67usize, 41usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.125 - 0.5).collect();
        let kn: Vec<f32> = (0..k * n).map(|i| (i % 17) as f32 * 0.0625 - 0.5).collect();
        let mut nk = vec![0.0f32; n * k];
        for p in 0..k {
            for j in 0..n {
                nk[j * k + p] = kn[p * n + j];
            }
        }

        let before = super::half_gemv::NK_GEMV_CALLS.with(|calls| calls.get());
        let (transposed, _) = run_half_gemm(m, k, n, &a, &nk, true);
        let took_the_gemv = super::half_gemv::NK_GEMV_CALLS.with(|calls| calls.get()) == before + 1;
        assert!(
            took_the_gemv || !super::half_gemv::simd_available(HalfFormat::F16),
            "an f16 transB decode must reach the [N, K] GEMV on an f16c host"
        );

        let (straight, _) = run_half_gemm(m, k, n, &a, &kn, false);
        for (lhs, rhs) in transposed.iter().zip(&straight) {
            assert!(
                (lhs - rhs).abs() <= 1e-3 * (1.0 + lhs.abs()),
                "transB={lhs} vs transB=0 {rhs}"
            );
        }
    }

    /// Prefill is untouched: `transB = 1` at `M > 1` still declines the f16
    /// fast path, because everything past the GEMV reads B as `[K, N]`.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn half_prefill_gemm_does_not_take_the_nk_gemv() {
        let (m, k, n) = (4usize, 67usize, 41usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.125 - 0.5).collect();
        let nk: Vec<f32> = (0..n * k).map(|i| (i % 17) as f32 * 0.0625 - 0.5).collect();

        let before = super::half_gemv::NK_GEMV_CALLS.with(|calls| calls.get());
        let (out, _) = run_half_gemm(m, k, n, &a, &nk, true);
        assert_eq!(
            super::half_gemv::NK_GEMV_CALLS.with(|calls| calls.get()),
            before,
            "the GEMV is a decode kernel and must not claim an M > 1 call"
        );

        let mut want = vec![0.0f32; m * n];
        for row in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += half::f16::from_f32(a[row * k + p]).to_f32()
                        * half::f16::from_f32(nk[j * k + p]).to_f32();
                }
                want[row * n + j] = acc;
            }
        }
        for (got, expected) in out.iter().zip(&want) {
            assert!(
                (got - expected).abs() <= 1e-2 * (1.0 + expected.abs()),
                "{got} vs {expected}"
            );
        }
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

    /// A transposed B at **`M > 1`** keeps the blocked half GEMM: the decode
    /// GEMV declines anything but a single row, and everything past it reads B
    /// as `[K, N]`, so a transpose would have to be materialised first.
    ///
    /// At `M == 1` this is no longer true -- see
    /// `half_decode_gemm_takes_the_nk_gemv_when_b_is_transposed`, which is why
    /// this test pins `m = 4`.
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

    /// Serializes the tests that read the process-global weight-transpose cache
    /// bytes or toggle its admission flag, so they never observe each other's
    /// entries or setting under Rust's parallel test harness (#1056). Analogous
    /// to `matmul_nbits`'s `CACHE_FLAG_TEST_LOCK`.
    static TRANSPOSE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Build a single-node `Gemm` graph with a constant `[n, k]` `B` initializer
    /// and `transB = 1`, for driving [`weight_transpose_cache_predicted_bytes`].
    /// The initializer's *bytes* are irrelevant (the predictor reads only its
    /// dims), so they are left zeroed; the executed kernel below uses its own
    /// `Owned` `B` carrying real data.
    fn trans_b_gemm_graph(n: usize, k: usize) -> Graph {
        use onnx_runtime_ir::static_shape;
        let mut graph = Graph::new();
        let a = graph.create_named_value("A", DataType::Float32, static_shape([1, k]));
        let b = graph.create_named_value("B", DataType::Float32, static_shape([n, k]));
        let y = graph.create_named_value("Y", DataType::Float32, static_shape([1, n]));
        graph.add_input(a);
        graph.add_input(b);
        let mut node = Node::new(NodeId(0), "Gemm", vec![Some(a), Some(b)], vec![y]);
        node.attributes.insert("transB".into(), Attribute::Int(1));
        graph.insert_node(node);
        graph.add_output(y);
        graph.set_initializer(
            b,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![n, k],
                vec![0u8; n * k * 4],
            )),
        );
        graph
    }

    /// #1056 acceptance criterion: the predicted transpose-cache bytes must
    /// equal the bytes actually held after a real run, ratio 1.00.
    ///
    /// This runs the constant-weight `transB` `Gemm` kernel through the `Kernel`
    /// trait at *two* activation shapes (prefill `m = 4`, decode `m = 1`) — the
    /// exact duplication the shape-keyed kernel cache produces in an
    /// autoregressive session — and asserts that the process-global cache grows
    /// by the predicted amount *once*, proving effect (b) from #1051 (per-
    /// instantiation copies) does **not** apply here: the global cache keys on
    /// the weight address, so the decode instance hits the prefill entry.
    #[test]
    fn predicted_transpose_bytes_equal_actual_after_gemm_execution() {
        let _guard = TRANSPOSE_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // Thread-local admit: never mutates the process-global the concurrent
        // `transposed_b` tests read (#1056 isolation).
        let _admit = crate::kernels::weight_transpose::CacheEnabledScope::new(true);

        // Distinctive geometry so a stray same-shaped entry cannot alias it.
        let (n, k) = (37usize, 91usize);
        let graph = trans_b_gemm_graph(n, k);
        let predicted = matmul::weight_transpose_cache_predicted_bytes(&graph);
        assert_eq!(
            predicted,
            (n as u64) * (k as u64) * 4,
            "one f32 [k,n] transpose per constant transB Gemm weight"
        );

        // One `B` buffer, shared by both kernel instances, so both key the
        // global cache identically.
        let b_data: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.013).sin()).collect();
        let b = Owned::f32(&[n, k], &b_data);
        let b_ptr = b.view().data_ptr::<f32>();

        // Evict any stale entry a since-freed weight of these dims may have left
        // at this recycled address, so the first `run` below is a genuine miss
        // and the byte total grows by `predicted` (rather than silently hitting a
        // pre-existing entry and asserting against no growth). Safe: no live
        // concurrent allocation can share `b_ptr`.
        crate::kernels::weight_transpose::f32_cache_evict(b_ptr, n, k);

        let run = |m: usize| -> Vec<f32> {
            let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.007).cos()).collect();
            let a = Owned::f32(&[m, k], &a_data);
            let mut y = Owned::zeros_f32(&[m, n]);
            let mut kernel = GemmKernel {
                alpha: 1.0,
                beta: 1.0,
                trans_a: false,
                trans_b: true,
                prepack: MatMulPrepack::default(),
            };
            kernel.set_constant_inputs(&[false, true]);
            kernel
                .execute(&[a.view(), b.view()], &mut [y.view_mut()])
                .unwrap();
            y.to_f32()
        };

        let before = matmul::weight_transpose_cache_bytes();
        let _prefill = run(4);
        let _decode = run(1);
        let after = matmul::weight_transpose_cache_bytes();

        // The process-global byte total is shared across the parallel test
        // harness, so a *concurrent* test caching an unrelated weight can also
        // grow it. Measure the bytes held for THIS weight directly by re-keying
        // the global cache with the exact `B` pointer the kernel used (a
        // constant, contiguous f32 input densifies to a borrow of its own
        // buffer, so `dense(1)` and this query share one address): a hit returns
        // the entry the two runs installed, never a fresh allocation.
        //
        // SAFETY: `b` is a live, contiguous `[n, k]` f32 tensor for the whole
        // test, so its data pointer addresses exactly `n * k` f32 values.
        let b_slice = unsafe { std::slice::from_raw_parts(b.view().data_ptr::<f32>(), n * k) };
        let entry = crate::kernels::weight_transpose::cached_transpose_f32(b_slice, n, k)
            .expect("the executed kernel cached this constant weight's transpose");
        let actual = (entry.len() * std::mem::size_of::<f32>()) as u64;
        assert_eq!(
            actual, predicted,
            "predicted transpose-cache bytes must equal the bytes actually held \
             for this weight (ratio 1.00); two shape instantiations retain one \
             shared copy"
        );
        // And that copy is genuinely part of the global total the plan budgets.
        assert!(
            after >= before + predicted as usize,
            "the cached transpose must be reflected in the global byte total"
        );
    }

    /// #1056 decline contract: when the cache is declined, a constant `transB`
    /// `Gemm` weight retains **nothing** (the transpose is recomputed per call
    /// and freed), and the result is byte-identical to the admitted run — a pure
    /// performance tradeoff, never a numerical one.
    #[test]
    fn declined_transpose_cache_retains_nothing_and_is_byte_identical() {
        let _guard = TRANSPOSE_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        // A fresh `B` never seen by the global cache. Built once so its data
        // pointer is stable across both runs and can be probed by exact key.
        let (n, k, m) = (29usize, 53usize, 3usize);
        let b_data: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.019).sin()).collect();
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.011).cos()).collect();
        let b = Owned::f32(&[n, k], &b_data);
        // `b` outlives every use of this pointer below and is a contiguous
        // `[n, k]` f32 tensor, so the key names exactly this weight.
        let b_ptr = b.view().data_ptr::<f32>();

        // Address reuse hygiene: the cache keys on `(addr, K, N)`, and an earlier
        // (now-freed) weight of the same dims could have been recycled onto this
        // exact address, leaving a stale entry that would make the probe below a
        // false positive. Evict that key first so the "before" state is
        // deterministically empty. Safe under the parallel harness: no *live*
        // concurrent allocation can share `b_ptr`, so the only entry this can
        // touch is a stale one nobody is using.
        crate::kernels::weight_transpose::f32_cache_evict(b_ptr, n, k);
        assert!(
            !crate::kernels::weight_transpose::f32_cache_contains(b_ptr, n, k),
            "precondition: this weight's key must start absent"
        );

        let run = || -> Vec<f32> {
            let a = Owned::f32(&[m, k], &a_data);
            let mut y = Owned::zeros_f32(&[m, n]);
            let mut kernel = GemmKernel {
                alpha: 1.0,
                beta: 1.0,
                trans_a: false,
                trans_b: true,
                prepack: MatMulPrepack::default(),
            };
            kernel.set_constant_inputs(&[false, true]);
            kernel
                .execute(&[a.view(), b.view()], &mut [y.view_mut()])
                .unwrap();
            y.to_f32()
        };

        // Declined: this weight's transpose must not be resident afterward.
        // Probe the exact `(addr, n, k)` key the kernel installs (it calls
        // `transposed_b(&b, n, k)`) rather than a global byte total, so a
        // concurrent test caching an unrelated weight cannot mask a leak.
        let declined = {
            let _decline = crate::kernels::weight_transpose::CacheEnabledScope::new(false);
            let out = run();
            assert!(
                !crate::kernels::weight_transpose::f32_cache_contains(b_ptr, n, k),
                "a declined transpose cache must retain nothing for this weight"
            );
            out
        };

        // Admitted: byte-identical output (transpose is the same math either
        // way) and now this weight *is* resident, confirming the two paths
        // differ only in retention, not in numerics.
        let admitted = {
            let _admit = crate::kernels::weight_transpose::CacheEnabledScope::new(true);
            let out = run();
            assert!(
                crate::kernels::weight_transpose::f32_cache_contains(b_ptr, n, k),
                "an admitted transpose cache must retain this weight"
            );
            out
        };
        assert_eq!(
            declined, admitted,
            "declining the transpose cache must not change results"
        );
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod half_decode_route_tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    /// `Gemm` at `m == 1` run through the real kernel. Returns the output and
    /// the number of decode-GEMV calls it made, so a test asserts *which*
    /// kernel ran and not only that the numbers are plausible.
    fn decode_gemm(k: usize, n: usize, a: &[f32], b: &[f32], trans_b: bool) -> (Vec<f32>, u64) {
        let a_t = Owned::f16(&[1, k], a);
        let b_t = if trans_b {
            Owned::f16(&[n, k], b)
        } else {
            Owned::f16(&[k, n], b)
        };
        let mut out = Owned::zeros_f32(&[1, n]);
        let mut kernel = GemmKernel {
            alpha: 1.0,
            beta: 1.0,
            trans_a: false,
            trans_b,
            prepack: MatMulPrepack::default(),
        };
        kernel.set_constant_inputs(&[false, true]);
        super::reset_half_decode_gemv_calls();
        kernel
            .execute(&[a_t.view(), b_t.view()], &mut [out.view_mut()])
            .unwrap();
        (out.to_f32(), super::half_decode_gemv_calls())
    }

    /// The same logical decode through `MatMul`, for the route-agreement test.
    fn decode_matmul(k: usize, n: usize, a: &[f32], b: &[f32]) -> (Vec<f32>, u64) {
        let a_t = Owned::f16(&[1, k], a);
        let b_t = Owned::f16(&[k, n], b);
        let mut out = Owned::zeros_f32(&[1, n]);
        let mut kernel = matmul::MatMulKernel::default();
        kernel.set_constant_inputs(&[false, true]);
        matmul::reset_half_decode_gemv_calls();
        kernel
            .execute(&[a_t.view(), b_t.view()], &mut [out.view_mut()])
            .unwrap();
        (out.to_f32(), matmul::half_decode_gemv_calls())
    }

    /// `f64` reference over the *narrowed* operand values, so the only error
    /// the tolerance has to absorb is the kernel's own accumulation order.
    fn reference(k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f64> {
        let narrow = |v: &[f32]| -> Vec<f64> {
            v.iter()
                .map(|&x| f64::from(half::f16::from_f32(x).to_f32()))
                .collect()
        };
        let (a, b) = (narrow(a), narrow(b));
        let mut out = vec![0.0f64; n];
        for (p, &av) in a.iter().enumerate().take(k) {
            for j in 0..n {
                out[j] += av * b[p * n + j];
            }
        }
        out
    }

    fn operand(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32) * 0.0137 + seed).sin() * 0.5)
            .collect()
    }

    fn worst_rel(got: &[f32], want: &[f64]) -> f64 {
        got.iter()
            .zip(want)
            .map(|(&got, &want)| (f64::from(got) - want).abs() / (1.0 + want.abs()))
            .fold(0.0f64, f64::max)
    }

    /// #1381, the part that survived: equivalent math must reach the same
    /// *kernel*, not merely the same module.
    ///
    /// Reaching one backend was not enough. `trans_b` still chose between the
    /// `[K, N]` and `[N, K]` GEMVs, which are not interchangeable: measured on
    /// AVX2 at `t = 1` the `[K, N]` kernel is 1.56x-2.98x slower (it strides
    /// `n * 2` bytes between consecutive `p`, so at `n = 6144` every read
    /// crosses a page and the prefetcher cannot run ahead) and 2.7x-9.3x less
    /// accurate (one serial accumulator per column against four combined
    /// pairwise). So the same logical projection was priced -- and rounded --
    /// by how the exporter happened to store the weight.
    ///
    /// With a constant weight transposed once into the prepack cache, all
    /// three spellings run the `[N, K]` kernel over identical values, so they
    /// now agree *bit for bit* rather than to a tolerance. That is the
    /// property worth pinning: a tolerance would pass even if they diverged
    /// again.
    #[test]
    fn every_operator_and_stored_order_decode_bit_identically() {
        // Thread-local admit, the #1056 isolation idiom: this test asserts the
        // *admitted* route, so it must not inherit whatever verdict another
        // test left on the process-global, and must not write that global
        // itself (which would race every concurrent reader).
        let _admit = crate::kernels::weight_transpose::CacheEnabledScope::new(true);
        for &(k, n) in &[(259usize, 131usize), (512, 512), (777, 333)] {
            let a = operand(k, 0.25);
            let b_kn = operand(k * n, 1.75);
            let mut b_nk = vec![0.0f32; k * n];
            for p in 0..k {
                for j in 0..n {
                    b_nk[j * k + p] = b_kn[p * n + j];
                }
            }

            matmul::reset_half_decode_transposed_calls();
            let (gemm_kn, gemm_kn_calls) = decode_gemm(k, n, &a, &b_kn, false);
            let transposed_after_kn = matmul::half_decode_transposed_calls();
            let (gemm_nk, gemm_nk_calls) = decode_gemm(k, n, &a, &b_nk, true);
            let (mm_kn, mm_calls) = decode_matmul(k, n, &a, &b_kn);

            assert_eq!(
                (gemm_kn_calls, gemm_nk_calls, mm_calls),
                (1, 1, 1),
                "{k}x{n}: every spelling must be served by the decode GEMV"
            );
            assert_eq!(
                transposed_after_kn, 1,
                "{k}x{n}: a constant [K, N] weight must reach the [N, K] kernel \
                 through the transpose cache, not be read in place"
            );

            for (index, ((&kn, &nk), &mm)) in gemm_kn
                .iter()
                .zip(gemm_nk.iter())
                .zip(mm_kn.iter())
                .enumerate()
            {
                assert_eq!(
                    kn.to_bits(),
                    nk.to_bits(),
                    "{k}x{n}[{index}]: Gemm transB=0 gave {kn}, transB=1 gave {nk}"
                );
                assert_eq!(
                    kn.to_bits(),
                    mm.to_bits(),
                    "{k}x{n}[{index}]: Gemm gave {kn}, MatMul gave {mm}"
                );
            }
        }
    }

    /// Declining the transpose cache (#1056) must fall back, not fail.
    ///
    /// The plan can refuse the footprint on a large model, and then the
    /// `[K, N]` weight has to be read in place again. That path must still be
    /// taken, still be correct, and -- unlike the admitted one -- must retain
    /// nothing.
    #[test]
    fn a_declined_transpose_cache_falls_back_to_reading_in_place() {
        use crate::kernels::weight_transpose::CacheEnabledScope;
        let (k, n) = (263usize, 137usize);
        let a = operand(k, 0.5);
        let b_kn = operand(k * n, 2.25);

        let _decline = CacheEnabledScope::new(false);
        matmul::reset_half_decode_transposed_calls();
        let (out, calls) = decode_gemm(k, n, &a, &b_kn, false);

        assert_eq!(calls, 1, "the decode GEMV must still serve the call");
        assert_eq!(
            matmul::half_decode_transposed_calls(),
            0,
            "a declined cache must not reach the transposed kernel"
        );
        let want = reference(k, n, &a, &b_kn);
        let worst = worst_rel(&out, &want);
        assert!(
            worst <= 2e-3,
            "declined-cache fallback drifted from the f64 reference: {worst:e}"
        );
    }

    /// #1381: `MatMul` and `Gemm` must take the *same* `m == 1` f16 route.
    ///
    /// They did not. `MatMul` diverted to the fused widen-pack GEBP once
    /// `k * n` reached 1M and `Gemm` stayed on the decode GEMV at every
    /// weight, so the identical operation had two kernels and two costs
    /// depending on which op the exporter emitted. Re-measuring picked the
    /// GEMV, so this asserts *both* ops reach it -- above and below the weight
    /// where the divert used to happen, and out to a 45M `mlp` shape.
    ///
    /// Fails loudly if either side re-acquires a weight-dependent divert: the
    /// counter is incremented by the GEMV arm alone.
    #[test]
    fn gemm_and_matmul_take_the_same_decode_route() {
        if !half_gemv::simd_available(HalfFormat::F16) {
            return;
        }
        // 1024x1024 is exactly the weight the retired divert triggered at;
        // 1024x768 the largest below it; the rest are decode shapes from a
        // 7B-class graph, every one of which the divert used to claim.
        for (k, n) in [
            (64usize, 64usize),
            (1024, 768),
            (1024, 1024),
            (1024, 2048),
            (4096, 4096),
        ] {
            let a = operand(k, 0.25);
            let b = operand(k * n, -0.5);

            let (gemm_out, gemm_gemv) = decode_gemm(k, n, &a, &b, false);
            assert_eq!(
                gemm_gemv, 1,
                "k={k} n={n}: Gemm must serve an m == 1 f16 decode with the GEMV"
            );

            let (matmul_out, matmul_gemv) = decode_matmul(k, n, &a, &b);
            assert_eq!(
                matmul_gemv, 1,
                "k={k} n={n}: MatMul must serve an m == 1 f16 decode with the GEMV"
            );

            assert_eq!(
                gemm_out, matmul_out,
                "k={k} n={n}: Gemm and MatMul decode must be bit-identical"
            );

            let worst = worst_rel(&gemm_out, &reference(k, n, &a, &b));
            assert!(
                worst <= 2e-3,
                "k={k} n={n}: decode disagrees with the f64 reference by {worst:e}"
            );
        }
    }

    /// Falsifier for #1731: a recycled weight address must not serve another
    /// weight's transpose.
    ///
    /// `WeightTransposeKey` is `(addr, k, n, tag)`. Nothing in that key is tied
    /// to the *contents* or the *lifetime* of the buffer it names, so once the
    /// first weight below is dropped and the allocator hands the same block to
    /// the second, the lookup hits and the GEMV multiplies by the wrong matrix.
    #[test]
    fn a_recycled_weight_address_must_not_serve_another_weights_transpose() {
        if !half_gemv::simd_available(HalfFormat::F16) {
            return;
        }
        let (k, n) = (4096usize, 4096usize);
        let a = operand(k, 0.25);

        // Two *different* weights of identical shape and dtype.
        let first = operand(k * n, -0.5);
        let second = operand(k * n, 0.125);

        // Populate the global cache for `first`, then let its buffer go.
        let (_, _) = decode_gemm(k, n, &a, &first, false);

        // `second` is very likely to land on the recycled block.
        let (got, _) = decode_gemm(k, n, &a, &second, false);

        let worst = worst_rel(&got, &reference(k, n, &a, &second));
        assert!(
            worst <= 2e-3,
            "a recycled address served the previous weight's transpose: {worst:e}"
        );
    }

    /// A transposed weight is `[N, K]` and keeps its own GEMV at every size.
    ///
    /// Worth pinning separately because the `[N, K]` arm is the layout every
    /// `nn.Linear` export produces, and because declining it drops `Gemm` into
    /// the portable blocked half GEMM -- there is no other fast path past this
    /// point without the `mlas` feature.
    #[test]
    fn transposed_decode_keeps_its_gemv_at_every_weight() {
        if !half_gemv::simd_available(HalfFormat::F16) {
            return;
        }
        for (k, n) in [(1024usize, 2048usize), (4096, 4096)] {
            let a = operand(k, 0.25);
            let kn = operand(k * n, -0.5);
            let mut nk = vec![0.0f32; n * k];
            for p in 0..k {
                for j in 0..n {
                    nk[j * k + p] = kn[p * n + j];
                }
            }
            let (out, gemv) = decode_gemm(k, n, &a, &nk, true);
            assert_eq!(gemv, 1, "k={k} n={n}: a transB decode must reach the GEMV");

            let worst = worst_rel(&out, &reference(k, n, &a, &kn));
            assert!(
                worst <= 2e-3,
                "k={k} n={n}: transB decode disagrees by {worst:e}"
            );
        }
    }

    /// A `bf16` decode through `Gemm`, which used to be excluded by dtype.
    ///
    /// `MatMul` served a `bf16` `m == 1` from the decode GEMV and `Gemm` did
    /// not -- its fast path required both operands to be `Float16` -- so the
    /// identical decode landed in the portable blocked half GEMM, the slowest
    /// dense region measured anywhere in this EP. This pins the pair the same
    /// way the `f16` test above does: both ops on the GEMV, bit-identical to
    /// each other, and inside tolerance of an `f64` reference taken over the
    /// *bf16-narrowed* operand values.
    #[test]
    fn bf16_decode_takes_the_same_gemv_in_both_ops() {
        if !half_gemv::simd_available(HalfFormat::Bf16) {
            return;
        }
        for (k, n) in [(64usize, 64usize), (1024, 1024), (2048, 3072)] {
            let a = operand(k, 0.25);
            let b = operand(k * n, -0.5);

            let (gemm_out, gemm_gemv) = decode_gemm_bf16(k, n, &a, &b, false);
            assert_eq!(
                gemm_gemv, 1,
                "k={k} n={n}: Gemm must serve an m == 1 bf16 decode with the GEMV"
            );

            let (matmul_out, matmul_gemv) = decode_matmul_bf16(k, n, &a, &b);
            assert_eq!(
                matmul_gemv, 1,
                "k={k} n={n}: MatMul must serve an m == 1 bf16 decode with the GEMV"
            );

            assert_eq!(
                gemm_out, matmul_out,
                "k={k} n={n}: Gemm and MatMul bf16 decode must be bit-identical"
            );

            let worst = worst_rel(&gemm_out, &reference_bf16(k, n, &a, &b));
            assert!(
                worst <= 8e-3,
                "k={k} n={n}: bf16 decode disagrees with the f64 reference by {worst:e}"
            );
        }
    }

    /// A *transposed* `bf16` decode now takes the `[N, K]` GEMV, and the
    /// numerics are what prove the weight is read as `bf16` and not as `f16`.
    ///
    /// This test used to assert the opposite. The decline was kernel coverage,
    /// not policy — the `[N, K]` GEMV existed only in an `f16` spelling — so
    /// the fix was to instantiate it per format rather than to keep gating on
    /// dtype. Reading a `bf16` weight through the `f16` kernel would not fault;
    /// it would silently reinterpret every bit pattern, which is why the route
    /// assertion alone is not enough and the error bound is checked against an
    /// `f64` reference built from the untransposed operand.
    #[test]
    fn a_transposed_bf16_decode_takes_the_same_gemv_as_f16() {
        if !half_gemv::simd_available(HalfFormat::Bf16) {
            return;
        }
        let (k, n) = (512usize, 256usize);
        let a = operand(k, 0.25);
        let kn = operand(k * n, -0.5);
        let mut nk = vec![0.0f32; n * k];
        for p in 0..k {
            for j in 0..n {
                nk[j * k + p] = kn[p * n + j];
            }
        }
        let (out, gemv) = decode_gemm_bf16(k, n, &a, &nk, true);
        assert_eq!(
            gemv, 1,
            "a transposed bf16 decode must reach the [N, K] GEMV, not the blocked GEMM"
        );
        let worst = worst_rel(&out, &reference_bf16(k, n, &a, &kn));
        assert!(
            worst <= 8e-3,
            "transposed bf16 decode disagrees by {worst:e} -- \
             an f16 reinterpretation of the weight would land here"
        );
    }

    /// `m > 1` is not decode and must not be counted as such in either dtype.
    ///
    /// The counter is the instrument the two tests above read, so a change
    /// that made it fire for a prefill would make them pass vacuously.
    #[test]
    fn a_multi_row_half_gemm_is_not_counted_as_decode() {
        if !half_gemv::simd_available(HalfFormat::F16) {
            return;
        }
        let (m, k, n) = (4usize, 256usize, 128usize);
        let a = Owned::f16(&[m, k], &operand(m * k, 0.25));
        let b = Owned::f16(&[k, n], &operand(k * n, -0.5));
        let mut out = Owned::zeros_f32(&[m, n]);
        let mut kernel = GemmKernel {
            alpha: 1.0,
            beta: 1.0,
            trans_a: false,
            trans_b: false,
            prepack: MatMulPrepack::default(),
        };
        kernel.set_constant_inputs(&[false, true]);
        super::reset_half_decode_gemv_calls();
        kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(
            super::half_decode_gemv_calls(),
            0,
            "an m = 4 half Gemm is a prefill and must not touch the decode GEMV"
        );
    }

    /// `bf16` twin of [`decode_gemm`].
    fn decode_gemm_bf16(
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        trans_b: bool,
    ) -> (Vec<f32>, u64) {
        let a_t = Owned::bf16(&[1, k], a);
        let b_t = if trans_b {
            Owned::bf16(&[n, k], b)
        } else {
            Owned::bf16(&[k, n], b)
        };
        let mut out = Owned::zeros_f32(&[1, n]);
        let mut kernel = GemmKernel {
            alpha: 1.0,
            beta: 1.0,
            trans_a: false,
            trans_b,
            prepack: MatMulPrepack::default(),
        };
        kernel.set_constant_inputs(&[false, true]);
        super::reset_half_decode_gemv_calls();
        kernel
            .execute(&[a_t.view(), b_t.view()], &mut [out.view_mut()])
            .unwrap();
        (out.to_f32(), super::half_decode_gemv_calls())
    }

    /// `bf16` twin of [`decode_matmul`].
    fn decode_matmul_bf16(k: usize, n: usize, a: &[f32], b: &[f32]) -> (Vec<f32>, u64) {
        let a_t = Owned::bf16(&[1, k], a);
        let b_t = Owned::bf16(&[k, n], b);
        let mut out = Owned::zeros_f32(&[1, n]);
        let mut kernel = matmul::MatMulKernel::default();
        kernel.set_constant_inputs(&[false, true]);
        matmul::reset_half_decode_gemv_calls();
        kernel
            .execute(&[a_t.view(), b_t.view()], &mut [out.view_mut()])
            .unwrap();
        (out.to_f32(), matmul::half_decode_gemv_calls())
    }

    /// `f64` reference over the *bf16-narrowed* operand values.
    fn reference_bf16(k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f64> {
        let narrow = |v: &[f32]| -> Vec<f64> {
            v.iter()
                .map(|&x| f64::from(half::bf16::from_f32(x).to_f32()))
                .collect()
        };
        let (a, b) = (narrow(a), narrow(b));
        let mut out = vec![0.0f64; n];
        for (p, &av) in a.iter().enumerate().take(k) {
            for j in 0..n {
                out[j] += av * b[p * n + j];
            }
        }
        out
    }
}

/// Production-path A/B for the `m == 1` f16 `Gemm` decode route.
///
/// Both arms come out of one build, selected by environment, because
/// `half_prefill_gebp_enabled` is a process-wide `OnceLock` -- so one arm per
/// process, exactly as `benches/half_decode_gemv_ab.rs` does it for `MatMul`:
///
/// ```text
/// ONNX_GENAI_CPU_MM_HALF_GEBP=0 cargo test -p onnx-runtime-ep-cpu --release --lib \
///     bench_gemm_half_decode_route -- --ignored --nocapture   # GEMV (pre-change)
/// cargo test -p onnx-runtime-ep-cpu --release --lib \
///     bench_gemm_half_decode_route -- --ignored --nocapture   # shipped routing
/// ```
///
/// The `f32` row runs the same shape through the same kernel on a path neither
/// arm can move, so it says whether a difference between two processes is the
/// route or the machine.
#[cfg(all(test, target_arch = "x86_64"))]
mod half_decode_route_bench {
    use super::*;
    use crate::kernels::testutil::Owned;
    use std::time::Instant;

    fn median(mut samples: Vec<f64>) -> f64 {
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        samples[samples.len() / 2]
    }

    fn operand(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32) * 0.0137 + seed).sin() * 0.5)
            .collect()
    }

    fn time_decode(dtype: DataType, k: usize, n: usize, reps: usize) -> (f64, f64, u64) {
        let a_data = operand(k, 0.25);
        let b_data = operand(k * n, -0.5);
        let (a, b) = match dtype {
            DataType::Float16 => (Owned::f16(&[1, k], &a_data), Owned::f16(&[k, n], &b_data)),
            DataType::BFloat16 => (Owned::bf16(&[1, k], &a_data), Owned::bf16(&[k, n], &b_data)),
            _ => (Owned::f32(&[1, k], &a_data), Owned::f32(&[k, n], &b_data)),
        };
        let mut out = Owned::zeros_f32(&[1, n]);
        let mut kernel = GemmKernel {
            alpha: 1.0,
            beta: 1.0,
            trans_a: false,
            trans_b: false,
            prepack: MatMulPrepack::default(),
        };
        kernel.set_constant_inputs(&[false, true]);

        // Cold: the first Run of a session, weight not yet resident.
        super::reset_half_decode_gemv_calls();
        let start = Instant::now();
        kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        let cold = start.elapsed().as_secs_f64() * 1e3;
        let gemv = super::half_decode_gemv_calls();

        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let start = Instant::now();
            kernel
                .execute(&[a.view(), b.view()], &mut [out.view_mut()])
                .unwrap();
            samples.push(start.elapsed().as_secs_f64() * 1e3);
        }
        (cold, median(samples), gemv)
    }

    #[test]
    #[ignore = "benchmark"]
    fn bench_gemm_half_decode_route() {
        let shapes: [(&str, usize, usize); 8] = [
            ("k1024n768", 1024, 768),
            ("k1024n1024", 1024, 1024),
            ("k1024n2048", 1024, 2048),
            ("k2048n1024", 2048, 1024),
            ("k512n4096", 512, 4096),
            ("qwen_qkv", 3584, 4608),
            ("llama_mlp", 4096, 11008),
            ("llama_qkv", 4096, 4096),
        ];
        let reps: usize = std::env::var("REPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(25);
        println!(
            "gemv={} gebp={} threads={}",
            std::env::var("ONNX_GENAI_CPU_MM_HALF_GEMV").unwrap_or_else(|_| "default(on)".into()),
            std::env::var("ONNX_GENAI_CPU_MM_HALF_GEBP").unwrap_or_else(|_| "default(on)".into()),
            rayon::current_num_threads()
        );
        let dtype = match std::env::var("PROBE_DTYPE").as_deref() {
            Ok("bf16") => DataType::BFloat16,
            _ => DataType::Float16,
        };
        println!(
            "{:>12} {:>7} {:>7} {:>9} {:>10} {:>8} {:>7}",
            "shape", "k", "n", "cold_ms", "steady_ms", "GB/s", "route"
        );
        for (name, k, n) in shapes {
            let (cold, steady, gemv) = time_decode(dtype, k, n, reps);
            let route = if gemv == 1 { "gemv" } else { "other" };
            let gb = (2 * k * n) as f64 / (steady * 1e-3) / 1e9;
            println!("{name:>12} {k:>7} {n:>7} {cold:>9.3} {steady:>10.3} {gb:>8.1} {route:>7}");
            let (cold, steady, _) = time_decode(DataType::Float32, k, n, reps.min(10));
            let gb = (4 * k * n) as f64 / (steady * 1e-3) / 1e9;
            println!(
                "{:>12} {k:>7} {n:>7} {cold:>9.3} {steady:>10.3} {gb:>8.1} {:>7}",
                "  ^f32 ctl", "ctl"
            );
        }
    }
}
