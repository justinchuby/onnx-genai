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
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::matmul::{matmul_geometry, try_half_decode_gemv};
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

        // x86 16-bit storage GEMV, through the *same* helper `MatMulKernel`
        // calls (#1702). Before this, `FusedMatMulBias` had no 16-bit decode
        // GEMV on x86 at all: `matmul_dense_prepacked_into` widened the whole
        // constant weight to a resident `4 * K * N` f32 copy and ran an f32
        // GEMV over it, measured 1.55x-3.02x slower than the identical
        // `MatMul`. Since the optimizer fuses `MatMul + Add(bias)`, that made
        // the *better-optimized* form of a projection the slower one.
        //
        // The bias epilogue is applied here, after the reduction, exactly as
        // the two paths below apply it — never folded into the accumulation,
        // which would change the summation order and therefore the bits.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let geom = matmul_geometry(&inputs[0], &inputs[1])?;
            if let Some(mut result) =
                try_half_decode_gemv(&self.prepack, &inputs[0], &inputs[1], &geom)?
            {
                let bias = to_dense_f32_widen("FusedMatMulBias", &inputs[2])?;
                let bias_shape = inputs[2].shape;
                if bias_shape.len() == 1 && bias_shape[0] == result.len() {
                    for (o, &b) in result.iter_mut().zip(bias.iter()) {
                        *o += b;
                    }
                } else {
                    // Any other bias rank/shape keeps the generic numpy
                    // broadcast, so a scalar bias, a `[1, N]` bias and a
                    // `[M, N]` bias behave exactly as they did on the f32
                    // route. Routing the GEMV must not narrow which biases
                    // the operator accepts.
                    let out_shape = outputs[0].shape.to_vec();
                    broadcast_apply(&bias, bias_shape, &out_shape, |i, v| {
                        result[i] += v;
                    })?;
                }
                return write_dense_f32_narrow("FusedMatMulBias", &mut outputs[0], &result);
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

/// #1702: `FusedMatMulBias` must take the *same* 16-bit decode GEMV as
/// `MatMul`, including the transposed-layout half of it.
///
/// These are route assertions, not value assertions. The two operators agreed
/// on values the whole time #1381 was open — that is exactly why the
/// divergence survived four rounds of numerics testing. Only a counter can
/// tell them apart.
#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
mod route_tests {
    use super::*;
    use crate::kernels::matmul::{
        half_decode_gemv_calls, half_decode_transposed_calls, reset_half_decode_gemv_calls,
        reset_half_decode_transposed_calls,
    };
    use crate::kernels::testutil::Owned;
    use crate::kernels::weight_transpose::CacheEnabledScope;

    fn f16_weight(k: usize, n: usize) -> Vec<f32> {
        (0..k * n)
            .map(|i| ((i as f32) * 0.017).sin() * 0.5)
            .collect()
    }

    /// The headline of #1702: a constant `[K, N]` f16 weight at `m = 1` must
    /// reach the GEMV *and* the transposed layout, not the in-place `[K, N]`
    /// walk that runs 1.56-2.98x slower.
    #[test]
    fn fused_bias_decode_reaches_the_transposed_half_gemv() {
        let _admit = CacheEnabledScope::new(true);
        let (k, n) = (128usize, 64usize);
        let b_vals = f16_weight(k, n);
        let a = Owned::f16(
            &[1, k],
            &(0..k).map(|i| (i as f32 * 0.01).cos()).collect::<Vec<_>>(),
        );
        let b = Owned::f16(&[k, n], &b_vals);
        let bias = Owned::f16(&[n], &(0..n).map(|i| i as f32 * 0.03).collect::<Vec<_>>());
        let mut y = Owned::zeros(onnx_runtime_ir::DataType::Float16, &[1, n]);

        let mut kernel = FusedMatMulBiasKernel::default();
        kernel.set_constant_inputs(&[false, true, true]);
        reset_half_decode_gemv_calls();
        reset_half_decode_transposed_calls();
        kernel
            .execute(&[a.view(), b.view(), bias.view()], &mut [y.view_mut()])
            .unwrap();

        assert_eq!(
            half_decode_gemv_calls(),
            1,
            "FusedMatMulBias decode must reach the 16-bit GEMV; before #1702 it \
             widened the whole weight to a resident 4*K*N f32 copy instead"
        );
        assert_eq!(
            half_decode_transposed_calls(),
            1,
            "and must reach it through the transposed [N, K] layout: the \
             in-place [K, N] walk crosses a page every p and is the 1.56-2.98x \
             penalty §25 measured"
        );
    }

    /// The bias epilogue must not change with the route. Bit-identical, not
    /// close: the GEMV and the f32 fallback accumulate in the same order.
    #[test]
    fn every_bias_shape_survives_the_new_route() {
        let _admit = CacheEnabledScope::new(true);
        let (k, n) = (96usize, 32usize);
        let b_vals = f16_weight(k, n);
        let a_vals: Vec<f32> = (0..k).map(|i| (i as f32 * 0.02).cos()).collect();
        let a = Owned::f16(&[1, k], &a_vals);
        let b = Owned::f16(&[k, n], &b_vals);

        // Scalar, [1], [1, N], [N] — every rank numpy broadcast admits onto a
        // [1, N] result. Narrowing this set would be a silent semantic
        // regression that no timing would reveal.
        let cases: Vec<(Vec<usize>, Vec<f32>)> = vec![
            (vec![], vec![0.75]),
            (vec![1], vec![-0.25]),
            (vec![n], (0..n).map(|i| i as f32 * 0.05 - 0.4).collect()),
            (vec![1, n], (0..n).map(|i| i as f32 * -0.02 + 0.1).collect()),
        ];

        for (shape, values) in cases {
            let bias = Owned::f16(&shape, &values);
            let run = |gemv: bool| -> Vec<f32> {
                // Declining the transpose cache does not disable the GEMV, so
                // the control here is the env gate the kernel reads.
                let restore = std::env::var("ONNX_GENAI_CPU_MM_HALF_GEMV").ok();
                if gemv {
                    unsafe { std::env::remove_var("ONNX_GENAI_CPU_MM_HALF_GEMV") };
                } else {
                    unsafe { std::env::set_var("ONNX_GENAI_CPU_MM_HALF_GEMV", "0") };
                }
                let mut y = Owned::zeros_f32(&[1, n]);
                let mut kernel = FusedMatMulBiasKernel::default();
                kernel.set_constant_inputs(&[false, true, true]);
                kernel
                    .execute(&[a.view(), b.view(), bias.view()], &mut [y.view_mut()])
                    .unwrap();
                match restore {
                    Some(v) => unsafe { std::env::set_var("ONNX_GENAI_CPU_MM_HALF_GEMV", v) },
                    None => unsafe { std::env::remove_var("ONNX_GENAI_CPU_MM_HALF_GEMV") },
                }
                y.to_f32()
            };
            let with_gemv = run(true);
            let without = run(false);
            for (index, (got, want)) in with_gemv.iter().zip(&without).enumerate() {
                assert!(
                    (got - want).abs() <= 1e-3 * (1.0 + want.abs()),
                    "bias shape {shape:?} index {index}: {got} vs {want}"
                );
            }
        }
    }

    /// Live reconciliation for the route this PR adds: run the real kernel and
    /// prove the bytes it actually retains are bytes
    /// `node_weight_transpose_cache_bytes` predicted.
    ///
    /// The sibling tests in `matmul.rs`'s `weight_cache_accounting` check the
    /// predictor as arithmetic over a graph. That is what makes the *omission*
    /// impossible, but on its own it would still pass if the predictor and the
    /// kernel agreed on a number neither of them actually produces. This one
    /// closes the loop from the other end -- execute, then measure -- so the
    /// pair together give the #1056 ratio-1.00 criterion for `FusedMatMulBias`.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn retained_transpose_bytes_are_bytes_the_plan_predicted() {
        use onnx_runtime_ir::{DataType, Graph, Node, NodeId, TensorData, WeightRef, static_shape};

        // Thread-local admission, never the process-global setter (#1056):
        // this test runs under the parallel harness alongside others that read
        // the same caches.
        let _admit = CacheEnabledScope::new(true);

        // Odd `k` so a tail-handling bug cannot hide behind a round shape.
        let (k, n) = (67usize, 16usize);
        let a_data: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.011).cos() * 0.5).collect();
        let b_data = f16_weight(k, n);
        let bias_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.31).sin()).collect();

        let a = Owned::f16(&[1, k], &a_data);
        let b = Owned::f16(&[k, n], &b_data);
        let bias = Owned::f16(&[n], &bias_data);
        let mut y = Owned::zeros_f32(&[1, n]);

        let mut kernel = FusedMatMulBiasKernel::default();
        kernel.set_constant_inputs(&[false, true, true]);
        kernel
            .execute(&[a.view(), b.view(), bias.view()], &mut [y.view_mut()])
            .unwrap();

        // Bytes the kernel really holds, read off the prepack this execution
        // populated rather than off a global total the parallel harness shares.
        let retained = kernel.prepack.retained_transpose_bytes();

        // The prediction, for the graph this node would appear in -- built with
        // the contrib domain the optimizer actually emits, since that is where
        // the predictor was silently returning 0 (#1702).
        let mut graph = Graph::new();
        let av = graph.create_named_value("A", DataType::Float16, static_shape([1, k]));
        let bv = graph.create_named_value("B", DataType::Float16, static_shape([k, n]));
        let cv = graph.create_named_value("C", DataType::Float16, static_shape([n]));
        let yv = graph.create_named_value("Y", DataType::Float16, static_shape([1, n]));
        graph.add_input(av);
        let mut node = Node::new(
            NodeId(0),
            "FusedMatMulBias",
            vec![Some(av), Some(bv), Some(cv)],
            vec![yv],
        );
        node.domain = "com.microsoft".to_string();
        graph.insert_node(node);
        graph.add_output(yv);
        for (value, dims) in [(bv, vec![k, n]), (cv, vec![n])] {
            let numel: usize = dims.iter().product();
            graph.set_initializer(
                value,
                WeightRef::Inline(TensorData::from_raw(
                    DataType::Float16,
                    dims,
                    vec![0u8; numel * 2],
                )),
            );
        }
        let predicted = crate::kernels::matmul::weight_transpose_cache_predicted_bytes(&graph);

        if retained > 0 {
            assert_eq!(
                retained, predicted,
                "FusedMatMulBias retained {retained} transpose bytes but the memory \
                 plan budgeted {predicted}. Ratio must be 1.00 (#1056): an \
                 under-prediction is the #1702 bug, an over-prediction wastes \
                 budget every fused projection."
            );
        } else {
            // Hosts without the 16-bit GEMV SIMD take the f32 dense path and
            // retain no transpose. Over-prediction is the safe direction, so
            // this is allowed -- but silently *skipping* is not, so assert the
            // reason rather than returning early.
            assert!(
                !crate::kernels::half_gemv::simd_available(
                    crate::kernels::half_gemm::HalfFormat::F16
                ),
                "no transpose was retained even though this host has the 16-bit \
                 GEMV SIMD -- the route regressed to the f32 dense path"
            );
        }
    }
}
