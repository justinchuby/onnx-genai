//! `Relu`: elementwise `max(0, x)` for f32 (`docs/ORT2.md` §4.4).
//!
//! Dispatch priority:
//! 1. MLAS (when `feature = "mlas"` — x86-64 Linux only)
//! 2. Zero-copy NEON SIMD (aarch64, contiguous f32, no MLAS dependency)
//! 3. Scalar reference via `to_dense_f32_widen` (all platforms, all dtypes)
//!
//! ## NaN semantics
//!
//! **NaN propagates on all platforms.** This matches MLAS, ONNX/numpy
//! `maximum(0, NaN) == NaN`, and the scalar `relu_in_place` path. It does **not**
//! match IEEE 754-2008 `maxNum` (which would return the non-NaN operand).
//!
//! - aarch64 NEON bulk: `vmaxq_f32` lowers to FMAX, which **propagates** NaN.
//!   (Note: `vmaxnmq_f32` lowers to FMAXNM, which *suppresses* — do not confuse.)
//! - Scalar tail and non-aarch64 fallback: explicit PartialOrd comparison
//!   `if x < 0.0 { 0.0 } else { x }`. The comparison returns false for NaN, so
//!   NaN falls through unchanged. **Do not use `f32::max`** — it suppresses NaN
//!   and would silently create a cross-platform semantic fork.
//!
//! ## Signed zero
//!
//! `vmaxq_f32(-0.0, +0.0)` returns `+0.0` on ARM (FMAX returns +0 when the
//! operands are ±0). The scalar comparison form preserves `-0.0` (since
//! `-0.0 < 0.0` is false in IEEE 754). This is an acceptable divergence: no
//! downstream consumer of Relu output distinguishes ±0, and both compare equal
//! via `==`. Tests assert equality, not bit-identity, for zero values.

use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};
use crate::strided::numel;
use onnx_runtime_ep_api::{Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::check_arity;

/// Dispatch counter proving the fast (non-MLAS) Relu path fires.
#[doc(hidden)]
pub static RELU_F32_FAST_TEST_HITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Apply `max(0, x)` in place. Shared with the fused `FusedGemm` kernel so the
/// ReLU activation has a single source of truth.
///
/// NaN is propagated (not clamped to 0): ONNX/numpy `maximum(0, NaN)` is NaN.
/// Uses PartialOrd comparison rather than `f32::max` (which suppresses NaN).
pub(crate) fn relu_in_place(data: &mut [f32]) {
    for v in data.iter_mut() {
        // `*v < 0.0` is false for NaN → NaN passes through unchanged.
        // -0.0 < 0.0 is false in IEEE 754 → -0.0 preserved (acceptable:
        // no consumer distinguishes ±0 in Relu output).
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Stateless f32 ReLU kernel.
pub struct ReluKernel;

/// Factory for [`ReluKernel`] (no attributes).
pub struct ReluFactory;

impl KernelFactory for ReluFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ReluKernel))
    }
}

impl Kernel for ReluKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Relu", inputs, outputs, 1, 1, 1)?;
        #[cfg(feature = "mlas")]
        if relu_contiguous_f32_mlas(&inputs[0], &mut outputs[0])? {
            return Ok(());
        }
        if relu_contiguous_f32_fast(&inputs[0], &mut outputs[0])? {
            return Ok(());
        }
        let x = to_dense_f32_widen("Relu", &inputs[0])?;
        let mut y = x.into_owned();
        relu_in_place(&mut y);
        write_dense_f32_narrow("Relu", &mut outputs[0], &y)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }

    fn estimated_flops(&self) -> Option<u64> {
        None
    }
}

/// Fast contiguous f32 Relu without MLAS. Uses NEON SIMD on aarch64,
/// plain scalar loop elsewhere. Avoids the `to_dense` allocation that
/// dominates the non-Conv portion of ResNet-18 inference (8 Relu nodes ×
/// up to 784 KB spatial tensors after Conv+BN+Relu fusion eliminates the rest).
fn relu_contiguous_f32_fast(input: &TensorView, output: &mut TensorMut) -> Result<bool> {
    if input.dtype != DataType::Float32
        || output.dtype != DataType::Float32
        || input.shape != output.shape
        || !input.is_contiguous()
        || !output.is_contiguous()
    {
        return Ok(false);
    }

    let len = numel(input.shape);

    // Overlap guard: if input and output alias, fall back to the generic path
    // which allocates a separate buffer.
    let input_start = input.data_ptr::<u8>() as usize;
    let input_end = input_start.saturating_add(input.byte_size());
    let output_start = output.data_ptr_mut::<u8>() as usize;
    let output_end = output_start.saturating_add(output.byte_size());
    if output_start < input_end && input_start < output_end {
        return Ok(false);
    }

    // SAFETY: validated contiguous Float32 views — `len` elements are accessible,
    // and the overlap check above proves the ranges are disjoint.
    let src = unsafe { std::slice::from_raw_parts(input.data_ptr::<f32>(), len) };
    let dst = unsafe { std::slice::from_raw_parts_mut(output.data_ptr_mut::<f32>(), len) };

    relu_f32_bulk(src, dst);
    RELU_F32_FAST_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(true)
}

/// Bulk f32 relu: dst[i] = max(src[i], 0).
/// Uses NEON on aarch64; scalar loop elsewhere.
#[inline]
fn relu_f32_bulk(src: &[f32], dst: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        relu_f32_neon(src, dst);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        relu_f32_scalar(src, dst);
    }
}

#[cfg(target_arch = "aarch64")]
fn relu_f32_neon(src: &[f32], dst: &mut [f32]) {
    use std::arch::aarch64::*;
    debug_assert_eq!(src.len(), dst.len());
    let n = src.len();
    // SAFETY: NEON is always available on aarch64. We process 16 elements per
    // iteration (4 lanes × 4 registers) for throughput, then handle the tail.
    // vmaxq_f32 propagates NaN (ARM FMAX semantics): max(NaN, 0) = NaN.
    unsafe {
        let vzero = vdupq_n_f32(0.0);
        let mut i = 0usize;
        let bulk_end = n & !15; // round down to multiple of 16
        while i < bulk_end {
            let a0 = vld1q_f32(src.as_ptr().add(i));
            let a1 = vld1q_f32(src.as_ptr().add(i + 4));
            let a2 = vld1q_f32(src.as_ptr().add(i + 8));
            let a3 = vld1q_f32(src.as_ptr().add(i + 12));
            let r0 = vmaxq_f32(a0, vzero);
            let r1 = vmaxq_f32(a1, vzero);
            let r2 = vmaxq_f32(a2, vzero);
            let r3 = vmaxq_f32(a3, vzero);
            vst1q_f32(dst.as_mut_ptr().add(i), r0);
            vst1q_f32(dst.as_mut_ptr().add(i + 4), r1);
            vst1q_f32(dst.as_mut_ptr().add(i + 8), r2);
            vst1q_f32(dst.as_mut_ptr().add(i + 12), r3);
            i += 16;
        }
        // Tail: process remaining elements 4 at a time, then scalar.
        while i + 4 <= n {
            let a = vld1q_f32(src.as_ptr().add(i));
            let r = vmaxq_f32(a, vzero);
            vst1q_f32(dst.as_mut_ptr().add(i), r);
            i += 4;
        }
        while i < n {
            let v = *src.get_unchecked(i);
            // PartialOrd: NaN < 0.0 is false → NaN passes through unchanged.
            // Do NOT use f32::max — it suppresses NaN.
            *dst.get_unchecked_mut(i) = if v < 0.0 { 0.0 } else { v };
            i += 1;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn relu_f32_scalar(src: &[f32], dst: &mut [f32]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        // PartialOrd: NaN < 0.0 is false → NaN passes through unchanged.
        // Do NOT use f32::max — it suppresses NaN.
        *d = if s < 0.0 { 0.0 } else { s };
    }
}

#[cfg(feature = "mlas")]
fn relu_contiguous_f32_mlas(input: &TensorView, output: &mut TensorMut) -> Result<bool> {
    if input.dtype != DataType::Float32
        || output.dtype != DataType::Float32
        || input.shape != output.shape
        || !input.is_contiguous()
        || !output.is_contiguous()
    {
        return Ok(false);
    }
    let input_start = input.data_ptr::<u8>() as usize;
    let input_end = input_start.saturating_add(input.byte_size());
    let output_start = output.data_ptr_mut::<u8>() as usize;
    let output_end = output_start.saturating_add(output.byte_size());
    if output_start < input_end && input_start < output_end {
        return Ok(false);
    }
    let input = to_dense_f32_widen("Relu", input)?;
    let output_len = output.numel();
    // SAFETY: equal contiguous Float32 shapes prove the output span, and the
    // range check proves it does not overlap the borrowed input.
    let output =
        unsafe { std::slice::from_raw_parts_mut(output.data_ptr_mut::<f32>(), output_len) };
    mlas_sys::compute_relu(&input, output);
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use std::sync::atomic::Ordering;

    #[test]
    fn relu_clamps_negatives() {
        let a = Owned::f32(&[2, 2], &[-1.0, 2.0, -3.0, 4.0]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        let inputs = [a.view()];
        let mut outs = [out.view_mut()];
        ReluKernel.execute(&inputs, &mut outs).unwrap();
        assert_eq!(out.to_f32(), vec![0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn relu_propagates_nan() {
        // ONNX/numpy maximum(0, NaN) == NaN; f32::max would wrongly yield 0.
        let mut data = vec![f32::NAN, -1.0, 2.0];
        relu_in_place(&mut data);
        assert!(data[0].is_nan());
        assert_eq!(data[1], 0.0);
        assert_eq!(data[2], 2.0);
    }

    #[test]
    fn relu_bf16_matches_widened_f32_reference_and_preserves_nan() {
        let x = Owned::bf16(&[5], &[f32::NAN, -80., -0., 1., 80.]);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[5]);
        ReluKernel
            .execute(&[x.view()], &mut [out.view_mut()])
            .unwrap();
        let result = out.to_bf16_as_f32();
        assert!(result[0].is_nan());
        assert_eq!(&result[1..], &[0., 0., 1., 80.]);
    }

    /// Proves the fast (non-MLAS) path fires on contiguous f32 input.
    #[test]
    fn relu_f32_fast_path_fires_on_contiguous_input() {
        let before = RELU_F32_FAST_TEST_HITS.load(Ordering::Relaxed);
        let a = Owned::f32(
            &[4, 4],
            &[
                -1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0, -13.0, 14.0,
                -15.0, 16.0,
            ],
        );
        let mut out = Owned::zeros_f32(&[4, 4]);
        ReluKernel
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap();
        let after = RELU_F32_FAST_TEST_HITS.load(Ordering::Relaxed);
        assert!(
            after > before,
            "RELU_F32_FAST_TEST_HITS did not increment: before={before} after={after}"
        );
        assert_eq!(
            out.to_f32(),
            vec![
                0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0, 0.0, 10.0, 0.0, 12.0, 0.0, 14.0, 0.0, 16.0
            ]
        );
    }

    /// Numerics parity: NEON/scalar fast path matches the reference `relu_in_place`
    /// for boundary lengths 1, 15, 16, 17, and a large non-multiple (1023).
    #[test]
    fn relu_f32_fast_path_matches_scalar_reference() {
        for len in [1, 15, 16, 17, 1023] {
            let data: Vec<f32> = (0..len)
                .map(|i| {
                    let v = (i as f32) - (len as f32 / 2.0);
                    if i == 0 {
                        f32::NAN
                    } else if i == 1 {
                        -0.0
                    } else if i == 2 {
                        0.0
                    } else if i == 3 {
                        f32::INFINITY
                    } else if i == 4 {
                        f32::NEG_INFINITY
                    } else {
                        v
                    }
                })
                .collect();

            // Reference
            let mut reference = data.clone();
            relu_in_place(&mut reference);

            // Fast path
            let a = Owned::f32(&[len], &data);
            let mut out = Owned::zeros_f32(&[len]);
            ReluKernel
                .execute(&[a.view()], &mut [out.view_mut()])
                .unwrap();
            let result = out.to_f32();

            for (idx, (&expected, &actual)) in reference.iter().zip(result.iter()).enumerate() {
                if expected.is_nan() {
                    assert!(
                        actual.is_nan(),
                        "len={len} idx={idx}: expected NaN, got {actual}"
                    );
                } else if expected == 0.0 {
                    // ±0 compare equal; NEON vmaxq_f32 maps -0→+0 while the
                    // scalar comparison form preserves -0. Both are correct Relu
                    // outputs — assert equality, not bit-identity.
                    assert_eq!(
                        actual, 0.0,
                        "len={len} idx={idx}: expected 0.0, got {actual}"
                    );
                } else {
                    assert_eq!(
                        expected.to_bits(),
                        actual.to_bits(),
                        "len={len} idx={idx}: expected {expected} got {actual}"
                    );
                }
            }
        }
    }

    /// NaN semantics through the fast path specifically.
    #[test]
    fn relu_f32_fast_path_nan_semantics() {
        let before = RELU_F32_FAST_TEST_HITS.load(Ordering::Relaxed);
        let data = vec![f32::NAN, -0.0, 0.0, -1.0, 1.0, f32::NAN];
        let a = Owned::f32(&[6], &data);
        let mut out = Owned::zeros_f32(&[6]);
        ReluKernel
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap();
        let after = RELU_F32_FAST_TEST_HITS.load(Ordering::Relaxed);
        assert!(after > before, "fast path did not fire");

        let result = out.to_f32();
        assert!(result[0].is_nan(), "NaN[0] not propagated");
        // -0.0: both comparison form and NEON vmaxq_f32 map it to a value == 0.0.
        // On aarch64 NEON bulk: vmaxq_f32(-0, +0) = +0. On scalar (comparison form):
        // -0.0 < 0.0 is false → -0.0 passed through. Both equal 0.0; we assert
        // equality not bit-identity so this test pins the same behaviour on all targets.
        assert_eq!(result[1], 0.0, "-0 should be non-negative");
        assert_eq!(result[2], 0.0);
        assert_eq!(result[3], 0.0); // -1 → 0
        assert_eq!(result[4], 1.0);
        assert!(result[5].is_nan(), "NaN[5] not propagated");
    }
}
