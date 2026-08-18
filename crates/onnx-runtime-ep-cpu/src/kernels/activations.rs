//! Attribute-driven float activation kernels.

use onnx_runtime_ep_api::{Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::check_arity;
use crate::dtype::{to_dense_f32_widen, to_dense_float, write_dense_f32_narrow, write_dense_float};

const SELU_ALPHA_DEFAULT: f32 = 1.673_263_2;
const SELU_GAMMA_DEFAULT: f32 = 1.050_701;

#[derive(Clone, Copy)]
enum Activation {
    Elu { alpha: f32 },
    LeakyRelu { alpha: f32 },
    HardSigmoid { alpha: f32, beta: f32 },
    Selu { alpha: f32, gamma: f32 },
    ThresholdedRelu { alpha: f32 },
    Swish { alpha: f32 },
    Silu,
    Celu { alpha: f32 },
    Mish,
}

impl Activation {
    fn name(self) -> &'static str {
        match self {
            Self::Elu { .. } => "Elu",
            Self::LeakyRelu { .. } => "LeakyRelu",
            Self::HardSigmoid { .. } => "HardSigmoid",
            Self::Selu { .. } => "Selu",
            Self::ThresholdedRelu { .. } => "ThresholdedRelu",
            Self::Swish { .. } => "Swish",
            Self::Silu => "Silu",
            Self::Celu { .. } => "Celu",
            Self::Mish => "Mish",
        }
    }

    fn apply(self, x: f32) -> f32 {
        match self {
            Self::Elu { alpha } => {
                if x >= 0.0 {
                    x
                } else {
                    alpha * x.exp_m1()
                }
            }
            Self::LeakyRelu { alpha } => {
                if x >= 0.0 {
                    x
                } else {
                    alpha * x
                }
            }
            Self::HardSigmoid { alpha, beta } => (alpha * x + beta).clamp(0.0, 1.0),
            Self::Selu { alpha, gamma } => gamma * if x > 0.0 { x } else { alpha * x.exp_m1() },
            Self::ThresholdedRelu { alpha } => {
                if x > alpha {
                    x
                } else {
                    0.0
                }
            }
            // Swish/SiLU: x·sigmoid(alpha·x), evaluated via the numerically
            // stable logistic to avoid overflow at large-magnitude inputs.
            Self::Swish { alpha } => {
                let z = alpha * x;
                let s = if z >= 0.0 {
                    1.0 / (1.0 + (-z).exp())
                } else {
                    let e = z.exp();
                    e / (1.0 + e)
                };
                x * s
            }
            Self::Silu => silu(x),
            // Shared with the slice path rather than transcribed again: the
            // literal ONNX formula loses NaN through `f32::max`/`min`, and one
            // copy of that guard is easier to keep true than two.
            Self::Celu { alpha } => crate::kernels::simd_activations::celu_scalar(x, alpha),
            // Mish(x) = x * tanh(softplus(x)), with softplus in its stable
            // form so large `x` neither overflows nor loses the identity.
            Self::Mish => x * (x.max(0.0) + (-x.abs()).exp().ln_1p()).tanh(),
        }
    }

    fn apply_f64(self, x: f64) -> f64 {
        match self {
            Self::Elu { alpha } => {
                if x >= 0.0 {
                    x
                } else {
                    f64::from(alpha) * x.exp_m1()
                }
            }
            Self::LeakyRelu { alpha } => {
                if x >= 0.0 {
                    x
                } else {
                    f64::from(alpha) * x
                }
            }
            Self::HardSigmoid { alpha, beta } => {
                (f64::from(alpha) * x + f64::from(beta)).clamp(0.0, 1.0)
            }
            Self::Selu { alpha, gamma } => {
                f64::from(gamma)
                    * if x > 0.0 {
                        x
                    } else {
                        f64::from(alpha) * x.exp_m1()
                    }
            }
            Self::ThresholdedRelu { alpha } => {
                if x > f64::from(alpha) {
                    x
                } else {
                    0.0
                }
            }
            Self::Swish { alpha } => {
                let z = f64::from(alpha) * x;
                let s = if z >= 0.0 {
                    1.0 / (1.0 + (-z).exp())
                } else {
                    let e = z.exp();
                    e / (1.0 + e)
                };
                x * s
            }
            Self::Silu => silu_f64(x),
            Self::Celu { alpha } => {
                // See `simd_activations::celu_scalar`: `max`/`min` return the
                // other operand for NaN, so without this the f64 path -- which
                // every `Float64` tensor takes -- would answer `0.0` where ORT
                // and the f32 paths answer NaN.
                if x.is_nan() {
                    return x;
                }
                let a = f64::from(alpha);
                x.max(0.0) + (a * ((x / a).exp() - 1.0)).min(0.0)
            }
            Self::Mish => x * (x.max(0.0) + (-x.abs()).exp().ln_1p()).tanh(),
        }
    }
}

fn silu(x: f32) -> f32 {
    // CUDA's device exp is evaluated in f64. Match that precision before the
    // f32 operation-order boundary so 1-ulp exp differences cannot be amplified
    // by downstream accuracy-level-4 activation quantization.
    if x >= 0.0 {
        // Covers +0.0 and +Inf: +Inf / (1 + exp(-Inf)) = +Inf / 1 = +Inf.
        x / (1.0 + ((-x) as f64).exp() as f32)
    } else if x == f32::NEG_INFINITY {
        // sigmoid(-Inf) = 0 and x * 0 would be NaN, so pin the limit SiLU(-Inf)=0.
        0.0
    } else {
        // Includes NaN, which propagates through exp and the product.
        let e = (x as f64).exp() as f32;
        x * e / (1.0 + e)
    }
}

fn silu_f64(x: f64) -> f64 {
    if x >= 0.0 {
        x / (1.0 + (-x).exp())
    } else if x == f64::NEG_INFINITY {
        0.0
    } else {
        let e = x.exp();
        x * e / (1.0 + e)
    }
}

/// Inputs whose magnitude exceeds this bound (or that are non-finite) fall
/// outside MLAS's internal logistic clamp of `[-18, 18]`, so SiLU must be
/// recomputed accurately for them. Inside the bound the MLAS approximation is
/// the same routine ORT uses and is accurate.
#[cfg(feature = "mlas")]
const SILU_MLAS_SAFE_BOUND: f32 = 18.0;

pub struct ActivationKernel {
    activation: Activation,
}

pub struct EluFactory;
pub struct LeakyReluFactory;
pub struct HardSigmoidFactory;
pub struct SeluFactory;
pub struct ThresholdedReluFactory;
pub struct SwishFactory;
pub struct SiluFactory;
pub struct CeluFactory;
pub struct MishFactory;

impl KernelFactory for EluFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ActivationKernel {
            activation: Activation::Elu {
                alpha: node.attr("alpha").and_then(|a| a.as_float()).unwrap_or(1.0),
            },
        }))
    }
}

impl KernelFactory for LeakyReluFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ActivationKernel {
            activation: Activation::LeakyRelu {
                alpha: node
                    .attr("alpha")
                    .and_then(|a| a.as_float())
                    .unwrap_or(0.01),
            },
        }))
    }
}

impl KernelFactory for HardSigmoidFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ActivationKernel {
            activation: Activation::HardSigmoid {
                alpha: node.attr("alpha").and_then(|a| a.as_float()).unwrap_or(0.2),
                beta: node.attr("beta").and_then(|a| a.as_float()).unwrap_or(0.5),
            },
        }))
    }
}

impl KernelFactory for SeluFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ActivationKernel {
            activation: Activation::Selu {
                alpha: node
                    .attr("alpha")
                    .and_then(|a| a.as_float())
                    .unwrap_or(SELU_ALPHA_DEFAULT),
                gamma: node
                    .attr("gamma")
                    .and_then(|a| a.as_float())
                    .unwrap_or(SELU_GAMMA_DEFAULT),
            },
        }))
    }
}

impl KernelFactory for ThresholdedReluFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ActivationKernel {
            activation: Activation::ThresholdedRelu {
                alpha: node.attr("alpha").and_then(|a| a.as_float()).unwrap_or(1.0),
            },
        }))
    }
}

impl KernelFactory for SwishFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let alpha = node.attr("alpha").and_then(|a| a.as_float()).unwrap_or(1.0);
        // Swish(alpha=1) ≡ SiLU. Canonicalize to SiLU to enable the contiguous
        // fast path and (future) NEON vectorization.
        let activation = if alpha == 1.0 {
            Activation::Silu
        } else {
            Activation::Swish { alpha }
        };
        Ok(Box::new(ActivationKernel { activation }))
    }
}

impl KernelFactory for CeluFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let alpha = node.attr("alpha").and_then(|a| a.as_float()).unwrap_or(1.0);
        // ONNX requires alpha != 0 (the definition divides by it). A model that
        // ships 0 would otherwise reach a division by zero in the kernel, so
        // fall back to the documented default rather than produce Inf/NaN.
        let alpha = if alpha == 0.0 { 1.0 } else { alpha };
        Ok(Box::new(ActivationKernel {
            activation: Activation::Celu { alpha },
        }))
    }
}

impl KernelFactory for MishFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ActivationKernel {
            activation: Activation::Mish,
        }))
    }
}

impl KernelFactory for SiluFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ActivationKernel {
            activation: Activation::Silu,
        }))
    }
}

impl Kernel for ActivationKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(self.activation.name(), inputs, outputs, 1, 1, 1)?;
        if matches!(self.activation, Activation::Silu)
            && silu_contiguous_f32(&inputs[0], &mut outputs[0])
        {
            return Ok(());
        }
        if inputs[0].dtype == DataType::Float64 {
            let y = to_dense_float::<f64>(&inputs[0])?
                .into_iter()
                .map(|x| self.activation.apply_f64(x))
                .collect::<Vec<_>>();
            return write_dense_float::<f64>(&mut outputs[0], &y);
        }
        let input = to_dense_f32_widen(self.activation.name(), &inputs[0])?;
        let y = if matches!(self.activation, Activation::Silu) {
            let mut output = vec![0.0; input.len()];
            silu_f32_slice(&input, &mut output);
            output
        } else if let Activation::Celu { alpha } = self.activation {
            let mut output = vec![0.0; input.len()];
            crate::kernels::simd_activations::celu_f32_slice(&input, &mut output, alpha);
            output
        } else if matches!(self.activation, Activation::Mish) {
            let mut output = vec![0.0; input.len()];
            crate::kernels::simd_activations::mish_f32_slice(&input, &mut output);
            output
        } else {
            input
                .iter()
                .map(|x| self.activation.apply(*x))
                .collect::<Vec<_>>()
        };
        write_dense_f32_narrow(self.activation.name(), &mut outputs[0], &y)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn silu_contiguous_f32(input: &TensorView, output: &mut TensorMut) -> bool {
    if input.dtype != DataType::Float32
        || output.dtype != DataType::Float32
        || input.shape != output.shape
        || input.strides != output.strides
        || !onnx_runtime_ir::is_dense(input.shape, input.strides)
    {
        return false;
    }

    let n = output.numel();
    let bytes = n.saturating_mul(std::mem::size_of::<f32>());
    let input_start = input.data_ptr::<f32>() as usize;
    let input_end = input_start.saturating_add(bytes);
    let output_start = output.data_ptr_mut::<f32>() as usize;
    let output_end = output_start.saturating_add(bytes);
    if output_start < input_end && input_start < output_end {
        return false;
    }

    // SAFETY: executor bounds checks plus equal contiguous f32 shapes prove both
    // pointers span n elements; the range check proves output does not alias input.
    let input = unsafe { std::slice::from_raw_parts(input.data_ptr::<f32>(), n) };
    let output = unsafe { std::slice::from_raw_parts_mut(output.data_ptr_mut::<f32>(), n) };
    silu_f32_slice(input, output);
    true
}

/// SiLU (`x * sigmoid(x)`) over equal-length contiguous f32 slices.
///
/// With the `mlas` feature this uses MLAS's fused one-pass SiLU, including its
/// AVX-512F runtime path. Without `mlas` we keep the scalar reference.
/// On aarch64 without `mlas`, a NEON-vectorized path processes 4 floats at a
/// time using a Cephes-style exp polynomial (~28 ULP worst-case on [-87, 88]).
/// Elements per MLAS + correction block.
///
/// 8192 f32 = 32 KiB, so a block and its input both stay inside L2 and the
/// correction scan reads what MLAS just touched rather than making a second
/// trip to DRAM.
#[cfg(feature = "mlas")]
const SILU_CORRECTION_BLOCK: usize = 8192;

/// MLAS's fused SiLU with its out-of-band results corrected, blocked so the
/// correction stays in cache.
///
/// MLAS's SIMD SiLU clamps its logistic input to [-18, 18] internally, so
/// out-of-range or non-finite results need correction: `SiLU(-1e30)` would
/// leak `sigmoid(-18) ~= 1.5e-8` instead of decaying to 0, and
/// `SiLU(+/-Inf)`/`SiLU(NaN)` would be corrupted.
///
/// Whether a correction is needed depends only on the *input*, so the common
/// all-in-band path never reads or writes the output a second time: a
/// branch-free OR-reduction over the input decides, and the write loop is
/// skipped entirely. The reduction is an accumulator rather than `any()`
/// because `any()`'s early exit is a loop-carried control dependency that
/// stops LLVM vectorising it.
#[cfg(feature = "mlas")]
fn silu_mlas_corrected(input: &[f32], output: &mut [f32]) {
    for (xs, ys) in input
        .chunks(SILU_CORRECTION_BLOCK)
        .zip(output.chunks_mut(SILU_CORRECTION_BLOCK))
    {
        mlas_sys::compute_silu(xs, ys);
        let mut needs_correction = 0u32;
        for &x in xs {
            needs_correction |= u32::from(!x.is_finite() || x.abs() > SILU_MLAS_SAFE_BOUND);
        }
        if needs_correction != 0 {
            for (o, &i) in ys.iter_mut().zip(xs) {
                if !i.is_finite() || i.abs() > SILU_MLAS_SAFE_BOUND {
                    *o = silu(i);
                }
            }
        }
    }
}

pub(crate) fn silu_f32_slice(input: &[f32], output: &mut [f32]) {
    #[cfg(feature = "mlas")]
    {
        crate::kernels::simd_activations::run_chunked_fn(input, output, silu_mlas_corrected);
    }
    #[cfg(all(not(feature = "mlas"), target_arch = "aarch64"))]
    {
        silu_f32_neon(input, output);
    }
    #[cfg(all(not(feature = "mlas"), not(target_arch = "aarch64")))]
    {
        for (output, &input) in output.iter_mut().zip(input) {
            *output = silu(input);
        }
    }
}

/// NEON-vectorized SiLU: `x / (1 + exp(-x))`, processing 4 floats per iteration.
///
/// Uses a Cephes-style exp polynomial with Cody-Waite range reduction. Measured
/// worst-case error ~28 ULP on the normal f32 range ([-87, 88]). Non-finite and
/// extreme values are handled by clamping and the scalar fallback for the tail.
#[cfg(all(not(feature = "mlas"), target_arch = "aarch64"))]
fn silu_f32_neon(input: &[f32], output: &mut [f32]) {
    use std::arch::aarch64::*;

    debug_assert_eq!(input.len(), output.len());

    let len = input.len();
    let mut i = 0;

    // Process 4 elements at a time using NEON intrinsics.
    // exp(-x) is computed via Cody-Waite range reduction + degree-5 polynomial.
    #[allow(clippy::excessive_precision)]
    unsafe {
        // Constants for exp computation:
        let log2ef = vdupq_n_f32(std::f32::consts::LOG2_E);
        let c1 = vdupq_n_f32(0.693359375_f32); // ln(2) high part (exact in f32)
        let c2 = vdupq_n_f32(-2.12194440e-4_f32); // ln(2) low part
        let one = vdupq_n_f32(1.0_f32);
        // Polynomial coefficients for exp(r) on [-ln2/2, ln2/2]:
        // exp(r) ≈ 1 + r*(1 + r*(c2 + r*(c3 + r*(c4 + r*c5))))
        let p2 = vdupq_n_f32(0.500000000_f32); // 1/2!
        let p3 = vdupq_n_f32(0.166666667_f32); // 1/3!
        let p4 = vdupq_n_f32(0.041666666_f32); // 1/4!
        let p5 = vdupq_n_f32(0.008333333_f32); // 1/5!
        // Clamp range to prevent overflow in 2^n reconstruction.
        let exp_lo = vdupq_n_f32(-87.3_f32);
        let exp_hi = vdupq_n_f32(88.7_f32);

        while i + 4 <= len {
            let x = vld1q_f32(input.as_ptr().add(i));
            let neg_x = vnegq_f32(x);

            // Clamp -x to prevent exp overflow/underflow.
            let clamped = vmaxq_f32(vminq_f32(neg_x, exp_hi), exp_lo);

            // Range reduction: n = round(clamped * log2(e))
            let nf = vrndnq_f32(vmulq_f32(clamped, log2ef));
            // r = clamped - n * ln(2), using Cody-Waite split for precision.
            let r = vsubq_f32(vsubq_f32(clamped, vmulq_f32(nf, c1)), vmulq_f32(nf, c2));

            // Horner's method: exp(r) ≈ 1 + r*(1 + r*(p2 + r*(p3 + r*(p4 + r*p5))))
            let poly = vfmaq_f32(p4, r, p5);
            let poly = vfmaq_f32(p3, r, poly);
            let poly = vfmaq_f32(p2, r, poly);
            let poly = vfmaq_f32(one, r, poly);
            let poly = vfmaq_f32(one, r, poly);

            // Reconstruct: exp(clamped) = poly * 2^n.
            // Add n to the IEEE754 exponent by adding n << 23 to the bit pattern.
            let ni = vcvtq_s32_f32(nf);
            let scale = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(ni, vdupq_n_s32(127))));
            let exp_neg_x = vmulq_f32(poly, scale);

            // SiLU: x / (1 + exp(-x))
            let denom = vaddq_f32(one, exp_neg_x);
            let result = vdivq_f32(x, denom);

            // Handle non-finite: if |x| > 87, use a simpler formula.
            // For very negative x: silu(x) ≈ 0. For very positive x: silu(x) ≈ x.
            // The clamping + polynomial already handles this correctly for normal
            // values, but NaN/Inf need the scalar path. Those are rare enough that
            // we skip vectorized NaN handling and correct in the tail.
            vst1q_f32(output.as_mut_ptr().add(i), result);
            i += 4;
        }
    }

    // Fix up non-finite values and handle the scalar tail.
    // Re-check the NEON-computed region for non-finite inputs and recompute those.
    let neon_end = i;
    for j in 0..neon_end {
        if !input[j].is_finite() {
            output[j] = silu(input[j]);
        }
    }
    // Scalar tail for remaining elements.
    while i < len {
        output[i] = silu(input[i]);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::{Attribute, NodeId};

    /// `Float64` tensors never touch the SIMD kernels, so `apply_f64` is the
    /// only implementation a `double` model sees. It has to agree with the
    /// other two about NaN -- `f64::max`/`min` drop it exactly as the f32 ones
    /// do, and ORT propagates it.
    #[test]
    fn celu_and_mish_propagate_nan_on_every_path() {
        for alpha in [0.5f32, 1.0, 3.0] {
            let act = Activation::Celu { alpha };
            assert!(act.apply(f32::NAN).is_nan(), "f32 Celu(NaN), alpha={alpha}");
            assert!(
                act.apply_f64(f64::NAN).is_nan(),
                "f64 Celu(NaN), alpha={alpha}"
            );
        }
        assert!(Activation::Mish.apply(f32::NAN).is_nan(), "f32 Mish(NaN)");
        assert!(
            Activation::Mish.apply_f64(f64::NAN).is_nan(),
            "f64 Mish(NaN)"
        );
    }

    #[test]
    fn activation_formulas_and_defaults() {
        let x = Owned::f32(&[3], &[-1.0, 0.0, 1.0]);
        let mut out = Owned::zeros_f32(&[3]);
        ActivationKernel {
            activation: Activation::Elu { alpha: 1.0 },
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
        assert!((out.to_f32()[0] - ((-1.0f32).exp() - 1.0)).abs() < 1e-6);
        ActivationKernel {
            activation: Activation::LeakyRelu { alpha: 0.1 },
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_f32(), vec![-0.1, 0.0, 1.0]);
        ActivationKernel {
            activation: Activation::HardSigmoid {
                alpha: 0.2,
                beta: 0.5,
            },
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_f32(), vec![0.3, 0.5, 0.7]);
    }

    #[test]
    fn selu_default_and_custom_parameters() {
        let x = Owned::f32(&[3], &[-1.0, 0.0, 2.0]);
        let mut out = Owned::zeros_f32(&[3]);
        let node = Node::new(NodeId(0), "Selu", vec![], vec![]);
        SeluFactory
            .create(&node, &[])
            .unwrap()
            .execute(&[x.view()], &mut [out.view_mut()])
            .unwrap();
        let expected = [
            SELU_GAMMA_DEFAULT * SELU_ALPHA_DEFAULT * (-1.0f32).exp_m1(),
            0.0,
            SELU_GAMMA_DEFAULT * 2.0,
        ];
        for (got, want) in out.to_f32().into_iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }

        let mut node = Node::new(NodeId(0), "Selu", vec![], vec![]);
        node.attributes
            .insert("alpha".into(), Attribute::Float(2.0));
        node.attributes
            .insert("gamma".into(), Attribute::Float(0.5));
        SeluFactory
            .create(&node, &[])
            .unwrap()
            .execute(&[x.view()], &mut [out.view_mut()])
            .unwrap();
        let expected = [(-1.0f32).exp_m1(), 0.0, 1.0];
        for (got, want) in out.to_f32().into_iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[test]
    fn thresholded_relu_default_custom_and_boundary() {
        let x = Owned::f32(&[5], &[-1.0, 0.5, 1.0, 1.5, 2.0]);
        let mut out = Owned::zeros_f32(&[5]);
        let node = Node::new(NodeId(0), "ThresholdedRelu", vec![], vec![]);
        ThresholdedReluFactory
            .create(&node, &[])
            .unwrap()
            .execute(&[x.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![0.0, 0.0, 0.0, 1.5, 2.0]);

        let mut node = Node::new(NodeId(0), "ThresholdedRelu", vec![], vec![]);
        node.attributes
            .insert("alpha".into(), Attribute::Float(0.5));
        ThresholdedReluFactory
            .create(&node, &[])
            .unwrap()
            .execute(&[x.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![0.0, 0.0, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn selu_supports_f16_f64_and_bf16() {
        let values = [-1.0f32, 0.0, 2.0];
        let expected: Vec<f32> = values
            .iter()
            .map(|&x| {
                Activation::Selu {
                    alpha: SELU_ALPHA_DEFAULT,
                    gamma: SELU_GAMMA_DEFAULT,
                }
                .apply(x)
            })
            .collect();
        let kernel = ActivationKernel {
            activation: Activation::Selu {
                alpha: SELU_ALPHA_DEFAULT,
                gamma: SELU_GAMMA_DEFAULT,
            },
        };

        let x16 = Owned::f16(&[3], &values);
        let mut out16 = Owned::zeros(DataType::Float16, &[3]);
        kernel
            .execute(&[x16.view()], &mut [out16.view_mut()])
            .unwrap();
        for (got, want) in out16.to_f16_as_f32().into_iter().zip(&expected) {
            assert!((got - want).abs() < 1e-3, "got {got}, want {want}");
        }

        let values64 = [-1.234_567_890_123_f64, 0.0, 2.345_678_901_234];
        let x64 = Owned::f64(&[3], &values64);
        let mut out64 = Owned::zeros(DataType::Float64, &[3]);
        kernel
            .execute(&[x64.view()], &mut [out64.view_mut()])
            .unwrap();
        let expected64 = values64.map(|x| {
            f64::from(SELU_GAMMA_DEFAULT)
                * if x > 0.0 {
                    x
                } else {
                    f64::from(SELU_ALPHA_DEFAULT) * x.exp_m1()
                }
        });
        for (got, want) in out64.to_f64().into_iter().zip(expected64) {
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }

        let xbf16 = Owned::bf16(&[3], &values);
        let mut outbf16 = Owned::zeros(DataType::BFloat16, &[3]);
        kernel
            .execute(&[xbf16.view()], &mut [outbf16.view_mut()])
            .unwrap();
        for (got, want) in outbf16.to_bf16_as_f32().into_iter().zip(&expected) {
            assert!((got - want).abs() < 1e-2, "got {got}, want {want}");
        }
    }

    #[test]
    fn thresholded_relu_supports_f16_f64_and_bf16() {
        let values = [-1.0f32, 1.0, 1.5];
        let expected = [0.0f32, 0.0, 1.5];
        let kernel = ActivationKernel {
            activation: Activation::ThresholdedRelu { alpha: 1.0 },
        };

        let x16 = Owned::f16(&[3], &values);
        let mut out16 = Owned::zeros(DataType::Float16, &[3]);
        kernel
            .execute(&[x16.view()], &mut [out16.view_mut()])
            .unwrap();
        assert_eq!(out16.to_f16_as_f32(), expected);

        let values64 = [-1.234_567_890_123_f64, 1.0, 1.234_567_890_123];
        let x64 = Owned::f64(&[3], &values64);
        let mut out64 = Owned::zeros(DataType::Float64, &[3]);
        kernel
            .execute(&[x64.view()], &mut [out64.view_mut()])
            .unwrap();
        let expected64 = [0.0, 0.0, values64[2]];
        for (got, want) in out64.to_f64().into_iter().zip(expected64) {
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }

        let xbf16 = Owned::bf16(&[3], &values);
        let mut outbf16 = Owned::zeros(DataType::BFloat16, &[3]);
        kernel
            .execute(&[xbf16.view()], &mut [outbf16.view_mut()])
            .unwrap();
        assert_eq!(outbf16.to_bf16_as_f32(), expected);
    }

    #[test]
    fn swish_default_and_alpha() {
        let x = Owned::f32(&[3], &[-1.0, 0.0, 2.0]);
        let mut out = Owned::zeros_f32(&[3]);
        // alpha=1 (SiLU): y = x·sigmoid(x).
        ActivationKernel {
            activation: Activation::Swish { alpha: 1.0 },
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
        let sig = |z: f32| 1.0 / (1.0 + (-z).exp());
        let want = [-sig(-1.0), 0.0, 2.0 * sig(2.0)];
        for (g, w) in out.to_f32().iter().zip(&want) {
            assert!((g - w).abs() < 1e-6, "got {g}, want {w}");
        }
        // alpha=2: y = x·sigmoid(2x).
        ActivationKernel {
            activation: Activation::Swish { alpha: 2.0 },
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
        let want2 = [-sig(-2.0), 0.0, 2.0 * sig(4.0)];
        for (g, w) in out.to_f32().iter().zip(&want2) {
            assert!((g - w).abs() < 1e-6, "got {g}, want {w}");
        }
    }

    #[test]
    fn silu_contiguous_matches_reference() {
        // Exact f64 reference SiLU used to pin the vectorized MLAS path.
        fn silu_ref(x: f64) -> f64 {
            if x >= 0.0 {
                x / (1.0 + (-x).exp())
            } else if x == f64::NEG_INFINITY {
                0.0
            } else {
                let e = x.exp();
                x * e / (1.0 + e)
            }
        }

        // Dense range spanning the MLAS clamp boundary plus extreme finite
        // magnitudes that fall well outside the [-18, 18] logistic clamp.
        let mut xs: Vec<f32> = Vec::new();
        let mut v = -50.0f32;
        while v <= 50.0 {
            xs.push(v);
            v += 0.25;
        }
        xs.extend_from_slice(&[
            1e30, -1e30, 1e-30, -1e-30, 18.0, -18.0, 18.5, -18.5, 17.5, -17.5, -0.0, 0.0,
        ]);

        let mut out = vec![0.0; xs.len()];
        silu_f32_slice(&xs, &mut out);

        for (got, input) in out.into_iter().zip(xs) {
            let want = silu_ref(input as f64);
            let abs_err = (got as f64 - want).abs();
            let rel_err = if want.abs() > 1.0 {
                abs_err / want.abs()
            } else {
                abs_err
            };
            // In-range region matches ORT's logistic approximation; extremes are
            // recomputed exactly. Both stay within a tight tolerance.
            assert!(
                abs_err <= 2e-6 || rel_err <= 2e-6,
                "silu({input}) = {got}, want {want}, abs_err {abs_err}, rel_err {rel_err}"
            );
        }
    }

    #[test]
    fn silu_in_range_region_is_bit_close() {
        // Pin the in-range band ([-18, 18]) against the exact reference. The MLAS
        // approximation is ORT's routine; hold it to a tight tolerance so future
        // regressions surface.
        let mut xs: Vec<f32> = Vec::new();
        let mut v = -18.0f32;
        while v <= 18.0 {
            xs.push(v);
            v += 0.1;
        }
        let n = xs.len();
        let x = Owned::f32(&[n], &xs);
        let mut out = Owned::zeros_f32(&[n]);
        ActivationKernel {
            activation: Activation::Silu,
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
        for (got, input) in out.to_f32().into_iter().zip(xs) {
            let want = (input as f64) / (1.0 + (-input as f64).exp());
            let abs_err = (got as f64 - want).abs();
            let rel_err = if want.abs() > 1.0 {
                abs_err / want.abs()
            } else {
                abs_err
            };
            assert!(
                abs_err <= 1e-5 || rel_err <= 1e-5,
                "silu({input}) = {got}, want {want}, abs_err {abs_err}"
            );
        }
    }

    #[test]
    fn silu_handles_infinities_and_nan() {
        let xs = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
        let x = Owned::f32(&[3], &xs);
        let mut out = Owned::zeros_f32(&[3]);
        ActivationKernel {
            activation: Activation::Silu,
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
        let got = out.to_f32();
        assert_eq!(got[0], f32::INFINITY, "SiLU(+Inf) must be +Inf");
        assert_eq!(got[1], 0.0, "SiLU(-Inf) must be 0");
        assert!(got[2].is_nan(), "SiLU(NaN) must be NaN");
    }

    #[test]
    fn silu_f16_and_bf16_match_scalar_reference() {
        // 329 values exercise SIMD remainders and the MLAS clamp boundary.
        let mut xs = Vec::new();
        let mut value = -20.0f32;
        while value <= 20.0 {
            xs.push(value);
            value += 0.125;
        }
        xs.extend_from_slice(&[
            -1e4,
            1e4,
            -18.5,
            18.5,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NAN,
            -0.0,
        ]);
        let kernel = ActivationKernel {
            activation: Activation::Silu,
        };

        let f16_input = Owned::f16(&[xs.len()], &xs);
        let f16_reference_input = f16_input.to_f16_as_f32();
        let f16_reference_values: Vec<_> = f16_reference_input.into_iter().map(silu).collect();
        let f16_reference = Owned::f16(&[xs.len()], &f16_reference_values);
        let mut f16_output = Owned::zeros(DataType::Float16, &[xs.len()]);
        kernel
            .execute(&[f16_input.view()], &mut [f16_output.view_mut()])
            .unwrap();
        for (got, expected) in f16_output
            .to_f16_as_f32()
            .into_iter()
            .zip(f16_reference.to_f16_as_f32())
        {
            assert!(
                got == expected
                    || (got - expected).abs() <= 1e-3 * expected.abs().max(1.0)
                    || (got.is_nan() && expected.is_nan()),
                "f16 SiLU got {got}, expected {expected}"
            );
        }

        let bf16_input = Owned::bf16(&[xs.len()], &xs);
        let bf16_reference_input = bf16_input.to_bf16_as_f32();
        let bf16_reference_values: Vec<_> = bf16_reference_input.into_iter().map(silu).collect();
        let bf16_reference = Owned::bf16(&[xs.len()], &bf16_reference_values);
        let mut bf16_output = Owned::zeros(DataType::BFloat16, &[xs.len()]);
        kernel
            .execute(&[bf16_input.view()], &mut [bf16_output.view_mut()])
            .unwrap();
        for (got, expected) in bf16_output
            .to_bf16_as_f32()
            .into_iter()
            .zip(bf16_reference.to_bf16_as_f32())
        {
            assert!(
                got == expected
                    || (got - expected).abs() <= 1e-3 * expected.abs().max(1.0)
                    || (got.is_nan() && expected.is_nan()),
                "bf16 SiLU got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn silu_strided_falls_back_correctly() {
        let mut x = Owned::f32(&[2, 2], &[-2.0, -1.0, 1.0, 2.0]);
        x.strides = vec![1, 2];
        let mut out = Owned::zeros_f32(&[2, 2]);
        ActivationKernel {
            activation: Activation::Silu,
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
        let logical = [-2.0f32, 1.0, -1.0, 2.0];
        for (got, input) in out.to_f32().into_iter().zip(logical) {
            let want = input * (1.0 / (1.0 + (-input).exp()));
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }
}

/// The MLAS-backed `SiLU` path must reach the pool and must keep the exact
/// semantics of the whole-tensor correction loop it replaced.
#[cfg(all(test, feature = "mlas"))]
mod silu_parallel {
    use super::*;
    use crate::kernels::simd_activations::{PAR_MIN_LEN, parallel_dispatches};

    /// Over `PAR_MIN_LEN` so the parallel branch is eligible, over several
    /// correction blocks, and not a multiple of the block, the chunk size or
    /// the lane count — so block and chunk boundaries land mid-run.
    const N: usize = PAR_MIN_LEN + 4099;

    /// What `silu_f32_slice` did before: MLAS over the whole tensor, then one
    /// unconditional correction pass over the whole tensor.
    fn reference(input: &[f32], output: &mut [f32]) {
        mlas_sys::compute_silu(input, output);
        for (o, &i) in output.iter_mut().zip(input) {
            if !i.is_finite() || i.abs() > SILU_MLAS_SAFE_BOUND {
                *o = silu(i);
            }
        }
    }

    /// Spans MLAS's safe band, both sides of the `+/-18` correction
    /// threshold, and the special values the correction exists for. The
    /// `+/-18` boundary itself is included, since `>` makes it in-band.
    fn probe(len: usize) -> Vec<f32> {
        let specials = [
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            18.0,
            -18.0,
            18.000002,
            -18.000002,
            1e30,
            -1e30,
            f32::MAX,
            f32::MIN,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
        ];
        (0..len)
            .map(|i| {
                // Sprinkle the special values thinly and at irregular stride so
                // they land at every lane, block and chunk offset, while the
                // bulk stays in band and exercises the fast path.
                if i % 1013 == 7 {
                    specials[(i / 1013) % specials.len()]
                } else {
                    ((i % 2003) as f32 - 1000.0) / 51.0
                }
            })
            .collect()
    }

    #[test]
    fn blocked_correction_matches_the_whole_tensor_loop_bit_for_bit() {
        let x = probe(N);
        let mut want = vec![0.0f32; N];
        reference(&x, &mut want);
        let mut got = vec![0.0f32; N];
        silu_f32_slice(&x, &mut got);

        for (i, (&w, &g)) in want.iter().zip(&got).enumerate() {
            if w.is_nan() && g.is_nan() {
                continue;
            }
            assert_eq!(
                w.to_bits(),
                g.to_bits(),
                "element {i} (input {:e}) differs: whole-tensor {w:e}, blocked {g:e}",
                x[i]
            );
        }
    }

    #[test]
    fn silu_reaches_run_chunked_parallel_branch() {
        if rayon::current_num_threads() < 2 {
            eprintln!(
                "skipped: global pool is single-threaded, so run_chunked would take \
                 its serial branch and this test could not fail"
            );
            return;
        }
        assert!(
            rayon::current_thread_index().is_none(),
            "this test must run outside the pool"
        );
        let x = vec![0.5f32; N];
        let mut y = vec![0.0f32; N];

        let before = parallel_dispatches();
        silu_f32_slice(&x, &mut y);
        assert!(
            parallel_dispatches() > before,
            "SiLU did not reach run_chunked's parallel branch, so it runs \
             single-threaded no matter how large the pool is"
        );
    }

    #[test]
    fn silu_is_thread_count_invariant() {
        let x = probe(N);
        let mut serial = vec![0.0f32; N];
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool")
            .install(|| silu_f32_slice(&x, &mut serial));

        let mut parallel = vec![0.0f32; N];
        silu_f32_slice(&x, &mut parallel);

        for (i, (&s, &p)) in serial.iter().zip(&parallel).enumerate() {
            if s.is_nan() && p.is_nan() {
                continue;
            }
            assert_eq!(
                s.to_bits(),
                p.to_bits(),
                "element {i}: splitting the slice changed the result \
                 (serial {s:e}, parallel {p:e})"
            );
        }
    }
}

/// What the chunked SiLU path is worth, measured against the serial one.
///
/// `Swish` (default domain, opset 24) is the ORT-visible spelling of SiLU and
/// ORT 1.28 does implement it, so a session-level A/B is available and is
/// reported in the PR that added this. This bench measures the narrower thing
/// the change actually controls: the same kernel with the parallel split on and
/// off, in one process, alternating to cancel drift, with no session, node
/// dispatch or ORT scheduling in the way.
#[cfg(all(test, feature = "mlas"))]
mod silu_bench {
    use super::*;
    use crate::kernels::simd_activations::serial_scope;
    use std::time::Instant;

    #[test]
    #[ignore = "benchmark; run explicitly with --release --ignored --nocapture"]
    fn silu_parallel_vs_serial() {
        println!(
            "n\tserial_us\tparallel_us\tspeedup\tthreads={}",
            rayon::current_num_threads()
        );
        for n in [1usize << 20, 1 << 22, 1 << 24] {
            let x: Vec<f32> = (0..n)
                .map(|i| ((i % 2003) as f32 - 1000.0) / 51.0)
                .collect();
            let mut y = vec![0.0f32; n];
            // Best-of, alternating, so a scheduling hiccup cannot land on one
            // arm only.
            let (mut ser, mut par) = (f64::MAX, f64::MAX);
            for _ in 0..9 {
                let t = Instant::now();
                serial_scope(|| silu_f32_slice(&x, &mut y));
                ser = ser.min(t.elapsed().as_secs_f64());
                let t = Instant::now();
                silu_f32_slice(&x, &mut y);
                par = par.min(t.elapsed().as_secs_f64());
            }
            let (s, p) = (ser * 1e6, par * 1e6);
            println!("{n}\t{s:.1}\t{p:.1}\t{:.2}x", s / p);
        }
    }
}
