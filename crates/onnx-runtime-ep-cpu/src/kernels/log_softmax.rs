//! Numerically stable `LogSoftmax` with ONNX-versioned axis semantics.

use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::check_arity;
use crate::strided::numel;

/// `LogSoftmax` kernel carrying its axis and whether it uses the pre-opset-13
/// coerce-to-2D definition.
pub struct LogSoftmaxKernel {
    axis: i64,
    coerce_2d: bool,
}

/// Factory for opset ≥13, where the default axis is -1 and normalization is
/// performed across the one selected axis.
pub struct LogSoftmaxFactory;

/// Factory for opset ≤12, where the default axis is 1 and the trailing axes are
/// flattened into a single row dimension.
pub struct LogSoftmaxLegacyFactory;

impl KernelFactory for LogSoftmaxFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(LogSoftmaxKernel {
            axis: node.attr("axis").and_then(|a| a.as_int()).unwrap_or(-1),
            coerce_2d: false,
        }))
    }
}

impl KernelFactory for LogSoftmaxLegacyFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(LogSoftmaxKernel {
            axis: node.attr("axis").and_then(|a| a.as_int()).unwrap_or(1),
            coerce_2d: true,
        }))
    }
}

/// Write `x - logsumexp(x)` over strided reduction slices. Subtracting the
/// slice maximum before exponentiation makes the logsumexp calculation stable.
fn log_softmax_slices(x: &[f32], out: &mut [f32], outer: usize, axis_dim: usize, inner: usize) {
    for o in 0..outer {
        for i in 0..inner {
            let base = o * axis_dim * inner + i;
            let mut max = f32::NEG_INFINITY;
            for a in 0..axis_dim {
                max = max.max(x[base + a * inner]);
            }
            let mut sum = 0.0;
            for a in 0..axis_dim {
                sum += (x[base + a * inner] - max).exp();
            }
            let log_sum = sum.ln();
            for a in 0..axis_dim {
                out[base + a * inner] = (x[base + a * inner] - max) - log_sum;
            }
        }
    }
}

impl Kernel for LogSoftmaxKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("LogSoftmax", inputs, outputs, 1, 1, 1)?;
        let x = to_dense_f32_widen("LogSoftmax", &inputs[0])?;
        let shape = inputs[0].shape;
        let rank = shape.len();
        if rank == 0 {
            return Err(EpError::KernelFailed(
                "LogSoftmax: input must have rank >= 1".into(),
            ));
        }
        let axis = if self.axis < 0 {
            self.axis + rank as i64
        } else {
            self.axis
        };
        if axis < 0 || axis as usize >= rank {
            return Err(EpError::KernelFailed(format!(
                "LogSoftmax: axis {} out of range for rank {rank}",
                self.axis
            )));
        }
        let axis = axis as usize;
        let mut out = vec![0.0; numel(shape)];
        if self.coerce_2d {
            let rows = shape[..axis].iter().product();
            let cols = shape[axis..].iter().product();
            log_softmax_slices(&x, &mut out, rows, cols, 1);
        } else {
            let outer = shape[..axis].iter().product();
            let inner = shape[axis + 1..].iter().product();
            log_softmax_slices(&x, &mut out, outer, shape[axis], inner);
        }
        write_dense_f32_narrow("LogSoftmax", &mut outputs[0], &out)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    fn run(axis: i64, coerce_2d: bool, x: &Owned, out: &mut Owned) {
        LogSoftmaxKernel { axis, coerce_2d }
            .execute(&[x.view()], &mut [out.view_mut()])
            .unwrap();
    }

    fn approx(got: &[f32], expected: &[f32]) {
        assert_eq!(got.len(), expected.len());
        for (&actual, &want) in got.iter().zip(expected) {
            assert!((actual - want).abs() < 1e-6, "{got:?} vs {expected:?}");
        }
    }

    #[test]
    fn log_softmax_opset13_last_axis_has_expected_values_and_normalizes() {
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 1., 2., 3.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        run(-1, false, &x, &mut out);
        let expected = [-2.407_606, -1.407_606, -0.407_605_95];
        let result = out.to_f32();
        approx(&result[..3], &expected);
        approx(&result[3..], &expected);
        for row in result.chunks_exact(3) {
            assert!((row.iter().map(|v| v.exp()).sum::<f32>() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn log_softmax_opset11_axis_one_coerces_trailing_dimensions() {
        let x = Owned::f32(&[2, 2, 2], &[1., 2., 3., 4., 1., 2., 3., 4.]);
        let mut legacy = Owned::zeros_f32(&[2, 2, 2]);
        let mut modern = Owned::zeros_f32(&[2, 2, 2]);
        run(1, true, &x, &mut legacy);
        run(1, false, &x, &mut modern);
        // Legacy treats each [2,2] block as one 4-element row.
        approx(
            &legacy.to_f32()[..4],
            &[-3.440_189_6, -2.440_189_6, -1.440_189_7, -0.440_189_7],
        );
        // Opset 13 reduces only axis 1, leaving the final dimension independent.
        approx(
            &modern.to_f32()[..4],
            &[-2.126_928, -2.126_928, -0.126_928_05, -0.126_928_05],
        );
    }

    #[test]
    fn log_softmax_is_translation_invariant_for_large_logits() {
        let small = Owned::f32(&[2], &[0., 1.]);
        let shifted = Owned::f32(&[2], &[1_000_000., 1_000_001.]);
        let equal_large = Owned::f32(&[2], &[1e10, 1e10]);
        let mut small_out = Owned::zeros_f32(&[2]);
        let mut shifted_out = Owned::zeros_f32(&[2]);
        let mut equal_large_out = Owned::zeros_f32(&[2]);

        run(-1, false, &small, &mut small_out);
        run(-1, false, &shifted, &mut shifted_out);
        run(-1, false, &equal_large, &mut equal_large_out);

        approx(&shifted_out.to_f32(), &small_out.to_f32());
        approx(&equal_large_out.to_f32(), &[-core::f32::consts::LN_2; 2]);
        for output in [&shifted_out, &equal_large_out] {
            assert!((output.to_f32().iter().map(|v| v.exp()).sum::<f32>() - 1.0).abs() < 1e-6);
        }
    }
    #[test]
    fn log_softmax_bf16_matches_widened_f32_reference() {
        let values = [-10.0, 0.0, 1.0, 80.0, -80.0, 2.0];
        let x = Owned::bf16(&[2, 3], &values);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[2, 3]);
        run(-1, false, &x, &mut out);
        let rounded = x.to_bf16_as_f32();
        let mut reference = vec![0.0; rounded.len()];
        log_softmax_slices(&rounded, &mut reference, 2, 3, 1);
        let expected: Vec<_> = reference
            .into_iter()
            .map(half::bf16::from_f32)
            .map(half::bf16::to_f32)
            .collect();
        assert_eq!(out.to_bf16_as_f32(), expected);
    }
}
