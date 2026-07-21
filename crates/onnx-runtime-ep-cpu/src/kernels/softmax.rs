//! `Softmax`: numerically stable softmax for f32 (`docs/ORT2.md` §4.4).
//!
//! ## Two opset semantics (both implemented, selected by opset)
//!
//! ONNX changed `Softmax`'s definition at opset 13:
//!
//! * **opset ≥ 13** ([`SoftmaxKernel`] with `coerce_2d = false`): `axis` is the
//!   single reduction axis — softmax is normalized along that one axis.
//! * **opset ≤ 12** ([`SoftmaxKernel`] with `coerce_2d = true`): the input is
//!   coerced to a 2D matrix `[d_0·…·d_{axis-1}, d_axis·…·d_{n-1}]` and softmax
//!   is taken over each row (the *entire* flattened trailing block), not just
//!   the `axis` dimension.
//!
//! The two definitions coincide exactly when `axis` is the last dimension
//! (every trailing block is then a single axis). They diverge for `axis != last`,
//! so applying the opset-13 kernel to an opset-12 node silently produced wrong
//! results — the advisory this kernel now closes. The registry keys the two
//! factories at `since_version` 1 (legacy) and 13 (per-axis); the provider's
//! opset-aware lookup selects the correct one.
//!
//! Stability: each reduction slice subtracts its max before `exp`, so large
//! logits (e.g. masked-attention `-inf`/`1e9` fills) never overflow.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::{check_arity, to_dense_f32, write_dense_f32};
use crate::strided::numel;

/// f32 Softmax kernel carrying the raw `axis` attribute and the opset semantics.
pub struct SoftmaxKernel {
    axis: i64,
    /// `true` for opset ≤ 12 (coerce-to-2D over the flattened trailing block);
    /// `false` for opset ≥ 13 (normalize over the single `axis`).
    coerce_2d: bool,
}

/// Factory for the opset ≥ 13 per-axis `Softmax` (`axis` default -1).
pub struct SoftmaxFactory;

/// Factory for the legacy opset ≤ 12 coerce-to-2D `Softmax` (`axis` default 1).
pub struct SoftmaxLegacyFactory;

impl KernelFactory for SoftmaxFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let axis = node.attr("axis").and_then(|a| a.as_int()).unwrap_or(-1);
        Ok(Box::new(SoftmaxKernel {
            axis,
            coerce_2d: false,
        }))
    }
}

impl KernelFactory for SoftmaxLegacyFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let axis = node.attr("axis").and_then(|a| a.as_int()).unwrap_or(1);
        Ok(Box::new(SoftmaxKernel {
            axis,
            coerce_2d: true,
        }))
    }
}

/// Softmax `n` independent contiguous rows of `axis_dim` elements each over the
/// stride-`inner` interleaving: element `a` of slice `(o, i)` lives at
/// `o·axis_dim·inner + a·inner + i`. With `inner == 1` this is a plain
/// row-major softmax; with `inner > 1` it reduces along an interior axis.
///
/// Shared with the `FusedAttention` kernel (`kernels::fused_attention`), which
/// softmaxes the last axis of the scaled/masked scores as its middle stage —
/// reusing this single numerically-stable implementation instead of duplicating
/// the max-subtract/exp/normalize loop.
pub(crate) fn softmax_slices(
    x: &[f32],
    out: &mut [f32],
    outer: usize,
    axis_dim: usize,
    inner: usize,
) {
    for o in 0..outer {
        for i in 0..inner {
            let base = o * axis_dim * inner + i;
            let mut max = f32::NEG_INFINITY;
            for a in 0..axis_dim {
                let v = x[base + a * inner];
                if v > max {
                    max = v;
                }
            }
            let mut sum = 0.0f32;
            for a in 0..axis_dim {
                let e = (x[base + a * inner] - max).exp();
                out[base + a * inner] = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            for a in 0..axis_dim {
                out[base + a * inner] *= inv;
            }
        }
    }
}

impl Kernel for SoftmaxKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Softmax", inputs, outputs, 1, 1, 1)?;
        let x = to_dense_f32(&inputs[0])?;
        let shape = inputs[0].shape;
        let rank = shape.len();
        if rank == 0 {
            return Err(EpError::KernelFailed(
                "Softmax: input must have rank >= 1".into(),
            ));
        }
        let axis = if self.axis < 0 {
            self.axis + rank as i64
        } else {
            self.axis
        };
        if axis < 0 || axis as usize >= rank {
            return Err(EpError::KernelFailed(format!(
                "Softmax: axis {} out of range for rank {rank}",
                self.axis
            )));
        }
        let axis = axis as usize;
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            let slices = if self.coerce_2d {
                crate::trace::product(shape[..axis].iter().copied())
            } else {
                crate::trace::product(shape[..axis].iter().chain(&shape[axis + 1..]).copied())
            };
            // Per element: subtract, exp, sum and normalization multiply; one
            // reciprocal per reduction slice. Max comparisons are not FLOPs.
            (inputs[0].numel() as u64)
                .saturating_mul(4)
                .saturating_add(slices)
        });

        let mut out = vec![0.0f32; numel(shape)];
        if self.coerce_2d {
            // opset ≤ 12: coerce to 2D `[d_0·…·d_{axis-1}, d_axis·…·d_{n-1}]`
            // and softmax each row over the whole flattened trailing block.
            let rows: usize = shape[..axis].iter().product();
            let cols: usize = shape[axis..].iter().product();
            // Trailing block is contiguous, so `inner == 1`.
            softmax_slices(&x, &mut out, rows, cols, 1);
        } else {
            // opset ≥ 13: normalize over the single `axis`, viewing the tensor
            // as `[outer, axis_dim, inner]`.
            let axis_dim = shape[axis];
            let outer: usize = shape[..axis].iter().product();
            let inner: usize = shape[axis + 1..].iter().product();
            softmax_slices(&x, &mut out, outer, axis_dim, inner);
        }
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

    fn run(axis: i64, x: &Owned, out: &mut Owned) {
        SoftmaxKernel {
            axis,
            coerce_2d: false,
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
    }

    fn run_legacy(axis: i64, x: &Owned, out: &mut Owned) {
        SoftmaxKernel {
            axis,
            coerce_2d: true,
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
    }

    fn approx(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-6, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn softmax_last_axis_2d() {
        // [2,3], axis 1. Row [1,2,3]: softmax = [0.09003, 0.24473, 0.66524].
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 1., 2., 3.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        run(1, &x, &mut out);
        let e = [0.090_030_57, 0.244_728_47, 0.665_240_96];
        let mut want = e.to_vec();
        want.extend_from_slice(&e);
        approx(&out.to_f32(), &want);
        // Each row sums to 1.
        let r = out.to_f32();
        assert!((r[0] + r[1] + r[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_axis0() {
        // [2,2], axis 0 reduces over rows (column-wise softmax).
        let x = Owned::f32(&[2, 2], &[1., 2., 1., 2.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        run(0, &x, &mut out);
        // Each column has equal entries -> 0.5 each.
        approx(&out.to_f32(), &[0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn softmax_negative_axis() {
        let x = Owned::f32(&[1, 3], &[0., 0., 0.]);
        let mut out = Owned::zeros_f32(&[1, 3]);
        run(-1, &x, &mut out);
        approx(&out.to_f32(), &[1. / 3., 1. / 3., 1. / 3.]);
    }

    #[test]
    fn softmax_numerically_stable_large_values() {
        // Without max-subtraction these overflow to inf; the result must stay
        // finite and (nearly) one-hot on the largest logit.
        let x = Owned::f32(&[1, 3], &[1000.0, 1001.0, 1002.0]);
        let mut out = Owned::zeros_f32(&[1, 3]);
        run(1, &x, &mut out);
        let r = out.to_f32();
        assert!(r.iter().all(|v| v.is_finite()));
        assert!((r.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // Same gaps as [0,1,2] -> [0.09003, 0.24473, 0.66524].
        approx(&r, &[0.090_030_57, 0.244_728_47, 0.665_240_96]);
    }

    #[test]
    fn softmax_batched_last_axis_4d() {
        // [1,1,2,2] with axis -1 — the BERT attention shape pattern.
        let x = Owned::f32(&[1, 1, 2, 2], &[1., 2., 3., 4.]);
        let mut out = Owned::zeros_f32(&[1, 1, 2, 2]);
        run(-1, &x, &mut out);
        let r = out.to_f32();
        // row [1,2] and row [3,4] both softmax to [0.26894, 0.73106].
        approx(&r, &[0.268_941_43, 0.731_058_6, 0.268_941_43, 0.731_058_6]);
    }

    #[test]
    fn softmax_opset13_default_axis_is_last_dimension() {
        let x = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        SoftmaxFactory
            .create(
                &Node::new(
                    onnx_runtime_ir::NodeId(0),
                    "Softmax",
                    Vec::new(),
                    Vec::new(),
                ),
                &[],
            )
            .unwrap()
            .execute(&[x.view()], &mut [out.view_mut()])
            .unwrap();
        approx(
            &out.to_f32(),
            &[0.268_941_43, 0.731_058_6, 0.268_941_43, 0.731_058_6],
        );
    }

    #[test]
    fn softmax_opset12_axis0_coerces_to_single_row() {
        // [2,2], axis 0. opset≤12 coerces to `[1, 4]` (rows before axis 0 = 1)
        // and softmaxes the ENTIRE flattened tensor as one row — unlike the
        // opset-13 per-axis (column-wise) definition.
        let x = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        run_legacy(0, &x, &mut out);
        let r = out.to_f32();
        approx(&r, &[0.032_058_6, 0.087_144_32, 0.236_882_82, 0.643_914_2]);
        // The whole tensor is one softmax row → all elements sum to 1.
        assert!((r.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_opset12_differs_from_opset13_when_axis_not_last() {
        // Same [2,2] axis-0 input: the opset-13 per-axis kernel normalizes each
        // column independently, so the two definitions must disagree here.
        let x = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let mut per_axis = Owned::zeros_f32(&[2, 2]);
        let mut legacy = Owned::zeros_f32(&[2, 2]);
        run(0, &x, &mut per_axis);
        run_legacy(0, &x, &mut legacy);
        // opset-13: each column [1,3] and [2,4] → [0.11920, 0.88080].
        approx(
            &per_axis.to_f32(),
            &[0.119_202_92, 0.119_202_92, 0.880_797_1, 0.880_797_1],
        );
        // The two kernels genuinely diverge (the bug this fix closes).
        let (a, b) = (per_axis.to_f32(), legacy.to_f32());
        assert!(a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-3));
    }

    #[test]
    fn softmax_opset12_matches_opset13_on_last_axis() {
        // When axis == last dim, the coerce-to-2D and per-axis definitions
        // coincide exactly — the BERT-attention case.
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let mut per_axis = Owned::zeros_f32(&[2, 3]);
        let mut legacy = Owned::zeros_f32(&[2, 3]);
        run(-1, &x, &mut per_axis);
        run_legacy(-1, &x, &mut legacy);
        approx(&per_axis.to_f32(), &legacy.to_f32());
    }
}
