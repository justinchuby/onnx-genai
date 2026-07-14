//! `Softmax`: numerically stable softmax over a single axis for f32
//! (`docs/ORT2.md` §4.4).
//!
//! ## Opset-12 axis semantics (decision)
//!
//! This kernel treats `axis` (default 1, negatives allowed) as **the single
//! reduction axis** — softmax is normalized along that one axis, matching the
//! opset-13+ definition. It deliberately does **not** implement the legacy
//! opset-<13 "coerce to 2D `[d0..axis, axis..]`" behavior. For the Phase-1
//! BERT target every `Softmax` reduces over the last axis, where the coerce and
//! per-axis definitions coincide exactly, so this matches ORT for the milestone
//! model. (See the decision note for the caveat about non-last-axis opset-12
//! graphs.)
//!
//! Stability: each reduction slice subtracts its max before `exp`, so large
//! logits (e.g. masked-attention `-inf`/`1e9` fills) never overflow.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::{check_arity, to_dense_f32, write_dense_f32};
use crate::strided::numel;

/// f32 Softmax kernel carrying the raw `axis` attribute.
pub struct SoftmaxKernel {
    axis: i64,
}

/// Factory reading `axis` (default 1).
pub struct SoftmaxFactory;

impl KernelFactory for SoftmaxFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let axis = node.attr("axis").and_then(|a| a.as_int()).unwrap_or(1);
        Ok(Box::new(SoftmaxKernel { axis }))
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

        // View the tensor as [outer, axis_dim, inner] with the reduction over
        // the middle axis; every (outer, inner) pair is an independent slice.
        let axis_dim = shape[axis];
        let outer: usize = shape[..axis].iter().product();
        let inner: usize = shape[axis + 1..].iter().product();

        let mut out = vec![0.0f32; numel(shape)];
        for o in 0..outer {
            for i in 0..inner {
                let base = o * axis_dim * inner + i;
                // max over the slice for numerical stability.
                let mut max = f32::NEG_INFINITY;
                for a in 0..axis_dim {
                    let v = x[base + a * inner];
                    if v > max {
                        max = v;
                    }
                }
                // exp(x - max) and running sum.
                let mut sum = 0.0f32;
                for a in 0..axis_dim {
                    let e = (x[base + a * inner] - max).exp();
                    out[base + a * inner] = e;
                    sum += e;
                }
                // normalize.
                let inv = 1.0 / sum;
                for a in 0..axis_dim {
                    out[base + a * inner] *= inv;
                }
            }
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
        SoftmaxKernel { axis }
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
}
