//! Additional f32 reduction kernels: `ReduceSum`, `ReduceMax`, `ReduceMin`,
//! `ReduceProd`, `ReduceSumSquare`, `ReduceL2` (`docs/ORT2.md` §4.4).
//!
//! These complement [`reduce::ReduceMeanKernel`](super::reduce) and share its
//! reduce-walk structure, but add support for the **modern input signature**:
//! at opset 13 (`ReduceSum`) / opset 18 (the others) the reduced axes moved
//! from the `axes` *attribute* to an optional second *input* tensor. This kernel
//! resolves axes from input 1 when present, falling back to the attribute, then
//! to "reduce all axes".
//!
//! `noop_with_empty_axes` (opset 18, default 0) selects identity vs reduce-all
//! when the axes set is explicitly empty.
//!
//! The output view's own shape (keepdims-aware) governs the write; the produced
//! dense buffer matches it element-for-element because reduced axes contribute
//! either a retained size-1 dim (keepdims) or nothing (squeezed).

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Node, compute_contiguous_strides};

use super::{check_arity, to_dense_f32, to_dense_i64, write_dense_f32};
use crate::strided::{next_index, numel};

/// The reduction to apply over the selected axes.
#[derive(Clone, Copy)]
enum ReduceOp {
    Sum,
    Max,
    Min,
    Prod,
    SumSquare,
    L2,
}

impl ReduceOp {
    fn name(self) -> &'static str {
        match self {
            ReduceOp::Sum => "ReduceSum",
            ReduceOp::Max => "ReduceMax",
            ReduceOp::Min => "ReduceMin",
            ReduceOp::Prod => "ReduceProd",
            ReduceOp::SumSquare => "ReduceSumSquare",
            ReduceOp::L2 => "ReduceL2",
        }
    }

    /// The identity/accumulator seed for an empty reduction group.
    fn init(self) -> f32 {
        match self {
            ReduceOp::Sum | ReduceOp::SumSquare | ReduceOp::L2 => 0.0,
            ReduceOp::Prod => 1.0,
            ReduceOp::Max => f32::NEG_INFINITY,
            ReduceOp::Min => f32::INFINITY,
        }
    }

    /// Fold accumulator `acc` with a new element `x`.
    fn fold(self, acc: f32, x: f32) -> f32 {
        match self {
            ReduceOp::Sum => acc + x,
            ReduceOp::Prod => acc * x,
            ReduceOp::SumSquare | ReduceOp::L2 => acc + x * x,
            // Max/Min propagate NaN (numpy semantics) — Rust's f32::max/min
            // suppress it, so guard explicitly.
            ReduceOp::Max => {
                if acc.is_nan() || x.is_nan() {
                    f32::NAN
                } else {
                    acc.max(x)
                }
            }
            ReduceOp::Min => {
                if acc.is_nan() || x.is_nan() {
                    f32::NAN
                } else {
                    acc.min(x)
                }
            }
        }
    }

    /// Final map applied to each accumulated group (only `ReduceL2` is nonlinear).
    fn finish(self, acc: f32) -> f32 {
        match self {
            ReduceOp::L2 => acc.sqrt(),
            _ => acc,
        }
    }
}

/// f32 reduction kernel carrying the op, the attribute `axes` (opset < 13/18),
/// `keepdims` and `noop_with_empty_axes`.
pub struct ReduceKernel {
    op: ReduceOp,
    axes_attr: Option<Vec<i64>>,
    keepdims: bool,
    noop_with_empty_axes: bool,
}

macro_rules! reduce_factory {
    ($factory:ident, $variant:expr) => {
        /// Factory reading `axes` (optional attribute), `keepdims` (default 1)
        /// and `noop_with_empty_axes` (default 0).
        pub struct $factory;
        impl KernelFactory for $factory {
            fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
                let axes_attr = node
                    .attr("axes")
                    .and_then(|a| a.as_ints())
                    .map(<[i64]>::to_vec);
                let keepdims = node.attr("keepdims").and_then(|a| a.as_int()).unwrap_or(1) != 0;
                let noop_with_empty_axes = node
                    .attr("noop_with_empty_axes")
                    .and_then(|a| a.as_int())
                    .unwrap_or(0)
                    != 0;
                Ok(Box::new(ReduceKernel {
                    op: $variant,
                    axes_attr,
                    keepdims,
                    noop_with_empty_axes,
                }))
            }
        }
    };
}

reduce_factory!(ReduceSumFactory, ReduceOp::Sum);
reduce_factory!(ReduceMaxFactory, ReduceOp::Max);
reduce_factory!(ReduceMinFactory, ReduceOp::Min);
reduce_factory!(ReduceProdFactory, ReduceOp::Prod);
reduce_factory!(ReduceSumSquareFactory, ReduceOp::SumSquare);
reduce_factory!(ReduceL2Factory, ReduceOp::L2);

impl Kernel for ReduceKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(self.op.name(), inputs, outputs, 1, 2, 1)?;
        let x = to_dense_f32(&inputs[0])?;
        let in_shape = inputs[0].shape;
        let rank = in_shape.len();

        // Resolve the raw axes list: input 1 (opset 13/18+) takes precedence
        // over the attribute; both absent means "reduce all axes" (unless
        // noop_with_empty_axes selects identity for an explicitly-empty set).
        let axes_raw: Option<Vec<i64>> = if inputs.len() == 2 {
            Some(to_dense_i64(&inputs[1])?)
        } else {
            self.axes_attr.clone()
        };

        let mut reduce = vec![false; rank];
        match &axes_raw {
            // Explicitly empty axes. With `noop_with_empty_axes` this reduces no
            // axis (each element is its own group); otherwise it reduces all.
            // Note "no reduction" is NOT a plain identity: the per-element pre-map
            // (square for SumSquare/L2) and post-map (sqrt for L2) still apply, so
            // we fall through to the normal loop with `reduce` all-false rather
            // than copying the input.
            Some(a) if a.is_empty() => {
                if !self.noop_with_empty_axes {
                    reduce.iter_mut().for_each(|r| *r = true);
                }
            }
            Some(axes) => {
                for &a in axes {
                    let ax = if a < 0 { a + rank as i64 } else { a };
                    if ax < 0 || ax as usize >= rank {
                        return Err(EpError::KernelFailed(format!(
                            "{}: axis {a} out of range for rank {rank}",
                            self.op.name()
                        )));
                    }
                    reduce[ax as usize] = true;
                }
            }
            None => {
                if !self.noop_with_empty_axes {
                    reduce.iter_mut().for_each(|r| *r = true);
                }
            }
        }

        let kept_shape: Vec<usize> = (0..rank)
            .filter(|&d| !reduce[d])
            .map(|d| in_shape[d])
            .collect();
        let kept_count = numel(&kept_shape);

        let in_strides = compute_contiguous_strides(in_shape);
        let kept_out_strides = compute_contiguous_strides(&kept_shape);

        let mut acc = vec![self.op.init(); kept_count.max(1)];
        if numel(in_shape) > 0 {
            let mut idx = vec![0usize; rank];
            loop {
                let mut in_off = 0usize;
                let mut out_off = 0usize;
                let mut kept_axis = 0usize;
                for d in 0..rank {
                    in_off += in_strides[d] as usize * idx[d];
                    if !reduce[d] {
                        out_off += kept_out_strides[kept_axis] as usize * idx[d];
                        kept_axis += 1;
                    }
                }
                acc[out_off] = self.op.fold(acc[out_off], x[in_off]);
                if !next_index(in_shape, &mut idx) {
                    break;
                }
            }
        }

        let out: Vec<f32> = acc.iter().map(|&a| self.op.finish(a)).collect();
        let _ = self.keepdims;
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

    fn run_attr(op: ReduceOp, axes: Option<Vec<i64>>, x: &Owned, out: &mut Owned) {
        ReduceKernel {
            op,
            axes_attr: axes,
            keepdims: true,
            noop_with_empty_axes: false,
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
    }

    fn run_axes_input(op: ReduceOp, x: &Owned, axes: &Owned, out: &mut Owned) {
        ReduceKernel {
            op,
            axes_attr: None,
            keepdims: true,
            noop_with_empty_axes: false,
        }
        .execute(&[x.view(), axes.view()], &mut [out.view_mut()])
        .unwrap();
    }

    #[test]
    fn sum_axis1() {
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let mut out = Owned::zeros_f32(&[2, 1]);
        run_attr(ReduceOp::Sum, Some(vec![1]), &x, &mut out);
        assert_eq!(out.to_f32(), vec![6., 15.]);
    }

    #[test]
    fn sum_axes_from_input() {
        // Modern opset-13 signature: axes come as input 1.
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let axes = Owned::i64(&[1], &[0]);
        let mut out = Owned::zeros_f32(&[1, 3]);
        run_axes_input(ReduceOp::Sum, &x, &axes, &mut out);
        assert_eq!(out.to_f32(), vec![5., 7., 9.]);
    }

    #[test]
    fn max_min_negative_axis() {
        let x = Owned::f32(&[2, 3], &[1., 9., 3., 4., 5., 0.]);
        let mut out = Owned::zeros_f32(&[2, 1]);
        run_attr(ReduceOp::Max, Some(vec![-1]), &x, &mut out);
        assert_eq!(out.to_f32(), vec![9., 5.]);
        run_attr(ReduceOp::Min, Some(vec![-1]), &x, &mut out);
        assert_eq!(out.to_f32(), vec![1., 0.]);
    }

    #[test]
    fn prod_and_sumsquare() {
        let x = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let mut out = Owned::zeros_f32(&[1, 2]);
        run_attr(ReduceOp::Prod, Some(vec![0]), &x, &mut out);
        assert_eq!(out.to_f32(), vec![3., 8.]);
        run_attr(ReduceOp::SumSquare, Some(vec![0]), &x, &mut out);
        // col0: 1+9=10, col1: 4+16=20
        assert_eq!(out.to_f32(), vec![10., 20.]);
    }

    #[test]
    fn l2_norm() {
        let x = Owned::f32(&[1, 2], &[3., 4.]);
        let mut out = Owned::zeros_f32(&[1, 1]);
        run_attr(ReduceOp::L2, Some(vec![1]), &x, &mut out);
        assert_eq!(out.to_f32(), vec![5.0]);
    }

    #[test]
    fn reduce_all_default() {
        let x = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let mut out = Owned::zeros_f32(&[1, 1]);
        run_attr(ReduceOp::Sum, None, &x, &mut out);
        assert_eq!(out.to_f32(), vec![10.]);
    }

    #[test]
    fn empty_axes_noop_applies_per_element_map() {
        // noop_with_empty_axes reduces NO axis, but the per-element pre-map still
        // applies: ReduceSumSquare on [1,2,3] with empty axes + noop -> [1,4,9].
        let x = Owned::f32(&[3], &[1., 2., 3.]);
        let mut out = Owned::zeros_f32(&[3]);
        ReduceKernel {
            op: ReduceOp::SumSquare,
            axes_attr: None,
            keepdims: true,
            noop_with_empty_axes: true,
        }
        .execute(
            &[x.view(), Owned::i64(&[0], &[]).view()],
            &mut [out.view_mut()],
        )
        .unwrap();
        assert_eq!(out.to_f32(), vec![1., 4., 9.]);
    }

    #[test]
    fn empty_axes_noop_sum_is_identity() {
        let x = Owned::f32(&[3], &[1., 2., 3.]);
        let mut out = Owned::zeros_f32(&[3]);
        ReduceKernel {
            op: ReduceOp::Sum,
            axes_attr: None,
            keepdims: true,
            noop_with_empty_axes: true,
        }
        .execute(
            &[x.view(), Owned::i64(&[0], &[]).view()],
            &mut [out.view_mut()],
        )
        .unwrap();
        assert_eq!(out.to_f32(), vec![1., 2., 3.]);
    }
}
