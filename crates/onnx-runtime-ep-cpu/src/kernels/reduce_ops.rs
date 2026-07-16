//! Additional f32 reduction kernels: `ReduceSum`, `ReduceMax`, `ReduceMin`,
//! `ReduceProd`, `ReduceSumSquare`, `ReduceL1`, `ReduceL2`, `ReduceLogSum`, and
//! `ReduceLogSumExp` (`docs/ORT2.md` §4.4).
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

use super::{check_arity, to_dense_f32, to_dense_i64, write_dense_bytes, write_dense_f32};
use crate::strided::{next_index, numel};

/// The reduction to apply over the selected axes.
#[derive(Clone, Copy)]
enum ReduceOp {
    Sum,
    Max,
    Min,
    Prod,
    SumSquare,
    L1,
    L2,
    LogSum,
    LogSumExp,
}

impl ReduceOp {
    fn name(self) -> &'static str {
        match self {
            ReduceOp::Sum => "ReduceSum",
            ReduceOp::Max => "ReduceMax",
            ReduceOp::Min => "ReduceMin",
            ReduceOp::Prod => "ReduceProd",
            ReduceOp::SumSquare => "ReduceSumSquare",
            ReduceOp::L1 => "ReduceL1",
            ReduceOp::L2 => "ReduceL2",
            ReduceOp::LogSum => "ReduceLogSum",
            ReduceOp::LogSumExp => "ReduceLogSumExp",
        }
    }

    /// The identity/accumulator seed for an empty reduction group.
    fn init(self) -> f32 {
        match self {
            ReduceOp::Sum
            | ReduceOp::SumSquare
            | ReduceOp::L1
            | ReduceOp::L2
            | ReduceOp::LogSum
            | ReduceOp::LogSumExp => 0.0,
            ReduceOp::Prod => 1.0,
            ReduceOp::Max => f32::NEG_INFINITY,
            ReduceOp::Min => f32::INFINITY,
        }
    }

    /// Fold accumulator `acc` with a new element `x`.
    fn fold(self, acc: f32, x: f32) -> f32 {
        match self {
            ReduceOp::Sum | ReduceOp::LogSum => acc + x,
            ReduceOp::Prod => acc * x,
            ReduceOp::SumSquare | ReduceOp::L2 => acc + x * x,
            ReduceOp::L1 => acc + x.abs(),
            // This is replaced by the stable two-accumulator path in `execute`.
            ReduceOp::LogSumExp => acc + x.exp(),
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

    /// Final map applied to each accumulated group.
    fn finish(self, acc: f32) -> f32 {
        match self {
            ReduceOp::L2 => acc.sqrt(),
            ReduceOp::LogSum | ReduceOp::LogSumExp => acc.ln(),
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
reduce_factory!(ReduceL1Factory, ReduceOp::L1);
reduce_factory!(ReduceL2Factory, ReduceOp::L2);
reduce_factory!(ReduceLogSumFactory, ReduceOp::LogSum);
reduce_factory!(ReduceLogSumExpFactory, ReduceOp::LogSumExp);

impl Kernel for ReduceKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(self.op.name(), inputs, outputs, 1, 2, 1)?;
        if matches!(self.op, ReduceOp::Sum) && inputs[0].dtype == onnx_runtime_ir::DataType::Int64 {
            return self.execute_i64_sum(inputs, &mut outputs[0]);
        }
        let x = to_dense_f32(&inputs[0])?;
        let in_shape = inputs[0].shape;
        let rank = in_shape.len();
        let reduce = self.resolve_axes(inputs, rank)?;

        let kept_shape: Vec<usize> = (0..rank)
            .filter(|&d| !reduce[d])
            .map(|d| in_shape[d])
            .collect();
        let kept_count = numel(&kept_shape);

        let in_strides = compute_contiguous_strides(in_shape);
        let kept_out_strides = compute_contiguous_strides(&kept_shape);

        let mut acc = vec![self.op.init(); kept_count.max(1)];
        // `log(sum(exp(x)))` must avoid overflowing for large finite inputs.
        // Track a running maximum and the sum in that maximum's exponent frame.
        let mut logsumexp_max = vec![f32::NEG_INFINITY; kept_count.max(1)];
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
                if matches!(self.op, ReduceOp::LogSumExp) {
                    let value = x[in_off];
                    let max = logsumexp_max[out_off];
                    if value.is_nan() || max.is_nan() {
                        logsumexp_max[out_off] = f32::NAN;
                        acc[out_off] = f32::NAN;
                    } else if max == f32::NEG_INFINITY {
                        acc[out_off] = 1.0;
                        logsumexp_max[out_off] = value;
                    } else if max == f32::INFINITY {
                        // Avoid computing `inf - inf` for another infinite
                        // value. Any group containing +inf reduces to +inf.
                    } else if value > max {
                        acc[out_off] = acc[out_off] * (max - value).exp() + 1.0;
                        logsumexp_max[out_off] = value;
                    } else {
                        acc[out_off] += (value - max).exp();
                    }
                } else {
                    acc[out_off] = self.op.fold(acc[out_off], x[in_off]);
                }
                if !next_index(in_shape, &mut idx) {
                    break;
                }
            }
        }

        let out: Vec<f32> = if matches!(self.op, ReduceOp::LogSumExp) {
            acc.iter()
                .zip(&logsumexp_max)
                .map(|(&sum, &max)| max + sum.ln())
                .collect()
        } else {
            acc.iter().map(|&a| self.op.finish(a)).collect()
        };
        let _ = self.keepdims;
        write_dense_f32(&mut outputs[0], &out)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

impl ReduceKernel {
    fn resolve_axes(&self, inputs: &[TensorView], rank: usize) -> Result<Vec<bool>> {
        let axes_raw = if inputs.len() == 2 && !inputs[1].is_absent() {
            Some(to_dense_i64(&inputs[1])?)
        } else {
            self.axes_attr.clone()
        };
        let mut reduce = vec![false; rank];
        match &axes_raw {
            Some(axes) if axes.is_empty() => {
                if !self.noop_with_empty_axes {
                    reduce.fill(true);
                }
            }
            Some(axes) => {
                for &axis in axes {
                    let normalized = if axis < 0 { axis + rank as i64 } else { axis };
                    if normalized < 0 || normalized as usize >= rank {
                        return Err(EpError::KernelFailed(format!(
                            "{}: axis {axis} out of range for rank {rank}",
                            self.op.name()
                        )));
                    }
                    reduce[normalized as usize] = true;
                }
            }
            None => {
                if !self.noop_with_empty_axes {
                    reduce.fill(true);
                }
            }
        }
        Ok(reduce)
    }

    fn execute_i64_sum(&self, inputs: &[TensorView], output: &mut TensorMut) -> Result<()> {
        let x = to_dense_i64(&inputs[0])?;
        let in_shape = inputs[0].shape;
        let rank = in_shape.len();
        let reduce = self.resolve_axes(inputs, rank)?;
        let kept_shape = (0..rank)
            .filter(|&axis| !reduce[axis])
            .map(|axis| in_shape[axis])
            .collect::<Vec<_>>();
        let in_strides = compute_contiguous_strides(in_shape);
        let kept_strides = compute_contiguous_strides(&kept_shape);
        let mut sums = vec![0_i64; numel(&kept_shape).max(1)];
        if numel(in_shape) > 0 {
            let mut index = vec![0; rank];
            loop {
                let mut input_offset = 0;
                let mut output_offset = 0;
                let mut kept_axis = 0;
                for axis in 0..rank {
                    input_offset += in_strides[axis] as usize * index[axis];
                    if !reduce[axis] {
                        output_offset += kept_strides[kept_axis] as usize * index[axis];
                        kept_axis += 1;
                    }
                }
                sums[output_offset] = sums[output_offset].wrapping_add(x[input_offset]);
                if !next_index(in_shape, &mut index) {
                    break;
                }
            }
        }
        let _ = self.keepdims;
        let bytes = sums
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        write_dense_bytes(output, &bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::DataType;

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

    fn run_omitted_axes(op: ReduceOp, x: &Owned, out: &mut Owned) {
        ReduceKernel {
            op,
            axes_attr: None,
            keepdims: true,
            noop_with_empty_axes: false,
        }
        .execute(
            &[x.view(), TensorView::absent(DataType::Int64)],
            &mut [out.view_mut()],
        )
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
    fn sum_int64_axis1() {
        let x = Owned::i64(&[2, 3], &[1, 2, 3, 4, 5, 6]);
        let mut out = Owned::zeros(DataType::Int64, &[2, 1]);
        run_attr(ReduceOp::Sum, Some(vec![1]), &x, &mut out);
        assert_eq!(out.to_i64(), vec![6, 15]);
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
    fn l1_and_log_sum() {
        let x = Owned::f32(&[2, 2], &[-1., 2., -3., 4.]);
        let mut l1 = Owned::zeros_f32(&[2, 1]);
        run_attr(ReduceOp::L1, Some(vec![1]), &x, &mut l1);
        assert_eq!(l1.to_f32(), vec![3., 7.]);

        let positive = Owned::f32(&[2], &[1., 3.]);
        let mut log_sum = Owned::zeros_f32(&[1]);
        run_attr(ReduceOp::LogSum, Some(vec![0]), &positive, &mut log_sum);
        assert!((log_sum.to_f32()[0] - 4_f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn log_sum_exp_is_stable_for_large_values() {
        let x = Owned::f32(&[2], &[1_000., 1_001.]);
        let mut out = Owned::zeros_f32(&[1]);
        run_attr(ReduceOp::LogSumExp, Some(vec![0]), &x, &mut out);
        assert!((out.to_f32()[0] - (1_001.0 + (-1_f32).exp().ln_1p())).abs() < 1e-5);
    }

    #[test]
    fn omitted_axes_reduce_all_for_opset18_reductions() {
        let x = Owned::f32(&[2, 2], &[-1., 2., -3., 4.]);
        let mut l1 = Owned::zeros_f32(&[1, 1]);
        run_omitted_axes(ReduceOp::L1, &x, &mut l1);
        assert_eq!(l1.to_f32(), vec![10.]);

        let positive = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let mut log_sum = Owned::zeros_f32(&[1, 1]);
        run_omitted_axes(ReduceOp::LogSum, &positive, &mut log_sum);
        assert!((log_sum.to_f32()[0] - 10_f32.ln()).abs() < 1e-6);

        let mut log_sum_exp = Owned::zeros_f32(&[1, 1]);
        run_omitted_axes(ReduceOp::LogSumExp, &positive, &mut log_sum_exp);
        assert!(
            (log_sum_exp.to_f32()[0]
                - (4.0 + (1.0 + (-1_f32).exp() + (-2_f32).exp() + (-3_f32).exp()).ln()))
            .abs()
                < 1e-6
        );
    }

    #[test]
    fn log_sum_exp_with_positive_infinities_is_infinite() {
        let x = Owned::f32(&[2], &[f32::INFINITY, f32::INFINITY]);
        let mut out = Owned::zeros_f32(&[1]);
        run_attr(ReduceOp::LogSumExp, Some(vec![0]), &x, &mut out);
        assert_eq!(out.to_f32(), vec![f32::INFINITY]);
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
