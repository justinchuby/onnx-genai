//! `ReduceMean`: mean over a set of axes for f32 (`docs/ORT2.md` §4.4).
//!
//! Two axis-signature forms are supported:
//! * **Legacy (opset ≤ 17):** the reduced axes are the `axes` **attribute**.
//! * **Modern (opset ≥ 18):** `axes` moved to an optional **second input**
//!   (`int64` tensor). When both are absent the axes default to *all* dims.
//!
//! `keepdims` (default 1) chooses whether the reduced axes are retained as
//! size-1 dims or squeezed out. `noop_with_empty_axes` (opset ≥ 18, default 0)
//! selects the behaviour when the axis set is empty: `0` reduces over every
//! axis, `1` is an identity (the input is copied through). Negative axes index
//! from the end.

use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Node, compute_contiguous_strides};

use super::{check_arity, to_dense_i64};
use crate::strided::{next_index, numel};

/// f32 ReduceMean kernel carrying the legacy `axes` attribute (may be negative;
/// `None` when unset or when axes arrive as the opset-18 input), `keepdims`, and
/// `noop_with_empty_axes`.
pub struct ReduceMeanKernel {
    axes: Option<Vec<i64>>,
    keepdims: bool,
    noop_with_empty_axes: bool,
}

/// Factory reading `axes` (optional attribute), `keepdims` (default 1), and
/// `noop_with_empty_axes` (default 0).
pub struct ReduceMeanFactory;

impl KernelFactory for ReduceMeanFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let axes = node
            .attr("axes")
            .and_then(|a| a.as_ints())
            .map(|v| v.to_vec());
        let keepdims = node.attr("keepdims").and_then(|a| a.as_int()).unwrap_or(1) != 0;
        let noop_with_empty_axes = node
            .attr("noop_with_empty_axes")
            .and_then(|a| a.as_int())
            .unwrap_or(0)
            != 0;
        Ok(Box::new(ReduceMeanKernel {
            axes,
            keepdims,
            noop_with_empty_axes,
        }))
    }
}

impl Kernel for ReduceMeanKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        // Opset ≥ 18 adds the optional `axes` input; opset ≤ 17 has data only.
        check_arity("ReduceMean", inputs, outputs, 1, 2, 1)?;
        let x = to_dense_f32_widen("ReduceMean", &inputs[0])?;
        let in_shape = inputs[0].shape;
        let rank = in_shape.len();

        // Resolve the axis set: the opset-18 `axes` input takes precedence over
        // the legacy attribute. An absent input falls back to the attribute.
        let axes: Option<Vec<i64>> = if inputs.len() >= 2 && !inputs[1].is_absent() {
            Some(to_dense_i64(&inputs[1])?)
        } else {
            self.axes.clone()
        };

        // An empty axis set means "no axes named". `noop_with_empty_axes`
        // decides: identity (leave every axis un-reduced) vs. reduce-all.
        let empty_axes = axes.as_ref().map(|a| a.is_empty()).unwrap_or(true);

        let mut reduce = vec![false; rank];
        if empty_axes {
            if !self.noop_with_empty_axes {
                reduce.iter_mut().for_each(|r| *r = true);
            }
            // else: all-false ⇒ identity (each element maps to a distinct
            // output slot below, with a divisor of 1).
        } else {
            for &a in axes.as_ref().unwrap() {
                let ax = if a < 0 { a + rank as i64 } else { a };
                if ax < 0 || ax as usize >= rank {
                    return Err(EpError::KernelFailed(format!(
                        "ReduceMean: axis {a} out of range for rank {rank}"
                    )));
                }
                reduce[ax as usize] = true;
            }
        }

        // Kept (output) axes and the count of elements folded into each output.
        let kept_shape: Vec<usize> = (0..rank)
            .filter(|&d| !reduce[d])
            .map(|d| in_shape[d])
            .collect();
        let reduced_count: usize = (0..rank)
            .filter(|&d| reduce[d])
            .map(|d| in_shape[d])
            .product();
        let kept_count = numel(&kept_shape);

        let in_strides = compute_contiguous_strides(in_shape);
        // Row-major stride of each *kept* axis into the flat output buffer.
        let kept_out_strides = compute_contiguous_strides(&kept_shape);

        let mut sums = vec![0.0f32; kept_count.max(1)];
        if numel(in_shape) > 0 {
            let mut idx = vec![0usize; rank];
            loop {
                // Flat input offset for this multi-index.
                let mut in_off = 0usize;
                // Flat output offset, skipping reduced axes.
                let mut out_off = 0usize;
                let mut kept_axis = 0usize;
                for d in 0..rank {
                    in_off += in_strides[d] as usize * idx[d];
                    if !reduce[d] {
                        out_off += kept_out_strides[kept_axis] as usize * idx[d];
                        kept_axis += 1;
                    }
                }
                sums[out_off] += x[in_off];
                if !next_index(in_shape, &mut idx) {
                    break;
                }
            }
        }

        let denom = reduced_count.max(1) as f32;
        let out: Vec<f32> = sums.iter().map(|&s| s / denom).collect();

        // The output view's own shape (keepdims-aware) governs the write; the
        // dense buffer already matches it element-for-element regardless of
        // whether reduced axes are retained as size-1 or squeezed.
        let _ = self.keepdims;
        write_dense_f32_narrow("ReduceMean", &mut outputs[0], &out)
    }

    fn supports_strided_input(&self, input_idx: usize) -> bool {
        // Only the data input (0) is materialized strided; the axes input (1)
        // goes through `to_dense_i64`, which handles strides itself.
        input_idx == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    fn run(axes: Option<Vec<i64>>, keepdims: bool, x: &Owned, out: &mut Owned) {
        ReduceMeanKernel {
            axes,
            keepdims,
            noop_with_empty_axes: false,
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
    }

    #[test]
    fn mean_axis1_keepdims() {
        // [2,3], mean over axis 1 -> [2,1]
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let mut out = Owned::zeros_f32(&[2, 1]);
        run(Some(vec![1]), true, &x, &mut out);
        assert_eq!(out.to_f32(), vec![2.0, 5.0]);
    }

    #[test]
    fn mean_axis0_no_keepdims() {
        // [2,3], mean over axis 0 -> [3]
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let mut out = Owned::zeros_f32(&[3]);
        run(Some(vec![0]), false, &x, &mut out);
        assert_eq!(out.to_f32(), vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn mean_multiple_axes() {
        // [2,2,2], reduce axes {0,2} -> keep axis 1, shape [1,2,1] keepdims.
        // data indexed [i,j,k]:
        //  (0,0,0)=1 (0,0,1)=2 (0,1,0)=3 (0,1,1)=4
        //  (1,0,0)=5 (1,0,1)=6 (1,1,0)=7 (1,1,1)=8
        // j=0: mean(1,2,5,6)=3.5 ; j=1: mean(3,4,7,8)=5.5
        let x = Owned::f32(&[2, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let mut out = Owned::zeros_f32(&[1, 2, 1]);
        run(Some(vec![0, 2]), true, &x, &mut out);
        assert_eq!(out.to_f32(), vec![3.5, 5.5]);
    }

    #[test]
    fn mean_negative_axis() {
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let mut out = Owned::zeros_f32(&[2, 1]);
        run(Some(vec![-1]), true, &x, &mut out);
        assert_eq!(out.to_f32(), vec![2.0, 5.0]);
    }

    #[test]
    fn mean_all_axes_default() {
        let x = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let mut out = Owned::zeros_f32(&[1, 1]);
        run(None, true, &x, &mut out);
        assert_eq!(out.to_f32(), vec![2.5]);
    }

    #[test]
    fn mean_axes_as_input_opset18() {
        // Opset-18 form: axes arrive as a second int64 input, not an attribute.
        // [2,3], mean over axis 1 -> [2,1].
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let axes = Owned::i64(&[1], &[1]);
        let mut out = Owned::zeros_f32(&[2, 1]);
        ReduceMeanKernel {
            axes: None,
            keepdims: true,
            noop_with_empty_axes: false,
        }
        .execute(&[x.view(), axes.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_f32(), vec![2.0, 5.0]);
    }

    #[test]
    fn mean_axes_input_negative() {
        // Negative axis via the opset-18 input path.
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let axes = Owned::i64(&[1], &[-1]);
        let mut out = Owned::zeros_f32(&[2, 1]);
        ReduceMeanKernel {
            axes: None,
            keepdims: true,
            noop_with_empty_axes: false,
        }
        .execute(&[x.view(), axes.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_f32(), vec![2.0, 5.0]);
    }

    #[test]
    fn mean_empty_axes_input_reduces_all() {
        // Empty axes input + noop_with_empty_axes=0 (default) -> reduce all.
        let x = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let axes = Owned::i64(&[0], &[]);
        let mut out = Owned::zeros_f32(&[1, 1]);
        ReduceMeanKernel {
            axes: None,
            keepdims: true,
            noop_with_empty_axes: false,
        }
        .execute(&[x.view(), axes.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_f32(), vec![2.5]);
    }

    #[test]
    fn mean_empty_axes_noop_is_identity() {
        // Empty axes + noop_with_empty_axes=1 -> identity (input copied through).
        let x = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let axes = Owned::i64(&[0], &[]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        ReduceMeanKernel {
            axes: None,
            keepdims: true,
            noop_with_empty_axes: true,
        }
        .execute(&[x.view(), axes.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_f32(), vec![1., 2., 3., 4.]);
    }

    #[test]
    fn mean_axes_input_absent_falls_back_to_attribute() {
        // An absent optional axes input falls back to the legacy attribute.
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let mut out = Owned::zeros_f32(&[2, 1]);
        ReduceMeanKernel {
            axes: Some(vec![1]),
            keepdims: true,
            noop_with_empty_axes: false,
        }
        .execute(
            &[
                x.view(),
                TensorView::absent(onnx_runtime_ir::DataType::Int64),
            ],
            &mut [out.view_mut()],
        )
        .unwrap();
        assert_eq!(out.to_f32(), vec![2.0, 5.0]);
    }
    #[test]
    fn reduce_mean_bf16_matches_widened_f32_reference() {
        let x = Owned::bf16(&[2, 3], &[-80., 0., 1., 80., -1., 2.]);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[2, 1]);
        run(Some(vec![1]), true, &x, &mut out);
        assert_eq!(
            out.to_bf16_as_f32(),
            vec![
                half::bf16::from_f32(-79. / 3.).to_f32(),
                half::bf16::from_f32(81. / 3.).to_f32()
            ]
        );
    }
}
