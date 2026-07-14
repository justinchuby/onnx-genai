//! `Slice`: extract a strided sub-tensor (`docs/ORT2.md` §4.4).
//!
//! Opset-10+ form: `starts`, `ends`, and the optional `axes` / `steps` are
//! **inputs** (int32/int64 tensors), not attributes. Clamping follows the ONNX
//! reference exactly, including negative indices, negative steps (reverse), and
//! the different clamp bounds for positive vs. negative steps. The op moves raw
//! element bytes and is dtype-agnostic.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{compute_contiguous_strides, Node};

use super::{check_arity, elem_size, to_dense_bytes, to_dense_i64};
use crate::strided::{next_index, numel};

/// Stateless Slice kernel.
pub struct SliceKernel;

/// The resolved per-axis Slice plan for one axis: the first source coordinate,
/// the step, and the number of output elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SliceAxisPlan {
    pub start: i64,
    pub step: i64,
    pub count: usize,
}

/// Compute the per-axis Slice plan (start, step, output-count) for every axis of
/// a rank-`in_shape.len()` input under opset-10+ semantics. `axes` and `steps`
/// must already be resolved to the same length as `starts`/`ends`. Non-sliced
/// axes take their full extent (`start = 0`, `step = 1`, `count = dim`).
///
/// This is the single source of truth for Slice output geometry — the CPU
/// [`SliceKernel`] and the session executor's data-dependent shape sizer both
/// call it, so the buffer the session sizes can never drift from what the
/// kernel writes. Clamping follows the ONNX reference exactly (negative
/// indices, negative steps, and step-sign-dependent bounds).
pub fn slice_plan(
    in_shape: &[usize],
    starts: &[i64],
    ends: &[i64],
    axes: &[i64],
    steps: &[i64],
) -> Result<Vec<SliceAxisPlan>> {
    let rank = in_shape.len();
    if starts.len() != ends.len() || starts.len() != axes.len() || starts.len() != steps.len() {
        return Err(EpError::KernelFailed(
            "Slice: starts/ends/axes/steps length mismatch".into(),
        ));
    }
    let mut plan = vec![
        SliceAxisPlan {
            start: 0,
            step: 1,
            count: 0,
        };
        rank
    ];
    for (ax, p) in plan.iter_mut().enumerate() {
        p.count = in_shape[ax];
    }
    for i in 0..starts.len() {
        let raw_axis = axes[i];
        let ax = if raw_axis < 0 {
            raw_axis + rank as i64
        } else {
            raw_axis
        };
        if ax < 0 || ax as usize >= rank {
            return Err(EpError::KernelFailed(format!(
                "Slice: axis {raw_axis} out of range for rank {rank}"
            )));
        }
        let ax = ax as usize;
        let d = in_shape[ax] as i64;
        let s = steps[i];
        if s == 0 {
            return Err(EpError::KernelFailed("Slice: step must be non-zero".into()));
        }
        let mut b = starts[i];
        let mut e = ends[i];
        if b < 0 {
            b += d;
        }
        if e < 0 {
            e += d;
        }
        // Clamp bounds differ by step sign (ONNX reference semantics).
        let (b, e) = if s < 0 {
            (b.clamp(0, d - 1), e.clamp(-1, d - 1))
        } else {
            (b.clamp(0, d), e.clamp(0, d))
        };
        // Element count along this axis under `step`.
        let c = if s > 0 {
            if e > b {
                ((e - b + s - 1) / s) as usize
            } else {
                0
            }
        } else if b > e {
            ((b - e + (-s) - 1) / (-s)) as usize
        } else {
            0
        };
        plan[ax] = SliceAxisPlan {
            start: b,
            step: s,
            count: c,
        };
    }
    Ok(plan)
}

/// Resolve the optional `axes` / `steps` Slice inputs to full-length vectors,
/// defaulting `axes` to `0..starts_len` and `steps` to all-ones. Shared so the
/// kernel and the executor's sizer resolve defaults identically.
pub fn slice_axes_steps(
    starts_len: usize,
    axes: Option<&[i64]>,
    steps: Option<&[i64]>,
) -> (Vec<i64>, Vec<i64>) {
    let axes = match axes {
        Some(a) => a.to_vec(),
        None => (0..starts_len as i64).collect(),
    };
    let steps = match steps {
        Some(s) => s.to_vec(),
        None => vec![1; starts_len],
    };
    (axes, steps)
}

/// Factory for [`SliceKernel`] (opset-10+ takes all parameters as inputs).
pub struct SliceFactory;

impl KernelFactory for SliceFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(SliceKernel))
    }
}

impl Kernel for SliceKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Slice", inputs, outputs, 3, 5, 1)?;
        let esize = elem_size(inputs[0].dtype)?;
        let src = to_dense_bytes(&inputs[0])?;
        let in_shape = inputs[0].shape;
        let rank = in_shape.len();

        let starts = to_dense_i64(&inputs[1])?;
        let ends = to_dense_i64(&inputs[2])?;
        let axes_in = if inputs.len() >= 4 {
            Some(to_dense_i64(&inputs[3])?)
        } else {
            None
        };
        let steps_in = if inputs.len() >= 5 {
            Some(to_dense_i64(&inputs[4])?)
        } else {
            None
        };
        let (axes, steps) =
            slice_axes_steps(starts.len(), axes_in.as_deref(), steps_in.as_deref());

        // Per-axis (start, step, count) — the single shared geometry helper.
        let plan = slice_plan(in_shape, &starts, &ends, &axes, &steps)?;
        let start: Vec<i64> = plan.iter().map(|p| p.start).collect();
        let step: Vec<i64> = plan.iter().map(|p| p.step).collect();
        let count: Vec<usize> = plan.iter().map(|p| p.count).collect();

        let out_shape = count;
        let in_strides = compute_contiguous_strides(in_shape);
        let n = numel(&out_shape);
        let mut out = vec![0u8; n * esize];
        if n > 0 {
            let mut idx = vec![0usize; rank];
            let mut w = 0usize;
            loop {
                let mut in_off = 0i64;
                for d in 0..rank {
                    let coord = start[d] + step[d] * idx[d] as i64;
                    in_off += in_strides[d] * coord;
                }
                let s = in_off as usize * esize;
                out[w..w + esize].copy_from_slice(&src[s..s + esize]);
                w += esize;
                if !next_index(&out_shape, &mut idx) {
                    break;
                }
            }
        }
        super::write_dense_bytes(&mut outputs[0], &out)
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
    fn slice_plan_matches_reference_geometry() {
        // Rank-2 input, slice axis 0 [1:3] step 1 → axis 0 count 2, axis 1 full.
        let plan = slice_plan(&[4, 2], &[1], &[3], &[0], &[1]).unwrap();
        assert_eq!(
            plan,
            vec![
                SliceAxisPlan { start: 1, step: 1, count: 2 },
                SliceAxisPlan { start: 0, step: 1, count: 2 },
            ]
        );
        // Negative step reverses: [4:-6:-1] over dim 5 → all 5 elements reversed.
        let rev = slice_plan(&[5], &[4], &[-6], &[0], &[-1]).unwrap();
        assert_eq!(rev, vec![SliceAxisPlan { start: 4, step: -1, count: 5 }]);
        // Zero step and length mismatch are rejected.
        assert!(slice_plan(&[5], &[0], &[5], &[0], &[0]).is_err());
        assert!(slice_plan(&[5], &[0, 1], &[5], &[0], &[1]).is_err());
    }

    #[test]
    fn slice_basic_axis0() {
        // data [4,2], slice rows [1:3] -> [2,2]
        let data = Owned::f32(&[4, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let starts = Owned::i64(&[1], &[1]);
        let ends = Owned::i64(&[1], &[3]);
        let axes = Owned::i64(&[1], &[0]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        SliceKernel
            .execute(
                &[data.view(), starts.view(), ends.view(), axes.view()],
                &mut [out.view_mut()],
            )
            .unwrap();
        assert_eq!(out.to_f32(), vec![3., 4., 5., 6.]);
    }

    #[test]
    fn slice_negative_indices() {
        // data [5], slice [-3:-1] -> indices 2,3 -> [3,4]
        let data = Owned::f32(&[5], &[1., 2., 3., 4., 5.]);
        let starts = Owned::i64(&[1], &[-3]);
        let ends = Owned::i64(&[1], &[-1]);
        let mut out = Owned::zeros_f32(&[2]);
        SliceKernel
            .execute(
                &[data.view(), starts.view(), ends.view()],
                &mut [out.view_mut()],
            )
            .unwrap();
        assert_eq!(out.to_f32(), vec![3., 4.]);
    }

    #[test]
    fn slice_step_two() {
        // data [6], slice [0:6:2] -> [1,3,5]
        let data = Owned::f32(&[6], &[1., 2., 3., 4., 5., 6.]);
        let starts = Owned::i64(&[1], &[0]);
        let ends = Owned::i64(&[1], &[6]);
        let axes = Owned::i64(&[1], &[0]);
        let steps = Owned::i64(&[1], &[2]);
        let mut out = Owned::zeros_f32(&[3]);
        SliceKernel
            .execute(
                &[
                    data.view(),
                    starts.view(),
                    ends.view(),
                    axes.view(),
                    steps.view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        assert_eq!(out.to_f32(), vec![1., 3., 5.]);
    }

    #[test]
    fn slice_negative_step_reverse() {
        // data [5], slice [4:-6:-1] (whole reversed) -> [5,4,3,2,1]
        let data = Owned::f32(&[5], &[1., 2., 3., 4., 5.]);
        let starts = Owned::i64(&[1], &[4]);
        let ends = Owned::i64(&[1], &[-6]);
        let axes = Owned::i64(&[1], &[0]);
        let steps = Owned::i64(&[1], &[-1]);
        let mut out = Owned::zeros_f32(&[5]);
        SliceKernel
            .execute(
                &[
                    data.view(),
                    starts.view(),
                    ends.view(),
                    axes.view(),
                    steps.view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        assert_eq!(out.to_f32(), vec![5., 4., 3., 2., 1.]);
    }

    #[test]
    fn slice_int64_data_multi_axis() {
        // data [3,3] int64, slice rows [0:2], cols [1:3] -> [2,2]
        let data = Owned::i64(&[3, 3], &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let starts = Owned::i64(&[2], &[0, 1]);
        let ends = Owned::i64(&[2], &[2, 3]);
        let axes = Owned::i64(&[2], &[0, 1]);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::Int64, &[2, 2]);
        SliceKernel
            .execute(
                &[data.view(), starts.view(), ends.view(), axes.view()],
                &mut [out.view_mut()],
            )
            .unwrap();
        assert_eq!(out.to_i64(), vec![2, 3, 5, 6]);
    }
}
