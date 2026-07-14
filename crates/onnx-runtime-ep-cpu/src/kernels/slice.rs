//! `Slice`: extract a strided sub-tensor (`docs/ORT2.md` §4.4).
//!
//! Opset-10+ form: `starts`, `ends`, and the optional `axes` / `steps` are
//! **inputs** (int32/int64 tensors), not attributes. Clamping follows the ONNX
//! reference exactly, including negative indices, negative steps (reverse), and
//! the different clamp bounds for positive vs. negative steps. The op moves raw
//! element bytes and is dtype-agnostic.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView, ViewOutput};
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
        // `axes` and `steps` are optional. An omitted optional input arrives as
        // an absent placeholder (ONNX empty-string input name), which preserves
        // positional arity — so `Slice(data, starts, ends, "", steps)` correctly
        // reads `steps` from slot 4 rather than misreading it as `axes`.
        let axes_in = if inputs.len() >= 4 && !inputs[3].is_absent() {
            Some(to_dense_i64(&inputs[3])?)
        } else {
            None
        };
        let steps_in = if inputs.len() >= 5 && !inputs[4].is_absent() {
            Some(to_dense_i64(&inputs[4])?)
        } else {
            None
        };
        let (axes, steps) =
            slice_axes_steps(starts.len(), axes_in.as_deref(), steps_in.as_deref());

        // Per-axis (start, step, count) — the single shared geometry helper.
        let plan = slice_plan(in_shape, &starts, &ends, &axes, &steps)?;
        let out_shape: Vec<_> = plan.iter().map(|p| p.count).collect();
        let n = numel(&out_shape);
        let mut out = vec![0u8; n * esize];
        if n == 0 {
            return super::write_dense_bytes(&mut outputs[0], &out);
        }

        // A full, unit-stride suffix remains contiguous in the dense source.
        // Copy each such suffix as one byte run rather than gathering its
        // elements individually. This covers the common case of slicing one of
        // the leading dimensions while retaining all trailing dimensions.
        let mut contiguous_suffix = rank;
        while contiguous_suffix > 0 {
            let p = plan[contiguous_suffix - 1];
            if p.start != 0 || p.step != 1 || p.count != in_shape[contiguous_suffix - 1] {
                break;
            }
            contiguous_suffix -= 1;
        }
        let inner_elems = numel(&in_shape[contiguous_suffix..]);
        let inner_bytes = inner_elems * esize;
        let outer_shape = &out_shape[..contiguous_suffix];
        let in_strides = compute_contiguous_strides(in_shape);
        let mut idx = vec![0usize; contiguous_suffix];
        let mut w = 0usize;
        loop {
            let mut in_off = 0i64;
            for d in 0..contiguous_suffix {
                in_off += in_strides[d] * (plan[d].start + plan[d].step * idx[d] as i64);
            }
            let s = in_off as usize * esize;
            out[w..w + inner_bytes].copy_from_slice(&src[s..s + inner_bytes]);
            w += inner_bytes;
            if !next_index(outer_shape, &mut idx) {
                break;
            }
        }
        super::write_dense_bytes(&mut outputs[0], &out)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }

    /// Zero-copy fast path: express the Slice result as a strided [`ViewOutput`]
    /// over `data` (input 0) instead of gathering bytes. A pure sub-view — any
    /// axis range with any step (positive, or negative → negative stride) — is a
    /// linear reindexing of the input, so it needs no copy: the output shares
    /// the input's buffer with adjusted shape/strides/offset. The executor keeps
    /// that buffer alive until the view's consumers run and materializes to
    /// contiguous only at a kernel that cannot take a strided input or at the
    /// graph-output boundary.
    ///
    /// Falls back to the copy path (returns `None`) when the geometry cannot be
    /// expressed as a view: sub-byte element types (no fixed-width element
    /// stride), an empty (zero-count) axis, or any parameter read failure. This
    /// is decided purely from `elem_size` and integer inputs — no dtype, vendor,
    /// or model special-casing (RULES §2/§4).
    fn view_outputs(&self, inputs: &[TensorView], num_outputs: usize) -> Option<Vec<ViewOutput>> {
        if num_outputs != 1 || inputs.len() < 3 {
            return None;
        }
        let data = &inputs[0];
        // Sub-byte / variable-width types have no fixed-width element stride, so
        // a byte-offset/element-stride view cannot address them: copy instead.
        let esize = data.dtype.byte_size();
        if esize == 0 {
            return None;
        }
        let in_shape = data.shape;
        let rank = in_shape.len();
        // `data`'s own (possibly already strided) geometry — the view we emit is
        // composed on top of it so a Slice-of-a-Slice stays a single view.
        let in_strides = data.strides;
        if in_strides.len() != rank {
            return None;
        }

        let starts = to_dense_i64(&inputs[1]).ok()?;
        let ends = to_dense_i64(&inputs[2]).ok()?;
        let axes_in = if inputs.len() >= 4 && !inputs[3].is_absent() {
            Some(to_dense_i64(&inputs[3]).ok()?)
        } else {
            None
        };
        let steps_in = if inputs.len() >= 5 && !inputs[4].is_absent() {
            Some(to_dense_i64(&inputs[4]).ok()?)
        } else {
            None
        };
        let (axes, steps) = slice_axes_steps(starts.len(), axes_in.as_deref(), steps_in.as_deref());
        let plan = slice_plan(in_shape, &starts, &ends, &axes, &steps).ok()?;

        let mut out_shape = Vec::with_capacity(rank);
        let mut out_strides = Vec::with_capacity(rank);
        // Element offset of the sub-view origin from `data`'s element origin.
        let mut origin_elems: i64 = 0;
        for (d, p) in plan.iter().enumerate() {
            // An empty axis makes offset math ambiguous (start clamps to the
            // boundary); the copy path already handles empties — fall back.
            if p.count == 0 {
                return None;
            }
            out_shape.push(p.count);
            out_strides.push(in_strides[d] * p.step);
            origin_elems += in_strides[d] * p.start;
        }
        // Compose onto `data`'s existing byte offset so we alias its base buffer.
        let byte_offset = (data.byte_offset as i64 + origin_elems * esize as i64) as usize;
        Some(vec![ViewOutput {
            input_index: 0,
            shape: out_shape,
            strides: out_strides,
            byte_offset,
        }])
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

    #[test]
    fn slice_omitted_axes_with_present_steps() {
        // Slice(data, starts, ends, "", steps): axes omitted (absent placeholder)
        // but steps supplied. The absent slot must NOT cause `steps` to be read
        // as `axes`. data [6], [0:6:2] over default axis 0 -> [1,3,5].
        let data = Owned::f32(&[6], &[1., 2., 3., 4., 5., 6.]);
        let starts = Owned::i64(&[1], &[0]);
        let ends = Owned::i64(&[1], &[6]);
        let steps = Owned::i64(&[1], &[2]);
        let mut out = Owned::zeros_f32(&[3]);
        SliceKernel
            .execute(
                &[
                    data.view(),
                    starts.view(),
                    ends.view(),
                    TensorView::absent(onnx_runtime_ir::DataType::Int64),
                    steps.view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap();
        assert_eq!(out.to_f32(), vec![1., 3., 5.]);
    }

    #[test]
    fn slice_large_contiguous_tail() {
        // Slice leading rows only, retaining a 256x64 trailing region. The
        // kernel copies each retained row as one contiguous byte run.
        let values: Vec<f32> = (0..3 * 256 * 64).map(|i| i as f32).collect();
        let data = Owned::f32(&[3, 256, 64], &values);
        let starts = Owned::i64(&[1], &[1]);
        let ends = Owned::i64(&[1], &[3]);
        let axes = Owned::i64(&[1], &[0]);
        let mut out = Owned::zeros_f32(&[2, 256, 64]);
        SliceKernel
            .execute(
                &[data.view(), starts.view(), ends.view(), axes.view()],
                &mut [out.view_mut()],
            )
            .unwrap();
        assert_eq!(out.to_f32(), values[256 * 64..]);
    }

    #[test]
    fn slice_negative_axis_with_steps() {
        // Negative axis (-1) addressing the last dim, with a negative step.
        // data [2,4], axis -1, [3:-5:-1] reverses each row -> [2,4] reversed cols.
        let data = Owned::f32(&[2, 4], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let starts = Owned::i64(&[1], &[3]);
        let ends = Owned::i64(&[1], &[-5]);
        let axes = Owned::i64(&[1], &[-1]);
        let steps = Owned::i64(&[1], &[-1]);
        let mut out = Owned::zeros_f32(&[2, 4]);
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
        assert_eq!(out.to_f32(), vec![4., 3., 2., 1., 8., 7., 6., 5.]);
    }

    #[test]
    fn slice_empty_output() {
        // A start == end slice yields a zero-length axis (empty output). The
        // kernel must produce an empty buffer without indexing errors.
        let data = Owned::f32(&[5], &[1., 2., 3., 4., 5.]);
        let starts = Owned::i64(&[1], &[2]);
        let ends = Owned::i64(&[1], &[2]);
        let mut out = Owned::zeros_f32(&[0]);
        SliceKernel
            .execute(
                &[data.view(), starts.view(), ends.view()],
                &mut [out.view_mut()],
            )
            .unwrap();
        assert_eq!(out.to_f32(), Vec::<f32>::new());
    }

    #[test]
    fn view_output_axis0_positive_step_is_subview() {
        // data [4,2] contiguous, slice rows [1:3] step 1 → view [2,2] over rows
        // 1..3 (byte offset = 2 elems * 4 bytes = 8; strides unchanged [2,1]).
        let data = Owned::f32(&[4, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let starts = Owned::i64(&[1], &[1]);
        let ends = Owned::i64(&[1], &[3]);
        let axes = Owned::i64(&[1], &[0]);
        let vo = SliceKernel
            .view_outputs(
                &[data.view(), starts.view(), ends.view(), axes.view()],
                1,
            )
            .expect("pure sub-view should be a view output");
        assert_eq!(vo.len(), 1);
        assert_eq!(vo[0].input_index, 0);
        assert_eq!(vo[0].shape, vec![2, 2]);
        assert_eq!(vo[0].strides, vec![2, 1]);
        assert_eq!(vo[0].byte_offset, 8);
    }

    #[test]
    fn view_output_step_two_scales_stride() {
        // data [6] step 2 → view [3] stride 2 (elements), offset 0.
        let data = Owned::f32(&[6], &[1., 2., 3., 4., 5., 6.]);
        let starts = Owned::i64(&[1], &[0]);
        let ends = Owned::i64(&[1], &[6]);
        let axes = Owned::i64(&[1], &[0]);
        let steps = Owned::i64(&[1], &[2]);
        let vo = SliceKernel
            .view_outputs(
                &[
                    data.view(),
                    starts.view(),
                    ends.view(),
                    axes.view(),
                    steps.view(),
                ],
                1,
            )
            .unwrap();
        assert_eq!(vo[0].shape, vec![3]);
        assert_eq!(vo[0].strides, vec![2]);
        assert_eq!(vo[0].byte_offset, 0);
    }

    #[test]
    fn view_output_negative_step_is_negative_stride() {
        // data [5] reversed [4:-6:-1] → view [5] stride -1, origin at last elem
        // (byte offset 4 elems * 4 bytes = 16).
        let data = Owned::f32(&[5], &[1., 2., 3., 4., 5.]);
        let starts = Owned::i64(&[1], &[4]);
        let ends = Owned::i64(&[1], &[-6]);
        let axes = Owned::i64(&[1], &[0]);
        let steps = Owned::i64(&[1], &[-1]);
        let vo = SliceKernel
            .view_outputs(
                &[
                    data.view(),
                    starts.view(),
                    ends.view(),
                    axes.view(),
                    steps.view(),
                ],
                1,
            )
            .unwrap();
        assert_eq!(vo[0].shape, vec![5]);
        assert_eq!(vo[0].strides, vec![-1]);
        assert_eq!(vo[0].byte_offset, 16);
    }

    #[test]
    fn view_output_empty_slice_falls_back_to_copy() {
        // A zero-count axis cannot be expressed as a view (ambiguous offset);
        // the kernel must fall back to the copy path (returns None).
        let data = Owned::f32(&[5], &[1., 2., 3., 4., 5.]);
        let starts = Owned::i64(&[1], &[2]);
        let ends = Owned::i64(&[1], &[2]);
        assert!(SliceKernel
            .view_outputs(&[data.view(), starts.view(), ends.view()], 1)
            .is_none());
    }

    #[test]
    fn view_output_composes_over_strided_input() {
        // A Slice-of-a-Slice: input is already a strided view (a [3] stride-2
        // view over a [6] buffer, offset 0 → elements 0,2,4). Slicing [1:3]
        // step 1 must compose: origin advances by 1*stride=2 elems (offset 8),
        // stride stays 2, shape [2] (picks original elements 2,4).
        let strided = Owned::f32(&[6], &[1., 2., 3., 4., 5., 6.]).with_view(&[3], &[2]);
        let starts = Owned::i64(&[1], &[1]);
        let ends = Owned::i64(&[1], &[3]);
        let axes = Owned::i64(&[1], &[0]);
        let vo = SliceKernel
            .view_outputs(
                &[strided.view(), starts.view(), ends.view(), axes.view()],
                1,
            )
            .unwrap();
        assert_eq!(vo[0].shape, vec![2]);
        assert_eq!(vo[0].strides, vec![2]);
        assert_eq!(vo[0].byte_offset, 8);
    }
}
