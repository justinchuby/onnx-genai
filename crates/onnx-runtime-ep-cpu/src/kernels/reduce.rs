//! `ReduceMean`: mean over a set of axes for f32 (`docs/ORT2.md` §4.4).
//!
//! Opset-12 form: the reduced axes are the `axes` **attribute** (they became a
//! second input only in opset 18). `keepdims` (default 1) chooses whether the
//! reduced axes are retained as size-1 dims or squeezed out. Absent `axes`
//! means reduce over every axis. Negative axes index from the end.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{compute_contiguous_strides, Node};

use super::{check_arity, to_dense_f32, write_dense_f32};
use crate::strided::{next_index, numel};

/// f32 ReduceMean kernel carrying the raw `axes` (may be negative) and
/// `keepdims`.
pub struct ReduceMeanKernel {
    axes: Option<Vec<i64>>,
    keepdims: bool,
}

/// Factory reading `axes` (optional) and `keepdims` (default 1).
pub struct ReduceMeanFactory;

impl KernelFactory for ReduceMeanFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let axes = node.attr("axes").and_then(|a| a.as_ints()).map(|v| v.to_vec());
        let keepdims = node.attr("keepdims").and_then(|a| a.as_int()).unwrap_or(1) != 0;
        Ok(Box::new(ReduceMeanKernel { axes, keepdims }))
    }
}

impl Kernel for ReduceMeanKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("ReduceMean", inputs, outputs, 1, 1, 1)?;
        let x = to_dense_f32(&inputs[0])?;
        let in_shape = inputs[0].shape;
        let rank = in_shape.len();

        // Resolve the reduced-axis set (default: all axes), normalizing negatives.
        let mut reduce = vec![false; rank];
        match &self.axes {
            None => reduce.iter_mut().for_each(|r| *r = true),
            Some(axes) => {
                for &a in axes {
                    let ax = if a < 0 { a + rank as i64 } else { a };
                    if ax < 0 || ax as usize >= rank {
                        return Err(EpError::KernelFailed(format!(
                            "ReduceMean: axis {a} out of range for rank {rank}"
                        )));
                    }
                    reduce[ax as usize] = true;
                }
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

    fn run(axes: Option<Vec<i64>>, keepdims: bool, x: &Owned, out: &mut Owned) {
        ReduceMeanKernel { axes, keepdims }
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
}
