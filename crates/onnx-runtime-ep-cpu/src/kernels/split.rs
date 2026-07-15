//! `Split`: divide one tensor along `axis` into several outputs
//! (`docs/ORT2.md` §4.4).
//!
//! The split sizes come from one of three sources, checked in ONNX precedence
//! order: the opset-13+ `split` **input** (an `int64` tensor), the legacy
//! `split` **attribute** (opset 1/2/11), or — when neither is present — an even
//! division across the outputs. Opset 18 adds the `num_outputs` attribute for
//! the even case; when the axis is not evenly divisible the final chunk takes
//! the remainder. Negative `axis` is resolved against the input rank. The op
//! moves raw element bytes, so every fixed-width dtype is supported.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, Node};

use super::{elem_size, to_dense_i64, write_dense_bytes};
use crate::strided::numel;

/// `Split` kernel carrying the resolved `axis` and the optional compile-time
/// split configuration (legacy `split` attribute / opset-18 `num_outputs`).
pub struct SplitKernel {
    axis: i64,
    split_attr: Option<Vec<i64>>,
    num_outputs: Option<i64>,
}

/// Factory reading `axis`, the legacy `split` attribute, and `num_outputs`.
pub struct SplitFactory;

impl KernelFactory for SplitFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let axis = node.attr("axis").and_then(Attribute::as_int).unwrap_or(0);
        let split_attr = node
            .attr("split")
            .and_then(Attribute::as_ints)
            .map(<[i64]>::to_vec);
        let num_outputs = node.attr("num_outputs").and_then(Attribute::as_int);
        Ok(Box::new(SplitKernel {
            axis,
            split_attr,
            num_outputs,
        }))
    }
}

/// Even division of `dim` into `n` parts (ONNX opset-18 `num_outputs`): equal
/// chunks when divisible, otherwise `ceil(dim/n)` for every chunk but the last,
/// which takes the (smaller) remainder.
fn even_split(dim: usize, n: usize) -> Result<Vec<usize>> {
    if n == 0 {
        return Err(EpError::KernelFailed(
            "Split: WHAT: zero outputs requested. WHY: Split needs at least one output. \
             HOW: give the Split node one or more outputs (or a positive `num_outputs`)."
                .into(),
        ));
    }
    if dim.is_multiple_of(n) {
        return Ok(vec![dim / n; n]);
    }
    let chunk = dim / n + 1;
    if chunk * (n - 1) > dim {
        return Err(EpError::KernelFailed(format!(
            "Split: WHAT: cannot split axis extent {dim} into {n} parts. \
             WHY: the even chunk size {chunk} leaves a negative final remainder. \
             HOW: reduce `num_outputs` to at most the axis extent {dim}."
        )));
    }
    let mut sizes = vec![chunk; n - 1];
    sizes.push(dim - chunk * (n - 1));
    Ok(sizes)
}

impl Kernel for SplitKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.is_empty() || inputs.len() > 2 {
            return Err(EpError::KernelFailed(format!(
                "Split: WHAT: expected 1..=2 inputs, got {}. \
                 WHY: Split takes `data` and an optional `split` tensor. \
                 HOW: connect `data` (and optionally a `split` int64 tensor).",
                inputs.len()
            )));
        }
        if outputs.is_empty() {
            return Err(EpError::KernelFailed(
                "Split: WHAT: no output views. WHY: Split writes each chunk into a distinct \
                 executor-provided output. HOW: give the Split node one or more outputs."
                    .into(),
            ));
        }

        let esize = elem_size(inputs[0].dtype)?;
        let in_shape = inputs[0].shape.to_vec();
        let rank = in_shape.len();
        if rank == 0 {
            return Err(EpError::KernelFailed(
                "Split: WHAT: input 0 is scalar (shape []). WHY: a scalar has no axis to split. \
                 HOW: provide a rank-1-or-higher input."
                    .into(),
            ));
        }
        let resolved = if self.axis < 0 {
            self.axis + rank as i64
        } else {
            self.axis
        };
        if resolved < 0 || resolved as usize >= rank {
            return Err(EpError::KernelFailed(format!(
                "Split: WHAT: axis {} is out of range for input shape {in_shape:?} (rank {rank}). \
                 WHY: the axis must identify an existing dimension. \
                 HOW: use an axis in [-{rank}, {rank}).",
                self.axis
            )));
        }
        let axis = resolved as usize;
        let axis_dim = in_shape[axis];
        let n_out = outputs.len();

        // Resolve split sizes in ONNX precedence: `split` input, then legacy
        // `split` attribute, then an even division (opset-18 `num_outputs`).
        let sizes: Vec<usize> = if inputs.len() == 2 && !inputs[1].is_absent() {
            to_dense_i64(&inputs[1])?
                .into_iter()
                .map(|v| {
                    usize::try_from(v).map_err(|_| {
                        EpError::KernelFailed(format!(
                            "Split: WHAT: `split` entry {v} is negative. \
                             WHY: split sizes must be non-negative. \
                             HOW: provide non-negative split sizes."
                        ))
                    })
                })
                .collect::<Result<_>>()?
        } else if let Some(attr) = &self.split_attr {
            attr.iter()
                .map(|&v| {
                    usize::try_from(v).map_err(|_| {
                        EpError::KernelFailed(format!(
                            "Split: WHAT: `split` attribute entry {v} is negative. \
                             WHY: split sizes must be non-negative. \
                             HOW: provide non-negative split sizes."
                        ))
                    })
                })
                .collect::<Result<_>>()?
        } else {
            let n = self
                .num_outputs
                .and_then(|v| usize::try_from(v).ok())
                .unwrap_or(n_out);
            even_split(axis_dim, n)?
        };

        if sizes.len() != n_out {
            return Err(EpError::KernelFailed(format!(
                "Split: WHAT: resolved {} split sizes but the node has {n_out} outputs. \
                 WHY: exactly one size is needed per output. \
                 HOW: align the split sizes / `num_outputs` with the output count.",
                sizes.len()
            )));
        }
        let total: usize = sizes.iter().sum();
        if total != axis_dim {
            return Err(EpError::KernelFailed(format!(
                "Split: WHAT: split sizes {sizes:?} sum to {total}, but axis {axis} has extent \
                 {axis_dim}. WHY: the chunks must cover the axis exactly once. \
                 HOW: make the split sizes sum to {axis_dim}."
            )));
        }

        let src = super::to_dense_bytes(&inputs[0])?;
        let outer = numel(&in_shape[..axis]);
        let inner_bytes = numel(&in_shape[axis + 1..]) * esize;
        let axis_stride_bytes = axis_dim * inner_bytes;

        let mut prefix = 0usize;
        for (k, &sz) in sizes.iter().enumerate() {
            let chunk_bytes = sz * inner_bytes;
            let mut buf = vec![0u8; outer * chunk_bytes];
            if chunk_bytes != 0 {
                for o in 0..outer {
                    let src_off = o * axis_stride_bytes + prefix * inner_bytes;
                    let dst_off = o * chunk_bytes;
                    buf[dst_off..dst_off + chunk_bytes]
                        .copy_from_slice(&src[src_off..src_off + chunk_bytes]);
                }
            }
            write_dense_bytes(&mut outputs[k], &buf)?;
            prefix += sz;
        }
        Ok(())
    }

    fn supports_strided_input(&self, input_idx: usize) -> bool {
        input_idx == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::DataType;

    fn kernel(axis: i64, split: Option<Vec<i64>>, num_outputs: Option<i64>) -> SplitKernel {
        SplitKernel {
            axis,
            split_attr: split,
            num_outputs,
        }
    }

    #[test]
    fn split_even_axis0() {
        // [4,2] split along axis 0 into two [2,2] chunks.
        let x = Owned::f32(&[4, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let mut o0 = Owned::zeros_f32(&[2, 2]);
        let mut o1 = Owned::zeros_f32(&[2, 2]);
        kernel(0, None, None)
            .execute(&[x.view()], &mut [o0.view_mut(), o1.view_mut()])
            .unwrap();
        assert_eq!(o0.to_f32(), vec![1., 2., 3., 4.]);
        assert_eq!(o1.to_f32(), vec![5., 6., 7., 8.]);
    }

    #[test]
    fn split_uneven_via_attribute_middle_axis() {
        // [2,3,2] split along axis 1 into sizes [1,2].
        let x = Owned::f32(&[2, 3, 2], &(1..=12).map(|v| v as f32).collect::<Vec<_>>());
        let mut o0 = Owned::zeros_f32(&[2, 1, 2]);
        let mut o1 = Owned::zeros_f32(&[2, 2, 2]);
        kernel(1, Some(vec![1, 2]), None)
            .execute(&[x.view()], &mut [o0.view_mut(), o1.view_mut()])
            .unwrap();
        assert_eq!(o0.to_f32(), vec![1., 2., 7., 8.]);
        assert_eq!(o1.to_f32(), vec![3., 4., 5., 6., 9., 10., 11., 12.]);
    }

    #[test]
    fn split_negative_axis() {
        // axis -1 on [2,4] → equivalent to axis 1; two [2,2] chunks.
        let x = Owned::f32(&[2, 4], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let mut o0 = Owned::zeros_f32(&[2, 2]);
        let mut o1 = Owned::zeros_f32(&[2, 2]);
        kernel(-1, None, None)
            .execute(&[x.view()], &mut [o0.view_mut(), o1.view_mut()])
            .unwrap();
        assert_eq!(o0.to_f32(), vec![1., 2., 5., 6.]);
        assert_eq!(o1.to_f32(), vec![3., 4., 7., 8.]);
    }

    #[test]
    fn split_num_outputs_remainder() {
        // opset-18 num_outputs=3 on extent 7 → sizes [3,3,1].
        assert_eq!(even_split(7, 3).unwrap(), vec![3, 3, 1]);
        assert_eq!(even_split(5, 2).unwrap(), vec![3, 2]);
        let x = Owned::f32(&[7], &[1., 2., 3., 4., 5., 6., 7.]);
        let mut o0 = Owned::zeros_f32(&[3]);
        let mut o1 = Owned::zeros_f32(&[3]);
        let mut o2 = Owned::zeros_f32(&[1]);
        kernel(0, None, Some(3))
            .execute(&[x.view()], &mut [o0.view_mut(), o1.view_mut(), o2.view_mut()])
            .unwrap();
        assert_eq!(o0.to_f32(), vec![1., 2., 3.]);
        assert_eq!(o1.to_f32(), vec![4., 5., 6.]);
        assert_eq!(o2.to_f32(), vec![7.]);
    }

    #[test]
    fn split_via_input_tensor_int64_dtype() {
        // opset-13+ `split` as an int64 input tensor; dtype-agnostic (i64 data).
        let x = Owned::i64(&[5], &[10, 20, 30, 40, 50]);
        let split = Owned::i64(&[2], &[3, 2]);
        let mut o0 = Owned::zeros(DataType::Int64, &[3]);
        let mut o1 = Owned::zeros(DataType::Int64, &[2]);
        kernel(0, None, None)
            .execute(&[x.view(), split.view()], &mut [o0.view_mut(), o1.view_mut()])
            .unwrap();
        assert_eq!(o0.to_i64(), vec![10, 20, 30]);
        assert_eq!(o1.to_i64(), vec![40, 50]);
    }

    #[test]
    fn split_rejects_sizes_not_summing_to_axis() {
        let x = Owned::f32(&[4], &[1., 2., 3., 4.]);
        let mut o0 = Owned::zeros_f32(&[1]);
        let mut o1 = Owned::zeros_f32(&[2]);
        let err = kernel(0, Some(vec![1, 2]), None)
            .execute(&[x.view()], &mut [o0.view_mut(), o1.view_mut()])
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("WHY:"));
        assert!(msg.contains("extent 4"));
    }
}
