//! `Concat`: join a list of tensors along one axis (`docs/ORT2.md` §4.4).
//!
//! Every input shares the output's shape except along the concatenation `axis`,
//! whose extents sum. The kernel is **dtype-agnostic**: it moves raw element
//! bytes through [`to_dense_bytes`]/[`write_dense_bytes`], so it serves every
//! fixed-width dtype uniformly (the concat pattern never inspects element
//! values).

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::{check_arity, elem_size, to_dense_bytes, write_dense_bytes};

/// Concat kernel carrying the raw `axis` attribute (may be negative).
pub struct ConcatKernel {
    axis: i64,
}

/// Factory reading the required `axis` attribute.
pub struct ConcatFactory;

impl KernelFactory for ConcatFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        // ONNX requires `axis`; real Concat nodes always carry it. Mirror the
        // crate convention (see `gather`, `reduce`) of defaulting an absent
        // attribute rather than failing kernel construction, and validate the
        // resolved axis against the input rank at execute time.
        let axis = node.attr("axis").and_then(|a| a.as_int()).unwrap_or(0);
        Ok(Box::new(ConcatKernel { axis }))
    }
}

impl Kernel for ConcatKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Concat", inputs, outputs, 1, usize::MAX, 1)?;
        let rank = inputs[0].shape.len();
        if rank == 0 {
            return Err(EpError::KernelFailed(
                "Concat: inputs must have rank >= 1. WHY: a scalar has no axis to join along. \
                 HOW: concatenate rank-1-or-higher tensors."
                    .into(),
            ));
        }
        let axis = if self.axis < 0 {
            self.axis + rank as i64
        } else {
            self.axis
        };
        if axis < 0 || axis as usize >= rank {
            return Err(EpError::KernelFailed(format!(
                "Concat: axis {} out of range for rank {rank}. WHY: the axis must index an \
                 existing dimension. HOW: pass an axis in [-{rank}, {rank}).",
                self.axis
            )));
        }
        let axis = axis as usize;

        let esize = elem_size(outputs[0].dtype)?;
        let out_shape = inputs[0].shape;
        // `outer` = product of dims strictly before `axis`; every input shares it.
        let outer: usize = out_shape[..axis].iter().product();
        // `inner` = product of dims strictly after `axis`, in bytes.
        let inner_bytes: usize = out_shape[axis + 1..].iter().product::<usize>() * esize;

        // Materialize each input's dense bytes and record its axis extent.
        let mut blocks: Vec<(Vec<u8>, usize)> = Vec::with_capacity(inputs.len());
        for (i, input) in inputs.iter().enumerate() {
            if input.shape.len() != rank {
                return Err(EpError::KernelFailed(format!(
                    "Concat: input {i} has rank {} but input 0 has rank {rank}. WHY: all inputs \
                     must share rank to join along one axis. HOW: align input ranks.",
                    input.shape.len()
                )));
            }
            if input.dtype != outputs[0].dtype {
                return Err(EpError::KernelFailed(format!(
                    "Concat: input {i} dtype {:?} differs from output dtype {:?}. WHY: Concat \
                     joins tensors of one dtype. HOW: cast inputs to a common dtype first.",
                    input.dtype, outputs[0].dtype
                )));
            }
            blocks.push((to_dense_bytes(input)?, input.shape[axis]));
        }

        // Interleave: for each `outer` slab, append every input's axis-slab in
        // input order. Each input's slab for a given `o` is `axis_dim * inner`.
        // When `outer` is 0 (an empty pre-axis dim) the output is empty and the
        // loop must not run — do NOT clamp to 1, or empty inputs over-read.
        let mut out = Vec::new();
        for o in 0..outer {
            for (bytes, axis_dim) in &blocks {
                let slab = axis_dim * inner_bytes;
                let start = o * slab;
                out.extend_from_slice(&bytes[start..start + slab]);
            }
        }

        write_dense_bytes(&mut outputs[0], &out)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::DataType;

    fn run(axis: i64, ins: &[&Owned], out: &mut Owned) {
        let views: Vec<_> = ins.iter().map(|o| o.view()).collect();
        ConcatKernel { axis }
            .execute(&views, &mut [out.view_mut()])
            .unwrap();
    }

    #[test]
    fn concat_axis0() {
        let a = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let b = Owned::f32(&[1, 2], &[5., 6.]);
        let mut out = Owned::zeros_f32(&[3, 2]);
        run(0, &[&a, &b], &mut out);
        assert_eq!(out.to_f32(), vec![1., 2., 3., 4., 5., 6.]);
    }

    #[test]
    fn concat_axis1() {
        let a = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let b = Owned::f32(&[2, 1], &[5., 6.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        run(1, &[&a, &b], &mut out);
        // rows: [1,2,5], [3,4,6]
        assert_eq!(out.to_f32(), vec![1., 2., 5., 3., 4., 6.]);
    }

    #[test]
    fn concat_negative_axis_three_inputs() {
        let a = Owned::f32(&[1, 1], &[1.]);
        let b = Owned::f32(&[1, 2], &[2., 3.]);
        let c = Owned::f32(&[1, 1], &[4.]);
        let mut out = Owned::zeros_f32(&[1, 4]);
        run(-1, &[&a, &b, &c], &mut out);
        assert_eq!(out.to_f32(), vec![1., 2., 3., 4.]);
    }

    #[test]
    fn concat_int64_dtype_agnostic() {
        let a = Owned::i64(&[2], &[10, 20]);
        let b = Owned::i64(&[3], &[30, 40, 50]);
        let mut out = Owned::zeros(DataType::Int64, &[5]);
        run(0, &[&a, &b], &mut out);
        assert_eq!(out.to_i64(), vec![10, 20, 30, 40, 50]);
    }
}
