//! `Gather`: index f32 `data` along `axis` with an integer `indices` tensor
//! (`docs/ORT2.md` §4.4). Output shape is
//! `data.shape[:axis] ++ indices.shape ++ data.shape[axis+1:]`.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::{check_arity, to_dense_f32, to_dense_i64, write_dense_f32};
use crate::strided::numel;

/// f32 Gather kernel carrying the resolved `axis`.
pub struct GatherKernel {
    /// Raw axis attribute (may be negative); normalized against data rank at run.
    axis: i64,
}

/// Factory reading the `axis` attribute (default 0) from the node.
pub struct GatherFactory;

impl KernelFactory for GatherFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let axis = node.attr("axis").and_then(|a| a.as_int()).unwrap_or(0);
        Ok(Box::new(GatherKernel { axis }))
    }
}

impl Kernel for GatherKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Gather", inputs, outputs, 2, 2, 1)?;
        let data = to_dense_f32(&inputs[0])?;
        let indices = to_dense_i64(&inputs[1])?;
        let data_shape = inputs[0].shape;
        let idx_shape = inputs[1].shape;
        let rank = data_shape.len();
        if rank == 0 {
            return Err(EpError::KernelFailed("Gather: data must have rank >= 1".into()));
        }

        // Normalize axis into [0, rank).
        let axis = if self.axis < 0 {
            self.axis + rank as i64
        } else {
            self.axis
        };
        if axis < 0 || axis as usize >= rank {
            return Err(EpError::KernelFailed(format!(
                "Gather: axis {} out of range for rank {rank}",
                self.axis
            )));
        }
        let axis = axis as usize;

        let axis_dim = data_shape[axis];
        let outer: usize = data_shape[..axis].iter().product();
        let inner: usize = data_shape[axis + 1..].iter().product();
        let num_idx = numel(idx_shape);

        let mut out = vec![0.0f32; outer * num_idx * inner];
        let mut w = 0usize;
        for o in 0..outer {
            for &raw in &indices {
                let idx = if raw < 0 { raw + axis_dim as i64 } else { raw };
                if idx < 0 || idx as usize >= axis_dim {
                    return Err(EpError::KernelFailed(format!(
                        "Gather: index {raw} out of range for axis dim {axis_dim}"
                    )));
                }
                let base = (o * axis_dim + idx as usize) * inner;
                out[w..w + inner].copy_from_slice(&data[base..base + inner]);
                w += inner;
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

    fn run(axis: i64, data: &Owned, idx: &Owned, out: &mut Owned) {
        GatherKernel { axis }
            .execute(&[data.view(), idx.view()], &mut [out.view_mut()])
            .unwrap();
    }

    #[test]
    fn gather_rows_axis0() {
        // data [3,2], indices [2] = [2,0] -> [2,2]
        let data = Owned::f32(&[3, 2], &[1., 2., 3., 4., 5., 6.]);
        let idx = Owned::i64(&[2], &[2, 0]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        run(0, &data, &idx, &mut out);
        assert_eq!(out.to_f32(), vec![5., 6., 1., 2.]);
    }

    #[test]
    fn gather_columns_axis1() {
        // data [2,3], indices [2] = [0,2], axis 1 -> [2,2]
        let data = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let idx = Owned::i64(&[2], &[0, 2]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        run(1, &data, &idx, &mut out);
        // cols 0 and 2: [1,3] and [4,6]
        assert_eq!(out.to_f32(), vec![1., 3., 4., 6.]);
    }

    #[test]
    fn gather_negative_index() {
        let data = Owned::f32(&[3, 2], &[1., 2., 3., 4., 5., 6.]);
        let idx = Owned::i64(&[1], &[-1]); // last row
        let mut out = Owned::zeros_f32(&[1, 2]);
        run(0, &data, &idx, &mut out);
        assert_eq!(out.to_f32(), vec![5., 6.]);
    }

    #[test]
    fn gather_2d_indices_embedding() {
        // Embedding-style: data [4,2] (vocab x dim), indices [1,3] -> [1,3,2]
        let data = Owned::f32(&[4, 2], &[0., 1., 2., 3., 4., 5., 6., 7.]);
        let idx = Owned::i64(&[1, 3], &[0, 2, 3]);
        let mut out = Owned::zeros_f32(&[1, 3, 2]);
        run(0, &data, &idx, &mut out);
        assert_eq!(out.to_f32(), vec![0., 1., 4., 5., 6., 7.]);
    }
}
