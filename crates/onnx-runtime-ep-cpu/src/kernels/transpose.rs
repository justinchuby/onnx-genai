//! `Transpose`: permute axes for f32. The `perm` attribute gives the axis
//! order; it defaults to reversing all axes (`docs/ORT2.md` §4.4).

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Node, compute_contiguous_strides};

use super::{check_arity, to_dense_f32, write_dense_f32};
use crate::strided::{next_index, numel};

/// f32 Transpose kernel carrying the resolved `perm`.
pub struct TransposeKernel {
    /// Axis permutation; `None` means reverse all axes.
    perm: Option<Vec<usize>>,
}

/// Factory reading the `perm` attribute from the node.
pub struct TransposeFactory;

impl KernelFactory for TransposeFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let perm = node
            .attr("perm")
            .and_then(|a| a.as_ints())
            .map(|ints| ints.iter().map(|&v| v as usize).collect::<Vec<_>>());
        Ok(Box::new(TransposeKernel { perm }))
    }
}

impl Kernel for TransposeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Transpose", inputs, outputs, 1, 1, 1)?;
        let in_shape = inputs[0].shape.to_vec();
        let rank = in_shape.len();
        let perm = match &self.perm {
            Some(p) => {
                if p.len() != rank {
                    return Err(EpError::KernelFailed(format!(
                        "Transpose: perm rank {} != input rank {rank}",
                        p.len()
                    )));
                }
                p.clone()
            }
            None => (0..rank).rev().collect(),
        };

        let din = to_dense_f32(&inputs[0])?;
        let in_strides = compute_contiguous_strides(&in_shape);
        // Output axis i corresponds to input axis perm[i].
        let out_shape: Vec<usize> = perm.iter().map(|&p| in_shape[p]).collect();
        let mut out = vec![0.0f32; numel(&out_shape)];

        if !out.is_empty() {
            let mut oidx = vec![0usize; rank];
            let mut flat = 0usize;
            loop {
                // input index: in[perm[i]] = out[i]
                let mut in_flat = 0i64;
                for (i, &p) in perm.iter().enumerate() {
                    in_flat += in_strides[p] * oidx[i] as i64;
                }
                out[flat] = din[in_flat as usize];
                flat += 1;
                if !next_index(&out_shape, &mut oidx) {
                    break;
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

    fn run(perm: Option<Vec<usize>>, input: &Owned, out: &mut Owned) {
        let k = TransposeKernel { perm };
        k.execute(&[input.view()], &mut [out.view_mut()]).unwrap();
    }

    #[test]
    fn transpose_2d_default_reverses() {
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let mut out = Owned::zeros_f32(&[3, 2]);
        run(None, &a, &mut out);
        // [[1,4],[2,5],[3,6]]
        assert_eq!(out.to_f32(), vec![1., 4., 2., 5., 3., 6.]);
    }

    #[test]
    fn transpose_3d_perm() {
        // shape [2,1,3], perm [1,0,2] -> [1,2,3]
        let a = Owned::f32(&[2, 1, 3], &[1., 2., 3., 4., 5., 6.]);
        let mut out = Owned::zeros_f32(&[1, 2, 3]);
        run(Some(vec![1, 0, 2]), &a, &mut out);
        assert_eq!(out.to_f32(), vec![1., 2., 3., 4., 5., 6.]);
    }

    #[test]
    fn transpose_3d_swap_last_two() {
        // shape [1,2,3], perm [0,2,1] -> [1,3,2]
        let a = Owned::f32(&[1, 2, 3], &[1., 2., 3., 4., 5., 6.]);
        let mut out = Owned::zeros_f32(&[1, 3, 2]);
        run(Some(vec![0, 2, 1]), &a, &mut out);
        // rows [1,2,3],[4,5,6] transposed -> [1,4],[2,5],[3,6]
        assert_eq!(out.to_f32(), vec![1., 4., 2., 5., 3., 6.]);
    }
}
