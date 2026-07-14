//! `Reshape`: a metadata-only op. The runtime pre-allocates the output with the
//! target shape, so the kernel copies the elements in row-major logical order
//! (`docs/ORT2.md` §4.4). The shape input is consumed upstream when the output
//! view is built; the kernel only moves data.

use onnx_runtime_ep_api::{Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::{check_arity, to_dense_f32, write_dense_f32};

/// Stateless f32 Reshape kernel (logical-order copy).
pub struct ReshapeKernel;

/// Factory for [`ReshapeKernel`].
pub struct ReshapeFactory;

impl KernelFactory for ReshapeFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ReshapeKernel))
    }
}

impl Kernel for ReshapeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        // data + shape tensor; the shape tensor is only metadata here.
        check_arity("Reshape", inputs, outputs, 1, 2, 1)?;
        let data = to_dense_f32(&inputs[0])?;
        write_dense_f32(&mut outputs[0], &data)
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
    fn reshape_preserves_row_major_order() {
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let mut out = Owned::zeros_f32(&[3, 2]);
        ReshapeKernel
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![1., 2., 3., 4., 5., 6.]);
    }
}
