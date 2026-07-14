//! `Shape`: return the input tensor's dimensions as a 1-D `int64` tensor
//! (`docs/ORT2.md` §4.4).
//!
//! Opset-12 `Shape` has no `start`/`end` attributes (those arrived in opset 15),
//! so it always yields the full shape vector. It reads no element data — only
//! the input view's shape metadata — and is therefore dtype-agnostic.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::{check_arity, write_dense_bytes};

/// Stateless Shape kernel.
pub struct ShapeKernel;

/// Factory for [`ShapeKernel`] (no attributes in opset 12).
pub struct ShapeFactory;

impl KernelFactory for ShapeFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ShapeKernel))
    }
}

impl Kernel for ShapeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Shape", inputs, outputs, 1, 1, 1)?;
        if outputs[0].dtype != DataType::Int64 {
            return Err(EpError::KernelFailed(format!(
                "Shape: output must be Int64, got {:?}",
                outputs[0].dtype
            )));
        }
        let mut bytes = Vec::with_capacity(inputs[0].shape.len() * 8);
        for &d in inputs[0].shape {
            bytes.extend_from_slice(&(d as i64).to_le_bytes());
        }
        write_dense_bytes(&mut outputs[0], &bytes)
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
    fn shape_of_3d_tensor() {
        let x = Owned::f32(&[2, 3, 4], &[0.0; 24]);
        let mut out = Owned::zeros(DataType::Int64, &[3]);
        ShapeKernel
            .execute(&[x.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i64(), vec![2, 3, 4]);
    }

    #[test]
    fn shape_of_scalar_is_empty() {
        let x = Owned::f32(&[], &[7.0]);
        let mut out = Owned::zeros(DataType::Int64, &[0]);
        ShapeKernel
            .execute(&[x.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i64(), Vec::<i64>::new());
    }
}
