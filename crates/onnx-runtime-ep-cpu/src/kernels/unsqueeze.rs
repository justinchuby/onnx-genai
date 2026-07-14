//! `Unsqueeze`: insert size-1 dimensions at the positions named by `axes`
//! (`docs/ORT2.md` §4.4).
//!
//! Opset-12 form: `axes` is an **attribute** (it became a second input in opset
//! 13). Unsqueeze never reorders elements — it only reshapes — so the kernel is
//! a dtype-agnostic row-major byte copy; the pre-allocated output view already
//! carries the unsqueezed shape. `axes` is read only to validate the rank
//! change.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::{check_arity, to_dense_bytes, write_dense_bytes};

/// Unsqueeze kernel carrying the opset-12 `axes` attribute.
pub struct UnsqueezeKernel {
    axes: Option<Vec<i64>>,
}

/// Factory reading the `axes` attribute.
pub struct UnsqueezeFactory;

impl KernelFactory for UnsqueezeFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let axes = node.attr("axes").and_then(|a| a.as_ints()).map(|v| v.to_vec());
        Ok(Box::new(UnsqueezeKernel { axes }))
    }
}

impl Kernel for UnsqueezeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Unsqueeze", inputs, outputs, 1, 1, 1)?;
        let axes = self.axes.as_ref().ok_or_else(|| {
            EpError::KernelFailed("Unsqueeze: missing `axes` attribute (opset 12)".into())
        })?;
        let expected_rank = inputs[0].shape.len() + axes.len();
        if outputs[0].shape.len() != expected_rank {
            return Err(EpError::KernelFailed(format!(
                "Unsqueeze: output rank {} != input rank {} + {} axes",
                outputs[0].shape.len(),
                inputs[0].shape.len(),
                axes.len()
            )));
        }
        let data = to_dense_bytes(&inputs[0])?;
        write_dense_bytes(&mut outputs[0], &data)
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
    fn unsqueeze_inserts_axis() {
        // [3] -> [1,3] (axes=[0]); data order preserved.
        let x = Owned::f32(&[3], &[1., 2., 3.]);
        let mut out = Owned::zeros_f32(&[1, 3]);
        UnsqueezeKernel {
            axes: Some(vec![0]),
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_f32(), vec![1., 2., 3.]);
    }

    #[test]
    fn unsqueeze_multiple_axes_int64() {
        // [2] -> [1,2,1] (axes=[0,2]) on an int64 tensor (dtype-agnostic).
        let x = Owned::i64(&[2], &[5, 9]);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::Int64, &[1, 2, 1]);
        UnsqueezeKernel {
            axes: Some(vec![0, 2]),
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_i64(), vec![5, 9]);
    }

    #[test]
    fn unsqueeze_missing_axes_errors() {
        let x = Owned::f32(&[3], &[1., 2., 3.]);
        let mut out = Owned::zeros_f32(&[1, 3]);
        assert!(UnsqueezeKernel { axes: None }
            .execute(&[x.view()], &mut [out.view_mut()])
            .is_err());
    }
}
