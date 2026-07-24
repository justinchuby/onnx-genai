//! `Unsqueeze`: insert size-1 dimensions at the positions named by `axes`
//! (`docs/ORT2.md` §4.4).
//!
//! Two axis-signature forms are supported:
//! * **Legacy (opset ≤ 12):** `axes` is an **attribute**.
//! * **Modern (opset ≥ 13):** `axes` moved to a required **second input**
//!   (`int64` tensor).
//!
//! Unsqueeze never reorders elements — it only reshapes — so the kernel is a
//! dtype-agnostic row-major byte copy; the pre-allocated output view already
//! carries the unsqueezed shape. `axes` is read only to validate the rank
//! change.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::{check_arity, to_dense_bytes, to_dense_i64, write_dense_bytes};

/// Unsqueeze kernel carrying the legacy `axes` attribute (`None` when axes
/// arrive as the opset-13 input).
pub struct UnsqueezeKernel {
    axes: Option<Vec<i64>>,
}

/// Factory reading the legacy `axes` attribute (absent in the opset-13 form).
pub struct UnsqueezeFactory;

impl KernelFactory for UnsqueezeFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let axes = node
            .attr("axes")
            .and_then(|a| a.as_ints())
            .map(|v| v.to_vec());
        Ok(Box::new(UnsqueezeKernel { axes }))
    }
}

impl Kernel for UnsqueezeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        // Opset ≥ 13 adds the required `axes` input; opset ≤ 12 has data only.
        check_arity("Unsqueeze", inputs, outputs, 1, 2, 1)?;

        // The opset-13 `axes` input takes precedence over the legacy attribute.
        let axes_len = if inputs.len() >= 2 && !inputs[1].is_absent() {
            to_dense_i64(&inputs[1])?.len()
        } else {
            self.axes
                .as_ref()
                .ok_or_else(|| {
                    EpError::KernelFailed(
                        "Unsqueeze: `axes` not supplied — provide it as the opset-13 second \
                         input or as the opset-12 `axes` attribute"
                            .into(),
                    )
                })?
                .len()
        };

        let expected_rank = inputs[0].shape.len() + axes_len;
        if outputs[0].shape.len() != expected_rank {
            return Err(EpError::KernelFailed(format!(
                "Unsqueeze: output rank {} != input rank {} + {} axes",
                outputs[0].shape.len(),
                inputs[0].shape.len(),
                axes_len
            )));
        }
        let data = to_dense_bytes(&inputs[0])?;
        write_dense_bytes(&mut outputs[0], &data)
    }

    fn supports_strided_input(&self, input_idx: usize) -> bool {
        // Only the data input (0) is materialized strided; the axes input (1)
        // goes through `to_dense_i64`, which handles strides itself.
        input_idx == 0
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
        assert!(
            UnsqueezeKernel { axes: None }
                .execute(&[x.view()], &mut [out.view_mut()])
                .is_err()
        );
    }

    #[test]
    fn unsqueeze_axes_as_input_opset13() {
        // Opset-13 form: axes arrive as a second int64 input. [3] -> [1,3].
        let x = Owned::f32(&[3], &[1., 2., 3.]);
        let axes = Owned::i64(&[1], &[0]);
        let mut out = Owned::zeros_f32(&[1, 3]);
        UnsqueezeKernel { axes: None }
            .execute(&[x.view(), axes.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![1., 2., 3.]);
    }

    #[test]
    fn unsqueeze_axes_input_multiple() {
        // [2] -> [1,2,1] via axes input [0,2] on an int64 tensor.
        let x = Owned::i64(&[2], &[5, 9]);
        let axes = Owned::i64(&[2], &[0, 2]);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::Int64, &[1, 2, 1]);
        UnsqueezeKernel { axes: None }
            .execute(&[x.view(), axes.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i64(), vec![5, 9]);
    }

    #[test]
    fn unsqueeze_axes_input_rank_mismatch_errors() {
        // Output rank must equal input rank + number of axes.
        let x = Owned::f32(&[3], &[1., 2., 3.]);
        let axes = Owned::i64(&[2], &[0, 1]);
        let mut out = Owned::zeros_f32(&[1, 3]);
        assert!(
            UnsqueezeKernel { axes: None }
                .execute(&[x.view(), axes.view()], &mut [out.view_mut()])
                .is_err()
        );
    }
    #[test]
    fn unsqueeze_bf16_preserves_element_bits() {
        let x = Owned::bf16(&[2], &[1., -2.]);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[1, 2]);
        UnsqueezeKernel {
            axes: Some(vec![0]),
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_u16_bits(), x.to_u16_bits());
    }
}
