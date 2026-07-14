//! Pure metadata data-movement ops: `Flatten`, `Squeeze`, `Size`
//! (`docs/ORT2.md` §4.4).
//!
//! `Flatten` and `Squeeze` only change a tensor's *shape*, never its row-major
//! element order, and the runtime pre-allocates the output with the target
//! shape (computed by shape inference). Each kernel therefore moves raw element
//! bytes through [`to_dense_bytes`]/[`write_dense_bytes`], serving every
//! fixed-width dtype uniformly. `Squeeze`'s optional `axes` input is consumed
//! upstream when the output shape is built, so the kernel ignores it.
//!
//! `Size` reports the input's total element count as a rank-0 `int64` scalar; it
//! reads only shape metadata and is dtype-agnostic on its input.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::{check_arity, to_dense_bytes, write_dense_bytes};
use crate::strided::numel;

/// Stateless `Flatten` kernel (row-major byte copy into the pre-shaped output).
pub struct FlattenKernel;

/// Factory for [`FlattenKernel`] (the `axis` attribute only affects the output
/// shape, which is resolved upstream).
pub struct FlattenFactory;

impl KernelFactory for FlattenFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(FlattenKernel))
    }
}

impl Kernel for FlattenKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Flatten", inputs, outputs, 1, 1, 1)?;
        let bytes = to_dense_bytes(&inputs[0])?;
        write_dense_bytes(&mut outputs[0], &bytes)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

/// Stateless `Squeeze` kernel (row-major byte copy into the pre-shaped output).
pub struct SqueezeKernel;

/// Factory for [`SqueezeKernel`] (`axes` from attribute or input 1 only affects
/// the output shape, which is resolved upstream).
pub struct SqueezeFactory;

impl KernelFactory for SqueezeFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(SqueezeKernel))
    }
}

impl Kernel for SqueezeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        // data (+ optional axes input, which is metadata only here).
        check_arity("Squeeze", inputs, outputs, 1, 2, 1)?;
        let bytes = to_dense_bytes(&inputs[0])?;
        write_dense_bytes(&mut outputs[0], &bytes)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

/// Stateless `Size` kernel: input element count as a rank-0 `int64` scalar.
pub struct SizeKernel;

/// Factory for [`SizeKernel`] (no attributes).
pub struct SizeFactory;

impl KernelFactory for SizeFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(SizeKernel))
    }
}

impl Kernel for SizeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Size", inputs, outputs, 1, 1, 1)?;
        if outputs[0].dtype != DataType::Int64 {
            return Err(EpError::KernelFailed(format!(
                "Size: output must be Int64, got {:?}. WHY: ONNX Size yields an int64 scalar. \
                 HOW: allocate the output as Int64.",
                outputs[0].dtype
            )));
        }
        let n = numel(inputs[0].shape) as i64;
        write_dense_bytes(&mut outputs[0], &n.to_le_bytes())
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

    #[test]
    fn flatten_copies_row_major() {
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        FlattenKernel
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![1., 2., 3., 4., 5., 6.]);
    }

    #[test]
    fn squeeze_copies_bytes_int64() {
        let a = Owned::i64(&[1, 3, 1], &[7, 8, 9]);
        let mut out = Owned::zeros(DataType::Int64, &[3]);
        SqueezeKernel
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i64(), vec![7, 8, 9]);
    }

    #[test]
    fn squeeze_with_axes_input_ignores_axes_data() {
        let a = Owned::f32(&[1, 2], &[3., 4.]);
        let axes = Owned::i64(&[1], &[0]);
        let mut out = Owned::zeros_f32(&[2]);
        SqueezeKernel
            .execute(&[a.view(), axes.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![3., 4.]);
    }

    #[test]
    fn size_reports_element_count() {
        let a = Owned::f32(&[2, 3, 4], &[0.0; 24]);
        let mut out = Owned::zeros(DataType::Int64, &[]);
        SizeKernel
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i64(), vec![24]);
    }

    #[test]
    fn size_of_scalar_is_one() {
        let a = Owned::f32(&[], &[5.0]);
        let mut out = Owned::zeros(DataType::Int64, &[]);
        SizeKernel
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i64(), vec![1]);
    }
}
