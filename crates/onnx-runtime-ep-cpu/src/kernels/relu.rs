//! `Relu`: elementwise `max(0, x)` for f32 (`docs/ORT2.md` §4.4).

use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};
use onnx_runtime_ep_api::{Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::check_arity;

/// Apply `max(0, x)` in place. Shared with the fused `FusedGemm` kernel so the
/// ReLU activation has a single source of truth.
///
/// NaN is propagated (not clamped to 0): ONNX/numpy `maximum(0, NaN)` is NaN,
/// whereas Rust's `f32::max` would return the non-NaN operand (`0.0`) and
/// silently drop the NaN.
pub(crate) fn relu_in_place(data: &mut [f32]) {
    for v in data.iter_mut() {
        if !v.is_nan() {
            *v = v.max(0.0);
        }
    }
}

/// Stateless f32 ReLU kernel.
pub struct ReluKernel;

/// Factory for [`ReluKernel`] (no attributes).
pub struct ReluFactory;

impl KernelFactory for ReluFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ReluKernel))
    }
}

impl Kernel for ReluKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Relu", inputs, outputs, 1, 1, 1)?;
        #[cfg(feature = "mlas")]
        if relu_contiguous_f32(&inputs[0], &mut outputs[0])? {
            return Ok(());
        }
        let x = to_dense_f32_widen("Relu", &inputs[0])?;
        let mut y = x.into_owned();
        relu_in_place(&mut y);
        write_dense_f32_narrow("Relu", &mut outputs[0], &y)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }

    fn estimated_flops(&self) -> Option<u64> {
        None
    }
}

#[cfg(feature = "mlas")]
fn relu_contiguous_f32(input: &TensorView, output: &mut TensorMut) -> Result<bool> {
    if input.dtype != onnx_runtime_ir::DataType::Float32
        || output.dtype != onnx_runtime_ir::DataType::Float32
        || input.shape != output.shape
        || !input.is_contiguous()
        || !output.is_contiguous()
    {
        return Ok(false);
    }
    let input_start = input.data_ptr::<u8>() as usize;
    let input_end = input_start.saturating_add(input.byte_size());
    let output_start = output.data_ptr_mut::<u8>() as usize;
    let output_end = output_start.saturating_add(output.byte_size());
    if output_start < input_end && input_start < output_end {
        return Ok(false);
    }
    let input = to_dense_f32_widen("Relu", input)?;
    let output_len = output.numel();
    // SAFETY: equal contiguous Float32 shapes prove the output span, and the
    // range check proves it does not overlap the borrowed input.
    let output =
        unsafe { std::slice::from_raw_parts_mut(output.data_ptr_mut::<f32>(), output_len) };
    mlas_sys::compute_relu(&input, output);
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    #[test]
    fn relu_clamps_negatives() {
        let a = Owned::f32(&[2, 2], &[-1.0, 2.0, -3.0, 4.0]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        let inputs = [a.view()];
        let mut outs = [out.view_mut()];
        ReluKernel.execute(&inputs, &mut outs).unwrap();
        assert_eq!(out.to_f32(), vec![0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn relu_propagates_nan() {
        // ONNX/numpy maximum(0, NaN) == NaN; f32::max would wrongly yield 0.
        let mut data = vec![f32::NAN, -1.0, 2.0];
        relu_in_place(&mut data);
        assert!(data[0].is_nan());
        assert_eq!(data[1], 0.0);
        assert_eq!(data[2], 2.0);
    }
    #[test]
    fn relu_bf16_matches_widened_f32_reference_and_preserves_nan() {
        let x = Owned::bf16(&[5], &[f32::NAN, -80., -0., 1., 80.]);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[5]);
        ReluKernel
            .execute(&[x.view()], &mut [out.view_mut()])
            .unwrap();
        let result = out.to_bf16_as_f32();
        assert!(result[0].is_nan());
        assert_eq!(&result[1..], &[0., 0., 1., 80.]);
    }
}
