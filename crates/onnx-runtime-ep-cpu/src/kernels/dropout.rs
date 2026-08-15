//! ONNX `Dropout` inference/evaluation kernel.
//!
//! Evaluation mode, and training with a zero ratio, are deterministic identity
//! operations. The optional mask is all true. Non-zero training dropout is
//! rejected rather than pretending to reproduce an implementation-specific RNG
//! stream.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::{elem_size, to_dense_bytes, write_dense_bytes};
use crate::strided::numel;

pub struct DropoutKernel;
pub struct DropoutFactory;

impl KernelFactory for DropoutFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(DropoutKernel))
    }
}

impl Kernel for DropoutKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.is_empty() || inputs.len() > 3 {
            return Err(EpError::KernelFailed(format!(
                "Dropout: expected 1..=3 inputs, got {}",
                inputs.len()
            )));
        }
        if !(1..=2).contains(&outputs.len()) {
            return Err(EpError::KernelFailed(format!(
                "Dropout: expected 1..=2 outputs, got {}",
                outputs.len()
            )));
        }

        let data = &inputs[0];
        validate_data_output(data, &outputs[0])?;
        if outputs.len() == 2 {
            let mask = &outputs[1];
            if mask.dtype != DataType::Bool || mask.shape != data.shape {
                return Err(EpError::KernelFailed(format!(
                    "Dropout: mask must be Bool with shape {:?}, got {:?}{:?}",
                    data.shape, mask.dtype, mask.shape
                )));
            }
        }

        let ratio = inputs
            .get(1)
            .filter(|input| !input.is_absent())
            .map(read_ratio)
            .transpose()?
            .unwrap_or(0.5);
        if !(0.0..1.0).contains(&ratio) {
            return Err(EpError::KernelFailed(format!(
                "Dropout: ratio must be in [0, 1), got {ratio}"
            )));
        }
        let training = inputs
            .get(2)
            .filter(|input| !input.is_absent())
            .map(read_training_mode)
            .transpose()?
            .unwrap_or(false);

        if training && ratio != 0.0 {
            return Err(EpError::KernelFailed(
                "Dropout: non-zero training dropout is unsupported because ONNX does not \
                 prescribe a portable RNG stream; use evaluation mode or ratio=0"
                    .into(),
            ));
        }

        let bytes = to_dense_bytes(data)?;
        write_dense_bytes(&mut outputs[0], &bytes)?;
        if outputs.len() == 2 {
            write_dense_bytes(&mut outputs[1], &vec![1; numel(data.shape)])?;
        }
        Ok(())
    }

    fn supports_strided_input(&self, input_idx: usize) -> bool {
        input_idx == 0
    }
}

fn validate_data_output(input: &TensorView, output: &TensorMut) -> Result<()> {
    elem_size(input.dtype)?;
    if input.dtype == DataType::String {
        return Err(EpError::KernelFailed(
            "Dropout: String data is not supported".into(),
        ));
    }
    if output.dtype != input.dtype || output.shape != input.shape {
        return Err(EpError::KernelFailed(format!(
            "Dropout: output must match input dtype/shape {:?}{:?}, got {:?}{:?}",
            input.dtype, input.shape, output.dtype, output.shape
        )));
    }
    Ok(())
}

fn read_ratio(input: &TensorView) -> Result<f64> {
    if !input.shape.is_empty() {
        return Err(EpError::KernelFailed(format!(
            "Dropout: ratio must be a scalar, got shape {:?}",
            input.shape
        )));
    }
    let bytes = to_dense_bytes(input)?;
    match input.dtype {
        DataType::Float16 => Ok(half::f16::from_le_bytes(bytes.try_into().unwrap()).to_f64()),
        DataType::BFloat16 => Ok(half::bf16::from_le_bytes(bytes.try_into().unwrap()).to_f64()),
        DataType::Float32 => Ok(f32::from_le_bytes(bytes.try_into().unwrap()) as f64),
        DataType::Float64 => Ok(f64::from_le_bytes(bytes.try_into().unwrap())),
        dtype => Err(EpError::KernelFailed(format!(
            "Dropout: ratio must have a floating-point scalar dtype, got {dtype:?}"
        ))),
    }
}

fn read_training_mode(input: &TensorView) -> Result<bool> {
    if input.dtype != DataType::Bool || !input.shape.is_empty() {
        return Err(EpError::KernelFailed(format!(
            "Dropout: training_mode must be a Bool scalar, got {:?}{:?}",
            input.dtype, input.shape
        )));
    }
    Ok(to_dense_bytes(input)?[0] != 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    #[test]
    fn inference_is_bit_exact_and_emits_true_mask() {
        let input = Owned::f32(&[2, 3], &[1.0, -2.5, 0.0, 4.25, f32::NAN, f32::INFINITY]);
        let ratio = Owned::f32(&[], &[0.75]);
        let training = Owned::bool_(&[], &[false]);
        let mut output = Owned::zeros_f32(&[2, 3]);
        let mut mask = Owned::zeros(DataType::Bool, &[2, 3]);

        DropoutKernel
            .execute(
                &[input.view(), ratio.view(), training.view()],
                &mut [output.view_mut(), mask.view_mut()],
            )
            .unwrap();

        assert_eq!(output.bytes, input.bytes);
        assert_eq!(mask.to_bool(), vec![true; 6]);
    }

    #[test]
    fn omitted_optional_inputs_default_to_evaluation() {
        let input = Owned::bf16_bits(&[3], &[0x3f80, 0x8000, 0x7fc1]);
        let mut output = Owned::zeros(DataType::BFloat16, &[3]);

        DropoutKernel
            .execute(&[input.view()], &mut [output.view_mut()])
            .unwrap();

        assert_eq!(output.bytes, input.bytes);
    }

    #[test]
    fn zero_ratio_training_is_identity_with_true_mask() {
        let input = Owned::f32(&[4], &[1.0, 2.0, 3.0, 4.0]);
        let ratio = Owned::f32(&[], &[0.0]);
        let training = Owned::bool_(&[], &[true]);
        let mut output = Owned::zeros_f32(&[4]);
        let mut mask = Owned::zeros(DataType::Bool, &[4]);

        DropoutKernel
            .execute(
                &[input.view(), ratio.view(), training.view()],
                &mut [output.view_mut(), mask.view_mut()],
            )
            .unwrap();

        assert_eq!(output.to_f32(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(mask.to_bool(), vec![true; 4]);
    }

    #[test]
    fn nonzero_training_mode_is_explicitly_unsupported() {
        let input = Owned::f32(&[2], &[1.0, 2.0]);
        let ratio = Owned::f32(&[], &[0.5]);
        let training = Owned::bool_(&[], &[true]);
        let mut output = Owned::zeros_f32(&[2]);

        let error = DropoutKernel
            .execute(
                &[input.view(), ratio.view(), training.view()],
                &mut [output.view_mut()],
            )
            .unwrap_err();

        assert!(error.to_string().contains("portable RNG stream"));
    }
}
