//! ONNX `Dropout` inference/evaluation kernel.
//!
//! Evaluation mode, and training with a zero ratio, are deterministic identity
//! operations. The optional mask is all true. Non-zero training dropout is
//! rejected rather than pretending to reproduce an implementation-specific RNG
//! stream.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::dropout::{validate_data_type, validate_ratio, validate_training_mode};
use onnx_runtime_ir::{DataType, Node};

use super::{to_dense_bytes, write_dense_bytes};
use crate::strided::numel;

pub struct DropoutKernel {
    opset: u64,
}
pub struct DropoutFactory {
    pub opset: u64,
}

impl KernelFactory for DropoutFactory {
    fn create(&self, _: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(DropoutKernel { opset: self.opset }))
    }
}

impl Kernel for DropoutKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if !(1..=3).contains(&inputs.len()) || inputs[0].is_absent() {
            return Err(EpError::KernelFailed(format!(
                "Dropout: expected data and up to two optional inputs, got {} input slots",
                inputs.len()
            )));
        }
        if !(1..=2).contains(&outputs.len()) || outputs[0].is_absent() {
            return Err(EpError::KernelFailed(format!(
                "Dropout: expected data output and optional mask, got {} output slots",
                outputs.len()
            )));
        }

        let data = &inputs[0];
        validate_data_type(self.opset, data.dtype).map_err(dropout_error)?;
        validate_data_output(data, &outputs[0])?;
        let (data_outputs, mask_outputs) = outputs.split_at_mut(1);
        let mask = mask_outputs.first_mut().filter(|v| !v.is_absent());
        if let Some(mask) = mask.as_ref() {
            if mask.dtype != DataType::Bool || mask.shape != data.shape {
                return Err(EpError::KernelFailed(format!(
                    "Dropout: mask must be Bool with shape {:?}, got {:?}{:?}",
                    data.shape, mask.dtype, mask.shape
                )));
            }
        }

        let ratio = inputs.get(1).filter(|input| !input.is_absent());
        if let Some(ratio) = ratio {
            validate_ratio(self.opset, ratio.dtype, ratio.shape.len()).map_err(dropout_error)?;
        }
        let training = inputs
            .get(2)
            .filter(|input| !input.is_absent())
            .map(read_training_mode)
            .transpose()?
            .unwrap_or(false);

        if training {
            let ratio = ratio.map(read_ratio).transpose()?.unwrap_or(0.5);
            if !(0.0..1.0).contains(&ratio) {
                return Err(dropout_error(format!(
                    "ratio must be in [0, 1), got {ratio}"
                )));
            }
            if ratio != 0.0 {
                return Err(dropout_error(
                    "non-zero training dropout is unsupported because ONNX does not \
                     prescribe a portable RNG stream; use evaluation mode or ratio=0"
                        .into(),
                ));
            }
        }

        let bytes = to_dense_bytes(data)?;
        write_dense_bytes(&mut data_outputs[0], &bytes)?;
        if let Some(mask) = mask {
            write_dense_bytes(mask, &vec![1; numel(data.shape)])?;
        }
        Ok(())
    }

    fn supports_strided_input(&self, input_idx: usize) -> bool {
        input_idx == 0
    }
}

fn validate_data_output(input: &TensorView, output: &TensorMut) -> Result<()> {
    if output.dtype != input.dtype || output.shape != input.shape {
        return Err(EpError::KernelFailed(format!(
            "Dropout: output must match input dtype/shape {:?}{:?}, got {:?}{:?}",
            input.dtype, input.shape, output.dtype, output.shape
        )));
    }
    Ok(())
}

fn read_ratio(input: &TensorView) -> Result<f64> {
    let bytes = to_dense_bytes(input)?;
    match input.dtype {
        DataType::Float16 => Ok(half::f16::from_le_bytes(bytes.try_into().unwrap()).to_f64()),
        DataType::BFloat16 => Ok(half::bf16::from_le_bytes(bytes.try_into().unwrap()).to_f64()),
        DataType::Float32 => Ok(f32::from_le_bytes(bytes.try_into().unwrap()) as f64),
        DataType::Float64 => Ok(f64::from_le_bytes(bytes.try_into().unwrap())),
        dtype => Err(dropout_error(format!(
            "training with ratio dtype {dtype:?} is unsupported; use evaluation mode or an f32 ratio of 0"
        ))),
    }
}

fn read_training_mode(input: &TensorView) -> Result<bool> {
    validate_training_mode(input.dtype, input.shape.len()).map_err(dropout_error)?;
    Ok(to_dense_bytes(input)?[0] != 0)
}

fn dropout_error(reason: String) -> EpError {
    EpError::KernelFailed(format!("Dropout: {reason}"))
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

        DropoutKernel { opset: 13 }
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

        DropoutKernel { opset: 13 }
            .execute(&[input.view()], &mut [output.view_mut()])
            .unwrap();

        assert_eq!(output.bytes, input.bytes);
    }

    #[test]
    fn opset22_inference_ignores_fp8_ratio_and_preserves_fp8_bytes() {
        let mut input = Owned::zeros(DataType::Float8E4M3FN, &[3]);
        input.bytes.copy_from_slice(&[0x38, 0x80, 0x7f]);
        let mut ratio = Owned::zeros(DataType::Float8E5M2, &[]);
        ratio.bytes[0] = 0x7f; // NaN: evaluation must not interpret the ratio.
        let training = Owned::bool_(&[], &[false]);
        let mut output = Owned::zeros(DataType::Float8E4M3FN, &[3]);
        let mut mask = Owned::zeros(DataType::Bool, &[3]);

        for ratio in [ratio.view(), TensorView::absent(DataType::Undefined)] {
            DropoutKernel { opset: 22 }
                .execute(
                    &[input.view(), ratio, training.view()],
                    &mut [output.view_mut(), mask.view_mut()],
                )
                .unwrap();
            assert_eq!(output.bytes, input.bytes);
            assert_eq!(mask.to_bool(), vec![true; 3]);
        }

        let error = DropoutKernel { opset: 13 }
            .execute(&[input.view()], &mut [output.view_mut()])
            .unwrap_err();
        assert!(error.to_string().contains("opset 13"), "{error}");
    }

    #[test]
    fn zero_ratio_training_is_identity_with_true_mask() {
        let input = Owned::f32(&[4], &[1.0, 2.0, 3.0, 4.0]);
        let ratio = Owned::f32(&[], &[0.0]);
        let training = Owned::bool_(&[], &[true]);
        let mut output = Owned::zeros_f32(&[4]);
        let mut mask = Owned::zeros(DataType::Bool, &[4]);

        DropoutKernel { opset: 13 }
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

        let error = DropoutKernel { opset: 13 }
            .execute(
                &[input.view(), ratio.view(), training.view()],
                &mut [output.view_mut()],
            )
            .unwrap_err();

        assert!(error.to_string().contains("portable RNG stream"));
    }
}
