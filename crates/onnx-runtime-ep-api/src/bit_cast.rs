//! Validate the shared contiguous-storage contract before a provider copy.

use crate::{EpError, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, checked_numel, opset26::validate_bit_cast};

pub fn copy_size(input: &TensorView, output: &TensorMut, target: DataType) -> Result<usize> {
    validate_bit_cast(input.dtype, target).map_err(EpError::KernelFailed)?;
    if output.dtype != target || output.shape != input.shape {
        return Err(EpError::KernelFailed(format!(
            "BitCast: output must have dtype {target:?} and input shape {:?}, got {:?} {:?}",
            input.shape, output.dtype, output.shape
        )));
    }
    input.validate()?;
    output.validate()?;
    let bytes = checked_numel(input.shape)
        .and_then(|n| input.dtype.checked_storage_bytes(n))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| {
            EpError::KernelFailed(format!(
                "BitCast: shape {:?} exceeds addressable storage",
                input.shape
            ))
        })?;
    if bytes != 0 && (!input.is_contiguous() || !output.is_contiguous()) {
        return Err(EpError::KernelFailed(
            "BitCast: materialize input and output as contiguous tensors before reinterpreting their storage".into(),
        ));
    }
    Ok(bytes)
}
