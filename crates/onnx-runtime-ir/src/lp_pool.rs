//! Type constraints for the native LpPool-18 and LpPool-22 revisions.

use crate::DataType;

pub fn validate_dtype(opset: u64, dtype: DataType) -> Result<(), String> {
    match dtype {
        DataType::Float16 | DataType::Float32 | DataType::Float64 => Ok(()),
        DataType::BFloat16 if opset >= 22 => Ok(()),
        DataType::BFloat16 => Err(format!(
            "BF16 input X is not allowed at opset {opset}; export at opset 22 or later or use a legacy float type"
        )),
        _ => Err(format!(
            "input X has type {dtype:?} at opset {opset}; use Float16, Float32, Float64, or BF16 at opset 22 or later"
        )),
    }
}
