//! Shared ONNX version constraint for the native TopK kernels.

use onnx_runtime_ir::DataType;

/// Check the BF16 revision independently of each provider's kernel dtype support.
pub fn validate_bfloat16_opset(opset: u64, input_dtypes: &[DataType]) -> Result<(), String> {
    if input_dtypes.first() == Some(&DataType::BFloat16) && opset < 24 {
        return Err(format!(
            "TopK: X dtype BFloat16 is not valid at opset {opset}; export at opset 24 or later, or supply Float16 or Float32 data"
        ));
    }
    Ok(())
}
