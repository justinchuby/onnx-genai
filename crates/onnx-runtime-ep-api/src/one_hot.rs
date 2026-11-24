//! Shared schema admission for native `OneHot` kernels.

use onnx_runtime_ir::DataType;

/// Check the opset-dependent constraint on `values` (which also sets the output dtype).
/// Each provider remains responsible for its own kernel's dtype and layout limits.
pub fn validate_values_dtype(opset: u64, input_dtypes: &[DataType]) -> Result<(), String> {
    if opset < 28 && input_dtypes.get(2) == Some(&DataType::BFloat16) {
        return Err(format!(
            "input 'values' (and the output) cannot use BFloat16 at opset {opset}; use OneHot-28 or later, or a dtype admitted by the earlier schema"
        ));
    }
    Ok(())
}
