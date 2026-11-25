//! Shared schema admission for native `OneHot` kernels.

use onnx_runtime_ir::DataType;

/// First schema admitting BF16 for `values` and the output.
pub const BF16_SINCE_VERSION: u64 = 28;

/// Check the opset-dependent constraint on `values` (which also sets the output dtype).
/// Each provider remains responsible for its own kernel's dtype and layout limits.
pub fn validate_values_dtype(opset: u64, input_dtypes: &[DataType]) -> Result<(), String> {
    if opset < BF16_SINCE_VERSION && input_dtypes.get(2) == Some(&DataType::BFloat16) {
        return Err(format!(
            "input 'values' (and the output) cannot use BFloat16 at opset {opset}; use OneHot-28 or later, or a dtype admitted by the earlier schema"
        ));
    }
    Ok(())
}
