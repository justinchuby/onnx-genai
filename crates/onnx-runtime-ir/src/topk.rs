//! ONNX version constraint for BF16 TopK.

use crate::{DataType, Node};

/// Validate the BF16 schema revision independently of kernel dtype support.
pub fn validate_bfloat16_opset(
    node: &Node,
    opset: u64,
    data_dtype: DataType,
) -> Result<(), String> {
    if node.is_default_domain()
        && node.op_type == "TopK"
        && opset < 24
        && data_dtype == DataType::BFloat16
    {
        return Err(format!(
            "TopK: X dtype BFloat16 is not valid at opset {opset}; export at opset 24 or later, or supply Float16 or Float32 data"
        ));
    }
    Ok(())
}
