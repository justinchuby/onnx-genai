use crate::{DataType, Node, is_default_domain};

/// Validate the ONNX version boundary for window-pooling element types.
pub fn validate_dtype_at_opset(
    node: &Node,
    opset: u64,
    input_dtypes: &[DataType],
) -> Result<(), String> {
    if is_default_domain(&node.domain)
        && matches!(node.op_type.as_str(), "AveragePool" | "MaxPool")
        && input_dtypes.first() == Some(&DataType::BFloat16)
        && opset < 22
    {
        return Err(format!(
            "{} bfloat16 requires ONNX opset 22 or later; got opset {opset}",
            node.op_type
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NodeId;

    #[test]
    fn bfloat16_window_pooling_starts_at_opset_22() {
        for op_type in ["AveragePool", "MaxPool"] {
            let node = Node::new(NodeId(0), op_type, vec![], vec![]);
            let reason = validate_dtype_at_opset(&node, 21, &[DataType::BFloat16]).unwrap_err();
            assert!(reason.contains("requires ONNX opset 22"), "{reason}");
            assert!(validate_dtype_at_opset(&node, 22, &[DataType::BFloat16]).is_ok());
            assert!(validate_dtype_at_opset(&node, 21, &[DataType::Float32]).is_ok());
        }
    }
}
