//! Shared claim-time ONNX activation schema boundaries.

use onnx_runtime_ir::{DataType, Node};

const BF16_SINCE_OPSET_22: &[&str] = &["Elu", "HardSigmoid", "Selu", "Softplus", "Softsign"];

/// Return why a standard activation cannot be claimed at the effective opset.
pub fn unsupported_reason(node: &Node, opset: u64, input_dtypes: &[DataType]) -> Option<String> {
    if !node.is_default_domain()
        || !BF16_SINCE_OPSET_22.contains(&node.op_type.as_str())
        || !input_dtypes.contains(&DataType::BFloat16)
        || opset >= 22
    {
        return None;
    }
    Some(format!(
        "{} does not admit BF16 at opset {opset}; export this node at opset 22 or later, or use FLOAT16, FLOAT, or DOUBLE",
        node.op_type
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::NodeId;

    #[test]
    fn bf16_activation_boundary_is_opset_22() {
        for op_type in BF16_SINCE_OPSET_22 {
            let node = Node::new(NodeId(0), *op_type, vec![], vec![]);
            let rejected = unsupported_reason(&node, 21, &[DataType::BFloat16]).unwrap();
            assert!(rejected.contains("opset 21"), "{op_type}: {rejected}");
            assert!(rejected.contains("opset 22"), "{op_type}: {rejected}");
            assert_eq!(
                unsupported_reason(&node, 22, &[DataType::BFloat16]),
                None,
                "{op_type}"
            );
            assert_eq!(
                unsupported_reason(&node, 21, &[DataType::Float16]),
                None,
                "{op_type}"
            );
        }
    }
}
