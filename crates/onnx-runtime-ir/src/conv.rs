//! Shared ONNX `Conv` schema admission rules.

use crate::{DataType, Node, is_default_domain};

pub const BF16_SINCE_OPSET: u64 = 22;
pub const PRE_BF16_DTYPES: &[DataType] = &[DataType::Float32, DataType::Float16];
pub const DTYPES: &[DataType] = &[DataType::Float32, DataType::Float16, DataType::BFloat16];

pub fn unsupported_reason(node: &Node, opset: u64, input_dtypes: &[DataType]) -> Option<String> {
    if node.op_type != "Conv"
        || !is_default_domain(&node.domain)
        || opset >= BF16_SINCE_OPSET
        || !input_dtypes.contains(&DataType::BFloat16)
    {
        return None;
    }
    Some(format!(
        "Conv: BFloat16 entered the ai.onnx Conv schema at opset {BF16_SINCE_OPSET}, \
         but this model uses opset {opset}; export Conv at opset {BF16_SINCE_OPSET} or later, \
         or use Float16/Float32 tensors"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NodeId;

    #[test]
    fn bf16_admission_starts_at_opset_22() {
        let node = Node::new(NodeId(0), "Conv", vec![], vec![]);
        let dtypes = [DataType::BFloat16, DataType::BFloat16];

        let reason = unsupported_reason(&node, 21, &dtypes).expect("opset 21 must reject BF16");
        assert!(reason.contains("opset 22"), "{reason}");
        assert!(unsupported_reason(&node, 22, &dtypes).is_none());
    }

    #[test]
    fn admission_rule_is_named_conv_only() {
        let node = Node::new(NodeId(0), "ConvTranspose", vec![], vec![]);
        assert!(unsupported_reason(&node, 21, &[DataType::BFloat16]).is_none());
    }
}
