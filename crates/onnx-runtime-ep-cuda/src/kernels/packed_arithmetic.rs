//! Host-only packed arithmetic admission shared by claim and direct factories.
use onnx_runtime_ep_api::{EpError, Result};
use onnx_runtime_ir::{Node, packed_arithmetic::ActivationArithmetic};
pub(super) fn validate(node: &Node) -> Result<()> {
    let arithmetic = ActivationArithmetic::from_node(node).map_err(EpError::KernelFailed)?;
    if arithmetic != ActivationArithmetic::Float32 {
        return Err(EpError::KernelFailed(format!(
            "{}: CUDA EP does not implement activation_arithmetic='{}'; select 'float32' with block_layout_version=1 or an EP implementing the requested arithmetic and weight format",
            node.op_type,
            arithmetic.name()
        )));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Attribute, NodeId};
    #[test]
    fn explicit_and_default_float32_remain_admitted() {
        for op in ["BlockQuantizedMatMul", "BlockQuantizedMoE"] {
            let mut node = Node::new(NodeId(0), op, vec![], vec![]);
            validate(&node).unwrap();
            node.attributes.insert(
                "activation_arithmetic".into(),
                Attribute::String(b"float32".to_vec()),
            );
            validate(&node).unwrap();
        }
    }
    #[test]
    fn quantized_and_malformed_arithmetic_are_rejected_without_a_cuda_device() {
        for op in ["BlockQuantizedMatMul", "BlockQuantizedMoE"] {
            for value in [
                Attribute::String(b"q8_0".to_vec()),
                Attribute::Int(1),
                Attribute::String(b"auto".to_vec()),
            ] {
                let mut node = Node::new(NodeId(0), op, vec![], vec![]);
                node.attributes
                    .insert("activation_arithmetic".into(), value);
                node.attributes
                    .insert("block_layout_version".into(), Attribute::Int(2));
                let error = validate(&node).unwrap_err().to_string();
                assert!(
                    error.contains(op)
                        && error.contains("activation_arithmetic")
                        && error.contains("float32"),
                    "{error}"
                );
            }
        }
    }
}
