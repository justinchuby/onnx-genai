#![deny(warnings)]

use onnx_runtime_ep_cuda::kernels::einsum::{unsupported_reason, unsupported_reason_for_opset};
use onnx_runtime_ir::{Attribute, DataType, Node, NodeId, TensorLayout, static_shape};

fn einsum_node() -> Node {
    let mut node = Node::new(NodeId(0), "Einsum", vec![], vec![]);
    node.attributes
        .insert("equation".into(), Attribute::String(b"i->i".to_vec()));
    node
}

#[test]
fn legacy_cuda_helper_keeps_its_einsum_12_signature_and_semantics() {
    let reason: Option<String> = unsupported_reason(
        &einsum_node(),
        &[static_shape([2])],
        &[DataType::BFloat16],
        &[TensorLayout::contiguous()],
    );
    let reason = reason.expect("the legacy helper must apply the Einsum-12 schema");

    assert!(reason.contains("not admitted by Einsum-12"), "{reason}");
}

#[test]
fn schema_aware_cuda_helper_uses_the_explicit_effective_opset() {
    let node = einsum_node();
    let shapes = [static_shape([2])];
    let layouts = [TensorLayout::contiguous()];

    assert!(
        unsupported_reason_for_opset(&node, 12, &shapes, &[DataType::Float32], &layouts,).is_none()
    );

    let opset27 =
        unsupported_reason_for_opset(&node, 27, &shapes, &[DataType::BFloat16], &layouts).unwrap();
    assert!(opset27.contains("not admitted by Einsum-12"), "{opset27}");

    let opset28 =
        unsupported_reason_for_opset(&node, 28, &shapes, &[DataType::BFloat16], &layouts).unwrap();
    assert!(
        opset28.contains("Einsum-28 admits BFloat16 semantically"),
        "{opset28}"
    );
}
