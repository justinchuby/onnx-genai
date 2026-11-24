#[path = "packed_shape_only/cases.rs"]
mod cases;

use cases::{CASES, TYPES, assert_transport, graph};
use onnx_runtime_ir::DeviceId;
use onnx_runtime_session::InferenceSession;

#[test]
fn packed_shape_only_cpu_transport() {
    assert_transport(
        |graph| InferenceSession::from_graph(graph).unwrap(),
        DeviceId::cpu(),
    );
}

#[test]
fn packed_shape_only_rejects_early_opsets() {
    for (dtype, since, _) in TYPES {
        for case @ (op, ..) in CASES {
            let opset = since - 1;
            let Err(error) = InferenceSession::from_graph(graph(opset, dtype, case)) else {
                panic!("{op} accepted {dtype:?} at opset {opset}, before opset {since}");
            };
            let message = error.to_string();
            assert!(
                message.contains(op) && message.contains(&format!("{dtype:?}")),
                "{message}"
            );
            assert!(
                message.contains(&format!("at opset {opset}"))
                    && message.contains(&format!("use opset {since}")),
                "{message}"
            );
        }
    }
}
