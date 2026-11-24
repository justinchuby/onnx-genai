#[path = "packed_shape_only/cases.rs"]
mod cases;

use cases::{
    CASES, IR9_FP8_TYPES, TYPES, assert_ir9_fp8_reshape, assert_transport, dynamic_reshape_graph,
    graph,
};
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
fn reshape_19_dynamic_fp8_cpu_transport() {
    assert_ir9_fp8_reshape(
        |graph| InferenceSession::from_graph(graph).unwrap(),
        DeviceId::cpu(),
    );
}

#[test]
fn shape_only_ir9_fp8_opset_boundaries() {
    for dtype in IR9_FP8_TYPES {
        let Err(error) = InferenceSession::from_graph(dynamic_reshape_graph(18, dtype)) else {
            panic!("Reshape-18 accepted {dtype:?}");
        };
        let message = error.to_string();
        assert!(
            message.contains("Reshape")
                && message.contains(&format!("{dtype:?}"))
                && message.contains("at opset 18")
                && message.contains("use opset 19"),
            "{message}"
        );
        for case @ (op, ..) in CASES.into_iter().skip(1) {
            let Err(error) = InferenceSession::from_graph(graph(20, dtype, case)) else {
                panic!("{op} accepted {dtype:?} before opset 21");
            };
            let message = error.to_string();
            assert!(
                message.contains(op)
                    && message.contains(&format!("{dtype:?}"))
                    && message.contains("at opset 20")
                    && message.contains("use opset 21"),
                "{message}"
            );
        }
    }
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
