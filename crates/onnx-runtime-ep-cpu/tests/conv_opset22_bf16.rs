#[path = "../benches/common/mod.rs"]
mod common;

use common::{FloatDType, Tensor};
use onnx_runtime_ep_api::{ExecutionProvider, TensorMut, TensorView};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{DataType, Node, NodeId, ValueId, static_shape};

fn conv_node() -> Node {
    Node::new(
        NodeId(0),
        "Conv",
        vec![Some(ValueId(0)), Some(ValueId(1)), Some(ValueId(2))],
        vec![ValueId(3)],
    )
}

#[test]
fn bf16_conv_is_rejected_at_opset_21_and_runs_at_opset_22() {
    let ep = CpuExecutionProvider::new();
    let node = conv_node();
    let shapes = [
        static_shape([1, 1, 3, 3]),
        static_shape([1, 1, 2, 2]),
        static_shape([1]),
    ];
    let dtypes = [DataType::BFloat16; 3];

    let rejected = ep.supports_op(&node, 21, &shapes, &dtypes, &[]);
    let reason = rejected
        .reason()
        .expect("opset 21 BF16 Conv must be rejected");
    assert!(reason.contains("opset 22"), "{reason}");
    assert!(
        ep.supports_op(&node, 22, &shapes, &dtypes, &[])
            .is_supported()
    );

    let input = Tensor::floats(
        FloatDType::Bf16,
        &[1, 1, 3, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );
    let weights = Tensor::floats(FloatDType::Bf16, &[1, 1, 2, 2], &[1.0, 2.0, -1.0, 0.5]);
    let bias = Tensor::floats(FloatDType::Bf16, &[1], &[0.25]);
    let mut output = Tensor::zeros(FloatDType::Bf16, &[1, 1, 2, 2]);
    let kernel = ep
        .get_kernel(&node, &[vec![1, 1, 3, 3], vec![1, 1, 2, 2], vec![1]], 22)
        .unwrap();
    let inputs: [TensorView<'_>; 3] = [input.view(), weights.view(), bias.view()];
    let mut outputs: [TensorMut<'_>; 1] = [output.view_mut()];
    kernel.execute(&inputs, &mut outputs).unwrap();

    assert_eq!(output.to_f32(), vec![3.75, 6.25, 11.25, 13.75]);
}

#[test]
fn internal_kernel_creation_remains_available_before_opset_22() {
    let ep = CpuExecutionProvider::new();
    assert!(
        ep.get_kernel(
            &conv_node(),
            &[vec![1, 1, 3, 3], vec![1, 1, 2, 2], vec![1]],
            21,
        )
        .is_ok(),
        "claim-time schema admission must not disable optimizer-created Conv kernels"
    );
}
