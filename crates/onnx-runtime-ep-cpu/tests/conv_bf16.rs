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
fn bf16_conv_runs_at_opset_11_when_semantics_match() {
    let ep = CpuExecutionProvider::new();
    let node = conv_node();
    let shapes = [
        static_shape([1, 1, 3, 3]),
        static_shape([1, 1, 2, 2]),
        static_shape([1]),
    ];
    let dtypes = [DataType::BFloat16; 3];

    let support = ep.supports_op(&node, 11, &shapes, &dtypes, &[]);
    assert!(
        support.is_supported(),
        "standalone CPU should accept kernel-supported BF16 for semantically compatible Conv-11: {:?}",
        support.reason()
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
        .get_kernel(&node, &[vec![1, 1, 3, 3], vec![1, 1, 2, 2], vec![1]], 11)
        .unwrap();
    let inputs: [TensorView<'_>; 3] = [input.view(), weights.view(), bias.view()];
    let mut outputs: [TensorMut<'_>; 1] = [output.view_mut()];
    kernel.execute(&inputs, &mut outputs).unwrap();

    assert_eq!(output.to_f32(), vec![3.75, 6.25, 11.25, 13.75]);
}
