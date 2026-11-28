use half::{bf16, f16};
use onnx_runtime_ep_api::negative_log_likelihood_loss::OP;
use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, ExecutionProvider, KernelMatch, TensorMut, TensorView,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{
    Attribute, DataType, DeviceId, Node, NodeId, compute_contiguous_strides, static_shape,
};

fn node(reduction: Option<&str>, ignore: Option<i64>) -> Node {
    let mut node = Node::new(NodeId(0), OP, vec![], vec![]);
    if let Some(reduction) = reduction {
        node.attributes.insert(
            "reduction".into(),
            Attribute::String(reduction.as_bytes().to_vec()),
        );
    }
    if let Some(ignore) = ignore {
        node.attributes
            .insert("ignore_index".into(), Attribute::Int(ignore));
    }
    node
}

fn view<'a, T>(
    values: &'a [T],
    dtype: DataType,
    shape: &'a [usize],
    strides: &'a [i64],
) -> TensorView<'a> {
    TensorView::new(
        DevicePtr(values.as_ptr().cast()),
        dtype,
        shape,
        strides,
        DeviceId::cpu(),
    )
}

fn run<T>(
    node: &Node,
    opset: u64,
    inputs: &[TensorView],
    dtype: DataType,
    shape: &[usize],
    output: &mut [T],
) -> onnx_runtime_ep_api::Result<()> {
    let ep = CpuExecutionProvider::new();
    let kernel = ep.get_kernel(node, &[], opset)?;
    let strides = compute_contiguous_strides(shape)
        .expect("NegativeLogLikelihoodLoss fixture output shape must have contiguous strides");
    kernel.execute(
        inputs,
        &mut [TensorMut::new(
            DevicePtrMut(output.as_mut_ptr().cast()),
            dtype,
            shape,
            &strides,
            DeviceId::cpu(),
        )],
    )
}

#[test]
fn weighted_spatial_reductions_ignore_outside_class_range() {
    let x = [-1_f32, -2., -3., -4., -5., -6., -7., -8.];
    let inputs = [
        view(&x, DataType::Float32, &[2, 2, 1, 2], &[4, 2, 2, 1]),
        view(&[1_i32, -99, 0, 1], DataType::Int32, &[2, 1, 2], &[2, 2, 1]),
        view(&[1_f32, 2.], DataType::Float32, &[2], &[1]),
    ];
    let mut losses = [f32::NAN; 4];
    run(
        &node(Some("none"), Some(-99)),
        19,
        &inputs,
        DataType::Float32,
        &[2, 1, 2],
        &mut losses,
    )
    .unwrap();
    assert_eq!(losses, [6., 0., 5., 16.]);
    for (reduction, expected) in [(Some("sum"), 27.), (None, 5.4)] {
        let mut output = [0_f32];
        run(
            &node(reduction, Some(-99)),
            21,
            &inputs,
            DataType::Float32,
            &[],
            &mut output,
        )
        .unwrap();
        assert_eq!(output, [expected]);
    }
}

#[test]
fn rank_two_i64_with_optional_weight_omitted_and_wide_ignore() {
    let ignore = i64::MAX - 1;
    let targets = [1_i64, ignore];
    let inputs = [
        view(
            &[0., -3.25_f64, -7.5, 0.],
            DataType::Float64,
            &[2, 2],
            &[2, 1],
        ),
        view(&targets, DataType::Int64, &[2], &[1]),
        TensorView::absent(DataType::Float64),
    ];
    let mut output = [0_f64];
    run(
        &node(None, Some(ignore)),
        12,
        &inputs,
        DataType::Float64,
        &[],
        &mut output,
    )
    .unwrap();
    assert_eq!(output, [3.25]);
    let error = run(
        &node(None, None),
        29,
        &inputs,
        DataType::Float64,
        &[],
        &mut output,
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains(&format!("target[1] is {ignore}")),
        "{error}"
    );
    let error = run(
        &node(None, Some(ignore)),
        29,
        &inputs,
        DataType::Float32,
        &[],
        &mut output,
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("loss must have dtype Float64"),
        "{error}"
    );
}

#[test]
fn half_accumulation_and_bf16_admission() {
    let ep = CpuExecutionProvider::new();
    let node = node(None, None);
    let shapes = [static_shape([1, 1]), static_shape([1])];
    let dtypes = [DataType::BFloat16, DataType::Int64];
    assert!(matches!(
        ep.supports_op(&node, 21, &shapes, &dtypes, &[]),
        KernelMatch::Unsupported { .. }
    ));
    assert!(matches!(
        ep.supports_op(&node, 22, &shapes, &dtypes, &[]),
        KernelMatch::Supported { .. }
    ));
    assert!(ep.get_kernel(&node, &[], 11).is_err());
    let scores = [bf16::from_f32(-2.)];
    let targets = [0_i64];
    let inputs = [
        view(&scores, DataType::BFloat16, &[1, 1], &[1, 1]),
        view(&targets, DataType::Int64, &[1], &[1]),
    ];
    let mut output = [bf16::ZERO];
    assert!(run(&node, 21, &inputs, DataType::BFloat16, &[], &mut output).is_err());
    run(&node, 22, &inputs, DataType::BFloat16, &[], &mut output).unwrap();
    assert_eq!(output, [bf16::from_f32(2.)]);
    run(&node, 29, &inputs, DataType::BFloat16, &[], &mut output).unwrap();
    assert_eq!(output, [bf16::from_f32(2.)]);

    let count = 70_000;
    let x = vec![f16::from_f32(-2.); count];
    let targets = vec![0_i32; count];
    let shape = [count, 1];
    let target_shape = [count];
    let inputs = [
        view(&x, DataType::Float16, &shape, &[1, 1]),
        view(&targets, DataType::Int32, &target_shape, &[1]),
    ];
    let mut output = [f16::ZERO];
    run(&node, 20, &inputs, DataType::Float16, &[], &mut output).unwrap();
    assert_eq!(output, [f16::from_f32(2.)]);
}

#[test]
fn all_ignored_mean_is_nan_without_reading_scores_or_weights() {
    let inputs = [
        view(&[f32::NAN], DataType::Float32, &[1, 1], &[1, 1]),
        view(&[0_i64], DataType::Int64, &[1], &[1]),
        view(&[f32::NAN], DataType::Float32, &[1], &[1]),
    ];
    let mut output = [0_f32];
    run(
        &node(None, Some(0)),
        29,
        &inputs,
        DataType::Float32,
        &[],
        &mut output,
    )
    .unwrap();
    assert!(output[0].is_nan());
    run(
        &node(Some("sum"), Some(0)),
        29,
        &inputs,
        DataType::Float32,
        &[],
        &mut output,
    )
    .unwrap();
    assert_eq!(output, [0.]);
}
