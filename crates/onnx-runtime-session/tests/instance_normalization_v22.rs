//! Standalone CPU and strict CUDA coverage for InstanceNormalization-22 BF16.

use half::{bf16, f16};
use onnx_runtime_ir::{Attribute, DataType, DeviceId, Graph, Node, NodeId, static_shape};
use onnx_runtime_loader::{Model, encode_model};
use onnx_runtime_session::{DevicePreference, InferenceSession, Tensor};

const SHAPE: [usize; 4] = [1, 2, 2, 2];
const X: [f32; 8] = [1.0, 1.0, 3.0, 3.0, -2.0, -2.0, 2.0, 2.0];
const SCALE: [f32; 2] = [2.0, 0.5];
const BIAS: [f32; 2] = [0.25, -1.0];
const EXPECTED: [f32; 8] = [-1.75, -1.75, 2.25, 2.25, -1.5, -1.5, -0.5, -0.5];

fn float_bytes(dtype: DataType, values: &[f32]) -> Vec<u8> {
    match dtype {
        DataType::BFloat16 => values
            .iter()
            .flat_map(|&value| bf16::from_f32(value).to_bits().to_le_bytes())
            .collect(),
        DataType::Float16 => values
            .iter()
            .flat_map(|&value| f16::from_f32(value).to_bits().to_le_bytes())
            .collect(),
        DataType::Float32 => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        DataType::Float64 => values
            .iter()
            .flat_map(|&value| f64::from(value).to_le_bytes())
            .collect(),
        other => panic!("unsupported test dtype {other:?}"),
    }
}

fn model(opset: u64, dtype: DataType) -> Vec<u8> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), opset);
    let x = graph.create_named_value("X", dtype, static_shape(SHAPE));
    let scale = graph.create_named_value("scale", dtype, static_shape([2]));
    let bias = graph.create_named_value("B", dtype, static_shape([2]));
    let y = graph.create_named_value("Y", dtype, static_shape(SHAPE));
    let mut node = Node::new(
        NodeId(0),
        "InstanceNormalization",
        vec![Some(x), Some(scale), Some(bias)],
        vec![y],
    );
    node.attributes
        .insert("epsilon".into(), Attribute::Float(0.0));
    graph.add_input(x);
    graph.add_input(scale);
    graph.add_input(bias);
    graph.insert_node(node);
    graph.add_output(y);
    encode_model(&Model::new(&graph)).expect("encode InstanceNormalization model")
}

fn inputs(dtype: DataType) -> [(&'static str, Tensor); 3] {
    [
        (
            "X",
            Tensor::from_raw(dtype, SHAPE.to_vec(), &float_bytes(dtype, &X)).unwrap(),
        ),
        (
            "scale",
            Tensor::from_raw(dtype, vec![2], &float_bytes(dtype, &SCALE)).unwrap(),
        ),
        (
            "B",
            Tensor::from_raw(dtype, vec![2], &float_bytes(dtype, &BIAS)).unwrap(),
        ),
    ]
}

fn run_golden(opset: u64, dtype: DataType, device: DevicePreference, expected_device: DeviceId) {
    let label = format!("InstanceNormalization-{opset} {dtype:?} on {expected_device:?}");
    let mut session = InferenceSession::builder()
        .model_bytes(&model(opset, dtype))
        .device(device)
        .option("optimization", "none")
        .build()
        .unwrap_or_else(|error| panic!("{label}: could not build session: {error}"));
    assert!(
        session.execution_provider_fallback_report().is_none(),
        "{label}: provider fallback: {:?}",
        session.execution_provider_fallback_report()
    );
    assert!(
        session.heterogeneous_placement_report().is_none(),
        "{label}: unexpected mixed placement: {:?}",
        session.heterogeneous_placement_report()
    );
    assert_eq!(session.device_id(), expected_device, "{label}");

    let inputs = inputs(dtype);
    let input_refs = inputs
        .iter()
        .map(|(name, tensor)| (*name, tensor))
        .collect::<Vec<_>>();
    let outputs = session
        .run(&input_refs)
        .unwrap_or_else(|error| panic!("{label}: execution failed: {error}"));
    assert_eq!(outputs.len(), 1, "{label}");
    assert_eq!(outputs[0].dtype, dtype, "{label}");
    assert_eq!(outputs[0].shape, SHAPE, "{label}");
    assert_eq!(
        outputs[0].as_bytes(),
        float_bytes(dtype, &EXPECTED),
        "{label}"
    );
}

#[test]
fn instance_normalization_v21_rejects_bf16() {
    let result = InferenceSession::builder()
        .model_bytes(&model(21, DataType::BFloat16))
        .device(DevicePreference::Cpu)
        .option("optimization", "none")
        .build();
    let error = match result {
        Ok(_) => panic!("InstanceNormalization-21 must reject BF16"),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(message.contains("InstanceNormalization"), "{message}");
    assert!(message.contains("BFloat16"), "{message}");
}

#[test]
fn instance_normalization_v6_legacy_float_dtypes_regression() {
    for dtype in [DataType::Float16, DataType::Float32, DataType::Float64] {
        run_golden(6, dtype, DevicePreference::Cpu, DeviceId::cpu());
    }
}

#[test]
fn instance_normalization_v22_bf16_dynamic_cpu_golden() {
    run_golden(
        22,
        DataType::BFloat16,
        DevicePreference::Cpu,
        DeviceId::cpu(),
    );
}

#[cfg(feature = "cuda")]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn instance_normalization_v22_bf16_strict_cuda_golden() {
    run_golden(
        22,
        DataType::BFloat16,
        DevicePreference::Gpu { index: Some(0) },
        DeviceId::cuda(0),
    );
}
