//! Standalone CPU and CUDA regression tests for the opset-22 BF16 addition.

use half::bf16;
use onnx_runtime_ir::{Attribute, DataType, DeviceId, Graph, Node, NodeId, static_shape};
use onnx_runtime_loader::{Model, encode_model};
use onnx_runtime_session::{DevicePreference, InferenceSession, Tensor};

const INPUT_SHAPE: [usize; 4] = [2, 2, 2, 2];
const OUTPUT_SHAPE: [usize; 4] = [2, 2, 1, 1];
const INPUT: [f32; 16] = [
    -3.0, 4.0, 0.0, 0.0, -8.0, -4.0, -2.0, -2.0, 1.0, 2.0, 2.0, 0.0, 0.0, -5.0, 12.0, 0.0,
];
const DTYPE: DataType = DataType::BFloat16;

struct Case {
    op: &'static str,
    p: Option<i64>,
    expected: [f32; 4],
}

const CASES: [Case; 4] = [
    Case {
        op: "GlobalAveragePool",
        p: None,
        expected: [0.25, -4.0, 1.25, 1.75],
    },
    Case {
        op: "GlobalMaxPool",
        p: None,
        expected: [4.0, -2.0, 2.0, 12.0],
    },
    Case {
        op: "GlobalLpPool",
        p: None,
        // sqrt(88) rounds to 9.375 in BF16.
        expected: [5.0, 9.375, 3.0, 13.0],
    },
    Case {
        op: "GlobalLpPool",
        p: Some(1),
        expected: [7.0, 16.0, 5.0, 17.0],
    },
];

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&value| bf16::from_f32(value).to_bits().to_le_bytes())
        .collect()
}

fn model(case: &Case) -> Vec<u8> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 22);
    let x = graph.create_named_value("X", DTYPE, static_shape(INPUT_SHAPE));
    let y = graph.create_named_value("Y", DTYPE, static_shape(OUTPUT_SHAPE));
    let mut node = Node::new(NodeId(0), case.op, vec![Some(x)], vec![y]);
    if let Some(p) = case.p {
        node.attributes.insert("p".into(), Attribute::Int(p));
    }
    graph.add_input(x);
    graph.insert_node(node);
    graph.add_output(y);
    encode_model(&Model::new(&graph)).expect("encode global pool v22 model")
}

fn check_cases(device: DevicePreference, expected_device: DeviceId) {
    let x = Tensor::from_raw(DTYPE, INPUT_SHAPE.to_vec(), &bf16_bytes(&INPUT)).unwrap();
    for case in &CASES {
        let label = format!("{} v22 BF16 p={:?} on {expected_device:?}", case.op, case.p);
        let mut session = InferenceSession::builder()
            .model_bytes(&model(case))
            .device(device.clone())
            .option("optimization", "none")
            .build()
            .unwrap_or_else(|err| panic!("{label}: could not build session: {err}"));
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
        let outputs = session
            .run(&[("X", &x)])
            .unwrap_or_else(|err| panic!("{label}: execution failed: {err}"));
        assert_eq!(outputs.len(), 1, "{label}");
        assert_eq!(outputs[0].dtype, DTYPE, "{label}");
        assert_eq!(outputs[0].shape, OUTPUT_SHAPE, "{label}");
        assert_eq!(outputs[0].as_bytes(), bf16_bytes(&case.expected), "{label}");
    }
}

#[test]
fn global_pool_v22_bf16_cpu() {
    check_cases(DevicePreference::Cpu, DeviceId::cpu());
}

#[cfg(feature = "cuda")]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn global_pool_v22_bf16_cuda_without_fallback() {
    check_cases(DevicePreference::Gpu { index: Some(0) }, DeviceId::cuda(0));
}
