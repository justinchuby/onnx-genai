//! Native LpPool conformance. Schema audited at ONNX commit
//! `622adf94d4e720142950aa8178f7357d52e63db0`; fixture values use ONNX 1.24.0
//! `ReferenceEvaluator`. To reproduce, feed each input and attribute list to a
//! single-node LpPool-22 model, applying the per-case adjustments below.

use half::bf16;
use onnx_runtime_ir::{Attribute, DataType, DeviceId, Graph, Node, NodeId, static_shape};
use onnx_runtime_loader::{Model, encode_model};
use onnx_runtime_session::{DevicePreference, InferenceSession, Tensor};

struct Case {
    name: &'static str,
    dtype: DataType,
    shape: [usize; 4],
    input: &'static [f64],
    output_shape: [usize; 4],
    expected: &'static [f64],
    attrs: Vec<(&'static str, Attribute)>,
}

fn cases() -> [Case; 3] {
    [
        Case {
            name: "asymmetric BF16 p=3",
            dtype: DataType::BFloat16,
            shape: [1, 1, 3, 4],
            input: &[-1., 2., -3., 4., 5., -6., 7., -8., 9., -10., 11., 12.],
            output_shape: [1, 1, 2, 2],
            // ReferenceEvaluator on f32 with BF16-exact inputs, rounded once
            // at output (the Python evaluator otherwise rounds its intermediates).
            expected: &[5.0, 8.375, 9.0, 13.25],
            attrs: vec![
                ("kernel_shape", Attribute::Ints(vec![2, 2])),
                ("strides", Attribute::Ints(vec![2, 2])),
                ("pads", Attribute::Ints(vec![0, 1, 1, 0])),
                ("p", Attribute::Int(3)),
            ],
        },
        Case {
            name: "double precision default p=2",
            dtype: DataType::Float64,
            shape: [1, 1, 2, 3],
            input: &[-1.000000000001, 2., -3., 4., 5., -6.],
            output_shape: [1, 1, 2, 2],
            expected: &[
                2.236_067_977_500_237,
                3.605_551_275_463_989,
                6.403_124_237_432_848_5,
                7.810_249_675_906_654,
            ],
            attrs: vec![("kernel_shape", Attribute::Ints(vec![1, 2]))],
        },
        Case {
            name: "VALID ceil-mode partial window",
            dtype: DataType::Float32,
            shape: [1, 1, 2, 5],
            input: &[-1., 2., -3., 4., 5., -6., 7., -8., 9., 10.],
            // ONNX shape inference on VALID + ceil_mode; values from
            // ReferenceEvaluator with equivalent explicit zero padding
            // (the evaluator rejects combining auto_pad with ceil_mode).
            output_shape: [1, 1, 2, 3],
            expected: &[2.236_068, 5., 5., 9.219_544, 12.041_594, 10.],
            attrs: vec![
                ("kernel_shape", Attribute::Ints(vec![1, 2])),
                ("strides", Attribute::Ints(vec![1, 2])),
                ("auto_pad", Attribute::String(b"VALID".to_vec())),
                ("ceil_mode", Attribute::Int(1)),
            ],
        },
    ]
}

fn model(case: &Case, opset: u64) -> Vec<u8> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), opset);
    let x = graph.create_named_value("X", case.dtype, static_shape(case.shape));
    let y = graph.create_named_value("Y", case.dtype, static_shape(case.output_shape));
    let mut node = Node::new(NodeId(0), "LpPool", vec![Some(x)], vec![y]);
    node.attributes.extend(
        case.attrs
            .iter()
            .map(|(name, attr)| ((*name).into(), attr.clone())),
    );
    graph.add_input(x);
    graph.insert_node(node);
    graph.add_output(y);
    encode_model(&Model::new(&graph)).expect("encode LpPool model")
}

fn tensor(case: &Case) -> Tensor {
    let bytes = match case.dtype {
        DataType::BFloat16 => case
            .input
            .iter()
            .flat_map(|&x| bf16::from_f64(x).to_bits().to_le_bytes())
            .collect::<Vec<_>>(),
        DataType::Float64 => case.input.iter().flat_map(|x| x.to_le_bytes()).collect(),
        DataType::Float32 => case
            .input
            .iter()
            .flat_map(|&x| (x as f32).to_le_bytes())
            .collect(),
        _ => unreachable!(),
    };
    Tensor::from_raw(case.dtype, case.shape.to_vec(), &bytes).unwrap()
}

fn check_cases(device: DevicePreference, expected_device: DeviceId) {
    for case in &cases() {
        let opsets: &[u64] = if case.dtype == DataType::BFloat16 {
            &[22]
        } else {
            &[18, 22]
        };
        for &opset in opsets {
            let label = format!("{} at opset {opset} on {expected_device:?}", case.name);
            let mut session = InferenceSession::builder()
                .model_bytes(&model(case, opset))
                .device(device.clone())
                .option("optimization", "none")
                .build()
                .unwrap_or_else(|err| panic!("{label}: {err}"));
            assert_eq!(session.device_id(), expected_device, "{label}");
            assert!(
                session.execution_provider_fallback_report().is_none(),
                "{label}"
            );
            assert!(
                session.heterogeneous_placement_report().is_none(),
                "{label}"
            );
            let outputs = session
                .run(&[("X", &tensor(case))])
                .unwrap_or_else(|err| panic!("{label}: {err}"));
            assert_eq!(outputs.len(), 1, "{label}");
            let y = &outputs[0];
            assert_eq!(y.dtype, case.dtype, "{label}");
            assert_eq!(y.shape, case.output_shape, "{label}");
            let (actual, tol): (Vec<f64>, f64) = match y.dtype {
                DataType::BFloat16 => (
                    y.as_bytes()
                        .chunks_exact(2)
                        .map(|b| {
                            bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f64()
                        })
                        .collect(),
                    0.0,
                ),
                DataType::Float64 => (
                    y.as_bytes()
                        .chunks_exact(8)
                        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                        .collect(),
                    1e-14,
                ),
                DataType::Float32 => (
                    y.as_bytes()
                        .chunks_exact(4)
                        .map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap())))
                        .collect(),
                    2e-6,
                ),
                _ => unreachable!(),
            };
            assert_eq!(actual.len(), case.expected.len(), "{label}");
            for (i, (&a, &b)) in actual.iter().zip(case.expected).enumerate() {
                assert!((a - b).abs() <= tol, "{label}: element {i}: {a} != {b}");
            }
        }
    }
}

#[test]
fn lp_pool_v22_cpu_matches_onnx_reference() {
    check_cases(DevicePreference::Cpu, DeviceId::cpu());
}

#[test]
fn lp_pool_v18_cpu_rejects_bf16() {
    let [case, ..] = cases();
    let err = InferenceSession::builder()
        .model_bytes(&model(&case, 18))
        .device(DevicePreference::Cpu)
        .option("optimization", "none")
        .build()
        .err()
        .expect("opset 18 must decline BF16");
    let detail = err.to_string();
    assert!(
        detail.contains("LpPool") && detail.contains("BF16"),
        "{detail}"
    );
}

#[cfg(feature = "cuda")]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires physical CUDA")]
#[test]
fn lp_pool_v22_cuda_matches_onnx_reference_without_fallback() {
    check_cases(DevicePreference::Gpu { index: Some(0) }, DeviceId::cuda(0));
}
