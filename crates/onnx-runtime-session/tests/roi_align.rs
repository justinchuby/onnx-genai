use std::{path::Path, sync::Arc};

use half::{bf16, f16};
use onnx_runtime_ep_api::ExecutionProvider;
use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, static_shape};
use onnx_runtime_session::{InferenceSession, Tensor};

#[path = "fixtures/roi_align_vectors.rs"]
mod roi_align_vectors;

use roi_align_vectors::{CASES, ROIS, X};

fn graph(opset: u64, dtype: DataType, count: usize, case: &roi_align_vectors::Case) -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), opset);
    let x = graph.create_named_value("X", dtype, static_shape([2, 2, 3, 3]));
    let rois = graph.create_named_value("rois", dtype, static_shape([count, 4]));
    let b = graph.create_named_value("batch_indices", DataType::Int64, static_shape([count]));
    let y = graph.create_named_value("Y", dtype, static_shape([count, 2, 1, 2]));
    for input in [x, rois, b] {
        graph.add_input(input);
    }
    let mut node = Node::new(
        NodeId(0),
        "RoiAlign",
        vec![Some(x), Some(rois), Some(b)],
        vec![y],
    );
    node.attributes
        .insert("output_width".into(), Attribute::Int(2));
    node.attributes
        .insert("spatial_scale".into(), Attribute::Float(0.5));
    node.attributes
        .insert("sampling_ratio".into(), Attribute::Int(case.ratio));
    node.attributes.insert(
        "mode".into(),
        Attribute::String(case.mode.as_bytes().to_vec()),
    );
    if let Some(coord) = case.coordinate {
        node.attributes.insert(
            "coordinate_transformation_mode".into(),
            Attribute::String(coord.as_bytes().to_vec()),
        );
    }
    graph.insert_node(node);
    graph.add_output(y);
    graph
}

fn floats(dtype: DataType, values: &[f64]) -> Vec<u8> {
    match dtype {
        DataType::Float64 => values.iter().flat_map(|x| x.to_le_bytes()).collect(),
        DataType::Float32 => values
            .iter()
            .flat_map(|&x| (x as f32).to_le_bytes())
            .collect(),
        DataType::Float16 => values
            .iter()
            .flat_map(|&x| f16::from_f64(x).to_bits().to_le_bytes())
            .collect(),
        DataType::BFloat16 => values
            .iter()
            .flat_map(|&x| bf16::from_f64(x).to_bits().to_le_bytes())
            .collect(),
        _ => unreachable!("fixture dtype"),
    }
}

fn make_session(
    provider: Arc<dyn ExecutionProvider>,
    opset: u64,
    dtype: DataType,
    count: usize,
    case: &roi_align_vectors::Case,
) -> InferenceSession {
    let session = InferenceSession::from_graph_with_provider(
        graph(opset, dtype, count, case),
        Arc::new(onnx_runtime_loader::WeightStore::new()),
        Path::new("."),
        provider,
        None, // no CPU fallback, including for the CUDA test
    )
    .unwrap();
    assert!(session.execution_provider_fallback_report().is_none());
    session
}

fn run(
    session: &mut InferenceSession,
    dtype: DataType,
    x: &[f64],
    b: &[i64],
) -> Result<Vec<Tensor>, onnx_runtime_session::SessionError> {
    let input = Tensor::from_raw(dtype, vec![2, 2, 3, 3], &floats(dtype, x)).unwrap();
    let rois = Tensor::from_raw(
        dtype,
        vec![b.len(), 4],
        &floats(dtype, &ROIS[..4 * b.len()]),
    )
    .unwrap();
    let indices = Tensor::from_raw(
        DataType::Int64,
        vec![b.len()],
        &b.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
    .unwrap();
    session.run(&[("X", &input), ("rois", &rois), ("batch_indices", &indices)])
}

fn check_values(y: &Tensor, dtype: DataType, case: &roi_align_vectors::Case) {
    if dtype == DataType::Float64 {
        for (i, (value, expected)) in y.as_bytes().chunks_exact(8).zip(case.expected).enumerate() {
            let value = f64::from_le_bytes(value.try_into().unwrap());
            assert!(
                (value - expected).abs() <= 1e-12,
                "lane {i}: {value} != {expected}"
            );
        }
    } else {
        assert_eq!(
            y.as_bytes(),
            floats(dtype, case.expected),
            "{dtype:?} / {:?} / {}",
            case.coordinate,
            case.mode
        );
    }
}

fn pinned(provider: Arc<dyn ExecutionProvider>) {
    for dtype in [
        DataType::Float16,
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float64,
    ] {
        for case in CASES {
            let mut session = make_session(Arc::clone(&provider), 22, dtype, 3, case);
            let outputs = run(&mut session, dtype, X, &[1, 0, 1]).unwrap();
            let [y] = outputs.as_slice() else {
                panic!("expected exactly one output")
            };
            assert_eq!(y.dtype, dtype);
            assert_eq!(y.shape, [3, 2, 1, 2]);
            check_values(y, dtype, case);
        }
    }
    // Opset 10 uses the unshifted coordinates without an attribute.
    let legacy = roi_align_vectors::Case {
        coordinate: None,
        ..CASES[1]
    };
    let mut session = make_session(Arc::clone(&provider), 10, DataType::Float64, 3, &legacy);
    let outputs = run(&mut session, DataType::Float64, X, &[1, 0, 1]).unwrap();
    check_values(&outputs[0], DataType::Float64, &legacy);

    // A constant field detects accidental f64 -> f32 narrowing.
    let precise = 1.0 + 2.0_f64.powi(-30);
    let mut session = make_session(provider, 22, DataType::Float64, 3, &CASES[0]);
    let outputs = run(&mut session, DataType::Float64, &[precise; 36], &[1, 0, 1]).unwrap();
    assert!(outputs[0].as_bytes().chunks_exact(8).any(|bytes| {
        let value = f64::from_le_bytes(bytes.try_into().unwrap());
        (value - precise).abs() <= 1e-14
    }));
}

fn indices(provider: Arc<dyn ExecutionProvider>) {
    let dtype = DataType::Float32;
    let mut session = make_session(Arc::clone(&provider), 22, dtype, 3, &CASES[0]);
    for bad in [-1, 2, i64::MAX] {
        let err = run(&mut session, dtype, X, &[1, bad, 0]).unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains("batch_indices") && text.contains("[0, 2)"),
            "{text}"
        );
    }
    let outputs = run(&mut session, dtype, X, &[1, 0, 1]).unwrap();
    assert_eq!(outputs[0].as_bytes(), floats(dtype, CASES[0].expected));
    let mut empty = make_session(provider, 22, dtype, 0, &CASES[0]);
    let outputs = run(&mut empty, dtype, X, &[]).unwrap();
    assert_eq!(outputs[0].shape, [0, 2, 1, 2]);
    assert!(outputs[0].as_bytes().is_empty());
}

#[test]
fn roi_align_cpu_pinned() {
    pinned(Arc::new(onnx_runtime_ep_cpu::CpuExecutionProvider::new()));
}

#[test]
fn roi_align_cpu_indices() {
    indices(Arc::new(onnx_runtime_ep_cpu::CpuExecutionProvider::new()));
}

#[cfg(feature = "gpu-tests")]
#[test]
fn roi_align_cuda_pinned() {
    pinned(Arc::new(
        onnx_runtime_ep_cuda::CudaExecutionProvider::initialized(0)
            .expect("physical CUDA device required"),
    ));
}

#[cfg(feature = "gpu-tests")]
#[test]
fn roi_align_cuda_indices() {
    indices(Arc::new(
        onnx_runtime_ep_cuda::CudaExecutionProvider::initialized(0)
            .expect("physical CUDA device required"),
    ));
}
