use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, dequantize_packed_gather, static_shape,
};
use onnx_runtime_session::{InferenceSession, Tensor};

const PROFILES: &[(&str, usize, usize)] = &[
    ("q4_0", 32, 18),
    ("q4_1", 32, 20),
    ("q5_0", 32, 22),
    ("q5_1", 32, 24),
    ("q8_0", 32, 34),
    ("q8_1", 32, 36),
    ("q2_k", 256, 84),
    ("q3_k", 256, 110),
    ("q4_k", 256, 144),
    ("q5_k", 256, 176),
    ("q6_k", 256, 210),
    ("q8_k", 256, 292),
    ("iq2_xxs", 256, 66),
    ("iq2_xs", 256, 74),
    ("iq3_xxs", 256, 98),
    ("iq1_s", 256, 50),
    ("iq4_nl", 32, 18),
    ("iq3_s", 256, 110),
    ("iq2_s", 256, 82),
    ("iq4_xs", 256, 136),
    ("iq1_m", 256, 56),
    ("tq1_0", 256, 54),
    ("tq2_0", 256, 66),
    ("mxfp4", 32, 17),
    ("nvfp4", 64, 36),
    ("q1_0", 128, 18),
    ("q2_0", 64, 18),
];

fn graph(profile: &str, width: usize, dtype: DataType, expose_decoded: bool) -> Graph {
    let rows = 4usize;
    let row_elements = width * 3;
    let bytes = PROFILES
        .iter()
        .find_map(|&(name, _, bytes)| (name == profile).then_some(bytes))
        .unwrap();
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 29);
    let packed =
        graph.create_named_value("packed", DataType::Uint8, static_shape([rows * 3 * bytes]));
    let indices = graph.create_named_value("indices", DataType::Int64, static_shape([3]));
    let decoded = graph.create_named_value("decoded", dtype, static_shape([rows, row_elements]));
    let output = graph.create_named_value("output", dtype, static_shape([3, row_elements]));
    graph.add_input(packed);
    graph.add_input(indices);
    let mut dq = Node::new(
        NodeId(0),
        "DequantizePacked",
        vec![Some(packed)],
        vec![decoded],
    );
    dq.attributes.insert(
        "format".into(),
        Attribute::String(format!("gguf:{profile}").into_bytes()),
    );
    dq.attributes.insert(
        "shape".into(),
        Attribute::Ints(vec![rows as i64, row_elements as i64]),
    );
    if dtype != DataType::Float32 {
        dq.attributes.insert(
            "output_dtype".into(),
            Attribute::Int(i64::from(dtype.to_onnx())),
        );
    }
    graph.insert_node(dq);
    let mut gather = Node::new(
        NodeId(0),
        "Gather",
        vec![Some(decoded), Some(indices)],
        vec![output],
    );
    gather.attributes.insert("axis".into(), Attribute::Int(0));
    graph.insert_node(gather);
    graph.add_output(output);
    if expose_decoded {
        graph.add_output(decoded);
    }
    graph
}

fn corpus(profile: &str, bytes: usize) -> Vec<u8> {
    let salt = profile
        .bytes()
        .fold(0u8, |state, byte| state.wrapping_mul(31).wrapping_add(byte));
    (0..4 * 3 * bytes)
        .map(|lane| {
            (lane as u8)
                .wrapping_mul(73)
                .wrapping_add(((lane / bytes) as u8).wrapping_mul(29))
                .wrapping_add(salt)
        })
        .collect()
}

fn ops(session: &InferenceSession, name: &str) -> usize {
    session
        .graph()
        .nodes
        .values()
        .filter(|node| node.op_type == name)
        .count()
}

fn verify(
    mut fused: InferenceSession,
    mut reference: InferenceSession,
    profile: &str,
    width: usize,
    bytes: usize,
) {
    assert_eq!(ops(&fused, "DequantizePackedGather"), 1, "{profile}");
    assert_eq!(ops(&fused, "DequantizePacked"), 0, "{profile}");
    assert_eq!(
        ops(&reference, dequantize_packed_gather::OP),
        0,
        "{profile}"
    );
    assert_eq!(ops(&reference, "DequantizePacked"), 1, "{profile}");
    let packed = corpus(profile, bytes);
    let weights = Tensor::from_raw(DataType::Uint8, vec![packed.len()], &packed).unwrap();
    let indices = Tensor::from_i64(&[3], &[2, 0, -1]).unwrap();
    let input = [("packed", &weights), ("indices", &indices)];
    let got = fused.run(&input).unwrap();
    let expected = reference.run(&input).unwrap();
    assert_eq!(
        got[0].as_bytes(),
        expected[0].as_bytes(),
        "{profile} must decode all three blocks in each selected row"
    );
    assert_eq!(got[0].shape, [3, width * 3]);

    // Row 1 is never selected. Changing its complete packed payload must not
    // affect the fused result, which proves the kernel addresses selected rows.
    let mut poisoned = packed;
    poisoned[3 * bytes..6 * bytes].fill(0xff);
    let poisoned = Tensor::from_raw(DataType::Uint8, vec![poisoned.len()], &poisoned).unwrap();
    let input = [("packed", &poisoned), ("indices", &indices)];
    assert_eq!(fused.run(&input).unwrap()[0].as_bytes(), got[0].as_bytes());
}

fn verify_bad_index(mut fused: InferenceSession, mut reference: InferenceSession) {
    let packed = corpus("q4_0", 18);
    let weights = Tensor::from_raw(DataType::Uint8, vec![packed.len()], &packed).unwrap();
    let indices = Tensor::from_i64(&[3], &[0, 4, -1]).unwrap();
    let input = [("packed", &weights), ("indices", &indices)];
    let fused_error = fused.run(&input).unwrap_err().to_string();
    let reference_error = reference.run(&input).unwrap_err().to_string();
    for error in [&fused_error, &reference_error] {
        assert!(
            error.contains("out of range") || error.contains("flags=0x40"),
            "{error}"
        );
    }
}

#[test]
fn packed_gather_cpu_all_profiles_fp32() {
    assert_eq!(PROFILES.len(), 27);
    for &(profile, width, bytes) in PROFILES {
        verify(
            InferenceSession::from_graph(graph(profile, width, DataType::Float32, false)).unwrap(),
            InferenceSession::from_graph(graph(profile, width, DataType::Float32, true)).unwrap(),
            profile,
            width,
            bytes,
        );
    }
}

#[test]
fn packed_gather_cpu_preserves_index_validation() {
    verify_bad_index(
        InferenceSession::from_graph(graph("q4_0", 32, DataType::Float32, false)).unwrap(),
        InferenceSession::from_graph(graph("q4_0", 32, DataType::Float32, true)).unwrap(),
    );
}

#[test]
fn packed_gather_preserves_visible_decoded_value() {
    let visible = InferenceSession::from_graph(graph("q4_0", 32, DataType::Float32, true)).unwrap();
    assert_eq!(ops(&visible, dequantize_packed_gather::OP), 0);
    assert_eq!(ops(&visible, "DequantizePacked"), 1);
}

#[test]
fn packed_gather_q4k_mxfp4_admitted_output_dtypes() {
    for &(profile, width, bytes) in &[("q4_k", 256, 144), ("mxfp4", 32, 17)] {
        for dtype in [DataType::Float16, DataType::BFloat16, DataType::Float64] {
            verify(
                InferenceSession::from_graph(graph(profile, width, dtype, false)).unwrap(),
                InferenceSession::from_graph(graph(profile, width, dtype, true)).unwrap(),
                profile,
                width,
                bytes,
            );
        }
    }
}

#[test]
fn packed_gather_rejects_integer_output_routes() {
    for dtype in [DataType::Int32, DataType::Int64] {
        let error = InferenceSession::from_graph(graph("q4_k", 256, dtype, false))
            .err()
            .expect("integer DequantizePacked output dtype must be rejected")
            .to_string();
        assert!(
            error.contains("output_dtype") && error.contains("FLOAT"),
            "{dtype:?}: {error}"
        );
    }
}

#[cfg(feature = "gpu-tests")]
#[test]
fn packed_gather_cuda_all_profiles_fp32() {
    use onnx_runtime_ep_cuda::CudaExecutionProvider;
    use std::{path::Path, sync::Arc};

    for &(profile, width, bytes) in PROFILES {
        let build = |expose_decoded| {
            let provider =
                Arc::new(CudaExecutionProvider::initialized(0).expect("initialize CUDA"));
            let session = InferenceSession::from_graph_with_provider(
                graph(profile, width, DataType::Float32, expose_decoded),
                Arc::new(onnx_runtime_loader::WeightStore::new()),
                Path::new("."),
                provider,
                None,
            )
            .unwrap();
            assert!(
                session.execution_provider_fallback_report().is_none(),
                "{profile}"
            );
            session
        };
        verify(build(false), build(true), profile, width, bytes);
    }
}

#[cfg(feature = "gpu-tests")]
#[test]
fn packed_gather_cuda_q4k_mxfp4_admitted_output_dtypes() {
    use onnx_runtime_ep_cuda::CudaExecutionProvider;
    use std::{path::Path, sync::Arc};

    for &(profile, width, bytes) in &[("q4_k", 256, 144), ("mxfp4", 32, 17)] {
        for dtype in [DataType::Float16, DataType::BFloat16, DataType::Float64] {
            let build = |expose_decoded| {
                let provider =
                    Arc::new(CudaExecutionProvider::initialized(0).expect("initialize CUDA"));
                let session = InferenceSession::from_graph_with_provider(
                    graph(profile, width, dtype, expose_decoded),
                    Arc::new(onnx_runtime_loader::WeightStore::new()),
                    Path::new("."),
                    provider,
                    None,
                )
                .unwrap();
                assert!(
                    session.execution_provider_fallback_report().is_none(),
                    "{profile} {dtype:?}"
                );
                session
            };
            verify(build(false), build(true), profile, width, bytes);
        }
    }
}

#[cfg(feature = "gpu-tests")]
#[test]
fn packed_gather_cuda_preserves_index_validation() {
    use onnx_runtime_ep_cuda::CudaExecutionProvider;
    use std::{path::Path, sync::Arc};

    let build = |expose_decoded| {
        InferenceSession::from_graph_with_provider(
            graph("q4_0", 32, DataType::Float32, expose_decoded),
            Arc::new(onnx_runtime_loader::WeightStore::new()),
            Path::new("."),
            Arc::new(CudaExecutionProvider::initialized(0).expect("initialize CUDA")),
            None,
        )
        .unwrap()
    };
    verify_bad_index(build(false), build(true));
}
