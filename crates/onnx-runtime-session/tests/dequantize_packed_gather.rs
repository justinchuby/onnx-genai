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
    gather_graph(profile, width, dtype, expose_decoded, width * 3, 3)
}

fn gather_graph(
    profile: &str,
    width: usize,
    dtype: DataType,
    expose_decoded: bool,
    row_elements: usize,
    index_count: usize,
) -> Graph {
    let rows = 4usize;
    let bytes = PROFILES
        .iter()
        .find_map(|&(name, _, bytes)| (name == profile).then_some(bytes))
        .unwrap();
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 29);
    let packed = graph.create_named_value(
        "packed",
        DataType::Uint8,
        static_shape([rows * (row_elements / width) * bytes]),
    );
    let indices = graph.create_named_value("indices", DataType::Int64, static_shape([index_count]));
    let decoded = graph.create_named_value("decoded", dtype, static_shape([rows, row_elements]));
    let output =
        graph.create_named_value("output", dtype, static_shape([index_count, row_elements]));
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

fn rank_one_graph(profile: &str, width: usize, expose_decoded: bool) -> Graph {
    let elements = width * 3;
    let bytes = PROFILES
        .iter()
        .find_map(|&(name, _, bytes)| (name == profile).then_some(bytes))
        .unwrap();
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 29);
    let packed = graph.create_named_value("packed", DataType::Uint8, static_shape([3 * bytes]));
    let indices = graph.create_named_value("indices", DataType::Int64, static_shape([4]));
    let decoded = graph.create_named_value("decoded", DataType::Float32, static_shape([elements]));
    let output = graph.create_named_value("output", DataType::Float32, static_shape([4]));
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
    dq.attributes
        .insert("shape".into(), Attribute::Ints(vec![elements as i64]));
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

#[test]
fn packed_gather_rank_one_declines_fusion_and_executes_cleanly() {
    for &(profile, width, bytes) in PROFILES {
        let mut private =
            InferenceSession::from_graph(rank_one_graph(profile, width, false)).unwrap();
        let mut reference =
            InferenceSession::from_graph(rank_one_graph(profile, width, true)).unwrap();
        assert_eq!(ops(&private, dequantize_packed_gather::OP), 0, "{profile}");
        assert_eq!(ops(&private, "DequantizePacked"), 1, "{profile}");
        let raw = &corpus(profile, bytes)[..3 * bytes];
        let weights = Tensor::from_raw(DataType::Uint8, vec![raw.len()], raw).unwrap();
        let indices =
            Tensor::from_i64(&[4], &[0, width as i64, (width * 2 + 1) as i64, -1]).unwrap();
        let input = [("packed", &weights), ("indices", &indices)];
        let got = private.run(&input).unwrap();
        let expected = reference.run(&input).unwrap();
        assert_eq!(got[0].shape, [4], "{profile}");
        assert_eq!(got[0].as_bytes(), expected[0].as_bytes(), "{profile}");
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

#[cfg(feature = "gpu-tests")]
#[test]
fn packed_gather_cuda_empty_output_preserves_index_validation() {
    use onnx_runtime_ep_cuda::CudaExecutionProvider;
    use onnx_runtime_session::DeviceGraphCaptureResult;
    use std::{path::Path, sync::Arc};

    let provider = Arc::new(CudaExecutionProvider::initialized(0).expect("initialize CUDA"));
    let runtime = provider.runtime();
    let build = |row_elements, count| {
        let session = InferenceSession::from_graph_with_provider(
            gather_graph("q4_0", 32, DataType::Float32, false, row_elements, count),
            Arc::new(onnx_runtime_loader::WeightStore::new()),
            Path::new("."),
            provider.clone(),
            None,
        )
        .unwrap();
        assert!(session.execution_provider_fallback_report().is_none());
        assert_eq!(ops(&session, dequantize_packed_gather::OP), 1);
        session
    };

    let mut session = build(0, 3);
    let packed = Tensor::from_raw(DataType::Uint8, vec![0], &[]).unwrap();
    let valid = Tensor::from_i64(&[3], &[0, -4, 3]).unwrap();
    let invalid = Tensor::from_i64(&[3], &[0, 1, 4]).unwrap();
    runtime.set_defer_eager_sync(false);
    let got = session
        .run(&[("packed", &packed), ("indices", &valid)])
        .unwrap();
    assert_eq!(got[0].shape, [3, 0]);
    assert!(got[0].as_bytes().is_empty());
    let error = session
        .run(&[("packed", &packed), ("indices", &invalid)])
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("index 4 is out of range for decoded axis 0 dimension 4"),
        "{error}"
    );

    runtime.set_defer_eager_sync(true);
    let got = session
        .run(&[("packed", &packed), ("indices", &valid)])
        .unwrap();
    assert_eq!(got[0].shape, [3, 0]);
    assert!(got[0].as_bytes().is_empty());
    let error = session
        .run(&[("packed", &packed), ("indices", &invalid)])
        .unwrap_err()
        .to_string();
    assert!(error.contains("flags=0x40"), "{error}");

    let mut bindings = [
        session
            .allocate_device_binding("packed", None::<String>, DataType::Uint8, vec![0], vec![0])
            .unwrap(),
        session
            .allocate_device_binding("indices", None::<String>, DataType::Int64, vec![3], vec![3])
            .unwrap(),
        session
            .allocate_device_output_binding("output", DataType::Float32, vec![3, 0], vec![3, 0])
            .unwrap(),
    ];
    bindings[1].write_bytes(0, valid.as_bytes()).unwrap();
    assert!(matches!(
        session
            .try_capture_with_device_bindings(&[], &mut bindings)
            .unwrap(),
        DeviceGraphCaptureResult::Captured(_)
    ));
    assert_eq!(session.captured_graph_segment_count(), 1);
    bindings[2].read_bytes_into(&mut []).unwrap();
    let below_range = Tensor::from_i64(&[3], &[0, 1, -5]).unwrap();
    bindings[1].write_bytes(0, below_range.as_bytes()).unwrap();
    let error = session
        .replay_device_graph(&mut bindings)
        .unwrap_err()
        .to_string();
    assert!(error.contains("flags=0x40"), "{error}");

    let mut empty = build(32, 0);
    let packed = Tensor::from_raw(DataType::Uint8, vec![4 * 18], &[0; 4 * 18]).unwrap();
    let indices = Tensor::from_i64(&[0], &[]).unwrap();
    let got = empty
        .run(&[("packed", &packed), ("indices", &indices)])
        .unwrap();
    assert_eq!(got[0].shape, [0, 32]);
    assert!(got[0].as_bytes().is_empty());
}
