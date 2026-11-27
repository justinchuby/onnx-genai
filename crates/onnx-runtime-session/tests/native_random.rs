use onnx_runtime_ir::{Attribute, DataType, Dim, Graph, Node, NodeId, static_shape};
use onnx_runtime_loader::{Model, encode_model};
use onnx_runtime_session::{DevicePreference, InferenceSession, Tensor};

fn random_model(op: &str, version: u64, seed: Option<f32>, dtype: Option<DataType>) -> Vec<u8> {
    random_model_with_input(op, version, seed, dtype, DataType::Int32)
}

fn random_model_with_input(
    op: &str,
    version: u64,
    seed: Option<f32>,
    dtype: Option<DataType>,
    input_dtype: DataType,
) -> Vec<u8> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), version);
    let like = op.ends_with("Like");
    let (inputs, shape) = if like {
        let shape = vec![Dim::Symbolic(graph.intern_symbol("count"))];
        let input = graph.create_named_value("input", input_dtype, shape.clone());
        graph.add_input(input);
        (vec![Some(input)], shape)
    } else {
        (vec![], static_shape([64]))
    };
    let out = graph.create_named_value(
        "output",
        dtype.unwrap_or(if like { input_dtype } else { DataType::Float32 }),
        shape,
    );
    graph.add_output(out);
    let mut node = Node::new(NodeId(0), op, inputs, vec![out]);
    if let Some(seed) = seed {
        node.attributes
            .insert("seed".into(), Attribute::Float(seed));
    }
    if let Some(dtype) = dtype {
        node.attributes
            .insert("dtype".into(), Attribute::Int(dtype as i64));
    }
    if !like {
        node.attributes
            .insert("shape".into(), Attribute::Ints(vec![64]));
    }
    graph.insert_node(node);
    encode_model(&Model::new(&graph)).expect("encode random-generator model")
}

fn cpu_session(bytes: &[u8]) -> Result<InferenceSession, onnx_runtime_session::SessionError> {
    InferenceSession::builder()
        .model_bytes(bytes)
        .device(DevicePreference::Cpu)
        .option("optimization", "all")
        .build()
}

#[test]
fn native_random_seeded_and_unseeded_runs_are_not_constant_folded() {
    let model = |seed| random_model("RandomUniform", 1, seed, None);
    let draw = |session: &mut InferenceSession| session.run(&[]).unwrap()[0].as_bytes().to_vec();
    let mut a = cpu_session(&model(Some(0.0))).unwrap();
    let mut b = cpu_session(&model(Some(-0.0))).unwrap();
    let first = draw(&mut a);
    assert_eq!(first, draw(&mut b));
    let second = draw(&mut a);
    assert_ne!(first, second);
    assert_eq!(second, draw(&mut b));
    let mut fraction = cpu_session(&model(Some(0.25))).unwrap();
    assert_ne!(first, draw(&mut fraction));
    let mut auto_a = cpu_session(&model(None)).unwrap();
    let mut auto_b = cpu_session(&model(None)).unwrap();
    let auto_first = draw(&mut auto_a);
    assert_ne!(auto_first, draw(&mut auto_a));
    assert_ne!(auto_first, draw(&mut auto_b));
}

#[test]
fn native_random_like_advances_one_stream_across_shapes() {
    assert_like_stream(cpu_session);
}

fn assert_like_stream(
    build: impl Fn(&[u8]) -> Result<InferenceSession, onnx_runtime_session::SessionError>,
) {
    let bytes = random_model("RandomNormalLike", 22, Some(-2.25), Some(DataType::Float64));
    let mut chunked = build(&bytes).unwrap();
    let mut whole = build(&bytes).unwrap();
    let draw = |session: &mut InferenceSession, count: usize, byte: u8| {
        let input = Tensor::from_raw(DataType::Int32, vec![count], &vec![byte; count * 4]).unwrap();
        let outputs = session.run(&[("input", &input)]).unwrap();
        assert_eq!(outputs[0].shape, [count]);
        assert_eq!(outputs[0].as_bytes().len(), count * 8);
        outputs[0].as_bytes().to_vec()
    };
    let chunks = [3, 0, 5, 6, 7, 8, 9, 3];
    let flat: Vec<_> = chunks
        .into_iter()
        .flat_map(|n| draw(&mut chunked, n, 0xff))
        .collect();
    assert_eq!(flat, draw(&mut whole, chunks.into_iter().sum(), 0));
}

#[cfg(feature = "cuda")]
#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires CUDA device")]
fn native_random_cuda_session_advances_one_stream_across_shapes() {
    use std::sync::Arc;

    use onnx_runtime_ep_cuda::CudaExecutionProvider;
    use onnx_runtime_ir::DeviceId;

    let ep =
        Arc::new(CudaExecutionProvider::initialized(0).expect("initialize physical CUDA device 0"));
    assert_like_stream(|bytes| {
        let session = InferenceSession::builder()
            .model_bytes(bytes)
            .execution_provider(ep.clone())
            .option("optimization", "all")
            .build()?;
        assert_eq!(session.device_id(), DeviceId::cuda(0));
        assert!(
            session.execution_provider_fallback_report().is_none(),
            "CUDA random graph must never fall back to CPU: {:?}",
            session.execution_provider_fallback_report()
        );
        Ok(session)
    });
}

#[test]
fn native_random_dtype_version_and_like_inheritance_are_validated() {
    let bf16 = Some(DataType::BFloat16);
    let model = |version| random_model("RandomNormal", version, Some(1.0), bf16);
    let error = cpu_session(&model(21)).err().expect("BF16 before opset 22");
    assert!(
        error.to_string().contains("BF16 starting at opset 22"),
        "{error}"
    );
    let outputs = cpu_session(&model(22)).unwrap().run(&[]).unwrap();
    assert_eq!(outputs[0].dtype, DataType::BFloat16);
    assert_eq!(outputs[0].as_bytes().len(), 128);

    let model = |dtype| random_model("RandomUniformLike", 22, Some(1.0), dtype);
    let input = Tensor::from_raw(DataType::Int32, vec![3], &[0; 12]).unwrap();
    // The symbolic input defers provider admission to the first run.
    let error = cpu_session(&model(None))
        .unwrap()
        .run(&[("input", &input)])
        .err()
        .expect("Int32 is only valid with a float override");
    assert!(
        error.to_string().contains("choose a float `dtype`"),
        "{error}"
    );
    let outputs = cpu_session(&model(Some(DataType::Float32)))
        .unwrap()
        .run(&[("input", &input)])
        .unwrap();
    assert_eq!(outputs[0].dtype, DataType::Float32);
    assert_eq!(outputs[0].shape, [3]);
}

#[test]
fn native_random_like_uses_only_managed_string_shape_on_cpu() {
    let bytes = |input_dtype| {
        random_model_with_input(
            "RandomUniformLike",
            1,
            Some(0.0),
            Some(DataType::Float32),
            input_dtype,
        )
    };
    let mut strings = cpu_session(&bytes(DataType::String)).unwrap();
    let mut ints = cpu_session(&bytes(DataType::Int32)).unwrap();
    let text = Tensor::from_strings(vec![3], vec!["x".into(), "".into(), "longer".into()]).unwrap();
    let numeric = Tensor::from_raw(DataType::Int32, vec![3], &[0; 12]).unwrap();
    for _ in 0..2 {
        let out = strings.run(&[("input", &text)]).unwrap();
        let reference = ints.run(&[("input", &numeric)]).unwrap();
        assert_eq!(out[0].shape, [3]);
        assert_eq!(out[0].dtype, DataType::Float32);
        assert_eq!(out[0].as_bytes(), reference[0].as_bytes());
    }
}
