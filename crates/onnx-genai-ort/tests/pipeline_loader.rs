use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use onnx_genai_ort::{
    DataType as OrtDataType, Environment, GraphIo, PipelineModelDirectory, PipelineModels, Session,
    SessionOptions, graph_io_from_model_path,
};
use onnx_std::Model;
use onnx_std::ir::{Attribute, DataType, Graph, Node, NodeId, TensorData, WeightRef};

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

struct FixtureDir(PathBuf);

impl FixtureDir {
    fn new() -> Self {
        let id = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join("multi-model-pipeline");
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/pipeline-loader-tests")
            .join(format!("{}-{id}", std::process::id()));
        if path.exists() {
            std::fs::remove_dir_all(&path).unwrap();
        }
        std::fs::create_dir_all(&path).unwrap();
        for filename in [
            "inference_metadata.yaml",
            "genai_config.json",
            "tokenizer.json",
            "decoder-tokenizer.json",
        ] {
            std::fs::copy(source.join(filename), path.join(filename)).unwrap();
        }
        write_models(&path);
        Self(path)
    }
}

impl Drop for FixtureDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn write_models(root: &Path) {
    let mut encoder = Graph::new();
    encoder.opset_imports.insert(String::new(), 13);
    let batch = encoder.intern_symbol("batch");
    let encoder_sequence = encoder.intern_symbol("encoder_sequence");
    let input = encoder.create_named_value(
        "input_features",
        DataType::Float32,
        vec![batch.into(), encoder_sequence.into(), 4.into()],
    );
    encoder.add_input(input);
    let hidden = encoder.create_named_value(
        "hidden_states",
        DataType::Float32,
        vec![batch.into(), encoder_sequence.into(), 4.into()],
    );
    encoder.insert_node(Node::new(
        NodeId(0),
        "Identity",
        vec![Some(input)],
        vec![hidden],
    ));
    encoder.add_output(hidden);
    write_model(&root.join("encoder.onnx.fixture"), encoder);

    let mut decoder = Graph::new();
    decoder.opset_imports.insert(String::new(), 13);
    let batch = decoder.intern_symbol("batch");
    let sequence = decoder.intern_symbol("sequence");
    let encoder_sequence = decoder.intern_symbol("encoder_sequence");
    let input_ids = decoder.create_named_value(
        "input_ids",
        DataType::Int64,
        vec![batch.into(), sequence.into()],
    );
    let attention_mask = decoder.create_named_value(
        "attention_mask",
        DataType::Int64,
        vec![batch.into(), sequence.into()],
    );
    let hidden = decoder.create_named_value(
        "encoder_hidden_states",
        DataType::Float32,
        vec![batch.into(), encoder_sequence.into(), 4.into()],
    );
    decoder.add_input(input_ids);
    decoder.add_input(attention_mask);
    decoder.add_input(hidden);
    let logits = decoder.create_named_value(
        "logits",
        DataType::Float32,
        vec![batch.into(), encoder_sequence.into(), 4.into()],
    );
    decoder.insert_node(Node::new(
        NodeId(0),
        "Identity",
        vec![Some(hidden)],
        vec![logits],
    ));
    decoder.add_output(logits);
    write_model(&root.join("decoder.onnx.fixture"), decoder);
}

fn write_model(path: &Path, graph: Graph) {
    let model = Model::new(graph);
    model.to_proto().unwrap();
    onnx_std::save_model(&model, path).unwrap();
}

/// Overwrite the fixture decoder with a graph authored for a native-only
/// operator: a node in a runtime-owned domain (`pkg.nxrt`) that ORT's session
/// builder has no kernel for and therefore rejects at load. The graph's declared
/// I/O is unchanged, so its contract is still resolvable from the ONNX graph.
fn write_native_only_decoder(root: &Path) {
    let mut decoder = Graph::new();
    decoder.opset_imports.insert(String::new(), 13);
    // Declaring the runtime's own operator domain is what makes ORT reject the
    // graph at session build: it has no kernel registered for it.
    decoder.opset_imports.insert("pkg.nxrt".to_string(), 1);
    let batch = decoder.intern_symbol("batch");
    let sequence = decoder.intern_symbol("sequence");
    let encoder_sequence = decoder.intern_symbol("encoder_sequence");
    let input_ids = decoder.create_named_value(
        "input_ids",
        DataType::Int64,
        vec![batch.into(), sequence.into()],
    );
    let attention_mask = decoder.create_named_value(
        "attention_mask",
        DataType::Int64,
        vec![batch.into(), sequence.into()],
    );
    let hidden = decoder.create_named_value(
        "encoder_hidden_states",
        DataType::Float32,
        vec![batch.into(), encoder_sequence.into(), 4.into()],
    );
    decoder.add_input(input_ids);
    decoder.add_input(attention_mask);
    decoder.add_input(hidden);
    let logits = decoder.create_named_value(
        "logits",
        DataType::Float32,
        vec![batch.into(), sequence.into(), 4.into()],
    );
    let mut node = Node::new(
        NodeId(0),
        "BlockQuantizedMatMul",
        vec![Some(hidden)],
        vec![logits],
    );
    node.domain = "pkg.nxrt".to_string();
    // The native op's `N` (output width) lets the graph shape-infer during
    // package admission, so the only thing that fails is ORT's session build —
    // which has no kernel for this runtime-owned operator. That is the exact
    // native-only shape this fix must let a pipeline load.
    node.attributes.insert("N".to_string(), Attribute::Int(4));
    decoder.insert_node(node);
    decoder.add_output(logits);
    write_model(&root.join("decoder.onnx.fixture"), decoder);
}

#[test]
fn resolves_multi_model_pipeline_directory() {
    let fixture = FixtureDir::new();
    let directory = PipelineModelDirectory::load(&fixture.0).expect("pipeline directory resolves");

    assert_eq!(directory.spec.models.len(), 2);
    assert!(directory.model_paths["encoder"].ends_with("encoder.onnx.fixture"));
    assert!(directory.model_paths["decoder"].ends_with("decoder.onnx.fixture"));
    assert!(directory.tokenizer_paths.shared.is_some());
    assert!(
        directory
            .tokenizer_paths
            .for_component("encoder")
            .expect("encoder uses shared tokenizer")
            .ends_with("tokenizer.json")
    );
    assert!(
        directory
            .tokenizer_paths
            .for_component("decoder")
            .expect("decoder uses component tokenizer")
            .ends_with("decoder-tokenizer.json")
    );
}

#[test]
fn native_metadata_precedes_invalid_genai_config_fallback() {
    let fixture = FixtureDir::new();
    let directory = PipelineModelDirectory::load(&fixture.0)
        .expect("native metadata must bypass the invalid compatibility file");

    assert!(
        directory
            .metadata_path
            .as_deref()
            .is_some_and(|path| path.ends_with("inference_metadata.yaml"))
    );
    assert_eq!(directory.spec.models.len(), 2);
}

// Resolving a pipeline already reads and validates the sidecar, so the settings
// it declares are carried on the directory. A consumer that re-opened the file
// could disagree with the spec built beside it, and one that swallowed the
// parse error would see a package that declares nothing.
#[test]
fn a_resolved_pipeline_carries_the_settings_its_metadata_declares() {
    let fixture = FixtureDir::new();
    let sidecar = fixture.0.join("inference_metadata.yaml");
    let mut source = std::fs::read_to_string(&sidecar).unwrap();
    source.push_str("model:\n  max_sequence_length: 4096\ntokens:\n  eos_token_id: [7, 11]\n");
    std::fs::write(&sidecar, source).unwrap();

    let directory =
        PipelineModelDirectory::load(&fixture.0).expect("the pipeline package must resolve");

    let metadata = directory
        .metadata
        .as_ref()
        .expect("a package with a native sidecar carries its parsed metadata");
    assert_eq!(
        metadata
            .model
            .as_ref()
            .and_then(|model| model.max_sequence_length),
        Some(4096)
    );
    assert_eq!(
        metadata
            .tokens
            .as_ref()
            .and_then(|tokens| tokens.eos_token_id.clone()),
        Some(vec![7, 11])
    );
}

/// A pipeline component authored for a native-only operator (one ORT's session
/// builder cannot load) must still load through the pipeline: its ORT `Session`
/// is skipped and its I/O contract is surfaced from the ONNX graph instead. This
/// locks the loader fix that lets native-backend components — for example a QMoE
/// decoder ORT rejects at load over its fp16/fp32 type contract — participate in
/// a pipeline without a redundant ORT session.
#[test]
fn native_only_component_loads_without_ort_session() {
    let fixture = FixtureDir::new();
    write_native_only_decoder(&fixture.0);

    // Building an ORT session for every component fails: ORT has no
    // `pkg.nxrt::BlockQuantizedMatMul` kernel, so the decoder is rejected at
    // load — exactly the failure the native-backend skip avoids.
    let all_ort = PipelineModels::load_with_options(&fixture.0, SessionOptions::default());
    assert!(
        all_ort.is_err(),
        "ORT must reject a native-only decoder when an ORT session is built for it"
    );

    // Skipping the decoder's ORT session loads the pipeline: the encoder keeps
    // its ORT session, the decoder is captured as session-free graph I/O, and
    // both expose their contract through the backend-neutral `graph_io` seam.
    let models = PipelineModels::load_with_ort_session_filter(
        &fixture.0,
        SessionOptions::default(),
        |component| component != "decoder",
    )
    .expect("pipeline loads when the native-only decoder's ORT session is skipped");

    assert!(
        models.session("decoder").is_none(),
        "the native-only decoder must not get an ORT session"
    );
    assert!(
        models.session("encoder").is_some(),
        "ORT-executed components keep their ORT session"
    );
    assert!(models.graph_io_metadata.contains_key("decoder"));
    assert!(!models.graph_io_metadata.contains_key("encoder"));

    let decoder = models
        .graph_io("decoder")
        .expect("the skipped decoder still exposes its graph I/O");
    let inputs: Vec<&str> = decoder.input_names().iter().map(String::as_str).collect();
    assert_eq!(
        inputs,
        ["input_ids", "attention_mask", "encoder_hidden_states"]
    );
    let outputs: Vec<&str> = decoder.output_names().iter().map(String::as_str).collect();
    assert_eq!(outputs, ["logits"]);

    // The ORT-backed encoder resolves through the same seam.
    assert!(
        models.graph_io("encoder").is_some(),
        "an ORT-backed component resolves graph I/O through the same seam"
    );
}

/// Author a small decoder ONNX with realistic paged-KV I/O AND a weight tensor
/// that is listed in `graph.input` while also being a `graph.initializer` — the
/// pre-IR-4 pattern ONNX still permits in IR>=4. `graph_io_from_model_path` must
/// exclude that initializer from its declared inputs (exactly as ORT's `Session`
/// and this repo's native graph loader do), so `GraphIoMetadata` never leaks a
/// weight as a graph input port.
///
/// The KV state is rank-4 fp16 `[batch, heads, seq, head_dim]` so the metadata
/// carries the geometry the engine's `infer_kv_model_info`/`resolve_kv_layers`
/// read structurally (num_kv_heads=2, head_dim=4), not just names. Every node
/// uses an ORT-loadable operator (`Identity`, `Cast`) so the SAME model can be
/// parsed both session-free (metadata) and through a real ORT `Session`, letting
/// the test prove metadata geometry == session geometry byte-for-byte.
fn write_kv_decoder_with_leaked_initializer(path: &Path) {
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 4;
    const VOCAB: usize = 8;

    let mut decoder = Graph::new();
    decoder.opset_imports.insert(String::new(), 13);
    let batch = decoder.intern_symbol("batch");
    let sequence = decoder.intern_symbol("sequence");

    let input_ids = decoder.create_named_value(
        "input_ids",
        DataType::Int64,
        vec![batch.into(), sequence.into()],
    );
    decoder.add_input(input_ids);

    // A weight that is BOTH a `graph.initializer` and listed in `graph.input`.
    // A correct loader must treat it as a constant and NOT surface it as a port.
    let embed = decoder.create_named_value(
        "model.embed_tokens.weight",
        DataType::Float16,
        vec![VOCAB.into(), HEAD_DIM.into()],
    );
    decoder.add_input(embed);
    decoder.set_initializer(
        embed,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float16,
            vec![VOCAB, HEAD_DIM],
            vec![0u8; VOCAB * HEAD_DIM * 2],
        )),
    );

    let kv_shape = || vec![batch.into(), HEADS.into(), sequence.into(), HEAD_DIM.into()];
    let past_key =
        decoder.create_named_value("past_key_values.0.key", DataType::Float16, kv_shape());
    let past_value =
        decoder.create_named_value("past_key_values.0.value", DataType::Float16, kv_shape());
    decoder.add_input(past_key);
    decoder.add_input(past_value);

    let present_key = decoder.create_named_value("present.0.key", DataType::Float16, kv_shape());
    let present_value =
        decoder.create_named_value("present.0.value", DataType::Float16, kv_shape());
    decoder.insert_node(Node::new(
        NodeId(0),
        "Identity",
        vec![Some(past_key)],
        vec![present_key],
    ));
    decoder.insert_node(Node::new(
        NodeId(1),
        "Identity",
        vec![Some(past_value)],
        vec![present_value],
    ));

    // logits = Cast(input_ids -> fp16); an ORT-loadable op that gives the graph a
    // real producer for its declared logits output.
    let logits = decoder.create_named_value(
        "logits",
        DataType::Float16,
        vec![batch.into(), sequence.into()],
    );
    let mut cast = Node::new(NodeId(2), "Cast", vec![Some(input_ids)], vec![logits]);
    cast.attributes.insert("to".to_string(), Attribute::Int(10)); // FLOAT16 elem_type
    decoder.insert_node(cast);

    decoder.add_output(logits);
    decoder.add_output(present_key);
    decoder.add_output(present_value);

    let model = Model::new(decoder);
    model.to_proto().unwrap();
    onnx_std::save_model(&model, path).unwrap();
}

/// The session-free `GraphIoMetadata` a native-backend component resolves must
/// recover the SAME KV/IO geometry a real ORT `Session` would — including
/// excluding weights that ONNX lets a graph list in both `graph.input` and
/// `graph.initializer`. A leaked weight would otherwise be routed as a spurious
/// port and would trip the decode float-rank>=3 native-load guard, falsely
/// rejecting a valid decoder.
#[test]
fn graph_io_metadata_excludes_initializers_and_matches_session_geometry() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/pipeline-loader-tests")
        .join(format!("kv-decoder-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("decoder_kv.onnx");
    write_kv_decoder_with_leaked_initializer(&path);

    // Session-free parse: the fixed loader excludes the initializer-backed input.
    let metadata = graph_io_from_model_path(&path).expect("graph I/O parses without a session");

    let input_names: Vec<&str> = metadata.input_names().iter().map(String::as_str).collect();
    assert!(
        !input_names.contains(&"model.embed_tokens.weight"),
        "a weight listed in graph.input but also a graph.initializer must NOT leak as an input port"
    );
    assert_eq!(
        input_names,
        [
            "input_ids",
            "past_key_values.0.key",
            "past_key_values.0.value"
        ],
        "declared inputs must be the real graph inputs, in graph order, with initializers excluded"
    );
    let output_names: Vec<&str> = metadata.output_names().iter().map(String::as_str).collect();
    assert_eq!(output_names, ["logits", "present.0.key", "present.0.value"]);

    // Dtypes must be recovered structurally: fp16 (elem_type 10), not fp32.
    let past_key = metadata
        .inputs()
        .iter()
        .find(|info| info.name == "past_key_values.0.key")
        .expect("past key input is present");
    assert_eq!(past_key.dtype, OrtDataType::Float16);
    assert_ne!(past_key.dtype, OrtDataType::Float32);

    // KV geometry the engine reads structurally: `[batch, heads, seq, head_dim]`
    // with symbolic batch/seq as -1 and static heads=2, head_dim=4.
    let present_key = metadata
        .outputs()
        .iter()
        .find(|info| info.name == "present.0.key")
        .expect("present key output is present");
    assert_eq!(present_key.dtype, OrtDataType::Float16);
    assert_eq!(
        present_key.shape,
        [-1, 2, -1, 4],
        "present-KV shape must expose num_kv_heads=2 and head_dim=4 for KV inference"
    );

    // Cross-check: a real ORT `Session` over the SAME model must recover the
    // identical geometry — proving the session-free metadata is a faithful
    // stand-in (and that ORT also excludes the initializer-backed input).
    let environment = Environment::new("kv-metadata-geometry-test").expect("ort environment");
    let session = Session::new(
        &environment,
        &path,
        SessionOptions::default().with_intra_op_threads(1),
    )
    .expect("ORT loads the twin decoder");

    assert_eq!(
        session.input_names(),
        metadata.input_names(),
        "session and metadata must declare the same input ports (both exclude initializers)"
    );
    assert_eq!(
        session.output_names(),
        metadata.output_names(),
        "session and metadata must declare the same output ports"
    );
    assert_eq!(
        session.inputs(),
        metadata.inputs(),
        "session and metadata input dtypes/shapes must match exactly"
    );
    assert_eq!(
        session.outputs(),
        metadata.outputs(),
        "session and metadata output dtypes/shapes (KV geometry) must match exactly"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
