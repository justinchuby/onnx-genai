use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use onnx_genai_ort::{PipelineModelDirectory, PipelineModels, SessionOptions};
use onnx_std::Model;
use onnx_std::ir::{Attribute, DataType, Graph, Node, NodeId};

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
