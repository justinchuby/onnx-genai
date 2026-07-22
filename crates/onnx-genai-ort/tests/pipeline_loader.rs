use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use onnx_genai_ort::PipelineModelDirectory;
use onnx_std::Model;
use onnx_std::ir::{DataType, Graph, Node, NodeId};

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
