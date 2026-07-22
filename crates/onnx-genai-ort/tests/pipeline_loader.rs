use std::path::{Path, PathBuf};

use onnx_genai_ort::PipelineModelDirectory;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn resolves_multi_model_pipeline_directory() {
    let directory = PipelineModelDirectory::load(fixture("multi-model-pipeline"))
        .expect("pipeline directory resolves");

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
    let directory = PipelineModelDirectory::load(fixture("multi-model-pipeline"))
        .expect("native metadata must bypass the invalid compatibility file");

    assert!(
        directory
            .metadata_path
            .as_deref()
            .is_some_and(|path| path.ends_with("inference_metadata.yaml"))
    );
    assert_eq!(directory.spec.models.len(), 2);
}
