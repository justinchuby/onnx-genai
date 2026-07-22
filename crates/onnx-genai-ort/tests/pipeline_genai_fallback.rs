use std::path::{Path, PathBuf};

use onnx_genai_ort::PipelineModelDirectory;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../onnx-genai-genai-config/tests/fixtures")
        .join(name)
}

#[test]
fn complete_genai_metadata_still_rejects_non_executable_edge_rank() {
    let error = PipelineModelDirectory::load(fixture("vlm-complete"))
        .expect_err("rank-mismatched compatibility package must fail admission")
        .to_string();

    assert!(error.contains("embedding.image_features"), "{error}");
    assert!(error.contains("incompatible ranks"), "{error}");
    assert!(
        error.contains("producer rank 2, consumer rank 3"),
        "{error}"
    );
    assert!(error.contains("regenerate the native sidecar"), "{error}");
}

#[test]
fn incomplete_genai_package_fails_with_regeneration_guidance() {
    let error = PipelineModelDirectory::load(fixture("vlm-incomplete"))
        .expect_err("incomplete compatibility package must fail")
        .to_string();

    assert!(error.contains("missing required semantics"));
    assert!(error.contains("mrope_section"));
    assert!(error.contains("Why:"));
    assert!(error.contains("never guesses from model.type"));
    assert!(error.contains("How to fix:"));
    assert!(error.contains("native inference_metadata.json"));
}

#[test]
fn real_foundry_package_fails_loudly_when_preprocessing_is_not_executable() {
    let model_dir =
        Path::new("/home/justinchu/.foundry/cache/models/Microsoft/qwen3.5-9b-generic-cpu-2/v2");
    if !model_dir.is_dir() {
        eprintln!("real Foundry Qwen3.5 package is not installed; validation is deferred");
        return;
    }

    let error = PipelineModelDirectory::load(model_dir)
        .expect_err("unsupported smart resize must not load")
        .to_string();
    assert!(error.contains("smart_resize=false"));
    assert!(error.contains("Why:"));
    assert!(error.contains("How to fix:"));
}
