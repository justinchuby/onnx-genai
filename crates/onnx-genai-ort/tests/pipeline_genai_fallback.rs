use std::path::{Path, PathBuf};

use onnx_genai_ort::PipelineModelDirectory;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../onnx-genai-genai-config/tests/fixtures")
        .join(name)
}

fn assert_complete_vlm(directory: &PipelineModelDirectory) {
    let preprocessing = directory
        .preprocessing
        .as_ref()
        .and_then(|preprocessing| preprocessing.image.as_ref())
        .expect("typed image preprocessing");
    let decoder_io = directory.spec.models["decoder"]
        .io
        .as_ref()
        .expect("decoder io");

    assert_eq!(
        preprocessing
            .outputs
            .iter()
            .map(|output| output.name.as_str())
            .collect::<Vec<_>>(),
        ["pixel_values", "image_grid_thw"]
    );
    assert_eq!(
        directory
            .spec
            .positions
            .as_ref()
            .expect("position program")
            .rank,
        3
    );
    assert_eq!(decoder_io.kv_inputs.as_ref().map(Vec::len), Some(16));
    assert_eq!(decoder_io.kv_outputs.as_ref().map(Vec::len), Some(16));
    assert_eq!(decoder_io.state_pairs.as_ref().map(Vec::len), Some(48));
}

#[test]
fn complete_genai_package_loads_as_pipeline_without_native_sidecar() {
    let directory = PipelineModelDirectory::load(fixture("vlm-complete"))
        .expect("complete compatibility package loads");

    assert!(directory.metadata_path.ends_with("genai_config.json"));
    assert_complete_vlm(&directory);
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
fn real_foundry_package_loads_when_installed() {
    let model_dir =
        Path::new("/home/justinchu/.foundry/cache/models/Microsoft/qwen3.5-9b-generic-cpu-2/v2");
    if !model_dir.is_dir() {
        eprintln!("real Foundry Qwen3.5 package is not installed; validation is deferred");
        return;
    }

    let directory =
        PipelineModelDirectory::load(model_dir).expect("real Foundry Qwen3.5 package loads");
    assert_complete_vlm(&directory);
}
