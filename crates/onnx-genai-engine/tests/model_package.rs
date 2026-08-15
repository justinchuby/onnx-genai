use std::path::{Path, PathBuf};

use onnx_genai_engine::{Engine, EngineConfig};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/model-package-cpu")
}

#[test]
fn engine_loads_ort_model_package_directory() -> anyhow::Result<()> {
    let _engine = Engine::from_dir(&fixture(), EngineConfig::default())?;
    Ok(())
}
