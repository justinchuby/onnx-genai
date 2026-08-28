use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use onnx_genai_engine::{Engine, EngineConfig};

const STATIC_CACHE: &str = include_str!(
    "../../../tests/fixtures/onnx_genai_workflows/static_cache/inference_metadata.yaml"
);

fn package(adapter: &str, version: &str) -> anyhow::Result<PathBuf> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target-extension-registry/checkpoint-engine-tests")
        .join(NEXT.fetch_add(1, Ordering::Relaxed).to_string());
    fs::create_dir_all(&root)?;
    let marker = "            ports:\n              model:";
    let replacement = format!(
        "            checkpoint:\n              adapter: {adapter}\n              version: '{version}'\n\
         {marker}"
    );
    let metadata = STATIC_CACHE.replacen(marker, &replacement, 1);
    assert_ne!(
        metadata, STATIC_CACHE,
        "fixture checkpoint insertion drifted"
    );
    fs::write(root.join("inference_metadata.yaml"), metadata)?;
    Ok(root)
}

#[test]
fn engine_rejects_checkpoint_extensions_before_component_or_state_allocation() -> anyhow::Result<()>
{
    for (adapter, version, expected) in [
        ("onnx-genai.kv-checkpoint", "1", "known, but unavailable"),
        ("onnx-genai.kv-checkpoint", "2", "version is not registered"),
        (
            "onnx-genai.tensor-checkpoint",
            "1",
            "identity 'onnx-genai.tensor-checkpoint' is not registered",
        ),
    ] {
        let root = package(adapter, version)?;
        let error = match Engine::from_dir(&root, EngineConfig::default()) {
            Err(error) => error,
            Ok(_) => panic!("checkpoint contract must fail before constructing an engine"),
        };
        let message = format!("{error:#}");
        assert!(
            message.contains("decoder_cache")
                && message.contains("FullAttention")
                && message.contains(&format!("{adapter}@{version}"))
                && message.contains(expected)
                && !message.contains("model.onnx.textproto"),
            "{message}"
        );
    }
    Ok(())
}
