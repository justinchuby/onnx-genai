use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use onnx_genai_metadata::{load_metadata_package, parse_metadata, validate_metadata};

const STATIC_CACHE: &str = include_str!(
    "../../../tests/fixtures/onnx_genai_workflows/static_cache/inference_metadata.yaml"
);

fn with_checkpoint(adapter: &str, version: &str) -> String {
    let marker = "            ports:\n              model:";
    let replacement = format!(
        "            checkpoint:\n              adapter: {adapter}\n              version: '{version}'\n\
         {marker}"
    );
    let document = STATIC_CACHE.replacen(marker, &replacement, 1);
    assert_ne!(
        document, STATIC_CACHE,
        "fixture checkpoint insertion drifted"
    );
    document
}

fn validation_message(adapter: &str, version: &str) -> String {
    let metadata =
        parse_metadata(&with_checkpoint(adapter, version), Some("yaml")).expect("syntax parses");
    validate_metadata(&metadata)
        .expect_err("checkpoint extension must fail closed")
        .join("; ")
}

fn fixture_root(name: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target-extension-registry/checkpoint-tests")
        .join(format!("{name}-{}", NEXT.fetch_add(1, Ordering::Relaxed)))
}

#[test]
fn parser_preserves_the_exact_checkpoint_pair_for_registry_validation() {
    let metadata = parse_metadata(
        &with_checkpoint("example.invalid/checkpoint", "37"),
        Some("yaml"),
    )
    .expect("the syntax parser preserves extension declarations");
    let checkpoint = metadata
        .pipeline
        .as_ref()
        .and_then(|pipeline| pipeline.workflow.serving.as_ref())
        .and_then(|serving| serving.state_service.groups.get("decoder_cache"))
        .and_then(|group| group.checkpoint.as_ref())
        .expect("checkpoint declaration remains typed");
    assert_eq!(checkpoint.adapter, "example.invalid/checkpoint");
    assert_eq!(checkpoint.version, "37");

    let message = validate_metadata(&metadata)
        .expect_err("semantic validation must reject an arbitrary pair")
        .join("; ");
    assert!(
        message.contains("decoder_cache")
            && message.contains("example.invalid/checkpoint@37")
            && message.contains("onnx-genai.kv-checkpoint@1"),
        "{message}"
    );
}

#[test]
fn checkpoint_identity_version_and_availability_fail_differently() {
    let unavailable = validation_message("onnx-genai.kv-checkpoint", "1");
    assert!(
        unavailable.contains("known extension 'onnx-genai.kv-checkpoint@1'")
            && unavailable.contains("known, but unavailable")
            && unavailable.contains("before artifact/session/state allocation"),
        "{unavailable}"
    );

    let version = validation_message("onnx-genai.kv-checkpoint", "2");
    assert!(
        version.contains("version is not registered")
            && version.contains("Registered versions")
            && version.contains("1"),
        "{version}"
    );

    let identity = validation_message("onnx-genai.tensor-checkpoint", "1");
    assert!(
        identity.contains("identity 'onnx-genai.tensor-checkpoint' is not registered")
            && identity.contains("onnx-genai.kv-checkpoint@1")
            && identity.contains("omit checkpoint"),
        "{identity}"
    );
}

#[test]
fn package_loading_rejects_checkpoint_before_artifact_resolution()
-> Result<(), Box<dyn std::error::Error>> {
    let root = fixture_root("pre-artifact");
    fs::create_dir_all(&root)?;
    fs::write(
        root.join("inference_metadata.yaml"),
        with_checkpoint("onnx-genai.kv-checkpoint", "1"),
    )?;

    let error = load_metadata_package(&root)
        .expect_err("checkpoint admission must precede missing component artifacts")
        .to_string();
    assert!(
        error.contains("onnx-genai.kv-checkpoint@1")
            && error.contains("before artifact/session/state allocation")
            && !error.contains("model.onnx.textproto"),
        "{error}"
    );
    Ok(())
}
