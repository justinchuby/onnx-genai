use std::path::{Path, PathBuf};

use onnx_model_package::{ModelPackage, PackageError, SelectionRequest};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn parses_valid_manifest_and_accepts_newer_minor_version() {
    let package = ModelPackage::open(fixture("valid-package")).unwrap();
    assert_eq!(package.manifest().schema_version, "1.7");
    assert_eq!(package.manifest().package_name.as_deref(), Some("tiny"));
    package.validate().unwrap();
}

#[test]
fn missing_required_field_is_an_error() {
    let error = ModelPackage::open(fixture("missing-required")).unwrap_err();
    assert!(matches!(error, PackageError::Json { .. }));
    assert!(error.to_string().contains("schema_version"));
}

#[test]
fn unsupported_major_version_is_an_error() {
    let error = ModelPackage::open(fixture("unsupported-version")).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("unsupported package schema major")
    );
}

#[test]
fn unknown_named_variant_is_an_error() {
    let package = ModelPackage::open(fixture("valid-package")).unwrap();
    let error = package
        .select("model", &SelectionRequest::named("missing"))
        .unwrap_err();
    assert!(matches!(error, PackageError::UnknownVariant { .. }));
}

#[test]
fn selection_matches_execution_provider_and_precision_in_manifest_order() {
    let package = ModelPackage::open(fixture("valid-package")).unwrap();
    let selected = package
        .select(
            "model",
            &SelectionRequest {
                execution_provider: Some("CPUExecutionProvider".to_string()),
                precision: Some("float16".to_string()),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(selected.variant_name, "cpu-fp16");
    assert_eq!(
        selected.model_path,
        fixture("valid-package").join("cpu-fp16/model.onnx")
    );
}

#[test]
fn execution_provider_boundary_does_not_fall_back_to_wrong_provider() {
    let package = ModelPackage::open(fixture("valid-package")).unwrap();
    let error = package
        .select(
            "model",
            &SelectionRequest::for_execution_provider("QNNExecutionProvider"),
        )
        .unwrap_err();
    assert!(matches!(error, PackageError::NoMatchingVariant { .. }));
}

#[test]
fn resolves_shared_tokenizer_directory() {
    let package = ModelPackage::open(fixture("valid-package")).unwrap();
    let selected = package
        .select("model", &SelectionRequest::named("cpu-fp32"))
        .unwrap();
    assert_eq!(
        selected.tokenizer_directory,
        Some(
            fixture("valid-package")
                .join("shared_assets")
                .join(format!("sha256-{}", "a".repeat(64)))
        )
    );
}

#[test]
fn validation_rejects_missing_referenced_file() {
    let package = ModelPackage::open(fixture("missing-model")).unwrap();
    let error = package.validate().unwrap_err();
    assert!(error.to_string().contains("referenced path does not exist"));
    assert!(error.to_string().contains("model.onnx"));
}
