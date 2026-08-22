use std::path::{Path, PathBuf};

use onnx_model_package::{HostTrust, ModelPackage, PackageError, SelectionRequest};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn fixture_canonical(name: &str) -> PathBuf {
    fixture(name).canonicalize().unwrap()
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
        fixture_canonical("valid-package").join("cpu-fp16/model.onnx.textproto")
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
            fixture_canonical("valid-package")
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

// ---------------------------------------------------------------------------
// Path-confinement / traversal security tests (issue #54, PR #322).
//
// The manifest is UNTRUSTED input: nothing inside it may cause a read outside
// the package root. These tests fail against the pre-fix code and pass after
// the confinement fixes.
// ---------------------------------------------------------------------------

use std::fs;

/// A unique, freshly emptied scratch directory under the crate's target dir
/// (never `/tmp`). `CARGO_TARGET_TMPDIR` lives inside the workspace target.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, contents).unwrap();
}

/// Build a single-variant package whose `model_file` (relative to the `cpu`
/// variant directory) is exactly `model_file`, with the declared `layout`.
fn build_package(root: &Path, layout: &str, model_file: &str) {
    let manifest = format!(
        r#"{{
  "schema_version": "1.0",
  "layout": "{layout}",
  "components": {{
    "model": {{
      "component_name": "model",
      "variants": {{
        "cpu": {{
          "variant_directory": "cpu",
          "ep": "CPUExecutionProvider",
          "executor_info": {{ "ort": {{ "model_file": "{model_file}" }} }}
        }}
      }}
    }}
  }}
}}"#
    );
    write(&root.join("manifest.json"), &manifest);
    fs::create_dir_all(root.join("cpu")).unwrap();
}

#[test]
fn installed_layout_absolute_model_file_is_rejected_by_default() {
    // Vulnerability #1: an untrusted manifest declaring `installed` layout must
    // NOT be able to read an absolute host path such as `/etc/passwd`.
    let root = scratch("installed-absolute");
    build_package(&root, "installed", "/etc/passwd");
    let package = ModelPackage::open(&root).unwrap();
    let error = package.validate().unwrap_err();
    assert!(
        matches!(error, PackageError::Invalid(_)),
        "expected confinement rejection, got: {error:?}"
    );
}

#[test]
fn installed_layout_parent_traversal_is_rejected_by_default() {
    // `installed` layout must not grant `..` escape from manifest content alone.
    let outside = scratch("installed-parent-outside");
    write(&outside.join("secret.onnx"), "secret");
    let root = outside.join("pkg");
    build_package(&root, "installed", "../../secret.onnx");
    let package = ModelPackage::open(&root).unwrap();
    let error = package.validate().unwrap_err();
    assert!(matches!(error, PackageError::Invalid(_)), "{error:?}");
}

#[test]
fn installed_layout_escape_requires_explicit_host_trust() {
    // Positive/negative control for the caller-supplied trust gate: the SAME
    // package escapes the root only when the host explicitly opts in.
    let base = scratch("installed-trust-gate");
    write(&base.join("outside/model.onnx"), "model-bytes");
    let root = base.join("pkg");
    build_package(&root, "installed", "../../outside/model.onnx");

    // Default (untrusted) open confines and rejects the escape.
    let confined = ModelPackage::open(&root).unwrap();
    assert!(confined.validate().is_err());

    // A trusted host may opt in to installed-layout escapes.
    let trusted = ModelPackage::open_with_trust(&root, HostTrust::AllowInstalledLayout).unwrap();
    let selected = trusted
        .select("model", &SelectionRequest::named("cpu"))
        .expect("trusted installed layout should resolve the escaped path");
    assert!(selected.model_path.ends_with("outside/model.onnx"));
}

#[test]
fn portable_absolute_reference_is_rejected() {
    let root = scratch("portable-absolute");
    build_package(&root, "portable", "/etc/passwd");
    let package = ModelPackage::open(&root).unwrap();
    assert!(package.validate().is_err());
}

#[test]
fn portable_parent_traversal_is_rejected() {
    let outside = scratch("portable-parent-outside");
    write(&outside.join("secret.onnx"), "secret");
    let root = outside.join("pkg");
    build_package(&root, "portable", "../../secret.onnx");
    let package = ModelPackage::open(&root).unwrap();
    assert!(package.validate().is_err());
}

#[test]
fn confined_package_built_at_runtime_still_loads() {
    // Positive control: a legitimate confined package loads successfully.
    let root = scratch("confined-positive");
    build_package(&root, "portable", "model.onnx");
    write(&root.join("cpu/model.onnx"), "model-bytes");
    let package = ModelPackage::open(&root).unwrap();
    package.validate().unwrap();
    let selected = package
        .select("model", &SelectionRequest::named("cpu"))
        .unwrap();
    assert!(selected.model_path.ends_with("cpu/model.onnx"));
}

#[cfg(unix)]
#[test]
fn symlinked_model_file_escaping_root_is_rejected() {
    // Vulnerability #1 (symlink variant): a symlink INSIDE the package pointing
    // OUTSIDE the root must be rejected via canonicalization.
    use std::os::unix::fs::symlink;

    let base = scratch("symlink-model-escape");
    write(&base.join("outside/secret.onnx"), "secret");
    let root = base.join("pkg");
    build_package(&root, "portable", "model.onnx");
    symlink(
        base.join("outside/secret.onnx"),
        root.join("cpu/model.onnx"),
    )
    .unwrap();

    let package = ModelPackage::open(&root).unwrap();
    let error = package.validate().unwrap_err();
    assert!(
        error.to_string().contains("resolves outside package root"),
        "expected confinement error, got: {error}"
    );
}

#[cfg(unix)]
#[test]
fn symlinked_external_component_json_escaping_root_is_rejected() {
    // Vulnerability #2: the implicit `component.json` under an external
    // component directory must pass the same confinement check. A symlinked
    // component.json pointing outside the root must be rejected.
    use std::os::unix::fs::symlink;

    let base = scratch("symlink-component-escape");
    // A structurally valid component definition placed outside the package.
    write(
        &base.join("outside/evil.json"),
        r#"{
  "component_name": "model",
  "variants": {
    "cpu": {
      "variant_directory": "cpu",
      "ep": "CPUExecutionProvider",
      "executor_info": { "ort": { "model_file": "model.onnx" } }
    }
  }
}"#,
    );

    let root = base.join("pkg");
    let manifest = r#"{
  "schema_version": "1.0",
  "layout": "portable",
  "components": { "model": "component-dir" }
}"#;
    write(&root.join("manifest.json"), manifest);
    fs::create_dir_all(root.join("component-dir")).unwrap();
    fs::create_dir_all(root.join("cpu")).unwrap();
    write(&root.join("cpu/model.onnx"), "model-bytes");
    // The component.json is a symlink escaping the package root.
    symlink(
        base.join("outside/evil.json"),
        root.join("component-dir/component.json"),
    )
    .unwrap();

    let package = ModelPackage::open(&root).unwrap();
    let error = package.validate().unwrap_err();
    assert!(
        error.to_string().contains("resolves outside package root"),
        "expected confinement error for symlinked component.json, got: {error}"
    );
}

#[cfg(unix)]
#[test]
fn confined_external_component_json_loads() {
    // Positive control for vuln #2 fix: a legitimate in-root component.json
    // loads normally (no false positive from the confinement check).
    let root = scratch("confined-external-component");
    let manifest = r#"{
  "schema_version": "1.0",
  "layout": "portable",
  "components": { "model": "component-dir" }
}"#;
    write(&root.join("manifest.json"), manifest);
    write(
        &root.join("component-dir/component.json"),
        r#"{
  "component_name": "model",
  "variants": {
    "cpu": {
      "variant_directory": "cpu",
      "ep": "CPUExecutionProvider",
      "executor_info": { "ort": { "model_file": "model.onnx" } }
    }
  }
}"#,
    );
    fs::create_dir_all(root.join("cpu")).unwrap();
    write(&root.join("cpu/model.onnx"), "model-bytes");

    let package = ModelPackage::open(&root).unwrap();
    package.validate().unwrap();
}

#[cfg(unix)]
#[test]
fn package_under_symlinked_root_still_loads() {
    // The package root itself may live under a symlinked directory (e.g. a
    // temp dir). Canonicalizing the root must not spuriously reject in-root
    // references (macOS /tmp -> /private/tmp analogue).
    use std::os::unix::fs::symlink;

    let base = scratch("symlinked-root");
    let real = base.join("real-pkg");
    build_package(&real, "portable", "model.onnx");
    write(&real.join("cpu/model.onnx"), "model-bytes");
    let link = base.join("linked-pkg");
    symlink(&real, &link).unwrap();

    let package = ModelPackage::open(&link).unwrap();
    package.validate().unwrap();
}
