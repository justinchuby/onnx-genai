use std::path::PathBuf;
use std::process::Command;

const FEATURE_MATRIX: &[(&str, &[&str])] = &[
    ("default", &[]),
    ("session", &["session"]),
    ("cuda", &["cuda"]),
    ("gpu-tests", &["gpu-tests"]),
    ("workspace-unified", &["workspace-unified"]),
];

const HOSTILE_BINS: &[(&str, &str)] = &[
    (
        "mint",
        "this function takes 1 argument but 0 arguments were supplied",
    ),
    (
        "bind",
        "this method takes 2 arguments but 1 argument was supplied",
    ),
    (
        "proof",
        "this method takes 2 arguments but 1 argument was supplied",
    ),
    ("construct_authority", "field `private`"),
    (
        "finalize",
        "this function takes 1 argument but 0 arguments were supplied",
    ),
];

#[test]
fn feature_unification_cannot_expose_safe_session_authority() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture = manifest_dir.join("tests/fixtures/authority-unification");
    let outer_target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("../../target"));
    let target = outer_target.join("authority-unification-fixture");

    for (matrix_name, features) in FEATURE_MATRIX {
        for (bin, expected_error) in HOSTILE_BINS {
            let mut command = Command::new(env!("CARGO"));
            command
                .current_dir(&fixture)
                .env("CARGO_TARGET_DIR", &target)
                .args(["check", "--quiet", "--bin", bin, "--no-default-features"]);
            if !features.is_empty() {
                command.arg("--features").arg(features.join(","));
            }
            let output = command.output().expect("run hostile Cargo fixture");
            assert!(
                !output.status.success(),
                "hostile {bin} unexpectedly compiled under {matrix_name}"
            );
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                stderr.contains(expected_error),
                "hostile {bin} under {matrix_name} failed for the wrong reason:\n{stderr}"
            );
        }
    }
}

#[test]
fn production_manifest_has_no_authority_feature_gate() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let manifest = std::fs::read_to_string(manifest_dir.join("Cargo.toml"))
        .expect("read ep-api Cargo manifest");
    let provider = std::fs::read_to_string(manifest_dir.join("src/provider.rs"))
        .expect("read provider API source");

    assert!(!manifest.contains("runtime-session-authority"));
    assert!(!provider.contains("feature = \"runtime-session-authority\""));
    assert!(provider.contains("pub struct ExecutorArtifactSessionAuthority"));
    assert!(!provider.contains("impl ExecutorArtifactSessionAuthority"));
    assert!(provider.contains("authority: &ExecutorArtifactSessionAuthority"));
}
