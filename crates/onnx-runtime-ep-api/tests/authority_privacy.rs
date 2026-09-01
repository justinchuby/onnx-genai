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
    (
        "construct_authority",
        "no `ExecutorArtifactSessionAuthority` in the root",
    ),
    (
        "finalize",
        "this function takes 1 argument but 0 arguments were supplied",
    ),
    (
        "maybe_uninit_authority",
        "does not permit being left uninitialized",
    ),
    ("zeroed_authority", "does not permit zero-initialization"),
    (
        "default_authority",
        "the trait bound `onnx_runtime_session_authority::ExecutorArtifactSessionAuthority: Default` is not satisfied",
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

    let mut positive_controls = 0;
    let mut hostile_failures = 0;
    for (matrix_name, features) in FEATURE_MATRIX {
        let mut control = Command::new(env!("CARGO"));
        control
            .current_dir(&fixture)
            .env("CARGO_TARGET_DIR", &target)
            .env("RUSTFLAGS", "-D warnings")
            .args([
                "check",
                "--quiet",
                "--bin",
                "surface_control",
                "--no-default-features",
            ]);
        if !features.is_empty() {
            control.arg("--features").arg(features.join(","));
        }
        let output = control
            .output()
            .expect("run hostile fixture positive control");
        assert!(
            output.status.success(),
            "authority API positive control failed under {matrix_name}:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        positive_controls += 1;

        for (bin, expected_error) in HOSTILE_BINS {
            let mut command = Command::new(env!("CARGO"));
            command
                .current_dir(&fixture)
                .env("CARGO_TARGET_DIR", &target)
                .env("RUSTFLAGS", "-D warnings")
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
            hostile_failures += 1;
        }
    }
    assert_eq!(positive_controls, FEATURE_MATRIX.len());
    assert_eq!(hostile_failures, FEATURE_MATRIX.len() * HOSTILE_BINS.len());
}

#[test]
fn public_api_has_no_authority_issuer_or_reexport() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let manifest = std::fs::read_to_string(manifest_dir.join("Cargo.toml"))
        .expect("read ep-api Cargo manifest");
    let provider = std::fs::read_to_string(manifest_dir.join("src/provider.rs"))
        .expect("read provider API source");
    let lib =
        std::fs::read_to_string(manifest_dir.join("src/lib.rs")).expect("read ep-api crate root");
    let authority_manifest =
        std::fs::read_to_string(manifest_dir.join("../onnx-runtime-session-authority/Cargo.toml"))
            .expect("read private authority Cargo manifest");
    let authority =
        std::fs::read_to_string(manifest_dir.join("../onnx-runtime-session-authority/src/lib.rs"))
            .expect("read private authority implementation");

    assert!(!manifest.contains("runtime-session-authority = []"));
    assert!(!provider.contains("feature = \"runtime-session-authority\""));
    assert!(!provider.contains("pub struct ExecutorArtifactSessionAuthority"));
    assert!(!lib.contains("ExecutorArtifactSessionAuthority"));
    assert!(provider.contains("authority: &ExecutorArtifactSessionAuthority"));
    assert!(authority_manifest.contains("publish = false"));
    assert!(authority.contains("seal: Arc<AuthoritySeal>"));
    assert!(authority.contains("allocation: Box<u8>"));
    assert!(!authority.contains("impl Default for ExecutorArtifactSessionAuthority"));
    assert!(!authority.contains("Serialize"));
    assert!(!authority.contains("Deserialize"));
    assert!(!authority.contains("#[repr("));
}
