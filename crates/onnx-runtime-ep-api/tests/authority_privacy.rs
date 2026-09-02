use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const FEATURE_MATRIX: &[(&str, &[&str])] = &[
    ("default", &[]),
    ("session", &["session"]),
    ("cuda", &["cuda"]),
    ("gpu-tests", &["gpu-tests"]),
    ("workspace-unified", &["workspace-unified"]),
];

const HOSTILE_BINS: &[(&str, &str)] = &[
    ("mint", "module `executor` is private"),
    ("bind", "no `ExecutorArtifactConfigTemplate` in the root"),
    ("proof", "no `ExecutorArtifactConfig` in the root"),
    ("construct_authority", "module `executor` is private"),
    ("finalize", "no method named `finalize_executor_artifacts`"),
    ("maybe_uninit_authority", "module `executor` is private"),
    ("zeroed_authority", "module `executor` is private"),
    ("default_authority", "module `executor` is private"),
    ("public_fields", "module `executor` is private"),
    (
        "trait_default",
        "no method named `finalize_executor_artifacts`",
    ),
    (
        "associated_type",
        "cannot find type `ExecutorArtifactFinalizationProof`",
    ),
    (
        "macro_reexport",
        "no `ExecutorArtifactSessionAuthority` in the root",
    ),
    ("clone_replay", "module `executor` is private"),
    ("report_complete", "no variant named `Complete`"),
    (
        "direct_provider",
        "no method named `resolve_executor_artifact_config`",
    ),
];

fn outer_target(manifest_dir: &Path) -> PathBuf {
    std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("../../target"))
}

fn cargo_check(fixture: &Path, target: &Path, bin: &str, features: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO"));
    command
        .current_dir(fixture)
        .env("CARGO_TARGET_DIR", target)
        .env("RUSTFLAGS", "-D warnings")
        .args(["check", "--quiet", "--bin", bin, "--no-default-features"]);
    if !features.is_empty() {
        command.arg("--features").arg(features.join(","));
    }
    command.output().expect("run hostile Cargo fixture")
}

#[test]
fn path_dependencies_and_feature_unification_expose_no_session_authority() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture = manifest_dir.join("tests/fixtures/authority-unification");
    let target = outer_target(&manifest_dir).join("authority-unification-fixture");

    let mut positive_controls = 0;
    let mut hostile_failures = 0;
    for (matrix_name, features) in FEATURE_MATRIX {
        let output = cargo_check(&fixture, &target, "surface_control", features);
        assert!(
            output.status.success(),
            "authority API positive control failed under {matrix_name}:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        positive_controls += 1;

        for (bin, expected_error) in HOSTILE_BINS {
            let output = cargo_check(&fixture, &target, bin, features);
            assert!(
                !output.status.success(),
                "hostile {bin} unexpectedly compiled under {matrix_name}"
            );
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                stderr.contains(expected_error),
                "hostile {bin} under {matrix_name} failed for the wrong reason; expected \
                 {expected_error:?}:\n{stderr}"
            );
            hostile_failures += 1;
        }
    }
    assert_eq!(positive_controls, 5);
    assert_eq!(hostile_failures, 75);
    eprintln!("authority_path_matrix: hostile=75 controls=5");
}

#[test]
fn removed_authority_package_cannot_be_path_depended_on() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let target_root = outer_target(&manifest_dir);
    let fixture = target_root.join("authority-removed-path-fixture");
    let source = fixture.join("src");
    std::fs::create_dir_all(&source).expect("create removed-authority path fixture");
    let removed = manifest_dir.join("../onnx-runtime-session-authority");
    std::fs::write(
        fixture.join("Cargo.toml"),
        format!(
            "[package]\nname = \"removed-authority-path-hostile\"\nversion = \"0.0.0\"\n\
             edition = \"2024\"\n\n[workspace]\n\n[dependencies]\n\
             onnx-runtime-session-authority = {{ path = {removed:?} }}\n"
        ),
    )
    .expect("write removed-authority path manifest");
    std::fs::write(source.join("main.rs"), "fn main() {}\n")
        .expect("write removed-authority path source");

    let output = Command::new(env!("CARGO"))
        .current_dir(&fixture)
        .env(
            "CARGO_TARGET_DIR",
            target_root.join("authority-removed-path-build"),
        )
        .args(["check", "--quiet"])
        .output()
        .expect("check removed authority path dependency");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("onnx-runtime-session-authority")
            && (stderr.contains("failed to read") || stderr.contains("No such file")),
        "removed authority path dependency failed for the wrong reason:\n{stderr}"
    );
    eprintln!("authority_removed_path: hostile=1");
}

#[test]
fn git_dependencies_expose_data_but_not_session_authority() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("EP API lives under <repo>/crates");
    let target_root = outer_target(&manifest_dir);
    let bare = target_root.join("authority-git-source.git");
    let fixture = target_root.join("authority-git-fixture");
    let source = fixture.join("src");
    if bare.exists() {
        std::fs::remove_dir_all(&bare).expect("replace owned authority git source");
    }
    if fixture.exists() {
        std::fs::remove_dir_all(&fixture).expect("replace owned authority git fixture");
    }
    let clone = Command::new("git")
        .args(["clone", "--quiet", "--bare"])
        .arg(repo)
        .arg(&bare)
        .status()
        .expect("clone current commit for git dependency proof");
    assert!(
        clone.success(),
        "clone current commit for git dependency proof"
    );
    let revision = Command::new("git")
        .current_dir(repo)
        .args(["rev-parse", "HEAD"])
        .output()
        .expect("resolve current git revision");
    assert!(revision.status.success());
    let revision = String::from_utf8(revision.stdout)
        .expect("git revision is UTF-8")
        .trim()
        .to_string();
    let git_url = format!("file://{}", bare.display());

    std::fs::create_dir_all(&source).expect("create git authority fixture");
    std::fs::write(
        fixture.join("Cargo.toml"),
        format!(
            "[package]\nname = \"authority-git-hostile\"\nversion = \"0.0.0\"\n\
             edition = \"2024\"\n\n[workspace]\n\n[dependencies]\n\
             onnx-runtime-ep-api = {{ git = {git_url:?}, rev = {revision:?} }}\n\
             onnx-runtime-session = {{ git = {git_url:?}, rev = {revision:?}, \
             default-features = false, features = [\"cuda\", \"cuda-13000\", \"gpu-tests\"] }}\n\
             onnx-runtime-ep-cuda = {{ git = {git_url:?}, rev = {revision:?}, \
             default-features = false, features = [\"cuda\", \"cuda-13000\", \"gpu-tests\"] }}\n\
             onnx-runtime-ir = {{ git = {git_url:?}, rev = {revision:?} }}\n\n\
             [[bin]]\nname = \"control\"\npath = \"src/control.rs\"\n\n\
             [[bin]]\nname = \"hostile\"\npath = \"src/hostile.rs\"\n"
        ),
    )
    .expect("write git authority manifest");
    std::fs::write(
        source.join("control.rs"),
        "use onnx_runtime_ep_api::{ExecutorArtifactPolicy, ExecutorArtifactProviderId, \
         ExecutorRouteResidencyConfig};\nuse onnx_runtime_ir::DeviceId;\nfn main() {\n    \
         let p = ExecutorArtifactPolicy::new(ExecutorArtifactProviderId::from_raw(1), \
         DeviceId::cuda(0), ExecutorRouteResidencyConfig::Disabled);\n    assert_eq!(p.device(), \
         DeviceId::cuda(0));\n    let _ = onnx_runtime_session::OpsetVersion::Known(17);\n    let \
         _ = std::mem::size_of::<onnx_runtime_ep_cuda::CudaExecutionProvider>();\n}\n",
    )
    .expect("write git positive control");
    std::fs::write(
        source.join("hostile.rs"),
        "fn main() {\n    let _ = \
         onnx_runtime_session::executor::issue_executor_instance_id();\n}\n",
    )
    .expect("write git hostile source");

    let build_target = target_root.join("authority-git-build");
    let control = cargo_check(&fixture, &build_target, "control", &[]);
    assert!(
        control.status.success(),
        "git dependency positive control failed:\n{}",
        String::from_utf8_lossy(&control.stderr)
    );
    let hostile = cargo_check(&fixture, &build_target, "hostile", &[]);
    assert!(!hostile.status.success());
    assert!(
        String::from_utf8_lossy(&hostile.stderr).contains("module `executor` is private"),
        "git hostile attack failed at the wrong boundary:\n{}",
        String::from_utf8_lossy(&hostile.stderr)
    );

    std::fs::write(
        fixture.join("Cargo.toml"),
        format!(
            "[package]\nname = \"removed-authority-git-hostile\"\nversion = \"0.0.0\"\n\
             edition = \"2024\"\n\n[workspace]\n\n[dependencies]\n\
             onnx-runtime-session-authority = {{ git = {git_url:?}, rev = {revision:?} }}\n"
        ),
    )
    .expect("write removed-authority git manifest");
    std::fs::write(source.join("main.rs"), "fn main() {}\n")
        .expect("write removed-authority git source");
    let removed = Command::new(env!("CARGO"))
        .current_dir(&fixture)
        .env("CARGO_TARGET_DIR", &build_target)
        .args(["check", "--quiet"])
        .output()
        .expect("check removed authority git dependency");
    assert!(!removed.status.success());
    let stderr = String::from_utf8_lossy(&removed.stderr);
    assert!(
        stderr.contains("onnx-runtime-session-authority") && stderr.contains("no matching package"),
        "removed authority git dependency failed for the wrong reason:\n{stderr}"
    );
    eprintln!("authority_git_matrix: hostile=2 controls=1");
}

#[test]
fn public_api_contains_no_authority_or_finalization_proof_surface() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace =
        std::fs::read_to_string(manifest_dir.join("../../Cargo.toml")).expect("read workspace");
    let manifest =
        std::fs::read_to_string(manifest_dir.join("Cargo.toml")).expect("read EP API manifest");
    let provider =
        std::fs::read_to_string(manifest_dir.join("src/provider.rs")).expect("read provider API");
    let lib = std::fs::read_to_string(manifest_dir.join("src/lib.rs")).expect("read EP API root");
    let session_executor =
        std::fs::read_to_string(manifest_dir.join("../onnx-runtime-session/src/executor/mod.rs"))
            .expect("read session executor authority owner");
    let session_root =
        std::fs::read_to_string(manifest_dir.join("../onnx-runtime-session/src/lib.rs"))
            .expect("read session crate root");
    let cuda_root =
        std::fs::read_to_string(manifest_dir.join("../onnx-runtime-ep-cuda/src/lib.rs"))
            .expect("read CUDA crate root");

    for source in [
        &workspace,
        &manifest,
        &provider,
        &lib,
        &session_root,
        &cuda_root,
    ] {
        assert!(!source.contains("onnx-runtime-session-authority"));
        assert!(!source.contains("ExecutorArtifactSessionAuthority"));
        assert!(!source.contains("ExecutorArtifactConfigTemplate"));
        assert!(!source.contains("ExecutorArtifactFinalizationProof"));
        assert!(!source.contains("ExecutorArtifactFinalizationOutcome"));
        assert!(!source.contains("finalize_executor_artifacts"));
    }
    assert!(provider.contains("This constructs data, not authority"));
    assert!(provider.contains("This method supplies data only"));
    assert!(provider.contains("fn inspect_executor_artifacts("));
    assert!(session_executor.contains("struct ExecutorArtifactConfig"));
    assert!(!session_executor.contains("pub struct ExecutorArtifactConfig"));
    assert!(session_executor.contains("fn issue("));
    assert!(!session_executor.contains("pub fn issue("));
    assert!(session_root.contains("mod executor;"));
    assert!(!session_root.contains("pub mod executor;"));
}
