//! End-to-end coverage for `--profile-json`.
//!
//! Asking for a profile sets `ONNX_GENAI_PROFILE` in the environment, which the
//! runtime config reads and freezes on first use. That is safe in a real run,
//! where it happens while the process is still parsing arguments, but a unit
//! test cannot promise it: another test in the same binary may already have
//! frozen the config, and the freeze guard then trips on a knob that changed
//! afterwards. The guard's own advice is to isolate a policy phase in its own
//! process, so these drive the real binary and read the profile it wrote.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture(name: &str) -> PathBuf {
    repository_root()
        .join("tests/fixtures")
        .join(name)
        .canonicalize()
        .expect("the fixture must exist")
}

/// A directory of this test's own, so parallel tests cannot read each other's
/// half-written profile.
fn scratch(name: &str) -> PathBuf {
    let directory = repository_root()
        .join("target/test-profiles")
        .join(format!("{}-{name}", std::process::id()));
    std::fs::create_dir_all(&directory).expect("the scratch directory must be created");
    directory
}

fn run_cli(arguments: &[&str]) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_onnx-genai"))
        .args(arguments)
        .env("ONNX_GENAI_EP", "cpu")
        .output()
        .expect("the onnx-genai binary must start");
    assert!(
        output.status.success(),
        "the command must succeed: {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// The provider the profile says the run used.
fn profile_execution_provider(path: &Path) -> String {
    let text = std::fs::read_to_string(path).expect("the profile must have been written");
    let value: serde_json::Value = serde_json::from_str(&text).expect("the profile must be JSON");
    value["execution_provider"]
        .as_str()
        .expect("profile must include execution_provider")
        .to_string()
}

#[test]
fn generate_profile_provider_comes_from_live_command_profile() {
    let directory = scratch("generate");
    let profile = directory.join("profile.json");

    run_cli(&[
        "--profile-json",
        &profile.display().to_string(),
        "generate",
        &fixture("tiny-llm").display().to_string(),
        "--prompt",
        "hi",
        "--max-new-tokens",
        "1",
        "--no-stats",
        "--cpu-cores",
        "1",
    ]);

    assert_eq!(profile_execution_provider(&profile), "cpu");
    std::fs::remove_dir_all(directory).expect("the scratch directory must be removed");
}

#[test]
fn transcribe_profile_provider_comes_from_live_command_profile() {
    let directory = scratch("transcribe");
    let profile = directory.join("profile.json");
    let model = fixture("tiny-whisper");

    run_cli(&[
        "--profile-json",
        &profile.display().to_string(),
        "transcribe",
        &model.display().to_string(),
        &model.join("tiny.wav").display().to_string(),
        "--format",
        "json",
        "--cpu-cores",
        "1",
    ]);

    assert_eq!(profile_execution_provider(&profile), "cpu");
    std::fs::remove_dir_all(directory).expect("the scratch directory must be removed");
}
