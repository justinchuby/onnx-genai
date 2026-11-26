//! Cargo-built subprocess coverage for `--profile-json` and `--profile-trace`.
//!
//! Asking for a profile sets `ONNX_GENAI_PROFILE` in the environment, which the
//! runtime config reads and freezes on first use. That is safe in a real run,
//! where it happens while the process is still parsing arguments, but a unit
//! test cannot promise it: another test in the same binary may already have
//! frozen the config, and the freeze guard then trips on a knob that changed
//! afterwards. The guard's own advice is to isolate a policy phase in its own
//! process, so these drive the real binary and read the profile it wrote.
//!
//! Generation profiles are covered here. `transcription_e2e` exercises maintained
//! frame/token speech packages and verifies the real CLI's Native profile too.

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

/// Isolated OS temporary storage, independent of Cargo's shared target directory.
fn scratch() -> tempfile::TempDir {
    tempfile::tempdir().expect("the profile scratch directory must be created")
}

fn run_cli(arguments: &[&str], environment: &[(&str, &str)]) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_onnx-genai"))
        .args(arguments)
        .env_remove("ONNX_GENAI_BACKEND")
        .env_remove("ONNX_GENAI_PROFILE")
        .env_remove("ONNX_GENAI_TRACE")
        .env("ONNX_GENAI_EP", "cpu")
        .envs(environment.iter().copied())
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
fn generate_profile_provider_comes_from_explicit_ort_reference_run() {
    let directory = scratch();
    let profile = directory.path().join("profile.json");

    run_cli(
        &[
            "--profile-json",
            &profile.display().to_string(),
            "generate",
            &fixture("tiny-llm").display().to_string(),
            "hi",
            "--backend",
            "ort",
            "--max-new-tokens",
            "1",
            "--no-stats",
            "--cpu-cores",
            "1",
        ],
        &[],
    );

    assert_eq!(profile_execution_provider(&profile), "cpu");
}

/// Exercise the process entry point as well as Native generation: Windows'
/// small process-main stack must not prevent the default command from running.
#[cfg(feature = "native-backend")]
#[test]
fn generate_defaults_to_native_and_profiles_the_actual_backend() {
    let directory = scratch();
    let profile = directory.path().join("profile.json");
    let output = run_cli(
        &[
            "--profile-json",
            &profile.display().to_string(),
            "generate",
            &fixture("tiny-llm").display().to_string(),
            "hi",
            "--max-new-tokens",
            "1",
            "--no-stats",
            "--cpu-cores",
            "1",
        ],
        &[(
            "ONNX_GENAI_ORT_LIB",
            "/missing/nxrt-native-reference-test/onnxruntime",
        )],
    );
    assert!(
        !output.trim().is_empty(),
        "Native generation must emit text"
    );
    assert_eq!(profile_execution_provider(&profile), "native");
    let value: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&profile).expect("Native profile must be written"),
    )
    .expect("Native profile must be valid JSON");
    assert_eq!(value["decode_backend"], "native");
    assert_eq!(value["generated_tokens"], 1);
}

/// This uses the Cargo-built executable and maintained tiny ONNX fixture. It
/// verifies live Native execution/output and trace export, not a wheel launcher.
#[cfg(feature = "native-backend")]
#[test]
fn native_generate_trace_contains_execution_events_matching_the_profile_backend() {
    let directory = scratch();
    let profile = directory.path().join("profile.json");
    let trace = directory.path().join("trace.json");
    let output = run_cli(
        &[
            "--profile-json",
            &profile.display().to_string(),
            "--profile-trace",
            &trace.display().to_string(),
            "generate",
            &fixture("tiny-llm").display().to_string(),
            "hi",
            "--device",
            "cpu",
            "--greedy",
            "--raw",
            "--max-new-tokens",
            "2",
            "--no-stats",
            "--cpu-cores",
            "1",
        ],
        &[
            (
                "ONNX_GENAI_ORT_LIB",
                "/missing/nxrt-native-trace-reference-test/onnxruntime",
            ),
            ("ONNX_GENAI_TRACE_VERBOSITY", "ops"),
        ],
    );
    assert!(
        !output.trim().is_empty(),
        "Native generation must emit text"
    );
    let profile: serde_json::Value =
        serde_json::from_slice(&std::fs::read(profile).expect("Native profile must be written"))
            .expect("Native profile must parse as JSON");
    assert_eq!(profile["decode_backend"], "native");
    assert_eq!(profile["execution_provider"], "native");
    assert_eq!(profile["generated_tokens"], 2);
    assert!(
        profile["prompt_tokens"]
            .as_u64()
            .is_some_and(|count| count > 0)
    );
    let timeline: serde_json::Value = serde_json::from_slice(
        &std::fs::read(trace).expect("--profile-trace must write a timeline"),
    )
    .expect("the emitted Chrome trace must parse as JSON");
    let events = timeline["traceEvents"]
        .as_array()
        .expect("traceEvents must be an array");
    assert!(
        !events.is_empty(),
        "a real Native inference trace must contain events"
    );
    for event in events {
        assert!(
            event["name"].as_str().is_some_and(|name| !name.is_empty()),
            "{event}"
        );
        assert!(event["cat"].is_string(), "{event}");
        assert!(event["ph"].is_string(), "{event}");
        assert!(event["pid"].as_u64().is_some(), "{event}");
        assert!(event["tid"].as_u64().is_some(), "{event}");
        assert!(
            event["ts"]
                .as_f64()
                .is_some_and(|value| value.is_finite() && value >= 0.0),
            "{event}"
        );
        if event["ph"] == "X" {
            assert!(
                event["dur"]
                    .as_f64()
                    .is_some_and(|value| value.is_finite() && value >= 0.0),
                "{event}"
            );
        }
    }
    assert!(
        events
            .iter()
            .any(|event| event["name"] == "native.session_run" && event["ph"] == "X"),
        "the Native backend must actually execute a decoder quantum: {timeline}"
    );
    assert!(
        events.iter().any(|event| event["cat"] == "op"
            && event["ph"] == "X"
            && event["args"]["device"] == "cpu"
            && event["args"]["node_id"].is_u64()),
        "merged trace must contain actual CPU operator work: {timeline}"
    );
    assert!(
        !events.iter().any(|event| event["name"]
            .as_str()
            .is_some_and(|name| name.starts_with("ort."))),
        "Native profile must not conceal ORT execution: {timeline}"
    );
}
