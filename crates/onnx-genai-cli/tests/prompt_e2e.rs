//! End-to-end coverage for `generate` prompt spelling.
//!
//! Parser tests cover exact argv edge cases. These tests drive the real binary
//! to prove that the long spelling reaches generation against a tiny model, and
//! that the prompt-selection errors remain actionable at the user boundary.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn tiny_model() -> PathBuf {
    repository_root()
        .join("tests/fixtures/tiny-llm")
        .canonicalize()
        .expect("the tiny text model fixture must exist")
}

fn run_cli(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_onnx-genai"))
        .args(arguments)
        .env("ONNX_GENAI_EP", "cpu")
        .output()
        .expect("the onnx-genai binary must start")
}

fn combined_output(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[test]
fn long_prompt_flag_loads_the_tiny_model_and_generates() {
    let model = tiny_model();
    let output = run_cli(&[
        "generate",
        &model.display().to_string(),
        "--prompt",
        "hello world",
        "--max-new-tokens",
        "1",
        "--no-stats",
        "--cpu-cores",
        "1",
    ]);
    let text = combined_output(&output);

    assert!(
        output.status.success(),
        "`--prompt` must get beyond clap, load the tiny fixture, and generate: {text}"
    );
}

#[test]
fn generate_prompt_selection_errors_name_both_spellings() {
    let model = tiny_model();
    let model = model.display().to_string();

    let absent = run_cli(&["generate", &model]);
    let both = run_cli(&["generate", &model, "positional", "--prompt", "flag"]);

    for (output, expected) in [
        (
            absent,
            "How: use either `generate MODEL \"your prompt\"` or `generate MODEL --prompt \"your prompt\"`.",
        ),
        (
            both,
            "How: use either `generate MODEL \"your prompt\"` or `generate MODEL --prompt \"your prompt\"`, but not both.",
        ),
    ] {
        let text = combined_output(&output);
        assert!(
            !output.status.success(),
            "the command must reject it: {text}"
        );
        assert!(
            text.contains(expected),
            "the error must name both prompt spellings: {text}"
        );
    }
}
