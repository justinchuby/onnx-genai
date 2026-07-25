//! End-to-end coverage for the CLI's multimodal input and image-output paths.
//!
//! These drive the real `onnx-genai` binary against committed tiny fixtures so
//! the whole user-facing chain is exercised: argument parsing, model loading,
//! image/audio preprocessing, image placeholder expansion, and pipeline decode.
//!
//! Fixtures (rebuild with the matching script under `scripts/`):
//! - `tests/fixtures/tiny-vlm-image-input` (`build_tiny_vlm_image_input.py`):
//!   declares `preprocessing.image` bound to `encoder.pixel_values` plus a
//!   `pipeline.vision` expansion contract mapping `<image>` (id 3) to `img` (id 4).
//! - `tests/fixtures/tiny-whisper` (`build_tiny_whisper.py`): declares an
//!   `encoder.input_features` audio input and ships a PCM16 WAV.
//! - `tests/fixtures/tiny-txt2img` (`build_tiny_txt2img.py`): a full
//!   text_encoder → denoiser → vae diffusion package.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture(name: &str) -> PathBuf {
    repository_root().join("tests/fixtures").join(name)
}

fn run(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_onnx-genai"))
        .args(arguments)
        .env("ONNX_GENAI_EP", "cpu")
        .output()
        .expect("the onnx-genai binary must run")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// A 4x4 solid-color PNG written to the target directory.
fn sample_png() -> PathBuf {
    let path = repository_root().join("target/test-fixtures/cli-sample.png");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    image::RgbImage::from_pixel(4, 4, image::Rgb([200, 30, 60]))
        .save(&path)
        .expect("the sample PNG must be written");
    path
}

#[test]
fn generate_sends_an_image_through_the_declared_vision_contract() {
    let image = sample_png();
    let output = run(&[
        "generate",
        fixture("tiny-vlm-image-input").to_str().unwrap(),
        "--image",
        image.to_str().unwrap(),
        "--prompt",
        "describe <image>",
        "--raw",
        "--max-new-tokens",
        "3",
    ]);

    assert!(
        output.status.success(),
        "image generation failed: {}",
        stderr(&output)
    );
    // The fixture's decoder always predicts the image token, which the
    // tokenizer renders as `img`. Seeing it proves the pixel tensor reached the
    // encoder and the expanded prompt reached the decoder.
    assert!(
        stdout(&output).contains("img"),
        "expected decoded text, got {:?}",
        stdout(&output)
    );
}

#[test]
fn an_image_without_a_matching_placeholder_is_rejected_with_an_actionable_error() {
    let image = sample_png();
    let output = run(&[
        "generate",
        fixture("tiny-vlm-image-input").to_str().unwrap(),
        "--image",
        image.to_str().unwrap(),
        "--prompt",
        "describe",
        "--raw",
        "--max-new-tokens",
        "2",
    ]);

    assert!(!output.status.success(), "a missing placeholder must fail");
    let message = stderr(&output);
    assert!(message.contains("What:"), "message: {message}");
    assert!(message.contains("How:"), "message: {message}");
}

#[test]
fn generate_transcribes_audio_through_the_declared_input_features_contract() {
    let model = fixture("tiny-whisper");
    let output = run(&[
        "generate",
        model.to_str().unwrap(),
        "--audio",
        model.join("tiny.wav").to_str().unwrap(),
        "--prompt",
        "",
        "--raw",
        "--max-new-tokens",
        "2",
    ]);

    assert!(
        output.status.success(),
        "audio generation failed: {}",
        stderr(&output)
    );
    assert!(
        !stdout(&output).trim().is_empty(),
        "the audio decoder must produce a transcript"
    );
}

#[test]
fn a_text_only_model_rejects_attachments_by_naming_what_it_accepts() {
    let image = sample_png();
    let output = run(&[
        "generate",
        fixture("tiny-llm").to_str().unwrap(),
        "--image",
        image.to_str().unwrap(),
        "--prompt",
        "hi",
        "--max-new-tokens",
        "2",
    ]);

    assert!(!output.status.success(), "a text model must reject images");
    let message = stderr(&output);
    assert!(
        message.contains("this model accepts text input"),
        "message: {message}"
    );
    assert!(message.contains("How:"), "message: {message}");
}

#[test]
fn generate_renders_a_prompt_to_a_png() {
    let out = repository_root().join("target/test-fixtures/cli-render.png");
    let _ = std::fs::remove_file(&out);

    let output = run(&[
        "generate",
        fixture("tiny-txt2img").to_str().unwrap(),
        "--prompt",
        "an astronaut riding a horse",
        "--negative-prompt",
        "blurry low quality",
        "--width",
        "8",
        "--height",
        "8",
        "--steps",
        "3",
        "--seed",
        "7",
        "--output-image",
        out.to_str().unwrap(),
    ]);

    assert!(
        output.status.success(),
        "image rendering failed: {}",
        stderr(&output)
    );
    assert!(
        stdout(&output).contains("8x8"),
        "stdout: {}",
        stdout(&output)
    );
    let decoded = image::open(&out).expect("the CLI must write a readable PNG");
    assert_eq!((decoded.width(), decoded.height()), (8, 8));
}
