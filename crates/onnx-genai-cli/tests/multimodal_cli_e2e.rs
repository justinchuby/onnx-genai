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

#[test]
fn encoder_decoder_fixtures_declare_ambiguous_decoder_roles() {
    for (fixture_name, expected_encoder_input) in [
        ("tiny-whisper", Some("encoder_hidden_states")),
        ("tiny-tts", None),
    ] {
        let metadata = onnx_genai_metadata::load_metadata(
            &fixture(fixture_name).join("inference_metadata.yaml"),
        )
        .expect("fixture metadata must load");
        let decoder_io = metadata
            .pipeline
            .as_ref()
            .and_then(|pipeline| pipeline.models.get("decoder"))
            .and_then(|decoder| decoder.io.as_ref())
            .expect("encoder-decoder fixture must declare decoder io");

        assert_eq!(decoder_io.token_input.as_deref(), Some("decoder_input_ids"));
        assert_eq!(
            decoder_io.position_ids_input.as_deref(),
            Some("position_ids")
        );
        assert_eq!(decoder_io.logits_output.as_deref(), Some("logits"));
        assert_eq!(
            decoder_io.encoder_hidden_states_input.as_deref(),
            expected_encoder_input
        );
        assert_eq!(decoder_io.kv_inputs.as_ref().map(Vec::len), Some(2));
        assert_eq!(decoder_io.kv_outputs.as_ref().map(Vec::len), Some(2));
    }
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

fn looks_like_compact_stats_line(line: &str) -> bool {
    let Some(inner) = line
        .trim()
        .strip_prefix("[ ")
        .and_then(|line| line.strip_suffix(" ]"))
    else {
        return false;
    };
    let fields: Vec<&str> = inner.split(" · ").collect();
    fields
        .iter()
        .any(|field| field.strip_suffix(" in").is_some_and(is_usize))
        && fields
            .iter()
            .any(|field| field.strip_suffix(" out").is_some_and(is_usize))
        && fields.iter().any(|field| field.starts_with("backend "))
        && fields
            .iter()
            .any(|field| field.strip_suffix(" tok/s").is_some_and(is_f64))
        && fields.iter().any(|field| {
            field
                .strip_prefix("ttft ")
                .and_then(|field| field.strip_suffix(" ms"))
                .is_some_and(is_f64)
        })
}

fn is_usize(text: &str) -> bool {
    text.parse::<usize>().is_ok()
}

fn is_f64(text: &str) -> bool {
    text.parse::<f64>().is_ok()
}

/// A 4x4 solid-color PNG written to the target directory.
fn sample_png() -> PathBuf {
    let directory = repository_root().join("target/test-fixtures");
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("cli-sample.png");
    // Several tests want this file and cargo runs them in parallel, so it is
    // written to a per-test temporary and renamed into place. Writing the shared
    // path directly lets one test read what another is still writing, which
    // surfaces as "the image format could not be determined".
    let staging = directory.join(format!(
        "cli-sample-{}-{:?}.png",
        std::process::id(),
        std::thread::current().id()
    ));
    image::RgbImage::from_pixel(4, 4, image::Rgb([200, 30, 60]))
        .save(&staging)
        .expect("the sample PNG must be written");
    std::fs::rename(&staging, &path).expect("the sample PNG must be published atomically");
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
fn a_prompt_with_more_placeholders_than_images_is_rejected() {
    let image = sample_png();
    // Two placeholders, one image: the caller positioned them deliberately, so
    // topping up or dropping one would silently change what the text refers to.
    let output = run(&[
        "generate",
        fixture("tiny-vlm-image-input").to_str().unwrap(),
        "--image",
        image.to_str().unwrap(),
        "--prompt",
        "compare <image> and <image>",
        "--raw",
        "--max-new-tokens",
        "2",
    ]);

    assert!(!output.status.success(), "a mismatched count must fail");
    let message = stderr(&output);
    assert!(message.contains("What:"), "message: {message}");
    assert!(message.contains("How:"), "message: {message}");
    assert!(
        message.contains("one placeholder per image"),
        "the fix must be spelled out: {message}"
    );
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
    // The wording comes from the shared admission policy the server uses too,
    // so the CLI and the HTTP API reject the same input the same way.
    assert!(
        message.contains("it accepts text input"),
        "message: {message}"
    );
    assert!(
        message.contains("single decoder graph"),
        "message: {message}"
    );
    assert!(message.contains("How:"), "message: {message}");
}

#[test]
fn piped_generate_stdout_is_byte_identical_with_or_without_default_stats() {
    let base = run(&[
        "generate",
        fixture("tiny-llm").to_str().unwrap(),
        "--prompt",
        "hello",
        "--raw",
        "--max-new-tokens",
        "4",
    ]);
    let no_stats = run(&[
        "generate",
        fixture("tiny-llm").to_str().unwrap(),
        "--prompt",
        "hello",
        "--raw",
        "--max-new-tokens",
        "4",
        "--no-stats",
    ]);

    assert!(base.status.success(), "failed: {}", stderr(&base));
    assert!(no_stats.status.success(), "failed: {}", stderr(&no_stats));
    let base_stdout = stdout(&base);
    let no_stats_stdout = stdout(&no_stats);
    assert_eq!(
        base_stdout, no_stats_stdout,
        "piped stdout must remain pure generated text"
    );
    assert!(
        !base_stdout.lines().any(looks_like_compact_stats_line),
        "stats line must not contaminate piped stdout: {base_stdout}"
    );
}

#[test]
fn piped_stream_output_always_ends_with_a_trailing_newline() {
    // `--stream` piped output has always ended with a trailing newline.
    // This pins the byte-stable guarantee: removing the unconditional
    // newline on the grounds that the model "already ended with one"
    // would silently break any script or pipeline that depends on the
    // line boundary.
    let output = run(&[
        "generate",
        fixture("tiny-llm").to_str().unwrap(),
        "--prompt",
        "hello",
        "--raw",
        "--max-new-tokens",
        "4",
        "--stream",
    ]);

    assert!(output.status.success(), "failed: {}", stderr(&output));
    assert!(
        output.stdout.last() == Some(&b'\n'),
        "piped --stream output must end with a trailing newline; got: {:?}",
        stdout(&output)
    );
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

#[test]
fn image_and_audio_cannot_be_combined_in_one_turn() {
    let image = sample_png();
    let model = fixture("tiny-whisper");
    let output = run(&[
        "generate",
        model.to_str().unwrap(),
        "--image",
        image.to_str().unwrap(),
        "--audio",
        model.join("tiny.wav").to_str().unwrap(),
        "--prompt",
        "",
        "--raw",
        "--max-new-tokens",
        "1",
    ]);

    assert!(!output.status.success(), "mixing modalities must fail");
    let message = stderr(&output);
    assert!(message.contains("What:"), "message: {message}");
    assert!(message.contains("How:"), "message: {message}");
}

#[test]
fn rendering_against_a_non_diffusion_model_explains_why() {
    let out = repository_root().join("target/test-fixtures/cli-never-written.png");
    let output = run(&[
        "generate",
        fixture("tiny-llm").to_str().unwrap(),
        "--prompt",
        "a cat",
        "--output-image",
        out.to_str().unwrap(),
    ]);

    assert!(!output.status.success(), "a text model cannot render");
    let message = stderr(&output);
    assert!(message.contains("What:"), "message: {message}");
    assert!(message.contains("How:"), "message: {message}");
}

#[test]
fn rendering_rejects_sizes_that_the_vae_cannot_produce() {
    let out = repository_root().join("target/test-fixtures/cli-bad-size.png");
    let output = run(&[
        "generate",
        fixture("tiny-txt2img").to_str().unwrap(),
        "--prompt",
        "a cat",
        "--width",
        "7",
        "--height",
        "8",
        "--output-image",
        out.to_str().unwrap(),
    ]);

    assert!(
        !output.status.success(),
        "a non-multiple-of-8 width must fail"
    );
    assert!(
        stderr(&output).contains("multiples of 8"),
        "message: {}",
        stderr(&output)
    );
}

#[test]
fn show_reports_a_models_resolved_files() {
    let output = run(&["show", fixture("tiny-llm").to_str().unwrap()]);

    assert!(output.status.success(), "show failed: {}", stderr(&output));
    let report = stdout(&output);
    assert!(report.contains("model directory:"), "{report}");
    assert!(report.contains("tokenizer:"), "{report}");
}

#[test]
fn show_accepts_a_config_file_inside_the_model_directory() {
    let output = run(&[
        "show",
        fixture("tiny-txt2img")
            .join("inference_metadata.yaml")
            .to_str()
            .unwrap(),
    ]);

    // A file argument resolves to its parent directory; a diffusion package has
    // no single decoder graph, so this must fail with a real message rather
    // than silently treating the file as a directory.
    let combined = format!("{}{}", stdout(&output), stderr(&output));
    assert!(
        combined.contains("tiny-txt2img"),
        "the resolved package must be named: {combined}"
    );
}

#[test]
fn list_enumerates_model_directories() {
    let output = run(&[
        "list",
        "--models-dir",
        repository_root().join("tests/fixtures").to_str().unwrap(),
    ]);

    assert!(output.status.success(), "list failed: {}", stderr(&output));
    let listing = stdout(&output);
    assert!(listing.contains("tiny-llm"), "{listing}");
    // tiny-txt2img is a multi-graph diffusion model (denoiser + text_encoder + vae);
    // the loader rejects it because it has multiple ONNX files and no decoder.onnx,
    // so it correctly does not appear in the listing.
    assert!(!listing.contains("tiny-txt2img"), "{listing}");
    assert!(listing.contains("tiny-llm-explicit-io"), "{listing}");
}

#[test]
fn version_reports_the_available_execution_providers() {
    let output = run(&["version"]);

    assert!(
        output.status.success(),
        "version failed: {}",
        stderr(&output)
    );
    let report = stdout(&output);
    assert!(report.contains("onnx-genai "), "{report}");
    assert!(report.contains("execution providers:"), "{report}");
}

#[test]
fn generate_synthesizes_a_prompt_to_a_wav() {
    let out = repository_root().join("target/test-fixtures/cli-speech.wav");
    let _ = std::fs::remove_file(&out);

    let output = run(&[
        "generate",
        fixture("tiny-tts").to_str().unwrap(),
        "--prompt",
        "hello there",
        "--max-new-tokens",
        "4",
        "--output-audio",
        out.to_str().unwrap(),
    ]);

    assert!(
        output.status.success(),
        "synthesis failed: {}",
        stderr(&output)
    );
    // The package's declared sample rate must be reported and written.
    assert!(stdout(&output).contains("16000 Hz"), "{}", stdout(&output));
    let wav = std::fs::read(&out).expect("the CLI must write a WAV file");
    assert_eq!(&wav[..4], b"RIFF");
    let decoded = onnx_genai::preprocess::audio::decode_wav_pcm16(&wav)
        .expect("the file must be readable PCM16 WAV");
    assert_eq!(decoded.sample_rate, 16_000);
    assert!(!decoded.samples.is_empty());
}

#[test]
fn synthesizing_with_a_non_speech_model_explains_why() {
    let out = repository_root().join("target/test-fixtures/cli-never-spoken.wav");
    let output = run(&[
        "generate",
        fixture("tiny-llm").to_str().unwrap(),
        "--prompt",
        "hello",
        "--output-audio",
        out.to_str().unwrap(),
    ]);

    assert!(!output.status.success(), "a text model cannot synthesize");
    let message = stderr(&output);
    assert!(message.contains("What:"), "message: {message}");
    assert!(message.contains("How:"), "message: {message}");
}

#[test]
fn image_and_audio_output_cannot_be_requested_together() {
    let output = run(&[
        "generate",
        fixture("tiny-tts").to_str().unwrap(),
        "--prompt",
        "hello",
        "--output-image",
        "/tmp/never.png",
        "--output-audio",
        "/tmp/never.wav",
    ]);

    assert!(!output.status.success(), "one invocation, one output");
    assert!(
        stderr(&output).contains("once per output"),
        "message: {}",
        stderr(&output)
    );
}

#[test]
fn a_single_image_needs_no_placeholder_in_the_prompt() {
    let image = sample_png();
    // No `<image>` typed: the CLI inserts the model's own placeholder, so a
    // caller never has to know its spelling.
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

    assert!(
        output.status.success(),
        "placeholder insertion failed: {}",
        stderr(&output)
    );
    assert!(stdout(&output).contains("img"), "{}", stdout(&output));
}

#[test]
fn an_explicitly_positioned_placeholder_is_honored() {
    let image = sample_png();
    let output = run(&[
        "generate",
        fixture("tiny-vlm-image-input").to_str().unwrap(),
        "--image",
        image.to_str().unwrap(),
        "--prompt",
        "describe <image> please",
        "--raw",
        "--max-new-tokens",
        "2",
    ]);

    assert!(
        output.status.success(),
        "explicit placement failed: {}",
        stderr(&output)
    );
}

#[test]
fn profile_reports_throughput_and_latency() {
    let output = run(&[
        "--profile",
        "generate",
        fixture("tiny-llm").to_str().unwrap(),
        "--prompt",
        "hello",
        "--raw",
        "--max-new-tokens",
        "4",
    ]);

    assert!(output.status.success(), "failed: {}", stderr(&output));
    let report = stderr(&output);
    for expected in [
        "time to first token",
        "decode throughput",
        "end-to-end throughput",
        "generated tokens",
        "prompt tokens",
        "model load",
    ] {
        assert!(
            report.contains(expected),
            "missing {expected} in:\n{report}"
        );
    }
    // The report is a diagnostic: it must not contaminate the generated text.
    assert!(
        !stdout(&output).contains("decode throughput"),
        "the profile must go to stderr: {}",
        stdout(&output)
    );
}

#[test]
fn profile_json_is_machine_readable() {
    let path = repository_root().join("target/test-fixtures/cli-profile.json");
    let _ = std::fs::remove_file(&path);

    let output = run(&[
        "--profile-json",
        path.to_str().unwrap(),
        "generate",
        fixture("tiny-llm").to_str().unwrap(),
        "--prompt",
        "hello",
        "--raw",
        "--max-new-tokens",
        "4",
    ]);

    assert!(output.status.success(), "failed: {}", stderr(&output));
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("the report must be written"))
            .expect("the report must be valid JSON");

    assert!(report["generated_tokens"].as_u64().unwrap() > 0);
    assert!(report["time_to_first_token_ms"].as_f64().unwrap() >= 0.0);
    assert!(report["model_load_ms"].as_f64().unwrap() > 0.0);
    assert_eq!(report["execution_provider"], "cpu");
}

#[test]
fn profile_trace_writes_a_chrome_timeline() {
    let path = repository_root().join("target/test-fixtures/cli-trace.json");
    let _ = std::fs::remove_file(&path);

    let output = run(&[
        "--profile-trace",
        path.to_str().unwrap(),
        "generate",
        fixture("tiny-llm").to_str().unwrap(),
        "--prompt",
        "hello",
        "--raw",
        "--max-new-tokens",
        "4",
    ]);

    assert!(output.status.success(), "failed: {}", stderr(&output));
    let trace: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("the trace must be written"))
            .expect("the trace must be valid JSON");

    let events = trace
        .get("traceEvents")
        .and_then(|events| events.as_array())
        .or_else(|| trace.as_array())
        .expect("a Chrome trace carries an event array");
    assert!(!events.is_empty(), "the timeline must record events");
    // Chrome Trace Event Format: every event needs a name and a phase.
    for event in events.iter().take(5) {
        assert!(event["name"].is_string(), "event: {event}");
        assert!(event["ph"].is_string(), "event: {event}");
    }
}

#[test]
fn profiling_a_render_reports_per_step_cost() {
    let out = repository_root().join("target/test-fixtures/cli-profile-render.png");
    let output = run(&[
        "--profile",
        "generate",
        fixture("tiny-txt2img").to_str().unwrap(),
        "--prompt",
        "a cat",
        "--width",
        "8",
        "--height",
        "8",
        "--steps",
        "3",
        "--output-image",
        out.to_str().unwrap(),
    ]);

    assert!(output.status.success(), "failed: {}", stderr(&output));
    let report = stderr(&output);
    assert!(report.contains("denoise steps"), "{report}");
    assert!(report.contains("per step"), "{report}");
}
