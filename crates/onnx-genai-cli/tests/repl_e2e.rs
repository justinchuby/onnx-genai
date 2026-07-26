//! End-to-end coverage for the interactive `onnx-genai run` REPL.
//!
//! The REPL is a stdin loop, so these drive the real binary with a piped script
//! and assert on what the user would actually see. They cover the slash-command
//! surface, attachment staging and clearing, multi-turn history, and the exit
//! paths — behavior that unit tests over `parse_repl_line` alone cannot reach.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture(name: &str) -> PathBuf {
    repository_root().join("tests/fixtures").join(name)
}

/// A 4x4 solid-color PNG written to the target directory.
fn sample_png() -> PathBuf {
    let path = repository_root().join("target/test-fixtures/repl-sample.png");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    image::RgbImage::from_pixel(4, 4, image::Rgb([10, 200, 90]))
        .save(&path)
        .expect("the sample PNG must be written");
    path
}

/// Run the REPL with `script` on stdin and return its output.
///
/// The script is fed all at once; the REPL exits on the trailing empty line or
/// on EOF, so every test terminates without needing a timeout.
fn repl(model: &Path, extra_arguments: &[&str], script: &str) -> Output {
    repl_with_global_flags(model, &[], extra_arguments, script)
}

/// Run the REPL with flags that belong before the subcommand (`--profile`).
fn repl_with_global_flags(
    model: &Path,
    global_arguments: &[&str],
    extra_arguments: &[&str],
    script: &str,
) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_onnx-genai"))
        .args(global_arguments)
        .arg("run")
        .arg(model)
        .args(extra_arguments)
        .env("ONNX_GENAI_EP", "cpu")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the onnx-genai binary must start");
    child
        .stdin
        .as_mut()
        .expect("stdin is piped")
        .write_all(script.as_bytes())
        .expect("the REPL must accept the script");
    child.wait_with_output().expect("the REPL must exit")
}

fn text(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn vlm() -> PathBuf {
    fixture("tiny-vlm-image-input")
}

fn text_model() -> PathBuf {
    fixture("tiny-llm")
}

#[test]
fn the_banner_reports_the_modalities_the_model_accepts() {
    let vlm_banner = text(&repl(&vlm(), &["--raw"], "\n"));
    assert!(
        vlm_banner.contains("text + image input"),
        "banner: {vlm_banner}"
    );

    let text_banner = text(&repl(&text_model(), &[], "\n"));
    assert!(
        text_banner.contains("(text input)"),
        "banner: {text_banner}"
    );
}

#[test]
fn help_lists_every_slash_command() {
    let output = text(&repl(&text_model(), &[], "/help\n\n"));

    for command in ["/help", "/reset", "/raw", "/system", "/image", "/audio"] {
        assert!(output.contains(command), "{command} missing from: {output}");
    }
}

#[test]
fn unknown_commands_are_reported_without_ending_the_session() {
    let output = text(&repl(&text_model(), &[], "/nope\n/help\n\n"));

    assert!(output.contains("unknown command: /nope"), "{output}");
    // The session survived: /help still ran afterwards.
    assert!(output.contains("/system <text>"), "{output}");
}

#[test]
fn system_and_raw_and_reset_acknowledge_their_effect() {
    let output = text(&repl(
        &text_model(),
        &[],
        "/system Be concise.\n/raw\n/raw\n/system\n/reset\n\n",
    ));

    assert!(output.contains("system message set"), "{output}");
    assert!(output.contains("raw mode enabled"), "{output}");
    assert!(output.contains("raw mode disabled"), "{output}");
    assert!(output.contains("system message cleared"), "{output}");
    assert!(
        output.contains("conversation history and pending attachments cleared"),
        "{output}"
    );
}

#[test]
fn an_image_attachment_runs_the_turn_and_is_cleared_afterwards() {
    let image = sample_png();
    let script = format!(
        "/image {} describe <image>\ndescribe <image>\n\n",
        image.display()
    );
    let output = text(&repl(&vlm(), &["--raw", "--max-new-tokens", "2"], &script));

    // First turn: the staged image is consumed and the model answers.
    assert!(
        output.contains(&format!("(sending {})", image.display())),
        "{output}"
    );
    assert!(output.contains("img"), "{output}");
    // Second turn: the attachment was cleared, so the same prompt now fails for
    // want of an image rather than silently reusing the previous one.
    assert!(
        output.contains("the turn carried no attachment"),
        "the staged image must not leak into the next turn: {output}"
    );
}

#[test]
fn reset_drops_a_staged_attachment_before_it_is_sent() {
    let image = sample_png();
    let script = format!("/image {} \n/reset\ndescribe <image>\n\n", image.display());
    let output = text(&repl(&vlm(), &["--raw", "--max-new-tokens", "2"], &script));

    assert!(
        output.contains("the turn carried no attachment"),
        "/reset must discard the staged image: {output}"
    );
}

#[test]
fn a_rejected_attachment_keeps_the_session_alive() {
    let script = "/audio /nonexistent.wav\n/help\n\n";
    let output = text(&repl(&vlm(), &["--raw"], script));

    // The VLM declares no audio contract, so the attachment is refused...
    assert!(output.contains("What:"), "{output}");
    assert!(output.contains("How:"), "{output}");
    // ...and the REPL is still accepting commands afterwards.
    assert!(output.contains("/system <text>"), "{output}");
}

#[test]
fn a_missing_attachment_path_is_reported_with_usage() {
    let output = text(&repl(&vlm(), &["--raw"], "/image\n/help\n\n"));

    assert!(output.contains("usage: /image <path>"), "{output}");
    assert!(output.contains("/system <text>"), "{output}");
}

#[test]
fn a_nonexistent_attachment_file_names_the_path() {
    let output = text(&repl(
        &vlm(),
        &["--raw"],
        "/image ./definitely-missing.png\n\n",
    ));

    assert!(output.contains("definitely-missing.png"), "{output}");
    assert!(output.contains("How:"), "{output}");
}

#[test]
fn preloaded_attachments_apply_to_the_first_turn() {
    let image = sample_png();
    let output = text(&repl(
        &vlm(),
        &[
            "--raw",
            "--max-new-tokens",
            "2",
            "--image",
            image.to_str().unwrap(),
        ],
        "describe <image>\n\n",
    ));

    assert!(
        output.contains(&format!("(sending {})", image.display())),
        "a --image passed to `run` must be staged for the first turn: {output}"
    );
    assert!(output.contains("img"), "{output}");
}

#[test]
fn multiple_turns_run_in_one_session() {
    let output = text(&repl(
        &text_model(),
        &["--raw", "--max-new-tokens", "2"],
        "first\nsecond\n\n",
    ));

    // Three prompts are printed: two turns plus the final empty line.
    assert!(
        output.matches(">>> ").count() >= 3,
        "each turn must re-prompt: {output}"
    );
}

#[test]
fn end_of_input_exits_cleanly() {
    // No trailing empty line: the REPL must exit on EOF, not hang or fail.
    let output = repl(
        &text_model(),
        &["--raw", "--max-new-tokens", "1"],
        "hello\n",
    );

    assert!(
        output.status.success(),
        "EOF must exit cleanly: {}",
        text(&output)
    );
}

/// Send SIGINT to `pid`, the same signal a terminal Ctrl-C delivers.
#[cfg(unix)]
fn send_interrupt(pid: u32) {
    let status = Command::new("kill")
        .arg("-INT")
        .arg(pid.to_string())
        .status()
        .expect("kill must run");
    assert!(status.success(), "SIGINT delivery failed");
}

/// Block until `path` contains `needle`, so a signal is never sent before the
/// REPL has installed its handler and reached the prompt.
#[cfg(unix)]
fn wait_for(path: &Path, needle: &str) -> String {
    for _ in 0..600 {
        if let Ok(text) = std::fs::read_to_string(path)
            && text.contains(needle)
        {
            return text;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    panic!(
        "timed out waiting for {needle:?}; saw: {:?}",
        std::fs::read_to_string(path).unwrap_or_default()
    );
}

/// One Ctrl-C at an idle prompt only warns; the second exits with 130.
#[cfg(unix)]
#[test]
fn two_ctrl_c_presses_are_needed_to_exit_an_idle_prompt() {
    let log = repository_root().join("target/test-fixtures/repl-ctrlc.log");
    std::fs::create_dir_all(log.parent().unwrap()).unwrap();
    let _ = std::fs::remove_file(&log);
    let stderr = std::fs::File::create(&log).unwrap();

    let mut child = Command::new(env!("CARGO_BIN_EXE_onnx-genai"))
        .arg("run")
        .arg(text_model())
        .arg("--raw")
        .env("ONNX_GENAI_EP", "cpu")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(stderr)
        .spawn()
        .expect("the onnx-genai binary must start");

    // The banner is printed after the Ctrl-C handler is installed.
    wait_for(&log, "Ctrl-C aborts the current generation");

    send_interrupt(child.id());
    let warned = wait_for(&log, "press Ctrl-C again to exit");
    assert!(
        child.try_wait().unwrap().is_none(),
        "one press must not exit: {warned}"
    );

    send_interrupt(child.id());
    // Poll rather than `wait()`: `Child::wait` drops the piped stdin first,
    // which would close the REPL's input and race a clean EOF exit against the
    // signal we are actually measuring.
    let status = (0..600)
        .find_map(|_| {
            std::thread::sleep(std::time::Duration::from_millis(50));
            child.try_wait().expect("the child status must be readable")
        })
        .expect("the second press must exit the REPL");
    assert_eq!(
        status.code(),
        Some(130),
        "the second press must exit with 128 + SIGINT"
    );
}

/// A copy of the tiny text model whose chat template declares reasoning
/// delimiters, the way a reasoning model's template does.
#[cfg(unix)]
fn reasoning_model() -> PathBuf {
    let source = fixture("tiny-llm");
    let dir = repository_root().join("target/test-fixtures/tiny-llm-reasoning");
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(&source).unwrap().flatten() {
        let name = entry.file_name();
        if name != "tokenizer_config.json" {
            let _ = std::fs::copy(entry.path(), dir.join(&name));
        }
    }
    // The template opens the span after the generation prompt, which is how
    // reasoning templates are written: the model only ever emits the close.
    std::fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"chat_template":"{% for m in messages %}<|{{ m.role }}|>\n{{ m.content }}\n{% endfor %}{% if add_generation_prompt %}<|assistant|>\n<think>\n{% endif %}"}"#,
    )
    .unwrap();
    dir
}

#[cfg(unix)]
#[test]
fn a_turn_that_stops_inside_the_reasoning_says_it_has_no_answer() {
    // Two tokens cannot reach the closing delimiter, so the turn genuinely has
    // no answer. The REPL must say so and drop the exchange, rather than record
    // an empty assistant message that teaches the model questions go unanswered.
    let output = text(&repl(
        &reasoning_model(),
        &["--max-new-tokens", "2"],
        "hello\n\n",
    ));

    assert!(
        output.contains("stopped inside the model's reasoning"),
        "the truncated turn must be reported: {output}"
    );
    assert!(
        output.contains("--max-new-tokens"),
        "the fix must be named: {output}"
    );
    assert!(
        output.contains("this turn is not kept"),
        "the user must be told the exchange was dropped: {output}"
    );
}

#[cfg(unix)]
#[test]
fn a_model_without_reasoning_delimiters_reports_nothing_about_them() {
    let output = text(&repl(
        &text_model(),
        &["--max-new-tokens", "2"],
        "hello\n\n",
    ));

    assert!(
        !output.contains("reasoning"),
        "a plain model must not mention reasoning: {output}"
    );
}

#[test]
fn a_second_question_about_the_same_image_reuses_the_encoder() {
    // The reason multimodal reuse exists: a follow-up about the same picture
    // should not re-run the vision encoder, and `--profile` is where a user can
    // see that it did not.
    let image = sample_png();
    let path = image.to_str().unwrap();
    let script = format!("/image {path} describe it\n/image {path} and again\n\n");
    let output = repl_with_global_flags(
        &fixture("tiny-vlm-image-input"),
        &["--profile"],
        &["--max-new-tokens", "2"],
        &script,
    );
    let text = text(&output);

    assert!(
        output.status.success(),
        "the REPL must exit cleanly: {text}"
    );
    assert!(
        text.contains("encoder cache"),
        "the profile must report encoder reuse: {text}"
    );
    // The first turn runs the encoder; the second must find it memoized.
    assert!(
        text.contains("1 hit / 0 run"),
        "the follow-up turn must hit the memoized encoder output: {text}"
    );
}

#[test]
fn stats_reports_per_turn_numbers_without_the_full_profile() {
    let output = text(&repl(
        &fixture("tiny-llm"),
        &["--max-new-tokens", "3"],
        "/stats\nhello\n\n",
    ));

    assert!(output.contains("per-turn stats enabled"), "{output}");
    assert!(
        output.contains(" in · ") && output.contains(" out"),
        "the stats line must report input and output tokens: {output}"
    );
    assert!(
        output.contains("tok/s"),
        "the stats line must report throughput: {output}"
    );
    assert!(
        !output.contains("per-stage breakdown"),
        "the compact line must not drag in the full profile report: {output}"
    );
}

#[test]
fn stats_is_off_until_asked_for() {
    let output = text(&repl(
        &fixture("tiny-llm"),
        &["--max-new-tokens", "3"],
        "hello\n\n",
    ));
    assert!(
        !output.contains("tok/s"),
        "an unasked-for stats line would be noise: {output}"
    );
}
