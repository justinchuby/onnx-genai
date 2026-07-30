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
    let directory = repository_root().join("target/test-fixtures");
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("repl-sample.png");
    // Several tests want this file and cargo runs them in parallel, so it is
    // written to a per-test temporary and renamed into place. Writing the shared
    // path directly lets one test read what another is still writing, which
    // surfaces as "the image format could not be determined".
    let staging = directory.join(format!(
        "repl-sample-{}-{:?}.png",
        std::process::id(),
        std::thread::current().id()
    ));
    image::RgbImage::from_pixel(4, 4, image::Rgb([10, 200, 90]))
        .save(&staging)
        .expect("the sample PNG must be written");
    std::fs::rename(&staging, &path).expect("the sample PNG must be published atomically");
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

fn stdout_text(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr_text(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// Parse the `completed turns: N` count from a `/session` summary.
fn completed_turns(output: &str) -> usize {
    output
        .lines()
        .find_map(|line| line.trim().strip_prefix("completed turns: "))
        .and_then(|value| value.trim().parse().ok())
        .expect("the /session summary must report a completed-turns count")
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
    let output = stdout_text(&repl(&text_model(), &[], "/help\n\n"));

    for command in [
        "/help", "/reset", "/raw", "/session", "/system", "/image", "/audio",
    ] {
        assert!(output.contains(command), "{command} missing from: {output}");
    }
}

#[test]
fn session_prints_structured_counts_without_message_content() {
    let output = text(&repl(
        &text_model(),
        &[],
        "/system private instruction\n/session\n\n",
    ));

    for field in [
        "session",
        "model:",
        "execution provider:",
        "decode backend:",
        "sampling:",
        "messages: 1 (system: 1, user: 0, assistant: 0)",
        "completed turns: 0",
        "tokens: prompt=0 generated=0",
    ] {
        assert!(output.contains(field), "{field} missing from: {output}");
    }
    assert!(
        !output.contains("private instruction"),
        "session summary must not print message content: {output}"
    );
}

#[test]
fn unknown_commands_are_reported_without_ending_the_session() {
    let output = repl(&text_model(), &[], "/nope\n/help\n\n");
    let stdout = stdout_text(&output);
    let stderr = stderr_text(&output);

    assert!(stderr.contains("unknown command: /nope"), "{stderr}");
    // The session survived: /help still ran afterwards.
    assert!(stdout.contains("/system <text>"), "{stdout}");
}

#[test]
#[cfg(unix)]
fn piped_double_slash_remains_an_unknown_command() {
    let output = repl(&text_model(), &[], "//foo\n/help\n\n");
    let stdout = stdout_text(&output);
    let stderr = stderr_text(&output);

    assert!(stderr.contains("unknown command: //foo"), "{stderr}");
    assert!(
        stdout.contains("/system <text>"),
        "the session must continue after reporting the unknown command: {stdout}"
    );
}

#[test]
#[cfg(unix)]
fn piped_help_with_an_argument_still_prints_full_help() {
    let bare_help = stdout_text(&repl(&text_model(), &[], "/help\n\n"));
    let help_with_argument = stdout_text(&repl(&text_model(), &[], "/help anything\n\n"));

    assert!(
        bare_help.contains("/system <text>"),
        "bare /help must print the full help listing: {bare_help}"
    );
    assert_eq!(
        help_with_argument, bare_help,
        "/help with an argument on the plain/piped path must print the same stdout help listing as bare /help"
    );
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
    let output = repl(&vlm(), &["--raw"], script);
    let stdout = stdout_text(&output);
    let stderr = stderr_text(&output);

    // The VLM declares no audio contract, so the attachment is refused...
    assert!(stderr.contains("What:"), "{stderr}");
    assert!(stderr.contains("How:"), "{stderr}");
    // ...and the REPL is still accepting commands afterwards.
    assert!(stdout.contains("/system <text>"), "{stdout}");
}

#[test]
fn a_missing_attachment_path_is_reported_with_usage() {
    let output = repl(&vlm(), &["--raw"], "/image\n/help\n\n");
    let stdout = stdout_text(&output);
    let stderr = stderr_text(&output);

    assert!(stderr.contains("usage: /image <path>"), "{stderr}");
    assert!(stdout.contains("/system <text>"), "{stdout}");
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

// Platform gates in this file are for genuine platform dependencies only: keep
// ordinary piped-stdin REPL tests cross-platform by default.
//
// The idle-prompt interrupt test remains Unix-only because it sends terminal
// Ctrl-C with SIGINT. Windows needs GenerateConsoleCtrlEvent with a compatible
// console process group, and targeting only the child process reliably from a
// non-interactive test runner is a separate platform implementation.
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

/// The committed tiny reasoning fixture, whose chat template opens a `<think>`
/// reasoning span after the generation prompt the way a real reasoning model's
/// template does. Its vocabulary *does* contain `</think>` (id 22), but greedy
/// decoding only reaches it on the `quick`/`fox`/`dog` prompts -- where it lands
/// at position 3 immediately before a real word -- so those close the span and
/// commit a non-empty answer, while every other prompt never reaches id 22 and
/// degenerates under greedy. That asymmetry is deliberate: it is one model with
/// two reachable greedy outcomes (close-and-commit vs degenerate-and-drop), so
/// the drop assertions mean something by contrast and the positive path is
/// covered too. See `tests/fixtures/tiny-reasoning/generate_tiny_reasoning.py`.
fn reasoning_model() -> PathBuf {
    fixture("tiny-reasoning")
}

#[test]
fn a_turn_that_stops_inside_the_reasoning_says_it_has_no_answer() {
    // Under greedy, "hello" is a degenerate prompt whose attractor never reaches
    // the close token, so a two-token budget deterministically stops *inside* the
    // reasoning with no answer. (A no-flag run would sample and could reach
    // `</think>` within two tokens, which is a different -- closed-but-empty --
    // drop, so this pins greedy to stay on the unclosed path.) The REPL must say
    // so and drop the exchange, rather than record an empty assistant message
    // that teaches the model questions go unanswered.
    let output = text(&repl(
        &reasoning_model(),
        &["--greedy", "--max-new-tokens", "2"],
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

#[test]
fn a_model_without_reasoning_delimiters_reports_nothing_about_them() {
    let output = text(&repl(
        &text_model(),
        &["--max-new-tokens", "2"],
        "hello\n\n",
    ));

    // Key on the actual reasoning diagnostic the runtime would emit, not the
    // bare word "reasoning": the captured output also carries the absolute model
    // path, so a checkout or worktree directory that happens to contain
    // "reasoning" must not be mistaken for the model reporting a reasoning span.
    assert!(
        !output.contains("the model's reasoning"),
        "a plain model must not report a reasoning stop: {output}"
    );
}

/// Ten distinct in-vocabulary prompts, keeping the reasoning session in the
/// design's 8-12 turn band (docs/research/testing/00-integration-stress-design.md,
/// scenario `ci_tiny_reasoning_pressure_cpu_ort`).
///
/// These are the *degenerate* prompts: on each of them the greedy attractor
/// never reaches the renamed close token (id 22), so the reasoning span stays
/// open and every turn is dropped. The "quick"/"fox"/"dog" family is
/// deliberately excluded because those *do* close (see the positive test);
/// mixing them in would break the "every turn drops" contract these prompts
/// pin. The exact words still do not matter to the assertions, which key on
/// properties (a drop note, a resource stop, empty history) not tokens.
const REASONING_TURNS: &[&str] = &[
    "hello", "world", "the", "brown", "jumps", "over", "lazy", ".", ",", "tok16",
];

/// Build a `run` script that sends every reasoning turn then asks for `/session`.
fn reasoning_session_script() -> String {
    let mut script = REASONING_TURNS.join("\n");
    script.push_str("\n/session\n\n");
    script
}

#[test]
fn greedy_reasoning_degenerates_and_no_turn_is_ever_committed() {
    // The user-visible defect this pins ("DeepSeek repeats its thinking and
    // won't stop"): a reasoning model decoded greedily stays inside its thinking,
    // never reaches an answer, and hits the token/context budget. Here a context
    // stop stands in, on CPU, for the CUDA KV-capacity stop the full model hit.
    // The REPL must drop every such exchange (reasoning-progress invariant) and
    // never commit an empty assistant turn (non-empty-committed invariant), while
    // still admitting the next turn after each drop (admission liveness).
    let output = text(&repl(
        &reasoning_model(),
        &["--greedy", "--max-new-tokens", "16"],
        &reasoning_session_script(),
    ));

    let drops = output
        .matches("stopped inside the model's reasoning")
        .count();
    assert_eq!(
        drops,
        REASONING_TURNS.len(),
        "every greedy reasoning turn must be dropped, got {drops}: {output}"
    );
    assert!(
        output.contains("this turn is not kept"),
        "the dropped exchange must be reported to the user: {output}"
    );
    // A classified resource stop, not a silent success: the decode ran out of
    // budget while still inside the reasoning span.
    assert!(
        output.contains("finish reason: Length"),
        "the drop must be a classified resource stop: {output}"
    );
    // Non-empty committed turns: nothing empty was kept, so history stays clean
    // and every later turn was still admitted (all ten ran to produce ten drops).
    assert!(
        output.contains("messages: 0 (system: 0, user: 0, assistant: 0)"),
        "no unclosed reasoning turn may be committed to history: {output}"
    );
}

#[test]
fn greedy_reasoning_is_reproducible_across_runs() {
    // Sampling-observability invariant, greedy half: a fixed (greedy) policy is
    // stable, so the same prompt yields byte-identical output on every run.
    let run = || {
        stdout_text(&repl(
            &reasoning_model(),
            &["--greedy", "--max-new-tokens", "16"],
            "hello\n\n",
        ))
    };
    let first = run();
    let second = run();
    assert!(
        first.contains(">>>"),
        "the run must have produced a reply prompt: {first}"
    );
    assert_eq!(
        first, second,
        "greedy decoding must be deterministic across runs"
    );
}

#[test]
fn sampling_reaches_the_decode_loop_not_only_the_session_summary() {
    // Finding 1 (Gaff, then Luv): the `/session` summary resolves the sampling
    // policy in `SessionSummary::fmt`, *independently* of generation. A test
    // keyed only on that summary passes even when generation silently reverts to
    // greedy -- the exact #385/#392 "declared defaults parsed then discarded,
    // greedy forced" defect. It cannot be caught by observing the token stream
    // either: at the fixture's declared temperature 0.6 / top_k 20 the decode
    // distribution is so peaked that sampling is effectively greedy, so the
    // emitted tokens almost never witness that sampling occurred (measured
    // ~99% collapse onto the greedy stream) -- a 5-run "at least one differs"
    // assertion is a ~95% false-fail, not a regression detector.
    //
    // Instead observe the policy the *decode loop actually used*. `run_generation_turn`
    // captures the sampling policy from the `turn.options` it moves into
    // `backend.generate`, and surfaces it in the `--stats` line on stderr. This is
    // deterministic: one run each way, no token-stream sampling.
    //
    // SCOPE: this pins the policy *handed to* the decode loop (greedy/temperature/
    // top_p/top_k the resolver produced), not the engine sampler's *behaviour*
    // under it. An engine-internal regression that ignores an honoured top_k or
    // temperature is out of reach here -- and inherently so on this fixture, whose
    // near-deterministic tokens cannot witness sampling (that is exactly why the
    // token-stream approach failed). This test's job is the resolution boundary.
    //
    // tiny-reasoning declares do_sample=true, temperature=0.6, top_k=20, so with
    // no sampling flag the decode loop must run stochastically at those declared
    // values. Commenting out the per-turn `resolve_sampling_defaults` call in
    // `interactive.rs` collapses the resolved policy to the runtime greedy
    // fallback (greedy=true, temperature=1, top_k=0) and turns every assertion
    // below red -- while the `/session`-keyed tests stay green, proving those
    // never witnessed generation.
    let sampled = stderr_text(&repl(
        &reasoning_model(),
        &["--max-new-tokens", "8"],
        "/stats\nquick\n\n",
    ));
    assert!(
        sampled.contains("sampling greedy=false"),
        "the model's declared do_sample must reach the decode loop, not just the \
         /session summary; the stats line did not report a stochastic policy: {sampled}"
    );
    assert!(
        sampled.contains("temperature=0.6"),
        "the decode loop must use the model's declared temperature, not the runtime \
         default; a regression that ignores declared temperature is invisible otherwise: {sampled}"
    );
    assert!(
        sampled.contains("top_k=20"),
        "the decode loop must use the model's declared top_k, not the runtime \
         default; a regression that ignores declared top_k is invisible otherwise: {sampled}"
    );

    // The other end of the precedence chain, same instrument: an explicit
    // --greedy must force the deterministic policy onto the decode loop, and the
    // stats line must report that -- so the summary and generation cannot silently
    // diverge in either direction.
    let greedy = stderr_text(&repl(
        &reasoning_model(),
        &["--greedy", "--max-new-tokens", "8"],
        "/stats\nquick\n\n",
    ));
    assert!(
        greedy.contains("sampling greedy=true"),
        "an explicit --greedy must force the decode loop greedy, and the stats line \
         must report it: {greedy}"
    );
}

#[test]
fn the_session_summary_reports_the_same_policy_generation_used() {
    // Unification guard (Gaff Finding 1, Luv (b)): the `/session` summary and the
    // decode loop resolve the sampling policy through one shared helper
    // (`resolve_session_sampling`) reading the live backend, so the summary is
    // structurally unable to report a policy generation did not use. Batty's
    // earlier ticket to defer this rested on the (false) premise that the
    // token-stream test already removed the harm; it never worked, so the
    // divergence was live and is closed here rather than ticketed.
    //
    // Observe both surfaces in one no-flag session: the `--stats` line (stderr)
    // reports what the turn used; the `/session` summary (stdout) reports what it
    // will use. Both must show the identical resolved policy -- greedy=false at
    // the declared temperature 0.6 / top_k 20.
    let output = repl(
        &reasoning_model(),
        &["--max-new-tokens", "8"],
        "/stats\nquick\n/session\n\n",
    );
    let stats = stderr_text(&output);
    let summary = stdout_text(&output);

    assert!(
        stats.contains("sampling greedy=false")
            && stats.contains("temperature=0.6")
            && stats.contains("top_k=20"),
        "the turn's --stats line must report the resolved policy generation used: {stats}"
    );
    // The `/session` summary line resolves through the same helper, so it must
    // report the same three values.
    let summary_line = summary
        .lines()
        .find(|line| line.trim_start().starts_with("sampling:"))
        .unwrap_or_default();
    assert!(
        summary_line.contains("greedy=false")
            && summary_line.contains("temperature=0.6")
            && summary_line.contains("top_k=20"),
        "the /session summary must report the same resolved policy the turn used, not a \
         second independent resolution: {summary}"
    );
}

#[test]
fn a_model_declaring_do_sample_is_not_forced_into_greedy() {
    // #385/#392: the runtime's greedy fallback must not override a default the
    // model actually published. tiny-reasoning declares `do_sample: true`, so
    // with no sampling flag the resolved policy is stochastic at the declared
    // temperature -- precisely the regime a reasoning model ships to avoid the
    // greedy loop above. This pins how `/session` *reports* the resolved policy;
    // that generation actually *uses* it is pinned by
    // `sampling_reaches_the_decode_loop_not_only_the_session_summary`.
    let output = text(&repl(&reasoning_model(), &[], "/session\n\n"));
    assert!(
        output.contains("greedy=false"),
        "a declared do_sample=true must resolve to stochastic decoding: {output}"
    );
    assert!(
        output.contains("temperature=0.6"),
        "the model's declared temperature must be honored: {output}"
    );
}

#[test]
fn an_explicit_greedy_flag_overrides_the_models_declared_do_sample() {
    // The other end of the precedence chain: an explicit caller flag wins over
    // the model's declared default. This is the only supported way to force the
    // degenerate greedy regime onto a model that declares it samples.
    let output = text(&repl(&reasoning_model(), &["--greedy"], "/session\n\n"));
    assert!(
        output.contains("greedy=true"),
        "an explicit --greedy must win over the model's declared do_sample: {output}"
    );
}

#[test]
fn temperature_zero_forces_greedy_even_when_the_model_declares_do_sample() {
    // Resolved temperature 0 has no stochastic meaning, so it collapses to greedy
    // regardless of the declared do_sample. This pins how `/session` reports that
    // resolution; the generation-observation half is covered by
    // `sampling_reaches_the_decode_loop_not_only_the_session_summary`.
    let output = text(&repl(
        &reasoning_model(),
        &["--temperature", "0"],
        "/session\n\n",
    ));
    assert!(
        output.contains("greedy=true"),
        "temperature 0 must resolve to greedy: {output}"
    );
}

#[test]
fn a_reasoning_turn_that_closes_its_span_commits_a_non_empty_answer() {
    // The positive half of the reasoning-progress invariant, and the reason the
    // fixture has a *reachable* close: a turn whose generated text closes the
    // </think> span with visible answer text after it must be committed, not
    // dropped. Without this a regression that drops *every* turn would pass
    // against a fixture that can only ever drop -- it could not tell "correctly
    // dropped" from "commit path broken". "quick" is the deterministic greedy
    // prompt whose attractor reaches the renamed close token followed by a real
    // word, so the span closes with a non-empty answer.
    let output = text(&repl(
        &reasoning_model(),
        &["--greedy", "--max-new-tokens", "8"],
        "quick\n/session\n\n",
    ));

    // The span actually closed: the close delimiter was emitted, and the turn
    // was not reported as stopping inside the reasoning.
    assert!(
        output.contains("</think>"),
        "the reasoning span must close for this prompt: {output}"
    );
    assert!(
        !output.contains("stopped inside the model's reasoning"),
        "a closed reasoning turn must not be dropped: {output}"
    );
    // The committed answer is non-empty: there is visible text after the close
    // delimiter. Keyed on the *presence* of answer text, not its token identity,
    // so regenerating the fixture (which can move the answer word) never breaks
    // this. The runtime does not itself guard a closed-but-empty answer (see the
    // decision record), so this asserts the property directly.
    let answer = output
        .rsplit_once("</think>")
        .map(|(_, tail)| tail.lines().next().unwrap_or("").trim())
        .unwrap_or("");
    assert!(
        !answer.is_empty(),
        "the committed answer after the close must be non-empty: {output}"
    );
    // /session shows history incrementing: exactly the user turn and its
    // committed assistant answer, and one completed turn.
    assert!(
        output.contains("messages: 2 (system: 0, user: 1, assistant: 1)"),
        "the committed turn must increment history to one user + one assistant: {output}"
    );
    assert!(
        output.contains("completed turns: 1"),
        "the closed turn must count as one completed turn: {output}"
    );
}

#[test]
fn a_reasoning_turn_that_closes_on_an_empty_answer_is_dropped_not_committed() {
    // Finding 2 (Gaff): the non-empty-committed invariant was enforced only on
    // the *unclosed* path. `quick --greedy --max-new-tokens 3` stops exactly on
    // `</think>` (the third greedy token) with nothing after it, so the span is
    // closed but the answer is empty. An empty assistant turn poisons later
    // context exactly as an unclosed turn does, so it must be dropped, not
    // committed. This is the three-token boundary case for the closed-but-empty
    // guard added in `interactive.rs`.
    let output = text(&repl(
        &reasoning_model(),
        &["--greedy", "--max-new-tokens", "3"],
        "quick\n/session\n\n",
    ));

    // The span did close -- this exercises the closed-but-empty guard, not the
    // unclosed one -- so `</think>` was emitted with no answer word after it.
    assert!(
        output.contains("</think>"),
        "the three-token boundary must close the span: {output}"
    );
    assert!(
        output.contains("closed its reasoning but produced no answer"),
        "a closed-but-empty answer must be reported as a dropped turn: {output}"
    );
    assert!(
        !output.contains("stopped inside the model's reasoning"),
        "the closed-but-empty case must not be misreported as stopping inside reasoning: {output}"
    );
    // Nothing was committed: no empty assistant turn reached history.
    assert!(
        output.contains("messages: 0 (system: 0, user: 0, assistant: 0)"),
        "a closed-but-empty turn must not be committed to history: {output}"
    );
    assert!(
        output.contains("completed turns: 0"),
        "a dropped empty turn must not count as a completed turn: {output}"
    );
}

#[test]
fn the_declared_stochastic_regime_still_terminates_without_hanging() {
    // The design asks for greedy plus a stochastic path. Under the model's
    // declared do_sample the generated tokens differ run to run, so -- unlike the
    // greedy tests -- the per-turn outcome (drop vs close) is not fixed: sampling
    // can reach the close token on prompts greedy would not. What is invariant is
    // termination: the stochastic regime must be the one in force, and every turn
    // must run to a classified stop and reach the /session prompt rather than
    // hanging inside an unbounded reasoning loop. Asserting a fixed drop count
    // here would be flaky for exactly the reason the sampling matters, so this
    // pins termination and policy, not an outcome tally.
    let output = repl(
        &reasoning_model(),
        &["--max-new-tokens", "16"],
        &reasoning_session_script(),
    );
    let combined = text(&output);
    assert!(
        output.status.success(),
        "the stochastic session must terminate cleanly, not hang: {combined}"
    );
    assert!(
        combined.contains("greedy=false"),
        "the declared regime must be stochastic: {combined}"
    );
    // Prove every one of the ten turns actually ran to a *classified* outcome --
    // merely reaching `/session` does not, because a turn refused at admission
    // still leaves the session live and the summary printed. Each turn ends in
    // exactly one of: a committed answer (counted in `completed turns`), a drop
    // inside its reasoning, a drop at the empty close of its reasoning, or a
    // refusal at admission because the context was full. Their sum equalling the
    // turn count is what pins that all ten ran and none was silently swallowed.
    let completed = completed_turns(&combined);
    let reasoning_drops = combined
        .matches("stopped inside the model's reasoning")
        .count();
    let empty_close_drops = combined
        .matches("closed its reasoning but produced no answer")
        .count();
    // Match the admission-drop message specifically. The literal
    // `context window is full (` (with the trailing `(` before the token count)
    // is unique to `full_repl_context_message`; the finite-fallback warning in
    // `warn_missing_context_limit` also contains "context window is full" but is
    // followed by " without", so the `(` disambiguates the two even if that
    // warning ever fires (it does not for this fixture, which declares
    // max_sequence_length).
    let admission_drops = combined.matches("context window is full (").count();
    assert_eq!(
        completed + reasoning_drops + empty_close_drops + admission_drops,
        REASONING_TURNS.len(),
        "every turn must reach a classified outcome (commit, reasoning drop, \
         empty-close drop, or admission drop); completed={completed} \
         reasoning_drops={reasoning_drops} empty_close_drops={empty_close_drops} \
         admission_drops={admission_drops}: {combined}"
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

#[test]
fn model_reports_the_current_session_and_switches_to_another() {
    // A second fixture with a different name proves the switch took effect.
    let script = format!(
        "/model\n/model {}\n/model\n\n",
        fixture("tiny-llm-scatter").display()
    );
    let output = text(&repl(
        &fixture("tiny-llm"),
        &["--max-new-tokens", "2"],
        &script,
    ));

    assert!(
        output.contains("tiny-llm"),
        "the starting model is reported: {output}"
    );
    assert!(
        output.contains("tiny-llm-scatter"),
        "the new model must be loaded and reported: {output}"
    );
    assert!(
        output.contains("conversation cleared"),
        "history belongs to the model that held it: {output}"
    );
}

#[test]
fn a_model_that_cannot_be_loaded_leaves_the_session_running() {
    let output = text(&repl(
        &fixture("tiny-llm"),
        &["--max-new-tokens", "2"],
        "/model /nonexistent-model-path\nhello\n\n",
    ));

    assert!(
        output.contains("error:"),
        "the failure must be reported: {output}"
    );
    assert!(
        output.matches(">>>").count() >= 3,
        "the REPL must keep prompting after a failed load: {output}"
    );
}

#[test]
fn an_execution_provider_this_build_lacks_is_rejected_before_reloading() {
    // Reported as an unavailable provider, not as a failure to load the model.
    let output = text(&repl(
        &fixture("tiny-llm"),
        &["--max-new-tokens", "2"],
        "/ep nosuchprovider
hello

",
    ));

    assert!(
        output.contains("is not an execution provider this build can select"),
        "{output}"
    );
    assert!(
        output.contains("auto, cpu"),
        "the choices must be named: {output}"
    );
    assert!(
        !output.contains("could not load"),
        "it must not surface as a model-loading failure: {output}"
    );
}

#[test]
fn ep_lists_what_this_build_can_select() {
    let output = text(&repl(
        &fixture("tiny-llm"),
        &["--max-new-tokens", "2"],
        "/ep\n\n",
    ));

    assert!(output.contains("execution provider"), "{output}");
    assert!(output.contains("cpu"), "cpu is always available: {output}");
    // Which providers appear beyond cpu depends on the build and the machine —
    // CUDA on a CUDA build, Metal where its plugin is configured — so only the
    // always-present one is asserted.
}

#[test]
fn ep_and_backend_switch_and_reload() {
    let output = text(&repl(
        &fixture("tiny-llm"),
        &["--max-new-tokens", "2"],
        "/ep cpu\n/backend ort\nhello\n\n",
    ));

    assert!(output.contains("execution provider cpu"), "{output}");
    assert!(output.contains("decode backend ort"), "{output}");
    assert!(
        !output.contains("error:"),
        "cpu and ort must both load this fixture: {output}"
    );
}

#[test]
fn an_unknown_backend_is_rejected_by_name() {
    let output = text(&repl(
        &fixture("tiny-llm"),
        &["--max-new-tokens", "2"],
        "/backend quantum\n\n",
    ));

    assert!(output.contains("not a decode backend"), "{output}");
    assert!(
        output.contains("auto, ort, or native"),
        "the choices must be named: {output}"
    );
}

#[test]
fn profile_can_be_turned_on_for_a_session_that_did_not_start_with_it() {
    let output = text(&repl(
        &fixture("tiny-llm"),
        &["--max-new-tokens", "2"],
        "/profile\n/profile on\nhello\n\n",
    ));

    assert!(
        output.contains("profile report off"),
        "it starts off: {output}"
    );
    assert!(output.contains("profile report on"), "{output}");
    assert!(
        output.contains("generated tokens"),
        "the report must actually be emitted: {output}"
    );
    assert!(
        output.contains("per-stage timings need --profile at startup"),
        "the one part that cannot be enabled later must be called out: {output}"
    );
}

#[test]
fn profile_rejects_a_setting_it_does_not_offer_and_says_what_it_does() {
    let output = text(&repl(
        &fixture("tiny-llm"),
        &["--max-new-tokens", "2"],
        "/profile maybe\n\n",
    ));
    assert!(output.contains("is not a /profile setting"), "{output}");
    // Refusing is not enough; the message has to name the alternatives.
    assert!(output.contains("trace <path>"), "{output}");
    assert!(output.contains("verbosity <level>"), "{output}");
}

/// `/profile on` must produce a timeline, not only a text report.
///
/// The trace destination used to be fixed from the environment before any
/// thread started, so an interactive session could never ask for a timeline
/// after the fact — exactly when someone decides they want one.
#[test]
fn profile_on_writes_a_timeline_to_a_chosen_path() {
    // Unique per test process/thread: cargo runs these in parallel and a
    // shared path would let one test read what another is still writing.
    let directory = repository_root().join("target/test-fixtures");
    std::fs::create_dir_all(&directory).expect("fixture directory");
    let trace = directory.join(format!(
        "repl-trace-{}-{:?}.perfetto.json",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_file(&trace);
    let output = text(&repl(
        &fixture("tiny-llm"),
        &["--max-new-tokens", "2"],
        &format!("/profile on\n/profile trace {}\nhello\n\n", trace.display()),
    ));
    assert!(
        output.contains("full detail"),
        "asking to profile should start at the most detailed level: {output}"
    );
    assert!(trace.is_file(), "no timeline was written: {output}");

    let document: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&trace).expect("read trace")).expect("parse trace");
    let events = document["traceEvents"]
        .as_array()
        .expect("traceEvents array");
    assert!(!events.is_empty(), "the timeline recorded nothing");
}

/// `/profile trace <path>` on its own must write the file.
///
/// Kept separate from the test that turns the report on first: that one passed
/// while this was broken, because asking for a report was what made anything
/// get emitted at all. Naming a destination is its own request.
#[test]
fn profile_trace_alone_writes_a_timeline() {
    let directory = repository_root().join("target/test-fixtures");
    std::fs::create_dir_all(&directory).expect("fixture directory");
    let trace = directory.join(format!(
        "repl-trace-only-{}-{:?}.perfetto.json",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_file(&trace);

    let output = text(&repl(
        &fixture("tiny-llm"),
        &["--max-new-tokens", "2"],
        &format!("/profile trace {}\nhello\n\n", trace.display()),
    ));
    assert!(
        trace.is_file(),
        "naming a destination did not produce one: {output}"
    );
    let document: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&trace).expect("read trace")).expect("parse trace");
    assert!(
        !document["traceEvents"]
            .as_array()
            .expect("traceEvents array")
            .is_empty(),
        "the timeline recorded nothing"
    );
}

/// `/profile trace off` must override a destination named at startup.
#[test]
fn profile_trace_off_overrides_a_startup_destination() {
    let directory = repository_root().join("target/test-fixtures");
    std::fs::create_dir_all(&directory).expect("fixture directory");
    let trace = directory.join(format!(
        "repl-trace-off-{}-{:?}.perfetto.json",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_file(&trace);

    let output = text(&repl_with_global_flags(
        &fixture("tiny-llm"),
        &["--profile-trace", trace.to_str().expect("utf-8 path")],
        &["--max-new-tokens", "2"],
        "/profile trace off\nhello\n\n",
    ));
    assert!(
        !trace.exists(),
        "turning the timeline off still wrote the startup destination: {output}"
    );
}

#[test]
fn profile_reports_and_changes_the_detail_level() {
    let output = text(&repl(
        &fixture("tiny-llm"),
        &["--max-new-tokens", "2"],
        "/profile verbosity ops\n/profile verbosity loud\n\n",
    ));
    assert!(output.contains("timeline detail ops"), "{output}");
    assert!(output.contains("is not a detail level"), "{output}");
    for level in ["decisions", "ops", "full"] {
        assert!(output.contains(level), "{output} should list {level}");
    }
}

#[test]
fn the_provider_menu_comes_from_the_runtime_not_the_cli() {
    // The CLI kept its own list once and it drifted: macOS auto-selects the
    // Metal plugin, but `/ep metal` was refused on a machine already running on
    // it. The menu is now whatever the runtime says it can resolve.
    let listed = text(&repl(
        &fixture("tiny-llm"),
        &["--max-new-tokens", "2"],
        "/ep\n\n",
    ));
    for provider in onnx_genai::ort::selectable_execution_providers() {
        assert!(
            listed.contains(provider),
            "the runtime offers {provider}, so the menu must list it: {listed}"
        );
    }
}

#[test]
fn pages_reports_the_pool_or_says_the_model_has_none() {
    let output = text(&repl(
        &fixture("tiny-llm"),
        &["--max-new-tokens", "2"],
        "/pages\n\n",
    ));

    // Either shape is correct depending on the model's KV layout; what must not
    // happen is a bare zero that reads as "an empty pool" when there is no pool.
    assert!(
        output.contains("kv pages") || output.contains("KV is not paged"),
        "{output}"
    );
}

#[test]
fn pages_is_offered_in_help() {
    let output = text(&repl(
        &fixture("tiny-llm"),
        &["--max-new-tokens", "2"],
        "/help\n\n",
    ));
    assert!(output.contains("/pages"), "{output}");
}

#[test]
fn stream_reply_to_a_pipe_always_ends_with_a_trailing_newline() {
    // The terminating newline after each streaming reply in piped mode is a
    // byte-stable guarantee for any script or pipeline consuming `run`.
    // This test pins it so a future reviewer cannot plausibly argue the
    // unconditional newline is "redundant" when the model reply already ends
    // with one.
    //
    // In piped mode the generated tokens arrive on stdout followed by the
    // trailing newline, then the next `>>> ` prompt. A two-turn session
    // therefore contains `\n>>> ` in its stdout.
    let output = stdout_text(&repl(
        &text_model(),
        &["--raw", "--max-new-tokens", "2"],
        "hello\n\n",
    ));

    assert!(
        output.contains("\n>>> "),
        "piped streaming output must end with a trailing newline before the next prompt: {output:?}"
    );
}
