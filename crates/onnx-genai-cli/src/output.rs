use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Instant;

use onnx_genai::ort::{ChatMessage, ChatTemplate};
use onnx_genai::reasoning::{ReasoningMarkers, ReasoningStream};
use onnx_genai::{GenerateToken, GenerationBudgetCap};

use super::interactive::{Backend, GENERATING, INTERRUPT_REQUESTED, Interrupted, TurnInput};
use super::{live_turn, profile};
use profile::RunProfile;

/// The reasoning convention a loaded model declares.
#[derive(Debug, Clone)]
pub(super) struct ReasoningConfig {
    pub(super) markers: ReasoningMarkers,
    /// True when the template writes the opening delimiter itself, so the
    /// model's output begins inside the span and never emits an opener.
    pub(super) opened_by_template: bool,
}

/// Reasoning delimiters this model declares through its chat template, if any.
pub(super) fn detect_reasoning(template: Option<&ChatTemplate>) -> Option<ReasoningConfig> {
    let template = template?;
    let source = template.source();
    let markers = ReasoningMarkers::from_chat_template(source)?;
    // The opener appearing after the assistant generation prompt means the
    // template opens the span for the model.
    let opened_by_template = source
        .rsplit_once("assistant")
        .map(|(_, tail)| tail.contains(&markers.start))
        .unwrap_or(false);
    Some(ReasoningConfig {
        markers,
        opened_by_template,
    })
}

/// Load the model's chat template unless `raw` is set. On a load failure this
/// degrades gracefully: it prints a one-line warning and returns `None`, causing
/// callers to fall back to the raw (untemplated) prompt rather than crash.
pub(super) fn load_chat_template(model_dir: &Path, raw: bool) -> Option<ChatTemplate> {
    if raw {
        return None;
    }
    match ChatTemplate::from_model_dir(model_dir) {
        Ok(template) => Some(template),
        Err(error) => {
            eprintln!(
                "warning: could not load chat template ({error}); sending the prompt untemplated"
            );
            None
        }
    }
}

/// How often the live status is refreshed while a reply streams. Fast enough to
/// look continuous, slow enough that redrawing is not part of the measurement.
const STATUS_REFRESH: std::time::Duration = std::time::Duration::from_millis(100);

/// Write one timeline holding both the engine's spans and the runtime's.
///
/// The engine records through the ORT profiler while the native runtime and its
/// execution providers record through `onnx-runtime-tracer`. Exporting only the
/// first is what left `native.session_run` an opaque block, so the two event
/// lists are concatenated here — the one place that already sees both. They
/// share a process and each thread keeps its own lane, so Perfetto nests the
/// runtime's per-operator spans inside the engine's step spans on its own.
pub(super) fn write_merged_trace(path: &Path) -> anyhow::Result<()> {
    let mut document = onnx_genai::ort::profile::trace_document();
    let runtime_events = runtime_trace_events();
    if !runtime_events.is_empty()
        && let Some(events) = document
            .get_mut("traceEvents")
            .and_then(serde_json::Value::as_array_mut)
    {
        events.extend(runtime_events);
    }
    std::fs::write(path, serde_json::to_vec(&document)?)?;
    Ok(())
}

/// The native runtime's spans as Chrome trace events, or empty when the native
/// backend is not compiled in.
fn runtime_trace_events() -> Vec<serde_json::Value> {
    #[cfg(feature = "native-backend")]
    {
        onnx_genai::engine::runtime_trace::collected_events()
            .into_iter()
            .filter_map(|event| serde_json::to_value(&event).ok())
            .collect()
    }
    #[cfg(not(feature = "native-backend"))]
    {
        Vec::new()
    }
}

/// Print the compact per-turn stats line, if the session asked for it.
///
/// Suppressed while `--profile` is on, which already prints every one of these
/// numbers and more; printing both would just repeat the turn twice.
pub(super) fn emit_stats_line(show_stats: bool, show_profile: bool, profile: &mut RunProfile) {
    if !should_emit_stats_line(show_stats, show_profile) {
        return;
    }
    profile.memory.sample_peak();
    // Stats go to stderr — probe stderr, not stdout.  Probing stdout inverts
    // the decision when stdout is redirected (`> out.txt` gives a file on
    // stdout but a terminal on stderr) or when stderr is redirected
    // (`2> log` gives a file on stderr but a terminal on stdout).
    let stats = stats_text(io::stderr().is_terminal(), profile);
    eprintln!("{stats}");
}

/// Choose the stats rendering for the given stderr terminal state.
///
/// Extracted from `emit_stats_line` so the branching logic is unit-testable
/// without a real terminal.  Pass `io::stderr().is_terminal()` at the call
/// site; pass `true`/`false` in tests.
fn stats_text(stderr_is_terminal: bool, profile: &RunProfile) -> String {
    if stderr_is_terminal {
        profile.to_stats_block()
    } else {
        profile.to_stats_line()
    }
}

fn budget_cap_notice(cap: GenerationBudgetCap) -> String {
    format!(
        "notice: scheduler capped --max-new-tokens from {} to {} because the KV byte budget cannot conservatively reserve the requested ceiling; raise --vram-limit or pass an explicit smaller --max-new-tokens to make this bound intentional",
        cap.requested_max_new_tokens, cap.admitted_max_new_tokens
    )
}

/// Whether a TTY streaming reply needs a trailing newline added by the caller.
///
/// Returns `true` when the reply is non-empty **and** its last character is
/// not `\n`.  Extracted from `run_generation_turn` so the decision is directly
/// unit-testable without a full generate pipeline or a real terminal.
///
/// The piped path always emits an unconditional newline (preserving historical
/// byte-stable behaviour for scripts); this predicate is only for the TTY path.
fn needs_trailing_newline(output: &str) -> bool {
    !output.is_empty() && !output.ends_with('\n')
}

fn should_emit_stats_line(show_stats: bool, show_profile: bool) -> bool {
    show_stats && !show_profile
}

/// Build the prompt string sent to the engine for the current turn.
///
/// With a chat `template`, the full `history` (all prior turns plus the current
/// user message) is rendered with `add_generation_prompt=true` so the model
/// continues as the assistant. Without a template (raw mode / load failure) the
/// last message's content is sent verbatim.
pub(super) fn build_turn_prompt(
    template: Option<&ChatTemplate>,
    history: &[ChatMessage],
) -> anyhow::Result<String> {
    match template {
        Some(template) => template
            .render(history, None, true)
            .map_err(|error| anyhow::anyhow!("chat template render failed: {error}")),
        None => Ok(history
            .last()
            .map(|message| message.content.clone())
            .unwrap_or_default()),
    }
}

/// Run one generation turn, streaming tokens through the terminal and returning
/// the accumulated assistant text.
///
/// The interrupt flag is reset at entry so a stale Ctrl-C cannot cancel this
/// turn, and the `GENERATING` flag is held for the duration so the Ctrl-C handler
/// soft-cancels instead of exiting. A Ctrl-C during the turn surfaces as an
/// [`Interrupted`] error (recognizable via [`is_interrupt_error`]).
pub(super) fn run_generation_turn(
    backend: &mut Backend,
    turn: TurnInput,
    stream: bool,
    profile: Option<&mut RunProfile>,
    reasoning: Option<&ReasoningConfig>,
    live: Option<&mut live_turn::LiveTurn>,
) -> anyhow::Result<String> {
    INTERRUPT_REQUESTED.store(false, Ordering::SeqCst);
    GENERATING.store(true, Ordering::SeqCst);
    backend.reset_reuse_stats();

    let prompt_tokens = turn.prompt_tokens;
    let context_limit = turn.context_limit;
    // Capture the sampling policy from `turn.options` here — the exact struct
    // moved into `backend.generate(turn, ...)` below. This is deliberately the
    // *only* capture site, and it reads the value at the point of use: there is
    // no separate resolved variable and no window between capture and use, so a
    // refactor cannot slip a re-resolution in between and leave stats reporting a
    // policy generation did not run with. The fields are `Copy`, so reading them
    // does not move `turn`. WARNING: this must read from `turn.options`, not a
    // value re-resolved for display — re-resolving is the #385/#392 defect this
    // instrument exists to catch, and it would make the sampling-policy test
    // green while pointing at the wrong thing. (It observes the policy handed to
    // the decode loop, not the engine sampler's behaviour under it.)
    let sampling_policy = profile::SamplingPolicy {
        greedy: turn.options.greedy,
        temperature: turn.options.temperature,
        top_p: turn.options.top_p,
        top_k: turn.options.top_k,
    };
    let mut output = String::new();
    let mut timings = profile::TokenTimings::default();
    timings.start();
    // Reasoning is dimmed as it streams so a reader can tell a model's thinking
    // from its answer. Only on a terminal: escape codes in a pipe would corrupt
    // the text a caller is parsing.
    let dim_reasoning = stream && reasoning.is_some() && io::stdout().is_terminal();
    let mut reasoning_stream = reasoning
        .map(|config| ReasoningStream::new(config.markers.clone(), config.opened_by_template));
    let mut dimmed = false;
    let mut live = live.filter(|live| stream && live.is_active());
    // Numbers move faster than a reader can follow, and every update costs a
    // frame, so the status is refreshed on a timer rather than per token.
    let mut last_status = Instant::now();
    let mut live_frames = 0usize;
    let mut callback = |token: GenerateToken| -> anyhow::Result<()> {
        if INTERRUPT_REQUESTED.load(Ordering::SeqCst) {
            return Err(anyhow::Error::new(Interrupted));
        }
        // Timed here, at the point the token reaches the caller, because that
        // is the latency a user actually experiences.
        timings.token();
        output.push_str(&token.text);
        if stream {
            let is_reasoning = reasoning_stream
                .as_mut()
                .map(|tracker| tracker.push(&token.text))
                .unwrap_or(false);
            if let Some(live) = live.as_deref_mut() {
                let first = live_frames == 0;
                live.push(&token.text, is_reasoning)?;
                live_frames += 1;
                // The first token draws immediately so the line is never empty
                // while a reply is on screen; after that the numbers are
                // throttled, since they move faster than they can be read and
                // every update costs a frame.
                if first || last_status.elapsed() >= STATUS_REFRESH {
                    let mut status = timings.live_summary();
                    if let (Some(prompt_tokens), Some(context_limit)) =
                        (prompt_tokens, context_limit)
                    {
                        let context = profile::ContextUsage {
                            used_tokens: prompt_tokens + timings.tokens(),
                            max_tokens: context_limit,
                        };
                        if status.is_empty() {
                            status = format!("ctx {context}");
                        } else {
                            status.push_str(&format!(" · ctx {context}"));
                        }
                    }
                    live.set_status(status)?;
                    last_status = Instant::now();
                }
            } else {
                if dim_reasoning && is_reasoning != dimmed {
                    print!("{}", if is_reasoning { "\x1b[2m" } else { "\x1b[0m" });
                    dimmed = is_reasoning;
                }
                print!("{}", token.text);
                io::stdout().flush()?;
            }
        }
        Ok(())
    };

    let result = backend.generate(turn, &mut callback);
    if dimmed {
        print!("\x1b[0m");
        let _ = io::stdout().flush();
    }
    let output_needs_trailing_newline = needs_trailing_newline(&output);
    if let Some(live) = live {
        live.finish(output_needs_trailing_newline)?;
    } else if stream {
        if io::stdout().is_terminal() {
            // TTY: conditional — skip when the reply already ended with a
            // newline. A model that terminates with `\n` would otherwise
            // produce a visible blank line before the next prompt.
            if output_needs_trailing_newline {
                println!();
            }
        } else {
            // Piped: always emit a trailing newline, unconditionally.
            //
            // Piped consumers of `--stream` output have always received a
            // terminating newline on every release. Making this conditional
            // (like the TTY path above) would silently break any script or
            // pipeline that depends on the line boundary — the extra newline
            // is invisible to the consumer, but its absence is a byte-stable
            // regression. The TTY path is conditional because a double blank
            // line is a visible defect there; here it is not.
            println!();
        }
    }
    GENERATING.store(false, Ordering::SeqCst);
    timings.finish();
    let budget_cap = result.as_ref().ok().and_then(|result| result.budget_cap);
    if let Some(cap) = budget_cap {
        eprintln!("{}", budget_cap_notice(cap));
    }
    if let Some(profile) = profile {
        profile.timings = timings;
        profile.multimodal_reuse = backend.multimodal_reuse();
        profile.sampling_policy = Some(sampling_policy);
        if let Ok(result) = &result {
            profile.finish_reason = Some(format!("{:?}", result.finish_reason));
            profile.prefix_cache_hit = Some(result.prefix_cache_hit_len);
            if let Some(cap) = budget_cap {
                profile.budget_cap = Some(profile::BudgetCap {
                    requested_max_new_tokens: cap.requested_max_new_tokens,
                    admitted_max_new_tokens: cap.admitted_max_new_tokens,
                });
            }
            if let (Some(prompt_tokens), Some(context_limit)) = (prompt_tokens, context_limit) {
                profile.context = Some(profile::ContextUsage {
                    used_tokens: prompt_tokens + result.token_ids.len(),
                    max_tokens: context_limit,
                });
            }
        }
    }
    result?;
    Ok(output)
}

pub(super) fn display_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_budget_cap_notice_names_values_and_levers() {
        let notice = budget_cap_notice(GenerationBudgetCap {
            requested_max_new_tokens: 3584,
            admitted_max_new_tokens: 128,
            requested_bytes: 4096,
            admitted_bytes: 640,
            available_bytes: 640,
        });

        assert!(notice.contains("scheduler capped --max-new-tokens from 3584 to 128"));
        assert!(notice.contains("KV byte budget"));
        assert!(notice.contains("--vram-limit"));
        assert!(notice.contains("explicit smaller --max-new-tokens"));
    }

    #[test]
    fn profile_text_report_suppresses_the_compact_stats_line() {
        assert!(should_emit_stats_line(true, false));
        assert!(!should_emit_stats_line(true, true));
        assert!(!should_emit_stats_line(false, false));
    }

    /// `needs_trailing_newline` is the predicate that determines whether the
    /// TTY streaming path adds a `\n` after the reply.  It must return `true`
    /// exactly when the output would leave the cursor in the middle of a line —
    /// that is, when the output is non-empty **and** does not already end with
    /// a newline.  Every case is testable in isolation because the predicate
    /// is a pure function with no I/O dependency.
    ///
    /// The PTY integration test (`pty_tty_e2e.rs`) confirms that this predicate
    /// is wired to a real terminal, but it cannot distinguish the conditional
    /// from an unconditional `println!()` for the current tiny-llm fixture
    /// (which never emits a trailing `\n`).  These unit tests close that gap.
    #[test]
    fn tty_trailing_newline_predicate_covers_all_cases() {
        // Normal reply with no trailing newline: cursor is mid-line → needs one.
        assert!(needs_trailing_newline("hello world"));

        // Reply that already ends with \n: cursor is at a new line → no extra.
        // This is the case the conditional prevents a visible blank line for.
        assert!(!needs_trailing_newline("hello world\n"));

        // Empty output (e.g. --max-new-tokens 0): nothing to terminate.
        assert!(!needs_trailing_newline(""));

        // Multi-line reply ending with \n — still no extra newline needed.
        assert!(!needs_trailing_newline("line one\nline two\n"));

        // Reply ending with \n followed by trailing spaces: the last character
        // is not \n, so the cursor is mid-line and a newline is required.
        assert!(needs_trailing_newline("hello\n "));

        // Newline is not the last character even if one appears inside.
        assert!(needs_trailing_newline("line one\nno newline at end"));
    }

    /// The stats format is determined by stderr's terminal state, not stdout's.
    ///
    /// `stats_text` takes only `stderr_is_terminal`; stdout's state is
    /// deliberately absent from the signature.  The two cases that motivated
    /// the fix:
    ///
    ///   A) `> out.txt` in a terminal: stdout=file, stderr=terminal → block
    ///   B) `2> stats.log` in a terminal: stdout=terminal, stderr=file → line
    ///
    /// A profile with both headline data (`prompt_tokens`) and resource data
    /// (`context`) is used so the block form produces two `[ … ]` lines joined
    /// by `\n`, making it structurally distinct from the single-line form.
    #[test]
    fn stats_format_follows_stderr_not_stdout() {
        use crate::profile::{ContextUsage, RunProfile};

        let profile = RunProfile {
            prompt_tokens: Some(10),
            context: Some(ContextUsage {
                used_tokens: 10,
                max_tokens: 4096,
            }),
            ..Default::default()
        };

        // stderr=terminal → two-line block (headline \n resources)
        let block = stats_text(true, &profile);
        assert!(
            block.contains('\n'),
            "stderr=terminal should produce two-line block; got: {block:?}"
        );

        // stderr=plain → single line (no embedded newline)
        let line = stats_text(false, &profile);
        assert!(
            !line.contains('\n'),
            "stderr=plain should produce single line; got: {line:?}"
        );

        // Case A: stdout redirected (not terminal), stderr on terminal → block
        // (the old stdout probe would have chosen line here)
        let case_a = stats_text(/* stderr */ true, &profile);
        assert!(
            case_a.contains('\n'),
            "case A: stderr=terminal must give block"
        );

        // Case B: stdout on terminal, stderr redirected (not terminal) → line
        // (the old stdout probe would have chosen block here)
        let case_b = stats_text(/* stderr */ false, &profile);
        assert!(
            !case_b.contains('\n'),
            "case B: stderr=plain must give line"
        );
    }
}
