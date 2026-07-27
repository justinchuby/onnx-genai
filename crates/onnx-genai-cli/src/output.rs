use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Instant;

use onnx_genai::GenerateToken;
use onnx_genai::ort::{ChatMessage, ChatTemplate};
use onnx_genai::reasoning::{ReasoningMarkers, ReasoningStream};

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
    if !show_stats || show_profile {
        return;
    }
    profile.memory.sample_peak();
    eprintln!("{}", profile.to_stats_line());
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
                    live.set_status(timings.live_summary())?;
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
    if let Some(live) = live {
        live.finish()?;
    }
    GENERATING.store(false, Ordering::SeqCst);
    timings.finish();
    if let Some(profile) = profile {
        profile.timings = timings;
        profile.multimodal_reuse = backend.multimodal_reuse();
        if let Ok(result) = &result {
            profile.finish_reason = Some(format!("{:?}", result.finish_reason));
            profile.prefix_cache_hit = Some(result.prefix_cache_hit_len);
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
