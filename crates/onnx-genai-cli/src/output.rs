use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Instant;

use onnx_genai::ort::{ChatMessage, ChatTemplate};
use onnx_genai::reasoning::{ReasoningChunk, ReasoningMarkers, ReasoningStream};
use onnx_genai::{GenerateToken, GenerationBudgetCap};

use super::interactive::{Backend, GENERATING, INTERRUPT_REQUESTED, Interrupted, TurnInput};
use super::{live_turn, profile};
use profile::RunProfile;

#[derive(Debug, Clone)]
pub(super) struct ResponseConfig {
    initial_header: String,
    content_header: String,
    reasoning_header: String,
    start_token: String,
    message_token: String,
    close_tokens: Vec<String>,
    start_token_id: Option<u32>,
    message_token_id: Option<u32>,
    close_token_ids: Vec<(u32, String)>,
}

pub(super) fn load_response_config(model_dir: &Path, raw: bool) -> Option<ResponseConfig> {
    if raw {
        return None;
    }
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(model_dir.join("tokenizer_config.json")).ok()?)
            .ok()?;
    let template = value.get("response_template")?;
    let start_anchor = template.get("start_anchor")?.as_str()?;
    let (start_token, initial_header) = split_leading_special_token(start_anchor)?;
    let fields = template.get("fields")?;
    let content_open = fields.get("content")?.get("open_pattern")?.as_str()?;
    let reasoning_open = fields
        .get("reasoning_content")?
        .get("open_pattern")?
        .as_str()?;
    let (content_header, message_token) = split_open_pattern(content_open)?;
    let (reasoning_header, reasoning_message_token) = split_open_pattern(reasoning_open)?;
    if reasoning_message_token != message_token {
        return None;
    }
    let mut close_tokens = Vec::new();
    for field_name in ["content", "reasoning_content"] {
        let close = fields.get(field_name)?.get("close")?;
        match close {
            serde_json::Value::String(token) => push_unique(&mut close_tokens, token),
            serde_json::Value::Array(tokens) => {
                for token in tokens {
                    push_unique(&mut close_tokens, token.as_str()?);
                }
            }
            _ => return None,
        }
    }
    Some(ResponseConfig {
        initial_header,
        content_header,
        reasoning_header,
        start_token,
        message_token,
        close_tokens,
        start_token_id: None,
        message_token_id: None,
        close_token_ids: Vec::new(),
    })
}

fn split_leading_special_token(value: &str) -> Option<(String, String)> {
    let end = value.find("|>")? + 2;
    Some((value[..end].to_string(), value[end..].to_string()))
}

fn split_open_pattern(pattern: &str) -> Option<(String, String)> {
    let literal = pattern.replace("\\|", "|");
    let marker_start = literal.rfind("<|")?;
    let marker = &literal[marker_start..];
    marker
        .ends_with("|>")
        .then(|| (literal[..marker_start].to_string(), marker.to_string()))
}

fn push_unique(values: &mut Vec<String>, value: &str) {
    if !values.iter().any(|existing| existing == value) {
        values.push(value.to_string());
    }
}

impl ResponseConfig {
    pub(super) fn bind_token_ids(&mut self, backend: &Backend) {
        self.start_token_id = backend.single_token_id(&self.start_token);
        self.message_token_id = backend.single_token_id(&self.message_token);
        self.close_token_ids = self
            .close_tokens
            .iter()
            .filter_map(|token| backend.single_token_id(token).map(|id| (id, token.clone())))
            .collect();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResponseState {
    Header,
    Content,
    Reasoning,
    Other,
}

struct ResponseStream<'a> {
    config: &'a ResponseConfig,
    state: ResponseState,
    header: String,
}

struct ResponseOutput {
    text: String,
    is_reasoning: bool,
    include_in_reply: bool,
}

impl<'a> ResponseStream<'a> {
    fn new(config: &'a ResponseConfig) -> Self {
        Self {
            config,
            state: ResponseState::Header,
            header: config.initial_header.clone(),
        }
    }

    fn push(&mut self, token: &GenerateToken) -> ResponseOutput {
        let empty = || ResponseOutput {
            text: String::new(),
            is_reasoning: false,
            include_in_reply: false,
        };
        if self.config.start_token_id == Some(token.token_id) {
            self.state = ResponseState::Header;
            self.header.clear();
            return ResponseOutput {
                text: self.config.start_token.clone(),
                is_reasoning: false,
                include_in_reply: false,
            };
        }
        if let Some((_, close_token)) = self
            .config
            .close_token_ids
            .iter()
            .find(|(id, _)| *id == token.token_id)
        {
            let was_reasoning = self.state == ResponseState::Reasoning;
            self.state = ResponseState::Other;
            return ResponseOutput {
                text: format!("{close_token}\n"),
                is_reasoning: was_reasoning,
                include_in_reply: false,
            };
        }
        if self.config.message_token_id == Some(token.token_id)
            && self.state == ResponseState::Header
        {
            self.state = if self.header.ends_with(&self.config.content_header) {
                ResponseState::Content
            } else if self.header.ends_with(&self.config.reasoning_header) {
                ResponseState::Reasoning
            } else {
                ResponseState::Other
            };
            return ResponseOutput {
                text: format!("{}\n", self.config.message_token),
                is_reasoning: self.state == ResponseState::Reasoning,
                include_in_reply: false,
            };
        }
        match self.state {
            ResponseState::Header => {
                self.header.push_str(&token.text);
                ResponseOutput {
                    text: token.text.clone(),
                    is_reasoning: false,
                    include_in_reply: false,
                }
            }
            ResponseState::Content => ResponseOutput {
                text: token.text.clone(),
                is_reasoning: false,
                include_in_reply: true,
            },
            ResponseState::Reasoning => ResponseOutput {
                text: token.text.clone(),
                is_reasoning: true,
                include_in_reply: false,
            },
            ResponseState::Other => empty(),
        }
    }
}

/// The reasoning convention a loaded model declares.
#[derive(Debug, Clone)]
pub(super) struct ReasoningConfig {
    pub(super) markers: ReasoningMarkers,
    /// True when the template writes the opening delimiter itself, so the
    /// model's output begins inside the span and never emits an opener.
    pub(super) opened_by_template: bool,
    marker_tokens: Vec<(u32, String)>,
}

/// Reasoning delimiters this model declares through its chat template, if any.
pub(super) fn detect_reasoning(template: Option<&ChatTemplate>) -> Option<ReasoningConfig> {
    let template = template?;
    let source = template.source();
    let markers = ReasoningMarkers::from_chat_template(source)?;
    let opened_by_template = template_opens_reasoning_for_generation(template, &markers);
    Some(ReasoningConfig {
        markers,
        opened_by_template,
        marker_tokens: Vec::new(),
    })
}

impl ReasoningConfig {
    pub(super) fn set_marker_token_ids(&mut self, start_id: Option<u32>, end_id: Option<u32>) {
        self.marker_tokens.clear();
        if let Some(id) = start_id {
            self.marker_tokens.push((id, self.markers.start.clone()));
        }
        if let Some(id) = end_id {
            self.marker_tokens.push((id, self.markers.end.clone()));
        }
    }

    fn marker_text_for_token(&self, token_id: u32) -> Option<&str> {
        self.marker_tokens
            .iter()
            .find_map(|(id, text)| (*id == token_id).then_some(text.as_str()))
    }
}

fn template_opens_reasoning_for_generation(
    template: &ChatTemplate,
    markers: &ReasoningMarkers,
) -> bool {
    let probe = [ChatMessage::user("__onnx_genai_reasoning_probe__")];
    let Ok(rendered) = template.render(&probe, None, true) else {
        return false;
    };
    let Some(last_start) = rendered.rfind(markers.start.as_str()) else {
        return false;
    };
    rendered
        .rfind(markers.end.as_str())
        .is_none_or(|last_end| last_start > last_end)
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

fn emit_reasoning_segment(
    segment: &ReasoningChunk,
    dim_reasoning: bool,
    dimmed: &mut bool,
    live: Option<&mut live_turn::LiveTurn>,
) -> anyhow::Result<()> {
    if segment.text.is_empty() {
        return Ok(());
    }
    if let Some(live) = live {
        live.push(&segment.text, segment.is_reasoning)?;
    } else {
        if dim_reasoning && segment.is_reasoning != *dimmed {
            print!(
                "{}",
                if segment.is_reasoning {
                    "\x1b[2m"
                } else {
                    "\x1b[0m"
                }
            );
            *dimmed = segment.is_reasoning;
        }
        print!("{}", segment.text);
        io::stdout().flush()?;
    }
    Ok(())
}

fn visible_token_text(token: &GenerateToken, reasoning: Option<&ReasoningConfig>) -> String {
    if !token.text.is_empty() {
        return token.text.clone();
    }
    reasoning
        .and_then(|config| config.marker_text_for_token(token.token_id))
        .unwrap_or_default()
        .to_string()
}

fn live_frame_due(draws: usize, last_draw: Instant, now: Instant) -> bool {
    draws == 0 || now.duration_since(last_draw) >= STATUS_REFRESH
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
    response: Option<&ResponseConfig>,
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
    let dim_reasoning =
        stream && (reasoning.is_some() || response.is_some()) && io::stdout().is_terminal();
    let mut reasoning_stream = reasoning
        .map(|config| ReasoningStream::new(config.markers.clone(), config.opened_by_template));
    let mut response_stream = response.map(ResponseStream::new);
    let mut dimmed = false;
    let mut live = live.filter(|live| stream && live.is_active());
    // Numbers move faster than a reader can follow, and every update costs a
    // frame, so the status is refreshed on a timer rather than per token.
    let mut last_live_draw = Instant::now();
    let mut live_draws = 0usize;
    let mut callback = |token: GenerateToken| -> anyhow::Result<()> {
        if INTERRUPT_REQUESTED.load(Ordering::SeqCst) {
            return Err(anyhow::Error::new(Interrupted));
        }
        // Timed here, at the point the token reaches the caller, because that
        // is the latency a user actually experiences.
        timings.token();
        let response_output = response_stream.as_mut().map(|tracker| tracker.push(&token));
        let token_text = if let Some(response_output) = response_output.as_ref() {
            response_output.text.clone()
        } else {
            visible_token_text(&token, reasoning)
        };
        if response_output
            .as_ref()
            .is_none_or(|response_output| response_output.include_in_reply)
        {
            output.push_str(&token_text);
        }
        if stream {
            let segments = if let Some(response_output) = response_output {
                vec![ReasoningChunk {
                    text: response_output.text,
                    is_reasoning: response_output.is_reasoning,
                }]
            } else if let Some(tracker) = reasoning_stream.as_mut() {
                tracker.push_segments(&token_text)
            } else {
                vec![ReasoningChunk {
                    text: token_text,
                    is_reasoning: false,
                }]
            };
            for segment in segments {
                let using_live = live.is_some() && !segment.text.is_empty();
                emit_reasoning_segment(&segment, dim_reasoning, &mut dimmed, live.as_deref_mut())?;
                let now = Instant::now();
                let draw_live_frame = using_live && live_frame_due(live_draws, last_live_draw, now);
                if draw_live_frame {
                    live_draws += 1;
                    last_live_draw = now;
                    // The first token draws immediately so the line is never empty
                    // while a reply is on screen; after that the numbers are
                    // throttled, since they move faster than they can be read and
                    // every update costs a frame.
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
                    if let Some(live) = live.as_deref_mut() {
                        live.set_status(status)?;
                        live.draw()?;
                    }
                }
            }
        }
        Ok(())
    };

    let result = backend.generate(turn, &mut callback);
    if stream && let Some(tracker) = reasoning_stream.as_mut() {
        for segment in tracker.finish() {
            emit_reasoning_segment(&segment, dim_reasoning, &mut dimmed, live.as_deref_mut())?;
        }
    }
    if let Some(live) = live.as_deref_mut() {
        live.draw()?;
    }
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
    use std::fs;

    #[test]
    fn response_stream_formats_atem_channels_and_keeps_only_answer_for_history() {
        let config = ResponseConfig {
            initial_header: "assistant".to_string(),
            content_header: "to=user".to_string(),
            reasoning_header: "to=self".to_string(),
            start_token: "<|start|>".to_string(),
            message_token: "<|message|>".to_string(),
            close_tokens: vec!["<|eom|>".to_string(), "<|eot|>".to_string()],
            start_token_id: Some(22),
            message_token_id: Some(23),
            close_token_ids: vec![(7, "<|eom|>".to_string()), (8, "<|eot|>".to_string())],
        };
        let tokens = [
            (10, " to=self"),
            (23, ""),
            (11, "thinking"),
            (7, ""),
            (22, ""),
            (12, "assistant to=user"),
            (23, ""),
            (13, "answer"),
            (8, ""),
        ];
        let mut stream = ResponseStream::new(&config);
        let outputs = tokens
            .into_iter()
            .map(|(token_id, text)| {
                stream.push(&GenerateToken {
                    token_id,
                    text: text.to_string(),
                    finish_reason: None,
                })
            })
            .collect::<Vec<_>>();
        let displayed = outputs
            .iter()
            .map(|output| output.text.as_str())
            .collect::<String>();
        let reply = outputs
            .iter()
            .filter(|output| output.include_in_reply)
            .map(|output| output.text.as_str())
            .collect::<String>();

        assert_eq!(
            displayed,
            " to=self<|message|>\nthinking<|eom|>\n\
             <|start|>assistant to=user<|message|>\nanswer<|eot|>\n"
        );
        assert!(outputs[2].is_reasoning);
        assert_eq!(reply, "answer");
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::current_dir().unwrap().join(format!(
            "output-test-{}-{}",
            std::process::id(),
            name
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

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

    #[test]
    fn empty_reasoning_marker_token_text_is_restored_for_display() {
        let mut config = ReasoningConfig {
            markers: ReasoningMarkers::new("<think>", "</think>"),
            opened_by_template: false,
            marker_tokens: Vec::new(),
        };
        config.set_marker_token_ids(Some(151667), Some(151668));
        let token = GenerateToken {
            token_id: 151667,
            text: String::new(),
            finish_reason: None,
        };

        assert_eq!(visible_token_text(&token, Some(&config)), "<think>");

        let mut stream = ReasoningStream::new(config.markers.clone(), config.opened_by_template);
        let chunks = stream.push_segments(&visible_token_text(&token, Some(&config)));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "<think>");
        assert!(chunks[0].is_reasoning);
    }

    #[test]
    fn unrelated_empty_special_token_stays_hidden() {
        let mut config = ReasoningConfig {
            markers: ReasoningMarkers::new("<think>", "</think>"),
            opened_by_template: false,
            marker_tokens: Vec::new(),
        };
        config.set_marker_token_ids(Some(151667), Some(151668));
        let eos = GenerateToken {
            token_id: 151645,
            text: String::new(),
            finish_reason: None,
        };

        assert_eq!(visible_token_text(&eos, Some(&config)), "");
    }

    #[test]
    fn live_frames_are_coalesced_over_a_token_burst() {
        let start = Instant::now();
        let mut last_draw = start;
        let mut draws = 0usize;

        for millisecond in 0..250 {
            let now = start + std::time::Duration::from_millis(millisecond);
            if live_frame_due(draws, last_draw, now) {
                draws += 1;
                last_draw = now;
            }
        }

        assert_eq!(
            draws, 3,
            "250 token callbacks at 1 ms spacing should draw at t=0, 100, and 200 ms, not once per token"
        );
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

    #[test]
    fn qwen3_template_does_not_open_reasoning_for_current_generation() {
        let dir = temp_dir("qwen3-reasoning-template");
        fs::write(
            dir.join("chat_template.jinja"),
            r#"{%- for message in messages %}
{%- if message.role == "user" %}
{{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}
{%- elif message.role == "assistant" %}
{{- '<|im_start|>assistant\n' }}
{%- if '</think>' in message.content %}
{{- '<think>\n' + message.content.split('</think>')[0].split('<think>')[-1] + '\n</think>\n\n' }}
{%- endif %}
{{- message.content.split('</think>')[-1].lstrip('\n') + '<|im_end|>\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{%- if enable_thinking is defined and enable_thinking is false %}
{{- '<think>\n\n</think>\n\n' }}
{%- endif %}
{%- endif %}"#,
        )
        .unwrap();

        let template = ChatTemplate::from_model_dir(&dir).unwrap();
        let old_source_tail_heuristic = template
            .source()
            .rsplit_once("assistant")
            .map(|(_, tail)| tail.contains("<think>"))
            .unwrap_or(false);
        let config = detect_reasoning(Some(&template)).expect("template declares think markers");
        let rendered = template
            .render(&[ChatMessage::user("what's the capital?")], None, true)
            .unwrap();
        let split = config
            .markers
            .split("<think>because</think>Olympia", config.opened_by_template);
        let mut stream = ReasoningStream::new(config.markers.clone(), config.opened_by_template);
        let trace = stream
            .push_segments("<think>because</think>Olympia")
            .into_iter()
            .map(|segment| (segment.text.to_string(), segment.is_reasoning))
            .collect::<Vec<_>>();

        assert!(
            old_source_tail_heuristic,
            "the old source-tail heuristic must reproduce the false positive"
        );
        assert!(
            !rendered.contains("<think>"),
            "the rendered current generation prompt must not open thinking: {rendered:?}"
        );
        assert!(
            !config.opened_by_template,
            "qwen3 opens prior assistant turns and an enable_thinking=false branch, not the current default generation turn"
        );
        assert!(split.complete);
        assert_eq!(split.answer, "Olympia");
        assert_eq!(
            trace,
            vec![
                ("<think>".to_string(), true),
                ("because".to_string(), true),
                ("</think>".to_string(), true),
                ("Olympia".to_string(), false),
            ]
        );

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn template_opening_current_generation_is_still_detected() {
        let dir = temp_dir("template-opens-reasoning");
        fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"chat_template":"{% for m in messages %}<|{{ m.role }}|>\n{{ m.content }}\n{% endfor %}{% if add_generation_prompt %}<|assistant|>\n<think>\n{% endif %}"}"#,
        )
        .unwrap();

        let template = ChatTemplate::from_model_dir(&dir).unwrap();
        let config = detect_reasoning(Some(&template)).expect("template declares think markers");

        assert!(config.opened_by_template);

        fs::remove_dir_all(dir).unwrap();
    }
}
