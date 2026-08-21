use std::borrow::Cow;
use std::fmt;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Context as _;
use nu_ansi_term::{Color as AnsiColor, Style as AnsiStyle};
use onnx_genai::engine::{
    EngineDecodeBackend, PipelineEngine, PipelineGenerateRequest, is_missing_required_input,
};
use onnx_genai::metadata::GenerationDefaults;
use onnx_genai::ort::profile::TraceVerbosity;
use onnx_genai::ort::{ChatMessage, ChatRole, SessionOptions, Tokenizer, ep_selection};
use onnx_genai::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest, GenerateResult,
    GenerateTokenCallback, SamplingOverrides,
};
use onnx_genai_server::multimodal::{self, MultimodalInput, MultimodalSpecs};
use reedline::{
    ColumnarMenu, Completer, DefaultHinter, Emacs, FileBackedHistory, Highlighter, KeyCode,
    KeyModifiers, MenuBuilder, Prompt, PromptEditMode, PromptHistorySearch,
    PromptHistorySearchStatus, Reedline, ReedlineEvent, ReedlineMenu, Signal, Span, StyledText,
    Suggestion, default_emacs_keybindings,
};

use super::commands::{
    ProfileSetting, ReplCommand, ReplLine, available_execution_providers, complete_repl_line,
    parse_decode_backend, parse_profile_setting, parse_repl_line, reload, render_command_help,
    render_repl_help, set_trace_recording,
};
use super::output::{
    bind_response_tokens, build_turn_prompt, detect_reasoning, display_paths, emit_stats_line,
    load_chat_template, load_response_config, run_generation_turn,
};
use super::{EngineArgs, ProfileArgs, RunArgs, decode_backend_name, resolve_model_dir};
use super::{live_turn, pages, profile};
use profile::RunProfile;

/// Process exit code for termination via SIGINT (Ctrl-C), matching the POSIX
/// convention of `128 + SIGINT`.
pub(super) const EXIT_INTERRUPTED: i32 = 130;

/// Set while a generation is running so the Ctrl-C handler can distinguish an
/// interrupt during generation (soft-cancel the current turn) from an interrupt
/// at an idle prompt (arm the exit).
pub(super) static GENERATING: AtomicBool = AtomicBool::new(false);

/// Set by the Ctrl-C handler when a generation should be aborted. The streaming
/// callback polls this and returns [`Interrupted`] to unwind out of the engine.
pub(super) static INTERRUPT_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Set once a Ctrl-C has already been observed, so the next one exits the
/// process. Cleared whenever the user submits a new REPL line, which proves
/// they meant to keep working rather than quit.
pub(super) static EXIT_ARMED: AtomicBool = AtomicBool::new(false);

/// Guards one-time installation of the Ctrl-C handler.
static CTRLC_HANDLER: Once = Once::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ReplInputMode {
    Plain,
    Tty,
}

pub(super) fn repl_input_mode(stdin_is_terminal: bool, stdout_is_terminal: bool) -> ReplInputMode {
    if stdin_is_terminal && stdout_is_terminal {
        ReplInputMode::Tty
    } else {
        ReplInputMode::Plain
    }
}

pub(super) fn initial_repl_show_stats(mode: ReplInputMode, no_stats: bool) -> bool {
    matches!(mode, ReplInputMode::Tty) && !no_stats
}

struct ReplPrompt;

impl Prompt for ReplPrompt {
    fn render_prompt_left(&self) -> Cow<'_, str> {
        Cow::Borrowed("")
    }

    fn render_prompt_right(&self) -> Cow<'_, str> {
        Cow::Borrowed("")
    }

    fn render_prompt_indicator(&self, _edit_mode: PromptEditMode) -> Cow<'_, str> {
        Cow::Borrowed(">>> ")
    }

    fn render_prompt_multiline_indicator(&self) -> Cow<'_, str> {
        Cow::Borrowed("... ")
    }

    fn render_prompt_history_search_indicator(
        &self,
        history_search: PromptHistorySearch,
    ) -> Cow<'_, str> {
        let status = match history_search.status {
            PromptHistorySearchStatus::Passing => "",
            PromptHistorySearchStatus::Failing => "failing ",
        };
        Cow::Owned(format!(
            "({status}reverse-search: {}) ",
            history_search.term
        ))
    }
}

struct ReplInputHighlighter;

impl Highlighter for ReplInputHighlighter {
    fn highlight(&self, line: &str, _cursor: usize) -> StyledText {
        let mut styled = StyledText::new();
        styled.push((
            AnsiStyle::new().bold().fg(AnsiColor::Cyan),
            line.to_string(),
        ));
        styled
    }
}

#[derive(Default)]
struct SlashCompleter;

impl Completer for SlashCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> Vec<Suggestion> {
        complete_repl_line(line, pos)
            .into_iter()
            .map(|completion| Suggestion {
                value: completion.replacement,
                display_override: Some(completion.display),
                description: completion.description,
                span: Span::new(completion.start, completion.end),
                append_whitespace: completion.append_space,
                ..Suggestion::default()
            })
            .collect()
    }
}

enum ReplReader {
    Plain,
    Tty {
        editor: Box<Reedline>,
        prompt: ReplPrompt,
    },
}

enum ReplRead {
    Line(String),
    Continue,
    Eof,
}

impl ReplReader {
    fn new(mode: ReplInputMode) -> Self {
        match mode {
            ReplInputMode::Plain => Self::Plain,
            ReplInputMode::Tty => Self::Tty {
                editor: Box::new(build_reedline_editor()),
                prompt: ReplPrompt,
            },
        }
    }

    fn read_line(&mut self, stdin: &io::Stdin) -> anyhow::Result<ReplRead> {
        match self {
            Self::Plain => {
                print!(">>> ");
                io::stdout().flush()?;

                let mut line = String::new();
                if stdin.lock().read_line(&mut line)? == 0 {
                    eprintln!();
                    return Ok(ReplRead::Eof);
                }
                Ok(ReplRead::Line(
                    line.trim_end_matches(['\n', '\r']).to_string(),
                ))
            }
            Self::Tty { editor, prompt } => match editor.read_line(prompt) {
                Ok(Signal::Success(line)) => Ok(ReplRead::Line(line)),
                Ok(Signal::CtrlD) => {
                    eprintln!();
                    Ok(ReplRead::Eof)
                }
                Ok(Signal::CtrlC) => {
                    match interrupt_action(false, false, EXIT_ARMED.load(Ordering::SeqCst)) {
                        InterruptAction::WarnThenExit => {
                            EXIT_ARMED.store(true, Ordering::SeqCst);
                            eprintln!("\n^C  (press Ctrl-C again to exit)");
                            Ok(ReplRead::Continue)
                        }
                        InterruptAction::Exit => std::process::exit(EXIT_INTERRUPTED),
                        InterruptAction::CancelGeneration => {
                            unreachable!("idle prompt is not generating")
                        }
                    }
                }
                Ok(Signal::ExternalBreak(line) | Signal::HostCommand(line)) => {
                    Ok(ReplRead::Line(line))
                }
                Ok(_) => Ok(ReplRead::Continue),
                Err(error) => Err(anyhow::anyhow!(error)),
            },
        }
    }
}

fn build_reedline_editor() -> Reedline {
    let mut keybindings = default_emacs_keybindings();
    keybindings.add_binding(
        KeyModifiers::NONE,
        KeyCode::Tab,
        ReedlineEvent::UntilFound(vec![
            ReedlineEvent::Menu("completion_menu".to_string()),
            ReedlineEvent::MenuNext,
        ]),
    );
    let edit_mode = Box::new(Emacs::new(keybindings));
    let completion_menu = Box::new(ColumnarMenu::default().with_name("completion_menu"));
    let history = Box::new(
        FileBackedHistory::with_file(1000, repl_history_path())
            .unwrap_or_else(|_| FileBackedHistory::default()),
    );

    Reedline::create()
        .with_history(history)
        .with_completer(Box::new(SlashCompleter))
        .with_menu(ReedlineMenu::EngineCompleter(completion_menu))
        .with_edit_mode(edit_mode)
        .with_highlighter(Box::new(ReplInputHighlighter))
        .with_hinter(Box::new(DefaultHinter::default()))
}

fn repl_history_path() -> PathBuf {
    if let Some(path) = std::env::var_os("ONNX_GENAI_REPL_HISTORY") {
        return PathBuf::from(path);
    }
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".onnx-genai")
        .join("repl-history.txt")
}

/// Marker error returned by the streaming callback when a Ctrl-C interrupt has
/// been requested. It propagates out of `generate_with_callback` as an
/// [`anyhow::Error`] and is recognized with [`is_interrupt_error`] so the REPL
/// can distinguish a user cancel from a genuine generation failure.
#[derive(Debug)]
pub(super) struct Interrupted;

impl fmt::Display for Interrupted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("generation interrupted")
    }
}

impl std::error::Error for Interrupted {}

/// Returns true when `error` was produced by a Ctrl-C interrupt (i.e. carries an
/// [`Interrupted`] marker), as opposed to a real generation failure.
pub(super) fn is_interrupt_error(error: &anyhow::Error) -> bool {
    error.downcast_ref::<Interrupted>().is_some()
}

/// What a Ctrl-C press should do, given the session's current state.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum InterruptAction {
    /// Abort the running generation and arm the exit.
    CancelGeneration,
    /// Warn that one more press exits, and arm the exit.
    WarnThenExit,
    /// A press already warned or cancelled: leave now.
    Exit,
}

/// Decide what a Ctrl-C means. One press stops what is happening; two in a row
/// exit the process.
///
/// `generating` is whether a turn is running, `already_cancelled` whether the
/// current turn was already asked to stop, and `exit_armed` whether an earlier
/// press has already offered the exit.
pub(super) fn interrupt_action(
    generating: bool,
    already_cancelled: bool,
    exit_armed: bool,
) -> InterruptAction {
    match (generating, already_cancelled, exit_armed) {
        (true, true, _) => InterruptAction::Exit,
        (true, false, _) => InterruptAction::CancelGeneration,
        (false, _, true) => InterruptAction::Exit,
        (false, _, false) => InterruptAction::WarnThenExit,
    }
}

/// Install the process-wide Ctrl-C handler exactly once.
///
/// One Ctrl-C stops what is happening; two in a row exit the process:
/// - during a generation, the first press soft-cancels the current turn (the
///   streaming callback observes the flag and aborts) and arms the exit, so a
///   second press while the turn is still unwinding exits immediately;
/// - at an idle prompt, the first press prints a hint and arms the exit, and
///   the second exits cleanly with code 130.
///
/// Submitting a new REPL line disarms the exit, so a Ctrl-C much later in the
/// session still needs two presses.
pub(super) fn install_ctrlc_handler() {
    CTRLC_HANDLER.call_once(|| {
        let result = ctrlc::set_handler(|| {
            let generating = GENERATING.load(Ordering::SeqCst);
            let already_cancelled = INTERRUPT_REQUESTED.load(Ordering::SeqCst);
            let exit_armed = EXIT_ARMED.load(Ordering::SeqCst);
            match interrupt_action(generating, already_cancelled, exit_armed) {
                InterruptAction::CancelGeneration => {
                    INTERRUPT_REQUESTED.store(true, Ordering::SeqCst);
                    EXIT_ARMED.store(true, Ordering::SeqCst);
                }
                InterruptAction::WarnThenExit => {
                    EXIT_ARMED.store(true, Ordering::SeqCst);
                    eprintln!("\n^C  (press Ctrl-C again to exit)");
                }
                InterruptAction::Exit => {
                    // This runs on the signal-handler thread and terminates the
                    // process without unwinding the generating thread, so the
                    // `FlushGuard` on that stack never drops. Flush the buffered
                    // diagnostics here so a double Ctrl-C during a turn does not
                    // silently discard the whole turn's logs.
                    let _ = crate::flush_deferred_tracing();
                    std::process::exit(EXIT_INTERRUPTED)
                }
            }
        });
        if let Err(error) = result {
            eprintln!("warning: could not install Ctrl-C handler: {error}");
        }
    });
}
/// One turn's input: the rendered prompt plus any attachments staged for it.
#[derive(Debug, Default)]
pub(super) struct TurnInput {
    pub(super) prompt: String,
    pub(super) images: Vec<PathBuf>,
    pub(super) audio: Vec<PathBuf>,
    pub(super) options: GenerateOptions,
    pub(super) prompt_tokens: Option<usize>,
    pub(super) context_limit: Option<usize>,
}

/// The loaded model, which is either a single decoder graph or a declared
/// multi-component pipeline. Only pipeline packages can accept image or audio
/// input, and only when their metadata declares the corresponding contract.
pub(super) enum Backend {
    Text(Box<Engine>),
    Pipeline(Box<PipelineBackend>),
}

/// A loaded pipeline package plus the contracts needed to feed it.
pub(super) struct PipelineBackend {
    engine: PipelineEngine,
    tokenizer: Tokenizer,
    multimodal: MultimodalSpecs,
    generation_defaults: Option<GenerationDefaults>,
}

/// Everything that decides how a model is loaded, so an interactive session can
/// change one part and rebuild.
///
/// The execution provider and decode backend are properties of a *loaded*
/// session, not of a request: an ONNX session is created against its providers
/// and cannot be moved between them. Changing either therefore reloads the
/// model, which is why they live here together with the directory.
#[derive(Debug, Clone)]
pub(super) struct SessionSettings {
    model_dir: PathBuf,
    /// Execution provider name, or `None` to keep whatever the environment and
    /// platform defaults select.
    execution_provider: Option<String>,
    decode_backend: EngineDecodeBackend,
    native_device: Option<onnx_genai::engine::native_decode_device::NativeDecodeDevice>,
    limits: onnx_genai::engine::ResourceLimits,
}

impl SessionSettings {
    pub(super) fn new(model_dir: PathBuf, engine: &EngineArgs) -> Self {
        let config = engine.to_config();
        Self {
            model_dir,
            execution_provider: None,
            decode_backend: config.decode_backend,
            native_device: config.native_device,
            limits: config.limits,
        }
    }

    pub(super) fn to_config(&self) -> EngineConfig {
        EngineConfig {
            decode_backend: self.decode_backend,
            native_device: self.native_device.clone(),
            limits: self.limits.clone(),
            ..EngineConfig::default()
        }
    }

    /// Session options for the chosen provider, or the environment's default
    /// when none was chosen.
    pub(super) fn to_session_options(&self) -> SessionOptions {
        match &self.execution_provider {
            Some(name) => SessionOptions::with_execution_provider(ep_selection(name.clone())),
            None => SessionOptions::default(),
        }
    }

    pub(super) fn backend_name(&self) -> &'static str {
        decode_backend_name(self.decode_backend)
    }
}

/// Cumulative, privacy-preserving usage for the conversation currently in the REPL.
#[derive(Debug, Default)]
pub(super) struct SessionUsage {
    pub(super) prompt_tokens: usize,
    pub(super) generated_tokens: usize,
    pub(super) completed_turns: usize,
}

impl SessionUsage {
    fn record(&mut self, prompt_tokens: Option<usize>, generated_tokens: usize) {
        self.prompt_tokens += prompt_tokens.unwrap_or_default();
        self.generated_tokens += generated_tokens;
        self.completed_turns += 1;
    }
}

/// Human-readable state for an interactive session.
///
/// Message text is deliberately not included: `/session` is useful for sharing
/// diagnostics without echoing conversation content.
pub(super) struct SessionSummary<'a> {
    pub(super) settings: &'a SessionSettings,
    pub(super) execution_provider: String,
    pub(super) resolved_decode_backend: EngineDecodeBackend,
    /// The sampling policy resolved for the current backend — the *same* value a
    /// generated turn uses, produced by [`resolve_session_sampling`]. Held
    /// resolved (not as base options + defaults + overrides) so the summary
    /// cannot resolve it a second, independent way and silently disagree with
    /// what generation does (#385/#392).
    pub(super) sampling: GenerateOptions,
    pub(super) history: &'a [ChatMessage],
    pub(super) usage: &'a SessionUsage,
}

/// Resolve the sampling policy for the live backend.
///
/// The single resolution site shared by the `/session` summary and every
/// generated turn, so the two cannot disagree about greedy/temperature/top_p/
/// top_k. It reads the backend's declared defaults and the session's explicit
/// overrides on demand, so there is nothing to cache and nothing to go stale
/// across a `/reload`, `/ep`, or `/backend`: the next call reads whichever
/// backend is live. `max_new_tokens`/`max_context` are left untouched (context
/// sizing happens separately, per turn).
pub(super) fn resolve_session_sampling(
    base: &GenerateOptions,
    backend: &Backend,
    overrides: &SamplingOverrides,
) -> GenerateOptions {
    let mut resolved = base.clone();
    resolved.resolve_sampling_defaults(backend.generation_defaults(), overrides);
    resolved
}

impl fmt::Display for SessionSummary<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let options = &self.sampling;
        let system_messages = self
            .history
            .iter()
            .filter(|message| matches!(message.role, ChatRole::System))
            .count();
        let user_messages = self
            .history
            .iter()
            .filter(|message| matches!(message.role, ChatRole::User))
            .count();
        let assistant_messages = self
            .history
            .iter()
            .filter(|message| matches!(message.role, ChatRole::Assistant))
            .count();

        writeln!(formatter, "session")?;
        writeln!(formatter, "  model: {}", self.settings.model_dir.display())?;
        writeln!(
            formatter,
            "  execution provider: {}",
            self.execution_provider
        )?;
        writeln!(
            formatter,
            "  decode backend: {}",
            decode_backend_name(self.resolved_decode_backend)
        )?;
        if self.settings.decode_backend != self.resolved_decode_backend {
            writeln!(
                formatter,
                "  requested backend: {}",
                self.settings.backend_name()
            )?;
        }
        writeln!(
            formatter,
            "  sampling: max_new_tokens={} max_context={} temperature={} top_p={} top_k={} greedy={}",
            options.max_new_tokens,
            options
                .max_context
                .map(|value| value.to_string())
                .unwrap_or_else(|| "auto".to_string()),
            options.temperature,
            options.top_p,
            options.top_k,
            options.greedy
        )?;
        writeln!(
            formatter,
            "  messages: {} (system: {system_messages}, user: {user_messages}, assistant: {assistant_messages})",
            self.history.len()
        )?;
        writeln!(
            formatter,
            "  completed turns: {}",
            self.usage.completed_turns
        )?;
        write!(
            formatter,
            "  tokens: prompt={} generated={}",
            self.usage.prompt_tokens, self.usage.generated_tokens
        )
    }
}

impl Backend {
    /// Load the model described by `settings`.
    pub(super) fn open(settings: &SessionSettings) -> anyhow::Result<Self> {
        Self::load_with_options(
            &settings.model_dir,
            settings.to_config(),
            settings.to_session_options(),
        )
    }

    /// Load `model_dir`, preferring its declared pipeline when it has one.
    pub(super) fn load(model_dir: &Path, config: EngineConfig) -> anyhow::Result<Self> {
        Self::load_with_options(model_dir, config, SessionOptions::default())
    }

    pub(super) fn load_with_options(
        model_dir: &Path,
        config: EngineConfig,
        session_options: SessionOptions,
    ) -> anyhow::Result<Self> {
        match multimodal::load(model_dir)? {
            Some(setup) => {
                let tokenizer = Tokenizer::from_file(&setup.tokenizer_path).map_err(|error| {
                    anyhow::anyhow!(
                        "What: the pipeline's prompt tokenizer could not be loaded from {}. \
                         Why: {error}. \
                         How: verify the package ships a valid tokenizer.json.",
                        setup.tokenizer_path.display()
                    )
                })?;
                let engine = PipelineEngine::from_dir_with_session_options(
                    model_dir,
                    config,
                    session_options,
                )?;
                Ok(Self::Pipeline(Box::new(PipelineBackend {
                    engine,
                    tokenizer,
                    multimodal: setup.multimodal,
                    generation_defaults: setup.generation_defaults,
                })))
            }
            None => Ok(Self::Text(Box::new(Engine::from_dir_with_session_options(
                model_dir,
                config,
                session_options,
            )?))),
        }
    }

    /// Declared input contracts, or `None` for a single decoder graph.
    pub(super) fn multimodal(&self) -> Option<&MultimodalSpecs> {
        match self {
            Self::Text(_) => None,
            Self::Pipeline(pipeline) => Some(&pipeline.multimodal),
        }
    }

    /// Human-readable summary of the modalities this model accepts.
    pub(super) fn accepted_modalities(&self) -> String {
        self.multimodal()
            .map_or_else(|| "text".to_string(), MultimodalSpecs::accepted_modalities)
    }

    pub(super) fn supports_images(&self) -> bool {
        self.multimodal()
            .is_some_and(|multimodal| multimodal.vision.is_some())
    }

    pub(super) fn supports_audio(&self) -> bool {
        self.multimodal()
            .is_some_and(|multimodal| multimodal.audio.is_some())
    }

    /// Run one turn, streaming tokens through `callback`.
    /// Clear the pipeline reuse counters so a profile covers only the next turn.
    pub(super) fn reset_reuse_stats(&self) {}

    /// What a multimodal pipeline avoided recomputing, or `None` for a single
    /// decoder graph, which has no encoder or attachments to reuse.
    pub(super) fn multimodal_reuse(&self) -> Option<profile::MultimodalReuse> {
        None
    }

    /// What the KV page pool holds right now, when the backend pages its KV.
    pub(super) fn page_usage(&self) -> Option<onnx_genai::kv::PageUsage> {
        match self {
            Self::Text(engine) => Some(engine.page_usage()),
            Self::Pipeline(_) => None,
        }
    }

    /// Cumulative KV page counters, when the backend keeps a page pool.
    pub(super) fn page_stats(&self) -> Option<onnx_genai::kv::PageStats> {
        match self {
            Self::Text(engine) => Some(engine.page_stats()),
            Self::Pipeline(_) => None,
        }
    }

    /// Concrete decoder backend selected for the loaded model.
    pub(super) fn decode_backend(&self) -> EngineDecodeBackend {
        match self {
            Self::Text(engine) => engine.decode_backend(),
            Self::Pipeline(pipeline) => pipeline.engine.decode_backend(),
        }
    }

    /// Execution-provider placement reported by the loaded model, not by the
    /// requested settings.
    pub(super) fn execution_provider_status(&self) -> String {
        match self {
            Self::Text(engine) => engine.execution_provider_status(),
            Self::Pipeline(pipeline) => pipeline.engine.execution_provider_status(),
        }
    }

    /// KV-cache accounting from the engine's resource governor.
    ///
    /// Only a single-model engine runs a governor; a pipeline reports nothing
    /// rather than a zero that would read as "no KV cache".
    pub(super) fn kv_usage(&self) -> Option<profile::MemoryUsage> {
        match self {
            Self::Text(engine) => {
                let snapshot = engine.resource_snapshot();
                let budget = snapshot.derived_budget;
                let breakdown = snapshot.breakdown;
                #[cfg(feature = "native-backend")]
                let weight_placement =
                    engine
                        .weight_placement_report()
                        .map(|report| profile::WeightPlacementMemory {
                            coordinated_weight_budget_bytes: report.coordinated_weight_budget_bytes,
                            effective_budget_bytes: report.effective_budget_bytes,
                            device_bytes: report.device_bytes,
                            host_bytes: report.host_bytes,
                            explanation: report.explanation.clone(),
                        });
                #[cfg(not(feature = "native-backend"))]
                let weight_placement = None;
                Some(profile::MemoryUsage {
                    kv_budget_bytes: Some(budget.kv_bytes),
                    kv_max_tokens: Some(budget.max_total_tokens),
                    host_ram_used_bytes: Some(snapshot.host_ram.used),
                    device_used_bytes: Some(snapshot.vram.used),
                    device_limit_bytes: snapshot.resolved_limits.vram_bytes,
                    device_oversubscribed_bytes: Some(engine.device_oversubscribed_bytes()),
                    peak_resident_bytes: None,
                    composition: Some(profile::DeviceComposition {
                        model_weights_bytes: breakdown.model_weights_bytes,
                        activations_bytes: breakdown.activations_bytes,
                        runtime_overhead_bytes: breakdown.ort_overhead_bytes,
                        kv_bytes: budget.kv_bytes,
                        kv_pages: budget.total_pages,
                        kv_page_bytes: budget.kv_bytes.checked_div(budget.total_pages).unwrap_or(0),
                    }),
                    activation_plan: engine.activation_memory_plan_stats().map(|stats| {
                        profile::ActivationPlanMemory {
                            complete: stats.complete,
                            peak_bytes: stats.peak_bytes,
                            naive_bytes: stats.naive_bytes,
                            savings_ratio: stats.savings_ratio,
                            unknown_sizes: stats.unknown_sizes,
                        }
                    }),
                    weight_placement,
                    memory_strategy_plan: Some(engine.memory_strategy_plan().clone()),
                    vmm_arena: engine.vmm_arena_stats().map(|stats| profile::VmmArena {
                        commits: stats.commits,
                        releases: stats.releases,
                        committed_bytes: stats.committed_bytes,
                        reserved_bytes: stats.reserved_bytes,
                        peak_committed_bytes: stats.peak_committed_bytes,
                        allocations: stats.allocations,
                        ref_underflows: stats.ref_underflows,
                        byte_underflows: stats.byte_underflows,
                        unaccounted_committed_bytes: stats.unaccounted_committed_bytes,
                    }),
                })
            }
            Self::Pipeline(pipeline) => Some(profile::MemoryUsage {
                memory_strategy_plan: Some(pipeline.engine.memory_strategy_plan().clone()),
                ..profile::MemoryUsage::default()
            }),
        }
    }

    /// Number of tokens the prompt occupies, when the backend can tell.
    pub(super) fn prompt_tokens(&self, prompt: &str) -> Option<usize> {
        match self {
            Self::Text(engine) => engine.tokenize(prompt).ok().map(|ids| ids.len()),
            Self::Pipeline(pipeline) => pipeline.tokenizer.encode(prompt).ok().map(|ids| ids.len()),
        }
    }

    /// Token id for an exact one-token marker string, when the loaded tokenizer
    /// represents it as one token.
    pub(super) fn single_token_id(&self, token: &str) -> Option<u32> {
        let ids = match self {
            Self::Text(engine) => engine.tokenize(token).ok()?,
            Self::Pipeline(pipeline) => pipeline
                .tokenizer
                .token_id(token)
                .map(|id| vec![id])
                .or_else(|| pipeline.tokenizer.encode(token).ok())?,
        };
        match ids.as_slice() {
            [id] => Some(*id),
            _ => None,
        }
    }

    pub(super) fn bind_reasoning_marker_tokens(
        &self,
        reasoning: &mut Option<super::output::ReasoningConfig>,
    ) {
        if let Some(config) = reasoning {
            let start_id = self.single_token_id(&config.markers.start);
            let end_id = self.single_token_id(&config.markers.end);
            config.set_marker_token_ids(start_id, end_id);
        }
    }

    pub(super) fn effective_max_context(&self, options: &GenerateOptions) -> Option<usize> {
        match self {
            Self::Text(engine) => engine.effective_max_context(options),
            Self::Pipeline(pipeline) => pipeline.engine.effective_max_context(options),
        }
    }

    /// The package's declared generation defaults, when it declared any.
    ///
    /// A package states its sampling regime (a reasoning model shipping
    /// `do_sample: true`, or a legacy `search` block imported into
    /// `generation.defaults`) as part of what the model *is*. Discarding it
    /// silently reinterprets every such package as greedy, which is not a
    /// neutral default: a model tuned to sample can degenerate into repetition
    /// under argmax. Explicit CLI flags still win — this only supplies the
    /// values the caller left unstated.
    pub(super) fn generation_defaults(&self) -> Option<&GenerationDefaults> {
        match self {
            Self::Text(engine) => engine
                .metadata()
                .generation
                .as_ref()
                .and_then(|generation| generation.defaults.as_ref()),
            Self::Pipeline(pipeline) => pipeline.generation_defaults.as_ref(),
        }
    }

    pub(super) fn generate(
        &mut self,
        turn: TurnInput,
        callback: &mut GenerateTokenCallback<'_>,
    ) -> anyhow::Result<GenerateResult> {
        multimodal::admit_attachments(
            self.multimodal(),
            "the loaded model",
            turn.images.len(),
            turn.audio.len(),
        )?;
        match self {
            Self::Text(engine) => {
                let request = GenerateRequest {
                    prompt: GeneratePrompt::Text(turn.prompt),
                    options: turn.options,
                };
                engine.generate_with_callback(request, Some(callback))
            }
            Self::Pipeline(pipeline) => {
                let attachments = turn.images.len() + turn.audio.len();
                let required = pipeline.multimodal.sole_modality();
                let request =
                    build_pipeline_request(&pipeline.tokenizer, &pipeline.multimodal, turn)?;
                pipeline
                    .engine
                    .generate_with_callback(request, Some(callback))
                    .map_err(|error| match required {
                        // A multimodal package can require its non-text input;
                        // say so rather than leaving a bare "missing input".
                        Some(modality)
                            if attachments == 0 && is_missing_required_input(&error) =>
                        {
                            error.context(format!(
                                "the turn carried no attachment, but this model declares {modality} input. \
                                 How: attach one with `/{modality} <path>` in the REPL, or `--{modality} <path>` on the command line."
                            ))
                        }
                        _ => error,
                    })
            }
        }
    }
}

pub(super) fn apply_context_sized_max_new_tokens(
    options: &mut GenerateOptions,
    max_new_tokens_was_explicit: bool,
    prompt_tokens: usize,
    effective_max_context: Option<usize>,
) -> bool {
    if max_new_tokens_was_explicit {
        return false;
    }
    match effective_max_context {
        Some(limit) => {
            // Leave the context-window check, not max_new_tokens, as the natural
            // stop so the engine reports FinishReason::Length at the boundary.
            options.max_new_tokens = limit.saturating_sub(prompt_tokens).saturating_add(1);
            false
        }
        None => {
            options.max_new_tokens = super::CLI_FALLBACK_MAX_NEW_TOKENS;
            true
        }
    }
}

pub(super) fn warn_missing_context_limit(max_new_tokens: usize) {
    eprintln!(
        "warning: model context length could not be inferred from inference metadata or the decode path; using finite fallback --max-new-tokens {max_new_tokens}. Configure --max-context, or declare model.max_sequence_length in inference metadata, to generate until the context window is full without risking an ORT out-of-bounds decode."
    );
}

pub(super) fn context_exhaustion_error(prompt_tokens: usize, limit: usize) -> anyhow::Error {
    anyhow::anyhow!(
        "What: the prompt already fills the context window ({prompt_tokens}/{limit} tokens). \
         Why: there is no room to generate even one token, so returning an empty answer would be misleading. \
         How: shorten the prompt or use a model/package with a larger context window."
    )
}

pub(super) fn full_repl_context_message(prompt_tokens: usize, limit: usize) -> String {
    format!(
        "context window is full ({prompt_tokens}/{limit} tokens); no answer was generated and this turn was not kept. Use /reset to clear the conversation history, shorten your prompt, or use a model/package with a larger context window."
    )
}

pub(super) fn context_window_is_full(
    prompt_tokens: usize,
    effective_max_context: Option<usize>,
) -> Option<usize> {
    effective_max_context.filter(|&limit| prompt_tokens >= limit)
}

pub(super) fn drop_exhausted_repl_turn(
    history: &mut Vec<ChatMessage>,
    prompt_tokens: usize,
    effective_max_context: Option<usize>,
) -> Option<String> {
    context_window_is_full(prompt_tokens, effective_max_context).map(|limit| {
        history.pop();
        full_repl_context_message(prompt_tokens, limit)
    })
}

fn reasoning_incomplete_note(
    span_closed: bool,
    finish_reason: Option<&str>,
    turn_max_new_tokens: usize,
    max_new_tokens_was_explicit: bool,
) -> String {
    let finish_reason = finish_reason.unwrap_or("unknown");
    let next_step = if max_new_tokens_was_explicit {
        format!(
            "If you want longer reasoning, try --max-new-tokens {}.",
            turn_max_new_tokens.saturating_mul(2)
        )
    } else {
        "If the context window is exhausted, use /reset to clear conversation history or shorten the prompt."
            .to_string()
    };
    // Two shapes reach here, and the diagnostic must name which so the user is
    // not told the decode "stopped inside" its reasoning when the span actually
    // closed. Both drop the turn for the same reason: no answer to keep.
    let cause = if span_closed {
        "generation closed its reasoning but produced no answer before stopping"
    } else {
        "generation stopped inside the model's reasoning"
    };
    format!(
        "note: {cause} (finish reason: {finish_reason}). No answer was produced, so this turn is not kept. {next_step}"
    )
}

/// Turn a prompt plus its attachments into a pipeline generation request.///
/// Audio replaces the prompt entirely: the transcription decoder is seeded with
/// the model's own transcription token sequence because the spoken audio, not
/// the typed text, carries the content. Images keep the prompt and expand each
/// placeholder token into the declared image-token run.
pub(super) fn build_pipeline_request(
    tokenizer: &Tokenizer,
    multimodal: &MultimodalSpecs,
    turn: TurnInput,
) -> anyhow::Result<PipelineGenerateRequest> {
    let TurnInput {
        prompt,
        images,
        audio,
        options,
        prompt_tokens: _,
        context_limit: _,
    } = turn;

    if let Some(path) = audio.first() {
        let spec = multimodal
            .audio
            .as_ref()
            .expect("audio support checked before building the request");
        let input = MultimodalInput::from_wav(spec, &read_attachment(path, "audio")?)
            .with_context(|| {
                format!(
                    "What: the audio file {} could not be preprocessed. \
                     Why: it is not usable as this model's declared audio input. \
                     How: provide a PCM16 WAV file.",
                    path.display()
                )
            })?;
        // Audio replaces the prompt: the transcription decoder is seeded with
        // the model's own token sequence because the clip, not the typed text,
        // carries the content.
        let token_ids = multimodal::audio_decoder_prompt(tokenizer, None)?;
        return input.bind(PipelineGenerateRequest::new(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(token_ids),
            options,
        }));
    }

    let mut token_ids = tokenizer.encode(&prompt).map_err(|error| {
        anyhow::anyhow!(
            "What: the prompt could not be tokenized. \
             Why: {error}. \
             How: verify the package's tokenizer.json matches its decoder."
        )
    })?;
    let input = match multimodal.vision.as_ref().filter(|_| !images.is_empty()) {
        None => None,
        Some(spec) => {
            let mut encoded = Vec::with_capacity(images.len());
            for path in &images {
                encoded.push(read_attachment(path, "image")?);
            }
            Some(MultimodalInput::from_images(
                spec,
                &encoded,
                &mut token_ids,
                multimodal::MAX_EXPANDED_PROMPT_TOKENS,
            )?)
        }
    };

    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(token_ids),
        options,
    });
    match input {
        Some(input) => input.bind(request),
        None => Ok(request),
    }
}

/// Read an attachment file, naming it and its modality on failure.
pub(super) fn read_attachment(path: &Path, kind: &str) -> anyhow::Result<Vec<u8>> {
    std::fs::read(path).map_err(|error| {
        anyhow::anyhow!(
            "What: the {kind} file {} could not be read. \
             Why: {error}. \
             How: check the path and that the file is readable.",
            path.display()
        )
    })
}

pub(super) fn run_repl(args: RunArgs, profiling: &ProfileArgs) -> anyhow::Result<()> {
    install_ctrlc_handler();
    args.cpu.apply().map_err(anyhow::Error::msg)?;
    let input_mode = repl_input_mode(io::stdin().is_terminal(), io::stdout().is_terminal());
    let mut settings = SessionSettings::new(resolve_model_dir(&args.model), &args.engine);
    let load_started = std::time::Instant::now();
    let mut backend = Backend::open(&settings)?;
    let load_elapsed = load_started.elapsed();
    let mut model_dir = settings.model_dir.clone();
    let mut raw_mode = args.sampling.raw;
    let mut show_profile = profiling.profile;
    // A session that asks to profile wants to see everything; detail can be
    // turned down afterwards. One-shot runs keep their own default, where the
    // intent is stated up front and the cost matters more.
    let mut trace_verbosity = TraceVerbosity::Full;
    // Where `/profile on` puts a timeline when the user has not named a file.
    let default_trace_path = PathBuf::from("onnx-genai-session.perfetto.json");
    let mut show_stats = initial_repl_show_stats(input_mode, args.no_stats);
    let sampling_options = args.sampling.to_options();
    let sampling_overrides = args.sampling.sampling_overrides();
    let mut session_usage = SessionUsage::default();
    // Inert unless stdout is a terminal, so a piped session is byte-for-byte
    // what it was before.
    let mut live = live_turn::LiveTurn::new();
    let mut template = load_chat_template(&model_dir, raw_mode);
    let mut reasoning = detect_reasoning(template.as_ref());
    backend.bind_reasoning_marker_tokens(&mut reasoning);
    let mut response = load_response_config(&model_dir, raw_mode);
    bind_response_tokens(&mut response, &backend);
    let mut warned_missing_context_limit = false;

    eprintln!(
        "onnx-genai interactive session ({} input). Enter a prompt, or an empty line / Ctrl-D to exit.\n\
         Ctrl-C aborts the current generation; press it twice to exit. Type /help for commands.",
        backend.accepted_modalities()
    );

    // Multi-turn conversation history. Each turn appends the user message, the
    // full history is rendered through the chat template, and the assistant's
    // reply is appended so later turns retain context. In raw mode there is no
    // template so only the latest user message is sent.
    let mut history: Vec<ChatMessage> = Vec::new();
    let mut image_attachments: Vec<PathBuf> = args.attachments.images.clone();
    let mut audio_attachments: Vec<PathBuf> = args.attachments.audio.clone();
    let stdin = io::stdin();
    let mut reader = ReplReader::new(input_mode);
    loop {
        let line = match reader.read_line(&stdin)? {
            ReplRead::Line(line) => line,
            ReplRead::Continue => continue,
            ReplRead::Eof => break,
        };
        // The user is still working, so a later Ctrl-C needs two presses again.
        EXIT_ARMED.store(false, Ordering::SeqCst);
        let prompt = match parse_repl_line(&line, input_mode) {
            ReplLine::Empty => break,
            ReplLine::Prompt(prompt) => Some(prompt),
            ReplLine::Command(ReplCommand::Help(command)) => {
                match command {
                    Some(command) => match render_command_help(&command) {
                        Some(help) => println!("{help}"),
                        None => eprintln!("unknown command: /{command} (try /help)"),
                    },
                    None => println!("{}", render_repl_help()),
                }
                None
            }
            ReplLine::Command(ReplCommand::Reset) => {
                history.clear();
                image_attachments.clear();
                audio_attachments.clear();
                session_usage = SessionUsage::default();
                println!("conversation history and pending attachments cleared");
                None
            }
            ReplLine::Command(ReplCommand::Profile(setting)) => {
                match parse_profile_setting(setting.as_deref()) {
                    Ok(ProfileSetting::Show) => {
                        println!("profile report {}", if show_profile { "on" } else { "off" });
                        match onnx_genai::ort::profile::trace_destination() {
                            Some(path) => println!(
                                "timeline {} -> {} ({} detail)",
                                if show_profile { "on" } else { "off" },
                                path.display(),
                                trace_verbosity
                            ),
                            None => println!("timeline off"),
                        }
                    }

                    Ok(ProfileSetting::Toggle(on)) => {
                        show_profile = on;
                        if on && onnx_genai::ort::profile::trace_destination().is_none() {
                            onnx_genai::ort::profile::set_trace_path(Some(
                                default_trace_path.clone(),
                            ));
                        }
                        set_trace_recording(on, trace_verbosity);
                        println!("profile report {}", if on { "on" } else { "off" });
                        match (on, onnx_genai::ort::profile::trace_destination()) {
                            (true, Some(path)) => println!(
                                "timeline on -> {} ({} detail); written when the session ends",
                                path.display(),
                                trace_verbosity
                            ),
                            _ => println!("timeline off"),
                        }
                        // The engine's per-stage breakdown is switched on from
                        // the environment before any thread starts, and cannot
                        // be turned on later. Say so rather than print a report
                        // that is quietly missing its most detailed section.
                        if on && !profiling.profile {
                            println!(
                                "note: per-stage timings need --profile at startup; this report covers timings, memory, and cache reuse"
                            );
                        }
                    }
                    Ok(ProfileSetting::Trace(path)) => {
                        onnx_genai::ort::profile::set_trace_path(Some(path.clone()));
                        set_trace_recording(true, trace_verbosity);
                        println!(
                            "timeline -> {} ({} detail)",
                            path.display(),
                            trace_verbosity
                        );
                    }
                    Ok(ProfileSetting::NoTrace) => {
                        set_trace_recording(false, trace_verbosity);
                        // Explicitly off, which also overrides a destination
                        // named at startup — otherwise "off" would keep writing
                        // wherever `--profile-trace` pointed.
                        onnx_genai::ort::profile::set_trace_path(None);
                        println!("timeline off");
                    }
                    Ok(ProfileSetting::Verbosity(level)) => {
                        trace_verbosity = level;
                        let recording =
                            onnx_genai::ort::profile::trace_destination().is_some() && show_profile;
                        set_trace_recording(recording, trace_verbosity);
                        println!("timeline detail {level}");
                        if matches!(level, TraceVerbosity::Full) {
                            println!(
                                "note: full detail adds a span per worker thread per operator; measured about 4% slower"
                            );
                        }
                    }
                    Err(error) => eprintln!("error: {error}"),
                }
                None
            }
            ReplLine::Command(ReplCommand::Model(path)) => {
                match path {
                    Some(path) => {
                        let mut next = settings.clone();
                        next.model_dir = resolve_model_dir(Path::new(&path));
                        match reload(&next) {
                            Ok(loaded) => {
                                backend = loaded;
                                backend.bind_reasoning_marker_tokens(&mut reasoning);
                                settings = next;
                                model_dir = settings.model_dir.clone();
                                template = load_chat_template(&model_dir, raw_mode);
                                reasoning = detect_reasoning(template.as_ref());
                                backend.bind_reasoning_marker_tokens(&mut reasoning);
                                response = load_response_config(&model_dir, raw_mode);
                                bind_response_tokens(&mut response, &backend);
                                warned_missing_context_limit = false;
                                // A conversation is about the model that held
                                // it; replaying it into a different model would
                                // attribute words to something that never said
                                // them.
                                history.clear();
                                image_attachments.clear();
                                audio_attachments.clear();
                                session_usage = SessionUsage::default();
                                println!(
                                    "loaded {} ({} input); conversation cleared",
                                    model_dir.display(),
                                    backend.accepted_modalities()
                                );
                            }
                            Err(error) => eprintln!("error: {error:#}"),
                        }
                    }
                    None => println!(
                        "{}",
                        SessionSummary {
                            settings: &settings,
                            execution_provider: backend.execution_provider_status(),
                            resolved_decode_backend: backend.decode_backend(),
                            sampling: resolve_session_sampling(
                                &sampling_options,
                                &backend,
                                &sampling_overrides
                            ),
                            history: &history,
                            usage: &session_usage,
                        }
                    ),
                }
                None
            }
            ReplLine::Command(ReplCommand::Session) => {
                println!(
                    "{}",
                    SessionSummary {
                        settings: &settings,
                        execution_provider: backend.execution_provider_status(),
                        resolved_decode_backend: backend.decode_backend(),
                        sampling: resolve_session_sampling(
                            &sampling_options,
                            &backend,
                            &sampling_overrides
                        ),
                        history: &history,
                        usage: &session_usage,
                    }
                );
                None
            }
            ReplLine::Command(ReplCommand::ExecutionProvider(name)) => {
                match name {
                    Some(name) if !available_execution_providers().contains(&name.as_str()) => {
                        // Rejected here rather than by the loader, which would
                        // report it as a failure to load the model.
                        eprintln!(
                            "error: What: {name:?} is not an execution provider this build can select. \
                             Why: provider support is compiled in, so a provider left out of the build \
                             cannot be chosen at runtime. \
                             How: use one of {}.",
                            available_execution_providers().join(", ")
                        );
                    }
                    Some(name) => {
                        let mut next = settings.clone();
                        next.execution_provider = (name != "auto").then(|| name.clone());
                        match reload(&next) {
                            Ok(loaded) => {
                                backend = loaded;
                                backend.bind_reasoning_marker_tokens(&mut reasoning);
                                settings = next;
                                history.clear();
                                session_usage = SessionUsage::default();
                                println!("execution provider {name}; conversation cleared");
                            }
                            Err(error) => eprintln!(
                                "error: {name} could not be selected: {error:#}\nthe previous session is still loaded"
                            ),
                        }
                    }
                    None => println!(
                        "execution provider {} (available: {})",
                        backend.execution_provider_status(),
                        available_execution_providers().join(", ")
                    ),
                }
                None
            }
            ReplLine::Command(ReplCommand::DecodeBackend(name)) => {
                match name {
                    Some(name) => match parse_decode_backend(&name) {
                        Ok(decode_backend) => {
                            let mut next = settings.clone();
                            next.decode_backend = decode_backend;
                            match reload(&next) {
                                Ok(loaded) => {
                                    backend = loaded;
                                    settings = next;
                                    history.clear();
                                    session_usage = SessionUsage::default();
                                    println!("decode backend {name}; conversation cleared");
                                }
                                Err(error) => eprintln!(
                                    "error: the {name} backend could not load this model: {error:#}\nthe previous session is still loaded"
                                ),
                            }
                        }
                        Err(error) => eprintln!("error: {error}"),
                    },
                    None => println!(
                        "{}",
                        SessionSummary {
                            settings: &settings,
                            execution_provider: backend.execution_provider_status(),
                            resolved_decode_backend: backend.decode_backend(),
                            sampling: resolve_session_sampling(
                                &sampling_options,
                                &backend,
                                &sampling_overrides
                            ),
                            history: &history,
                            usage: &session_usage,
                        }
                    ),
                }
                None
            }
            ReplLine::Command(ReplCommand::Pages) => {
                match backend.page_usage() {
                    Some(usage) => print!("{}", pages::render(&usage)),
                    // Absent rather than an empty pool: this decoder's KV is not
                    // paged at all, which is a different thing from holding
                    // nothing.
                    None => println!("this model's KV is not paged, so there are no pages to show"),
                }
                None
            }
            ReplLine::Command(ReplCommand::ToggleStats) => {
                show_stats = !show_stats;
                println!(
                    "per-turn stats {}",
                    if show_stats { "enabled" } else { "disabled" }
                );
                None
            }
            ReplLine::Command(ReplCommand::ToggleRaw) => {
                raw_mode = !raw_mode;
                template = load_chat_template(&model_dir, raw_mode);
                reasoning = detect_reasoning(template.as_ref());
                backend.bind_reasoning_marker_tokens(&mut reasoning);
                response = load_response_config(&model_dir, raw_mode);
                bind_response_tokens(&mut response, &backend);
                println!("raw mode {}", if raw_mode { "enabled" } else { "disabled" });
                None
            }
            ReplLine::Command(ReplCommand::System(system_message)) => {
                if history
                    .first()
                    .is_some_and(|message| matches!(message.role, ChatRole::System))
                {
                    history.remove(0);
                }
                match system_message {
                    Some(system_message) => {
                        history.insert(0, ChatMessage::system(system_message));
                        println!("system message set");
                    }
                    None => println!("system message cleared"),
                }
                None
            }
            ReplLine::Command(ReplCommand::Image { path, prompt }) => {
                match stage_attachment(path, "image", backend.supports_images()) {
                    Ok(Some(path)) => {
                        image_attachments.push(path);
                        prompt
                    }
                    Ok(None) => prompt,
                    Err(error) => {
                        eprintln!("{error:#}");
                        None
                    }
                }
            }
            ReplLine::Command(ReplCommand::Audio { path, prompt }) => {
                match stage_attachment(path, "audio", backend.supports_audio()) {
                    Ok(Some(path)) => {
                        audio_attachments.push(path);
                        prompt
                    }
                    Ok(None) => prompt,
                    Err(error) => {
                        eprintln!("{error:#}");
                        None
                    }
                }
            }
            ReplLine::Command(ReplCommand::Unknown(command)) => {
                eprintln!("unknown command: {command} (try /help)");
                None
            }
        };
        let Some(prompt) = prompt else {
            continue;
        };

        history.push(ChatMessage::user(prompt));
        let staged_images = std::mem::take(&mut image_attachments);
        let staged_audio = std::mem::take(&mut audio_attachments);
        if !staged_audio.is_empty() {
            println!(
                "(transcribing {} — the model's audio decoder prompt replaces the typed text)",
                display_paths(&staged_audio)
            );
        } else if !staged_images.is_empty() {
            println!("(sending {})", display_paths(&staged_images));
        }
        let rendered = build_turn_prompt(template.as_ref(), &history)?;
        // Resolve against the *current* backend each turn through the same helper
        // the `/session` summary uses, so the model's declared sampling regime is
        // honored, stays correct across a `/reload` that swaps in different
        // declared defaults, and can never disagree with what `/session` reports.
        let mut turn_options =
            resolve_session_sampling(&sampling_options, &backend, &sampling_overrides);
        let prompt_tokens = backend.prompt_tokens(&rendered).unwrap_or_default();
        let effective_max_context = backend.effective_max_context(&turn_options);
        if let Some(message) =
            drop_exhausted_repl_turn(&mut history, prompt_tokens, effective_max_context)
        {
            eprintln!("{message}");
            image_attachments = staged_images;
            audio_attachments = staged_audio;
            continue;
        }
        let used_fallback = apply_context_sized_max_new_tokens(
            &mut turn_options,
            args.sampling.max_new_tokens.is_some(),
            prompt_tokens,
            effective_max_context,
        );
        if used_fallback && !warned_missing_context_limit {
            warn_missing_context_limit(turn_options.max_new_tokens);
            warned_missing_context_limit = true;
        }
        let turn_max_new_tokens = turn_options.max_new_tokens;
        let turn = TurnInput {
            prompt: rendered,
            images: staged_images,
            audio: staged_audio,
            options: turn_options,
            prompt_tokens: Some(prompt_tokens),
            context_limit: effective_max_context,
        };

        let mut profile = RunProfile::new(model_dir.display().to_string());
        profile.execution_provider = backend.execution_provider_status();
        profile.decode_backend = Some(decode_backend_name(backend.decode_backend()).to_string());
        profile.phase("model load", load_elapsed);
        profile.prompt_tokens = Some(prompt_tokens);
        profile.context = effective_max_context.map(|max_tokens| profile::ContextUsage {
            used_tokens: prompt_tokens,
            max_tokens,
        });
        if let Some(memory) = backend.kv_usage() {
            profile.memory = memory;
        }
        let pages_before = backend.page_stats();
        match run_generation_turn(
            &mut backend,
            turn,
            true,
            Some(&mut profile),
            reasoning.as_ref(),
            response.as_ref(),
            // Live rendering follows `/stats`: it is what puts moving numbers
            // under the reply, and a session that did not ask for them keeps the
            // plain streaming path untouched.
            show_stats.then_some(&mut live),
        ) {
            Ok(output) => {
                // Reasoning models are trained with earlier turns' thinking
                // removed, so replaying it degrades quality and inflates the
                // context. Only the answer becomes history.
                let reply = match reasoning.as_ref() {
                    Some(config) => {
                        let split = config.markers.split(&output, config.opened_by_template);
                        // Drop the exchange when there is no answer to keep. Two
                        // cases qualify: the decode budget ran out mid-thought so
                        // the span never closed (`!complete`), or the span closed
                        // with only whitespace after it (`answer` empty). Both
                        // would otherwise record an empty assistant turn, which
                        // teaches the model that questions go unanswered and
                        // poisons later turns' context. Emptiness was historically
                        // guarded only on the unclosed path; the closed-but-empty
                        // case (e.g. a decode that stops exactly on `</think>`) is
                        // the same defect and is guarded here too.
                        if !split.complete || split.answer.trim().is_empty() {
                            eprintln!(
                                "{}",
                                reasoning_incomplete_note(
                                    split.complete,
                                    profile.finish_reason.as_deref(),
                                    turn_max_new_tokens,
                                    args.sampling.max_new_tokens.is_some(),
                                )
                            );
                            history.pop();
                            profiling.emit_when(show_profile, &mut profile)?;
                            emit_stats_line(show_stats, show_profile, &mut profile);
                            continue;
                        }
                        split.answer
                    }
                    None => output,
                };
                history.push(ChatMessage::assistant(reply));
                session_usage.record(profile.prompt_tokens, profile.timings.tokens());
                if let (Some(before), Some(after)) = (pages_before, backend.page_stats()) {
                    profile.pages = Some(profile::PageActivity::since(before, after));
                }
                // Report per turn: in a session the interesting comparison is
                // between turns, not a single number at exit.
                profiling.emit_when(show_profile, &mut profile)?;
                emit_stats_line(show_stats, show_profile, &mut profile);
            }
            Err(error) if is_interrupt_error(&error) => {
                // Drop the interrupted turn from history so a partial/aborted
                // reply never pollutes the conversation context, then return to
                // the prompt instead of exiting.
                eprintln!("^C interrupted (press Ctrl-C again to exit)");
                history.pop();
            }
            Err(error) => {
                // A rejected attachment or unreadable file is a user error, not
                // a reason to end the session: report it and keep the REPL alive.
                eprintln!("error: {error:#}");
                history.pop();
            }
        }
    }
    Ok(())
}

/// Validate a `/image` or `/audio` argument, returning the path to stage.
///
/// `Ok(None)` means the command was informational (usage was printed) and the
/// turn should continue without an attachment.
pub(super) fn stage_attachment(
    path: Option<String>,
    kind: &str,
    supported: bool,
) -> anyhow::Result<Option<PathBuf>> {
    let Some(path) = path else {
        anyhow::bail!("usage: /{kind} <path> [prompt text]");
    };
    if !supported {
        anyhow::bail!(
            "What: the /{kind} attachment was rejected. \
             Why: the loaded model declares no {kind} input contract in its metadata. \
             How: load a package that declares one, or send the prompt as text."
        );
    }
    let path = PathBuf::from(path);
    if !path.is_file() {
        anyhow::bail!(
            "What: the {kind} file {} could not be staged. \
             Why: no readable file exists at that path. \
             How: pass a path relative to the current directory, or an absolute path.",
            path.display()
        );
    }
    Ok(Some(path))
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tty_mode_enables_live_stats_by_default() {
        assert!(matches!(repl_input_mode(true, true), ReplInputMode::Tty));
        assert!(initial_repl_show_stats(ReplInputMode::Tty, false));
        assert!(!initial_repl_show_stats(ReplInputMode::Tty, true));
        assert!(!initial_repl_show_stats(ReplInputMode::Plain, false));
    }
}
