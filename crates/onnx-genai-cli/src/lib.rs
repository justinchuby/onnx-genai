//! Unified `onnx-genai` command-line interface.
//!
//! Subcommands:
//! - `serve`    — start the OpenAI-compatible HTTP server
//! - `generate` — one-shot text generation, or text-to-image with `--output-image`
//! - `run`      — interactive generation REPL
//! - `show`     — inspect a model's resolved files and metadata
//! - `list`     — list model directories under a models directory
//! - `version`  — print version and available execution providers
//!
//! `generate`, `run`, and `show` accept either a model directory or a config
//! file inside it (a file resolves to its parent directory).
//!
//! `generate` and `run` accept image and audio input on any package whose
//! metadata declares the corresponding contract, reusing the same preprocessing
//! and placeholder-expansion path as the server (`onnx_genai_server::multimodal`)
//! so both front ends behave identically.
//!
//! This crate is built two ways:
//! - as the `onnx-genai` binary (`src/main.rs`) for local development, and
//! - as the `onnx-genai-server` wheel's private `_onnx_genai_server` extension
//!   module (behind the `python` feature). The wheel ships a Python console
//!   entry point (`onnx_genai_server:main`) that loads the ONNX Runtime shared
//!   library from the installed `onnxruntime` wheel and then calls
//!   [`run`] through `_run_cli`. A raw binary cannot run that loader shim, so
//!   the Python entry point is how the wheel finds ONNX Runtime at exec time.

use std::fmt;
use std::io::{self, BufRead, IsTerminal, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use onnx_genai::ort::profile::TraceVerbosity;
use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::Context as _;
use clap::{Args, Parser, Subcommand};

mod commands;
mod generate;
mod live_turn;
mod memory;
mod model_inspection;
mod output;
mod pages;
mod profile;
mod transcribe;
use onnx_genai::engine::{EngineDecodeBackend, PipelineEngine, PipelineGenerateRequest};
use onnx_genai::metadata::load_metadata;
use onnx_genai::ort::{ChatMessage, ChatRole, ChatTemplate, ModelDirectory, Tokenizer};
use onnx_genai::ort::{SessionOptions, ep_selection};
use onnx_genai::preprocess::audio::{
    AudioSegment, SegmentConfig, StreamSegmenter, decode_wav_pcm16,
};
use onnx_genai::reasoning::{ReasoningMarkers, ReasoningStream};
use onnx_genai::text_to_audio::{self, TextToAudioRequest};
use onnx_genai::text_to_image::{self, TextToImageRequest, VaeDecoder};
use onnx_genai::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest, GenerateResult,
    GenerateToken, GenerateTokenCallback, StopSequence,
};
use onnx_genai_server::multimodal::{self, MultimodalInput, MultimodalSpecs};
use onnx_genai_server::{ServeArgs, from_models_dir, run_serve};
use commands::{
    ProfileSetting, ReplCommand, ReplLine, available_execution_providers, parse_decode_backend,
    parse_profile_setting, parse_repl_line, reload, resolved_default_providers, set_trace_recording,
};
use generate::generate;
use model_inspection::{list, show, version};
use output::{
    ReasoningConfig, build_turn_prompt, detect_reasoning, display_paths, emit_stats_line,
    load_chat_template, run_generation_turn, write_merged_trace,
};
use profile::RunProfile;
use transcribe::transcribe;

/// Process exit code for termination via SIGINT (Ctrl-C), matching the POSIX
/// convention of `128 + SIGINT`.
const EXIT_INTERRUPTED: i32 = 130;

/// Set while a generation is running so the Ctrl-C handler can distinguish an
/// interrupt during generation (soft-cancel the current turn) from an interrupt
/// at an idle prompt (arm the exit).
static GENERATING: AtomicBool = AtomicBool::new(false);

/// Set by the Ctrl-C handler when a generation should be aborted. The streaming
/// callback polls this and returns [`Interrupted`] to unwind out of the engine.
static INTERRUPT_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Set once a Ctrl-C has already been observed, so the next one exits the
/// process. Cleared whenever the user submits a new REPL line, which proves
/// they meant to keep working rather than quit.
static EXIT_ARMED: AtomicBool = AtomicBool::new(false);

/// Guards one-time installation of the Ctrl-C handler.
static CTRLC_HANDLER: Once = Once::new();

/// Marker error returned by the streaming callback when a Ctrl-C interrupt has
/// been requested. It propagates out of `generate_with_callback` as an
/// [`anyhow::Error`] and is recognized with [`is_interrupt_error`] so the REPL
/// can distinguish a user cancel from a genuine generation failure.
#[derive(Debug)]
struct Interrupted;

impl fmt::Display for Interrupted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("generation interrupted")
    }
}

impl std::error::Error for Interrupted {}

/// Returns true when `error` was produced by a Ctrl-C interrupt (i.e. carries an
/// [`Interrupted`] marker), as opposed to a real generation failure.
fn is_interrupt_error(error: &anyhow::Error) -> bool {
    error.downcast_ref::<Interrupted>().is_some()
}

/// What a Ctrl-C press should do, given the session's current state.
#[derive(Debug, PartialEq, Eq)]
enum InterruptAction {
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
fn interrupt_action(
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
fn install_ctrlc_handler() {
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
                InterruptAction::Exit => std::process::exit(EXIT_INTERRUPTED),
            }
        });
        if let Err(error) = result {
            eprintln!("warning: could not install Ctrl-C handler: {error}");
        }
    });
}
/// One turn's input: the rendered prompt plus any attachments staged for it.
#[derive(Debug, Default)]
struct TurnInput {
    prompt: String,
    images: Vec<PathBuf>,
    audio: Vec<PathBuf>,
    options: GenerateOptions,
}

/// The loaded model, which is either a single decoder graph or a declared
/// multi-component pipeline. Only pipeline packages can accept image or audio
/// input, and only when their metadata declares the corresponding contract.
enum Backend {
    Text(Box<Engine>),
    Pipeline(Box<PipelineBackend>),
}

/// A loaded pipeline package plus the contracts needed to feed it.
struct PipelineBackend {
    engine: PipelineEngine,
    tokenizer: Tokenizer,
    multimodal: MultimodalSpecs,
}

/// Everything that decides how a model is loaded, so an interactive session can
/// change one part and rebuild.
///
/// The execution provider and decode backend are properties of a *loaded*
/// session, not of a request: an ONNX session is created against its providers
/// and cannot be moved between them. Changing either therefore reloads the
/// model, which is why they live here together with the directory.
#[derive(Debug, Clone)]
struct SessionSettings {
    model_dir: PathBuf,
    /// Execution provider name, or `None` to keep whatever the environment and
    /// platform defaults select.
    execution_provider: Option<String>,
    decode_backend: EngineDecodeBackend,
    limits: onnx_genai::engine::ResourceLimits,
}

impl SessionSettings {
    fn new(model_dir: PathBuf, engine: &EngineArgs) -> Self {
        Self {
            model_dir,
            execution_provider: None,
            decode_backend: EngineDecodeBackend::Auto,
            limits: engine.to_config().limits,
        }
    }

    fn to_config(&self) -> EngineConfig {
        EngineConfig {
            decode_backend: self.decode_backend,
            limits: self.limits.clone(),
            ..EngineConfig::default()
        }
    }

    /// Session options for the chosen provider, or the environment's default
    /// when none was chosen.
    fn to_session_options(&self) -> SessionOptions {
        match &self.execution_provider {
            Some(name) => SessionOptions::with_execution_provider(ep_selection(name.clone())),
            None => SessionOptions::default(),
        }
    }

    /// The providers a session built from these settings actually runs on, in
    /// priority order.
    ///
    /// Resolved rather than echoed back: with no explicit choice the provider
    /// comes from the environment *or* from platform auto-selection (Metal on
    /// Apple Silicon), and reporting the request instead of the result would
    /// name CPU for a session running on the GPU.
    fn resolved_providers(&self) -> String {
        let options = self.to_session_options();
        let names = options
            .execution_providers
            .iter()
            .map(|provider| provider.selection.name.as_str())
            .collect::<Vec<_>>();
        if names.is_empty() {
            "cpu".to_string()
        } else {
            names.join(", ")
        }
    }

    fn backend_name(&self) -> &'static str {
        match self.decode_backend {
            EngineDecodeBackend::Auto => "auto",
            EngineDecodeBackend::Ort => "ort",
            EngineDecodeBackend::Native => "native",
        }
    }

    /// How the current selection reads back to a user.
    fn describe(&self) -> String {
        format!(
            "model {} · ep {} · backend {}",
            self.model_dir.display(),
            self.resolved_providers(),
            self.backend_name()
        )
    }
}

impl Backend {
    /// Load the model described by `settings`.
    fn open(settings: &SessionSettings) -> anyhow::Result<Self> {
        Self::load_with_options(
            &settings.model_dir,
            settings.to_config(),
            settings.to_session_options(),
        )
    }

    /// Load `model_dir`, preferring its declared pipeline when it has one.
    fn load(model_dir: &Path, config: EngineConfig) -> anyhow::Result<Self> {
        Self::load_with_options(model_dir, config, SessionOptions::default())
    }

    fn load_with_options(
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
    fn multimodal(&self) -> Option<&MultimodalSpecs> {
        match self {
            Self::Text(_) => None,
            Self::Pipeline(pipeline) => Some(&pipeline.multimodal),
        }
    }

    /// Human-readable summary of the modalities this model accepts.
    fn accepted_modalities(&self) -> String {
        self.multimodal()
            .map_or_else(|| "text".to_string(), MultimodalSpecs::accepted_modalities)
    }

    fn supports_images(&self) -> bool {
        self.multimodal()
            .is_some_and(|multimodal| multimodal.vision.is_some())
    }

    fn supports_audio(&self) -> bool {
        self.multimodal()
            .is_some_and(|multimodal| multimodal.audio.is_some())
    }

    /// Run one turn, streaming tokens through `callback`.
    /// Clear the pipeline reuse counters so a profile covers only the next turn.
    fn reset_reuse_stats(&self) {
        if let Self::Pipeline(pipeline) = self {
            pipeline.engine.reset_cache_stats();
        }
    }

    /// What a multimodal pipeline avoided recomputing, or `None` for a single
    /// decoder graph, which has no encoder or attachments to reuse.
    fn multimodal_reuse(&self) -> Option<profile::MultimodalReuse> {
        let Self::Pipeline(pipeline) = self else {
            return None;
        };
        let stats = pipeline.engine.cache_stats();
        Some(profile::MultimodalReuse {
            encoder_hits: stats.encoder_hits,
            encoder_misses: stats.encoder_misses,
            encoder_bytes: stats.encoder_bytes,
            prefix_reused_tokens: stats.prefix_reused_tokens,
            prefill_tokens: stats.prefill_tokens,
        })
    }

    /// What the KV page pool holds right now, when the backend pages its KV.
    fn page_usage(&self) -> Option<onnx_genai::kv::PageUsage> {
        match self {
            Self::Text(engine) => Some(engine.page_usage()),
            Self::Pipeline(pipeline) => pipeline.engine.page_usage(),
        }
    }

    /// Cumulative KV page counters, when the backend keeps a page pool.
    fn page_stats(&self) -> Option<onnx_genai::kv::PageStats> {
        match self {
            Self::Text(engine) => Some(engine.page_stats()),
            Self::Pipeline(_) => None,
        }
    }

    /// KV-cache accounting from the engine's resource governor.
    ///
    /// Only a single-model engine runs a governor; a pipeline reports nothing
    /// rather than a zero that would read as "no KV cache".
    fn kv_usage(&self) -> Option<profile::MemoryUsage> {
        match self {
            Self::Text(engine) => {
                let snapshot = engine.resource_snapshot();
                let budget = snapshot.derived_budget;
                let breakdown = snapshot.breakdown;
                Some(profile::MemoryUsage {
                    kv_budget_bytes: Some(budget.kv_bytes),
                    kv_max_tokens: Some(budget.max_total_tokens),
                    host_ram_used_bytes: Some(snapshot.host_ram.used),
                    device_used_bytes: Some(snapshot.vram.used),
                    device_limit_bytes: Some(snapshot.resolved_limits.vram_bytes),
                    peak_resident_bytes: None,
                    composition: Some(profile::DeviceComposition {
                        model_weights_bytes: breakdown.model_weights_bytes,
                        activations_bytes: breakdown.activations_bytes,
                        runtime_overhead_bytes: breakdown.ort_overhead_bytes,
                        kv_bytes: budget.kv_bytes,
                        kv_pages: budget.total_pages,
                        kv_page_bytes: budget.kv_bytes.checked_div(budget.total_pages).unwrap_or(0),
                    }),
                })
            }
            Self::Pipeline(_) => None,
        }
    }

    /// Number of tokens the prompt occupies, when the backend can tell.
    fn prompt_tokens(&self, prompt: &str) -> Option<usize> {
        match self {
            Self::Text(engine) => engine.tokenize(prompt).ok().map(|ids| ids.len()),
            Self::Pipeline(pipeline) => pipeline.tokenizer.encode(prompt).ok().map(|ids| ids.len()),
        }
    }

    fn generate(
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
                        Some(modality) if attachments == 0 => error.context(format!(
                            "the turn carried no attachment, but this model declares {modality} input. \
                             How: attach one with `/{modality} <path>` in the REPL, or `--{modality} <path>` on the command line."
                        )),
                        _ => error,
                    })
            }
        }
    }
}

/// Turn a prompt plus its attachments into a pipeline generation request.///
/// Audio replaces the prompt entirely: the transcription decoder is seeded with
/// the model's own transcription token sequence because the spoken audio, not
/// the typed text, carries the content. Images keep the prompt and expand each
/// placeholder token into the declared image-token run.
fn build_pipeline_request(
    tokenizer: &Tokenizer,
    multimodal: &MultimodalSpecs,
    turn: TurnInput,
) -> anyhow::Result<PipelineGenerateRequest> {
    let TurnInput {
        prompt,
        images,
        audio,
        options,
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
fn read_attachment(path: &Path, kind: &str) -> anyhow::Result<Vec<u8>> {
    std::fs::read(path).map_err(|error| {
        anyhow::anyhow!(
            "What: the {kind} file {} could not be read. \
             Why: {error}. \
             How: check the path and that the file is readable.",
            path.display()
        )
    })
}


#[derive(Debug, Parser)]
#[command(
    name = "onnx-genai",
    version,
    about = "Run generative AI models with ONNX Runtime"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[command(flatten)]
    profiling: ProfileArgs,
}

// `Serve` and `Generate` carry much larger argument structs than the rest, so
// the variants are boxed to keep the enum from being sized by its widest arm.
#[derive(Debug, Subcommand)]
enum Commands {
    /// Start an OpenAI-compatible HTTP server.
    Serve(Box<ServeArgs>),
    /// Generate text from a single prompt and exit.
    Generate(Box<GenerateArgs>),
    /// Start an interactive generation REPL (one prompt per line).
    Run(RunArgs),
    /// Show a model's resolved files and metadata.
    Show(ShowArgs),
    /// List model directories under a models directory.
    #[command(alias = "ls")]
    List(ListArgs),
    /// Transcribe speech to text, from files or a live stream.
    Transcribe(Box<TranscribeArgs>),
    /// Print version and available execution providers.
    Version,
}

/// Shared sampling flags for `generate` and `run`.
#[derive(Debug, Args)]
struct SamplingArgs {
    /// Maximum number of new tokens to generate.
    #[arg(long)]
    max_new_tokens: Option<usize>,

    /// Temperature applied before token selection.
    #[arg(long)]
    temperature: Option<f32>,

    /// Nucleus sampling probability. Values >= 1 disable top-p filtering.
    #[arg(long)]
    top_p: Option<f32>,

    /// Keep only the top-k logits before token selection. Zero disables top-k filtering.
    #[arg(long)]
    top_k: Option<usize>,

    /// Text stop sequence. May be provided multiple times.
    #[arg(long)]
    stop: Vec<String>,

    /// Send the prompt to the model verbatim, bypassing the model's chat
    /// template. Use for base (non-chat) models or to reproduce raw prompts.
    #[arg(long)]
    raw: bool,
}

/// Shared CPU resource controls for `generate` and `run`.
#[derive(Debug, Args)]
struct CpuArgs {
    /// Cap native CPU decode to N worker cores. Overrides
    /// ONNX_GENAI_CPU_DECODE_THREADS; when neither is set, automatic sizing is
    /// unchanged. Setting N now also bounds prefill/MLAS: the global Rayon pool
    /// is built with N workers (not all logical CPUs) and, on Linux, the process
    /// is pinned to N CPUs (packed on one NUMA node where possible), so
    /// `--cpu-cores N` alone makes the engine coexist with other programs -- no
    /// external `taskset` needed. An explicit ONNX_GENAI_CPU_DECODE_AFFINITY
    /// still wins over the automatic pinning.
    #[arg(long, value_name = "N")]
    cpu_cores: Option<NonZeroUsize>,
}

impl CpuArgs {
    fn apply(&self) -> anyhow::Result<()> {
        #[cfg(feature = "native-backend")]
        {
            onnx_genai_engine::set_cpu_decode_thread_budget(self.cpu_cores.map(NonZeroUsize::get))
                .map_err(anyhow::Error::msg)?;
        }
        #[cfg(not(feature = "native-backend"))]
        {
            let _ = self.cpu_cores;
        }
        Ok(())
    }
}

impl SamplingArgs {
    fn to_options(&self) -> GenerateOptions {
        let mut options = GenerateOptions::default();
        if let Some(max_new_tokens) = self.max_new_tokens {
            options.max_new_tokens = max_new_tokens;
        }
        if let Some(temperature) = self.temperature {
            options.temperature = temperature;
        }
        if let Some(top_p) = self.top_p {
            options.top_p = top_p;
        }
        if let Some(top_k) = self.top_k {
            options.top_k = top_k;
        }
        options.stop_sequences = self.stop.iter().cloned().map(StopSequence::Text).collect();
        options
    }
}

/// Shared engine-tuning flags.
#[derive(Debug, Args, Default, Clone)]
struct EngineArgs {
    /// Memory ceiling the engine may use for weights and KV cache: a byte count
    /// (`8GiB`), a fraction of detected capacity (`0.9`), or `auto`.
    ///
    /// An explicit byte value is authoritative — the runtime's device-capacity
    /// probe is still provisional, so this is how you tell it what is really
    /// available. Raising it enlarges the KV cache, and therefore the context
    /// that fits.
    #[arg(long, value_name = "LIMIT", value_parser = parse_limit)]
    vram_limit: Option<onnx_genai::engine::ResourceLimit>,

    /// Host RAM ceiling for the warm offload tier, in the same format.
    #[arg(long, value_name = "LIMIT", value_parser = parse_limit)]
    host_ram_limit: Option<onnx_genai::engine::ResourceLimit>,
}

impl EngineArgs {
    fn to_config(&self) -> EngineConfig {
        let mut config = EngineConfig::default();
        if let Some(limit) = self.vram_limit {
            config.limits.vram_limit = limit;
        }
        if let Some(limit) = self.host_ram_limit {
            config.limits.host_ram_limit = limit;
        }
        config
    }
}

/// Parse a `--vram-limit` / `--host-ram-limit` value.
fn parse_limit(input: &str) -> Result<onnx_genai::engine::ResourceLimit, String> {
    onnx_genai::engine::parse_resource_limit(input).map_err(|error| {
        format!(
            "What: the memory limit {input:?} was rejected. \
             Why: {error}. \
             How: pass a byte count such as 8GiB, a fraction such as 0.9, or auto."
        )
    })
}

/// Shared profiling flags.
#[derive(Debug, Args, Default, Clone)]
struct ProfileArgs {
    /// Report timing and throughput to stderr after the run: time to first
    /// token, decode tok/s, inter-token latency percentiles, and per-phase time.
    #[arg(long)]
    profile: bool,

    /// Also write the report as one JSON object, for diffing runs or plotting.
    /// Use `-` for stdout.
    #[arg(long, value_name = "PATH")]
    profile_json: Option<PathBuf>,

    /// Write a Chrome Trace Event timeline, viewable at https://ui.perfetto.dev.
    #[arg(long, value_name = "PATH")]
    profile_trace: Option<PathBuf>,
}

impl ProfileArgs {
    /// Whether anything at all was requested.
    fn requested(&self) -> bool {
        self.profile || self.profile_json.is_some() || self.profile_trace.is_some()
    }

    /// Turn on the engine's stage profiler and timeline tracer.
    ///
    /// Both are read once from the environment by the runtime config, so this
    /// must run before any engine call. It is called from `run` before the
    /// command dispatches, while the process is still single-threaded.
    fn install(&self) {
        if !self.requested() {
            return;
        }
        // SAFETY: called at startup from the single-threaded argument-parsing
        // path, before any engine, driver, or runtime thread exists.
        unsafe {
            if self.profile || self.profile_json.is_some() {
                std::env::set_var("ONNX_GENAI_PROFILE", "1");
            }
            if let Some(path) = &self.profile_trace {
                std::env::set_var("ONNX_GENAI_TRACE", path);
            }
        }
    }

    /// Emit whatever the caller asked for.
    fn emit(&self, profile: &mut RunProfile) -> anyhow::Result<()> {
        self.emit_when(self.profile, profile)
    }

    /// Emit with the text report gated on `show_text` rather than on the
    /// startup flag, so an interactive session can turn it on and off.
    ///
    /// The JSON output stays tied to its flag, which names a file chosen at
    /// startup. The timeline does not: a session can pick a destination later
    /// with `/profile trace`, and having asked for one *is* the request.
    fn emit_when(&self, show_text: bool, profile: &mut RunProfile) -> anyhow::Result<()> {
        let trace = onnx_genai::ort::profile::trace_destination();
        if !self.requested() && !show_text && trace.is_none() {
            return Ok(());
        }
        profile.memory.sample_peak();
        if show_text {
            eprint!("{}", profile.to_text());
        }
        if let Some(path) = &self.profile_json {
            let json = format!("{}\n", profile.to_json());
            if path.as_os_str() == "-" {
                print!("{json}");
                io::stdout().flush()?;
            } else {
                std::fs::write(path, json).map_err(|error| {
                    anyhow::anyhow!(
                        "What: the profile report could not be written to {}. \
                         Why: {error}. \
                         How: choose a path in a writable directory.",
                        path.display()
                    )
                })?;
            }
        }
        // Not `self.profile_trace`: an interactive session can choose a
        // destination after startup with `/profile trace`, and the startup flag
        // sets the same place, so asking where the timeline goes covers both.
        if let Some(path) = trace {
            let path = path.as_path();
            write_merged_trace(path).map_err(|error| {
                anyhow::anyhow!(
                    "What: the timeline trace could not be written to {}. \
                     Why: {error}. \
                     How: choose a path in a writable directory.",
                    path.display()
                )
            })?;
            eprintln!(
                "[profile] wrote {} (open it at https://ui.perfetto.dev)",
                path.display()
            );
        }
        Ok(())
    }
}

/// Shared multimodal attachment flags for `generate` and `run`.
#[derive(Debug, Args, Default, Clone)]
struct AttachmentArgs {
    /// Image file sent with the prompt. May be provided multiple times.
    /// Requires a pipeline package that declares an image preprocessing program.
    #[arg(long = "image", value_name = "PATH")]
    images: Vec<PathBuf>,

    /// PCM16 WAV file sent with the prompt. Requires a pipeline package that
    /// declares an `input_features` audio input.
    #[arg(long = "audio", value_name = "PATH")]
    audio: Vec<PathBuf>,
}

/// Text-to-image flags for `generate --output-image`.
#[derive(Debug, Args)]
struct ImageOutputArgs {
    /// Render the prompt to this PNG file instead of generating text. Requires a
    /// diffusion package whose metadata declares `strategy.kind: iterative`.
    /// With `--batch-size > 1` the images are written as `<stem>_0.png`, ...
    #[arg(long, value_name = "PATH")]
    output_image: Option<PathBuf>,

    /// Negative prompt, used as the classifier-free-guidance unconditional embedding.
    #[arg(long, default_value = "")]
    negative_prompt: String,

    /// Number of denoise steps. Defaults to the package's declared `num_steps`.
    #[arg(long)]
    steps: Option<usize>,

    /// Classifier-free-guidance scale; 1.0 disables guidance. Defaults to the
    /// package's declared `guidance_scale`.
    #[arg(long)]
    guidance_scale: Option<f32>,

    /// Seed for the initial latent.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Output image height in pixels (must be a multiple of 8).
    #[arg(long, default_value_t = 512)]
    height: usize,

    /// Output image width in pixels (must be a multiple of 8).
    #[arg(long, default_value_t = 512)]
    width: usize,

    /// Number of images to render in one batch.
    #[arg(long, default_value_t = 1)]
    batch_size: usize,

    /// CLIP tokenizer.json (defaults to `<model>/tokenizer.json`).
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// Prompt encoder ONNX file (defaults to the component declared in metadata).
    #[arg(long)]
    text_encoder: Option<PathBuf>,

    /// Standalone latent→image ONNX decoder, for packages whose pipeline stops
    /// at the latent instead of declaring a final image phase.
    #[arg(long)]
    vae_decoder: Option<PathBuf>,

    /// Latent scaling factor applied before `--vae-decoder` (Stable Diffusion 1.x uses 0.18215).
    #[arg(long, default_value_t = 0.18215)]
    vae_scaling_factor: f32,
}

impl ImageOutputArgs {
    fn to_request(&self, prompt: String) -> TextToImageRequest {
        TextToImageRequest {
            prompt,
            negative_prompt: self.negative_prompt.clone(),
            steps: self.steps,
            guidance_scale: self.guidance_scale,
            start_step: None,
            seed: self.seed,
            height: self.height,
            width: self.width,
            batch_size: self.batch_size,
            tokenizer_path: self.tokenizer.clone(),
            text_encoder_path: self.text_encoder.clone(),
            vae_decoder: self.vae_decoder.clone().map(|model_path| VaeDecoder {
                model_path,
                scaling_factor: self.vae_scaling_factor,
            }),
        }
    }
}

/// Text-to-speech flags for `generate --output-audio`.
#[derive(Debug, Args)]
struct AudioOutputArgs {
    /// Synthesize the prompt to this WAV file instead of generating text.
    /// Requires a package whose pipeline ends in a waveform stage.
    #[arg(long, value_name = "PATH")]
    output_audio: Option<PathBuf>,

    /// Override the package's declared output sample rate, in hertz. Only
    /// needed for a package whose metadata omits `pipeline.audio.sample_rate`.
    #[arg(long)]
    sample_rate: Option<u32>,
}

impl AudioOutputArgs {
    fn to_request(&self, text: String, sampling: &SamplingArgs) -> TextToAudioRequest {
        TextToAudioRequest {
            text,
            max_new_tokens: sampling.max_new_tokens,
            temperature: sampling.temperature,
            sample_rate: self.sample_rate,
            seed: None,
        }
    }
}

#[derive(Debug, Args)]
struct GenerateArgs {
    /// Model directory, or a config file inside it (e.g. inference_metadata.yaml).
    model: PathBuf,

    #[command(flatten)]
    sampling: SamplingArgs,

    #[command(flatten)]
    attachments: AttachmentArgs,

    #[command(flatten)]
    engine: EngineArgs,

    #[command(flatten)]
    cpu: CpuArgs,

    #[command(flatten)]
    image_output: ImageOutputArgs,

    #[command(flatten)]
    audio_output: AudioOutputArgs,

    /// Print generated tokens as they arrive.
    #[arg(long)]
    stream: bool,

    /// Prompt text.
    #[arg(long, short = 'p')]
    prompt: String,
}

#[derive(Debug, Args)]
struct RunArgs {
    /// Model directory, or a config file inside it (e.g. inference_metadata.yaml).
    model: PathBuf,

    #[command(flatten)]
    sampling: SamplingArgs,

    #[command(flatten)]
    attachments: AttachmentArgs,

    #[command(flatten)]
    engine: EngineArgs,

    #[command(flatten)]
    cpu: CpuArgs,
}

/// Output shape for `transcribe`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum TranscriptFormat {
    /// One line of text per segment.
    Text,
    /// One JSON object per segment, with timings.
    Json,
    /// SubRip subtitles.
    Srt,
}

#[derive(Debug, Args)]
struct TranscribeArgs {
    /// Model directory, or a config file inside it (e.g. inference_metadata.yaml).
    model: PathBuf,

    /// PCM16 WAV files to transcribe. Use `-` to read a stream from standard
    /// input, which is transcribed live: each segment is printed as soon as it
    /// is recognized. Defaults to `-` when no path is given.
    #[arg(value_name = "AUDIO")]
    audio: Vec<PathBuf>,

    /// Spoken language token, e.g. `en` or `zh`. Defaults to the model's own choice.
    #[arg(long)]
    language: Option<String>,

    /// Transcript format.
    #[arg(long, value_enum, default_value_t = TranscriptFormat::Text)]
    format: TranscriptFormat,

    /// Longest segment to transcribe at once, in seconds. Defaults to the
    /// model's declared input window, which is also the hard maximum.
    #[arg(long)]
    segment_seconds: Option<f32>,

    /// Silence needed to end a segment early, in seconds. Zero cuts on the
    /// window alone.
    #[arg(long, default_value_t = 0.5)]
    silence_seconds: f32,

    /// Amplitude (RMS) at or below which audio counts as silence.
    #[arg(long, default_value_t = 0.01)]
    silence_threshold: f32,

    /// Shortest segment worth transcribing, in seconds. Ignores brief clicks.
    #[arg(long, default_value_t = 0.1)]
    min_segment_seconds: f32,

    /// Sample rate of raw PCM16 arriving on standard input. Ignored when the
    /// stream begins with a WAV header, which declares its own rate.
    #[arg(long, default_value_t = 16_000)]
    sample_rate: u32,

    /// Channel count of raw PCM16 on standard input; channels are mixed to mono.
    #[arg(long, default_value_t = 1)]
    channels: u16,

    /// Maximum tokens to decode per segment.
    #[arg(long)]
    max_new_tokens: Option<usize>,

    #[command(flatten)]
    engine: EngineArgs,

    #[command(flatten)]
    cpu: CpuArgs,
}

#[derive(Debug, Args)]
struct ShowArgs {
    /// Model directory, or a config file inside it (e.g. inference_metadata.yaml).
    model: PathBuf,
}

#[derive(Debug, Args)]
struct ListArgs {
    /// Parent directory whose immediate subdirectories are each treated as one
    /// model. Falls back to ONNX_GENAI_MODELS_DIR.
    #[arg(long, env = "ONNX_GENAI_MODELS_DIR")]
    models_dir: PathBuf,
}

/// Accept either a model directory or a config file inside it. A file resolves
/// to its parent directory so `show ./model/genai_config.json` and
/// `show ./model` behave identically.
fn resolve_model_dir(path: &Path) -> PathBuf {
    if path.is_file() {
        path.parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| path.to_path_buf())
    } else {
        path.to_path_buf()
    }
}

/// Initialize tracing once. Safe to call from either the binary or the Python
/// entry point; a second call is ignored.
fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        // Diagnostics go to stderr: stdout carries the command's actual output
        // (generated text, transcripts), which is routinely piped and parsed.
        .with_writer(io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
}

/// Parse `args` (an argv vector whose first element is the program name) and run
/// the requested subcommand. Clap prints `--help`/`--version` and exits the
/// process on a usage error, matching a normal binary.
pub fn run(args: Vec<String>) -> anyhow::Result<()> {
    init_tracing();
    let cli = Cli::parse_from(args);
    cli.profiling.install();
    let profiling = cli.profiling;
    match cli.command {
        Commands::Serve(serve_args) => {
            let runtime = tokio::runtime::Runtime::new()?;
            runtime.block_on(run_serve(*serve_args))
        }
        Commands::Generate(generate_args) => generate(*generate_args, &profiling),
        Commands::Run(run_args) => run_repl(run_args, &profiling),
        Commands::Transcribe(transcribe_args) => transcribe(*transcribe_args, &profiling),
        Commands::Show(show_args) => show(&show_args.model),
        Commands::List(list_args) => list(&list_args.models_dir),
        Commands::Version => {
            version();
            Ok(())
        }
    }
}


fn run_repl(args: RunArgs, profiling: &ProfileArgs) -> anyhow::Result<()> {
    install_ctrlc_handler();
    args.cpu.apply()?;
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
    // Per-turn numbers are opt-in: a line after every reply is noise until a
    // reader is actually watching throughput or cache behavior.
    let mut show_stats = false;
    // Inert unless stdout is a terminal, so a piped session is byte-for-byte
    // what it was before.
    let mut live = live_turn::LiveTurn::new();
    let mut template = load_chat_template(&model_dir, raw_mode);
    let mut reasoning = detect_reasoning(template.as_ref());

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
    loop {
        print!(">>> ");
        io::stdout().flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            eprintln!();
            break;
        }
        // The user is still working, so a later Ctrl-C needs two presses again.
        EXIT_ARMED.store(false, Ordering::SeqCst);
        let line = line.trim_end_matches(['\n', '\r']);
        let prompt = match parse_repl_line(line) {
            ReplLine::Empty => break,
            ReplLine::Prompt(prompt) => Some(prompt),
            ReplLine::Command(ReplCommand::Help) => {
                println!(
                    "/help\n/reset\n/raw\n/stats\n/pages\n/profile [on|off|trace <path>|verbosity <decisions|ops|full>]\n/model [path]\n/ep [name]\n/backend [auto|ort|native]\n/system <text>\n/image <path> [prompt text]\n/audio <path> [prompt text]"
                );
                None
            }
            ReplLine::Command(ReplCommand::Reset) => {
                history.clear();
                image_attachments.clear();
                audio_attachments.clear();
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
                                settings = next;
                                model_dir = settings.model_dir.clone();
                                template = load_chat_template(&model_dir, raw_mode);
                                reasoning = detect_reasoning(template.as_ref());
                                // A conversation is about the model that held
                                // it; replaying it into a different model would
                                // attribute words to something that never said
                                // them.
                                history.clear();
                                image_attachments.clear();
                                audio_attachments.clear();
                                println!(
                                    "loaded {} ({} input); conversation cleared",
                                    model_dir.display(),
                                    backend.accepted_modalities()
                                );
                            }
                            Err(error) => eprintln!("error: {error:#}"),
                        }
                    }
                    None => println!("{}", settings.describe()),
                }
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
                                settings = next;
                                history.clear();
                                println!("execution provider {name}; conversation cleared");
                            }
                            Err(error) => eprintln!(
                                "error: {name} could not be selected: {error:#}\nthe previous session is still loaded"
                            ),
                        }
                    }
                    None => println!(
                        "execution provider {} (available: {})",
                        settings.resolved_providers(),
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
                                    println!("decode backend {name}; conversation cleared");
                                }
                                Err(error) => eprintln!(
                                    "error: the {name} backend could not load this model: {error:#}\nthe previous session is still loaded"
                                ),
                            }
                        }
                        Err(error) => eprintln!("error: {error}"),
                    },
                    None => println!("{}", settings.describe()),
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
        let turn = TurnInput {
            prompt: rendered,
            images: staged_images,
            audio: staged_audio,
            options: args.sampling.to_options(),
        };

        let mut profile = RunProfile::new(model_dir.display().to_string());
        profile.execution_provider = settings.resolved_providers();
        profile.decode_backend = Some(settings.backend_name().to_string());
        profile.phase("model load", load_elapsed);
        profile.prompt_tokens = backend.prompt_tokens(&turn.prompt);
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
            // Live rendering follows `/stats`: it is what puts moving numbers
            // under the reply, and a session that did not ask for them keeps the
            // plain streaming path untouched.
            show_stats.then_some(&mut live),
        ) {
            Ok(output) => {
                if !live.is_active() {
                    println!();
                }
                // Reasoning models are trained with earlier turns' thinking
                // removed, so replaying it degrades quality and inflates the
                // context. Only the answer becomes history.
                let reply = match reasoning.as_ref() {
                    Some(config) => {
                        let split = config.markers.split(&output, config.opened_by_template);
                        if !split.complete {
                            // The decode budget ran out mid-thought, so there is
                            // no answer. Drop the whole exchange rather than
                            // record an empty assistant turn, which would teach
                            // the model that questions go unanswered.
                            eprintln!(
                                "note: generation stopped inside the model's reasoning, so this turn is not kept. Raise --max-new-tokens."
                            );
                            history.pop();
                            profiling.emit_when(show_profile, &mut profile)?;
                            emit_stats_line(show_stats, show_profile, &mut profile);
                            continue;
                        }
                        split.answer.to_string()
                    }
                    None => output,
                };
                history.push(ChatMessage::assistant(reply));
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
                eprintln!("\n^C interrupted (press Ctrl-C again to exit)");
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
fn stage_attachment(
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
    use std::fs;

    #[test]
    fn generate_accepts_positional_model_and_prompt_flag() {
        let parsed_command_line =
            Cli::try_parse_from(["onnx-genai", "generate", "./m", "--prompt", "hi"]).unwrap();

        match parsed_command_line.command {
            Commands::Generate(args) => {
                assert_eq!(args.model, PathBuf::from("./m"));
                assert_eq!(args.prompt, "hi");
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn generate_accepts_prompt_short_flag() {
        let parsed_command_line =
            Cli::try_parse_from(["onnx-genai", "generate", "./m", "-p", "hi"]).unwrap();

        match parsed_command_line.command {
            Commands::Generate(args) => {
                assert_eq!(args.model, PathBuf::from("./m"));
                assert_eq!(args.prompt, "hi");
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn cpu_cores_is_shared_by_generate_and_run() {
        let generate = Cli::try_parse_from([
            "onnx-genai",
            "generate",
            "./m",
            "--prompt",
            "hi",
            "--cpu-cores",
            "8",
        ])
        .unwrap();
        let run = Cli::try_parse_from(["onnx-genai", "run", "./m", "--cpu-cores", "4"]).unwrap();

        match generate.command {
            Commands::Generate(args) => {
                assert_eq!(args.cpu.cpu_cores.map(NonZeroUsize::get), Some(8));
            }
            _ => panic!("expected generate command"),
        }
        match run.command {
            Commands::Run(args) => {
                assert_eq!(args.cpu.cpu_cores.map(NonZeroUsize::get), Some(4));
            }
            _ => panic!("expected run command"),
        }
    }

    #[test]
    fn cpu_cores_rejects_zero() {
        assert!(
            Cli::try_parse_from([
                "onnx-genai",
                "generate",
                "./m",
                "--prompt",
                "hi",
                "--cpu-cores",
                "0",
            ])
            .is_err()
        );
    }

    #[test]
    fn generate_requires_prompt_flag() {
        assert!(Cli::try_parse_from(["onnx-genai", "generate", "./m"]).is_err());
    }

    #[test]
    fn generate_rejects_model_flag() {
        assert!(
            Cli::try_parse_from(["onnx-genai", "generate", "--model", "./m", "--prompt", "hi"])
                .is_err()
        );
    }

    #[test]
    fn run_accepts_positional_model() {
        let parsed_command_line = Cli::try_parse_from(["onnx-genai", "run", "./m"]).unwrap();

        match parsed_command_line.command {
            Commands::Run(args) => assert_eq!(args.model, PathBuf::from("./m")),
            _ => panic!("expected run command"),
        }
    }

    #[test]
    fn run_rejects_model_flag() {
        assert!(Cli::try_parse_from(["onnx-genai", "run", "--model", "./m"]).is_err());
    }

    #[test]
    fn one_ctrl_c_stops_the_generation_and_two_exit() {
        // Idle prompt: the first press only warns, the second leaves.
        assert_eq!(
            interrupt_action(false, false, false),
            InterruptAction::WarnThenExit
        );
        assert_eq!(interrupt_action(false, false, true), InterruptAction::Exit);

        // During a generation: the first press cancels the turn, a second while
        // it is still unwinding leaves immediately.
        assert_eq!(
            interrupt_action(true, false, false),
            InterruptAction::CancelGeneration
        );
        assert_eq!(interrupt_action(true, true, true), InterruptAction::Exit);
    }

    #[test]
    fn a_cancelled_turn_leaves_the_exit_armed_for_the_next_press() {
        // After `run_generation_turn` returns, GENERATING is false again but the
        // cancelling press already armed the exit, so the next press exits —
        // "one press stops the generation, two exit".
        assert_eq!(interrupt_action(false, true, true), InterruptAction::Exit);
    }

    #[test]
    fn a_new_prompt_disarms_the_exit() {
        // `run_repl` clears EXIT_ARMED after each submitted line, so a much
        // later Ctrl-C warns again instead of quitting outright.
        assert_eq!(
            interrupt_action(false, true, false),
            InterruptAction::WarnThenExit
        );
    }

    #[test]
    fn generate_accepts_repeated_image_and_audio_attachments() {
        let parsed = Cli::try_parse_from([
            "onnx-genai",
            "generate",
            "./m",
            "--prompt",
            "describe",
            "--image",
            "a.png",
            "--image",
            "b.png",
            "--audio",
            "c.wav",
        ])
        .unwrap();

        match parsed.command {
            Commands::Generate(args) => {
                assert_eq!(
                    args.attachments.images,
                    vec![PathBuf::from("a.png"), PathBuf::from("b.png")]
                );
                assert_eq!(args.attachments.audio, vec![PathBuf::from("c.wav")]);
                assert!(args.image_output.output_image.is_none());
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn generate_accepts_text_to_image_flags() {
        let parsed = Cli::try_parse_from([
            "onnx-genai",
            "generate",
            "./m",
            "--prompt",
            "an astronaut riding a horse",
            "--output-image",
            "out.png",
            "--negative-prompt",
            "blurry",
            "--steps",
            "20",
            "--guidance-scale",
            "7.5",
            "--seed",
            "42",
            "--width",
            "768",
            "--height",
            "512",
        ])
        .unwrap();

        match parsed.command {
            Commands::Generate(args) => {
                let request = args.image_output.to_request(args.prompt.clone());
                assert_eq!(
                    args.image_output.output_image,
                    Some(PathBuf::from("out.png"))
                );
                assert_eq!(request.negative_prompt, "blurry");
                assert_eq!(request.steps, Some(20));
                assert_eq!(request.guidance_scale, Some(7.5));
                assert_eq!(request.seed, 42);
                assert_eq!((request.width, request.height), (768, 512));
                assert!(request.vae_decoder.is_none());
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn image_output_args_carry_the_standalone_vae_decoder() {
        let parsed = Cli::try_parse_from([
            "onnx-genai",
            "generate",
            "./m",
            "--prompt",
            "a cat",
            "--output-image",
            "out.png",
            "--vae-decoder",
            "vae_decoder/model.onnx",
            "--vae-scaling-factor",
            "0.13025",
        ])
        .unwrap();

        match parsed.command {
            Commands::Generate(args) => {
                let decoder = args
                    .image_output
                    .to_request(args.prompt.clone())
                    .vae_decoder
                    .expect("--vae-decoder must be carried through");
                assert_eq!(decoder.model_path, PathBuf::from("vae_decoder/model.onnx"));
                assert!((decoder.scaling_factor - 0.13025).abs() < 1e-6);
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn stage_attachment_requires_a_path() {
        let error =
            stage_attachment(None, "image", true).expect_err("a bare /image must report its usage");

        assert!(error.to_string().contains("usage: /image <path>"));
    }

    #[test]
    fn stage_attachment_rejects_unsupported_modalities_before_touching_the_disk() {
        let error = stage_attachment(Some("nonexistent.png".to_string()), "image", false)
            .expect_err("a text-only model must reject image attachments");

        let message = error.to_string();
        assert!(message.contains("What:"), "message: {message}");
        assert!(message.contains("How:"), "message: {message}");
    }

    #[test]
    fn stage_attachment_rejects_missing_files() {
        let error = stage_attachment(
            Some("definitely-not-a-real-file.png".to_string()),
            "image",
            true,
        )
        .expect_err("a missing file must fail closed");

        assert!(
            error.to_string().contains("definitely-not-a-real-file.png"),
            "the rejected path must be named: {error}"
        );
    }

    #[test]
    fn stage_attachment_accepts_an_existing_file() {
        let dir = temp_dir("stage-attachment");
        let path = dir.join("cat.png");
        fs::write(&path, b"not really a png").unwrap();

        let staged = stage_attachment(Some(path.display().to_string()), "image", true).unwrap();

        assert_eq!(staged, Some(path));
        fs::remove_dir_all(dir).unwrap();
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::current_dir().unwrap().join(format!(
            "cli-test-{}-{}",
            std::process::id(),
            name
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn is_interrupt_error_detects_marker_and_rejects_others() {
        let interrupt: anyhow::Error = anyhow::Error::new(Interrupted);
        assert!(is_interrupt_error(&interrupt));

        let other = anyhow::anyhow!("model failed to load");
        assert!(!is_interrupt_error(&other));

        // The marker must remain detectable after being wrapped with context,
        // mirroring how it unwinds through the engine.
        let wrapped = interrupt.context("while generating");
        assert!(is_interrupt_error(&wrapped));
    }

    #[test]
    fn repl_loop_continues_on_interrupt_but_propagates_real_errors() {
        // Mirrors the classification `run_repl` performs on each turn's result:
        // an interrupt keeps the loop alive; any other error aborts it.
        fn should_continue(result: anyhow::Result<()>) -> anyhow::Result<bool> {
            match result {
                Ok(()) => Ok(true),
                Err(error) if is_interrupt_error(&error) => Ok(true),
                Err(error) => Err(error),
            }
        }

        assert!(should_continue(Ok(())).unwrap());
        assert!(should_continue(Err(anyhow::Error::new(Interrupted))).unwrap());
        assert!(should_continue(Err(anyhow::anyhow!("boom"))).is_err());
    }

    #[test]
    fn parse_repl_line_recognizes_control_commands() {
        assert_eq!(
            parse_repl_line("/help"),
            ReplLine::Command(ReplCommand::Help)
        );
        assert_eq!(
            parse_repl_line("/reset"),
            ReplLine::Command(ReplCommand::Reset)
        );
        assert_eq!(
            parse_repl_line("/raw"),
            ReplLine::Command(ReplCommand::ToggleRaw)
        );
    }

    #[test]
    fn parse_repl_line_recognizes_system_commands() {
        assert_eq!(
            parse_repl_line("/system keep answers short"),
            ReplLine::Command(ReplCommand::System(Some("keep answers short".to_string())))
        );
        assert_eq!(
            parse_repl_line("/system   "),
            ReplLine::Command(ReplCommand::System(None))
        );
    }

    #[test]
    fn parse_repl_line_recognizes_image_and_audio_attachments() {
        assert_eq!(
            parse_repl_line("/image cat.png"),
            ReplLine::Command(ReplCommand::Image {
                path: Some("cat.png".to_string()),
                prompt: None,
            })
        );
        assert_eq!(
            parse_repl_line("/image cat.png describe this"),
            ReplLine::Command(ReplCommand::Image {
                path: Some("cat.png".to_string()),
                prompt: Some("describe this".to_string()),
            })
        );
        assert_eq!(
            parse_repl_line("/audio speech.wav summarize it"),
            ReplLine::Command(ReplCommand::Audio {
                path: Some("speech.wav".to_string()),
                prompt: Some("summarize it".to_string()),
            })
        );
        assert_eq!(
            parse_repl_line("/audio"),
            ReplLine::Command(ReplCommand::Audio {
                path: None,
                prompt: None,
            })
        );
    }

    #[test]
    fn parse_repl_line_preserves_prompts_and_rejects_unknown_commands() {
        assert_eq!(
            parse_repl_line("  explain this"),
            ReplLine::Prompt("  explain this".to_string())
        );
        assert_eq!(
            parse_repl_line("/unsupported extra"),
            ReplLine::Command(ReplCommand::Unknown("/unsupported".to_string()))
        );
    }

    #[test]
    fn session_control_commands_parse_with_and_without_an_argument() {
        assert_eq!(
            parse_repl_line("/profile on"),
            ReplLine::Command(ReplCommand::Profile(Some("on".to_string())))
        );
        assert_eq!(
            parse_repl_line("/profile"),
            ReplLine::Command(ReplCommand::Profile(None)),
            "a bare command reports the current state"
        );
        assert_eq!(
            parse_repl_line("/ep  cuda "),
            ReplLine::Command(ReplCommand::ExecutionProvider(Some("cuda".to_string()))),
            "surrounding whitespace is not part of the name"
        );
        assert_eq!(
            parse_repl_line("/backend native"),
            ReplLine::Command(ReplCommand::DecodeBackend(Some("native".to_string())))
        );
        assert_eq!(
            parse_repl_line("/model ./m"),
            ReplLine::Command(ReplCommand::Model(Some("./m".to_string())))
        );
    }

    #[test]
    fn a_toggle_reports_when_given_nothing_and_refuses_nonsense() {}

    #[test]
    fn decode_backends_are_named_by_the_engine_not_guessed() {
        assert_eq!(parse_decode_backend("auto"), Ok(EngineDecodeBackend::Auto));
        assert_eq!(parse_decode_backend("ort"), Ok(EngineDecodeBackend::Ort));
        assert_eq!(
            parse_decode_backend("native"),
            Ok(EngineDecodeBackend::Native)
        );
        let error = parse_decode_backend("cuda").expect_err("not a backend");
        assert!(error.contains("auto, ort, or native"), "{error}");
    }

    #[test]
    fn parse_repl_line_treats_empty_and_whitespace_lines_as_empty() {
        assert_eq!(parse_repl_line(""), ReplLine::Empty);
        assert_eq!(parse_repl_line(" \t "), ReplLine::Empty);
    }

    #[test]
    fn build_turn_prompt_applies_chat_template() {
        let dir = temp_dir("chat-template");
        fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"chat_template":"{% for m in messages %}<|{{ m.role }}|>\n{{ m.content }}\n{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}"}"#,
        )
        .unwrap();

        let template = load_chat_template(&dir, false).expect("template should load");
        let history = vec![ChatMessage::user("hello there")];
        let rendered = build_turn_prompt(Some(&template), &history).unwrap();

        assert!(rendered.contains("<|user|>"), "rendered: {rendered}");
        assert!(rendered.contains("hello there"));
        assert!(
            rendered.contains("<|assistant|>"),
            "generation prompt marker missing: {rendered}"
        );
        assert_ne!(rendered, "hello there");

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn build_turn_prompt_renders_multi_turn_history() {
        let dir = temp_dir("multi-turn");
        fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"chat_template":"{% for m in messages %}<|{{ m.role }}|>\n{{ m.content }}\n{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}"}"#,
        )
        .unwrap();

        let template = load_chat_template(&dir, false).unwrap();
        let history = vec![
            ChatMessage::user("first question"),
            ChatMessage::assistant("first answer"),
            ChatMessage::user("second question"),
        ];
        let rendered = build_turn_prompt(Some(&template), &history).unwrap();

        assert!(rendered.contains("first question"));
        assert!(rendered.contains("first answer"));
        assert!(rendered.contains("second question"));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn raw_mode_bypasses_chat_template() {
        let dir = temp_dir("raw-mode");
        fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"chat_template":"<|user|>\n{{ messages[0].content }}"}"#,
        )
        .unwrap();

        // `raw = true` must yield no template regardless of what the model ships.
        let template = load_chat_template(&dir, true);
        assert!(template.is_none());

        let history = vec![ChatMessage::user("verbatim prompt")];
        let rendered = build_turn_prompt(template.as_ref(), &history).unwrap();
        assert_eq!(rendered, "verbatim prompt");

        fs::remove_dir_all(dir).unwrap();
    }
}

/// PyO3 extension module backing the `onnx-genai-server` wheel.
///
/// The wheel's Python console entry point (`onnx_genai_server:main`) preloads
/// libonnxruntime from the installed `onnxruntime` wheel and then calls
/// `_run_cli(sys.argv)`. This is how a wheel-installed `onnx-genai` command
/// finds ONNX Runtime without bundling it (a plain binary has no import-time
/// hook to locate the onnxruntime package).
#[cfg(feature = "python")]
#[pyo3::pymodule]
fn _onnx_genai_server(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::*;

    /// Run the CLI with `args` (an argv list including the program name) and
    /// return a process exit code. The GIL is released while the command runs.
    #[pyfn(module)]
    #[pyo3(name = "_run_cli")]
    fn run_cli(python: Python<'_>, args: Vec<String>) -> i32 {
        python.detach(|| match run(args) {
            Ok(()) => 0,
            Err(error) => {
                eprintln!("error: {error:#}");
                1
            }
        })
    }

    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[cfg(test)]
mod profile_setting_tests {
    use super::*;

    #[test]
    fn parses_every_profile_setting_and_explains_the_rest() {
        assert_eq!(parse_profile_setting(None), Ok(ProfileSetting::Show));
        assert_eq!(parse_profile_setting(Some("  ")), Ok(ProfileSetting::Show));
        assert_eq!(
            parse_profile_setting(Some("on")),
            Ok(ProfileSetting::Toggle(true))
        );
        assert_eq!(
            parse_profile_setting(Some("off")),
            Ok(ProfileSetting::Toggle(false))
        );
        assert_eq!(
            parse_profile_setting(Some("trace  run.json")),
            Ok(ProfileSetting::Trace(PathBuf::from("run.json")))
        );
        assert_eq!(
            parse_profile_setting(Some("trace off")),
            Ok(ProfileSetting::NoTrace)
        );
        assert_eq!(
            parse_profile_setting(Some("verbosity full")),
            Ok(ProfileSetting::Verbosity(TraceVerbosity::Full))
        );
        assert_eq!(
            parse_profile_setting(Some("detail DECISIONS")),
            Ok(ProfileSetting::Verbosity(TraceVerbosity::Decisions))
        );

        // A path with spaces survives, since the rest of the line is the path.
        assert_eq!(
            parse_profile_setting(Some("trace my traces/run.json")),
            Ok(ProfileSetting::Trace(PathBuf::from("my traces/run.json")))
        );

        // Every rejection names the valid choices rather than only refusing.
        let unknown = parse_profile_setting(Some("bogus")).unwrap_err();
        assert!(unknown.contains("trace <path>"), "{unknown}");
        assert!(unknown.contains("verbosity <level>"), "{unknown}");

        let bad_level = parse_profile_setting(Some("verbosity loud")).unwrap_err();
        for level in ["decisions", "ops", "full"] {
            assert!(bad_level.contains(level), "{bad_level} should list {level}");
        }

        let no_path = parse_profile_setting(Some("trace")).unwrap_err();
        assert!(no_path.contains("needs a file"), "{no_path}");
    }
}
