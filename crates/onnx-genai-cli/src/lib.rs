//! Unified `onnx-genai` command-line interface.
//!
//! Subcommands:
//! - `serve`    — start the OpenAI-compatible HTTP server
//! - `generate` — one-shot text generation; image and speech generation use the server APIs
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

use std::io::{self, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Mutex, OnceLock};

use clap::{ArgAction, Args, Parser, Subcommand};

mod commands;
mod generate;
mod interactive;
mod live_turn;
mod memory;
mod model_inspection;
mod output;
mod pages;
mod profile;
mod transcribe;
use generate::generate;
use interactive::run_repl;
#[cfg(test)]
use interactive::{
    InterruptAction, Interrupted, ReplInputMode, apply_context_sized_max_new_tokens,
    context_exhaustion_error, context_window_is_full, drop_exhausted_repl_turn,
    initial_repl_show_stats, interrupt_action, is_interrupt_error, repl_input_mode,
    stage_attachment,
};
use model_inspection::{list, show, version};
#[cfg(test)]
use onnx_genai::engine::EngineDecodeBackend;
#[cfg(all(test, feature = "native-backend"))]
use onnx_genai::engine::native_decode_device::NativeDecodeDevice;
use onnx_genai::{GenerateOptions, SamplingOverrides, StopSequence};
#[cfg(test)]
use onnx_genai_server::runtime_args::DeviceChoice;
use onnx_genai_server::{
    ServeArgs, run_serve,
    runtime_args::{CpuArgs, EngineArgs, decode_backend_name},
};
use output::write_merged_trace;
use profile::RunProfile;
use transcribe::transcribe;

const CLI_FALLBACK_MAX_NEW_TOKENS: usize = 512;

#[cfg(test)]
use commands::{
    ProfileSetting, ReplCommand, ReplLine, command_registry, complete_repl_line,
    parse_profile_setting, parse_repl_line, render_repl_help,
};
#[cfg(test)]
use onnx_genai::ort::{ChatMessage, profile::TraceVerbosity};
#[cfg(test)]
use output::{build_turn_prompt, load_chat_template};

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

    /// Maximum total context length (prompt + generated tokens).
    #[arg(long, value_name = "TOKENS")]
    max_context: Option<NonZeroUsize>,

    /// Temperature applied before token selection.
    #[arg(long)]
    temperature: Option<f32>,

    /// Nucleus sampling probability. Values >= 1 disable top-p filtering.
    #[arg(long)]
    top_p: Option<f32>,

    /// Keep only the top-k logits before token selection. Zero disables top-k filtering.
    #[arg(long)]
    top_k: Option<usize>,

    /// Force deterministic argmax token selection.
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "no_greedy")]
    greedy: bool,

    /// Use stochastic sampling. Sampling flags also imply this unless temperature is 0.
    #[arg(long = "no-greedy", action = ArgAction::SetTrue)]
    no_greedy: bool,

    /// Text stop sequence. May be provided multiple times.
    #[arg(long)]
    stop: Vec<String>,

    /// Send the prompt to the model verbatim, bypassing the model's chat
    /// template. Use for base (non-chat) models or to reproduce raw prompts.
    #[arg(long)]
    raw: bool,
}

impl SamplingArgs {
    fn to_options(&self) -> GenerateOptions {
        let mut options = GenerateOptions {
            max_new_tokens: self.max_new_tokens.unwrap_or(CLI_FALLBACK_MAX_NEW_TOKENS),
            max_context: self.max_context.map(NonZeroUsize::get),
            ..GenerateOptions::default()
        };
        if self
            .temperature
            .is_some_and(|temperature| temperature > 0.0)
            || self.top_p.is_some()
            || self.top_k.is_some_and(|top_k| top_k > 0)
        {
            options.greedy = false;
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
        if self.no_greedy {
            options.greedy = false;
        }
        if self.greedy {
            options.greedy = true;
        }
        if self.temperature == Some(0.0) {
            options.greedy = true;
        }
        options.stop_sequences = self.stop.iter().cloned().map(StopSequence::Text).collect();
        options
    }

    /// The caller's *explicit* sampling selections, for
    /// [`GenerateOptions::resolve_sampling_defaults`].
    ///
    /// This is the "flags win" half of the precedence contract. A greedy
    /// decision is explicit when the user forced determinism (`--greedy` or
    /// `--temperature 0`) or requested sampling (`--no-greedy`, or any positive
    /// sampling control). When the user is silent about greediness the decision
    /// is deferred (`None`) so the model's declared `do_sample` can drive it.
    fn sampling_overrides(&self) -> SamplingOverrides {
        let forces_greedy = self.greedy || self.temperature == Some(0.0);
        let requests_sampling = self.no_greedy
            || self
                .temperature
                .is_some_and(|temperature| temperature > 0.0)
            || self.top_p.is_some()
            || self.top_k.is_some_and(|top_k| top_k > 0);
        let greedy = if forces_greedy {
            Some(true)
        } else if requests_sampling {
            Some(false)
        } else {
            None
        };
        SamplingOverrides {
            greedy,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
        }
    }
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

    /// Print generated tokens as they arrive.
    #[arg(long)]
    stream: bool,

    /// Do not show compact per-turn stats by default in an interactive terminal.
    #[arg(long)]
    no_stats: bool,

    /// Prompt text, given after the model: `generate MODEL "your prompt"`.
    ///
    /// Optional at the parser so an omitted prompt is answered with guidance
    /// rather than clap's bare "required argument"; [`GenerateArgs::prompt`]
    /// is what the rest of the command reads.
    #[arg(value_name = "PROMPT")]
    prompt: Option<String>,
}

impl GenerateArgs {
    /// The prompt to generate from.
    ///
    /// The model is positional, so the prompt is too: one way to pass each of
    /// this subcommand's two required arguments, in the order they read.
    fn take_prompt(&mut self) -> Result<String, String> {
        self.prompt.take().ok_or_else(|| {
            "What: no prompt was given. Why: `generate` needs text to continue. \
             How: pass it after the model (`generate MODEL \"your prompt\"`)."
                .to_string()
        })
    }
}

#[derive(Debug, Args)]
struct RunArgs {
    /// Model directory, or a config file inside it (e.g. inference_metadata.yaml).
    model: PathBuf,

    /// Do not show compact per-turn stats by default in an interactive terminal.
    #[arg(long)]
    no_stats: bool,

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
    #[arg(value_name = "MODELS_DIR", env = "ONNX_GENAI_MODELS_DIR")]
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
        .with_writer(DeferredStderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
}

static DEFERRED_TRACING: OnceLock<Mutex<Vec<u8>>> = OnceLock::new();

struct DeferredStderr;

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for DeferredStderr {
    type Writer = DeferredStderrWriter;

    fn make_writer(&'a self) -> Self::Writer {
        DeferredStderrWriter
    }
}

struct DeferredStderrWriter;

impl Write for DeferredStderrWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if interactive::GENERATING.load(Ordering::SeqCst) {
            DEFERRED_TRACING
                .get_or_init(|| Mutex::new(Vec::new()))
                .lock()
                .map_err(|_| io::Error::other("deferred tracing buffer is poisoned"))?
                .extend_from_slice(bytes);
            return Ok(bytes.len());
        }
        flush_deferred_tracing()?;
        io::stderr().write(bytes)
    }

    fn flush(&mut self) -> io::Result<()> {
        if !interactive::GENERATING.load(Ordering::SeqCst) {
            flush_deferred_tracing()?;
            io::stderr().flush()?;
        }
        Ok(())
    }
}

pub(crate) fn flush_deferred_tracing() -> io::Result<()> {
    let Some(buffer) = DEFERRED_TRACING.get() else {
        return Ok(());
    };
    let bytes = {
        let mut buffer = buffer
            .lock()
            .map_err(|_| io::Error::other("deferred tracing buffer is poisoned"))?;
        std::mem::take(&mut *buffer)
    };
    if !bytes.is_empty() {
        let mut stderr = io::stderr().lock();
        stderr.write_all(&bytes)?;
        stderr.flush()?;
    }
    Ok(())
}

/// Clears the streaming flag and flushes any diagnostics buffered during a turn
/// when it drops, so the buffer is drained on *every* exit from
/// [`run_generation_turn`](crate::output::run_generation_turn) — a normal
/// return, a `?` early-return while finalizing the reply, or a panic unwind —
/// not just the happy path.
///
/// The forced `process::exit` taken on a double Ctrl-C cannot run this: it
/// terminates the process from the signal-handler thread without unwinding the
/// generating thread's stack, so that path flushes explicitly before exiting.
pub(crate) struct FlushGuard;

impl Drop for FlushGuard {
    fn drop(&mut self) {
        interactive::GENERATING.store(false, Ordering::SeqCst);
        let _ = flush_deferred_tracing();
    }
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
        Commands::Generate(mut generate_args) => {
            let prompt = generate_args
                .take_prompt()
                .map_err(|message| anyhow::anyhow!(message))?;
            generate(*generate_args, prompt, &profiling)
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn flush_guard_drains_buffered_diagnostics_when_a_turn_exits_early() {
        // Simulate a turn in progress: the streaming flag is set and diagnostics
        // have been buffered by the deferred tracing writer instead of reaching
        // stderr live.
        interactive::GENERATING.store(true, Ordering::SeqCst);
        DEFERRED_TRACING
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .unwrap()
            .extend_from_slice(b"buffered diagnostic\n");

        // Reproduce a `?` early-return out of the finalization sequence while a
        // `FlushGuard` is live on the stack, exactly as a broken-pipe write error
        // from `emit_reasoning_segment` / `live.draw` / `live.finish` would.
        fn finalize_then_fail() -> io::Result<()> {
            let _flush_guard = FlushGuard;
            Err(io::Error::other("stdout write failed during finalization"))?;
            Ok(())
        }
        assert!(finalize_then_fail().is_err());

        // The guard must have drained the buffer (flushed it to the sink) and
        // cleared the streaming flag despite the early return, so diagnostics are
        // never silently lost.
        let buffered = DEFERRED_TRACING
            .get()
            .expect("buffer was initialized")
            .lock()
            .unwrap()
            .len();
        assert_eq!(
            buffered, 0,
            "FlushGuard left diagnostics buffered after an early return"
        );
        assert!(
            !interactive::GENERATING.load(Ordering::SeqCst),
            "FlushGuard left the streaming flag set after an early return"
        );
    }

    /// The model is positional, so the prompt is too: one way to pass each of
    /// this subcommand's two required arguments, in the order they read.
    #[test]
    fn generate_takes_the_model_then_the_prompt_positionally() {
        let parsed_command_line =
            Cli::try_parse_from(["onnx-genai", "generate", "./m", "hi"]).unwrap();

        match parsed_command_line.command {
            Commands::Generate(mut args) => {
                assert_eq!(args.model, PathBuf::from("./m"));
                assert_eq!(args.take_prompt().expect("a prompt was given"), "hi");
            }
            _ => panic!("expected generate command"),
        }
    }

    /// Clap's own "required argument" text names a placeholder; someone who
    /// typed only the model needs to be told what the second word is.
    #[test]
    fn generate_without_a_prompt_says_what_to_type() {
        let parsed_command_line = Cli::try_parse_from(["onnx-genai", "generate", "./m"]).unwrap();

        match parsed_command_line.command {
            Commands::Generate(mut args) => {
                let error = args.take_prompt().expect_err("no prompt was given");
                assert!(error.contains("generate MODEL"), "{error}");
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn cpu_cores_is_shared_by_generate_and_run() {
        let generate =
            Cli::try_parse_from(["onnx-genai", "generate", "./m", "hi", "--cpu-cores", "8"])
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
            Cli::try_parse_from(["onnx-genai", "generate", "./m", "hi", "--cpu-cores", "0",])
                .is_err()
        );
    }

    #[test]
    fn cli_uses_finite_fallback_until_a_model_context_is_known() {
        let parsed = Cli::try_parse_from(["onnx-genai", "generate", "./m", "hi"]).unwrap();

        match parsed.command {
            Commands::Generate(args) => {
                assert_eq!(
                    args.sampling.to_options().max_new_tokens,
                    CLI_FALLBACK_MAX_NEW_TOKENS
                );
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn absent_max_new_tokens_fills_remaining_context_when_known() {
        let mut options = GenerateOptions::default();

        let used_fallback =
            apply_context_sized_max_new_tokens(&mut options, false, 128, Some(2048));

        assert!(!used_fallback);
        assert_eq!(options.max_new_tokens, 1921);
    }

    #[test]
    fn explicit_max_new_tokens_is_honored_exactly() {
        let mut options = GenerateOptions {
            max_new_tokens: 2,
            ..GenerateOptions::default()
        };

        let used_fallback = apply_context_sized_max_new_tokens(&mut options, true, 128, Some(2048));

        assert!(!used_fallback);
        assert_eq!(options.max_new_tokens, 2);
    }

    #[test]
    fn absent_max_new_tokens_uses_finite_fallback_when_context_is_unknown() {
        let mut options = GenerateOptions::default();

        let used_fallback = apply_context_sized_max_new_tokens(&mut options, false, 128, None);

        assert!(used_fallback);
        assert_eq!(options.max_new_tokens, CLI_FALLBACK_MAX_NEW_TOKENS);
    }

    #[test]
    fn context_exhaustion_at_equal_limit_drops_only_current_repl_turn() {
        let mut history = vec![
            ChatMessage::user("first question"),
            ChatMessage::assistant("first answer"),
            ChatMessage::user("second question"),
        ];

        let message = drop_exhausted_repl_turn(&mut history, 2048, Some(2048))
            .expect("equal limit should be exhausted");

        assert!(message.contains("/reset"), "{message}");
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].content, "first question");
        assert_eq!(history[1].content, "first answer");
        assert!(
            !history.iter().any(|message| {
                message.role.as_str() == "assistant" && message.content.is_empty()
            })
        );
    }

    #[test]
    fn context_exhaustion_above_limit_drops_only_current_repl_turn() {
        let mut history = vec![
            ChatMessage::user("first question"),
            ChatMessage::assistant("first answer"),
            ChatMessage::user("second question"),
        ];

        let message = drop_exhausted_repl_turn(&mut history, 2050, Some(2048))
            .expect("above limit should be exhausted");

        assert!(message.contains("2050/2048"), "{message}");
        assert_eq!(history.len(), 2);
        assert!(
            !history.iter().any(|message| {
                message.role.as_str() == "assistant" && message.content.is_empty()
            })
        );
    }

    #[test]
    fn context_exhaustion_one_shot_reports_actionable_error() {
        let limit = context_window_is_full(2048, Some(2048))
            .expect("equal limit should reject one-shot generation");
        let error = context_exhaustion_error(2048, limit).to_string();

        assert!(error.contains("2048/2048"), "{error}");
        assert!(error.contains("shorten the prompt"), "{error}");
        assert!(error.contains("larger context window"), "{error}");
    }

    #[test]
    fn context_exhaustion_guard_leaves_healthy_turns_unchanged() {
        let mut history = vec![ChatMessage::user("still has room")];
        let mut options = GenerateOptions::default();

        let message = drop_exhausted_repl_turn(&mut history, 2047, Some(2048));
        let used_fallback =
            apply_context_sized_max_new_tokens(&mut options, false, 2047, Some(2048));

        assert!(message.is_none());
        assert!(!used_fallback);
        assert_eq!(options.max_new_tokens, 2);
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].content, "still has room");
    }

    #[test]
    fn max_context_is_shared_by_generate_and_run() {
        let generate = Cli::try_parse_from([
            "onnx-genai",
            "generate",
            "./m",
            "hi",
            "--max-context",
            "1024",
        ])
        .unwrap();
        let run =
            Cli::try_parse_from(["onnx-genai", "run", "./m", "--max-context", "2048"]).unwrap();

        match generate.command {
            Commands::Generate(args) => {
                assert_eq!(args.sampling.to_options().max_context, Some(1024));
            }
            _ => panic!("expected generate command"),
        }
        match run.command {
            Commands::Run(args) => {
                assert_eq!(args.sampling.to_options().max_context, Some(2048));
            }
            _ => panic!("expected run command"),
        }
    }

    #[test]
    fn max_context_rejects_zero() {
        assert!(
            Cli::try_parse_from(["onnx-genai", "generate", "./m", "hi", "--max-context", "0",])
                .is_err()
        );
    }

    #[test]
    fn backend_is_shared_by_generate_run_and_transcribe() {
        let generate =
            Cli::try_parse_from(["onnx-genai", "generate", "./m", "hi", "--backend", "native"])
                .unwrap();
        let run = Cli::try_parse_from(["onnx-genai", "run", "./m", "--backend", "ort"]).unwrap();
        let transcribe =
            Cli::try_parse_from(["onnx-genai", "transcribe", "./m", "--backend", "native"])
                .unwrap();

        match generate.command {
            Commands::Generate(args) => {
                assert_eq!(args.engine.backend, EngineDecodeBackend::Native);
                assert_eq!(
                    args.engine.to_config().decode_backend,
                    EngineDecodeBackend::Native
                );
            }
            _ => panic!("expected generate command"),
        }
        match run.command {
            Commands::Run(args) => {
                assert_eq!(args.engine.backend, EngineDecodeBackend::Ort);
                assert_eq!(
                    args.engine.to_config().decode_backend,
                    EngineDecodeBackend::Ort
                );
            }
            _ => panic!("expected run command"),
        }
        match transcribe.command {
            Commands::Transcribe(args) => {
                assert_eq!(args.engine.backend, EngineDecodeBackend::Native);
                assert_eq!(
                    args.engine.to_config().decode_backend,
                    EngineDecodeBackend::Native
                );
            }
            _ => panic!("expected transcribe command"),
        }
    }

    #[test]
    fn transcribe_rejects_unknown_backend_loudly() {
        let error = Cli::try_parse_from(["onnx-genai", "transcribe", "./m", "--backend", "cuda"])
            .expect_err("cuda is an execution provider, not a decode backend")
            .to_string();

        assert!(error.contains("auto, ort, or native"), "{error}");
    }

    #[test]
    fn backend_defaults_to_auto() {
        let parsed = Cli::try_parse_from(["onnx-genai", "generate", "./m", "hi"]).unwrap();

        match parsed.command {
            Commands::Generate(args) => {
                assert_eq!(args.engine.backend, EngineDecodeBackend::Auto);
                assert_eq!(
                    args.engine.to_config().decode_backend,
                    EngineDecodeBackend::Auto
                );
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn backend_reuses_repl_parser_error() {
        let error =
            Cli::try_parse_from(["onnx-genai", "generate", "./m", "hi", "--backend", "cuda"])
                .expect_err("cuda is an execution provider, not a decode backend")
                .to_string();

        assert!(
            error.contains("What: \"cuda\" is not a decode backend"),
            "{error}"
        );
        assert!(error.contains("How: use auto, ort, or native."), "{error}");
    }

    #[test]
    fn sampling_flags_disable_greedy_unless_temperature_is_zero_or_greedy_is_forced() {
        let stochastic = Cli::try_parse_from([
            "onnx-genai",
            "generate",
            "./m",
            "hi",
            "--temperature",
            "0.7",
        ])
        .unwrap();
        let top_p = Cli::try_parse_from(["onnx-genai", "run", "./m", "--top-p", "0.9"]).unwrap();
        let top_k =
            Cli::try_parse_from(["onnx-genai", "generate", "./m", "hi", "--top-k", "40"]).unwrap();
        let temperature_zero =
            Cli::try_parse_from(["onnx-genai", "generate", "./m", "hi", "--temperature", "0"])
                .unwrap();
        let forced_greedy = Cli::try_parse_from([
            "onnx-genai",
            "generate",
            "./m",
            "hi",
            "--temperature",
            "0.7",
            "--greedy",
        ])
        .unwrap();

        match stochastic.command {
            Commands::Generate(args) => assert!(!args.sampling.to_options().greedy),
            _ => panic!("expected generate command"),
        }
        match top_p.command {
            Commands::Run(args) => assert!(!args.sampling.to_options().greedy),
            _ => panic!("expected run command"),
        }
        match top_k.command {
            Commands::Generate(args) => assert!(!args.sampling.to_options().greedy),
            _ => panic!("expected generate command"),
        }
        match temperature_zero.command {
            Commands::Generate(args) => assert!(args.sampling.to_options().greedy),
            _ => panic!("expected generate command"),
        }
        match forced_greedy.command {
            Commands::Generate(args) => assert!(args.sampling.to_options().greedy),
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn generate_rejects_model_and_prompt_flags() {
        assert!(Cli::try_parse_from(["onnx-genai", "generate", "--model", "./m", "hi"]).is_err());
        assert!(Cli::try_parse_from(["onnx-genai", "generate", "./m", "--prompt", "hi"]).is_err());
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
    fn run_accepts_no_stats_opt_out() {
        let parsed_command_line =
            Cli::try_parse_from(["onnx-genai", "run", "./m", "--no-stats"]).unwrap();

        match parsed_command_line.command {
            Commands::Run(args) => assert!(args.no_stats),
            _ => panic!("expected run command"),
        }
    }

    #[test]
    fn generate_accepts_no_stats_opt_out() {
        let parsed_command_line =
            Cli::try_parse_from(["onnx-genai", "generate", "./m", "hi", "--no-stats"]).unwrap();

        match parsed_command_line.command {
            Commands::Generate(args) => assert!(args.no_stats),
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn tty_split_keeps_pipes_plain_and_stats_tty_only() {
        assert_eq!(repl_input_mode(true, true), ReplInputMode::Tty);
        assert_eq!(repl_input_mode(false, true), ReplInputMode::Plain);
        assert_eq!(repl_input_mode(true, false), ReplInputMode::Plain);

        assert!(initial_repl_show_stats(ReplInputMode::Tty, false));
        assert!(!initial_repl_show_stats(ReplInputMode::Tty, true));
        assert!(!initial_repl_show_stats(ReplInputMode::Plain, false));
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
            parse_repl_line("/help", ReplInputMode::Tty),
            ReplLine::Command(ReplCommand::Help(None))
        );
        assert_eq!(
            parse_repl_line("/reset", ReplInputMode::Tty),
            ReplLine::Command(ReplCommand::Reset)
        );
        assert_eq!(
            parse_repl_line("/raw", ReplInputMode::Tty),
            ReplLine::Command(ReplCommand::ToggleRaw)
        );
    }

    #[test]
    fn parse_repl_line_recognizes_system_commands() {
        assert_eq!(
            parse_repl_line("/system keep answers short", ReplInputMode::Tty),
            ReplLine::Command(ReplCommand::System(Some("keep answers short".to_string())))
        );
        assert_eq!(
            parse_repl_line("/system   ", ReplInputMode::Tty),
            ReplLine::Command(ReplCommand::System(None))
        );
    }

    #[test]
    fn parse_repl_line_recognizes_image_and_audio_attachments() {
        assert_eq!(
            parse_repl_line("/image cat.png", ReplInputMode::Tty),
            ReplLine::Command(ReplCommand::Image {
                path: Some("cat.png".to_string()),
                prompt: None,
            })
        );
        assert_eq!(
            parse_repl_line("/image cat.png describe this", ReplInputMode::Tty),
            ReplLine::Command(ReplCommand::Image {
                path: Some("cat.png".to_string()),
                prompt: Some("describe this".to_string()),
            })
        );
        assert_eq!(
            parse_repl_line("/audio speech.wav summarize it", ReplInputMode::Tty),
            ReplLine::Command(ReplCommand::Audio {
                path: Some("speech.wav".to_string()),
                prompt: Some("summarize it".to_string()),
            })
        );
        assert_eq!(
            parse_repl_line("/audio", ReplInputMode::Tty),
            ReplLine::Command(ReplCommand::Audio {
                path: None,
                prompt: None,
            })
        );
    }

    #[test]
    fn parse_repl_line_preserves_prompts_and_rejects_unknown_commands() {
        assert_eq!(
            parse_repl_line("  explain this", ReplInputMode::Tty),
            ReplLine::Prompt("  explain this".to_string())
        );
        assert_eq!(
            parse_repl_line("//literal slash", ReplInputMode::Tty),
            ReplLine::Prompt("/literal slash".to_string())
        );
        assert_eq!(
            parse_repl_line("/unsupported extra", ReplInputMode::Tty),
            ReplLine::Command(ReplCommand::Unknown("/unsupported".to_string()))
        );
    }

    #[test]
    fn parse_repl_line_keeps_plain_mode_compatible_with_piped_repl() {
        assert_eq!(
            parse_repl_line("//literal slash", ReplInputMode::Plain),
            ReplLine::Command(ReplCommand::Unknown("//literal".to_string()))
        );
        assert_eq!(
            parse_repl_line("/help anything", ReplInputMode::Plain),
            ReplLine::Command(ReplCommand::Help(None))
        );
        assert_eq!(
            parse_repl_line("/help anything", ReplInputMode::Tty),
            ReplLine::Command(ReplCommand::Help(Some("anything".to_string())))
        );
    }

    #[test]
    fn command_registry_drives_help_and_parser() {
        let help = render_repl_help();
        for command in command_registry() {
            assert!(
                help.lines().any(|line| line == command.usage),
                "{} missing from help",
                command.usage
            );
            assert!(
                !matches!(
                    parse_repl_line(&format!("/{}", command.name), ReplInputMode::Tty),
                    ReplLine::Command(ReplCommand::Unknown(_))
                ),
                "{} did not parse through registry",
                command.name
            );
        }
        assert_eq!(
            help,
            "/help\n/reset\n/raw\n/stats\n/pages\n/profile [on|off|trace <path>|verbosity <decisions|ops|full>]\n/model [path]\n/session\n/ep [name]\n/backend [auto|ort|native]\n/system <text>\n/image <path> [prompt text]\n/audio <path> [prompt text]"
        );
    }

    #[test]
    fn slash_completion_covers_commands_and_arguments() {
        let command_names = complete_repl_line("/ba", 3)
            .into_iter()
            .map(|item| item.replacement)
            .collect::<Vec<_>>();
        assert_eq!(command_names, vec!["/backend"]);

        let backends = complete_repl_line("/backend n", 10)
            .into_iter()
            .map(|item| item.replacement)
            .collect::<Vec<_>>();
        assert_eq!(backends, vec!["native"]);

        let providers = complete_repl_line("/ep a", 5)
            .into_iter()
            .map(|item| item.replacement)
            .collect::<Vec<_>>();
        assert!(providers.contains(&"auto".to_string()));

        let verbosity = complete_repl_line("/profile verbosity f", 20)
            .into_iter()
            .map(|item| item.replacement)
            .collect::<Vec<_>>();
        assert_eq!(verbosity, vec!["full"]);
    }

    #[test]
    fn session_control_commands_parse_with_and_without_an_argument() {
        assert_eq!(
            parse_repl_line("/profile on", ReplInputMode::Tty),
            ReplLine::Command(ReplCommand::Profile(Some("on".to_string())))
        );
        assert_eq!(
            parse_repl_line("/profile", ReplInputMode::Tty),
            ReplLine::Command(ReplCommand::Profile(None)),
            "a bare command reports the current state"
        );
        assert_eq!(
            parse_repl_line("/ep  cuda ", ReplInputMode::Tty),
            ReplLine::Command(ReplCommand::ExecutionProvider(Some("cuda".to_string()))),
            "surrounding whitespace is not part of the name"
        );
        assert_eq!(
            parse_repl_line("/backend native", ReplInputMode::Tty),
            ReplLine::Command(ReplCommand::DecodeBackend(Some("native".to_string())))
        );
        assert_eq!(
            parse_repl_line("/model ./m", ReplInputMode::Tty),
            ReplLine::Command(ReplCommand::Model(Some("./m".to_string())))
        );
        assert_eq!(
            parse_repl_line("/session", ReplInputMode::Tty),
            ReplLine::Command(ReplCommand::Session)
        );
    }

    #[test]
    fn session_summary_is_structured_and_redacts_message_content() {
        let settings =
            interactive::SessionSettings::new(PathBuf::from("models/tiny"), &EngineArgs::default());
        let options = GenerateOptions {
            max_new_tokens: 32,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            greedy: false,
            ..GenerateOptions::default()
        };
        let history = vec![
            ChatMessage::system("private instruction"),
            ChatMessage::user("private question"),
            ChatMessage::assistant("private answer"),
        ];
        let usage = interactive::SessionUsage {
            prompt_tokens: 14,
            generated_tokens: 5,
            completed_turns: 1,
        };
        let sampling_overrides = SamplingOverrides::default();
        let mut sampling = options;
        sampling.resolve_sampling_defaults(None, &sampling_overrides);

        let summary = interactive::SessionSummary {
            settings: &settings,
            execution_provider: "cpu".to_string(),
            resolved_decode_backend: EngineDecodeBackend::Ort,
            sampling,
            history: &history,
            usage: &usage,
        }
        .to_string();

        assert_eq!(
            summary,
            "session\n\
             \x20\x20model: models/tiny\n\
             \x20\x20execution provider: cpu\n\
             \x20\x20decode backend: ort\n\
             \x20\x20requested backend: auto\n\
             \x20\x20sampling: max_new_tokens=32 max_context=auto temperature=0.7 top_p=0.9 top_k=40 greedy=false\n\
             \x20\x20messages: 3 (system: 1, user: 1, assistant: 1)\n\
             \x20\x20completed turns: 1\n\
             \x20\x20tokens: prompt=14 generated=5"
        );
        assert!(!summary.contains("private"), "{summary}");
    }

    #[test]
    fn session_summary_displays_effective_model_sampling_defaults() {
        let settings =
            interactive::SessionSettings::new(PathBuf::from("models/tiny"), &EngineArgs::default());
        let options = GenerateOptions {
            max_new_tokens: 32,
            ..GenerateOptions::default()
        };
        let defaults = onnx_genai::metadata::GenerationDefaults {
            do_sample: Some(true),
            temperature: Some(0.6),
            top_k: Some(20),
            top_p: Some(0.95),
            repetition_penalty: None,
            num_beams: None,
            num_return_sequences: None,
            min_length: None,
            max_length: None,
            length_penalty: None,
            no_repeat_ngram_size: None,
            diversity_penalty: None,
            early_stopping: None,
        };
        let usage = interactive::SessionUsage::default();
        let sampling_overrides = SamplingOverrides::default();
        let mut sampling = options;
        sampling.resolve_sampling_defaults(Some(&defaults), &sampling_overrides);

        let summary = interactive::SessionSummary {
            settings: &settings,
            execution_provider: "cpu".to_string(),
            resolved_decode_backend: EngineDecodeBackend::Ort,
            sampling,
            history: &[],
            usage: &usage,
        }
        .to_string();

        assert!(
            summary.contains(
                "sampling: max_new_tokens=32 max_context=auto temperature=0.6 top_p=0.95 top_k=20 greedy=false"
            ),
            "{summary}"
        );
    }

    #[test]
    fn session_summary_reports_loaded_provider_status() {
        let settings =
            interactive::SessionSettings::new(PathBuf::from("models/tiny"), &EngineArgs::default());
        let usage = interactive::SessionUsage::default();
        let summary = interactive::SessionSummary {
            settings: &settings,
            execution_provider: "cpu (CPU session fallback); skipped: webgpu, coreml".to_string(),
            resolved_decode_backend: EngineDecodeBackend::Ort,
            sampling: GenerateOptions::default(),
            history: &[],
            usage: &usage,
        }
        .to_string();

        assert!(
            summary.contains(
                "execution provider: cpu (CPU session fallback); skipped: webgpu, coreml"
            ),
            "{summary}"
        );
    }

    #[test]
    fn interactive_session_preserves_the_requested_native_device() {
        let args = EngineArgs {
            backend: EngineDecodeBackend::Native,
            device: Some(DeviceChoice::Cuda(Some(3))),
            ..EngineArgs::default()
        };
        let settings = interactive::SessionSettings::new(PathBuf::from("models/tiny"), &args);

        let config = settings.to_config();
        assert_eq!(config.decode_backend, EngineDecodeBackend::Native);
        // A device only reaches the engine when the native decoder is compiled
        // in; without it there is nothing that could honor one.
        #[cfg(feature = "native-backend")]
        assert_eq!(
            config.native_device,
            Some(NativeDecodeDevice::Cuda { index: Some(3) })
        );
    }

    #[test]
    fn parse_repl_line_treats_empty_and_whitespace_lines_as_empty() {
        assert_eq!(parse_repl_line("", ReplInputMode::Tty), ReplLine::Empty);
        assert_eq!(parse_repl_line(" \t ", ReplInputMode::Tty), ReplLine::Empty);
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
