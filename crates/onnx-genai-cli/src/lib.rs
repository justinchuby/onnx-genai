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

use std::io::{self, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

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
use commands::parse_decode_backend;
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
use onnx_genai::engine::EngineDecodeBackend;
use onnx_genai::engine::native_decode_device::NativeDecodeDevice;
use onnx_genai::text_to_audio::TextToAudioRequest;
use onnx_genai::text_to_image::{TextToImageRequest, VaeDecoder};
use onnx_genai::{EngineConfig, GenerateOptions, SamplingOverrides, StopSequence};
use onnx_genai_server::{ServeArgs, run_serve};
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

/// Shared engine-tuning flags.
#[derive(Debug, Args, Default, Clone)]
struct EngineArgs {
    /// Decoder backend for text generation.
    #[arg(long, value_name = "auto|ort|native", value_parser = parse_decode_backend, default_value = "auto")]
    backend: EngineDecodeBackend,

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

    /// Device the native decode backend runs on: `cpu`, `cuda`, `cuda:N`, or
    /// `auto`.
    ///
    /// `auto` (the default) takes the device from the model's declared execution
    /// providers. Most exported models declare none, which resolves to the CPU —
    /// so on a machine with a GPU, `--backend native` alone will still run on the
    /// CPU unless you say `--device cuda` (#1064). Ignored by the ORT backend,
    /// which selects providers from the model's own session options.
    #[arg(long, value_name = "auto|cpu|cuda[:N]", value_parser = parse_native_device)]
    device: Option<NativeDeviceChoice>,
}

/// A `--device` value. `Auto` is distinct from an absent flag only in intent;
/// both defer to the model's declared providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeDeviceChoice {
    Auto,
    Cpu,
    Cuda(Option<u32>),
}

/// Parse a `--device` value.
///
/// Refuses anything else rather than silently falling back: a user who asks for
/// a device they cannot have should be told, not quietly given the CPU. That
/// silent fallback is exactly what #1064 documents.
fn parse_native_device(input: &str) -> Result<NativeDeviceChoice, String> {
    let raw = input.trim();
    let lowered = raw.to_ascii_lowercase();
    match lowered.as_str() {
        "auto" => Ok(NativeDeviceChoice::Auto),
        "cpu" => Ok(NativeDeviceChoice::Cpu),
        "cuda" | "gpu" => Ok(NativeDeviceChoice::Cuda(None)),
        _ => match lowered.strip_prefix("cuda:") {
            Some(index) => index
                .parse::<u32>()
                .map(|index| NativeDeviceChoice::Cuda(Some(index)))
                .map_err(|_| {
                    format!("'{raw}' is not a valid device: expected a CUDA index, as in 'cuda:0'")
                }),
            None => Err(format!(
                "'{raw}' is not a valid device: expected 'auto', 'cpu', 'cuda', or 'cuda:N'"
            )),
        },
    }
}

impl EngineArgs {
    fn to_config(&self) -> EngineConfig {
        let mut config = EngineConfig {
            decode_backend: self.backend,
            ..EngineConfig::default()
        };
        if let Some(limit) = self.vram_limit {
            config.limits.vram_limit = limit;
        }
        if let Some(limit) = self.host_ram_limit {
            config.limits.host_ram_limit = limit;
        }
        match self.device {
            None | Some(NativeDeviceChoice::Auto) => {}
            Some(NativeDeviceChoice::Cpu) => {
                config.native_device = Some(NativeDecodeDevice::Cpu);
            }
            Some(NativeDeviceChoice::Cuda(index)) => {
                config.native_device = Some(NativeDecodeDevice::Cuda { index });
            }
        }
        config
    }
}

fn decode_backend_name(backend: EngineDecodeBackend) -> &'static str {
    match backend {
        EngineDecodeBackend::Auto => "auto",
        EngineDecodeBackend::Ort => "ort",
        EngineDecodeBackend::Native => "native",
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
            source_image: None,
            vae_encoder: None,
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

    /// Do not show compact per-turn stats by default in an interactive terminal.
    #[arg(long)]
    no_stats: bool,

    /// Prompt text. May also be given positionally after the model
    /// (`generate MODEL "your prompt"`).
    #[arg(long = "prompt", short = 'p', value_name = "PROMPT")]
    prompt_flag: Option<String>,

    /// Prompt text, given positionally.
    #[arg(value_name = "PROMPT")]
    prompt_positional: Option<String>,

    /// The prompt resolved from either spelling. Not parsed from the command
    /// line; filled by [`GenerateArgs::resolve_prompt`] before use.
    #[arg(skip)]
    prompt: String,
}

impl GenerateArgs {
    /// Accept the prompt positionally or as a flag, and fill [`Self::prompt`].
    ///
    /// `model` is positional while `prompt` was flag-only, so the two required
    /// arguments of the same subcommand disagreed about how to be passed. That
    /// asymmetry has no reason behind it and is not memorable: the coordinator
    /// wrote `--model X --prompt Y` in a measurement runbook handed to the
    /// repository owner, and clap rejected it.
    ///
    /// Supplying both spellings is an error rather than a silent precedence
    /// rule, because someone who typed two different prompts has no preference
    /// for us to guess at.
    fn resolve_prompt(&mut self) -> Result<(), String> {
        self.prompt = match (self.prompt_flag.take(), self.prompt_positional.take()) {
            (Some(_), Some(_)) => {
                return Err(
                    "What: the prompt was given twice, once with --prompt/-p and once \
                     positionally. Why: only one prompt can be generated from. How: drop \
                     either the flag or the positional argument."
                        .to_string(),
                );
            }
            (Some(prompt), None) | (None, Some(prompt)) => prompt,
            (None, None) => {
                return Err(
                    "What: no prompt was given. Why: `generate` needs text to continue. \
                     How: pass it positionally (`generate MODEL \"your prompt\"`) or as \
                     `--prompt \"your prompt\"`."
                        .to_string(),
                );
            }
        };
        Ok(())
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
                // INFO spans can arrive between streamed tokens and visually
                // split a reply on a shared terminal. Keep normal interactive
                // output stable; operators can still opt in with RUST_LOG=info.
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
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
        Commands::Generate(mut generate_args) => {
            generate_args
                .resolve_prompt()
                .map_err(|message| anyhow::anyhow!(message))?;
            generate(*generate_args, &profiling)
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
    fn generate_accepts_positional_model_and_prompt_flag() {
        let parsed_command_line =
            Cli::try_parse_from(["onnx-genai", "generate", "./m", "--prompt", "hi"]).unwrap();

        match parsed_command_line.command {
            Commands::Generate(mut args) => {
                args.resolve_prompt().expect("flag prompt resolves");
                assert_eq!(args.model, PathBuf::from("./m"));
                assert_eq!(args.prompt, "hi");
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn generate_accepts_a_positional_prompt() {
        // `model` was positional while `prompt` was flag-only, so the two
        // required arguments of one subcommand disagreed about how to be
        // passed. That cost a real user a failed command, so both spellings
        // work now.
        let parsed_command_line =
            Cli::try_parse_from(["onnx-genai", "generate", "./m", "hi"]).unwrap();

        match parsed_command_line.command {
            Commands::Generate(mut args) => {
                args.resolve_prompt().expect("positional prompt resolves");
                assert_eq!(args.model, PathBuf::from("./m"));
                assert_eq!(args.prompt, "hi");
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn generate_rejects_a_prompt_given_twice() {
        // Two different prompts is a mistake with no correct precedence rule,
        // so it is an error rather than a silent choice.
        let parsed_command_line =
            Cli::try_parse_from(["onnx-genai", "generate", "./m", "positional", "-p", "flag"])
                .unwrap();

        match parsed_command_line.command {
            Commands::Generate(mut args) => {
                let error = args
                    .resolve_prompt()
                    .expect_err("two prompts must be rejected");
                assert!(error.contains("twice"), "{error}");
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn generate_accepts_prompt_short_flag() {
        let parsed_command_line =
            Cli::try_parse_from(["onnx-genai", "generate", "./m", "-p", "hi"]).unwrap();

        match parsed_command_line.command {
            Commands::Generate(mut args) => {
                args.resolve_prompt().expect("short-flag prompt resolves");
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
    fn cli_uses_finite_fallback_until_a_model_context_is_known() {
        let parsed =
            Cli::try_parse_from(["onnx-genai", "generate", "./m", "--prompt", "hi"]).unwrap();

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
            "--prompt",
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
            Cli::try_parse_from([
                "onnx-genai",
                "generate",
                "./m",
                "--prompt",
                "hi",
                "--max-context",
                "0",
            ])
            .is_err()
        );
    }

    #[test]
    fn backend_is_shared_by_generate_run_and_transcribe() {
        let generate = Cli::try_parse_from([
            "onnx-genai",
            "generate",
            "./m",
            "--prompt",
            "hi",
            "--backend",
            "native",
        ])
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
        let parsed =
            Cli::try_parse_from(["onnx-genai", "generate", "./m", "--prompt", "hi"]).unwrap();

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
        let error = Cli::try_parse_from([
            "onnx-genai",
            "generate",
            "./m",
            "--prompt",
            "hi",
            "--backend",
            "cuda",
        ])
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
            "--prompt",
            "hi",
            "--temperature",
            "0.7",
        ])
        .unwrap();
        let top_p = Cli::try_parse_from(["onnx-genai", "run", "./m", "--top-p", "0.9"]).unwrap();
        let top_k = Cli::try_parse_from([
            "onnx-genai",
            "generate",
            "./m",
            "--prompt",
            "hi",
            "--top-k",
            "40",
        ])
        .unwrap();
        let temperature_zero = Cli::try_parse_from([
            "onnx-genai",
            "generate",
            "./m",
            "--prompt",
            "hi",
            "--temperature",
            "0",
        ])
        .unwrap();
        let forced_greedy = Cli::try_parse_from([
            "onnx-genai",
            "generate",
            "./m",
            "--prompt",
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
    fn generate_requires_a_prompt_in_either_spelling() {
        // Parsing now succeeds with neither spelling present -- the prompt is
        // resolved afterwards -- so the rejection moved from clap to
        // `resolve_prompt`, and the message has to carry its own weight.
        let parsed = Cli::try_parse_from(["onnx-genai", "generate", "./m"]).unwrap();
        match parsed.command {
            Commands::Generate(mut args) => {
                let error = args
                    .resolve_prompt()
                    .expect_err("a missing prompt must be rejected");
                assert!(error.contains("no prompt was given"), "{error}");
                assert!(error.contains("positionally"), "{error}");
            }
            _ => panic!("expected generate command"),
        }
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
        let parsed_command_line = Cli::try_parse_from([
            "onnx-genai",
            "generate",
            "./m",
            "--prompt",
            "hi",
            "--no-stats",
        ])
        .unwrap();

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
            Commands::Generate(mut args) => {
                args.resolve_prompt().expect("prompt resolves");
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
            Commands::Generate(mut args) => {
                args.resolve_prompt().expect("prompt resolves");
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
    fn generate_profile_provider_comes_from_live_command_profile() -> anyhow::Result<()> {
        let model = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm")
            .canonicalize()?;
        let dir = temp_dir("generate-live-profile");
        let profile = dir.join("profile.json");

        run(vec![
            "onnx-genai".to_string(),
            "--profile-json".to_string(),
            profile.display().to_string(),
            "generate".to_string(),
            model.display().to_string(),
            "--prompt".to_string(),
            "hi".to_string(),
            "--max-new-tokens".to_string(),
            "1".to_string(),
            "--no-stats".to_string(),
            "--cpu-cores".to_string(),
            "1".to_string(),
        ])?;

        assert_eq!(profile_execution_provider(&profile)?, "cpu");
        fs::remove_dir_all(dir).unwrap();
        Ok(())
    }

    #[test]
    fn transcribe_profile_provider_comes_from_live_command_profile() -> anyhow::Result<()> {
        let model = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-whisper")
            .canonicalize()?;
        let audio = model.join("tiny.wav");
        let dir = temp_dir("transcribe-live-profile");
        let profile = dir.join("profile.json");

        run(vec![
            "onnx-genai".to_string(),
            "--profile-json".to_string(),
            profile.display().to_string(),
            "transcribe".to_string(),
            model.display().to_string(),
            audio.display().to_string(),
            "--format".to_string(),
            "json".to_string(),
            "--cpu-cores".to_string(),
            "1".to_string(),
        ])?;

        assert_eq!(profile_execution_provider(&profile)?, "cpu");
        fs::remove_dir_all(dir).unwrap();
        Ok(())
    }

    fn profile_execution_provider(path: &Path) -> anyhow::Result<String> {
        let value: serde_json::Value = serde_json::from_str(&fs::read_to_string(path)?)?;
        Ok(value["execution_provider"]
            .as_str()
            .expect("profile must include execution_provider")
            .to_string())
    }

    #[test]
    fn a_toggle_reports_when_given_nothing_and_refuses_nonsense() {}

    /// `--device` exists because `--backend native` alone silently ran on the CPU
    /// on a GPU machine: device selection came only from the model's declared
    /// execution providers, and typical exports declare none (#1064). Measured
    /// 0 MiB of GPU memory across a whole native run before this flag existed.
    #[test]
    fn a_device_can_be_asked_for_explicitly_and_nonsense_is_refused() {
        assert_eq!(parse_native_device("auto"), Ok(NativeDeviceChoice::Auto));
        assert_eq!(parse_native_device("cpu"), Ok(NativeDeviceChoice::Cpu));
        assert_eq!(
            parse_native_device("cuda"),
            Ok(NativeDeviceChoice::Cuda(None))
        );
        assert_eq!(
            parse_native_device("CUDA:1"),
            Ok(NativeDeviceChoice::Cuda(Some(1))),
            "device names are case-insensitive"
        );

        // Refused rather than quietly resolved to the CPU: a silent fallback is
        // the behaviour this flag exists to end.
        let error = parse_native_device("cuda:x").expect_err("not a device index");
        assert!(error.contains("cuda:0"), "{error}");
        let error = parse_native_device("tpu").expect_err("not a device");
        assert!(
            error.contains("'auto', 'cpu', 'cuda', or 'cuda:N'"),
            "{error}"
        );
    }

    /// The flag has to reach `EngineConfig`, not merely parse. Absent and `auto`
    /// both leave the engine's own resolution untouched.
    #[test]
    fn the_device_flag_reaches_the_engine_config() {
        let args = EngineArgs::default();
        assert!(
            args.to_config().native_device.is_none(),
            "an absent --device must not override the model's declared providers"
        );

        let auto = EngineArgs {
            device: Some(NativeDeviceChoice::Auto),
            ..EngineArgs::default()
        };
        assert!(auto.to_config().native_device.is_none());

        let cuda = EngineArgs {
            device: Some(NativeDeviceChoice::Cuda(Some(1))),
            ..EngineArgs::default()
        };
        assert_eq!(
            cuda.to_config().native_device,
            Some(NativeDecodeDevice::Cuda { index: Some(1) })
        );

        let cpu = EngineArgs {
            device: Some(NativeDeviceChoice::Cpu),
            ..EngineArgs::default()
        };
        assert_eq!(cpu.to_config().native_device, Some(NativeDecodeDevice::Cpu));
    }

    #[test]
    fn interactive_session_preserves_the_requested_native_device() {
        let args = EngineArgs {
            backend: EngineDecodeBackend::Native,
            device: Some(NativeDeviceChoice::Cuda(Some(3))),
            ..EngineArgs::default()
        };
        let settings = interactive::SessionSettings::new(PathBuf::from("models/tiny"), &args);

        let config = settings.to_config();
        assert_eq!(config.decode_backend, EngineDecodeBackend::Native);
        assert_eq!(
            config.native_device,
            Some(NativeDecodeDevice::Cuda { index: Some(3) })
        );
    }

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
