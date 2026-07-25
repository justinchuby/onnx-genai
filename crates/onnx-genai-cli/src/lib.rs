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
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Context as _;
use clap::{Args, Parser, Subcommand};
use onnx_genai::engine::{PipelineEngine, PipelineGenerateRequest};
use onnx_genai::metadata::load_metadata;
use onnx_genai::ort::{ChatMessage, ChatRole, ChatTemplate, ModelDirectory, Tokenizer};
use onnx_genai::text_to_audio::{self, TextToAudioRequest};
use onnx_genai::text_to_image::{self, TextToImageRequest, VaeDecoder};
use onnx_genai::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest, GenerateToken,
    GenerateTokenCallback, StopSequence,
};
use onnx_genai_server::multimodal::{self, MultimodalInput, MultimodalSpecs};
use onnx_genai_server::{ServeArgs, from_models_dir, run_serve};

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

/// Load the model's chat template unless `raw` is set. On a load failure this
/// degrades gracefully: it prints a one-line warning and returns `None`, causing
/// callers to fall back to the raw (untemplated) prompt rather than crash.
fn load_chat_template(model_dir: &Path, raw: bool) -> Option<ChatTemplate> {
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

/// Build the prompt string sent to the engine for the current turn.
///
/// With a chat `template`, the full `history` (all prior turns plus the current
/// user message) is rendered with `add_generation_prompt=true` so the model
/// continues as the assistant. Without a template (raw mode / load failure) the
/// last message's content is sent verbatim.
fn build_turn_prompt(
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

impl Backend {
    /// Load `model_dir`, preferring its declared pipeline when it has one.
    fn load(model_dir: &Path) -> anyhow::Result<Self> {
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
                let engine = Engine::from_pipeline_dir(model_dir, EngineConfig::default())?;
                Ok(Self::Pipeline(Box::new(PipelineBackend {
                    engine,
                    tokenizer,
                    multimodal: setup.multimodal,
                })))
            }
            None => Ok(Self::Text(Box::new(Engine::from_dir(
                model_dir,
                EngineConfig::default(),
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
    fn generate(
        &mut self,
        turn: TurnInput,
        callback: &mut GenerateTokenCallback<'_>,
    ) -> anyhow::Result<()> {
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
                engine.generate_with_callback(request, Some(callback))?;
                Ok(())
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
                    })?;
                Ok(())
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

/// Run one generation turn, streaming tokens through the terminal and returning
/// the accumulated assistant text.
///
/// The interrupt flag is reset at entry so a stale Ctrl-C cannot cancel this
/// turn, and the `GENERATING` flag is held for the duration so the Ctrl-C handler
/// soft-cancels instead of exiting. A Ctrl-C during the turn surfaces as an
/// [`Interrupted`] error (recognizable via [`is_interrupt_error`]).
fn run_generation_turn(
    backend: &mut Backend,
    turn: TurnInput,
    stream: bool,
) -> anyhow::Result<String> {
    INTERRUPT_REQUESTED.store(false, Ordering::SeqCst);
    GENERATING.store(true, Ordering::SeqCst);

    let mut output = String::new();
    let mut callback = |token: GenerateToken| -> anyhow::Result<()> {
        if INTERRUPT_REQUESTED.load(Ordering::SeqCst) {
            return Err(anyhow::Error::new(Interrupted));
        }
        output.push_str(&token.text);
        if stream {
            print!("{}", token.text);
            io::stdout().flush()?;
        }
        Ok(())
    };

    let result = backend.generate(turn, &mut callback);
    GENERATING.store(false, Ordering::SeqCst);
    result?;
    Ok(output)
}

#[derive(Debug, PartialEq, Eq)]
enum ReplCommand {
    Help,
    Reset,
    ToggleRaw,
    System(Option<String>),
    Image {
        path: Option<String>,
        prompt: Option<String>,
    },
    Audio {
        path: Option<String>,
        prompt: Option<String>,
    },
    Unknown(String),
}

#[derive(Debug, PartialEq, Eq)]
enum ReplLine {
    Command(ReplCommand),
    Prompt(String),
    Empty,
}

fn parse_repl_line(line: &str) -> ReplLine {
    if line.trim().is_empty() {
        return ReplLine::Empty;
    }
    let Some(command_line) = line.strip_prefix('/') else {
        return ReplLine::Prompt(line.to_string());
    };

    let mut parts = command_line.splitn(2, char::is_whitespace);
    let command = parts.next().unwrap_or_default();
    let arguments = parts.next().unwrap_or_default().trim();
    let attachment_command = |is_image| {
        let mut attachment_parts = arguments.splitn(2, char::is_whitespace);
        let path = attachment_parts
            .next()
            .filter(|path| !path.is_empty())
            .map(ToString::to_string);
        let prompt = attachment_parts
            .next()
            .map(str::trim)
            .filter(|prompt| !prompt.is_empty())
            .map(ToString::to_string);
        if is_image {
            ReplCommand::Image { path, prompt }
        } else {
            ReplCommand::Audio { path, prompt }
        }
    };

    let command = match command {
        "help" => ReplCommand::Help,
        "reset" => ReplCommand::Reset,
        "raw" => ReplCommand::ToggleRaw,
        "system" => ReplCommand::System((!arguments.is_empty()).then(|| arguments.to_string())),
        "image" => attachment_command(true),
        "audio" => attachment_command(false),
        _ => ReplCommand::Unknown(format!("/{command}")),
    };
    ReplLine::Command(command)
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
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Start an OpenAI-compatible HTTP server.
    Serve(ServeArgs),
    /// Generate text from a single prompt and exit.
    Generate(GenerateArgs),
    /// Start an interactive generation REPL (one prompt per line).
    Run(RunArgs),
    /// Show a model's resolved files and metadata.
    Show(ShowArgs),
    /// List model directories under a models directory.
    #[command(alias = "ls")]
    List(ListArgs),
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
    image_output: ImageOutputArgs,

    #[command(flatten)]
    audio_output: AudioOutputArgs,

    /// Print generated tokens as they arrive.
    #[arg(long)]
    stream: bool,

    /// Prompt text.
    #[arg(long)]
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
    match cli.command {
        Commands::Serve(serve_args) => {
            let runtime = tokio::runtime::Runtime::new()?;
            runtime.block_on(run_serve(serve_args))
        }
        Commands::Generate(generate_args) => generate(generate_args),
        Commands::Run(run_args) => run_repl(run_args),
        Commands::Show(show_args) => show(&show_args.model),
        Commands::List(list_args) => list(&list_args.models_dir),
        Commands::Version => {
            version();
            Ok(())
        }
    }
}

fn generate(args: GenerateArgs) -> anyhow::Result<()> {
    install_ctrlc_handler();
    let model_dir = resolve_model_dir(&args.model);
    if args.image_output.output_image.is_some() && args.audio_output.output_audio.is_some() {
        anyhow::bail!(
            "What: --output-image and --output-audio were combined. \
             Why: one invocation produces one kind of output. \
             How: run the command once per output."
        );
    }
    if args.image_output.output_image.is_some() {
        return generate_image(&model_dir, args);
    }
    if args.audio_output.output_audio.is_some() {
        return generate_audio(&model_dir, args);
    }
    let options = args.sampling.to_options();

    let template = load_chat_template(&model_dir, args.sampling.raw);
    let history = vec![ChatMessage::user(args.prompt)];
    let prompt = build_turn_prompt(template.as_ref(), &history)?;
    let turn = TurnInput {
        prompt,
        images: args.attachments.images.clone(),
        audio: args.attachments.audio.clone(),
        options,
    };

    let mut backend = Backend::load(&model_dir)?;
    match run_generation_turn(&mut backend, turn, args.stream) {
        Ok(output) => {
            if args.stream {
                println!();
            } else {
                println!("{output}");
            }
            Ok(())
        }
        Err(error) if is_interrupt_error(&error) => {
            // A Ctrl-C during a one-shot generation aborts and exits non-zero.
            eprintln!("\n^C interrupted");
            std::process::exit(EXIT_INTERRUPTED);
        }
        Err(error) => Err(error),
    }
}

/// Render `--prompt` to PNG(s) through the model's declared diffusion pipeline.
fn generate_image(model_dir: &Path, args: GenerateArgs) -> anyhow::Result<()> {
    let output = args
        .image_output
        .output_image
        .clone()
        .expect("image output path checked by the caller");
    let request = args.image_output.to_request(args.prompt.clone());
    let mut engine = PipelineEngine::from_dir_with_config(model_dir, EngineConfig::default())
        .map_err(|error| {
            anyhow::anyhow!(
                "What: {} could not be loaded as a diffusion pipeline. \
                 Why: {error:#}. \
                 How: point --output-image at a package whose inference metadata declares a `pipeline` with `strategy.kind: iterative`.",
                model_dir.display()
            )
        })?;

    let images = text_to_image::render(model_dir, &mut engine, &request)?;
    if images.is_empty() {
        anyhow::bail!(
            "What: no image was produced. \
             Why: the pipeline returned fewer images than the requested batch size of {}. \
             How: render with --batch-size 1, or report this as a pipeline output-shape bug.",
            request.batch_size
        );
    }
    for (index, image) in images.iter().enumerate() {
        let path = if images.len() == 1 {
            output.clone()
        } else {
            let stem = output
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_else(|| "out".to_string());
            let extension = output
                .extension()
                .map(|extension| extension.to_string_lossy().into_owned())
                .unwrap_or_else(|| "png".to_string());
            output.with_file_name(format!("{stem}_{index}.{extension}"))
        };
        text_to_image::save_png(image, &path)?;
        println!(
            "saved {} ({}x{})",
            path.display(),
            image.width,
            image.height
        );
    }
    Ok(())
}

/// Synthesize `--prompt` to a WAV file through the model's declared TTS pipeline.
fn generate_audio(model_dir: &Path, args: GenerateArgs) -> anyhow::Result<()> {
    let output = args
        .audio_output
        .output_audio
        .clone()
        .expect("audio output path checked by the caller");
    let setup = multimodal::load(model_dir)?.with_context(|| {
        format!(
            "What: {} could not be loaded as a speech pipeline. \
             Why: it declares no `pipeline`, so it has no vocoder stage to run. \
             How: point --output-audio at a text-to-speech package.",
            model_dir.display()
        )
    })?;
    let tokenizer = Tokenizer::from_file(&setup.tokenizer_path).map_err(|error| {
        anyhow::anyhow!(
            "What: the package's tokenizer could not be loaded from {}. \
             Why: {error}. \
             How: verify the package ships a valid tokenizer.json.",
            setup.tokenizer_path.display()
        )
    })?;
    let mut engine = PipelineEngine::from_dir_with_config(model_dir, EngineConfig::default())?;

    let request = args
        .audio_output
        .to_request(args.prompt.clone(), &args.sampling);
    let audio = text_to_audio::synthesize(&mut engine, &tokenizer, &request)?;
    text_to_audio::save_wav(&audio, &output)?;
    println!(
        "saved {} ({:.2}s, {} Hz, {} channel{})",
        output.display(),
        audio.duration_secs(),
        audio.sample_rate,
        audio.channels,
        if audio.channels == 1 { "" } else { "s" }
    );
    Ok(())
}

fn run_repl(args: RunArgs) -> anyhow::Result<()> {
    install_ctrlc_handler();
    let model_dir = resolve_model_dir(&args.model);
    let mut backend = Backend::load(&model_dir)?;
    let mut raw_mode = args.sampling.raw;
    let mut template = load_chat_template(&model_dir, raw_mode);

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
                    "/help\n/reset\n/raw\n/system <text>\n/image <path> [prompt text]\n/audio <path> [prompt text]"
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
            ReplLine::Command(ReplCommand::ToggleRaw) => {
                raw_mode = !raw_mode;
                template = load_chat_template(&model_dir, raw_mode);
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

        match run_generation_turn(&mut backend, turn, true) {
            Ok(output) => {
                println!();
                history.push(ChatMessage::assistant(output));
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

fn display_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn show(model: &Path) -> anyhow::Result<()> {
    let model_dir = resolve_model_dir(model);
    let directory = ModelDirectory::load(&model_dir)?;

    println!("model directory: {}", directory.root.display());
    println!("model file:      {}", directory.model_path.display());
    println!("tokenizer:       {}", directory.tokenizer_path.display());
    match &directory.metadata_path {
        Some(path) => println!("metadata:        {}", path.display()),
        None => println!("metadata:        (none)"),
    }
    let genai_config = model_dir.join("genai_config.json");
    if genai_config.is_file() {
        println!("genai config:    {}", genai_config.display());
    }
    if directory.speculator.is_some() {
        println!("speculator:      detected");
    }

    if let Some(metadata_path) = &directory.metadata_path {
        let metadata = load_metadata(metadata_path)?;
        if !metadata.required_capabilities.is_empty() {
            println!(
                "capabilities:    {}",
                metadata.required_capabilities.join(", ")
            );
        }
        if let Some(model_caps) = &metadata.model {
            if let Some(max_len) = model_caps.max_sequence_length {
                println!("max sequence:    {max_len}");
            }
            if let Some(attention) = &model_caps.attention {
                println!("attention:       {attention:?}");
            }
        }
        if let Some(quantization) = &metadata.quantization {
            println!("quantization:    {quantization:?}");
        }
    }
    Ok(())
}

fn list(models_dir: &Path) -> anyhow::Result<()> {
    let specs = from_models_dir(models_dir)?;
    if specs.is_empty() {
        println!("no models found under {}", models_dir.display());
        return Ok(());
    }
    for spec in specs {
        println!("{}\t{}", spec.id, spec.path.display());
    }
    Ok(())
}

fn version() {
    println!("onnx-genai {}", env!("CARGO_PKG_VERSION"));
    let mut providers = vec!["cpu"];
    if cfg!(feature = "cuda") {
        providers.push("cuda");
    }
    println!("execution providers: {}", providers.join(", "));
    println!("select an execution provider at runtime with ONNX_GENAI_EP (e.g. cpu, cuda).");
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
