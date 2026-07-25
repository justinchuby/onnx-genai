//! Unified `onnx-genai` command-line interface.
//!
//! Subcommands:
//! - `serve`    — start the OpenAI-compatible HTTP server
//! - `generate` — one-shot text generation
//! - `run`      — interactive generation REPL
//! - `show`     — inspect a model's resolved files and metadata
//! - `list`     — list model directories under a models directory
//! - `version`  — print version and available execution providers
//!
//! `generate`, `run`, and `show` accept either a model directory or a config
//! file inside it (a file resolves to its parent directory).
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

use clap::{Args, Parser, Subcommand};
use onnx_genai::metadata::load_metadata;
use onnx_genai::ort::{ChatMessage, ChatRole, ChatTemplate, ModelDirectory};
use onnx_genai::{
    Engine, EngineConfig, GenerateOptions, GenerateRequest, GenerateToken, StopSequence,
};
use onnx_genai_server::{ServeArgs, from_models_dir, run_serve};

/// Process exit code for termination via SIGINT (Ctrl-C), matching the POSIX
/// convention of `128 + SIGINT`.
const EXIT_INTERRUPTED: i32 = 130;

/// Set while a generation is running so the Ctrl-C handler can distinguish an
/// interrupt during generation (soft-cancel the current turn) from an interrupt
/// at an idle prompt (exit the process).
static GENERATING: AtomicBool = AtomicBool::new(false);

/// Set by the Ctrl-C handler when a generation should be aborted. The streaming
/// callback polls this and returns [`Interrupted`] to unwind out of the engine.
static INTERRUPT_REQUESTED: AtomicBool = AtomicBool::new(false);

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

/// Install the process-wide Ctrl-C handler exactly once.
///
/// While a generation is running, Ctrl-C requests a soft-cancel of the current
/// turn (the streaming callback observes the flag and aborts). At an idle prompt
/// it exits the process cleanly with code 130, matching typical REPL semantics.
fn install_ctrlc_handler() {
    CTRLC_HANDLER.call_once(|| {
        let result = ctrlc::set_handler(|| {
            if GENERATING.load(Ordering::SeqCst) {
                INTERRUPT_REQUESTED.store(true, Ordering::SeqCst);
            } else {
                std::process::exit(EXIT_INTERRUPTED);
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

/// Run one generation turn, streaming tokens through `callback` and returning the
/// accumulated assistant text.
///
/// The interrupt flag is reset at entry so a stale Ctrl-C cannot cancel this
/// turn, and the `GENERATING` flag is held for the duration so the Ctrl-C handler
/// soft-cancels instead of exiting. A Ctrl-C during the turn surfaces as an
/// [`Interrupted`] error (recognizable via [`is_interrupt_error`]).
fn run_generation_turn(
    engine: &mut Engine,
    request: GenerateRequest,
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

    let result = engine.generate_with_callback(request, Some(&mut callback));
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

#[derive(Debug, Args)]
struct GenerateArgs {
    /// Model directory, or a config file inside it (e.g. inference_metadata.yaml).
    model: PathBuf,

    #[command(flatten)]
    sampling: SamplingArgs,

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
    let options = args.sampling.to_options();

    let template = load_chat_template(&model_dir, args.sampling.raw);
    let history = vec![ChatMessage::user(args.prompt)];
    let prompt = build_turn_prompt(template.as_ref(), &history)?;
    let request = GenerateRequest {
        prompt: prompt.into(),
        options,
    };

    let mut engine = Engine::from_dir(&model_dir, EngineConfig::default())?;
    match run_generation_turn(&mut engine, request, args.stream) {
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

fn run_repl(args: RunArgs) -> anyhow::Result<()> {
    install_ctrlc_handler();
    let model_dir = resolve_model_dir(&args.model);
    let mut engine = Engine::from_dir(&model_dir, EngineConfig::default())?;
    let mut raw_mode = args.sampling.raw;
    let mut template = load_chat_template(&model_dir, raw_mode);

    eprintln!(
        "onnx-genai interactive session. Enter a prompt, or an empty line / Ctrl-D to exit.\n\
         Ctrl-C aborts the current generation; press it again at the prompt to exit."
    );

    // Multi-turn conversation history. Each turn appends the user message, the
    // full history is rendered through the chat template, and the assistant's
    // reply is appended so later turns retain context. In raw mode there is no
    // template so only the latest user message is sent.
    let mut history: Vec<ChatMessage> = Vec::new();
    let mut image_attachments: Vec<PathBuf> = Vec::new();
    let mut audio_attachments: Vec<PathBuf> = Vec::new();
    let stdin = io::stdin();
    loop {
        print!(">>> ");
        io::stdout().flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            eprintln!();
            break;
        }
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
                if let Some(path) = path {
                    let path = PathBuf::from(path);
                    if path.exists() {
                        image_attachments.push(path);
                    } else {
                        eprintln!("warning: image path does not exist: {}", path.display());
                    }
                } else {
                    eprintln!("usage: /image <path> [prompt text]");
                }
                prompt
            }
            ReplLine::Command(ReplCommand::Audio { path, prompt }) => {
                if let Some(path) = path {
                    let path = PathBuf::from(path);
                    if path.exists() {
                        audio_attachments.push(path);
                    } else {
                        eprintln!("warning: audio path does not exist: {}", path.display());
                    }
                } else {
                    eprintln!("usage: /audio <path> [prompt text]");
                }
                prompt
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
        if !staged_images.is_empty() {
            let paths = staged_images
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            println!(
                "⚠ image input staged ({}) but multimodal execution is not yet wired — sending text only for now.",
                paths
            );
        }
        if !staged_audio.is_empty() {
            let paths = staged_audio
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            println!(
                "⚠ audio input staged ({}) but multimodal execution is not yet wired — sending text only for now.",
                paths
            );
        }
        let rendered = build_turn_prompt(template.as_ref(), &history)?;
        let request = GenerateRequest {
            prompt: rendered.into(),
            options: args.sampling.to_options(),
        };

        match run_generation_turn(&mut engine, request, true) {
            Ok(output) => {
                println!();
                history.push(ChatMessage::assistant(output));
            }
            Err(error) if is_interrupt_error(&error) => {
                // Drop the interrupted turn from history so a partial/aborted
                // reply never pollutes the conversation context, then return to
                // the prompt instead of exiting.
                eprintln!("\n^C interrupted");
                history.pop();
            }
            Err(error) => return Err(error),
        }
    }
    Ok(())
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
