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

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand};
use onnx_genai::metadata::load_metadata;
use onnx_genai::ort::ModelDirectory;
use onnx_genai::{Engine, EngineConfig, GenerateOptions, GenerateRequest, GenerateToken, StopSequence};
use onnx_genai_server::{ServeArgs, from_models_dir, run_serve};

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
        options.stop_sequences = self
            .stop
            .iter()
            .cloned()
            .map(StopSequence::Text)
            .collect();
        options
    }
}

#[derive(Debug, Args)]
struct GenerateArgs {
    /// Model directory, or a config file inside it (e.g. inference_metadata.yaml).
    #[arg(long)]
    model: PathBuf,

    #[command(flatten)]
    sampling: SamplingArgs,

    /// Print generated tokens as they arrive.
    #[arg(long)]
    stream: bool,

    /// Prompt text.
    prompt: String,
}

#[derive(Debug, Args)]
struct RunArgs {
    /// Model directory, or a config file inside it (e.g. inference_metadata.yaml).
    #[arg(long)]
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Serve(args) => run_serve(args).await,
        Commands::Generate(args) => generate(args),
        Commands::Run(args) => run_repl(args),
        Commands::Show(args) => show(&args.model),
        Commands::List(args) => list(&args.models_dir),
        Commands::Version => {
            version();
            Ok(())
        }
    }
}

fn generate(args: GenerateArgs) -> anyhow::Result<()> {
    let model_dir = resolve_model_dir(&args.model);
    let options = args.sampling.to_options();
    let request = GenerateRequest {
        prompt: args.prompt.into(),
        options,
    };

    let mut engine = Engine::from_dir(&model_dir, EngineConfig::default())?;
    if args.stream {
        let mut callback = |token: GenerateToken| -> anyhow::Result<()> {
            print!("{}", token.text);
            io::stdout().flush()?;
            Ok(())
        };
        engine.generate_with_callback(request, Some(&mut callback))?;
        println!();
    } else {
        let result = engine.generate(request)?;
        println!("{}", result.text);
    }
    Ok(())
}

fn run_repl(args: RunArgs) -> anyhow::Result<()> {
    let model_dir = resolve_model_dir(&args.model);
    let mut engine = Engine::from_dir(&model_dir, EngineConfig::default())?;

    eprintln!("onnx-genai interactive session. Enter a prompt, or an empty line / Ctrl-D to exit.");
    let stdin = io::stdin();
    loop {
        print!(">>> ");
        io::stdout().flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            eprintln!();
            break;
        }
        let prompt = line.trim_end_matches(['\n', '\r']);
        if prompt.is_empty() {
            break;
        }

        let request = GenerateRequest {
            prompt: prompt.into(),
            options: args.sampling.to_options(),
        };
        let mut callback = |token: GenerateToken| -> anyhow::Result<()> {
            print!("{}", token.text);
            io::stdout().flush()?;
            Ok(())
        };
        engine.generate_with_callback(request, Some(&mut callback))?;
        println!();
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
