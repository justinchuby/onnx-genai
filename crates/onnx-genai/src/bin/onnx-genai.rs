use std::{
    io::{self, Write},
    path::PathBuf,
};

use clap::{Args, Parser, Subcommand};
use onnx_genai::{
    Engine, EngineConfig, GenerateOptions, GenerateRequest, GenerateToken, StopSequence,
};

#[derive(Debug, Parser)]
#[command(
    name = "onnx-genai",
    about = "Run generative AI models with ONNX Runtime"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Generate text from a prompt.
    Generate(GenerateArgs),
}

#[derive(Debug, Args)]
struct GenerateArgs {
    /// Model directory containing the ONNX model, tokenizer, and optional metadata.
    #[arg(long)]
    model: PathBuf,

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

    /// Print generated tokens as they arrive.
    #[arg(long)]
    stream: bool,

    /// Prompt text.
    prompt: String,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate(args) => generate(args),
    }
}

fn generate(args: GenerateArgs) -> anyhow::Result<()> {
    let mut options = GenerateOptions::default();
    if let Some(max_new_tokens) = args.max_new_tokens {
        options.max_new_tokens = max_new_tokens;
    }
    if let Some(temperature) = args.temperature {
        options.temperature = temperature;
    }
    if let Some(top_p) = args.top_p {
        options.top_p = top_p;
    }
    if let Some(top_k) = args.top_k {
        options.top_k = top_k;
    }
    options.stop_sequences = args.stop.into_iter().map(StopSequence::Text).collect();

    let request = GenerateRequest {
        prompt: args.prompt.into(),
        options,
    };

    let mut engine = Engine::from_dir(&args.model, EngineConfig::default())?;

    if args.stream {
        let mut callback = |token: GenerateToken| -> anyhow::Result<()> {
            print!("{}", token.text);
            io::stdout().flush()?;
            Ok(())
        };
        engine.generate_with_callback(request, Some(&mut callback))?;
    } else {
        let result = engine.generate(request)?;
        print!("{}", result.text);
        io::stdout().flush()?;
    }

    Ok(())
}
