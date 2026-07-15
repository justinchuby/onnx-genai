//! Native nxrt token-generation profiler using the engine's shared decode loop.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use onnx_genai_engine::{GenerateOptions, NativeDecodeDevice, NativeDecodeSession, ProcessorChain};
use onnx_genai_ort::Tokenizer;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ExecutionProvider {
    Cpu,
    Cuda,
}

#[derive(Debug, Parser)]
#[command(about = "Profile native nxrt token generation through the engine decode loop")]
struct Args {
    /// ONNX model file, or a directory containing model.onnx and tokenizer.json.
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value_t = 128)]
    tokens: usize,
    #[arg(long, default_value_t = 1)]
    warmups: usize,
    #[arg(long, default_value_t = 1)]
    runs: usize,
    #[arg(long, value_enum, default_value_t = ExecutionProvider::Cpu)]
    ep: ExecutionProvider,
    #[arg(long, default_value = "Hello")]
    prompt: String,
}

fn model_file(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.join("model.onnx")
    } else {
        path.to_path_buf()
    }
}

fn tokenizer_file(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.join("tokenizer.json")
    } else {
        path.parent()
            .unwrap_or_else(|| Path::new("."))
            .join("tokenizer.json")
    }
}

fn generate(
    session: &mut NativeDecodeSession,
    prompt_tokens: &[u32],
    tokenizer: &Tokenizer,
    tokens: usize,
) -> Result<usize> {
    let mut options = GenerateOptions::default();
    options.max_new_tokens = tokens;
    options.temperature = 0.0;
    options.greedy = true;
    options.stop_on_eos = false;
    let result = session.generate(prompt_tokens, &options, &ProcessorChain::new(), tokenizer)?;
    Ok(result.token_ids.len())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.tokens == 0 || args.runs == 0 {
        bail!("--tokens and --runs must be greater than zero");
    }
    let device = match args.ep {
        ExecutionProvider::Cpu => NativeDecodeDevice::Cpu,
        ExecutionProvider::Cuda => NativeDecodeDevice::Cuda { index: None },
    };
    let model = model_file(&args.model);
    let tokenizer = Tokenizer::from_file(tokenizer_file(&args.model))
        .context("load tokenizer.json beside native decoder")?;
    let prompt_tokens = tokenizer.encode(&args.prompt).context("tokenize prompt")?;
    if prompt_tokens.is_empty() {
        bail!("prompt tokenized to an empty sequence");
    }
    let mut session = NativeDecodeSession::load(&model, device)
        .with_context(|| format!("load native decoder {}", model.display()))?;

    println!(
        "profile_native: model={} ep={:?} tokens={} warmups={} runs={}",
        model.display(),
        args.ep,
        args.tokens,
        args.warmups,
        args.runs
    );
    for _ in 0..args.warmups {
        std::hint::black_box(generate(
            &mut session,
            &prompt_tokens,
            &tokenizer,
            args.tokens,
        )?);
    }

    let mut generated = 0usize;
    let mut elapsed = Duration::ZERO;
    for _ in 0..args.runs {
        let start = Instant::now();
        generated += generate(&mut session, &prompt_tokens, &tokenizer, args.tokens)?;
        elapsed += start.elapsed();
    }
    let tok_per_s = generated as f64 / elapsed.as_secs_f64();
    let ms_per_step = elapsed.as_secs_f64() * 1_000.0 / generated as f64;
    println!(
        "throughput: {tok_per_s:.2} tok/s, {ms_per_step:.3} ms/step \
         ({generated} generated tokens in {:.3} ms)",
        elapsed.as_secs_f64() * 1_000.0
    );
    Ok(())
}
