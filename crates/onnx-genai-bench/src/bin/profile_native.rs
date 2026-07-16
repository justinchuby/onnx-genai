//! Native nxrt token-generation profiler using the engine's shared decode loop.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use onnx_genai_bench::{fixture_path, synthetic_decoder};
use onnx_genai_engine::{GenerateOptions, NativeDecodeDevice, NativeDecodeSession, ProcessorChain};
use onnx_genai_ort::Tokenizer;
use onnx_runtime_session::InferenceSession;

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
    model: Option<PathBuf>,
    /// Build and profile the architecture-representative two-layer cached decoder.
    #[arg(long)]
    synthetic: bool,
    /// Inspection ONNX path written by --synthetic; timing uses the equivalent IR graph.
    #[arg(long, default_value = "target/native-synthetic-decoder.onnx")]
    synthetic_model_out: PathBuf,
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
) -> Result<Vec<u32>> {
    let mut options = GenerateOptions::default();
    options.max_new_tokens = tokens;
    options.temperature = 0.0;
    options.greedy = true;
    options.stop_on_eos = false;
    let result = session.generate(prompt_tokens, &options, &ProcessorChain::new(), tokenizer)?;
    Ok(result.token_ids)
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.tokens == 0 || args.runs == 0 {
        bail!("--tokens and --runs must be greater than zero");
    }
    if !args.synthetic && args.model.is_none() {
        bail!("--model is required unless --synthetic is used");
    }
    let device = match args.ep {
        ExecutionProvider::Cpu => NativeDecodeDevice::Cpu,
        ExecutionProvider::Cuda => NativeDecodeDevice::Cuda { index: None },
    };
    let model = if args.synthetic {
        if let Some(parent) = args.synthetic_model_out.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("create synthetic model directory {}", parent.display())
            })?;
        }
        synthetic_decoder::write_synthetic_decoder(&args.synthetic_model_out)
            .context("write synthetic decoder ONNX")?;
        args.synthetic_model_out.clone()
    } else {
        model_file(args.model.as_deref().expect("validated model argument"))
    };
    let tokenizer_path = if args.synthetic {
        fixture_path("tiny-gemma4-assistant").join("tokenizer.json")
    } else {
        tokenizer_file(args.model.as_deref().expect("validated model argument"))
    };
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .context("load tokenizer.json beside native decoder")?;
    let prompt_tokens = tokenizer.encode(&args.prompt).context("tokenize prompt")?;
    if prompt_tokens.is_empty() {
        bail!("prompt tokenized to an empty sequence");
    }
    let mut session = if args.synthetic {
        if !matches!(device, NativeDecodeDevice::Cpu) {
            bail!("native CUDA decode is unavailable in onnx-runtime-session");
        }
        let native = InferenceSession::from_graph(synthetic_decoder::build_synthetic_decoder())
            .context("build synthetic native session")?;
        NativeDecodeSession::from_session(native).context("wrap synthetic native decoder")?
    } else {
        NativeDecodeSession::load(&model, device)
            .with_context(|| format!("load native decoder {}", model.display()))?
    };

    println!(
        "profile_native: model={} ep={:?} layers={} prompt_tokens={prompt_tokens:?} \
         tokens={} warmups={} runs={}",
        model.display(),
        args.ep,
        session.kv_layer_count(),
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
    let mut reference_tokens = None;
    for _ in 0..args.runs {
        let start = Instant::now();
        let tokens = generate(&mut session, &prompt_tokens, &tokenizer, args.tokens)?;
        elapsed += start.elapsed();
        generated += tokens.len();
        if let Some(reference) = &reference_tokens {
            if reference != &tokens {
                bail!(
                    "native greedy decode was not deterministic: first={reference:?}, rerun={tokens:?}"
                );
            }
        } else {
            reference_tokens = Some(tokens);
        }
    }
    let tok_per_s = generated as f64 / elapsed.as_secs_f64();
    let ms_per_step = elapsed.as_secs_f64() * 1_000.0 / generated as f64;
    println!(
        "throughput: {tok_per_s:.2} tok/s, {ms_per_step:.3} ms/step \
         ({generated} generated tokens in {:.3} ms)",
        elapsed.as_secs_f64() * 1_000.0
    );
    if let Some(tokens) = reference_tokens {
        println!("generated_token_ids: {tokens:?}");
    }
    Ok(())
}
