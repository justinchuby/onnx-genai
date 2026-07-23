//! Native nxrt token-generation profiler using the engine's shared decode loop.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use onnx_genai_bench::{fixture_path, synthetic_decoder};
use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GenerateOptions, GenerateRequest,
    NativeDecodeDevice, NativeDecodeSession, PipelineEngine, PipelineGenerateRequest,
    ProcessorChain,
};
use onnx_genai_engine::logits::{MinPProcessor, RepetitionPenaltyProcessor};
use onnx_genai_ort::{Tokenizer, available_execution_providers};
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
    /// Load a metadata-declared multi-model pipeline instead of a single decoder.
    #[arg(long)]
    pipeline: bool,
    /// Inspection ONNX path written by --synthetic; timing uses the equivalent IR graph.
    #[arg(long, default_value = "target/native-synthetic-decoder.onnx")]
    synthetic_model_out: PathBuf,
    #[arg(long, default_value_t = 128)]
    tokens: usize,
    #[arg(long, default_value_t = 1)]
    warmups: usize,
    #[arg(long, default_value_t = 1)]
    runs: usize,
    /// Time steady decode from token callbacks, excluding the first N emitted
    /// tokens so prefill, eager warmup, and graph capture are outside the window.
    #[arg(long)]
    steady: bool,
    #[arg(long, default_value_t = 8)]
    decode_skip: usize,
    #[arg(long, value_enum, default_value_t = ExecutionProvider::Cpu)]
    ep: ExecutionProvider,
    #[arg(long, default_value = "Hello")]
    prompt: String,
    /// When set, capture an `onnx-runtime-tracer` timeline of a single traced
    /// generation and write it as Chrome JSON to this path. Surfaces the per-op
    /// executor spans with `kernel_variant` / `capture_status` fields. Tracing
    /// is left OFF for the timed warmup/measurement runs so throughput is
    /// unaffected.
    #[arg(long)]
    trace: Option<PathBuf>,
    /// Dump native token-0 top-K log-probabilities (log-softmax) as JSON to this
    /// path for a single-token greedy forward, then exit. Used to bisect
    /// native-vs-ORT logit divergence.
    #[arg(long)]
    dump_logprobs: Option<PathBuf>,
    #[arg(long, default_value_t = 40)]
    logprobs_k: usize,
    /// Override the text prompt with an explicit JSON array of token ids (e.g.
    /// "[9707, 12824, 13]"). Enables exact teacher-forced logit comparison
    /// against ORT without tokenizer round-trip drift. Only honored with
    /// --dump-logprobs.
    #[arg(long)]
    prompt_ids: Option<PathBuf>,
    /// HF-style repetition penalty applied host-side to the output logits before
    /// token selection (divides positive / multiplies negative logits of tokens
    /// already in the prompt+generated stream). Default 1.0 is OFF and keeps the
    /// captured device-argmax greedy fast path byte-identical.
    #[arg(long, default_value_t = 1.0)]
    repetition_penalty: f32,
    /// Optional window: only penalize the most recent N tokens of the combined
    /// prompt+generated stream. Unset penalizes the whole history.
    #[arg(long)]
    repetition_window: Option<usize>,
    /// Min-p nucleus threshold (relative to the top token's probability). Default
    /// 0.0 is OFF. Only affects categorical (non-greedy) sampling.
    #[arg(long, default_value_t = 0.0)]
    min_p: f32,
}

/// Whether any host-side sampling policy (penalty / min-p) is enabled. When
/// false the decode path is byte-identical to the default greedy benchmark and,
/// on CUDA, keeps the captured device-argmax fast path.
fn sampling_enabled(args: &Args) -> bool {
    args.repetition_penalty != 1.0 || args.min_p > 0.0
}

/// Copy the CLI sampling policy onto generation options (default values are
/// no-ops, preserving existing greedy behavior exactly).
fn apply_sampling_options(options: &mut GenerateOptions, args: &Args) {
    options.repetition_penalty = args.repetition_penalty;
    options.repetition_window = args.repetition_window;
    options.min_p = args.min_p;
}

/// Build the host-side processor chain from the CLI sampling policy. Empty when
/// sampling is OFF, so the greedy fast path stays armed.
fn sampling_chain(args: &Args) -> ProcessorChain {
    let mut chain = ProcessorChain::new();
    if args.repetition_penalty != 1.0 {
        chain.add(Box::new(RepetitionPenaltyProcessor {
            penalty: args.repetition_penalty,
            window: args.repetition_window,
        }));
    }
    if args.min_p > 0.0 {
        chain.add(Box::new(MinPProcessor { min_p: args.min_p }));
    }
    chain
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
    args: &Args,
) -> Result<Vec<u32>> {
    let mut options = GenerateOptions {
        max_new_tokens: tokens,
        temperature: 0.0,
        greedy: true,
        stop_on_eos: false,
        ..GenerateOptions::default()
    };
    apply_sampling_options(&mut options, args);
    // Empty when sampling is OFF, so the greedy device fast path stays armed;
    // otherwise the penalty/min-p run host-side on the output logits, outside
    // the captured graph replay.
    let chain = sampling_chain(args);
    let result = session.generate(prompt_tokens, &options, &chain, tokenizer)?;
    Ok(result.token_ids)
}

fn request(args: &Args, tokens: usize) -> GenerateRequest {
    let mut request = GenerateRequest::new(args.prompt.clone());
    request.options.max_new_tokens = tokens;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    apply_sampling_options(&mut request.options, args);
    request
}

fn pipeline_request(args: &Args, tokens: usize) -> PipelineGenerateRequest {
    PipelineGenerateRequest::new(request(args, tokens))
}

fn describe_sampling(args: &Args) -> String {
    if !sampling_enabled(args) {
        return "sampling: OFF (greedy, byte-identical fast path)".to_string();
    }
    let window = args
        .repetition_window
        .map_or_else(|| "all".to_string(), |w| w.to_string());
    format!(
        "sampling: ON repetition_penalty={} repetition_window={} min_p={}",
        args.repetition_penalty, window, args.min_p
    )
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn run_steady(args: &Args, model_dir: &Path, device: NativeDecodeDevice) -> Result<()> {
    if args.synthetic {
        bail!("--steady requires a real model directory");
    }
    if args.tokens <= args.decode_skip {
        bail!("--tokens must be greater than --decode-skip");
    }
    println!("profile_native: {}", describe_sampling(args));

    let mut config = EngineConfig {
        decode_backend: EngineDecodeBackend::Native,
        ..EngineConfig::default()
    };
    config.native_device = Some(device);
    let mut engine = Engine::from_dir(model_dir, config)
        .with_context(|| format!("load native engine {}", model_dir.display()))?;

    for _ in 0..args.warmups {
        std::hint::black_box(
            engine
                .generate(request(args, args.tokens))
                .context("steady warmup generation")?,
        );
    }

    let mut prefills_ms = Vec::with_capacity(args.runs);
    let mut decode_ms_per_token = Vec::with_capacity(args.runs);
    let mut throughputs = Vec::with_capacity(args.runs);
    let mut reference_tokens = None;
    for run in 1..=args.runs {
        let start = Instant::now();
        let mut token_times = Vec::with_capacity(args.tokens);
        let mut callback = |_| {
            token_times.push(start.elapsed());
            Ok(())
        };
        let result = engine
            .generate_with_callback(request(args, args.tokens), Some(&mut callback))
            .context("steady measured generation")?;
        if token_times.len() <= args.decode_skip {
            bail!(
                "generation emitted {} tokens, not enough for --decode-skip {}",
                token_times.len(),
                args.decode_skip
            );
        }
        if let Some(reference) = &reference_tokens {
            if reference != &result.token_ids {
                bail!("native greedy decode was not deterministic across measured runs");
            }
        } else {
            reference_tokens = Some(result.token_ids.clone());
            println!("generated_text: {:?}", result.text);
        }

        let prefill_ms = token_times[0].as_secs_f64() * 1_000.0;
        let decode_tokens = token_times.len() - args.decode_skip;
        let decode_wall = token_times[token_times.len() - 1] - token_times[args.decode_skip - 1];
        let ms_per_token = decode_wall.as_secs_f64() * 1_000.0 / decode_tokens as f64;
        let tok_per_s = decode_tokens as f64 / decode_wall.as_secs_f64();
        println!(
            "steady_run {run}: prefill={prefill_ms:.3} ms decode_tokens={decode_tokens} \
             decode_wall={:.3} ms decode={ms_per_token:.3} ms/token throughput={tok_per_s:.2} tok/s",
            decode_wall.as_secs_f64() * 1_000.0
        );
        prefills_ms.push(prefill_ms);
        decode_ms_per_token.push(ms_per_token);
        throughputs.push(tok_per_s);
    }

    println!(
        "steady_median: prefill={:.3} ms decode={:.3} ms/token throughput={:.2} tok/s \
         (runs={} warmups={} decode_skip={})",
        median(&mut prefills_ms),
        median(&mut decode_ms_per_token),
        median(&mut throughputs),
        args.runs,
        args.warmups,
        args.decode_skip
    );
    if let Some(tokens) = reference_tokens {
        println!("generated_token_ids: {tokens:?}");
    }
    Ok(())
}

fn run_pipeline(args: &Args, model_dir: &Path) -> Result<()> {
    if args.synthetic {
        bail!("--pipeline cannot be combined with --synthetic");
    }
    if !model_dir.is_dir() {
        bail!("--pipeline requires --model to name a pipeline directory");
    }
    if args.steady && args.tokens <= args.decode_skip {
        bail!("--tokens must be greater than --decode-skip");
    }
    let requested_provider = match args.ep {
        ExecutionProvider::Cpu => "cpu",
        ExecutionProvider::Cuda => "cuda",
    };
    // This single-threaded CLI sets provider selection before the process-wide
    // runtime configuration is first read while constructing pipeline sessions.
    unsafe {
        std::env::set_var("ONNX_GENAI_EP", requested_provider);
    }
    let available_providers =
        available_execution_providers().context("query linked ONNX Runtime providers")?;
    println!("ort_available_execution_providers: {available_providers:?}");
    if matches!(args.ep, ExecutionProvider::Cuda)
        && !available_providers
            .iter()
            .any(|provider| provider.eq_ignore_ascii_case("CUDAExecutionProvider"))
    {
        bail!(
            "--ep cuda requested for a pipeline, but the linked ONNX Runtime does not expose \
             CUDAExecutionProvider (available: {available_providers:?}); put the CUDA-enabled \
             ONNX Runtime library directory first in LD_LIBRARY_PATH"
        );
    }

    let mut engine = PipelineEngine::from_dir(model_dir)
        .with_context(|| format!("load pipeline engine {}", model_dir.display()))?;
    for _ in 0..args.warmups {
        std::hint::black_box(
            engine
                .generate_with_pipeline_request(pipeline_request(args, args.tokens))
                .context("pipeline warmup generation")?,
        );
    }

    if args.steady {
        let mut prefills_ms = Vec::with_capacity(args.runs);
        let mut decode_ms_per_token = Vec::with_capacity(args.runs);
        let mut throughputs = Vec::with_capacity(args.runs);
        let mut reference_tokens = None;
        let mut reference_text = None;
        for run in 1..=args.runs {
            let start = Instant::now();
            let mut token_times = Vec::with_capacity(args.tokens);
            let mut callback = |_| {
                token_times.push(start.elapsed());
                Ok(())
            };
            let result = engine
                .generate_with_callback(
                    pipeline_request(args, args.tokens),
                    Some(&mut callback),
                )
                .context("steady pipeline measured generation")?;
            if token_times.len() <= args.decode_skip {
                bail!(
                    "pipeline generation emitted {} tokens, not enough for --decode-skip {}",
                    token_times.len(),
                    args.decode_skip
                );
            }
            if let Some(reference) = &reference_tokens {
                if reference != &result.token_ids {
                    bail!("pipeline greedy decode was not deterministic across measured runs");
                }
            } else {
                reference_tokens = Some(result.token_ids);
                reference_text = Some(result.text);
            }

            let prefill_ms = token_times[0].as_secs_f64() * 1_000.0;
            let decode_tokens = token_times.len() - args.decode_skip;
            let decode_wall =
                token_times[token_times.len() - 1] - token_times[args.decode_skip - 1];
            let ms_per_token = decode_wall.as_secs_f64() * 1_000.0 / decode_tokens as f64;
            let tok_per_s = decode_tokens as f64 / decode_wall.as_secs_f64();
            println!(
                "steady_run {run}: prefill={prefill_ms:.3} ms decode_tokens={decode_tokens} \
                 decode_wall={:.3} ms decode={ms_per_token:.3} ms/token \
                 throughput={tok_per_s:.2} tok/s",
                decode_wall.as_secs_f64() * 1_000.0
            );
            prefills_ms.push(prefill_ms);
            decode_ms_per_token.push(ms_per_token);
            throughputs.push(tok_per_s);
        }
        println!(
            "steady_median: prefill={:.3} ms decode={:.3} ms/token throughput={:.2} tok/s \
             (runs={} warmups={} decode_skip={})",
            median(&mut prefills_ms),
            median(&mut decode_ms_per_token),
            median(&mut throughputs),
            args.runs,
            args.warmups,
            args.decode_skip
        );
        if let Some(tokens) = reference_tokens {
            println!("generated_token_ids: {tokens:?}");
        }
        if let Some(text) = reference_text {
            println!("generated_text: {text:?}");
        }
        return Ok(());
    }

    let mut generated = 0usize;
    let mut elapsed = Duration::ZERO;
    let mut reference_tokens = None;
    let mut reference_text = None;
    for _ in 0..args.runs {
        let start = Instant::now();
        let result = engine
            .generate_with_pipeline_request(pipeline_request(args, args.tokens))
            .context("pipeline measured generation")?;
        elapsed += start.elapsed();
        generated += result.token_ids.len();
        if let Some(reference) = &reference_tokens {
            if reference != &result.token_ids {
                bail!(
                    "pipeline greedy decode was not deterministic: first={reference:?}, \
                     rerun={:?}",
                    result.token_ids
                );
            }
        } else {
            reference_tokens = Some(result.token_ids);
            reference_text = Some(result.text);
        }
    }
    if generated == 0 {
        bail!("pipeline generation produced no tokens");
    }

    let tok_per_s = generated as f64 / elapsed.as_secs_f64();
    let ms_per_step = elapsed.as_secs_f64() * 1_000.0 / generated as f64;
    println!(
        "profile_native: pipeline={} ep={:?} tokens={} warmups={} runs={}",
        model_dir.display(),
        args.ep,
        args.tokens,
        args.warmups,
        args.runs
    );
    println!(
        "throughput: {tok_per_s:.2} tok/s, {ms_per_step:.3} ms/step \
         ({generated} generated tokens in {:.3} ms)",
        elapsed.as_secs_f64() * 1_000.0
    );
    if let Some(tokens) = reference_tokens {
        println!("generated_token_ids: {tokens:?}");
    }
    if let Some(text) = reference_text {
        println!("generated_text: {text:?}");
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.tokens == 0 || args.runs == 0 {
        bail!("--tokens and --runs must be greater than zero");
    }
    if !args.synthetic && args.model.is_none() {
        bail!("--model is required unless --synthetic is used");
    }
    if args.pipeline {
        return run_pipeline(
            &args,
            args.model.as_deref().expect("validated model argument"),
        );
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
    if args.steady {
        return run_steady(
            &args,
            args.model.as_deref().expect("validated model argument"),
            device,
        );
    }
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
            bail!("the in-memory synthetic session constructor is CPU-only");
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
    println!("profile_native: {}", describe_sampling(&args));
    if let Some(dump_path) = args.dump_logprobs.as_ref() {
        let dump_prompt_tokens = if let Some(ids_path) = args.prompt_ids.as_ref() {
            let raw = std::fs::read_to_string(ids_path)
                .with_context(|| format!("read prompt ids from {}", ids_path.display()))?;
            let ids: Vec<u32> = serde_json::from_str(raw.trim())
                .with_context(|| format!("parse prompt ids JSON from {}", ids_path.display()))?;
            if ids.is_empty() {
                bail!("--prompt-ids must contain at least one token id");
            }
            println!("dump_prompt_ids: {ids:?}");
            ids
        } else {
            prompt_tokens.clone()
        };
        let options = GenerateOptions {
            max_new_tokens: 1,
            temperature: 0.0,
            greedy: true,
            stop_on_eos: false,
            top_logprobs: Some(args.logprobs_k),
            ..GenerateOptions::default()
        };
        let result =
            session.generate(&dump_prompt_tokens, &options, &ProcessorChain::new(), &tokenizer)?;
        let logprobs = result
            .logprobs
            .and_then(|entries| entries.into_iter().next())
            .context("native generation did not return token-0 logprobs")?;
        let top: Vec<serde_json::Value> = logprobs
            .top
            .iter()
            .map(|(id, lp)| serde_json::json!([*id, *lp]))
            .collect();
        let payload = serde_json::json!({
            "n_prompt_tokens": dump_prompt_tokens.len(),
            "selected_token": logprobs.token_id,
            "selected_logprob": logprobs.logprob,
            "top": top,
        });
        std::fs::write(dump_path, serde_json::to_string(&payload)?)
            .with_context(|| format!("write logprobs to {}", dump_path.display()))?;
        println!(
            "dumped native token-0 top-{} logprobs (selected={}) to {}",
            args.logprobs_k,
            logprobs.token_id,
            dump_path.display()
        );
        return Ok(());
    }
    if let Some(trace_path) = args.trace.as_ref() {
        // Capture one *traced* generation before the timed runs. Enabling the
        // tracer opens a per-op executor span for every node it dispatches,
        // which is what lets the CUDA kernels attach their `kernel_variant` /
        // `capture_status` annotations. This traced pass exercises the graph
        // capture path (which runs every op eagerly through `exec_plan_node`),
        // so the resulting timeline contains real decode-op variant + capture
        // reasons. We disable tracing again immediately afterwards so the timed
        // warmup/measurement loops below run with zero tracing overhead.
        let (ctx, collector) = onnx_runtime_tracer::TraceContext::in_memory();
        session.set_trace_context(ctx);
        std::hint::black_box(generate(
            &mut session,
            &prompt_tokens,
            &tokenizer,
            args.tokens,
            &args,
        )?);
        session.set_trace_context(onnx_runtime_tracer::TraceContext::noop());
        let json = collector.to_chrome_json();
        std::fs::write(trace_path, &json)
            .with_context(|| format!("failed to write trace to {}", trace_path.display()))?;
        println!(
            "profile_native: wrote {} trace events to {}",
            collector.len(),
            trace_path.display()
        );
    }

    for _ in 0..args.warmups {
        std::hint::black_box(generate(
            &mut session,
            &prompt_tokens,
            &tokenizer,
            args.tokens,
            &args,
        )?);
    }

    let stats_before = session.cuda_kv_debug_stats();
    let mut generated = 0usize;
    let mut elapsed = Duration::ZERO;
    let mut reference_tokens = None;
    for _ in 0..args.runs {
        let start = Instant::now();
        let tokens = generate(&mut session, &prompt_tokens, &tokenizer, args.tokens, &args)?;
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
    if let Some(stats) = session.cuda_kv_debug_stats() {
        let before = stats_before
            .as_ref()
            .expect("CUDA stats before measurement");
        println!(
            "cuda_graph: enabled={} captures={} replays={} fallbacks={}",
            stats.graph.enabled, stats.graph.captures, stats.graph.replays, stats.graph.fallbacks
        );
        println!(
            "cuda_graph_measured: captures={} replays={} fallbacks={}",
            stats.graph.captures - before.graph.captures,
            stats.graph.replays - before.graph.replays,
            stats.graph.fallbacks - before.graph.fallbacks
        );
        println!(
            "device_kv_measured: h2d_calls={} h2d_bytes={} d2h_calls={} d2h_bytes={}",
            stats.kv_transfers.host_upload_calls - before.kv_transfers.host_upload_calls,
            stats.kv_transfers.host_upload_bytes - before.kv_transfers.host_upload_bytes,
            stats.kv_transfers.host_download_calls - before.kv_transfers.host_download_calls,
            stats.kv_transfers.host_download_bytes - before.kv_transfers.host_download_bytes
        );
        if let Some(reason) = session.cuda_graph_fallback_reason() {
            println!("cuda_graph_fallback_reason: {reason}");
        }
    }
    if let Some(tokens) = reference_tokens {
        println!("generated_token_ids: {tokens:?}");
        println!(
            "generated_text: {:?}",
            tokenizer
                .decode(&tokens)
                .context("decode generated tokens")?
        );
    }
    Ok(())
}
