//! Native/ORT token-generation profiler using the engine's shared decode loop.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use onnx_genai_bench::{fixture_path, synthetic_decoder};
use onnx_genai_engine::logits::{
    MinPProcessor, RepetitionPenaltyProcessor, TemperatureProcessor, TopKProcessor, TopPProcessor,
};
use onnx_genai_engine::{
    DecodePrecision, Engine, EngineConfig, EngineDecodeBackend, GenerateOptions, GeneratePrompt,
    GenerateRequest, NativeDecodeDevice, NativeDecodeSession, PipelineEngine,
    PipelineGenerateRequest, ProcessorChain, parse_resource_limit,
};
use onnx_genai_ort::{Tokenizer, available_execution_providers, profile};
use onnx_runtime_session::InferenceSession;

/// Honor `ONNX_GENAI_VRAM_LIMIT` in the profiler, mirroring the server CLI.
///
/// The real fix for large-model residency is real CUDA device-capacity
/// detection in the governor (so the default `Fraction(0.90)` just works), but
/// this convenience lets an operator pin an explicit ceiling (bytes, `8GiB`,
/// `0.9`, or `auto`) without going through the server.
fn apply_vram_limit_env(config: &mut EngineConfig) -> Result<()> {
    match std::env::var("ONNX_GENAI_VRAM_LIMIT") {
        Ok(raw) if !raw.trim().is_empty() => {
            let limit = parse_resource_limit(raw.trim())
                .map_err(|error| anyhow::anyhow!("invalid ONNX_GENAI_VRAM_LIMIT: {error}"))?;
            eprintln!("profile_native: ONNX_GENAI_VRAM_LIMIT -> vram_limit={limit:?}");
            config.limits.vram_limit = limit;
        }
        _ => {}
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ExecutionProvider {
    Cpu,
    Cuda,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum DecodeBackend {
    Native,
    Ort,
    Auto,
}

impl DecodeBackend {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Ort => "ort",
            Self::Auto => "auto",
        }
    }
}

impl From<DecodeBackend> for EngineDecodeBackend {
    fn from(backend: DecodeBackend) -> Self {
        match backend {
            DecodeBackend::Native => Self::Native,
            DecodeBackend::Ort => Self::Ort,
            DecodeBackend::Auto => Self::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum DecodePrecisionArg {
    Model,
    Fp16,
}

impl From<DecodePrecisionArg> for DecodePrecision {
    fn from(precision: DecodePrecisionArg) -> Self {
        match precision {
            DecodePrecisionArg::Model => Self::Model,
            DecodePrecisionArg::Fp16 => Self::Fp16,
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Profile token generation through the shared engine decode loop")]
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
    /// Decoder backend used by the engine timing path. ORT and auto require
    /// --features bench-native,bench-ort (plus cuda for --ep cuda).
    #[arg(long, value_enum, default_value_t = DecodeBackend::Native)]
    backend: DecodeBackend,
    /// Decoder numeric precision. Fp16 is opt-in; model preserves authored precision.
    #[arg(long, value_enum, default_value_t = DecodePrecisionArg::Model)]
    decode_precision: DecodePrecisionArg,
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
    /// "[9707, 12824, 13]"). Applies to native, pipeline, and log-probability
    /// runs so paired benchmarks avoid tokenizer round-trip drift.
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
    /// Temperature for categorical sampling. Values > 0 switch the benchmark
    /// from greedy argmax to seeded categorical sampling.
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    /// Nucleus sampling probability. Values >= 1 disable top-p filtering.
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    /// Keep only the top-k logits before token selection. Zero disables top-k.
    #[arg(long, default_value_t = 0)]
    top_k: usize,
    /// Seed for reproducible categorical sampling across measured runs.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Stage 2a (#750): run the batch-N stateless fused forward
    /// ([`NativeDecodeSession::run_fused_batch_prefill`]) for each batch size in
    /// this comma-separated list, resetting the CUDA weight-offload counters
    /// around each so `htod_bytes_per_token` / `page_ins_per_token` isolate the
    /// weight-streaming amortization. Leads the before/after ~1/N table.
    #[arg(long, value_delimiter = ',')]
    fused_forward_amortization: Option<Vec<usize>>,
    /// Stage 2b (#750): drive the batch-N fused forward with a **non-empty**
    /// length-L batched KV past for each `N@L` pair in this comma-separated list
    /// (e.g. `1@0,8@0,1@512,8@512`). Resets the weight-offload counters around
    /// each so `htod_bytes_per_token` / `page_ins_per_token` show how the
    /// weight-residency reclaim under the elastic budget (#866) rises as `N` and
    /// `L` grow the committed KV — the KV-multiplication trade the stage 2a
    /// empty-past probe could not surface.
    #[arg(long, value_delimiter = ',')]
    fused_forward_kv_sweep: Option<Vec<String>>,
}

fn categorical_sampling_enabled(args: &Args) -> bool {
    args.temperature > 0.0
}

/// Whether any host-side sampling policy (penalty / min-p) is enabled. When
/// false the decode path is byte-identical to the default greedy benchmark and,
/// on CUDA, keeps the captured device-argmax fast path.
fn sampling_enabled(args: &Args) -> bool {
    args.repetition_penalty != 1.0
        || args.min_p > 0.0
        || categorical_sampling_enabled(args)
        || args.top_p < 1.0
        || args.top_k > 0
}

/// Copy the CLI sampling policy onto generation options (default values are
/// no-ops, preserving existing greedy behavior exactly).
fn apply_sampling_options(options: &mut GenerateOptions, args: &Args) {
    let categorical = categorical_sampling_enabled(args);
    options.temperature = if categorical { args.temperature } else { 0.0 };
    options.greedy = !categorical;
    options.seed = categorical.then_some(args.seed);
    options.top_p = args.top_p;
    options.top_k = args.top_k;
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
    if args.temperature > 0.0 && args.temperature != 1.0 {
        chain.add(Box::new(TemperatureProcessor {
            temperature: args.temperature,
        }));
    }
    if args.top_k > 0 {
        chain.add(Box::new(TopKProcessor { top_k: args.top_k }));
    }
    if args.top_p < 1.0 {
        chain.add(Box::new(TopPProcessor { top_p: args.top_p }));
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
    request.options.stop_on_eos = false;
    apply_sampling_options(&mut request.options, args);
    request
}

fn pipeline_request(args: &Args, tokens: usize, prompt_tokens: &[u32]) -> PipelineGenerateRequest {
    let mut request = request(args, tokens);
    request.prompt = GeneratePrompt::TokenIds(prompt_tokens.to_vec());
    PipelineGenerateRequest::new(request)
}

fn describe_sampling(args: &Args) -> String {
    if !sampling_enabled(args) {
        return "sampling: OFF (greedy, byte-identical fast path)".to_string();
    }
    let window = args
        .repetition_window
        .map_or_else(|| "all".to_string(), |w| w.to_string());
    format!(
        "sampling: ON temperature={} top_p={} top_k={} seed={} repetition_penalty={} \
         repetition_window={} min_p={}",
        args.temperature,
        args.top_p,
        args.top_k,
        args.seed,
        args.repetition_penalty,
        window,
        args.min_p
    )
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn validate_backend(args: &Args) -> Result<()> {
    #[cfg(not(feature = "bench-ort"))]
    if matches!(args.backend, DecodeBackend::Ort | DecodeBackend::Auto) {
        bail!(
            "ORT backend requires building profile_native with \
             `--features bench-native,bench-ort,cuda` for CUDA \
             (omit `,cuda` for CPU)"
        );
    }

    if !args.steady && !args.pipeline && args.backend != DecodeBackend::Native {
        bail!("--backend {} requires --steady", args.backend.as_str());
    }
    Ok(())
}

fn print_backend_label(backend: DecodeBackend) {
    if backend != DecodeBackend::Native {
        println!("profile_native: backend={}", backend.as_str());
    }
}

fn print_memory_observability(engine: &Engine) {
    let plan = engine.memory_strategy_plan();
    let application = plan.runtime_application();
    println!(
        "memory_strategy: strategy={:?} inferred={:?} access={:?} total_weight_bytes={} \
         kv_bytes_per_token={:?} resolved_device_budget_bytes={:?} \
         fits_resolved_device_budget={:?}",
        plan.strategy,
        plan.inferred_strategy,
        plan.weight_access_pattern,
        plan.total_weight_bytes,
        plan.kv_bytes_per_token,
        plan.resolved_device_budget_bytes,
        plan.fits_resolved_device_budget
    );
    println!(
        "memory_policy: weight_offload_enabled={} managed_no_spill={} \
         scan_resistant_dense={} device_budget_bytes={:?} managed_limit_bytes={:?} \
         auto_enabled_from_vram_limit={}",
        application.weight_offload_enabled,
        application.managed_no_spill,
        application.scan_resistant_dense,
        application.device_budget_bytes,
        application.managed_limit_bytes,
        application.auto_enabled_from_vram_limit
    );
    let resources = engine.resource_snapshot();
    println!(
        "resource_vram: used_bytes={} limit_bytes={} headroom_bytes={} oversubscribed_bytes={}",
        resources.vram.used,
        resources.vram.limit,
        resources.vram.headroom,
        engine.device_oversubscribed_bytes()
    );
}

fn print_cuda_observability(
    engine: &Engine,
    before: Option<&onnx_genai_engine::native_decode::CudaKvDebugStats>,
) {
    if let Some(stats) = engine.native_cuda_debug_stats() {
        println!(
            "cuda_graph: enabled={} captures={} replays={} fallbacks={} invalidations={}",
            stats.graph.enabled,
            stats.graph.captures,
            stats.graph.replays,
            stats.graph.fallbacks,
            stats.graph.invalidations
        );
        if let Some(before) = before {
            println!(
                "cuda_graph_measured: captures={} replays={} fallbacks={} invalidations={}",
                stats.graph.captures.saturating_sub(before.graph.captures),
                stats.graph.replays.saturating_sub(before.graph.replays),
                stats.graph.fallbacks.saturating_sub(before.graph.fallbacks),
                stats
                    .graph
                    .invalidations
                    .saturating_sub(before.graph.invalidations)
            );
        }
        if let Some(reason) = &stats.graph.decline_reason {
            println!("cuda_graph_decline_reason: {reason}");
        }
        if let Some(reason) = &stats.graph.growth_decision {
            println!("cuda_graph_growth_decision: {reason}");
        }
        if let Some(report) = &stats.graph.fallback_report {
            println!("cuda_graph_fallback_report: {report}");
        }
    }
}

fn weight_offload_hit_rate(stats: &onnx_runtime_ep_cuda::GlobalOffloadStats) -> Option<f64> {
    let lookups = stats.page_ins.saturating_add(stats.hits);
    (lookups > 0).then(|| stats.hits as f64 / lookups as f64 * 100.0)
}

/// Ratio of an accumulated counter to the number of emitted tokens.
///
/// This is the batch-invariant quantity the #750 measurement protocol requires
/// leading every batch-1 vs batch-N comparison: on a streaming-bound model the
/// weight bytes streamed per decode step are (near-)constant in batch size `B`,
/// so `htod_bytes / emitted_tokens` and `page_ins / emitted_tokens` should fall
/// ~1/B while wall-clock throughput stays noisy. Returns `None` when no tokens
/// were emitted so the caller reports `n/a` rather than dividing by zero.
fn per_emitted_token(total: u64, emitted_tokens: u64) -> Option<f64> {
    (emitted_tokens > 0).then(|| total as f64 / emitted_tokens as f64)
}

fn print_weight_offload_amortization(
    stats: &onnx_runtime_ep_cuda::GlobalOffloadStats,
    emitted_tokens: u64,
) {
    let fmt = |value: Option<f64>| {
        value
            .map(|ratio| format!("{ratio:.1}"))
            .unwrap_or_else(|| "n/a".to_string())
    };
    println!(
        "weight_offload_amortization: emitted_tokens={} htod_bytes_per_token={} \
         page_ins_per_token={}",
        emitted_tokens,
        fmt(per_emitted_token(stats.htod_bytes, emitted_tokens)),
        fmt(per_emitted_token(stats.page_ins, emitted_tokens))
    );
}

fn print_weight_offload_observability(emitted_tokens: u64) {
    let stats = onnx_runtime_ep_cuda::global_offload_stats();
    let hit_rate = weight_offload_hit_rate(&stats)
        .map(|rate| format!("{rate:.2}%"))
        .unwrap_or_else(|| "n/a".to_string());
    // The byte-weighted rate is the one residency policy must be judged on: the
    // count-based rate weights a 10 KiB norm like an 11 MiB projection, so it can
    // improve while streamed bytes get worse (#857, #837 item 3).
    let byte_hit_rate = stats
        .byte_hit_rate()
        .map(|rate| format!("{:.2}%", rate * 100.0))
        .unwrap_or_else(|| "n/a".to_string());
    // Byte-weighted attribution of the bypass count: what share of streamed
    // bytes is bypass traffic that residency policy keeps no benefit from and
    // re-streams every step (#837 item 3).
    let bypassed_byte_share = stats
        .bypassed_byte_share()
        .map(|rate| format!("{:.2}%", rate * 100.0))
        .unwrap_or_else(|| "n/a".to_string());
    println!(
        "weight_offload_cache: page_ins={} hits={} hit_rate={} byte_hit_rate={} \
         hit_bytes={} evictions={} bypassed_page_ins={} bypassed_page_in_bytes={} \
         bypassed_byte_share={}",
        stats.page_ins,
        stats.hits,
        hit_rate,
        byte_hit_rate,
        stats.hit_bytes,
        stats.evictions,
        stats.bypassed_page_ins,
        stats.bypassed_page_in_bytes,
        bypassed_byte_share
    );
    print_weight_offload_amortization(&stats, emitted_tokens);
    println!(
        "weight_offload_timing: materialize_ms={:.3} htod_ms={:.3} \
         admit_sync_ms={:.3} vram_alloc_ms={:.3} vram_free_ms={:.3}",
        stats.materialize_ns as f64 / 1_000_000.0,
        stats.htod_ns as f64 / 1_000_000.0,
        stats.admit_sync_ns as f64 / 1_000_000.0,
        stats.vram_alloc_ns as f64 / 1_000_000.0,
        stats.vram_free_ns as f64 / 1_000_000.0
    );
    println!(
        "weight_offload_physical: budget_bytes={} mapped_physical_bytes={} \
         physical_owned_bytes={} htod_bytes={}",
        stats.budget_bytes,
        stats.mapped_physical_bytes,
        stats.physical_owned_bytes,
        stats.htod_bytes
    );
    let htod_gbps = if stats.htod_ns > 0 {
        stats.htod_bytes as f64 / (stats.htod_ns as f64 / 1_000_000_000.0) / 1e9
    } else {
        0.0
    };
    println!(
        "weight_offload_staging: pinned_alloc_calls={} pinned_reuses={} \
         effective_htod_gbps={:.3}",
        stats.pinned_alloc_calls, stats.pinned_reuses, htod_gbps
    );
}

fn print_vmm_observability(engine: &Engine) {
    if let Some(stats) = engine.vmm_arena_stats() {
        println!(
            "vmm_arena: committed_physical_bytes={} reserved_va_bytes={} \
             peak_committed_physical_bytes={} commits={} releases={} allocations={} \
             ref_underflows={} byte_underflows={} unaccounted_committed_bytes={}",
            stats.committed_bytes,
            stats.reserved_bytes,
            stats.peak_committed_bytes,
            stats.commits,
            stats.releases,
            stats.allocations,
            stats.ref_underflows,
            stats.byte_underflows,
            stats.unaccounted_committed_bytes
        );
    }
}

fn configure_ort_provider(args: &Args) -> Result<()> {
    let requested_provider = match args.ep {
        ExecutionProvider::Cpu => "cpu",
        ExecutionProvider::Cuda => "cuda",
    };
    // This single-threaded CLI sets provider selection before the process-wide
    // runtime configuration is first read while constructing ORT sessions.
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
            "--backend {} --ep cuda requested, but the linked ONNX Runtime does not expose \
             CUDAExecutionProvider (available: {available_providers:?}); put the CUDA-enabled \
             ONNX Runtime library directory first in LD_LIBRARY_PATH",
            args.backend.as_str()
        );
    }
    Ok(())
}

/// Stage 2a (#750): measure weight-streaming amortization of the batch-N
/// stateless fused forward. For each batch size `N`, reset the global CUDA
/// weight-offload counters, run exactly one fused forward over `N` rows, and
/// report `htod_bytes` / `page_ins` both in total and per emitted row. Because
/// the offload residency is keyed purely by weight identity (no batch axis), a
/// single forward pages each weight in at most once regardless of `N`, so the
/// per-row figures fall ~`1/N` — the deterministic, batch-invariant signature
/// that "amortization happened", led ahead of any wall-clock number.
fn run_fused_forward_amortization(
    session: &mut NativeDecodeSession,
    prompt_tokens: &[u32],
    batch_sizes: &[usize],
) -> Result<()> {
    if batch_sizes.is_empty() {
        bail!("--fused-forward-amortization requires at least one batch size");
    }
    let own_pid = std::process::id();
    println!(
        "fused_forward_amortization: own_pid={own_pid} batch_sizes={batch_sizes:?} \
         (rows are independent length-1 sequences, empty past; stateless eager forward)"
    );
    report_foreign_compute_apps(own_pid);

    // One-time warmup outside every measurement window so first-touch admission
    // and any lazy CUDA setup are not attributed to a batch size.
    let warm = fused_tokens(prompt_tokens, 1);
    let _ = session
        .run_fused_batch_prefill(&warm)
        .context("warmup fused forward")?;

    for &batch in batch_sizes {
        if batch == 0 {
            bail!("--fused-forward-amortization batch sizes must be > 0");
        }
        report_foreign_compute_apps(own_pid);
        let tokens = fused_tokens(prompt_tokens, batch);
        onnx_runtime_ep_cuda::reset_global_offload_stats();
        let rows = session
            .run_fused_batch_prefill(&tokens)
            .with_context(|| format!("fused forward at batch {batch}"))?;
        assert_eq!(
            rows.len(),
            batch,
            "fused forward must emit one row per token"
        );
        let emitted = batch as u64;
        println!("--- fused_forward batch={batch} ---");
        print_weight_offload_observability(emitted);
    }
    Ok(())
}

/// Stage 2b (#750): sweep the batch-N fused forward across `N@L` (batch @
/// past_len) pairs, resetting the weight-offload counters around each so the
/// KV-multiplication trade is visible — as `N` and `L` grow the committed KV,
/// the elastic weight budget (#866) reclaims weight residency and
/// `htod_bytes_per_token` rises. Leads with the deterministic per-token counters
/// exactly as the amortization sweep does.
fn run_fused_forward_kv_sweep(
    session: &mut NativeDecodeSession,
    prompt_tokens: &[u32],
    pairs: &[String],
) -> Result<()> {
    if pairs.is_empty() {
        bail!("--fused-forward-kv-sweep requires at least one N@L pair");
    }
    let parsed = pairs
        .iter()
        .map(|pair| {
            let (batch, past) = pair
                .split_once('@')
                .with_context(|| format!("--fused-forward-kv-sweep pair '{pair}' must be N@L"))?;
            let batch: usize = batch
                .trim()
                .parse()
                .with_context(|| format!("invalid batch N in '{pair}'"))?;
            let past: usize = past
                .trim()
                .parse()
                .with_context(|| format!("invalid past_len L in '{pair}'"))?;
            if batch == 0 {
                bail!("--fused-forward-kv-sweep batch N must be > 0 in '{pair}'");
            }
            Ok((batch, past))
        })
        .collect::<Result<Vec<_>>>()?;

    let own_pid = std::process::id();
    println!(
        "fused_forward_kv_sweep: own_pid={own_pid} pairs={parsed:?} \
         (rows are independent length-1 sequences over a zero-seeded length-L batched past; stateless eager forward)"
    );
    report_foreign_compute_apps(own_pid);

    // One-time warmup outside every measurement window.
    let warm = fused_tokens(prompt_tokens, 1);
    let _ = session
        .run_fused_batch_forward(&warm, 0)
        .context("warmup fused forward")?;

    for (batch, past_len) in parsed {
        report_foreign_compute_apps(own_pid);
        let tokens = fused_tokens(prompt_tokens, batch);
        onnx_runtime_ep_cuda::reset_global_offload_stats();
        let rows = session
            .run_fused_batch_forward(&tokens, past_len)
            .with_context(|| format!("fused forward at batch {batch} past_len {past_len}"))?;
        assert_eq!(
            rows.len(),
            batch,
            "fused forward must emit one row per token"
        );
        println!("--- fused_forward batch={batch} past_len={past_len} ---");
        print_weight_offload_observability(batch as u64);
    }
    Ok(())
}

/// Build `batch` token ids for the fused-forward probe by cycling the prompt.
/// Token *values* never affect weight-streaming counters (residency is keyed by
/// weight identity, not by token), so cycling a real prompt is representative.
fn fused_tokens(prompt_tokens: &[u32], batch: usize) -> Vec<u32> {
    (0..batch)
        .map(|index| prompt_tokens[index % prompt_tokens.len()])
        .collect()
}

/// Print any CUDA compute processes that are NOT this profiler, so a contended
/// run is labeled as such in its own log (per #851). Our own PID is expected
/// and filtered out; a non-empty foreign list is flagged loudly.
fn report_foreign_compute_apps(own_pid: u32) {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader",
        ])
        .output();
    match output {
        Ok(output) if output.status.success() => {
            let text = String::from_utf8_lossy(&output.stdout);
            let foreign: Vec<&str> = text
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .filter(|line| {
                    line.split(',')
                        .next()
                        .and_then(|pid| pid.trim().parse::<u32>().ok())
                        != Some(own_pid)
                })
                .collect();
            if foreign.is_empty() {
                println!("gpu_contention_check: clear (only own_pid={own_pid})");
            } else {
                println!(
                    "gpu_contention_check: CONTENDED — foreign compute apps present: {foreign:?} \
                     (label this run contended)"
                );
            }
        }
        Ok(output) => println!(
            "gpu_contention_check: nvidia-smi exited with status {} (skipping check)",
            output.status
        ),
        Err(error) => println!("gpu_contention_check: nvidia-smi unavailable ({error})"),
    }
}

fn run_steady(args: &Args, model_dir: &Path, device: NativeDecodeDevice) -> Result<()> {
    if args.synthetic {
        bail!("--steady requires a real model directory");
    }
    if args.tokens <= args.decode_skip {
        bail!("--tokens must be greater than --decode-skip");
    }
    print_backend_label(args.backend);
    println!("profile_native: {}", describe_sampling(args));
    if matches!(args.backend, DecodeBackend::Ort | DecodeBackend::Auto) {
        configure_ort_provider(args)?;
    }

    if !matches!(args.backend, DecodeBackend::Native) {
        let requested_provider = match args.ep {
            ExecutionProvider::Cpu => "cpu",
            ExecutionProvider::Cuda => "cuda",
        };
        // This single-threaded CLI sets provider selection before the process-wide
        // runtime configuration is first read while constructing the ORT session.
        unsafe {
            std::env::set_var("ONNX_GENAI_EP", requested_provider);
        }
    }
    let mut config = EngineConfig {
        decode_backend: args.backend.into(),
        decode_precision: args.decode_precision.into(),
        ..EngineConfig::default()
    };
    config.native_device = Some(device);
    apply_vram_limit_env(&mut config)?;
    let mut engine = Engine::from_dir(model_dir, config).with_context(|| {
        format!(
            "load {} engine {}",
            args.backend.as_str(),
            model_dir.display()
        )
    })?;
    println!(
        "profile_native: model={} ep={:?} backend={}",
        model_dir.display(),
        args.ep,
        args.backend.as_str()
    );
    if args.backend != DecodeBackend::Native {
        println!(
            "profile_native: resolved_backend={}",
            match engine.decode_backend() {
                EngineDecodeBackend::Native => "native",
                EngineDecodeBackend::Ort => "ort",
                EngineDecodeBackend::Auto => "auto",
            }
        );
    }
    print_memory_observability(&engine);

    for _ in 0..args.warmups {
        std::hint::black_box(
            engine
                .generate(request(args, args.tokens))
                .context("steady warmup generation")?,
        );
    }
    profile::reset();
    onnx_runtime_session::reset_exec_phase_profile();
    onnx_runtime_session::reset_dense_prefetch_gap_stats();
    onnx_runtime_ep_cuda::reset_global_offload_stats();
    let cuda_before = engine.native_cuda_debug_stats();

    let mut prefills_ms = Vec::with_capacity(args.runs);
    let mut decode_ms_per_token = Vec::with_capacity(args.runs);
    let mut throughputs = Vec::with_capacity(args.runs);
    let mut reference_tokens = None;
    let mut generated = 0usize;
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
                bail!(
                    "{} decode was not deterministic across measured runs",
                    args.backend.as_str()
                );
            }
        } else {
            reference_tokens = Some(result.token_ids.clone());
            println!("generated_text: {:?}", result.text);
        }
        generated += result.token_ids.len();

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
        "steady_median: backend={} prefill={:.3} ms decode={:.3} ms/token throughput={:.2} tok/s \
         (runs={} warmups={} decode_skip={})",
        args.backend.as_str(),
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
    print_cuda_observability(&engine, cuda_before.as_ref());
    print_weight_offload_observability(generated as u64);
    print_vmm_observability(&engine);
    if profile::enabled() {
        println!("{}", profile::report(generated as u64));
    }
    onnx_runtime_session::print_exec_phase_profile();
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
    println!(
        "profile_native: pipeline={} ep={:?} tokens={} warmups={} runs={}",
        model_dir.display(),
        args.ep,
        args.tokens,
        args.warmups,
        args.runs
    );
    print_backend_label(args.backend);
    if matches!(args.backend, DecodeBackend::Ort | DecodeBackend::Auto) {
        configure_ort_provider(args)?;
    }

    let mut config = EngineConfig {
        decode_backend: args.backend.into(),
        decode_precision: args.decode_precision.into(),
        ..EngineConfig::default()
    };
    config.native_device = Some(match args.ep {
        ExecutionProvider::Cpu => NativeDecodeDevice::Cpu,
        ExecutionProvider::Cuda => NativeDecodeDevice::Cuda { index: None },
    });
    apply_vram_limit_env(&mut config)?;
    let mut engine = PipelineEngine::from_dir_with_config(model_dir, config)
        .with_context(|| format!("load pipeline engine {}", model_dir.display()))?;
    let tokenizer =
        Tokenizer::from_file(tokenizer_file(model_dir)).context("load pipeline tokenizer.json")?;
    let prompt_tokens = if let Some(ids_path) = args.prompt_ids.as_ref() {
        let raw = std::fs::read_to_string(ids_path)
            .with_context(|| format!("read prompt ids from {}", ids_path.display()))?;
        serde_json::from_str::<Vec<u32>>(raw.trim())
            .with_context(|| format!("parse prompt ids JSON from {}", ids_path.display()))?
    } else {
        tokenizer
            .encode(&args.prompt)
            .context("tokenize pipeline prompt")?
    };
    if prompt_tokens.is_empty() {
        bail!("pipeline prompt tokenized to an empty sequence");
    }
    for _ in 0..args.warmups {
        std::hint::black_box(
            engine
                .generate_with_pipeline_request(pipeline_request(args, args.tokens, &prompt_tokens))
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
                    pipeline_request(args, args.tokens, &prompt_tokens),
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
            let generated_tokens = result.token_ids.len();
            if let Some(reference) = &reference_tokens {
                if reference != &result.token_ids {
                    bail!("pipeline greedy decode was not deterministic across measured runs");
                }
            } else {
                reference_tokens = Some(result.token_ids);
                reference_text = Some(result.text);
            }

            let diagnostic = engine.workflow_performance_diagnostic();
            if diagnostic.last_emit_timestamps_ns.len() != generated_tokens {
                bail!(
                    "workflow emitted {} timing events for {} generated tokens",
                    diagnostic.last_emit_timestamps_ns.len(),
                    generated_tokens
                );
            }
            let prefill_ms = diagnostic.last_emit_timestamps_ns[0] as f64 / 1_000_000.0;
            let decode_tokens = diagnostic.last_emit_timestamps_ns.len() - args.decode_skip;
            let decode_ns = diagnostic.last_emit_timestamps_ns
                [diagnostic.last_emit_timestamps_ns.len() - 1]
                - diagnostic.last_emit_timestamps_ns[args.decode_skip - 1];
            let decode_wall = Duration::from_nanos(u64::try_from(decode_ns)?);
            let ms_per_token = decode_ns as f64 / 1_000_000.0 / decode_tokens as f64;
            let tok_per_s = decode_tokens as f64 * 1_000_000_000.0 / decode_ns as f64;
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
        let diagnostic = engine.workflow_performance_diagnostic();
        for island in diagnostic.islands {
            println!(
                "workflow_island {}: components={:?} runs={} session_runs={} eager={} stable={} \
                 captures={} replays={} syncs={} h2d={}/{}B d2h={}/{}B d2d={}/{}B \
                 elapsed={:.3}ms fallback={:?}",
                island.id,
                island.components,
                island.runs,
                island.session_runs,
                island.eager_runs,
                island.stable_binding_runs,
                island.captures,
                island.replays,
                island.device_synchronizations,
                island.host_to_device_copies,
                island.host_to_device_bytes,
                island.device_to_host_copies,
                island.device_to_host_bytes,
                island.device_to_device_copies,
                island.device_to_device_bytes,
                island.total_run_ns as f64 / 1_000_000.0,
                island.fallback_reason
            );
        }
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
            .generate_with_pipeline_request(pipeline_request(args, args.tokens, &prompt_tokens))
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
    validate_backend(&args)?;
    eprintln!(
        "profile_native: WEIGHT_OFFLOAD_BYTE_AWARE={:?} WEIGHT_OFFLOAD_EVICT_ORDER={:?} \
         MANAGED_WEIGHT_STREAMING={:?} CUDA_GRAPH={:?}",
        std::env::var("ONNX_GENAI_WEIGHT_OFFLOAD_BYTE_AWARE").ok(),
        std::env::var("ONNX_GENAI_WEIGHT_OFFLOAD_EVICT_ORDER").ok(),
        std::env::var("ONNX_GENAI_MANAGED_WEIGHT_STREAMING").ok(),
        std::env::var("ONNX_GENAI_CUDA_GRAPH").ok(),
    );
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
    let prompt_tokens = if let Some(ids_path) = args.prompt_ids.as_ref() {
        let raw = std::fs::read_to_string(ids_path)
            .with_context(|| format!("read prompt ids from {}", ids_path.display()))?;
        serde_json::from_str::<Vec<u32>>(raw.trim())
            .with_context(|| format!("parse prompt ids JSON from {}", ids_path.display()))?
    } else {
        tokenizer.encode(&args.prompt).context("tokenize prompt")?
    };
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
        NativeDecodeSession::load_with_resolved_io(&model, device)
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
    if let Some(batch_sizes) = args.fused_forward_amortization.clone() {
        return run_fused_forward_amortization(&mut session, &prompt_tokens, &batch_sizes);
    }
    if let Some(pairs) = args.fused_forward_kv_sweep.clone() {
        return run_fused_forward_kv_sweep(&mut session, &prompt_tokens, &pairs);
    }
    if let Some(dump_path) = args.dump_logprobs.as_ref() {
        let dump_prompt_tokens = prompt_tokens.clone();
        if args.prompt_ids.is_some() {
            println!("dump_prompt_ids: {dump_prompt_tokens:?}");
        }
        let options = GenerateOptions {
            max_new_tokens: 1,
            temperature: 0.0,
            greedy: true,
            stop_on_eos: false,
            top_logprobs: Some(args.logprobs_k),
            ..GenerateOptions::default()
        };
        let result = session.generate(
            &dump_prompt_tokens,
            &options,
            &ProcessorChain::new(),
            &tokenizer,
        )?;
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
    profile::reset();
    onnx_runtime_ep_cuda::reset_global_offload_stats();

    onnx_runtime_session::reset_dense_prefetch_gap_stats();
    onnx_runtime_ep_cuda::reset_global_offload_stats();

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
            "cuda_graph: enabled={} captures={} replays={} fallbacks={} invalidations={}",
            stats.graph.enabled,
            stats.graph.captures,
            stats.graph.replays,
            stats.graph.fallbacks,
            stats.graph.invalidations
        );
        println!(
            "cuda_graph_measured: captures={} replays={} fallbacks={} invalidations={}",
            stats.graph.captures - before.graph.captures,
            stats.graph.replays - before.graph.replays,
            stats.graph.fallbacks - before.graph.fallbacks,
            stats.graph.invalidations - before.graph.invalidations
        );
        if let Some(reason) = &stats.graph.decline_reason {
            println!("cuda_graph_decline_reason: {reason}");
        }
        println!(
            "cuda_kv_growth_measured: events={} d2d_copy_bytes={}",
            stats.kv_growth_events - before.kv_growth_events,
            stats.kv_growth_d2d_copy_bytes - before.kv_growth_d2d_copy_bytes
        );
        println!(
            "cuda_kv: logical_len={} max_len={} committed_len={} hard_max_len={} committed_bytes={} physical_bytes={}",
            stats.logical_len,
            stats.max_len,
            stats.kv_committed_len,
            stats.hard_max_len,
            stats.kv_committed_bytes,
            stats.kv_physical_bytes_by_binding.iter().sum::<usize>()
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
    let offload = onnx_runtime_ep_cuda::global_offload_stats();
    println!(
        "weight_offload_cache: page_ins={} hits={} evictions={}",
        offload.page_ins, offload.hits, offload.evictions
    );
    print_weight_offload_amortization(&offload, generated as u64);
    if profile::enabled() {
        println!("{}", profile::report(generated as u64));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_backend_values_and_preserves_native_default() {
        let default = Args::try_parse_from(["profile_native", "--synthetic"]).unwrap();
        assert_eq!(default.backend, DecodeBackend::Native);
        assert_eq!(default.decode_precision, DecodePrecisionArg::Model);

        for (value, expected) in [
            ("native", DecodeBackend::Native),
            ("ort", DecodeBackend::Ort),
            ("auto", DecodeBackend::Auto),
        ] {
            let args = Args::try_parse_from(["profile_native", "--synthetic", "--backend", value])
                .unwrap();
            assert_eq!(args.backend, expected);
        }
    }

    #[test]
    fn weight_offload_hit_rate_counts_hits_and_misses() {
        let stats = onnx_runtime_ep_cuda::GlobalOffloadStats {
            page_ins: 7,
            hits: 21,
            ..Default::default()
        };
        assert_eq!(weight_offload_hit_rate(&stats), Some(75.0));
        assert_eq!(
            weight_offload_hit_rate(&onnx_runtime_ep_cuda::GlobalOffloadStats::default()),
            None
        );
    }

    #[test]
    fn per_emitted_token_divides_counter_by_tokens() {
        // 65,772,419,072 htod bytes over 16 emitted tokens is the #837 baseline;
        // the ratio (~4.11 GB/token) is the batch-invariant quantity a batch-N
        // run must drive down, and is far more stable than wall-clock tok/s.
        assert_eq!(per_emitted_token(65_772_419_072, 16), Some(4_110_776_192.0));
        assert_eq!(per_emitted_token(5_535, 16), Some(345.9375));
        // No emitted tokens must report `n/a`, never divide by zero.
        assert_eq!(per_emitted_token(1_000, 0), None);
        assert_eq!(per_emitted_token(0, 8), Some(0.0));
    }

    #[test]
    fn parses_decode_precision_values_and_preserves_model_default() {
        for (value, expected) in [
            ("model", DecodePrecisionArg::Model),
            ("fp16", DecodePrecisionArg::Fp16),
        ] {
            let args = Args::try_parse_from([
                "profile_native",
                "--synthetic",
                "--decode-precision",
                value,
            ])
            .unwrap();
            assert_eq!(args.decode_precision, expected);
        }
    }

    #[test]
    fn rejects_invalid_backend_value() {
        let error = Args::try_parse_from(["profile_native", "--synthetic", "--backend", "bogus"])
            .unwrap_err();
        assert_eq!(error.kind(), clap::error::ErrorKind::InvalidValue);
    }

    #[cfg(not(feature = "bench-ort"))]
    #[test]
    fn rejects_ort_without_bench_ort_feature() {
        let args = Args::try_parse_from([
            "profile_native",
            "--model",
            "unused",
            "--steady",
            "--backend",
            "ort",
        ])
        .unwrap();
        let error = validate_backend(&args).unwrap_err().to_string();
        assert!(error.contains("bench-native,bench-ort,cuda"), "{error}");
    }
}
