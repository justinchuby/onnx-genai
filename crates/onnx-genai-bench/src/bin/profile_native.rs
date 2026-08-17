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
    DecodePrecision, Engine, EngineConfig, EngineDecodeBackend, GenerateOptions, GenerateRequest,
    NativeDecodeDevice, NativeDecodeSession, PipelineEngine, PipelineGenerateRequest,
    ProcessorChain, SpeculativeMode, SpeculativeStats, parse_resource_limit,
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

/// Speculative decoding mode selected on the command line. Only the native
/// single-model engine path (`--steady` without `--pipeline`) wires speculation
/// today; see the guard in `run_pipeline`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum SpeculativeArg {
    /// Plain M=1 greedy decode (default).
    None,
    /// Prompt-lookup (n-gram) speculation: copy continuations from the most
    /// recent matching context n-gram, then exact-verify against the target.
    PromptLookup,
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
    /// Must be at least 1: the steady window is timed from the token
    /// immediately before the first measured token, so index `decode_skip - 1`
    /// has to exist.
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
    /// Stage 2b-impl-4 (#750): sweep the **persistent batch-N decode path**
    /// ([`NativeDecodeSession::decode_greedy_batch`], the real captured decode
    /// step — not the stateless fused forward) across the comma-separated batch
    /// sizes in this list (e.g. `1,2,4,8`). For each `N` a fresh session is built
    /// pinned to batch `N` (`ONNX_GENAI_NATIVE_DECODE_BATCH`), `--tokens` batch-N
    /// greedy steps are run from an empty past feeding an identical token to all
    /// `N` rows, and the weight-offload counters + CUDA-graph recapture counters
    /// are reported per shape. Because every row sees identical inputs from an
    /// identical (empty) initial state, all `N` output rows must be byte-identical
    /// each step (a cross-row corruption guard), and row 0 must match the batch-1
    /// stream (the batch-N row-identity guard). Token *values* are content-
    /// arbitrary (no real prompt is seeded — that needs the stage 2c caller), but
    /// the weight/page/capture counters are content-invariant (#884) and are the
    /// measurement the whole batch-N line was after.
    #[arg(long, value_delimiter = ',')]
    native_decode_batch_sweep: Option<Vec<usize>>,
    /// Unmeasured context length to seed before the measured `--native-decode-batch-sweep`
    /// window (default 0). Each batch runs this many decode steps first so the
    /// committed KV occupancy grows to `N × context × kv_bytes_per_token`,
    /// shrinking the elastic weight budget (#866) enough that an over-budget model
    /// actually streams weights during the measured steps. 0 leaves the KV empty
    /// (weights stay fully resident, `htod_bytes_per_token = 0`).
    #[arg(long, default_value_t = 0)]
    native_decode_batch_context: usize,
    /// Stage 2c (#750): the batch-N **solo-equivalence** correctness gate. Give a
    /// `||`-separated list of K distinct text prompts (e.g.
    /// `"The capital of France is||Once upon a time,||2 + 2 ="`). Each prompt is
    /// tokenized and — because the native persistent decode path is a *uniform*
    /// batch (one shared mask window and one shared position per step, see
    /// `DecodeCudaState::extend_mask`) — every prompt is truncated to the shortest
    /// common token length `L` so all K rows step in lockstep. The tool then
    /// (1) runs each prompt **alone** at batch 1 (fresh
    /// `ONNX_GENAI_NATIVE_DECODE_BATCH=1` session, reset between prompts),
    /// recording its `--tokens` greedy token stream, then (2) seeds all K
    /// genuinely-different prompts into one batch-K session (row `b` carries
    /// prompt `b`'s own tokens, position, and KV row) and runs the same
    /// `--tokens` steps, and asserts every batch row is **byte-identical** to
    /// that prompt's solo stream. Unlike the identical-token
    /// `--native-decode-batch-sweep`, the rows here carry different content, so
    /// cross-row KV bleed, a shared mask, a wrong stride, or a position error
    /// would all diverge — the real test that rows do not observe each other.
    #[arg(long)]
    solo_equivalence_prompts: Option<String>,
    /// Stage 3a (#750): the batch-N **ragged** solo-equivalence correctness gate.
    /// Same idea as `--solo-equivalence-prompts` but the prompts are kept at their
    /// genuinely different token lengths (NOT truncated to a common `L`), so the
    /// batch is ragged — rows sit at different logical lengths within one fused
    /// forward. Give a `||`-separated list of K prompts whose tokenizations differ
    /// in length. The tool right-aligns the prefills (each row is held with an
    /// `advance=false` step until it is time for its prompt to finish alongside
    /// the longest, so from the moment a shorter row starts it is at a different
    /// length than its peers), then (1) runs each prompt **alone** at batch 1 and
    /// (2) seeds all K prompts into one ragged batch-K session, and asserts every
    /// batch row is **byte-identical** to its solo stream. If the per-row length,
    /// position, or mask geometry is wrong, the different-length rows diverge —
    /// that is the signal. The per-row prompt lengths and token ids are printed so
    /// it is visible the lengths really differ.
    #[arg(long)]
    ragged_solo_equivalence_prompts: Option<String>,
    /// Stage 3b (#750): the **mid-flight** continuous-batch solo-equivalence gate.
    /// This is the gate for what 3b adds on top of 3a's ragged geometry —
    /// admitting a fresh request into a freed slot *between steps* while its peers
    /// keep decoding. Give a `||`-separated list of N prompts (N MUST exceed the
    /// batch width `--mid-flight-batch` so real backfill happens). The tool runs
    /// each prompt **alone** at batch 1 as the reference, then drives a continuous
    /// batch of width K: it seeds the first K prompts, and as rows finish it
    /// retires them (`deactivate_batch_row`) and admits a waiting prompt into the
    /// freed slot (`assign_batch_row`) mid-flight, sampling every row from host
    /// `[B,1,vocab]` logits (`decode_greedy_batch_ragged_logits`, the stage-3b
    /// host-logits seam). It asserts every admitted row is **byte-identical** to
    /// its solo stream — if slot reuse leaks stale KV, mask, or position, the
    /// admitted row diverges. The per-row admission step and lengths are printed
    /// so it is visible rows were admitted mid-flight, not all at step 0; a gate
    /// where every row started at step 0 is worthless and is rejected.
    #[arg(long)]
    mid_flight_solo_equivalence_prompts: Option<String>,
    /// Batch width K for `--mid-flight-solo-equivalence-prompts` (physical decode
    /// rows). Must be at least 2 and strictly less than the number of prompts so
    /// backfill actually occurs.
    #[arg(long, default_value_t = 2)]
    mid_flight_batch: usize,
    /// Drive `--mid-flight-solo-equivalence-prompts` through the real
    /// `ContinuousBatchManager` on the native backend (#750 stage 4) instead of
    /// the hand-rolled slot driver. This proves the *manager* — not a bespoke
    /// harness — admits rows mid-flight and reproduces each solo stream, and it
    /// reports the manager's capture stats and honest logits D2H cost. The gate
    /// still rejects a run where every row was admitted at step 0.
    #[arg(long, default_value_t = false)]
    mid_flight_via_manager: bool,
    /// Speculative decoding mode. `prompt-lookup` enables native n-gram
    /// speculation (exact verification; lossless vs greedy). Only supported on
    /// the single-model native `--steady` path; rejected with `--pipeline`.
    #[arg(long, value_enum, default_value_t = SpeculativeArg::None)]
    speculative: SpeculativeArg,
    /// Prompt-lookup n-gram key length (trailing context tokens matched to find
    /// a continuation). Only used when `--speculative prompt-lookup`.
    #[arg(long, default_value_t = 3)]
    spec_ngram: usize,
    /// Prompt-lookup draft width K: max continuation tokens proposed and
    /// verified per step. Only used when `--speculative prompt-lookup`.
    #[arg(long, default_value_t = 4)]
    spec_tokens: usize,
    /// Run the standalone base-decode on-GPU-argmax A/B benchmark: greedy decode
    /// with the host argmax (logits D2H + host reduction) vs the on-GPU argmax
    /// (`decode_greedy_batch`, no logits D2H), reporting tok/s for both and the
    /// token divergence between them. Pure base-decode measurement — no
    /// speculative / draft machinery. Selects the primary reported path via the
    /// `ONNX_GENAI_ONGPU_ARGMAX` env flag (default OFF = host argmax).
    #[arg(long, default_value_t = false)]
    ongpu_argmax_bench: bool,
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
    apply_speculative_options(&mut request.options, args);
    request
}

/// Wire the prompt-lookup speculative mode into the request options. A no-op for
/// `--speculative none`, so the default greedy fast path is untouched.
fn apply_speculative_options(options: &mut GenerateOptions, args: &Args) {
    match args.speculative {
        SpeculativeArg::None => {}
        SpeculativeArg::PromptLookup => {
            options.speculative_mode = Some(SpeculativeMode::PromptLookup {
                ngram: args.spec_ngram,
                max_tokens: args.spec_tokens,
            });
            options.num_speculative_tokens = Some(args.spec_tokens);
        }
    }
}

fn describe_speculative(args: &Args) -> String {
    match args.speculative {
        SpeculativeArg::None => "speculative: OFF (plain M=1 greedy)".to_string(),
        SpeculativeArg::PromptLookup => format!(
            "speculative: prompt-lookup ngram={} K={}",
            args.spec_ngram, args.spec_tokens
        ),
    }
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

/// Report prompt-lookup speculation diagnostics: proposals, exact-verify
/// acceptance rate, and the mean accepted-run length per verify step.
fn print_speculative_observability(stats: &SpeculativeStats) {
    let acceptance = if stats.proposed_tokens > 0 {
        stats.accepted_tokens as f64 / stats.proposed_tokens as f64 * 100.0
    } else {
        0.0
    };
    // Each verify step commits `accepted` draft tokens plus one free bonus
    // token, so mean tokens committed per verify = (accepted + steps) / steps.
    let tokens_per_step = if stats.verification_steps > 0 {
        (stats.accepted_tokens + stats.verification_steps) as f64 / stats.verification_steps as f64
    } else {
        0.0
    };
    println!(
        "speculative_stats: verify_steps={} proposed={} accepted={} acceptance={acceptance:.1}% \
         multi_token_accepts={} tokens_per_verify_step={tokens_per_step:.2}",
        stats.verification_steps,
        stats.proposed_tokens,
        stats.accepted_tokens,
        stats.multi_token_accepts,
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
        if stats.graph.device_token_loop_k > 0 || stats.graph.device_token_loop_steps > 0 {
            println!(
                "device_token_loop: k={} chained_steps={}",
                stats.graph.device_token_loop_k, stats.graph.device_token_loop_steps
            );
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
         page_ins_per_token={} zero_copy_bytes_per_token={} zero_copy_reads_per_token={} \
         zero_copy_binds={} host_registered_bytes={}",
        emitted_tokens,
        fmt(per_emitted_token(stats.htod_bytes, emitted_tokens)),
        fmt(per_emitted_token(stats.page_ins, emitted_tokens)),
        fmt(per_emitted_token(stats.zero_copy_bytes, emitted_tokens)),
        fmt(per_emitted_token(stats.zero_copy_reads, emitted_tokens)),
        stats.zero_copy_binds,
        stats.host_registered_bytes,
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
    // When the hybrid reads cold weights zero-copy in place, those bytes never
    // appear in htod traffic, so byte_hit_rate() falsely reads ~100%. Include
    // zero-copy PCIe traffic in the denominator for an honest residency figure.
    let zero_copy_byte_hit_rate = stats
        .zero_copy_byte_hit_rate()
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
         zero_copy_byte_hit_rate={} hit_bytes={} evictions={} bypassed_page_ins={} \
         bypassed_page_in_bytes={} bypassed_byte_share={}",
        stats.page_ins,
        stats.hits,
        hit_rate,
        byte_hit_rate,
        zero_copy_byte_hit_rate,
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
    print_weight_paging_key_trace(emitted_tokens);
    let (pinned_keys, pinned_bytes) = onnx_runtime_ep_cuda::pinned_hot_set();
    if pinned_keys > 0 {
        println!(
            "weight_offload_pin: pinned_keys={pinned_keys} pinned_bytes={pinned_bytes} \
             pinned_bytes_saved_per_step={pinned_bytes}"
        );
    }
}

/// Dump the per-key weight-paging trace (#837 item 3 characterisation). Empty
/// unless `ONNX_GENAI_WEIGHT_PAGING_KEY_TRACE` was set for the run, so this
/// prints nothing on a normal profile. `reads_per_step = reads / emitted_tokens`
/// (one decode step per emitted token) is the discriminator: a bypassed key with
/// `reads_per_step ~= 1` is read once per step, so admitting it saves its `len`
/// bytes per step but a reuse-frequency reservation has nothing to rank it on.
fn print_weight_paging_key_trace(emitted_tokens: u64) {
    let rows = onnx_runtime_ep_cuda::weight_paging_key_trace();
    if rows.is_empty() {
        return;
    }
    let steps = emitted_tokens.max(1) as f64;
    let mut bypass_keys = 0u64;
    let mut bypass_bytes_per_step = 0.0f64;
    let mut retained_keys = 0u64;
    for (_, row) in &rows {
        if row.bypass_page_ins > 0 {
            bypass_keys += 1;
            bypass_bytes_per_step += (row.bypass_page_ins * row.len) as f64 / steps;
        }
        if row.retained_page_ins > 0 && row.bypass_page_ins == 0 {
            retained_keys += 1;
        }
    }
    println!(
        "weight_paging_key_trace_summary: distinct_keys={} bypass_keys={} retained_only_keys={} \
         bypass_bytes_per_step={:.0} emitted_tokens={}",
        rows.len(),
        bypass_keys,
        retained_keys,
        bypass_bytes_per_step,
        emitted_tokens
    );
    // Head of the list is sorted by bytes re-streamed (bypass_page_ins * len).
    for (key, row) in rows.iter().take(40) {
        println!(
            "weight_paging_key: key={key} len={} reads={} reads_per_step={:.3} hits={} \
             retained_page_ins={} bypass_page_ins={} bypass_bytes_per_step={:.0}",
            row.len,
            row.reads(),
            row.reads() as f64 / steps,
            row.hits,
            row.retained_page_ins,
            row.bypass_page_ins,
            (row.bypass_page_ins * row.len) as f64 / steps,
        );
    }
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

/// Stage 2b-impl-4 (#750): sweep the **persistent batch-N decode path** — the
/// real captured decode step [`NativeDecodeSession::decode_greedy_batch`], not
/// the stateless fused forward the stage 2a/2b sweeps measure. For each batch
/// size `N` this rebuilds a fresh session pinned to batch `N`
/// (`ONNX_GENAI_NATIVE_DECODE_BATCH`), runs `tokens` batch-N greedy decode steps
/// from an empty past feeding an identical token to all `N` rows, and reports
/// the weight-offload counters and the CUDA-graph recapture counters per shape.
///
/// Because every row sees identical inputs from an identical (empty) initial KV
/// state, all `N` output rows must be byte-identical every step (a cross-row
/// corruption guard, the #892 detector) and row 0 must equal the batch-1 stream
/// (the batch-N row-identity guard the scoping doc §3.8 asks for). Token *values*
/// are content-arbitrary (a real per-sequence prompt needs the stage 2c seeding
/// caller), but the weight/page/capture counters are content-invariant (#884),
/// so they are the trustworthy batch-N measurement.
fn run_native_decode_batch_sweep(
    model_dir: &Path,
    device: NativeDecodeDevice,
    decode_precision: DecodePrecision,
    prompt_tokens: &[u32],
    batch_sizes: &[usize],
    tokens: usize,
    context: usize,
) -> Result<()> {
    if batch_sizes.is_empty() {
        bail!("--native-decode-batch-sweep requires at least one batch size");
    }
    let own_pid = std::process::id();
    println!(
        "native_decode_batch_sweep: own_pid={own_pid} batch_sizes={batch_sizes:?} tokens={tokens} \
         context={context} (persistent captured decode path; identical token fanned to all N rows; \
         `context` unmeasured decode steps build a length-N×context KV occupancy to create weight \
         pressure, then the weight-offload counters are reset and `tokens` steps are measured; token \
         values are content-arbitrary, weight/page/capture counters are content-invariant #884)"
    );

    // Row-0 token stream of the batch-1 pass, used as the cross-batch identity
    // reference for every N > 1 pass.
    let mut batch1_reference: Option<Vec<u32>> = None;

    for &batch in batch_sizes {
        if batch == 0 {
            bail!("--native-decode-batch-sweep batch sizes must be > 0");
        }
        report_foreign_compute_apps(own_pid);

        // SAFETY: single-threaded benchmark setup; the batch extent is read once
        // at session construction below.
        unsafe {
            std::env::set_var("ONNX_GENAI_NATIVE_DECODE_BATCH", batch.to_string());
        }
        // Build the *fully governed* engine so the native decode session carries
        // the effective weight-offload policy (managed no-spill + stable-VA
        // authority) instead of the conservative pointer-unstable default that
        // `load_with_resolved_io` hardcodes. Only through this path can whole-step
        // CUDA-graph capture stay ON while weights stream: the earlier bare-load
        // sweep reported a capture decline that was a harness artifact of the
        // default, not a runtime property. The batch extent is read from
        // ONNX_GENAI_NATIVE_DECODE_BATCH during the native session load below.
        let mut config = EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            decode_precision,
            ..EngineConfig::default()
        };
        config.native_device = Some(device.clone());
        apply_vram_limit_env(&mut config)?;
        let mut engine = Engine::from_dir(model_dir, config).with_context(|| {
            format!(
                "load governed engine {} at batch {batch}",
                model_dir.display()
            )
        })?;
        let session = engine.native_decode_session_mut().ok_or_else(|| {
            anyhow::anyhow!(
                "engine did not build a native decode session at batch {batch}; the model likely \
                 resolved to the ORT backend (batch-N needs the native CUDA persistent path)"
            )
        })?;
        let bound = session.native_decode_batch();
        if bound != batch {
            bail!(
                "requested batch {batch} but the session bound batch {bound}; the model likely \
                 declined a CUDA persistent decode session (batch-N needs the captured CUDA path)"
            );
        }

        onnx_runtime_ep_cuda::reset_global_offload_stats();
        let before = session.cuda_kv_debug_stats();

        let step_token = prompt_tokens[0];

        // Context warmup (unmeasured): grow the KV to `context` tokens per row so
        // the committed KV occupancy is `N × context × kv_bytes_per_token`. This
        // shrinks the elastic weight budget (#866) and forces the over-budget
        // model to stream weights during the *measured* steps below. All rows see
        // the identical token, so row identity is preserved into the measured
        // window. Stats are reset AFTER this warmup so first-touch admission is
        // not attributed to the measurement.
        for step in 0..context {
            let inputs = vec![step_token; batch];
            let rows = session
                .decode_greedy_batch(&inputs, step)
                .with_context(|| format!("batch {batch} context step {step}"))?;
            debug_assert_eq!(rows.len(), batch);
        }

        onnx_runtime_ep_cuda::reset_global_offload_stats();
        let before_measured = session.cuda_kv_debug_stats();

        let mut row0_stream = Vec::with_capacity(tokens);
        for step in 0..tokens {
            let inputs = vec![step_token; batch];
            let past_len = context + step;
            let rows = session
                .decode_greedy_batch(&inputs, past_len)
                .with_context(|| format!("batch {batch} decode step {step}"))?;
            if rows.len() != batch {
                bail!(
                    "batch {batch} decode step {step} returned {} rows, expected {batch}",
                    rows.len()
                );
            }
            for (row, &token) in rows.iter().enumerate() {
                if token != rows[0] {
                    bail!(
                        "batch {batch} row-identity VIOLATION at step {step}: row {row} produced \
                         token {token} but row 0 produced {} — the batch grid corrupted a row \
                         (the #892 failure class)",
                        rows[0]
                    );
                }
            }
            row0_stream.push(rows[0]);
        }

        println!("--- native_decode batch={batch} ---");
        println!(
            "native_decode_batch_row_identity: batch={batch} all_rows_equal_row0=true steps={tokens}"
        );
        println!("native_decode_batch_row0_stream: batch={batch} {row0_stream:?}");
        match &batch1_reference {
            None if batch == 1 => batch1_reference = Some(row0_stream.clone()),
            Some(reference) => {
                let matches = *reference == row0_stream;
                println!(
                    "native_decode_batch_cross_identity: batch={batch} row0_matches_batch1={matches}"
                );
                if !matches {
                    println!(
                        "native_decode_batch_cross_identity_detail: batch1={reference:?} \
                         batch{batch}_row0={row0_stream:?}"
                    );
                }
            }
            None => {
                println!(
                    "native_decode_batch_cross_identity: batch={batch} row0_matches_batch1=unknown \
                     (batch 1 not in this sweep; include 1 to anchor the reference)"
                );
            }
        }

        // Recapture-per-shape: the CUDA-graph counters. The *measured-window*
        // delta (post-warmup) answers whether capture survives steady batch-N
        // decode at the seeded context; the *total* delta (from load) also folds
        // in the warmup's bucket-growth recaptures. `invalidations`/`captures` > 1
        // in the measured window means capture did not survive the batch shape as
        // implemented (a first-class result).
        if let Some(stats) = session.cuda_kv_debug_stats() {
            let graph_delta =
                |base: &Option<onnx_genai_engine::native_decode::CudaKvDebugStats>| {
                    base.as_ref()
                        .map(|b| {
                            (
                                b.graph.captures,
                                b.graph.replays,
                                b.graph.fallbacks,
                                b.graph.invalidations,
                            )
                        })
                        .unwrap_or((0, 0, 0, 0))
                };
            let (mc, mr, mf, mi) = graph_delta(&before_measured);
            println!(
                "native_decode_batch_cuda_graph_measured: batch={batch} enabled={} captures={} \
                 replays={} fallbacks={} invalidations={}",
                stats.graph.enabled,
                stats.graph.captures.saturating_sub(mc),
                stats.graph.replays.saturating_sub(mr),
                stats.graph.fallbacks.saturating_sub(mf),
                stats.graph.invalidations.saturating_sub(mi),
            );
            let (tc, tr, tf, ti) = graph_delta(&before);
            println!(
                "native_decode_batch_cuda_graph_total: batch={batch} captures={} replays={} \
                 fallbacks={} invalidations={}",
                stats.graph.captures.saturating_sub(tc),
                stats.graph.replays.saturating_sub(tr),
                stats.graph.fallbacks.saturating_sub(tf),
                stats.graph.invalidations.saturating_sub(ti),
            );
            if let Some(reason) = &stats.graph.decline_reason {
                println!("native_decode_batch_cuda_graph_decline: batch={batch} {reason}");
            }
            if let Some(decision) = &stats.graph.growth_decision {
                println!("native_decode_batch_cuda_graph_growth: batch={batch} {decision}");
            }
        }

        // Weight-streaming amortization: emitted tokens = steps × rows, so
        // `htod_bytes_per_token` is per produced token and should fall ~1/N,
        // partly offset upward as N× KV occupancy shrinks the elastic weight
        // budget (#866). `byte_hit_rate` carries that #866 offset.
        let emitted = (tokens as u64).saturating_mul(batch as u64);
        print_weight_offload_observability(emitted);

        // `session` borrows `engine`; drop the engine to release the native
        // session (and its VRAM) before the next batch shape reloads.
        let _ = session;
        drop(engine);
    }

    // SAFETY: single-threaded teardown.
    unsafe {
        std::env::remove_var("ONNX_GENAI_NATIVE_DECODE_BATCH");
    }
    Ok(())
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

/// Build a fully governed native engine pinned to `batch` decode rows and hand
/// back its native decode session mutably. Mirrors the construction in
/// [`run_native_decode_batch_sweep`] so the weight-offload policy and stable-VA
/// authority are the governed ones (trap #1: ask the runtime, do not tell it).
/// The batch extent is read from `ONNX_GENAI_NATIVE_DECODE_BATCH`, which the
/// caller sets before invoking.
fn build_governed_batch_engine(
    model_dir: &Path,
    device: &NativeDecodeDevice,
    decode_precision: DecodePrecision,
    batch: usize,
) -> Result<Engine> {
    let mut config = EngineConfig {
        decode_backend: EngineDecodeBackend::Native,
        decode_precision,
        ..EngineConfig::default()
    };
    config.native_device = Some(device.clone());
    apply_vram_limit_env(&mut config)?;
    let mut engine = Engine::from_dir(model_dir, config).with_context(|| {
        format!(
            "load governed engine {} at batch {batch}",
            model_dir.display()
        )
    })?;
    let bound = engine
        .native_decode_session_mut()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "engine did not build a native decode session at batch {batch}; the model likely \
                 resolved to the ORT backend (batch-N needs the native CUDA persistent path)"
            )
        })?
        .native_decode_batch();
    if bound != batch {
        bail!(
            "requested batch {batch} but the session bound batch {bound}; the model likely declined \
             a CUDA persistent decode session (batch-N needs the captured CUDA path)"
        );
    }
    Ok(engine)
}

/// Seed `prompts` (row `b` = prompt `b`) into `session`'s `batch` decode rows by
/// lockstep single-token stepping, then greedily generate `gen_tokens` more per
/// row. Returns one `gen_tokens`-long token stream per row. All prompts must be
/// the same length `L` (uniform batch): every step feeds one token per row at
/// the shared position and the shared mask window advances by one, so ragged
/// lengths are not representable — the caller truncates to a common `L`.
///
/// The greedy token returned by the final prefill step (consuming each row's
/// last prompt token) is that row's first generated token; each subsequent step
/// feeds the row its own previous token, so rows never share input.
fn drive_uniform_batch(
    session: &mut NativeDecodeSession,
    prompts: &[Vec<u32>],
    gen_tokens: usize,
) -> Result<Vec<Vec<u32>>> {
    let batch = prompts.len();
    if batch == 0 {
        bail!("drive_uniform_batch requires at least one prompt");
    }
    let prompt_len = prompts[0].len();
    if prompt_len == 0 {
        bail!("drive_uniform_batch requires a non-empty prompt");
    }
    if prompts.iter().any(|prompt| prompt.len() != prompt_len) {
        bail!("drive_uniform_batch requires every prompt to share the uniform length {prompt_len}");
    }
    session.reset()?;

    // Prefill lockstep: at position `p` every row is fed its own prompt token.
    let mut row_tokens = vec![0u32; batch];
    for position in 0..prompt_len {
        let inputs: Vec<u32> = prompts.iter().map(|prompt| prompt[position]).collect();
        let rows = session
            .decode_greedy_batch(&inputs, position)
            .with_context(|| format!("uniform batch prefill step at position {position}"))?;
        if rows.len() != batch {
            bail!(
                "prefill step {position} returned {} rows, expected {batch}",
                rows.len()
            );
        }
        row_tokens = rows;
    }

    // The final prefill step's greedy result is each row's first generated token.
    let mut streams: Vec<Vec<u32>> = row_tokens.iter().map(|&token| vec![token]).collect();
    for step in 0..gen_tokens.saturating_sub(1) {
        let inputs: Vec<u32> = streams
            .iter()
            .map(|stream| *stream.last().expect("stream seeded with the first token"))
            .collect();
        let past_len = prompt_len + step;
        let rows = session
            .decode_greedy_batch(&inputs, past_len)
            .with_context(|| format!("uniform batch decode step {step} at past_len {past_len}"))?;
        if rows.len() != batch {
            bail!(
                "decode step {step} returned {} rows, expected {batch}",
                rows.len()
            );
        }
        for (row, &token) in rows.iter().enumerate() {
            streams[row].push(token);
        }
    }
    Ok(streams)
}

/// Stage 2c (#750) batch-N solo-equivalence gate: seed K genuinely different
/// prompts into one batch-K decode session and assert every row reproduces the
/// exact token stream that prompt produces when run alone at batch 1. This is a
/// strictly stronger correctness bar than the identical-token sweep's row
/// identity: the rows carry different content, so cross-row KV bleed, a shared
/// mask, a wrong stride, or a position error all diverge here.
fn run_native_decode_solo_equivalence(
    model_dir: &Path,
    device: NativeDecodeDevice,
    decode_precision: DecodePrecision,
    tokenizer: &Tokenizer,
    prompt_spec: &str,
    gen_tokens: usize,
) -> Result<()> {
    let own_pid = std::process::id();
    report_foreign_compute_apps(own_pid);

    let prompt_texts: Vec<&str> = prompt_spec
        .split("||")
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .collect();
    if prompt_texts.len() < 2 {
        bail!(
            "--solo-equivalence-prompts needs at least 2 distinct prompts separated by '||' (got {})",
            prompt_texts.len()
        );
    }
    let mut prompts: Vec<Vec<u32>> = Vec::with_capacity(prompt_texts.len());
    for text in &prompt_texts {
        let tokens = tokenizer
            .encode(text)
            .with_context(|| format!("tokenize solo-equivalence prompt {text:?}"))?;
        if tokens.is_empty() {
            bail!("solo-equivalence prompt {text:?} tokenized to an empty sequence");
        }
        prompts.push(tokens);
    }
    // Uniform batch requires equal lengths, so truncate every prompt to the
    // shortest common token length. This is a real, named constraint of the
    // native uniform-batch path (one shared mask window / shared position per
    // step); a ragged batch is not representable without per-row masks.
    let uniform_len = prompts
        .iter()
        .map(Vec::len)
        .min()
        .expect("non-empty prompt set");
    if uniform_len == 0 {
        bail!("solo-equivalence prompts share no common non-empty prefix");
    }
    for prompt in &mut prompts {
        prompt.truncate(uniform_len);
    }
    let batch = prompts.len();
    println!(
        "native_decode_batch_solo_equivalence: own_pid={own_pid} prompts={batch} uniform_len={uniform_len} \
         gen_tokens={gen_tokens} (uniform-batch constraint: every prompt truncated to the shortest \
         common token length; rows carry genuinely different content)"
    );
    for (row, (text, prompt)) in prompt_texts.iter().zip(prompts.iter()).enumerate() {
        println!(
            "native_decode_batch_solo_equivalence_prompt: row={row} text={text:?} tokens={prompt:?}"
        );
    }

    // 1) Solo reference: each prompt alone at batch 1 (reset between prompts).
    // SAFETY: single-threaded benchmark setup.
    unsafe {
        std::env::set_var("ONNX_GENAI_NATIVE_DECODE_BATCH", "1");
    }
    let mut solo_engine = build_governed_batch_engine(model_dir, &device, decode_precision, 1)?;
    let solo_session = solo_engine
        .native_decode_session_mut()
        .expect("batch-1 native session");
    let mut solo_streams: Vec<Vec<u32>> = Vec::with_capacity(batch);
    for (row, prompt) in prompts.iter().enumerate() {
        let stream = drive_uniform_batch(solo_session, std::slice::from_ref(prompt), gen_tokens)?;
        println!(
            "native_decode_batch_solo_equivalence_solo: row={row} stream={:?}",
            stream[0]
        );
        solo_streams.push(stream.into_iter().next().expect("one solo row"));
    }
    drop(solo_engine);

    // 2) Batch-K: all K prompts seeded into one session, stepping together.
    // SAFETY: single-threaded benchmark setup.
    unsafe {
        std::env::set_var("ONNX_GENAI_NATIVE_DECODE_BATCH", batch.to_string());
    }
    let mut batch_engine =
        build_governed_batch_engine(model_dir, &device, decode_precision, batch)?;
    let batch_session = batch_engine
        .native_decode_session_mut()
        .expect("batch-K native session");
    let batch_streams = drive_uniform_batch(batch_session, &prompts, gen_tokens)?;
    drop(batch_engine);

    // SAFETY: single-threaded teardown.
    unsafe {
        std::env::remove_var("ONNX_GENAI_NATIVE_DECODE_BATCH");
    }

    // 3) Assert each batch row reproduces its solo stream byte-for-byte.
    let mut all_match = true;
    for (row, (solo, batched)) in solo_streams.iter().zip(batch_streams.iter()).enumerate() {
        let matches = solo == batched;
        all_match &= matches;
        println!(
            "native_decode_batch_solo_equivalence_row: row={row} matches_solo={matches} \
             batch_stream={batched:?}"
        );
        if !matches {
            println!(
                "native_decode_batch_solo_equivalence_detail: row={row} solo={solo:?} batch={batched:?}"
            );
        }
    }
    println!(
        "native_decode_batch_solo_equivalence_result: prompts={batch} uniform_len={uniform_len} \
         gen_tokens={gen_tokens} all_rows_match_solo={all_match}"
    );
    if !all_match {
        bail!(
            "solo-equivalence FAILED: at least one batch row diverged from its solo batch-1 stream \
             — rows are observing each other (cross-row KV bleed, shared mask, wrong stride, or \
             position error)"
        );
    }
    Ok(())
}

/// Seed `prompts` of *genuinely different* token lengths into `session`'s
/// `batch` decode rows using the ragged per-row geometry (stage 3a, #750), then
/// greedily generate `gen_tokens` per row. Returns one `gen_tokens`-long token
/// stream per row.
///
/// Prefills are **right-aligned**: a row with a shorter prompt is held (an
/// `advance=false` step that reprocesses its first token at position 0 without
/// growing its length) for `max_len − len_r` steps, then prefills its prompt so
/// every row consumes its last prompt token on the same final prefill step. From
/// the moment a shorter row begins, it sits at a different logical length than
/// its still-prefilling peers — so the batch is genuinely ragged (different
/// per-row mask windows and positions in one fused forward), not a uniform batch
/// of equal-length rows. A held row's inert write lands at its own offset 0 and
/// is overwritten byte-for-byte when its real prefill starts, so a row's KV
/// state after prefill is identical to running it alone.
fn drive_ragged_batch(
    session: &mut NativeDecodeSession,
    prompts: &[Vec<u32>],
    gen_tokens: usize,
) -> Result<Vec<Vec<u32>>> {
    let batch = prompts.len();
    if batch == 0 {
        bail!("drive_ragged_batch requires at least one prompt");
    }
    let lens: Vec<usize> = prompts.iter().map(Vec::len).collect();
    if lens.iter().any(|&len| len == 0) {
        bail!("drive_ragged_batch requires every prompt to be non-empty");
    }
    let lmax = lens.iter().copied().max().expect("non-empty prompt set");
    session.reset()?;

    // Per-row committed length (== the `past_len` fed on the row's next step).
    let mut row_len = vec![0usize; batch];
    // Each row's first generated token, produced when it consumes its last
    // prompt token (all rows reach that on the final right-aligned prefill step).
    let mut first_gen = vec![0u32; batch];

    for i in 0..lmax {
        let mut tokens = vec![0u32; batch];
        let mut past_lens = vec![0usize; batch];
        let mut advances = vec![false; batch];
        for row in 0..batch {
            let offset = lmax - lens[row];
            if i < offset {
                // Held: reprocess this row's first token at position 0; length
                // does not grow, so the row waits at length 0 while its peers
                // prefill ahead of it.
                tokens[row] = prompts[row][0];
                past_lens[row] = 0;
                advances[row] = false;
            } else {
                let local = i - offset;
                tokens[row] = prompts[row][local];
                past_lens[row] = local;
                advances[row] = true;
            }
        }
        let rows = session
            .decode_greedy_batch_ragged(&tokens, &past_lens, &advances)
            .with_context(|| format!("ragged batch prefill step {i}"))?;
        if rows.len() != batch {
            bail!(
                "prefill step {i} returned {} rows, expected {batch}",
                rows.len()
            );
        }
        // Prove the device saw genuinely different per-row lengths in one fused
        // forward: on the final prefill step every row is at its own prompt
        // length, so `past_lens` spans distinct values (unless all prompts were
        // the same length, which the gate refuses up front).
        if i + 1 == lmax {
            println!(
                "native_decode_ragged_geometry: step={i} (final prefill) per_row_past_lens={past_lens:?} \
                 advances={advances:?} — one fused forward, rows at distinct lengths"
            );
        }
        for row in 0..batch {
            if advances[row] {
                row_len[row] += 1;
                first_gen[row] = rows[row];
            }
        }
    }

    // Every row has now prefilled its whole prompt (length == its prompt length),
    // and `first_gen` holds each row's first generated token.
    for (row, &len) in lens.iter().enumerate() {
        if row_len[row] != len {
            bail!(
                "ragged prefill left row {row} at length {} (expected {len})",
                row_len[row]
            );
        }
    }
    let mut streams: Vec<Vec<u32>> = first_gen.iter().map(|&token| vec![token]).collect();

    // Generation: every row advances each step, staying ragged (the per-row
    // lengths keep their prefill offsets).
    for _ in 0..gen_tokens.saturating_sub(1) {
        let tokens: Vec<u32> = streams
            .iter()
            .map(|stream| *stream.last().expect("stream seeded with first token"))
            .collect();
        let past_lens = row_len.clone();
        let advances = vec![true; batch];
        let rows = session
            .decode_greedy_batch_ragged(&tokens, &past_lens, &advances)
            .context("ragged batch decode step")?;
        if rows.len() != batch {
            bail!("decode step returned {} rows, expected {batch}", rows.len());
        }
        for (row, &token) in rows.iter().enumerate() {
            row_len[row] += 1;
            streams[row].push(token);
        }
    }
    Ok(streams)
}

/// Stage 3a (#750) batch-N **ragged** solo-equivalence gate: seed K prompts of
/// genuinely different token lengths into one ragged batch-K decode session and
/// assert every row reproduces the exact token stream that prompt produces run
/// alone at batch 1. Because the rows sit at different logical lengths in the
/// same fused forward, a wrong per-row mask window, position, or length would
/// diverge — the direct test of the stage-3a geometry.
fn run_native_decode_ragged_solo_equivalence(
    model_dir: &Path,
    device: NativeDecodeDevice,
    decode_precision: DecodePrecision,
    tokenizer: &Tokenizer,
    prompt_spec: &str,
    gen_tokens: usize,
) -> Result<()> {
    let own_pid = std::process::id();
    report_foreign_compute_apps(own_pid);

    let prompt_texts: Vec<&str> = prompt_spec
        .split("||")
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .collect();
    if prompt_texts.len() < 2 {
        bail!(
            "--ragged-solo-equivalence-prompts needs at least 2 distinct prompts separated by \
             '||' (got {})",
            prompt_texts.len()
        );
    }
    let mut prompts: Vec<Vec<u32>> = Vec::with_capacity(prompt_texts.len());
    for text in &prompt_texts {
        let tokens = tokenizer
            .encode(text)
            .with_context(|| format!("tokenize ragged solo-equivalence prompt {text:?}"))?;
        if tokens.is_empty() {
            bail!("ragged solo-equivalence prompt {text:?} tokenized to an empty sequence");
        }
        prompts.push(tokens);
    }
    let lens: Vec<usize> = prompts.iter().map(Vec::len).collect();
    // The whole point of the ragged gate is different-length rows: refuse a run
    // that would silently degenerate into a uniform batch. A gate that passes
    // because every prompt happened to be the same length is worthless.
    let min_len = lens.iter().copied().min().expect("non-empty prompt set");
    let max_len = lens.iter().copied().max().expect("non-empty prompt set");
    let batch = prompts.len();
    println!(
        "native_decode_batch_ragged_solo_equivalence: own_pid={own_pid} prompts={batch} \
         row_lens={lens:?} min_len={min_len} max_len={max_len} gen_tokens={gen_tokens} \
         (ragged batch: prompts kept at their genuinely different token lengths, NOT truncated)"
    );
    for (row, (text, prompt)) in prompt_texts.iter().zip(prompts.iter()).enumerate() {
        println!(
            "native_decode_batch_ragged_solo_equivalence_prompt: row={row} len={} text={text:?} \
             tokens={prompt:?}",
            prompt.len()
        );
    }
    if min_len == max_len {
        bail!(
            "ragged solo-equivalence requires prompts of genuinely different token lengths, but \
             every prompt tokenized to {min_len} tokens — this would degenerate into a uniform \
             batch and prove nothing about ragged geometry. Choose prompts whose tokenizations \
             differ in length."
        );
    }

    // 1) Solo reference: each prompt alone at batch 1 (reset between prompts).
    // SAFETY: single-threaded benchmark setup.
    unsafe {
        std::env::set_var("ONNX_GENAI_NATIVE_DECODE_BATCH", "1");
    }
    let mut solo_engine = build_governed_batch_engine(model_dir, &device, decode_precision, 1)?;
    let solo_session = solo_engine
        .native_decode_session_mut()
        .expect("batch-1 native session");
    let mut solo_streams: Vec<Vec<u32>> = Vec::with_capacity(batch);
    for (row, prompt) in prompts.iter().enumerate() {
        let stream = drive_uniform_batch(solo_session, std::slice::from_ref(prompt), gen_tokens)?;
        println!(
            "native_decode_batch_ragged_solo_equivalence_solo: row={row} len={} stream={:?}",
            prompt.len(),
            stream[0]
        );
        solo_streams.push(stream.into_iter().next().expect("one solo row"));
    }
    drop(solo_engine);

    // 2) Ragged batch-K: all K different-length prompts seeded into one session.
    // SAFETY: single-threaded benchmark setup.
    unsafe {
        std::env::set_var("ONNX_GENAI_NATIVE_DECODE_BATCH", batch.to_string());
    }
    let mut batch_engine =
        build_governed_batch_engine(model_dir, &device, decode_precision, batch)?;
    let batch_session = batch_engine
        .native_decode_session_mut()
        .expect("batch-K native session");
    let batch_streams = drive_ragged_batch(batch_session, &prompts, gen_tokens)?;
    // Prove the captured decode graph survived the ragged per-row geometry: only
    // mask/position *values* vary per step (the mask width stays frozen to the
    // physical bucket), so capture must hold exactly as it does for a uniform
    // batch (captures>0, invalidations==0, fallbacks==0). A ragged run that
    // silently fell back to eager would show up here.
    if let Some(stats) = batch_session.cuda_kv_debug_stats() {
        println!(
            "native_decode_batch_ragged_solo_equivalence_capture: graph_enabled={} captures={} \
             replays={} fallbacks={} invalidations={} final_logical_len={}",
            stats.graph.enabled,
            stats.graph.captures,
            stats.graph.replays,
            stats.graph.fallbacks,
            stats.graph.invalidations,
            stats.logical_len
        );
    }
    drop(batch_engine);

    // SAFETY: single-threaded teardown.
    unsafe {
        std::env::remove_var("ONNX_GENAI_NATIVE_DECODE_BATCH");
    }

    // 3) Assert each ragged batch row reproduces its solo stream byte-for-byte.
    let mut all_match = true;
    for (row, (solo, batched)) in solo_streams.iter().zip(batch_streams.iter()).enumerate() {
        let matches = solo == batched;
        all_match &= matches;
        println!(
            "native_decode_batch_ragged_solo_equivalence_row: row={row} len={} matches_solo={matches} \
             batch_stream={batched:?}",
            lens[row]
        );
        if !matches {
            println!(
                "native_decode_batch_ragged_solo_equivalence_detail: row={row} solo={solo:?} \
                 batch={batched:?}"
            );
        }
    }
    println!(
        "native_decode_batch_ragged_solo_equivalence_result: prompts={batch} row_lens={lens:?} \
         min_len={min_len} max_len={max_len} gen_tokens={gen_tokens} all_rows_match_solo={all_match}"
    );
    if !all_match {
        bail!(
            "ragged solo-equivalence FAILED: at least one different-length batch row diverged from \
             its solo batch-1 stream — the ragged per-row geometry (mask window, position, or \
             length) is wrong or rows are observing each other"
        );
    }
    Ok(())
}

/// Lowest-index argmax over a host `[vocab]` logits row, matching the native
/// device-argmax tie-break (ties resolve to the lowest token id). This is used
/// by the stage-3b mid-flight gate so a host-logits selection is byte-comparable
/// to the device-argmax solo reference — the cross-check that the host-logits
/// seam returns the same distribution the device argmax reduces.
fn host_argmax(logits: &[f32]) -> u32 {
    let mut best_index = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in logits.iter().enumerate() {
        if value > best_value {
            best_value = value;
            best_index = index;
        }
    }
    best_index as u32
}

/// Opt-in on-GPU-argmax base-decode flag (`ONNX_GENAI_ONGPU_ARGMAX`). Default
/// OFF. When ON, the base-decode greedy A/B benchmark reports the on-GPU-argmax
/// (`decode_greedy_batch`) path as its primary throughput number; when OFF it
/// reports the host-argmax path. Both paths are always measured and their token
/// streams cross-checked for byte-identity regardless of the flag.
fn ongpu_argmax_enabled() -> bool {
    matches!(
        std::env::var("ONNX_GENAI_ONGPU_ARGMAX").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("on") | Some("ON")
    )
}

/// Lowest-index-on-ties argmax: strict `>` from `-inf` keeps the FIRST maximum,
/// matching the ONNX/ORT canonical greedy tie-break (`ArgMax` with
/// `select_last_index=false`), the host sampler `sample_greedy` ("ties keep the
/// lowest token id"), and the reconciled on-GPU `device_argmax` (lowest global
/// index). The host reference and the device path are therefore byte-identical
/// even on the rare exact fp16 tie at the argmax.
fn argmax_lowest_index(row: &[f32]) -> u32 {
    let mut best = f32::NEG_INFINITY;
    let mut best_index = 0u32;
    for (index, &value) in row.iter().enumerate() {
        if value > best {
            best = value;
            best_index = index as u32;
        }
    }
    best_index
}

/// Steady-window tok/s from per-token timestamps, excluding the first `skip`
/// emitted tokens (prefill / capture warmup) from the timed window.
fn steady_tok_s(times: &[Duration], skip: usize) -> f64 {
    if times.len() <= skip + 1 {
        if times.len() < 2 {
            return 0.0;
        }
        let wall = (times[times.len() - 1] - times[0]).as_secs_f64();
        return if wall > 0.0 {
            (times.len() - 1) as f64 / wall
        } else {
            0.0
        };
    }
    let decode_tokens = times.len() - skip;
    let wall = (times[times.len() - 1] - times[skip - 1]).as_secs_f64();
    if wall > 0.0 {
        decode_tokens as f64 / wall
    } else {
        0.0
    }
}

/// Fraction of positions where two token streams differ (byte-identity check).
fn token_divergence(a: &[u32], b: &[u32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return if a.len() == b.len() { 0.0 } else { 1.0 };
    }
    let mut diff = 0usize;
    for i in 0..n {
        if a[i] != b[i] {
            diff += 1;
        }
    }
    diff += a.len().max(b.len()) - n;
    diff as f64 / a.len().max(b.len()) as f64
}

/// Base-decode greedy with the HOST argmax: each step copies the full
/// `[vocab]` fp16 logits to the host (`decode`) and reduces on the CPU. This is
/// the pre-existing sampler path and the byte-identity reference.
fn greedy_decode_host(
    session: &mut NativeDecodeSession,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    eos: &[u32],
    decode_skip: usize,
) -> Result<(Vec<u32>, f64)> {
    session.reset()?;
    let mut logits = session
        .decode(prompt_tokens, 0)?
        .pop()
        .context("greedy prefill produced no logits")?;
    let mut generated = Vec::with_capacity(max_new_tokens);
    let mut times = Vec::with_capacity(max_new_tokens);
    let start = Instant::now();
    while generated.len() < max_new_tokens {
        let token = argmax_lowest_index(&logits);
        generated.push(token);
        times.push(start.elapsed());
        if eos.contains(&token) {
            break;
        }
        let past = session.current_len();
        logits = session
            .decode(&[token], past)?
            .pop()
            .context("greedy decode produced no logits")?;
    }
    Ok((generated, steady_tok_s(&times, decode_skip)))
}

/// Base-decode greedy with the ON-GPU argmax: each step selects the next token
/// with `decode_greedy_batch` (batch 1), which reduces over the vocab on the
/// GPU and returns only the token id — no `[vocab]` logits D2H, no host
/// reduction. Byte-identical to [`greedy_decode_host`] (device_argmax resolves
/// ties to the same lowest index — the ONNX/ORT canonical). The prefill first
/// token still comes from the
/// prefill logits (one-time, outside the steady window) so both streams start
/// from the same bootstrap.
fn greedy_decode_ongpu(
    session: &mut NativeDecodeSession,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    eos: &[u32],
    decode_skip: usize,
) -> Result<(Vec<u32>, f64)> {
    session.reset()?;
    let base_logits = session
        .decode(prompt_tokens, 0)?
        .pop()
        .context("greedy prefill produced no logits")?;
    let mut generated = Vec::with_capacity(max_new_tokens);
    let mut times = Vec::with_capacity(max_new_tokens);
    let start = Instant::now();
    let mut token = argmax_lowest_index(&base_logits);
    loop {
        generated.push(token);
        times.push(start.elapsed());
        if eos.contains(&token) || generated.len() >= max_new_tokens {
            break;
        }
        let past = session.current_len();
        token = session
            .decode_greedy_batch(std::slice::from_ref(&token), past)?
            .pop()
            .context("device-argmax greedy decode produced no token")?;
    }
    Ok((generated, steady_tok_s(&times, decode_skip)))
}

/// Wrap a user prompt in the Qwen2.5 chat template so the instruct target
/// produces a realistic assistant continuation.
fn qwen_chat_wrap(user: &str) -> String {
    format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
         <|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    )
}

/// Standalone base-decode on-GPU-argmax A/B benchmark. Measures greedy decode
/// with the host argmax (logits D2H + host reduction) vs the on-GPU argmax
/// (`decode_greedy_batch`, no logits D2H) over `--runs` runs, reports median
/// tok/s for both, the net ratio, and the token divergence between the two
/// streams (must be 0.000% — byte-identical). No speculative / draft machinery.
fn run_ongpu_argmax_bench(args: &Args, model_dir: &Path, device: NativeDecodeDevice) -> Result<()> {
    let tokenizer_path = tokenizer_file(model_dir);
    let tokenizer =
        Tokenizer::from_file(&tokenizer_path).context("load tokenizer.json beside decoder")?;
    let mut session = NativeDecodeSession::load_with_resolved_io(&model_file(model_dir), device)
        .with_context(|| format!("load native decoder {}", model_dir.display()))?;

    let prompt_text = qwen_chat_wrap(&args.prompt);
    let prompt_tokens = tokenizer.encode(&prompt_text).context("tokenize prompt")?;
    if prompt_tokens.is_empty() {
        bail!("prompt tokenized to an empty sequence");
    }
    let eos: Vec<u32> = vec![151645, 151643];
    let max_new = args.tokens;
    let skip = args.decode_skip.max(1);
    let primary_ongpu = ongpu_argmax_enabled();

    println!(
        "profile_native: ongpu-argmax-bench model={} layers={} prompt_tokens={} tokens={} \
         warmups={} runs={} decode_skip={} ONNX_GENAI_ONGPU_ARGMAX={} (primary_path={})",
        model_dir.display(),
        session.kv_layer_count(),
        prompt_tokens.len(),
        max_new,
        args.warmups,
        args.runs,
        skip,
        if primary_ongpu { "1" } else { "0" },
        if primary_ongpu { "ongpu" } else { "host" },
    );

    // Warmup both paths to arm CUDA-graph capture before timing.
    for _ in 0..args.warmups {
        let _ = greedy_decode_host(&mut session, &prompt_tokens, max_new.min(16), &eos, 1)?;
        let _ = greedy_decode_ongpu(&mut session, &prompt_tokens, max_new.min(16), &eos, 1)?;
    }

    let mut host_rates = Vec::with_capacity(args.runs);
    let mut ongpu_rates = Vec::with_capacity(args.runs);
    let mut host_tokens_ref: Option<Vec<u32>> = None;
    let mut ongpu_tokens_ref: Option<Vec<u32>> = None;
    for _ in 0..args.runs {
        let (host_tokens, host_rate) =
            greedy_decode_host(&mut session, &prompt_tokens, max_new, &eos, skip)?;
        let (ongpu_tokens, ongpu_rate) =
            greedy_decode_ongpu(&mut session, &prompt_tokens, max_new, &eos, skip)?;
        host_rates.push(host_rate);
        ongpu_rates.push(ongpu_rate);
        host_tokens_ref.get_or_insert(host_tokens);
        ongpu_tokens_ref.get_or_insert(ongpu_tokens);
    }

    let host_median = median(&mut host_rates);
    let ongpu_median = median(&mut ongpu_rates);
    let host_tokens = host_tokens_ref.expect("at least one run");
    let ongpu_tokens = ongpu_tokens_ref.expect("at least one run");
    let divergence = token_divergence(&ongpu_tokens, &host_tokens);
    let ratio = if host_median > 0.0 {
        ongpu_median / host_median
    } else {
        0.0
    };

    println!(
        "ongpu_argmax_bench: host_argmax={host_median:.2} tok/s ongpu_argmax={ongpu_median:.2} \
         tok/s net={ratio:.4}x divergence={:.3}% generated_tokens={}",
        divergence * 100.0,
        host_tokens.len(),
    );
    println!(
        "ongpu_argmax_bench: byte_identity={} (ongpu vs host greedy must be 0.000%)",
        if divergence == 0.0 { "PASS" } else { "FAIL" },
    );
    Ok(())
}

/// One live sequence occupying a continuous-batch decode slot in the stage-3b
/// mid-flight driver.
struct MidFlightJob {
    /// Index into the original request list (so its stream can be checked
    /// against the matching solo reference).
    req_index: usize,
    prompt: Vec<u32>,
    /// Logical length of this row (== the `past_len` fed on its next step). This
    /// mirrors `NativeDecodeSession::batch_row_len(slot)`.
    cursor: usize,
    /// Generated tokens produced so far (excludes the prompt).
    stream: Vec<u32>,
    /// How many tokens this request generates before it retires.
    gen_target: usize,
    /// Global step index at which this request was admitted into its slot. `> 0`
    /// proves it was backfilled mid-flight rather than seeded at the start.
    admitted_step: usize,
}

/// Stage 3b (#750) mid-flight continuous-batch solo-equivalence gate.
///
/// Drives a native CUDA batch of width `batch` as a *continuous* batch: it seeds
/// the first `batch` prompts, samples every row from host `[B,1,vocab]` logits
/// (the stage-3b host-logits seam), and as rows finish it retires them and
/// admits a waiting prompt into the freed slot **between steps** while the peers
/// keep decoding. Every admitted row must reproduce, byte-for-byte, the stream
/// that prompt produces run alone at batch 1; a stale-KV / stale-mask /
/// stale-position leak across the slot-reuse boundary would diverge. This is the
/// direct test of what 3b adds over 3a's ragged geometry.
fn run_native_decode_mid_flight_solo_equivalence(
    model_dir: &Path,
    device: NativeDecodeDevice,
    decode_precision: DecodePrecision,
    tokenizer: &Tokenizer,
    prompt_spec: &str,
    batch: usize,
    gen_tokens: usize,
) -> Result<()> {
    let own_pid = std::process::id();
    report_foreign_compute_apps(own_pid);

    if batch < 2 {
        bail!("--mid-flight-batch must be at least 2 (a single slot cannot demonstrate backfill)");
    }
    if gen_tokens < 2 {
        bail!("--tokens must be at least 2 for the mid-flight gate");
    }

    let prompt_texts: Vec<&str> = prompt_spec
        .split("||")
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .collect();
    if prompt_texts.len() <= batch {
        bail!(
            "--mid-flight-solo-equivalence-prompts needs strictly more than --mid-flight-batch \
             ({batch}) prompts so a waiting request is actually admitted into a freed slot \
             mid-flight (got {})",
            prompt_texts.len()
        );
    }
    let mut prompts: Vec<Vec<u32>> = Vec::with_capacity(prompt_texts.len());
    for text in &prompt_texts {
        let tokens = tokenizer
            .encode(text)
            .with_context(|| format!("tokenize mid-flight prompt {text:?}"))?;
        if tokens.is_empty() {
            bail!("mid-flight prompt {text:?} tokenized to an empty sequence");
        }
        prompts.push(tokens);
    }
    let requests = prompts.len();
    // Stagger the per-request generation lengths so rows finish at genuinely
    // different steps — otherwise every row would retire together and backfill
    // would happen in a single synchronized wave, which proves nothing about
    // mid-flight admission while peers decode. The lengths cycle in [1, gen_tokens].
    let gen_targets: Vec<usize> = (0..requests).map(|i| 1 + (i % gen_tokens)).collect();
    let lens: Vec<usize> = prompts.iter().map(Vec::len).collect();
    println!(
        "native_decode_mid_flight_solo_equivalence: own_pid={own_pid} requests={requests} \
         batch={batch} row_lens={lens:?} gen_targets={gen_targets:?} max_gen_tokens={gen_tokens}"
    );

    // 1) Solo reference: each prompt alone at batch 1 for its own gen_target.
    // SAFETY: single-threaded benchmark setup.
    unsafe {
        std::env::set_var("ONNX_GENAI_NATIVE_DECODE_BATCH", "1");
    }
    let mut solo_engine = build_governed_batch_engine(model_dir, &device, decode_precision, 1)?;
    let solo_session = solo_engine
        .native_decode_session_mut()
        .expect("batch-1 native session");
    let mut solo_streams: Vec<Vec<u32>> = Vec::with_capacity(requests);
    for (req, prompt) in prompts.iter().enumerate() {
        let stream =
            drive_uniform_batch(solo_session, std::slice::from_ref(prompt), gen_targets[req])?;
        let stream = stream.into_iter().next().expect("one solo row");
        println!(
            "native_decode_mid_flight_solo_equivalence_solo: req={req} prompt_len={} \
             gen_target={} stream={stream:?}",
            prompt.len(),
            gen_targets[req]
        );
        solo_streams.push(stream);
    }
    drop(solo_engine);

    // 2) Continuous batch of width `batch`: seed the first `batch`, backfill the
    // rest mid-flight as rows finish.
    // SAFETY: single-threaded benchmark setup.
    unsafe {
        std::env::set_var("ONNX_GENAI_NATIVE_DECODE_BATCH", batch.to_string());
    }
    let mut batch_engine =
        build_governed_batch_engine(model_dir, &device, decode_precision, batch)?;
    let batch_session = batch_engine
        .native_decode_session_mut()
        .expect("batch-K native session");
    batch_session.reset()?;

    // Start every physical slot inactive, then admit the first `batch` requests —
    // exercising deactivate_batch_row + assign_batch_row exactly as the
    // ContinuousBatchManager does at construction and admission.
    for slot in 0..batch {
        batch_session.deactivate_batch_row(slot)?;
    }
    let mut waiting: std::collections::VecDeque<usize> = (0..requests).collect();
    let mut slots: Vec<Option<MidFlightJob>> = (0..batch).map(|_| None).collect();
    let mut completed: Vec<Option<Vec<u32>>> = vec![None; requests];
    // Per-request admission/finish bookkeeping for the visibility print.
    let mut admitted_at: Vec<Option<usize>> = vec![None; requests];
    let mut finished_at: Vec<Option<usize>> = vec![None; requests];

    let admit = |session: &mut NativeDecodeSession,
                 slots: &mut Vec<Option<MidFlightJob>>,
                 waiting: &mut std::collections::VecDeque<usize>,
                 admitted_at: &mut [Option<usize>],
                 slot: usize,
                 step: usize|
     -> Result<()> {
        if let Some(req) = waiting.pop_front() {
            session.assign_batch_row(slot)?;
            admitted_at[req] = Some(step);
            slots[slot] = Some(MidFlightJob {
                req_index: req,
                prompt: prompts[req].clone(),
                cursor: 0,
                stream: Vec::new(),
                gen_target: gen_targets[req],
                admitted_step: step,
            });
        }
        Ok(())
    };

    // Initial admission wave (step 0). These necessarily start at step 0; the
    // gate's proof is that the *remaining* requests are admitted at step > 0.
    for slot in 0..batch {
        admit(
            batch_session,
            &mut slots,
            &mut waiting,
            &mut admitted_at,
            slot,
            0,
        )?;
    }

    let mut d2h_bytes_total: u128 = 0;
    let mut d2h_time_total = std::time::Duration::ZERO;
    let mut d2h_steps: u64 = 0;
    let mut mid_flight_admissions = 0usize;

    let max_steps = requests * (gen_tokens + 64) + 1024;
    let mut step = 0usize;
    while slots.iter().any(Option::is_some) || !waiting.is_empty() {
        if step >= max_steps {
            bail!(
                "mid-flight gate exceeded {max_steps} steps without draining — likely a stuck row"
            );
        }
        // Build the per-row inputs for this fused step. Empty slots are held
        // (advance=false) so they neither grow nor perturb the active rows.
        let mut tokens = vec![0u32; batch];
        let mut past_lens = vec![0usize; batch];
        let mut advances = vec![false; batch];
        for (slot, job) in slots.iter().enumerate() {
            if let Some(job) = job {
                let token = if job.cursor < job.prompt.len() {
                    job.prompt[job.cursor]
                } else {
                    *job.stream.last().expect("generating row has a last token")
                };
                // The row length the session tracks must match our cursor.
                let session_len = batch_session.batch_row_len(slot)?;
                if session_len != job.cursor {
                    bail!(
                        "mid-flight row {slot} cursor {} disagrees with session row_len {session_len}",
                        job.cursor
                    );
                }
                tokens[slot] = token;
                past_lens[slot] = job.cursor;
                advances[slot] = true;
            }
        }

        let result = batch_session
            .decode_greedy_batch_ragged_logits(&tokens, &past_lens, &advances)
            .with_context(|| format!("mid-flight host-logits step {step}"))?;
        d2h_bytes_total += result.d2h_bytes as u128;
        d2h_time_total += result.d2h_time;
        d2h_steps += 1;
        if result.logits.len() != batch {
            bail!(
                "mid-flight step {step} returned {} logit rows, expected {batch}",
                result.logits.len()
            );
        }

        // Consume each active row's logits, advance its cursor, record generated
        // tokens, and retire + backfill finished rows.
        for slot in 0..batch {
            let Some(job) = slots[slot].as_mut() else {
                continue;
            };
            let cursor_before = job.cursor;
            let token = host_argmax(&result.logits[slot]);
            job.cursor += 1;
            // Once the row has consumed its whole prompt, each step's logits row
            // is a genuine next-token prediction (the final prompt step yields the
            // first generated token, exactly as drive_uniform_batch records it).
            if cursor_before + 1 >= job.prompt.len() {
                job.stream.push(token);
            }
            if job.stream.len() >= job.gen_target {
                let req = job.req_index;
                let admitted_step = job.admitted_step;
                completed[req] = Some(std::mem::take(&mut job.stream));
                finished_at[req] = Some(step + 1);
                if admitted_step > 0 {
                    mid_flight_admissions += 1;
                }
                // Retire the slot, then admit a waiting request into it for the
                // NEXT step — the mid-flight backfill this gate exists to prove.
                batch_session.deactivate_batch_row(slot)?;
                slots[slot] = None;
                admit(
                    batch_session,
                    &mut slots,
                    &mut waiting,
                    &mut admitted_at,
                    slot,
                    step + 1,
                )?;
            }
        }
        step += 1;
    }

    // Capture must survive mid-flight admission: assign/deactivate are host-side
    // mask/length writes into the already-bound persistent bindings, so no
    // recapture is forced. Any invalidations here come from KV-bucket growth, not
    // admission; report the count so the reader can attribute it.
    let capture = batch_session.cuda_kv_debug_stats().map(|stats| {
        (
            stats.graph.enabled,
            stats.graph.captures,
            stats.graph.replays,
            stats.graph.fallbacks,
            stats.graph.invalidations,
        )
    });
    if let Some((enabled, captures, replays, fallbacks, invalidations)) = capture {
        println!(
            "native_decode_mid_flight_solo_equivalence_capture: graph_enabled={enabled} \
             captures={captures} replays={replays} fallbacks={fallbacks} \
             invalidations={invalidations} (assign/deactivate are host-side writes; any \
             invalidations are KV-growth boundaries, not admissions)"
        );
    }
    drop(batch_engine);
    // SAFETY: single-threaded teardown.
    unsafe {
        std::env::remove_var("ONNX_GENAI_NATIVE_DECODE_BATCH");
    }

    // Report the honest D2H cost of the host-logits seam.
    let d2h_ms = d2h_time_total.as_secs_f64() * 1000.0;
    let per_step_kb = if d2h_steps > 0 {
        (d2h_bytes_total as f64 / d2h_steps as f64) / 1024.0
    } else {
        0.0
    };
    println!(
        "native_decode_mid_flight_solo_equivalence_d2h: steps={d2h_steps} \
         total_logits_d2h_bytes={d2h_bytes_total} total_d2h_ms={d2h_ms:.3} \
         per_step_logits_kb={per_step_kb:.1} (host [B,1,vocab] logits read each step; this is the \
         cost the device-argmax fast path avoids)"
    );

    // Visibility: prove rows were admitted mid-flight (step > 0), not all at step 0.
    for req in 0..requests {
        println!(
            "native_decode_mid_flight_solo_equivalence_admission: req={req} prompt_len={} \
             gen_target={} admitted_at_step={:?} finished_at_step={:?}",
            lens[req], gen_targets[req], admitted_at[req], finished_at[req]
        );
    }
    if mid_flight_admissions == 0 {
        bail!(
            "mid-flight gate is worthless: every request was admitted at step 0 (no row was \
             backfilled into a freed slot while peers kept decoding). Increase the prompt count \
             or reduce --mid-flight-batch."
        );
    }

    // Assert every request's continuous-batch stream is byte-identical to solo.
    let mut all_match = true;
    for req in 0..requests {
        let solo = &solo_streams[req];
        let batched = completed[req]
            .as_ref()
            .with_context(|| format!("mid-flight request {req} never completed"))?;
        let matches = solo == batched;
        all_match &= matches;
        println!(
            "native_decode_mid_flight_solo_equivalence_row: req={req} prompt_len={} \
             admitted_at_step={:?} matches_solo={matches} batch_stream={batched:?}",
            lens[req], admitted_at[req]
        );
        if !matches {
            println!(
                "native_decode_mid_flight_solo_equivalence_detail: req={req} solo={solo:?} \
                 batch={batched:?}"
            );
        }
    }
    println!(
        "native_decode_mid_flight_solo_equivalence_result: requests={requests} batch={batch} \
         mid_flight_admissions={mid_flight_admissions} all_rows_match_solo={all_match}"
    );
    if !all_match {
        bail!(
            "mid-flight solo-equivalence FAILED: a row admitted into a recycled slot diverged from \
             its solo batch-1 stream — slot reuse leaked stale KV, mask, or position across the \
             admission boundary"
        );
    }
    Ok(())
}

/// Stage 4 (#750) mid-flight solo-equivalence gate driven through the **real**
/// [`onnx_genai_engine::ContinuousBatchManager`] on the native CUDA backend.
///
/// The sibling `run_native_decode_mid_flight_solo_equivalence` hand-drives the
/// native seams (`assign_batch_row`/`deactivate_batch_row`/
/// `decode_greedy_batch_ragged_logits`) directly, so it proves the seams are
/// correct but *not* that the manager wires them correctly. This gate closes
/// that gap: it builds a batch-`K` native engine, calls
/// `engine.continuous_batch_manager(K)`, submits every prompt as a greedy
/// `GenerateRequest`, and drives `manager.step()` to completion. The manager —
/// not this harness — owns admission (`admit_available_rows` → `assign_row`),
/// sampling (host `[B, 1, vocab]` logits through `BatchStepLogits::HostRows`),
/// and eviction (`deactivate_row`). Each request's `GenerateResult::token_ids`
/// must be byte-identical to the same prompt run **alone** at batch 1, and the
/// manager must admit at least one request at step > 0 (a run where every row
/// started at step 0 is rejected). Capture stats and the manager's logits D2H
/// cost are reported so the reader can confirm capture survived admission and
/// see the honest per-step transfer cost.
fn run_native_decode_mid_flight_manager_solo_equivalence(
    model_dir: &Path,
    device: NativeDecodeDevice,
    decode_precision: DecodePrecision,
    tokenizer: &Tokenizer,
    prompt_spec: &str,
    batch: usize,
    gen_tokens: usize,
) -> Result<()> {
    use onnx_genai_engine::{ContinuousBatchAdmission, ContinuousBatchEvent, GeneratePrompt};

    let own_pid = std::process::id();
    report_foreign_compute_apps(own_pid);

    if batch < 2 {
        bail!("--mid-flight-batch must be at least 2 (a single slot cannot demonstrate backfill)");
    }
    if gen_tokens < 2 {
        bail!("--tokens must be at least 2 for the mid-flight gate");
    }

    let prompt_texts: Vec<&str> = prompt_spec
        .split("||")
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .collect();
    if prompt_texts.len() <= batch {
        bail!(
            "--mid-flight-solo-equivalence-prompts needs strictly more than --mid-flight-batch \
             ({batch}) prompts so a waiting request is actually admitted into a freed slot \
             mid-flight (got {})",
            prompt_texts.len()
        );
    }
    let mut prompts: Vec<Vec<u32>> = Vec::with_capacity(prompt_texts.len());
    for text in &prompt_texts {
        let tokens = tokenizer
            .encode(text)
            .with_context(|| format!("tokenize mid-flight prompt {text:?}"))?;
        if tokens.is_empty() {
            bail!("mid-flight prompt {text:?} tokenized to an empty sequence");
        }
        prompts.push(tokens);
    }
    let requests = prompts.len();
    // Stagger the per-request generation lengths so rows retire at different
    // steps and backfill is genuinely mid-flight, matching the hand-driven gate.
    let gen_targets: Vec<usize> = (0..requests).map(|i| 1 + (i % gen_tokens)).collect();
    let lens: Vec<usize> = prompts.iter().map(Vec::len).collect();
    println!(
        "native_decode_mid_flight_manager_solo_equivalence: own_pid={own_pid} requests={requests} \
         batch={batch} row_lens={lens:?} gen_targets={gen_targets:?} max_gen_tokens={gen_tokens}"
    );

    // 1) Solo reference: each prompt alone at batch 1 for its own gen_target,
    // using the device-argmax path (decode_greedy_batch). Byte-identical to the
    // hand-driven gate's reference.
    // SAFETY: single-threaded benchmark setup.
    unsafe {
        std::env::set_var("ONNX_GENAI_NATIVE_DECODE_BATCH", "1");
    }
    let mut solo_engine = build_governed_batch_engine(model_dir, &device, decode_precision, 1)?;
    let solo_session = solo_engine
        .native_decode_session_mut()
        .expect("batch-1 native session");
    let mut solo_streams: Vec<Vec<u32>> = Vec::with_capacity(requests);
    for (req, prompt) in prompts.iter().enumerate() {
        let stream =
            drive_uniform_batch(solo_session, std::slice::from_ref(prompt), gen_targets[req])?;
        let stream = stream.into_iter().next().expect("one solo row");
        println!(
            "native_decode_mid_flight_manager_solo_equivalence_solo: req={req} prompt_len={} \
             gen_target={} stream={stream:?}",
            prompt.len(),
            gen_targets[req]
        );
        solo_streams.push(stream);
    }
    drop(solo_engine);

    // 2) Real manager over a batch-K native engine. The manager owns every
    // physical decode row: it deactivates all rows at construction, admits from
    // its FIFO queue into freed slots at the start of every step, samples from
    // host logits, and evicts finished rows.
    // SAFETY: single-threaded benchmark setup.
    unsafe {
        std::env::set_var("ONNX_GENAI_NATIVE_DECODE_BATCH", batch.to_string());
    }
    let mut batch_engine =
        build_governed_batch_engine(model_dir, &device, decode_precision, batch)?;
    let mut manager = batch_engine
        .continuous_batch_manager(batch)
        .context("build native ContinuousBatchManager (the #750 stage-4 wiring under test)")?;

    // Submit every request as a greedy generation. `temperature = 0.0` and
    // `greedy = true` force lowest-index argmax (the exact tie-break the native
    // device argmax uses), and `stop_on_eos = false` makes the manager generate
    // exactly `gen_target` tokens like `drive_uniform_batch` — so the streams are
    // compared on equal footing.
    let mut handle_to_req: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::with_capacity(requests);
    for (req, prompt) in prompts.iter().enumerate() {
        let options = GenerateOptions {
            max_new_tokens: gen_targets[req],
            temperature: 0.0,
            greedy: true,
            stop_on_eos: false,
            ..GenerateOptions::default()
        };
        let request = GenerateRequest {
            prompt: GeneratePrompt::TokenIds(prompt.clone()),
            options,
        };
        let handle = manager
            .submit(request)
            .with_context(|| format!("submit mid-flight request {req} to the manager"))?;
        if handle_to_req.insert(handle.id, req).is_some() {
            bail!(
                "manager returned a duplicate handle id {} for req {req}",
                handle.id
            );
        }
    }

    let mut admitted_at: Vec<Option<usize>> = vec![None; requests];
    let mut finished_at: Vec<Option<usize>> = vec![None; requests];
    let mut completed: Vec<Option<Vec<u32>>> = vec![None; requests];
    let mut mid_flight_admissions = 0usize;

    let record_admissions = |admissions: Vec<ContinuousBatchAdmission>,
                             step: usize,
                             admitted_at: &mut [Option<usize>],
                             mid_flight_admissions: &mut usize|
     -> Result<()> {
        for admission in admissions {
            match admission {
                ContinuousBatchAdmission::Assigned { handle } => {
                    let req = *handle_to_req
                        .get(&handle.id)
                        .with_context(|| format!("admission for unknown handle {}", handle.id))?;
                    if admitted_at[req].is_none() {
                        admitted_at[req] = Some(step);
                        if step > 0 {
                            *mid_flight_admissions += 1;
                        }
                    }
                }
                ContinuousBatchAdmission::Rejected { handle, error } => {
                    let req = handle_to_req.get(&handle.id).copied();
                    bail!("manager rejected admission for req {req:?}: {error:#}");
                }
            }
        }
        Ok(())
    };

    // Drain anything queued at submit time (a prompt at the context limit is
    // admitted + finished during submit). These count as step-0 admissions.
    record_admissions(
        manager.poll_admissions(),
        0,
        &mut admitted_at,
        &mut mid_flight_admissions,
    )?;
    for event in manager.poll() {
        if let ContinuousBatchEvent::Finished { handle, result } = event {
            let req = *handle_to_req
                .get(&handle.id)
                .with_context(|| format!("finish for unknown handle {}", handle.id))?;
            completed[req] = Some(result.token_ids.clone());
            finished_at[req] = Some(0);
        }
    }

    let max_steps = requests * (gen_tokens + 64) + 1024;
    let mut step = 0usize;
    while manager.has_pending_work() {
        if step >= max_steps {
            bail!(
                "manager mid-flight gate exceeded {max_steps} steps without draining — likely a \
                 stuck row"
            );
        }
        manager
            .step()
            .with_context(|| format!("manager.step() at step {step}"))?;
        record_admissions(
            manager.poll_admissions(),
            step,
            &mut admitted_at,
            &mut mid_flight_admissions,
        )?;
        for event in manager.poll() {
            if let ContinuousBatchEvent::Finished { handle, result } = event {
                let req = *handle_to_req
                    .get(&handle.id)
                    .with_context(|| format!("finish for unknown handle {}", handle.id))?;
                completed[req] = Some(result.token_ids.clone());
                finished_at[req] = Some(step);
            }
        }
        step += 1;
    }

    // Report the manager's honest logits D2H cost (the native host-logits seam
    // reads `[B, 1, vocab]` each step). `BatchStepLogits::HostRows` is *moved*
    // into the manager's rows, so the manager adds no copy on top of this read.
    if let Some(d2h) = manager.logits_d2h_stats() {
        let d2h_ms = d2h.time.as_secs_f64() * 1000.0;
        let per_step_kb = if d2h.steps > 0 {
            (d2h.bytes as f64 / d2h.steps as f64) / 1024.0
        } else {
            0.0
        };
        println!(
            "native_decode_mid_flight_manager_solo_equivalence_d2h: steps={} \
             total_logits_d2h_bytes={} total_d2h_ms={d2h_ms:.3} per_step_logits_kb={per_step_kb:.1} \
             (manager moves each host row into its slot; no copy on top of this read)",
            d2h.steps, d2h.bytes
        );
    } else {
        println!(
            "native_decode_mid_flight_manager_solo_equivalence_d2h: backend reported no logits \
             D2H (unexpected for the native host-logits seam)"
        );
    }

    // Capture must survive mid-flight admission: assign/deactivate are host-side
    // mask/length writes, so no recapture is forced. Read the stats after the
    // manager is dropped to release its &mut borrow of the session.
    drop(manager);
    let capture = batch_engine
        .native_decode_session_mut()
        .and_then(|session| session.cuda_kv_debug_stats())
        .map(|stats| {
            (
                stats.graph.enabled,
                stats.graph.captures,
                stats.graph.replays,
                stats.graph.fallbacks,
                stats.graph.invalidations,
            )
        });
    if let Some((enabled, captures, replays, fallbacks, invalidations)) = capture {
        println!(
            "native_decode_mid_flight_manager_solo_equivalence_capture: graph_enabled={enabled} \
             captures={captures} replays={replays} fallbacks={fallbacks} \
             invalidations={invalidations} (assign/deactivate are host-side writes through the \
             manager; any invalidations are KV-growth boundaries, not admissions)"
        );
    }
    drop(batch_engine);
    // SAFETY: single-threaded teardown.
    unsafe {
        std::env::remove_var("ONNX_GENAI_NATIVE_DECODE_BATCH");
    }

    // Visibility: prove rows were admitted mid-flight (step > 0), not all at step 0.
    for req in 0..requests {
        println!(
            "native_decode_mid_flight_manager_solo_equivalence_admission: req={req} prompt_len={} \
             gen_target={} admitted_at_step={:?} finished_at_step={:?}",
            lens[req], gen_targets[req], admitted_at[req], finished_at[req]
        );
    }
    if mid_flight_admissions == 0 {
        bail!(
            "manager mid-flight gate is worthless: every request was admitted at step 0 (no row \
             was backfilled into a freed slot while peers kept decoding). Increase the prompt \
             count or reduce --mid-flight-batch."
        );
    }

    // Assert every request's manager-produced stream is byte-identical to solo.
    let mut all_match = true;
    for req in 0..requests {
        let solo = &solo_streams[req];
        let batched = completed[req]
            .as_ref()
            .with_context(|| format!("manager mid-flight request {req} never finished"))?;
        let matches = solo == batched;
        all_match &= matches;
        println!(
            "native_decode_mid_flight_manager_solo_equivalence_row: req={req} prompt_len={} \
             admitted_at_step={:?} matches_solo={matches} batch_stream={batched:?}",
            lens[req], admitted_at[req]
        );
        if !matches {
            println!(
                "native_decode_mid_flight_manager_solo_equivalence_detail: req={req} solo={solo:?} \
                 batch={batched:?}"
            );
        }
    }
    println!(
        "native_decode_mid_flight_manager_solo_equivalence_result: requests={requests} \
         batch={batch} mid_flight_admissions={mid_flight_admissions} all_rows_match_solo={all_match}"
    );
    if !all_match {
        bail!(
            "manager mid-flight solo-equivalence FAILED: a row admitted into a recycled slot by \
             the manager diverged from its solo batch-1 stream — the manager wiring leaked stale \
             KV, mask, or position across the admission boundary"
        );
    }
    Ok(())
}

fn run_steady(args: &Args, model_dir: &Path, device: NativeDecodeDevice) -> Result<()> {
    if args.synthetic {
        bail!("--steady requires a real model directory");
    }
    if args.decode_skip == 0 {
        bail!(
            "--decode-skip must be at least 1: the steady window is timed from the token \
             immediately before the first measured token"
        );
    }
    if args.tokens <= args.decode_skip {
        bail!("--tokens must be greater than --decode-skip");
    }
    print_backend_label(args.backend);
    println!("profile_native: {}", describe_sampling(args));
    println!("profile_native: {}", describe_speculative(args));
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
    if args.speculative != SpeculativeArg::None {
        print_speculative_observability(&engine.last_speculative_stats());
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
    if args.speculative != SpeculativeArg::None {
        bail!(
            "--speculative is not supported on the --pipeline path: PipelineEngine drives its \
             autoregressive decode through the strict one-token-per-step run_decode_loop, which \
             has no k-token verify/rewind hook. Native prompt-lookup speculation is wired only \
             into the single-model Engine path (use --steady without --pipeline). Requesting \
             speculation here would be silently ignored and report misleading (greedy) numbers."
        );
    }
    if args.steady && args.decode_skip == 0 {
        bail!(
            "--decode-skip must be at least 1: the steady window is timed from the token \
             immediately before the first measured token"
        );
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
                .generate_with_callback(pipeline_request(args, args.tokens), Some(&mut callback))
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
    if args.ongpu_argmax_bench {
        return run_ongpu_argmax_bench(
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
        NativeDecodeSession::load_with_resolved_io(&model, device.clone())
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
    if let Some(batch_sizes) = args.native_decode_batch_sweep.clone() {
        drop(session);
        let model_dir = args
            .model
            .as_deref()
            .expect("validated model argument")
            .to_path_buf();
        return run_native_decode_batch_sweep(
            &model_dir,
            device.clone(),
            args.decode_precision.into(),
            &prompt_tokens,
            &batch_sizes,
            args.tokens,
            args.native_decode_batch_context,
        );
    }
    if let Some(prompt_spec) = args.solo_equivalence_prompts.clone() {
        drop(session);
        let model_dir = args
            .model
            .as_deref()
            .expect("validated model argument")
            .to_path_buf();
        return run_native_decode_solo_equivalence(
            &model_dir,
            device.clone(),
            args.decode_precision.into(),
            &tokenizer,
            &prompt_spec,
            args.tokens,
        );
    }
    if let Some(prompt_spec) = args.ragged_solo_equivalence_prompts.clone() {
        drop(session);
        let model_dir = args
            .model
            .as_deref()
            .expect("validated model argument")
            .to_path_buf();
        return run_native_decode_ragged_solo_equivalence(
            &model_dir,
            device.clone(),
            args.decode_precision.into(),
            &tokenizer,
            &prompt_spec,
            args.tokens,
        );
    }
    if let Some(prompt_spec) = args.mid_flight_solo_equivalence_prompts.clone() {
        drop(session);
        let model_dir = args
            .model
            .as_deref()
            .expect("validated model argument")
            .to_path_buf();
        if args.mid_flight_via_manager {
            return run_native_decode_mid_flight_manager_solo_equivalence(
                &model_dir,
                device.clone(),
                args.decode_precision.into(),
                &tokenizer,
                &prompt_spec,
                args.mid_flight_batch,
                args.tokens,
            );
        }
        return run_native_decode_mid_flight_solo_equivalence(
            &model_dir,
            device.clone(),
            args.decode_precision.into(),
            &tokenizer,
            &prompt_spec,
            args.mid_flight_batch,
            args.tokens,
        );
    }
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
