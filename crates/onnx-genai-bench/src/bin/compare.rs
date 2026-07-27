use std::{
    cmp::Ordering,
    env, fmt, fs,
    path::{Path, PathBuf},
    process::Command as StdCommand,
    sync::Arc,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, ValueEnum};
use serde_json::{Value, json};
use tokenizers::Tokenizer;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    process::Command as TokioCommand,
};

#[cfg(feature = "bench-native")]
use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GenerateRequest, NativeDecodeDevice,
};
#[cfg(feature = "bench-native")]
use onnx_genai_ort::{ChatMessage, ChatTemplate, available_execution_providers};

const SYSTEM_PROMPT: &str = "You are a concise benchmark assistant.";
const DEFAULT_PROMPT: &str = "Write a short Rust function that reverses a string.";

#[derive(Debug, Parser)]
#[command(
    name = "onnx-genai-compare",
    about = "Compare OpenAI-compatible local inference runtimes"
)]
struct Args {
    /// Measured runs per runtime and prompt after warmup.
    #[arg(long, default_value_t = 5)]
    runs: usize,

    /// Discarded warmup runs per runtime and prompt.
    #[arg(long, default_value_t = 1)]
    warmups: usize,

    /// Maximum generated tokens per request.
    #[arg(long, visible_alias = "tokens", default_value_t = 50)]
    max_tokens: usize,

    /// Generated tokens to discard from the decode-throughput window in direct mode.
    #[arg(long, default_value_t = 2)]
    decode_skip: usize,

    /// Model directory for direct native CPU EP versus ORT CPU EP comparison.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Prompt used by the direct native-vs-ORT comparison.
    #[arg(long, default_value = DEFAULT_PROMPT)]
    prompt: String,

    /// Do not apply the model chat template in direct comparison mode.
    #[arg(long)]
    raw: bool,

    /// Direct backend to run. Repeat to override the default native+ORT pair.
    #[arg(long = "direct-backend", value_enum)]
    direct_backends: Vec<DirectBackend>,

    /// Common Qwen tokenizer used to count prompt and generated tokens.
    #[arg(long, default_value = "models/qwen2.5-0.5b/tokenizer.json")]
    tokenizer: PathBuf,

    /// Runtime as NAME|BASE_URL|MODEL|FORMAT|SETTINGS. Repeat to override defaults.
    #[arg(long = "runtime", value_parser = parse_runtime)]
    runtimes: Vec<RuntimeSpec>,

    /// Also write the Markdown report to this path.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Also write the machine-readable direct comparison report as JSON. Use '-' for stdout.
    #[arg(long, value_name = "PATH")]
    profile_json: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum DirectBackend {
    Native,
    Ort,
}

impl DirectBackend {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Ort => "ort",
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Native => "native CPU EP",
            Self::Ort => "ORT CPU EP",
        }
    }
}

#[derive(Clone, Debug)]
struct RuntimeSpec {
    name: String,
    base_url: String,
    model: String,
    format: String,
    settings: String,
}

impl RuntimeSpec {
    fn defaults() -> Vec<Self> {
        vec![
            Self {
                name: "onnx-genai".into(),
                base_url: "http://127.0.0.1:8080/v1".into(),
                model: "qwen2.5-0.5b".into(),
                format: "ONNX f32".into(),
                settings: "CPU EP; ORT default threads".into(),
            },
            Self {
                name: "Ollama (llama.cpp)".into(),
                base_url: "http://127.0.0.1:11434/v1".into(),
                model: "qwen2.5:0.5b".into(),
                format: "GGUF (record exact quant in command)".into(),
                settings: "runtime defaults".into(),
            },
            Self {
                name: "LM Studio".into(),
                base_url: "http://127.0.0.1:1234/v1".into(),
                model: "qwen2.5-0.5b-instruct".into(),
                format: "GGUF (record exact quant in command)".into(),
                settings: "runtime defaults".into(),
            },
        ]
    }

    fn models_url(&self) -> String {
        format!("{}/models", self.base_url.trim_end_matches('/'))
    }

    fn completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }
}

fn parse_runtime(value: &str) -> std::result::Result<RuntimeSpec, String> {
    let fields = value.split('|').collect::<Vec<_>>();
    if fields.len() != 5 || fields.iter().any(|field| field.trim().is_empty()) {
        return Err("expected NAME|BASE_URL|MODEL|FORMAT|SETTINGS".into());
    }
    Ok(RuntimeSpec {
        name: fields[0].trim().into(),
        base_url: fields[1].trim().trim_end_matches('/').into(),
        model: fields[2].trim().into(),
        format: fields[3].trim().into(),
        settings: fields[4].trim().into(),
    })
}

#[derive(Clone, Debug)]
struct PromptCase {
    name: &'static str,
    text: String,
    prompt_tokens: usize,
}

impl PromptCase {
    fn all(tokenizer: &Tokenizer) -> Result<Vec<Self>> {
        let short = "Explain why reproducible inference benchmarks matter. Cover hardware, \
            software versions, model format, prompts, warmup, statistics, and power state in \
            detail; do not conclude before covering every item."
            .to_string();
        let paragraph = "A reproducible inference benchmark fixes the model, tokenizer, prompt, \
            generation settings, runtime version, hardware, power mode, and warmup procedure. \
            It distinguishes prompt processing from autoregressive decoding, reports distributions \
            rather than a single best run, and records enough metadata for another engineer to \
            repeat the measurement. ";
        let mut long = String::from(
            "Read the following benchmark protocol. After the protocol, summarize its three most \
             important controls in exactly three concise bullet points.\n\n",
        );
        for index in 1..=12 {
            long.push_str(&format!("Protocol section {index}: {paragraph}\n"));
        }
        long.push_str("\nNow provide the requested three bullet points.");

        [("short", short), ("long", long)]
            .into_iter()
            .map(|(name, text)| {
                let rendered = render_qwen_prompt(&text);
                let prompt_tokens = tokenizer
                    .encode(rendered, false)
                    .map_err(|err| anyhow!("failed to tokenize {name} prompt: {err}"))?
                    .len();
                Ok(Self {
                    name,
                    text,
                    prompt_tokens,
                })
            })
            .collect()
    }
}

fn render_qwen_prompt(user_prompt: &str) -> String {
    format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n\
         <|im_start|>user\n{user_prompt}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

#[derive(Clone, Debug)]
struct Sample {
    ttft: Duration,
    total: Duration,
    output_tokens: usize,
    decode_tokens_per_second: f64,
    estimated_prefill_tokens_per_second: f64,
}

#[derive(Clone, Copy, Debug)]
struct Distribution {
    min: f64,
    p10: f64,
    median: f64,
    p90: f64,
    p95: f64,
    max: f64,
}

impl Distribution {
    fn from_values(mut values: Vec<f64>) -> Self {
        values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
        Self {
            min: values.first().copied().unwrap_or(f64::NAN),
            p10: percentile(&values, 0.1),
            median: percentile(&values, 0.5),
            p90: percentile(&values, 0.9),
            p95: percentile(&values, 0.95),
            max: values.last().copied().unwrap_or(f64::NAN),
        }
    }
}

#[derive(Clone, Debug)]
struct Summary {
    runtime_index: usize,
    prompt_index: usize,
    ttft_ms: Distribution,
    decode_tokens_per_second: Distribution,
    total_ms: Distribution,
    estimated_prefill_tokens_per_second: Distribution,
    output_tokens: Distribution,
}

impl Summary {
    fn from_samples(runtime_index: usize, prompt_index: usize, samples: &[Sample]) -> Self {
        Self {
            runtime_index,
            prompt_index,
            ttft_ms: Distribution::from_values(
                samples
                    .iter()
                    .map(|sample| sample.ttft.as_secs_f64() * 1_000.0)
                    .collect(),
            ),
            decode_tokens_per_second: Distribution::from_values(
                samples
                    .iter()
                    .map(|sample| sample.decode_tokens_per_second)
                    .collect(),
            ),
            total_ms: Distribution::from_values(
                samples
                    .iter()
                    .map(|sample| sample.total.as_secs_f64() * 1_000.0)
                    .collect(),
            ),
            estimated_prefill_tokens_per_second: Distribution::from_values(
                samples
                    .iter()
                    .map(|sample| sample.estimated_prefill_tokens_per_second)
                    .collect(),
            ),
            output_tokens: Distribution::from_values(
                samples
                    .iter()
                    .map(|sample| sample.output_tokens as f64)
                    .collect(),
            ),
        }
    }
}

#[derive(Debug)]
enum RuntimeState {
    Available { advertised_models: Vec<String> },
    Skipped { reason: String },
}

#[derive(Clone, Debug)]
struct DirectSample {
    backend: DirectBackend,
    run: usize,
    model_load: Duration,
    ttft: Duration,
    total: Duration,
    generated_tokens: usize,
    decode_tokens_per_second: f64,
    end_to_end_tokens_per_second: f64,
}

#[derive(Clone, Debug)]
struct DirectSummary {
    backend: DirectBackend,
    model_load_ms: Distribution,
    ttft_ms: Distribution,
    decode_tokens_per_second: Distribution,
    end_to_end_tokens_per_second: Distribution,
    total_ms: Distribution,
    generated_tokens: Distribution,
}

#[derive(Clone, Debug)]
struct DirectRoofline {
    measured_bandwidth_gbps: f64,
    model_weight_bytes: u64,
    ceiling_tokens_per_second: f64,
    threads: usize,
    bytes_per_thread: usize,
    repetitions: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if args.runs == 0 {
        bail!("--runs must be at least 1");
    }
    if args.max_tokens == 0 {
        bail!("--max-tokens must be at least 1");
    }
    if let Some(model) = args.model.as_ref() {
        return run_direct_compare(&args, model);
    }

    let tokenizer = Tokenizer::from_file(&args.tokenizer)
        .map_err(|err| anyhow!("failed to load {}: {err}", args.tokenizer.display()))?;
    let prompts = PromptCase::all(&tokenizer)?;
    let runtimes = if args.runtimes.is_empty() {
        RuntimeSpec::defaults()
    } else {
        args.runtimes.clone()
    };
    let mut states = Vec::with_capacity(runtimes.len());
    for runtime in &runtimes {
        states.push(probe_runtime(runtime).await);
    }

    let mut summaries = Vec::new();
    let mut run_notes = Vec::new();
    for (runtime_index, runtime) in runtimes.iter().enumerate() {
        if let RuntimeState::Skipped { reason } = &states[runtime_index] {
            run_notes.push(format!("{} skipped: {reason}", runtime.name));
            continue;
        }
        for (prompt_index, prompt) in prompts.iter().enumerate() {
            eprintln!(
                "benchmarking {} / {} ({} warmup, {} measured)",
                runtime.name, prompt.name, args.warmups, args.runs
            );
            let mut failed = None;
            for warmup in 0..args.warmups {
                if let Err(err) = run_once(runtime, prompt, args.max_tokens).await {
                    failed = Some(format!("warmup {} failed: {err:#}", warmup + 1));
                    break;
                }
            }
            if let Some(reason) = failed {
                run_notes.push(format!(
                    "{} / {} skipped after probe: {reason}",
                    runtime.name, prompt.name
                ));
                continue;
            }

            let mut samples = Vec::with_capacity(args.runs);
            for run in 0..args.runs {
                match run_once(runtime, prompt, args.max_tokens).await {
                    Ok(sample) => samples.push(sample),
                    Err(err) => {
                        failed = Some(format!("measured run {} failed: {err:#}", run + 1));
                        break;
                    }
                }
            }
            if let Some(reason) = failed {
                run_notes.push(format!(
                    "{} / {} omitted: {reason}",
                    runtime.name, prompt.name
                ));
            } else {
                summaries.push(Summary::from_samples(runtime_index, prompt_index, &samples));
            }
        }
    }

    let report = render_report(&args, &runtimes, &states, &prompts, &summaries, &run_notes);
    print!("{report}");
    if let Some(path) = &args.profile_json {
        write_json_arg(path, &json!({ "report_markdown": report }))?;
    }
    if let Some(output) = args.output {
        if let Some(parent) = output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        fs::write(&output, &report)
            .with_context(|| format!("failed to write {}", output.display()))?;
        eprintln!("wrote {}", output.display());
    }
    Ok(())
}

#[cfg(not(feature = "bench-native"))]
fn run_direct_compare(_args: &Args, _model: &Path) -> Result<()> {
    bail!(
        "direct native-vs-ORT comparison requires `cargo run -p onnx-genai-bench \
         --features bench-native --bin compare -- --model {}`",
        "models/qwen2.5-0.5b"
    )
}

#[cfg(feature = "bench-native")]
fn run_direct_compare(args: &Args, model: &Path) -> Result<()> {
    if args.max_tokens < 2 {
        bail!("direct comparison needs at least two generated tokens for decode throughput");
    }
    if args.decode_skip >= args.max_tokens {
        bail!("--decode-skip must be smaller than --tokens/--max-tokens");
    }
    let model_dir = if model.is_file() {
        model
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .context("--model file has no parent directory")?
    } else {
        model
    };
    if !model_dir.is_dir() {
        bail!("model directory does not exist: {}", model_dir.display());
    }
    let backends = if args.direct_backends.is_empty() {
        vec![DirectBackend::Native, DirectBackend::Ort]
    } else {
        args.direct_backends.clone()
    };
    if backends.is_empty() {
        bail!("at least one direct backend is required");
    }
    let (prompt_mode, rendered_prompt) = resolve_direct_prompt(model_dir, &args.prompt, args.raw)?;
    let prompt_tokens = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|err| {
            anyhow!(
                "failed to load tokenizer beside {}: {err}",
                model_dir.display()
            )
        })?
        .encode(rendered_prompt.clone(), false)
        .map_err(|err| anyhow!("failed to tokenize direct comparison prompt: {err}"))?
        .len();

    let roofline = measure_direct_roofline(model_dir)?;

    for warmup in 1..=args.warmups {
        for &backend in &backends {
            eprintln!(
                "direct warmup {warmup}/{}: {}",
                args.warmups,
                backend.label()
            );
            std::hint::black_box(run_direct_once(
                args,
                model_dir,
                &rendered_prompt,
                backend,
                0,
            )?);
        }
    }

    let mut samples = Vec::with_capacity(args.runs * backends.len());
    for run in 1..=args.runs {
        for &backend in &backends {
            eprintln!(
                "direct measured run {run}/{}: {}",
                args.runs,
                backend.label()
            );
            samples.push(run_direct_once(
                args,
                model_dir,
                &rendered_prompt,
                backend,
                run,
            )?);
        }
    }
    let summaries = summarize_direct_samples(&backends, &samples)?;
    let report = render_direct_report(
        args,
        model_dir,
        &prompt_mode,
        prompt_tokens,
        &summaries,
        roofline.as_ref(),
    );
    let json_report = direct_json_report(
        args,
        model_dir,
        &prompt_mode,
        prompt_tokens,
        &samples,
        &summaries,
        roofline.as_ref(),
    );

    if args
        .profile_json
        .as_ref()
        .is_some_and(|path| path.as_os_str() == "-")
    {
        eprint!("{report}");
        write_json_arg(Path::new("-"), &json_report)?;
    } else {
        print!("{report}");
        if let Some(path) = &args.profile_json {
            write_json_arg(path, &json_report)?;
        }
    }
    Ok(())
}

#[cfg(feature = "bench-native")]
fn resolve_direct_prompt(model_dir: &Path, prompt: &str, raw: bool) -> Result<(String, String)> {
    if raw {
        return Ok(("raw (--raw)".to_string(), prompt.to_string()));
    }
    let standalone = model_dir.join("chat_template.jinja");
    let tokenizer_config = model_dir.join("tokenizer_config.json");
    let has_template = standalone.is_file()
        || (tokenizer_config.is_file()
            && std::fs::read_to_string(&tokenizer_config)
                .ok()
                .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
                .and_then(|value| value.get("chat_template").cloned())
                .and_then(|value| value.as_str().map(ToString::to_string))
                .is_some());
    if !has_template {
        return Ok((
            "raw (no chat template in model dir)".to_string(),
            prompt.to_string(),
        ));
    }
    let template = ChatTemplate::from_model_dir(model_dir)
        .with_context(|| format!("load chat template from {}", model_dir.display()))?;
    let rendered = template
        .render(&[ChatMessage::new("user", prompt.to_string())], None, true)
        .with_context(|| format!("render chat template from {}", model_dir.display()))?;
    Ok(("chat template".to_string(), rendered))
}

#[cfg(feature = "bench-native")]
fn run_direct_once(
    args: &Args,
    model_dir: &Path,
    prompt: &str,
    backend: DirectBackend,
    run: usize,
) -> Result<DirectSample> {
    configure_direct_backend(backend)?;
    let mut config = EngineConfig {
        decode_backend: match backend {
            DirectBackend::Native => EngineDecodeBackend::Native,
            DirectBackend::Ort => EngineDecodeBackend::Ort,
        },
        ..EngineConfig::default()
    };
    config.native_device = Some(NativeDecodeDevice::Cpu);

    let load_started = Instant::now();
    let mut engine = Engine::from_dir(model_dir, config)
        .with_context(|| format!("load {} from {}", backend.label(), model_dir.display()))?;
    let model_load = load_started.elapsed();
    if backend == DirectBackend::Ort && engine.decode_backend() != EngineDecodeBackend::Ort {
        bail!(
            "requested ORT CPU backend but engine resolved {:?}",
            engine.decode_backend()
        );
    }
    if backend == DirectBackend::Native && engine.decode_backend() != EngineDecodeBackend::Native {
        bail!(
            "requested native CPU backend but engine resolved {:?}",
            engine.decode_backend()
        );
    }

    let mut request = GenerateRequest::new(prompt.to_string());
    request.options.max_new_tokens = args.max_tokens;
    request.options.temperature = 0.0;
    request.options.top_p = 1.0;
    request.options.greedy = true;
    request.options.seed = Some(0);
    request.options.stop_on_eos = false;

    let generate_started = Instant::now();
    let mut token_times = Vec::with_capacity(args.max_tokens);
    let mut callback = |_| {
        token_times.push(generate_started.elapsed());
        Ok(())
    };
    let result = engine
        .generate_with_callback(request, Some(&mut callback))
        .with_context(|| format!("generate with {}", backend.label()))?;
    let total = generate_started.elapsed();
    let generated_tokens = result.token_ids.len();
    if generated_tokens == 0 || token_times.is_empty() {
        bail!("{} generated no tokens", backend.label());
    }
    let ttft = token_times[0];
    if token_times.len() <= args.decode_skip {
        bail!(
            "{} emitted {} timed tokens, not enough for --decode-skip {}",
            backend.label(),
            token_times.len(),
            args.decode_skip
        );
    }
    let decode_tokens = generated_tokens.saturating_sub(args.decode_skip);
    let decode_window = token_times[token_times.len() - 1]
        .saturating_sub(token_times[args.decode_skip.saturating_sub(1)]);
    if decode_tokens == 0 || decode_window.is_zero() {
        bail!(
            "{} emitted too few timed tokens for decode throughput: tokens={} decode_skip={} window={:?}",
            backend.label(),
            generated_tokens,
            args.decode_skip,
            decode_window
        );
    }
    Ok(DirectSample {
        backend,
        run,
        model_load,
        ttft,
        total,
        generated_tokens,
        decode_tokens_per_second: decode_tokens as f64 / decode_window.as_secs_f64(),
        end_to_end_tokens_per_second: generated_tokens as f64 / total.as_secs_f64(),
    })
}

#[cfg(feature = "bench-native")]
fn configure_direct_backend(backend: DirectBackend) -> Result<()> {
    unsafe {
        std::env::set_var("ONNX_GENAI_EP", "cpu");
    }
    if backend == DirectBackend::Ort {
        let available =
            available_execution_providers().context("query linked ONNX Runtime providers")?;
        if !available
            .iter()
            .any(|provider| provider.eq_ignore_ascii_case("CPUExecutionProvider"))
        {
            bail!(
                "ORT CPU backend requested, but CPUExecutionProvider is unavailable \
                 (available: {available:?})"
            );
        }
    }
    Ok(())
}

fn measure_direct_roofline(model_dir: &Path) -> Result<Option<DirectRoofline>> {
    if !(cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")) {
        return Ok(None);
    }
    let model_weight_bytes = model_weight_bytes(model_dir)?;
    if model_weight_bytes == 0 {
        bail!(
            "cannot compute roofline fraction: no model weight bytes found under {}",
            model_dir.display()
        );
    }
    let available_threads = thread::available_parallelism().map_or(1, usize::from);
    let threads = sysctl_usize("hw.perflevel0.physicalcpu")
        .unwrap_or(available_threads)
        .clamp(1, available_threads);
    let bytes_per_thread = 256 * 1024 * 1024;
    let repetitions = 3;
    let measured_bandwidth_gbps =
        measure_sequential_read_bandwidth_gbps(threads, bytes_per_thread, repetitions);
    if !measured_bandwidth_gbps.is_finite() || measured_bandwidth_gbps <= 0.0 {
        bail!("measured invalid Apple-Silicon memory bandwidth: {measured_bandwidth_gbps}");
    }
    Ok(Some(DirectRoofline {
        measured_bandwidth_gbps,
        model_weight_bytes,
        ceiling_tokens_per_second: measured_bandwidth_gbps * 1_000_000_000.0
            / model_weight_bytes as f64,
        threads,
        bytes_per_thread,
        repetitions,
    }))
}

fn model_weight_bytes(model_dir: &Path) -> Result<u64> {
    let mut bytes = 0_u64;
    for entry in fs::read_dir(model_dir).with_context(|| format!("read {}", model_dir.display()))? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name == "model.onnx" || name.starts_with("model.onnx.data") {
            bytes += entry.metadata()?.len();
        }
    }
    Ok(bytes)
}

fn sysctl_usize(name: &str) -> Option<usize> {
    command_output("sysctl", &["-n", name])?.trim().parse().ok()
}

fn measure_sequential_read_bandwidth_gbps(
    threads: usize,
    bytes_per_thread: usize,
    repetitions: usize,
) -> f64 {
    let len = bytes_per_thread / std::mem::size_of::<u64>();
    let buffers = Arc::new(
        (0..threads)
            .map(|thread_index| {
                (0..len)
                    .map(|index| (index as u64).wrapping_mul(0x9E37_79B9) ^ thread_index as u64)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
    );
    let mut best = 0.0_f64;
    for _ in 0..repetitions {
        let started = Instant::now();
        thread::scope(|scope| {
            let mut handles = Vec::with_capacity(threads);
            for buffer in buffers.iter() {
                handles.push(scope.spawn(move || {
                    let mut sum = 0_u64;
                    for value in buffer {
                        sum = sum.wrapping_add(std::hint::black_box(*value));
                    }
                    std::hint::black_box(sum);
                }));
            }
            for handle in handles {
                handle.join().expect("bandwidth worker panicked");
            }
        });
        let elapsed = started.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            best = best.max((threads * bytes_per_thread) as f64 / elapsed / 1_000_000_000.0);
        }
    }
    best
}

fn summarize_direct_samples(
    backends: &[DirectBackend],
    samples: &[DirectSample],
) -> Result<Vec<DirectSummary>> {
    backends
        .iter()
        .copied()
        .map(|backend| {
            let backend_samples = samples
                .iter()
                .filter(|sample| sample.backend == backend)
                .collect::<Vec<_>>();
            if backend_samples.is_empty() {
                bail!("no samples recorded for {}", backend.label());
            }
            Ok(DirectSummary {
                backend,
                model_load_ms: Distribution::from_values(
                    backend_samples
                        .iter()
                        .map(|sample| millis(sample.model_load))
                        .collect(),
                ),
                ttft_ms: Distribution::from_values(
                    backend_samples
                        .iter()
                        .map(|sample| millis(sample.ttft))
                        .collect(),
                ),
                decode_tokens_per_second: Distribution::from_values(
                    backend_samples
                        .iter()
                        .map(|sample| sample.decode_tokens_per_second)
                        .collect(),
                ),
                end_to_end_tokens_per_second: Distribution::from_values(
                    backend_samples
                        .iter()
                        .map(|sample| sample.end_to_end_tokens_per_second)
                        .collect(),
                ),
                total_ms: Distribution::from_values(
                    backend_samples
                        .iter()
                        .map(|sample| millis(sample.total))
                        .collect(),
                ),
                generated_tokens: Distribution::from_values(
                    backend_samples
                        .iter()
                        .map(|sample| sample.generated_tokens as f64)
                        .collect(),
                ),
            })
        })
        .collect()
}

fn render_direct_report(
    args: &Args,
    model_dir: &Path,
    prompt_mode: &str,
    prompt_tokens: usize,
    summaries: &[DirectSummary],
    roofline: Option<&DirectRoofline>,
) -> String {
    let mut report = String::new();
    report.push_str("# Native CPU EP vs ORT CPU EP comparison\n\n");
    report.push_str("| field | value |\n|---|---|\n");
    metadata_row(&mut report, "model", model_dir.display().to_string());
    metadata_row(&mut report, "prompt", args.prompt.clone());
    metadata_row(&mut report, "prompt mode", prompt_mode.to_string());
    metadata_row(&mut report, "prompt tokens", prompt_tokens.to_string());
    metadata_row(&mut report, "generated tokens", args.max_tokens.to_string());
    metadata_row(
        &mut report,
        "methodology",
        format!(
            "{} warmup pair(s), {} measured pair(s), discard first {} generated token(s) \
             from decode throughput, median with p10-p95 spread",
            args.warmups, args.runs, args.decode_skip
        ),
    );
    metadata_row(
        &mut report,
        "machine",
        command_output("uname", &["-srmp"]).unwrap_or_else(|| env::consts::OS.into()),
    );
    metadata_row(
        &mut report,
        "CPU",
        command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
            .unwrap_or_else(|| "unknown".into()),
    );
    metadata_row(
        &mut report,
        "git commit",
        command_output("git", &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into()),
    );
    if let Some(roofline) = roofline {
        metadata_row(
            &mut report,
            "measured CPU bandwidth",
            format!(
                "{:.1} GB/s ({} thread(s), {} MiB/thread, best of {})",
                roofline.measured_bandwidth_gbps,
                roofline.threads,
                roofline.bytes_per_thread / (1024 * 1024),
                roofline.repetitions
            ),
        );
        metadata_row(
            &mut report,
            "model weight bytes",
            roofline.model_weight_bytes.to_string(),
        );
        metadata_row(
            &mut report,
            "decode roofline ceiling",
            format!("{:.2} tok/s", roofline.ceiling_tokens_per_second),
        );
    } else {
        metadata_row(
            &mut report,
            "decode roofline ceiling",
            "unavailable on non-Apple-Silicon host".to_string(),
        );
    }
    report.push('\n');
    report.push_str("| backend | model load ms | TTFT ms | decode tok/s | decode roofline % | end-to-end tok/s | total ms | output tokens |\n");
    report.push_str("|---|---:|---:|---:|---:|---:|---:|---:|\n");
    for summary in summaries {
        let roofline_cell = roofline
            .map(|roofline| {
                percent_spread_cell(
                    distribution_scale(
                        summary.decode_tokens_per_second,
                        roofline.model_weight_bytes as f64
                            / (roofline.measured_bandwidth_gbps * 1_000_000_000.0),
                    ),
                    2,
                )
            })
            .unwrap_or_else(|| "N/A".to_string());
        report.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
            summary.backend.label(),
            spread_cell(summary.model_load_ms, 1),
            spread_cell(summary.ttft_ms, 1),
            spread_cell(summary.decode_tokens_per_second, 2),
            roofline_cell,
            spread_cell(summary.end_to_end_tokens_per_second, 2),
            spread_cell(summary.total_ms, 1),
            spread_cell(summary.generated_tokens, 0),
        ));
    }
    if let Some(ratios) = direct_ratios(summaries) {
        report.push('\n');
        report.push_str("| native/ORT ratio | value |\n|---|---:|\n");
        report.push_str(&format!(
            "| decode tok/s | {:.3}x |\n",
            ratios.decode_tokens_per_second
        ));
        report.push_str(&format!(
            "| end-to-end tok/s | {:.3}x |\n",
            ratios.end_to_end_tokens_per_second
        ));
        report.push_str(&format!("| TTFT ms | {:.3}x |\n", ratios.ttft_ms));
        report.push_str(&format!(
            "| model load ms | {:.3}x |\n",
            ratios.model_load_ms
        ));
    }
    report
}

#[derive(Clone, Copy, Debug)]
struct DirectRatios {
    model_load_ms: f64,
    ttft_ms: f64,
    decode_tokens_per_second: f64,
    end_to_end_tokens_per_second: f64,
}

fn direct_ratios(summaries: &[DirectSummary]) -> Option<DirectRatios> {
    let native = summaries
        .iter()
        .find(|summary| summary.backend == DirectBackend::Native)?;
    let ort = summaries
        .iter()
        .find(|summary| summary.backend == DirectBackend::Ort)?;
    Some(DirectRatios {
        model_load_ms: native.model_load_ms.median / ort.model_load_ms.median,
        ttft_ms: native.ttft_ms.median / ort.ttft_ms.median,
        decode_tokens_per_second: native.decode_tokens_per_second.median
            / ort.decode_tokens_per_second.median,
        end_to_end_tokens_per_second: native.end_to_end_tokens_per_second.median
            / ort.end_to_end_tokens_per_second.median,
    })
}

fn direct_json_report(
    args: &Args,
    model_dir: &Path,
    prompt_mode: &str,
    prompt_tokens: usize,
    samples: &[DirectSample],
    summaries: &[DirectSummary],
    roofline: Option<&DirectRoofline>,
) -> Value {
    json!({
        "model": model_dir.display().to_string(),
        "prompt": args.prompt.clone(),
        "prompt_mode": prompt_mode,
        "prompt_tokens": prompt_tokens,
        "generated_tokens_requested": args.max_tokens,
        "decode_skip": args.decode_skip,
        "warmups": args.warmups,
        "runs": args.runs,
        "methodology": {
            "warmups": args.warmups,
            "measured_repetitions": args.runs,
            "statistic": "median with p10/p95 spread",
            "sebastian_methodology": "defaults follow Sebastian's 1 warmup, 5 runs, 50 tokens, decode-skip 2 protocol"
        },
        "roofline": roofline.map(direct_roofline_json),
        "summaries": summaries.iter().map(|summary| direct_summary_json(summary, roofline)).collect::<Vec<_>>(),
        "ratios": direct_ratios(summaries).map(|ratios| json!({
            "native_over_ort_model_load_ms": ratios.model_load_ms,
            "native_over_ort_ttft_ms": ratios.ttft_ms,
            "native_over_ort_decode_tokens_per_second": ratios.decode_tokens_per_second,
            "native_over_ort_end_to_end_tokens_per_second": ratios.end_to_end_tokens_per_second,
        })),
        "samples": samples.iter().map(direct_sample_json).collect::<Vec<_>>(),
    })
}

fn direct_roofline_json(roofline: &DirectRoofline) -> Value {
    json!({
        "measured_bandwidth_gbps": roofline.measured_bandwidth_gbps,
        "model_weight_bytes": roofline.model_weight_bytes,
        "ceiling_tokens_per_second": roofline.ceiling_tokens_per_second,
        "threads": roofline.threads,
        "bytes_per_thread": roofline.bytes_per_thread,
        "repetitions": roofline.repetitions,
    })
}

fn direct_summary_json(summary: &DirectSummary, roofline: Option<&DirectRoofline>) -> Value {
    json!({
        "backend": summary.backend.as_str(),
        "model_load_ms": distribution_json(summary.model_load_ms),
        "ttft_ms": distribution_json(summary.ttft_ms),
        "decode_tokens_per_second": distribution_json(summary.decode_tokens_per_second),
        "decode_roofline_fraction": roofline.map(|roofline| distribution_json(distribution_scale(
            summary.decode_tokens_per_second,
            roofline.model_weight_bytes as f64 / (roofline.measured_bandwidth_gbps * 1_000_000_000.0),
        ))),
        "end_to_end_tokens_per_second": distribution_json(summary.end_to_end_tokens_per_second),
        "total_ms": distribution_json(summary.total_ms),
        "generated_tokens": distribution_json(summary.generated_tokens),
    })
}

fn direct_sample_json(sample: &DirectSample) -> Value {
    json!({
        "backend": sample.backend.as_str(),
        "run": sample.run,
        "model_load_ms": millis(sample.model_load),
        "ttft_ms": millis(sample.ttft),
        "decode_tokens_per_second": sample.decode_tokens_per_second,
        "end_to_end_tokens_per_second": sample.end_to_end_tokens_per_second,
        "total_ms": millis(sample.total),
        "generated_tokens": sample.generated_tokens,
    })
}

fn distribution_json(distribution: Distribution) -> Value {
    json!({
        "min": distribution.min,
        "p10": distribution.p10,
        "median": distribution.median,
        "p90": distribution.p90,
        "p95": distribution.p95,
        "max": distribution.max,
    })
}

fn distribution_scale(distribution: Distribution, scale: f64) -> Distribution {
    Distribution {
        min: distribution.min * scale,
        p10: distribution.p10 * scale,
        median: distribution.median * scale,
        p90: distribution.p90 * scale,
        p95: distribution.p95 * scale,
        max: distribution.max * scale,
    }
}

async fn probe_runtime(runtime: &RuntimeSpec) -> RuntimeState {
    let output = match TokioCommand::new("curl")
        .args([
            "--silent",
            "--show-error",
            "--fail-with-body",
            "--max-time",
            "2",
            &runtime.models_url(),
        ])
        .output()
        .await
    {
        Ok(output) => output,
        Err(err) => {
            return RuntimeState::Skipped {
                reason: format!(
                    "failed to execute curl for {} ({err})",
                    runtime.models_url()
                ),
            };
        }
    };
    if !output.status.success() {
        return RuntimeState::Skipped {
            reason: format!(
                "{} is unavailable ({})",
                runtime.models_url(),
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        };
    }
    let payload = match serde_json::from_slice::<Value>(&output.stdout) {
        Ok(payload) => payload,
        Err(err) => {
            return RuntimeState::Skipped {
                reason: format!("model probe returned invalid JSON ({err})"),
            };
        }
    };
    let advertised_models = payload["data"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|entry| entry["id"].as_str().map(ToString::to_string))
        .collect::<Vec<_>>();
    if !advertised_models
        .iter()
        .any(|model| model == &runtime.model)
    {
        return RuntimeState::Skipped {
            reason: format!(
                "model `{}` is not loaded; advertised models: {}",
                runtime.model,
                if advertised_models.is_empty() {
                    "(none)".into()
                } else {
                    advertised_models.join(", ")
                }
            ),
        };
    }
    RuntimeState::Available { advertised_models }
}

async fn run_once(runtime: &RuntimeSpec, prompt: &PromptCase, max_tokens: usize) -> Result<Sample> {
    let request = json!({
        "model": runtime.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt.text}
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "stream": true,
        "stream_options": {"include_usage": true}
    });
    let started = Instant::now();
    let mut child = TokioCommand::new("curl")
        .args([
            "--silent",
            "--show-error",
            "--fail-with-body",
            "--no-buffer",
            "--http1.1",
            "--max-time",
            "600",
            "--header",
            "Content-Type: application/json",
            "--request",
            "POST",
            "--data-binary",
            "@-",
            &runtime.completions_url(),
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("failed to start curl for {}", runtime.name))?;
    let mut stdin = child.stdin.take().context("curl stdin was unavailable")?;
    stdin
        .write_all(request.to_string().as_bytes())
        .await
        .context("failed to send request body to curl")?;
    drop(stdin);
    let mut stdout = child.stdout.take().context("curl stdout was unavailable")?;
    let mut stderr = child.stderr.take().context("curl stderr was unavailable")?;
    let stderr_task = tokio::spawn(async move {
        let mut bytes = Vec::new();
        stderr.read_to_end(&mut bytes).await.map(|_| bytes)
    });

    let mut pending = Vec::new();
    let mut raw_tail = Vec::new();
    let mut content_events = 0_usize;
    let mut usage_completion_tokens = None;
    let mut ttft = None;
    let mut saw_done = false;
    let mut chunk = [0_u8; 8 * 1024];
    loop {
        let read = stdout.read(&mut chunk).await?;
        if read == 0 {
            break;
        }
        pending.extend_from_slice(&chunk[..read]);
        for data in take_sse_events(&mut pending)? {
            if data == "[DONE]" {
                saw_done = true;
                continue;
            }
            let payload: Value = serde_json::from_str(&data)
                .with_context(|| format!("invalid SSE JSON from {}: {data}", runtime.name))?;
            if let Some(error) = payload.get("error") {
                bail!("stream error: {error}");
            }
            if let Some(tokens) = payload
                .pointer("/usage/completion_tokens")
                .and_then(Value::as_u64)
            {
                usage_completion_tokens = Some(tokens as usize);
            }
            if let Some(content) = payload
                .pointer("/choices/0/delta/content")
                .and_then(Value::as_str)
                .or_else(|| payload.pointer("/choices/0/text").and_then(Value::as_str))
                && !content.is_empty()
            {
                ttft.get_or_insert_with(|| started.elapsed());
                content_events += 1;
            }
        }
    }
    raw_tail.extend_from_slice(&pending);
    if !pending.is_empty() {
        for data in take_final_sse_event(&mut pending)? {
            if data != "[DONE]" {
                let payload: Value = serde_json::from_str(&data)?;
                if let Some(content) = payload
                    .pointer("/choices/0/delta/content")
                    .and_then(Value::as_str)
                    .or_else(|| payload.pointer("/choices/0/text").and_then(Value::as_str))
                    && !content.is_empty()
                {
                    ttft.get_or_insert_with(|| started.elapsed());
                    content_events += 1;
                }
                if let Some(tokens) = payload
                    .pointer("/usage/completion_tokens")
                    .and_then(Value::as_u64)
                {
                    usage_completion_tokens = Some(tokens as usize);
                }
            }
        }
    }
    let status = child.wait().await?;
    let stderr = stderr_task
        .await
        .context("curl stderr task failed")??
        .into_iter()
        .collect::<Vec<_>>();
    if !status.success() {
        let stderr = String::from_utf8_lossy(&stderr);
        let tail = String::from_utf8_lossy(&raw_tail);
        bail!(
            "curl exited with {status}: {}{}",
            stderr.trim(),
            if tail.trim().is_empty() {
                String::new()
            } else {
                format!("; response: {}", tail.trim())
            }
        );
    }
    let total = started.elapsed();
    let ttft = ttft.ok_or_else(|| {
        anyhow!(
            "{} returned no generated content{}",
            runtime.name,
            if saw_done { " before [DONE]" } else { "" }
        )
    })?;
    let output_tokens = usage_completion_tokens.unwrap_or(content_events);
    if output_tokens == 0 {
        bail!("{} generated text but no countable tokens", runtime.name);
    }
    let decode_duration = total.saturating_sub(ttft);
    let decode_tokens = output_tokens.saturating_sub(1);
    let decode_tokens_per_second = if decode_tokens == 0 {
        output_tokens as f64 / total.as_secs_f64()
    } else {
        decode_tokens as f64 / decode_duration.as_secs_f64()
    };
    Ok(Sample {
        ttft,
        total,
        output_tokens,
        decode_tokens_per_second,
        estimated_prefill_tokens_per_second: prompt.prompt_tokens as f64 / ttft.as_secs_f64(),
    })
}

fn take_sse_events(pending: &mut Vec<u8>) -> Result<Vec<String>> {
    let mut events = Vec::new();
    while let Some((end, delimiter_len)) = find_event_end(pending) {
        let event = pending.drain(..end + delimiter_len).collect::<Vec<_>>();
        events.extend(parse_sse_event(&event[..end])?);
    }
    Ok(events)
}

fn take_final_sse_event(pending: &mut Vec<u8>) -> Result<Vec<String>> {
    let bytes = std::mem::take(pending);
    parse_sse_event(&bytes)
}

fn find_event_end(bytes: &[u8]) -> Option<(usize, usize)> {
    let lf = bytes.windows(2).position(|window| window == b"\n\n");
    let crlf = bytes.windows(4).position(|window| window == b"\r\n\r\n");
    match (lf, crlf) {
        (Some(left), Some(right)) if left <= right => Some((left, 2)),
        (Some(_), Some(right)) => Some((right, 4)),
        (Some(left), None) => Some((left, 2)),
        (None, Some(right)) => Some((right, 4)),
        (None, None) => None,
    }
}

fn parse_sse_event(bytes: &[u8]) -> Result<Vec<String>> {
    let text = std::str::from_utf8(bytes).context("SSE event was not valid UTF-8")?;
    let data = text
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(|line| line.strip_prefix(' ').unwrap_or(line))
        .collect::<Vec<_>>();
    if data.is_empty() {
        Ok(Vec::new())
    } else {
        Ok(vec![data.join("\n")])
    }
}

fn percentile(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    if values.len() == 1 {
        return values[0];
    }
    let rank = percentile * (values.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        values[lower]
    } else {
        values[lower] + (values[upper] - values[lower]) * (rank - lower as f64)
    }
}

fn render_report(
    args: &Args,
    runtimes: &[RuntimeSpec],
    states: &[RuntimeState],
    prompts: &[PromptCase],
    summaries: &[Summary],
    run_notes: &[String],
) -> String {
    let hostname = command_output("hostname", &["-s"]).unwrap_or_else(|| "unknown-host".into());
    let timestamp = unix_timestamp();
    let mut report = format!(
        "# Cross-runtime benchmark — {} ({hostname})\n\n",
        calendar_date()
    );
    report.push_str("## Machine and run metadata\n\n");
    report.push_str("| field | value |\n|---|---|\n");
    metadata_row(
        &mut report,
        "CPU",
        command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
            .unwrap_or_else(|| "unknown".into()),
    );
    metadata_row(
        &mut report,
        "cores",
        command_output("sysctl", &["-n", "hw.logicalcpu"])
            .or_else(|| command_output("getconf", &["_NPROCESSORS_ONLN"]))
            .unwrap_or_else(|| "unknown".into()),
    );
    metadata_row(
        &mut report,
        "OS",
        command_output("uname", &["-srmp"]).unwrap_or_else(|| env::consts::OS.into()),
    );
    metadata_row(
        &mut report,
        "rustc",
        command_output("rustc", &["--version"]).unwrap_or_else(|| "unknown".into()),
    );
    metadata_row(
        &mut report,
        "git commit",
        command_output("git", &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into()),
    );
    metadata_row(
        &mut report,
        "working tree",
        if command_output("git", &["status", "--porcelain"]).is_some_and(|status| status.is_empty())
        {
            "clean".into()
        } else {
            "dirty".into()
        },
    );
    metadata_row(
        &mut report,
        "power",
        power_metadata().unwrap_or_else(|| "unknown; record power profile manually".into()),
    );
    metadata_row(&mut report, "run timestamp (Unix)", timestamp.to_string());
    metadata_row(
        &mut report,
        "harness",
        format!(
            "{} warmup(s), {} measured run(s), max_tokens={}, greedy",
            args.warmups, args.runs, args.max_tokens
        ),
    );

    report.push_str("\n## Runtime configuration\n\n");
    report.push_str(
        "| runtime | endpoint | model | format / quantization | execution settings | status |\n",
    );
    report.push_str("|---|---|---|---|---|---|\n");
    for (runtime, state) in runtimes.iter().zip(states) {
        let status = match state {
            RuntimeState::Available { advertised_models } => {
                format!("available ({})", advertised_models.join(", "))
            }
            RuntimeState::Skipped { reason } => format!("skipped: {reason}"),
        };
        report.push_str(&format!(
            "| {} | `{}` | `{}` | {} | {} | {} |\n",
            escape_table(&runtime.name),
            escape_table(&runtime.base_url),
            escape_table(&runtime.model),
            escape_table(&runtime.format),
            escape_table(&runtime.settings),
            escape_table(&status)
        ));
    }

    report.push_str("\n## Methodology\n\n");
    report.push_str(&format!(
        "- OpenAI `POST /v1/chat/completions` streaming API; fixed system prompt; `temperature=0`, `top_p=1`, `seed=0`, and `max_tokens={}`.\n",
        args.max_tokens
    ));
    report.push_str(&format!(
        "- {} warmup run(s) were discarded, followed by {} measured run(s). Cells show **median / p90**; the bold cell is the median winner for that prompt and metric.\n",
        args.warmups, args.runs
    ));
    report.push_str("- TTFT is request start to first non-empty streamed content. Total latency ends when the response stream closes after `[DONE]`.\n");
    report.push_str("- Decode throughput excludes TTFT: `(generated_tokens - 1) / (total - TTFT)`. Generated tokens use streamed `usage.completion_tokens` when supplied, otherwise one non-empty content event is counted as one token.\n");
    report.push_str("- Estimated prefill throughput is `rendered_prompt_tokens / TTFT`; it includes HTTP, queueing, chat-template processing, and first-token decode, so treat it as an API-level estimate rather than kernel-only prefill speed.\n");
    report.push_str("- All runtimes receive the same explicit system/user messages. Prompt token counts use the Qwen2.5 chat template and tokenizer.\n");

    report.push_str("\n## Results\n\n");
    report.push_str("| prompt | prompt tokens | runtime | TTFT ms ↓ | decode tok/s ↑ | total ms ↓ | estimated prefill tok/s ↑ | output tokens |\n");
    report.push_str("|---|---:|---|---:|---:|---:|---:|---:|\n");
    for (prompt_index, prompt) in prompts.iter().enumerate() {
        let prompt_summaries = summaries
            .iter()
            .filter(|summary| summary.prompt_index == prompt_index)
            .collect::<Vec<_>>();
        for summary in &prompt_summaries {
            let ttft = metric_cell(
                summary.ttft_ms,
                is_winner(
                    summary.ttft_ms.median,
                    prompt_summaries.iter().map(|entry| entry.ttft_ms.median),
                    Direction::Lower,
                ),
                1,
            );
            let decode = metric_cell(
                summary.decode_tokens_per_second,
                is_winner(
                    summary.decode_tokens_per_second.median,
                    prompt_summaries
                        .iter()
                        .map(|entry| entry.decode_tokens_per_second.median),
                    Direction::Higher,
                ),
                2,
            );
            let total = metric_cell(
                summary.total_ms,
                is_winner(
                    summary.total_ms.median,
                    prompt_summaries.iter().map(|entry| entry.total_ms.median),
                    Direction::Lower,
                ),
                1,
            );
            let prefill = metric_cell(
                summary.estimated_prefill_tokens_per_second,
                is_winner(
                    summary.estimated_prefill_tokens_per_second.median,
                    prompt_summaries
                        .iter()
                        .map(|entry| entry.estimated_prefill_tokens_per_second.median),
                    Direction::Higher,
                ),
                1,
            );
            report.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {:.0} / {:.0} |\n",
                prompt.name,
                prompt.prompt_tokens,
                escape_table(&runtimes[summary.runtime_index].name),
                ttft,
                decode,
                total,
                prefill,
                summary.output_tokens.median,
                summary.output_tokens.p90
            ));
        }
    }

    report.push_str("\n## Automatic comparison against onnx-genai\n\n");
    report.push_str(
        "| prompt | competitor | TTFT | decode throughput | total latency | estimated prefill |\n",
    );
    report.push_str("|---|---|---:|---:|---:|---:|\n");
    if let Some(onnx_index) = runtimes
        .iter()
        .position(|runtime| runtime.name == "onnx-genai")
    {
        for (prompt_index, prompt) in prompts.iter().enumerate() {
            let onnx = summaries.iter().find(|summary| {
                summary.runtime_index == onnx_index && summary.prompt_index == prompt_index
            });
            if let Some(onnx) = onnx {
                for competitor in summaries.iter().filter(|summary| {
                    summary.prompt_index == prompt_index && summary.runtime_index != onnx_index
                }) {
                    report.push_str(&format!(
                        "| {} | {} | {} | {} | {} | {} |\n",
                        prompt.name,
                        escape_table(&runtimes[competitor.runtime_index].name),
                        relative_latency(onnx.ttft_ms.median, competitor.ttft_ms.median),
                        relative_throughput(
                            onnx.decode_tokens_per_second.median,
                            competitor.decode_tokens_per_second.median
                        ),
                        relative_latency(onnx.total_ms.median, competitor.total_ms.median),
                        relative_throughput(
                            onnx.estimated_prefill_tokens_per_second.median,
                            competitor.estimated_prefill_tokens_per_second.median
                        )
                    ));
                }
            }
        }
    } else {
        report.push_str("| — | — | onnx-genai result unavailable | — | — | — |\n");
    }

    if !run_notes.is_empty() {
        report.push_str("\n## Skips and run failures\n\n");
        for note in run_notes {
            report.push_str(&format!("- {}\n", note));
        }
    }
    report.push_str("\n## Fairness caveats\n\n");
    report.push_str("- The HTTP layer, model family/size, prompts, and generation policy are common, but ONNX and GGUF quantizations can differ. The runtime table is part of the result and must identify the exact formats; do not compare unlabeled or selectively chosen quants.\n");
    report.push_str("- This is single-request latency/decode performance, not concurrent serving throughput. Background load, thermals, power source, and model residency affect results.\n");
    report.push_str("- Default runtime threading is intentional because this measures deployment behavior. The single-thread ORT setting used by exact-equality tests is not used here.\n");
    report
}

fn metadata_row(report: &mut String, field: &str, value: String) {
    report.push_str(&format!(
        "| {} | {} |\n",
        escape_table(field),
        escape_table(value.trim())
    ));
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = StdCommand::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn power_metadata() -> Option<String> {
    let battery = command_output("pmset", &["-g", "batt"])?;
    let source = battery
        .lines()
        .next()
        .and_then(|line| line.split('\'').nth(1))
        .unwrap_or("unknown power source");
    let mode = command_output("pmset", &["-g", "custom"])
        .and_then(|output| {
            output
                .lines()
                .find(|line| line.trim_start().starts_with("powermode"))
                .and_then(|line| line.split_whitespace().last())
                .map(ToString::to_string)
        })
        .unwrap_or_else(|| "unknown".into());
    Some(format!("{source}; macOS powermode={mode}"))
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn calendar_date() -> String {
    command_output("date", &["+%Y-%m-%d"]).unwrap_or_else(|| "unknown-date".into())
}

fn escape_table(value: impl fmt::Display) -> String {
    value.to_string().replace('|', "\\|").replace('\n', " ")
}

#[derive(Clone, Copy)]
enum Direction {
    Lower,
    Higher,
}

fn is_winner(value: f64, values: impl Iterator<Item = f64>, direction: Direction) -> bool {
    let winner = match direction {
        Direction::Lower => values.fold(f64::INFINITY, f64::min),
        Direction::Higher => values.fold(f64::NEG_INFINITY, f64::max),
    };
    (value - winner).abs() <= f64::EPSILON * winner.abs().max(1.0)
}

fn metric_cell(distribution: Distribution, winner: bool, precision: usize) -> String {
    let value = format!(
        "{:.*} / {:.*}",
        precision, distribution.median, precision, distribution.p90
    );
    if winner {
        format!("**{value}**")
    } else {
        value
    }
}

fn spread_cell(distribution: Distribution, precision: usize) -> String {
    format!(
        "{:.*} [{:.*}, {:.*}]",
        precision, distribution.median, precision, distribution.p10, precision, distribution.p95
    )
}

fn percent_spread_cell(distribution: Distribution, precision: usize) -> String {
    spread_cell(distribution_scale(distribution, 100.0), precision)
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn write_json_arg(path: &Path, value: &Value) -> Result<()> {
    let json = format!("{}\n", serde_json::to_string_pretty(value)?);
    if path.as_os_str() == "-" {
        print!("{json}");
        return Ok(());
    }
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))
}

fn relative_latency(onnx: f64, competitor: f64) -> String {
    let percent = (onnx / competitor - 1.0) * 100.0;
    if percent <= 0.0 {
        format!("{:.1}% lower", -percent)
    } else {
        format!("{percent:.1}% higher")
    }
}

fn relative_throughput(onnx: f64, competitor: f64) -> String {
    let percent = (onnx / competitor - 1.0) * 100.0;
    if percent >= 0.0 {
        format!("{percent:.1}% faster")
    } else {
        format!("{:.1}% slower", -percent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_runtime_spec() {
        let runtime = parse_runtime("test|http://localhost:1/v1/|model|Q8_0|CPU").unwrap();
        assert_eq!(runtime.name, "test");
        assert_eq!(runtime.base_url, "http://localhost:1/v1");
        assert_eq!(
            runtime.completions_url(),
            "http://localhost:1/v1/chat/completions"
        );
    }

    #[test]
    fn rejects_incomplete_runtime_spec() {
        assert!(parse_runtime("test|http://localhost|model").is_err());
    }

    #[test]
    fn computes_interpolated_percentiles() {
        let distribution = Distribution::from_values(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(distribution.median, 3.0);
        assert_eq!(distribution.p90, 4.6);
    }

    #[test]
    fn parses_fragmented_lf_sse_events() {
        let mut bytes = br#"data: {"choices":[{"delta":{"content":"hi"}}]}

data: [DONE]

tail"#
            .to_vec();
        let events = take_sse_events(&mut bytes).unwrap();
        assert_eq!(
            events,
            vec![r#"{"choices":[{"delta":{"content":"hi"}}]}"#, "[DONE]"]
        );
        assert_eq!(bytes, b"tail");
    }

    #[test]
    fn parses_crlf_and_multiline_sse() {
        let mut bytes = b"event: message\r\ndata: first\r\ndata: second\r\n\r\n".to_vec();
        assert_eq!(take_sse_events(&mut bytes).unwrap(), vec!["first\nsecond"]);
        assert!(bytes.is_empty());
    }

    #[test]
    fn formats_relative_results() {
        assert_eq!(relative_latency(80.0, 100.0), "20.0% lower");
        assert_eq!(relative_latency(120.0, 100.0), "20.0% higher");
        assert_eq!(relative_throughput(120.0, 100.0), "20.0% faster");
        assert_eq!(relative_throughput(80.0, 100.0), "20.0% slower");
    }

    #[test]
    fn renders_explicit_qwen_chat_prompt() {
        let rendered = render_qwen_prompt("hello");
        assert!(rendered.contains(SYSTEM_PROMPT));
        assert!(rendered.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn winner_respects_metric_direction() {
        assert!(is_winner(1.0, [1.0, 2.0].into_iter(), Direction::Lower));
        assert!(is_winner(2.0, [1.0, 2.0].into_iter(), Direction::Higher));
    }
}
