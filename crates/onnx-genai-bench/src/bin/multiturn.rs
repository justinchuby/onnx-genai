//! Multi-turn LLM benchmark: measures the break-even point where ORT's
//! amortised pre-packing cost overtakes native's faster load.
//!
//! Design: Model loaded once per backend, then N sequential turns with growing
//! conversation context (as in real multi-turn chat). Reports per-turn TTFT,
//! decode throughput, and cumulative wall-clock. Interleaved A/B in one process.
//!
//! Both native and ORT use their session APIs for KV reuse by default.
//! Pass `--native-stateless` to measure the old stateless path (full re-prefill
//! each turn) for comparison.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command as StdCommand,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde_json::{Value, json};

#[cfg(feature = "bench-native")]
use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GenerateRequest, NativeDecodeDevice,
};
#[cfg(feature = "bench-native")]
use onnx_genai_ort::{ChatMessage, ChatTemplate, available_execution_providers};
#[cfg(feature = "bench-native")]
use onnx_runtime_ep_cpu::kernels::matmul::weight_transpose_cache_sizes;

const DEFAULT_TURNS: usize = 10;
const DEFAULT_TOKENS_PER_TURN: usize = 30;

/// Conversation prompts that simulate a real multi-turn chat session.
const TURN_PROMPTS: &[&str] = &[
    "What is ONNX and why does it matter for model portability?",
    "How does weight pre-packing improve inference latency?",
    "What are the tradeoffs of pre-packing during model load vs lazy preparation?",
    "Explain KV cache reuse across conversation turns.",
    "How does batch size affect throughput for transformer models?",
    "Compare memory-mapped weight loading versus session graph construction.",
    "What is the roofline model and how does it apply to LLM decode?",
    "Describe the difference between prefill and decode phases.",
    "How do Apple Silicon's AMX and NEON units differ for GEMM workloads?",
    "What metrics matter most for real-world LLM serving latency?",
    "Explain how speculative decoding can improve generation throughput.",
    "What is the relationship between model size and memory bandwidth utilization?",
    "How does context length growth affect per-turn latency?",
    "Compare greedy vs top-p sampling in terms of computational cost.",
    "What are the key differences between ONNX Runtime and a native inference engine?",
    "How do thread pools affect inference under host contention?",
    "Explain prefix caching and when it helps multi-turn workloads.",
    "What is the impact of FP16 vs FP32 on decode throughput?",
    "Describe how model quantization affects latency and accuracy.",
    "Summarize the key performance factors for multi-turn LLM inference.",
];

#[derive(Debug, Parser)]
#[command(
    name = "onnx-genai-multiturn",
    about = "Multi-turn LLM benchmark: find break-even turn count for native vs ORT"
)]
struct Args {
    /// Model directory.
    #[arg(long)]
    model: PathBuf,

    /// Number of conversation turns.
    #[arg(long, default_value_t = DEFAULT_TURNS)]
    turns: usize,

    /// Maximum tokens generated per turn.
    #[arg(long, default_value_t = DEFAULT_TOKENS_PER_TURN)]
    tokens_per_turn: usize,

    /// Number of full multi-turn session repetitions (median reported).
    #[arg(long, default_value_t = 3)]
    repetitions: usize,

    /// Use the stateless native path (full re-prefill each turn, no KV reuse).
    /// Default is to use create_session + generate_in_session
    /// for KV persistence across turns, matching ORT's session API.
    #[arg(long)]
    native_stateless: bool,

    /// Write JSON results to this path. Use '-' for stdout.
    #[arg(long, value_name = "PATH")]
    profile_json: Option<PathBuf>,
}

#[derive(Clone, Debug)]
struct TurnResult {
    turn: usize,
    ttft: Duration,
    decode_duration: Duration,
    total_turn_duration: Duration,
    generated_tokens: usize,
    prompt_tokens: usize,
}

#[derive(Clone, Debug)]
struct SessionResult {
    backend: &'static str,
    model_load: Duration,
    turns: Vec<TurnResult>,
    total_session_wall_clock: Duration,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.turns == 0 {
        bail!("--turns must be at least 1");
    }
    if args.tokens_per_turn < 2 {
        bail!("--tokens-per-turn must be at least 2");
    }
    if args.repetitions == 0 {
        bail!("--repetitions must be at least 1");
    }
    #[cfg(not(feature = "bench-native"))]
    {
        bail!(
            "multi-turn benchmark requires `cargo run -p onnx-genai-bench \
             --features bench-native --bin multiturn -- --model <path>`"
        );
    }
    #[cfg(feature = "bench-native")]
    run_multiturn(&args)
}

#[cfg(feature = "bench-native")]
fn run_multiturn(args: &Args) -> Result<()> {
    let model_dir = if args.model.is_file() {
        args.model
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .context("--model file has no parent directory")?
    } else {
        args.model.as_path()
    };
    if !model_dir.is_dir() {
        bail!("model directory does not exist: {}", model_dir.display());
    }

    let load_avg = get_load_avg();
    eprintln!("host load: {load_avg}");
    let native_mode = if args.native_stateless {
        "stateless (full re-prefill)"
    } else {
        "session (KV reuse)"
    };
    eprintln!(
        "model: {} | turns: {} | tokens/turn: {} | repetitions: {} | native: {}",
        model_dir.display(),
        args.turns,
        args.tokens_per_turn,
        args.repetitions,
        native_mode,
    );

    // Resolve chat template for multi-turn rendering
    let template = ChatTemplate::from_model_dir(model_dir)
        .with_context(|| format!("load chat template from {}", model_dir.display()))?;

    // Run interleaved repetitions: native then ORT per rep to control for thermal drift
    let mut native_results = Vec::with_capacity(args.repetitions);
    let mut ort_results = Vec::with_capacity(args.repetitions);

    for rep in 1..=args.repetitions {
        eprintln!("\n--- repetition {rep}/{} ---", args.repetitions);

        // Native: session API with KV reuse (default) or stateless re-prefill
        eprintln!(
            "  native ({}): running {}-turn session...",
            if args.native_stateless {
                "stateless"
            } else {
                "session"
            },
            args.turns
        );
        let native = run_native_session(args, model_dir, &template)?;
        native_results.push(native);

        // ORT: uses create_session + generate_in_session (KV reuse)
        eprintln!("  ORT: running {}-turn session...", args.turns);
        let ort = run_ort_session(args, model_dir, &template)?;
        ort_results.push(ort);
    }

    let load_avg_after = get_load_avg();

    // Pick median by total session wall-clock
    let native_median = pick_median(&native_results);
    let ort_median = pick_median(&ort_results);

    // Find break-even point
    let break_even = find_break_even(native_median, ort_median);

    // Render report
    let report = render_report(
        args,
        model_dir,
        native_median,
        ort_median,
        break_even,
        &load_avg,
        &load_avg_after,
    );

    if args
        .profile_json
        .as_ref()
        .is_some_and(|p| p.as_os_str() == "-")
    {
        eprint!("{report}");
        let json = build_json(
            args,
            model_dir,
            native_median,
            ort_median,
            break_even,
            &load_avg,
            &load_avg_after,
        );
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        print!("{report}");
        if let Some(path) = &args.profile_json {
            let json = build_json(
                args,
                model_dir,
                native_median,
                ort_median,
                break_even,
                &load_avg,
                &load_avg_after,
            );
            write_json(path, &json)?;
        }
    }
    Ok(())
}

#[cfg(feature = "bench-native")]
fn run_native_session(
    args: &Args,
    model_dir: &Path,
    template: &ChatTemplate,
) -> Result<SessionResult> {
    unsafe {
        std::env::set_var("ONNX_GENAI_EP", "cpu");
    }

    let mut config = EngineConfig {
        decode_backend: EngineDecodeBackend::Native,
        ..EngineConfig::default()
    };
    config.native_device = Some(NativeDecodeDevice::Cpu);

    let load_start = Instant::now();
    let mut engine = Engine::from_dir(model_dir, config)
        .with_context(|| format!("load native from {}", model_dir.display()))?;
    let model_load = load_start.elapsed();

    // Cache verification: record weight transpose cache state after model load
    let cache_after_load = weight_transpose_cache_sizes();
    eprintln!(
        "    cache after load: f16={}, f32={}",
        cache_after_load.0, cache_after_load.1
    );

    if engine.decode_backend() != EngineDecodeBackend::Native {
        bail!(
            "requested native but engine resolved {:?}",
            engine.decode_backend()
        );
    }

    let backend_label = if args.native_stateless {
        "native-stateless"
    } else {
        "native"
    };

    // Create a persistent session for KV reuse (unless stateless mode). The
    // backend (native here) is selected by the engine's decode_backend.
    let session_id = if args.native_stateless {
        None
    } else {
        Some(engine.create_session()?)
    };

    let session_start = Instant::now();
    let mut conversation: Vec<ChatMessage> = Vec::new();
    let mut turns = Vec::with_capacity(args.turns);

    for turn_idx in 0..args.turns {
        let prompt_text = TURN_PROMPTS[turn_idx % TURN_PROMPTS.len()];
        conversation.push(ChatMessage::new("user", prompt_text.to_string()));

        let rendered = template
            .render(&conversation, None, true)
            .with_context(|| format!("render template for turn {turn_idx}"))?;

        let prompt_tokens = rendered.len() / 4; // approximate

        let mut request = GenerateRequest::new(rendered);
        request.options.max_new_tokens = args.tokens_per_turn;
        request.options.temperature = 0.0;
        request.options.top_p = 1.0;
        request.options.greedy = true;
        request.options.seed = Some(0);
        request.options.stop_on_eos = false;

        let turn_start = Instant::now();
        let mut token_times = Vec::with_capacity(args.tokens_per_turn);
        let mut callback = |_| {
            token_times.push(turn_start.elapsed());
            Ok(())
        };

        let result = if let Some(sid) = session_id {
            // Session API: KV state persists across turns, only new tokens prefilled
            engine
                .generate_in_session_with_callback(sid, request, Some(&mut callback))
                .with_context(|| format!("native session generate turn {turn_idx}"))?
        } else {
            // Stateless: full re-prefill each turn (legacy behaviour)
            engine
                .generate_with_callback(request, Some(&mut callback))
                .with_context(|| format!("native stateless generate turn {turn_idx}"))?
        };
        let total_turn = turn_start.elapsed();

        let ttft = token_times.first().copied().unwrap_or(total_turn);
        let decode_duration = total_turn.saturating_sub(ttft);

        // Add assistant response to conversation for context growth
        conversation.push(ChatMessage::new("assistant", result.text.clone()));

        turns.push(TurnResult {
            turn: turn_idx,
            ttft,
            decode_duration,
            total_turn_duration: total_turn,
            generated_tokens: result.token_ids.len(),
            prompt_tokens,
        });
    }

    // Close the session if we created one
    if let Some(sid) = session_id {
        engine.close_session(sid)?;
    }

    // Cache verification: check cache state after all turns
    let cache_after_turns = weight_transpose_cache_sizes();
    eprintln!(
        "    cache after {} turns: f16={}, f32={} (stable={})",
        args.turns,
        cache_after_turns.0,
        cache_after_turns.1,
        cache_after_load == cache_after_turns
    );

    Ok(SessionResult {
        backend: backend_label,
        model_load,
        turns,
        total_session_wall_clock: session_start.elapsed(),
    })
}

#[cfg(feature = "bench-native")]
fn run_ort_session(
    args: &Args,
    model_dir: &Path,
    template: &ChatTemplate,
) -> Result<SessionResult> {
    unsafe {
        std::env::set_var("ONNX_GENAI_EP", "cpu");
    }

    let available =
        available_execution_providers().context("query linked ONNX Runtime providers")?;
    if !available
        .iter()
        .any(|p| p.eq_ignore_ascii_case("CPUExecutionProvider"))
    {
        bail!("ORT CPU EP unavailable (available: {available:?})");
    }

    let config = EngineConfig {
        decode_backend: EngineDecodeBackend::Ort,
        ..EngineConfig::default()
    };

    let load_start = Instant::now();
    let mut engine = Engine::from_dir(model_dir, config)
        .with_context(|| format!("load ORT from {}", model_dir.display()))?;
    let model_load = load_start.elapsed();

    if engine.decode_backend() != EngineDecodeBackend::Ort {
        bail!(
            "requested ORT but engine resolved {:?}",
            engine.decode_backend()
        );
    }

    let session_start = Instant::now();
    let session_id = engine.create_session()?;
    let mut conversation: Vec<ChatMessage> = Vec::new();
    let mut turns = Vec::with_capacity(args.turns);

    for turn_idx in 0..args.turns {
        let prompt_text = TURN_PROMPTS[turn_idx % TURN_PROMPTS.len()];
        conversation.push(ChatMessage::new("user", prompt_text.to_string()));

        // For ORT with persistent session, pass only the incremental content.
        // The session already has KV for previous turns. We render:
        // - Turn 1: full conversation (system + user1 + gen prompt)
        // - Turn N>1: just the new assistant response marker + user message + gen prompt
        // The engine appends new tokens to the session's existing context.
        let turn_prompt = if turn_idx == 0 {
            template
                .render(&conversation, None, true)
                .with_context(|| format!("render template for turn {turn_idx}"))?
        } else {
            // Build incremental: previous assistant response + new user turn + generation prompt
            // This matches how a server would use the session API
            let prev_assistant = &conversation[conversation.len() - 2]; // prior assistant msg
            let new_user = &conversation[conversation.len() - 1]; // current user msg
            format!(
                "{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                prev_assistant.content, new_user.content
            )
        };

        let prompt_tokens = turn_prompt.len() / 4; // approximate

        let mut request = GenerateRequest::new(turn_prompt);
        request.options.max_new_tokens = args.tokens_per_turn;
        request.options.temperature = 0.0;
        request.options.top_p = 1.0;
        request.options.greedy = true;
        request.options.seed = Some(0);
        request.options.stop_on_eos = false;

        let turn_start = Instant::now();
        let mut token_times = Vec::with_capacity(args.tokens_per_turn);
        let mut callback = |_| {
            token_times.push(turn_start.elapsed());
            Ok(())
        };
        let result = engine
            .generate_in_session_with_callback(session_id, request, Some(&mut callback))
            .with_context(|| format!("ORT generate turn {turn_idx}"))?;
        let total_turn = turn_start.elapsed();

        let ttft = token_times.first().copied().unwrap_or(total_turn);
        let decode_duration = total_turn.saturating_sub(ttft);

        conversation.push(ChatMessage::new("assistant", result.text.clone()));

        turns.push(TurnResult {
            turn: turn_idx,
            ttft,
            decode_duration,
            total_turn_duration: total_turn,
            generated_tokens: result.token_ids.len(),
            prompt_tokens,
        });
    }

    engine.close_session(session_id)?;

    Ok(SessionResult {
        backend: "ort",
        model_load,
        turns,
        total_session_wall_clock: session_start.elapsed(),
    })
}

fn pick_median(results: &[SessionResult]) -> &SessionResult {
    if results.len() == 1 {
        return &results[0];
    }
    let mut indexed: Vec<(usize, Duration)> = results
        .iter()
        .enumerate()
        .map(|(i, r)| (i, r.total_session_wall_clock))
        .collect();
    indexed.sort_by_key(|(_, d)| *d);
    &results[indexed[indexed.len() / 2].0]
}

/// Find the turn at which ORT's cumulative time (load + all turns up to N)
/// becomes lower than native's cumulative time.
fn find_break_even(native: &SessionResult, ort: &SessionResult) -> Option<usize> {
    let num_turns = native.turns.len().min(ort.turns.len());
    let mut native_cumulative = native.model_load;
    let mut ort_cumulative = ort.model_load;

    for i in 0..num_turns {
        native_cumulative += native.turns[i].total_turn_duration;
        ort_cumulative += ort.turns[i].total_turn_duration;
        if ort_cumulative < native_cumulative {
            return Some(i + 1); // 1-indexed turn number
        }
    }
    None
}

fn render_report(
    args: &Args,
    model_dir: &Path,
    native: &SessionResult,
    ort: &SessionResult,
    break_even: Option<usize>,
    load_before: &str,
    load_after: &str,
) -> String {
    let mut report = String::new();
    report.push_str("# Multi-turn LLM Benchmark: Native vs ORT\n\n");

    // Metadata
    report.push_str("| field | value |\n|---|---|\n");
    meta(&mut report, "model", &model_dir.display().to_string());
    meta(&mut report, "turns", &args.turns.to_string());
    meta(
        &mut report,
        "tokens/turn",
        &args.tokens_per_turn.to_string(),
    );
    meta(&mut report, "repetitions", &args.repetitions.to_string());
    meta(
        &mut report,
        "native mode",
        if native.backend == "native-stateless" {
            "stateless (full re-prefill each turn)"
        } else {
            "session (KV reuse via create_session)"
        },
    );
    meta(&mut report, "host load (before)", load_before);
    meta(&mut report, "host load (after)", load_after);
    meta(
        &mut report,
        "machine",
        &command_output("uname", &["-srmp"]).unwrap_or_else(|| "unknown".into()),
    );
    meta(
        &mut report,
        "git commit",
        &command_output("git", &["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "?".into()),
    );
    report.push('\n');

    // Break-even headline
    report.push_str("## Break-even analysis\n\n");
    match break_even {
        Some(turn) => {
            report.push_str(&format!(
                "**Break-even turn: {turn}** — after {turn} turn(s), ORT's cumulative \
                 wall-clock (including its slower model load) drops below native's. \
                 ORT's pre-packing cost is amortised by turn {turn}.\n\n"
            ));
        }
        None => {
            report.push_str(&format!(
                "**No break-even within {} turns** — native's cumulative wall-clock \
                 remains lower than ORT's across all measured turns. ORT's pre-packing \
                 cost is NOT amortised within this session length.\n\n",
                args.turns
            ));
        }
    }

    // Model load comparison
    report.push_str("## Model load\n\n");
    report.push_str("| backend | model load ms |\n|---|---:|\n");
    report.push_str(&format!(
        "| native | {:.1} |\n",
        native.model_load.as_secs_f64() * 1000.0
    ));
    report.push_str(&format!(
        "| ORT | {:.1} |\n",
        ort.model_load.as_secs_f64() * 1000.0
    ));
    let load_ratio = ort.model_load.as_secs_f64() / native.model_load.as_secs_f64();
    report.push_str(&format!(
        "\nORT model load is {load_ratio:.1}× slower than native.\n\n"
    ));

    // Per-turn table
    report.push_str("## Per-turn results\n\n");
    report.push_str(
        "| turn | native TTFT ms | ORT TTFT ms | native decode ms | ORT decode ms | \
         native total ms | ORT total ms | native cumul ms | ORT cumul ms | winner |\n",
    );
    report.push_str("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n");

    let num_turns = native.turns.len().min(ort.turns.len());
    let mut native_cumul = native.model_load;
    let mut ort_cumul = ort.model_load;

    for i in 0..num_turns {
        native_cumul += native.turns[i].total_turn_duration;
        ort_cumul += ort.turns[i].total_turn_duration;
        let winner = if native_cumul <= ort_cumul {
            "native"
        } else {
            "**ORT**"
        };
        report.push_str(&format!(
            "| {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {} |\n",
            i + 1,
            ms(native.turns[i].ttft),
            ms(ort.turns[i].ttft),
            ms(native.turns[i].decode_duration),
            ms(ort.turns[i].decode_duration),
            ms(native.turns[i].total_turn_duration),
            ms(ort.turns[i].total_turn_duration),
            ms(native_cumul),
            ms(ort_cumul),
            winner,
        ));
    }
    report.push('\n');

    // Summary
    report.push_str("## Summary\n\n");
    let native_total = ms(native.model_load)
        + native
            .turns
            .iter()
            .map(|t| ms(t.total_turn_duration))
            .sum::<f64>();
    let ort_total = ms(ort.model_load)
        + ort
            .turns
            .iter()
            .map(|t| ms(t.total_turn_duration))
            .sum::<f64>();
    report.push_str(&format!(
        "- Native total session: {native_total:.0} ms (load {:.0} ms + {} turns)\n",
        ms(native.model_load),
        num_turns
    ));
    report.push_str(&format!(
        "- ORT total session: {ort_total:.0} ms (load {:.0} ms + {} turns)\n",
        ms(ort.model_load),
        num_turns
    ));
    let session_ratio = native_total / ort_total;
    if session_ratio < 1.0 {
        report.push_str(&format!(
            "- Over {} turns: native {:.2}× faster overall\n",
            num_turns,
            1.0 / session_ratio
        ));
    } else {
        report.push_str(&format!(
            "- Over {num_turns} turns: **ORT {session_ratio:.2}× faster overall**\n"
        ));
    }

    // TTFT trend (does native TTFT grow with context while ORT stays flat?)
    if num_turns >= 3 {
        let native_first_ttft = ms(native.turns[0].ttft);
        let native_last_ttft = ms(native.turns[num_turns - 1].ttft);
        let ort_first_ttft = ms(ort.turns[0].ttft);
        let ort_last_ttft = ms(ort.turns[num_turns - 1].ttft);
        report.push_str(&format!(
            "- Native TTFT trend: {native_first_ttft:.1} ms (turn 1) → {native_last_ttft:.1} ms (turn {num_turns}) \
             [{:.1}× growth]\n",
            native_last_ttft / native_first_ttft.max(0.001)
        ));
        report.push_str(&format!(
            "- ORT TTFT trend: {ort_first_ttft:.1} ms (turn 1) → {ort_last_ttft:.1} ms (turn {num_turns}) \
             [{:.1}× growth]\n",
            ort_last_ttft / ort_first_ttft.max(0.001)
        ));
    }

    // Steady-state per-prefill analysis (turns 3+ to exclude warm-up effects)
    report.push_str("\n## Steady-state per-prefill analysis\n\n");
    if num_turns >= 4 {
        let steady_start = 2; // 0-indexed, skip first 2 turns
        let native_steady_ttfts: Vec<f64> = native.turns[steady_start..num_turns]
            .iter()
            .map(|t| ms(t.ttft))
            .collect();
        let ort_steady_ttfts: Vec<f64> = ort.turns[steady_start..num_turns]
            .iter()
            .map(|t| ms(t.ttft))
            .collect();
        let native_avg_ttft =
            native_steady_ttfts.iter().sum::<f64>() / native_steady_ttfts.len() as f64;
        let ort_avg_ttft = ort_steady_ttfts.iter().sum::<f64>() / ort_steady_ttfts.len() as f64;
        let native_steady_decode: Vec<f64> = native.turns[steady_start..num_turns]
            .iter()
            .map(|t| ms(t.decode_duration))
            .collect();
        let ort_steady_decode: Vec<f64> = ort.turns[steady_start..num_turns]
            .iter()
            .map(|t| ms(t.decode_duration))
            .collect();
        let native_avg_decode =
            native_steady_decode.iter().sum::<f64>() / native_steady_decode.len() as f64;
        let ort_avg_decode = ort_steady_decode.iter().sum::<f64>() / ort_steady_decode.len() as f64;

        report.push_str(&format!(
            "Steady-state (turns {}-{}, excluding warm-up):\n\n",
            steady_start + 1,
            num_turns
        ));
        report.push_str("| metric | native ms | ORT ms | ratio |\n|---|---:|---:|---:|\n");
        report.push_str(&format!(
            "| avg TTFT (prefill) | {native_avg_ttft:.1} | {ort_avg_ttft:.1} | {:.2}× |\n",
            native_avg_ttft / ort_avg_ttft.max(0.001)
        ));
        report.push_str(&format!(
            "| avg decode | {native_avg_decode:.1} | {ort_avg_decode:.1} | {:.2}× |\n",
            native_avg_decode / ort_avg_decode.max(0.001)
        ));
        let native_avg_total = native_avg_ttft + native_avg_decode;
        let ort_avg_total = ort_avg_ttft + ort_avg_decode;
        report.push_str(&format!(
            "| avg total/turn | {native_avg_total:.1} | {ort_avg_total:.1} | {:.2}× |\n\n",
            native_avg_total / ort_avg_total.max(0.001)
        ));

        // Decompose: how much of ORT's advantage is amortised pre-packing vs faster kernel?
        report.push_str("### Advantage decomposition\n\n");
        let load_advantage_ms = ms(ort.model_load).max(0.0) - ms(native.model_load).max(0.0);
        let per_turn_advantage_ms = native_avg_total - ort_avg_total;
        report.push_str(&format!(
            "- ORT load penalty (one-time): {load_advantage_ms:.0} ms slower\n"
        ));
        report.push_str(&format!(
            "- ORT per-turn advantage (steady-state): {per_turn_advantage_ms:.0} ms/turn faster\n"
        ));
        if per_turn_advantage_ms > 0.0 {
            let turns_to_amortise = (load_advantage_ms / per_turn_advantage_ms).ceil() as usize;
            report.push_str(&format!(
                "- Turns to amortise load penalty: ~{turns_to_amortise}\n"
            ));
            report.push_str(&format!(
                "- **Root cause: ORT is {:.1}× faster per prefill (not just amortisation)**\n",
                native_avg_ttft / ort_avg_ttft.max(0.001)
            ));
            report.push_str(&format!(
                "- ORT decode is also {:.1}× faster per turn\n\n",
                native_avg_decode / ort_avg_decode.max(0.001)
            ));
        }

        // Target: what native TTFT would need to be to win at every turn count
        report.push_str("### Target to beat ORT at every turn count\n\n");
        report.push_str(
            "For native to win at ALL turn counts, per-turn time must be ≤ ORT's. \
             Given the load advantage, native needs:\n\n",
        );
        report.push_str(&format!(
            "- Per-prefill TTFT target: ≤ {ort_avg_ttft:.1} ms (currently {native_avg_ttft:.1} ms, need {:.0}% reduction)\n",
            ((native_avg_ttft - ort_avg_ttft) / native_avg_ttft) * 100.0
        ));
        report.push_str(&format!(
            "- Per-turn decode target: ≤ {ort_avg_decode:.1} ms (currently {native_avg_decode:.1} ms, need {:.0}% reduction)\n\n",
            ((native_avg_decode - ort_avg_decode) / native_avg_decode) * 100.0
        ));

        // Derive the KV narrative from the actual TTFT growth observed
        let first_ttft_n = ms(native.turns[0].ttft);
        let last_ttft_n = ms(native.turns[num_turns - 1].ttft);
        let first_ttft_o = ms(ort.turns[0].ttft);
        let last_ttft_o = ms(ort.turns[num_turns - 1].ttft);
        let native_ttft_growth = last_ttft_n / first_ttft_n.max(0.001);
        let ort_ttft_growth = last_ttft_o / first_ttft_o.max(0.001);
        if native_ttft_growth > 2.0 && ort_ttft_growth < 1.5 {
            report.push_str(
                "**NOTE:** Native TTFT grows significantly with context length while ORT's stays \
                 ~flat. This pattern indicates native is re-prefilling growing context each turn \
                 (no KV persistence), while ORT incrementally extends its KV cache. \
                 Session-persistent KV for the native backend would eliminate this growth.\n",
            );
        } else if native_ttft_growth < 1.5 && ort_ttft_growth < 1.5 {
            report.push_str(
                "**NOTE:** Both backends show ~flat TTFT across turns, consistent with \
                 KV persistence (only new tokens are prefilled each turn). \
                 The session API is active for both backends.\n",
            );
        } else {
            report.push_str(&format!(
                "**NOTE:** Native TTFT growth: {native_ttft_growth:.1}×, ORT TTFT growth: \
                 {ort_ttft_growth:.1}×. Interpret per-turn cost differences in light of \
                 whether each backend is using session-persistent KV.\n",
            ));
        }
    }

    report
}

fn build_json(
    args: &Args,
    model_dir: &Path,
    native: &SessionResult,
    ort: &SessionResult,
    break_even: Option<usize>,
    load_before: &str,
    load_after: &str,
) -> Value {
    json!({
        "benchmark": "multiturn",
        "model": model_dir.display().to_string(),
        "turns": args.turns,
        "tokens_per_turn": args.tokens_per_turn,
        "repetitions": args.repetitions,
        "native_stateless": args.native_stateless,
        "host_load_before": load_before,
        "host_load_after": load_after,
        "break_even_turn": break_even,
        "native": session_json(native),
        "ort": session_json(ort),
    })
}

fn session_json(result: &SessionResult) -> Value {
    json!({
        "backend": result.backend,
        "model_load_ms": ms(result.model_load),
        "total_session_wall_clock_ms": ms(result.total_session_wall_clock),
        "turns": result.turns.iter().map(|t| json!({
            "turn": t.turn,
            "ttft_ms": ms(t.ttft),
            "decode_duration_ms": ms(t.decode_duration),
            "total_turn_ms": ms(t.total_turn_duration),
            "generated_tokens": t.generated_tokens,
            "prompt_tokens_approx": t.prompt_tokens,
        })).collect::<Vec<_>>(),
    })
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn meta(report: &mut String, field: &str, value: &str) {
    report.push_str(&format!("| {field} | {value} |\n"));
}

fn get_load_avg() -> String {
    command_output("sysctl", &["-n", "vm.loadavg"]).unwrap_or_else(|| "unknown".into())
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = StdCommand::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    let json = format!("{}\n", serde_json::to_string_pretty(value)?);
    if path.as_os_str() == "-" {
        print!("{json}");
        return Ok(());
    }
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).with_context(|| format!("create dir {}", parent.display()))?;
    }
    fs::write(path, json).with_context(|| format!("write {}", path.display()))
}
