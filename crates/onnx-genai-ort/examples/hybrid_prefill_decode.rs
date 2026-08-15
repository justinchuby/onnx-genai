//! Hybrid execution prototype: PREFILL on the GPU (Metal EP), DECODE on the CPU EP.
//!
//! Premise (Justin): single-token decode is a memory-bandwidth + dispatch-latency
//! bound GEMV where GPU parallelism does not pay and per-token command-buffer sync
//! dominates, so the CPU EP wins decode. PREFILL is a compute-bound GEMM over the
//! whole prompt (large M), where GPU parallelism *should* pay — and long-prompt
//! TTFT is our weakest metric. So: prefill on the Metal EP, decode on the CPU EP.
//!
//! This example runs the same model through two ORT sessions (one Metal, one CPU),
//! measures prefill/TTFT and decode tok/s separately for pure-CPU, pure-Metal, and
//! the hybrid path, and verifies the hybrid token stream stays coherent against a
//! pure-CPU run. The KV handoff uses `DecodeSession::export_kv`/`import_kv`: after
//! the Metal prefill produces `present.*`, the KV is materialized on host (cheap on
//! Apple unified memory) and adopted by the CPU decode session, which continues
//! generation from the prompt length.
//!
//! Usage:
//!   ONNX_GENAI_METAL_EP_LIB=<abs path to libonnxruntime_mlx_ep.dylib> \
//!   cargo run -p onnx-genai-ort --release --example hybrid_prefill_decode -- \
//!     --model models/qwen2.5-0.5b-cpu-recipe \
//!     --prompt "What is the capital of France?" --pad-to 512 --max-tokens 32 \
//!     --mode all

use std::{
    env,
    path::PathBuf,
    time::{Duration, Instant},
};

use onnx_genai_ort::{
    ChatMessage, ChatTemplate, DecodeSession, DecodeSessionOptions, Environment, ModelDirectory,
    OrtError, Result, Session, SessionOptions, Tokenizer, Value, ep_selection,
};

#[derive(Debug)]
struct Args {
    model: PathBuf,
    prompt: String,
    prompt_tokens: Option<Vec<i64>>,
    pad_to: Option<usize>,
    max_tokens: usize,
    mode: Mode,
    warmup: bool,
    chat: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Cpu,
    Metal,
    Hybrid,
    All,
}

/// One end-to-end run's measurements.
struct RunReport {
    label: &'static str,
    prompt_len: usize,
    ttft: Duration,
    handoff: Option<Duration>,
    decode_total: Duration,
    decode_tokens: usize,
    tokens: Vec<i64>,
    text: String,
}

impl RunReport {
    fn decode_tok_per_s(&self) -> f64 {
        if self.decode_total.as_secs_f64() == 0.0 {
            return f64::NAN;
        }
        self.decode_tokens as f64 / self.decode_total.as_secs_f64()
    }

    fn total(&self) -> Duration {
        self.ttft + self.handoff.unwrap_or_default() + self.decode_total
    }

    fn print(&self) {
        let ttft_ms = self.ttft.as_secs_f64() * 1000.0;
        let total_ms = self.total().as_secs_f64() * 1000.0;
        println!("── {} ──", self.label);
        println!("  prompt_len      : {}", self.prompt_len);
        println!("  prefill/TTFT    : {ttft_ms:.1} ms");
        if let Some(handoff) = self.handoff {
            println!(
                "  kv_handoff      : {:.2} ms",
                handoff.as_secs_f64() * 1000.0
            );
        }
        println!(
            "  decode          : {} tokens, {:.1} tok/s ({:.2} ms/tok)",
            self.decode_tokens,
            self.decode_tok_per_s(),
            self.decode_total.as_secs_f64() * 1000.0 / self.decode_tokens.max(1) as f64,
        );
        println!("  end-to-end      : {total_ms:.1} ms");
        let preview = self.tokens.iter().take(24).collect::<Vec<_>>();
        println!("  first_tokens    : {preview:?}");
        println!("  text            : {:?}", truncate(&self.text, 160));
    }
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let env = Environment::new("hybrid-prefill-decode")?;
    let dir = ModelDirectory::load(&args.model)?;
    let tokenizer = Tokenizer::from_file(&dir.tokenizer_path)?;

    let prompt_tokens = build_prompt(&tokenizer, &dir, &args)?;
    println!(
        "model={} prompt_len={} decode_tokens={} mode={:?}",
        dir.model_path.display(),
        prompt_tokens.len(),
        args.max_tokens,
        args.mode
    );

    // CPU session is always needed (decode side of hybrid + pure-CPU baseline).
    let cpu_session = Session::new(
        &env,
        &dir.model_path,
        SessionOptions::with_execution_provider(ep_selection("cpu")),
    )?;

    // Metal session is optional: if the plugin is not configured we skip the
    // Metal-dependent runs and report why, instead of aborting.
    let metal_session = match Session::new(
        &env,
        &dir.model_path,
        SessionOptions::with_execution_provider(ep_selection("metal")),
    ) {
        Ok(session) => Some(session),
        Err(err) => {
            eprintln!(
                "warning: Metal EP session unavailable ({err}); running CPU-only. \
                 Set ONNX_GENAI_METAL_EP_LIB to the built plugin to enable Metal/hybrid."
            );
            None
        }
    };

    let want_cpu = matches!(args.mode, Mode::Cpu | Mode::All);
    let want_metal = matches!(args.mode, Mode::Metal | Mode::All) && metal_session.is_some();
    let want_hybrid = matches!(args.mode, Mode::Hybrid | Mode::All) && metal_session.is_some();

    if args.warmup {
        // Warm both sessions (shader compile, weight upload, thread pools) so the
        // measured TTFT is steady-state rather than one-time initialization.
        run_pure(&cpu_session, &tokenizer, &prompt_tokens, 2, "warmup-cpu")?;
        if let Some(metal) = metal_session.as_ref() {
            run_pure(metal, &tokenizer, &prompt_tokens, 2, "warmup-metal")?;
        }
    }

    let mut reports = Vec::new();
    let mut cpu_reference: Option<RunReport> = None;

    if want_cpu {
        let report = run_pure(
            &cpu_session,
            &tokenizer,
            &prompt_tokens,
            args.max_tokens,
            "pure-cpu",
        )?;
        cpu_reference = Some(clone_stream(&report));
        reports.push(report);
    }
    if want_metal {
        let metal = metal_session.as_ref().unwrap();
        let report = run_pure(
            metal,
            &tokenizer,
            &prompt_tokens,
            args.max_tokens,
            "pure-metal",
        )?;
        reports.push(report);
    }
    if want_hybrid {
        let metal = metal_session.as_ref().unwrap();
        let report = run_hybrid(
            metal,
            &cpu_session,
            &tokenizer,
            &prompt_tokens,
            args.max_tokens,
        )?;
        reports.push(report);
    }

    println!("\n=== RESULTS ===");
    for report in &reports {
        report.print();
    }

    // Coherence: compare each run against the pure-CPU stream (the correctness
    // reference for the hybrid: decode is on CPU, only prefill moved to GPU).
    if let Some(reference) = cpu_reference.as_ref() {
        println!("\n=== COHERENCE (vs pure-cpu) ===");
        for report in &reports {
            if report.label == "pure-cpu" {
                continue;
            }
            let divergence = first_divergence(&reference.tokens, &report.tokens);
            match divergence {
                None => println!(
                    "  {:<11}: IDENTICAL to pure-cpu for all {} tokens",
                    report.label,
                    report.tokens.len()
                ),
                Some(index) => println!(
                    "  {:<11}: matches pure-cpu for first {} tokens, diverges at token {}",
                    report.label, index, index
                ),
            }
        }
    }

    Ok(())
}

/// Run a full prefill + greedy decode on a single session/EP.
fn run_pure(
    session: &Session,
    tokenizer: &Tokenizer,
    prompt: &[i64],
    max_tokens: usize,
    label: &'static str,
) -> Result<RunReport> {
    let mut decode = DecodeSession::new(session, DecodeSessionOptions::default())?;

    // Prefill: whole prompt in one forward pass. TTFT = time to the first token.
    let (ttft, first_token) = timed_prefill(&mut decode, prompt)?;

    let mut tokens = vec![first_token];
    let decode_started = Instant::now();
    greedy_decode(&mut decode, &mut tokens, prompt.len(), max_tokens)?;
    let decode_total = decode_started.elapsed();

    Ok(RunReport {
        label,
        prompt_len: prompt.len(),
        ttft,
        handoff: None,
        decode_total,
        decode_tokens: tokens.len(),
        text: detokenize(tokenizer, &tokens),
        tokens,
    })
}

/// Hybrid: prefill on `metal`, hand the KV to `cpu`, decode on `cpu`.
fn run_hybrid(
    metal: &Session,
    cpu: &Session,
    tokenizer: &Tokenizer,
    prompt: &[i64],
    max_tokens: usize,
) -> Result<RunReport> {
    let mut prefill = DecodeSession::new(metal, DecodeSessionOptions::default())?;
    let (ttft, first_token) = timed_prefill(&mut prefill, prompt)?;

    // KV handoff: export the Metal-produced present KV as host-owned tensors and
    // adopt them into the CPU decode session at the prompt length.
    let handoff_started = Instant::now();
    let kv = prefill.export_kv()?;
    let mut decode = DecodeSession::new(cpu, DecodeSessionOptions::default())?;
    decode.import_kv(prompt.len(), kv)?;
    let handoff = handoff_started.elapsed();

    let mut tokens = vec![first_token];
    let decode_started = Instant::now();
    greedy_decode(&mut decode, &mut tokens, prompt.len(), max_tokens)?;
    let decode_total = decode_started.elapsed();

    Ok(RunReport {
        label: "hybrid",
        prompt_len: prompt.len(),
        ttft,
        handoff: Some(handoff),
        decode_total,
        decode_tokens: tokens.len(),
        text: detokenize(tokenizer, &tokens),
        tokens,
    })
}

/// Run the prefill step and return (TTFT, first generated token).
fn timed_prefill(decode: &mut DecodeSession, prompt: &[i64]) -> Result<(Duration, i64)> {
    let mask = vec![1_i64; prompt.len()];
    let position_ids = (0..prompt.len() as i64).collect::<Vec<_>>();
    let started = Instant::now();
    let logits = decode.step(prompt, &mask, &position_ids)?;
    let first_token = argmax_last(&logits, prompt.len())?;
    Ok((started.elapsed(), first_token))
}

/// Greedy single-token decode loop; appends generated tokens to `tokens`.
fn greedy_decode(
    decode: &mut DecodeSession,
    tokens: &mut Vec<i64>,
    prompt_len: usize,
    max_tokens: usize,
) -> Result<()> {
    let mut token = *tokens.last().expect("first token present");
    for offset in 0..max_tokens.saturating_sub(1) {
        // KV length before this step: prompt + tokens already committed to KV.
        // `token` (produced by prefill/prev step) has not yet been fed, so it is
        // placed at absolute position `past_len`.
        let past_len = prompt_len + offset;
        let mask = vec![1_i64; past_len + 1];
        let logits = decode.step(&[token], &mask, &[past_len as i64])?;
        token = argmax_last(&logits, 1)?;
        tokens.push(token);
    }
    Ok(())
}

/// Argmax over the vocabulary of the last sequence position of `[1, S, V]` logits.
fn argmax_last(logits: &Value, seq_len: usize) -> Result<i64> {
    let data = logits.to_vec_f32()?;
    if data.is_empty() || seq_len == 0 {
        return Err(invalid("empty logits"));
    }
    let vocab = data.len() / seq_len;
    let last = &data[(seq_len - 1) * vocab..seq_len * vocab];
    let index = last
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .ok_or_else(|| invalid("empty logits row"))?;
    Ok(index as i64)
}

fn build_prompt(tokenizer: &Tokenizer, dir: &ModelDirectory, args: &Args) -> Result<Vec<i64>> {
    // Explicit prompt token ids bypass tokenization/chat-templating entirely.
    // This is the reproducible path: pass the exact ChatML-encoded ids so greedy
    // output is a recognizable sentence (e.g. the Metal EP e2e fixture yields
    // "The capital of France is Paris.").
    let mut tokens = if let Some(ids) = args.prompt_tokens.clone() {
        ids
    } else if args.chat {
        let template = ChatTemplate::from_model_dir(&dir.root)?;
        let text = template.render(&[ChatMessage::user(args.prompt.clone())], None, true)?;
        tokenizer.encode_i64(&text)?
    } else {
        tokenizer.encode_i64(&args.prompt)?
    };
    if tokens.is_empty() {
        return Err(invalid("prompt tokenized to zero tokens"));
    }
    if let Some(target) = args.pad_to {
        if target == 0 {
            return Err(invalid("--pad-to must be positive"));
        }
        // Lengthen a short prompt into a long-context prompt by repeating its
        // tokens (long-prompt TTFT is the metric under test). Truncate if longer.
        let base = tokens.clone();
        while tokens.len() < target {
            let remaining = target - tokens.len();
            tokens.extend(base.iter().take(remaining).copied());
        }
        tokens.truncate(target);
    }
    Ok(tokens)
}

fn detokenize(tokenizer: &Tokenizer, tokens: &[i64]) -> String {
    tokenizer
        .decode_i64(tokens)
        .unwrap_or_else(|_| "<detokenize failed>".to_string())
}

fn first_divergence(a: &[i64], b: &[i64]) -> Option<usize> {
    let len = a.len().min(b.len());
    (0..len).find(|&i| a[i] != b[i])
}

fn clone_stream(report: &RunReport) -> RunReport {
    RunReport {
        label: report.label,
        prompt_len: report.prompt_len,
        ttft: report.ttft,
        handoff: report.handoff,
        decode_total: report.decode_total,
        decode_tokens: report.decode_tokens,
        tokens: report.tokens.clone(),
        text: report.text.clone(),
    }
}

fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut out = text.chars().take(max).collect::<String>();
    out.push('…');
    out
}

fn parse_args() -> Result<Args> {
    let mut model = None;
    let mut prompt = "What is the capital of France?".to_string();
    let mut prompt_tokens = None;
    let mut pad_to = None;
    let mut max_tokens = 32;
    let mut mode = Mode::All;
    let mut warmup = true;
    let mut chat = false;

    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--model" => model = Some(PathBuf::from(next_value(&mut iter, "--model")?)),
            "--prompt" => prompt = next_value(&mut iter, "--prompt")?,
            "--prompt-tokens" => {
                let raw = next_value(&mut iter, "--prompt-tokens")?;
                let ids = raw
                    .split_whitespace()
                    .map(|part| {
                        part.parse::<i64>()
                            .map_err(|_| invalid(format!("invalid prompt token id '{part}'")))
                    })
                    .collect::<Result<Vec<_>>>()?;
                if ids.is_empty() {
                    return Err(invalid("--prompt-tokens requires at least one id"));
                }
                prompt_tokens = Some(ids);
            }
            "--prompt-tokens-file" => {
                let path = next_value(&mut iter, "--prompt-tokens-file")?;
                let raw = std::fs::read_to_string(&path)
                    .map_err(|err| invalid(format!("cannot read {path}: {err}")))?;
                let ids = raw
                    .split_whitespace()
                    .map(|part| {
                        part.parse::<i64>()
                            .map_err(|_| invalid(format!("invalid prompt token id '{part}'")))
                    })
                    .collect::<Result<Vec<_>>>()?;
                if ids.is_empty() {
                    return Err(invalid("--prompt-tokens-file is empty"));
                }
                prompt_tokens = Some(ids);
            }
            "--pad-to" => pad_to = Some(parse_usize(&mut iter, "--pad-to")?),
            "--max-tokens" => max_tokens = parse_usize(&mut iter, "--max-tokens")?,
            "--chat" => chat = true,
            "--mode" => {
                mode = match next_value(&mut iter, "--mode")?.as_str() {
                    "cpu" => Mode::Cpu,
                    "metal" => Mode::Metal,
                    "hybrid" => Mode::Hybrid,
                    "all" => Mode::All,
                    other => return Err(invalid(format!("invalid --mode '{other}'"))),
                };
            }
            "--no-warmup" => warmup = false,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(invalid(format!("unknown argument '{other}'"))),
        }
    }

    Ok(Args {
        model: model.ok_or_else(|| invalid("--model is required"))?,
        prompt,
        prompt_tokens,
        pad_to,
        max_tokens: max_tokens.max(1),
        mode,
        warmup,
        chat,
    })
}

fn print_help() {
    println!(
        "Usage: cargo run -p onnx-genai-ort --release --example hybrid_prefill_decode -- \\\n  \
         --model <DIR> [--prompt TEXT] [--pad-to N] [--max-tokens N] \\\n  \
         [--mode cpu|metal|hybrid|all] [--no-warmup]\n\n\
         Set ONNX_GENAI_METAL_EP_LIB to the built libonnxruntime_mlx_ep.dylib for Metal/hybrid."
    );
}

fn next_value(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    iter.next()
        .ok_or_else(|| invalid(format!("{flag} requires a value")))
}

fn parse_usize(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<usize> {
    next_value(iter, flag)?
        .parse()
        .map_err(|_| invalid(format!("{flag} requires a non-negative integer")))
}

fn invalid(message: impl Into<String>) -> OrtError {
    OrtError::InvalidArgument(message.into())
}
