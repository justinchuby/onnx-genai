//! Direct-engine decode profiler.
//!
//! Loads a model with the real `onnx-genai` engine (no HTTP/SSE), runs a fixed
//! number of decode steps, and prints the env-gated per-stage profiler report.
//! This isolates the per-token decode cost inside our runtime so it can be
//! attributed to ORT kernel time (`ort.session_run`) versus our orchestration
//! (tensor binding, KV rotation, logits copy, sampling, detokenization).
//!
//! Usage:
//!   ONNX_GENAI_PROFILE=1 cargo run --release -p onnx-genai-bench \
//!     --features bench-ort --bin profile_decode -- \
//!     --model models/qwen2.5-0.5b-q4-onnx-fixed --tokens 128 [--threads N] \
//!     [--prompt "..."] [--warmups 1] [--runs 1] [--raw]
//!
//! By default the `--prompt` is wrapped as a single user turn and rendered
//! through the model's chat template (same path the server uses), so the
//! measured prompt matches real serving and is comparable to onnxruntime-genai
//! run with `apply_chat_template`. Pass `--raw` to feed the prompt untemplated.

use std::path::{Path, PathBuf};
use std::time::Instant;

use onnx_genai_engine::{Engine, EngineConfig, GenerateRequest};
use onnx_genai_ort::{ChatMessage, ChatTemplate, SessionOptions, profile};

struct Args {
    model: PathBuf,
    tokens: usize,
    threads: Option<i32>,
    prompt: String,
    warmups: usize,
    runs: usize,
    raw: bool,
}

fn parse_args() -> Args {
    let mut model = PathBuf::from("models/qwen2.5-0.5b-q4-onnx-fixed");
    let mut tokens = 128usize;
    let mut threads: Option<i32> = None;
    let mut prompt = String::from("Write a short paragraph about the history of computing.");
    let mut warmups = 1usize;
    let mut runs = 1usize;
    let mut raw = false;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => model = PathBuf::from(it.next().expect("--model needs a value")),
            "--tokens" => tokens = it.next().and_then(|v| v.parse().ok()).expect("--tokens N"),
            "--threads" => {
                threads = Some(it.next().and_then(|v| v.parse().ok()).expect("--threads N"));
            }
            "--prompt" => prompt = it.next().expect("--prompt needs a value"),
            "--warmups" => warmups = it.next().and_then(|v| v.parse().ok()).expect("--warmups N"),
            "--runs" => runs = it.next().and_then(|v| v.parse().ok()).expect("--runs N"),
            "--raw" => raw = true,
            other => panic!("unknown arg: {other}"),
        }
    }

    Args {
        model,
        tokens,
        threads,
        prompt,
        warmups,
        runs,
        raw,
    }
}

/// Load the model's chat template only when the directory actually ships one
/// (standalone `chat_template.jinja` or a `chat_template` key in
/// `tokenizer_config.json`) — mirrors the server's `load_chat_template` so the
/// profiler never falls back to the generic built-in template silently.
fn load_real_chat_template(model_dir: &Path) -> Option<ChatTemplate> {
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
    if has_template {
        ChatTemplate::from_model_dir(model_dir).ok()
    } else {
        None
    }
}

/// Build the prompt actually fed to the engine. Unless `--raw` was given, the
/// `prompt` is wrapped as one user turn and rendered through the model's chat
/// template with `add_generation_prompt=true`, matching the server path.
fn resolve_prompt(args: &Args) -> String {
    if args.raw {
        println!("prompt: raw (chat template NOT applied; --raw)");
        return args.prompt.clone();
    }
    match load_real_chat_template(&args.model) {
        Some(template) => {
            let messages = [ChatMessage::new("user", args.prompt.clone())];
            match template.render(&messages, None, true) {
                Ok(rendered) => {
                    println!("prompt: chat-templated ({} chars)", rendered.len());
                    rendered
                }
                Err(err) => {
                    println!("prompt: chat template render failed ({err}); using raw prompt");
                    args.prompt.clone()
                }
            }
        }
        None => {
            println!("prompt: no chat template in model dir; using raw prompt");
            args.prompt.clone()
        }
    }
}

fn build_engine(args: &Args) -> Engine {
    match args.threads {
        Some(n) => Engine::from_dir_with_session_options(
            &args.model,
            EngineConfig::default(),
            SessionOptions::default().with_intra_op_threads(n),
        ),
        None => Engine::from_dir(&args.model, EngineConfig::default()),
    }
    .expect("model must load")
}

fn request(prompt: &str, tokens: usize) -> GenerateRequest {
    let mut request = GenerateRequest::new(prompt.to_string());
    request.options.max_new_tokens = tokens;
    request.options.temperature = 0.0;
    request.options.stop_on_eos = false;
    request
}

fn main() {
    let args = parse_args();
    println!(
        "profile_decode: model={} tokens={} threads={:?} warmups={} runs={} profile_enabled={}",
        args.model.display(),
        args.tokens,
        args.threads,
        args.warmups,
        args.runs,
        profile::enabled()
    );

    let mut engine = build_engine(&args);
    let prompt = resolve_prompt(&args);

    for _ in 0..args.warmups {
        let result = engine
            .generate(request(&prompt, args.tokens))
            .expect("warmup generate");
        std::hint::black_box(&result);
    }

    // Discard warmup measurements; only the measured runs count.
    profile::reset();

    let mut total_tokens = 0u64;
    let mut last_text = String::new();
    let start = Instant::now();
    for _ in 0..args.runs {
        let result = engine
            .generate(request(&prompt, args.tokens))
            .expect("measured generate");
        total_tokens += result.token_ids.len() as u64;
        last_text = result.text.clone();
        std::hint::black_box(&result);
    }
    let elapsed = start.elapsed();

    let per_token_us = (elapsed.as_secs_f64() * 1_000_000.0) / total_tokens.max(1) as f64;
    let tok_per_s = total_tokens as f64 / elapsed.as_secs_f64();
    println!(
        "\nwall: {:.3} ms over {} tokens ({} run(s)) -> {:.2} tok/s, {:.2} us/token\n",
        elapsed.as_secs_f64() * 1000.0,
        total_tokens,
        args.runs,
        tok_per_s,
        per_token_us
    );
    println!("--- generated text (coherence check) ---\n{last_text}\n---");

    if profile::enabled() {
        println!("{}", profile::report(total_tokens));
    } else {
        println!("(set ONNX_GENAI_PROFILE=1 for the per-stage breakdown)");
    }

    if profile::tracing_enabled() {
        match profile::write_trace() {
            Ok(()) => println!(
                "wrote Perfetto timeline to {} (open at https://ui.perfetto.dev)",
                onnx_genai_runtime_config::runtime_config()
                    .trace
                    .as_deref()
                    .map_or_else(String::new, |path| path.display().to_string())
            ),
            Err(err) => eprintln!("failed to write trace: {err}"),
        }
    }
}
