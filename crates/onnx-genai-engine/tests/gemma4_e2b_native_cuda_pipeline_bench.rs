//! Inc3c perf measurement harness (issue #384): quantify the eager/uncaptured
//! per-step cost of the native CUDA decoder against the graph-captured ceiling
//! and the ORT-CUDA reference bar.
//!
//! Two controlled experiments, mode-selected by `GEMMA4_BENCH_MODE` (each mode
//! runs in its own process — ORT env/EP + lib are process-global one-shot init,
//! so mixing native and ORT-CUDA in one process is unsafe):
//!
//! Single-graph Qwen3-0.6B (real mask-consuming decoder; isolates the pure
//! capture-vs-eager launch-overhead lever — identical graph, only capture
//! toggled):
//!   - `qwen3-captured` : native decoder, CUDA EP, CUDA-graph capture ON
//!   - `qwen3-eager`    : native decoder, CUDA EP, capture OFF (`ONNX_GENAI_CUDA_GRAPH=0`)
//!   - `qwen3-ort`      : ORT decoder (set `ONNX_GENAI_EP=cuda` for the CUDA bar)
//!
//! Gemma 3n E2B multi-component pipeline (the *actual* eager inputs_embeds +
//! routed `per_layer_inputs` path under Inc3c):
//!   - `gemma-native-cuda` / `gemma-native-cpu` / `gemma-ort`
//!
//! NOTE: the gemma modes are currently blocked by an unrelated pipeline
//! limitation — gemma-3n's audio `input_features_mask` is `Bool`, and the value/
//! cache path errors `unsupported cached ORT value dtype: Bool`, so a direct
//! captured-vs-eager gemma tok/s number cannot be produced yet. The qwen3 modes
//! (a real mask-consuming decoder) are the working capture-vs-eager measurement
//! and gave the Part A numbers (captured 612 / eager 220 / ORT 443 tok/s).
//!
//! Steady-state decode tok/s uses the two-length method
//! (`tok/s = (N2 - N1) / (t2 - t1)`), cancelling prefill + fixed per-call
//! overhead. The engine is built ONCE and reused across timed generations.
//!
//! ```bash
//! source .cudaenv.sh
//! export CUDA_VISIBLE_DEVICES=4
//! export ONNX_GENAI_ORT_LIB=$ORT_ROOT/lib/libonnxruntime.so.1.27.0
//! GEMMA4_BENCH_MODE=qwen3-captured cargo test -p onnx-genai-engine \
//!   --features cuda,native-backend --test gemma4_e2b_native_cuda_pipeline_bench \
//!   -- --ignored --nocapture
//! ```
#![cfg(all(feature = "cuda", feature = "native-backend"))]

use std::path::PathBuf;
use std::time::Instant;

use onnx_genai_engine::pipeline::PipelineGenerateRequest;
use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GenerateOptions, GeneratePrompt, GenerateRequest,
    NativeDecodeDevice, PipelineEngine,
};
use onnx_genai_ort::{DataType, Value};

const QWEN3_DIR: &str = "/home/justinchu/mobius/.scratch/qwen3-0.6b-int4-cuda-postfix";
const GEMMA_DIR: &str = "/home/justinchu/mobius/.scratch/gemma4-e2b-native";
const NATIVE_DECODER_ENV: &str = "ONNX_GENAI_PIPELINE_NATIVE_DECODER";
const NATIVE_DECODER_DEVICE_ENV: &str = "ONNX_GENAI_PIPELINE_NATIVE_DECODER_DEVICE";

const N1: usize = 32;
const N2: usize = 160;
const REPS: usize = 3;
/// A short deterministic text prompt (Qwen3 token ids for "The capital of").
const QWEN3_PROMPT: &[u32] = &[785, 6722, 315];
/// Gemma text prompt: BOS + a few tokens; text-only decode.
const GEMMA_PROMPT: &[u32] = &[2, 651, 6996, 576, 8698, 603];

fn dir_if_present(path: &str) -> Option<PathBuf> {
    let dir = PathBuf::from(path);
    if dir.join("inference_metadata.yaml").is_file() {
        Some(dir)
    } else {
        eprintln!("skipping bench: {path} has no inference_metadata.yaml");
        None
    }
}

/// Report steady-state decode tok/s for a `run(new_tokens) -> seconds` closure.
fn measure(mode: &str, mut run: impl FnMut(usize) -> anyhow::Result<f64>) -> anyhow::Result<()> {
    let n1 = env_usize("BENCH_N1", N1);
    let n2 = env_usize("BENCH_N2", N2);
    let reps = env_usize("BENCH_REPS", REPS);
    let _ = run(n1)?; // warm-up (JIT, cuDNN algo pick, capture).
    let mut best_short = f64::MAX;
    let mut best_long = f64::MAX;
    for _ in 0..reps {
        best_short = best_short.min(run(n1)?);
        best_long = best_long.min(run(n2)?);
    }
    let decode_tok_s = (n2 - n1) as f64 / (best_long - best_short);
    let full_tok_s = n2 as f64 / best_long;
    eprintln!("=== decode bench: mode={mode} ===");
    eprintln!("  N1={n1} best={best_short:.4}s   N2={n2} best={best_long:.4}s");
    eprintln!("  steady-state decode tok/s (two-length) = {decode_tok_s:.2}");
    eprintln!("  full-run tok/s (incl prefill)          = {full_tok_s:.2}");
    Ok(())
}

fn qwen3_config(backend: EngineDecodeBackend) -> EngineConfig {
    EngineConfig {
        decode_backend: backend,
        native_device: match backend {
            EngineDecodeBackend::Native => Some(NativeDecodeDevice::Cuda { index: Some(0) }),
            _ => None,
        },
        ..EngineConfig::default()
    }
}

fn qwen3_run(engine: &mut Engine, new_tokens: usize) -> anyhow::Result<f64> {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(QWEN3_PROMPT.to_vec()));
    request.options = GenerateOptions {
        max_new_tokens: new_tokens,
        temperature: 0.0,
        greedy: true,
        stop_on_eos: false,
        ..GenerateOptions::default()
    };
    let start = Instant::now();
    let result = engine.generate(request)?;
    let elapsed = start.elapsed().as_secs_f64();
    assert_eq!(result.token_ids.len(), new_tokens, "early stop");
    Ok(elapsed)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(default)
}

/// Minimal dummy audio inputs so the Gemma 3n E2B pipeline (which declares the
/// audio encoder `input_features` as a required request input) runs text-only:
/// a short zero spectrogram with an all-false validity mask contributes no audio
/// tokens, leaving the decode driven by the text embedding component.
fn dummy_audio() -> anyhow::Result<(Value, Value)> {
    const TIME: usize = 4;
    const MELS: usize = 128;
    let features =
        Value::from_vec_f16_bits(vec![0u16; TIME * MELS], &[1, TIME as i64, MELS as i64])?;
    let mask = Value::from_raw_bytes(vec![0u8; TIME], &[1, TIME as i64], DataType::Bool)?;
    Ok((features, mask))
}

fn gemma_run(engine: &mut PipelineEngine, new_tokens: usize) -> anyhow::Result<f64> {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(GEMMA_PROMPT.to_vec()));
    request.options = GenerateOptions {
        max_new_tokens: new_tokens,
        temperature: 0.0,
        greedy: true,
        stop_on_eos: false,
        ..GenerateOptions::default()
    };
    let (features, mask) = dummy_audio()?;
    let pipeline_request = PipelineGenerateRequest::new(request)
        .with_input("audio_encoder.input_features", features)
        .with_input("audio_encoder.input_features_mask", mask);
    let start = Instant::now();
    let result = engine.generate_with_pipeline_request(pipeline_request)?;
    let elapsed = start.elapsed().as_secs_f64();
    assert_eq!(result.token_ids.len(), new_tokens, "early stop");
    Ok(elapsed)
}

#[test]
#[ignore = "manual perf measurement; requires the real exports and a CUDA device"]
fn native_cuda_decode_toks() -> anyhow::Result<()> {
    let mode = std::env::var("GEMMA4_BENCH_MODE").unwrap_or_else(|_| "qwen3-captured".to_string());

    match mode.as_str() {
        "qwen3-captured" | "qwen3-eager" => {
            let Some(dir) = dir_if_present(QWEN3_DIR) else {
                return Ok(());
            };
            if mode == "qwen3-captured" {
                unsafe { std::env::set_var("ONNX_GENAI_CUDA_GRAPH", "1") };
            } else {
                unsafe { std::env::set_var("ONNX_GENAI_CUDA_GRAPH", "0") };
            }
            let mut engine = Engine::from_dir(&dir, qwen3_config(EngineDecodeBackend::Native))?;
            measure(&mode, |n| qwen3_run(&mut engine, n))?;
        }
        "qwen3-ort" => {
            let Some(dir) = dir_if_present(QWEN3_DIR) else {
                return Ok(());
            };
            let mut engine = Engine::from_dir(&dir, qwen3_config(EngineDecodeBackend::Ort))?;
            measure(&mode, |n| qwen3_run(&mut engine, n))?;
        }
        "gemma-native-cuda" | "gemma-native-cpu" => {
            let Some(dir) = dir_if_present(GEMMA_DIR) else {
                return Ok(());
            };
            let device = if mode == "gemma-native-cuda" {
                "cuda:0"
            } else {
                "cpu"
            };
            unsafe {
                std::env::set_var(NATIVE_DECODER_ENV, "decoder");
                std::env::set_var(NATIVE_DECODER_DEVICE_ENV, device);
            }
            let mut engine = Engine::from_pipeline_dir(
                &dir,
                EngineConfig {
                    pipeline_cache_bytes: 0,
                    ..EngineConfig::default()
                },
            )?;
            measure(&mode, |n| gemma_run(&mut engine, n))?;
        }
        "gemma-ort" => {
            let Some(dir) = dir_if_present(GEMMA_DIR) else {
                return Ok(());
            };
            let mut engine = Engine::from_pipeline_dir(
                &dir,
                EngineConfig {
                    pipeline_cache_bytes: 0,
                    ..EngineConfig::default()
                },
            )?;
            measure(&mode, |n| gemma_run(&mut engine, n))?;
        }
        other => anyhow::bail!("unknown GEMMA4_BENCH_MODE={other}"),
    }
    Ok(())
}
