//! Diagnostic probe (not an assertion lock) for the opt-in fp16-fused decode
//! path (`DecodePrecision::Fp16`). For the model in `PHI35_MINI_E2E_DIR` it runs
//! three 64-token greedy decodes — ORT fp32, native `DecodePrecision::Model`
//! (default), native `DecodePrecision::Fp16` — and reports each token stream,
//! decode throughput, and the leading-match counts. It never fails; the parity
//! assertion lives in `phi35_mini_fp16_decode_lock.rs`.
//!
//! Point `PHI35_MINI_E2E_DIR` at any model to reuse it: for an fp16-activation
//! model (e.g. Qwen2.5-0.5b), native Model and native Fp16 produce identical
//! streams because the fp16-fingerprint gate makes `Fp16` a no-op — the
//! default-path-unchanged evidence.
//!
//! ```bash
//! PHI35_MINI_E2E_DIR=~/.foundry/cache/models/Microsoft/Phi-3.5-mini-instruct-generic-cpu-2/v2 \
//!   cargo test -p onnx-genai-engine --features cuda,native-backend \
//!   --test phi35_fp16_decode_parity -- --ignored --nocapture
//! # Select a single config in a fresh process: PARITY_ONE=ort|native_model|native_fp16
//! ```
#![cfg(all(feature = "cuda", feature = "native-backend"))]

use onnx_genai_engine::{
    DecodePrecision, Engine, EngineConfig, EngineDecodeBackend, GeneratePrompt, GenerateRequest,
    NativeDecodeDevice,
};

const DEFAULT_MODEL_DIR: &str =
    "/home/justinchu/.foundry/cache/models/Microsoft/Phi-3.5-mini-instruct-generic-cpu-2/v2";

const PROMPT: &str = "Hello";
const HORIZON: usize = 64;

struct Run {
    tokens: Vec<u32>,
    decode_secs: f64,
}

fn generate(
    model_dir: &std::path::Path,
    backend: EngineDecodeBackend,
    native_device: Option<NativeDecodeDevice>,
    decode_precision: DecodePrecision,
) -> anyhow::Result<Run> {
    let mut engine = Engine::from_dir(
        model_dir,
        EngineConfig {
            decode_backend: backend,
            native_device,
            decode_precision,
            ..EngineConfig::default()
        },
    )?;
    let mut request = GenerateRequest::new(GeneratePrompt::Text(PROMPT.to_string()));
    request.options.max_new_tokens = HORIZON;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    let t0 = std::time::Instant::now();
    let result = engine.generate(request)?;
    let decode_secs = t0.elapsed().as_secs_f64();
    Ok(Run {
        tokens: result.token_ids,
        decode_secs,
    })
}

fn leading_match(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

fn model_dir() -> Option<std::path::PathBuf> {
    let dir = std::env::var_os("PHI35_MINI_E2E_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(DEFAULT_MODEL_DIR));
    if !dir.is_dir() {
        eprintln!("skipping fp16 decode probe: model dir absent: {}", dir.display());
        return None;
    }
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping fp16 decode probe: CUDA unavailable: {error}");
        return None;
    }
    unsafe { std::env::set_var("ONNX_GENAI_EP", "cuda") };
    Some(dir)
}

fn cuda(device: Option<u32>) -> Option<NativeDecodeDevice> {
    Some(NativeDecodeDevice::Cuda { index: device })
}

fn report(label: &str, run: &Run) {
    eprintln!(
        "[{label}] {HORIZON} tokens in {:.3}s => {:.2} tok/s (incl. prefill; host may be contended)",
        run.decode_secs,
        HORIZON as f64 / run.decode_secs
    );
    eprintln!("[{label}] tokens: {:?}", run.tokens);
}

#[test]
#[ignore = "requires the real Phi-3.5-mini int4 model + CUDA; run with --ignored --nocapture"]
fn phi35_fp16_decode_parity_report() -> anyhow::Result<()> {
    let Some(dir) = model_dir() else {
        return Ok(());
    };

    let ort = generate(&dir, EngineDecodeBackend::Ort, None, DecodePrecision::Model)?;
    let native_model = generate(
        &dir,
        EngineDecodeBackend::Native,
        cuda(Some(0)),
        DecodePrecision::Model,
    )?;
    let native_fp16 = generate(
        &dir,
        EngineDecodeBackend::Native,
        cuda(Some(0)),
        DecodePrecision::Fp16,
    )?;

    eprintln!("\n==== fp16 decode parity report ({HORIZON}-token greedy, prompt {PROMPT:?}) ====");
    report("ort-fp32", &ort);
    report("native-model", &native_model);
    report("native-fp16", &native_fp16);
    eprintln!(
        "leading match native-fp16 vs ort-fp32: {}/{HORIZON}",
        leading_match(&native_fp16.tokens, &ort.tokens)
    );
    eprintln!(
        "leading match native-model vs ort-fp32: {}/{HORIZON}",
        leading_match(&native_model.tokens, &ort.tokens)
    );
    if native_fp16.decode_secs > 0.0 {
        eprintln!(
            "native fp16 vs native model speedup: {:.2}x",
            native_model.decode_secs / native_fp16.decode_secs
        );
    }
    eprintln!("=================================================================\n");
    Ok(())
}

/// Isolated single-config run (fresh process) selected by
/// `PARITY_ONE=ort|native_model|native_fp16`, to rule out in-process
/// cross-session interference and to time one path at a time.
#[test]
#[ignore = "requires the real Phi-3.5-mini int4 model + CUDA; select via PARITY_ONE"]
fn phi35_fp16_decode_parity_isolated() -> anyhow::Result<()> {
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    let mode = std::env::var("PARITY_ONE").unwrap_or_else(|_| "native_fp16".to_string());
    let run = match mode.as_str() {
        "ort" => generate(&dir, EngineDecodeBackend::Ort, None, DecodePrecision::Model)?,
        "native_model" => generate(
            &dir,
            EngineDecodeBackend::Native,
            cuda(Some(0)),
            DecodePrecision::Model,
        )?,
        "native_fp16" => generate(
            &dir,
            EngineDecodeBackend::Native,
            cuda(Some(0)),
            DecodePrecision::Fp16,
        )?,
        other => anyhow::bail!("unknown PARITY_ONE={other}"),
    };
    eprintln!("\n==== ISOLATED [{mode}] ({HORIZON}-token greedy, prompt {PROMPT:?}) ====");
    report(&mode, &run);
    eprintln!("=====================================================\n");
    Ok(())
}
