//! Qwen2.5-1.5B int4 native-CUDA accuracy locks.
//!
//! For the raw prompt `"The capital of France is"`, native and ORT CUDA first
//! diverge after the common generated prefix ending in `"France."`: native
//! selects token 576 (`" The"`), while the deployed ORT CUDA path selects 15920
//! (`" Which"`). Rewriting every deployed MatMulNBits to `accuracy_level=1`
//! makes the ORT CPU fp32-activation oracle select 576, adjudicating native as
//! the more accurate backend at the first divergence.
//!
//! The Foundry steady-decode sweep's default prompt (`"Hello"`, token `9707`)
//! has the same verdict. Native and ORT CUDA agree for 26 generated tokens, then
//! split at decode index 26: native selects token 1909 (`" top"`), while ORT CUDA
//! selects token 821 (`" data"`). The accuracy-level-1 ORT CPU oracle also
//! selects 1909 with token 821 in the top set, so this is another acc-4
//! int8-activation near tie where native stays on the fp32-correct side. Asking
//! the engine for `top_logprobs` intentionally bypasses the device greedy fast
//! path and makes ORT's returned logits/host argmax select the oracle token too;
//! the measured split is therefore in the default greedy decode path, not in the
//! fp32 oracle or host-logit view.
//!
//! Build the oracle and run this real-model lock with:
//!
//! ```bash
//! python3 scripts/qwen_q4_f32_oracle.py --case qwen2.5-1.5b \
//!   --rewrite-acc1-dir target/qwen15b-acc1
//! ONNX_GENAI_QWEN15B_CUDA_DIR=/path/to/deployed/model \
//! ONNX_GENAI_QWEN15B_ACC1_DIR=target/qwen15b-acc1 \
//! CUDA_VISIBLE_DEVICES=0 cargo test -p onnx-genai-engine \
//!   --features native-backend,cuda --test qwen2_5_1_5b_divergence \
//!   -- --ignored --nocapture
//! ```
#![cfg(feature = "native-cuda")]

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GenerateOptions, GeneratePrompt, GenerateRequest,
    NativeDecodeDevice,
};

const DEFAULT_MODEL_DIR: &str =
    "/home/justinchu/.foundry/cache/models/Microsoft/qwen2.5-1.5b-instruct-cuda-gpu-4/v4";
const NATIVE_ORACLE_TOKEN: u32 = 576;
const ORT_CUDA_TOKEN: u32 = 15920;
const DIVERGENT_PREFIX: &[u32] = &[
    785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315, 9625, 374, 304, 279, 3146, 315, 9625, 13,
];
const BENCH_NATIVE_ORACLE_TOKEN: u32 = 1909;
const BENCH_ORT_CUDA_TOKEN: u32 = 821;
const BENCH_DIVERGENCE_INDEX: usize = 26;
const BENCH_PROMPT: &str = "Hello";
const BENCH_NATIVE_ORACLE_TOKENS: &[u32] = &[
    12824,
    13,
    576,
    2701,
    374,
    264,
    1140,
    315,
    279,
    1909,
    220,
    16,
    15,
    1429,
    5411,
    6467,
    315,
    678,
    882,
    13,
    576,
    1140,
    374,
    19697,
    504,
    279,
    BENCH_NATIVE_ORACLE_TOKEN,
];

fn next_token_from_prefix(
    dir: &Path,
    prefix: &[u32],
    backend: EngineDecodeBackend,
    native_device: Option<NativeDecodeDevice>,
) -> anyhow::Result<onnx_genai_engine::GenerateResult> {
    let mut engine = Engine::from_dir(
        dir,
        EngineConfig {
            decode_backend: backend,
            native_device,
            ..EngineConfig::default()
        },
    )?;
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(prefix.to_vec()));
    request.options.max_new_tokens = 1;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    request.options.top_logprobs = Some(8);
    engine.generate(request)
}

fn next_token(
    dir: &Path,
    backend: EngineDecodeBackend,
    native_device: Option<NativeDecodeDevice>,
) -> anyhow::Result<onnx_genai_engine::GenerateResult> {
    next_token_from_prefix(dir, DIVERGENT_PREFIX, backend, native_device)
}

fn generate_benchmark_prompt(
    dir: &Path,
    backend: EngineDecodeBackend,
    native_device: Option<NativeDecodeDevice>,
    top_logprobs: Option<usize>,
) -> anyhow::Result<onnx_genai_engine::GenerateResult> {
    let mut engine = Engine::from_dir(
        dir,
        EngineConfig {
            decode_backend: backend,
            native_device,
            ..EngineConfig::default()
        },
    )?;
    let mut request = GenerateRequest::new(GeneratePrompt::Text(BENCH_PROMPT.to_string()));
    request.options = GenerateOptions {
        max_new_tokens: BENCH_DIVERGENCE_INDEX + 1,
        temperature: 0.0,
        greedy: true,
        stop_on_eos: false,
        top_logprobs,
        ..GenerateOptions::default()
    };
    engine.generate(request)
}

fn assert_oracle_token(label: &str, result: &onnx_genai_engine::GenerateResult) {
    assert_eq!(result.token_ids, [NATIVE_ORACLE_TOKEN], "{label}");
    let top = &result
        .logprobs
        .as_ref()
        .expect("top_logprobs requested but absent")[0]
        .top;
    assert_eq!(
        top.first().map(|(token, _)| *token),
        Some(NATIVE_ORACLE_TOKEN),
        "{label}: {top:?}"
    );
    eprintln!("{label}: selected={NATIVE_ORACLE_TOKEN}, top={top:?}");
}

fn assert_bench_oracle_token(label: &str, result: &onnx_genai_engine::GenerateResult) -> f32 {
    assert_eq!(
        result.token_ids, BENCH_NATIVE_ORACLE_TOKENS,
        "{label} must select the fp32-oracle token at benchmark divergence index {BENCH_DIVERGENCE_INDEX}",
    );
    let top = &result
        .logprobs
        .as_ref()
        .expect("top_logprobs requested but absent")[BENCH_DIVERGENCE_INDEX]
        .top;
    assert_eq!(
        top.first().map(|(token, _)| *token),
        Some(BENCH_NATIVE_ORACLE_TOKEN),
        "{label}: {top:?}"
    );
    let native_lp = top
        .iter()
        .find(|(token, _)| *token == BENCH_NATIVE_ORACLE_TOKEN)
        .map(|(_, value)| *value)
        .expect("oracle/native token missing from top logprobs");
    let ort_lp = top
        .iter()
        .find(|(token, _)| *token == BENCH_ORT_CUDA_TOKEN)
        .map(|(_, value)| *value)
        .expect("ORT-CUDA alternative token missing from top logprobs");
    let margin = native_lp - ort_lp;
    eprintln!(
        "{label}: selected={BENCH_NATIVE_ORACLE_TOKEN}, token {BENCH_NATIVE_ORACLE_TOKEN} logprob={native_lp}, token {BENCH_ORT_CUDA_TOKEN} logprob={ort_lp}, margin={margin}, top={top:?}"
    );
    margin
}

#[test]
#[ignore = "requires the deployed and accuracy-level-1 Qwen2.5-1.5B models plus CUDA"]
fn qwen2_5_1_5b_native_cuda_matches_acc1_fp32_oracle() -> anyhow::Result<()> {
    let deployed_dir = std::env::var_os("ONNX_GENAI_QWEN15B_CUDA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR));
    let Some(oracle_dir) = std::env::var_os("ONNX_GENAI_QWEN15B_ACC1_DIR").map(PathBuf::from)
    else {
        eprintln!("skipping Qwen2.5-1.5B divergence lock: set ONNX_GENAI_QWEN15B_ACC1_DIR");
        return Ok(());
    };
    if !deployed_dir.is_dir() || !oracle_dir.is_dir() {
        eprintln!(
            "skipping Qwen2.5-1.5B divergence lock: deployed={} oracle={}",
            deployed_dir.display(),
            oracle_dir.display()
        );
        return Ok(());
    }
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping Qwen2.5-1.5B divergence lock: CUDA unavailable: {error}");
        return Ok(());
    }

    unsafe {
        std::env::set_var("ONNX_GENAI_EP", "cpu");
    }
    let fp32_oracle = next_token(&oracle_dir, EngineDecodeBackend::Ort, None)?;
    assert_oracle_token("ORT CPU accuracy-level-1 oracle", &fp32_oracle);

    let native = next_token(
        &deployed_dir,
        EngineDecodeBackend::Native,
        Some(NativeDecodeDevice::Cuda { index: Some(0) }),
    )?;
    assert_oracle_token("native CUDA", &native);

    let oracle_top = &fp32_oracle.logprobs.as_ref().unwrap()[0].top;
    assert!(
        oracle_top.iter().any(|(token, _)| *token == ORT_CUDA_TOKEN),
        "ORT CUDA token {ORT_CUDA_TOKEN} left the oracle top-8: {oracle_top:?}"
    );
    Ok(())
}

#[test]
#[ignore = "requires the deployed and accuracy-level-1 Qwen2.5-1.5B models plus CUDA"]
fn qwen2_5_1_5b_benchmark_prompt_native_cuda_matches_acc1_fp32_oracle() -> anyhow::Result<()> {
    let deployed_dir = std::env::var_os("ONNX_GENAI_QWEN15B_CUDA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR));
    let Some(oracle_dir) = std::env::var_os("ONNX_GENAI_QWEN15B_ACC1_DIR").map(PathBuf::from)
    else {
        eprintln!(
            "skipping Qwen2.5-1.5B benchmark divergence lock: set ONNX_GENAI_QWEN15B_ACC1_DIR"
        );
        return Ok(());
    };
    if !deployed_dir.is_dir() || !oracle_dir.is_dir() {
        eprintln!(
            "skipping Qwen2.5-1.5B benchmark divergence lock: deployed={} oracle={}",
            deployed_dir.display(),
            oracle_dir.display()
        );
        return Ok(());
    }
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping Qwen2.5-1.5B benchmark divergence lock: CUDA unavailable: {error}");
        return Ok(());
    }

    unsafe {
        std::env::set_var("ONNX_GENAI_EP", "cpu");
    }
    let fp32_oracle =
        generate_benchmark_prompt(&oracle_dir, EngineDecodeBackend::Ort, None, Some(8))?;
    let oracle_margin = assert_bench_oracle_token("ORT CPU accuracy-level-1 oracle", &fp32_oracle);

    unsafe {
        std::env::set_var("ONNX_GENAI_EP", "cuda");
    }
    let ort_cuda_logits =
        generate_benchmark_prompt(&deployed_dir, EngineDecodeBackend::Ort, None, Some(8))?;
    let ort_logits_margin = assert_bench_oracle_token("ORT CUDA acc-4 logits", &ort_cuda_logits);

    let native = generate_benchmark_prompt(
        &deployed_dir,
        EngineDecodeBackend::Native,
        Some(NativeDecodeDevice::Cuda { index: Some(0) }),
        None,
    )?;
    assert_eq!(
        native.token_ids, BENCH_NATIVE_ORACLE_TOKENS,
        "native CUDA default greedy fast path must keep the fp32-oracle token at index {BENCH_DIVERGENCE_INDEX}"
    );

    let native_logits = generate_benchmark_prompt(
        &deployed_dir,
        EngineDecodeBackend::Native,
        Some(NativeDecodeDevice::Cuda { index: Some(0) }),
        Some(8),
    )?;
    let native_margin = assert_bench_oracle_token("native CUDA logits", &native_logits);

    assert!(
        oracle_margin > 0.0 && native_margin > 0.0 && ort_logits_margin > 0.0,
        "expected fp32 oracle/native/ORT-logits to prefer {BENCH_NATIVE_ORACLE_TOKEN}; margins: oracle={oracle_margin}, native={native_margin}, ort_logits={ort_logits_margin}",
    );
    Ok(())
}
