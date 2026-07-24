//! Qwen2.5-1.5B int4 native-CUDA accuracy lock.
//!
//! For the raw prompt `"The capital of France is"`, native and ORT CUDA first
//! diverge after the common generated prefix ending in `"France."`: native
//! selects token 576 (`" The"`), while the deployed ORT CUDA path selects 15920
//! (`" Which"`). Rewriting every deployed MatMulNBits to `accuracy_level=1`
//! makes the ORT CPU fp32-activation oracle select 576, adjudicating native as
//! the more accurate backend at the first divergence.
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
#![cfg(all(feature = "native-backend", feature = "cuda"))]

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GeneratePrompt, GenerateRequest, NativeDecodeDevice,
};

const DEFAULT_MODEL_DIR: &str =
    "/home/justinchu/.foundry/cache/models/Microsoft/qwen2.5-1.5b-instruct-cuda-gpu-4/v4";
const NATIVE_ORACLE_TOKEN: u32 = 576;
const ORT_CUDA_TOKEN: u32 = 15920;
const DIVERGENT_PREFIX: &[u32] = &[
    785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315, 9625, 374, 304, 279, 3146, 315, 9625, 13,
];

fn next_token(
    dir: &Path,
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
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(DIVERGENT_PREFIX.to_vec()));
    request.options.max_new_tokens = 1;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    request.options.top_logprobs = Some(8);
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
