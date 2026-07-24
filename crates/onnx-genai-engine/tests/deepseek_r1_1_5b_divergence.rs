//! Greedy-decode accuracy lock for DeepSeek-R1-Distill-Qwen-1.5B int4.
//!
//! With the model's chat template and prompt `"The capital of France is"`,
//! native and ORT CUDA agree for seven generated tokens, then diverge:
//!
//! | backend | token | logit(374) - logit(315) |
//! |---------|-------|--------------------------|
//! | native CUDA | **374** | +0.453125 |
//! | ORT CUDA | 315 | -0.125000 |
//! | ORT CPU, all MatMulNBits explicitly `accuracy_level=1` | **374** | +0.468750 |
//!
//! The fp32 MatMulNBits oracle therefore adjudicates token 374 as the true
//! argmax. The deployed graph has no explicit `accuracy_level` attributes
//! (equivalent to the fp32/default path for the native and CPU kernels);
//! explicitly setting all 141 nodes to level 1 leaves the CPU oracle on 374.
//! ORT CUDA ignores that accuracy-level rewrite and remains on 315, so matching
//! ORT CUDA here would be an accuracy regression.
//!
//! Run the real-model CPU + CUDA lock with:
//!
//! ```bash
//! DEEPSEEK_R1_1_5B_E2E_DIR=/path/to/model \
//! CUDA_VISIBLE_DEVICES=0 cargo test -p onnx-genai-engine --features native-backend,cuda \
//!   --test deepseek_r1_1_5b_divergence -- --ignored --nocapture
//! ```
#![cfg(all(feature = "native-backend", feature = "cuda"))]

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GenerateRequest, NativeDecodeDevice,
};

const DEFAULT_MODEL_DIR: &str =
    "/home/justinchu/glm-e2e-artifacts/deepseek-r1-distill-qwen-1.5b-int4-cuda";
const ORACLE_TOKEN: u32 = 374;
const ORT_CUDA_TOKEN: u32 = 315;
const EXPECTED_TOKENS: [u32; 8] = [3070, 34, 5367, 334, 13, 576, 6722, ORACLE_TOKEN];

fn generate(
    dir: &std::path::Path,
    backend: EngineDecodeBackend,
    device: Option<NativeDecodeDevice>,
) -> anyhow::Result<onnx_genai_engine::GenerateResult> {
    let config = EngineConfig {
        decode_backend: backend,
        native_device: device,
        ..EngineConfig::default()
    };
    let mut engine = Engine::from_dir(dir, config)?;
    let mut request = GenerateRequest::new("The capital of France is".to_string());
    request.options.max_new_tokens = EXPECTED_TOKENS.len();
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    request.options.top_logprobs = Some(8);
    engine.generate(request)
}

fn assert_oracle_argmax(label: &str, result: &onnx_genai_engine::GenerateResult) {
    assert_eq!(
        result.token_ids, EXPECTED_TOKENS,
        "{label} greedy stream drifted from the fp32-oracle-correct sequence"
    );
    let top = &result
        .logprobs
        .as_ref()
        .expect("top_logprobs requested but absent")[7]
        .top;
    assert_eq!(top.first().map(|(id, _)| *id), Some(ORACLE_TOKEN));
    let logprob = |token| {
        top.iter()
            .find(|(id, _)| *id == token)
            .map(|(_, value)| *value)
            .unwrap_or_else(|| panic!("token {token} missing from {label} top-8: {top:?}"))
    };
    let margin = logprob(ORACLE_TOKEN) - logprob(ORT_CUDA_TOKEN);
    assert!(
        (0.40..0.55).contains(&margin),
        "{label} must preserve the oracle's clear 374-over-315 margin; got {margin}"
    );
    eprintln!("{label}: token={ORACLE_TOKEN}, logit margin over {ORT_CUDA_TOKEN}={margin}");
}

#[test]
#[ignore = "requires the real DeepSeek-R1-Distill-Qwen-1.5B int4 model and a CUDA device"]
fn deepseek_r1_1_5b_native_cuda_matches_fp32_cpu_oracle() -> anyhow::Result<()> {
    let dir = std::env::var_os("DEEPSEEK_R1_1_5B_E2E_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(DEFAULT_MODEL_DIR));
    if !dir.is_dir() {
        eprintln!(
            "skipping DeepSeek-R1 divergence lock: model directory absent: {}",
            dir.display()
        );
        return Ok(());
    }
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping DeepSeek-R1 divergence lock: CUDA unavailable: {error}");
        return Ok(());
    }

    // The graph's absent accuracy_level attributes select the same fp32
    // MatMulNBits CPU path as the explicit level-1 oracle rewrite.
    unsafe {
        std::env::set_var("ONNX_GENAI_EP", "cpu");
    }
    let cpu_oracle = generate(&dir, EngineDecodeBackend::Ort, None)?;
    assert_oracle_argmax("ORT CPU fp32 oracle", &cpu_oracle);

    let cuda = generate(
        &dir,
        EngineDecodeBackend::Native,
        Some(NativeDecodeDevice::Cuda { index: Some(0) }),
    )?;
    assert_oracle_argmax("native CUDA", &cuda);
    Ok(())
}
