//! Greedy-decode accuracy lock for DeepSeek-R1-Distill-Qwen-1.5B int4.
//!
//! With the raw prompt `"The capital of France is"`, native and ORT CUDA agree
//! for seven generated tokens, then diverge:
//!
//! | backend | token | logit(374) - logit(315) |
//! |---------|-------|--------------------------|
//! | native CUDA | **374** | +0.453125 |
//! | ORT CUDA | 315 | -0.125000 |
//! | ORT CPU, fp32 `MatMulNBits` oracle | **374** | +0.468750 |
//!
//! The fp32 MatMulNBits oracle therefore adjudicates token 374 as the true
//! argmax. The deployed graph has no explicit `accuracy_level` attributes
//! (equivalent to the fp32/default path for the native and CPU kernels), so the
//! ORT CPU run is the fp32 oracle without any rewrite; explicitly setting all
//! 141 nodes to level 1 leaves the CPU oracle on 374. ORT CUDA ignores that
//! accuracy-level rewrite and remains on 315, so matching ORT CUDA here would be
//! an accuracy regression.
//!
//! The shared adjudication harness lives in `common/oracle_lock.rs` and is
//! reused by the benchmark-prompt lock in
//! `deepseek_r1_1_5b_benchprompt_divergence.rs`.
//!
//! Run the real-model CPU + CUDA lock with:
//!
//! ```bash
//! DEEPSEEK_R1_1_5B_E2E_DIR=/path/to/model \
//! CUDA_VISIBLE_DEVICES=0 cargo test -p onnx-genai-engine --features native-backend,cuda \
//!   --test deepseek_r1_1_5b_divergence -- --ignored --nocapture
//! ```
#![cfg(all(feature = "native-backend", feature = "cuda"))]

#[path = "common/oracle_lock.rs"]
mod oracle_lock;

use onnx_genai_engine::{EngineDecodeBackend, NativeDecodeDevice};

const DEFAULT_MODEL_DIR: &str =
    "/home/justinchu/glm-e2e-artifacts/deepseek-r1-distill-qwen-1.5b-int4-cuda";
const PROMPT: &str = "The capital of France is";
const ORACLE_TOKEN: u32 = 374;
const ORT_CUDA_TOKEN: u32 = 315;
const DIVERGENCE_INDEX: usize = 7;
const EXPECTED_TOKENS: [u32; 8] = [3070, 34, 5367, 334, 13, 576, 6722, ORACLE_TOKEN];
// Both the native and CPU-oracle margins for 374 over 315 sit near +0.46.
const MARGIN_BAND: std::ops::RangeInclusive<f32> = 0.40..=0.55;

#[test]
#[ignore = "requires the real DeepSeek-R1-Distill-Qwen-1.5B int4 model and a CUDA device"]
fn deepseek_r1_1_5b_native_cuda_matches_fp32_cpu_oracle() -> anyhow::Result<()> {
    let Some(dir) = oracle_lock::resolve_model_dir("DEEPSEEK_R1_1_5B_E2E_DIR", DEFAULT_MODEL_DIR)
    else {
        return Ok(());
    };
    if !oracle_lock::cuda_available() {
        return Ok(());
    }

    // The graph's absent accuracy_level attributes select the same fp32
    // MatMulNBits CPU path as an explicit level-1 oracle rewrite.
    unsafe {
        std::env::set_var("ONNX_GENAI_EP", "cpu");
    }
    let cpu_oracle = oracle_lock::generate(
        &dir,
        PROMPT,
        EXPECTED_TOKENS.len(),
        EngineDecodeBackend::Ort,
        None,
    )?;
    oracle_lock::assert_oracle_argmax(
        "ORT CPU fp32 oracle",
        &cpu_oracle,
        &EXPECTED_TOKENS,
        DIVERGENCE_INDEX,
        ORACLE_TOKEN,
        ORT_CUDA_TOKEN,
        MARGIN_BAND,
    );

    let cuda = oracle_lock::generate(
        &dir,
        PROMPT,
        EXPECTED_TOKENS.len(),
        EngineDecodeBackend::Native,
        Some(NativeDecodeDevice::Cuda { index: Some(0) }),
    )?;
    oracle_lock::assert_oracle_argmax(
        "native CUDA",
        &cuda,
        &EXPECTED_TOKENS,
        DIVERGENCE_INDEX,
        ORACLE_TOKEN,
        ORT_CUDA_TOKEN,
        MARGIN_BAND,
    );
    Ok(())
}
