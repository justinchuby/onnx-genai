//! Benchmark-prompt greedy-decode accuracy lock for
//! DeepSeek-R1-Distill-Qwen-1.5B int4.
//!
//! This is the second, independent DeepSeek-R1 divergence adjudicated by the
//! fp32 `MatMulNBits` oracle. It uses the repository's native-vs-ORT parity
//! benchmark prompt (`tests/parity/native_ort_cuda_golden.json`):
//!
//! > "Explain why deterministic greedy decoding is useful when validating a new
//! >  transformer inference backend. Include concrete numerical failure modes
//! >  and how you would distinguish harmless floating-point near-ties from
//! >  implementation bugs."
//!
//! Native and ORT CUDA agree for the first 14 generated tokens, then diverge at
//! generated index 14:
//!
//! | backend | token 14 | logit(47116) - logit(10519) |
//! |---------|----------|------------------------------|
//! | native CUDA | **47116** | +0.375000 |
//! | ORT CUDA | 10519 | (selects 10519) |
//! | ORT CPU, fp32 `MatMulNBits` oracle | **47116** | +0.390625 |
//! | ORT CPU, explicit `accuracy_level=1` rewrite (141 nodes) | **47116** | +0.390625 |
//!
//! The fp32 oracle adjudicates token 47116 as the true argmax, so native CUDA is
//! the *more accurate* backend at this razor-thin tie; matching ORT CUDA (10519)
//! would be an accuracy regression. This is the same accuracy-level phenomenon
//! locked for the `"capital of France"` prompt in
//! `deepseek_r1_1_5b_divergence.rs`, now independently oracle-adjudicated for the
//! benchmark prompt.
//!
//! Historical note: the 2026-07-25 status doc recorded a *different* divergence
//! on the chat-templated `"history of computing"` prompt (native 374 vs ORT CUDA
//! 594 at generated index 15). At the current HEAD the decode kernels have
//! converged and that prompt no longer diverges (native == ORT CUDA for 40
//! tokens); the benchmark-parity prompt above is the divergence that reproduces
//! and is locked here.
//!
//! Run the real-model CPU + CUDA lock with:
//!
//! ```bash
//! DEEPSEEK_R1_1_5B_E2E_DIR=/path/to/model \
//! CUDA_VISIBLE_DEVICES=0 cargo test -p onnx-genai-engine --features native-backend,cuda \
//!   --test deepseek_r1_1_5b_benchprompt_divergence -- --ignored --nocapture
//! ```
#![cfg(all(feature = "native-backend", feature = "cuda"))]

#[path = "common/oracle_lock.rs"]
mod oracle_lock;

use onnx_genai_engine::{EngineDecodeBackend, NativeDecodeDevice};

const DEFAULT_MODEL_DIR: &str =
    "/home/justinchu/glm-e2e-artifacts/deepseek-r1-distill-qwen-1.5b-int4-cuda";
const PROMPT: &str = "Explain why deterministic greedy decoding is useful when validating a new \
transformer inference backend. Include concrete numerical failure modes and how you would \
distinguish harmless floating-point near-ties from implementation bugs.";
const ORACLE_TOKEN: u32 = 47116;
const ORT_CUDA_TOKEN: u32 = 10519;
const DIVERGENCE_INDEX: usize = 14;
const EXPECTED_TOKENS: [u32; 15] = [
    2014,
    10339,
    11,
    358,
    1184,
    311,
    1156,
    3535,
    279,
    18940,
    315,
    42578,
    44378,
    11,
    ORACLE_TOKEN,
];
// Native (+0.375) and CPU-oracle (+0.390625) margins for 47116 over 10519.
const MARGIN_BAND: std::ops::RangeInclusive<f32> = 0.30..=0.45;

#[test]
#[ignore = "requires the real DeepSeek-R1-Distill-Qwen-1.5B int4 model and a CUDA device"]
fn deepseek_r1_1_5b_benchprompt_native_cuda_matches_fp32_cpu_oracle() -> anyhow::Result<()> {
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
