//! Regression guard for the decode KV-bucket-growth workspace abort (Bug 1).
//!
//! Open-ended native-CUDA generation past the first KV power-of-two capacity
//! boundary (256) from a short prompt used to abort with the governed-workspace
//! invariant (`required > prepared`): the single-token decode step grew the KV
//! bucket but never re-prepared the persistent `::Attention` fp32 score scratch
//! (`batch·q_heads·q_seq·total_seq·4`) for the new, larger bucket, so the first
//! execute after the 256→512 jump needed exactly 2× the reserved workspace.
//!
//! Every native model doing open-ended generation past ~256 tokens with the
//! default KV configuration hit this; benches only dodged it by pre-sizing KV
//! (`ONNX_GENAI_KV_MIN_BUCKET`). These tests generate 320 tokens from a short
//! prompt with the DEFAULT KV config and assert completion — no golden stream,
//! just "does it survive the rebucket". They cover both a QMoE model
//! (DeepSeek-V2-Lite) and a dense model (Qwen2.5-0.5B) so the fix is proven for
//! MoE and non-MoE decode alike.
//!
//! ```bash
//! DEEPSEEK_V2_LITE_CUDA_DIR=/path/to/deepseek-v2-lite-real-int4-post434 \
//! ONNX_GENAI_QWEN05B_CUDA_DIR=/path/to/qwen2.5-0.5b-int4 \
//! CUDA_VISIBLE_DEVICES=0 cargo test -p onnx-genai-engine \
//!   --features native-backend,cuda --test decode_workspace_bucket_growth \
//!   -- --ignored --test-threads=1 --nocapture
//! ```
#![cfg(all(feature = "native-backend", feature = "cuda"))]

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GeneratePrompt, GenerateRequest, NativeDecodeDevice,
    SpeculativeMode,
};

#[path = "common/decode_lock.rs"]
mod decode_lock;

/// Generate past the first two KV bucket boundaries (256, 512) so the rebucket
/// path is exercised at least twice.
const TOKEN_COUNT: usize = 320;

/// Single-token-bucket prompt: keeps prefill inside the initial 256 bucket so
/// the growth is driven entirely by decode, matching the minimal repro.
const PROMPT: &str = "Hello";

#[test]
#[ignore = "requires the real DeepSeek-V2-Lite int4 export and a CUDA device"]
fn deepseek_v2_lite_native_cuda_generates_past_kv_bucket_growth() -> anyhow::Result<()> {
    decode_lock::assert_generation_completes("DEEPSEEK_V2_LITE_CUDA_DIR", PROMPT, TOKEN_COUNT)
}

#[test]
#[ignore = "requires the deployed Qwen2.5-0.5B int4 model and a CUDA device"]
fn qwen2_5_0_5b_native_cuda_generates_past_kv_bucket_growth() -> anyhow::Result<()> {
    decode_lock::assert_generation_completes("ONNX_GENAI_QWEN05B_CUDA_DIR", PROMPT, TOKEN_COUNT)
}

/// Eager-path guard: native prompt-lookup speculative decode past the 256 KV
/// bucket boundary exercises the eager multi-token verify forward
/// (`decode_cuda_eager`) — the residual Bug 1 site #1189 did not guard. A
/// verify pass whose `past + draft.len()` crosses a power-of-two boundary must
/// re-prepare the governed `::Attention` workspace for the grown bucket instead
/// of tripping the prepared-workspace invariant. This asserts open-ended
/// speculative generation completes (placement/liveness, not token identity —
/// the eager forward's logits leave the numerically coherent window well before
/// 256, so the speculative stream is not expected to match greedy at this
/// depth).
#[test]
#[ignore = "requires the deployed Qwen2.5-0.5B int4 model and a CUDA device"]
fn qwen2_5_0_5b_native_cuda_speculative_generates_past_kv_bucket_growth() -> anyhow::Result<()> {
    let Some(model_dir) = std::env::var_os("ONNX_GENAI_QWEN05B_CUDA_DIR") else {
        eprintln!("skipping: set ONNX_GENAI_QWEN05B_CUDA_DIR");
        return Ok(());
    };
    let model_dir = std::path::PathBuf::from(model_dir);
    if onnx_runtime_ep_cuda::CudaExecutionProvider::new(0).is_err() {
        eprintln!("skipping: CUDA unavailable");
        return Ok(());
    }
    unsafe {
        std::env::set_var("ONNX_GENAI_EP", "cuda");
    }
    let mut engine = Engine::from_dir(
        &model_dir,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(NativeDecodeDevice::Cuda { index: Some(0) }),
            ..EngineConfig::default()
        },
    )?;
    // A repetitive prompt so prompt-lookup finds matching n-grams and engages
    // the propose -> eager-verify -> accept path across the bucket boundary.
    let prompt = "The cat sat on the mat. ".repeat(4);
    let mut request = GenerateRequest::new(GeneratePrompt::Text(prompt));
    request.options.max_new_tokens = TOKEN_COUNT;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    request.options.speculative_mode = Some(SpeculativeMode::PromptLookup {
        ngram: 2,
        max_tokens: 4,
    });
    let out = engine.generate(request)?;
    assert_eq!(
        out.token_ids.len(),
        TOKEN_COUNT,
        "native speculative generation returned {} of {TOKEN_COUNT} tokens \
         (a short return means it aborted mid-stream, e.g. the eager verify \
         KV-bucket-growth workspace invariant abort)",
        out.token_ids.len(),
    );
    Ok(())
}
