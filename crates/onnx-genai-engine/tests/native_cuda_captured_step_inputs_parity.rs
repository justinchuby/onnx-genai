//! Native multi-component pipeline — increment 3c (issue #384).
//!
//! Proves the **captured per-step-input decode path** is (a) behavior-identical
//! to the default eager owned-input path and (b) genuinely engaged — not a
//! silent decline back to eager.
//!
//! Inc3a/Inc3b bind the decoder's `inputs_embeds`/routed ports on-device per
//! step but rebuild *owned* inputs and never capture, so every step pays the
//! uncaptured kernel-launch cost (measured ~2.8x below the graph-captured ceiling
//! and ~2x below ORT-CUDA on qwen3-0.6b). Inc3c writes those one-token per-step
//! tensors into *persistent* device bindings and reuses the captured
//! `run_one_token` graph, gated behind `ONNX_GENAI_NATIVE_DECODER_CAPTURE_STEP_INPUTS`
//! (default off, so the eager path stays byte-identical).
//!
//! Non-tautological proof: the captured and eager decode steps are distinct
//! functions. The process-global counter
//! [`NATIVE_DECODER_CAPTURED_STEP_INPUT_DECODES`] is bumped only inside the
//! captured branch, so a non-zero delta with the flag on — and a zero delta with
//! it off — is direct evidence the intended path executed, while identical token
//! IDs prove it stayed correct.
//!
//! Fixture: `tiny-gqa-embeds-cuda` — an `inputs_embeds` composite pipeline whose
//! decoder routes its KV through a real `GroupQueryAttention` op, so the native
//! CUDA decode path *engages* whole-graph CUDA-graph capture (the naive-Concat KV
//! tiny fixtures structurally decline it). Closed-form tokens `[0, 5, 6, 7]`.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use onnx_genai_engine::pipeline::PipelineGenerateRequest;
use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest,
    NATIVE_DECODER_CAPTURED_STEP_INPUT_DECODES,
};
use onnx_genai_ort::Value;

const NATIVE_DECODER_ENV: &str = "ONNX_GENAI_PIPELINE_NATIVE_DECODER";
const NATIVE_DECODER_DEVICE_ENV: &str = "ONNX_GENAI_PIPELINE_NATIVE_DECODER_DEVICE";
const CAPTURE_STEP_INPUTS_ENV: &str = "ONNX_GENAI_NATIVE_DECODER_CAPTURE_STEP_INPUTS";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-gqa-embeds-cuda")
}

fn tiny_pixels() -> anyhow::Result<Value> {
    Value::from_vec_f32((0..12).map(|i| i as f32 / 12.0).collect(), &[1, 3, 2, 2])
        .map_err(Into::into)
}

fn cuda_device_index() -> Option<u32> {
    std::env::var_os("CUDA_VISIBLE_DEVICES")?;
    Some(0)
}

/// Run the native-CUDA pipeline decoder for `max_new_tokens` steps, optionally
/// enabling the captured per-step-input path, and return `(tokens, captured
/// step-input decode count observed during this run)`.
fn generate_tokens_capturing(index: u32, capture: bool) -> anyhow::Result<(Vec<u32>, u64)> {
    unsafe {
        std::env::set_var(NATIVE_DECODER_ENV, "decoder");
        std::env::set_var(NATIVE_DECODER_DEVICE_ENV, format!("cuda:{index}"));
        if capture {
            std::env::set_var(CAPTURE_STEP_INPUTS_ENV, "1");
        } else {
            std::env::remove_var(CAPTURE_STEP_INPUTS_ENV);
        }
    }

    let before = NATIVE_DECODER_CAPTURED_STEP_INPUT_DECODES.load(Ordering::Relaxed);

    let result = (|| {
        let mut engine = Engine::from_pipeline_dir(&fixture_dir(), EngineConfig::default())?;
        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![3, 7]));
        request.options = GenerateOptions {
            max_new_tokens: 4,
            temperature: 0.0,
            stop_on_eos: false,
            ..GenerateOptions::default()
        };
        let pipeline_request = PipelineGenerateRequest::new(request)
            .with_input("vision_encoder.pixel_values", tiny_pixels()?);
        let result = engine.generate_with_pipeline_request(pipeline_request)?;
        Ok::<_, anyhow::Error>(result.token_ids)
    })();

    let after = NATIVE_DECODER_CAPTURED_STEP_INPUT_DECODES.load(Ordering::Relaxed);

    unsafe {
        std::env::remove_var(NATIVE_DECODER_ENV);
        std::env::remove_var(NATIVE_DECODER_DEVICE_ENV);
        std::env::remove_var(CAPTURE_STEP_INPUTS_ENV);
    }

    result.map(|tokens| (tokens, after.saturating_sub(before)))
}

#[test]
fn native_cuda_captured_step_inputs_match_eager_and_engage() -> anyhow::Result<()> {
    let Some(index) = cuda_device_index() else {
        eprintln!("skipping: no CUDA GPU visible (set CUDA_VISIBLE_DEVICES)");
        return Ok(());
    };

    // Default eager owned-input path: the reference token stream. No decode step
    // should take the captured branch, so the counter must not move.
    let (eager_tokens, eager_captured) = generate_tokens_capturing(index, false)?;
    assert_eq!(
        eager_tokens,
        vec![0, 5, 6, 7],
        "native CUDA eager routed baseline drifted"
    );
    assert_eq!(
        eager_captured, 0,
        "the captured per-step-input path must stay dormant with the flag off"
    );

    // Captured per-step-input path: identical tokens (correctness preserved) and
    // a non-zero counter delta (the captured branch genuinely executed instead of
    // silently declining to eager). One captured decode per generated token.
    let (captured_tokens, captured_count) = generate_tokens_capturing(index, true)?;
    assert_eq!(
        captured_tokens, eager_tokens,
        "captured per-step-input decode diverged from the eager owned-input path"
    );
    assert!(
        captured_count >= 1,
        "expected the captured per-step-input path to run at least once, saw {captured_count}"
    );
    eprintln!(
        "inc3c captured-step-inputs parity: tokens={captured_tokens:?} captured_decodes={captured_count}"
    );
    Ok(())
}
