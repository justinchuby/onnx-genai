//! Inc3c real-model validation (issue #384): concretely demonstrate — with the
//! on-main capture counter — that **qwen3-0.6b does NOT engage the
//! capture-step-inputs path**, because it is a *single-component* `input_ids`
//! model, not a multi-component `inputs_embeds`/routed pipeline decoder.
//!
//! This is the honest "counter proof on the real qwen3-0.6b" the flag path was
//! probed for: the flag `ONNX_GENAI_NATIVE_DECODER_CAPTURE_STEP_INPUTS` only
//! gates `native_decode/cuda.rs::decode_cuda_captured_step_inputs`, which is
//! reached solely by a pipeline decoder that declares an `inputs_embeds`/`Routed`
//! port. qwen3-0.6b's decoder consumes `input_ids` directly and loads via the
//! single-model `Engine::from_dir`, so:
//!   1. `Engine::from_pipeline_dir` refuses it (no multi-component pipeline), and
//!   2. its native-CUDA single-graph decode leaves the captured-step-inputs
//!      counter at 0 even with the flag ON (it captures via the *token-id*
//!      `run_one_token` path, a different mechanism — the Part A 612/220/443
//!      lever — not the step-inputs path).
//!
//! The capture-step-inputs win therefore applies to multi-component
//! `inputs_embeds` decoders with capacity-aware KV (GroupQueryAttention), i.e.
//! the gemma-3n / 35B-A3B class — proven on the synthetic `tiny-gqa-embeds-cuda`
//! fixture in `native_cuda_captured_step_inputs_parity`.
//!
//! ```bash
//! source .cudaenv.sh
//! export CUDA_VISIBLE_DEVICES=4
//! export ONNX_GENAI_ORT_LIB=$ORT_ROOT/lib/libonnxruntime.so.1.27.0
//! QWEN3_0_6B_CUDA_E2E_DIR=/home/justinchu/mobius/.scratch/qwen3-0.6b-int4-cuda-postfix \
//! cargo test -p onnx-genai-engine --features cuda,native-backend \
//!   --test qwen3_0_6b_capture_step_inputs_decline -- --ignored --nocapture
//! ```
#![cfg(all(feature = "cuda", feature = "native-backend"))]

use std::path::PathBuf;
use std::sync::atomic::Ordering;

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GenerateOptions, GeneratePrompt, GenerateRequest,
    NATIVE_DECODER_CAPTURED_STEP_INPUT_DECODES, NativeDecodeDevice,
};

const DEFAULT_DIR: &str = "/home/justinchu/mobius/.scratch/qwen3-0.6b-int4-cuda-postfix";
const CAPTURE_ENV: &str = "ONNX_GENAI_NATIVE_DECODER_CAPTURE_STEP_INPUTS";
const PROMPT: &[u32] = &[785, 6722, 315];
const MAX_NEW_TOKENS: usize = 8;

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("QWEN3_0_6B_CUDA_E2E_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_DIR));
    if dir.join("model.onnx").is_file() && dir.join("inference_metadata.yaml").is_file() {
        Some(dir)
    } else {
        eprintln!(
            "skipping qwen3-0.6b capture-decline test: {} not present",
            dir.display()
        );
        None
    }
}

#[test]
#[ignore = "requires the real qwen3-0.6b int4 model via QWEN3_0_6B_CUDA_E2E_DIR and a CUDA device"]
fn qwen3_0_6b_single_component_declines_capture_step_inputs() -> anyhow::Result<()> {
    let Some(dir) = model_dir() else {
        return Ok(());
    };

    // (1) It is NOT a multi-component pipeline: `from_pipeline_dir` must refuse.
    let pipeline_load = Engine::from_pipeline_dir(&dir, EngineConfig::default());
    assert!(
        pipeline_load.is_err(),
        "qwen3-0.6b unexpectedly loaded as a multi-component pipeline; it is single-component"
    );
    eprintln!(
        "qwen3-0.6b is single-component: from_pipeline_dir refused it: {}",
        pipeline_load.err().unwrap()
    );

    // (2) Its native-CUDA single-graph decode leaves the captured-step-inputs
    // counter at 0 under the DEFAULT (capture default-on, no env set) — the
    // step-inputs capture path is pipeline-only, so an ineligible single-graph
    // decoder auto-declines even though capture is on by default.
    unsafe { std::env::remove_var(CAPTURE_ENV) };
    let before = NATIVE_DECODER_CAPTURED_STEP_INPUT_DECODES.load(Ordering::Relaxed);
    let tokens = (|| {
        let mut engine = Engine::from_dir(
            &dir,
            EngineConfig {
                decode_backend: EngineDecodeBackend::Native,
                native_device: Some(NativeDecodeDevice::Cuda { index: Some(0) }),
                ..EngineConfig::default()
            },
        )?;
        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(PROMPT.to_vec()));
        request.options = GenerateOptions {
            max_new_tokens: MAX_NEW_TOKENS,
            temperature: 0.0,
            greedy: true,
            stop_on_eos: false,
            ..GenerateOptions::default()
        };
        Ok::<_, anyhow::Error>(engine.generate(request)?.token_ids)
    })();
    let after = NATIVE_DECODER_CAPTURED_STEP_INPUT_DECODES.load(Ordering::Relaxed);
    let tokens = tokens?;

    assert_eq!(
        tokens.len(),
        MAX_NEW_TOKENS,
        "native decode did not emit the requested tokens"
    );
    assert_eq!(
        after - before,
        0,
        "qwen3-0.6b single-graph decode unexpectedly engaged the capture-step-inputs path"
    );
    eprintln!(
        "PROVEN: qwen3-0.6b native-CUDA decode with capture default-on → captured_step_input_decodes=0 \
         (single-component input_ids model does NOT route through the pipeline capture-step-inputs path); tokens={tokens:?}"
    );
    Ok(())
}
