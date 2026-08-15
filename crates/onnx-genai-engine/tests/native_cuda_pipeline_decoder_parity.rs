//! Native multi-component pipeline — increment 3a (issue #384).
//!
//! Proves the **native CUDA device-KV decoder** slice: the autoregressive
//! decoder of the Gemma4-style VLM composite pipeline produces **identical
//! generated token ids** when the native nxrt decoder runs on the **CUDA EP**
//! (device-resident KV, one-token `inputs_embeds` uploaded per step) as it does
//! on CPU. This lifts the prior CUDA refusal of `inputs_embeds`/routed step
//! inputs — the 35B-A3B GPU native-decode unblock.
//!
//! Both runs drive the same stateful
//! [`PipelineDecoderComponent`](../src/pipeline/decoder_component.rs) native
//! adapter introduced in inc2b (#479); only the decode **device** differs,
//! selected via `ONNX_GENAI_PIPELINE_NATIVE_DECODER_DEVICE` (`cpu` vs
//! `cuda:<index>`). The native decoder itself is selected with
//! `ONNX_GENAI_PIPELINE_NATIVE_DECODER=decoder`.
//!
//! Fixture: `scripts/build_tiny_gemma4_vlm_cuda.py` — closed-form identical to
//! `tiny-gemma4-vlm` (prompt `[3, 7]` -> `[0, 5, 6, 7]`) but the decoder also
//! declares `attention_mask` + `position_ids` (consumed via a zero term so the
//! logits are unchanged), which the CUDA decode path requires for its device
//! mask/KV bindings. Comparing native-CPU vs native-CUDA keeps the proof
//! independent of the ORT decoder handling the extra declared inputs.

use std::path::{Path, PathBuf};

use onnx_genai_engine::pipeline::PipelineGenerateRequest;
use onnx_genai_engine::{Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest};
use onnx_genai_ort::Value;

const NATIVE_DECODER_ENV: &str = "ONNX_GENAI_PIPELINE_NATIVE_DECODER";
const NATIVE_DECODER_DEVICE_ENV: &str = "ONNX_GENAI_PIPELINE_NATIVE_DECODER_DEVICE";

fn tiny_gemma4_vlm_cuda_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-gemma4-vlm-cuda")
}

fn tiny_pixels() -> anyhow::Result<Value> {
    Value::from_vec_f32((0..12).map(|i| i as f32 / 12.0).collect(), &[1, 3, 2, 2])
        .map_err(Into::into)
}

/// Returns the CUDA device index to test on, or `None` if no CUDA GPU is
/// visible (in which case the test skips gracefully).
fn cuda_device_index() -> Option<u32> {
    // Respect an explicit device pin from the environment; otherwise default to
    // device 0 of whatever `CUDA_VISIBLE_DEVICES` exposes.
    std::env::var_os("CUDA_VISIBLE_DEVICES")?;
    Some(0)
}

/// One composite generation over the CUDA fixture with the native decoder
/// pinned to `device` (`cpu` or e.g. `cuda:0`).
fn generate_tokens(device: &str) -> anyhow::Result<Vec<u32>> {
    unsafe {
        std::env::set_var(NATIVE_DECODER_ENV, "decoder");
        std::env::set_var(NATIVE_DECODER_DEVICE_ENV, device);
    }

    let result = (|| {
        let mut engine =
            Engine::from_pipeline_dir(&tiny_gemma4_vlm_cuda_dir(), EngineConfig::default())?;
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

    unsafe {
        std::env::remove_var(NATIVE_DECODER_ENV);
        std::env::remove_var(NATIVE_DECODER_DEVICE_ENV);
    }
    result
}

#[test]
fn native_cuda_pipeline_decoder_matches_cpu_token_ids() -> anyhow::Result<()> {
    let Some(index) = cuda_device_index() else {
        eprintln!("skipping: no CUDA GPU visible (set CUDA_VISIBLE_DEVICES)");
        return Ok(());
    };

    // Native decoder on CPU: the inc2b device-KV path, our reference here.
    let cpu_tokens = generate_tokens("cpu")?;
    assert_eq!(
        cpu_tokens,
        vec![0, 5, 6, 7],
        "native CPU decoder baseline drifted"
    );

    // Native decoder on the CUDA EP: inputs_embeds uploaded per step, KV kept
    // device-resident on the GPU.
    let cuda_tokens = generate_tokens(&format!("cuda:{index}"))?;
    assert_eq!(
        cuda_tokens, cpu_tokens,
        "native CUDA device-KV decoder diverged from the native CPU baseline"
    );
    Ok(())
}
