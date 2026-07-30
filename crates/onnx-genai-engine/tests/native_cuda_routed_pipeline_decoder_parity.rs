//! Native multi-component pipeline — increment 3b (issue #384).
//!
//! Proves the **native CUDA decoder binds a generic `Routed` port on-device**:
//! the Gemma4-style VLM composite pipeline gains a second cross-component edge —
//! the every_step `embedding` component emits a `router_state` output routed to a
//! `router_state` input on the decoder. That port has no generated role and is
//! not `inputs_embeds`; it is a `NativeStepInputSource::Routed` input, exactly
//! the class the CUDA decoder refused before Inc3b.
//!
//! Inc3a lifted the CUDA refusal only for `inputs_embeds`; Inc3b generalizes the
//! eager owned-input build so **any** routed port is uploaded per step and bound
//! on-device while the attention mask and KV cache stay device-resident. This
//! test drives the native decoder on CPU vs the CUDA EP (device 4 via
//! `CUDA_VISIBLE_DEVICES`) through the pipeline and asserts identical tokens,
//! proving the routed port reaches the GPU correctly with the KV kept resident.
//!
//! Fixture: `scripts/build_tiny_gemma4_vlm_cuda_routed.py` — closed-form tokens
//! `[0, 5, 6, 7]` (the routed `router_state` is consumed through a real `MatMul`
//! by a zero matrix, so it flows through a CUDA op but contributes nothing).

use std::path::{Path, PathBuf};

use onnx_genai_engine::pipeline::PipelineGenerateRequest;
use onnx_genai_engine::{Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest};
use onnx_genai_ort::Value;

const NATIVE_DECODER_ENV: &str = "ONNX_GENAI_PIPELINE_NATIVE_DECODER";
const NATIVE_DECODER_DEVICE_ENV: &str = "ONNX_GENAI_PIPELINE_NATIVE_DECODER_DEVICE";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-gemma4-vlm-cuda-routed")
}

fn tiny_pixels() -> anyhow::Result<Value> {
    Value::from_vec_f32((0..12).map(|i| i as f32 / 12.0).collect(), &[1, 3, 2, 2])
        .map_err(Into::into)
}

fn cuda_device_index() -> Option<u32> {
    std::env::var_os("CUDA_VISIBLE_DEVICES")?;
    Some(0)
}

fn generate_tokens(device: &str) -> anyhow::Result<Vec<u32>> {
    unsafe {
        std::env::set_var(NATIVE_DECODER_ENV, "decoder");
        std::env::set_var(NATIVE_DECODER_DEVICE_ENV, device);
    }

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

    unsafe {
        std::env::remove_var(NATIVE_DECODER_ENV);
        std::env::remove_var(NATIVE_DECODER_DEVICE_ENV);
    }
    result
}

#[test]
fn native_cuda_routed_pipeline_decoder_matches_cpu_token_ids() -> anyhow::Result<()> {
    let Some(index) = cuda_device_index() else {
        eprintln!("skipping: no CUDA GPU visible (set CUDA_VISIBLE_DEVICES)");
        return Ok(());
    };

    // Native decoder on CPU binding the routed port host-side — the reference.
    let cpu_tokens = generate_tokens("cpu")?;
    assert_eq!(
        cpu_tokens,
        vec![0, 5, 6, 7],
        "native CPU routed-port baseline drifted"
    );

    // Native decoder on the CUDA EP: the routed `router_state` is uploaded per
    // step and bound on-device; the KV cache stays device-resident on the GPU.
    let cuda_tokens = generate_tokens(&format!("cuda:{index}"))?;
    assert_eq!(
        cuda_tokens, cpu_tokens,
        "native CUDA decoder with a generic routed port diverged from the native CPU baseline"
    );
    Ok(())
}
