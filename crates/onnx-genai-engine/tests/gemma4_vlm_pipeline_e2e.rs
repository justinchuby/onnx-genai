//! End-to-end test for the **composite** pipeline strategy over a Gemma4-style
//! VLM with **inputs_embeds fusion** — the multimodal decoder path from
//! DESIGN.md §20 that the 2-model `tiny-vlm` fixture does *not* exercise.
//!
//! Runs `Engine::from_pipeline_dir` + `generate_with_pipeline_request` on the
//! deterministic fixture built by `scripts/build_tiny_gemma4_vlm.py`:
//!
//!   * `vision_encoder`: `pixel_values[1,3,2,2] -> image_features[1,1,4]`
//!   * `embedding` (fusion): `input_ids + image_features -> inputs_embeds`
//!     (`inputs_embeds = E[input_ids] + 1{input_ids==7} * image_features`)
//!   * `decoder`: `inputs_embeds -> logits + KV` (no `input_ids` input)
//!
//! wired `vision_encoder.image_features -> embedding.image_features` and
//! `embedding.inputs_embeds -> decoder.inputs_embeds` via the pipeline
//! `dataflow`. This proves two engine seams unique to inputs_embeds fusion:
//!
//!   1. the prompt token ids are seeded into the shared pool as
//!      `embedding.input_ids` so the fusion model runs in the prompt phase, and
//!   2. each decode step re-embeds the running token through the fusion model to
//!      produce a single-token `inputs_embeds` (the decoder has no token input).
//!
//! The closed-form head (`logits = inputs_embeds @ W`, `W[:,v] = E[(v-1)%8]`)
//! makes the generated token ids exact: prompt `[3, <image>]` -> `[0, 5, 6, 7]`,
//! where the first token additionally depends on the fused image features.

use std::path::{Path, PathBuf};

use onnx_genai_engine::pipeline::PipelineGenerateRequest;
use onnx_genai_engine::{Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest};
use onnx_genai_ort::Value;

fn tiny_gemma4_vlm_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-gemma4-vlm")
}

fn tiny_pixels() -> anyhow::Result<Value> {
    // pixel_values[1,3,2,2] = i/12; the vision encoder means over channels.
    Value::from_vec_f32((0..12).map(|i| i as f32 / 12.0).collect(), &[1, 3, 2, 2])
        .map_err(Into::into)
}

#[test]
fn gemma4_vlm_composite_generates_inputs_embeds_tokens() -> anyhow::Result<()> {
    let model_dir = tiny_gemma4_vlm_dir();

    let mut engine = Engine::from_pipeline_dir(&model_dir, EngineConfig::default())?;

    // Prompt: a text token (3) followed by the image placeholder token (7). The
    // trailing placeholder is where the fusion scatters the image features, so
    // the first generated token depends on the vision -> fusion -> decode chain.
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

    // Exact closed-form chain (also asserted by the Python builder against ORT).
    assert_eq!(result.token_ids, vec![0, 5, 6, 7]);
    Ok(())
}
