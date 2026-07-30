//! Native multi-component pipeline — increment 2b (issue #384).
//!
//! Proves the **native device-KV decoder** slice: the autoregressive **decoder**
//! of the Gemma4-style VLM composite pipeline produces **identical generated
//! token ids** whether it runs through ONNX Runtime (the default) or through the
//! native nxrt backend keeping its KV cache session-resident across steps.
//!
//! The same pipeline decode loop drives both backends through the stateful
//! [`PipelineDecoderComponent`](../src/pipeline/decoder_component.rs) trait
//! introduced in inc2a (#478) — there is no forked native copy of the decode
//! loop. The native decoder is selected at runtime via
//! `ONNX_GENAI_PIPELINE_NATIVE_DECODER=decoder`.
//!
//! Fixture: `scripts/build_tiny_gemma4_vlm.py`. Its closed-form head makes the
//! generated ids exact: prompt `[3, 7]` -> `[0, 5, 6, 7]`. The decoder consumes
//! `inputs_embeds` (produced each step by the `embedding` every_step component)
//! and grows one KV pair; fed identical inputs, the native and ORT decoders
//! sample identical tokens.
//!
//! The ORT baseline runs with the default paged decode path; the native decoder
//! runs the non-paged, fresh-decode path (native present-KV exposure + paged
//! reuse is inc3). Paging is a cross-request KV-reuse cache and does not change
//! the tokens produced within a generation, so the token ids must still match.

use std::path::{Path, PathBuf};

use onnx_genai_engine::pipeline::PipelineGenerateRequest;
use onnx_genai_engine::{Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest};
use onnx_genai_ort::Value;

const NATIVE_DECODER_ENV: &str = "ONNX_GENAI_PIPELINE_NATIVE_DECODER";

fn tiny_gemma4_vlm_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-gemma4-vlm")
}

fn tiny_pixels() -> anyhow::Result<Value> {
    // pixel_values[1,3,2,2] = i/12; the vision encoder means over channels.
    Value::from_vec_f32((0..12).map(|i| i as f32 / 12.0).collect(), &[1, 3, 2, 2])
        .map_err(Into::into)
}

/// One composite generation over the fixture. `native_decoder` selects whether
/// the autoregressive `decoder` runs on the native device-KV backend.
fn generate_tokens(native_decoder: bool) -> anyhow::Result<Vec<u32>> {
    // Process-global env: set/clear around the single engine construction that
    // reads it, so the two runs in this test do not interleave. This test's
    // integration binary owns the process.
    if native_decoder {
        unsafe { std::env::set_var(NATIVE_DECODER_ENV, "decoder") };
    } else {
        unsafe { std::env::remove_var(NATIVE_DECODER_ENV) };
    }

    let result = (|| {
        let mut engine =
            Engine::from_pipeline_dir(&tiny_gemma4_vlm_dir(), EngineConfig::default())?;
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

    unsafe { std::env::remove_var(NATIVE_DECODER_ENV) };
    result
}

#[test]
fn native_pipeline_decoder_matches_ort_token_ids() -> anyhow::Result<()> {
    let ort_tokens = generate_tokens(false)?;
    let native_tokens = generate_tokens(true)?;

    // Baseline the ORT path against the fixture's known closed-form ids so a
    // regression that changes *both* backends identically still fails.
    assert_eq!(ort_tokens, vec![0, 5, 6, 7], "ORT decoder baseline drifted");
    assert_eq!(
        native_tokens, ort_tokens,
        "native device-KV decoder diverged from the ORT baseline"
    );
    Ok(())
}
