//! Native multi-component pipeline — increment 1 (issue #384).
//!
//! Proves the value-type seam: the `every_step` embedding component of the
//! Gemma4-style VLM composite pipeline produces **identical generated token
//! ids** whether it runs through ONNX Runtime (the default) or through the
//! native nxrt backend, while the decoder stays on ORT in both runs.
//!
//! The same decode loop drives both backends through the backend-neutral
//! [`ComponentSession`](onnx_genai_metadata::ComponentSession) trait — there is
//! no forked native copy of `run_step_components`. The native every_step
//! component is selected at runtime via
//! `ONNX_GENAI_PIPELINE_NATIVE_STEP_COMPONENTS=embedding`.
//!
//! Fixture: `scripts/build_tiny_gemma4_vlm.py`. Its closed-form head makes the
//! generated ids exact: prompt `[3, 7]` -> `[0, 5, 6, 7]`. The embedding graph
//! is an integer `Gather` plus exact-valued `Mul`/`Add` over a 0/1 table, so the
//! fused `inputs_embeds` is byte-identical across backends and the decoder — fed
//! identical inputs — samples identical tokens.

use std::path::{Path, PathBuf};

use onnx_genai_engine::pipeline::PipelineGenerateRequest;
use onnx_genai_engine::{Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest};
use onnx_genai_ort::Value;

const NATIVE_STEP_ENV: &str = "ONNX_GENAI_PIPELINE_NATIVE_STEP_COMPONENTS";

fn tiny_gemma4_vlm_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-gemma4-vlm")
}

fn tiny_pixels() -> anyhow::Result<Value> {
    // pixel_values[1,3,2,2] = i/12; the vision encoder means over channels.
    Value::from_vec_f32((0..12).map(|i| i as f32 / 12.0).collect(), &[1, 3, 2, 2])
        .map_err(Into::into)
}

/// One composite generation over the fixture. `native_embedding` selects whether
/// the `embedding` every_step component runs on the native backend.
fn generate_tokens(native_embedding: bool) -> anyhow::Result<Vec<u32>> {
    // Process-global env: set/clear around the single engine construction that
    // reads it, so the two runs in this test do not interleave. This test's
    // integration binary owns the process.
    if native_embedding {
        unsafe { std::env::set_var(NATIVE_STEP_ENV, "embedding") };
    } else {
        unsafe { std::env::remove_var(NATIVE_STEP_ENV) };
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

    unsafe { std::env::remove_var(NATIVE_STEP_ENV) };
    result
}

#[test]
fn native_every_step_embedding_matches_ort_token_ids() -> anyhow::Result<()> {
    let ort_tokens = generate_tokens(false)?;
    let native_tokens = generate_tokens(true)?;

    // Baseline the ORT path against the fixture's known closed-form ids so a
    // regression that changes *both* backends identically still fails.
    assert_eq!(
        ort_tokens,
        vec![0, 5, 6, 7],
        "ORT every_step baseline drifted"
    );
    assert_eq!(
        native_tokens, ort_tokens,
        "native every_step embedding diverged from the ORT baseline"
    );
    Ok(())
}
