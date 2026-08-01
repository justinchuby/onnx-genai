//! Native multi-component pipeline — full-native composite decode (issue #384).
//!
//! The prior increments proved the backend-neutral ownership seam one slice at a
//! time: inc1 (#450) drove the `every_step` embedding component natively while
//! the decoder stayed on ORT, and inc2b (#479) drove the decoder natively while
//! the embedding stayed on ORT. This test closes the loop and locks the seam for
//! the **keystone** case both slices exist to enable: **every declared component
//! session running on the native backend at once**, which is exactly the shape a
//! large multi-component package (e.g. an `inputs_embeds` embedding feeding a
//! routed decoder, up to the 35B-A3B 3-component package) decodes through.
//!
//! One pipeline decode loop drives both the every_step
//! [`ComponentSession`](onnx_genai_metadata::ComponentSession) and the stateful
//! [`PipelineDecoderComponent`](../src/pipeline/decoder_component.rs) on the
//! native nxrt backend simultaneously — there is no forked native decode loop.
//! Both are selected at runtime:
//! `ONNX_GENAI_PIPELINE_NATIVE_STEP_COMPONENTS=embedding` and
//! `ONNX_GENAI_PIPELINE_NATIVE_DECODER=decoder`.
//!
//! Fixture: `scripts/build_tiny_gemma4_vlm.py`. Its closed-form head makes the
//! generated ids exact: prompt `[3, 7]` -> `[0, 5, 6, 7]`. The embedding graph
//! is an integer `Gather` plus exact-valued `Mul`/`Add` over a 0/1 table, so the
//! fused `inputs_embeds` is byte-identical across backends; the decoder consumes
//! that embedding each step and grows one KV pair session-resident. Fed identical
//! inputs, the fully-native pipeline samples the same tokens as the ORT baseline.
//!
//! The ORT baseline runs the default paged decode path; native selection runs the
//! non-paged, fresh-decode path (native present-KV exposure + paged cross-request
//! reuse is the next increment). Paging is a cross-request KV-reuse cache and does
//! not change the tokens produced within a single generation, so the ids match.

use std::path::{Path, PathBuf};

use onnx_genai_engine::pipeline::PipelineGenerateRequest;
use onnx_genai_engine::{Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest};
use onnx_genai_ort::Value;

const NATIVE_STEP_ENV: &str = "ONNX_GENAI_PIPELINE_NATIVE_STEP_COMPONENTS";
const NATIVE_DECODER_ENV: &str = "ONNX_GENAI_PIPELINE_NATIVE_DECODER";

fn tiny_gemma4_vlm_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-gemma4-vlm")
}

fn tiny_pixels() -> anyhow::Result<Value> {
    // pixel_values[1,3,2,2] = i/12; the vision encoder means over channels.
    Value::from_vec_f32((0..12).map(|i| i as f32 / 12.0).collect(), &[1, 3, 2, 2])
        .map_err(Into::into)
}

/// One composite generation over the fixture. When `full_native` is set, *both*
/// the every_step `embedding` component and the autoregressive `decoder` run on
/// the native backend; otherwise the whole pipeline runs on ORT.
fn generate_tokens(full_native: bool) -> anyhow::Result<Vec<u32>> {
    // Process-global env: set/clear around the single engine construction that
    // reads it, so the two runs in this test do not interleave. This test's
    // integration binary owns the process.
    if full_native {
        unsafe { std::env::set_var(NATIVE_STEP_ENV, "embedding") };
        unsafe { std::env::set_var(NATIVE_DECODER_ENV, "decoder") };
    } else {
        unsafe { std::env::remove_var(NATIVE_STEP_ENV) };
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

    unsafe { std::env::remove_var(NATIVE_STEP_ENV) };
    unsafe { std::env::remove_var(NATIVE_DECODER_ENV) };
    result
}

#[test]
fn full_native_pipeline_matches_ort_token_ids() -> anyhow::Result<()> {
    let ort_tokens = generate_tokens(false)?;
    let native_tokens = generate_tokens(true)?;

    // Baseline the ORT path against the fixture's known closed-form ids so a
    // regression that changes *both* backends identically still fails.
    assert_eq!(
        ort_tokens,
        vec![0, 5, 6, 7],
        "ORT composite baseline drifted"
    );
    assert_eq!(
        native_tokens, ort_tokens,
        "fully-native composite pipeline (native embedding + native decoder) diverged from the \
         ORT baseline"
    );
    Ok(())
}
