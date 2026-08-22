//! End-to-end proof that an imported ComfyUI workflow executes on the generic
//! workflow runtime.
//!
//! The package under `tests/fixtures/comfyui_workflows/txt2img_sd15` carries a
//! ComfyUI API-format `workflow.json`, the canonical `inference_metadata.yaml`
//! the importer lowered it into, tiny ONNX components in the ABI that metadata
//! references, and `reference.json`: an independent numpy simulation of what the
//! emitted metadata says should happen.
//!
//! Nothing in this file executes diffusion. It converts, loads, runs, and
//! compares. Every step of the loop — schedule, guidance, solver, decode — is
//! performed by the generic workflow engine from the metadata alone.

use std::path::PathBuf;

use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest,
    PipelineGenerateRequest,
    pipeline::{PipelineOutputs, WorkflowOutputRole},
};
use onnx_genai_ort::Value;

fn package() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/comfyui_workflows/txt2img_sd15")
}

fn reference() -> serde_json::Value {
    let path = package().join("reference.json");
    serde_json::from_str(&std::fs::read_to_string(&path).expect("reference.json")).expect("json")
}

fn floats(reference: &serde_json::Value, key: &str) -> Vec<f32> {
    reference[key]
        .as_array()
        .expect("array")
        .iter()
        .map(|value| value.as_f64().expect("float") as f32)
        .collect()
}

fn tokens(reference: &serde_json::Value, key: &str) -> Vec<i64> {
    reference[key]
        .as_array()
        .expect("array")
        .iter()
        .map(|value| value.as_i64().expect("int"))
        .collect()
}

/// Build the request the emitted metadata declares, by SSA value name.
fn request(
    prompt: &[i64],
    negative: &[i64],
    seed: i64,
    guidance: f32,
    steps: usize,
) -> anyhow::Result<PipelineGenerateRequest> {
    let options = GenerateOptions {
        max_new_tokens: steps,
        ..GenerateOptions::default()
    };
    Ok(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(Vec::new()),
        options,
    })
    .with_input(
        "request.input_ids",
        Value::from_slice_i64(prompt, &[1, i64::try_from(prompt.len())?])?,
    )
    .with_input(
        "request.negative_input_ids",
        Value::from_slice_i64(negative, &[1, i64::try_from(negative.len())?])?,
    )
    .with_input("request.seed", Value::from_slice_i64(&[seed], &[1])?)
    .with_input(
        "request.guidance_scale",
        Value::from_slice_f32(&[guidance], &[1])?,
    ))
}

fn image(outputs: &PipelineOutputs) -> &Value {
    outputs
        .aggregate("image")
        .expect("the converted workflow declares an image output")
}

/// The imported workflow runs to an image, and the image is the one the
/// metadata's own semantics predict.
#[test]
fn imported_comfyui_workflow_produces_the_referenced_image() -> anyhow::Result<()> {
    let reference = reference();
    let prompt = tokens(&reference, "prompt_tokens");
    let negative = tokens(&reference, "negative_tokens");
    let seed = reference["seed"].as_i64().expect("seed");
    let guidance = reference["guidance_scale"].as_f64().expect("guidance") as f32;
    let steps = reference["steps"].as_u64().expect("steps") as usize;
    // The reference is computed in double precision, so the bound is float32
    // execution error rather than an exact match. Any structural mis-wiring —
    // a swapped conditioning branch, an off-by-one schedule index, guidance
    // applied after the solver — moves these values by order 0.1, not 1e-3.
    let tolerance = reference["tolerance"].as_f64().expect("tolerance") as f32;

    let mut engine = Engine::from_pipeline_dir(&package(), EngineConfig::default())?;
    let outputs =
        engine.run_pipeline_outputs(request(&prompt, &negative, seed, guidance, steps)?)?;

    assert_eq!(image(&outputs).shape(), [1, 3, 4, 4]);
    let produced = image(&outputs).to_vec_f32()?;
    let expected = floats(&reference, "image");
    assert_eq!(produced.len(), expected.len());
    for (index, (actual, expected)) in produced.iter().zip(&expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "image element {index}: {actual} != {expected}"
        );
    }

    let latent = outputs
        .aggregate("latent")
        .expect("latent output")
        .to_vec_f32()?;
    for (index, (actual, expected)) in latent.iter().zip(&floats(&reference, "latent")).enumerate()
    {
        assert!(
            (actual - expected).abs() <= tolerance,
            "latent element {index}: {actual} != {expected}"
        );
    }
    Ok(())
}

/// Guidance is not decoration: the scale the ComfyUI graph declared reaches the
/// loop, and changing it changes the image.
#[test]
fn guidance_scale_reaches_the_converted_loop() -> anyhow::Result<()> {
    let reference = reference();
    let prompt = tokens(&reference, "prompt_tokens");
    let negative = tokens(&reference, "negative_tokens");
    let steps = reference["steps"].as_u64().expect("steps") as usize;
    let mut engine = Engine::from_pipeline_dir(&package(), EngineConfig::default())?;

    let guided = engine
        .run_pipeline_outputs(request(&prompt, &negative, 7, 7.5, steps)?)?
        .aggregate("image")
        .expect("image")
        .to_vec_f32()?;
    let unguided = engine
        .run_pipeline_outputs(request(&prompt, &negative, 7, 0.0, steps)?)?
        .aggregate("image")
        .expect("image")
        .to_vec_f32()?;
    assert!(
        guided
            .iter()
            .zip(&unguided)
            .any(|(left, right)| (left - right).abs() > 1e-4),
        "a guidance scale that never reaches the denoiser would leave the image unchanged"
    );

    // Guidance of exactly zero keeps only the unconditional branch, so it must
    // equal conditioning the single pass on the negative prompt instead.
    let negative_only = engine
        .run_pipeline_outputs(request(&negative, &negative, 7, 0.0, steps)?)?
        .aggregate("image")
        .expect("image")
        .to_vec_f32()?;
    for (index, (left, right)) in unguided.iter().zip(&negative_only).enumerate() {
        assert!(
            (left - right).abs() <= 1e-5,
            "element {index}: zero guidance must select the unconditional branch"
        );
    }
    Ok(())
}

/// The seed the ComfyUI graph declared drives a reproducible draw.
#[test]
fn the_converted_workflow_is_seed_deterministic() -> anyhow::Result<()> {
    let reference = reference();
    let prompt = tokens(&reference, "prompt_tokens");
    let negative = tokens(&reference, "negative_tokens");
    let steps = reference["steps"].as_u64().expect("steps") as usize;
    let mut engine = Engine::from_pipeline_dir(&package(), EngineConfig::default())?;

    let first = engine
        .run_pipeline_outputs(request(&prompt, &negative, 11, 7.5, steps)?)?
        .aggregate("image")
        .expect("image")
        .to_vec_f32()?;
    let again = engine
        .run_pipeline_outputs(request(&prompt, &negative, 11, 7.5, steps)?)?
        .aggregate("image")
        .expect("image")
        .to_vec_f32()?;
    assert_eq!(first, again, "the same seed must reproduce the same image");

    let other = engine
        .run_pipeline_outputs(request(&prompt, &negative, 12, 7.5, steps)?)?
        .aggregate("image")
        .expect("image")
        .to_vec_f32()?;
    assert!(
        first
            .iter()
            .zip(&other)
            .any(|(left, right)| (left - right).abs() > 1e-4),
        "a different seed must draw a different latent"
    );
    Ok(())
}

/// Step count is a run parameter of the emitted workflow, not a baked constant.
#[test]
fn the_converted_loop_honours_the_requested_step_count() -> anyhow::Result<()> {
    let reference = reference();
    let prompt = tokens(&reference, "prompt_tokens");
    let negative = tokens(&reference, "negative_tokens");
    let mut engine = Engine::from_pipeline_dir(&package(), EngineConfig::default())?;

    let full = engine
        .run_pipeline_outputs(request(&prompt, &negative, 5, 7.5, 4)?)?
        .aggregate("latent")
        .expect("latent")
        .to_vec_f32()?;
    let partial = engine
        .run_pipeline_outputs(request(&prompt, &negative, 5, 7.5, 2)?)?
        .aggregate("latent")
        .expect("latent")
        .to_vec_f32()?;
    assert!(
        full.iter()
            .zip(&partial)
            .any(|(left, right)| (left - right).abs() > 1e-4),
        "stopping the loop early must leave the latent at a different point"
    );
    Ok(())
}

/// The image role is what a consumer reads, not the output's spelling.
#[test]
fn the_image_output_is_reachable_by_role() -> anyhow::Result<()> {
    let reference = reference();
    let prompt = tokens(&reference, "prompt_tokens");
    let negative = tokens(&reference, "negative_tokens");
    let engine = Engine::from_pipeline_dir(&package(), EngineConfig::default())?;
    let mut engine = engine;
    let outputs = engine.run_pipeline_outputs(request(&prompt, &negative, 3, 7.5, 4)?)?;
    let by_role = engine
        .structured_output_for_role(&outputs, WorkflowOutputRole::Image)
        .expect("an imported diffusion workflow declares an image-role output");
    assert_eq!(by_role.shape(), [1, 3, 4, 4]);
    Ok(())
}
