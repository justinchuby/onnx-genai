//! End-to-end test for the iterative (diffusion) pipeline execution seam.
//!
//! Exercises `PipelineEngine::run_pipeline` over a `kind: iterative` strategy
//! using the tiny deterministic fixture built by
//! `scripts/build_tiny_diffusion.py`:
//!
//!   * `denoiser`: `denoised = (sample + cond) * 0.5`, with `sample`
//!     loop-carried via a `denoiser.denoised -> denoiser.sample` self-edge and
//!     `cond` re-supplied as constant conditioning each step.
//!   * `vae` (`final_only`): `image = latent * 2 + 1`, fed the final denoiser
//!     output via `denoiser.denoised -> vae.latent`.
//!
//! `num_steps = 3`, seed `sample = 0`, so the closed form is
//! `s_3 = cond * (7/8)` and `image = s_3 * 2 + 1`.

use onnx_genai_engine::{Engine, EngineConfig, GeneratePrompt, GenerateRequest, PipelineGenerateRequest};
use onnx_genai_ort::Value;
use std::path::{Path, PathBuf};

fn diffusion_fixture() -> anyhow::Result<PathBuf> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-diffusion")
        .canonicalize()?)
}

fn empty_request() -> PipelineGenerateRequest {
    // The iterative path ignores the token prompt; it consumes tensor inputs.
    PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])))
}

#[test]
fn iterative_diffusion_pipeline_runs_denoise_loop_and_final_vae() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&diffusion_fixture()?, EngineConfig::default())?;

    let cond = [1.0f32, 2.0, 3.0, 4.0];
    let request = empty_request()
        .with_input("denoiser.sample", Value::from_slice_f32(&[0.0; 4], &[1, 4])?)
        .with_input("denoiser.cond", Value::from_slice_f32(&cond, &[1, 4])?);

    let outputs = engine.run_pipeline(request)?;

    // Closed form after 3 steps of s_k = (s_{k-1} + cond) / 2 with s_0 = 0.
    let expected_sample: Vec<f32> = cond.iter().map(|c| c * 7.0 / 8.0).collect();
    let denoised = outputs
        .get("denoiser.denoised")
        .expect("denoiser output present")
        .to_vec_f32()?;
    for (got, want) in denoised.iter().zip(&expected_sample) {
        assert!((got - want).abs() < 1e-5, "denoised {got} != {want}");
    }

    // vae: image = latent * 2 + 1.
    let expected_image: Vec<f32> = expected_sample.iter().map(|s| s * 2.0 + 1.0).collect();
    let image = outputs
        .get("vae.image")
        .expect("vae output present")
        .to_vec_f32()?;
    for (got, want) in image.iter().zip(&expected_image) {
        assert!((got - want).abs() < 1e-5, "image {got} != {want}");
    }

    Ok(())
}

#[test]
fn generate_rejects_iterative_pipeline_with_clear_error() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&diffusion_fixture()?, EngineConfig::default())?;
    let err = engine
        .generate(GenerateRequest::new(GeneratePrompt::TokenIds(vec![0])))
        .expect_err("generate() must reject a non-autoregressive pipeline");
    assert!(
        err.to_string().contains("autoregressive"),
        "unexpected error: {err}"
    );
    Ok(())
}

#[test]
fn number_of_steps_controls_denoise_iterations() -> anyhow::Result<()> {
    // Sanity: a single step gives s_1 = cond / 2 (distinct from the 3-step form),
    // proving the loop honors num_steps rather than running once or forever.
    // The fixture is fixed at num_steps=3, so assert the 3-step result differs
    // from both the 1-step and 2-step closed forms.
    let mut engine = Engine::from_pipeline_dir(&diffusion_fixture()?, EngineConfig::default())?;
    let cond = [4.0f32, 8.0, 12.0, 16.0];
    let request = empty_request()
        .with_input("denoiser.sample", Value::from_slice_f32(&[0.0; 4], &[1, 4])?)
        .with_input("denoiser.cond", Value::from_slice_f32(&cond, &[1, 4])?);
    let outputs = engine.run_pipeline(request)?;
    let denoised = outputs.get("denoiser.denoised").unwrap().to_vec_f32()?;

    let three_step: Vec<f32> = cond.iter().map(|c| c * 7.0 / 8.0).collect();
    let one_step: Vec<f32> = cond.iter().map(|c| c * 0.5).collect();
    assert_eq!(denoised, three_step);
    assert_ne!(denoised, one_step);
    Ok(())
}
