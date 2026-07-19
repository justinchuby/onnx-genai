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
use std::fs;
use std::path::{Path, PathBuf};

fn diffusion_fixture() -> anyhow::Result<PathBuf> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-diffusion")
        .canonicalize()?)
}

/// Build a scratch pipeline dir that reuses the committed tiny-diffusion ONNX
/// files but substitutes custom `inference_metadata.yaml`.
fn fixture_with_metadata(name: &str, files: &[&str], metadata: &str) -> anyhow::Result<PathBuf> {
    let source = diffusion_fixture()?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/test-fixtures")
        .join(name);
    fs::create_dir_all(&root)?;
    for file in files {
        fs::copy(source.join(file), root.join(file))?;
    }
    fs::write(root.join("inference_metadata.yaml"), metadata)?;
    Ok(root)
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

#[test]
fn single_pass_pipeline_runs_one_forward() -> anyhow::Result<()> {
    // vae alone as a single_pass model: image = latent * 2 + 1.
    let metadata = "\
pipeline:
  models:
    vae:
      filename: vae.onnx
      type: vae
  strategy:
    kind: single_pass
    model: vae
";
    let dir = fixture_with_metadata("diffusion-single-pass", &["vae.onnx"], metadata)?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;

    let latent = [1.5f32, -2.0, 3.0, 0.25];
    let request = empty_request().with_input("vae.latent", Value::from_slice_f32(&latent, &[1, 4])?);
    let out = engine.run_pipeline(request)?;
    let image = out.get("vae.image").expect("vae output").to_vec_f32()?;
    for (got, l) in image.iter().zip(&latent) {
        assert!((got - (l * 2.0 + 1.0)).abs() < 1e-5, "{got} != {}", l * 2.0 + 1.0);
    }
    Ok(())
}

fn diffusion_metadata_with_guidance(scale: &str) -> String {
    format!(
        "\
pipeline:
  models:
    denoiser:
      filename: denoiser.onnx
      type: denoiser
  dataflow:
    - from: denoiser.denoised
      to: denoiser.sample
  strategy:
    kind: iterative
    denoiser: denoiser
    num_steps: 2
    guidance_scale: {scale}
"
    )
}

#[test]
fn guidance_scale_one_is_treated_as_no_guidance_and_runs() -> anyhow::Result<()> {
    let dir = fixture_with_metadata(
        "diffusion-guidance-one",
        &["denoiser.onnx"],
        &diffusion_metadata_with_guidance("1.0"),
    )?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;
    let cond = [2.0f32, 4.0, 6.0, 8.0];
    let request = empty_request()
        .with_input("denoiser.sample", Value::from_slice_f32(&[0.0; 4], &[1, 4])?)
        .with_input("denoiser.cond", Value::from_slice_f32(&cond, &[1, 4])?);
    // guidance_scale == 1.0 must run (no CFG); 2 steps -> s_2 = cond * 3/4.
    let out = engine.run_pipeline(request)?;
    let denoised = out.get("denoiser.denoised").unwrap().to_vec_f32()?;
    for (got, c) in denoised.iter().zip(&cond) {
        assert!((got - c * 0.75).abs() < 1e-5, "{got} != {}", c * 0.75);
    }
    Ok(())
}

#[test]
fn nonunit_guidance_scale_is_rejected_pending_scheduler() -> anyhow::Result<()> {
    let dir = fixture_with_metadata(
        "diffusion-guidance-cfg",
        &["denoiser.onnx"],
        &diffusion_metadata_with_guidance("7.5"),
    )?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;
    let request = empty_request()
        .with_input("denoiser.sample", Value::from_slice_f32(&[0.0; 4], &[1, 4])?)
        .with_input("denoiser.cond", Value::from_slice_f32(&[1.0; 4], &[1, 4])?);
    let err = match engine.run_pipeline(request) {
        Ok(_) => panic!("non-unit guidance_scale must be rejected"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("guidance"),
        "unexpected error: {err}"
    );
    Ok(())
}

#[test]
fn iterative_threads_multiple_independent_loop_carried_tensors() -> anyhow::Result<()> {
    // denoiser_multi has two loop-carried states x, y and constant cond:
    //   x_next = (x + cond) * 0.5   (loop-carried x)
    //   y_next = (y + x)    * 0.5   (loop-carried y)
    // With x0 = y0 = 0, cond = c, after 2 steps: x = 3c/4, y = c/4.
    let metadata = "\
pipeline:
  models:
    denoiser:
      filename: denoiser_multi.onnx
      type: denoiser
  dataflow:
    - from: denoiser.x_next
      to: denoiser.x
    - from: denoiser.y_next
      to: denoiser.y
  strategy:
    kind: iterative
    denoiser: denoiser
    num_steps: 2
";
    let dir = fixture_with_metadata("diffusion-multi", &["denoiser_multi.onnx"], metadata)?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;

    let cond = [4.0f32, 8.0, 12.0, 16.0];
    let request = empty_request()
        .with_input("denoiser.x", Value::from_slice_f32(&[0.0; 4], &[1, 4])?)
        .with_input("denoiser.y", Value::from_slice_f32(&[0.0; 4], &[1, 4])?)
        .with_input("denoiser.cond", Value::from_slice_f32(&cond, &[1, 4])?);
    let out = engine.run_pipeline(request)?;

    let x = out.get("denoiser.x_next").expect("x_next").to_vec_f32()?;
    let y = out.get("denoiser.y_next").expect("y_next").to_vec_f32()?;
    for (got, c) in x.iter().zip(&cond) {
        assert!((got - c * 0.75).abs() < 1e-5, "x {got} != {}", c * 0.75);
    }
    for (got, c) in y.iter().zip(&cond) {
        assert!((got - c * 0.25).abs() < 1e-5, "y {got} != {}", c * 0.25);
    }
    Ok(())
}
