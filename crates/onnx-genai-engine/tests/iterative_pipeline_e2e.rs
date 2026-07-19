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
fn nonunit_guidance_without_cfg_input_is_rejected_at_load() -> anyhow::Result<()> {
    // guidance_scale != 1.0 requires a declared cfg_conditioning_input; the
    // metadata here omits it, so loading must fail with an actionable error.
    let dir = fixture_with_metadata(
        "diffusion-guidance-nocfg",
        &["denoiser.onnx"],
        &diffusion_metadata_with_guidance("7.5"),
    )?;
    let err = Engine::from_pipeline_dir(&dir, EngineConfig::default())
        .err()
        .expect("non-unit guidance without cfg_conditioning_input must be rejected");
    assert!(
        err.to_string().contains("cfg_conditioning_input"),
        "unexpected error: {err}"
    );
    Ok(())
}

#[test]
fn classifier_free_guidance_combines_conditional_and_unconditional() -> anyhow::Result<()> {
    // denoiser: denoised = (sample + cond) * 0.5.
    //   conditional (cond=c):  (sample + c) * 0.5
    //   unconditional (cond=0): sample * 0.5
    //   guided = uncond + scale*(cond - uncond) = 0.5*sample + 0.5*scale*c
    // With seed sample=0, scale=2: guided = c. Identity feedback publishes it.
    let metadata = "\
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
    num_steps: 1
    guidance_scale: 2.0
    cfg_conditioning_input: cond
";
    let dir = fixture_with_metadata("diffusion-cfg", &["denoiser.onnx"], metadata)?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;
    let cond = [1.0f32, 2.0, 3.0, 4.0];
    let request = empty_request()
        .with_input("denoiser.sample", Value::from_slice_f32(&[0.0; 4], &[1, 4])?)
        .with_input("denoiser.cond", Value::from_slice_f32(&cond, &[1, 4])?);
    let out = engine.run_pipeline(request)?;
    let denoised = out.get("denoiser.denoised").expect("guided output").to_vec_f32()?;
    for (got, c) in denoised.iter().zip(&cond) {
        assert!((got - c).abs() < 1e-5, "guided {got} != {c}");
    }
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

#[test]
fn iterative_injects_explicit_timestep_schedule() -> anyhow::Result<()> {
    // Step-aware denoiser: denoised = sample + t. With sample_0 = 0 and loop
    // edge denoised->sample, the result is the running sum of timesteps.
    let metadata = "\
pipeline:
  models:
    denoiser:
      filename: denoiser_step.onnx
      type: denoiser
  dataflow:
    - from: denoiser.denoised
      to: denoiser.sample
  strategy:
    kind: iterative
    denoiser: denoiser
    num_steps: 3
    timestep_input: t
    timesteps: [10.0, 20.0, 30.0]
";
    let dir = fixture_with_metadata("diffusion-timestep", &["denoiser_step.onnx"], metadata)?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;
    let request =
        empty_request().with_input("denoiser.sample", Value::from_slice_f32(&[0.0; 4], &[1, 4])?);
    let out = engine.run_pipeline(request)?;
    let denoised = out.get("denoiser.denoised").unwrap().to_vec_f32()?;
    // 10 + 20 + 30 = 60.
    for got in &denoised {
        assert!((got - 60.0).abs() < 1e-4, "{got} != 60");
    }
    Ok(())
}

#[test]
fn iterative_defaults_timestep_to_step_index() -> anyhow::Result<()> {
    // No `timesteps`: the loop injects the 0-based step index (0,1,2) -> sum 3.
    let metadata = "\
pipeline:
  models:
    denoiser:
      filename: denoiser_step.onnx
      type: denoiser
  dataflow:
    - from: denoiser.denoised
      to: denoiser.sample
  strategy:
    kind: iterative
    denoiser: denoiser
    num_steps: 3
    timestep_input: t
";
    let dir = fixture_with_metadata("diffusion-timestep-default", &["denoiser_step.onnx"], metadata)?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;
    let request =
        empty_request().with_input("denoiser.sample", Value::from_slice_f32(&[0.0; 4], &[1, 4])?);
    let out = engine.run_pipeline(request)?;
    let denoised = out.get("denoiser.denoised").unwrap().to_vec_f32()?;
    for got in &denoised {
        assert!((got - 3.0).abs() < 1e-4, "{got} != 3");
    }
    Ok(())
}

#[test]
fn iterative_dpmpp_2m_scheduler_runs_multistep_loop() -> anyhow::Result<()> {
    // DPM++ 2M is a multistep scheduler (order-2 uses the previous step's data
    // prediction). This exercises registration + the first-order (step 0) and
    // second-order (step 1+) branches + the lower-order final step, and checks
    // the loop produces a finite, correctly-shaped result. Numerical parity with
    // diffusers is covered by scripts/dpmpp_parity.py + scripts/dpmpp_e2e.py.
    let metadata = "\
pipeline:
  models:
    denoiser:
      filename: denoiser_step.onnx
      type: denoiser
  dataflow:
    - from: denoiser.denoised
      to: denoiser.sample
  strategy:
    kind: iterative
    denoiser: denoiser
    num_steps: 4
    timestep_input: t
    scheduler_config:
      kind: dpmpp_2m
      num_train_timesteps: 1000
      beta_start: 0.00085
      beta_end: 0.012
      beta_schedule: scaled_linear
";
    let dir = fixture_with_metadata("diffusion-dpmpp", &["denoiser_step.onnx"], metadata)?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;
    let request =
        empty_request().with_input("denoiser.sample", Value::from_slice_f32(&[0.1; 4], &[1, 4])?);
    let out = engine.run_pipeline(request)?;
    let sample = out.get("denoiser.sample").expect("scheduled sample").to_vec_f32()?;
    assert_eq!(sample.len(), 4);
    for got in &sample {
        assert!(got.is_finite(), "dpm++ produced non-finite value {got}");
    }
    Ok(())
}

#[test]
fn iterative_start_step_runs_partial_loop() -> anyhow::Result<()> {
    // img2img: start_step skips the earliest (noisiest) steps. denoiser_step
    // outputs `denoised = sample + t` with identity feedback (no scheduler).
    // timesteps=[10,20,30,40], start_step=2 runs only steps 2,3:
    //   seed 0 -> 0+30=30 -> 30+40=70   (vs the full loop's 100).
    let metadata = "\
pipeline:
  models:
    denoiser:
      filename: denoiser_step.onnx
      type: denoiser
  dataflow:
    - from: denoiser.denoised
      to: denoiser.sample
  strategy:
    kind: iterative
    denoiser: denoiser
    num_steps: 4
    start_step: 2
    timestep_input: t
    timesteps:
      - 10.0
      - 20.0
      - 30.0
      - 40.0
";
    let dir = fixture_with_metadata("diffusion-img2img", &["denoiser_step.onnx"], metadata)?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;
    let request =
        empty_request().with_input("denoiser.sample", Value::from_slice_f32(&[0.0; 4], &[1, 4])?);
    let out = engine.run_pipeline(request)?;
    let denoised = out.get("denoiser.denoised").unwrap().to_vec_f32()?;
    for got in &denoised {
        assert!((got - 70.0).abs() < 1e-4, "{got} != 70 (partial loop from start_step=2)");
    }
    Ok(())
}

#[test]
fn iterative_euler_ancestral_consumes_per_step_noise() -> anyhow::Result<()> {
    // Euler Ancestral is stochastic: the loop must supply a per-step noise tensor
    // `denoiser.sample.noise` shaped [num_steps, *sample_shape]. Zero noise makes
    // the step deterministic; verify it threads through and runs. Numerical parity
    // (with matched noise) is covered by scripts/euler_a_e2e.py.
    let metadata = "\
pipeline:
  models:
    denoiser:
      filename: denoiser_step.onnx
      type: denoiser
  dataflow:
    - from: denoiser.denoised
      to: denoiser.sample
  strategy:
    kind: iterative
    denoiser: denoiser
    num_steps: 3
    timestep_input: t
    scheduler_config:
      kind: euler_ancestral
      num_train_timesteps: 1000
      beta_start: 0.00085
      beta_end: 0.012
      beta_schedule: scaled_linear
";
    let dir = fixture_with_metadata("diffusion-euler-a", &["denoiser_step.onnx"], metadata)?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;
    let request = empty_request()
        .with_input("denoiser.sample", Value::from_slice_f32(&[0.1; 4], &[1, 4])?)
        .with_input("denoiser.sample.noise", Value::from_slice_f32(&[0.0; 12], &[3, 1, 4])?);
    let out = engine.run_pipeline(request)?;
    let sample = out.get("denoiser.sample").expect("scheduled sample").to_vec_f32()?;
    for got in &sample {
        assert!(got.is_finite(), "euler_ancestral produced non-finite value {got}");
    }
    Ok(())
}

#[test]
fn iterative_dpmpp_2m_karras_runs() -> anyhow::Result<()> {
    // use_karras_sigmas swaps in the Karras (rho=7) sigma schedule; verify it
    // threads through the schema and runs. Numerical parity is covered by
    // scripts/karras_parity.py + scripts/dpmpp_e2e.py (ONNX_GENAI_KARRAS=1).
    let metadata = "\
pipeline:
  models:
    denoiser:
      filename: denoiser_step.onnx
      type: denoiser
  dataflow:
    - from: denoiser.denoised
      to: denoiser.sample
  strategy:
    kind: iterative
    denoiser: denoiser
    num_steps: 5
    timestep_input: t
    scheduler_config:
      kind: dpmpp_2m
      num_train_timesteps: 1000
      beta_start: 0.00085
      beta_end: 0.012
      beta_schedule: scaled_linear
      use_karras_sigmas: true
";
    let dir = fixture_with_metadata("diffusion-dpmpp-karras", &["denoiser_step.onnx"], metadata)?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;
    let request =
        empty_request().with_input("denoiser.sample", Value::from_slice_f32(&[0.1; 4], &[1, 4])?);
    let out = engine.run_pipeline(request)?;
    let sample = out.get("denoiser.sample").expect("scheduled sample").to_vec_f32()?;
    for got in &sample {
        assert!(got.is_finite(), "dpm++ karras produced non-finite value {got}");
    }
    Ok(())
}

#[test]
fn iterative_euler_scheduler_scales_input_and_steps() -> anyhow::Result<()> {
    // denoiser_step outputs `denoised = sample + t`. Euler scales the loop input
    // by 1/sqrt(sigma^2+1) BEFORE the denoiser, so with t=0 the model output
    // (eps) = sample/sqrt(2). num_train=2, beta=0.5 (linear), num_steps=1 gives
    // sigmas=[1, 0]; step advances the RAW sample: x + eps*(0-1).
    //   scaled  = 1/sqrt(2)
    //   eps     = 1/sqrt(2)              (published under the output port)
    //   sample' = 1 + (1/sqrt(2))*(-1) = 1 - 1/sqrt(2)   (published under input port)
    let metadata = "\
pipeline:
  models:
    denoiser:
      filename: denoiser_step.onnx
      type: denoiser
  dataflow:
    - from: denoiser.denoised
      to: denoiser.sample
  strategy:
    kind: iterative
    denoiser: denoiser
    num_steps: 1
    timestep_input: t
    scheduler_config:
      kind: euler
      num_train_timesteps: 2
      beta_start: 0.5
      beta_end: 0.5
      beta_schedule: linear
";
    let dir = fixture_with_metadata("diffusion-euler", &["denoiser_step.onnx"], metadata)?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;
    let request =
        empty_request().with_input("denoiser.sample", Value::from_slice_f32(&[1.0; 4], &[1, 4])?);
    let out = engine.run_pipeline(request)?;

    let inv_sqrt2 = 1.0 / std::f32::consts::SQRT_2;
    let sample = out.get("denoiser.sample").expect("scheduled sample").to_vec_f32()?;
    let expected_sample = 1.0 - inv_sqrt2;
    for got in &sample {
        assert!((got - expected_sample).abs() < 1e-5, "sample {got} != {expected_sample}");
    }
    let eps = out.get("denoiser.denoised").expect("model output").to_vec_f32()?;
    for got in &eps {
        assert!((got - inv_sqrt2).abs() < 1e-5, "eps {got} != {inv_sqrt2}");
    }
    Ok(())
}

#[test]
fn cfg_conditioning_port_that_is_also_a_loop_input_is_rejected() -> anyhow::Result<()> {
    // If cfg_conditioning_input names a loop-carried input port, the
    // unconditional override would clobber the scheduler's loop sample. Reject
    // at load time.
    let metadata = "\
pipeline:
  models:
    denoiser:
      filename: denoiser_step.onnx
      type: denoiser
  dataflow:
    - from: denoiser.denoised
      to: denoiser.sample
  strategy:
    kind: iterative
    denoiser: denoiser
    num_steps: 2
    guidance_scale: 7.5
    cfg_conditioning_input: sample
";
    let dir = fixture_with_metadata("diffusion-cfg-collision", &["denoiser_step.onnx"], metadata)?;
    let err = Engine::from_pipeline_dir(&dir, EngineConfig::default())
        .err()
        .expect("cfg conditioning port colliding with a loop input must be rejected");
    assert!(
        err.to_string().contains("loop-carried input"),
        "unexpected: {err}"
    );
    Ok(())
}

#[test]
fn iterative_ddim_scheduler_transforms_loop_carried_sample() -> anyhow::Result<()> {
    // denoiser_step outputs `denoised = sample + t`. With timestep_input=t and
    // no explicit schedule, step 0 injects t=0, so the model output (treated as
    // the noise prediction eps) equals the input sample. DDIM(num_train=2,
    // beta=0.5, num_steps=1) then maps sample=1, eps=1 -> sqrt(2) - 1.
    let metadata = "\
pipeline:
  models:
    denoiser:
      filename: denoiser_step.onnx
      type: denoiser
  dataflow:
    - from: denoiser.denoised
      to: denoiser.sample
  strategy:
    kind: iterative
    denoiser: denoiser
    num_steps: 1
    timestep_input: t
    scheduler_config:
      kind: ddim
      num_train_timesteps: 2
      beta_start: 0.5
      beta_end: 0.5
";
    let dir = fixture_with_metadata("diffusion-ddim", &["denoiser_step.onnx"], metadata)?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;
    let request =
        empty_request().with_input("denoiser.sample", Value::from_slice_f32(&[1.0; 4], &[1, 4])?);
    let out = engine.run_pipeline(request)?;

    // Post-scheduler sample is published under the input-port key.
    let sample = out.get("denoiser.sample").expect("scheduled sample").to_vec_f32()?;
    let expected = std::f32::consts::SQRT_2 - 1.0;
    for got in &sample {
        assert!((got - expected).abs() < 1e-5, "sample {got} != {expected}");
    }
    // The raw model output (eps) is still published under the output-port key.
    let eps = out.get("denoiser.denoised").expect("model output").to_vec_f32()?;
    for got in &eps {
        assert!((got - 1.0).abs() < 1e-5, "eps {got} != 1.0");
    }
    Ok(())
}

#[test]
fn unsupported_scheduler_kind_is_rejected_at_load() -> anyhow::Result<()> {
    let metadata = "\
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
    scheduler_config:
      kind: no_such_scheduler
";
    let dir = fixture_with_metadata("diffusion-bad-scheduler", &["denoiser.onnx"], metadata)?;
    let err = Engine::from_pipeline_dir(&dir, EngineConfig::default())
        .err()
        .expect("unsupported scheduler kind must be rejected");
    assert!(err.to_string().contains("scheduler kind"), "unexpected: {err}");
    Ok(())
}

#[test]
fn real_dit_denoiser_runs_through_ddim_iterative_pipeline() -> anyhow::Result<()> {
    // A REAL diffusion-transformer denoiser built by Mobius
    // (scripts/build_tiny_dit_diffusion.py): patch-embed + AdaLN + self/cross
    // attention + FFN, with the standard contract
    //   sample[B,4,H,W] + timestep[B](int64) + encoder_hidden_states[B,S,16]
    //     -> noise_pred[B,4,H,W].
    // This exercises rank-4 latents, an INT64 timestep injection, and the DDIM
    // scheduler end-to-end (the metadata declares num_steps=3 + ddim).
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-dit-diffusion")
        .canonicalize()?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;

    let sample: Vec<f32> = (0..4 * 8 * 8).map(|i| (i as f32 % 7.0 - 3.0) * 0.1).collect();
    let ehs: Vec<f32> = (0..4 * 16).map(|i| (i as f32 % 5.0 - 2.0) * 0.1).collect();
    let request = empty_request()
        .with_input("denoiser.sample", Value::from_slice_f32(&sample, &[1, 4, 8, 8])?)
        .with_input(
            "denoiser.encoder_hidden_states",
            Value::from_slice_f32(&ehs, &[1, 4, 16])?,
        );

    let out = engine.run_pipeline(request)?;
    // Post-scheduler latent, published under the input-port key.
    let final_sample = out
        .get("denoiser.sample")
        .expect("scheduled latent")
        .to_vec_f32()?;
    assert_eq!(final_sample.len(), 4 * 8 * 8);
    assert!(
        final_sample.iter().all(|v| v.is_finite()),
        "DiT diffusion produced non-finite latent"
    );
    // The raw noise prediction is also published under the output-port key.
    let noise = out.get("denoiser.noise_pred").expect("noise_pred").to_vec_f32()?;
    assert_eq!(noise.len(), 4 * 8 * 8);
    Ok(())
}

#[test]
fn cfg_uses_caller_supplied_unconditional_conditioning() -> anyhow::Result<()> {
    // denoiser: denoised = (sample + cond) * 0.5.
    //   cond pass (cond=c):        (sample + c) * 0.5
    //   uncond pass (cond=u):      (sample + u) * 0.5   [supplied, NOT zeroed]
    //   guided = uncond + s*(cond-uncond) = 0.5*sample + 0.5*u + s*0.5*(c-u)
    // seed sample=0, c=2, u=1, s=2 -> 0.5 + 2*0.5 = 1.5  (zeros-fallback would give 2.0).
    let metadata = "\
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
    num_steps: 1
    guidance_scale: 2.0
    cfg_conditioning_input: cond
";
    let dir = fixture_with_metadata("diffusion-cfg-uncond", &["denoiser.onnx"], metadata)?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;
    let request = empty_request()
        .with_input("denoiser.sample", Value::from_slice_f32(&[0.0; 4], &[1, 4])?)
        .with_input("denoiser.cond", Value::from_slice_f32(&[2.0; 4], &[1, 4])?)
        .with_input("denoiser.cond.uncond", Value::from_slice_f32(&[1.0; 4], &[1, 4])?);
    let out = engine.run_pipeline(request)?;
    let denoised = out.get("denoiser.denoised").unwrap().to_vec_f32()?;
    for got in &denoised {
        assert!((got - 1.5).abs() < 1e-5, "{got} != 1.5 (uncond input not honored?)");
    }
    Ok(())
}

#[test]
fn masked_language_diffusion_refines_masked_sequence() -> anyhow::Result<()> {
    // Synthetic masked-diffusion LM (scripts/build_tiny_masked_diffusion.py):
    // fixed logits with argmax = [2,3,4,5] and decreasing confidence. The
    // masked_diffusion scheduler (mask_token_id=1, num_steps=4) unmasks one
    // highest-confidence position per step, refining the all-mask seed
    // [1,1,1,1] to [2,3,4,5].
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-masked-diffusion")
        .canonicalize()?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;
    let request = empty_request()
        .with_input("denoiser.input_ids", Value::from_slice_i64(&[1, 1, 1, 1], &[1, 4])?);
    let out = engine.run_pipeline(request)?;
    let tokens = out
        .get("denoiser.input_ids")
        .expect("refined token sequence")
        .to_vec_i64()?;
    assert_eq!(tokens, vec![2, 3, 4, 5]);
    Ok(())
}

#[test]
fn custom_user_scheduler_can_be_registered_and_run() -> anyhow::Result<()> {
    use onnx_genai_engine::{Scheduler, SchedulerRegistry};
    use onnx_genai_ort::Value as OrtValue;
    use std::sync::Arc;

    // A user-defined scheduler: next = sample + model_output (proves the trait
    // + registry are extensible without touching the engine).
    #[derive(Debug)]
    struct AddScheduler;
    impl Scheduler for AddScheduler {
        fn step(
            &self,
            _step: usize,
            _num_steps: usize,
            sample: &OrtValue,
            model_output: &OrtValue,
        ) -> anyhow::Result<OrtValue> {
            let s = sample.to_vec_f32()?;
            let m = model_output.to_vec_f32()?;
            let out: Vec<f32> = s.iter().zip(&m).map(|(a, b)| a + b).collect();
            Ok(OrtValue::from_slice_f32(&out, sample.shape())?)
        }
    }

    let metadata = "\
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
    scheduler_config:
      kind: my_adder
";
    let dir = fixture_with_metadata("diffusion-custom-sched", &["denoiser.onnx"], metadata)?;
    let mut registry = SchedulerRegistry::builtin();
    registry.register(
        "my_adder",
        Arc::new(|_cfg, _num_steps| Ok(Arc::new(AddScheduler) as Arc<dyn Scheduler>)),
    );
    let mut engine =
        Engine::from_pipeline_dir_with_schedulers(&dir, EngineConfig::default(), &registry)?;

    // denoiser: denoised = (sample + cond) * 0.5.  With cond=c, custom step is
    //   next = sample + denoised.   seed sample=0:
    //   step0: denoised=(0+c)/2=c/2; next=0+c/2=c/2
    //   step1: denoised=(c/2+c)/2=3c/4; next=c/2+3c/4=5c/4
    let cond = [4.0f32, 8.0, 12.0, 16.0];
    let request = empty_request()
        .with_input("denoiser.sample", Value::from_slice_f32(&[0.0; 4], &[1, 4])?)
        .with_input("denoiser.cond", Value::from_slice_f32(&cond, &[1, 4])?);
    let out = engine.run_pipeline(request)?;
    let sample = out.get("denoiser.sample").unwrap().to_vec_f32()?;
    for (got, c) in sample.iter().zip(&cond) {
        assert!((got - c * 1.25).abs() < 1e-5, "{got} != {}", c * 1.25);
    }
    Ok(())
}
