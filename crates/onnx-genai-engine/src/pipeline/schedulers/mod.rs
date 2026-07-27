//! Diffusion loop schedulers for the iterative pipeline.
//!
//! Houses the [`Scheduler`] trait, the [`SchedulerRegistry`] factory, the
//! shared `PredictionType` / model-output conversions, the sigma-schedule
//! helpers, and one submodule per built-in scheduler algorithm. Extracted
//! verbatim from `pipeline.rs`; no algorithm or numeric changes.

use anyhow::Context;
use onnx_genai_metadata::SchedulerSpec;
use onnx_genai_ort::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

mod ddim;
mod ddpm;
mod dpmpp;
mod euler;
mod flow_matching;
mod masked_diffusion;

use ddim::DdimSchedule;
use ddpm::DdpmSchedule;
use dpmpp::Dpmpp2m;
use euler::{EulerAncestral, EulerSchedule};
use flow_matching::FlowMatching;
use masked_diffusion::{MaskedDiffusion, Remasking};

fn mix_noise(
    original: &Value,
    noise: &Value,
    original_scale: f32,
    noise_scale: f32,
) -> anyhow::Result<Value> {
    if original.shape() != noise.shape() {
        anyhow::bail!(
            "original/noise shape mismatch: {:?} vs {:?}",
            original.shape(),
            noise.shape()
        );
    }
    let shape = original.shape().to_vec();
    let original = original.to_vec_f32_lossy()?;
    let noise = noise.to_vec_f32_lossy()?;
    let mixed: Vec<f32> = original
        .iter()
        .zip(noise)
        .map(|(&sample, noise)| original_scale * sample + noise_scale * noise)
        .collect();
    Value::from_slice_f32(&mixed, &shape).map_err(Into::into)
}

/// A loop-carried transform applied to a denoiser's output at each iterative
/// step. **Implement this trait to plug in a custom scheduler** and register it
/// with a [`SchedulerRegistry`]; the built-in `ddim` (continuous latents) and
/// `masked_diffusion` (discrete tokens) are just implementations.
///
/// `sample` is the value currently fed to the loop-carried input; `model_output`
/// is the denoiser's output this step. Return the next loop-carried value.
pub trait Scheduler: Send + Sync + std::fmt::Debug {
    fn step(
        &self,
        step: usize,
        num_steps: usize,
        sample: &Value,
        model_output: &Value,
    ) -> anyhow::Result<Value>;

    /// Reset any per-loop internal state (e.g. a multistep scheduler's previous
    /// prediction). Called once before each denoise loop. Default no-op.
    fn reset(&self) {}

    /// Whether this scheduler consumes fresh Gaussian noise each step (ancestral /
    /// stochastic samplers). When `true`, the loop supplies per-step noise via
    /// [`Scheduler::step_with_noise`]. Default `false` (deterministic).
    fn needs_noise(&self) -> bool {
        false
    }

    /// Like [`Scheduler::step`] but with the per-step noise an ancestral sampler
    /// needs. The default ignores `noise` and delegates to `step`, so existing
    /// deterministic schedulers are unaffected.
    fn step_with_noise(
        &self,
        step: usize,
        num_steps: usize,
        sample: &Value,
        model_output: &Value,
        _noise: Option<&Value>,
    ) -> anyhow::Result<Value> {
        self.step(step, num_steps, sample, model_output)
    }

    /// Per-step transform applied to the loop-carried input BEFORE the denoiser
    /// (e.g. Euler's `sample / sqrt(sigma^2 + 1)`). `Ok(None)` = identity (the
    /// denoiser sees the raw loop-carried value, as DDIM requires).
    fn scale_input(
        &self,
        _step: usize,
        _num_steps: usize,
        _sample: &Value,
    ) -> anyhow::Result<Option<Value>> {
        Ok(None)
    }

    /// The factor by which the caller must scale the initial random latent so it
    /// lives in this scheduler's sigma space. Sigma-space samplers (Euler,
    /// Euler-Ancestral) return their maximum sigma (`sigmas[0]`); DDIM and
    /// DPM-Solver++ leave the seed unscaled and return `1.0` (the default).
    fn init_noise_sigma(&self) -> f32 {
        1.0
    }

    /// Noise an encoded sample to the scheduler state used at `step`.
    fn add_noise(
        &self,
        _step: usize,
        _num_steps: usize,
        _original: &Value,
        _noise: &Value,
    ) -> anyhow::Result<Value> {
        anyhow::bail!("this scheduler does not support continuous-latent noise initialization")
    }

    /// The per-step denoiser timesteps this scheduler feeds to the model, matching
    /// the diffusers scheduler it emulates (e.g. `[999.0, 966.0, ..., 33.0]` for a
    /// 30-step DPM-Solver++ linspace schedule). Length equals the loop step count.
    ///
    /// The pipeline uses these when the plan does not carry an explicit
    /// `strategy.timesteps` schedule, so from-scratch packages that omit the
    /// timestep table still drive the denoiser with the correct diffusion
    /// timesteps rather than the raw `0..num_steps` step index. Returns `None`
    /// for schedulers with no meaningful timestep (e.g. discrete token diffusion),
    /// leaving the pipeline to fall back to the step index.
    fn timesteps(&self) -> Option<Vec<f32>> {
        None
    }

    /// Build the unconditional loop-carried sample for classifier-free guidance
    /// from the current (conditional) one, when the guidance direction is a
    /// transform of the loop state rather than a separate conditioning input.
    ///
    /// Discrete language diffusion (LLaDA) forms its unconditional pass by
    /// re-masking the prompt tokens of the current sequence (`un_x[prompt] =
    /// mask_id`); the pipeline feeds the returned value as the denoiser's
    /// loop-carried input on the unconditional pass. Continuous (image)
    /// schedulers return `None` (their unconditional direction comes from a
    /// zeroed / `.uncond` conditioning input instead).
    fn cfg_uncond_sample(&self, _sample: &Value) -> anyhow::Result<Option<Value>> {
        Ok(None)
    }
}

/// Builds a [`Scheduler`] from a declared [`SchedulerSpec`] and the loop length.
pub type SchedulerFactory =
    Arc<dyn Fn(&SchedulerSpec, usize) -> anyhow::Result<Arc<dyn Scheduler>> + Send + Sync>;

/// Registry mapping a `scheduler_config.kind` string to a factory. Users extend
/// it with [`register`](Self::register) to support their own schedulers, then
/// load a pipeline via [`PipelineEngine::from_pipeline_dir_with_schedulers`].
#[derive(Clone)]
pub struct SchedulerRegistry {
    factories: HashMap<String, SchedulerFactory>,
}

impl std::fmt::Debug for SchedulerRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchedulerRegistry")
            .field("kinds", &self.factories.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl SchedulerRegistry {
    /// Registry with the built-in diffusion, flow-matching, and masked-token schedulers.
    pub fn builtin() -> Self {
        let mut factories: HashMap<String, SchedulerFactory> = HashMap::new();
        factories.insert(
            "ddpm".to_string(),
            Arc::new(|cfg: &SchedulerSpec, num_steps: usize| {
                let prediction = PredictionType::parse(cfg.prediction_type.as_deref())?;
                let sched = DdpmSchedule::with_schedule(
                    cfg.num_train_timesteps.unwrap_or(1000),
                    cfg.beta_start.unwrap_or(0.0001),
                    cfg.beta_end.unwrap_or(0.02),
                    cfg.beta_schedule.as_deref().unwrap_or("linear"),
                    num_steps,
                )?
                .with_prediction(prediction);
                Ok(Arc::new(sched) as Arc<dyn Scheduler>)
            }),
        );
        factories.insert(
            "ddim".to_string(),
            Arc::new(|cfg: &SchedulerSpec, num_steps: usize| {
                let prediction = PredictionType::parse(cfg.prediction_type.as_deref())?;
                let sched = DdimSchedule::with_schedule(
                    cfg.num_train_timesteps.unwrap_or(1000),
                    cfg.beta_start.unwrap_or(0.00085),
                    cfg.beta_end.unwrap_or(0.012),
                    cfg.beta_schedule.as_deref().unwrap_or("linear"),
                    num_steps,
                )?
                .with_prediction(prediction);
                Ok(Arc::new(sched) as Arc<dyn Scheduler>)
            }),
        );
        factories.insert(
            "euler".to_string(),
            Arc::new(|cfg: &SchedulerSpec, num_steps: usize| {
                let prediction = PredictionType::parse(cfg.prediction_type.as_deref())?;
                let sched = EulerSchedule::with_schedule(
                    cfg.num_train_timesteps.unwrap_or(1000),
                    cfg.beta_start.unwrap_or(0.00085),
                    cfg.beta_end.unwrap_or(0.012),
                    cfg.beta_schedule.as_deref().unwrap_or("scaled_linear"),
                    num_steps,
                    sigma_spacing(cfg)?,
                )?
                .with_prediction(prediction);
                Ok(Arc::new(sched) as Arc<dyn Scheduler>)
            }),
        );
        factories.insert(
            "euler_ancestral".to_string(),
            Arc::new(|cfg: &SchedulerSpec, num_steps: usize| {
                let prediction = PredictionType::parse(cfg.prediction_type.as_deref())?;
                let sched = EulerAncestral::with_schedule(
                    cfg.num_train_timesteps.unwrap_or(1000),
                    cfg.beta_start.unwrap_or(0.00085),
                    cfg.beta_end.unwrap_or(0.012),
                    cfg.beta_schedule.as_deref().unwrap_or("scaled_linear"),
                    num_steps,
                    sigma_spacing(cfg)?,
                )?
                .with_prediction(prediction);
                Ok(Arc::new(sched) as Arc<dyn Scheduler>)
            }),
        );
        factories.insert(
            "dpmpp_2m".to_string(),
            Arc::new(|cfg: &SchedulerSpec, num_steps: usize| {
                let prediction = PredictionType::parse(cfg.prediction_type.as_deref())?;
                let sched = Dpmpp2m::with_schedule(
                    cfg.num_train_timesteps.unwrap_or(1000),
                    cfg.beta_start.unwrap_or(0.00085),
                    cfg.beta_end.unwrap_or(0.012),
                    cfg.beta_schedule.as_deref().unwrap_or("scaled_linear"),
                    num_steps,
                    sigma_spacing(cfg)?,
                )?
                .with_prediction(prediction);
                Ok(Arc::new(sched) as Arc<dyn Scheduler>)
            }),
        );
        factories.insert(
            "flow_matching".to_string(),
            Arc::new(|cfg: &SchedulerSpec, num_steps: usize| {
                if let Some(prediction) = cfg.prediction_type.as_deref()
                    && !prediction.is_empty()
                    && prediction != "flow"
                    && prediction != "velocity"
                {
                    anyhow::bail!(
                        "flow_matching prediction_type must be omitted or 'flow'/'velocity', got '{prediction}'"
                    );
                }
                let sched = FlowMatching::with_schedule(
                    cfg.num_train_timesteps.unwrap_or(1000),
                    num_steps,
                    cfg.shift.unwrap_or(1.0),
                )?;
                Ok(Arc::new(sched) as Arc<dyn Scheduler>)
            }),
        );
        factories.insert(
            "masked_diffusion".to_string(),
            Arc::new(|cfg: &SchedulerSpec, _num_steps: usize| {
                let mask_token_id = cfg
                    .mask_token_id
                    .context("masked_diffusion scheduler requires 'mask_token_id'")?;
                let temperature = cfg.temperature.unwrap_or(0.0);
                if temperature < 0.0 {
                    anyhow::bail!("masked_diffusion temperature must be >= 0");
                }
                if let Some(block_length) = cfg.block_length
                    && block_length == 0
                {
                    anyhow::bail!("masked_diffusion block_length must be >= 1");
                }
                let remasking = match cfg.remasking.as_deref() {
                    None | Some("low_confidence") => Remasking::LowConfidence,
                    Some("random") => Remasking::Random,
                    Some(other) => anyhow::bail!(
                        "masked_diffusion remasking must be 'low_confidence' or 'random', \
                         got '{other}'"
                    ),
                };
                Ok(Arc::new(MaskedDiffusion {
                    mask_token_id,
                    temperature,
                    block_length: cfg.block_length,
                    remasking,
                    generation_start: Mutex::new(None),
                }) as Arc<dyn Scheduler>)
            }),
        );
        Self { factories }
    }

    /// Register (or override) a scheduler kind with a factory.
    pub fn register(&mut self, kind: impl Into<String>, factory: SchedulerFactory) {
        self.factories.insert(kind.into(), factory);
    }

    pub(crate) fn build(
        &self,
        spec: &SchedulerSpec,
        num_steps: usize,
    ) -> anyhow::Result<Arc<dyn Scheduler>> {
        let factory = self.factories.get(&spec.kind).with_context(|| {
            format!(
                "unknown scheduler kind '{}' (registered: {:?})",
                spec.kind,
                self.factories.keys().collect::<Vec<_>>()
            )
        })?;
        factory(spec, num_steps)
    }
}

impl Default for SchedulerRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

/// The parameterization the denoiser was trained with, i.e. what the model's
/// raw output represents. All built-in schedulers internally drive an
/// epsilon/x0 update, so a model output in any of these parameterizations is
/// first converted to epsilon (and/or x0) via [`epsilon_from_model_output`] /
/// [`x0_from_model_output`] before the existing step math runs. This keeps the
/// `epsilon` path byte-identical while adding `v_prediction` and `sample`/`x0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PredictionType {
    /// The model predicts the noise `epsilon` (Stable Diffusion 1.x default).
    Epsilon,
    /// The model predicts the velocity `v` (Salimans & Ho 2022; SD 2.x, SDXL
    /// refiner, many fine-tunes).
    VPrediction,
    /// The model predicts the clean sample `x0` directly (`sample`/`x0`).
    Sample,
}

impl PredictionType {
    /// Parse a scheduler/diffusion-config `prediction_type` string. `None` and
    /// the empty string default to `epsilon` (diffusers' default). Both the
    /// diffusers spelling `sample` and the common alias `x0` map to
    /// [`PredictionType::Sample`].
    fn parse(value: Option<&str>) -> anyhow::Result<Self> {
        match value.unwrap_or("epsilon") {
            "epsilon" => Ok(Self::Epsilon),
            "v_prediction" => Ok(Self::VPrediction),
            "sample" | "x0" => Ok(Self::Sample),
            other => anyhow::bail!(
                "unsupported prediction_type '{other}' (expected 'epsilon', 'v_prediction', or 'sample'/'x0')"
            ),
        }
    }
}

/// Convert one model-output element to the epsilon (noise) the epsilon-form step
/// math expects, given the DDPM per-timestep `alpha_t = sqrt(alpha_cumprod_t)`
/// and `sigma_t = sqrt(1 - alpha_cumprod_t)` and the current noisy sample `x_t`.
///
/// Matches diffusers' `DDIMScheduler` conventions
/// (`beta_prod_t = 1 - alpha_prod_t`, `sigma_t = beta_prod_t ** 0.5`):
/// - `epsilon`:      `eps = model_out`                       (identity)
/// - `v_prediction`: `eps = alpha_t * model_out + sigma_t * x_t`
/// - `sample`/`x0`:  `eps = (x_t - alpha_t * model_out) / sigma_t`
#[inline]
pub(crate) fn epsilon_from_model_output(
    model_out: f32,
    x_t: f32,
    alpha_t: f32,
    sigma_t: f32,
    prediction: PredictionType,
) -> f32 {
    match prediction {
        PredictionType::Epsilon => model_out,
        PredictionType::VPrediction => alpha_t * model_out + sigma_t * x_t,
        PredictionType::Sample => (x_t - alpha_t * model_out) / sigma_t,
    }
}

/// Convert one model-output element to the clean-sample estimate `x0`
/// (`pred_original_sample`), given the DDPM per-timestep `alpha_t` / `sigma_t`
/// (see [`epsilon_from_model_output`]) and the current noisy sample `x_t`.
///
/// Matches diffusers' `DPMSolverMultistepScheduler.convert_model_output`:
/// - `epsilon`:      `x0 = (x_t - sigma_t * model_out) / alpha_t`
/// - `v_prediction`: `x0 = alpha_t * x_t - sigma_t * model_out`
/// - `sample`/`x0`:  `x0 = model_out`                        (identity)
#[inline]
pub(crate) fn x0_from_model_output(
    model_out: f32,
    x_t: f32,
    alpha_t: f32,
    sigma_t: f32,
    prediction: PredictionType,
) -> f32 {
    match prediction {
        PredictionType::Epsilon => (x_t - sigma_t * model_out) / alpha_t,
        PredictionType::VPrediction => alpha_t * x_t - sigma_t * model_out,
        PredictionType::Sample => model_out,
    }
}

/// `alpha_t = 1/sqrt(sigma^2+1)`, `sigma_t = sigma * alpha_t` (diffusers convention).
pub(crate) fn dpm_alpha_sigma(sigma: f32) -> (f32, f32) {
    let alpha_t = 1.0 / (sigma * sigma + 1.0).sqrt();
    (alpha_t, sigma * alpha_t)
}

/// Training sigmas `((1-alpha_cumprod)/alpha_cumprod)^0.5` over the beta schedule.
pub(crate) fn training_sigmas(
    num_train_timesteps: usize,
    beta_start: f32,
    beta_end: f32,
    beta_schedule: &str,
) -> anyhow::Result<Vec<f32>> {
    Ok(
        training_alpha_cumprod(num_train_timesteps, beta_start, beta_end, beta_schedule)?
            .into_iter()
            .map(|alpha| ((1.0 - alpha) / alpha).sqrt())
            .collect(),
    )
}

/// Cumulative alpha products for the configured beta schedule.
pub(crate) fn training_alpha_cumprod(
    num_train_timesteps: usize,
    beta_start: f32,
    beta_end: f32,
    beta_schedule: &str,
) -> anyhow::Result<Vec<f32>> {
    if num_train_timesteps < 2 {
        anyhow::bail!("scheduler num_train_timesteps must be >= 2");
    }
    let denom = (num_train_timesteps - 1) as f32;
    let (lo, hi, square) = match beta_schedule {
        "linear" => (beta_start, beta_end, false),
        "scaled_linear" => (beta_start.sqrt(), beta_end.sqrt(), true),
        other => anyhow::bail!(
            "unsupported scheduler beta_schedule '{other}' (expected 'linear' or 'scaled_linear')"
        ),
    };
    let mut out = Vec::with_capacity(num_train_timesteps);
    let mut prod = 1.0f32;
    for i in 0..num_train_timesteps {
        let mut beta = lo + (hi - lo) * (i as f32) / denom;
        if square {
            beta *= beta;
        }
        prod *= 1.0 - beta;
        out.push(prod);
    }
    Ok(out)
}

/// Interpolate a diffusion timestep from a sigma, matching diffusers
/// `SchedulerMixin._sigma_to_t`. `train` holds the ascending training sigmas
/// (index = training timestep). Finds the bracketing training sigmas in log space
/// and linearly interpolates the (fractional) timestep. Used to recover the
/// denoiser timesteps for sigma-space schedules (Karras / exponential) where the
/// timesteps are not the sigma indices.
pub(crate) fn sigma_to_t(train: &[f32], sigma: f32) -> f32 {
    let log_sigma = sigma.max(1e-10).ln();
    let count = train
        .iter()
        .filter(|&&s| s.max(1e-10).ln() <= log_sigma)
        .count();
    let low_idx = count.saturating_sub(1).min(train.len().saturating_sub(2));
    let high_idx = low_idx + 1;
    let low = train[low_idx].max(1e-10).ln();
    let high = train[high_idx].max(1e-10).ln();
    let weight = ((low - log_sigma) / (low - high)).clamp(0.0, 1.0);
    (1.0 - weight) * low_idx as f32 + weight * high_idx as f32
}

/// Karras (rho=7) sigma schedule from the training sigma range, descending, with
/// a trailing `0.0`. Length `num_steps + 1`. Matches diffusers `_convert_to_karras`
/// (identical for Euler and DPM++ since both derive min/max from the full range).
fn karras_sigmas(
    num_train_timesteps: usize,
    beta_start: f32,
    beta_end: f32,
    beta_schedule: &str,
    num_steps: usize,
) -> anyhow::Result<Vec<f32>> {
    const RHO: f32 = 7.0;
    let train = training_sigmas(num_train_timesteps, beta_start, beta_end, beta_schedule)?;
    let sigma_min = train[0];
    let sigma_max = train[num_train_timesteps - 1];
    let min_inv = sigma_min.powf(1.0 / RHO);
    let max_inv = sigma_max.powf(1.0 / RHO);
    let mut sigmas = Vec::with_capacity(num_steps + 1);
    for k in 0..num_steps {
        let ramp = if num_steps > 1 {
            k as f32 / (num_steps - 1) as f32
        } else {
            0.0
        };
        sigmas.push((max_inv + ramp * (min_inv - max_inv)).powf(RHO));
    }
    sigmas.push(0.0);
    Ok(sigmas)
}

/// Exponential sigma schedule: `exp(linspace(log(sigma_max), log(sigma_min), n))`,
/// descending, trailing `0.0`. Same training-sigma min/max as Karras. Matches
/// diffusers `_convert_to_exponential`.
fn exponential_sigmas(
    num_train_timesteps: usize,
    beta_start: f32,
    beta_end: f32,
    beta_schedule: &str,
    num_steps: usize,
) -> anyhow::Result<Vec<f32>> {
    let train = training_sigmas(num_train_timesteps, beta_start, beta_end, beta_schedule)?;
    let log_min = train[0].ln();
    let log_max = train[num_train_timesteps - 1].ln();
    let mut sigmas = Vec::with_capacity(num_steps + 1);
    for k in 0..num_steps {
        let ramp = if num_steps > 1 {
            k as f32 / (num_steps - 1) as f32
        } else {
            0.0
        };
        sigmas.push((log_max + ramp * (log_min - log_max)).exp());
    }
    sigmas.push(0.0);
    Ok(sigmas)
}

/// Select the sigma schedule the spec requests: `"karras"`, `"exponential"`, or
/// the default `"linspace"`. Rejects the conflicting case where both Karras and
/// exponential are requested.
fn sigma_spacing(cfg: &SchedulerSpec) -> anyhow::Result<&'static str> {
    let karras = cfg.use_karras_sigmas.unwrap_or(false);
    let exponential = cfg.use_exponential_sigmas.unwrap_or(false);
    if karras && exponential {
        anyhow::bail!("scheduler cannot set both use_karras_sigmas and use_exponential_sigmas");
    }
    Ok(if karras {
        "karras"
    } else if exponential {
        "exponential"
    } else {
        "linspace"
    })
}

/// Precomputed sigmas for a non-linspace spacing (`karras`/`exponential`), or
/// `None` for the default `linspace` (which each scheduler builds itself).
pub(crate) fn spacing_sigmas(
    spacing: &str,
    num_train_timesteps: usize,
    beta_start: f32,
    beta_end: f32,
    beta_schedule: &str,
    num_steps: usize,
) -> anyhow::Result<Option<Vec<f32>>> {
    match spacing {
        "karras" => Ok(Some(karras_sigmas(
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
            num_steps,
        )?)),
        "exponential" => Ok(Some(exponential_sigmas(
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
            num_steps,
        )?)),
        "linspace" | "" => Ok(None),
        other => anyhow::bail!("unsupported sigma spacing '{other}' (karras/exponential/linspace)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prediction_type_parse_accepts_known_aliases() {
        assert_eq!(
            PredictionType::parse(None).unwrap(),
            PredictionType::Epsilon
        );
        assert_eq!(
            PredictionType::parse(Some("epsilon")).unwrap(),
            PredictionType::Epsilon
        );
        assert_eq!(
            PredictionType::parse(Some("v_prediction")).unwrap(),
            PredictionType::VPrediction
        );
        assert_eq!(
            PredictionType::parse(Some("sample")).unwrap(),
            PredictionType::Sample
        );
        assert_eq!(
            PredictionType::parse(Some("x0")).unwrap(),
            PredictionType::Sample
        );
        assert!(PredictionType::parse(Some("nonsense")).is_err());
    }

    #[test]
    fn model_output_conversion_matches_diffusers_formulas() {
        // Representative mid-timestep with alpha_cumprod = 0.5 so
        // alpha_t = sqrt(0.5) and sigma_t = sqrt(1 - 0.5) = sqrt(0.5).
        let alpha_t = 0.5f32.sqrt();
        let sigma_t = (1.0f32 - 0.5).sqrt();
        let x_t = 0.7f32;
        let model_out = 0.3f32;

        // epsilon: both conversions reduce to the diffusers closed form and the
        // epsilon path is the identity.
        assert!(
            (epsilon_from_model_output(model_out, x_t, alpha_t, sigma_t, PredictionType::Epsilon)
                - model_out)
                .abs()
                < 1e-6
        );
        let x0_eps =
            x0_from_model_output(model_out, x_t, alpha_t, sigma_t, PredictionType::Epsilon);
        assert!((x0_eps - (x_t - sigma_t * model_out) / alpha_t).abs() < 1e-6);

        // v_prediction: diffusers DDIMScheduler v_prediction branch.
        //   pred_epsilon = alpha_t * model_out + sigma_t * x_t
        //   pred_x0      = alpha_t * x_t - sigma_t * model_out
        let eps_v = epsilon_from_model_output(
            model_out,
            x_t,
            alpha_t,
            sigma_t,
            PredictionType::VPrediction,
        );
        assert!((eps_v - (alpha_t * model_out + sigma_t * x_t)).abs() < 1e-6);
        let x0_v = x0_from_model_output(
            model_out,
            x_t,
            alpha_t,
            sigma_t,
            PredictionType::VPrediction,
        );
        assert!((x0_v - (alpha_t * x_t - sigma_t * model_out)).abs() < 1e-6);
        // Internal consistency: eps and x0 satisfy x_t = alpha_t*x0 + sigma_t*eps.
        assert!((alpha_t * x0_v + sigma_t * eps_v - x_t).abs() < 1e-6);

        // sample / x0: model_out IS x0, and eps = (x_t - alpha_t*x0)/sigma_t.
        let x0_s = x0_from_model_output(model_out, x_t, alpha_t, sigma_t, PredictionType::Sample);
        assert!((x0_s - model_out).abs() < 1e-6);
        let eps_s =
            epsilon_from_model_output(model_out, x_t, alpha_t, sigma_t, PredictionType::Sample);
        assert!((eps_s - (x_t - alpha_t * model_out) / sigma_t).abs() < 1e-6);
        assert!((alpha_t * x0_s + sigma_t * eps_s - x_t).abs() < 1e-6);
    }

    #[test]
    fn model_output_conversion_endpoints() {
        // alpha_cumprod -> 1 (t = 0): alpha_t = 1, sigma_t = 0. eps == model_out
        // for epsilon and v_prediction (the sigma_t term vanishes); x0 == x_t for
        // v_prediction.
        let (alpha_t, sigma_t) = (1.0f32, 0.0f32);
        let x_t = 0.9f32;
        let model_out = -0.4f32;
        assert!(
            (epsilon_from_model_output(
                model_out,
                x_t,
                alpha_t,
                sigma_t,
                PredictionType::VPrediction
            ) - model_out)
                .abs()
                < 1e-6
        );
        assert!(
            (x0_from_model_output(
                model_out,
                x_t,
                alpha_t,
                sigma_t,
                PredictionType::VPrediction
            ) - x_t)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn v_prediction_epsilon_path_stays_byte_identical() {
        // A scheduler built with prediction_type=epsilon must produce EXACTLY the
        // same bytes as before the v-prediction change (regression guard).
        let registry = SchedulerRegistry::builtin();
        for kind in ["ddim", "euler", "dpmpp_2m"] {
            let eps_spec = SchedulerSpec {
                kind: kind.to_string(),
                prediction_type: Some("epsilon".to_string()),
                ..SchedulerSpec::default()
            };
            let default_spec = SchedulerSpec {
                kind: kind.to_string(),
                prediction_type: None,
                ..SchedulerSpec::default()
            };
            let eps = registry.build(&eps_spec, 6).expect("epsilon builds");
            let dflt = registry.build(&default_spec, 6).expect("default builds");
            let sample = Value::from_slice_f32(&[0.1, -0.2, 0.3, 0.4], &[1, 4]).unwrap();
            let model_out = Value::from_slice_f32(&[0.5, 0.6, -0.7, 0.8], &[1, 4]).unwrap();
            eps.reset();
            dflt.reset();
            let a = eps.step(0, 6, &sample, &model_out).unwrap();
            let b = dflt.step(0, 6, &sample, &model_out).unwrap();
            assert_eq!(
                a.to_vec_f32_lossy().unwrap(),
                b.to_vec_f32_lossy().unwrap(),
                "{kind}: explicit epsilon must equal the default"
            );
        }
    }

    #[test]
    fn v_prediction_schedulers_construct_and_step_finite() {
        // Previously each scheduler hard-rejected v_prediction. Now they must
        // construct and produce finite, shape-preserving latent updates over a
        // couple of synthetic steps with random-ish model output.
        let registry = SchedulerRegistry::builtin();
        let num_steps = 4usize;
        let shape = [1i64, 2, 2, 2];
        let elems = 8usize;
        for kind in ["ddpm", "ddim", "euler", "euler_ancestral", "dpmpp_2m"] {
            for prediction in ["v_prediction", "x0", "sample"] {
                let spec = SchedulerSpec {
                    kind: kind.to_string(),
                    prediction_type: Some(prediction.to_string()),
                    num_train_timesteps: Some(1000),
                    ..SchedulerSpec::default()
                };
                let sched = registry
                    .build(&spec, num_steps)
                    .unwrap_or_else(|e| panic!("{kind}/{prediction} must construct: {e}"));
                sched.reset();
                let init = sched.init_noise_sigma();
                let mut latent: Vec<f32> = (0..elems)
                    .map(|i| ((i as f32 * 0.37).sin()) * init)
                    .collect();
                for step in 0..num_steps {
                    let sample = Value::from_slice_f32(&latent, &shape).unwrap();
                    let model_out: Vec<f32> = (0..elems)
                        .map(|i| ((step as f32 + 1.0) * 0.11 + i as f32 * 0.19).cos())
                        .collect();
                    let model_value = Value::from_slice_f32(&model_out, &shape).unwrap();
                    let noise = Value::from_slice_f32(
                        &(0..elems)
                            .map(|i| ((i as f32 + step as f32) * 0.53).sin())
                            .collect::<Vec<_>>(),
                        &shape,
                    )
                    .unwrap();
                    let next = sched
                        .step_with_noise(step, num_steps, &sample, &model_value, Some(&noise))
                        .unwrap_or_else(|e| panic!("{kind}/{prediction} step {step}: {e}"));
                    assert_eq!(next.shape(), shape, "{kind}/{prediction} preserves shape");
                    latent = next.to_vec_f32_lossy().unwrap();
                    for (i, v) in latent.iter().enumerate() {
                        assert!(
                            v.is_finite(),
                            "{kind}/{prediction} step {step} elem {i} non-finite: {v}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn registry_accepts_modern_prediction_contracts() {
        let registry = SchedulerRegistry::builtin();
        for prediction in ["epsilon", "v_prediction", "sample", "x0"] {
            let spec = SchedulerSpec {
                kind: "ddpm".to_string(),
                prediction_type: Some(prediction.to_string()),
                num_train_timesteps: Some(8),
                ..SchedulerSpec::default()
            };
            registry
                .build(&spec, 4)
                .unwrap_or_else(|error| panic!("ddpm/{prediction} must build: {error}"));
        }
        for prediction in [None, Some("flow"), Some("velocity")] {
            let spec = SchedulerSpec {
                kind: "flow_matching".to_string(),
                prediction_type: prediction.map(str::to_string),
                shift: Some(3.0),
                ..SchedulerSpec::default()
            };
            registry
                .build(&spec, 4)
                .unwrap_or_else(|error| panic!("flow_matching/{prediction:?} must build: {error}"));
        }
        let invalid = SchedulerSpec {
            kind: "flow_matching".to_string(),
            prediction_type: Some("epsilon".to_string()),
            ..SchedulerSpec::default()
        };
        assert!(registry.build(&invalid, 4).is_err());
    }
}
