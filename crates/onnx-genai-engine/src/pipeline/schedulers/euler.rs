//! Euler and Euler-Ancestral sigma-space schedulers.
//!
//! `EulerDiscreteScheduler` (deterministic) and its ancestral (stochastic)
//! variant. Extracted verbatim from `pipeline.rs`.

use anyhow::Context;
use onnx_genai_ort::Value;

use super::{
    PredictionType, Scheduler, epsilon_from_model_output, sigma_to_t, spacing_sigmas,
    training_sigmas,
};

/// Euler (`EulerDiscreteScheduler`, epsilon prediction) — a sigma-space
/// scheduler. Unlike DDIM it rescales the loop-carried sample before the
/// denoiser (`x / sqrt(sigma^2 + 1)`), then advances the *raw* sample along the
/// noise derivative: `x_next = x + eps * (sigma_next - sigma)`. Matches diffusers
/// `EulerDiscreteScheduler(timestep_spacing="linspace", interpolation_type="linear")`.
/// The initial seed must be pre-scaled by `init_noise_sigma` (= `sigmas[0]`).
#[derive(Debug, Clone)]
pub(super) struct EulerSchedule {
    /// Inference sigmas, descending, with a trailing `0.0`. Length `num_steps + 1`.
    sigmas: Vec<f32>,
    /// Per-step denoiser timesteps (fractional), length `num_steps`.
    timesteps: Vec<f32>,
    /// Model parameterization (`epsilon` by default).
    prediction: PredictionType,
}

impl EulerSchedule {
    pub(super) fn with_schedule(
        num_train_timesteps: usize,
        beta_start: f32,
        beta_end: f32,
        beta_schedule: &str,
        num_steps: usize,
        spacing: &str,
    ) -> anyhow::Result<Self> {
        if num_train_timesteps < 2 {
            anyhow::bail!("scheduler num_train_timesteps must be >= 2");
        }
        if num_steps == 0 || num_steps > num_train_timesteps {
            anyhow::bail!("scheduler num_steps ({num_steps}) must be in 1..={num_train_timesteps}");
        }
        if let Some(sigmas) = spacing_sigmas(
            spacing,
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
            num_steps,
        )? {
            let train = training_sigmas(num_train_timesteps, beta_start, beta_end, beta_schedule)?;
            let timesteps = sigmas[..num_steps]
                .iter()
                .map(|&s| sigma_to_t(&train, s))
                .collect();
            return Ok(Self {
                sigmas,
                timesteps,
                prediction: PredictionType::Epsilon,
            });
        }
        let denom = (num_train_timesteps - 1) as f32;
        let (lo, hi, square) = match beta_schedule {
            "linear" => (beta_start, beta_end, false),
            "scaled_linear" => (beta_start.sqrt(), beta_end.sqrt(), true),
            other => anyhow::bail!(
                "unsupported scheduler beta_schedule '{other}' (expected 'linear' or 'scaled_linear')"
            ),
        };
        // Training sigmas: sigma_i = sqrt((1 - alpha_cumprod_i) / alpha_cumprod_i).
        let mut train_sigmas = Vec::with_capacity(num_train_timesteps);
        let mut prod = 1.0f32;
        for i in 0..num_train_timesteps {
            let mut beta = lo + (hi - lo) * (i as f32) / denom;
            if square {
                beta *= beta;
            }
            prod *= 1.0 - beta;
            train_sigmas.push(((1.0 - prod) / prod).sqrt());
        }
        // "linspace" timesteps: evenly spaced over [0, N-1], taken descending,
        // with sigmas linearly interpolated at each (fractional) timestep.
        let ts_denom = if num_steps > 1 {
            (num_steps - 1) as f32
        } else {
            1.0
        };
        let interp = |t: f32| -> f32 {
            let low = t.floor().max(0.0) as usize;
            let high = (low + 1).min(num_train_timesteps - 1);
            let frac = t - low as f32;
            train_sigmas[low] * (1.0 - frac) + train_sigmas[high] * frac
        };
        let mut sigmas = Vec::with_capacity(num_steps + 1);
        let mut timesteps = Vec::with_capacity(num_steps);
        for k in 0..num_steps {
            let idx = num_steps - 1 - k;
            let t = idx as f32 * denom / ts_denom;
            timesteps.push(t);
            sigmas.push(interp(t));
        }
        sigmas.push(0.0);
        Ok(Self {
            sigmas,
            timesteps,
            prediction: PredictionType::Epsilon,
        })
    }

    /// Set the model parameterization (`epsilon` by default).
    pub(super) fn with_prediction(mut self, prediction: PredictionType) -> Self {
        self.prediction = prediction;
        self
    }

    /// `x / sqrt(sigma^2 + 1)` — scale the raw sample for the denoiser input.
    fn scale(&self, step: usize, sample: &[f32]) -> Vec<f32> {
        let factor = (self.sigmas[step] * self.sigmas[step] + 1.0).sqrt();
        sample.iter().map(|&x| x / factor).collect()
    }

    /// `x_next = x + eps * (sigma_next - sigma)` on the raw sample. The raw
    /// `model_out` is first converted to the epsilon derivative per
    /// [`Self::prediction`] (byte-identical for `epsilon`).
    fn step_vec(&self, step: usize, sample: &[f32], model_out: &[f32]) -> anyhow::Result<Vec<f32>> {
        if sample.len() != model_out.len() {
            anyhow::bail!(
                "scheduler sample/model_output length mismatch: {} vs {}",
                sample.len(),
                model_out.len()
            );
        }
        let sigma = self.sigmas[step];
        // Convert to DDPM alpha/sigma: alpha_t = 1/sqrt(sigma^2+1),
        // sigma_t = sigma * alpha_t, and the DDPM latent x_t = alpha_t * sample.
        // The epsilon derivative diffusers feeds is `(sample - x0) / sigma`,
        // which equals the DDPM epsilon; it reduces to `model_out` for epsilon.
        let alpha_t = 1.0 / (sigma * sigma + 1.0).sqrt();
        let sigma_t = sigma * alpha_t;
        let dt = self.sigmas[step + 1] - self.sigmas[step];
        Ok(sample
            .iter()
            .zip(model_out)
            .map(|(&x, &m)| {
                let e =
                    epsilon_from_model_output(m, alpha_t * x, alpha_t, sigma_t, self.prediction);
                x + e * dt
            })
            .collect())
    }
}

impl Scheduler for EulerSchedule {
    fn step(
        &self,
        step: usize,
        _num_steps: usize,
        sample: &Value,
        model_output: &Value,
    ) -> anyhow::Result<Value> {
        let shape = sample.shape().to_vec();
        let stepped = self.step_vec(
            step,
            &sample.to_vec_f32_lossy()?,
            &model_output.to_vec_f32_lossy()?,
        )?;
        Value::from_slice_f32(&stepped, &shape).map_err(Into::into)
    }

    fn scale_input(
        &self,
        step: usize,
        _num_steps: usize,
        sample: &Value,
    ) -> anyhow::Result<Option<Value>> {
        let shape = sample.shape().to_vec();
        let scaled = self.scale(step, &sample.to_vec_f32_lossy()?);
        Ok(Some(Value::from_slice_f32(&scaled, &shape)?))
    }

    fn init_noise_sigma(&self) -> f32 {
        self.sigmas[0]
    }

    fn timesteps(&self) -> Option<Vec<f32>> {
        Some(self.timesteps.clone())
    }

    fn add_noise(
        &self,
        step: usize,
        num_steps: usize,
        original: &Value,
        noise: &Value,
    ) -> anyhow::Result<Value> {
        let sigma = if step == num_steps {
            0.0
        } else {
            self.sigmas[step]
        };
        super::mix_noise(original, noise, 1.0, sigma)
    }
}

/// Euler Ancestral (`EulerAncestralDiscreteScheduler`, epsilon) — a *stochastic*
/// sampler (one of the most-used in ComfyUI). Like Euler it scales the model
/// input and seeds at `sigmas[0]`, but each step advances to an intermediate
/// `sigma_down` and injects fresh noise scaled by `sigma_up`:
///   `sigma_up   = sqrt(sigma_to^2 (sigma_from^2 - sigma_to^2) / sigma_from^2)`
///   `sigma_down = sqrt(sigma_to^2 - sigma_up^2)`
///   `x_next = x + eps*(sigma_down - sigma) + noise*sigma_up`.
/// Matches diffusers when fed the same per-step noise sequence.
#[derive(Debug, Clone)]
pub(super) struct EulerAncestral {
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    prediction: PredictionType,
}

impl EulerAncestral {
    pub(super) fn with_schedule(
        num_train_timesteps: usize,
        beta_start: f32,
        beta_end: f32,
        beta_schedule: &str,
        num_steps: usize,
        spacing: &str,
    ) -> anyhow::Result<Self> {
        // Same sigma schedule as Euler (linspace interp / Karras / exponential).
        let euler = EulerSchedule::with_schedule(
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
            num_steps,
            spacing,
        )?;
        Ok(Self {
            sigmas: euler.sigmas,
            timesteps: euler.timesteps,
            prediction: PredictionType::Epsilon,
        })
    }

    /// Set the model parameterization (`epsilon` by default).
    pub(super) fn with_prediction(mut self, prediction: PredictionType) -> Self {
        self.prediction = prediction;
        self
    }
}

impl Scheduler for EulerAncestral {
    fn step(
        &self,
        _step: usize,
        _num_steps: usize,
        _sample: &Value,
        _model_output: &Value,
    ) -> anyhow::Result<Value> {
        anyhow::bail!("euler_ancestral is stochastic; the loop must call step_with_noise")
    }

    fn needs_noise(&self) -> bool {
        true
    }

    fn step_with_noise(
        &self,
        step: usize,
        _num_steps: usize,
        sample: &Value,
        model_output: &Value,
        noise: Option<&Value>,
    ) -> anyhow::Result<Value> {
        let shape = sample.shape().to_vec();
        let x = sample.to_vec_f32_lossy()?;
        let model_out = model_output.to_vec_f32_lossy()?;
        let sigma_from = self.sigmas[step];
        let sigma_to = self.sigmas[step + 1];
        let sigma_up = (sigma_to * sigma_to * (sigma_from * sigma_from - sigma_to * sigma_to)
            / (sigma_from * sigma_from))
            .max(0.0)
            .sqrt();
        let sigma_down = (sigma_to * sigma_to - sigma_up * sigma_up).max(0.0).sqrt();
        let dt = sigma_down - sigma_from;
        // DDPM alpha/sigma for the current sigma; DDPM latent x_t = alpha_t * x.
        let alpha_t = 1.0 / (sigma_from * sigma_from + 1.0).sqrt();
        let sigma_t = sigma_from * alpha_t;
        let noise = noise
            .context("euler_ancestral requires per-step noise")?
            .to_vec_f32_lossy()?;
        if noise.len() != x.len() {
            anyhow::bail!(
                "euler_ancestral noise length {} != sample {}",
                noise.len(),
                x.len()
            );
        }
        let out: Vec<f32> = (0..x.len())
            .map(|i| {
                let e = epsilon_from_model_output(
                    model_out[i],
                    alpha_t * x[i],
                    alpha_t,
                    sigma_t,
                    self.prediction,
                );
                x[i] + e * dt + noise[i] * sigma_up
            })
            .collect();
        Value::from_slice_f32(&out, &shape).map_err(Into::into)
    }

    fn scale_input(
        &self,
        step: usize,
        _num_steps: usize,
        sample: &Value,
    ) -> anyhow::Result<Option<Value>> {
        let factor = (self.sigmas[step] * self.sigmas[step] + 1.0).sqrt();
        let scaled: Vec<f32> = sample
            .to_vec_f32_lossy()?
            .iter()
            .map(|&x| x / factor)
            .collect();
        Ok(Some(Value::from_slice_f32(&scaled, sample.shape())?))
    }

    fn init_noise_sigma(&self) -> f32 {
        self.sigmas[0]
    }

    fn timesteps(&self) -> Option<Vec<f32>> {
        Some(self.timesteps.clone())
    }

    fn add_noise(
        &self,
        step: usize,
        num_steps: usize,
        original: &Value,
        noise: &Value,
    ) -> anyhow::Result<Value> {
        let sigma = if step == num_steps {
            0.0
        } else {
            self.sigmas[step]
        };
        super::mix_noise(original, noise, 1.0, sigma)
    }
}
