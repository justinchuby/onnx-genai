//! DDPM ancestral scheduler.

use anyhow::Context;
use onnx_genai_ort::Value;

use super::{PredictionType, Scheduler, training_alpha_cumprod, x0_from_model_output};

/// Standard fixed-small-variance DDPM ancestral sampler.
#[derive(Debug, Clone)]
pub(super) struct DdpmSchedule {
    /// Per-step `(alpha_cumprod_t, alpha_cumprod_prev)`, in inference order.
    steps: Vec<(f32, f32)>,
    timesteps: Vec<f32>,
    prediction: PredictionType,
}

impl DdpmSchedule {
    pub(super) fn with_schedule(
        num_train_timesteps: usize,
        beta_start: f32,
        beta_end: f32,
        beta_schedule: &str,
        num_steps: usize,
    ) -> anyhow::Result<Self> {
        if num_train_timesteps < 2 {
            anyhow::bail!("scheduler num_train_timesteps must be >= 2");
        }
        if num_steps == 0 || num_steps > num_train_timesteps {
            anyhow::bail!("scheduler num_steps ({num_steps}) must be in 1..={num_train_timesteps}");
        }
        let alpha_cumprod =
            training_alpha_cumprod(num_train_timesteps, beta_start, beta_end, beta_schedule)?;

        let step_ratio = num_train_timesteps / num_steps;
        let ascending: Vec<usize> = (0..num_steps).map(|i| i * step_ratio).collect();
        let mut steps = Vec::with_capacity(num_steps);
        let mut timesteps = Vec::with_capacity(num_steps);
        for k in 0..num_steps {
            let t = ascending[num_steps - 1 - k];
            let prev_t = if k + 1 < num_steps {
                Some(ascending[num_steps - 2 - k])
            } else {
                None
            };
            timesteps.push(t as f32);
            steps.push((
                alpha_cumprod[t],
                prev_t.map_or(1.0, |index| alpha_cumprod[index]),
            ));
        }
        Ok(Self {
            steps,
            timesteps,
            prediction: PredictionType::Epsilon,
        })
    }

    pub(super) fn with_prediction(mut self, prediction: PredictionType) -> Self {
        self.prediction = prediction;
        self
    }

    fn step_vec(
        &self,
        step: usize,
        sample: &[f32],
        model_out: &[f32],
        noise: &[f32],
    ) -> anyhow::Result<Vec<f32>> {
        if sample.len() != model_out.len() || sample.len() != noise.len() {
            anyhow::bail!(
                "ddpm sample/model_output/noise length mismatch: {}/{}/{}",
                sample.len(),
                model_out.len(),
                noise.len()
            );
        }
        let (alpha_prod_t, alpha_prod_prev) = self.steps[step];
        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_prev = 1.0 - alpha_prod_prev;
        let current_alpha = alpha_prod_t / alpha_prod_prev;
        let current_beta = 1.0 - current_alpha;
        let x0_coeff = alpha_prod_prev.sqrt() * current_beta / beta_prod_t;
        let sample_coeff = current_alpha.sqrt() * beta_prod_prev / beta_prod_t;
        let variance = (beta_prod_prev / beta_prod_t * current_beta).max(0.0);
        let noise_scale = variance.sqrt();
        let alpha_t = alpha_prod_t.sqrt();
        let sigma_t = beta_prod_t.sqrt();

        Ok(sample
            .iter()
            .zip(model_out)
            .zip(noise)
            .map(|((&x, &model), &z)| {
                let x0 = x0_from_model_output(model, x, alpha_t, sigma_t, self.prediction);
                x0_coeff * x0 + sample_coeff * x + noise_scale * z
            })
            .collect())
    }
}

impl Scheduler for DdpmSchedule {
    fn step(
        &self,
        _step: usize,
        _num_steps: usize,
        _sample: &Value,
        _model_output: &Value,
    ) -> anyhow::Result<Value> {
        anyhow::bail!("ddpm is stochastic; the loop must call step_with_noise")
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
        let stepped = self.step_vec(
            step,
            &sample.to_vec_f32_lossy()?,
            &model_output.to_vec_f32_lossy()?,
            &noise
                .context("ddpm requires per-step noise")?
                .to_vec_f32_lossy()?,
        )?;
        Value::from_slice_f32(&stepped, &shape).map_err(Into::into)
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
        if step == num_steps {
            return Value::from_slice_f32(&original.to_vec_f32_lossy()?, original.shape())
                .map_err(Into::into);
        }
        let (alpha, _) = self.steps[step];
        super::mix_noise(original, noise, alpha.sqrt(), (1.0 - alpha).sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(got: f32, want: f32) {
        assert!((got - want).abs() < 1e-6, "{got} != {want}");
    }

    #[test]
    fn ddpm_ancestral_step_matches_hand_computed_reference() {
        // Betas [0.2, 0.2, 0.2], inference timesteps [2, 1, 0].
        // At t=2: alpha_bar=0.512, alpha_bar_prev=0.64.
        let scheduler =
            DdpmSchedule::with_schedule(3, 0.2, 0.2, "linear", 3).expect("schedule builds");
        let out = scheduler
            .step_vec(0, &[0.7], &[0.1], &[-0.25])
            .expect("step succeeds");
        close(out[0], 0.654_586_9);

        // The same epsilon encoded as v_prediction must produce the same result.
        let v = -0.543_642_6;
        let v_scheduler = scheduler
            .clone()
            .with_prediction(PredictionType::VPrediction);
        close(
            v_scheduler.step_vec(0, &[0.7], &[v], &[-0.25]).unwrap()[0],
            0.654_586_9,
        );
    }

    #[test]
    fn ddpm_final_single_step_is_x0_and_ignores_noise() {
        let scheduler =
            DdpmSchedule::with_schedule(2, 0.2, 0.2, "linear", 1).expect("schedule builds");
        close(
            scheduler.step_vec(0, &[-0.3], &[0.2], &[99.0]).unwrap()[0],
            -0.435_410_2,
        );
        let x0_scheduler = scheduler.with_prediction(PredictionType::Sample);
        close(
            x0_scheduler
                .step_vec(0, &[-0.3], &[0.625], &[-99.0])
                .unwrap()[0],
            0.625,
        );
    }
}
