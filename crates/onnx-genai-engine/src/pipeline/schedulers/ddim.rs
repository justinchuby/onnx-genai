//! DDIM (denoising diffusion implicit models) scheduler.
//!
//! Deterministic (eta = 0) continuous-latent scheduler. Extracted verbatim
//! from `pipeline.rs`.

use onnx_genai_ort::Value;

use super::{PredictionType, Scheduler, epsilon_from_model_output, training_alpha_cumprod};

/// DDIM (η = 0) noise schedule, precomputed per inference
/// step as `(alpha_cumprod_t, alpha_cumprod_prev)`.
///
/// Diffusion-standard update for a model that predicts noise `eps`:
///   `x0_hat = (x_t - sqrt(1 - a_t) * eps) / sqrt(a_t)`
///   `x_prev = sqrt(a_prev) * x0_hat + sqrt(1 - a_prev) * eps`
#[derive(Debug, Clone)]
pub(super) struct DdimSchedule {
    steps: Vec<(f32, f32)>,
    timesteps: Vec<f32>,
    prediction: PredictionType,
}

impl DdimSchedule {
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
        // Evenly spaced inference timesteps, descending (diffusers convention).
        let step_ratio = num_train_timesteps / num_steps;
        let ascending: Vec<usize> = (0..num_steps).map(|i| i * step_ratio).collect();
        let mut steps = Vec::with_capacity(num_steps);
        let mut timesteps = Vec::with_capacity(num_steps);
        for k in 0..num_steps {
            let t = ascending[num_steps - 1 - k];
            timesteps.push(t as f32);
            let a_t = alpha_cumprod[t];
            let a_prev = if k + 1 < num_steps {
                alpha_cumprod[ascending[num_steps - 1 - (k + 1)]]
            } else {
                1.0
            };
            steps.push((a_t, a_prev));
        }
        Ok(Self {
            steps,
            timesteps,
            prediction: PredictionType::Epsilon,
        })
    }

    /// Set the model parameterization (`epsilon` by default).
    pub(super) fn with_prediction(mut self, prediction: PredictionType) -> Self {
        self.prediction = prediction;
        self
    }

    /// Apply one DDIM step to `sample` given the raw `model_out`. The model
    /// output is first converted to epsilon per [`Self::prediction`], then the
    /// epsilon-form DDIM update runs (byte-identical for `epsilon`).
    fn step(&self, k: usize, sample: &[f32], model_out: &[f32]) -> anyhow::Result<Vec<f32>> {
        if sample.len() != model_out.len() {
            anyhow::bail!(
                "scheduler sample/model_output length mismatch: {} vs {}",
                sample.len(),
                model_out.len()
            );
        }
        let (a_t, a_prev) = self.steps[k];
        let sqrt_a_t = a_t.sqrt();
        let sqrt_one_minus_a_t = (1.0 - a_t).sqrt();
        let sqrt_a_prev = a_prev.sqrt();
        let sqrt_one_minus_a_prev = (1.0 - a_prev).sqrt();
        Ok(sample
            .iter()
            .zip(model_out)
            .map(|(&x, &m)| {
                let e =
                    epsilon_from_model_output(m, x, sqrt_a_t, sqrt_one_minus_a_t, self.prediction);
                let x0_hat = (x - sqrt_one_minus_a_t * e) / sqrt_a_t;
                sqrt_a_prev * x0_hat + sqrt_one_minus_a_prev * e
            })
            .collect())
    }
}

impl Scheduler for DdimSchedule {
    fn step(
        &self,
        step: usize,
        _num_steps: usize,
        sample: &Value,
        model_output: &Value,
    ) -> anyhow::Result<Value> {
        let shape = sample.shape().to_vec();
        let stepped = DdimSchedule::step(
            self,
            step,
            &sample.to_vec_f32_lossy()?,
            &model_output.to_vec_f32_lossy()?,
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

    #[test]
    fn ddim_step_matches_hand_computed_closed_form() {
        // num_train=2, beta_start=beta_end=0.5 => betas=[0.5,0.5],
        // alphas=[0.5,0.5], alpha_cumprod=[0.5,0.25].
        // num_steps=1 => timestep t=0 => a_t=0.5, a_prev=1.0 (final step).
        //   x0_hat = (x - sqrt(0.5)*e) / sqrt(0.5)
        //   next   = sqrt(1)*x0_hat + sqrt(0)*e = x0_hat
        let sched = DdimSchedule::with_schedule(2, 0.5, 0.5, "linear", 1).expect("schedule builds");
        // x=1, e=0 -> next = 1/sqrt(0.5) = sqrt(2) ~= 1.41421356
        let n0 = sched.step(0, &[1.0], &[0.0]).unwrap();
        assert!((n0[0] - std::f32::consts::SQRT_2).abs() < 1e-5, "{}", n0[0]);
        // x=1, e=1 -> x0_hat = (1 - sqrt(0.5))/sqrt(0.5) = sqrt(2) - 1 ~= 0.41421356
        let n1 = sched.step(0, &[1.0], &[1.0]).unwrap();
        assert!(
            (n1[0] - (std::f32::consts::SQRT_2 - 1.0)).abs() < 1e-5,
            "{}",
            n1[0]
        );
    }

    #[test]
    fn ddim_add_noise_matches_hand_computed_alpha_mix() {
        let scheduler = DdimSchedule::with_schedule(2, 0.5, 0.5, "linear", 1).unwrap();
        let original = Value::from_slice_f32(&[2.0], &[1]).unwrap();
        let noise = Value::from_slice_f32(&[3.0], &[1]).unwrap();
        let noised = Scheduler::add_noise(&scheduler, 0, 1, &original, &noise).unwrap();
        let expected = 2.0 * 0.5f32.sqrt() + 3.0 * 0.5f32.sqrt();
        assert!((noised.to_vec_f32_lossy().unwrap()[0] - expected).abs() < 1e-6);
        assert_eq!(
            Scheduler::add_noise(&scheduler, 1, 1, &original, &noise)
                .unwrap()
                .to_vec_f32_lossy()
                .unwrap(),
            vec![2.0]
        );
    }

    #[test]
    fn ddim_new_rejects_invalid_step_counts() {
        assert!(DdimSchedule::with_schedule(1, 0.1, 0.2, "linear", 1).is_err()); // num_train < 2
        assert!(DdimSchedule::with_schedule(4, 0.1, 0.2, "linear", 0).is_err()); // num_steps == 0
        assert!(DdimSchedule::with_schedule(4, 0.1, 0.2, "linear", 5).is_err()); // num_steps > num_train
    }

    #[test]
    fn ddim_exposes_descending_integer_timesteps() {
        let sched =
            DdimSchedule::with_schedule(1000, 0.00085, 0.012, "scaled_linear", 4).expect("builds");
        // step_ratio = 250, ascending = [0, 250, 500, 750], reversed for inference.
        assert_eq!(sched.timesteps(), Some(vec![750.0, 500.0, 250.0, 0.0]));
    }
}
