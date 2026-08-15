//! DPM-Solver++ (2M) multistep scheduler.
//!
//! Order-2 (in log-SNR space) deterministic multistep sampler. Extracted
//! verbatim from `pipeline.rs`.

use onnx_genai_ort::Value;
use std::sync::Mutex;

use super::{
    PredictionType, Scheduler, dpm_alpha_sigma, sigma_to_t, spacing_sigmas, training_sigmas,
    x0_from_model_output,
};

/// DPM-Solver++ (2M) — a fast *multistep* deterministic scheduler and the default
/// sampler in most Stable Diffusion / ComfyUI workflows. Order-2 in log-SNR (λ)
/// space using the previous step's data prediction (`x0`), with a first-order step
/// at the start and a first-order final step (when `<15` steps or the final sigma
/// is zero, matching diffusers `final_sigmas_type="zero"`). Matches diffusers
/// `DPMSolverMultistepScheduler(algorithm_type="dpmsolver++", solver_type="midpoint")`.
/// Unlike Euler it does NOT scale the model input (`scale_model_input` is identity)
/// and its `init_noise_sigma` is 1.0 (the seed is unscaled).
#[derive(Debug)]
pub(super) struct Dpmpp2m {
    /// Inference sigmas, descending, with a trailing `0.0`. Length `num_steps + 1`.
    sigmas: Vec<f32>,
    /// Per-step denoiser timesteps, length `num_steps`.
    timesteps: Vec<f32>,
    /// Previous step's data prediction (`x0`) for the multistep update. Reset at
    /// step 0 of each denoise loop; interior-mutable so `step` keeps `&self`.
    prev_x0: Mutex<Option<Vec<f32>>>,
    /// Model parameterization (`epsilon` by default).
    prediction: PredictionType,
}

impl Dpmpp2m {
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
                prev_x0: Mutex::new(None),
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
        let mut train = Vec::with_capacity(num_train_timesteps);
        let mut prod = 1.0f32;
        for i in 0..num_train_timesteps {
            let mut beta = lo + (hi - lo) * (i as f32) / denom;
            if square {
                beta *= beta;
            }
            prod *= 1.0 - beta;
            train.push(((1.0 - prod) / prod).sqrt());
        }
        // Timesteps: linspace(0, num_train-1, num_steps+1) rounded to int, reversed,
        // drop the last (the 0). Sigmas interpolate at those integer timesteps
        // (integer => exact lookup). Trailing 0 for final_sigmas_type="zero".
        let mut ts_int: Vec<usize> = (0..=num_steps)
            .map(|j| (j as f32 * denom / num_steps as f32).round_ties_even() as usize)
            .collect();
        ts_int.reverse();
        ts_int.pop();
        let timesteps: Vec<f32> = ts_int.iter().map(|&t| t as f32).collect();
        let mut sigmas: Vec<f32> = ts_int
            .iter()
            .map(|&t| train[t.min(num_train_timesteps - 1)])
            .collect();
        sigmas.push(0.0);
        Ok(Self {
            sigmas,
            timesteps,
            prev_x0: Mutex::new(None),
            prediction: PredictionType::Epsilon,
        })
    }

    /// Set the model parameterization (`epsilon` by default).
    pub(super) fn with_prediction(mut self, prediction: PredictionType) -> Self {
        self.prediction = prediction;
        self
    }
}

impl Scheduler for Dpmpp2m {
    fn step(
        &self,
        step: usize,
        num_steps: usize,
        sample: &Value,
        model_output: &Value,
    ) -> anyhow::Result<Value> {
        let shape = sample.shape().to_vec();
        let x = sample.to_vec_f32_lossy()?;
        let model_out = model_output.to_vec_f32_lossy()?;
        if x.len() != model_out.len() {
            anyhow::bail!(
                "dpm++ sample/model_output length mismatch: {} vs {}",
                x.len(),
                model_out.len()
            );
        }

        let sigma = self.sigmas[step];
        let (alpha_t0, sigma_t0) = dpm_alpha_sigma(sigma);
        // Data prediction (x0) from the raw model output per the parameterization.
        // For epsilon this is the byte-identical (x - sigma_t*eps)/alpha_t.
        let x0: Vec<f32> = x
            .iter()
            .zip(&model_out)
            .map(|(&xi, &mi)| x0_from_model_output(mi, xi, alpha_t0, sigma_t0, self.prediction))
            .collect();

        let s_next = self.sigmas[step + 1];
        let (a_t, sig_t) = dpm_alpha_sigma(s_next);
        let (a_s0, sig_s0) = dpm_alpha_sigma(sigma);
        let lam_t = a_t.ln() - sig_t.ln(); // +inf at the final step (sig_t == 0)
        let lam_s0 = a_s0.ln() - sig_s0.ln();
        let h = lam_t - lam_s0;
        let neg_expm1 = (-h).exp() - 1.0; // exp(-h) - 1  (== -1 at the final step)

        let mut prev = self
            .prev_x0
            .lock()
            .map_err(|_| anyhow::anyhow!("dpm++ scheduler state poisoned"))?;
        // Match diffusers `DPMSolverMultistepScheduler`: the final step drops to
        // the first-order update when `lower_order_final` applies. diffusers sets
        // that at the last step whenever `num_steps < 15` OR the final sigma is
        // zero (`final_sigmas_type="zero"`, the default this schedule uses). The
        // second-order update divides by the log-SNR step `h`, which is infinite
        // when the final sigma is zero — so skipping it there also avoids the
        // resulting non-finite latent.
        let lower_order_final = step + 1 == num_steps && (num_steps < 15 || s_next <= 0.0);
        // First step of the loop (prev cleared by reset) or the low-order final
        // step both use the first-order update.
        let first_order = lower_order_final || prev.is_none();

        let out: Vec<f32> = if first_order {
            x.iter()
                .zip(&x0)
                .map(|(&xi, &d0)| (sig_t / sig_s0) * xi - a_t * neg_expm1 * d0)
                .collect()
        } else {
            let prev_x0 = prev.as_ref().unwrap();
            let s_prev = self.sigmas[step - 1];
            let (a_s1, sig_s1) = dpm_alpha_sigma(s_prev);
            let lam_s1 = a_s1.ln() - sig_s1.ln();
            let h0 = lam_s0 - lam_s1;
            let r0 = h0 / h;
            x.iter()
                .enumerate()
                .map(|(i, &xi)| {
                    let d0 = x0[i];
                    let d1 = (1.0 / r0) * (x0[i] - prev_x0[i]);
                    (sig_t / sig_s0) * xi - a_t * neg_expm1 * d0 - 0.5 * a_t * neg_expm1 * d1
                })
                .collect()
        };
        *prev = Some(x0);
        drop(prev);
        Value::from_slice_f32(&out, &shape).map_err(Into::into)
    }

    fn reset(&self) {
        if let Ok(mut prev) = self.prev_x0.lock() {
            *prev = None;
        }
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
        let (alpha, sigma) = dpm_alpha_sigma(sigma);
        super::mix_noise(original, noise, alpha, sigma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dpmpp_timesteps_match_diffusers_linspace() {
        // Classic Stable Diffusion 1.x schedule (1000 train steps, scaled_linear).
        // diffusers `DPMSolverMultistepScheduler(timestep_spacing="linspace")`
        // uses `linspace(0, num_train-1, num_steps+1).round()[::-1][:-1]`.
        let num_train = 1000usize;
        let num_steps = 25usize;
        let sched =
            Dpmpp2m::with_schedule(num_train, 0.00085, 0.012, "scaled_linear", num_steps, "")
                .expect("schedule builds");
        let timesteps = sched.timesteps().expect("dpm++ exposes timesteps");
        let denom = (num_train - 1) as f32;
        let mut expected: Vec<f32> = (0..=num_steps)
            .map(|j| (j as f32 * denom / num_steps as f32).round_ties_even())
            .collect();
        expected.reverse();
        expected.pop();
        assert_eq!(timesteps.len(), num_steps);
        assert!(
            (timesteps[0] - 999.0).abs() < 1e-3,
            "first timestep {}",
            timesteps[0]
        );
        for (got, want) in timesteps.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-3, "timestep {got} != {want}");
        }
    }

    #[test]
    fn dpmpp_final_step_stays_finite_with_zero_final_sigma() {
        // With >= 15 steps and a zero final sigma (final_sigmas_type="zero"), the
        // last step must drop to the first-order update; the second-order update
        // divides by an infinite log-SNR step at sigma=0 and would emit NaN/inf.
        let num_steps = 20usize;
        let sched = Dpmpp2m::with_schedule(1000, 0.00085, 0.012, "scaled_linear", num_steps, "")
            .expect("schedule builds");
        sched.reset();
        let mut sample = Value::from_slice_f32(&[1.0, -0.5, 0.25], &[3]).unwrap();
        for step in 0..num_steps {
            let eps = Value::from_slice_f32(&[0.3, -0.2, 0.1], &[3]).unwrap();
            sample = sched.step(step, num_steps, &sample, &eps).unwrap();
        }
        assert!(
            sample
                .to_vec_f32()
                .unwrap()
                .iter()
                .all(|value| value.is_finite()),
            "final dpm++ sample must be finite"
        );
    }
}
