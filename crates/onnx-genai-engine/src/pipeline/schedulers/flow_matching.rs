//! Rectified-flow / flow-matching Euler scheduler.

use onnx_genai_ort::Value;

use super::Scheduler;

/// First-order integration of a model-predicted velocity/vector field.
#[derive(Debug, Clone)]
pub(super) struct FlowMatching {
    /// Shifted sigmas, descending, followed by terminal zero.
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
}

impl FlowMatching {
    pub(super) fn with_schedule(
        num_train_timesteps: usize,
        num_steps: usize,
        shift: f32,
    ) -> anyhow::Result<Self> {
        if num_train_timesteps < 2 {
            anyhow::bail!("scheduler num_train_timesteps must be >= 2");
        }
        if num_steps == 0 {
            anyhow::bail!("scheduler num_steps must be >= 1");
        }
        if !shift.is_finite() || shift <= 0.0 {
            anyhow::bail!("flow_matching shift must be finite and > 0");
        }
        let sigma_min = 1.0 / num_train_timesteps as f32;
        let mut sigmas = Vec::with_capacity(num_steps + 1);
        for step in 0..num_steps {
            let ramp = if num_steps > 1 {
                step as f32 / (num_steps - 1) as f32
            } else {
                0.0
            };
            let sigma = 1.0 + ramp * (sigma_min - 1.0);
            sigmas.push(shift * sigma / (1.0 + (shift - 1.0) * sigma));
        }
        let timesteps = sigmas
            .iter()
            .map(|sigma| sigma * num_train_timesteps as f32)
            .collect();
        sigmas.push(0.0);
        Ok(Self { sigmas, timesteps })
    }

    fn step_vec(&self, step: usize, sample: &[f32], velocity: &[f32]) -> anyhow::Result<Vec<f32>> {
        if sample.len() != velocity.len() {
            anyhow::bail!(
                "flow_matching sample/model_output length mismatch: {} vs {}",
                sample.len(),
                velocity.len()
            );
        }
        let dt = self.sigmas[step + 1] - self.sigmas[step];
        Ok(sample
            .iter()
            .zip(velocity)
            .map(|(&x, &v)| x + dt * v)
            .collect())
    }
}

impl Scheduler for FlowMatching {
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
        super::mix_noise(original, noise, 1.0 - sigma, sigma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(got: f32, want: f32) {
        assert!((got - want).abs() < 1e-6, "{got} != {want}");
    }

    #[test]
    fn shifted_flow_matching_steps_match_hand_computed_euler_updates() {
        // N=4, steps=3, shift=2:
        // base sigmas [1, .625, .25] -> shifted [1, 10/13, .4], then terminal 0.
        let scheduler = FlowMatching::with_schedule(4, 3, 2.0).expect("schedule builds");
        assert_eq!(scheduler.timesteps(), Some(vec![4.0, 3.076_923_1, 1.6]));
        let first = scheduler.step_vec(0, &[0.75], &[-0.4]).unwrap();
        close(first[0], 0.842_307_7);
        let middle = scheduler.step_vec(1, &first, &[-0.4]).unwrap();
        close(middle[0], 0.99);
        let last = scheduler.step_vec(2, &middle, &[-0.4]).unwrap();
        close(last[0], 1.15);
    }

    #[test]
    fn single_step_flow_matching_integrates_from_noise_to_data() {
        let scheduler = FlowMatching::with_schedule(1000, 1, 3.0).expect("schedule builds");
        assert_eq!(scheduler.timesteps(), Some(vec![1000.0]));
        let out = scheduler
            .step_vec(0, &[0.25, -0.5], &[0.75, -0.25])
            .unwrap();
        assert_eq!(out, vec![-0.5, -0.25]);
    }
}
