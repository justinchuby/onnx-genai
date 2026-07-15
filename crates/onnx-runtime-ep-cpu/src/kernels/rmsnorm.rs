//! Standard `ai.onnx::RMSNormalization` (opset 23): root-mean-square layer
//! normalization without mean subtraction or bias.
//!
//! Per the ONNX reference (`onnx/reference/ops/op_rms_normalization.py`):
//!
//! ```text
//! rms = sqrt(mean(X², axes) + epsilon)      # axes = axis..rank
//! Y   = (X / rms) * scale                    # scale broadcasts over the axes
//! ```
//!
//! Unlike [`super::layernorm::LayerNormKernel`] there is **no** mean removal and
//! **no** bias term. Statistics are computed in f32 (`stash_type=1`, the only
//! supported/default value). `scale` has the shape of the normalized axes
//! (`X.shape[axis:]`) and broadcasts over the leading (group) axes.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::{check_arity, to_dense_f32, write_dense_f32};

/// f32 RMSNormalization kernel carrying `axis` and `epsilon`.
pub struct RmsNormKernel {
    axis: i64,
    epsilon: f32,
}

/// Factory reading `axis` (default -1), `epsilon` (default 1e-5) and
/// `stash_type` (default 1; only 1 = compute-in-float is supported).
pub struct RmsNormFactory;

impl KernelFactory for RmsNormFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let axis = node.attr("axis").and_then(|a| a.as_int()).unwrap_or(-1);
        let epsilon = node
            .attr("epsilon")
            .and_then(|a| a.as_float())
            .unwrap_or(1e-5);
        let stash_type = node
            .attr("stash_type")
            .and_then(|a| a.as_int())
            .unwrap_or(1);
        if stash_type != 1 {
            return Err(EpError::KernelFailed(format!(
                "RMSNormalization: stash_type {stash_type} unsupported (only 1 = float)"
            )));
        }
        Ok(Box::new(RmsNormKernel { axis, epsilon }))
    }
}

impl Kernel for RmsNormKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("RMSNormalization", inputs, outputs, 2, 2, 1)?;
        let x = to_dense_f32(&inputs[0])?;
        let scale = to_dense_f32(&inputs[1])?;

        let x_shape = inputs[0].shape;
        let rank = x_shape.len();
        let axis = if self.axis < 0 {
            self.axis + rank as i64
        } else {
            self.axis
        };
        if axis < 0 || axis as usize > rank {
            return Err(EpError::KernelFailed(format!(
                "RMSNormalization: axis {} out of range for rank {rank}",
                self.axis
            )));
        }
        let axis = axis as usize;

        let norm_size: usize = x_shape[axis..].iter().product();
        let num_groups: usize = x_shape[..axis].iter().product();
        if norm_size == 0 {
            return Err(EpError::KernelFailed(
                "RMSNormalization: empty normalization axis".into(),
            ));
        }
        if scale.len() != norm_size {
            return Err(EpError::KernelFailed(format!(
                "RMSNormalization: scale has {} elements, expected {norm_size}",
                scale.len()
            )));
        }

        let mut y = vec![0.0f32; x.len()];
        for g in 0..num_groups {
            let base = g * norm_size;
            let slice = &x[base..base + norm_size];
            let mean_sq = slice.iter().map(|&v| v * v).sum::<f32>() / norm_size as f32;
            let inv_rms = 1.0 / (mean_sq + self.epsilon).sqrt();
            for e in 0..norm_size {
                y[base + e] = slice[e] * inv_rms * scale[e];
            }
        }

        write_dense_f32(&mut outputs[0], &y)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    /// Reference RMSNorm for a single row (scale applied elementwise).
    fn reference(row: &[f32], scale: &[f32], eps: f32) -> Vec<f32> {
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        row.iter()
            .zip(scale)
            .map(|(v, s)| v * inv * s)
            .collect()
    }

    #[test]
    fn rmsnorm_last_axis_matches_reference() {
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 2., 4., 6.]);
        let scale = Owned::f32(&[3], &[1., 1., 1.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        RmsNormKernel {
            axis: -1,
            epsilon: 1e-5,
        }
        .execute(&[x.view(), scale.view()], &mut [out.view_mut()])
        .unwrap();
        let mut want = reference(&[1., 2., 3.], &[1., 1., 1.], 1e-5);
        want.extend(reference(&[2., 4., 6.], &[1., 1., 1.], 1e-5));
        for (g, w) in out.to_f32().iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
        }
    }

    #[test]
    fn rmsnorm_applies_scale() {
        let x = Owned::f32(&[1, 4], &[1., 2., 3., 4.]);
        let scale = Owned::f32(&[4], &[2., 0.5, 1., 3.]);
        let mut out = Owned::zeros_f32(&[1, 4]);
        RmsNormKernel {
            axis: -1,
            epsilon: 1e-5,
        }
        .execute(&[x.view(), scale.view()], &mut [out.view_mut()])
        .unwrap();
        let want = reference(&[1., 2., 3., 4.], &[2., 0.5, 1., 3.], 1e-5);
        for (g, w) in out.to_f32().iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
        }
    }

    #[test]
    fn rmsnorm_axis_and_epsilon() {
        // axis=1 over a [2,2,2]: norm_size=4, two groups.
        let x = Owned::f32(&[2, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let scale = Owned::f32(&[2, 2], &[1., 1., 1., 1.]);
        let mut out = Owned::zeros_f32(&[2, 2, 2]);
        let eps = 1e-2;
        RmsNormKernel { axis: 1, epsilon: eps }
            .execute(&[x.view(), scale.view()], &mut [out.view_mut()])
            .unwrap();
        let mut want = reference(&[1., 2., 3., 4.], &[1., 1., 1., 1.], eps);
        want.extend(reference(&[5., 6., 7., 8.], &[1., 1., 1., 1.], eps));
        for (g, w) in out.to_f32().iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
        }
    }

    #[test]
    fn rmsnorm_no_mean_subtraction() {
        // A constant row: RMSNorm(k) = k/|k| * scale = sign(k) (eps→0), which
        // differs from LayerNorm (which would give 0 after mean removal).
        let x = Owned::f32(&[1, 3], &[5., 5., 5.]);
        let scale = Owned::f32(&[3], &[1., 1., 1.]);
        let mut out = Owned::zeros_f32(&[1, 3]);
        RmsNormKernel {
            axis: -1,
            epsilon: 0.0,
        }
        .execute(&[x.view(), scale.view()], &mut [out.view_mut()])
        .unwrap();
        for &g in out.to_f32().iter() {
            assert!((g - 1.0).abs() < 1e-5, "got {g}");
        }
    }
}
