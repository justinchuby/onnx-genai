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
//! supported/default value). `scale` may be **any** shape unidirectionally
//! (NumPy-style, right-aligned) broadcastable to `X` — scalar, the normalized
//! axes shape (`X.shape[axis:]`), or any intermediate — and is broadcast over
//! `X` before the elementwise multiply.

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
        let y = rms_norm_dense(
            &x,
            inputs[0].shape,
            &scale,
            inputs[1].shape,
            self.axis,
            self.epsilon,
        )?;
        write_dense_f32(&mut outputs[0], &y)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

/// Shared RMSNorm math for contrib kernels.
pub(crate) fn rms_norm_dense(
    x: &[f32],
    x_shape: &[usize],
    scale: &[f32],
    scale_shape: &[usize],
    axis: i64,
    epsilon: f32,
) -> Result<Vec<f32>> {
    let rank = x_shape.len();
    let axis = if axis < 0 { axis + rank as i64 } else { axis };
    if axis < 0 || axis as usize >= rank {
        return Err(EpError::KernelFailed(format!(
            "RMSNormalization: axis {} out of range for rank {rank}",
            axis
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

    // `scale` may be any shape unidirectionally broadcastable to `X`
    // (NumPy-style, right-aligned). Precompute per-axis multipliers so a
    // flat `X` index maps to the matching `scale` element in O(rank).
    if scale_shape.len() > rank {
        return Err(EpError::KernelFailed(format!(
            "RMSNormalization: scale rank {} exceeds X rank {rank}",
            scale_shape.len()
        )));
    }
    // Right-align scale dims against X; validate broadcastability and build
    // a per-X-axis stride into the flat scale buffer (0 where broadcast).
    let offset = rank - scale_shape.len();
    let mut scale_strides = vec![0usize; rank];
    {
        let mut stride = 1usize;
        for i in (0..scale_shape.len()).rev() {
            let sdim = scale_shape[i];
            let xdim = x_shape[offset + i];
            if sdim != xdim && sdim != 1 {
                return Err(EpError::KernelFailed(format!(
                    "RMSNormalization: scale shape {scale_shape:?} not broadcastable to X shape {x_shape:?}"
                )));
            }
            scale_strides[offset + i] = if sdim == 1 { 0 } else { stride };
            stride *= sdim;
        }
    }
    if scale.len() != scale_shape.iter().product::<usize>() {
        return Err(EpError::KernelFailed(format!(
            "RMSNormalization: scale has {} elements, expected {} for shape {scale_shape:?}",
            scale.len(),
            scale_shape.iter().product::<usize>()
        )));
    }

    // Row-major strides for X to unravel a flat index into coordinates.
    let mut x_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        x_strides[i] = x_strides[i + 1] * x_shape[i + 1];
    }
    let scale_index = |flat: usize| -> usize {
        let mut si = 0usize;
        let mut rem = flat;
        for d in 0..rank {
            let coord = rem / x_strides[d];
            rem %= x_strides[d];
            si += coord * scale_strides[d];
        }
        si
    };

    let mut y = vec![0.0f32; x.len()];
    for g in 0..num_groups {
        let base = g * norm_size;
        let slice = &x[base..base + norm_size];
        let mean_sq = slice.iter().map(|&v| v * v).sum::<f32>() / norm_size as f32;
        let inv_rms = 1.0 / (mean_sq + epsilon).sqrt();
        for e in 0..norm_size {
            let idx = base + e;
            y[idx] = x[idx] * inv_rms * scale[scale_index(idx)];
        }
    }

    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    /// Reference RMSNorm for a single row (scale applied elementwise).
    fn reference(row: &[f32], scale: &[f32], eps: f32) -> Vec<f32> {
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        row.iter().zip(scale).map(|(v, s)| v * inv * s).collect()
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
        RmsNormKernel {
            axis: 1,
            epsilon: eps,
        }
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

    /// Group-wise RMS over the last `norm_size` elements, then multiply by a
    /// scale that is right-aligned broadcast against `x_shape`.
    fn reference_bcast(
        x: &[f32],
        x_shape: &[usize],
        scale: &[f32],
        scale_shape: &[usize],
        axis: usize,
        eps: f32,
    ) -> Vec<f32> {
        let rank = x_shape.len();
        let norm_size: usize = x_shape[axis..].iter().product();
        let num_groups: usize = x_shape[..axis].iter().product();
        // Row-major X strides.
        let mut xs = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            xs[i] = xs[i + 1] * x_shape[i + 1];
        }
        // Right-aligned scale strides (0 where broadcast).
        let offset = rank - scale_shape.len();
        let mut ss = vec![0usize; rank];
        let mut stride = 1usize;
        for i in (0..scale_shape.len()).rev() {
            ss[offset + i] = if scale_shape[i] == 1 { 0 } else { stride };
            stride *= scale_shape[i];
        }
        let mut out = vec![0.0f32; x.len()];
        for g in 0..num_groups {
            let base = g * norm_size;
            let slice = &x[base..base + norm_size];
            let inv =
                1.0 / (slice.iter().map(|v| v * v).sum::<f32>() / norm_size as f32 + eps).sqrt();
            for e in 0..norm_size {
                let flat = base + e;
                let mut rem = flat;
                let mut si = 0usize;
                for d in 0..rank {
                    let c = rem / xs[d];
                    rem %= xs[d];
                    si += c * ss[d];
                }
                out[flat] = x[flat] * inv * scale[si];
            }
        }
        out
    }

    #[test]
    fn rmsnorm_scalar_scale_broadcasts() {
        let x_data = [1., 2., 3., 4., 5., 6.];
        let x = Owned::f32(&[2, 3], &x_data);
        let scale = Owned::f32(&[], &[2.0]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        RmsNormKernel {
            axis: -1,
            epsilon: 1e-5,
        }
        .execute(&[x.view(), scale.view()], &mut [out.view_mut()])
        .unwrap();
        let want = reference_bcast(&x_data, &[2, 3], &[2.0], &[], 1, 1e-5);
        for (g, w) in out.to_f32().iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
        }
    }

    #[test]
    fn rmsnorm_scale_broadcasts_last_axis() {
        // X=[2,3,4], axis=1, Scale=[4] → broadcast over groups and the axis dim.
        let x_data: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let x = Owned::f32(&[2, 3, 4], &x_data);
        let scale_data = [1., 2., 3., 4.];
        let scale = Owned::f32(&[4], &scale_data);
        let mut out = Owned::zeros_f32(&[2, 3, 4]);
        RmsNormKernel {
            axis: 1,
            epsilon: 1e-5,
        }
        .execute(&[x.view(), scale.view()], &mut [out.view_mut()])
        .unwrap();
        let want = reference_bcast(&x_data, &[2, 3, 4], &scale_data, &[4], 1, 1e-5);
        for (g, w) in out.to_f32().iter().zip(&want) {
            assert!((g - w).abs() < 1e-4, "got {g}, want {w}");
        }
    }

    #[test]
    fn rmsnorm_scale_broadcasts_partial_shape() {
        // X=[2,3,4], axis=1, Scale=[3,1] → broadcast over the last dim.
        let x_data: Vec<f32> = (0..24).map(|v| (v as f32) * 0.5).collect();
        let x = Owned::f32(&[2, 3, 4], &x_data);
        let scale_data = [1., 2., 3.];
        let scale = Owned::f32(&[3, 1], &scale_data);
        let mut out = Owned::zeros_f32(&[2, 3, 4]);
        RmsNormKernel {
            axis: 1,
            epsilon: 1e-5,
        }
        .execute(&[x.view(), scale.view()], &mut [out.view_mut()])
        .unwrap();
        let want = reference_bcast(&x_data, &[2, 3, 4], &scale_data, &[3, 1], 1, 1e-5);
        for (g, w) in out.to_f32().iter().zip(&want) {
            assert!((g - w).abs() < 1e-4, "got {g}, want {w}");
        }
    }

    #[test]
    fn rmsnorm_scale_full_normalized_shape() {
        // X=[2,3,4], axis=1, Scale=[3,4] → full normalized-axes shape.
        let x_data: Vec<f32> = (0..24).map(|v| (v as f32) - 12.0).collect();
        let x = Owned::f32(&[2, 3, 4], &x_data);
        let scale_data: Vec<f32> = (1..13).map(|v| v as f32 * 0.25).collect();
        let scale = Owned::f32(&[3, 4], &scale_data);
        let mut out = Owned::zeros_f32(&[2, 3, 4]);
        RmsNormKernel {
            axis: 1,
            epsilon: 1e-5,
        }
        .execute(&[x.view(), scale.view()], &mut [out.view_mut()])
        .unwrap();
        let want = reference_bcast(&x_data, &[2, 3, 4], &scale_data, &[3, 4], 1, 1e-5);
        for (g, w) in out.to_f32().iter().zip(&want) {
            assert!((g - w).abs() < 1e-4, "got {g}, want {w}");
        }
    }

    #[test]
    fn rmsnorm_non_broadcastable_scale_errors() {
        // Scale=[3] cannot broadcast to X's last dim of 4.
        let x = Owned::f32(&[2, 4], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let scale = Owned::f32(&[3], &[1., 1., 1.]);
        let mut out = Owned::zeros_f32(&[2, 4]);
        let err = RmsNormKernel {
            axis: -1,
            epsilon: 1e-5,
        }
        .execute(&[x.view(), scale.view()], &mut [out.view_mut()]);
        assert!(err.is_err());
    }

    #[test]
    fn rmsnorm_axis_equal_rank_rejected() {
        // axis == rank is out of the valid [-rank, rank-1] range.
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let scale = Owned::f32(&[3], &[1., 1., 1.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        let err = RmsNormKernel {
            axis: 2,
            epsilon: 1e-5,
        }
        .execute(&[x.view(), scale.view()], &mut [out.view_mut()]);
        assert!(err.is_err(), "axis == rank must be rejected");
    }

    #[test]
    fn rmsnorm_axis_below_negative_rank_rejected() {
        // axis == -rank-1 normalises below 0 and is out of range.
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let scale = Owned::f32(&[3], &[1., 1., 1.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        let err = RmsNormKernel {
            axis: -3,
            epsilon: 1e-5,
        }
        .execute(&[x.view(), scale.view()], &mut [out.view_mut()]);
        assert!(err.is_err(), "axis == -rank-1 must be rejected");
    }
}
