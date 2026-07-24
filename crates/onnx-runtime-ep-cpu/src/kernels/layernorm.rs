//! `LayerNormalization`: normalize f32 `X` over the axes from `axis` onward,
//! then scale and shift (`docs/ORT2.md` §4.4).
//!
//! `Y = (X - mean) / sqrt(var + epsilon) * Scale + B`, where `mean`/`var` are
//! the population statistics over the normalized axes. The optional `Mean` and
//! `InvStdDev` outputs are filled when the caller provides those output slots.

use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::check_arity;

/// f32 LayerNormalization kernel carrying `axis` and `epsilon`.
pub struct LayerNormKernel {
    axis: i64,
    epsilon: f32,
}

/// Factory reading `axis` (default -1) and `epsilon` (default 1e-5).
pub struct LayerNormFactory;

impl KernelFactory for LayerNormFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let axis = node.attr("axis").and_then(|a| a.as_int()).unwrap_or(-1);
        let epsilon = node
            .attr("epsilon")
            .and_then(|a| a.as_float())
            .unwrap_or(1e-5);
        Ok(Box::new(LayerNormKernel { axis, epsilon }))
    }
}

impl Kernel for LayerNormKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("LayerNormalization", inputs, outputs, 2, 3, 1)?;
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            let elements = inputs[0].numel() as u64;
            // Estimate the explicit arithmetic in mean/variance/normalization.
            // sqrt is counted as one operation; bias contributes one add/element.
            let groups = normalization_groups(inputs[0].shape, self.axis).unwrap_or(0) as u64;
            let mut flops = elements
                .saturating_mul(7)
                .saturating_add(groups.saturating_mul(5));
            if inputs.len() == 3 {
                flops = flops.saturating_add(elements);
            }
            flops
        });
        let x = to_dense_f32_widen("LayerNormalization", &inputs[0])?;
        let scale = to_dense_f32_widen("LayerNormalization", &inputs[1])?;
        let bias = if inputs.len() == 3 {
            Some(to_dense_f32_widen("LayerNormalization", &inputs[2])?)
        } else {
            None
        };

        let (y, means, inv_stds) = layer_norm_dense(
            &x,
            inputs[0].shape,
            &scale,
            bias.as_deref(),
            self.axis,
            self.epsilon,
        )?;

        write_dense_f32_narrow("LayerNormalization", &mut outputs[0], &y)?;
        // Optional Mean / InvStdDev outputs (shape = X.shape[:axis] with the
        // normalized axes as 1s; element count == num_groups).
        if outputs.len() > 1 {
            write_dense_f32_narrow("LayerNormalization", &mut outputs[1], &means)?;
        }
        if outputs.len() > 2 {
            write_dense_f32_narrow("LayerNormalization", &mut outputs[2], &inv_stds)?;
        }
        Ok(())
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn normalization_groups(shape: &[usize], axis: i64) -> Option<usize> {
    let axis = if axis < 0 {
        axis.checked_add(shape.len() as i64)?
    } else {
        axis
    };
    (axis >= 0 && axis as usize <= shape.len()).then(|| shape[..axis as usize].iter().product())
}

/// Shared LayerNorm math for fused contrib kernels. Returns the normalized
/// output plus one mean and inverse standard deviation per normalized group.
pub(crate) fn layer_norm_dense(
    x: &[f32],
    x_shape: &[usize],
    scale: &[f32],
    bias: Option<&[f32]>,
    axis: i64,
    epsilon: f32,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let rank = x_shape.len();
    let axis = if axis < 0 { axis + rank as i64 } else { axis };
    if axis < 0 || axis as usize > rank {
        return Err(EpError::KernelFailed(format!(
            "LayerNormalization: axis {} out of range for rank {rank}",
            axis
        )));
    }
    let axis = axis as usize;

    let norm_size: usize = x_shape[axis..].iter().product();
    let num_groups: usize = x_shape[..axis].iter().product();
    if norm_size == 0 {
        return Err(EpError::KernelFailed(
            "LayerNormalization: empty normalization axis".into(),
        ));
    }
    if scale.len() != norm_size {
        return Err(EpError::KernelFailed(format!(
            "LayerNormalization: scale has {} elements, expected {norm_size}",
            scale.len()
        )));
    }
    if let Some(b) = bias
        && b.len() != norm_size
    {
        return Err(EpError::KernelFailed(format!(
            "LayerNormalization: bias has {} elements, expected {norm_size}",
            b.len()
        )));
    }
    let mut y = vec![0.0f32; x.len()];
    let mut means = vec![0.0f32; num_groups];
    let mut inv_stds = vec![0.0f32; num_groups];

    for g in 0..num_groups {
        let base = g * norm_size;
        let slice = &x[base..base + norm_size];
        let mean = slice.iter().sum::<f32>() / norm_size as f32;
        let var = slice.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / norm_size as f32;
        let inv_std = 1.0 / (var + epsilon).sqrt();
        means[g] = mean;
        inv_stds[g] = inv_std;
        for e in 0..norm_size {
            let norm = (slice[e] - mean) * inv_std;
            let mut out = norm * scale[e];
            if let Some(b) = bias {
                out += b[e];
            }
            y[base + e] = out;
        }
    }

    Ok((y, means, inv_stds))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    #[test]
    fn layernorm_last_axis_matches_reference() {
        // X [2,3], normalize last axis. Scale=1, Bias=0, eps=1e-5.
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 2., 4., 6.]);
        let scale = Owned::f32(&[3], &[1., 1., 1.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        let k = LayerNormKernel {
            axis: -1,
            epsilon: 1e-5,
        };
        k.execute(&[x.view(), scale.view()], &mut [out.view_mut()])
            .unwrap();

        // Hand-computed: row0 mean=2, var=2/3, inv=1/sqrt(0.66667+1e-5)
        let expect = |row: &[f32]| -> Vec<f32> {
            let mean = row.iter().sum::<f32>() / 3.0;
            let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 3.0;
            let inv = 1.0 / (var + 1e-5f32).sqrt();
            row.iter().map(|v| (v - mean) * inv).collect()
        };
        let mut want = expect(&[1., 2., 3.]);
        want.extend(expect(&[2., 4., 6.]));
        let got = out.to_f32();
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
        }
    }

    #[test]
    fn layernorm_applies_scale_and_bias() {
        let x = Owned::f32(&[1, 4], &[1., 2., 3., 4.]);
        let scale = Owned::f32(&[4], &[2., 2., 2., 2.]);
        let bias = Owned::f32(&[4], &[1., 1., 1., 1.]);
        let mut out = Owned::zeros_f32(&[1, 4]);
        let k = LayerNormKernel {
            axis: -1,
            epsilon: 1e-5,
        };
        k.execute(
            &[x.view(), scale.view(), bias.view()],
            &mut [out.view_mut()],
        )
        .unwrap();

        let row = [1., 2., 3., 4.];
        let mean = 2.5;
        let var = row.iter().map(|v: &f32| (v - mean).powi(2)).sum::<f32>() / 4.0;
        let inv = 1.0 / (var + 1e-5f32).sqrt();
        let want: Vec<f32> = row.iter().map(|v| (v - mean) * inv * 2.0 + 1.0).collect();
        for (g, w) in out.to_f32().iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
        }
    }

    #[test]
    fn layernorm_writes_optional_mean_and_invstd() {
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 2., 4., 6.]);
        let scale = Owned::f32(&[3], &[1., 1., 1.]);
        let mut y = Owned::zeros_f32(&[2, 3]);
        let mut mean = Owned::zeros_f32(&[2, 1]);
        let mut invstd = Owned::zeros_f32(&[2, 1]);
        let k = LayerNormKernel {
            axis: -1,
            epsilon: 1e-5,
        };
        k.execute(
            &[x.view(), scale.view()],
            &mut [y.view_mut(), mean.view_mut(), invstd.view_mut()],
        )
        .unwrap();
        assert!((mean.to_f32()[0] - 2.0).abs() < 1e-6);
        assert!((mean.to_f32()[1] - 4.0).abs() < 1e-6);
    }
    #[test]
    fn layernorm_bf16_matches_widened_f32_reference() {
        let x = Owned::bf16(&[2, 3], &[-80., 0., 1., 80., -1., 2.]);
        let scale = Owned::bf16(&[3], &[1., 0.5, 2.]);
        let bias = Owned::bf16(&[3], &[0., -1., 1.]);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[2, 3]);
        LayerNormKernel {
            axis: -1,
            epsilon: 1e-5,
        }
        .execute(
            &[x.view(), scale.view(), bias.view()],
            &mut [out.view_mut()],
        )
        .unwrap();
        let (reference, _, _) = layer_norm_dense(
            &x.to_bf16_as_f32(),
            &[2, 3],
            &scale.to_bf16_as_f32(),
            Some(&bias.to_bf16_as_f32()),
            -1,
            1e-5,
        )
        .unwrap();
        let expected: Vec<_> = reference
            .into_iter()
            .map(half::bf16::from_f32)
            .map(half::bf16::to_f32)
            .collect();
        assert_eq!(out.to_bf16_as_f32(), expected);
    }
}
