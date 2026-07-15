//! `com.microsoft` fused transformer kernels composed from the shared CPU
//! GELU, LayerNorm, and RMSNorm implementations.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::gelu::{exact_gelu, tanh_gelu};
use super::layernorm::layer_norm_dense;
use super::rmsnorm::rms_norm_dense;
use super::{check_arity, to_dense_f32, write_dense_f32};

fn last_dim_bias(x_shape: &[usize], bias: &[f32], op: &str) -> Result<usize> {
    let Some(&width) = x_shape.last() else {
        return Err(EpError::KernelFailed(format!(
            "{op}: X must have rank at least 1"
        )));
    };
    if bias.len() != width {
        return Err(EpError::KernelFailed(format!(
            "{op}: bias has {} elements, expected last dimension {width}",
            bias.len()
        )));
    }
    Ok(width)
}

pub struct BiasGeluKernel;
pub struct BiasGeluFactory;

impl KernelFactory for BiasGeluFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(BiasGeluKernel))
    }
}

impl Kernel for BiasGeluKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("BiasGelu", inputs, outputs, 2, 2, 1)?;
        let x = to_dense_f32(&inputs[0])?;
        let bias = to_dense_f32(&inputs[1])?;
        let width = last_dim_bias(inputs[0].shape, &bias, "BiasGelu")?;
        let y = x
            .iter()
            .enumerate()
            .map(|(i, &v)| exact_gelu(v + bias[i % width]))
            .collect::<Vec<_>>();
        write_dense_f32(&mut outputs[0], &y)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

pub struct FastGeluKernel;
pub struct FastGeluFactory;

impl KernelFactory for FastGeluFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(FastGeluKernel))
    }
}

impl Kernel for FastGeluKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("FastGelu", inputs, outputs, 1, 2, 1)?;
        let x = to_dense_f32(&inputs[0])?;
        let bias = if inputs.len() == 2 {
            Some(to_dense_f32(&inputs[1])?)
        } else {
            None
        };
        let width = bias
            .as_deref()
            .map(|b| last_dim_bias(inputs[0].shape, b, "FastGelu"))
            .transpose()?;
        let y = x
            .iter()
            .enumerate()
            .map(|(i, &v)| tanh_gelu(v + bias.as_ref().map_or(0.0, |b| b[i % width.unwrap()])))
            .collect::<Vec<_>>();
        write_dense_f32(&mut outputs[0], &y)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

pub struct QuickGeluKernel {
    alpha: f32,
}

pub struct QuickGeluFactory;

impl KernelFactory for QuickGeluFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(QuickGeluKernel {
            alpha: node
                .attr("alpha")
                .and_then(|a| a.as_float())
                .unwrap_or(1.702),
        }))
    }
}

impl Kernel for QuickGeluKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("QuickGelu", inputs, outputs, 1, 1, 1)?;
        let y = to_dense_f32(&inputs[0])?
            .into_iter()
            .map(|x| {
                let z = self.alpha * x;
                let sigmoid = if z >= 0.0 {
                    1.0 / (1.0 + (-z).exp())
                } else {
                    let e = z.exp();
                    e / (1.0 + e)
                };
                x * sigmoid
            })
            .collect::<Vec<_>>();
        write_dense_f32(&mut outputs[0], &y)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

pub struct SkipLayerNormKernel {
    epsilon: f32,
}

pub struct SkipLayerNormFactory;

impl KernelFactory for SkipLayerNormFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(SkipLayerNormKernel {
            epsilon: node
                .attr("epsilon")
                .and_then(|a| a.as_float())
                .unwrap_or(1e-12),
        }))
    }
}

impl Kernel for SkipLayerNormKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        // The existing contrib shape handler declares the primary output plus
        // optional Mean and InvStdDev, but not InputSkipBiasSum.
        check_arity("SkipLayerNormalization", inputs, outputs, 3, 5, 3)?;
        let x = to_dense_f32(&inputs[0])?;
        let skip = to_dense_f32(&inputs[1])?;
        if inputs[0].shape != inputs[1].shape || x.len() != skip.len() {
            return Err(EpError::KernelFailed(
                "SkipLayerNormalization: skip must have the same shape as X".into(),
            ));
        }
        let gamma = to_dense_f32(&inputs[2])?;
        let beta = if inputs.len() >= 4 {
            Some(to_dense_f32(&inputs[3])?)
        } else {
            None
        };
        let bias = if inputs.len() == 5 {
            Some(to_dense_f32(&inputs[4])?)
        } else {
            None
        };
        let width = bias
            .as_deref()
            .map(|b| last_dim_bias(inputs[0].shape, b, "SkipLayerNormalization"))
            .transpose()?;
        let sum = x
            .iter()
            .zip(&skip)
            .enumerate()
            .map(|(i, (&a, &b))| a + b + bias.as_ref().map_or(0.0, |v| v[i % width.unwrap()]))
            .collect::<Vec<_>>();
        let (y, means, inv_stds) = layer_norm_dense(
            &sum,
            inputs[0].shape,
            &gamma,
            beta.as_deref(),
            -1,
            self.epsilon,
        )?;
        write_dense_f32(&mut outputs[0], &y)?;
        if outputs.len() > 1 {
            write_dense_f32(&mut outputs[1], &means)?;
        }
        if outputs.len() > 2 {
            write_dense_f32(&mut outputs[2], &inv_stds)?;
        }
        Ok(())
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

pub struct SimplifiedLayerNormKernel {
    epsilon: f32,
}

pub struct SimplifiedLayerNormFactory;

impl KernelFactory for SimplifiedLayerNormFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(SimplifiedLayerNormKernel {
            epsilon: node
                .attr("epsilon")
                .and_then(|a| a.as_float())
                .unwrap_or(1e-5),
        }))
    }
}

impl Kernel for SimplifiedLayerNormKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("SimplifiedLayerNormalization", inputs, outputs, 2, 2, 1)?;
        let x = to_dense_f32(&inputs[0])?;
        let scale = to_dense_f32(&inputs[1])?;
        let y = rms_norm_dense(
            &x,
            inputs[0].shape,
            &scale,
            inputs[1].shape,
            -1,
            self.epsilon,
        )?;
        write_dense_f32(&mut outputs[0], &y)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::elementwise::erf;
    use crate::kernels::testutil::Owned;

    fn assert_close(got: &[f32], want: &[f32]) {
        for (&g, &w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
        }
    }

    #[test]
    fn bias_gelu_matches_exact_erf_reference() {
        let x = Owned::f32(&[2, 3], &[-1., 0., 1., 2., -2., 0.5]);
        let bias = Owned::f32(&[3], &[0.5, -0.25, 1.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        BiasGeluKernel
            .execute(&[x.view(), bias.view()], &mut [out.view_mut()])
            .unwrap();
        let want = [-0.5f32, -0.25, 2., 2.5, -2.25, 1.5].map(|v| {
            let vf = v as f64;
            (0.5 * vf * (1. + erf(vf / std::f64::consts::SQRT_2))) as f32
        });
        assert_close(&out.to_f32(), &want);
    }

    #[test]
    fn fast_gelu_matches_tanh_reference_with_bias() {
        let x = Owned::f32(&[2, 2], &[-1., 0.5, 1., 2.]);
        let bias = Owned::f32(&[2], &[0.25, -0.5]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        FastGeluKernel
            .execute(&[x.view(), bias.view()], &mut [out.view_mut()])
            .unwrap();
        let want = [-0.75f32, 0., 1.25, 1.5].map(|v| {
            let vf = v as f64;
            (0.5 * vf * (1. + (0.797_884_560_802_865_f64 * (vf + 0.044_715 * vf * vf * vf)).tanh()))
                as f32
        });
        assert_close(&out.to_f32(), &want);
    }

    #[test]
    fn quick_gelu_matches_alpha_reference() {
        let x = Owned::f32(&[3], &[-1., 0.5, 2.]);
        let mut out = Owned::zeros_f32(&[3]);
        QuickGeluKernel { alpha: 2. }
            .execute(&[x.view()], &mut [out.view_mut()])
            .unwrap();
        let want = [-1f32, 0.5, 2.].map(|v| v / (1. + (-2. * v).exp()));
        assert_close(&out.to_f32(), &want);
    }

    #[test]
    fn skip_layer_norm_matches_reference_and_writes_optional_outputs() {
        let x = Owned::f32(&[2, 4], &[1., 2., 3., 4., 2., 3., 4., 5.]);
        let skip = Owned::f32(&[2, 4], &[0.5, -1., 1., 0., 1., 0., -1., 2.]);
        let gamma = Owned::f32(&[4], &[1., 2., 0.5, 1.5]);
        let beta = Owned::f32(&[4], &[0., 1., -1., 0.5]);
        let bias = Owned::f32(&[4], &[0.25, 0., -0.5, 1.]);
        let mut y = Owned::zeros_f32(&[2, 4]);
        let mut mean = Owned::zeros_f32(&[2, 1]);
        let mut inv_std = Owned::zeros_f32(&[2, 1]);
        SkipLayerNormKernel { epsilon: 1e-5 }
            .execute(
                &[
                    x.view(),
                    skip.view(),
                    gamma.view(),
                    beta.view(),
                    bias.view(),
                ],
                &mut [y.view_mut(), mean.view_mut(), inv_std.view_mut()],
            )
            .unwrap();
        let sum = [1.75, 1., 3.5, 5., 3.25, 3., 2.5, 8.];
        let gamma_data = [1., 2., 0.5, 1.5];
        let beta_data = [0., 1., -1., 0.5];
        let mut want = Vec::new();
        let mut means = Vec::new();
        let mut invs = Vec::new();
        for row in sum.chunks_exact(4) {
            let m = row.iter().sum::<f32>() / 4.;
            let inv = 1. / (row.iter().map(|v| (v - m).powi(2)).sum::<f32>() / 4. + 1e-5).sqrt();
            means.push(m);
            invs.push(inv);
            want.extend((0..4).map(|i| (row[i] - m) * inv * gamma_data[i] + beta_data[i]));
        }
        assert_close(&y.to_f32(), &want);
        assert_close(&mean.to_f32(), &means);
        assert_close(&inv_std.to_f32(), &invs);
    }

    #[test]
    fn simplified_layer_norm_matches_rms_reference() {
        let x = Owned::f32(&[2, 4], &[1., 2., 3., 4., -2., 0., 2., 4.]);
        let scale = Owned::f32(&[4], &[1., 2., 0.5, 1.5]);
        let mut out = Owned::zeros_f32(&[2, 4]);
        SimplifiedLayerNormKernel { epsilon: 1e-5 }
            .execute(&[x.view(), scale.view()], &mut [out.view_mut()])
            .unwrap();
        let scale_data = [1., 2., 0.5, 1.5];
        let mut want = Vec::new();
        for row in [1., 2., 3., 4., -2., 0., 2., 4.].chunks_exact(4) {
            let inv = 1. / (row.iter().map(|v| v * v).sum::<f32>() / 4. + 1e-5).sqrt();
            want.extend((0..4).map(|i| row[i] * inv * scale_data[i]));
        }
        assert_close(&out.to_f32(), &want);
    }
}
