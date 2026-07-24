//! `com.microsoft` fused transformer kernels composed from the shared CPU
//! GELU, LayerNorm, and RMSNorm implementations. Floating activation kernels
//! widen f16/bf16 to f32 for compute and narrow their result back to storage.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::gelu::{exact_gelu, tanh_gelu};
use super::layernorm::layer_norm_dense;
use super::rmsnorm::rms_norm_dense;
use super::{check_arity, to_dense_f32, write_dense_f32};
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};

fn require_fused_float_dtype(op: &str, inputs: &[TensorView], output: &TensorMut) -> Result<()> {
    let dtype = inputs[0].dtype;
    if !matches!(
        dtype,
        DataType::Float16 | DataType::BFloat16 | DataType::Float32
    ) {
        return Err(EpError::KernelFailed(format!(
            "{op}: unsupported dtype {dtype:?}; expected Float16, BFloat16, or Float32"
        )));
    }
    if inputs.iter().any(|input| input.dtype != dtype) || output.dtype != dtype {
        return Err(EpError::KernelFailed(format!(
            "{op}: input and output dtypes must all match (got input {dtype:?}, output {:?})",
            output.dtype
        )));
    }
    Ok(())
}

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
        require_fused_float_dtype("BiasGelu", inputs, &outputs[0])?;
        let x = to_dense_f32_widen("BiasGelu", &inputs[0])?;
        let bias = to_dense_f32_widen("BiasGelu", &inputs[1])?;
        let width = last_dim_bias(inputs[0].shape, &bias, "BiasGelu")?;
        let y = x
            .iter()
            .enumerate()
            .map(|(i, &v)| exact_gelu(v + bias[i % width]))
            .collect::<Vec<_>>();
        write_dense_f32_narrow("BiasGelu", &mut outputs[0], &y)
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
        require_fused_float_dtype("FastGelu", inputs, &outputs[0])?;
        let x = to_dense_f32_widen("FastGelu", &inputs[0])?;
        let bias = if inputs.len() == 2 {
            Some(to_dense_f32_widen("FastGelu", &inputs[1])?)
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
        write_dense_f32_narrow("FastGelu", &mut outputs[0], &y)
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
        require_fused_float_dtype("QuickGelu", inputs, &outputs[0])?;
        let y = to_dense_f32_widen("QuickGelu", &inputs[0])?
            .iter()
            .map(|&x| {
                if x == f32::NEG_INFINITY {
                    return 0.0;
                }
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
        write_dense_f32_narrow("QuickGelu", &mut outputs[0], &y)
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
        // ORT SkipLayerNormalization: `output` (0, required) plus optional
        // `mean` (1), `inv_std_var` (2), and `input_skip_bias_sum` (3). A valid
        // node may request as few as one output, so only require output 0.
        check_arity("SkipLayerNormalization", inputs, outputs, 3, 5, 1)?;
        let x = to_dense_f32(&inputs[0])?;
        let skip = to_dense_f32(&inputs[1])?;
        if inputs[0].shape != inputs[1].shape || x.len() != skip.len() {
            return Err(EpError::KernelFailed(
                "SkipLayerNormalization: skip must have the same shape as X".into(),
            ));
        }
        let gamma = to_dense_f32(&inputs[2])?;
        // `beta` (slot 3) and `bias` (slot 4) are independently optional: the
        // executor may pass an absent placeholder for either while the other is
        // present, so guard each slot separately instead of by input count.
        let beta = if inputs.len() >= 4 && !inputs[3].is_absent() {
            Some(to_dense_f32(&inputs[3])?)
        } else {
            None
        };
        let bias = if inputs.len() >= 5 && !inputs[4].is_absent() {
            Some(to_dense_f32(&inputs[4])?)
        } else {
            None
        };
        let width = bias
            .as_deref()
            .map(|b| last_dim_bias(inputs[0].shape, b, "SkipLayerNormalization"))
            .transpose()?;
        // sum = X + skip + bias (bias broadcasts over the last dimension).
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
        // Output 3 (`input_skip_bias_sum`) is the pre-normalization X-shaped sum.
        if outputs.len() > 3 {
            write_dense_f32(&mut outputs[3], &sum)?;
        }
        Ok(())
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

pub struct SimplifiedLayerNormKernel {
    axis: i64,
    epsilon: f32,
}

pub struct SimplifiedLayerNormFactory;

impl KernelFactory for SimplifiedLayerNormFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(SimplifiedLayerNormKernel {
            axis: node.attr("axis").and_then(|a| a.as_int()).unwrap_or(-1),
            epsilon: node
                .attr("epsilon")
                .and_then(|a| a.as_float())
                .unwrap_or(1e-5),
        }))
    }
}

impl Kernel for SimplifiedLayerNormKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        const OP: &str = "SimplifiedLayerNormalization";
        // ORT SimplifiedLayerNormalization: `output` (0, required) plus optional
        // `inv_std_var` (1). Only the primary output is mandatory.
        check_arity(OP, inputs, outputs, 2, 2, 1)?;
        let x = to_dense_f32_widen(OP, &inputs[0])?;
        let scale = to_dense_f32_widen(OP, &inputs[1])?;
        // Reuse the shared RMSNorm core, honouring `axis` (default -1) so the
        // group spans dims `[axis..rank)` and `scale` broadcasts over it.
        let y = rms_norm_dense(
            &x,
            inputs[0].shape,
            &scale,
            inputs[1].shape,
            self.axis,
            self.epsilon,
        )?;
        write_dense_f32_narrow(OP, &mut outputs[0], &y)?;
        // Optional InvStdDev output: the per-group `1 / sqrt(mean(x²) + eps)`,
        // one value per normalized group (reduced shape). `rms_norm_dense`
        // already validated `axis`, so normalization here cannot fail.
        if outputs.len() > 1 {
            let rank = inputs[0].shape.len();
            let axis = if self.axis < 0 {
                (self.axis + rank as i64) as usize
            } else {
                self.axis as usize
            };
            let norm_size: usize = inputs[0].shape[axis..].iter().product();
            let num_groups: usize = inputs[0].shape[..axis].iter().product();
            let inv_std = (0..num_groups)
                .map(|g| {
                    let slice = &x[g * norm_size..g * norm_size + norm_size];
                    let mean_sq = slice.iter().map(|&v| v * v).sum::<f32>() / norm_size as f32;
                    1.0 / (mean_sq + self.epsilon).sqrt()
                })
                .collect::<Vec<_>>();
            write_dense_f32_narrow(OP, &mut outputs[1], &inv_std)?;
        }
        Ok(())
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuExecutionProvider;
    use crate::kernels::elementwise::erf;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, static_shape};
    use onnx_runtime_loader::{Model, encode_model_proto};

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
    fn fused_gelu_activations_bf16_handle_special_values() {
        const NAN: u16 = 0x7fc0;
        let values = [f32::NEG_INFINITY, -0.0, 0.0, f32::INFINITY, f32::NAN];
        let x = Owned::bf16(&[values.len()], &values);
        let zero_bias = Owned::bf16(&[values.len()], &[0.0; 5]);

        // GELU(-inf) = +0 and GELU(+inf) = +inf. Adding the +0 Bias changes
        // -0 to +0 before BiasGelu/FastGelu; QuickGelu preserves its -0.
        // NaN payloads are intentionally not prescribed.
        let biased_expected = [0, 0, 0, 0x7f80, NAN];
        let quick_expected = [0, 0x8000, 0, 0x7f80, NAN];
        let mut output = Owned::zeros(DataType::BFloat16, &[values.len()]);
        BiasGeluKernel
            .execute(&[x.view(), zero_bias.view()], &mut [output.view_mut()])
            .unwrap();
        assert_bf16_special_values("BiasGelu", &output.to_u16_bits(), &biased_expected);

        let mut output = Owned::zeros(DataType::BFloat16, &[values.len()]);
        FastGeluKernel
            .execute(&[x.view(), zero_bias.view()], &mut [output.view_mut()])
            .unwrap();
        assert_bf16_special_values("FastGelu", &output.to_u16_bits(), &biased_expected);

        let mut output = Owned::zeros(DataType::BFloat16, &[values.len()]);
        QuickGeluKernel { alpha: 1.702 }
            .execute(&[x.view()], &mut [output.view_mut()])
            .unwrap();
        assert_bf16_special_values("QuickGelu", &output.to_u16_bits(), &quick_expected);
    }

    #[test]
    fn fused_gelu_activations_bf16_match_independent_finite_goldens() {
        // Generated out-of-band from the ORT formulas in f64 and rounded to
        // nearest-even BF16. Exact and Fast GELU happen to have the same
        // rounded values for these inputs.
        let values = [-1.25, -0.5, 0.75, 1.5];
        let x = Owned::bf16(&[values.len()], &values);
        let bias = Owned::bf16(&[values.len()], &[0.0; 4]);
        let mut output = Owned::zeros(DataType::BFloat16, &[values.len()]);
        BiasGeluKernel
            .execute(&[x.view(), bias.view()], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(output.to_u16_bits(), vec![0xbe07, 0xbe1e, 0x3f14, 0x3fb3]);

        let mut output = Owned::zeros(DataType::BFloat16, &[values.len()]);
        FastGeluKernel
            .execute(&[x.view(), bias.view()], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(output.to_u16_bits(), vec![0xbe07, 0xbe1e, 0x3f14, 0x3fb3]);

        let mut output = Owned::zeros(DataType::BFloat16, &[values.len()]);
        QuickGeluKernel { alpha: 1.702 }
            .execute(&[x.view()], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(output.to_u16_bits(), vec![0xbe08, 0xbe19, 0x3f16, 0x3fb2]);
    }

    fn assert_bf16_special_values(op: &str, got: &[u16], expected: &[u16]) {
        const NAN: u16 = 0x7fc0;
        for (&got, &expected) in got.iter().zip(expected) {
            if expected == NAN {
                assert!(
                    got & 0x7f80 == 0x7f80 && got & 0x007f != 0,
                    "{op}: expected NaN, got 0x{got:04x}"
                );
            } else {
                assert_eq!(got, expected, "{op}: expected 0x{expected:04x}");
            }
        }
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
        SimplifiedLayerNormKernel {
            axis: -1,
            epsilon: 1e-5,
        }
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

    fn owned_float(dtype: DataType, shape: &[usize], data: &[f32]) -> Owned {
        match dtype {
            DataType::Float16 => Owned::f16(shape, data),
            DataType::BFloat16 => Owned::bf16(shape, data),
            DataType::Float32 => Owned::f32(shape, data),
            DataType::Float64 => {
                let data = data.iter().map(|&value| value as f64).collect::<Vec<_>>();
                Owned::f64(shape, &data)
            }
            _ => panic!("test helper requires a supported floating-point dtype"),
        }
    }

    fn owned_float_as_f32(value: &Owned) -> Vec<f32> {
        match value.dtype {
            DataType::Float16 => value.to_f16_as_f32(),
            DataType::BFloat16 => value.to_bf16_as_f32(),
            DataType::Float32 => value.to_f32(),
            DataType::Float64 => value
                .to_f64()
                .into_iter()
                .map(|element| element as f32)
                .collect(),
            _ => panic!("test helper requires a supported floating-point dtype"),
        }
    }

    fn simplified_layer_norm_reference(
        x: &[f32],
        scale: &[f32],
        norm_size: usize,
        epsilon: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut output = Vec::with_capacity(x.len());
        let mut inv_std = Vec::with_capacity(x.len() / norm_size);
        for group in x.chunks_exact(norm_size) {
            let inv = 1.0
                / (group.iter().map(|value| value * value).sum::<f32>() / norm_size as f32
                    + epsilon)
                    .sqrt();
            inv_std.push(inv);
            output.extend(
                group
                    .iter()
                    .zip(scale)
                    .map(|(&value, &weight)| value * inv * weight),
            );
        }
        (output, inv_std)
    }

    #[test]
    fn simplified_layer_norm_supports_floating_dtypes_and_shapes() {
        const EPSILON: f32 = 1e-5;
        let cases = [
            (
                vec![2, 3],
                -1,
                vec![3],
                vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
                vec![1.0, 0.5, 1.5],
            ),
            (
                vec![2, 2, 3],
                1,
                vec![2, 3],
                vec![
                    0.5, 1.0, 1.5, 2.0, 2.5, 3.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0,
                ],
                vec![1.0, 0.5, 1.5, 2.0, 0.25, 0.75],
            ),
        ];

        for dtype in [
            DataType::Float16,
            DataType::BFloat16,
            DataType::Float32,
            DataType::Float64,
        ] {
            for (x_shape, axis, scale_shape, x_data, scale_data) in &cases {
                let axis_index = if *axis < 0 {
                    (*axis + x_shape.len() as i64) as usize
                } else {
                    *axis as usize
                };
                let norm_size = x_shape[axis_index..].iter().product();
                let (want, want_inv_std) =
                    simplified_layer_norm_reference(x_data, scale_data, norm_size, EPSILON);
                let x = owned_float(dtype, x_shape, x_data);
                let scale = owned_float(dtype, scale_shape, scale_data);
                let mut output = Owned::zeros(dtype, x_shape);
                let mut stats_shape = x_shape.clone();
                stats_shape[axis_index..].fill(1);
                let mut inv_std = Owned::zeros(dtype, &stats_shape);

                SimplifiedLayerNormKernel {
                    axis: *axis,
                    epsilon: EPSILON,
                }
                .execute(
                    &[x.view(), scale.view()],
                    &mut [output.view_mut(), inv_std.view_mut()],
                )
                .unwrap();

                let tolerance = match dtype {
                    DataType::Float16 => 2e-3,
                    DataType::BFloat16 => 2e-2,
                    DataType::Float32 | DataType::Float64 => 1e-5,
                    _ => unreachable!(),
                };
                for (got, want) in owned_float_as_f32(&output).iter().zip(&want) {
                    assert!(
                        (got - want).abs() < tolerance,
                        "{dtype:?} output: got {got}, want {want}"
                    );
                }
                for (got, want) in owned_float_as_f32(&inv_std).iter().zip(&want_inv_std) {
                    assert!(
                        (got - want).abs() < tolerance,
                        "{dtype:?} inv_std: got {got}, want {want}"
                    );
                }
            }
        }
    }

    fn simplified_layer_norm_kernel(domain: &str, opset: u64) -> Box<dyn Kernel> {
        let mut graph = Graph::new();
        graph.opset_imports.insert(domain.into(), opset);
        let x = graph.create_named_value("x", DataType::Float32, static_shape([2, 4]));
        let scale = graph.create_named_value("scale", DataType::Float32, static_shape([4]));
        let output = graph.create_named_value("output", DataType::Float32, static_shape([2, 4]));
        graph.add_input(x);
        graph.add_input(scale);
        let mut node = Node::new(
            NodeId(0),
            "SimplifiedLayerNormalization",
            vec![Some(x), Some(scale)],
            vec![output],
        );
        node.domain = domain.into();
        node.attributes
            .insert("epsilon".into(), Attribute::Float(1e-5));
        let node_id = graph.insert_node(node);
        graph.add_output(output);
        let model = Model::new(&graph);
        let proto = encode_model_proto(&model).unwrap();
        assert_eq!(
            proto.graph.as_ref().unwrap().node[0].domain,
            domain,
            "IR-to-proto conversion must preserve the operator domain"
        );
        CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node_id), &[], opset)
            .unwrap()
    }

    #[test]
    fn standard_simplified_layer_norm_matches_contrib_variant() {
        let x = Owned::f32(&[2, 4], &[1., 2., 3., 4., -2., 0., 2., 4.]);
        let scale = Owned::f32(&[4], &[1., 2., 0.5, 1.5]);
        let mut standard = Owned::zeros_f32(&[2, 4]);
        let mut contrib = Owned::zeros_f32(&[2, 4]);
        simplified_layer_norm_kernel("", 21)
            .execute(&[x.view(), scale.view()], &mut [standard.view_mut()])
            .unwrap();
        simplified_layer_norm_kernel("com.microsoft", 1)
            .execute(&[x.view(), scale.view()], &mut [contrib.view_mut()])
            .unwrap();
        assert_close(&standard.to_f32(), &contrib.to_f32());
    }

    /// A valid output-only SkipLayerNorm node (single output) must succeed:
    /// previously `check_arity` required ≥3 outputs and rejected it.
    #[test]
    fn skip_layer_norm_output_only_node_succeeds() {
        let x = Owned::f32(&[2, 4], &[1., 2., 3., 4., 2., 3., 4., 5.]);
        let skip = Owned::f32(&[2, 4], &[0.5, -1., 1., 0., 1., 0., -1., 2.]);
        let gamma = Owned::f32(&[4], &[1., 2., 0.5, 1.5]);
        let beta = Owned::f32(&[4], &[0., 1., -1., 0.5]);
        let mut y = Owned::zeros_f32(&[2, 4]);
        SkipLayerNormKernel { epsilon: 1e-5 }
            .execute(
                &[x.view(), skip.view(), gamma.view(), beta.view()],
                &mut [y.view_mut()],
            )
            .unwrap();

        let sum = [1.5, 1., 4., 4., 3., 3., 3., 7.];
        let gamma_data = [1., 2., 0.5, 1.5];
        let beta_data = [0., 1., -1., 0.5];
        let mut want = Vec::new();
        for row in sum.chunks_exact(4) {
            let m = row.iter().sum::<f32>() / 4.;
            let inv = 1. / (row.iter().map(|v| (v - m).powi(2)).sum::<f32>() / 4. + 1e-5).sqrt();
            want.extend((0..4).map(|i| (row[i] - m) * inv * gamma_data[i] + beta_data[i]));
        }
        assert_close(&y.to_f32(), &want);
    }

    /// A 4-output SkipLayerNorm node writes `output`, `mean`, `inv_std_var`, and
    /// `input_skip_bias_sum` (= X + skip + bias), all numerically correct.
    #[test]
    fn skip_layer_norm_writes_input_skip_bias_sum() {
        let x = Owned::f32(&[2, 4], &[1., 2., 3., 4., 2., 3., 4., 5.]);
        let skip = Owned::f32(&[2, 4], &[0.5, -1., 1., 0., 1., 0., -1., 2.]);
        let gamma = Owned::f32(&[4], &[1., 2., 0.5, 1.5]);
        let beta = Owned::f32(&[4], &[0., 1., -1., 0.5]);
        let bias = Owned::f32(&[4], &[0.25, 0., -0.5, 1.]);
        let mut y = Owned::zeros_f32(&[2, 4]);
        let mut mean = Owned::zeros_f32(&[2, 1]);
        let mut inv_std = Owned::zeros_f32(&[2, 1]);
        let mut skip_sum = Owned::zeros_f32(&[2, 4]);
        SkipLayerNormKernel { epsilon: 1e-5 }
            .execute(
                &[
                    x.view(),
                    skip.view(),
                    gamma.view(),
                    beta.view(),
                    bias.view(),
                ],
                &mut [
                    y.view_mut(),
                    mean.view_mut(),
                    inv_std.view_mut(),
                    skip_sum.view_mut(),
                ],
            )
            .unwrap();
        // sum = X + skip + bias, hand-computed for the [2,4] reference.
        let sum = [1.75f32, 1., 3.5, 5., 3.25, 3., 2.5, 8.];
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
        assert_close(&skip_sum.to_f32(), &sum);
    }

    /// `beta` absent while `bias` present: beta must be treated as 0 (no shift)
    /// and the present bias slot must still be added to the sum.
    #[test]
    fn skip_layer_norm_beta_absent_bias_present() {
        let x = Owned::f32(&[2, 4], &[1., 2., 3., 4., 2., 3., 4., 5.]);
        let skip = Owned::f32(&[2, 4], &[0.5, -1., 1., 0., 1., 0., -1., 2.]);
        let gamma = Owned::f32(&[4], &[1., 2., 0.5, 1.5]);
        let bias = Owned::f32(&[4], &[0.25, 0., -0.5, 1.]);
        let mut y = Owned::zeros_f32(&[2, 4]);
        SkipLayerNormKernel { epsilon: 1e-5 }
            .execute(
                &[
                    x.view(),
                    skip.view(),
                    gamma.view(),
                    TensorView::absent(DataType::Float32),
                    bias.view(),
                ],
                &mut [y.view_mut()],
            )
            .unwrap();
        let sum = [1.75f32, 1., 3.5, 5., 3.25, 3., 2.5, 8.];
        let gamma_data = [1., 2., 0.5, 1.5];
        let mut want = Vec::new();
        for row in sum.chunks_exact(4) {
            let m = row.iter().sum::<f32>() / 4.;
            let inv = 1. / (row.iter().map(|v| (v - m).powi(2)).sum::<f32>() / 4. + 1e-5).sqrt();
            want.extend((0..4).map(|i| (row[i] - m) * inv * gamma_data[i]));
        }
        assert_close(&y.to_f32(), &want);
    }

    /// `beta` present while `bias` absent: bias must be treated as 0 while the
    /// present beta slot still shifts the normalized output.
    #[test]
    fn skip_layer_norm_beta_present_bias_absent() {
        let x = Owned::f32(&[2, 4], &[1., 2., 3., 4., 2., 3., 4., 5.]);
        let skip = Owned::f32(&[2, 4], &[0.5, -1., 1., 0., 1., 0., -1., 2.]);
        let gamma = Owned::f32(&[4], &[1., 2., 0.5, 1.5]);
        let beta = Owned::f32(&[4], &[0., 1., -1., 0.5]);
        let mut y = Owned::zeros_f32(&[2, 4]);
        SkipLayerNormKernel { epsilon: 1e-5 }
            .execute(
                &[
                    x.view(),
                    skip.view(),
                    gamma.view(),
                    beta.view(),
                    TensorView::absent(DataType::Float32),
                ],
                &mut [y.view_mut()],
            )
            .unwrap();
        // bias absent → sum = X + skip only.
        let sum = [1.5f32, 1., 4., 4., 3., 3., 3., 7.];
        let gamma_data = [1., 2., 0.5, 1.5];
        let beta_data = [0., 1., -1., 0.5];
        let mut want = Vec::new();
        for row in sum.chunks_exact(4) {
            let m = row.iter().sum::<f32>() / 4.;
            let inv = 1. / (row.iter().map(|v| (v - m).powi(2)).sum::<f32>() / 4. + 1e-5).sqrt();
            want.extend((0..4).map(|i| (row[i] - m) * inv * gamma_data[i] + beta_data[i]));
        }
        assert_close(&y.to_f32(), &want);
    }

    /// SimplifiedLayerNorm must honour `axis` over multiple trailing dims and
    /// write the optional `InvStdDev` (per-group inv_rms, reduced shape).
    #[test]
    fn simplified_layer_norm_axis_multi_dim_and_inv_std() {
        // X=[2,2,2], axis=1 → norm_size=4, two groups. Scale broadcasts over
        // the normalized [2,2] block.
        let x_data = [1., 2., 3., 4., 5., 6., 7., 8.];
        let x = Owned::f32(&[2, 2, 2], &x_data);
        let scale = Owned::f32(&[2, 2], &[1., 2., 0.5, 1.5]);
        let mut out = Owned::zeros_f32(&[2, 2, 2]);
        let mut inv_std = Owned::zeros_f32(&[2, 1, 1]);
        let eps = 1e-5;
        SimplifiedLayerNormKernel {
            axis: 1,
            epsilon: eps,
        }
        .execute(
            &[x.view(), scale.view()],
            &mut [out.view_mut(), inv_std.view_mut()],
        )
        .unwrap();
        let scale_data = [1., 2., 0.5, 1.5];
        let mut want = Vec::new();
        let mut want_inv = Vec::new();
        for group in x_data.chunks_exact(4) {
            let inv = 1. / (group.iter().map(|v| v * v).sum::<f32>() / 4. + eps).sqrt();
            want_inv.push(inv);
            want.extend((0..4).map(|i| group[i] * inv * scale_data[i]));
        }
        assert_close(&out.to_f32(), &want);
        assert_close(&inv_std.to_f32(), &want_inv);
    }
}
