//! Fused `Silu(x) * y` for same-shape SwiGLU gates.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::activations::silu_f32_slice;
use super::check_arity;
use crate::dtype::{to_dense_f32_widen, to_dense_float, write_dense_f32_narrow, write_dense_float};

pub struct SiluMulFactory;

impl KernelFactory for SiluMulFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(SiluMulKernel))
    }
}

pub struct SiluMulKernel;

impl Kernel for SiluMulKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("SiluMul", inputs, outputs, 2, 2, 1)?;
        let dtype = inputs[0].dtype;
        if !matches!(
            dtype,
            DataType::Float16 | DataType::BFloat16 | DataType::Float32 | DataType::Float64
        ) || inputs[1].dtype != dtype
            || outputs[0].dtype != dtype
            || inputs[0].shape != inputs[1].shape
            || inputs[0].shape != outputs[0].shape
        {
            return Err(EpError::KernelFailed(
                "SiluMul: inputs and output must have one equal floating-point shape and dtype"
                    .into(),
            ));
        }

        if dtype == DataType::Float64 {
            let x = to_dense_float::<f64>(&inputs[0])?;
            let y = to_dense_float::<f64>(&inputs[1])?;
            let output = x
                .iter()
                .zip(y.iter())
                .map(|(&x, &y)| silu_f64(x) * y)
                .collect::<Vec<_>>();
            return write_dense_float::<f64>(&mut outputs[0], &output);
        }

        let x = to_dense_f32_widen("SiluMul", &inputs[0])?;
        let y = to_dense_f32_widen("SiluMul", &inputs[1])?;
        let mut output = vec![0.0; x.len()];
        // This is the same SIMD/runtime-gated routine as standalone SiLU, so its
        // approximation and exceptional-value behavior are preserved exactly.
        silu_f32_slice(&x, &mut output);
        match dtype {
            DataType::Float32 => {
                for (output, &y) in output.iter_mut().zip(y.iter()) {
                    *output *= y;
                }
            }
            // The unfused graph stores SiLU's result before Mul. Reproduce that
            // storage rounding before multiplying so f16/bf16 results are
            // byte-identical to the two-node graph.
            DataType::Float16 => {
                for (output, &y) in output.iter_mut().zip(y.iter()) {
                    *output = half::f16::from_f32(*output).to_f32() * y;
                }
            }
            DataType::BFloat16 => {
                for (output, &y) in output.iter_mut().zip(y.iter()) {
                    *output = half::bf16::from_f32(*output).to_f32() * y;
                }
            }
            _ => unreachable!("dtype was validated above"),
        }
        write_dense_f32_narrow("SiluMul", &mut outputs[0], &output)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn silu_f64(x: f64) -> f64 {
    if x >= 0.0 {
        x / (1.0 + (-x).exp())
    } else if x == f64::NEG_INFINITY {
        0.0
    } else {
        let e = x.exp();
        x * e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::activations::SiluFactory;
    use crate::kernels::elementwise::MulFactory;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::NodeId;

    fn random_values(n: usize) -> Vec<f32> {
        let mut state = 0x5EED_1234_u32;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state as i32 as f32) / i32::MAX as f32 * 24.0
            })
            .collect()
    }

    fn assert_fused_matches_unfused(dtype: DataType, aliased_operand: bool) {
        let mut x_values = random_values(257);
        x_values.extend_from_slice(&[-100.0, -18.5, -0.0, 0.0, 18.5, 100.0]);
        let y_values = if aliased_operand {
            x_values.clone()
        } else {
            random_values(x_values.len())
        };
        let x = match dtype {
            DataType::Float16 => Owned::f16(&[x_values.len()], &x_values),
            DataType::BFloat16 => Owned::bf16(&[x_values.len()], &x_values),
            DataType::Float32 => Owned::f32(&[x_values.len()], &x_values),
            _ => unreachable!(),
        };
        let y = match dtype {
            DataType::Float16 => Owned::f16(&[y_values.len()], &y_values),
            DataType::BFloat16 => Owned::bf16(&[y_values.len()], &y_values),
            DataType::Float32 => Owned::f32(&[y_values.len()], &y_values),
            _ => unreachable!(),
        };
        let mut silu = Owned::zeros(dtype, &[x_values.len()]);
        let mut unfused = Owned::zeros(dtype, &[x_values.len()]);
        let mut fused = Owned::zeros(dtype, &[x_values.len()]);
        let node = Node::new(NodeId(0), "Silu", vec![], vec![]);
        SiluFactory
            .create(&node, &[])
            .unwrap()
            .execute(&[x.view()], &mut [silu.view_mut()])
            .unwrap();
        MulFactory
            .create(&Node::new(NodeId(0), "Mul", vec![], vec![]), &[])
            .unwrap()
            .execute(&[silu.view(), y.view()], &mut [unfused.view_mut()])
            .unwrap();
        SiluMulFactory
            .create(&Node::new(NodeId(0), "SiluMul", vec![], vec![]), &[])
            .unwrap()
            .execute(&[x.view(), y.view()], &mut [fused.view_mut()])
            .unwrap();
        assert_eq!(fused.bytes, unfused.bytes, "dtype {dtype:?}");
    }

    #[test]
    fn silu_mul_is_byte_identical_to_silu_then_mul() {
        for dtype in [DataType::Float16, DataType::BFloat16, DataType::Float32] {
            assert_fused_matches_unfused(dtype, false);
        }
    }

    #[test]
    fn aliased_silu_mul_is_byte_identical_to_silu_then_mul() {
        for dtype in [DataType::Float16, DataType::BFloat16, DataType::Float32] {
            assert_fused_matches_unfused(dtype, true);
        }
    }
}
