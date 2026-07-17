//! Attribute-driven f32 activation kernels.

use onnx_runtime_ep_api::{Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::{check_arity, to_dense_f32, write_dense_f32};

#[derive(Clone, Copy)]
enum Activation {
    Elu { alpha: f32 },
    LeakyRelu { alpha: f32 },
    HardSigmoid { alpha: f32, beta: f32 },
    Swish { alpha: f32 },
    Silu,
}

impl Activation {
    fn name(self) -> &'static str {
        match self {
            Self::Elu { .. } => "Elu",
            Self::LeakyRelu { .. } => "LeakyRelu",
            Self::HardSigmoid { .. } => "HardSigmoid",
            Self::Swish { .. } => "Swish",
            Self::Silu => "Silu",
        }
    }

    fn apply(self, x: f32) -> f32 {
        match self {
            Self::Elu { alpha } => {
                if x >= 0.0 {
                    x
                } else {
                    alpha * x.exp_m1()
                }
            }
            Self::LeakyRelu { alpha } => {
                if x >= 0.0 {
                    x
                } else {
                    alpha * x
                }
            }
            Self::HardSigmoid { alpha, beta } => (alpha * x + beta).clamp(0.0, 1.0),
            // Swish/SiLU: x·sigmoid(alpha·x), evaluated via the numerically
            // stable logistic to avoid overflow at large-magnitude inputs.
            Self::Swish { alpha } => {
                let z = alpha * x;
                let s = if z >= 0.0 {
                    1.0 / (1.0 + (-z).exp())
                } else {
                    let e = z.exp();
                    e / (1.0 + e)
                };
                x * s
            }
            Self::Silu => silu(x),
        }
    }
}

fn silu(x: f32) -> f32 {
    // CUDA's device exp is evaluated in f64. Match that precision before the
    // f32 operation-order boundary so 1-ulp exp differences cannot be amplified
    // by downstream accuracy-level-4 activation quantization.
    if x >= 0.0 {
        x / (1.0 + ((-x) as f64).exp() as f32)
    } else {
        let e = (x as f64).exp() as f32;
        x * e / (1.0 + e)
    }
}

pub struct ActivationKernel {
    activation: Activation,
}

pub struct EluFactory;
pub struct LeakyReluFactory;
pub struct HardSigmoidFactory;
pub struct SwishFactory;
pub struct SiluFactory;

impl KernelFactory for EluFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ActivationKernel {
            activation: Activation::Elu {
                alpha: node.attr("alpha").and_then(|a| a.as_float()).unwrap_or(1.0),
            },
        }))
    }
}

impl KernelFactory for LeakyReluFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ActivationKernel {
            activation: Activation::LeakyRelu {
                alpha: node
                    .attr("alpha")
                    .and_then(|a| a.as_float())
                    .unwrap_or(0.01),
            },
        }))
    }
}

impl KernelFactory for HardSigmoidFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ActivationKernel {
            activation: Activation::HardSigmoid {
                alpha: node.attr("alpha").and_then(|a| a.as_float()).unwrap_or(0.2),
                beta: node.attr("beta").and_then(|a| a.as_float()).unwrap_or(0.5),
            },
        }))
    }
}

impl KernelFactory for SwishFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ActivationKernel {
            activation: Activation::Swish {
                alpha: node.attr("alpha").and_then(|a| a.as_float()).unwrap_or(1.0),
            },
        }))
    }
}

impl KernelFactory for SiluFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ActivationKernel {
            activation: Activation::Silu,
        }))
    }
}

impl Kernel for ActivationKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(self.activation.name(), inputs, outputs, 1, 1, 1)?;
        if matches!(self.activation, Activation::Silu)
            && silu_contiguous_f32(&inputs[0], &mut outputs[0])
        {
            return Ok(());
        }
        let y = to_dense_f32(&inputs[0])?
            .into_iter()
            .map(|x| self.activation.apply(x))
            .collect::<Vec<_>>();
        write_dense_f32(&mut outputs[0], &y)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn silu_contiguous_f32(input: &TensorView, output: &mut TensorMut) -> bool {
    if input.dtype != DataType::Float32
        || output.dtype != DataType::Float32
        || input.shape != output.shape
        || !input.is_contiguous()
        || !output.is_contiguous()
    {
        return false;
    }

    let n = output.numel();
    let bytes = n.saturating_mul(std::mem::size_of::<f32>());
    let input_start = input.data_ptr::<f32>() as usize;
    let input_end = input_start.saturating_add(bytes);
    let output_start = output.data_ptr_mut::<f32>() as usize;
    let output_end = output_start.saturating_add(bytes);
    if output_start < input_end && input_start < output_end {
        return false;
    }

    // SAFETY: executor bounds checks plus equal contiguous f32 shapes prove both
    // pointers span n elements; the range check proves output does not alias input.
    let input = unsafe { std::slice::from_raw_parts(input.data_ptr::<f32>(), n) };
    let output = unsafe { std::slice::from_raw_parts_mut(output.data_ptr_mut::<f32>(), n) };
    for (output, &input) in output.iter_mut().zip(input) {
        *output = silu(input);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    #[test]
    fn activation_formulas_and_defaults() {
        let x = Owned::f32(&[3], &[-1.0, 0.0, 1.0]);
        let mut out = Owned::zeros_f32(&[3]);
        ActivationKernel {
            activation: Activation::Elu { alpha: 1.0 },
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
        assert!((out.to_f32()[0] - ((-1.0f32).exp() - 1.0)).abs() < 1e-6);
        ActivationKernel {
            activation: Activation::LeakyRelu { alpha: 0.1 },
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_f32(), vec![-0.1, 0.0, 1.0]);
        ActivationKernel {
            activation: Activation::HardSigmoid {
                alpha: 0.2,
                beta: 0.5,
            },
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_f32(), vec![0.3, 0.5, 0.7]);
    }

    #[test]
    fn swish_default_and_alpha() {
        let x = Owned::f32(&[3], &[-1.0, 0.0, 2.0]);
        let mut out = Owned::zeros_f32(&[3]);
        // alpha=1 (SiLU): y = x·sigmoid(x).
        ActivationKernel {
            activation: Activation::Swish { alpha: 1.0 },
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
        let sig = |z: f32| 1.0 / (1.0 + (-z).exp());
        let want = [-1.0 * sig(-1.0), 0.0, 2.0 * sig(2.0)];
        for (g, w) in out.to_f32().iter().zip(&want) {
            assert!((g - w).abs() < 1e-6, "got {g}, want {w}");
        }
        // alpha=2: y = x·sigmoid(2x).
        ActivationKernel {
            activation: Activation::Swish { alpha: 2.0 },
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
        let want2 = [-1.0 * sig(-2.0), 0.0, 2.0 * sig(4.0)];
        for (g, w) in out.to_f32().iter().zip(&want2) {
            assert!((g - w).abs() < 1e-6, "got {g}, want {w}");
        }
    }

    #[test]
    fn silu_contiguous_matches_reference() {
        let x = Owned::f32(&[6], &[-100.0, -2.0, -0.0, 0.0, 2.0, 100.0]);
        let mut out = Owned::zeros_f32(&[6]);
        ActivationKernel {
            activation: Activation::Silu,
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
        for (got, input) in out
            .to_f32()
            .into_iter()
            .zip([-100.0f32, -2.0, -0.0, 0.0, 2.0, 100.0])
        {
            let want = input * (1.0 / (1.0 + (-input).exp()));
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[test]
    fn silu_strided_falls_back_correctly() {
        let mut x = Owned::f32(&[2, 2], &[-2.0, -1.0, 1.0, 2.0]);
        x.strides = vec![1, 2];
        let mut out = Owned::zeros_f32(&[2, 2]);
        ActivationKernel {
            activation: Activation::Silu,
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
        let logical = [-2.0f32, 1.0, -1.0, 2.0];
        for (got, input) in out.to_f32().into_iter().zip(logical) {
            let want = input * (1.0 / (1.0 + (-input).exp()));
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }
}
