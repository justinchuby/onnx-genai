//! Attribute-driven f32 activation kernels.

use onnx_runtime_ep_api::{Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::{check_arity, to_dense_f32, write_dense_f32};

#[derive(Clone, Copy)]
enum Activation {
    Elu { alpha: f32 },
    LeakyRelu { alpha: f32 },
    HardSigmoid { alpha: f32, beta: f32 },
}

impl Activation {
    fn name(self) -> &'static str {
        match self {
            Self::Elu { .. } => "Elu",
            Self::LeakyRelu { .. } => "LeakyRelu",
            Self::HardSigmoid { .. } => "HardSigmoid",
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
        }
    }
}

pub struct ActivationKernel {
    activation: Activation,
}

pub struct EluFactory;
pub struct LeakyReluFactory;
pub struct HardSigmoidFactory;

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

impl Kernel for ActivationKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(self.activation.name(), inputs, outputs, 1, 1, 1)?;
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
}
