//! `com.microsoft::Gelu`: the exact (error-function) Gaussian Error Linear
//! Unit, `0.5·x·(1 + erf(x / √2))`, elementwise for f32.
//!
//! Emitted by the optimizer's GELU fusion (`onnx_runtime_optimizer::fusion`),
//! which collapses the op-by-op `Mul/Div → Erf → Add → Mul` decomposition into
//! a single node. The error function is the SAME `erf` helper the standalone
//! `Erf` kernel uses (`super::elementwise::erf`), so the fused op is numerically
//! identical to the decomposition it replaces.

use onnx_runtime_ep_api::{Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::elementwise::erf;
use super::{check_arity, to_dense_f32, write_dense_f32};

/// `1/√2`, evaluated in `f64` to match the `erf` argument precision.
const FRAC_1_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Exact GELU of one element: `0.5·x·(1 + erf(x / √2))`. The inner scale and
/// error function are computed in `f64` (as the `Erf` kernel does) and the
/// result rounded to `f32`.
fn gelu(x: f32) -> f32 {
    let xf = x as f64;
    (0.5 * xf * (1.0 + erf(xf * FRAC_1_SQRT_2))) as f32
}

/// Stateless f32 exact-GELU kernel.
pub struct GeluKernel;

/// Factory for [`GeluKernel`] (no attributes).
pub struct GeluFactory;

impl KernelFactory for GeluFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(GeluKernel))
    }
}

impl Kernel for GeluKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Gelu", inputs, outputs, 1, 1, 1)?;
        let x = to_dense_f32(&inputs[0])?;
        let y: Vec<f32> = x.iter().map(|&v| gelu(v)).collect();
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

    /// Reference GELU computed independently in `f64` with the crate's `erf`.
    fn reference(x: f32) -> f32 {
        let xf = x as f64;
        (0.5 * xf * (1.0 + erf(xf / std::f64::consts::SQRT_2))) as f32
    }

    #[test]
    fn gelu_known_values() {
        // gelu(0)=0; gelu is monotone & odd-ish: gelu(-x) = x - gelu(x) is NOT
        // exact, so assert against the reference and a couple of hand values.
        let xs = [-3.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
        let a = Owned::f32(&[xs.len()], &xs);
        let mut out = Owned::zeros_f32(&[xs.len()]);
        GeluKernel
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap();
        let got = out.to_f32();
        for (&x, &g) in xs.iter().zip(got.iter()) {
            assert!((g - reference(x)).abs() <= 1e-6, "gelu({x}) = {g}");
        }
        // gelu(0) is exactly 0.
        assert_eq!(got[xs.iter().position(|&v| v == 0.0).unwrap()], 0.0);
    }

    #[test]
    fn gelu_reads_strided_view() {
        // Backing [2,2] row-major; expose transposed [2,2] with strides [1,2].
        let a = Owned::f32(&[2, 2], &[-1.0, 1.0, 2.0, -2.0]).with_view(&[2, 2], &[1, 2]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        GeluKernel
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap();
        // Transposed read order: [-1, 2, 1, -2].
        let expect: Vec<f32> = [-1.0f32, 2.0, 1.0, -2.0]
            .iter()
            .map(|&v| reference(v))
            .collect();
        assert_eq!(out.to_f32(), expect);
    }
}
