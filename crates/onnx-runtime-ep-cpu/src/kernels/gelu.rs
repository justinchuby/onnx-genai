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

use super::check_arity;
use super::elementwise::erf;
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};

/// `1/√2`, evaluated in `f64` to match the `erf` argument precision.
const FRAC_1_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Exact GELU of one element: `0.5·x·(1 + erf(x / √2))`. The inner scale and
/// error function are computed in `f64` (as the `Erf` kernel does) and the
/// result rounded to `f32`.
pub(crate) fn exact_gelu(x: f32) -> f32 {
    if x == f32::NEG_INFINITY {
        return 0.0;
    }
    let xf = x as f64;
    (0.5 * xf * (1.0 + erf(xf * FRAC_1_SQRT_2))) as f32
}

/// `√(2/π)`, the outer scale of the tanh GELU approximation, in `f64`.
const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;

/// Tanh-approximation GELU of one element (the standard `ai.onnx::Gelu`
/// `approximate="tanh"` path): `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`.
/// Computed in `f64` and rounded to `f32` to mirror the exact path's precision.
pub(crate) fn tanh_gelu(x: f32) -> f32 {
    if x == f32::NEG_INFINITY {
        return 0.0;
    }
    let xf = x as f64;
    let inner = SQRT_2_OVER_PI * (xf + 0.044_715 * xf * xf * xf);
    (0.5 * xf * (1.0 + inner.tanh())) as f32
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
        let x = to_dense_f32_widen("Gelu", &inputs[0])?;
        let y: Vec<f32> = x.iter().map(|&v| exact_gelu(v)).collect();
        write_dense_f32_narrow("Gelu", &mut outputs[0], &y)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

/// Standard-domain `ai.onnx::Gelu` (opset 20) kernel honoring the `approximate`
/// attribute: `"none"` (default) uses the exact erf path, `"tanh"` uses the
/// tanh approximation. Distinct from [`GeluKernel`] (the `com.microsoft` exact
/// op) so the contrib op's behavior is untouched.
pub struct StdGeluKernel {
    /// `true` for `approximate="tanh"`, `false` for the exact erf path.
    tanh: bool,
}

/// Factory for [`StdGeluKernel`], reading the `approximate` attribute.
pub struct StdGeluFactory;

impl KernelFactory for StdGeluFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let tanh = node
            .attr("approximate")
            .and_then(|a| a.as_str())
            .map(|s| s == "tanh")
            .unwrap_or(false);
        Ok(Box::new(StdGeluKernel { tanh }))
    }
}

impl Kernel for StdGeluKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Gelu", inputs, outputs, 1, 1, 1)?;
        let x = to_dense_f32_widen("Gelu", &inputs[0])?;
        let y: Vec<f32> = if self.tanh {
            x.iter().map(|&v| tanh_gelu(v)).collect()
        } else {
            x.iter().map(|&v| exact_gelu(v)).collect()
        };
        write_dense_f32_narrow("Gelu", &mut outputs[0], &y)
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
    fn gelu_bf16_matches_widened_f32_reference() {
        let xs = [-3.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
        let a = Owned::f32(&[xs.len()], &xs);
        let mut ref_out = Owned::zeros_f32(&[xs.len()]);
        GeluKernel
            .execute(&[a.view()], &mut [ref_out.view_mut()])
            .unwrap();

        let a = Owned::bf16(&[xs.len()], &xs);
        let mut bf16_out = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[xs.len()]);
        GeluKernel
            .execute(&[a.view()], &mut [bf16_out.view_mut()])
            .unwrap();
        for (&r, &g) in ref_out
            .to_f32()
            .iter()
            .zip(bf16_out.to_bf16_as_f32().iter())
        {
            assert!(
                (r - g).abs() <= 0.03 * r.abs().max(1.0),
                "gelu bf16 {g} vs f32 {r}"
            );
        }
    }

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

    /// Reference tanh-approximation GELU computed independently in `f64`.
    fn reference_tanh(x: f32) -> f32 {
        let xf = x as f64;
        let inner = (2.0f64 / std::f64::consts::PI).sqrt() * (xf + 0.044_715 * xf * xf * xf);
        (0.5 * xf * (1.0 + inner.tanh())) as f32
    }

    #[test]
    fn std_gelu_approximate_none_matches_exact() {
        let xs = [-3.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
        let a = Owned::f32(&[xs.len()], &xs);
        let mut out = Owned::zeros_f32(&[xs.len()]);
        StdGeluKernel { tanh: false }
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap();
        for (&x, &g) in xs.iter().zip(out.to_f32().iter()) {
            assert!((g - reference(x)).abs() <= 1e-6, "gelu({x}) = {g}");
        }
    }

    #[test]
    fn std_gelu_approximate_tanh_matches_reference() {
        let xs = [-3.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
        let a = Owned::f32(&[xs.len()], &xs);
        let mut out = Owned::zeros_f32(&[xs.len()]);
        StdGeluKernel { tanh: true }
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap();
        let got = out.to_f32();
        for (&x, &g) in xs.iter().zip(got.iter()) {
            assert!(
                (g - reference_tanh(x)).abs() <= 1e-6,
                "gelu_tanh({x}) = {g}"
            );
        }
        // gelu(0) is exactly 0 in both paths.
        assert_eq!(got[3], 0.0);
        // Hand value: gelu_tanh(1) ≈ 0.841192.
        assert!(
            (got[5] - 0.841_192).abs() < 1e-4,
            "gelu_tanh(1) = {}",
            got[5]
        );
    }

    #[test]
    fn std_gelu_factory_reads_approximate_attr() {
        use onnx_runtime_ir::{Attribute, Node, NodeId};
        let mut node = Node::new(NodeId(0), "Gelu", vec![], vec![]);
        node.attributes.insert(
            "approximate".to_string(),
            Attribute::String(b"tanh".to_vec()),
        );
        let k = StdGeluFactory.create(&node, &[]).unwrap();
        let a = Owned::f32(&[1], &[1.0]);
        let mut out = Owned::zeros_f32(&[1]);
        k.execute(&[a.view()], &mut [out.view_mut()]).unwrap();
        assert!((out.to_f32()[0] - reference_tanh(1.0)).abs() < 1e-6);
    }
}
