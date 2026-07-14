//! `Relu`: elementwise `max(0, x)` for f32 (`docs/ORT2.md` §4.4).

use onnx_runtime_ep_api::{Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::{check_arity, to_dense_f32, write_dense_f32};

/// Apply `max(0, x)` in place. Shared with the fused `FusedGemm` kernel so the
/// ReLU activation has a single source of truth.
pub(crate) fn relu_in_place(data: &mut [f32]) {
    for v in data.iter_mut() {
        *v = v.max(0.0);
    }
}

/// Stateless f32 ReLU kernel.
pub struct ReluKernel;

/// Factory for [`ReluKernel`] (no attributes).
pub struct ReluFactory;

impl KernelFactory for ReluFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ReluKernel))
    }
}

impl Kernel for ReluKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Relu", inputs, outputs, 1, 1, 1)?;
        let x = to_dense_f32(&inputs[0])?;
        let mut y = x;
        relu_in_place(&mut y);
        write_dense_f32(&mut outputs[0], &y)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }

    fn estimated_flops(&self) -> Option<u64> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    #[test]
    fn relu_clamps_negatives() {
        let a = Owned::f32(&[2, 2], &[-1.0, 2.0, -3.0, 4.0]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        let inputs = [a.view()];
        let mut outs = [out.view_mut()];
        ReluKernel.execute(&inputs, &mut outs).unwrap();
        assert_eq!(out.to_f32(), vec![0.0, 2.0, 0.0, 4.0]);
    }
}
