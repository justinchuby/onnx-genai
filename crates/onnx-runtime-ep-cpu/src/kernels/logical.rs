//! `Not`: logical negation of a boolean tensor (`docs/ORT2.md` §4.4).
//!
//! ONNX `Bool` tensors store one byte per element (`0` = false, non-zero =
//! true). `Not` flips each element, emitting canonical `1`/`0` bytes. It is a
//! straight per-element map over the raw bytes.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::{check_arity, to_dense_bytes, write_dense_bytes};

/// Stateless `Not` kernel (boolean element negation).
pub struct NotKernel;

/// Factory for [`NotKernel`] (no attributes).
pub struct NotFactory;

impl KernelFactory for NotFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(NotKernel))
    }
}

impl Kernel for NotKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Not", inputs, outputs, 1, 1, 1)?;
        if inputs[0].dtype != DataType::Bool || outputs[0].dtype != DataType::Bool {
            return Err(EpError::KernelFailed(format!(
                "Not: requires Bool input and output, got input {:?} / output {:?}. WHY: `Not` \
                 is a logical op defined only on booleans. HOW: feed a Bool tensor.",
                inputs[0].dtype, outputs[0].dtype
            )));
        }
        let bytes = to_dense_bytes(&inputs[0])?;
        let out: Vec<u8> = bytes.iter().map(|&b| u8::from(b == 0)).collect();
        write_dense_bytes(&mut outputs[0], &out)
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
    fn not_flips_bools() {
        let a = Owned::bool_(&[4], &[true, false, true, false]);
        let mut out = Owned::zeros(DataType::Bool, &[4]);
        NotKernel
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_bool(), vec![false, true, false, true]);
    }

    #[test]
    fn not_rejects_non_bool() {
        let a = Owned::f32(&[2], &[1., 0.]);
        let mut out = Owned::zeros_f32(&[2]);
        let err = NotKernel.execute(&[a.view()], &mut [out.view_mut()]);
        assert!(err.is_err());
    }
}
