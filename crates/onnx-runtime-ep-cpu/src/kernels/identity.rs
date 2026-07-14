//! `Identity`: copy the input to the output unchanged (`docs/ORT2.md` §4.4).
//!
//! Unlike the arithmetic kernels, `Identity` is **dtype-agnostic**: it moves the
//! raw element bytes without interpreting them, so it works for every raw-layout
//! [`DataType`](onnx_runtime_ir::DataType) accepted by runtime tensor views
//! (f32/f64, all int/uint widths, bool, f16/bf16, and packed sub-byte types).
//! `String` tensors are represented out-of-band in the IR and are rejected at
//! the CPU tensor-view boundary rather than silently treated as zero-byte data.
//! This is what the graph builders used by conformance suites (e.g. spox) emit
//! to *name* a graph output, so supporting it unblocks running those models
//! end-to-end.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::check_arity;

/// Stateless, dtype-agnostic `Identity` kernel (raw byte copy).
pub struct IdentityKernel;

/// Factory for [`IdentityKernel`].
pub struct IdentityFactory;

impl KernelFactory for IdentityFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(IdentityKernel))
    }
}

impl Kernel for IdentityKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Identity", inputs, outputs, 1, 1, 1)?;
        let input = &inputs[0];
        let output = &mut outputs[0];

        if input.dtype == DataType::String || output.dtype == DataType::String {
            return Err(EpError::KernelFailed(format!(
                "Identity: String tensors are not supported by CPU execution tensor views \
                 because their payloads are stored out-of-band in the IR; refusing to copy \
                 {:?}{:?} to {:?}{:?}, which would otherwise silently lose all string data",
                input.dtype, input.shape, output.dtype, output.shape
            )));
        }

        // The executor always hands kernels contiguous row-major views, so a
        // flat byte copy is exact for any dtype. Guard the invariant instead of
        // silently corrupting data if that ever changes.
        if !input.is_contiguous() || !output.is_contiguous() {
            return Err(EpError::KernelFailed(
                "Identity: expected contiguous input/output views".to_string(),
            ));
        }
        let in_bytes = input.byte_size();
        let out_bytes = output.byte_size();
        if in_bytes != out_bytes {
            return Err(EpError::KernelFailed(format!(
                "Identity: input has {in_bytes} bytes but output has {out_bytes} \
                 (dtype/shape mismatch: in {:?}{:?} vs out {:?}{:?})",
                input.dtype, input.shape, output.dtype, output.shape
            )));
        }
        if in_bytes == 0 {
            return Ok(());
        }

        let src = input.data_ptr::<u8>();
        let dst = output.data_ptr_mut::<u8>();
        // SAFETY: both views are contiguous host allocations of `in_bytes`
        // (checked equal above); `src`/`dst` are the element origins the
        // executor bounds-gated before dispatch, and SSA guarantees the output
        // buffer is disjoint from the input, so the ranges do not overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, in_bytes);
        }
        Ok(())
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::{NodeId, compute_contiguous_strides};

    fn raw(dtype: DataType, shape: &[usize], bytes: &[u8]) -> Owned {
        Owned {
            bytes: bytes.to_vec(),
            shape: shape.to_vec(),
            strides: compute_contiguous_strides(shape),
            dtype,
        }
    }

    fn identity_kernel() -> Box<dyn Kernel> {
        let node = Node::new(NodeId(0), "Identity", vec![], vec![]);
        IdentityFactory.create(&node, &[]).unwrap()
    }

    #[test]
    fn identity_copies_f32() {
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        IdentityKernel
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![1., 2., 3., 4., 5., 6.]);
    }

    #[test]
    fn identity_is_bit_exact_for_bf16_f16_and_i32() {
        let cases = [
            (
                DataType::BFloat16,
                vec![0x00, 0x00, 0x01, 0x00, 0x80, 0x7f, 0xc1, 0x7f, 0x80, 0xff],
            ),
            (
                DataType::Float16,
                vec![0x00, 0x00, 0x01, 0x00, 0x00, 0x7c, 0x01, 0x7e, 0x00, 0xfc],
            ),
            (
                DataType::Int32,
                vec![
                    0x00, 0x00, 0x00, 0x80, 0xff, 0xff, 0xff, 0x7f, 0xef, 0xbe, 0xad, 0xde,
                ],
            ),
        ];

        for (dtype, expected) in cases {
            let elements = expected.len() / dtype.byte_size();
            let input = raw(dtype, &[elements], &expected);
            let mut output = Owned::zeros(dtype, &[elements]);

            identity_kernel()
                .execute(&[input.view()], &mut [output.view_mut()])
                .unwrap();

            assert_eq!(output.bytes, expected, "{dtype:?} bytes changed");
        }
    }

    #[test]
    fn identity_rejects_string_before_zero_byte_copy() {
        let input = raw(DataType::String, &[2], &[]);
        let mut output = raw(DataType::String, &[2], &[]);

        let error = identity_kernel()
            .execute(&[input.view()], &mut [output.view_mut()])
            .unwrap_err();

        let message = error.to_string();
        assert!(message.contains("String tensors are not supported"));
        assert!(message.contains("silently lose all string data"));
    }
}
