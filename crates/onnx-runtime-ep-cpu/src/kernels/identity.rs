//! `Identity`: copy the input to the output unchanged (`docs/architecture/ORT2.md` §4.4).
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

        input.validate().map_err(|e| {
            EpError::KernelFailed(format!(
                "Identity: invalid input: {e}; bind a valid raw tensor"
            ))
        })?;
        output.validate().map_err(|e| {
            EpError::KernelFailed(format!(
                "Identity: invalid output: {e}; bind a valid raw tensor"
            ))
        })?;
        // This raw byte-copy implementation requires contiguous logical views.
        // Explicit bindings may carry strides, so refuse unsupported layouts.
        if !input.is_contiguous() || !output.is_contiguous() {
            return Err(EpError::KernelFailed(format!(
                "Identity: expected contiguous input/output views, got strides {:?}/{:?}; bind contiguous views",
                input.strides, output.strides
            )));
        }
        // Shared view validation already proves concrete raw dtype and complete
        // dense count/byte arithmetic, including packed storage and empty shapes.
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

        // Complete both Host/span proofs before reading or writing any bytes.
        for (name, base, offset, device) in [
            (
                "input",
                input.data.0 as usize,
                input.byte_offset,
                input.device,
            ),
            (
                "output",
                output.data.0 as usize,
                output.byte_offset,
                output.device,
            ),
        ] {
            if !device.is_host_accessible() {
                return Err(EpError::KernelFailed(format!(
                    "Identity: {name} on {device:?} is not Host-accessible; explicitly transfer before CPU copying"
                )));
            }
            if std::alloc::Layout::array::<u8>(in_bytes).is_err()
                || base
                    .checked_add(offset)
                    .and_then(|p| p.checked_add(in_bytes))
                    .is_none()
            {
                return Err(EpError::KernelFailed(format!(
                    "Identity: {name} byte_offset {offset} and {in_bytes} storage bytes exceed addressable Host backing; bind valid smaller storage"
                )));
            }
        }
        let src = input.data_ptr::<u8>();
        let dst = output.data_ptr_mut::<u8>();
        // SAFETY: validated contiguous Host spans of equal checked byte count;
        // the owner supplies live backing bounds. Explicit I/O bindings may
        // alias, so copying must support identical and shifted overlaps.
        unsafe {
            std::ptr::copy(src, dst, in_bytes);
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
    #[test]
    fn identity_preserves_raw_bits_for_exact_and_shifted_overlap() {
        use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut};
        use onnx_runtime_ir::DeviceId;
        for (dtype, count, size, shifts) in [
            (DataType::Float32, 2usize, 8usize, [0usize, 4, 8]),
            (DataType::Uint4, 5, 3, [3, 4, 5]),
        ] {
            for dst in shifts {
                let mut backing = (0..24).map(|n| (n * 17 + 3) as u8).collect::<Vec<_>>();
                let mut expected = backing.clone();
                expected.copy_within(4..4 + size, dst);
                let shape = [count];
                let input = TensorView::new(
                    DevicePtr(backing.as_ptr().cast()),
                    dtype,
                    &shape,
                    &[1],
                    DeviceId::cpu(),
                )
                .with_byte_offset(4);
                let output = TensorMut::new(
                    DevicePtrMut(backing.as_mut_ptr().cast()),
                    dtype,
                    &shape,
                    &[1],
                    DeviceId::cpu(),
                )
                .with_byte_offset(dst);
                IdentityKernel.execute(&[input], &mut [output]).unwrap();
                assert_eq!(backing, expected, "{dtype:?} dst={dst}");
            }
        }
    }
    #[test]
    fn identity_refuses_nonhost_or_wrapping_storage_before_copying() {
        use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut};
        use onnx_runtime_ir::DeviceId;
        for bad in 0..4 {
            let mut backing = [123u8; 8];
            let mut input = TensorView::new(
                DevicePtr(8usize as *const std::ffi::c_void),
                DataType::Uint8,
                &[3],
                &[1],
                DeviceId::cpu(),
            );
            let mut output = TensorMut::new(
                DevicePtrMut(backing.as_mut_ptr().cast()),
                DataType::Uint8,
                &[3],
                &[1],
                DeviceId::cpu(),
            );
            match bad {
                0 => input.device = DeviceId::cuda(0),
                1 => {
                    output.device = DeviceId::cuda(0);
                    output.data = DevicePtrMut(8usize as *mut std::ffi::c_void);
                }
                2 => input.data = DevicePtr((usize::MAX - 1) as *const std::ffi::c_void),
                _ => output.data = DevicePtrMut((usize::MAX - 1) as *mut std::ffi::c_void),
            }
            let text = IdentityKernel
                .execute(&[input], &mut [output])
                .unwrap_err()
                .to_string();
            assert!(
                text.contains("Identity")
                    && text.contains(if bad < 2 {
                        "Host-accessible"
                    } else {
                        "addressable Host backing"
                    }),
                "{text}"
            );
            assert_eq!(backing, [123; 8]);
        }
        let input = TensorView::new(
            DevicePtr(8usize as *const std::ffi::c_void),
            DataType::Float64,
            &[usize::MAX],
            &[1],
            DeviceId::cpu(),
        );
        let output = TensorMut::new(
            DevicePtrMut(8usize as *mut std::ffi::c_void),
            DataType::Float64,
            &[usize::MAX],
            &[1],
            DeviceId::cpu(),
        );
        let text = IdentityKernel
            .execute(&[input], &mut [output])
            .unwrap_err()
            .to_string();
        assert!(
            text.contains("input")
                && text.contains("dense byte count overflows usize")
                && text.contains("Float64")
                && text.contains(&usize::MAX.to_string()),
            "{text}"
        );
    }
}
