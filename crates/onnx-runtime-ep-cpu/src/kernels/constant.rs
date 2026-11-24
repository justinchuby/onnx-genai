//! `Constant`: materialize a compile-time constant tensor from an attribute
//! (`docs/architecture/ORT2.md` §4.4).
//!
//! Handles tensor-valued `value` (including packed numeric tensors) and the
//! numeric `value_float`, `value_floats`, `value_int`, and `value_ints` forms.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node, TensorData};

use super::{check_arity, write_dense_bytes};

enum ConstantValue {
    ByteAligned(Vec<u8>),
    Packed(TensorData),
}

/// Constant kernel carrying the decoded value (`None` if no supported
/// value attribute was present; execution then errors).
pub struct ConstantKernel {
    value: Option<ConstantValue>,
}

/// Factory decoding the constant's value attribute.
pub struct ConstantFactory;

impl KernelFactory for ConstantFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ConstantKernel {
            value: decode_value(node),
        }))
    }
}

fn decode_value(node: &Node) -> Option<ConstantValue> {
    if let Some(Attribute::Tensor(t)) = node.attr("value") {
        // Only the packed types in the pinned ONNX Constant schema. The IR also
        // represents FLOAT6, which that schema does not admit.
        return Some(
            if matches!(
                t.dtype,
                DataType::Int4
                    | DataType::Uint4
                    | DataType::Float4E2M1
                    | DataType::Int2
                    | DataType::Uint2
            ) {
                ConstantValue::Packed(t.clone())
            } else {
                ConstantValue::ByteAligned(t.data.clone())
            },
        );
    }
    decode_numeric_bytes(node).map(ConstantValue::ByteAligned)
}

fn decode_numeric_bytes(node: &Node) -> Option<Vec<u8>> {
    if let Some(Attribute::Float(f)) = node.attr("value_float") {
        return Some(f.to_le_bytes().to_vec());
    }
    if let Some(Attribute::Floats(v)) = node.attr("value_floats") {
        return Some(v.iter().flat_map(|f| f.to_le_bytes()).collect());
    }
    if let Some(Attribute::Int(i)) = node.attr("value_int") {
        return Some(i.to_le_bytes().to_vec());
    }
    if let Some(Attribute::Ints(v)) = node.attr("value_ints") {
        return Some(v.iter().flat_map(|i| i.to_le_bytes()).collect());
    }
    None
}

impl Kernel for ConstantKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Constant", inputs, outputs, 0, 0, 1)?;
        let out = &mut outputs[0];
        match &self.value {
            Some(ConstantValue::ByteAligned(bytes)) => write_dense_bytes(out, bytes),
            Some(ConstantValue::Packed(value)) => write_packed_value(out, value),
            None => Err(EpError::KernelFailed(
                "Constant: no supported value attribute".into(),
            )),
        }
    }
}

fn write_packed_value(out: &mut TensorMut<'_>, value: &TensorData) -> Result<()> {
    if out.dtype != value.dtype || out.shape != value.dims.as_slice() {
        return Err(EpError::KernelFailed(format!(
            "Constant: packed value has dtype {:?} and shape {:?}, but output declares {:?} {:?}; the output must match the value tensor",
            value.dtype, value.dims, out.dtype, out.shape
        )));
    }
    let expected = value.checked_expected_bytes().ok_or_else(|| {
        EpError::KernelFailed(format!(
            "Constant: packed storage size overflows for dtype {:?} and shape {:?}",
            value.dtype, value.dims
        ))
    })?;
    if value.data.len() != expected {
        return Err(EpError::KernelFailed(format!(
            "Constant: packed value for {:?} {:?} has {} bytes; expected {expected}",
            value.dtype,
            value.dims,
            value.data.len()
        )));
    }
    out.validate()?;
    if expected == 0 {
        return Ok(());
    }
    if !out.is_contiguous() {
        return Err(EpError::KernelFailed(
            "Constant: packed output must be contiguous row-major; use a contiguous output buffer"
                .into(),
        ));
    }
    // SAFETY: the CPU caller supplies a writable allocation for the contiguous
    // output at its byte origin. Its dtype/shape imply exactly `expected` bytes
    // there, and the attribute storage is owned separately by the kernel.
    unsafe {
        std::ptr::copy_nonoverlapping(value.data.as_ptr(), out.data_ptr_mut::<u8>(), expected);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ep_api::{DevicePtrMut, ExecutionProvider};
    use onnx_runtime_ir::{DeviceId, NodeId, compute_contiguous_strides};

    use crate::CpuExecutionProvider;

    fn kernel_from(node: &Node) -> Box<dyn Kernel> {
        ConstantFactory.create(node, &[]).unwrap()
    }

    fn node_with(name: &str, attr: Attribute) -> Node {
        let mut n = Node::new(NodeId(0), "Constant", vec![], vec![]);
        n.attributes.insert(name.to_string(), attr);
        n
    }

    #[test]
    fn constant_from_tensor_value() {
        let t = TensorData::from_raw(
            DataType::Float32,
            vec![2, 2],
            [1.0f32, 2.0, 3.0, 4.0]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect(),
        );
        let node = node_with("value", Attribute::Tensor(t));
        let k = kernel_from(&node);
        let mut out = Owned::zeros_f32(&[2, 2]);
        k.execute(&[], &mut [out.view_mut()]).unwrap();
        assert_eq!(out.to_f32(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn constant_from_value_float_scalar() {
        let node = node_with("value_float", Attribute::Float(3.5));
        let k = kernel_from(&node);
        let mut out = Owned::zeros_f32(&[]);
        k.execute(&[], &mut [out.view_mut()]).unwrap();
        assert_eq!(out.to_f32(), vec![3.5]);
    }

    #[test]
    fn constant_from_value_ints() {
        let node = node_with("value_ints", Attribute::Ints(vec![7, 8, 9]));
        let k = kernel_from(&node);
        let mut out = Owned::zeros(DataType::Int64, &[3]);
        k.execute(&[], &mut [out.view_mut()]).unwrap();
        assert_eq!(out.to_i64(), vec![7, 8, 9]);
    }

    #[test]
    fn constant_missing_value_errors() {
        let node = Node::new(NodeId(0), "Constant", vec![], vec![]);
        let k = kernel_from(&node);
        let mut out = Owned::zeros_f32(&[1]);
        assert!(k.execute(&[], &mut [out.view_mut()]).is_err());
    }
    #[test]
    fn constant_tensor_bf16_preserves_raw_bits() {
        let bits = [0x3f80u16, 0x7fc1, 0x8000];
        let bytes = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
        let node = node_with(
            "value",
            Attribute::Tensor(TensorData::from_raw(DataType::BFloat16, vec![3], bytes)),
        );
        let k = kernel_from(&node);
        let mut out = Owned::zeros(DataType::BFloat16, &[3]);
        k.execute(&[], &mut [out.view_mut()]).unwrap();
        assert_eq!(out.to_u16_bits(), bits);
    }

    #[test]
    fn constant_packed_tensor_opsets_preserve_exact_storage() {
        let ep = CpuExecutionProvider::new();
        // First Constant schema admitting each dtype at the pinned ONNX commit.
        let cases: &[(u64, DataType, &[usize], &[u8])] = &[
            (21, DataType::Int4, &[3], &[0x78, 0x0f]),
            (21, DataType::Uint4, &[], &[0x0b]),
            (23, DataType::Float4E2M1, &[1, 3], &[0x95, 0x02]),
            (25, DataType::Int2, &[5], &[0xe4, 0x03]),
            (25, DataType::Uint2, &[0, 3], &[]),
        ];
        for &(opset, dtype, shape, payload) in cases {
            let node = node_with(
                "value",
                Attribute::Tensor(TensorData::from_raw(
                    dtype,
                    shape.to_vec(),
                    payload.to_vec(),
                )),
            );
            // Direct provider dispatch: the optimizer would fold this node.
            let claim = ep.supports_op(&node, opset, &[], &[], &[]);
            assert!(claim.is_supported(), "{dtype:?}: {:?}", claim.reason());
            let kernel = ep.get_kernel(&node, &[], opset).unwrap();
            let strides = compute_contiguous_strides(shape);
            let mut storage = vec![0xA5u8; payload.len() + 2];
            let out = TensorMut::new(
                DevicePtrMut(storage.as_mut_ptr().cast()),
                dtype,
                shape,
                &strides,
                DeviceId::cpu(),
            )
            .with_byte_offset(1);
            kernel.execute(&[], &mut [out]).unwrap();
            assert_eq!(storage[0], 0xA5, "{dtype:?}: prefix overwritten");
            assert_eq!(&storage[1..storage.len() - 1], payload, "{dtype:?}");
            assert_eq!(
                storage[storage.len() - 1],
                0xA5,
                "{dtype:?}: suffix overwritten"
            );
        }
    }

    #[test]
    fn constant_packed_tensor_rejects_mismatched_output_or_storage() {
        let cases: &[(DataType, &[usize], &[i64], &[u8], &str)] = &[
            (
                DataType::Uint4,
                &[3],
                &[1],
                &[0x78, 0x0f],
                "output declares",
            ),
            (DataType::Int4, &[4], &[1], &[0x78, 0x0f], "output declares"),
            (
                DataType::Int4,
                &[3],
                &[2],
                &[0x78, 0x0f],
                "contiguous row-major",
            ),
            (DataType::Int4, &[3], &[1], &[0x78], "expected 2"),
        ];
        for &(dtype, shape, strides, payload, expected_error) in cases {
            let node = node_with(
                "value",
                Attribute::Tensor(TensorData::from_raw(
                    DataType::Int4,
                    vec![3],
                    payload.to_vec(),
                )),
            );
            let kernel = kernel_from(&node);
            let mut storage = [0xA5u8; 3];
            let out = TensorMut::new(
                DevicePtrMut(storage.as_mut_ptr().cast()),
                dtype,
                shape,
                strides,
                DeviceId::cpu(),
            );
            let err = kernel.execute(&[], &mut [out]).unwrap_err().to_string();
            assert!(err.contains(expected_error), "{err}");
            assert_eq!(storage, [0xA5; 3]);
        }
    }
}
