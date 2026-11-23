use onnx_runtime_quantization::dequantize_packed_conformance as oracle;

use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{
    DataType, DeviceId, Node, NodeId, ValueId, compute_contiguous_strides, static_shape,
};

#[test]
fn cpu_dequantize_packed_conformance() {
    let ep = CpuExecutionProvider::new();
    oracle::check_conformance(|name, packed, shape, dtype| {
        let mut node = Node::new(
            NodeId(0),
            "DequantizePacked",
            vec![Some(ValueId(0))],
            vec![ValueId(1)],
        );
        node.version = Some(29);
        node.attributes.extend(
            oracle::attributes(name, shape, dtype)
                .into_iter()
                .map(|(key, attr)| (key.to_owned(), attr)),
        );
        let input_shape = vec![packed.len()];
        let claim = ep.supports_op(
            &node,
            29,
            &[static_shape([packed.len()])],
            &[DataType::Uint8],
            &[],
        );
        assert!(claim.is_supported(), "{:?}", claim.reason());
        let kernel = ep
            .get_kernel(&node, std::slice::from_ref(&input_shape), 29)
            .unwrap();
        let dtype = dtype.unwrap_or(DataType::Float32);
        let mut bytes = vec![0; dtype.storage_bytes(shape.iter().product())];
        kernel
            .execute(
                &[TensorView::new(
                    DevicePtr(packed.as_ptr().cast()),
                    DataType::Uint8,
                    &input_shape,
                    &[1],
                    DeviceId::cpu(),
                )],
                &mut [TensorMut::new(
                    DevicePtrMut(bytes.as_mut_ptr().cast()),
                    dtype,
                    shape,
                    &compute_contiguous_strides(shape),
                    DeviceId::cpu(),
                )],
            )
            .unwrap();
        if cfg!(target_endian = "big") {
            for lane in bytes.chunks_exact_mut(dtype.byte_size()) {
                lane.reverse();
            }
        }
        bytes
    });
}
