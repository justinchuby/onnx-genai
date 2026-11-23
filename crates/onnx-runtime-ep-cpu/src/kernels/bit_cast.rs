use onnx_runtime_ep_api::{
    Kernel, KernelFactory, Result, TensorMut, TensorView, bit_cast::copy_size,
};
use onnx_runtime_ir::{DataType, Node, opset26::bit_cast_target};

use super::check_arity;

pub struct BitCastFactory;

impl KernelFactory for BitCastFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(BitCastKernel(
            bit_cast_target(node).map_err(onnx_runtime_ep_api::EpError::KernelFailed)?,
        )))
    }
}

struct BitCastKernel(DataType);

impl Kernel for BitCastKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("BitCast", inputs, outputs, 1, 1, 1)?;
        let input = &inputs[0];
        let output = &mut outputs[0];
        let bytes = copy_size(input, output, self.0)?;
        if bytes != 0 && !std::ptr::eq(input.data_ptr::<u8>(), output.data_ptr_mut::<u8>()) {
            // Both views cover `bytes` by the EP backing-storage contract.
            unsafe { std::ptr::copy(input.data_ptr::<u8>(), output.data_ptr_mut::<u8>(), bytes) };
        }

        Ok(())
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }

    fn can_run_in_place(&self, input_index: usize) -> bool {
        input_index == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut};
    use onnx_runtime_ir::{Attribute, DeviceId, NodeId};

    #[test]
    fn bitcast_copies_raw_payload_at_byte_offset() {
        let bytes = [0xaa, 0x21, 0x03, 0xbb];
        let mut dest = [0xcc; 4];
        let input = TensorView::new(
            DevicePtr(bytes.as_ptr().cast()),
            DataType::Uint4,
            &[3],
            &[1],
            DeviceId::cpu(),
        )
        .with_byte_offset(1);
        let output = TensorMut::new(
            DevicePtrMut(dest.as_mut_ptr().cast()),
            DataType::Float4E2M1,
            &[3],
            &[1],
            DeviceId::cpu(),
        )
        .with_byte_offset(1);
        BitCastKernel(DataType::Float4E2M1)
            .execute(&[input], &mut [output])
            .unwrap();
        assert_eq!(dest, [0xcc, 0x21, 0x03, 0xcc]);
    }

    #[test]
    fn bitcast_rejects_bad_target_even_if_i64_truncation_would_be_valid() {
        for attr in [
            None,
            Some(Attribute::Int(1_i64 << 32 | 1)),
            Some(Attribute::Int(8)),
            Some(Attribute::Float(1.0)),
        ] {
            let mut node = Node::new(NodeId(0), "BitCast", vec![], vec![]);
            if let Some(attr) = attr {
                node.attributes.insert("to".into(), attr);
            }
            assert!(BitCastFactory.create(&node, &[]).is_err());
        }
    }
}
