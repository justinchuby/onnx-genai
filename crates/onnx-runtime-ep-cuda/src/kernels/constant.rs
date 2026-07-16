//! `Constant`: decode an ONNX value attribute once, then upload it to the GPU.

use std::sync::Arc;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, Node};

use crate::runtime::{CudaRuntime, cuptr};

pub struct ConstantFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for ConstantFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(ConstantKernel {
            runtime: self.runtime.clone(),
            bytes: decode_value(node),
        }))
    }
}

#[derive(Debug)]
pub struct ConstantKernel {
    runtime: Arc<CudaRuntime>,
    bytes: Option<Vec<u8>>,
}

fn decode_value(node: &Node) -> Option<Vec<u8>> {
    if let Some(Attribute::Tensor(tensor)) = node.attr("value") {
        return Some(tensor.data.clone());
    }
    if let Some(Attribute::Float(value)) = node.attr("value_float") {
        return Some(value.to_le_bytes().to_vec());
    }
    if let Some(Attribute::Floats(values)) = node.attr("value_floats") {
        return Some(
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        );
    }
    if let Some(Attribute::Int(value)) = node.attr("value_int") {
        return Some(value.to_le_bytes().to_vec());
    }
    node.attr("value_ints")
        .and_then(|attribute| match attribute {
            Attribute::Ints(values) => Some(
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            ),
            _ => None,
        })
}

impl Kernel for ConstantKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if !inputs.is_empty() || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Constant: expected 0 inputs and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let output = &mut outputs[0];
        if !output.is_contiguous() {
            return Err(EpError::KernelFailed(
                "cuda_ep Constant: output must be contiguous".into(),
            ));
        }
        let bytes = self.bytes.as_ref().ok_or_else(|| {
            EpError::KernelFailed("cuda_ep Constant: no supported value attribute".into())
        })?;
        if bytes.len() != output.dtype.storage_bytes(output.numel()) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Constant: attribute has {} bytes but output requires {}",
                bytes.len(),
                output.dtype.storage_bytes(output.numel())
            )));
        }
        if !bytes.is_empty() {
            // SAFETY: the output allocation is live and exactly `bytes.len()` long.
            unsafe {
                self.runtime
                    .htod(bytes, cuptr(output.data_ptr_mut::<u8>() as *const u8 as _))?
            };
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::NodeId;

    #[test]
    fn decodes_numeric_value_attributes() {
        let mut node = Node::new(NodeId(0), "Constant", vec![], vec![]);
        node.attributes
            .insert("value_float".into(), Attribute::Float(1.5));
        assert_eq!(decode_value(&node), Some(1.5_f32.to_le_bytes().to_vec()));
        node.attributes.clear();
        node.attributes
            .insert("value_floats".into(), Attribute::Floats(vec![1.0, -2.0]));
        assert_eq!(
            decode_value(&node),
            Some([1.0_f32.to_le_bytes(), (-2.0_f32).to_le_bytes()].concat())
        );
        node.attributes.clear();
        node.attributes
            .insert("value_int".into(), Attribute::Int(-3));
        assert_eq!(decode_value(&node), Some((-3_i64).to_le_bytes().to_vec()));
        node.attributes.clear();
        node.attributes
            .insert("value_ints".into(), Attribute::Ints(vec![4, 5]));
        assert_eq!(
            decode_value(&node),
            Some([4_i64.to_le_bytes(), 5_i64.to_le_bytes()].concat())
        );
    }
}
