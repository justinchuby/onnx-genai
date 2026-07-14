//! `Constant`: materialize a compile-time constant tensor from an attribute
//! (`docs/ORT2.md` §4.4).
//!
//! Opset-12 supports several mutually-exclusive value attributes; this kernel
//! handles the numeric ones the BERT target uses: `value` (a full tensor),
//! `value_float`, `value_floats`, `value_int`, and `value_ints`. The attribute
//! is decoded once at factory time into the dense little-endian element bytes,
//! which are then copied into the pre-allocated output view (whose shape/dtype
//! the loader derives from the same attribute).

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, Node};

use super::{check_arity, write_dense_bytes};

/// Constant kernel carrying the decoded element bytes (`None` if no supported
/// value attribute was present; execution then errors).
pub struct ConstantKernel {
    bytes: Option<Vec<u8>>,
}

/// Factory decoding the constant's value attribute.
pub struct ConstantFactory;

impl KernelFactory for ConstantFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let bytes = decode_value(node);
        Ok(Box::new(ConstantKernel { bytes }))
    }
}

/// Decode the first recognized `value*` attribute into dense LE element bytes.
fn decode_value(node: &Node) -> Option<Vec<u8>> {
    if let Some(Attribute::Tensor(t)) = node.attr("value") {
        return Some(t.data.clone());
    }
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
        let bytes = self.bytes.as_ref().ok_or_else(|| {
            EpError::KernelFailed("Constant: no supported value attribute".into())
        })?;
        write_dense_bytes(&mut outputs[0], bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::{DataType, NodeId, TensorData};

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
}
