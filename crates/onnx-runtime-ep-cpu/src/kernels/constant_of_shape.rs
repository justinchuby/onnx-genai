//! `ConstantOfShape`: build a tensor of a runtime-supplied shape, every element
//! set to the scalar `value` attribute (`docs/ORT2.md` §4.4).
//!
//! Input 0 is a 1-D `int64` tensor giving the output dimensions; the output
//! rank equals that tensor's length (an empty length yields a scalar). The
//! `value` attribute is a one-element tensor carrying both the fill value and
//! the output dtype (default: a single `float32` `0`). The op is dtype-generic:
//! the decoded fill bytes are simply tiled across the output, so every
//! fixed-width element type is supported without special-casing.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use super::{check_arity, elem_size, to_dense_i64, write_dense_bytes};
use crate::strided::numel;

/// `ConstantOfShape` kernel carrying the decoded fill element (`dtype` plus its
/// little-endian bytes). Output shape is read from input 0 at execution time.
pub struct ConstantOfShapeKernel {
    dtype: DataType,
    fill: Vec<u8>,
}

/// Factory decoding the `value` attribute into `(dtype, fill bytes)`.
pub struct ConstantOfShapeFactory;

impl KernelFactory for ConstantOfShapeFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let (dtype, fill) = decode_value(node)?;
        Ok(Box::new(ConstantOfShapeKernel { dtype, fill }))
    }
}

/// Decode the `value` attribute (a one-element tensor). ONNX defaults to a
/// single `float32` `0` when the attribute is absent.
fn decode_value(node: &Node) -> Result<(DataType, Vec<u8>)> {
    match node.attr("value") {
        Some(Attribute::Tensor(t)) => {
            let esize = elem_size(t.dtype)?;
            if t.data.len() < esize {
                return Err(EpError::KernelFailed(format!(
                    "ConstantOfShape: WHAT: `value` tensor holds {} bytes but dtype {:?} needs \
                     {esize}. WHY: `value` must carry exactly one element. \
                     HOW: provide a one-element `value` tensor.",
                    t.data.len(),
                    t.dtype
                )));
            }
            // Exactly the first element's bytes are the fill; any extra are ignored.
            Ok((t.dtype, t.data[..esize].to_vec()))
        }
        _ => Ok((DataType::Float32, 0.0f32.to_le_bytes().to_vec())),
    }
}

impl Kernel for ConstantOfShapeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("ConstantOfShape", inputs, outputs, 1, 1, 1)?;
        if inputs[0].shape.len() > 1 {
            return Err(EpError::KernelFailed(format!(
                "ConstantOfShape: WHAT: shape input has rank {} (shape {:?}). \
                 WHY: the shape input must be a 1-D int64 tensor. \
                 HOW: provide a rank-1 int64 shape tensor.",
                inputs[0].shape.len(),
                inputs[0].shape
            )));
        }
        let dims = to_dense_i64(&inputs[0])?;
        let mut out_shape = Vec::with_capacity(dims.len());
        for &d in &dims {
            if d < 0 {
                return Err(EpError::KernelFailed(format!(
                    "ConstantOfShape: WHAT: shape entry {d} is negative. \
                     WHY: output dimensions must be non-negative. \
                     HOW: supply a shape tensor with non-negative dimensions."
                )));
            }
            out_shape.push(d as usize);
        }

        let out = &mut outputs[0];
        if out.dtype != self.dtype {
            return Err(EpError::KernelFailed(format!(
                "ConstantOfShape: WHAT: output dtype {:?} differs from `value` dtype {:?}. \
                 WHY: the output element type is dictated by the `value` attribute. \
                 HOW: allocate the output with dtype {:?}.",
                out.dtype, self.dtype, self.dtype
            )));
        }
        if out.shape != out_shape.as_slice() {
            return Err(EpError::KernelFailed(format!(
                "ConstantOfShape: WHAT: executor supplied output shape {:?}, but the shape input \
                 requires {out_shape:?}. WHY: the output must match the runtime shape tensor. \
                 HOW: allocate the output with shape {out_shape:?}.",
                out.shape
            )));
        }

        let esize = elem_size(self.dtype)?;
        let n = numel(&out_shape);
        let mut buf = vec![0u8; n * esize];
        for chunk in buf.chunks_exact_mut(esize) {
            chunk.copy_from_slice(&self.fill);
        }
        write_dense_bytes(out, &buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::{NodeId, TensorData};

    fn kernel_with(value: Option<Attribute>) -> Box<dyn Kernel> {
        let mut n = Node::new(NodeId(0), "ConstantOfShape", vec![], vec![]);
        if let Some(v) = value {
            n.attributes.insert("value".to_string(), v);
        }
        ConstantOfShapeFactory.create(&n, &[]).unwrap()
    }

    fn tensor_value(dtype: DataType, bytes: Vec<u8>) -> Attribute {
        Attribute::Tensor(TensorData::from_raw(dtype, vec![1], bytes))
    }

    #[test]
    fn constant_of_shape_default_float_zero() {
        // No `value`: default float32 0 fill. Shape [2,3] → six zeros.
        let k = kernel_with(None);
        let shape = Owned::i64(&[2], &[2, 3]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        k.execute(&[shape.view()], &mut [out.view_mut()]).unwrap();
        assert_eq!(out.to_f32(), vec![0.0; 6]);
        assert_eq!(out.shape, vec![2, 3]);
    }

    #[test]
    fn constant_of_shape_float_fill() {
        let k = kernel_with(Some(tensor_value(
            DataType::Float32,
            2.5f32.to_le_bytes().to_vec(),
        )));
        let shape = Owned::i64(&[2], &[2, 2]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        k.execute(&[shape.view()], &mut [out.view_mut()]).unwrap();
        assert_eq!(out.to_f32(), vec![2.5, 2.5, 2.5, 2.5]);
    }

    #[test]
    fn constant_of_shape_int64_fill() {
        let k = kernel_with(Some(tensor_value(
            DataType::Int64,
            7i64.to_le_bytes().to_vec(),
        )));
        let shape = Owned::i64(&[1], &[4]);
        let mut out = Owned::zeros(DataType::Int64, &[4]);
        k.execute(&[shape.view()], &mut [out.view_mut()]).unwrap();
        assert_eq!(out.to_i64(), vec![7, 7, 7, 7]);
    }

    #[test]
    fn constant_of_shape_zero_dim_yields_empty_tensor() {
        // A zero dimension yields an empty (0-element) tensor of the right shape.
        let k = kernel_with(Some(tensor_value(
            DataType::Float32,
            1.0f32.to_le_bytes().to_vec(),
        )));
        let shape = Owned::i64(&[2], &[0, 3]);
        let mut out = Owned::zeros_f32(&[0, 3]);
        k.execute(&[shape.view()], &mut [out.view_mut()]).unwrap();
        assert_eq!(out.to_f32(), Vec::<f32>::new());
        assert_eq!(out.shape, vec![0, 3]);
    }

    #[test]
    fn constant_of_shape_empty_shape_input_yields_scalar() {
        // A length-0 shape tensor → rank-0 (scalar) output with one filled element.
        let k = kernel_with(Some(tensor_value(
            DataType::Float32,
            9.0f32.to_le_bytes().to_vec(),
        )));
        let shape = Owned::i64(&[0], &[]);
        let mut out = Owned::zeros_f32(&[]);
        k.execute(&[shape.view()], &mut [out.view_mut()]).unwrap();
        assert_eq!(out.to_f32(), vec![9.0]);
        assert_eq!(out.shape, Vec::<usize>::new());
    }

    #[test]
    fn constant_of_shape_rejects_dtype_mismatch() {
        let k = kernel_with(Some(tensor_value(
            DataType::Int64,
            1i64.to_le_bytes().to_vec(),
        )));
        let shape = Owned::i64(&[1], &[2]);
        let mut out = Owned::zeros_f32(&[2]);
        let err = k.execute(&[shape.view()], &mut [out.view_mut()]).unwrap_err();
        assert!(err.to_string().contains("WHY:"));
    }
}
