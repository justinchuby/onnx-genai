//! `Shape`: return the input tensor's dimensions as a 1-D `int64` tensor
//! (`docs/ORT2.md` §4.4).
//!
//! Opset-15 added the optional `start`/`end` attributes, which select a
//! (possibly negative-indexed) slice of the dim list; when omitted the full
//! shape vector is returned. It reads no element data — only the input view's
//! shape metadata — and is therefore dtype-agnostic. The start/end clamping
//! mirrors the shape-inference handler (`data_ops::shape`) so the kernel's
//! output length always matches the executor's pre-sized buffer.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::{check_arity, write_dense_bytes};

/// Stateless Shape kernel carrying the resolved `start`/`end` attributes.
pub struct ShapeKernel {
    /// Raw `start` attribute (default `0`); negative indexes from the end.
    start: i64,
    /// Raw `end` attribute (`None` → up to and including the last dim).
    end: Option<i64>,
}

/// Factory for [`ShapeKernel`], reading the opset-15 `start`/`end` attributes.
pub struct ShapeFactory;

impl KernelFactory for ShapeFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let start = node.attr("start").and_then(|a| a.as_int()).unwrap_or(0);
        let end = node.attr("end").and_then(|a| a.as_int());
        Ok(Box::new(ShapeKernel { start, end }))
    }
}

impl Kernel for ShapeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Shape", inputs, outputs, 1, 1, 1)?;
        if outputs[0].dtype != DataType::Int64 {
            return Err(EpError::KernelFailed(format!(
                "Shape: output must be Int64, got {:?}",
                outputs[0].dtype
            )));
        }
        let shape = inputs[0].shape;
        let rank = shape.len() as i64;
        // Clamp a (possibly negative) index into `[0, rank]`, matching the
        // shape-inference handler.
        let clamp = |v: i64| -> usize {
            let v = if v < 0 { v + rank } else { v };
            v.clamp(0, rank) as usize
        };
        let start = clamp(self.start);
        let end = clamp(self.end.unwrap_or(rank));
        let dims = shape.get(start..end.max(start)).unwrap_or(&[]);
        let mut bytes = Vec::with_capacity(dims.len() * 8);
        for &d in dims {
            bytes.extend_from_slice(&(d as i64).to_le_bytes());
        }
        write_dense_bytes(&mut outputs[0], &bytes)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    fn shape_kernel(start: i64, end: Option<i64>) -> ShapeKernel {
        ShapeKernel { start, end }
    }

    #[test]
    fn shape_of_3d_tensor() {
        let x = Owned::f32(&[2, 3, 4], &[0.0; 24]);
        let mut out = Owned::zeros(DataType::Int64, &[3]);
        shape_kernel(0, None)
            .execute(&[x.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i64(), vec![2, 3, 4]);
    }

    #[test]
    fn shape_of_scalar_is_empty() {
        let x = Owned::f32(&[], &[7.0]);
        let mut out = Owned::zeros(DataType::Int64, &[0]);
        shape_kernel(0, None)
            .execute(&[x.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i64(), Vec::<i64>::new());
    }

    #[test]
    fn shape_start_end_slices_dims() {
        // start=0,end=1 → [batch]; the expanded-Attention decomposition relies
        // on this to extract single dims (BatchSize, QSeqLen, …).
        let x = Owned::f32(&[2, 3, 4, 8], &[0.0; 192]);
        let mut out = Owned::zeros(DataType::Int64, &[1]);
        shape_kernel(0, Some(1))
            .execute(&[x.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i64(), vec![2]);
    }

    #[test]
    fn shape_negative_start_end() {
        // start=-2,end=-1 → the second-to-last dim only (QSeqLen/KVSeqLen).
        let x = Owned::f32(&[2, 3, 4, 8], &[0.0; 192]);
        let mut out = Owned::zeros(DataType::Int64, &[1]);
        shape_kernel(-2, Some(-1))
            .execute(&[x.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i64(), vec![4]);
    }

    #[test]
    fn shape_start_only_to_end() {
        // start=1 with no end → all dims from index 1 onward.
        let x = Owned::f32(&[2, 3, 4], &[0.0; 24]);
        let mut out = Owned::zeros(DataType::Int64, &[2]);
        shape_kernel(1, None)
            .execute(&[x.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i64(), vec![3, 4]);
    }
}
