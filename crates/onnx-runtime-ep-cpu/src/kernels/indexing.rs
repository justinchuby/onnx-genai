//! Indexed byte-copy kernels: `GatherElements` and `GatherND`.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::{check_arity, elem_size, to_dense_bytes, to_dense_i64, write_dense_bytes};
use crate::strided::numel;

pub struct GatherElementsKernel {
    axis: i64,
}
pub struct GatherElementsFactory;
impl KernelFactory for GatherElementsFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(GatherElementsKernel {
            axis: node.attr("axis").and_then(|a| a.as_int()).unwrap_or(0),
        }))
    }
}
impl Kernel for GatherElementsKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("GatherElements", inputs, outputs, 2, 2, 1)?;
        let data = &inputs[0];
        let indices = &inputs[1];
        if data.shape.len() != indices.shape.len() {
            return Err(EpError::KernelFailed(
                "GatherElements: data and indices must have equal rank".into(),
            ));
        }
        let rank = data.shape.len();
        let axis = normalize_axis("GatherElements", self.axis, rank)?;
        for d in 0..rank {
            if d != axis && indices.shape[d] > data.shape[d] {
                return Err(EpError::KernelFailed(format!(
                    "GatherElements: indices dimension {} exceeds data dimension {} at axis {d}",
                    indices.shape[d], data.shape[d]
                )));
            }
        }
        let esize = elem_size(data.dtype)?;
        if outputs[0].dtype != data.dtype {
            return Err(EpError::KernelFailed(
                "GatherElements: output dtype must match data".into(),
            ));
        }
        let src = to_dense_bytes(data)?;
        let idx = to_dense_i64(indices)?;
        let n = numel(indices.shape);
        let strides = contiguous(indices.shape);
        let data_strides = contiguous(data.shape);
        let mut out = vec![0; n * esize];
        for linear in 0..n {
            let mut rem = linear;
            let mut data_off = 0;
            for d in 0..rank {
                let coordinate = rem / strides[d];
                rem %= strides[d];
                let c = if d == axis {
                    let raw = idx[linear];
                    let adjusted = if raw < 0 {
                        raw + data.shape[d] as i64
                    } else {
                        raw
                    };
                    if adjusted < 0 || adjusted as usize >= data.shape[d] {
                        return Err(EpError::KernelFailed(format!(
                            "GatherElements: index {raw} out of range at axis {d}"
                        )));
                    }
                    adjusted as usize
                } else {
                    coordinate
                };
                data_off += c * data_strides[d];
            }
            out[linear * esize..(linear + 1) * esize]
                .copy_from_slice(&src[data_off * esize..(data_off + 1) * esize]);
        }
        write_dense_bytes(&mut outputs[0], &out)
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

pub struct GatherNDKernel {
    batch_dims: i64,
}
pub struct GatherNDFactory;
impl KernelFactory for GatherNDFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(GatherNDKernel {
            batch_dims: node
                .attr("batch_dims")
                .and_then(|a| a.as_int())
                .unwrap_or(0),
        }))
    }
}
impl Kernel for GatherNDKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("GatherND", inputs, outputs, 2, 2, 1)?;
        let data = &inputs[0];
        let indices = &inputs[1];
        if indices.shape.is_empty() {
            return Err(EpError::KernelFailed(
                "GatherND: indices must have rank >= 1".into(),
            ));
        }
        let batch = self.batch_dims;
        if batch < 0 || batch as usize > data.shape.len() || batch as usize >= indices.shape.len() {
            return Err(EpError::KernelFailed("GatherND: invalid batch_dims".into()));
        }
        let batch = batch as usize;
        if data.shape[..batch] != indices.shape[..batch] {
            return Err(EpError::KernelFailed(
                "GatherND: batch dimensions must match".into(),
            ));
        }
        let k = *indices.shape.last().unwrap();
        if k > data.shape.len() - batch {
            return Err(EpError::KernelFailed(
                "GatherND: index tuple is longer than data suffix rank".into(),
            ));
        }
        if outputs[0].dtype != data.dtype {
            return Err(EpError::KernelFailed(
                "GatherND: output dtype must match data".into(),
            ));
        }
        let src = to_dense_bytes(data)?;
        let idx = to_dense_i64(indices)?;
        let esize = elem_size(data.dtype)?;
        let batches = numel(&data.shape[..batch]);
        let tuples_per_batch = numel(&indices.shape[batch..indices.shape.len() - 1]);
        let data_batch_len = numel(&data.shape[batch..]);
        let tail_len = numel(&data.shape[batch + k..]);
        let mut out = Vec::with_capacity(batches * tuples_per_batch * tail_len * esize);
        for b in 0..batches {
            for t in 0..tuples_per_batch {
                let mut base = b * data_batch_len;
                for d in 0..k {
                    let raw = idx[(b * tuples_per_batch + t) * k + d];
                    let dim = data.shape[batch + d];
                    let v = if raw < 0 { raw + dim as i64 } else { raw };
                    if v < 0 || v as usize >= dim {
                        return Err(EpError::KernelFailed(format!(
                            "GatherND: index {raw} out of range at tuple dimension {d}"
                        )));
                    }
                    base += v as usize * numel(&data.shape[batch + d + 1..]);
                }
                out.extend_from_slice(&src[base * esize..(base + tail_len) * esize]);
            }
        }
        write_dense_bytes(&mut outputs[0], &out)
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

fn normalize_axis(op: &str, axis: i64, rank: usize) -> Result<usize> {
    let axis = if axis < 0 { axis + rank as i64 } else { axis };
    if axis < 0 || axis as usize >= rank {
        Err(EpError::KernelFailed(format!("{op}: axis out of range")))
    } else {
        Ok(axis as usize)
    }
}
fn contiguous(shape: &[usize]) -> Vec<usize> {
    let mut result = vec![1; shape.len()];
    for d in (0..shape.len()).rev().skip(1) {
        result[d] = result[d + 1] * shape[d + 1];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    #[test]
    fn gather_elements_negative_axis_and_index() {
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let i = Owned::i64(&[2, 2], &[2, 0, -1, 1]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        GatherElementsKernel { axis: -1 }
            .execute(&[x.view(), i.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![3., 1., 6., 5.]);
    }
    #[test]
    fn gather_nd_tuples_and_negative_indices() {
        let x = Owned::f32(&[2, 2, 2], &[0., 1., 2., 3., 4., 5., 6., 7.]);
        let i = Owned::i64(&[2, 2], &[0, 1, -1, 0]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        GatherNDKernel { batch_dims: 0 }
            .execute(&[x.view(), i.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![2., 3., 4., 5.]);
    }
    #[test]
    fn gather_nd_with_batch_dims() {
        let x = Owned::f32(&[2, 2, 2], &[0., 1., 2., 3., 4., 5., 6., 7.]);
        let i = Owned::i64(&[2, 2, 1], &[1, 0, 0, 1]);
        let mut out = Owned::zeros_f32(&[2, 2, 2]);
        GatherNDKernel { batch_dims: 1 }
            .execute(&[x.view(), i.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![2., 3., 0., 1., 4., 5., 6., 7.]);
    }
}
