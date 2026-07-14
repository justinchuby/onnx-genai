//! Sequence construction and scans: `Tile`, `Range`, and `CumSum`.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::{
    check_arity, elem_size, to_dense_bytes, to_dense_f32, to_dense_i64, write_dense_bytes,
    write_dense_f32,
};
use crate::strided::numel;

pub struct TileKernel;
pub struct TileFactory;
impl KernelFactory for TileFactory {
    fn create(&self, _: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(TileKernel))
    }
}
impl Kernel for TileKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Tile", inputs, outputs, 2, 2, 1)?;
        let x = &inputs[0];
        let repeats = to_dense_i64(&inputs[1])?;
        if repeats.len() != x.shape.len() || repeats.iter().any(|&r| r < 0) {
            return Err(EpError::KernelFailed(
                "Tile: repeats must be non-negative and match input rank".into(),
            ));
        }
        if outputs[0].dtype != x.dtype {
            return Err(EpError::KernelFailed(
                "Tile: output dtype must match input".into(),
            ));
        }
        let esize = elem_size(x.dtype)?;
        let src = to_dense_bytes(x)?;
        let out_shape = outputs[0].shape.to_vec();
        if out_shape.len() != x.shape.len()
            || out_shape
                .iter()
                .zip(x.shape)
                .zip(&repeats)
                .any(|((&o, &d), &r)| o != d * r as usize)
        {
            return Err(EpError::KernelFailed(
                "Tile: output shape does not match repeats".into(),
            ));
        }
        let in_strides = strides(x.shape);
        let out_strides = strides(&out_shape);
        let mut out = vec![0; numel(&out_shape) * esize];
        for linear in 0..numel(&out_shape) {
            let mut rem = linear;
            let mut source = 0;
            for d in 0..out_shape.len() {
                let coordinate = rem / out_strides[d];
                rem %= out_strides[d];
                source += (coordinate % x.shape[d]) * in_strides[d];
            }
            out[linear * esize..(linear + 1) * esize]
                .copy_from_slice(&src[source * esize..(source + 1) * esize]);
        }
        write_dense_bytes(&mut outputs[0], &out)
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

pub struct RangeKernel;
pub struct RangeFactory;
impl KernelFactory for RangeFactory {
    fn create(&self, _: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(RangeKernel))
    }
}
impl Kernel for RangeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Range", inputs, outputs, 3, 3, 1)?;
        if inputs.iter().any(|v| !v.shape.is_empty()) {
            return Err(EpError::KernelFailed(
                "Range: inputs must be scalars".into(),
            ));
        }
        if inputs.iter().any(|v| v.dtype != inputs[0].dtype) || outputs[0].dtype != inputs[0].dtype
        {
            return Err(EpError::KernelFailed(
                "Range: input and output dtypes must match".into(),
            ));
        }
        match inputs[0].dtype {
            DataType::Float32 => {
                let (start, limit, delta) = (
                    to_dense_f32(&inputs[0])?[0],
                    to_dense_f32(&inputs[1])?[0],
                    to_dense_f32(&inputs[2])?[0],
                );
                if delta == 0. {
                    return Err(EpError::KernelFailed(
                        "Range: delta must not be zero".into(),
                    ));
                }
                let mut out = Vec::new();
                let mut value = start;
                while (delta > 0. && value < limit) || (delta < 0. && value > limit) {
                    out.push(value);
                    value += delta;
                }
                write_dense_f32(&mut outputs[0], &out)
            }
            DataType::Int64 => {
                let (start, limit, delta) = (
                    to_dense_i64(&inputs[0])?[0],
                    to_dense_i64(&inputs[1])?[0],
                    to_dense_i64(&inputs[2])?[0],
                );
                if delta == 0 {
                    return Err(EpError::KernelFailed(
                        "Range: delta must not be zero".into(),
                    ));
                }
                let mut out = Vec::new();
                let mut value = start;
                while (delta > 0 && value < limit) || (delta < 0 && value > limit) {
                    out.push(value);
                    value = value
                        .checked_add(delta)
                        .ok_or_else(|| EpError::KernelFailed("Range: value overflow".into()))?;
                }
                let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
                write_dense_bytes(&mut outputs[0], &bytes)
            }
            dtype => Err(EpError::KernelFailed(format!(
                "Range: unsupported dtype {dtype:?}"
            ))),
        }
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

pub struct CumSumKernel {
    exclusive: bool,
    reverse: bool,
}
pub struct CumSumFactory;
impl KernelFactory for CumSumFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(CumSumKernel {
            exclusive: node.attr("exclusive").and_then(|a| a.as_int()).unwrap_or(0) != 0,
            reverse: node.attr("reverse").and_then(|a| a.as_int()).unwrap_or(0) != 0,
        }))
    }
}
impl Kernel for CumSumKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("CumSum", inputs, outputs, 2, 2, 1)?;
        let axis = to_dense_i64(&inputs[1])?;
        if axis.len() != 1 {
            return Err(EpError::KernelFailed(
                "CumSum: axis must be a scalar".into(),
            ));
        }
        let rank = inputs[0].shape.len();
        let raw = axis[0];
        let axis = if raw < 0 { raw + rank as i64 } else { raw };
        if axis < 0 || axis as usize >= rank {
            return Err(EpError::KernelFailed("CumSum: axis out of range".into()));
        }
        let axis = axis as usize;
        if outputs[0].dtype != inputs[0].dtype {
            return Err(EpError::KernelFailed(
                "CumSum: output dtype must match input".into(),
            ));
        }
        match inputs[0].dtype {
            DataType::Float32 => {
                let mut out = to_dense_f32(&inputs[0])?;
                scan(
                    &mut out,
                    inputs[0].shape,
                    axis,
                    self.exclusive,
                    self.reverse,
                    |a, b| a + b,
                );
                write_dense_f32(&mut outputs[0], &out)
            }
            DataType::Int64 => {
                let mut out = to_dense_i64(&inputs[0])?;
                scan(
                    &mut out,
                    inputs[0].shape,
                    axis,
                    self.exclusive,
                    self.reverse,
                    i64::wrapping_add,
                );
                let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
                write_dense_bytes(&mut outputs[0], &bytes)
            }
            dtype => Err(EpError::KernelFailed(format!(
                "CumSum: unsupported dtype {dtype:?}"
            ))),
        }
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}
fn scan<T: Copy + Default>(
    values: &mut [T],
    shape: &[usize],
    axis: usize,
    exclusive: bool,
    reverse: bool,
    add: impl Fn(T, T) -> T,
) {
    let inner = numel(&shape[axis + 1..]);
    let width = shape[axis];
    for outer in 0..numel(&shape[..axis]) {
        for i in 0..inner {
            let mut total = T::default();
            for n in 0..width {
                let d = if reverse { width - 1 - n } else { n };
                let offset = (outer * width + d) * inner + i;
                let v = values[offset];
                if exclusive {
                    values[offset] = total;
                    total = add(total, v);
                } else {
                    total = add(total, v);
                    values[offset] = total;
                }
            }
        }
    }
}
fn strides(shape: &[usize]) -> Vec<usize> {
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
    fn tile_repeats_each_axis() {
        let x = Owned::f32(&[1, 2], &[1., 2.]);
        let r = Owned::i64(&[2], &[2, 3]);
        let mut y = Owned::zeros_f32(&[2, 6]);
        TileKernel
            .execute(&[x.view(), r.view()], &mut [y.view_mut()])
            .unwrap();
        assert_eq!(
            y.to_f32(),
            vec![1., 2., 1., 2., 1., 2., 1., 2., 1., 2., 1., 2.]
        );
    }
    #[test]
    fn range_descending_int64() {
        let a = Owned::i64(&[], &[5]);
        let b = Owned::i64(&[], &[-1]);
        let d = Owned::i64(&[], &[-2]);
        let mut y = Owned::zeros(DataType::Int64, &[3]);
        RangeKernel
            .execute(&[a.view(), b.view(), d.view()], &mut [y.view_mut()])
            .unwrap();
        assert_eq!(y.to_i64(), vec![5, 3, 1]);
    }
    #[test]
    fn cumsum_reverse_exclusive() {
        let x = Owned::f32(&[3], &[1., 2., 3.]);
        let a = Owned::i64(&[], &[0]);
        let mut y = Owned::zeros_f32(&[3]);
        CumSumKernel {
            exclusive: true,
            reverse: true,
        }
        .execute(&[x.view(), a.view()], &mut [y.view_mut()])
        .unwrap();
        assert_eq!(y.to_f32(), vec![5., 3., 0.]);
    }
}
