//! Sequence construction and scans: `Tile`, `Range`, `CumSum`, and `CumProd`.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::{
    check_arity, elem_size, to_dense_bytes, to_dense_f32, to_dense_i64, write_dense_bytes,
    write_dense_f32,
};
use crate::dtype::{ComputeDomain, NumericElem, to_dense, write_dense};
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
                let count = float_range_count(start, limit, delta)?;
                let mut out = alloc_range_output::<f32>(count)?;
                out.extend((0..count).map(|i| start + i as f32 * delta));
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
                let count = int_range_count(start, limit, delta)?;
                let mut out = alloc_range_output::<i64>(count)?;
                for i in 0..count {
                    let v = i64::try_from(start as i128 + i as i128 * delta as i128)
                        .map_err(|_| EpError::KernelFailed("Range: value overflow".into()))?;
                    out.push(v);
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
        execute_cumulative(
            "CumSum",
            inputs,
            outputs,
            self.exclusive,
            self.reverse,
            CumulativeOp::Sum,
        )
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

pub struct CumProdKernel {
    exclusive: bool,
    reverse: bool,
}
pub struct CumProdFactory;
impl KernelFactory for CumProdFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(CumProdKernel {
            exclusive: node.attr("exclusive").and_then(|a| a.as_int()).unwrap_or(0) != 0,
            reverse: node.attr("reverse").and_then(|a| a.as_int()).unwrap_or(0) != 0,
        }))
    }
}
impl Kernel for CumProdKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        execute_cumulative(
            "CumProd",
            inputs,
            outputs,
            self.exclusive,
            self.reverse,
            CumulativeOp::Product,
        )
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

#[derive(Clone, Copy)]
enum CumulativeOp {
    Sum,
    Product,
}

fn execute_cumulative(
    op: &str,
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
    exclusive: bool,
    reverse: bool,
    cumulative_op: CumulativeOp,
) -> Result<()> {
    check_arity(op, inputs, outputs, 2, 2, 1)?;
    let axis_values = to_dense_i64(&inputs[1])?;
    if axis_values.len() != 1 {
        return Err(EpError::KernelFailed(format!(
            "{op}: axis must contain exactly one element"
        )));
    }
    let rank = inputs[0].shape.len();
    let raw = axis_values[0];
    let axis = if raw < 0 { raw + rank as i64 } else { raw };
    if axis < 0 || axis as usize >= rank {
        return Err(EpError::KernelFailed(format!("{op}: axis out of range")));
    }
    if outputs[0].dtype != inputs[0].dtype {
        return Err(EpError::KernelFailed(format!(
            "{op}: output dtype must match input"
        )));
    }

    macro_rules! run {
        ($ty:ty) => {
            execute_cumulative_typed::<$ty>(
                &inputs[0],
                &mut outputs[0],
                axis as usize,
                exclusive,
                reverse,
                cumulative_op,
            )
        };
    }
    match inputs[0].dtype {
        DataType::Uint32 => run!(u32),
        DataType::Uint64 => run!(u64),
        DataType::Int32 => run!(i32),
        DataType::Int64 => run!(i64),
        DataType::Float16 => run!(half::f16),
        DataType::Float32 => run!(f32),
        DataType::Float64 => run!(f64),
        DataType::BFloat16 => run!(half::bf16),
        dtype => Err(EpError::KernelFailed(format!(
            "{op}: unsupported dtype {dtype:?}"
        ))),
    }
}

fn execute_cumulative_typed<T>(
    input: &TensorView,
    output: &mut TensorMut,
    axis: usize,
    exclusive: bool,
    reverse: bool,
    cumulative_op: CumulativeOp,
) -> Result<()>
where
    T: NumericElem,
{
    let mut values = to_dense::<T>(input)?;
    let identity = match cumulative_op {
        CumulativeOp::Sum => 0.0,
        CumulativeOp::Product => 1.0,
    };
    scan(
        &mut values,
        input.shape,
        axis,
        exclusive,
        reverse,
        T::from_f32_scalar(identity).to_acc(),
        |left, right| match cumulative_op {
            CumulativeOp::Sum => left.c_add(right),
            CumulativeOp::Product => left.c_mul(right),
        },
    );
    write_dense::<T>(output, &values)
}

fn float_range_count(start: f32, limit: f32, delta: f32) -> Result<usize> {
    let count = ((limit - start) / delta).ceil().max(0.0);
    if !count.is_finite() || count >= usize::MAX as f32 {
        return Err(EpError::KernelFailed(
            "Range: element count exceeds addressable memory".into(),
        ));
    }
    Ok(count as usize)
}

fn int_range_count(start: i64, limit: i64, delta: i64) -> Result<usize> {
    let count = if delta > 0 && start < limit {
        let distance = limit as i128 - start as i128;
        (distance + delta as i128 - 1) / delta as i128
    } else if delta < 0 && start > limit {
        let distance = start as i128 - limit as i128;
        let step = -(delta as i128);
        (distance + step - 1) / step
    } else {
        0
    };
    usize::try_from(count).map_err(|_| {
        EpError::KernelFailed("Range: element count exceeds addressable memory".into())
    })
}

/// Validate a Range output's element count against byte addressability and
/// pre-allocate its backing buffer without ever panicking.
///
/// `count * size_of::<T>()` must not exceed `isize::MAX` (the largest size any
/// Rust allocation can address); otherwise `Vec` allocation/index math would
/// panic on user-controlled inputs. We check the bound up front and fall back
/// to `try_reserve` so an over-large request fails as a clean kernel error.
fn alloc_range_output<T>(count: usize) -> Result<Vec<T>> {
    let elem = std::mem::size_of::<T>();
    let bytes = count.checked_mul(elem);
    if bytes.is_none_or(|b| b > isize::MAX as usize) {
        return Err(EpError::KernelFailed(format!(
            "Range output too large: {count} elements ({} bytes) exceeds addressable limit",
            bytes.map_or_else(|| "overflow".to_string(), |b| b.to_string()),
        )));
    }
    let mut out = Vec::new();
    out.try_reserve(count).map_err(|_| {
        EpError::KernelFailed(format!(
            "Range output too large: failed to allocate {count} elements ({} bytes)",
            count * elem,
        ))
    })?;
    Ok(out)
}

fn scan<T, F>(
    values: &mut [T],
    shape: &[usize],
    axis: usize,
    exclusive: bool,
    reverse: bool,
    identity: T::Acc,
    combine: F,
) where
    T: NumericElem,
    F: Fn(T::Acc, T::Acc) -> T::Acc,
{
    let inner = numel(&shape[axis + 1..]);
    let width = shape[axis];
    for outer in 0..numel(&shape[..axis]) {
        for i in 0..inner {
            let mut total = identity;
            for n in 0..width {
                let d = if reverse { width - 1 - n } else { n };
                let offset = (outer * width + d) * inner + i;
                let value = values[offset].to_acc();
                if exclusive {
                    values[offset] = T::from_acc(total);
                    total = combine(total, value);
                } else {
                    total = combine(total, value);
                    values[offset] = T::from_acc(total);
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
    fn range_float32_generates_by_index() {
        let a = Owned::f32(&[], &[0.]);
        let b = Owned::f32(&[], &[1.]);
        let d = Owned::f32(&[], &[0.25]);
        let mut y = Owned::zeros_f32(&[4]);
        RangeKernel
            .execute(&[a.view(), b.view(), d.view()], &mut [y.view_mut()])
            .unwrap();
        assert_eq!(y.to_f32(), vec![0., 0.25, 0.5, 0.75]);
    }
    #[test]
    fn range_float32_descending() {
        let a = Owned::f32(&[], &[1.]);
        let b = Owned::f32(&[], &[-0.1]);
        let d = Owned::f32(&[], &[-0.25]);
        let mut y = Owned::zeros_f32(&[5]);
        RangeKernel
            .execute(&[a.view(), b.view(), d.view()], &mut [y.view_mut()])
            .unwrap();
        assert_eq!(y.to_f32(), vec![1., 0.75, 0.5, 0.25, 0.]);
    }
    #[test]
    fn range_float32_no_progress_terminates() {
        let a = Owned::f32(&[], &[16_777_216.]);
        let b = Owned::f32(&[], &[16_777_218.]);
        let d = Owned::f32(&[], &[1.]);
        let mut y = Owned::zeros_f32(&[2]);
        RangeKernel
            .execute(&[a.view(), b.view(), d.view()], &mut [y.view_mut()])
            .unwrap();
        assert_eq!(y.to_f32().len(), 2);
        assert_eq!(y.to_f32().first(), Some(&16_777_216.));
        assert_eq!(y.to_f32().last(), Some(&16_777_216.));
    }
    #[test]
    fn range_int64_overflow_returns_error() {
        // count ~= i64::MAX elements; count * size_of::<i64>() overflows the
        // addressable-byte limit, so the kernel must return an Err before any
        // allocation is attempted instead of panicking.
        let a = Owned::i64(&[], &[0]);
        let b = Owned::i64(&[], &[i64::MAX]);
        let d = Owned::i64(&[], &[1]);
        let mut y = Owned::zeros(DataType::Int64, &[1]);
        let result = RangeKernel.execute(&[a.view(), b.view(), d.view()], &mut [y.view_mut()]);
        assert!(
            matches!(result, Err(EpError::KernelFailed(_))),
            "expected KernelFailed, got {result:?}"
        );
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

    #[test]
    fn cumsum_negative_axis_int32() {
        let x = Owned::i32(&[2, 3], &[1, 2, 3, 4, 5, 6]);
        let axis = Owned::i64(&[1], &[-1]);
        let mut y = Owned::zeros(DataType::Int32, &[2, 3]);
        CumSumKernel {
            exclusive: false,
            reverse: false,
        }
        .execute(&[x.view(), axis.view()], &mut [y.view_mut()])
        .unwrap();
        assert_eq!(y.to_i32(), vec![1, 3, 6, 4, 9, 15]);
    }

    #[test]
    fn cumsum_exclusive_axis_zero() {
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let axis = Owned::i64(&[], &[0]);
        let mut y = Owned::zeros_f32(&[2, 3]);
        CumSumKernel {
            exclusive: true,
            reverse: false,
        }
        .execute(&[x.view(), axis.view()], &mut [y.view_mut()])
        .unwrap();
        assert_eq!(y.to_f32(), vec![0., 0., 0., 1., 2., 3.]);
    }

    #[test]
    fn cumprod_negative_axis_int32_exclusive() {
        let x = Owned::i32(&[2, 3], &[1, 2, 3, 4, 5, 6]);
        let axis = Owned::i64(&[], &[-1]);
        let mut y = Owned::zeros(DataType::Int32, &[2, 3]);
        CumProdKernel {
            exclusive: true,
            reverse: false,
        }
        .execute(&[x.view(), axis.view()], &mut [y.view_mut()])
        .unwrap();
        assert_eq!(y.to_i32(), vec![1, 1, 2, 1, 4, 20]);
    }

    #[test]
    fn cumprod_reverse_exclusive() {
        let x = Owned::f32(&[3], &[2., 3., 4.]);
        let axis = Owned::i64(&[], &[0]);
        let mut y = Owned::zeros_f32(&[3]);
        CumProdKernel {
            exclusive: true,
            reverse: true,
        }
        .execute(&[x.view(), axis.view()], &mut [y.view_mut()])
        .unwrap();
        assert_eq!(y.to_f32(), vec![12., 4., 1.]);
    }

    #[test]
    fn cumprod_2d_axis_zero() {
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let axis = Owned::i64(&[1], &[0]);
        let mut y = Owned::zeros_f32(&[2, 3]);
        CumProdKernel {
            exclusive: false,
            reverse: false,
        }
        .execute(&[x.view(), axis.view()], &mut [y.view_mut()])
        .unwrap();
        assert_eq!(y.to_f32(), vec![1., 2., 3., 4., 10., 18.]);
    }
}
