//! Sequence construction and scans: `Tile`, `Range`, `CumSum`, and `CumProd`.

use onnx_runtime_ep_api::standard_math::{Scan, validate};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::standard_math::Math;
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
pub struct Range27Factory;

struct Range27Kernel {
    stash_type: i64,
}

impl KernelFactory for Range27Factory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(Range27Kernel {
            stash_type: node
                .attr("stash_type")
                .and_then(|a| a.as_int())
                .unwrap_or(1),
        }))
    }
}

impl Kernel for Range27Kernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        execute_range(inputs, outputs, Some(self.stash_type))
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

impl KernelFactory for RangeFactory {
    fn create(&self, _: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(RangeKernel))
    }
}
impl Kernel for RangeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        execute_range(inputs, outputs, None)
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

fn execute_range(
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
    stash_type: Option<i64>,
) -> Result<()> {
    check_arity("Range", inputs, outputs, 3, 3, 1)?;
    if inputs.iter().any(|v| !v.shape.is_empty()) {
        return Err(EpError::KernelFailed(
            "Range: inputs must be scalars".into(),
        ));
    }
    if inputs.iter().any(|v| v.dtype != inputs[0].dtype) || outputs[0].dtype != inputs[0].dtype {
        return Err(EpError::KernelFailed(
            "Range: input and output dtypes must match".into(),
        ));
    }
    if stash_type.is_some() && outputs[0].shape.len() != 1 {
        return Err(EpError::KernelFailed(
            "Range-27: output must be a rank-1 tensor".into(),
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
        DataType::Float16 | DataType::BFloat16 => {
            let start = crate::dtype::to_dense_f32_widen("Range", &inputs[0])?[0];
            let limit = crate::dtype::to_dense_f32_widen("Range", &inputs[1])?[0];
            let delta = crate::dtype::to_dense_f32_widen("Range", &inputs[2])?[0];
            if delta == 0. {
                return Err(EpError::KernelFailed(
                    "Range: delta must not be zero".into(),
                ));
            }
            match stash_type.unwrap_or(1) {
                1 => {
                    let count = float_range_count(start, limit, delta)?;
                    let mut out = alloc_range_output::<f32>(count)?;
                    out.extend((0..count).map(|i| start + i as f32 * delta));
                    crate::dtype::write_dense_f32_narrow("Range", &mut outputs[0], &out)
                }
                11 => {
                    let (start, limit, delta) =
                        (f64::from(start), f64::from(limit), f64::from(delta));
                    let count = double_range_count(start, limit, delta)?;
                    if inputs[0].dtype == DataType::Float16 {
                        let mut out = alloc_range_output::<half::f16>(count)?;
                        out.extend(
                            (0..count).map(|i| half::f16::from_f64(start + i as f64 * delta)),
                        );
                        write_dense(&mut outputs[0], &out)
                    } else {
                        let mut out = alloc_range_output::<half::bf16>(count)?;
                        out.extend(
                            (0..count).map(|i| half::bf16::from_f64(start + i as f64 * delta)),
                        );
                        write_dense(&mut outputs[0], &out)
                    }
                }
                invalid => Err(EpError::KernelFailed(format!(
                    "Range-27: stash_type={invalid} is invalid for FP16/BF16; use FLOAT (1) or DOUBLE (11)"
                ))),
            }
        }
        DataType::Float64 => {
            let start = to_dense::<f64>(&inputs[0])?[0];
            let limit = to_dense::<f64>(&inputs[1])?[0];
            let delta = to_dense::<f64>(&inputs[2])?[0];
            let count = double_range_count(start, limit, delta)?;
            let mut out = alloc_range_output::<f64>(count)?;
            out.extend((0..count).map(|i| start + i as f64 * delta));
            write_dense(&mut outputs[0], &out)
        }
        DataType::Int16 => {
            let start = i64::from(to_dense::<i16>(&inputs[0])?[0]);
            let limit = i64::from(to_dense::<i16>(&inputs[1])?[0]);
            let delta = i64::from(to_dense::<i16>(&inputs[2])?[0]);
            if delta == 0 {
                return Err(EpError::KernelFailed(
                    "Range: delta must not be zero".into(),
                ));
            }
            let count = int_range_count(start, limit, delta)?;
            let mut out = alloc_range_output::<i16>(count)?;
            out.extend((0..count).map(|i| (start + i as i64 * delta) as i16));
            write_dense(&mut outputs[0], &out)
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
        DataType::Int32 => {
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
            let mut out = alloc_range_output::<i32>(count)?;
            for i in 0..count {
                let value = i32::try_from(start as i128 + i as i128 * delta as i128)
                    .map_err(|_| EpError::KernelFailed("Range: value overflow".into()))?;
                out.push(value);
            }
            let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
            write_dense_bytes(&mut outputs[0], &bytes)
        }
        dtype => Err(EpError::KernelFailed(format!(
            "Range: unsupported dtype {dtype:?}"
        ))),
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
        let (exclusive, reverse) =
            onnx_runtime_ir::opset26::cum_prod_flags(node).map_err(EpError::KernelFailed)?;
        Ok(Box::new(CumProdKernel { exclusive, reverse }))
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

    fn can_run_in_place(&self, input_index: usize) -> bool {
        input_index == 0
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
    if matches!(cumulative_op, CumulativeOp::Product)
        && (!inputs[1].shape.is_empty()
            || !matches!(inputs[1].dtype, DataType::Int32 | DataType::Int64))
    {
        return Err(EpError::KernelFailed(format!(
            "CumProd: axis must be an Int32 or Int64 rank-0 tensor, got {:?} {:?}",
            inputs[1].dtype, inputs[1].shape
        )));
    }
    let rank = inputs[0].shape.len();
    let axis = if matches!(cumulative_op, CumulativeOp::Product) {
        let axis = &inputs[1];
        axis.validate()?;
        // The validated scalar is host-resident; alignment is not guaranteed.
        let raw = unsafe {
            if axis.dtype == DataType::Int32 {
                i64::from(axis.data_ptr::<i32>().read_unaligned())
            } else {
                axis.data_ptr::<i64>().read_unaligned()
            }
        };
        onnx_runtime_ir::opset26::cum_prod_axis(raw, rank).map_err(EpError::KernelFailed)?
    } else {
        let axis_values = to_dense_i64(&inputs[1])?;
        let [raw] = axis_values[..] else {
            return Err(EpError::KernelFailed(format!(
                "{op}: axis must contain exactly one element"
            )));
        };
        let axis = if raw < 0 { raw + rank as i64 } else { raw };
        if axis < 0 || axis as usize >= rank {
            return Err(EpError::KernelFailed(format!("{op}: axis out of range")));
        }
        axis as usize
    };
    if outputs[0].dtype != inputs[0].dtype
        || (matches!(cumulative_op, CumulativeOp::Product) && outputs[0].shape != inputs[0].shape)
    {
        return Err(EpError::KernelFailed(format!(
            "{op}: expected output {:?} {:?} matching the input, got {:?} {:?}",
            inputs[0].dtype, inputs[0].shape, outputs[0].dtype, outputs[0].shape
        )));
    }

    macro_rules! run {
        ($ty:ty) => {
            execute_cumulative_typed::<$ty>(
                &inputs[0],
                &mut outputs[0],
                axis,
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
    if matches!(cumulative_op, CumulativeOp::Product) {
        input.validate()?;
        output.validate()?;
        if input.shape.contains(&0) {
            return Ok(());
        }
        if input.is_contiguous() && output.is_contiguous() {
            let src = input.data_ptr::<T>();
            let dst = output.data_ptr_mut::<T>();
            // Both views have the same shape and dtype. The scan reads each
            // logical element before writing that slot, including in-place.
            scan_product::<T>(
                input.shape,
                axis,
                exclusive,
                reverse,
                |i| unsafe { src.add(i).read_unaligned() },
                |i, value| unsafe { dst.add(i).write_unaligned(value) },
            );
            return Ok(());
        }
        let mut values = to_dense::<T>(input)?;
        let ptr = values.as_mut_ptr();
        // The temporary is dense and uniquely owned; each slot is read before
        // it is overwritten by the same lane.
        scan_product::<T>(
            input.shape,
            axis,
            exclusive,
            reverse,
            |i| unsafe { ptr.add(i).read() },
            |i, value| unsafe { ptr.add(i).write(value) },
        );
        return write_dense::<T>(output, &values);
    }
    let mut values = to_dense::<T>(input)?;
    if values.is_empty() {
        return write_dense::<T>(output, &values);
    }
    scan(
        &mut values,
        input.shape,
        axis,
        exclusive,
        reverse,
        T::from_f32_scalar(0.0).to_acc(),
        |left, right| left.c_add(right),
    );
    write_dense::<T>(output, &values)
}

#[inline]
fn scan_product<T: NumericElem>(
    shape: &[usize],
    axis: usize,
    exclusive: bool,
    reverse: bool,
    read: impl Fn(usize) -> T,
    mut write: impl FnMut(usize, T),
) {
    let inner = numel(&shape[axis + 1..]);
    let width = shape[axis];
    let one = T::from_f32_scalar(1.0);
    for outer in 0..numel(&shape[..axis]) {
        for i in 0..inner {
            let base = outer * width * inner + i;
            let offset = |n| base + (if reverse { width - 1 - n } else { n }) * inner;
            let mut total = read(offset(0));
            write(offset(0), if exclusive { one } else { total });
            for n in 1..width {
                let i = offset(n);
                let next = read(i);
                // Feed back the product rounded in the input dtype.
                let current = T::from_acc(total.to_acc().c_mul(next.to_acc()));
                write(i, if exclusive { total } else { current });
                total = current;
            }
        }
    }
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

fn double_range_count(start: f64, limit: f64, delta: f64) -> Result<usize> {
    if delta == 0.0 {
        return Err(EpError::KernelFailed(
            "Range: delta must not be zero".into(),
        ));
    }
    let count = ((limit - start) / delta).ceil();
    if !count.is_finite() || count >= usize::MAX as f64 {
        return Err(EpError::KernelFailed(
            "Range: output length is non-finite or exceeds addressable memory".into(),
        ));
    }
    Ok(count.max(0.0) as usize)
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
    fn range_int32_produces_int32_output() {
        // The Muse Glimmer vision tower builds int32 window offsets this way.
        let a = Owned::i32(&[], &[2]);
        let b = Owned::i32(&[], &[8]);
        let d = Owned::i32(&[], &[2]);
        let mut y = Owned::zeros(DataType::Int32, &[3]);
        RangeKernel
            .execute(&[a.view(), b.view(), d.view()], &mut [y.view_mut()])
            .unwrap();
        assert_eq!(y.to_i32(), vec![2, 4, 6]);
    }
    #[test]
    fn range_int32_negative_delta_counts_down() {
        let a = Owned::i32(&[], &[5]);
        let b = Owned::i32(&[], &[1]);
        let d = Owned::i32(&[], &[-2]);
        let mut y = Owned::zeros(DataType::Int32, &[2]);
        RangeKernel
            .execute(&[a.view(), b.view(), d.view()], &mut [y.view_mut()])
            .unwrap();
        assert_eq!(y.to_i32(), vec![5, 3]);
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
        let axis = Owned::i32(&[], &[-1]);
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
    fn cumprod_rejects_vector_and_out_of_range_axes_even_when_empty() {
        let x = Owned::zeros_f32(&[2, 0, 3]);
        let mut y = Owned::zeros_f32(&[2, 0, 3]);
        let k = CumProdKernel {
            exclusive: false,
            reverse: false,
        };
        for (axis, msg) in [
            (Owned::i64(&[1], &[0]), "rank-0"),
            (Owned::i64(&[], &[-4]), "out of range"),
            (Owned::f32(&[], &[1.0]), "Int32 or Int64"),
        ] {
            let err = k
                .execute(&[x.view(), axis.view()], &mut [y.view_mut()])
                .unwrap_err();
            assert!(err.to_string().contains(msg), "{err}");
        }
        let axis = Owned::i64(&[], &[-2]);
        k.execute(&[x.view(), axis.view()], &mut [y.view_mut()])
            .unwrap();
    }

    #[test]
    fn cumprod_preserves_first_nan_bits_and_matches_numpy_step_rounding() {
        let k = CumProdKernel {
            exclusive: false,
            reverse: false,
        };
        let f32_data = Owned::f32(&[2], &[f32::from_bits(0xff81_2345), 1.0]);
        let a = Owned::i32(&[], &[0]);
        let mut f32_out = Owned::zeros_f32(&[2]);
        k.execute(&[f32_data.view(), a.view()], &mut [f32_out.view_mut()])
            .unwrap();
        assert_eq!(f32_out.to_f32()[0].to_bits(), 0xff81_2345);

        let x = Owned::f16(&[5], &[1.13; 5]);
        let mut y = Owned::zeros(DataType::Float16, &[5]);
        k.execute(&[x.view(), a.view()], &mut [y.view_mut()])
            .unwrap();
        let y = to_dense::<half::f16>(&y.view()).unwrap();
        // np.cumprod(np.full(5, 1.13, dtype=np.float16)).view(np.uint16)
        assert_eq!(
            y.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            [0x3c85, 0x3d1b, 0x3dc5, 0x3e85, 0x3f5e]
        );
    }
    #[test]
    fn cumprod_2d_axis_zero() {
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let axis = Owned::i64(&[], &[0]);
        let mut y = Owned::zeros_f32(&[2, 3]);
        CumProdKernel {
            exclusive: false,
            reverse: false,
        }
        .execute(&[x.view(), axis.view()], &mut [y.view_mut()])
        .unwrap();
        assert_eq!(y.to_f32(), vec![1., 2., 3., 4., 10., 18.]);
    }

    #[test]
    #[ignore = "release microbenchmark; run under the shared host lock"]
    fn cumprod_cpu_microbenchmark() {
        use std::{hint::black_box, time::Instant};

        let axis = Owned::i64(&[], &[-1]);
        let kernel = CumProdKernel {
            exclusive: false,
            reverse: false,
        };
        for (shape, iterations) in [([16, 8], 20_000usize), ([128, 4096], 50)] {
            let [rows, width] = shape;
            let x = Owned::f32(&shape, &vec![1.0001; rows * width]);
            let mut out = Owned::zeros_f32(&shape);
            for strided in [false, true] {
                let mut input = x.view();
                let transposed = [1, rows as i64];
                if strided {
                    input.strides = &transposed;
                }
                let mut baseline = || {
                    let axis_slot = to_dense_i64(&axis.view()).unwrap();
                    let ax = (axis_slot[0] + shape.len() as i64) as usize;
                    let mut data = to_dense::<f32>(black_box(&input)).unwrap();
                    scan(&mut data, &shape, ax, false, false, 1.0, |a, b| a * b);
                    write_dense::<f32>(black_box(&mut out.view_mut()), &data).unwrap();
                };
                baseline();
                let expected = out.to_f32();
                kernel
                    .execute(&[input, axis.view()], &mut [out.view_mut()])
                    .unwrap();
                assert_eq!(out.to_f32(), expected);
                let mut samples = [Vec::new(), Vec::new()];
                for rep in 0..5 {
                    for arm in [rep % 2, 1 - rep % 2] {
                        let start = Instant::now();
                        for _ in 0..iterations {
                            if arm == 0 {
                                let ax = (to_dense_i64(black_box(&axis.view())).unwrap()[0] + 2)
                                    as usize;
                                let mut data = to_dense::<f32>(black_box(&input)).unwrap();
                                scan(&mut data, &shape, ax, false, false, 1.0, |a, b| a * b);
                                write_dense::<f32>(black_box(&mut out.view_mut()), &data).unwrap();
                            } else {
                                kernel
                                    .execute(
                                        black_box(&[input, axis.view()]),
                                        black_box(&mut [out.view_mut()]),
                                    )
                                    .unwrap();
                            }
                        }
                        samples[arm].push(start.elapsed().as_secs_f64() * 1e6 / iterations as f64);
                    }
                }
                for (arm, times) in ["dense-buffer baseline", "native scan"]
                    .into_iter()
                    .zip(&mut samples)
                {
                    times.sort_by(f64::total_cmp);
                    eprintln!(
                        "CumProd CPU {shape:?} strided={strided} {arm}: {:.3} us [{:.3}, {:.3}]",
                        times[2], times[0], times[4]
                    );
                }
            }
        }
    }

    #[test]
    fn range_bf16_matches_widened_f32_reference() {
        let start = Owned::bf16(&[], &[-1.]);
        let limit = Owned::bf16(&[], &[1.]);
        let delta = Owned::bf16(&[], &[0.5]);
        let mut out = Owned::zeros(DataType::BFloat16, &[4]);
        RangeKernel
            .execute(
                &[start.view(), limit.view(), delta.view()],
                &mut [out.view_mut()],
            )
            .unwrap();
        assert_eq!(out.to_bf16_as_f32(), vec![-1., -0.5, 0., 0.5]);
    }

    #[test]
    fn range_v27_stash_precision_controls_low_precision_length() {
        for dtype in [DataType::Float16, DataType::BFloat16] {
            let make_scalar = |value: f32| {
                let bits = if dtype == DataType::Float16 {
                    half::f16::from_f32(value).to_bits()
                } else {
                    half::bf16::from_f32(value).to_bits()
                };
                (bits, dtype)
            };
            let (start, _) = make_scalar(-2.0_f32.powi(-24));
            let (limit, _) = make_scalar(16.0);
            let (delta, _) = make_scalar(4.0);
            let input = |word: &u16| {
                TensorView::new(
                    onnx_runtime_ep_api::DevicePtr(std::ptr::from_ref(word).cast()),
                    dtype,
                    &[],
                    &[],
                    onnx_runtime_ir::DeviceId::cpu(),
                )
            };
            let inputs = [input(&start), input(&limit), input(&delta)];
            for (stash_type, len) in [(1, 4), (11, 5)] {
                let mut actual = vec![0u16; len];
                let shape = [len];
                let output = TensorMut::new(
                    onnx_runtime_ep_api::DevicePtrMut(actual.as_mut_ptr().cast()),
                    dtype,
                    &shape,
                    &[1],
                    onnx_runtime_ir::DeviceId::cpu(),
                );
                Range27Kernel { stash_type }
                    .execute(&inputs, &mut [output])
                    .unwrap();
                let mut expected = vec![start];
                expected.extend(
                    [4.0, 8.0, 12.0, 16.0][..len - 1]
                        .iter()
                        .map(|&v| make_scalar(v).0),
                );
                assert_eq!(actual, expected);
            }
        }
    }
    #[test]
    fn tile_bf16_preserves_element_bits() {
        let x = Owned::bf16(&[2], &[1., -2.]);
        let repeats = Owned::i64(&[1], &[2]);
        let mut out = Owned::zeros(DataType::BFloat16, &[4]);
        TileKernel
            .execute(&[x.view(), repeats.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(
            out.to_u16_bits(),
            vec![
                x.to_u16_bits()[0],
                x.to_u16_bits()[1],
                x.to_u16_bits()[0],
                x.to_u16_bits()[1]
            ]
        );
    }
}

pub(super) struct LogCumSumExpFactory;

impl KernelFactory for LogCumSumExpFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(LogCumSumExpKernel(
            Math::from_node(node).map_err(EpError::KernelFailed)?,
        )))
    }
}

struct LogCumSumExpKernel(Math);

impl Kernel for LogCumSumExpKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let scan = validate(self.0, inputs, outputs)?.expect("validated scan axis");
        super::unary_math::run_dense_unary(
            self.0.name(),
            &inputs[0],
            &mut outputs[0],
            |x| scan_f32(x, scan),
            |x| scan_f64(x, scan),
        )
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

macro_rules! log_cum_sum_exp {
    ($name:ident, $float:ty) => {
        fn $name(data: &mut [$float], scan: Scan) {
            let Scan {
                lanes,
                length,
                inner,
            } = scan;
            for lane in 0..lanes {
                let start = (lane / inner) * length * inner + lane % inner;
                let mut max = <$float>::NEG_INFINITY;
                // Keep the leading 1 separate so log1p retains small prefixes.
                let mut rest = 0.;
                for k in 0..length {
                    let x = &mut data[start + k * inner];
                    if x.is_nan() {
                        max = *x;
                    } else if !max.is_nan() {
                        if *x == <$float>::INFINITY {
                            max = *x;
                        } else if x.is_finite() && max != <$float>::INFINITY {
                            if *x > max {
                                rest = (1. + rest) * (max - *x).exp();
                                max = *x;
                            } else {
                                rest += (*x - max).exp();
                            }
                        }
                    }
                    *x = if max.is_finite() {
                        max + rest.ln_1p()
                    } else {
                        max
                    };
                }
            }
        }
    };
}
log_cum_sum_exp!(scan_f32, f32);
log_cum_sum_exp!(scan_f64, f64);

#[cfg(test)]
mod log_cum_sum_exp_tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    #[test]
    fn scan_preserves_small_prefix_and_handles_interleaved_lanes() {
        let x = Owned::f32(&[2, 2], &[0., 1000., -30., 1000.]);
        let mut y = Owned::zeros_f32(&[2, 2]);
        LogCumSumExpKernel(Math::LogCumSumExp { axis: -2 })
            .execute(&[x.view()], &mut [y.view_mut()])
            .unwrap();
        let y = y.to_f32();
        assert_eq!(&y[..2], &[0., 1000.]);
        assert!((y[2] - 9.357_623e-14).abs() < 1e-19);
        assert!((y[3] - 1_000.693_2).abs() < 1e-4);
    }
}
