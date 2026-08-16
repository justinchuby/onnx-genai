//! `QLinearMatMul`: integer matrix multiplication with linear quantization.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Dim, Node, Shape, broadcast_shapes, compute_contiguous_strides};
use rayon::prelude::*;

use super::{check_arity, to_dense_bytes, write_dense_bytes};
use crate::strided::numel;

/// Multiply-accumulate count below which the integer accumulation runs on the
/// calling thread. A rayon fork costs on the order of microseconds; this is the
/// point where the accumulation itself is comfortably past that, so the guard
/// never trades measurable throughput for it.
const PARALLEL_MIN_WORK: usize = 1 << 16;

pub struct QLinearMatMulKernel;
pub struct QLinearMatMulFactory;

impl KernelFactory for QLinearMatMulFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(QLinearMatMulKernel))
    }
}

/// Return a claim-time denial for metadata the CPU reference kernel cannot run.
pub(crate) fn unsupported_reason(
    input_dtypes: &[DataType],
    input_shapes: &[Shape],
) -> Option<String> {
    if !input_dtypes.is_empty() {
        if input_dtypes.len() != 8 {
            return Some(format!(
                "QLinearMatMul requires 8 inputs, got {}",
                input_dtypes.len()
            ));
        }
        for &(index, name) in &[(0, "A"), (3, "B"), (7, "y_zero_point")] {
            if !is_quantized(input_dtypes[index]) {
                return Some(format!(
                    "QLinearMatMul: {name} must have Int8 or Uint8 dtype, got {:?}",
                    input_dtypes[index]
                ));
            }
        }
        for &(integer, value, name) in &[(0, 2, "a_zero_point"), (3, 5, "b_zero_point")] {
            if input_dtypes[value] != input_dtypes[integer] {
                return Some(format!(
                    "QLinearMatMul: {name} dtype {:?} must match input dtype {:?}",
                    input_dtypes[value], input_dtypes[integer]
                ));
            }
        }
        for &index in &[1, 4, 6] {
            if input_dtypes[index] != DataType::Float32 {
                return Some(format!(
                    "QLinearMatMul: scale input {index} must be Float32, got {:?}",
                    input_dtypes[index]
                ));
            }
        }
    }
    if input_shapes.is_empty() {
        return None;
    }
    if input_shapes.len() != 8 {
        return Some(format!(
            "QLinearMatMul requires 8 input shapes, got {}",
            input_shapes.len()
        ));
    }
    if let Err(reason) = validate_claim_shapes(input_shapes) {
        return Some(reason);
    }
    None
}

fn validate_claim_shapes(shapes: &[Shape]) -> std::result::Result<(), String> {
    let a = &shapes[0];
    let b = &shapes[3];
    if a.is_empty() || b.is_empty() {
        return Err("QLinearMatMul: operands must be at least 1-D".into());
    }
    if !dims_compatible(
        a[a.len() - 1],
        b[if b.len() == 1 { 0 } else { b.len() - 2 }],
    ) {
        return Err("QLinearMatMul: inner dimensions are not provably equal".into());
    }
    validate_batch_broadcast(
        &a[..a.len().saturating_sub(2)],
        &b[..b.len().saturating_sub(2)],
    )?;
    validate_claim_quant_pair("a", &shapes[1], &shapes[2], a, QuantAxis::Row)?;
    validate_claim_quant_pair("b", &shapes[4], &shapes[5], b, QuantAxis::Column)?;
    if shapes[6] != shapes[7] {
        return Err("QLinearMatMul: y_scale and y_zero_point shapes must match".into());
    }
    if !is_claim_scalar_shape(&shapes[6]) {
        return Err("QLinearMatMul: output scale and zero point must be scalar".into());
    }
    Ok(())
}

fn validate_batch_broadcast(a: &[Dim], b: &[Dim]) -> std::result::Result<(), String> {
    let rank = a.len().max(b.len());
    for trailing in 0..rank {
        let a_dim = a
            .len()
            .checked_sub(trailing + 1)
            .map_or(Dim::Static(1), |index| a[index]);
        let b_dim = b
            .len()
            .checked_sub(trailing + 1)
            .map_or(Dim::Static(1), |index| b[index]);
        if !dims_broadcastable(a_dim, b_dim) {
            return Err("QLinearMatMul: batch dimensions are not provably broadcastable".into());
        }
    }
    Ok(())
}

fn validate_claim_quant_pair(
    name: &str,
    scale: &Shape,
    zero_point: &Shape,
    operand: &Shape,
    axis: QuantAxis,
) -> std::result::Result<(), String> {
    if scale != zero_point {
        return Err(format!(
            "QLinearMatMul: {name}_scale and {name}_zero_point shapes must match"
        ));
    }
    if is_claim_scalar_shape(scale) || is_claim_axis_shape(scale, operand, axis) {
        Ok(())
    } else {
        Err(format!(
            "QLinearMatMul: invalid {name} scale/zero-point shape"
        ))
    }
}

fn is_claim_scalar_shape(shape: &[Dim]) -> bool {
    shape.is_empty() || shape == [Dim::Static(1)]
}

fn is_claim_axis_shape(shape: &[Dim], operand: &[Dim], axis: QuantAxis) -> bool {
    match operand.len() {
        0 | 1 => false,
        2 => {
            shape.len() == 1
                && dims_equal(
                    shape[0],
                    operand[match axis {
                        QuantAxis::Row => 0,
                        QuantAxis::Column => 1,
                    }],
                )
        }
        rank => {
            if shape.len() != rank {
                return false;
            }
            let batch = rank - 2;
            if !shape[..batch]
                .iter()
                .zip(&operand[..batch])
                .all(|(&left, &right)| dims_equal(left, right))
            {
                return false;
            }
            match axis {
                QuantAxis::Row => {
                    dims_equal(shape[batch], operand[batch]) && shape[batch + 1] == Dim::Static(1)
                }
                QuantAxis::Column => {
                    shape[batch] == Dim::Static(1)
                        && dims_equal(shape[batch + 1], operand[batch + 1])
                }
            }
        }
    }
}

fn dims_equal(left: Dim, right: Dim) -> bool {
    left == right
}

fn dims_compatible(left: Dim, right: Dim) -> bool {
    dims_equal(left, right)
}

fn dims_broadcastable(left: Dim, right: Dim) -> bool {
    dims_equal(left, right) || left == Dim::Static(1) || right == Dim::Static(1)
}

impl Kernel for QLinearMatMulKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("QLinearMatMul", inputs, outputs, 8, 8, 1)?;
        let a = &inputs[0];
        let b = &inputs[3];
        if !is_quantized(a.dtype) || !is_quantized(b.dtype) || !is_quantized(outputs[0].dtype) {
            return Err(EpError::KernelFailed(
                "QLinearMatMul: A, B, and output must have Int8 or Uint8 dtype".into(),
            ));
        }
        if inputs[2].dtype != a.dtype || inputs[5].dtype != b.dtype {
            return Err(EpError::KernelFailed(
                "QLinearMatMul: each input zero_point must match its quantized input dtype".into(),
            ));
        }
        if inputs[7].dtype != outputs[0].dtype {
            return Err(EpError::KernelFailed(
                "QLinearMatMul: output dtype must match y_zero_point dtype".into(),
            ));
        }
        for &index in &[1, 4, 6] {
            if inputs[index].dtype != DataType::Float32 {
                return Err(EpError::KernelFailed(format!(
                    "QLinearMatMul: scale input {index} must be Float32"
                )));
            }
        }

        let geometry = Geometry::new(a.shape, b.shape)?;
        if outputs[0].shape != geometry.output_shape {
            return Err(EpError::KernelFailed(format!(
                "QLinearMatMul: output shape {:?} must be {:?}",
                outputs[0].shape, geometry.output_shape
            )));
        }
        let a_quant = QuantParams::load("a", &inputs[1], &inputs[2], a.shape, QuantAxis::Row)?;
        let b_quant = QuantParams::load("b", &inputs[4], &inputs[5], b.shape, QuantAxis::Column)?;
        let (y_scale, y_zero_point) = output_quant_params(&inputs[6], &inputs[7])?;

        let a = read_quantized(&inputs[0])?;
        let b = read_quantized(&inputs[3])?;
        let (m, k, n) = (geometry.m, geometry.k, geometry.n);
        let mut output = Vec::with_capacity(geometry.result_len);
        let mut batch_index = vec![0; geometry.batch_shape.len()];
        // Hoisted out of the batch loop: both are re-filled per batch, so a
        // many-batch call allocates once rather than once per batch.
        let mut products: Vec<i32> = Vec::new();
        let mut b_zero_points: Vec<i32> = Vec::new();
        for batch in 0..geometry.batch_count {
            let a_batch = geometry.a_batch_offset(&batch_index);
            let b_batch = geometry.b_batch_offset(&batch_index);
            let a_offset = a_batch * m * k;
            let b_offset = b_batch * k * n;
            // `n == 0` produces no output for this batch, and both
            // `par_chunks_mut(0)` and `chunks_exact(0)` panic, so leave early.
            if n == 0 {
                if batch + 1 < geometry.batch_count {
                    next_index(&geometry.batch_shape, &mut batch_index);
                }
                continue;
            }
            b_zero_points.clear();
            b_zero_points.extend((0..n).map(|column| b_quant.at(b_batch, column).1));

            // Accumulate the integer product with `k` outermost so `B` is read
            // along its rows. The previous order walked `B` down a column with
            // stride `n`, which touches a fresh cache line per element.
            //
            // The zero points are lifted out of the inner loop by expanding
            //   sum_k (a_k - az) * (b_kn - bz_n)
            //     = sum_k (a_k - az) * b_kn  -  bz_n * sum_k (a_k - az)
            // which is an identity over the integers, so under wrapping
            // arithmetic (exactly arithmetic mod 2^32) both sides reduce to the
            // same `i32`. The result is bit-identical to the previous loop,
            // including on overflow, and the inner loop becomes a plain
            // multiply-accumulate over two contiguous slices.
            //
            // Rows are independent, so integer accumulation stays deterministic
            // whether the rows are walked serially or under `par_chunks_mut`:
            // neither changes a summation order.
            products.clear();
            products.resize(m * n, 0);
            let b_zero_points = &b_zero_points[..];
            let accumulate_row = |row: usize, accumulators: &mut [i32]| {
                let a_zero_point = a_quant.at(a_batch, row).1;
                let a_row = &a[a_offset + row * k..a_offset + row * k + k];
                let mut a_sum = 0i32;
                for (inner, &a_value) in a_row.iter().enumerate() {
                    let centered = a_value.wrapping_sub(a_zero_point);
                    a_sum = a_sum.wrapping_add(centered);
                    if centered == 0 {
                        continue;
                    }
                    let start = b_offset + inner * n;
                    let b_row = &b[start..start + n];
                    for (accumulator, &b_value) in accumulators.iter_mut().zip(b_row) {
                        *accumulator = accumulator.wrapping_add(centered.wrapping_mul(b_value));
                    }
                }
                for (accumulator, &b_zero_point) in accumulators.iter_mut().zip(b_zero_points) {
                    *accumulator = accumulator.wrapping_sub(a_sum.wrapping_mul(b_zero_point));
                }
            };
            // One chunk cannot be split, and a fork is pure overhead below the
            // point where the accumulation dominates it. `m == 1` is the decode
            // shape, so this is the common case rather than a corner.
            if m <= 1 || rayon::current_num_threads() <= 1 || m * n * k < PARALLEL_MIN_WORK {
                for (row, accumulators) in products.chunks_mut(n).enumerate() {
                    accumulate_row(row, accumulators);
                }
            } else {
                products
                    .par_chunks_mut(n)
                    .enumerate()
                    .for_each(|(row, accumulators)| accumulate_row(row, accumulators));
            }

            for (row, accumulators) in products.chunks_exact(n).enumerate() {
                let a_scale = a_quant.at(a_batch, row).0;
                for (column, &accumulated) in accumulators.iter().enumerate() {
                    let b_scale = b_quant.at(b_batch, column).0;
                    let scale = a_scale * b_scale / y_scale;
                    let value = (accumulated as f32 * scale).round_ties_even() as i64
                        + i64::from(y_zero_point);
                    push_quantized(&mut output, outputs[0].dtype, value)?;
                }
            }
            if batch + 1 < geometry.batch_count {
                next_index(&geometry.batch_shape, &mut batch_index);
            }
        }
        write_dense_bytes(&mut outputs[0], &output)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn is_quantized(dtype: DataType) -> bool {
    matches!(dtype, DataType::Int8 | DataType::Uint8)
}

#[derive(Clone, Copy)]
enum QuantAxis {
    Row,
    Column,
}

struct QuantParams {
    scales: Vec<f32>,
    zero_points: Vec<i32>,
    axis_len: usize,
    per_axis: bool,
}

impl QuantParams {
    fn load(
        name: &str,
        scale: &TensorView,
        zero_point: &TensorView,
        operand_shape: &[usize],
        axis: QuantAxis,
    ) -> Result<Self> {
        if scale.shape != zero_point.shape {
            return Err(EpError::KernelFailed(format!(
                "QLinearMatMul: {name}_scale and {name}_zero_point shapes must match"
            )));
        }
        let per_axis = if is_scalar_shape(scale.shape) {
            false
        } else if is_axis_shape(scale.shape, operand_shape, axis) {
            true
        } else {
            return Err(EpError::KernelFailed(format!(
                "QLinearMatMul: invalid {name} scale/zero-point shape {:?} for operand shape {:?}",
                scale.shape, operand_shape
            )));
        };
        let scales = read_scales(scale)?;
        let zero_points = read_quantized(zero_point)?;
        let axis_len = match axis {
            QuantAxis::Row => {
                if operand_shape.len() == 1 {
                    1
                } else {
                    operand_shape[operand_shape.len() - 2]
                }
            }
            QuantAxis::Column => *operand_shape.last().unwrap_or(&1),
        };
        Ok(Self {
            scales,
            zero_points,
            axis_len,
            per_axis,
        })
    }

    fn at(&self, source_batch: usize, axis_index: usize) -> (f32, i32) {
        let index = if self.per_axis {
            source_batch * self.axis_len + axis_index
        } else {
            0
        };
        (self.scales[index], self.zero_points[index])
    }
}

fn is_scalar_shape(shape: &[usize]) -> bool {
    shape.is_empty() || shape == [1]
}

fn is_axis_shape(shape: &[usize], operand: &[usize], axis: QuantAxis) -> bool {
    match operand.len() {
        0 | 1 => false,
        2 => {
            shape
                == [operand[match axis {
                    QuantAxis::Row => 0,
                    QuantAxis::Column => 1,
                }]]
        }
        rank => {
            if shape.len() != rank || shape[..rank - 2] != operand[..rank - 2] {
                return false;
            }
            match axis {
                QuantAxis::Row => shape[rank - 2] == operand[rank - 2] && shape[rank - 1] == 1,
                QuantAxis::Column => shape[rank - 2] == 1 && shape[rank - 1] == operand[rank - 1],
            }
        }
    }
}

fn output_quant_params(scale: &TensorView, zero_point: &TensorView) -> Result<(f32, i32)> {
    if scale.shape != zero_point.shape {
        return Err(EpError::KernelFailed(
            "QLinearMatMul: y_scale and y_zero_point shapes must match".into(),
        ));
    }
    if !is_scalar_shape(scale.shape) {
        return Err(EpError::KernelFailed(
            "QLinearMatMul: output scale and zero point must be scalar".into(),
        ));
    }
    Ok((read_scales(scale)?[0], read_quantized(zero_point)?[0]))
}

fn read_scales(view: &TensorView) -> Result<Vec<f32>> {
    let bytes = to_dense_bytes(view)?;
    let scales: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    if scales
        .iter()
        .any(|value| *value <= 0.0 || !value.is_finite())
    {
        return Err(EpError::KernelFailed(
            "QLinearMatMul: scales must be finite and positive".into(),
        ));
    }
    Ok(scales)
}

fn read_quantized(view: &TensorView) -> Result<Vec<i32>> {
    let bytes = to_dense_bytes(view)?;
    match view.dtype {
        DataType::Int8 => Ok(bytes.into_iter().map(|value| value as i8 as i32).collect()),
        DataType::Uint8 => Ok(bytes.into_iter().map(i32::from).collect()),
        other => Err(EpError::KernelFailed(format!(
            "QLinearMatMul: expected Int8 or Uint8 tensor, got {other:?}"
        ))),
    }
}

fn push_quantized(bytes: &mut Vec<u8>, dtype: DataType, value: i64) -> Result<()> {
    match dtype {
        DataType::Int8 => {
            bytes.push(value.clamp(i8::MIN as i64, i8::MAX as i64) as i8 as u8);
            Ok(())
        }
        DataType::Uint8 => {
            bytes.push(value.clamp(0, u8::MAX as i64) as u8);
            Ok(())
        }
        other => Err(EpError::KernelFailed(format!(
            "QLinearMatMul: unsupported output dtype {other:?}"
        ))),
    }
}

struct Geometry {
    m: usize,
    k: usize,
    n: usize,
    a_batch: Vec<usize>,
    b_batch: Vec<usize>,
    a_batch_strides: Vec<i64>,
    b_batch_strides: Vec<i64>,
    batch_shape: Vec<usize>,
    batch_count: usize,
    result_len: usize,
    output_shape: Vec<usize>,
}

impl Geometry {
    fn new(a: &[usize], b: &[usize]) -> Result<Self> {
        let a_1d = a.len() == 1;
        let b_1d = b.len() == 1;
        let a = if a_1d { vec![1, a[0]] } else { a.to_vec() };
        let b = if b_1d { vec![b[0], 1] } else { b.to_vec() };
        if a.len() < 2 || b.len() < 2 {
            return Err(EpError::KernelFailed(
                "QLinearMatMul: operands must be at least 1-D".into(),
            ));
        }
        let m = a[a.len() - 2];
        let k = a[a.len() - 1];
        let b_k = b[b.len() - 2];
        let n = b[b.len() - 1];
        if k != b_k {
            return Err(EpError::KernelFailed(format!(
                "QLinearMatMul: inner dims disagree ({k} vs {b_k})"
            )));
        }
        let a_batch = a[..a.len() - 2].to_vec();
        let b_batch = b[..b.len() - 2].to_vec();
        let batch_shape = broadcast_shapes(&a_batch, &b_batch)?;
        let batch_count = numel(&batch_shape);
        let mut output_shape = batch_shape.clone();
        if !a_1d {
            output_shape.push(m);
        }
        if !b_1d {
            output_shape.push(n);
        }
        Ok(Self {
            m,
            k,
            n,
            a_batch_strides: compute_contiguous_strides(&a_batch),
            b_batch_strides: compute_contiguous_strides(&b_batch),
            a_batch,
            b_batch,
            batch_shape,
            batch_count,
            result_len: batch_count * m * n,
            output_shape,
        })
    }

    fn a_batch_offset(&self, batch_index: &[usize]) -> usize {
        broadcast_offset(batch_index, &self.a_batch, &self.a_batch_strides)
    }

    fn b_batch_offset(&self, batch_index: &[usize]) -> usize {
        broadcast_offset(batch_index, &self.b_batch, &self.b_batch_strides)
    }
}

fn broadcast_offset(batch_index: &[usize], shape: &[usize], strides: &[i64]) -> usize {
    let leading = batch_index.len() - shape.len();
    shape
        .iter()
        .zip(strides)
        .enumerate()
        .map(|(index, (&dimension, &stride))| {
            if dimension == 1 {
                0
            } else {
                batch_index[leading + index] * stride as usize
            }
        })
        .sum()
}

fn next_index(shape: &[usize], index: &mut [usize]) {
    for (dimension, coordinate) in shape.iter().zip(index).rev() {
        *coordinate += 1;
        if *coordinate < *dimension {
            return;
        }
        *coordinate = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::compute_contiguous_strides;

    fn i8(shape: &[usize], values: &[i8]) -> Owned {
        Owned {
            bytes: values.iter().map(|&value| value as u8).collect(),
            shape: shape.to_vec(),
            strides: compute_contiguous_strides(shape),
            dtype: DataType::Int8,
        }
    }

    struct Reference<'a> {
        a: &'a [i32],
        a_shape: &'a [usize],
        a_scales: &'a [f32],
        a_zeros: &'a [i32],
        b: &'a [i32],
        b_shape: &'a [usize],
        b_scales: &'a [f32],
        b_zeros: &'a [i32],
        y_scale: f32,
        y_zero: i32,
        output_dtype: DataType,
    }

    fn reference(input: Reference<'_>) -> Vec<i64> {
        let geometry = Geometry::new(input.a_shape, input.b_shape).unwrap();
        let a_per_row = input.a_scales.len() > 1 || input.a_zeros.len() > 1;
        let b_per_column = input.b_scales.len() > 1 || input.b_zeros.len() > 1;
        let mut batch_index = vec![0; geometry.batch_shape.len()];
        let mut output = Vec::with_capacity(geometry.result_len);
        for batch in 0..geometry.batch_count {
            let a_batch = geometry.a_batch_offset(&batch_index);
            let b_batch = geometry.b_batch_offset(&batch_index);
            for row in 0..geometry.m {
                for column in 0..geometry.n {
                    let a_quant_index = if a_per_row {
                        a_batch * geometry.m + row
                    } else {
                        0
                    };
                    let b_quant_index = if b_per_column {
                        b_batch * geometry.n + column
                    } else {
                        0
                    };
                    let mut product = 0.0f64;
                    for inner in 0..geometry.k {
                        let a_index = a_batch * geometry.m * geometry.k + row * geometry.k + inner;
                        let b_index =
                            b_batch * geometry.k * geometry.n + inner * geometry.n + column;
                        let a = f64::from(input.a[a_index] - input.a_zeros[a_quant_index])
                            * f64::from(input.a_scales[a_quant_index]);
                        let b = f64::from(input.b[b_index] - input.b_zeros[b_quant_index])
                            * f64::from(input.b_scales[b_quant_index]);
                        product += a * b;
                    }
                    let quantized = (product / f64::from(input.y_scale)).round_ties_even() as i64
                        + i64::from(input.y_zero);
                    output.push(match input.output_dtype {
                        DataType::Int8 => quantized.clamp(i8::MIN as i64, i8::MAX as i64),
                        DataType::Uint8 => quantized.clamp(0, u8::MAX as i64),
                        _ => unreachable!(),
                    });
                }
            }
            if batch + 1 < geometry.batch_count {
                next_index(&geometry.batch_shape, &mut batch_index);
            }
        }
        output
    }

    fn execute(inputs: [&Owned; 8], output_dtype: DataType, output_shape: &[usize]) -> Owned {
        let mut output = Owned::zeros(output_dtype, output_shape);
        QLinearMatMulKernel
            .execute(&inputs.map(|input| input.view()), &mut [output.view_mut()])
            .unwrap();
        output
    }

    fn output_values(output: &Owned) -> Vec<i64> {
        match output.dtype {
            DataType::Int8 => output
                .bytes
                .iter()
                .map(|&value| i64::from(value as i8))
                .collect(),
            DataType::Uint8 => output.bytes.iter().map(|&value| i64::from(value)).collect(),
            _ => unreachable!(),
        }
    }

    #[test]
    fn qlinear_matmul_uint8_per_tensor_matches_dequant_matmul_requant_reference() {
        let a = Owned::u8(&[2, 3], &[130, 125, 140, 120, 135, 128]);
        let a_scale = Owned::f32(&[], &[0.25]);
        let a_zero = Owned::u8(&[], &[128]);
        let b = Owned::u8(&[3, 2], &[131, 126, 120, 140, 128, 130]);
        let b_scale = Owned::f32(&[], &[0.5]);
        let b_zero = Owned::u8(&[], &[128]);
        let y_scale = Owned::f32(&[], &[0.125]);
        let y_zero = Owned::u8(&[], &[127]);
        let out = execute(
            [
                &a, &a_scale, &a_zero, &b, &b_scale, &b_zero, &y_scale, &y_zero,
            ],
            DataType::Uint8,
            &[2, 2],
        );
        let expected = reference(Reference {
            a: &[130, 125, 140, 120, 135, 128],
            a_shape: &[2, 3],
            a_scales: &[0.25],
            a_zeros: &[128],
            b: &[131, 126, 120, 140, 128, 130],
            b_shape: &[3, 2],
            b_scales: &[0.5],
            b_zeros: &[128],
            y_scale: 0.125,
            y_zero: 127,
            output_dtype: DataType::Uint8,
        });
        assert_eq!(output_values(&out), expected);
    }

    #[test]
    fn qlinear_matmul_int8_per_column_scales_matches_reference() {
        let a = i8(&[1, 2], &[-2, 5]);
        let a_scale = Owned::f32(&[], &[0.25]);
        let a_zero = i8(&[], &[-1]);
        let b = i8(&[2, 3], &[3, -4, 7, 2, 5, -6]);
        let b_scale = Owned::f32(&[3], &[0.5, 0.25, 0.125]);
        let b_zero = i8(&[3], &[1, -2, 3]);
        let y_scale = Owned::f32(&[], &[0.25]);
        let y_zero = i8(&[], &[2]);
        let out = execute(
            [
                &a, &a_scale, &a_zero, &b, &b_scale, &b_zero, &y_scale, &y_zero,
            ],
            DataType::Int8,
            &[1, 3],
        );
        let expected = reference(Reference {
            a: &[-2, 5],
            a_shape: &[1, 2],
            a_scales: &[0.25],
            a_zeros: &[-1],
            b: &[3, -4, 7, 2, 5, -6],
            b_shape: &[2, 3],
            b_scales: &[0.5, 0.25, 0.125],
            b_zeros: &[1, -2, 3],
            y_scale: 0.25,
            y_zero: 2,
            output_dtype: DataType::Int8,
        });
        assert_eq!(output_values(&out), expected);
    }

    #[test]
    fn qlinear_matmul_uint8_per_row_a_scales_matches_reference() {
        let a_values = [10, 14, 7, 20];
        let a = Owned::u8(&[2, 2], &a_values.map(|value| value as u8));
        let a_scale = Owned::f32(&[2], &[0.5, 0.125]);
        let a_zero = Owned::u8(&[2], &[8, 6]);
        let b_values = [3, 9, 5, 1];
        let b = Owned::u8(&[2, 2], &b_values.map(|value| value as u8));
        let b_scale = Owned::f32(&[2], &[0.25, 0.5]);
        let b_zero = Owned::u8(&[2], &[2, 4]);
        let y_scale = Owned::f32(&[], &[0.125]);
        let y_zero = Owned::u8(&[], &[100]);
        let out = execute(
            [
                &a, &a_scale, &a_zero, &b, &b_scale, &b_zero, &y_scale, &y_zero,
            ],
            DataType::Uint8,
            &[2, 2],
        );
        let expected = reference(Reference {
            a: &a_values,
            a_shape: &[2, 2],
            a_scales: &[0.5, 0.125],
            a_zeros: &[8, 6],
            b: &b_values,
            b_shape: &[2, 2],
            b_scales: &[0.25, 0.5],
            b_zeros: &[2, 4],
            y_scale: 0.125,
            y_zero: 100,
            output_dtype: DataType::Uint8,
        });
        assert_eq!(output_values(&out), expected);
    }

    #[test]
    fn qlinear_matmul_batched_per_row_and_per_column_broadcasts_match_reference() {
        let a_values = [12, 8, 7, 15, 5, 20, 9, 4];
        let a = Owned::u8(&[2, 2, 2], &a_values.map(|value| value as u8));
        let a_scale = Owned::f32(&[2, 2, 1], &[0.5, 0.25, 0.125, 0.75]);
        let a_zero = Owned::u8(&[2, 2, 1], &[10, 8, 6, 5]);
        let b_values = [3, -4, 6, 2];
        let b = i8(&[1, 2, 2], &b_values);
        let b_scale = Owned::f32(&[1, 1, 2], &[0.5, 0.25]);
        let b_zero = i8(&[1, 1, 2], &[1, -2]);
        let y_scale = Owned::f32(&[1], &[0.125]);
        let y_zero = Owned::u8(&[1], &[120]);
        let out = execute(
            [
                &a, &a_scale, &a_zero, &b, &b_scale, &b_zero, &y_scale, &y_zero,
            ],
            DataType::Uint8,
            &[2, 2, 2],
        );
        let a_scales = [0.5, 0.25, 0.125, 0.75];
        let a_zeros = [10, 8, 6, 5];
        let b_scales = [0.5, 0.25];
        let b_zeros = [1, -2];
        let mut expected = Vec::with_capacity(8);
        for batch in 0..2 {
            for row in 0..2 {
                for column in 0..2 {
                    let mut product = 0.0f64;
                    for inner in 0..2 {
                        let a_index = batch * 4 + row * 2 + inner;
                        let b_index = inner * 2 + column;
                        let a = f64::from(a_values[a_index] - a_zeros[batch * 2 + row])
                            * a_scales[batch * 2 + row];
                        let b = f64::from(b_values[b_index] - b_zeros[column]) * b_scales[column];
                        product += a * b;
                    }
                    expected.push(((product / 0.125).round_ties_even() as i64 + 120).clamp(0, 255));
                }
            }
        }
        assert_eq!(expected, vec![108, 108, 153, 135, 154, 134, 129, 102]);
        assert_eq!(output_values(&out), expected);
    }

    #[test]
    fn qlinear_matmul_rounds_ties_to_even_and_saturates_int8() {
        let a_values = [1, 1, 1, 1];
        let a = i8(&[1, 4], &a_values);
        let a_scale = Owned::f32(&[], &[1.0]);
        let a_zero = i8(&[], &[0]);
        let b_values = [
            1, 3, 127, -128, 0, 0, 127, -128, 0, 0, 127, -128, 0, 0, 127, -128,
        ];
        let b = i8(&[4, 4], &b_values);
        let b_scale = Owned::f32(&[], &[1.0]);
        let b_zero = i8(&[], &[0]);
        let y_scale = Owned::f32(&[], &[2.0]);
        let y_zero = i8(&[], &[0]);
        let out = execute(
            [
                &a, &a_scale, &a_zero, &b, &b_scale, &b_zero, &y_scale, &y_zero,
            ],
            DataType::Int8,
            &[1, 4],
        );
        let expected = reference(Reference {
            a: &a_values.map(i32::from),
            a_shape: &[1, 4],
            a_scales: &[1.0],
            a_zeros: &[0],
            b: &b_values.map(i32::from),
            b_shape: &[4, 4],
            b_scales: &[1.0],
            b_zeros: &[0],
            y_scale: 2.0,
            y_zero: 0,
            output_dtype: DataType::Int8,
        });
        assert_eq!(output_values(&out), expected);
        assert_eq!(expected, vec![0, 2, 127, -128]);
    }

    #[test]
    fn qlinear_matmul_rejects_mismatched_scale_and_zero_point_shapes() {
        let a = Owned::u8(&[2, 2], &[1, 2, 3, 4]);
        let a_scale = Owned::f32(&[2], &[0.5, 0.25]);
        let a_zero = Owned::u8(&[], &[0]);
        let b = Owned::u8(&[2, 1], &[1, 1]);
        let b_scale = Owned::f32(&[], &[1.0]);
        let b_zero = Owned::u8(&[], &[0]);
        let y_scale = Owned::f32(&[], &[1.0]);
        let y_zero = Owned::u8(&[], &[0]);
        let mut out = Owned::zeros(DataType::Uint8, &[2, 1]);
        let error = QLinearMatMulKernel
            .execute(
                &[
                    a.view(),
                    a_scale.view(),
                    a_zero.view(),
                    b.view(),
                    b_scale.view(),
                    b_zero.view(),
                    y_scale.view(),
                    y_zero.view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap_err();
        assert!(error.to_string().contains("shapes must match"), "{error}");
    }

    /// Transcription of the accumulation loop this change replaced, kept as the
    /// oracle for bit-identity. The shared `reference` helper sums in `f64` and
    /// so cannot answer "did we reproduce the *previous kernel* exactly", which
    /// is the property that matters once the `i32` accumulator can wrap.
    ///
    /// Mirrors the removed code: per-`(row, column)` accumulation over `k`,
    /// both zero points subtracted inside the loop, `wrapping_add`, then the
    /// unchanged scale / `round_ties_even` / clamp epilogue.
    #[allow(clippy::too_many_arguments)]
    fn previous_loop_oracle(
        a: &[i32],
        b: &[i32],
        m: usize,
        k: usize,
        n: usize,
        a_scales: &[f32],
        a_zeros: &[i32],
        b_scales: &[f32],
        b_zeros: &[i32],
        y_scale: f32,
        y_zero: i32,
        output_dtype: DataType,
    ) -> Vec<i64> {
        let pick = |values: &[f32], index: usize| values[if values.len() > 1 { index } else { 0 }];
        let pick_zero =
            |values: &[i32], index: usize| values[if values.len() > 1 { index } else { 0 }];
        let mut out = Vec::with_capacity(m * n);
        for row in 0..m {
            for column in 0..n {
                let a_scale = pick(a_scales, row);
                let a_zero_point = pick_zero(a_zeros, row);
                let b_scale = pick(b_scales, column);
                let b_zero_point = pick_zero(b_zeros, column);
                let mut accumulated = 0i32;
                for inner in 0..k {
                    let av = a[row * k + inner] - a_zero_point;
                    let bv = b[inner * n + column] - b_zero_point;
                    accumulated = accumulated.wrapping_add(av * bv);
                }
                let scale = a_scale * b_scale / y_scale;
                let value =
                    (accumulated as f32 * scale).round_ties_even() as i64 + i64::from(y_zero);
                out.push(match output_dtype {
                    DataType::Int8 => value.clamp(i8::MIN as i64, i8::MAX as i64),
                    DataType::Uint8 => value.clamp(0, u8::MAX as i64),
                    _ => unreachable!(),
                });
            }
        }
        out
    }

    /// The reordered accumulation must be **bit**-identical to the loop it
    /// replaces, not merely close: `QLinearMatMul` output is integer, so a
    /// one-LSB difference is a wrong answer.
    ///
    /// The rewrite lifts the zero points out of the inner loop using
    ///   sum_k (a_k - az) * (b_kn - bz_n)
    ///     = sum_k (a_k - az) * b_kn  -  bz_n * sum_k (a_k - az)
    /// which is an identity over the integers, so under wrapping arithmetic
    /// (arithmetic mod 2^32) both sides reduce to the same `i32`.
    ///
    /// Compared against a transcription of the previous loop over shapes that
    /// miss every tile boundary, both dtypes, and per-tensor as well as
    /// per-axis (per-row `A`, per-column `B`) quantization. Overflow of the
    /// accumulator is covered separately by
    /// `qlinear_matmul_overflowing_accumulator_matches_the_previous_loop`,
    /// which needs a much larger `K` than is practical to sweep here.
    #[test]
    fn qlinear_matmul_reordered_accumulation_is_bit_identical() {
        for &(m, k, n) in &[
            (1usize, 1usize, 1usize),
            (1, 37, 65),
            (3, 64, 16),
            (5, 129, 33),
        ] {
            for per_axis in [false, true] {
                // --- Uint8 ---
                let a_u8: Vec<u8> = (0..m * k).map(|i| ((i * 37 + 11) % 256) as u8).collect();
                let b_u8: Vec<u8> = (0..k * n).map(|i| ((i * 91 + 7) % 256) as u8).collect();
                let (a_scales, a_zeros): (Vec<f32>, Vec<i32>) = if per_axis {
                    (
                        (0..m).map(|i| 0.02 + i as f32 * 0.003).collect(),
                        (0..m).map(|i| ((i * 13) % 256) as i32).collect(),
                    )
                } else {
                    (vec![0.02], vec![128])
                };
                let (b_scales, b_zeros): (Vec<f32>, Vec<i32>) = if per_axis {
                    (
                        (0..n).map(|i| 0.01 + i as f32 * 0.002).collect(),
                        (0..n).map(|i| ((i * 29 + 3) % 256) as i32).collect(),
                    )
                } else {
                    (vec![0.01], vec![127])
                };
                let axis_shape = |len: usize| if len > 1 { vec![len] } else { Vec::new() };
                let output = execute(
                    [
                        &Owned::u8(&[m, k], &a_u8),
                        &Owned::f32(&axis_shape(a_scales.len()), &a_scales),
                        &Owned::u8(
                            &axis_shape(a_zeros.len()),
                            &a_zeros.iter().map(|&z| z as u8).collect::<Vec<_>>(),
                        ),
                        &Owned::u8(&[k, n], &b_u8),
                        &Owned::f32(&axis_shape(b_scales.len()), &b_scales),
                        &Owned::u8(
                            &axis_shape(b_zeros.len()),
                            &b_zeros.iter().map(|&z| z as u8).collect::<Vec<_>>(),
                        ),
                        &Owned::f32(&[], &[0.05]),
                        &Owned::u8(&[], &[120]),
                    ],
                    DataType::Uint8,
                    &[m, n],
                );
                let a_i32: Vec<i32> = a_u8.iter().map(|&v| i32::from(v)).collect();
                let b_i32: Vec<i32> = b_u8.iter().map(|&v| i32::from(v)).collect();
                assert_eq!(
                    output_values(&output),
                    previous_loop_oracle(
                        &a_i32,
                        &b_i32,
                        m,
                        k,
                        n,
                        &a_scales,
                        &a_zeros,
                        &b_scales,
                        &b_zeros,
                        0.05,
                        120,
                        DataType::Uint8
                    ),
                    "u8 m={m} k={k} n={n} per_axis={per_axis}"
                );

                // --- Int8 ---
                let a_i8: Vec<i8> = (0..m * k)
                    .map(|i| (((i * 37 + 11) % 256) as i32 - 128) as i8)
                    .collect();
                let b_i8: Vec<i8> = (0..k * n)
                    .map(|i| (((i * 91 + 7) % 256) as i32 - 128) as i8)
                    .collect();
                let (a_scales_i, a_zeros_i): (Vec<f32>, Vec<i32>) = if per_axis {
                    (
                        (0..m).map(|i| 0.02 + i as f32 * 0.003).collect(),
                        (0..m).map(|i| ((i * 13) % 256) as i32 - 128).collect(),
                    )
                } else {
                    (vec![0.02], vec![-5])
                };
                let (b_scales_i, b_zeros_i): (Vec<f32>, Vec<i32>) = if per_axis {
                    (
                        (0..n).map(|i| 0.01 + i as f32 * 0.002).collect(),
                        (0..n).map(|i| ((i * 29 + 3) % 256) as i32 - 128).collect(),
                    )
                } else {
                    (vec![0.01], vec![7])
                };
                let output = execute(
                    [
                        &i8(&[m, k], &a_i8),
                        &Owned::f32(&axis_shape(a_scales_i.len()), &a_scales_i),
                        &i8(
                            &axis_shape(a_zeros_i.len()),
                            &a_zeros_i.iter().map(|&z| z as i8).collect::<Vec<_>>(),
                        ),
                        &i8(&[k, n], &b_i8),
                        &Owned::f32(&axis_shape(b_scales_i.len()), &b_scales_i),
                        &i8(
                            &axis_shape(b_zeros_i.len()),
                            &b_zeros_i.iter().map(|&z| z as i8).collect::<Vec<_>>(),
                        ),
                        &Owned::f32(&[], &[0.05]),
                        &i8(&[], &[-3]),
                    ],
                    DataType::Int8,
                    &[m, n],
                );
                let a_i32: Vec<i32> = a_i8.iter().map(|&v| i32::from(v)).collect();
                let b_i32: Vec<i32> = b_i8.iter().map(|&v| i32::from(v)).collect();
                assert_eq!(
                    output_values(&output),
                    previous_loop_oracle(
                        &a_i32,
                        &b_i32,
                        m,
                        k,
                        n,
                        &a_scales_i,
                        &a_zeros_i,
                        &b_scales_i,
                        &b_zeros_i,
                        0.05,
                        -3,
                        DataType::Int8
                    ),
                    "i8 m={m} k={k} n={n} per_axis={per_axis}"
                );
            }
        }
    }

    /// `N == 0` produces an empty output. `par_chunks_mut(0)` and
    /// `chunks_exact(0)` both panic, so the kernel has to skip the batch
    /// outright -- the previous `for column in 0..0` loop degenerated
    /// harmlessly and this must keep doing so.
    #[test]
    fn qlinear_matmul_zero_width_output_is_empty_not_a_panic() {
        let out = execute(
            [
                &Owned::u8(&[2, 3], &[1, 2, 3, 4, 5, 6]),
                &Owned::f32(&[], &[0.5]),
                &Owned::u8(&[], &[3]),
                &Owned::u8(&[3, 0], &[]),
                &Owned::f32(&[], &[0.25]),
                &Owned::u8(&[], &[2]),
                &Owned::f32(&[], &[0.125]),
                &Owned::u8(&[], &[10]),
            ],
            DataType::Uint8,
            &[2, 0],
        );
        assert!(output_values(&out).is_empty());
    }

    /// `K` large enough that the `i32` accumulator wraps.
    ///
    /// The removed loop accumulated with `wrapping_add`, so on overflow its
    /// answer is defined but not the mathematical dot product -- the shared
    /// `reference` helper here sums in `f64` and legitimately disagrees. What
    /// must hold is that the rewrite reproduces the *old kernel* exactly,
    /// because the zero-point identity it relies on is only valid modulo 2^32.
    /// So this compares against a transcription of the old inner loop.
    #[test]
    fn qlinear_matmul_overflowing_accumulator_matches_the_previous_loop() {
        let (m, k, n) = (2usize, 40_000usize, 3usize);
        let a_values: Vec<u8> = (0..m * k)
            .map(|i| if i % 3 == 0 { 255 } else { 254 })
            .collect();
        let b_values: Vec<u8> = (0..k * n)
            .map(|i| if i % 5 == 0 { 255 } else { 253 })
            .collect();
        let (a_zero, b_zero) = (17i32, 250i32);
        let (a_scale, b_scale, y_scale, y_zero) = (1.0f32, 1.0f32, 1.0e6f32, 40i32);

        let output = execute(
            [
                &Owned::u8(&[m, k], &a_values),
                &Owned::f32(&[], &[a_scale]),
                &Owned::u8(&[], &[a_zero as u8]),
                &Owned::u8(&[k, n], &b_values),
                &Owned::f32(&[], &[b_scale]),
                &Owned::u8(&[], &[b_zero as u8]),
                &Owned::f32(&[], &[y_scale]),
                &Owned::u8(&[], &[y_zero as u8]),
            ],
            DataType::Uint8,
            &[m, n],
        );

        // The zero-point correction term this rewrite introduces is
        // `a_sum * b_zero_point`, and the test is only meaningful if that term
        // actually leaves `i32`.
        let a_sum: i64 = (0..k)
            .map(|inner| i64::from(a_values[inner]) - i64::from(a_zero))
            .sum();
        assert!(
            a_sum * i64::from(b_zero) > i64::from(i32::MAX),
            "test data no longer overflows the correction term ({})",
            a_sum * i64::from(b_zero)
        );

        // Transcription of the loop this change replaced.
        let mut expected = Vec::with_capacity(m * n);
        for row in 0..m {
            for column in 0..n {
                let mut accumulated = 0i32;
                for inner in 0..k {
                    let av = i32::from(a_values[row * k + inner]) - a_zero;
                    let bv = i32::from(b_values[inner * n + column]) - b_zero;
                    accumulated = accumulated.wrapping_add(av * bv);
                }
                let scale = a_scale * b_scale / y_scale;
                let value =
                    (accumulated as f32 * scale).round_ties_even() as i64 + i64::from(y_zero);
                expected.push(value.clamp(0, i64::from(u8::MAX)));
            }
        }
        assert_eq!(
            output_values(&output),
            expected,
            "the zero-point identity did not survive i32 overflow"
        );
    }

    /// Row parallelism must not make the result depend on the thread count:
    /// each output row is accumulated by exactly one worker, so repeated runs
    /// -- and runs under a narrower pool -- have to agree exactly.
    #[test]
    fn qlinear_matmul_is_deterministic_across_repeated_runs() {
        let (m, k, n) = (9usize, 71usize, 23usize);
        let a_values: Vec<u8> = (0..m * k).map(|i| ((i * 53 + 3) % 256) as u8).collect();
        let b_values: Vec<u8> = (0..k * n).map(|i| ((i * 17 + 29) % 256) as u8).collect();
        let a = Owned::u8(&[m, k], &a_values);
        let b = Owned::u8(&[k, n], &b_values);
        let a_scale = Owned::f32(&[], &[0.03]);
        let a_zero = Owned::u8(&[], &[130]);
        let b_scale = Owned::f32(&[], &[0.007]);
        let b_zero = Owned::u8(&[], &[110]);
        let y_scale = Owned::f32(&[], &[0.04]);
        let y_zero = Owned::u8(&[], &[100]);
        let inputs = [
            &a, &a_scale, &a_zero, &b, &b_scale, &b_zero, &y_scale, &y_zero,
        ];

        let first = output_values(&execute(inputs, DataType::Uint8, &[m, n]));
        for round in 1..4 {
            let again = output_values(&execute(inputs, DataType::Uint8, &[m, n]));
            assert_eq!(first, again, "round {round} disagreed with the first run");
        }

        // Vary the pool width so the row partition genuinely changes. Row
        // accumulation is sequential and integer, so the answer must not depend
        // on how many workers split the rows -- including the serial path the
        // single-thread pool takes.
        for threads in [1usize, 2, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let under_pool =
                pool.install(|| output_values(&execute(inputs, DataType::Uint8, &[m, n])));
            assert_eq!(
                first, under_pool,
                "a {threads}-thread pool produced a different result"
            );
        }
    }
}
