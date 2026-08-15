//! `QLinearMatMul`: integer matrix multiplication with linear quantization.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Dim, Node, Shape, broadcast_shapes, compute_contiguous_strides};

use super::{check_arity, to_dense_bytes, write_dense_bytes};
use crate::strided::numel;

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
        let mut output = Vec::with_capacity(geometry.result_len);
        let mut batch_index = vec![0; geometry.batch_shape.len()];
        for batch in 0..geometry.batch_count {
            let a_batch = geometry.a_batch_offset(&batch_index);
            let b_batch = geometry.b_batch_offset(&batch_index);
            let a_offset = a_batch * geometry.m * geometry.k;
            let b_offset = b_batch * geometry.k * geometry.n;
            for row in 0..geometry.m {
                for column in 0..geometry.n {
                    let (a_scale, a_zero_point) = a_quant.at(a_batch, row);
                    let (b_scale, b_zero_point) = b_quant.at(b_batch, column);
                    let mut accumulated = 0i32;
                    for inner in 0..geometry.k {
                        let av = a[a_offset + row * geometry.k + inner] - a_zero_point;
                        let bv = b[b_offset + inner * geometry.n + column] - b_zero_point;
                        accumulated = accumulated.wrapping_add(av * bv);
                    }
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
}
