//! `QLinearMatMul`: integer matrix multiplication with linear quantization.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node, broadcast_shapes, compute_contiguous_strides};

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
    _input_shapes: &[onnx_runtime_ir::Shape],
) -> Option<String> {
    if input_dtypes.is_empty() {
        return None;
    }
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
    None
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
        let a_scale = scalar_scale("a_scale", &inputs[1])?;
        let b_scale = b_scales(&inputs[4], geometry.n)?;
        let y_scale = scalar_scale("y_scale", &inputs[6])?;
        let a_zero_point = scalar_integer("a_zero_point", &inputs[2])?;
        let b_zero_point = b_zero_points(&inputs[5], geometry.n)?;
        let y_zero_point = scalar_integer("y_zero_point", &inputs[7])?;

        let a = read_quantized(&inputs[0])?;
        let b = read_quantized(&inputs[3])?;
        let mut output = Vec::with_capacity(geometry.result_len);
        let mut batch_index = vec![0; geometry.batch_shape.len()];
        for batch in 0..geometry.batch_count {
            let a_offset = geometry.a_offset(&batch_index);
            let b_offset = geometry.b_offset(&batch_index);
            for row in 0..geometry.m {
                for column in 0..geometry.n {
                    let mut accumulated = 0i64;
                    for inner in 0..geometry.k {
                        let av = a[a_offset + row * geometry.k + inner] - a_zero_point;
                        let bv = b[b_offset + inner * geometry.n + column]
                            - b_zero_point[if b_zero_point.len() == 1 { 0 } else { column }];
                        accumulated += av * bv;
                    }
                    let scale =
                        a_scale * b_scale[if b_scale.len() == 1 { 0 } else { column }] / y_scale;
                    let value =
                        (accumulated as f32 * scale).round_ties_even() as i64 + y_zero_point;
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

fn scalar_scale(name: &str, view: &TensorView) -> Result<f32> {
    if view.numel() != 1 {
        return Err(EpError::KernelFailed(format!(
            "QLinearMatMul: {name} must be a scalar"
        )));
    }
    let value = f32::from_le_bytes(to_dense_bytes(view)?[..4].try_into().unwrap());
    if value <= 0.0 || !value.is_finite() {
        return Err(EpError::KernelFailed(format!(
            "QLinearMatMul: {name} must be finite and positive"
        )));
    }
    Ok(value)
}

fn b_scales(view: &TensorView, n: usize) -> Result<Vec<f32>> {
    if view.shape.len() > 1 || !(view.numel() == 1 || view.numel() == n) {
        return Err(EpError::KernelFailed(format!(
            "QLinearMatMul: b_scale must be scalar or a 1-D tensor of length {n}"
        )));
    }
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
            "QLinearMatMul: b_scale values must be finite and positive".into(),
        ));
    }
    Ok(scales)
}

fn scalar_integer(name: &str, view: &TensorView) -> Result<i64> {
    if view.numel() != 1 {
        return Err(EpError::KernelFailed(format!(
            "QLinearMatMul: {name} must be a scalar"
        )));
    }
    Ok(read_quantized(view)?[0])
}

fn b_zero_points(view: &TensorView, n: usize) -> Result<Vec<i64>> {
    if view.shape.len() > 1 || !(view.numel() == 1 || view.numel() == n) {
        return Err(EpError::KernelFailed(format!(
            "QLinearMatMul: b_zero_point must be scalar or a 1-D tensor of length {n}"
        )));
    }
    read_quantized(view)
}

fn read_quantized(view: &TensorView) -> Result<Vec<i64>> {
    let bytes = to_dense_bytes(view)?;
    match view.dtype {
        DataType::Int8 => Ok(bytes.into_iter().map(|value| value as i8 as i64).collect()),
        DataType::Uint8 => Ok(bytes.into_iter().map(i64::from).collect()),
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

    fn a_offset(&self, batch_index: &[usize]) -> usize {
        broadcast_offset(batch_index, &self.a_batch, &self.a_batch_strides) * self.m * self.k
    }

    fn b_offset(&self, batch_index: &[usize]) -> usize {
        broadcast_offset(batch_index, &self.b_batch, &self.b_batch_strides) * self.k * self.n
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

    fn reference(
        a: &[i64],
        b: &[i64],
        a_scale: f32,
        b_scales: &[f32],
        y_scale: f32,
        a_zero: i64,
        b_zeros: &[i64],
        y_zero: i64,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<i64> {
        (0..m)
            .flat_map(|row| {
                (0..n).map(move |column| {
                    let sum: i64 = (0..k)
                        .map(|inner| {
                            (a[row * k + inner] - a_zero)
                                * (b[inner * n + column]
                                    - b_zeros[if b_zeros.len() == 1 { 0 } else { column }])
                        })
                        .sum();
                    (sum as f32 * a_scale * b_scales[if b_scales.len() == 1 { 0 } else { column }]
                        / y_scale)
                        .round_ties_even() as i64
                        + y_zero
                })
            })
            .collect()
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
        let mut out = Owned::zeros(DataType::Uint8, &[2, 2]);
        QLinearMatMulKernel
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
            .unwrap();
        let expected = reference(
            &[130, 125, 140, 120, 135, 128],
            &[131, 126, 120, 140, 128, 130],
            0.25,
            &[0.5],
            0.125,
            128,
            &[128],
            127,
            2,
            3,
            2,
        );
        assert_eq!(
            out.to_u8(),
            expected
                .into_iter()
                .map(|value| value as u8)
                .collect::<Vec<_>>()
        );
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
        let mut out = Owned::zeros(DataType::Int8, &[1, 3]);
        QLinearMatMulKernel
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
            .unwrap();
        let expected = reference(
            &[-2, 5],
            &[3, -4, 7, 2, 5, -6],
            0.25,
            &[0.5, 0.25, 0.125],
            0.25,
            -1,
            &[1, -2, 3],
            2,
            1,
            2,
            3,
        );
        assert_eq!(
            out.bytes
                .iter()
                .map(|&value| value as i8 as i64)
                .collect::<Vec<_>>(),
            expected
        );
    }
}
