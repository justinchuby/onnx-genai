//! Shared ONNX Det contract for native providers.

use crate::{EpError, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Dim, Node, Shape};

pub const DTYPES: &[DataType] = &[
    DataType::Float16,
    DataType::Float32,
    DataType::Float64,
    DataType::BFloat16,
];

fn dtype_contract(dtype: DataType, opset: i64) -> std::result::Result<(), String> {
    if dtype == DataType::BFloat16 && opset < 22 {
        Err(format!(
            "Det: BFloat16 requires opset 22 or later (got {opset}); re-export or cast to Float32"
        ))
    } else if !DTYPES.contains(&dtype) {
        Err(format!(
            "Det: input type {dtype:?} is not supported; use Float16, Float32, Float64 or (opset 22+) BFloat16"
        ))
    } else {
        Ok(())
    }
}

pub fn unsupported_reason(
    node: &Node,
    opset: u64,
    shapes: &[Shape],
    dtypes: &[DataType],
) -> Option<String> {
    if node.op_type != "Det" || !matches!(node.domain.as_str(), "" | "ai.onnx") {
        return None;
    }
    let check = || {
        if !matches!(node.inputs.as_slice(), [Some(_)]) || node.outputs.len() != 1 {
            return Err("Det: expected one present input and one output".into());
        }
        if let Some(&dtype) = dtypes.first()
            && dtype != DataType::Undefined
        {
            dtype_contract(
                dtype,
                node.version.unwrap_or(if opset >= 22 { 22 } else { 11 }),
            )?;
        }
        if let Some(shape) = shapes.first() {
            matrix_axes(shape, |d| Dim::as_static(*d))?;
        }
        Ok(())
    };
    check().err()
}

fn matrix_axes<T>(
    shape: &[T],
    extent: impl Fn(&T) -> Option<usize>,
) -> std::result::Result<(&[T], &T), String> {
    let [batch @ .., rows, cols] = shape else {
        return Err(format!("Det: expected rank >= 2, got rank {}", shape.len()));
    };
    if let (Some(rows), Some(cols)) = (extent(rows), extent(cols))
        && rows != cols
    {
        return Err(format!(
            "Det: expected a square matrix on the last two axes, got {rows}x{cols}"
        ));
    }
    Ok((batch, cols))
}

pub struct Geometry {
    pub batch: usize,
    pub order: usize,
    pub matrix_elements: usize,
}

impl Geometry {
    pub fn from_shape(shape: &[usize]) -> Result<Self> {
        Self::from_axes(matrix_axes(shape, |&d| Some(d)).map_err(EpError::KernelFailed)?)
    }

    fn from_axes((prefix, &order): (&[usize], &usize)) -> Result<Self> {
        let overflow = || EpError::KernelFailed("Det: tensor element count overflows usize".into());
        let batch = if prefix.contains(&0) {
            0
        } else {
            prefix
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or_else(overflow)?
        };
        let matrix_elements = if batch == 0 {
            0
        } else {
            order.checked_mul(order).ok_or_else(overflow)?
        };
        batch.checked_mul(matrix_elements).ok_or_else(overflow)?;
        Ok(Self {
            batch,
            order,
            matrix_elements,
        })
    }
}

pub fn validate(inputs: &[TensorView], outputs: &[TensorMut], opset: i64) -> Result<Geometry> {
    let ([x], [y]) = (inputs, outputs) else {
        return Err(EpError::KernelFailed(
            "Det: expected one input and one output".into(),
        ));
    };
    dtype_contract(x.dtype, opset).map_err(EpError::KernelFailed)?;
    let axes = matrix_axes(x.shape, |&d| Some(d)).map_err(EpError::KernelFailed)?;
    if x.dtype != y.dtype || y.shape != axes.0 {
        return Err(EpError::KernelFailed(format!(
            "Det: expected output type {:?} and batch shape {:?}, got {:?} {:?}",
            x.dtype, axes.0, y.dtype, y.shape
        )));
    }
    Geometry::from_axes(axes)
}
