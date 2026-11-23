use crate::{EpError, Result, TensorMut, TensorView};
use onnx_runtime_ir::DataType;

pub struct Layout {
    pub outer: usize,
    pub axis: usize,
    pub inner: usize,
    pub indices: usize,
    pub lanes: usize,
}

pub fn layout(axis: i64, x: &[TensorView], y: &[TensorMut]) -> Result<Layout> {
    let fail = |detail: &str| EpError::KernelFailed(format!("AffineQuantizedGather: {detail}"));
    let (data, indices, scale) = match x {
        [data, indices, scale] | [data, indices, scale, _] => (data, indices, scale),
        _ => {
            return Err(fail(
                "expected data, indices, scale and optional zero point",
            ));
        }
    };
    let [out] = y else {
        return Err(fail("expected one output"));
    };
    if !matches!(data.dtype, DataType::Int8 | DataType::Uint8)
        || !matches!(indices.dtype, DataType::Int32 | DataType::Int64)
        || scale.dtype != DataType::Float32
        || !scale.shape.is_empty()
        || out.dtype != DataType::Float32
        || x.get(3)
            .is_some_and(|z| z.dtype != data.dtype || !z.shape.is_empty())
    {
        return Err(fail(
            "requires INT8/UINT8 data with matching scalar zero point, integer indices, and scalar FLOAT scale",
        ));
    }
    let rank = data.shape.len() as i64;
    if !(-rank..rank).contains(&axis) {
        return Err(fail(&format!(
            "axis {axis} is out of range for data rank {rank}"
        )));
    }
    let a = axis.rem_euclid(rank) as usize;
    let expected: Vec<_> = data.shape[..a]
        .iter()
        .chain(indices.shape)
        .chain(&data.shape[a + 1..])
        .copied()
        .collect();
    if out.shape != expected {
        return Err(fail(&format!(
            "expected output shape {expected:?}, got {:?}",
            out.shape
        )));
    }
    let product = |shape: &[usize]| {
        shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| fail(&format!("tensor shape {shape:?} overflows host indexing")))
    };
    product(data.shape)?;
    Ok(Layout {
        outer: product(&data.shape[..a])?,
        axis: data.shape[a],
        inner: product(&data.shape[a + 1..])?,
        indices: product(indices.shape)?,
        lanes: product(out.shape)?,
    })
}
