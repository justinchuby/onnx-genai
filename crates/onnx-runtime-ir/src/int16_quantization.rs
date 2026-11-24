//! Shared ONNX contract for fixed-width 16-bit integer linear quantization.

use crate::{DataType, Node};

pub fn is_code(dtype: DataType) -> bool {
    matches!(dtype, DataType::Int16 | DataType::Uint16)
}

fn is_float(dtype: DataType) -> bool {
    matches!(
        dtype,
        DataType::Float32 | DataType::Float16 | DataType::BFloat16
    )
}

pub fn attribute_dtype(node: &Node, name: &str) -> Result<Option<DataType>, String> {
    match node.attr(name).and_then(|attr| attr.as_int()) {
        None | Some(0) => Ok(None),
        Some(raw) => i32::try_from(raw)
            .ok()
            .and_then(DataType::from_onnx)
            .map(Some)
            .ok_or_else(|| format!("{name}={raw} is not an ONNX dtype")),
    }
}

pub fn validate_opset(node: &Node, opset: u64) -> Result<(), String> {
    if opset < 21 {
        return Err(format!(
            "INT16 and UINT16 require opset 21 or later (got {opset})"
        ));
    }
    let newer_attribute = if node.op_type == "QuantizeLinear" {
        "precision"
    } else {
        "output_dtype"
    };
    if opset < 23 && node.attr(newer_attribute).is_some() {
        return Err(format!(
            "{newer_attribute} requires opset 23 or later (got {opset})"
        ));
    }
    Ok(())
}

pub fn quantize_precision(
    explicit: Option<DataType>,
    x: DataType,
    scale: DataType,
) -> Result<DataType, String> {
    if !(is_float(x) || x == DataType::Int32) || !(is_float(scale) || scale == DataType::Int32) {
        return Err(format!(
            "INT16/UINT16 quantization requires FLOAT, FLOAT16, BFLOAT16 or INT32 x and scale; got {x:?} and {scale:?}"
        ));
    }
    let precision = explicit.unwrap_or(if scale == DataType::Int32 {
        DataType::Float64
    } else {
        scale
    });
    if !is_float(precision) && precision != DataType::Float64 {
        return Err(format!(
            "precision must select FLOAT, FLOAT16, BFLOAT16 or DOUBLE division; got {precision:?}"
        ));
    }
    Ok(precision)
}

pub fn dequantize_output(node: &Node, scale: DataType) -> Result<DataType, String> {
    let output = attribute_dtype(node, "output_dtype")?.unwrap_or(scale);
    if !is_float(scale) || !is_float(output) {
        return Err(format!(
            "INT16/UINT16 dequantization requires FLOAT, FLOAT16 or BFLOAT16 scale and output; got {scale:?} and {output:?}"
        ));
    }
    Ok(output)
}
