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

pub fn validate_revision(node: &Node, opset: u64, scale_rank: Option<usize>) -> Result<(), String> {
    // Keep revision-specific forms and attributes, not the schema's storage-type floor.
    let quantize = node.op_type == "QuantizeLinear";
    for (name, since) in [
        ("axis", 13),
        ("block_size", 21),
        ("output_dtype", if quantize { 21 } else { 23 }),
        ("precision", 23),
        ("saturate", 19),
    ] {
        if !quantize && matches!(name, "precision" | "saturate") {
            continue;
        }
        if opset < since && node.attr(name).is_some() {
            return Err(format!(
                "{name} requires opset {since} or later (got {opset})"
            ));
        }
    }
    if opset < 13 && scale_rank.is_some_and(|rank| rank != 0) {
        return Err(format!(
            "opset {opset} requires a scalar scale; per-axis quantization requires opset 13 or later"
        ));
    }
    Ok(())
}

pub fn validate_quantize_types(opset: u64, x: DataType, scale: DataType) -> Result<(), String> {
    if opset < 19 {
        if !matches!(x, DataType::Float32 | DataType::Int32) || scale != DataType::Float32 {
            return Err(format!(
                "opset {opset} requires FLOAT or INT32 x and FLOAT scale; got {x:?} and {scale:?}"
            ));
        }
    } else if opset < 23 && x != scale {
        return Err(format!(
            "opset {opset} requires x and scale to share a dtype; got {x:?} and {scale:?}"
        ));
    }
    Ok(())
}

pub fn validate_dequantize_scale(opset: u64, scale: DataType) -> Result<(), String> {
    if opset < 19 && scale != DataType::Float32 {
        return Err(format!(
            "opset {opset} requires FLOAT scale and output; got scale {scale:?}"
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
