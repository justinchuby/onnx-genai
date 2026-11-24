//! Shared semantic checks for the standard ONNX opset-26 operators.

use crate::{DataType, Node};

pub const BIT_CAST_TYPES: &[DataType] = &[
    DataType::Uint8,
    DataType::Uint16,
    DataType::Uint32,
    DataType::Uint64,
    DataType::Int8,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
    DataType::BFloat16,
    DataType::Float16,
    DataType::Float32,
    DataType::Float64,
    DataType::Bool,
    DataType::Complex64,
    DataType::Complex128,
    DataType::Float8E4M3FN,
    DataType::Float8E4M3FNUZ,
    DataType::Float8E5M2,
    DataType::Float8E5M2FNUZ,
    DataType::Float8E8M0,
    DataType::Uint4,
    DataType::Int4,
    DataType::Float4E2M1,
    DataType::Uint2,
    DataType::Int2,
];

pub const CUM_PROD_TYPES: &[DataType] = &[
    DataType::Uint32,
    DataType::Uint64,
    DataType::Int32,
    DataType::Int64,
    DataType::Float16,
    DataType::Float32,
    DataType::Float64,
    DataType::BFloat16,
];

pub const CUM_PROD_AXIS_TYPES: &[DataType] = &[DataType::Int32, DataType::Int64];
pub const CUM_PROD_INPUT_TYPES: &[(usize, &[DataType])] = &[(1, CUM_PROD_AXIS_TYPES)];

pub fn bit_cast_target(node: &Node) -> Result<DataType, String> {
    let raw = node.attr("to").and_then(|a| a.as_int()).ok_or_else(|| {
        "BitCast: supply the required integer `to` attribute (ONNX tensor dtype)".to_string()
    })?;
    i32::try_from(raw)
        .ok()
        .and_then(DataType::from_onnx)
        .filter(|t| BIT_CAST_TYPES.contains(t))
        .ok_or_else(|| format!("BitCast: `to`={raw} is not a dtype in the BitCast-26 schema"))
}

pub fn validate_bit_cast(input: DataType, target: DataType) -> Result<(), String> {
    if !BIT_CAST_TYPES.contains(&input) || !BIT_CAST_TYPES.contains(&target) {
        return Err(format!(
            "BitCast: {input:?} -> {target:?} requires two fixed-width BitCast-26 dtypes"
        ));
    }
    if input.bit_size() != target.bit_size() {
        return Err(format!(
            "BitCast: {input:?} is {} bits but {target:?} is {} bits; choose an equal-width `to` dtype",
            input.bit_size(),
            target.bit_size(),
        ));
    }
    Ok(())
}

pub fn cum_prod_flags(node: &Node) -> Result<(bool, bool), String> {
    let flag = |name| match node.attr(name) {
        None | Some(crate::Attribute::Int(0)) => Ok(false),
        Some(crate::Attribute::Int(1)) => Ok(true),
        Some(value) => Err(format!(
            "CumProd: `{name}` must be integer 0 or 1, got {value:?}"
        )),
    };
    Ok((flag("exclusive")?, flag("reverse")?))
}

pub fn cum_prod_axis(raw: i64, rank: usize) -> Result<usize, String> {
    let axis = if raw < 0 { raw + rank as i64 } else { raw };
    if axis >= 0 && (axis as u64) < rank as u64 {
        Ok(axis as usize)
    } else {
        Err(format!(
            "CumProd: axis {raw} is out of range for input rank {rank}; use [-rank, rank-1]"
        ))
    }
}
