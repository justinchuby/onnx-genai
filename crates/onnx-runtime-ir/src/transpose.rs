//! Native storage contract for ONNX Transpose.

use crate::DataType;

/// Fixed-width dtypes from Transpose-25. Variable-width strings cannot use
/// either native raw-buffer kernel.
pub const DTYPES: &[DataType] = &[
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
    DataType::Uint4,
    DataType::Int4,
    DataType::Float4E2M1,
    DataType::Float8E8M0,
    DataType::Uint2,
    DataType::Int2,
];

/// Packed elements that occupy whole lanes within a byte, never crossing a
/// byte boundary. Widths that need a different layout (e.g. FLOAT6) are not
/// supported by the native packed transpose.
pub fn packed_lane_bits(dtype: DataType) -> Option<usize> {
    let bits = dtype.bit_size();
    (bits > 0 && bits < 8 && 8 % bits == 0).then_some(bits)
}

/// Validate the input type for native placement (the output has the same type).
pub fn validate_dtype(dtype: DataType, opset: u64) -> Result<(), String> {
    if !DTYPES.contains(&dtype) {
        return Err(format!(
            "Transpose does not support native storage for {dtype:?}; use a supported fixed-width dtype"
        ));
    }
    let since = match dtype {
        DataType::Int4 | DataType::Uint4 => 21,
        DataType::Float4E2M1 => 23,
        DataType::Float8E8M0 => 24,
        DataType::Int2 | DataType::Uint2 => 25,
        _ => return Ok(()),
    };
    if opset < since {
        return Err(format!(
            "Transpose at opset {opset} does not admit {dtype:?}; use opset {since} or later"
        ));
    }
    Ok(())
}
