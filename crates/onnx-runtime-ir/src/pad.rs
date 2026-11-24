//! Shared ONNX Pad dtype and packed-coordinate contracts for native EPs.

use crate::DataType;

pub const DTYPES: &[DataType] = &[
    DataType::Float32,
    DataType::Float16,
    DataType::BFloat16,
    DataType::Float64,
    DataType::Int8,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
    DataType::Uint8,
    DataType::Uint16,
    DataType::Uint32,
    DataType::Uint64,
    DataType::Bool,
    DataType::Int4,
    DataType::Uint4,
    DataType::Float4E2M1,
    DataType::Float8E8M0,
    DataType::Int2,
    DataType::Uint2,
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Constant,
    Reflect,
    Edge,
    Wrap,
}

/// Packed elements can be moved without conversion, but their raw encoding
/// must be in the Pad schema at the effective opset.
pub fn validate_dtype(dtype: DataType, opset: u64) -> Result<(), String> {
    let since = match dtype {
        DataType::Int4 | DataType::Uint4 => 21,
        DataType::Float4E2M1 => 23,
        DataType::Float8E8M0 => 24,
        DataType::Int2 | DataType::Uint2 => 25,
        other if other.byte_size() != 0 => return Ok(()),
        other => {
            return Err(format!(
                "Pad has no native storage path for {other:?}; use a supported fixed-width dtype"
            ));
        }
    };
    if opset < since {
        return Err(format!(
            "Pad: {dtype:?} requires opset {since} or later, got {opset}; export the model with a newer opset"
        ));
    }
    Ok(())
}

/// The pinned ONNX reference casts numeric zero to the tensor's dtype.
/// E8M0 has no zero; its reference conversion produces the NaN code 0xff.
pub fn default_fill_byte(dtype: DataType) -> u8 {
    if dtype == DataType::Float8E8M0 {
        0xff
    } else {
        0
    }
}

pub struct PackedAxis {
    pub crop_start: usize,
    pub cropped: usize,
    pub pad_begin: i64,
    pub output: usize,
}

/// For indexed modes, the sampling domain is the cropped input, not the
/// original axis. A constant pad of a fully cropped axis has no source lanes.
pub fn packed_axis(
    mode: Mode,
    axis: usize,
    dim: usize,
    begin: i64,
    end: i64,
) -> Result<PackedAxis, String> {
    if i64::try_from(dim).is_err() {
        return Err(format!(
            "Pad: input extent {dim} on axis {axis} exceeds signed coordinate range; reduce the input shape"
        ));
    }
    let output = dim as i128 + i128::from(begin) + i128::from(end);
    let output = usize::try_from(output)
        .ok()
        .filter(|&n| i64::try_from(n).is_ok())
        .ok_or_else(|| format!(
            "Pad: axis {axis} (extent {dim}) with pads ({begin}, {end}) has an invalid output extent; reduce padding/cropping"
        ))?;
    let crop_start = u128::from(begin.min(0).unsigned_abs());
    let crop_end = u128::from(end.min(0).unsigned_abs());
    let cropped = (dim as u128).saturating_sub(crop_start + crop_end) as usize;
    let pad_begin = begin.max(0);
    let pad_end = end.max(0);
    if output != 0 && mode != Mode::Constant && cropped == 0 {
        return Err(format!(
            "Pad: {mode:?} cannot sample axis {axis} after cropping removes every element; reduce the crop or use constant mode"
        ));
    }
    if mode == Mode::Reflect
        && (pad_begin != 0 || pad_end != 0)
        && (cropped < 2
            || pad_begin as u128 >= cropped as u128
            || pad_end as u128 >= cropped as u128
            || 2 * (cropped as u128 - 1) > i64::MAX as u128)
    {
        return Err(format!(
            "Pad: reflect pads at axis {axis} must fit the cropped extent {cropped} (at least two elements, each pad smaller than the extent); reduce padding/cropping"
        ));
    }
    Ok(PackedAxis {
        crop_start: crop_start.min(dim as u128) as usize,
        cropped,
        pad_begin,
        output,
    })
}
