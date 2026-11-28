//! Shared ONNX `Mod` mode and dtype rules for execution providers.

use crate::{Attribute, DataType, Node};

pub const FLOAT_FLOOR_SINCE: u64 = 28;
pub const DTYPES: &[DataType] = &[
    DataType::Float16,
    DataType::BFloat16,
    DataType::Float32,
    DataType::Float64,
    DataType::Int8,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
    DataType::Uint8,
    DataType::Uint16,
    DataType::Uint32,
    DataType::Uint64,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModMode {
    Floor,
    Trunc,
}

impl ModMode {
    pub fn parse(node: &Node) -> Result<Self, String> {
        match node.attr("fmod") {
            None | Some(Attribute::Int(0)) => Ok(Self::Floor),
            Some(Attribute::Int(1)) => Ok(Self::Trunc),
            Some(Attribute::Int(value)) => Err(format!("`fmod` must be 0 or 1, got {value}")),
            Some(_) => Err("`fmod` must be an integer attribute".into()),
        }
    }

    pub fn validate_dtype(self, opset: u64, dtype: DataType) -> Result<(), String> {
        if !DTYPES.contains(&dtype) {
            return Err(format!(
                "expected an ONNX numeric dtype for Mod, got {dtype:?}"
            ));
        }
        if self == Self::Floor && dtype.is_float() && opset < FLOAT_FLOOR_SINCE {
            return Err(format!(
                "at opset {opset}, fmod=0 requires an integer dtype, got {dtype:?}; use fmod=1 for floating-point truncation or opset 28+ for floor remainder"
            ));
        }
        Ok(())
    }
}
