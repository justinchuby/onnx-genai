//! Contracts shared by the native implementations of the opset 28/29 math operators.

use crate::{Attribute, DataType, Node};

/// Standard-domain name and introduction version.
pub const OPS: [(&str, u64); 5] = [
    ("SwiGLU", 28),
    ("GeGLU", 29),
    ("Expm1", 29),
    ("Log1p", 29),
    ("LogCumSumExp", 29),
];

/// The complete ONNX type constraint for these operators.
pub const DTYPES: &[DataType] = &[
    DataType::Float16,
    DataType::BFloat16,
    DataType::Float32,
    DataType::Float64,
];

/// Validated attributes for a standard math operator.
#[derive(Clone, Copy, Debug)]
pub enum Math {
    SwiGlu { alpha: f32 },
    GeGlu { tanh: bool },
    Expm1,
    Log1p,
    LogCumSumExp { axis: i64 },
}

impl Math {
    pub fn from_node(node: &Node) -> Result<Self, String> {
        let op = match node.op_type.as_str() {
            "SwiGLU" => Self::SwiGlu {
                alpha: match node.attr("alpha") {
                    None => 1.,
                    Some(Attribute::Float(value)) => *value,
                    _ => return Err("SwiGLU: alpha must be a FLOAT attribute".into()),
                },
            },
            "GeGLU" => Self::GeGlu {
                tanh: match node.attr("approximate") {
                    None => false,
                    Some(Attribute::String(s)) if s == b"none" => false,
                    Some(Attribute::String(s)) if s == b"tanh" => true,
                    _ => return Err("GeGLU: approximate must be 'none' or 'tanh'".into()),
                },
            },
            "Expm1" => Self::Expm1,
            "Log1p" => Self::Log1p,
            "LogCumSumExp" => Self::LogCumSumExp {
                axis: match node.attr("axis") {
                    None => 0,
                    Some(Attribute::Int(value)) => *value,
                    _ => return Err("LogCumSumExp: axis must be an INT attribute".into()),
                },
            },
            name => return Err(format!("{name}: not an opset 28/29 math operator")),
        };
        Ok(op)
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::SwiGlu { .. } => "SwiGLU",
            Self::GeGlu { .. } => "GeGLU",
            Self::Expm1 => "Expm1",
            Self::Log1p => "Log1p",
            Self::LogCumSumExp { .. } => "LogCumSumExp",
        }
    }

    pub fn input_count(self) -> usize {
        match self {
            Self::SwiGlu { .. } | Self::GeGlu { .. } => 2,
            _ => 1,
        }
    }

    pub fn axis(self, rank: usize) -> Result<Option<usize>, String> {
        let Self::LogCumSumExp { axis } = self else {
            return Ok(None);
        };
        let r = i64::try_from(rank).map_err(|_| format!("LogCumSumExp: invalid rank {rank}"))?;
        if !(-r..r).contains(&axis) {
            return Err(format!(
                "LogCumSumExp: axis {axis} is out of range for rank {rank}; use a rank >= 1 input and an axis in [-rank, rank)"
            ));
        }
        Ok(Some(if axis < 0 {
            (r + axis) as usize
        } else {
            axis as usize
        }))
    }
}
