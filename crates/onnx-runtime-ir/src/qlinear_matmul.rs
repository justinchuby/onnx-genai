//! Shared scale type constraint for integer `QLinearMatMul` kernels.

use crate::DataType;

/// First ONNX schema with half scales; not the standalone capability floor.
pub const HALF_SCALE_SINCE_VERSION: u64 = 21;

/// Scale storage implemented by the provider for the integer `QLinearMatMul` path.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ScaleSupport {
    FloatOnly,
    FloatAndHalf,
}

impl ScaleSupport {
    pub fn for_standalone(standalone: bool) -> Self {
        if standalone {
            Self::FloatAndHalf
        } else {
            Self::FloatOnly
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScaleType {
    Float32,
    Float16,
    BFloat16,
}

impl ScaleType {
    /// Resolve `TS` once: all three scales must use the same storage type.
    pub fn resolve(
        support: ScaleSupport,
        a_scale: DataType,
        b_scale: DataType,
        y_scale: DataType,
    ) -> Result<Self, String> {
        let scale_type = match a_scale {
            DataType::Float32 => Self::Float32,
            DataType::Float16 if support == ScaleSupport::FloatAndHalf => Self::Float16,
            DataType::BFloat16 if support == ScaleSupport::FloatAndHalf => Self::BFloat16,
            _ => {
                let expected = if support == ScaleSupport::FloatAndHalf {
                    "Float32, Float16, or BFloat16"
                } else {
                    "Float32 (half scales are only supported by standalone nxrt)"
                };
                return Err(format!(
                    "requires a_scale to be {expected}, got {a_scale:?}"
                ));
            }
        };
        for (name, dtype) in [("b_scale", b_scale), ("y_scale", y_scale)] {
            if dtype != a_scale {
                return Err(format!(
                    "requires {name} dtype to match a_scale ({a_scale:?}), got {dtype:?}"
                ));
            }
        }
        Ok(scale_type)
    }
}
