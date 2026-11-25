//! Shared ONNX packed-integer storage and Cast revision rules.

use crate::DataType;

/// Storage metadata for an ONNX packed integer element type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedInteger {
    bits: u8,
    signed: bool,
    cast_since: u64,
}

impl PackedInteger {
    pub const fn bits(self) -> u8 {
        self.bits
    }

    pub const fn lanes_per_byte(self) -> usize {
        8 / self.bits as usize
    }

    pub const fn mask(self) -> u8 {
        (1_u8 << self.bits) - 1
    }

    pub const fn cast_since(self) -> u64 {
        self.cast_since
    }

    /// Decode one low-bit-first storage code, sign-extending signed formats.
    pub const fn decode(self, code: u8) -> i64 {
        let code = code & self.mask();
        if self.signed {
            ((code << (8 - self.bits)) as i8 >> (8 - self.bits)) as i64
        } else {
            code as i64
        }
    }

    /// Encode an integer with ONNX integer-to-integer wrap semantics.
    pub const fn encode_wrapping_i64(self, value: i64) -> u8 {
        value as u8 & self.mask()
    }

    pub const fn encode_wrapping_u64(self, value: u64) -> u8 {
        value as u8 & self.mask()
    }
}

pub const fn packed_integer(dtype: DataType) -> Option<PackedInteger> {
    Some(match dtype {
        DataType::Uint4 => PackedInteger {
            bits: 4,
            signed: false,
            cast_since: 21,
        },
        DataType::Int4 => PackedInteger {
            bits: 4,
            signed: true,
            cast_since: 21,
        },
        DataType::Uint2 => PackedInteger {
            bits: 2,
            signed: false,
            cast_since: 25,
        },
        DataType::Int2 => PackedInteger {
            bits: 2,
            signed: true,
            cast_since: 25,
        },
        _ => return None,
    })
}

/// Fixed-width peer types accepted when either Cast endpoint is packed.
pub const fn packed_cast_peer_supported(dtype: DataType) -> bool {
    matches!(
        dtype,
        DataType::Float32
            | DataType::Float64
            | DataType::Float16
            | DataType::BFloat16
            | DataType::Int8
            | DataType::Uint8
            | DataType::Int16
            | DataType::Uint16
            | DataType::Int32
            | DataType::Uint32
            | DataType::Int64
            | DataType::Uint64
            | DataType::Bool
            | DataType::Uint4
            | DataType::Int4
            | DataType::Uint2
            | DataType::Int2
    )
}

/// Explain why a Cast/CastLike dtype pair is not legal at `opset`.
pub fn cast_unsupported_reason(
    op: &str,
    opset: u64,
    source: Option<DataType>,
    target: Option<DataType>,
) -> Option<String> {
    for (role, dtype) in [("source", source), ("target", target)] {
        let Some(dtype) = dtype else {
            continue;
        };
        let Some(packed) = packed_integer(dtype) else {
            continue;
        };
        if opset < packed.cast_since() {
            return Some(format!(
                "{op}: packed {role} dtype {dtype:?} requires opset {} or newer; \
                 export this node at that opset or cast through an 8-bit integer tensor",
                packed.cast_since()
            ));
        }
    }
    if source.is_some_and(|dtype| packed_integer(dtype).is_some())
        || target.is_some_and(|dtype| packed_integer(dtype).is_some())
    {
        for (role, dtype) in [("source", source), ("target", target)] {
            let Some(dtype) = dtype else {
                continue;
            };
            if !packed_cast_peer_supported(dtype) {
                return Some(format!(
                    "{op}: packed integer conversion does not support {role} dtype {dtype:?}; \
                     use f16/bf16/f32/f64, bool, or an integer dtype from 2 to 64 bits"
                ));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_codes_are_low_bit_first_and_sign_extended() {
        let int4 = packed_integer(DataType::Int4).unwrap();
        assert_eq!(int4.decode(0x7), 7);
        assert_eq!(int4.decode(0x8), -8);
        assert_eq!(int4.decode(0xf), -1);
        assert_eq!(int4.encode_wrapping_i64(-9), 0x7);

        let int2 = packed_integer(DataType::Int2).unwrap();
        assert_eq!(int2.decode(0x1), 1);
        assert_eq!(int2.decode(0x2), -2);
        assert_eq!(int2.decode(0x3), -1);
        assert_eq!(int2.encode_wrapping_i64(3), 0x3);
    }

    #[test]
    fn cast_revisions_follow_the_onnx_schema_history() {
        assert!(cast_unsupported_reason("Cast", 20, Some(DataType::Uint4), None).is_some());
        assert!(cast_unsupported_reason("Cast", 21, Some(DataType::Uint4), None).is_none());
        assert!(cast_unsupported_reason("CastLike", 24, None, Some(DataType::Int2)).is_some());
        assert!(cast_unsupported_reason("CastLike", 25, None, Some(DataType::Int2)).is_none());
        assert!(
            cast_unsupported_reason(
                "Cast",
                25,
                Some(DataType::Uint4),
                Some(DataType::Float8E4M3FN)
            )
            .is_some()
        );
        assert!(
            cast_unsupported_reason("Cast", 25, Some(DataType::Float4E2M1), Some(DataType::Int2))
                .is_some()
        );
    }
}
