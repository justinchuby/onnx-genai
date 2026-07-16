//! Element data types, matching `onnx.TensorProto.DataType`.

/// Supported tensor element types.
///
/// The discriminant values match ONNX `TensorProto.DataType` so that the
/// loader can cast the protobuf integer directly (see [`DataType::from_onnx`]).
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DataType {
    Undefined = 0,
    Float32 = 1,
    Uint8 = 2,
    Int8 = 3,
    Uint16 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    String = 8,
    Bool = 9,
    Float16 = 10,
    Float64 = 11,
    Uint32 = 12,
    Uint64 = 13,
    Complex64 = 14,
    Complex128 = 15,
    BFloat16 = 16,
    Float8E4M3FN = 17,
    Float8E4M3FNUZ = 18,
    Float8E5M2 = 19,
    Float8E5M2FNUZ = 20,
    Uint4 = 21,
    Int4 = 22,
    Float4E2M1 = 23,
}

impl DataType {
    /// Byte size per element. Sub-byte (packed) and variable-width types
    /// return `0`; use [`DataType::bit_size`] for those.
    pub fn byte_size(self) -> usize {
        match self {
            Self::Float32 | Self::Int32 | Self::Uint32 => 4,
            Self::Float64 | Self::Int64 | Self::Uint64 | Self::Complex64 => 8,
            Self::Complex128 => 16,
            Self::Float16 | Self::BFloat16 | Self::Int16 | Self::Uint16 => 2,
            Self::Int8
            | Self::Uint8
            | Self::Bool
            | Self::Float8E4M3FN
            | Self::Float8E4M3FNUZ
            | Self::Float8E5M2
            | Self::Float8E5M2FNUZ => 1,
            Self::Int4 | Self::Uint4 | Self::Float4E2M1 => 0, // packed: 2 elements per byte
            Self::String | Self::Undefined => 0, // variable-width / no concrete storage
        }
    }

    /// Bit size per element. Sub-byte types return their true width (e.g. 4).
    pub fn bit_size(self) -> usize {
        match self {
            Self::Int4 | Self::Uint4 | Self::Float4E2M1 => 4,
            other => other.byte_size() * 8,
        }
    }

    /// Whether this is a floating-point type (any precision).
    pub fn is_float(self) -> bool {
        matches!(
            self,
            Self::Float32
                | Self::Float64
                | Self::Float16
                | Self::BFloat16
                | Self::Float8E4M3FN
                | Self::Float8E4M3FNUZ
                | Self::Float8E5M2
                | Self::Float8E5M2FNUZ
                | Self::Float4E2M1
        )
    }

    /// Whether this is a signed or unsigned integer type (excludes `Bool`).
    pub fn is_int(self) -> bool {
        matches!(
            self,
            Self::Uint8
                | Self::Int8
                | Self::Uint16
                | Self::Int16
                | Self::Int32
                | Self::Int64
                | Self::Uint32
                | Self::Uint64
                | Self::Int4
                | Self::Uint4
        )
    }

    /// Whether elements contain real and imaginary floating-point components.
    pub fn is_complex(self) -> bool {
        matches!(self, Self::Complex64 | Self::Complex128)
    }

    /// Whether elements are packed multiple-per-byte (4-bit types).
    pub fn is_sub_byte(self) -> bool {
        matches!(self, Self::Int4 | Self::Uint4 | Self::Float4E2M1)
    }

    /// Number of bytes needed to store `count` elements, accounting for
    /// sub-byte packing (two 4-bit elements per byte, rounded up).
    ///
    /// Panics on `usize` overflow of the `count * byte_size` product. Callers
    /// that size heap allocations from an untrusted shape MUST use
    /// [`DataType::checked_storage_bytes`] instead so a crafted static shape
    /// cannot wrap the multiply and under-allocate.
    pub fn storage_bytes(self, count: usize) -> usize {
        self.checked_storage_bytes(count)
            .expect("storage_bytes overflow")
    }

    /// Number of bytes needed to store `count` elements, returning `None` when
    /// the `count * byte_size` product overflows `usize`.
    ///
    /// This is the overflow-safe counterpart of [`DataType::storage_bytes`]:
    /// even though an element *count* may fit in `usize`, the element-count →
    /// bytes multiply can still wrap for a fixed-width dtype (e.g. `2^61`
    /// elements of an 8-byte type). Allocation sites route through this so a
    /// wrapped product becomes a clean error instead of a 1-byte allocation
    /// followed by an out-of-bounds access.
    pub fn checked_storage_bytes(self, count: usize) -> Option<usize> {
        if self == Self::Undefined {
            None
        } else if self.is_sub_byte() {
            Some(count.div_ceil(2))
        } else {
            count.checked_mul(self.byte_size())
        }
    }

    /// Convert from the raw ONNX `TensorProto.DataType` integer.
    ///
    /// Returns `None` for `UNDEFINED` (0) and any out-of-range or future value
    /// the runtime does not model. The discriminants below mirror the vendored
    /// `onnx.proto3` `TensorProto.DataType` enum verbatim.
    pub fn from_onnx(raw: i32) -> Option<Self> {
        Some(match raw {
            1 => Self::Float32,
            2 => Self::Uint8,
            3 => Self::Int8,
            4 => Self::Uint16,
            5 => Self::Int16,
            6 => Self::Int32,
            7 => Self::Int64,
            8 => Self::String,
            9 => Self::Bool,
            10 => Self::Float16,
            11 => Self::Float64,
            12 => Self::Uint32,
            13 => Self::Uint64,
            14 => Self::Complex64,
            15 => Self::Complex128,
            16 => Self::BFloat16,
            17 => Self::Float8E4M3FN,
            18 => Self::Float8E4M3FNUZ,
            19 => Self::Float8E5M2,
            20 => Self::Float8E5M2FNUZ,
            21 => Self::Uint4,
            22 => Self::Int4,
            23 => Self::Float4E2M1,
            _ => return None,
        })
    }

    /// The raw ONNX `TensorProto.DataType` integer for this type.
    pub fn to_onnx(self) -> i32 {
        self as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_and_bit_sizes() {
        assert_eq!(DataType::Undefined.checked_storage_bytes(1), None);
        assert_eq!(DataType::Float32.byte_size(), 4);
        assert_eq!(DataType::Float32.bit_size(), 32);
        assert_eq!(DataType::Int64.byte_size(), 8);
        assert_eq!(DataType::Float16.byte_size(), 2);
        assert_eq!(DataType::Bool.byte_size(), 1);
        assert_eq!(DataType::Int4.byte_size(), 0);
        assert_eq!(DataType::Int4.bit_size(), 4);
    }

    #[test]
    fn sub_byte_storage_rounds_up() {
        assert_eq!(DataType::Int4.storage_bytes(4), 2);
        assert_eq!(DataType::Int4.storage_bytes(5), 3);
        assert_eq!(DataType::Float32.storage_bytes(4), 16);
    }

    #[test]
    fn checked_storage_bytes_normal_counts() {
        assert_eq!(DataType::Float32.checked_storage_bytes(4), Some(16));
        assert_eq!(DataType::Float64.checked_storage_bytes(3), Some(24));
        assert_eq!(DataType::Int4.checked_storage_bytes(5), Some(3));
        assert_eq!(DataType::Bool.checked_storage_bytes(0), Some(0));
    }

    #[test]
    fn checked_storage_bytes_detects_byte_overflow() {
        // The element *count* fits in usize but count * byte_size wraps: this
        // is the exploited path (a `[2^61]`-of-8-byte shape passes the numel
        // check yet `2^61 * 8` wraps to 0 → 1-byte allocation).
        assert_eq!(
            DataType::Float64.checked_storage_bytes(usize::MAX / 4),
            None
        );
        assert_eq!(DataType::Int64.checked_storage_bytes(usize::MAX), None);
        // Sub-byte packing can never overflow (div_ceil shrinks the count).
        assert_eq!(
            DataType::Int4.checked_storage_bytes(usize::MAX),
            Some(usize::MAX / 2 + 1)
        );
    }

    #[test]
    fn classification() {
        assert!(DataType::Float16.is_float());
        assert!(!DataType::Int32.is_float());
        assert!(DataType::Int32.is_int());
        assert!(!DataType::Bool.is_int());
        assert!(DataType::Complex64.is_complex());
        assert!(DataType::Complex128.is_complex());
        assert!(!DataType::Float32.is_complex());
        assert!(DataType::Uint4.is_sub_byte());
        // float8 / float4 variants are floats but not ints.
        for dt in [
            DataType::Float8E4M3FN,
            DataType::Float8E4M3FNUZ,
            DataType::Float8E5M2,
            DataType::Float8E5M2FNUZ,
            DataType::Float4E2M1,
        ] {
            assert!(dt.is_float(), "{dt:?} should be float");
            assert!(!dt.is_int(), "{dt:?} should not be int");
        }
        // 4-bit types (incl. Float4E2M1) are sub-byte; float8s are full bytes.
        assert!(DataType::Int4.is_sub_byte());
        assert!(DataType::Float4E2M1.is_sub_byte());
        assert!(!DataType::Float8E5M2.is_sub_byte());
        assert!(!DataType::Float8E4M3FNUZ.is_sub_byte());
    }

    #[test]
    fn float4_sub_byte_storage_rounds_up() {
        assert_eq!(DataType::Float4E2M1.byte_size(), 0);
        assert_eq!(DataType::Float4E2M1.bit_size(), 4);
        assert_eq!(DataType::Float4E2M1.storage_bytes(4), 2);
        assert_eq!(DataType::Float4E2M1.storage_bytes(5), 3);
        assert_eq!(
            DataType::Float4E2M1.checked_storage_bytes(usize::MAX),
            Some(usize::MAX / 2 + 1)
        );
    }

    #[test]
    fn float8_variants_are_one_byte() {
        for dt in [
            DataType::Float8E4M3FN,
            DataType::Float8E4M3FNUZ,
            DataType::Float8E5M2,
            DataType::Float8E5M2FNUZ,
        ] {
            assert_eq!(dt.byte_size(), 1, "{dt:?} should be 1 byte");
            assert_eq!(dt.bit_size(), 8, "{dt:?} should be 8 bits");
            assert!(!dt.is_sub_byte(), "{dt:?} is not packed");
        }
    }

    /// Pins every `DataType` variant to its authoritative ONNX
    /// `TensorProto.DataType` integer (vendored `onnx.proto3`). A regression
    /// here silently corrupts weights, so the table is exhaustive.
    #[test]
    fn onnx_discriminants_match_spec() {
        assert_eq!(DataType::Undefined.to_onnx(), 0);
        let table: &[(DataType, i32)] = &[
            (DataType::Float32, 1),
            (DataType::Uint8, 2),
            (DataType::Int8, 3),
            (DataType::Uint16, 4),
            (DataType::Int16, 5),
            (DataType::Int32, 6),
            (DataType::Int64, 7),
            (DataType::String, 8),
            (DataType::Bool, 9),
            (DataType::Float16, 10),
            (DataType::Float64, 11),
            (DataType::Uint32, 12),
            (DataType::Uint64, 13),
            (DataType::Complex64, 14),
            (DataType::Complex128, 15),
            (DataType::BFloat16, 16),
            (DataType::Float8E4M3FN, 17),
            (DataType::Float8E4M3FNUZ, 18),
            (DataType::Float8E5M2, 19),
            (DataType::Float8E5M2FNUZ, 20),
            (DataType::Uint4, 21),
            (DataType::Int4, 22),
            (DataType::Float4E2M1, 23),
        ];
        for &(dt, raw) in table {
            assert_eq!(dt.to_onnx(), raw, "{dt:?} to_onnx mismatch");
            assert_eq!(
                DataType::from_onnx(raw),
                Some(dt),
                "from_onnx({raw}) mismatch"
            );
        }
    }

    #[test]
    fn onnx_roundtrip() {
        for dt in [
            DataType::Float32,
            DataType::Int64,
            DataType::BFloat16,
            DataType::Uint4,
            DataType::Float4E2M1,
            DataType::Float8E5M2FNUZ,
        ] {
            assert_eq!(DataType::from_onnx(dt.to_onnx()), Some(dt));
        }
    }

    /// Unsupported / unknown raw values must return `None` (clean error path),
    /// never a wrong variant or a panic.
    #[test]
    fn unknown_raw_values_return_none() {
        // UNDEFINED and out-of-range/future values.
        for raw in [0, 24, 100, 9999, -1, i32::MAX, i32::MIN] {
            assert_eq!(DataType::from_onnx(raw), None, "raw {raw} should be None");
        }
    }
}
