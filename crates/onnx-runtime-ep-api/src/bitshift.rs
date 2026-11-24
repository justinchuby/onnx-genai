//! Shared dtype and version contract for native ONNX `BitShift` kernels.

use onnx_runtime_ir::DataType;

pub const SIGNED_SINCE: u64 = 28;
const DTYPES: [DataType; 8] = [
    DataType::Uint8,
    DataType::Uint16,
    DataType::Uint32,
    DataType::Uint64,
    DataType::Int8,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitShiftTypes {
    UnsignedOnly,
    SignedAndUnsigned,
}

impl BitShiftTypes {
    pub fn for_opset(opset: u64) -> Self {
        if opset >= SIGNED_SINCE {
            Self::SignedAndUnsigned
        } else {
            Self::UnsignedOnly
        }
    }

    pub fn dtypes(self) -> &'static [DataType] {
        match self {
            Self::UnsignedOnly => &DTYPES[..4],
            Self::SignedAndUnsigned => &DTYPES,
        }
    }

    pub fn validate_dtype(self, dtype: DataType) -> Result<(), String> {
        if self.dtypes().contains(&dtype) {
            Ok(())
        } else if DTYPES.contains(&dtype) {
            Err(format!(
                "BitShift: signed operand dtype {dtype:?} requires opset {SIGNED_SINCE} or later"
            ))
        } else {
            Err(format!(
                "BitShift: expected an 8-, 16-, 32-, or 64-bit integer operand, got {dtype:?}"
            ))
        }
    }
}
