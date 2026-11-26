//! Shared dtype and version contract for native ONNX `BitShift` kernels.

use onnx_runtime_ir::DataType;

pub const SINCE_VERSION: u64 = 11;

/// Keep the shipped ORT descriptor unsigned even though native execution also
/// supports signed integers.
pub const ORT_DTYPES: &[DataType] = &[
    DataType::Uint8,
    DataType::Uint16,
    DataType::Uint32,
    DataType::Uint64,
];

pub fn native_supports_dtype(dtype: DataType) -> bool {
    ORT_DTYPES.contains(&dtype)
        || matches!(
            dtype,
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
        )
}

pub fn unsupported_dtype_reason(dtype: DataType) -> String {
    format!("BitShift: expected an 8-, 16-, 32-, or 64-bit integer operand, got {dtype:?}")
}
