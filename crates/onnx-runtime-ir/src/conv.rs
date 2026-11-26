//! Shared ONNX `Conv` dtype contracts.

use crate::DataType;

/// Schema boundary used when advertising kernels through schema-governed ABIs.
///
/// Standalone execution may accept supported dtypes on earlier revisions when
/// their operator semantics are unchanged.
pub const SCHEMA_BF16_SINCE_OPSET: u64 = 22;
pub const SCHEMA_PRE_BF16_DTYPES: &[DataType] = &[DataType::Float32, DataType::Float16];
pub const DTYPES: &[DataType] = &[DataType::Float32, DataType::Float16, DataType::BFloat16];
