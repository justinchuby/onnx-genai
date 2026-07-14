//! ONNX protobuf decoding (§19.1).
//!
//! In the full implementation this module wraps `prost`-generated types from
//! `onnx.proto3` (built via a `build.rs`). For the Phase 1 skeleton it exposes
//! only the decode entry point.

use crate::LoaderError;

/// A decoded ONNX `ModelProto`. Placeholder until the `prost` types land.
#[derive(Debug, Default)]
pub struct ModelProto {
    pub ir_version: i64,
    pub producer_name: String,
    /// Opset imports as (domain, version) pairs.
    pub opset_import: Vec<(String, i64)>,
}

/// Decode a `ModelProto` from protobuf bytes.
pub fn decode_model(bytes: &[u8]) -> Result<ModelProto, LoaderError> {
    let _ = bytes;
    todo!("ort2-loader: prost-decode onnx.ModelProto")
}
