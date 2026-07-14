//! ONNX protobuf decoding (§19.1).
//!
//! The `onnx` submodule contains the `prost`-generated types compiled from the
//! vendored `proto/onnx.proto3` (see `build.rs`). [`decode_model`] parses a
//! serialized `ModelProto` from bytes.

use prost::Message;

use crate::LoaderError;

/// The `prost`-generated ONNX protobuf types (package `onnx`).
#[allow(clippy::all, missing_docs, non_snake_case)]
pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub use onnx::ModelProto;

/// Decode a [`ModelProto`] from serialized protobuf bytes.
pub fn decode_model(bytes: &[u8]) -> Result<ModelProto, LoaderError> {
    ModelProto::decode(bytes).map_err(|e| LoaderError::ProtobufParse(e.to_string()))
}
