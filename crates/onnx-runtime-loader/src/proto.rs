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

/// Encoded `FileDescriptorSet` for the exact vendored ONNX schema used to
/// generate [`onnx`]. Textual codecs use this descriptor so every present and
/// future field in the bound proto is handled from one source of truth.
pub const FILE_DESCRIPTOR_SET: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/onnx_descriptor.bin"));

/// Decode a [`ModelProto`] from serialized protobuf bytes.
pub fn decode_model(bytes: &[u8]) -> Result<ModelProto, LoaderError> {
    ModelProto::decode(bytes).map_err(|e| LoaderError::ProtobufParse(e.to_string()))
}

/// Convert an ONNX protobuf **TextFormat** (`.textproto`) document into the
/// binary protobuf wire encoding of a `ModelProto`.
///
/// Parsing is descriptor-driven from the exact vendored ONNX schema (via
/// [`FILE_DESCRIPTOR_SET`]), so every bound message, field, oneof, and enum is
/// covered — the same source of truth used to generate the prost types. The
/// returned bytes are byte-for-byte loadable by [`decode_model`] and the
/// weight/graph build path, so a textproto fixture flows through the identical
/// binary-decode pipeline as a real `.onnx` model.
///
/// Because this yields a self-contained binary buffer with no model-directory
/// context, textproto fixtures must inline all weights (no external
/// `.onnx.data`).
pub fn textproto_to_binary(text: &str) -> Result<Vec<u8>, LoaderError> {
    use prost_reflect::{DescriptorPool, DynamicMessage};
    use std::sync::OnceLock;

    static POOL: OnceLock<DescriptorPool> = OnceLock::new();
    let pool = POOL.get_or_init(|| {
        DescriptorPool::decode(FILE_DESCRIPTOR_SET)
            .expect("the generated ONNX descriptor set must be valid")
    });
    let descriptor = pool
        .get_message_by_name("onnx.ModelProto")
        .expect("the ONNX descriptor set must define onnx.ModelProto");
    let dynamic = DynamicMessage::parse_text_format(descriptor, text)
        .map_err(|e| LoaderError::TextProtoParse(e.to_string()))?;
    Ok(dynamic.encode_to_vec())
}
