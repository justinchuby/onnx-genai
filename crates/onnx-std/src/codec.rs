//! Uniform string-codec API for ONNX model interchange.

use crate::{Model, Result, json, text, textproto};

/// A textual serialization format for [`Model`].
///
/// Binary protobuf file I/O remains separate because it also manages paths,
/// external weights, and file-format detection.
pub trait TextCodec {
    /// Format-specific serialization options.
    type Options: Default;

    /// Serialize `model` with explicit format options.
    fn serialize(model: &Model, options: &Self::Options) -> Result<String>;

    /// Deserialize a model from this format.
    fn deserialize(source: &str) -> Result<Model>;
}

/// The human-readable ONNX text DSL (ONNX_RS §5).
#[derive(Clone, Copy, Debug, Default)]
pub struct Text;

impl TextCodec for Text {
    type Options = text::PrintOptions;

    fn serialize(model: &Model, options: &Self::Options) -> Result<String> {
        Ok(text::to_text_with(model, options))
    }

    fn deserialize(source: &str) -> Result<Model> {
        text::from_text(source)
    }
}

/// Canonical protobuf JSON mapping (ONNX_RS §6).
#[derive(Clone, Copy, Debug, Default)]
pub struct Json;

impl TextCodec for Json {
    type Options = ();

    fn serialize(model: &Model, _options: &Self::Options) -> Result<String> {
        json::to_json(model)
    }

    fn deserialize(source: &str) -> Result<Model> {
        json::from_json(source)
    }
}

/// Protobuf TextFormat (`.onnxtxt` / `.pbtxt`) (ONNX_RS §6).
#[derive(Clone, Copy, Debug, Default)]
pub struct TextProto;

impl TextCodec for TextProto {
    type Options = ();

    fn serialize(model: &Model, _options: &Self::Options) -> Result<String> {
        textproto::to_textproto(model)
    }

    fn deserialize(source: &str) -> Result<Model> {
        textproto::from_textproto(source)
    }
}
