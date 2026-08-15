//! Descriptor-driven ONNX protobuf TextFormat (`.onnxtxt` / `.pbtxt`).
//!
//! Parsing and printing use the same generated descriptor as JSON, eliminating
//! per-codec field allowlists and covering the complete bound ONNX schema.

use prost_reflect::{DynamicMessage, text_format::FormatOptions};

use crate::{Error, Model, Result};

/// Serialize a model as canonical, pretty protobuf TextFormat.
pub fn to_textproto(model: &Model) -> Result<String> {
    let dynamic = crate::proto_serde::to_dynamic(&model.to_proto()?)
        .map_err(|error| textproto_error(error.to_string()))?;
    Ok(dynamic.to_text_format_with_options(&FormatOptions::new().pretty(true)))
}

/// Parse protobuf TextFormat into an ONNX model.
pub fn from_textproto(source: &str) -> Result<Model> {
    let dynamic = DynamicMessage::parse_text_format(crate::proto_serde::descriptor(), source)
        .map_err(|error| textproto_error(error.to_string()))?;
    let proto = crate::proto_serde::from_dynamic(&dynamic)
        .map_err(|error| textproto_error(error.to_string()))?;
    Model::from_proto(proto)
}

/// Convert a protobuf TextFormat (`.textproto`) document into the binary
/// protobuf wire encoding of its `ModelProto`.
///
/// This is the conversion used by runtime loaders to accept git-friendly
/// textproto fixtures: parse the text and re-encode as the exact binary bytes a
/// runtime's binary-decode path already expects. It is deliberately
/// *lightweight* — it does not build the runtime graph or run shape inference
/// (unlike [`from_textproto`]), so it faithfully reproduces the model bytes for
/// a downstream runtime (e.g. ONNX Runtime) to load and validate itself.
///
/// Because the returned buffer carries no model-directory context, textproto
/// documents must inline all weights (no external `.onnx.data`).
pub fn to_binary(source: &str) -> Result<Vec<u8>> {
    use prost::Message;
    let dynamic = DynamicMessage::parse_text_format(crate::proto_serde::descriptor(), source)
        .map_err(|error| textproto_error(error.to_string()))?;
    Ok(dynamic.encode_to_vec())
}

fn textproto_error(message: impl Into<String>) -> Error {
    Error::TextProto(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_comments_and_alternate_delimiters() {
        let source = r#"
            # protobuf comment
            ir_version: 10
            opset_import < domain: "" version: 21 >
            graph {
              name: "main"
              input {
                name: "X"
                type { tensor_type { elem_type: 1 shape { dim { dim_value: 2 } } } }
              }
              output {
                name: "Y"
                type { tensor_type { elem_type: 1 shape { dim { dim_value: 2 } } } }
              }
              node { input: "X" output: "Y" op_type: "Identity" }
            }
        "#;
        let model = from_textproto(source).unwrap();
        assert_eq!(model.graph.num_nodes(), 1);
        let printed = to_textproto(&model).unwrap();
        assert!(printed.contains("elem_type: 1"));
        assert!(printed.contains("op_type: \"Identity\""));
    }
}
