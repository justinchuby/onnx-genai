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
