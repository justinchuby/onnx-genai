//! Canonical descriptor-driven ONNX protobuf JSON interchange.
//!
//! The generated descriptor for the crate's vendored `onnx.proto3` is the only
//! field table. This keeps JSON and protobuf TextFormat automatically complete
//! and consistent for every message, field, oneof, and enum in the bound spec.

use prost_reflect::DynamicMessage;

use crate::{Error, Model, Result};

/// Serialize a model using protobuf's canonical JSON mapping.
pub fn to_json(model: &Model) -> Result<String> {
    let dynamic = crate::proto_serde::to_dynamic(&model.to_proto()?)?;
    serde_json::to_string_pretty(&dynamic).map_err(json_error)
}

/// Parse a canonical ONNX protobuf JSON document.
pub fn from_json(source: &str) -> Result<Model> {
    let mut deserializer = serde_json::Deserializer::from_str(source);
    let dynamic = DynamicMessage::deserialize(crate::proto_serde::descriptor(), &mut deserializer)
        .map_err(json_error)?;
    deserializer.end().map_err(json_error)?;
    Model::from_proto(crate::proto_serde::from_dynamic(&dynamic)?)
}

fn json_error(error: impl std::fmt::Display) -> Error {
    Error::Json(error.to_string())
}

#[cfg(test)]
mod tests {
    use onnx_runtime_ir::{DataType, Graph, Node, NodeId, static_shape};

    use super::*;

    #[test]
    fn simple_model_round_trips() {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 21);
        let input = graph.create_named_value("X", DataType::Float32, static_shape([2, 3]));
        let output = graph.create_named_value("Y", DataType::Float32, static_shape([2, 3]));
        graph.add_input(input);
        graph.add_output(output);
        graph.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(input)],
            vec![output],
        ));

        let json = to_json(&Model::new(graph)).unwrap();
        let decoded = from_json(&json).unwrap();
        assert_eq!(to_json(&decoded).unwrap(), json);
        assert_eq!(decoded.graph.num_nodes(), 1);
    }

    #[test]
    fn rejects_unknown_fields() {
        let error = match from_json(r#"{"unknownOnnxField": true}"#) {
            Ok(_) => panic!("unknown field must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("unknownOnnxField"));
    }
}
