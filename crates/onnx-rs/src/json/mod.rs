//! ONNX protobuf-JSON interchange.
//!
//! The public API converts through the loader's generated `ModelProto`, keeping
//! protobuf and JSON model I/O on the same shared IR conversion path.

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use onnx_runtime_loader::proto::onnx::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, StringStringEntryProto,
    TensorProto, TensorShapeProto, TypeProto, ValueInfoProto, attribute_proto, tensor_proto,
    tensor_shape_proto, type_proto,
};
use onnx_runtime_loader::{
    Model as EncoderModel, encode_model_proto, load_model_bytes_with_weights,
};
use prost::Message;
use serde_json::{Map, Number, Value};

use crate::{Error, Model, Result};

/// Serialize a model using the canonical protobuf JSON field and scalar mapping.
pub fn to_json(model: &Model) -> Result<String> {
    let mut encoder = EncoderModel::new(&model.graph).with_metadata(model.metadata.clone());
    if let Some(weights) = model.weights() {
        encoder = encoder.with_weights(weights);
    }
    let proto = encode_model_proto(&encoder)?;
    serde_json::to_string_pretty(&model_to_value(&proto)).map_err(json_error)
}

/// Parse an ONNX protobuf JSON document into the shared IR model.
pub fn from_json(source: &str) -> Result<Model> {
    let value: Value = serde_json::from_str(source).map_err(json_error)?;
    let proto = parse_model(&value)?;
    let metadata = onnx_runtime_loader::ModelMetadata {
        ir_version: proto.ir_version,
        producer_name: proto.producer_name.clone(),
        producer_version: proto.producer_version.clone(),
        domain: proto.domain.clone(),
        model_version: proto.model_version,
        doc_string: nonempty(proto.doc_string.clone()),
        graph_name: proto
            .graph
            .as_ref()
            .map(|g| g.name.clone())
            .unwrap_or_default(),
        metadata_props: proto
            .metadata_props
            .iter()
            .map(|entry| (entry.key.clone(), entry.value.clone()))
            .collect(),
    };
    let bytes = proto.encode_to_vec();
    let (graph, weights) = load_model_bytes_with_weights(&bytes, ".")?;
    let mut model = Model::with_metadata(graph, metadata);
    model.set_weights(weights);
    Ok(model)
}

fn json_error(error: impl std::fmt::Display) -> Error {
    Error::Json(error.to_string())
}

fn nonempty(value: String) -> Option<String> {
    (!value.is_empty()).then_some(value)
}

fn object(value: &Value) -> Result<&Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| json_error("expected JSON object"))
}

fn field<'a>(map: &'a Map<String, Value>, camel: &str, snake: &str) -> Option<&'a Value> {
    map.get(camel).or_else(|| map.get(snake))
}

fn reject_unsupported_field(map: &Map<String, Value>, camel: &str, snake: &str) -> Result<()> {
    let populated = field(map, camel, snake).is_some_and(|value| match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64() != Some(0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    });
    if populated {
        Err(json_error(format!("unsupported ONNX JSON field: {camel}")))
    } else {
        Ok(())
    }
}

fn string(map: &Map<String, Value>, camel: &str, snake: &str) -> Result<String> {
    match field(map, camel, snake) {
        None => Ok(String::new()),
        Some(Value::String(value)) => Ok(value.clone()),
        Some(_) => Err(json_error(format!("{camel} must be a string"))),
    }
}

fn i64_value(value: &Value, name: &str) -> Result<i64> {
    match value {
        Value::String(value) => value
            .parse()
            .map_err(|_| json_error(format!("{name} must be an int64 string"))),
        Value::Number(value) => value
            .as_i64()
            .ok_or_else(|| json_error(format!("{name} must be an integer"))),
        _ => Err(json_error(format!("{name} must be an int64 string"))),
    }
}

fn u64_value(value: &Value, name: &str) -> Result<u64> {
    match value {
        Value::String(value) => value
            .parse()
            .map_err(|_| json_error(format!("{name} must be a uint64 string"))),
        Value::Number(value) => value
            .as_u64()
            .ok_or_else(|| json_error(format!("{name} must be an unsigned integer"))),
        _ => Err(json_error(format!("{name} must be a uint64 string"))),
    }
}

fn i32_value(value: &Value, name: &str) -> Result<i32> {
    let integer = i64_value(value, name)?;
    i32::try_from(integer).map_err(|_| json_error(format!("{name} is outside int32 range")))
}

fn float_value(value: &Value, name: &str) -> Result<f32> {
    match value {
        Value::Number(value) => value
            .as_f64()
            .map(|value| value as f32)
            .ok_or_else(|| json_error(format!("{name} must be a number"))),
        Value::String(value) if value == "NaN" => Ok(f32::NAN),
        Value::String(value) if value == "Infinity" => Ok(f32::INFINITY),
        Value::String(value) if value == "-Infinity" => Ok(f32::NEG_INFINITY),
        _ => Err(json_error(format!("{name} must be a number"))),
    }
}

fn double_value(value: &Value, name: &str) -> Result<f64> {
    match value {
        Value::Number(value) => value
            .as_f64()
            .ok_or_else(|| json_error(format!("{name} must be a number"))),
        Value::String(value) if value == "NaN" => Ok(f64::NAN),
        Value::String(value) if value == "Infinity" => Ok(f64::INFINITY),
        Value::String(value) if value == "-Infinity" => Ok(f64::NEG_INFINITY),
        _ => Err(json_error(format!("{name} must be a number"))),
    }
}

fn array<'a>(map: &'a Map<String, Value>, camel: &str, snake: &str) -> Result<&'a [Value]> {
    match field(map, camel, snake) {
        None => Ok(&[]),
        Some(Value::Array(values)) => Ok(values),
        Some(_) => Err(json_error(format!("{camel} must be an array"))),
    }
}

fn bytes_value(value: &Value, name: &str) -> Result<Vec<u8>> {
    let encoded = value
        .as_str()
        .ok_or_else(|| json_error(format!("{name} must be a base64 string")))?;
    BASE64
        .decode(encoded)
        .map_err(|error| json_error(format!("invalid base64 in {name}: {error}")))
}

fn put(map: &mut Map<String, Value>, name: &str, value: Value, include: bool) {
    if include {
        map.insert(name.to_string(), value);
    }
}

fn strings_value(values: &[String]) -> Value {
    Value::Array(values.iter().cloned().map(Value::String).collect())
}

fn int64_json(value: i64) -> Value {
    Value::String(value.to_string())
}

fn uint64_json(value: u64) -> Value {
    Value::String(value.to_string())
}

fn float_json(value: f64) -> Value {
    if value.is_nan() {
        Value::String("NaN".into())
    } else if value == f64::INFINITY {
        Value::String("Infinity".into())
    } else if value == f64::NEG_INFINITY {
        Value::String("-Infinity".into())
    } else {
        Number::from_f64(value)
            .map(Value::Number)
            .unwrap_or_else(|| Value::String("NaN".into()))
    }
}

fn model_to_value(proto: &ModelProto) -> Value {
    let mut map = Map::new();
    put(
        &mut map,
        "irVersion",
        int64_json(proto.ir_version),
        proto.ir_version != 0,
    );
    put(
        &mut map,
        "opsetImport",
        Value::Array(proto.opset_import.iter().map(opset_to_value).collect()),
        !proto.opset_import.is_empty(),
    );
    put(
        &mut map,
        "producerName",
        Value::String(proto.producer_name.clone()),
        !proto.producer_name.is_empty(),
    );
    put(
        &mut map,
        "producerVersion",
        Value::String(proto.producer_version.clone()),
        !proto.producer_version.is_empty(),
    );
    put(
        &mut map,
        "domain",
        Value::String(proto.domain.clone()),
        !proto.domain.is_empty(),
    );
    put(
        &mut map,
        "modelVersion",
        int64_json(proto.model_version),
        proto.model_version != 0,
    );
    put(
        &mut map,
        "docString",
        Value::String(proto.doc_string.clone()),
        !proto.doc_string.is_empty(),
    );
    if let Some(graph) = &proto.graph {
        map.insert("graph".into(), graph_to_value(graph));
    }
    put(
        &mut map,
        "metadataProps",
        Value::Array(proto.metadata_props.iter().map(entry_to_value).collect()),
        !proto.metadata_props.is_empty(),
    );
    Value::Object(map)
}

fn parse_model(value: &Value) -> Result<ModelProto> {
    let map = object(value)?;
    reject_unsupported_field(map, "trainingInfo", "training_info")?;
    reject_unsupported_field(map, "functions", "functions")?;
    let graph = field(map, "graph", "graph").map(parse_graph).transpose()?;
    Ok(ModelProto {
        ir_version: field(map, "irVersion", "ir_version")
            .map(|v| i64_value(v, "irVersion"))
            .transpose()?
            .unwrap_or_default(),
        opset_import: array(map, "opsetImport", "opset_import")?
            .iter()
            .map(parse_opset)
            .collect::<Result<_>>()?,
        producer_name: string(map, "producerName", "producer_name")?,
        producer_version: string(map, "producerVersion", "producer_version")?,
        domain: string(map, "domain", "domain")?,
        model_version: field(map, "modelVersion", "model_version")
            .map(|v| i64_value(v, "modelVersion"))
            .transpose()?
            .unwrap_or_default(),
        doc_string: string(map, "docString", "doc_string")?,
        graph,
        metadata_props: array(map, "metadataProps", "metadata_props")?
            .iter()
            .map(parse_entry)
            .collect::<Result<_>>()?,
        ..Default::default()
    })
}

fn opset_to_value(opset: &OperatorSetIdProto) -> Value {
    let mut map = Map::new();
    put(
        &mut map,
        "domain",
        Value::String(opset.domain.clone()),
        !opset.domain.is_empty(),
    );
    map.insert("version".into(), int64_json(opset.version));
    Value::Object(map)
}

fn parse_opset(value: &Value) -> Result<OperatorSetIdProto> {
    let map = object(value)?;
    Ok(OperatorSetIdProto {
        domain: string(map, "domain", "domain")?,
        version: field(map, "version", "version")
            .map(|v| i64_value(v, "version"))
            .transpose()?
            .unwrap_or_default(),
    })
}

fn entry_to_value(entry: &StringStringEntryProto) -> Value {
    let mut map = Map::new();
    put(
        &mut map,
        "key",
        Value::String(entry.key.clone()),
        !entry.key.is_empty(),
    );
    put(
        &mut map,
        "value",
        Value::String(entry.value.clone()),
        !entry.value.is_empty(),
    );
    Value::Object(map)
}

fn parse_entry(value: &Value) -> Result<StringStringEntryProto> {
    let map = object(value)?;
    Ok(StringStringEntryProto {
        key: string(map, "key", "key")?,
        value: string(map, "value", "value")?,
    })
}

fn graph_to_value(graph: &GraphProto) -> Value {
    let mut map = Map::new();
    put(
        &mut map,
        "node",
        Value::Array(graph.node.iter().map(node_to_value).collect()),
        !graph.node.is_empty(),
    );
    put(
        &mut map,
        "name",
        Value::String(graph.name.clone()),
        !graph.name.is_empty(),
    );
    put(
        &mut map,
        "initializer",
        Value::Array(graph.initializer.iter().map(tensor_to_value).collect()),
        !graph.initializer.is_empty(),
    );
    put(
        &mut map,
        "docString",
        Value::String(graph.doc_string.clone()),
        !graph.doc_string.is_empty(),
    );
    put(
        &mut map,
        "input",
        Value::Array(graph.input.iter().map(value_info_to_value).collect()),
        !graph.input.is_empty(),
    );
    put(
        &mut map,
        "output",
        Value::Array(graph.output.iter().map(value_info_to_value).collect()),
        !graph.output.is_empty(),
    );
    put(
        &mut map,
        "valueInfo",
        Value::Array(graph.value_info.iter().map(value_info_to_value).collect()),
        !graph.value_info.is_empty(),
    );
    put(
        &mut map,
        "metadataProps",
        Value::Array(graph.metadata_props.iter().map(entry_to_value).collect()),
        !graph.metadata_props.is_empty(),
    );
    Value::Object(map)
}

fn parse_graph(value: &Value) -> Result<GraphProto> {
    parse_graph_with_context(value, true)
}

fn parse_nested_graph(value: &Value) -> Result<GraphProto> {
    parse_graph_with_context(value, false)
}

fn parse_graph_with_context(value: &Value, is_top_level: bool) -> Result<GraphProto> {
    let map = object(value)?;
    reject_unsupported_field(map, "sparseInitializer", "sparse_initializer")?;
    reject_unsupported_field(map, "quantizationAnnotation", "quantization_annotation")?;
    reject_unsupported_field(map, "docString", "doc_string")?;
    reject_unsupported_field(map, "metadataProps", "metadata_props")?;
    if !is_top_level {
        reject_unsupported_field(map, "name", "name")?;
    }
    Ok(GraphProto {
        node: array(map, "node", "node")?
            .iter()
            .map(parse_node)
            .collect::<Result<_>>()?,
        name: string(map, "name", "name")?,
        initializer: array(map, "initializer", "initializer")?
            .iter()
            .map(parse_tensor)
            .collect::<Result<_>>()?,
        doc_string: string(map, "docString", "doc_string")?,
        input: array(map, "input", "input")?
            .iter()
            .map(parse_value_info)
            .collect::<Result<_>>()?,
        output: array(map, "output", "output")?
            .iter()
            .map(parse_value_info)
            .collect::<Result<_>>()?,
        value_info: array(map, "valueInfo", "value_info")?
            .iter()
            .map(parse_value_info)
            .collect::<Result<_>>()?,
        metadata_props: array(map, "metadataProps", "metadata_props")?
            .iter()
            .map(parse_entry)
            .collect::<Result<_>>()?,
        ..Default::default()
    })
}

fn node_to_value(node: &NodeProto) -> Value {
    let mut map = Map::new();
    put(
        &mut map,
        "input",
        strings_value(&node.input),
        !node.input.is_empty(),
    );
    put(
        &mut map,
        "output",
        strings_value(&node.output),
        !node.output.is_empty(),
    );
    put(
        &mut map,
        "name",
        Value::String(node.name.clone()),
        !node.name.is_empty(),
    );
    put(
        &mut map,
        "opType",
        Value::String(node.op_type.clone()),
        !node.op_type.is_empty(),
    );
    put(
        &mut map,
        "domain",
        Value::String(node.domain.clone()),
        !node.domain.is_empty(),
    );
    put(
        &mut map,
        "attribute",
        Value::Array(node.attribute.iter().map(attribute_to_value).collect()),
        !node.attribute.is_empty(),
    );
    put(
        &mut map,
        "docString",
        Value::String(node.doc_string.clone()),
        !node.doc_string.is_empty(),
    );
    Value::Object(map)
}

fn parse_string_array(map: &Map<String, Value>, camel: &str, snake: &str) -> Result<Vec<String>> {
    array(map, camel, snake)?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| json_error(format!("{camel} entries must be strings")))
        })
        .collect()
}

fn parse_node(value: &Value) -> Result<NodeProto> {
    let map = object(value)?;
    reject_unsupported_field(map, "overload", "overload")?;
    reject_unsupported_field(map, "metadataProps", "metadata_props")?;
    Ok(NodeProto {
        input: parse_string_array(map, "input", "input")?,
        output: parse_string_array(map, "output", "output")?,
        name: string(map, "name", "name")?,
        op_type: string(map, "opType", "op_type")?,
        domain: string(map, "domain", "domain")?,
        attribute: array(map, "attribute", "attribute")?
            .iter()
            .map(parse_attribute)
            .collect::<Result<_>>()?,
        doc_string: string(map, "docString", "doc_string")?,
        ..Default::default()
    })
}

fn attribute_to_value(attribute: &AttributeProto) -> Value {
    let mut map = Map::new();
    put(
        &mut map,
        "name",
        Value::String(attribute.name.clone()),
        !attribute.name.is_empty(),
    );
    put(
        &mut map,
        "docString",
        Value::String(attribute.doc_string.clone()),
        !attribute.doc_string.is_empty(),
    );
    let attr_type = attribute_proto::AttributeType::try_from(attribute.r#type)
        .map(|value| Value::String(value.as_str_name().into()))
        .unwrap_or_else(|_| Value::Number(attribute.r#type.into()));
    put(&mut map, "type", attr_type, attribute.r#type != 0);
    put(
        &mut map,
        "f",
        float_json(attribute.f as f64),
        attribute.f != 0.0,
    );
    put(&mut map, "i", int64_json(attribute.i), attribute.i != 0);
    put(
        &mut map,
        "s",
        Value::String(BASE64.encode(&attribute.s)),
        !attribute.s.is_empty(),
    );
    if let Some(tensor) = &attribute.t {
        map.insert("t".into(), tensor_to_value(tensor));
    }
    if let Some(graph) = &attribute.g {
        map.insert("g".into(), graph_to_value(graph));
    }
    if let Some(tp) = &attribute.tp {
        map.insert("tp".into(), type_to_value(tp));
    }
    put(
        &mut map,
        "floats",
        Value::Array(
            attribute
                .floats
                .iter()
                .map(|&v| float_json(v as f64))
                .collect(),
        ),
        !attribute.floats.is_empty(),
    );
    put(
        &mut map,
        "ints",
        Value::Array(attribute.ints.iter().map(|&v| int64_json(v)).collect()),
        !attribute.ints.is_empty(),
    );
    put(
        &mut map,
        "strings",
        Value::Array(
            attribute
                .strings
                .iter()
                .map(|v| Value::String(BASE64.encode(v)))
                .collect(),
        ),
        !attribute.strings.is_empty(),
    );
    put(
        &mut map,
        "graphs",
        Value::Array(attribute.graphs.iter().map(graph_to_value).collect()),
        !attribute.graphs.is_empty(),
    );
    Value::Object(map)
}

fn parse_attribute_type(value: &Value) -> Result<i32> {
    if let Some(name) = value.as_str() {
        return attribute_proto::AttributeType::from_str_name(name)
            .map(|value| value as i32)
            .ok_or_else(|| json_error(format!("unknown AttributeProto type {name:?}")));
    }
    i32_value(value, "type")
}

fn parse_attribute(value: &Value) -> Result<AttributeProto> {
    let map = object(value)?;
    Ok(AttributeProto {
        name: string(map, "name", "name")?,
        doc_string: string(map, "docString", "doc_string")?,
        r#type: field(map, "type", "type")
            .map(parse_attribute_type)
            .transpose()?
            .unwrap_or_default(),
        f: field(map, "f", "f")
            .map(|v| float_value(v, "f"))
            .transpose()?
            .unwrap_or_default(),
        i: field(map, "i", "i")
            .map(|v| i64_value(v, "i"))
            .transpose()?
            .unwrap_or_default(),
        s: field(map, "s", "s")
            .map(|v| bytes_value(v, "s"))
            .transpose()?
            .unwrap_or_default(),
        t: field(map, "t", "t").map(parse_tensor).transpose()?,
        g: field(map, "g", "g").map(parse_nested_graph).transpose()?,
        tp: field(map, "tp", "tp").map(parse_type).transpose()?,
        floats: array(map, "floats", "floats")?
            .iter()
            .map(|v| float_value(v, "floats"))
            .collect::<Result<_>>()?,
        ints: array(map, "ints", "ints")?
            .iter()
            .map(|v| i64_value(v, "ints"))
            .collect::<Result<_>>()?,
        strings: array(map, "strings", "strings")?
            .iter()
            .map(|v| bytes_value(v, "strings"))
            .collect::<Result<_>>()?,
        graphs: array(map, "graphs", "graphs")?
            .iter()
            .map(parse_nested_graph)
            .collect::<Result<_>>()?,
        ..Default::default()
    })
}

fn tensor_to_value(tensor: &TensorProto) -> Value {
    let mut map = Map::new();
    put(
        &mut map,
        "dims",
        Value::Array(tensor.dims.iter().map(|&v| int64_json(v)).collect()),
        !tensor.dims.is_empty(),
    );
    let dtype = tensor_proto::DataType::try_from(tensor.data_type)
        .map(|value| Value::String(value.as_str_name().into()))
        .unwrap_or_else(|_| Value::Number(tensor.data_type.into()));
    put(&mut map, "dataType", dtype, tensor.data_type != 0);
    put(
        &mut map,
        "floatData",
        Value::Array(
            tensor
                .float_data
                .iter()
                .map(|&v| float_json(v as f64))
                .collect(),
        ),
        !tensor.float_data.is_empty(),
    );
    put(
        &mut map,
        "int32Data",
        Value::Array(
            tensor
                .int32_data
                .iter()
                .map(|&v| Value::Number(v.into()))
                .collect(),
        ),
        !tensor.int32_data.is_empty(),
    );
    put(
        &mut map,
        "stringData",
        Value::Array(
            tensor
                .string_data
                .iter()
                .map(|v| Value::String(BASE64.encode(v)))
                .collect(),
        ),
        !tensor.string_data.is_empty(),
    );
    put(
        &mut map,
        "int64Data",
        Value::Array(tensor.int64_data.iter().map(|&v| int64_json(v)).collect()),
        !tensor.int64_data.is_empty(),
    );
    put(
        &mut map,
        "name",
        Value::String(tensor.name.clone()),
        !tensor.name.is_empty(),
    );
    put(
        &mut map,
        "docString",
        Value::String(tensor.doc_string.clone()),
        !tensor.doc_string.is_empty(),
    );
    put(
        &mut map,
        "rawData",
        Value::String(BASE64.encode(&tensor.raw_data)),
        !tensor.raw_data.is_empty(),
    );
    put(
        &mut map,
        "doubleData",
        Value::Array(tensor.double_data.iter().map(|&v| float_json(v)).collect()),
        !tensor.double_data.is_empty(),
    );
    put(
        &mut map,
        "uint64Data",
        Value::Array(tensor.uint64_data.iter().map(|&v| uint64_json(v)).collect()),
        !tensor.uint64_data.is_empty(),
    );
    put(
        &mut map,
        "externalData",
        Value::Array(tensor.external_data.iter().map(entry_to_value).collect()),
        !tensor.external_data.is_empty(),
    );
    let location = tensor_proto::DataLocation::try_from(tensor.data_location)
        .map(|value| Value::String(value.as_str_name().into()))
        .unwrap_or_else(|_| Value::Number(tensor.data_location.into()));
    put(
        &mut map,
        "dataLocation",
        location,
        tensor.data_location != 0,
    );
    Value::Object(map)
}

fn parse_tensor_enum(value: &Value, name: &str) -> Result<i32> {
    if let Some(enum_name) = value.as_str() {
        return tensor_proto::DataType::from_str_name(enum_name)
            .map(|value| value as i32)
            .ok_or_else(|| json_error(format!("unknown {name} {enum_name:?}")));
    }
    i32_value(value, name)
}

fn parse_location(value: &Value) -> Result<i32> {
    if let Some(name) = value.as_str() {
        return tensor_proto::DataLocation::from_str_name(name)
            .map(|value| value as i32)
            .ok_or_else(|| json_error(format!("unknown dataLocation {name:?}")));
    }
    i32_value(value, "dataLocation")
}

fn parse_tensor(value: &Value) -> Result<TensorProto> {
    let map = object(value)?;
    reject_unsupported_field(map, "segment", "segment")?;
    reject_unsupported_field(map, "docString", "doc_string")?;
    reject_unsupported_field(map, "metadataProps", "metadata_props")?;
    Ok(TensorProto {
        dims: array(map, "dims", "dims")?
            .iter()
            .map(|v| i64_value(v, "dims"))
            .collect::<Result<_>>()?,
        data_type: field(map, "dataType", "data_type")
            .map(|v| parse_tensor_enum(v, "dataType"))
            .transpose()?
            .unwrap_or_default(),
        float_data: array(map, "floatData", "float_data")?
            .iter()
            .map(|v| float_value(v, "floatData"))
            .collect::<Result<_>>()?,
        int32_data: array(map, "int32Data", "int32_data")?
            .iter()
            .map(|v| i32_value(v, "int32Data"))
            .collect::<Result<_>>()?,
        string_data: array(map, "stringData", "string_data")?
            .iter()
            .map(|v| bytes_value(v, "stringData"))
            .collect::<Result<_>>()?,
        int64_data: array(map, "int64Data", "int64_data")?
            .iter()
            .map(|v| i64_value(v, "int64Data"))
            .collect::<Result<_>>()?,
        name: string(map, "name", "name")?,
        doc_string: string(map, "docString", "doc_string")?,
        raw_data: field(map, "rawData", "raw_data")
            .map(|v| bytes_value(v, "rawData"))
            .transpose()?
            .unwrap_or_default(),
        double_data: array(map, "doubleData", "double_data")?
            .iter()
            .map(|v| double_value(v, "doubleData"))
            .collect::<Result<_>>()?,
        uint64_data: array(map, "uint64Data", "uint64_data")?
            .iter()
            .map(|v| u64_value(v, "uint64Data"))
            .collect::<Result<_>>()?,
        external_data: array(map, "externalData", "external_data")?
            .iter()
            .map(parse_entry)
            .collect::<Result<_>>()?,
        data_location: field(map, "dataLocation", "data_location")
            .map(parse_location)
            .transpose()?
            .unwrap_or_default(),
        ..Default::default()
    })
}

fn value_info_to_value(info: &ValueInfoProto) -> Value {
    let mut map = Map::new();
    put(
        &mut map,
        "name",
        Value::String(info.name.clone()),
        !info.name.is_empty(),
    );
    if let Some(value_type) = &info.r#type {
        map.insert("type".into(), type_to_value(value_type));
    }
    put(
        &mut map,
        "docString",
        Value::String(info.doc_string.clone()),
        !info.doc_string.is_empty(),
    );
    Value::Object(map)
}

fn parse_value_info(value: &Value) -> Result<ValueInfoProto> {
    let map = object(value)?;
    Ok(ValueInfoProto {
        name: string(map, "name", "name")?,
        r#type: field(map, "type", "type").map(parse_type).transpose()?,
        doc_string: string(map, "docString", "doc_string")?,
        ..Default::default()
    })
}

fn type_to_value(value_type: &TypeProto) -> Value {
    let mut map = Map::new();
    put(
        &mut map,
        "denotation",
        Value::String(value_type.denotation.clone()),
        !value_type.denotation.is_empty(),
    );
    match &value_type.value {
        Some(type_proto::Value::TensorType(value)) => {
            map.insert(
                "tensorType".into(),
                tensor_type_to_value(value.elem_type, value.shape.as_ref()),
            );
        }
        Some(type_proto::Value::SparseTensorType(value)) => {
            map.insert(
                "sparseTensorType".into(),
                tensor_type_to_value(value.elem_type, value.shape.as_ref()),
            );
        }
        Some(type_proto::Value::SequenceType(value)) => {
            let mut inner = Map::new();
            if let Some(elem) = &value.elem_type {
                inner.insert("elemType".into(), type_to_value(elem));
            }
            map.insert("sequenceType".into(), Value::Object(inner));
        }
        Some(type_proto::Value::OptionalType(value)) => {
            let mut inner = Map::new();
            if let Some(elem) = &value.elem_type {
                inner.insert("elemType".into(), type_to_value(elem));
            }
            map.insert("optionalType".into(), Value::Object(inner));
        }
        Some(type_proto::Value::MapType(value)) => {
            let mut inner = Map::new();
            inner.insert("keyType".into(), Value::Number(value.key_type.into()));
            if let Some(elem) = &value.value_type {
                inner.insert("valueType".into(), type_to_value(elem));
            }
            map.insert("mapType".into(), Value::Object(inner));
        }
        None => {}
    }
    Value::Object(map)
}

fn tensor_type_to_value(elem_type: i32, shape: Option<&TensorShapeProto>) -> Value {
    let mut map = Map::new();
    let dtype = tensor_proto::DataType::try_from(elem_type)
        .map(|value| Value::String(value.as_str_name().into()))
        .unwrap_or_else(|_| Value::Number(elem_type.into()));
    put(&mut map, "elemType", dtype, elem_type != 0);
    if let Some(shape) = shape {
        map.insert("shape".into(), shape_to_value(shape));
    }
    Value::Object(map)
}

fn parse_type(value: &Value) -> Result<TypeProto> {
    let map = object(value)?;
    let value = if let Some(value) = field(map, "tensorType", "tensor_type") {
        let (elem_type, shape) = parse_tensor_type(value)?;
        Some(type_proto::Value::TensorType(type_proto::Tensor {
            elem_type,
            shape,
        }))
    } else if let Some(value) = field(map, "sparseTensorType", "sparse_tensor_type") {
        let (elem_type, shape) = parse_tensor_type(value)?;
        Some(type_proto::Value::SparseTensorType(
            type_proto::SparseTensor { elem_type, shape },
        ))
    } else if let Some(value) = field(map, "sequenceType", "sequence_type") {
        let inner = object(value)?;
        let elem_type = field(inner, "elemType", "elem_type")
            .map(parse_type)
            .transpose()?
            .map(Box::new);
        Some(type_proto::Value::SequenceType(Box::new(
            type_proto::Sequence { elem_type },
        )))
    } else if let Some(value) = field(map, "optionalType", "optional_type") {
        let inner = object(value)?;
        let elem_type = field(inner, "elemType", "elem_type")
            .map(parse_type)
            .transpose()?
            .map(Box::new);
        Some(type_proto::Value::OptionalType(Box::new(
            type_proto::Optional { elem_type },
        )))
    } else if let Some(value) = field(map, "mapType", "map_type") {
        let inner = object(value)?;
        let key_type = field(inner, "keyType", "key_type")
            .map(|v| parse_tensor_enum(v, "keyType"))
            .transpose()?
            .unwrap_or_default();
        let value_type = field(inner, "valueType", "value_type")
            .map(parse_type)
            .transpose()?
            .map(Box::new);
        Some(type_proto::Value::MapType(Box::new(type_proto::Map {
            key_type,
            value_type,
        })))
    } else {
        None
    };
    Ok(TypeProto {
        denotation: string(map, "denotation", "denotation")?,
        value,
    })
}

fn parse_tensor_type(value: &Value) -> Result<(i32, Option<TensorShapeProto>)> {
    let map = object(value)?;
    let elem_type = field(map, "elemType", "elem_type")
        .map(|v| parse_tensor_enum(v, "elemType"))
        .transpose()?
        .unwrap_or_default();
    let shape = field(map, "shape", "shape").map(parse_shape).transpose()?;
    Ok((elem_type, shape))
}

fn shape_to_value(shape: &TensorShapeProto) -> Value {
    let dimensions = shape
        .dim
        .iter()
        .map(|dimension| {
            let mut map = Map::new();
            match &dimension.value {
                Some(tensor_shape_proto::dimension::Value::DimValue(value)) => {
                    map.insert("dimValue".into(), int64_json(*value));
                }
                Some(tensor_shape_proto::dimension::Value::DimParam(value)) => {
                    map.insert("dimParam".into(), Value::String(value.clone()));
                }
                None => {}
            }
            put(
                &mut map,
                "denotation",
                Value::String(dimension.denotation.clone()),
                !dimension.denotation.is_empty(),
            );
            Value::Object(map)
        })
        .collect();
    Value::Object(Map::from_iter([("dim".into(), Value::Array(dimensions))]))
}

fn parse_shape(value: &Value) -> Result<TensorShapeProto> {
    let map = object(value)?;
    let dim = array(map, "dim", "dim")?
        .iter()
        .map(|value| {
            let map = object(value)?;
            let value = if let Some(value) = field(map, "dimValue", "dim_value") {
                Some(tensor_shape_proto::dimension::Value::DimValue(i64_value(
                    value, "dimValue",
                )?))
            } else {
                field(map, "dimParam", "dim_param")
                    .map(|value| {
                        value
                            .as_str()
                            .map(|value| {
                                tensor_shape_proto::dimension::Value::DimParam(value.into())
                            })
                            .ok_or_else(|| json_error("dimParam must be a string"))
                    })
                    .transpose()?
            };
            Ok(tensor_shape_proto::Dimension {
                denotation: string(map, "denotation", "denotation")?,
                value,
            })
        })
        .collect::<Result<_>>()?;
    Ok(TensorShapeProto { dim })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use onnx_runtime_ir::{
        Attribute, DataType, Graph, Node, NodeId, TensorData, WeightRef, static_shape,
    };
    use onnx_runtime_loader::ModelMetadata;

    use super::*;

    fn round_trip(model: &Model) -> Model {
        let json = to_json(model).expect("serialize model");
        let _: Value = serde_json::from_str(&json).expect("output is valid JSON");
        let decoded = from_json(&json).expect("deserialize model");
        assert_eq!(
            to_json(&decoded).expect("re-serialize model"),
            json,
            "protobuf structure changed across the JSON round-trip"
        );
        decoded
    }

    fn base_graph() -> Graph {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 21);
        graph
    }

    #[test]
    fn populated_unrepresentable_proto_fields_are_rejected() {
        let cases = [
            (
                r#"{"graph":{"sparseInitializer":[{}]}}"#,
                "sparseInitializer",
            ),
            (
                r#"{"graph":{"quantizationAnnotation":[{}]}}"#,
                "quantizationAnnotation",
            ),
            (r#"{"graph":{"docString":"documentation"}}"#, "docString"),
            (
                r#"{"graph":{"metadataProps":[{"key":"k","value":"v"}]}}"#,
                "metadataProps",
            ),
            (r#"{"graph":{"node":[{"overload":"local"}]}}"#, "overload"),
            (
                r#"{"graph":{"node":[{"metadataProps":[{"key":"k","value":"v"}]}]}}"#,
                "metadataProps",
            ),
            (r#"{"trainingInfo":[{}]}"#, "trainingInfo"),
            (r#"{"functions":[{}]}"#, "functions"),
            (
                r#"{"graph":{"initializer":[{"segment":{"begin":"0","end":"1"}}]}}"#,
                "segment",
            ),
            (
                r#"{"graph":{"initializer":[{"docString":"documentation"}]}}"#,
                "docString",
            ),
            (
                r#"{"graph":{"initializer":[{"metadataProps":[{"key":"k","value":"v"}]}]}}"#,
                "metadataProps",
            ),
            (
                r#"{"graph":{"node":[{"attribute":[{"g":{"name":"nested"}}]}]}}"#,
                "name",
            ),
        ];

        for (source, field) in cases {
            let error = match from_json(source) {
                Ok(_) => panic!("expected {field} to be rejected"),
                Err(error) => error,
            };
            assert!(
                error
                    .to_string()
                    .contains(&format!("unsupported ONNX JSON field: {field}")),
                "unexpected error for {field}: {error}"
            );
        }
    }

    #[test]
    fn simple_model_round_trips() {
        let mut graph = base_graph();
        let input = graph.create_named_value("X", DataType::Float32, static_shape([2, 3]));
        let output = graph.create_named_value("Y", DataType::Float32, static_shape([2, 3]));
        graph.add_input(input);
        graph.add_output(output);
        let mut node = Node::new(NodeId(0), "Identity", vec![Some(input)], vec![output]);
        node.name = "identity".into();
        graph.insert_node(node);

        let mut metadata = ModelMetadata::default();
        metadata.producer_name = "onnx-rs".into();
        metadata.model_version = 7;
        metadata.graph_name = "simple".into();
        metadata.metadata_props = vec![("purpose".into(), "json-test".into())];
        let decoded = round_trip(&Model::with_metadata(graph, metadata.clone()));

        assert_eq!(decoded.metadata, metadata);
        assert_eq!(decoded.graph.num_nodes(), 1);
        assert_eq!(decoded.graph.inputs.len(), 1);
        assert_eq!(decoded.graph.outputs.len(), 1);
    }

    #[test]
    fn initializers_and_typed_attributes_round_trip() {
        let mut graph = base_graph();
        graph.opset_imports.insert("test".into(), 1);
        let input = graph.create_named_value("X", DataType::Float32, static_shape([2]));
        let weight = graph.create_named_value("W", DataType::Float32, static_shape([2]));
        let output = graph.create_named_value("Y", DataType::Float32, static_shape([2]));
        graph.add_input(input);
        graph.add_output(output);
        graph.set_initializer(
            weight,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![2],
                [1.25f32.to_le_bytes(), (-2.5f32).to_le_bytes()].concat(),
            )),
        );

        let mut node = Node::new(
            NodeId(0),
            "TypedAttributes",
            vec![Some(input), Some(weight)],
            vec![output],
        );
        node.domain = "test".into();
        node.attributes = HashMap::from([
            ("axis".into(), Attribute::Int(-1)),
            ("alpha".into(), Attribute::Float(0.25)),
            ("label".into(), Attribute::String(b"hello".to_vec())),
            ("axes".into(), Attribute::Ints(vec![0, 1])),
            ("scales".into(), Attribute::Floats(vec![0.5, 2.0])),
            (
                "labels".into(),
                Attribute::Strings(vec![b"a".to_vec(), b"b".to_vec()]),
            ),
        ]);
        graph.insert_node(node);

        let decoded = round_trip(&Model::new(graph));
        assert_eq!(decoded.graph.initializers.len(), 1);
        let (_, node) = decoded.graph.nodes.iter().next().expect("node");
        assert_eq!(node.attr("axis").and_then(Attribute::as_int), Some(-1));
        assert_eq!(node.attr("alpha").and_then(Attribute::as_float), Some(0.25));
        assert_eq!(
            node.attr("label").and_then(Attribute::as_bytes),
            Some(&b"hello"[..])
        );
        assert_eq!(
            node.attr("axes").and_then(Attribute::as_ints),
            Some(&[0, 1][..])
        );
    }

    fn branch_graph(name: &str, value: f32) -> Graph {
        let mut graph = Graph::new();
        let output = graph.create_named_value(name, DataType::Float32, static_shape([1]));
        graph.set_initializer(
            output,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![1],
                value.to_le_bytes().to_vec(),
            )),
        );
        graph.add_output(output);
        graph
    }

    #[test]
    fn control_flow_subgraphs_round_trip() {
        let mut graph = base_graph();
        let condition = graph.create_named_value("cond", DataType::Bool, static_shape([]));
        let output = graph.create_named_value("result", DataType::Float32, static_shape([1]));
        graph.add_input(condition);
        graph.add_output(output);

        let then_branch = branch_graph("then_value", 1.0);
        let else_branch = branch_graph("else_value", -1.0);
        let mut node = Node::new(NodeId(0), "If", vec![Some(condition)], vec![output]);
        node.name = "choose".into();
        node.attributes.insert(
            "then_branch".into(),
            Attribute::Graph(Box::new(then_branch)),
        );
        node.attributes.insert(
            "else_branch".into(),
            Attribute::Graph(Box::new(else_branch)),
        );
        graph.insert_node(node);

        let decoded = round_trip(&Model::new(graph));
        let (_, node) = decoded.graph.nodes.iter().next().expect("If node");
        assert!(matches!(
            node.attr("then_branch"),
            Some(Attribute::Graph(_))
        ));
        assert!(matches!(
            node.attr("else_branch"),
            Some(Attribute::Graph(_))
        ));
        assert_eq!(decoded.graph.subgraphs.len(), 2);
    }

    #[test]
    fn malformed_json_returns_error() {
        assert!(from_json("{ definitely not JSON").is_err());
        assert!(from_json(r#"{"irVersion":[]}"#).is_err());
        assert!(from_json(r#"{"irVersion":"10","graph":"not an object"}"#).is_err());
    }

    #[test]
    fn typed_tensor_fields_are_accepted() {
        let source = r#"{
          "irVersion": "10",
          "opsetImport": [{"version": "21"}],
          "graph": {
            "initializer": [{
              "dims": ["2"],
              "dataType": "INT64",
              "int64Data": ["4", "9"],
              "name": "values"
            }],
            "output": [{
              "name": "values",
              "type": {"tensorType": {
                "elemType": "INT64",
                "shape": {"dim": [{"dimValue": "2"}]}
              }}
            }]
          }
        }"#;
        let model = from_json(source).expect("typed TensorProto fields parse");
        assert_eq!(model.graph.initializers.len(), 1);
        assert!(
            to_json(&model)
                .expect("serialize parsed model")
                .contains("\"rawData\"")
        );
    }

    fn assert_low_bit_typed_tensors_round_trip(
        dtype_name: &str,
        dtype: DataType,
        int32_data: &[i32],
        expected_data: &[u8],
        dims: usize,
    ) {
        let source = format!(
            r#"{{
              "irVersion": "10",
              "opsetImport": [{{"version": "21"}}],
              "graph": {{
                "initializer": [{{
                  "dims": ["{dims}"],
                  "dataType": "{dtype_name}",
                  "int32Data": {int32_data:?},
                  "name": "typed_initializer"
                }}],
                "node": [{{
                  "output": ["typed_attribute_output"],
                  "opType": "Constant",
                  "attribute": [{{
                    "name": "value",
                    "type": "TENSOR",
                    "t": {{
                      "dims": ["{dims}"],
                      "dataType": "{dtype_name}",
                      "int32Data": {int32_data:?}
                    }}
                  }}]
                }}],
                "output": [
                  {{
                    "name": "typed_initializer",
                    "type": {{"tensorType": {{
                      "elemType": "{dtype_name}",
                      "shape": {{"dim": [{{"dimValue": "{dims}"}}]}}
                    }}}}
                  }},
                  {{
                    "name": "typed_attribute_output",
                    "type": {{"tensorType": {{
                      "elemType": "{dtype_name}",
                      "shape": {{"dim": [{{"dimValue": "{dims}"}}]}}
                    }}}}
                  }}
                ]
              }}
            }}"#
        );

        let assert_data = |model: &Model| {
            let initializer = model
                .graph
                .initializers
                .iter()
                .find(|(id, _)| {
                    model.graph.value(**id).name.as_deref() == Some("typed_initializer")
                })
                .map(|(_, weight)| weight)
                .expect("typed initializer");
            let WeightRef::Inline(initializer) = initializer else {
                panic!("typed initializer must be inline");
            };
            assert_eq!(initializer.dtype, dtype);
            assert_eq!(initializer.data, expected_data);

            let (_, node) = model.graph.nodes.iter().next().expect("Constant node");
            let Some(Attribute::Tensor(attribute)) = node.attr("value") else {
                panic!("Constant value must be a tensor attribute");
            };
            assert_eq!(attribute.dtype, dtype);
            assert_eq!(attribute.data, expected_data);
        };

        let parsed = from_json(&source).expect("parse typed low-bit tensors");
        assert_data(&parsed);

        let canonical = to_json(&parsed).expect("serialize low-bit tensors");
        assert!(canonical.contains("\"rawData\""));
        let decoded = from_json(&canonical).expect("reparse serialized low-bit tensors");
        assert_data(&decoded);
        assert_eq!(
            to_json(&decoded).expect("re-serialize low-bit tensors"),
            canonical
        );
    }

    #[test]
    fn float8_initializer_and_attribute_tensor_round_trip() {
        assert_low_bit_typed_tensors_round_trip(
            "FLOAT8E4M3FN",
            DataType::Float8E4M3FN,
            &[0x01, 0x7f, 0xff],
            &[0x01, 0x7f, 0xff],
            3,
        );
    }

    #[test]
    fn int4_initializer_and_attribute_tensor_round_trip() {
        assert_low_bit_typed_tensors_round_trip(
            "INT4",
            DataType::Int4,
            &[0x21, 0x03],
            &[0x21, 0x03],
            3,
        );
    }
}
