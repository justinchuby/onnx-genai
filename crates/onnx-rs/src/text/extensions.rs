//! Lossless patches for protobuf fields outside the native graph DSL.

use onnx_runtime_loader::proto::onnx::{
    AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, TensorShapeProto, TypeProto,
    ValueInfoProto, attribute_proto, type_proto,
};
use prost_reflect::{DynamicMessage, text_format::FormatOptions};

use crate::{Error, Result};

pub(super) const BEGIN: &str = "__onnx_extensions_begin__";
pub(super) const END: &str = "__onnx_extensions_end__";

/// Append only the schema fields that the compact graph DSL cannot express.
///
/// The extension is protobuf TextFormat rather than an opaque binary sidecar.
/// Fields represented by the DSL are removed from this patch and are restored
/// from the parsed body, so edits to graph signatures, nodes, attributes, and
/// initializer declarations remain authoritative.
pub(super) fn append(source: &ModelProto, output: &mut String) {
    let residual = residual_model(source.clone());
    let dynamic = crate::proto_serde::to_dynamic(&residual)
        .expect("generated ModelProto must match its generated descriptor");
    let text = dynamic.to_text_format_with_options(&FormatOptions::new().pretty(true));
    output.push_str(BEGIN);
    output.push('\n');
    output.push_str(&text);
    if !text.ends_with('\n') {
        output.push('\n');
    }
    output.push_str(END);
    output.push('\n');
}

pub(super) fn split(source: &str) -> Result<(&str, Option<ModelProto>)> {
    let Some(begin) = source.find(BEGIN) else {
        return Ok((source, None));
    };
    let extension_start = begin + BEGIN.len();
    let extension_tail = &source[extension_start..];
    let end_offset = extension_tail
        .find(END)
        .ok_or_else(|| Error::TextProto("unterminated ONNX extension block".into()))?;
    if !extension_tail[end_offset + END.len()..].trim().is_empty() {
        return Err(Error::TextProto(
            "unexpected content after ONNX extension block".into(),
        ));
    }
    let text = extension_tail[..end_offset].trim();
    let dynamic = DynamicMessage::parse_text_format(crate::proto_serde::descriptor(), text)
        .map_err(|error| Error::TextProto(format!("invalid ONNX extension block: {error}")))?;
    let residual = crate::proto_serde::from_dynamic(&dynamic)
        .map_err(|error| Error::TextProto(format!("invalid ONNX extension block: {error}")))?;
    Ok((source[..begin].trim_end(), Some(residual)))
}

pub(super) fn merge(mut residual: ModelProto, native: ModelProto) -> ModelProto {
    residual.ir_version = native.ir_version;
    residual.opset_import = native.opset_import;
    residual.graph = match (residual.graph.take(), native.graph) {
        (Some(residual), Some(native)) => Some(merge_graph(residual, native, true)),
        (_, native) => native,
    };
    residual
}

fn residual_model(mut proto: ModelProto) -> ModelProto {
    proto.ir_version = 0;
    proto.opset_import.clear();
    proto.graph = proto.graph.map(|graph| residual_graph(graph, true));
    proto
}

fn residual_graph(mut graph: GraphProto, name_is_native: bool) -> GraphProto {
    if name_is_native {
        graph.name.clear();
    }
    graph.node = graph.node.into_iter().map(residual_node).collect();
    graph.initializer = graph
        .initializer
        .into_iter()
        .map(|mut tensor| {
            tensor.name.clear();
            residual_tensor(tensor)
        })
        .collect();
    graph.input = graph.input.into_iter().map(residual_value_info).collect();
    graph.output = graph.output.into_iter().map(residual_value_info).collect();
    graph
}

fn residual_node(mut node: NodeProto) -> NodeProto {
    node.input.clear();
    node.output.clear();
    node.op_type.clear();
    node.domain.clear();
    node.attribute = node.attribute.into_iter().map(residual_attribute).collect();
    node
}

fn residual_attribute(mut attr: AttributeProto) -> AttributeProto {
    use attribute_proto::AttributeType;

    match AttributeType::try_from(attr.r#type).unwrap_or(AttributeType::Undefined) {
        AttributeType::Float => attr.f = 0.0,
        AttributeType::Int => attr.i = 0,
        AttributeType::String if std::str::from_utf8(&attr.s).is_ok() => attr.s.clear(),
        AttributeType::Tensor => attr.t = attr.t.map(residual_tensor),
        AttributeType::Graph => attr.g = attr.g.map(|graph| residual_graph(graph, false)),
        AttributeType::Floats => attr.floats.clear(),
        AttributeType::Ints => attr.ints.clear(),
        AttributeType::Strings
            if attr
                .strings
                .iter()
                .all(|value| std::str::from_utf8(value).is_ok()) =>
        {
            attr.strings.clear();
        }
        AttributeType::Graphs => {
            attr.graphs = attr
                .graphs
                .into_iter()
                .map(|graph| residual_graph(graph, false))
                .collect()
        }
        AttributeType::Undefined
        | AttributeType::String
        | AttributeType::SparseTensor
        | AttributeType::TypeProto
        | AttributeType::Strings
        | AttributeType::Tensors
        | AttributeType::SparseTensors
        | AttributeType::TypeProtos => {}
    }
    attr
}

fn residual_tensor(mut tensor: TensorProto) -> TensorProto {
    tensor.dims.clear();
    tensor.data_type = 0;
    tensor
}

fn residual_value_info(mut value: ValueInfoProto) -> ValueInfoProto {
    value.name.clear();
    value.r#type = value.r#type.map(residual_type);
    value
}

fn residual_type(mut value: TypeProto) -> TypeProto {
    if let Some(type_proto::Value::TensorType(tensor)) = &mut value.value {
        tensor.elem_type = 0;
        if let Some(shape) = &mut tensor.shape {
            for dim in &mut shape.dim {
                dim.value = None;
            }
        }
    }
    value
}

fn merge_graph(mut residual: GraphProto, native: GraphProto, name_is_native: bool) -> GraphProto {
    if name_is_native {
        residual.name = native.name;
    }
    residual.node = native
        .node
        .into_iter()
        .enumerate()
        .map(|(index, node)| match residual.node.get(index).cloned() {
            Some(patch) => merge_node(patch, node),
            None => node,
        })
        .collect();
    residual.initializer = native
        .initializer
        .into_iter()
        .enumerate()
        .map(
            |(index, tensor)| match residual.initializer.get(index).cloned() {
                Some(patch) => merge_initializer(patch, tensor),
                None => tensor,
            },
        )
        .collect();
    residual.input = merge_value_infos(residual.input, native.input);
    residual.output = merge_value_infos(residual.output, native.output);
    residual
}

fn merge_node(mut residual: NodeProto, native: NodeProto) -> NodeProto {
    residual.input = native.input;
    residual.output = native.output;
    residual.op_type = native.op_type;
    residual.domain = native.domain;

    let mut native_attrs = native.attribute;
    let mut attributes = Vec::with_capacity(native_attrs.len());
    for patch in residual.attribute {
        if let Some(index) = native_attrs
            .iter()
            .position(|attribute| attribute.name == patch.name)
        {
            attributes.push(merge_attribute(patch, native_attrs.remove(index)));
        }
    }
    attributes.extend(native_attrs);
    residual.attribute = attributes;
    residual
}

fn merge_attribute(mut residual: AttributeProto, native: AttributeProto) -> AttributeProto {
    use attribute_proto::AttributeType;

    residual.name = native.name;
    residual.r#type = native.r#type;
    match AttributeType::try_from(native.r#type).unwrap_or(AttributeType::Undefined) {
        AttributeType::Float => residual.f = native.f,
        AttributeType::Int => residual.i = native.i,
        AttributeType::String if residual.s.is_empty() => residual.s = native.s,
        AttributeType::Tensor => {
            residual.t = match (residual.t.take(), native.t) {
                (Some(patch), Some(native)) => Some(merge_tensor(patch, native, false)),
                (_, native) => native,
            }
        }
        AttributeType::Graph => {
            residual.g = match (residual.g.take(), native.g) {
                (Some(patch), Some(native)) => Some(merge_graph(patch, native, false)),
                (_, native) => native,
            }
        }
        AttributeType::Floats => residual.floats = native.floats,
        AttributeType::Ints => residual.ints = native.ints,
        AttributeType::Strings if residual.strings.is_empty() => residual.strings = native.strings,
        AttributeType::Tensors => {
            residual.tensors = native
                .tensors
                .into_iter()
                .enumerate()
                .map(
                    |(index, tensor)| match residual.tensors.get(index).cloned() {
                        Some(patch) => patch,
                        None => tensor,
                    },
                )
                .collect();
        }
        AttributeType::Graphs => {
            residual.graphs = native
                .graphs
                .into_iter()
                .enumerate()
                .map(|(index, graph)| match residual.graphs.get(index).cloned() {
                    Some(patch) => merge_graph(patch, graph, false),
                    None => graph,
                })
                .collect();
        }
        AttributeType::Undefined
        | AttributeType::String
        | AttributeType::SparseTensor
        | AttributeType::TypeProto
        | AttributeType::Strings
        | AttributeType::SparseTensors
        | AttributeType::TypeProtos => {}
    }
    residual
}

fn merge_initializer(residual: TensorProto, native: TensorProto) -> TensorProto {
    merge_tensor(residual, native, true)
}

fn merge_tensor(
    mut residual: TensorProto,
    native: TensorProto,
    name_is_native: bool,
) -> TensorProto {
    residual.dims = native.dims;
    residual.data_type = native.data_type;
    if name_is_native {
        residual.name = native.name;
    }
    residual
}

fn merge_value_infos(
    residual: Vec<ValueInfoProto>,
    native: Vec<ValueInfoProto>,
) -> Vec<ValueInfoProto> {
    native
        .into_iter()
        .enumerate()
        .map(|(index, value)| match residual.get(index).cloned() {
            Some(patch) => merge_value_info(patch, value),
            None => value,
        })
        .collect()
}

fn merge_value_info(mut residual: ValueInfoProto, native: ValueInfoProto) -> ValueInfoProto {
    residual.name = native.name;
    residual.r#type = match (residual.r#type.take(), native.r#type) {
        (Some(patch), Some(native)) => Some(merge_type(patch, native)),
        (_, native) => native,
    };
    residual
}

fn merge_type(mut residual: TypeProto, native: TypeProto) -> TypeProto {
    match (&mut residual.value, native.value) {
        (
            Some(type_proto::Value::TensorType(patch)),
            Some(type_proto::Value::TensorType(native)),
        ) => {
            patch.elem_type = native.elem_type;
            patch.shape = match (patch.shape.take(), native.shape) {
                (Some(patch), Some(native)) => Some(merge_shape(patch, native)),
                (_, native) => native,
            };
        }
        (None, native) => residual.value = native,
        (Some(_), _) => {}
    }
    residual
}

fn merge_shape(residual: TensorShapeProto, native: TensorShapeProto) -> TensorShapeProto {
    TensorShapeProto {
        dim: native
            .dim
            .into_iter()
            .enumerate()
            .map(|(index, mut dim)| {
                if let Some(patch) = residual.dim.get(index) {
                    dim.denotation = patch.denotation.clone();
                }
                dim
            })
            .collect(),
    }
}
