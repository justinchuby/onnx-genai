//! ONNX protobuf **encoding** — the inverse of [`graph_builder`](crate::graph_builder)
//! and [`weights`](crate::weights) (§19.1, §55.4 dump path).
//!
//! Serialises an [`onnx_runtime_ir::Graph`] (plus model-level metadata that the
//! IR does not itself store) back into an ONNX `ModelProto`, then to protobuf
//! bytes via `prost`. This is the foundational capability the EPContext writer
//! (§55.4) builds on, but it is deliberately **model-agnostic**: it hardcodes no
//! op type, vendor, or model name.
//!
//! ## Round-trip contract
//!
//! Everything the load path (`decode → build → weights`) preserves survives an
//! `encode → decode` round-trip byte-for-byte:
//!
//! * nodes: `op_type`, `domain`, `input`/`output` order (incl. skipped optional
//!   slots), attributes, `doc_string`;
//! * graph inputs / outputs / interior `value_info` (dtype + static & symbolic
//!   dims, symbols re-emitted by their interned name);
//! * initializers: all supported dtypes, raw little-endian bytes byte-exact
//!   (including `STRING` payloads);
//! * opset imports, `ir_version`, producer fields, model `doc_string`,
//!   `metadata_props`.
//!
//! ### EPContext opaque blob (critical, §55.3/§55.4)
//!
//! The load path stores an `EPContext` node's `ep_cache_context` attribute — an
//! ONNX `STRING` attribute whose bytes are an *opaque* compiled-vendor blob or a
//! relative path — losslessly as a `UINT8` tensor, so lossy UTF-8 decoding never
//! corrupts a binary blob. The encoder performs the exact inverse: it writes
//! that `UINT8` tensor back out as a `STRING` attribute (`AttributeProto.s`) with
//! its bytes intact, so the produced model both round-trips through the load path
//! and matches what upstream ORT emits.
//!
//! ## Fields deliberately not encoded
//!
//! The IR `Graph` does not model these, so they cannot be reproduced and are
//! emitted empty / default:
//!
//! * per-node `name` (the IR [`Node`](onnx_runtime_ir::Node) has no name field);
//! * `TrainingInfoProto`, `FunctionProto`, sparse initializers, quantization
//!   annotations (not represented in the IR);
//! * control-flow **subgraph** formal `input`/`output` lists — the load path
//!   does not register them as graph I/O for nested graphs, so only the
//!   subgraph's nodes / value_info are re-emitted. (No Phase-1 op uses these.)

use std::collections::HashSet;
use std::path::Path;

use prost::Message;

use onnx_runtime_ir::{
    Attribute, DataType, Dim, Graph, Node, Shape, TensorData, TypeProto, ValueId, WeightRef,
};

use crate::proto::onnx::{
    self, attribute_proto::AttributeType, tensor_shape_proto, type_proto, AttributeProto,
    GraphProto, ModelProto, NodeProto, OperatorSetIdProto, StringStringEntryProto, TensorProto,
    TensorShapeProto, ValueInfoProto,
};
use crate::weights::WeightStore;
use crate::LoaderError;

/// Default ONNX `ir_version` stamped when [`ModelMetadata`] does not override it
/// (IR version 10, the version paired with opset 21).
pub const DEFAULT_IR_VERSION: i64 = 10;

/// Model-level metadata that the IR [`Graph`] does not itself carry.
///
/// The load path drops these (it only keeps `opset_imports` on the `Graph`), so
/// a caller that wants a faithful `ModelProto` supplies them here. All fields
/// default to empty/zero except [`ir_version`](ModelMetadata::ir_version).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelMetadata {
    /// `ModelProto.ir_version`.
    pub ir_version: i64,
    /// `ModelProto.producer_name`.
    pub producer_name: String,
    /// `ModelProto.producer_version`.
    pub producer_version: String,
    /// `ModelProto.domain`.
    pub domain: String,
    /// `ModelProto.model_version`.
    pub model_version: i64,
    /// `ModelProto.doc_string`.
    pub doc_string: Option<String>,
    /// `GraphProto.name` for the top-level graph.
    pub graph_name: String,
    /// `ModelProto.metadata_props` (`key → value`), emitted in order.
    pub metadata_props: Vec<(String, String)>,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            ir_version: DEFAULT_IR_VERSION,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: None,
            graph_name: String::new(),
            metadata_props: Vec::new(),
        }
    }
}

/// An IR [`Graph`] bundled with the model-level metadata and live weight bytes
/// needed to encode a complete ONNX `ModelProto`.
///
/// Construct with [`Model::new`] and refine via [`Model::with_metadata`] /
/// [`Model::with_weights`]. A [`WeightStore`] is required only when the graph
/// has `External`-backed initializers (inline initializers carry their own
/// bytes).
pub struct Model<'a> {
    /// The graph to encode.
    pub graph: &'a Graph,
    /// Model-level metadata (see [`ModelMetadata`]).
    pub metadata: ModelMetadata,
    /// Live weight store backing any [`WeightRef::External`] initializers.
    pub weights: Option<&'a WeightStore>,
}

impl<'a> Model<'a> {
    /// A model over `graph` with default metadata and no external weight store.
    pub fn new(graph: &'a Graph) -> Self {
        Self {
            graph,
            metadata: ModelMetadata::default(),
            weights: None,
        }
    }

    /// Attach model-level metadata.
    pub fn with_metadata(mut self, metadata: ModelMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Attach the live [`WeightStore`] backing external initializers.
    pub fn with_weights(mut self, weights: &'a WeightStore) -> Self {
        self.weights = Some(weights);
        self
    }
}

/// Encode `model` into serialized ONNX protobuf bytes.
pub fn encode_model(model: &Model) -> Result<Vec<u8>, LoaderError> {
    Ok(encode_model_proto(model)?.encode_to_vec())
}

/// Encode `model` and write the serialized bytes to `path`.
pub fn write_model(model: &Model, path: impl AsRef<Path>) -> Result<(), LoaderError> {
    let bytes = encode_model(model)?;
    let path = path.as_ref();
    std::fs::write(path, bytes).map_err(|source| LoaderError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Build the [`ModelProto`] for `model` without serialising it (useful when the
/// caller wants to mutate the proto before encoding, e.g. the §55.4 writer
/// splicing in `EPContext` nodes).
pub fn encode_model_proto(model: &Model) -> Result<ModelProto, LoaderError> {
    let meta = &model.metadata;
    let graph = encode_graph_proto(model.graph, model.weights, true, &meta.graph_name)?;

    // Opset imports sorted by domain for deterministic output.
    let mut opset_import: Vec<OperatorSetIdProto> = model
        .graph
        .opset_imports
        .iter()
        .map(|(domain, &version)| OperatorSetIdProto {
            domain: domain.clone(),
            version: version as i64,
        })
        .collect();
    opset_import.sort_by(|a, b| a.domain.cmp(&b.domain));

    let metadata_props = meta
        .metadata_props
        .iter()
        .map(|(key, value)| StringStringEntryProto {
            key: key.clone(),
            value: value.clone(),
        })
        .collect();

    Ok(ModelProto {
        ir_version: meta.ir_version,
        opset_import,
        producer_name: meta.producer_name.clone(),
        producer_version: meta.producer_version.clone(),
        domain: meta.domain.clone(),
        model_version: meta.model_version,
        doc_string: meta.doc_string.clone().unwrap_or_default(),
        graph: Some(graph),
        metadata_props,
        ..Default::default()
    })
}

/// Encode a [`Graph`] into a `GraphProto`. `is_top_level` reserved for future
/// subgraph-specific behaviour; currently the same fields are emitted for both.
fn encode_graph_proto(
    graph: &Graph,
    weights: Option<&WeightStore>,
    _is_top_level: bool,
    name: &str,
) -> Result<GraphProto, LoaderError> {
    // 1. Initializers, ordered by value id for determinism.
    let mut init_ids: Vec<ValueId> = graph.initializers.keys().copied().collect();
    init_ids.sort_by_key(|v| v.0);
    let mut initializer = Vec::with_capacity(init_ids.len());
    for vid in &init_ids {
        let weight = &graph.initializers[vid];
        let iname = value_name(graph, *vid).unwrap_or_default().to_string();
        initializer.push(encode_weight(iname, weight, weights)?);
    }

    // 2. Graph inputs / outputs as ValueInfoProtos.
    let input: Vec<ValueInfoProto> = graph
        .inputs
        .iter()
        .map(|&vid| encode_value_info(graph, vid))
        .collect();
    let output: Vec<ValueInfoProto> = graph
        .outputs
        .iter()
        .map(|&vid| encode_value_info(graph, vid))
        .collect();

    // 3. Interior value_info: every named value that is not a graph input,
    //    output, or initializer. Anonymous values (skipped optional outputs,
    //    unnamed SSA edges) carry no name and are omitted.
    let mut excluded: HashSet<ValueId> = HashSet::new();
    excluded.extend(graph.inputs.iter().copied());
    excluded.extend(graph.outputs.iter().copied());
    excluded.extend(init_ids.iter().copied());
    let mut value_info = Vec::new();
    for (vid, value) in graph.values.iter() {
        if excluded.contains(&vid) {
            continue;
        }
        if value.name.as_deref().is_some_and(|n| !n.is_empty()) {
            value_info.push(encode_value_info(graph, vid));
        }
    }

    // 4. Nodes, in ascending node-id (== load) order.
    let mut node = Vec::with_capacity(graph.num_nodes());
    for (_, n) in graph.nodes.iter() {
        node.push(encode_node(graph, n)?);
    }

    Ok(GraphProto {
        node,
        name: name.to_string(),
        initializer,
        input,
        output,
        value_info,
        ..Default::default()
    })
}

/// Encode a single node into a `NodeProto`.
fn encode_node(graph: &Graph, node: &Node) -> Result<NodeProto, LoaderError> {
    let input: Vec<String> = node
        .inputs
        .iter()
        .map(|slot| match slot {
            Some(vid) => value_name(graph, *vid).unwrap_or_default().to_string(),
            None => String::new(),
        })
        .collect();
    let output: Vec<String> = node
        .outputs
        .iter()
        .map(|&vid| value_name(graph, vid).unwrap_or_default().to_string())
        .collect();

    let is_ep_ctx = crate::epcontext::is_ep_context_op(&node.op_type, &node.domain);

    // Sort attributes by name so the encoding is deterministic (the IR stores
    // them in a HashMap).
    let mut keys: Vec<&String> = node.attributes.keys().collect();
    keys.sort();
    let mut attribute = Vec::with_capacity(keys.len());
    for key in keys {
        attribute.push(encode_attribute(graph, key, &node.attributes[key], is_ep_ctx)?);
    }

    Ok(NodeProto {
        input,
        output,
        name: String::new(),
        op_type: node.op_type.clone(),
        domain: node.domain.clone(),
        attribute,
        doc_string: node.doc_string.clone().unwrap_or_default(),
        ..Default::default()
    })
}

/// Encode one IR [`Attribute`] into an `AttributeProto`, setting the `type`
/// discriminator to match the populated field (required for IR ≥ 0.0.2).
fn encode_attribute(
    graph: &Graph,
    name: &str,
    attr: &Attribute,
    is_ep_ctx: bool,
) -> Result<AttributeProto, LoaderError> {
    let mut ap = AttributeProto {
        name: name.to_string(),
        ..Default::default()
    };
    match attr {
        Attribute::Int(v) => {
            ap.i = *v;
            ap.r#type = AttributeType::Int as i32;
        }
        Attribute::Float(v) => {
            ap.f = *v;
            ap.r#type = AttributeType::Float as i32;
        }
        Attribute::String(s) => {
            ap.s = s.clone().into_bytes();
            ap.r#type = AttributeType::String as i32;
        }
        Attribute::Ints(v) => {
            ap.ints = v.clone();
            ap.r#type = AttributeType::Ints as i32;
        }
        Attribute::Floats(v) => {
            ap.floats = v.clone();
            ap.r#type = AttributeType::Floats as i32;
        }
        Attribute::Strings(v) => {
            ap.strings = v.iter().map(|s| s.clone().into_bytes()).collect();
            ap.r#type = AttributeType::Strings as i32;
        }
        Attribute::Tensor(t) => {
            // Inverse of the load path's EPContext special-case: an
            // `ep_cache_context` payload is held in the IR as an opaque UINT8
            // tensor but is an ONNX STRING attribute on the wire. Emit its bytes
            // back into `AttributeProto.s` byte-exactly (§55.3/§55.4).
            if is_ep_ctx && name == "ep_cache_context" && t.dtype == DataType::Uint8 {
                ap.s = t.data.clone();
                ap.r#type = AttributeType::String as i32;
            } else {
                ap.t = Some(encode_tensor(t));
                ap.r#type = AttributeType::Tensor as i32;
            }
        }
        Attribute::Graph(g) => {
            ap.g = Some(encode_graph_proto(g, None, false, "")?);
            ap.r#type = AttributeType::Graph as i32;
        }
        Attribute::Graphs(gs) => {
            ap.graphs = gs
                .iter()
                .map(|g| encode_graph_proto(g, None, false, ""))
                .collect::<Result<_, _>>()?;
            ap.r#type = AttributeType::Graphs as i32;
        }
        Attribute::TypeProto(tp) => {
            ap.tp = Some(encode_type_proto(graph, tp));
            ap.r#type = AttributeType::TypeProto as i32;
        }
        // The IR carries a SparseTensor attribute variant, but the load path
        // never builds one (it errors on sparse attribute kinds). Surface a
        // clean error rather than emit a malformed proto.
        Attribute::SparseTensor(_) => {
            return Err(LoaderError::GraphBuild(format!(
                "attribute {name:?}: SparseTensor encoding is unsupported"
            )));
        }
    }
    Ok(ap)
}

/// Encode a [`TensorData`] into a `TensorProto`, preserving raw little-endian
/// bytes (or `STRING` payloads) byte-exactly.
fn encode_tensor(t: &TensorData) -> TensorProto {
    let mut tp = TensorProto {
        dims: t.dims.iter().map(|&d| d as i64).collect(),
        data_type: t.dtype.to_onnx(),
        name: t.name.clone().unwrap_or_default(),
        ..Default::default()
    };
    if t.dtype == DataType::String {
        tp.string_data = t.strings.iter().map(|s| s.clone().into_bytes()).collect();
    } else {
        tp.raw_data = t.data.clone();
    }
    tp
}

/// Encode an initializer [`WeightRef`] into a `TensorProto` named `name`.
///
/// Inline weights are emitted directly; external weights are materialised inline
/// (as `raw_data`) from the provided [`WeightStore`]. Preserving the external
/// `data_location` reference on write is a follow-up (see the decision note).
fn encode_weight(
    name: String,
    weight: &WeightRef,
    weights: Option<&WeightStore>,
) -> Result<TensorProto, LoaderError> {
    match weight {
        WeightRef::Inline(t) => {
            let mut tp = encode_tensor(t);
            tp.name = name;
            Ok(tp)
        }
        WeightRef::External { dtype, dims, .. } => {
            if *dtype == DataType::String {
                return Err(LoaderError::GraphBuild(format!(
                    "external initializer {name:?}: STRING external data is unsupported"
                )));
            }
            let bytes = weights.and_then(|s| s.bytes(weight)).ok_or_else(|| {
                LoaderError::GraphBuild(format!(
                    "external initializer {name:?}: weight bytes unavailable \
                     (attach a WeightStore via Model::with_weights)"
                ))
            })?;
            Ok(TensorProto {
                name,
                data_type: dtype.to_onnx(),
                dims: dims.iter().map(|&d| d as i64).collect(),
                raw_data: bytes.to_vec(),
                ..Default::default()
            })
        }
    }
}

/// Encode a value's `(dtype, shape)` into a `ValueInfoProto`.
fn encode_value_info(graph: &Graph, vid: ValueId) -> ValueInfoProto {
    let value = graph.value(vid);
    ValueInfoProto {
        name: value.name.clone().unwrap_or_default(),
        r#type: Some(encode_tensor_type(graph, value.dtype, &value.shape)),
        ..Default::default()
    }
}

/// Build a tensor `TypeProto` from an element type and shape.
fn encode_tensor_type(graph: &Graph, dtype: DataType, shape: &Shape) -> onnx::TypeProto {
    onnx::TypeProto {
        value: Some(type_proto::Value::TensorType(type_proto::Tensor {
            elem_type: dtype.to_onnx(),
            shape: Some(encode_shape(graph, shape)),
        })),
        ..Default::default()
    }
}

/// Encode an IR [`Shape`] into a `TensorShapeProto`. Static dims become
/// `dim_value`; symbolic dims are re-emitted by their interned name as
/// `dim_param`, or as a valueless (unknown) dimension when unnamed.
fn encode_shape(graph: &Graph, shape: &Shape) -> TensorShapeProto {
    use tensor_shape_proto::{dimension::Value as DV, Dimension};
    let dim = shape
        .iter()
        .map(|d| {
            let value = match d {
                Dim::Static(n) => Some(DV::DimValue(*n as i64)),
                Dim::Symbolic(sym) => graph
                    .symbol_constraints
                    .get(sym)
                    .and_then(|c| c.name.clone())
                    .map(DV::DimParam),
            };
            Dimension {
                value,
                ..Default::default()
            }
        })
        .collect();
    TensorShapeProto { dim }
}

/// Encode an IR [`TypeProto`] (the inverse of `graph_builder::convert_type_proto`).
fn encode_type_proto(graph: &Graph, tp: &TypeProto) -> onnx::TypeProto {
    let value = match tp {
        TypeProto::Tensor { dtype, shape } => type_proto::Value::TensorType(type_proto::Tensor {
            elem_type: dtype.to_onnx(),
            shape: Some(encode_shape(graph, shape)),
        }),
        TypeProto::SparseTensor { dtype, shape } => {
            type_proto::Value::SparseTensorType(type_proto::SparseTensor {
                elem_type: dtype.to_onnx(),
                shape: Some(encode_shape(graph, shape)),
            })
        }
        TypeProto::Sequence(inner) => type_proto::Value::SequenceType(Box::new(type_proto::Sequence {
            elem_type: Some(Box::new(encode_type_proto(graph, inner))),
        })),
        TypeProto::Optional(inner) => type_proto::Value::OptionalType(Box::new(type_proto::Optional {
            elem_type: Some(Box::new(encode_type_proto(graph, inner))),
        })),
        TypeProto::Map { key, value } => type_proto::Value::MapType(Box::new(type_proto::Map {
            key_type: key.to_onnx(),
            value_type: Some(Box::new(encode_type_proto(graph, value))),
        })),
    };
    onnx::TypeProto {
        value: Some(value),
        ..Default::default()
    }
}

/// The name of a graph value, if it has one.
fn value_name(graph: &Graph, vid: ValueId) -> Option<&str> {
    graph.try_value(vid).and_then(|v| v.name.as_deref())
}
