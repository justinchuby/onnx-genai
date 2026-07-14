//! Build an [`onnx_runtime_ir::Graph`] from a decoded `ModelProto` (§19.1).
//!
//! Responsible for the graph-construction invariants of `docs/ORT2.md` §3.5:
//! stable value ids, unique node outputs (SSA), source values for inputs and
//! initializers, and interning symbolic dims that share a protobuf name.

use std::collections::HashMap;

use onnx_runtime_ir::{
    Attribute, DataType, Dim, Graph, Node, NodeId, Shape, TensorData, TypeProto, ValueId,
};

use crate::proto::onnx::{
    self, attribute_proto, tensor_shape_proto, type_proto, AttributeProto, GraphProto, ModelProto,
    TensorShapeProto,
};
use crate::weights::tensor_data_from_proto;
use crate::LoaderError;

/// The result of building a graph: the IR graph plus the mapping from ONNX
/// tensor names to the value ids they were assigned (needed by the weight
/// loader).
pub struct BuiltGraph {
    pub graph: Graph,
    pub name_map: HashMap<String, ValueId>,
}

/// Build the IR graph (nodes, values, symbols, opsets) from a `ModelProto`.
///
/// Weights and shape inference are applied by later pipeline stages.
pub fn build_graph(model: &ModelProto) -> Result<BuiltGraph, LoaderError> {
    let mut graph = Graph::new();

    // Opset imports: domain -> version.
    for opset in &model.opset_import {
        if opset.version > 0 {
            graph
                .opset_imports
                .insert(opset.domain.clone(), opset.version as u64);
        }
    }

    let graph_proto = model
        .graph
        .as_ref()
        .ok_or_else(|| LoaderError::GraphBuild("ModelProto has no graph".into()))?;

    let name_map = build_graph_proto(&mut graph, graph_proto, true)?;

    graph
        .validate()
        .map_err(|errs| LoaderError::GraphBuild(format!("{errs:?}")))?;

    Ok(BuiltGraph { graph, name_map })
}

/// Populate `graph` from a `GraphProto`. When `is_top_level` is true, inputs
/// and outputs are registered as graph I/O. Returns the name→value map for the
/// values created in this graph scope.
fn build_graph_proto(
    graph: &mut Graph,
    gp: &GraphProto,
    is_top_level: bool,
) -> Result<HashMap<String, ValueId>, LoaderError> {
    let mut names: HashMap<String, ValueId> = HashMap::new();

    // 1. Initializers: fully-typed source values (producer = None).
    for init in &gp.initializer {
        if init.name.is_empty() {
            continue;
        }
        let dtype = decode_dtype(init.data_type, || {
            format!("initializer '{}'", init.name)
        })?;
        let dims_vec: Vec<usize> = init.dims.iter().map(|&d| d.max(0) as usize).collect();
        let shape: Shape = dims_vec.iter().copied().map(Dim::Static).collect();
        let vid = graph.create_named_value(init.name.clone(), dtype, shape);
        names.insert(init.name.clone(), vid);
        // Subgraph (non-top-level) initializers are not visited by the model's
        // top-level weight loader (`weights::load_weights` only walks the root
        // `GraphProto`), so a control-flow body's own constants would otherwise
        // be producer-less values with no data — indistinguishable from an
        // outer-scope capture, and unrunnable. Inline their bytes here so the
        // subgraph is a self-contained runnable graph. Only inline-encoded
        // initializers are supported inside subgraphs; an external-data body
        // initializer is left unbound and surfaces as a clear missing-source
        // error at execution rather than being silently mis-read.
        if !is_top_level
            && init.data_location != crate::proto::onnx::tensor_proto::DataLocation::External as i32
        {
            let td = crate::weights::tensor_data_from_proto(init, dtype, &dims_vec)?;
            graph.set_initializer(vid, onnx_runtime_ir::WeightRef::Inline(td));
        }
    }

    // 2. Graph inputs. Names that are also initializers are constants, not
    //    real graph inputs (invariant §3.5.3). Both the top-level graph and
    //    control-flow subgraph bodies record their formal input signature in
    //    `graph.inputs`, in declared order: a subgraph body's formal parameters
    //    (e.g. Loop's `iter_num`/`cond`/loop-carried, Scan's state/scan slices)
    //    are bound positionally at execution, so their order must survive load.
    for vi in &gp.input {
        if vi.name.is_empty() {
            continue;
        }
        if names.contains_key(&vi.name) {
            continue; // initializer-backed constant input
        }
        let (dtype, shape) = value_info_type(graph, vi)?;
        let vid = graph.create_named_value(vi.name.clone(), dtype, shape);
        names.insert(vi.name.clone(), vid);
        graph.add_input(vid);
    }

    // 3. Declared value_info type hints for interior values.
    for vi in &gp.value_info {
        if vi.name.is_empty() || names.contains_key(&vi.name) {
            continue;
        }
        let (dtype, shape) = value_info_type(graph, vi)?;
        let vid = graph.create_named_value(vi.name.clone(), dtype, shape);
        names.insert(vi.name.clone(), vid);
    }

    // 4. Graph outputs: typed values that a node will later produce by name.
    for vi in &gp.output {
        if vi.name.is_empty() {
            continue;
        }
        if !names.contains_key(&vi.name) {
            let (dtype, shape) = value_info_type(graph, vi)?;
            let vid = graph.create_named_value(vi.name.clone(), dtype, shape);
            names.insert(vi.name.clone(), vid);
        }
    }

    // 5. Nodes: wire inputs/outputs, converting attributes.
    for np in &gp.node {
        let inputs: Vec<Option<ValueId>> = np
            .input
            .iter()
            .map(|name| {
                if name.is_empty() {
                    None
                } else {
                    Some(get_or_create(graph, &mut names, name))
                }
            })
            .collect();

        let outputs: Vec<ValueId> = np
            .output
            .iter()
            .map(|name| {
                if name.is_empty() {
                    // An omitted (unused) optional output: give it an anonymous
                    // value to preserve positional arity.
                    graph.create_value(DataType::Float32, Vec::new())
                } else {
                    get_or_create(graph, &mut names, name)
                }
            })
            .collect();

        let mut node = Node::new(NodeId(0), np.op_type.clone(), inputs, outputs);
        node.name = np.name.clone();
        node.domain = np.domain.clone();
        if !np.doc_string.is_empty() {
            node.doc_string = Some(np.doc_string.clone());
        }
        for ap in &np.attribute {
            if let Some((key, attr)) = convert_attribute(graph, ap)? {
                node.attributes.insert(key, attr);
            }
        }

        let nid = graph.insert_node(node);
        register_subgraphs(graph, nid);
    }

    // 6. Register graph outputs in order. Both the top-level graph and each
    //    control-flow subgraph body record their formal output signature here,
    //    in declared order: a body's outputs (e.g. Loop's
    //    `cond_out`/loop-carried/scan-outputs) are consumed positionally by the
    //    control-flow executor, so the order must survive load.
    for vi in &gp.output {
        if let Some(&vid) = names.get(&vi.name) {
            graph.add_output(vid);
        }
    }

    Ok(names)
}

/// After a node is inserted, move any `Graph`/`Graphs` attribute bodies into the
/// graph's `subgraphs` index so traversal/validation can reach them (§3.3).
fn register_subgraphs(graph: &mut Graph, nid: NodeId) {
    let attrs: Vec<(String, usize)> = graph
        .node(nid)
        .attributes
        .iter()
        .filter_map(|(k, v)| match v {
            Attribute::Graph(_) => Some((k.clone(), 1)),
            Attribute::Graphs(gs) => Some((k.clone(), gs.len())),
            _ => None,
        })
        .collect();
    for (key, count) in attrs {
        match graph.node(nid).attributes.get(&key) {
            Some(Attribute::Graph(g)) => {
                let sub = (**g).clone();
                graph.subgraphs.insert((nid, key), sub);
            }
            Some(Attribute::Graphs(_)) => {
                for i in 0..count {
                    if let Some(Attribute::Graphs(gs)) = graph.node(nid).attributes.get(&key) {
                        let sub = gs[i].clone();
                        graph.subgraphs.insert((nid, format!("{key}[{i}]")), sub);
                    }
                }
            }
            _ => {}
        }
    }
}

/// Fetch the value id for `name`, creating a placeholder value if it does not
/// exist yet (interior SSA value with as-yet-unknown type).
fn get_or_create(
    graph: &mut Graph,
    names: &mut HashMap<String, ValueId>,
    name: &str,
) -> ValueId {
    if let Some(&vid) = names.get(name) {
        return vid;
    }
    let vid = graph.create_named_value(name.to_string(), DataType::Float32, Vec::new());
    names.insert(name.to_string(), vid);
    vid
}

/// Decode a raw ONNX `TensorProto.DataType` integer into an IR [`DataType`],
/// failing closed when the runtime does not model the type. This prevents an
/// unmodeled dtype (e.g. `COMPLEX64` = 14) from being silently mislabeled as
/// `Float32` at any tensor-type decode site (§19.1).
fn decode_dtype(
    raw: i32,
    context: impl FnOnce() -> String,
) -> Result<DataType, LoaderError> {
    DataType::from_onnx(raw).ok_or_else(|| LoaderError::UnsupportedDataType {
        raw,
        context: context(),
    })
}

/// Extract `(dtype, shape)` from a `ValueInfoProto`'s tensor type, interning
/// symbolic dims into `graph` by name.
fn value_info_type(
    graph: &mut Graph,
    vi: &onnx::ValueInfoProto,
) -> Result<(DataType, Shape), LoaderError> {
    match vi.r#type.as_ref() {
        Some(tp) => type_proto_to_dtype_shape(graph, tp, &vi.name),
        // A value-info with no type at all is genuinely untyped, not an
        // unmodeled dtype: keep the tensor-centric placeholder default.
        None => Ok((DataType::Float32, Vec::new())),
    }
}

fn type_proto_to_dtype_shape(
    graph: &mut Graph,
    tp: &onnx::TypeProto,
    name: &str,
) -> Result<(DataType, Shape), LoaderError> {
    match tp.value.as_ref() {
        Some(type_proto::Value::TensorType(t)) => {
            let dtype = decode_dtype(t.elem_type, || format!("value-info '{name}'"))?;
            let shape = t
                .shape
                .as_ref()
                .map(|s| tensor_shape_to_shape(graph, s))
                .unwrap_or_default();
            Ok((dtype, shape))
        }
        Some(type_proto::Value::SparseTensorType(t)) => {
            let dtype = decode_dtype(t.elem_type, || format!("value-info '{name}'"))?;
            let shape = t
                .shape
                .as_ref()
                .map(|s| tensor_shape_to_shape(graph, s))
                .unwrap_or_default();
            Ok((dtype, shape))
        }
        // Non-tensor containers (sequence/map/optional): the IR value model is
        // tensor-centric; record a placeholder type. These do not occur in the
        // Phase-1 (BERT) op set.
        _ => Ok((DataType::Float32, Vec::new())),
    }
}

/// Convert an ONNX `TensorShapeProto` to an IR [`Shape`], interning dim-params
/// by name (invariant §3.5.4) and allocating fresh anonymous symbols for
/// unknown dims.
fn tensor_shape_to_shape(graph: &mut Graph, tsp: &TensorShapeProto) -> Shape {
    tsp.dim
        .iter()
        .map(|d| match d.value.as_ref() {
            Some(tensor_shape_proto::dimension::Value::DimValue(v)) if *v >= 0 => {
                Dim::Static(*v as usize)
            }
            Some(tensor_shape_proto::dimension::Value::DimParam(name)) if !name.is_empty() => {
                Dim::Symbolic(graph.intern_symbol(name))
            }
            // Unknown dim (no value, negative, or empty param): fresh symbol.
            _ => Dim::Symbolic(graph.create_symbol(None)),
        })
        .collect()
}

/// Convert an `AttributeProto` to an IR `(name, Attribute)`. Returns `Ok(None)`
/// for empty/absent attributes, and errors (rather than silently dropping) on
/// tensor/sparse-tensor/type-proto **list** kinds the IR does not model, which
/// do not appear in the Phase-1 op set.
fn convert_attribute(
    graph: &mut Graph,
    ap: &AttributeProto,
) -> Result<Option<(String, Attribute)>, LoaderError> {
    use attribute_proto::AttributeType as AT;

    // Determine the attribute kind, falling back to field-presence heuristics
    // for IR<0.0.2 protos where `type` may be unset.
    let ty = AT::try_from(ap.r#type).unwrap_or(AT::Undefined);

    let attr = match ty {
        AT::Float => Attribute::Float(ap.f),
        AT::Int => Attribute::Int(ap.i),
        // STRING attributes are arbitrary byte strings on the wire (an opaque
        // blob, a path, or text). Preserve the exact bytes rather than lossily
        // decoding as UTF-8, so encode is a byte-exact inverse of decode.
        AT::String => Attribute::String(ap.s.clone()),
        AT::Floats => Attribute::Floats(ap.floats.clone()),
        AT::Ints => Attribute::Ints(ap.ints.clone()),
        AT::Strings => Attribute::Strings(ap.strings.clone()),
        AT::Tensor => match ap.t.as_ref() {
            Some(t) => Attribute::Tensor(convert_tensor(t)?),
            None => return Ok(None),
        },
        AT::Graph => match ap.g.as_ref() {
            Some(g) => {
                let mut sub = Graph::new();
                build_graph_proto(&mut sub, g, false)?;
                Attribute::Graph(Box::new(sub))
            }
            None => return Ok(None),
        },
        AT::Graphs => {
            let mut subs = Vec::new();
            for g in &ap.graphs {
                let mut sub = Graph::new();
                build_graph_proto(&mut sub, g, false)?;
                subs.push(sub);
            }
            Attribute::Graphs(subs)
        }
        AT::TypeProto => match ap.tp.as_ref() {
            Some(tp) => Attribute::TypeProto(convert_type_proto(graph, tp)?),
            None => return Ok(None),
        },
        // Field-presence fallback when `type` is UNDEFINED.
        AT::Undefined => {
            if let Some(t) = ap.t.as_ref() {
                Attribute::Tensor(convert_tensor(t)?)
            } else if !ap.floats.is_empty() {
                Attribute::Floats(ap.floats.clone())
            } else if !ap.ints.is_empty() {
                Attribute::Ints(ap.ints.clone())
            } else if !ap.strings.is_empty() {
                Attribute::Strings(ap.strings.clone())
            } else if !ap.s.is_empty() {
                Attribute::String(ap.s.clone())
            } else if ap.i != 0 {
                Attribute::Int(ap.i)
            } else if ap.f != 0.0 {
                Attribute::Float(ap.f)
            } else {
                return Ok(None);
            }
        }
        // Tensor / sparse-tensor / type-proto list attributes have no IR
        // variant. Surface a clean error rather than silently dropping the
        // attribute, which could otherwise mis-model an op (§19.1).
        AT::Tensors | AT::SparseTensor | AT::SparseTensors | AT::TypeProtos => {
            return Err(LoaderError::GraphBuild(format!(
                "attribute {:?} has unmodeled type {:?}",
                ap.name, ty
            )));
        }
    };
    Ok(Some((ap.name.clone(), attr)))
}

fn convert_tensor(t: &onnx::TensorProto) -> Result<TensorData, LoaderError> {
    let dtype = decode_dtype(t.data_type, || format!("attribute tensor '{}'", t.name))?;
    let dims: Vec<usize> = t.dims.iter().map(|&d| d.max(0) as usize).collect();
    tensor_data_from_proto(t, dtype, &dims)
}

fn convert_type_proto(
    graph: &mut Graph,
    tp: &onnx::TypeProto,
) -> Result<TypeProto, LoaderError> {
    let ty = match tp.value.as_ref() {
        Some(type_proto::Value::TensorType(t)) => {
            let dtype = decode_dtype(t.elem_type, || "type-proto attribute (tensor)".to_string())?;
            let shape = t
                .shape
                .as_ref()
                .map(|s| tensor_shape_to_shape(graph, s))
                .unwrap_or_default();
            TypeProto::Tensor { dtype, shape }
        }
        Some(type_proto::Value::SparseTensorType(t)) => {
            let dtype =
                decode_dtype(t.elem_type, || "type-proto attribute (sparse tensor)".to_string())?;
            let shape = t
                .shape
                .as_ref()
                .map(|s| tensor_shape_to_shape(graph, s))
                .unwrap_or_default();
            TypeProto::SparseTensor { dtype, shape }
        }
        Some(type_proto::Value::SequenceType(s)) => {
            let inner = s
                .elem_type
                .as_ref()
                .map(|e| convert_type_proto(graph, e))
                .transpose()?
                .unwrap_or(TypeProto::Tensor {
                    dtype: DataType::Float32,
                    shape: Vec::new(),
                });
            TypeProto::Sequence(Box::new(inner))
        }
        Some(type_proto::Value::OptionalType(o)) => {
            let inner = o
                .elem_type
                .as_ref()
                .map(|e| convert_type_proto(graph, e))
                .transpose()?
                .unwrap_or(TypeProto::Tensor {
                    dtype: DataType::Float32,
                    shape: Vec::new(),
                });
            TypeProto::Optional(Box::new(inner))
        }
        Some(type_proto::Value::MapType(m)) => {
            let key = decode_dtype(m.key_type, || "type-proto attribute (map key)".to_string())?;
            let value = m
                .value_type
                .as_ref()
                .map(|e| convert_type_proto(graph, e))
                .transpose()?
                .unwrap_or(TypeProto::Tensor {
                    dtype: DataType::Float32,
                    shape: Vec::new(),
                });
            TypeProto::Map {
                key,
                value: Box::new(value),
            }
        }
        None => TypeProto::Tensor {
            dtype: DataType::Float32,
            shape: Vec::new(),
        },
    };
    Ok(ty)
}
