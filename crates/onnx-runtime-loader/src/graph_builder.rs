//! Build an [`onnx_runtime_ir::Graph`] from a decoded `ModelProto` (§19.1).
//!
//! Responsible for the graph-construction invariants of `docs/architecture/ORT2.md` §3.5:
//! stable value ids, unique node outputs (SSA), source values for inputs and
//! initializers, and interning symbolic dims that share a protobuf name.

use std::collections::HashMap;

use onnx_runtime_ir::{
    Attribute, DataType, Dim, Graph, ModelFunction, ModelFunctionKey, Node, NodeId, Shape,
    TensorData, TypeProto, ValueId, normalize_domain,
};

use crate::LoaderError;
use crate::proto::onnx::{
    self, AttributeProto, GraphProto, ModelProto, TensorShapeProto, attribute_proto,
    tensor_shape_proto, type_proto,
};
use crate::weights::tensor_data_from_proto;

/// The result of building a graph: the IR graph plus the mapping from ONNX
/// tensor names to the value ids they were assigned (needed by the weight
/// loader).
pub(crate) struct BuiltGraph {
    pub(crate) graph: Graph,
    pub(crate) name_map: HashMap<String, ValueId>,
}

pub(crate) enum FunctionLoading<'a> {
    InlineAll,
    InlineFiltered(&'a crate::function_inline::KeepAsOp<'a>),
    Preserve,
}

/// Build the IR graph (nodes, values, symbols, opsets) from a `ModelProto`.
///
/// Weights and shape inference are applied by later pipeline stages.
///
/// The graph returned here is **structurally incomplete**: top-level
/// initializers are created as named values, but their [`WeightRef`] source
/// data is attached later by the weight loader
/// ([`crate::weights::load_weights`] → `Graph::set_initializer`). Structural
/// validation ([`Graph::validate`]) is therefore deliberately deferred to the
/// full pipeline (see `build_from_bytes_with_weights`), where it runs *after*
/// initializers are registered — otherwise a legal initializer that is also a
/// graph output (or a pre-IR-4 input that is also an initializer) would be
/// mis-flagged as a producer-less value, because `Graph::validate` recognizes
/// an initializer as a valid value source only once it appears in
/// `graph.initializers`.
///
/// [`WeightRef`]: onnx_runtime_ir::WeightRef
/// [`Graph::validate`]: onnx_runtime_ir::Graph::validate
pub(crate) fn build_graph(
    model: &ModelProto,
    function_loading: FunctionLoading<'_>,
) -> Result<BuiltGraph, LoaderError> {
    let model_functions = build_model_function_catalog(model)?;
    // Runtime loading explicitly lowers function calls. Fidelity loading keeps
    // the source call graph intact and carries the declarations in the IR.
    let inlined = match function_loading {
        FunctionLoading::InlineAll => crate::function_inline::inline_functions(model)?,
        FunctionLoading::InlineFiltered(keep) => {
            crate::function_inline::inline_functions_filtered(model, keep)?
        }
        FunctionLoading::Preserve => std::borrow::Cow::Borrowed(model),
    };
    let model = inlined.as_ref();

    let mut graph = Graph::new();
    graph.ir_version = Some(model.ir_version);
    graph.model_functions = model_functions;

    graph.opset_imports = import_opsets(&model.opset_import, "model")?;

    let graph_proto = model
        .graph
        .as_ref()
        .ok_or_else(|| LoaderError::GraphBuild("ModelProto has no graph".into()))?;

    let name_map = build_graph_proto(&mut graph, graph_proto, true)?;

    Ok(BuiltGraph { graph, name_map })
}

fn build_model_function_catalog(
    model: &ModelProto,
) -> Result<HashMap<ModelFunctionKey, ModelFunction>, LoaderError> {
    let mut functions: HashMap<ModelFunctionKey, ModelFunction> = HashMap::new();
    for func in &model.functions {
        let key = ModelFunctionKey::from_function(func);
        if functions.contains_key(&key) {
            return Err(LoaderError::DuplicateModelFunction {
                domain: key.domain,
                name: key.name,
                overload: key.overload,
            });
        }

        validate_function_attribute_bindings(func)?;
        let value_info: HashMap<&str, &onnx::ValueInfoProto> = func
            .value_info
            .iter()
            .map(|value| (value.name.as_str(), value))
            .collect();
        let gp = GraphProto {
            input: func
                .input
                .iter()
                .map(|name| {
                    value_info.get(name.as_str()).map_or_else(
                        || onnx::ValueInfoProto {
                            name: name.clone(),
                            ..Default::default()
                        },
                        |value| (*value).clone(),
                    )
                })
                .collect(),
            output: func
                .output
                .iter()
                .map(|name| {
                    value_info.get(name.as_str()).map_or_else(
                        || onnx::ValueInfoProto {
                            name: name.clone(),
                            ..Default::default()
                        },
                        |value| (*value).clone(),
                    )
                })
                .collect(),
            value_info: func.value_info.clone(),
            node: func.node.clone(),
            ..Default::default()
        };
        let mut body = build_subgraph(&gp, Some(model.ir_version))?;
        body.opset_imports = import_opsets(&func.opset_import, &function_label(&key))?;
        onnx_runtime_ir::pattern::MatchSession::new(&body).map_err(|error| {
            LoaderError::GraphBuild(format!(
                "model-local function {} has invalid graph structure: {error}",
                function_label(&key)
            ))
        })?;
        functions.insert(
            key,
            ModelFunction {
                declaration: func.clone(),
                body,
            },
        );
    }
    Ok(functions)
}

fn import_opsets(
    imports: &[onnx::OperatorSetIdProto],
    context: &str,
) -> Result<HashMap<String, u64>, LoaderError> {
    let mut resolved = HashMap::new();
    for import in imports {
        let domain = normalize_domain(&import.domain).to_string();
        if import.version <= 0 {
            return Err(LoaderError::GraphBuild(format!(
                "{context}: domain {:?} imports invalid opset {}; declare a positive version",
                import.domain, import.version
            )));
        }
        if resolved.insert(domain, import.version as u64).is_some() {
            return Err(LoaderError::GraphBuild(format!(
                "{context}: domain {:?} has duplicate opset imports; declare each canonical domain once",
                import.domain
            )));
        }
    }
    Ok(resolved)
}

fn validate_function_attribute_bindings(func: &onnx::FunctionProto) -> Result<(), LoaderError> {
    let key = ModelFunctionKey::from_function(func);
    let function = function_label(&key);
    let mut formals = std::collections::HashSet::new();
    for name in &func.attribute {
        if name.is_empty() || !formals.insert(name.as_str()) {
            return Err(LoaderError::MalformedFunctionAttribute {
                function,
                detail: format!("required formal attribute {name:?} is empty or duplicated"),
            });
        }
    }
    for default in &func.attribute_proto {
        if default.name.is_empty() || !formals.insert(default.name.as_str()) {
            return Err(LoaderError::MalformedFunctionAttribute {
                function,
                detail: format!(
                    "defaulted formal attribute {:?} is empty, duplicated, or also required",
                    default.name
                ),
            });
        }
    }

    fn validate_graph_refs(
        graph: &GraphProto,
        formals: &std::collections::HashSet<&str>,
        function: &str,
    ) -> Result<(), LoaderError> {
        for node in &graph.node {
            for attribute in &node.attribute {
                if !attribute.ref_attr_name.is_empty()
                    && !formals.contains(attribute.ref_attr_name.as_str())
                {
                    return Err(LoaderError::UnresolvedFunctionAttributeRef {
                        function: function.to_string(),
                        node: if node.name.is_empty() {
                            format!("{}::{}", node.domain, node.op_type)
                        } else {
                            node.name.clone()
                        },
                        attribute: attribute.name.clone(),
                        reference: attribute.ref_attr_name.clone(),
                    });
                }
                for nested in attribute.g.iter().chain(&attribute.graphs) {
                    validate_graph_refs(nested, formals, function)?;
                }
            }
        }
        Ok(())
    }

    let body = GraphProto {
        node: func.node.clone(),
        ..Default::default()
    };
    validate_graph_refs(&body, &formals, &function)
}

fn function_label(key: &ModelFunctionKey) -> String {
    let domain = if key.domain.is_empty() {
        "ai.onnx"
    } else {
        &key.domain
    };
    if key.overload.is_empty() {
        format!("{domain}::{}", key.name)
    } else {
        format!("{domain}::{}:{}", key.name, key.overload)
    }
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
            return Err(LoaderError::GraphBuild(
                "initializer has no name; assign a unique value name before loading".into(),
            ));
        }
        if names.contains_key(&init.name) {
            return Err(LoaderError::GraphBuild(format!(
                "duplicate initializer name {:?}; assign unique initializer identities",
                init.name
            )));
        }
        let dtype = decode_dtype(init.data_type, || format!("initializer '{}'", init.name))?;
        let dims_vec = checked_dims(&init.dims, &format!("initializer {:?}", init.name))?;
        let shape: Shape = dims_vec.iter().copied().map(Dim::Static).collect();
        let vid = graph.create_named_value(init.name.clone(), dtype, shape);
        names.insert(init.name.clone(), vid);
        // Subgraph (non-top-level) initializers are not visited by the model's
        // top-level weight loader (`weights::load_weights` only walks the root
        // `GraphProto`), so a control-flow body's own constants would otherwise
        // be producer-less values with no data — indistinguishable from an
        // outer-scope capture. Attach inline bytes or a model-relative external
        // descriptor now; the loader resolves nested external mappings later.
        if !is_top_level {
            let weight = if init.data_location == onnx::tensor_proto::DataLocation::External as i32
            {
                crate::weights::external_initializer_reference(
                    init,
                    dtype,
                    dims_vec,
                    std::path::Path::new(""),
                )?
            } else {
                onnx_runtime_ir::WeightRef::Inline(crate::weights::tensor_data_from_proto(
                    init, dtype, &dims_vec,
                )?)
            };
            graph.set_initializer(vid, weight);
        }
    }

    for init in &gp.sparse_initializer {
        let values = init.values.as_ref().ok_or_else(|| {
            LoaderError::GraphBuild(
                "sparse initializer is missing values; provide a named TensorProto".into(),
            )
        })?;
        if values.name.is_empty() || names.contains_key(&values.name) {
            return Err(LoaderError::GraphBuild(format!(
                "sparse initializer name {:?} is empty or duplicated; assign a unique values.name",
                values.name
            )));
        }
        let indices = init.indices.as_ref().ok_or_else(|| {
            LoaderError::GraphBuild(format!(
                "sparse initializer {:?} is missing indices; provide int64 COO indices",
                values.name
            ))
        })?;
        let dtype = decode_dtype(values.data_type, || {
            format!("sparse initializer {:?}", values.name)
        })?;
        let dims = checked_dims(&init.dims, &format!("sparse initializer {:?}", values.name))?;
        checked_dims(&values.dims, &format!("sparse values {:?}", values.name))?;
        checked_dims(&indices.dims, &format!("sparse indices {:?}", values.name))?;
        if dtype == DataType::Undefined
            || indices.data_type != DataType::Int64.to_onnx()
            || values.dims.len() != 1
            || !(indices.dims == values.dims || indices.dims == [values.dims[0], dims.len() as i64])
        {
            return Err(LoaderError::GraphBuild(format!(
                "sparse initializer {:?} has invalid dtype/COO shapes; use values [nnz] and int64 indices [nnz] or [nnz,rank]",
                values.name
            )));
        }
        validate_sparse_payload(init, &dims)?;
        let vid = graph.create_named_value(
            values.name.clone(),
            dtype,
            dims.into_iter().map(Dim::Static).collect(),
        );
        names.insert(values.name.clone(), vid);
        graph.sparse_initializers.insert(vid, init.clone());
    }

    // 2. Graph inputs, including overridable initializer defaults. Both the top-level graph and
    //    control-flow subgraph bodies record their formal input signature in
    //    `graph.inputs`, in declared order: a subgraph body's formal parameters
    //    (e.g. Loop's `iter_num`/`cond`/loop-carried, Scan's state/scan slices)
    //    are bound positionally at execution, so their order must survive load.
    for vi in &gp.input {
        if vi.name.is_empty() {
            continue;
        }
        if let Some(&vid) = names.get(&vi.name) {
            if graph.inputs.contains(&vid) {
                return Err(LoaderError::GraphBuild(format!(
                    "duplicate graph input {:?}; declare each formal once",
                    vi.name
                )));
            }
            if let Some(tp) = &vi.r#type {
                validate_type_proto(tp, &vi.name, 0)?;
                validate_initializer_type(graph, vid, tp, &vi.name)?;
            }
            if graph.ir_version.is_some_and(|version| version < 4) {
                graph.value_mut(vid).declared_type = vi.r#type.clone();
            } else {
                let (dtype, shape, type_known, shape_known) = value_info_type(graph, vi)?;
                let value = graph.value_mut(vid);
                value.declared_type = vi.r#type.clone();
                value.dtype = dtype;
                value.shape = shape;
                mark_unknown_type_info(graph, vid, type_known, shape_known);
            }
            graph.add_input(vid);
            continue;
        }
        let (dtype, shape, type_known, shape_known) = value_info_type(graph, vi)?;
        let vid = graph.create_named_value(vi.name.clone(), dtype, shape);
        graph.value_mut(vid).declared_type = vi.r#type.clone();
        mark_unknown_type_info(graph, vid, type_known, shape_known);
        names.insert(vi.name.clone(), vid);
        graph.add_input(vid);
    }

    // 3. Declared value_info type hints for interior values.
    for vi in &gp.value_info {
        if vi.name.is_empty() || names.contains_key(&vi.name) {
            continue;
        }
        let (dtype, shape, type_known, shape_known) = value_info_type(graph, vi)?;
        let vid = graph.create_named_value(vi.name.clone(), dtype, shape);
        graph.value_mut(vid).declared_type = vi.r#type.clone();
        mark_unknown_type_info(graph, vid, type_known, shape_known);
        names.insert(vi.name.clone(), vid);
    }

    // 4. Graph outputs: typed values that a node will later produce by name.
    for vi in &gp.output {
        if vi.name.is_empty() {
            continue;
        }
        if !names.contains_key(&vi.name) {
            let (dtype, shape, type_known, shape_known) = value_info_type(graph, vi)?;
            let vid = graph.create_named_value(vi.name.clone(), dtype, shape);
            graph.value_mut(vid).declared_type = vi.r#type.clone();
            mark_unknown_type_info(graph, vid, type_known, shape_known);
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

        let outputs: Vec<Option<ValueId>> = np
            .output
            .iter()
            .map(|name| {
                if name.is_empty() {
                    None
                } else {
                    Some(get_or_create(graph, &mut names, name))
                }
            })
            .collect();

        let mut node = Node::new_with_output_slots(NodeId(0), np.op_type.clone(), inputs, outputs);
        node.name = np.name.clone();
        node.domain = normalize_domain(&np.domain).to_string();
        node.overload = np.overload.clone();
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
pub(crate) fn register_subgraphs(graph: &mut Graph, nid: NodeId) {
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
                graph.subgraphs.insert((nid, key.into()), sub);
            }
            Some(Attribute::Graphs(_)) => {
                for i in 0..count {
                    if let Some(Attribute::Graphs(gs)) = graph.node(nid).attributes.get(&key) {
                        let sub = gs[i].clone();
                        graph.subgraphs.insert(
                            (
                                nid,
                                onnx_runtime_ir::GraphAttribute::element(key.clone(), i),
                            ),
                            sub,
                        );
                    }
                }
            }
            _ => {}
        }
    }
}

/// Fetch the value id for `name`, creating a placeholder value if it does not
/// exist yet (interior SSA value with as-yet-unknown type).
fn get_or_create(graph: &mut Graph, names: &mut HashMap<String, ValueId>, name: &str) -> ValueId {
    if let Some(&vid) = names.get(name) {
        return vid;
    }
    let vid = graph.create_named_value(name.to_string(), DataType::Undefined, Vec::new());
    graph.mark_value_type_unknown(vid);
    graph.mark_value_shape_unknown(vid);
    names.insert(name.to_string(), vid);
    vid
}

fn mark_unknown_type_info(graph: &mut Graph, value: ValueId, type_known: bool, shape_known: bool) {
    if !type_known {
        graph.mark_value_type_unknown(value);
    }
    if !shape_known {
        graph.mark_value_shape_unknown(value);
    }
}

/// Decode a raw ONNX `TensorProto.DataType` integer into an IR [`DataType`],
/// failing closed when the runtime does not model the type. This prevents an
/// unmodeled dtype (e.g. `COMPLEX64` = 14) from being silently mislabeled as
/// `Float32` at any tensor-type decode site (§19.1).
fn decode_dtype(raw: i32, context: impl FnOnce() -> String) -> Result<DataType, LoaderError> {
    DataType::from_onnx(raw).ok_or_else(|| LoaderError::UnsupportedDataType {
        raw,
        context: context(),
    })
}

/// Extract `(dtype, shape, type_known, shape_known)` from a `ValueInfoProto`,
/// interning symbolic dims into `graph` by name.
fn value_info_type(
    graph: &mut Graph,
    vi: &onnx::ValueInfoProto,
) -> Result<(DataType, Shape, bool, bool), LoaderError> {
    match vi.r#type.as_ref() {
        Some(tp) => type_proto_to_dtype_shape(graph, tp, &vi.name),
        None => Ok((DataType::Undefined, Vec::new(), false, false)),
    }
}

fn type_proto_to_dtype_shape(
    graph: &mut Graph,
    tp: &onnx::TypeProto,
    name: &str,
) -> Result<(DataType, Shape, bool, bool), LoaderError> {
    validate_type_proto(tp, &format!("value-info '{name}'"), 0)?;
    match tp.value.as_ref() {
        Some(type_proto::Value::TensorType(t)) => {
            let dtype = decode_dtype(t.elem_type, || format!("value-info '{name}'"))?;
            let shape_known = t.shape.is_some();
            let shape = t
                .shape
                .as_ref()
                .map(|s| tensor_shape_to_shape(graph, s))
                .unwrap_or_default();
            Ok((dtype, shape, dtype != DataType::Undefined, shape_known))
        }
        Some(type_proto::Value::SparseTensorType(_)) => {
            // Sparse values are not dense-tensor evidence.
            Ok((DataType::Undefined, Vec::new(), false, false))
        }
        _ => Ok((DataType::Undefined, Vec::new(), false, false)),
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
/// for empty/absent attributes.
pub(crate) fn convert_attribute(
    graph: &mut Graph,
    ap: &AttributeProto,
) -> Result<Option<(String, Attribute)>, LoaderError> {
    use attribute_proto::AttributeType as AT;

    // Determine the attribute kind, falling back to field-presence heuristics
    // for IR<0.0.2 protos where `type` may be unset.
    let ty = AT::try_from(ap.r#type).map_err(|_| LoaderError::GraphBuild(format!(
        "attribute {:?} declares unknown kind {}; supply a supported ONNX AttributeType instead of guessing its payload",
        ap.name, ap.r#type
    )))?;
    if !ap.ref_attr_name.is_empty() {
        if ty == AT::Undefined {
            return Err(LoaderError::GraphBuild(format!(
                "attribute {:?} references formal {:?} without a type; specify its ONNX AttributeType",
                ap.name, ap.ref_attr_name
            )));
        }
        return Ok(Some((
            ap.name.clone(),
            Attribute::Ref {
                name: ap.ref_attr_name.clone(),
                kind: ty,
            },
        )));
    }

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
        AT::Tensors => Attribute::Tensors(
            ap.tensors
                .iter()
                .map(convert_tensor)
                .collect::<Result<_, _>>()?,
        ),
        AT::SparseTensor => match ap.sparse_tensor.as_ref() {
            Some(t) => Attribute::SparseTensor(convert_sparse_tensor(t)?),
            None => return Ok(None),
        },
        AT::SparseTensors => Attribute::SparseTensors(
            ap.sparse_tensors
                .iter()
                .map(convert_sparse_tensor)
                .collect::<Result<_, _>>()?,
        ),
        AT::Graph => match ap.g.as_ref() {
            Some(g) => Attribute::Graph(Box::new(build_subgraph(g, graph.ir_version)?)),
            None => return Ok(None),
        },
        AT::Graphs => Attribute::Graphs(
            ap.graphs
                .iter()
                .map(|body| build_subgraph(body, graph.ir_version))
                .collect::<Result<_, _>>()?,
        ),
        AT::TypeProto => match ap.tp.as_ref() {
            Some(tp) => Attribute::TypeProto(convert_type_proto(graph, tp)?),
            None => {
                return Err(LoaderError::GraphBuild(format!(
                    "TYPE_PROTO attribute {:?} is missing tp; supply a TypeProto (empty means unknown)",
                    ap.name
                )));
            }
        },
        AT::TypeProtos => Attribute::TypeProtos(
            ap.type_protos
                .iter()
                .map(|tp| convert_type_proto(graph, tp))
                .collect::<Result<_, _>>()?,
        ),
        // Field-presence fallback when `type` is UNDEFINED.
        AT::Undefined => {
            if let Some(g) = ap.g.as_ref() {
                Attribute::Graph(Box::new(build_subgraph(g, graph.ir_version)?))
            } else if !ap.graphs.is_empty() {
                Attribute::Graphs(
                    ap.graphs
                        .iter()
                        .map(|body| build_subgraph(body, graph.ir_version))
                        .collect::<Result<_, _>>()?,
                )
            } else if let Some(t) = ap.t.as_ref() {
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
    };
    Ok(Some((ap.name.clone(), attr)))
}

fn build_subgraph(gp: &GraphProto, ir_version: Option<i64>) -> Result<Graph, LoaderError> {
    let mut graph = Graph::new();
    graph.ir_version = ir_version;
    build_graph_proto(&mut graph, gp, false)?;
    Ok(graph)
}

fn convert_tensor(t: &onnx::TensorProto) -> Result<TensorData, LoaderError> {
    let dtype = decode_dtype(t.data_type, || format!("attribute tensor '{}'", t.name))?;
    let dims = checked_dims(&t.dims, &format!("attribute tensor {:?}", t.name))?;
    tensor_data_from_proto(t, dtype, &dims)
}

fn convert_sparse_tensor(
    tensor: &onnx::SparseTensorProto,
) -> Result<onnx_runtime_ir::SparseTensorData, LoaderError> {
    let values = tensor
        .values
        .as_ref()
        .ok_or_else(|| LoaderError::GraphBuild("sparse tensor is missing values".into()))
        .and_then(convert_tensor)?;
    let indices = tensor
        .indices
        .as_ref()
        .ok_or_else(|| LoaderError::GraphBuild("sparse tensor is missing indices".into()))
        .and_then(convert_tensor)?;
    Ok(onnx_runtime_ir::SparseTensorData {
        values,
        indices,
        dims: checked_dims(&tensor.dims, "sparse tensor attribute")?,
    })
}

fn checked_dims(dims: &[i64], context: &str) -> Result<Vec<usize>, LoaderError> {
    dims.iter().map(|&dim| usize::try_from(dim).map_err(|_| {
        LoaderError::GraphBuild(format!(
            "{context}: dimension {dim} is negative or exceeds the host index range; repair the concrete tensor shape"
        ))
    })).collect()
}

fn validate_sparse_payload(
    sparse: &onnx::SparseTensorProto,
    dims: &[usize],
) -> Result<(), LoaderError> {
    let values = sparse.values.as_ref().expect("checked sparse values");
    let indices = sparse.indices.as_ref().expect("checked sparse indices");
    let malformed = |detail: &str| {
        LoaderError::GraphBuild(format!(
            "sparse initializer {:?}: {detail}; repair the COO payload and dimensions",
            values.name
        ))
    };
    let nnz = usize::try_from(values.dims[0]).map_err(|_| malformed("negative nnz"))?;
    let total =
        onnx_runtime_ir::checked_numel(dims).ok_or_else(|| malformed("dense shape overflows"))?;
    if nnz > total {
        return Err(malformed("nnz exceeds dense element count"));
    }
    for tensor in [values, indices] {
        if tensor.data_location == onnx::tensor_proto::DataLocation::External as i32 {
            crate::weights::external_initializer_reference(
                tensor,
                decode_dtype(tensor.data_type, || {
                    format!("sparse initializer {:?}", values.name)
                })?,
                checked_dims(&tensor.dims, "external sparse tensor")?,
                std::path::Path::new(""),
            )?;
            continue;
        }
        let data = convert_tensor(tensor)?;
        let valid = if data.dtype == DataType::String {
            data.checked_numel() == Some(data.strings.len())
        } else {
            data.checked_expected_bytes() == Some(data.data.len())
        };
        if !valid {
            return Err(malformed(
                "tensor element count disagrees with its stored payload",
            ));
        }
    }
    if indices.data_location == onnx::tensor_proto::DataLocation::External as i32 {
        return Ok(());
    }
    let data = convert_tensor(indices)?;
    let coordinates = onnx_runtime_ir::read_vec_le::<i64>(&data.data)
        .map_err(|_| malformed("indices are not complete int64 elements"))?;
    if indices.dims.len() == 1 {
        let mut previous = None;
        for coordinate in coordinates {
            let index =
                usize::try_from(coordinate).map_err(|_| malformed("negative linear index"))?;
            if index >= total || previous.is_some_and(|prior| index <= prior) {
                return Err(malformed(
                    "linear indices are out of range, unsorted or duplicated",
                ));
            }
            previous = Some(index);
        }
    } else if !dims.is_empty() {
        let mut previous: Option<&[i64]> = None;
        for coordinate in coordinates.chunks_exact(dims.len()) {
            if coordinate
                .iter()
                .zip(dims)
                .any(|(&index, &dim)| usize::try_from(index).map_or(true, |index| index >= dim))
                || previous.is_some_and(|prior| coordinate <= prior)
            {
                return Err(malformed(
                    "coordinate indices are out of range, unsorted or duplicated",
                ));
            }
            previous = Some(coordinate);
        }
    }
    Ok(())
}

fn validate_initializer_type(
    graph: &Graph,
    value: ValueId,
    tp: &TypeProto,
    name: &str,
) -> Result<(), LoaderError> {
    let Some(kind) = tp.value.as_ref() else {
        return Ok(());
    };
    let type_proto::Value::TensorType(tensor) = kind else {
        return Err(LoaderError::GraphBuild(format!(
            "initializer input {name:?} declares a non-tensor type; declare the dense value type of its initializer default"
        )));
    };
    let default = graph.value(value);
    if tensor.elem_type != DataType::Undefined.to_onnx()
        && tensor.elem_type != default.dtype.to_onnx()
    {
        return Err(LoaderError::GraphBuild(format!(
            "initializer input {name:?} dtype {} disagrees with default {:?}; align the declaration and initializer",
            tensor.elem_type, default.dtype
        )));
    }
    if let Some(shape) = &tensor.shape {
        let compatible = shape.dim.len() == default.shape.len()
            && shape
                .dim
                .iter()
                .zip(&default.shape)
                .all(|(declared, actual)| match (&declared.value, actual) {
                    (Some(tensor_shape_proto::dimension::Value::DimValue(a)), Dim::Static(b)) => {
                        usize::try_from(*a).ok() == Some(*b)
                    }
                    _ => true,
                });
        if !compatible {
            return Err(LoaderError::GraphBuild(format!(
                "initializer input {name:?} shape disagrees with its default {:?}; align the declaration and initializer",
                default.shape
            )));
        }
    }
    Ok(())
}

fn convert_type_proto(_graph: &mut Graph, tp: &onnx::TypeProto) -> Result<TypeProto, LoaderError> {
    validate_type_proto(tp, "TYPE_PROTO attribute", 0)?;
    Ok(tp.clone())
}

fn validate_type_proto(tp: &TypeProto, context: &str, depth: usize) -> Result<(), LoaderError> {
    if depth > 64 {
        return Err(LoaderError::GraphBuild(format!(
            "{context}: recursive type exceeds 64 levels; simplify the type descriptor"
        )));
    }
    let missing = || {
        LoaderError::GraphBuild(format!(
            "{context}: container type is missing its element/value descriptor; supply an explicit TypeProto (empty means unknown)"
        ))
    };
    match tp.value.as_ref() {
        Some(type_proto::Value::TensorType(t)) => {
            decode_dtype(t.elem_type, || context.to_string())?;
            validate_tensor_shape(t.shape.as_ref(), context)?;
        }
        Some(type_proto::Value::SparseTensorType(t)) => {
            decode_dtype(t.elem_type, || context.to_string())?;
            validate_tensor_shape(t.shape.as_ref(), context)?;
        }
        Some(type_proto::Value::SequenceType(t)) => {
            validate_type_proto(
                t.elem_type.as_deref().ok_or_else(missing)?,
                context,
                depth + 1,
            )?;
        }
        Some(type_proto::Value::OptionalType(t)) => {
            validate_type_proto(
                t.elem_type.as_deref().ok_or_else(missing)?,
                context,
                depth + 1,
            )?;
        }
        Some(type_proto::Value::MapType(t)) => {
            if !matches!(t.key_type, 2..=7 | 8 | 12 | 13) {
                return Err(LoaderError::GraphBuild(format!(
                    "{context}: map key type {} is not an ONNX integer or string type; repair key_type",
                    t.key_type
                )));
            }
            validate_type_proto(
                t.value_type.as_deref().ok_or_else(missing)?,
                context,
                depth + 1,
            )?;
        }
        Some(type_proto::Value::OpaqueType(_)) | None => {}
    }
    Ok(())
}

fn validate_tensor_shape(
    shape: Option<&TensorShapeProto>,
    context: &str,
) -> Result<(), LoaderError> {
    if shape.is_some_and(|s| {
        s.dim.iter().any(
            |d| matches!(d.value, Some(tensor_shape_proto::dimension::Value::DimValue(n)) if n < 0),
        )
    }) {
        return Err(LoaderError::GraphBuild(format!(
            "{context}: tensor type declares a negative dimension; use an absent dimension for unknown extent"
        )));
    }
    Ok(())
}
