use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString, c_char, c_void};
use std::path::Path;
use std::ptr;
use std::sync::{Arc, OnceLock};

use memmap2::Mmap;
use onnx_genai_ort_sys as ort;
use onnx_runtime_ir::{Attribute, DataType, Dim, Graph, Node, NodeId, ValueId};

use super::SubgraphClaim;
use super::ffi_helpers::*;
use super::weights::host_ort_value_for_weight;
use crate::error::{EpError, Result};
use crate::provider::EpId;
use crate::tensor::{TensorMut, TensorView};

#[repr(C)]
pub(super) struct HostStatus {
    code: ort::OrtErrorCode,
    msg: CString,
}

#[repr(C)]
pub(super) struct HostTensorTypeAndShapeInfo {
    pub(super) dtype: ort::ONNXTensorElementDataType,
    pub(super) dims: Option<Vec<i64>>,
    symbols: Vec<CString>,
}

impl HostTensorTypeAndShapeInfo {
    pub(super) fn concrete(dtype: DataType, dims: &[usize]) -> std::result::Result<Self, String> {
        Ok(Self { dtype: dtype_to_ort(dtype), dims: Some(dims.iter().map(|&dim| i64::try_from(dim).map_err(|_| format!("tensor dimension {dim} exceeds ORT's i64 range; reduce the dimensions"))).collect::<std::result::Result<_, _>>()?), symbols: Vec::new() })
    }

    fn for_value(
        graph: &Graph,
        value: &onnx_runtime_ir::Value,
    ) -> std::result::Result<Self, String> {
        let shape = value.tensor_shape();
        let symbols = shape.map(|shape| shape.iter().map(|dim| {
            let name = match dim {
                Dim::Symbolic(id) => graph.symbol_constraints.get(id).and_then(|constraint| constraint.name.as_deref()).unwrap_or(""),
                Dim::Static(_) => "",
            };
            CString::new(name).map_err(|_| format!("value {:?} dimension symbol contains an embedded NUL; use native IR compilation for this symbol", value.id))
        }).collect::<std::result::Result<Vec<_>, _>>()).transpose()?.unwrap_or_default();
        let dims = shape.map(|shape| shape.iter().map(|dim| match dim {
            Dim::Static(dim) => i64::try_from(*dim).map_err(|_| format!("value {:?} dimension {dim} exceeds ORT's i64 range; reduce the shape", value.id)),
            Dim::Symbolic(_) => Ok(-1),
        }).collect::<std::result::Result<Vec<_>, _>>()).transpose()?;
        Ok(Self {
            dtype: dtype_to_ort(value.tensor_dtype().unwrap_or(DataType::Undefined)),
            dims,
            symbols,
        })
    }
}

#[repr(C)]
pub(super) struct HostTypeInfo {
    kind: ort::ONNXType,
    tensor: HostTensorTypeAndShapeInfo,
}

impl HostTypeInfo {
    fn for_value(
        graph: &Graph,
        value: &onnx_runtime_ir::Value,
    ) -> std::result::Result<Self, String> {
        use onnx_runtime_ir::onnx::type_proto::Value as Type;
        let kind = match value
            .declared_type
            .as_ref()
            .and_then(|ty| ty.value.as_ref())
        {
            Some(Type::SequenceType(_)) => ort::ONNX_TYPE_SEQUENCE,
            Some(Type::OptionalType(_)) => ort::ONNX_TYPE_OPTIONAL,
            Some(Type::MapType(_)) => ort::ONNX_TYPE_MAP,
            Some(Type::SparseTensorType(_)) => ort::ONNX_TYPE_SPARSETENSOR,
            Some(Type::OpaqueType(_)) => ort::ONNX_TYPE_OPAQUE,
            Some(Type::TensorType(_)) => ort::ONNX_TYPE_TENSOR,
            None if value.tensor_dtype().is_some() || value.tensor_shape().is_some() => {
                ort::ONNX_TYPE_TENSOR
            }
            None => ort::ONNX_TYPE_UNKNOWN,
        };
        Ok(Self {
            kind,
            tensor: HostTensorTypeAndShapeInfo::for_value(graph, value)?,
        })
    }
}

#[repr(C)]
pub(super) struct HostOrtValue {
    pub(super) tensor: HostTensorTypeAndShapeInfo,
    pub(super) storage: HostOrtValueStorage,
}

pub(super) enum HostOrtValueStorage {
    Owned(Vec<u8>),
    Strings(Vec<Vec<u8>>),
    Borrowed {
        ptr: *mut c_void,
        len: usize,
    },
    /// A window onto a mapped weight file.
    ///
    /// Copying instead would mean a heap copy of every initializer for every
    /// subgraph a plugin claims -- on a 0.5B model claimed as 97 subgraphs
    /// that is the whole weight set nearly a hundred times over, which the
    /// kernel answers by killing the process. The `Arc` keeps the mapping
    /// alive for as long as any value points into it.
    Mapped {
        map: Arc<Mmap>,
        offset: usize,
        len: usize,
    },
}

impl HostOrtValue {
    fn data_ptr(&self) -> *const c_void {
        match &self.storage {
            HostOrtValueStorage::Owned(data) => data.as_ptr().cast(),
            HostOrtValueStorage::Strings(_) => ptr::null(),
            HostOrtValueStorage::Borrowed { ptr, .. } => (*ptr).cast_const(),
            HostOrtValueStorage::Mapped { map, offset, .. } => map[*offset..].as_ptr().cast(),
        }
    }

    fn data_mut_ptr(&mut self) -> *mut c_void {
        match &mut self.storage {
            HostOrtValueStorage::Owned(data) => data.as_mut_ptr().cast(),
            HostOrtValueStorage::Strings(_) => ptr::null_mut(),
            HostOrtValueStorage::Borrowed { ptr, .. } => *ptr,
            // No mutable pointer exists for a shared, read-only mapping, and
            // aliasing one mutably to satisfy a caller would be worse than
            // refusing. The accessor turns this into an error status rather
            // than reporting success with a null pointer.
            HostOrtValueStorage::Mapped { .. } => ptr::null_mut(),
        }
    }

    fn byte_len(&self) -> usize {
        match &self.storage {
            HostOrtValueStorage::Owned(data) => data.len(),
            HostOrtValueStorage::Strings(data) => data.iter().map(Vec::len).sum(),
            HostOrtValueStorage::Borrowed { len, .. } => *len,
            HostOrtValueStorage::Mapped { len, .. } => *len,
        }
    }
}

#[repr(C)]
pub(super) struct HostKernelContext {
    inputs: Vec<Option<HostOrtValue>>,
    outputs: Vec<HostOrtValue>,
    requested_outputs: Option<Vec<bool>>,
}

impl HostKernelContext {
    pub(super) fn new(inputs: &[TensorView<'_>], outputs: &mut [TensorMut<'_>]) -> Result<Self> {
        let inputs = inputs
            .iter()
            .map(|input| {
                if input.is_absent() {
                    return Ok(None);
                }
                if !input.device.is_host_accessible() {
                    return Err(EpError::KernelFailed(format!(
                        "plugin fused-subgraph input is on non-host-accessible device {:?}; fix by inserting a host/device copy boundary before plugin dispatch",
                        input.device
                    )));
                }
                if !onnx_runtime_ir::is_contiguous(input.shape, input.strides) {
                    return Err(EpError::KernelFailed(
                        "plugin fused-subgraph input is strided; fix by materializing it before plugin dispatch".into(),
                    ));
                }
                let ptr = (input.data.0 as *mut u8).wrapping_add(input.byte_offset).cast::<c_void>();
                Ok(Some(HostOrtValue {
                    tensor: HostTensorTypeAndShapeInfo::concrete(input.dtype, input.shape).map_err(EpError::KernelFailed)?,
                    storage: HostOrtValueStorage::Borrowed {
                        ptr,
                        len: input.byte_size(),
                    },
                }))
            })
            .collect::<Result<Vec<_>>>()?;
        let outputs = outputs
            .iter_mut()
            .map(|output| {
                if !output.device.is_host_accessible() {
                    return Err(EpError::KernelFailed(format!(
                        "plugin fused-subgraph output is on non-host-accessible device {:?}; fix by selecting a host-accessible plugin device or adding copy-out support",
                        output.device
                    )));
                }
                if !onnx_runtime_ir::is_contiguous(output.shape, output.strides) {
                    return Err(EpError::KernelFailed(
                        "plugin fused-subgraph output is strided; fix by allocating contiguous plugin outputs".into(),
                    ));
                }
                Ok(HostOrtValue {
                    tensor: HostTensorTypeAndShapeInfo::concrete(output.dtype, output.shape).map_err(EpError::KernelFailed)?,
                    storage: HostOrtValueStorage::Borrowed {
                        ptr: output.data.0,
                        len: output.byte_size(),
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            inputs,
            outputs,
            requested_outputs: None,
        })
    }

    pub(super) fn dynamic(
        inputs: &[TensorView<'_>],
        dtypes: &[DataType],
        requested: &[bool],
    ) -> Result<Self> {
        if dtypes.len() != requested.len() {
            return Err(EpError::KernelFailed(format!(
                "plugin dynamic output count mismatch: {} declared, {} requested",
                dtypes.len(),
                requested.len()
            )));
        }
        let mut context = Self::new(inputs, &mut [])?;
        context.outputs = dtypes
            .iter()
            .map(|&dtype| HostOrtValue {
                tensor: HostTensorTypeAndShapeInfo {
                    dtype: dtype_to_ort(dtype),
                    dims: None,
                    symbols: Vec::new(),
                },
                storage: HostOrtValueStorage::Owned(Vec::new()),
            })
            .collect();
        context.requested_outputs = Some(requested.to_vec());
        Ok(context)
    }

    pub(super) fn into_dynamic_outputs(self) -> Result<Vec<Option<crate::KernelSizedOutput>>> {
        let requested = self.requested_outputs.ok_or_else(|| {
            EpError::KernelFailed("plugin context was not allocated for dynamic outputs".into())
        })?;
        self.outputs.into_iter().zip(requested).enumerate().map(|(index, (value, requested))| {
            if !requested { return Ok(None); }
            let dims = value.tensor.dims.ok_or_else(|| EpError::KernelFailed(format!("plugin did not allocate requested dynamic output {index}; call KernelContext_GetOutput during Compute")))?;
            let shape = dims.into_iter().map(|dim| usize::try_from(dim).map_err(|_| EpError::KernelFailed(format!("plugin output {index} has invalid extent {dim}")))).collect::<Result<Vec<_>>>()?;
            let dtype = DataType::from_onnx(value.tensor.dtype as _).ok_or_else(|| EpError::KernelFailed(format!("plugin output {index} has unknown dtype {}", value.tensor.dtype)))?;
            let HostOrtValueStorage::Owned(bytes) = value.storage else { return Err(EpError::KernelFailed(format!("plugin dynamic output {index} does not own its payload"))); };
            Ok(Some(crate::KernelSizedOutput { shape, dtype, bytes }))
        }).collect()
    }

    pub(super) fn byte_size(&self) -> usize {
        self.inputs
            .iter()
            .flatten()
            .map(HostOrtValue::byte_len)
            .chain(self.outputs.iter().map(HostOrtValue::byte_len))
            .sum()
    }
}

#[repr(C)]
pub(super) struct HostValueInfo {
    id: ValueId,
    name: CString,
    type_info: HostTypeInfo,
    initializer: Option<Box<HostOrtValue>>,
    producer: Option<*const ort::OrtNode>,
    is_constant_initializer: bool,
}

#[repr(C)]
pub(super) struct HostOpAttr {
    name: CString,
    attr: Attribute,
}

#[repr(C)]
pub(super) struct HostNode {
    id: NodeId,
    name: CString,
    op_type: CString,
    domain: CString,
    since_version: i32,
    inputs: Vec<*const ort::OrtValueInfo>,
    implicit_inputs: Vec<*const ort::OrtValueInfo>,
    outputs: Vec<*const ort::OrtValueInfo>,
    attrs: Vec<HostOpAttr>,
    subgraphs: Vec<(CString, Box<HostGraph>)>,
    /// Value infos this node's `inputs`/`outputs` point into.
    ///
    /// Held here rather than leaked. The boxes must outlive the pointers we
    /// hand the plugin, and tying them to the node does that exactly: they die
    /// when the node does, which is when the compiled plan is dropped. Leaking
    /// them was also correct in the narrow sense and grew without bound in a
    /// process that builds sessions repeatedly.
    #[allow(clippy::vec_box)]
    owned_values: Vec<Box<HostValueInfo>>,
}

#[repr(C)]
pub(super) struct HostGraph {
    ir_version: i64,
    opsets: Vec<(CString, i64)>,
    parent: *const ort::OrtNode,
    values: HashMap<ValueId, Box<HostValueInfo>>,
    // Boxed nodes keep the `OrtNode` pointers handed to the plugin stable even
    // if the vector moves during construction.
    #[allow(clippy::vec_box)]
    nodes: Vec<Box<HostNode>>,
    node_index: HashMap<NodeId, usize>,
    inputs: Vec<*const ort::OrtValueInfo>,
    outputs: Vec<*const ort::OrtValueInfo>,
    initializers: Vec<*const ort::OrtValueInfo>,
}

#[derive(Default)]
#[repr(C)]
pub(super) struct HostSupportInfo {
    pub(super) fused_groups: Vec<Vec<NodeId>>,
}

impl HostSupportInfo {
    pub(super) fn into_claims(self, graph: &Graph, ep_id: EpId) -> Vec<SubgraphClaim> {
        let captures = graph.captured_inputs_by_node();
        let mut capture_users: HashMap<ValueId, Vec<NodeId>> = HashMap::new();
        for (&owner, values) in &captures {
            for &value in values {
                capture_users.entry(value).or_default().push(owner);
            }
        }
        self.fused_groups
            .into_iter()
            .map(|node_ids| {
                let node_set: HashSet<_> = node_ids.iter().copied().collect();
                let mut input_values = Vec::new();
                let mut output_values = Vec::new();
                for &node_id in &node_ids {
                    let node = graph.node(node_id);
                    for value_id in node
                        .input_values()
                        .chain(captures.get(&node_id).into_iter().flatten().copied())
                    {
                        let outside = graph
                            .value(value_id)
                            .producer
                            .is_none_or(|p| !node_set.contains(&p));
                        if outside && !input_values.contains(&value_id) {
                            input_values.push(value_id);
                        }
                    }
                    for value_id in node.output_values() {
                        let value = graph.value(value_id);
                        let leaves = value.is_graph_output
                            || value
                                .consumers
                                .nodes()
                                .iter()
                                .any(|n| !node_set.contains(n))
                            || capture_users.get(&value_id).is_some_and(|users| {
                                users.iter().any(|node| !node_set.contains(node))
                            });
                        if leaves && !output_values.contains(&value_id) {
                            output_values.push(value_id);
                        }
                    }
                }
                SubgraphClaim {
                    ep_id,
                    node_ids,
                    input_values,
                    output_values,
                    meta_def: None,
                }
            })
            .collect()
    }
}

/// Model metadata and lexical bindings inherited by one projected scope.
struct HostGraphScope<'a> {
    opsets: &'a HashMap<String, u64>,
    values: &'a HashMap<String, *const ort::OrtValueInfo>,
    ir_version: Option<i64>,
}

impl HostGraph {
    pub(super) fn new(graph: &Graph) -> std::result::Result<Self, String> {
        let order = graph.topological_order().map_err(|err| {
            format!("cannot expose graph to plugin because topological ordering failed: {err}")
        })?;
        Self::new_with_order(
            graph,
            order,
            graph.inputs.clone(),
            graph.outputs.clone(),
            0,
            HostGraphScope {
                opsets: &HashMap::new(),
                values: &HashMap::new(),
                ir_version: None,
            },
        )
    }

    pub(super) fn new_for_claim(
        graph: &Graph,
        claim: &SubgraphClaim,
    ) -> std::result::Result<Self, String> {
        let selected: HashSet<NodeId> = claim.node_ids.iter().copied().collect();
        let order = graph
            .topological_order()
            .map_err(|err| {
                format!("cannot expose claimed subgraph to plugin because topological ordering failed: {err}")
            })?
            .into_iter()
            .filter(|node| selected.contains(node))
            .collect::<Vec<_>>();
        Self::new_with_order(
            graph,
            order,
            claim.input_values.clone(),
            claim.output_values.clone(),
            0,
            HostGraphScope {
                opsets: &HashMap::new(),
                values: &HashMap::new(),
                ir_version: None,
            },
        )
    }

    fn new_with_order(
        graph: &Graph,
        order: Vec<NodeId>,
        graph_inputs: Vec<ValueId>,
        graph_outputs: Vec<ValueId>,
        depth: usize,
        inherited: HostGraphScope<'_>,
    ) -> std::result::Result<Self, String> {
        if depth > super::MAX_GRAPH_NESTING {
            return Err(format!(
                "ORT graph nesting depth {depth} exceeds the supported projection limit {}; reduce nested graph depth",
                super::MAX_GRAPH_NESTING
            ));
        }
        let ir_version = graph.ir_version.or(inherited.ir_version);
        let mut scope_opsets = inherited.opsets.clone();
        scope_opsets.extend(graph.opset_imports.clone());
        let mut values: HashMap<ValueId, Box<HostValueInfo>> = HashMap::new();
        for (value_id, value) in graph.values.iter() {
            let initializer = graph
                .initializers
                .get(&value_id)
                .and_then(host_ort_value_for_weight);
            values.insert(
                value_id,
                Box::new(HostValueInfo {
                    id: value_id,
                    name: CString::new(value.name.clone().unwrap_or_default()).map_err(|_| {
                        format!("value {value_id:?} name contains an interior NUL byte")
                    })?,
                    type_info: HostTypeInfo::for_value(graph, value)?,
                    initializer,
                    producer: None,
                    is_constant_initializer: graph.is_constant_initializer(value_id),
                }),
            );
        }

        let mut lexical_values = inherited.values.clone();
        for (id, value) in graph.values.iter() {
            if let Some(name) = &value.name
                && (value.producer.is_some()
                    || graph.inputs.contains(&id)
                    || graph.is_initializer(id)
                    || !lexical_values.contains_key(name))
            {
                lexical_values.insert(name.clone(), value_ptr(&values[&id]));
            }
        }
        let mut nodes = Vec::with_capacity(order.len());
        let mut node_index = HashMap::new();
        for node_id in order {
            let node = graph.node(node_id);
            let mut host_node = Box::new(HostNode {
                id: node_id,
                name: cstring_lossless(&node.name, "node name", node_id)?,
                op_type: cstring_lossless(&node.op_type, "node op_type", node_id)?,
                domain: cstring_lossless(&node.domain, "node domain", node_id)?,
                since_version: {
                    let version = since_version(graph, node);
                    if version > 0 { version } else {
                        scope_opsets.get(&node.domain)
                            .and_then(|version| i32::try_from(*version).ok()).unwrap_or(0)
                    }
                },
                inputs: node
                    .inputs
                    .iter()
                    .map(|slot| {
                        slot.and_then(|v| {
                            graph.value(v).name.as_ref().and_then(|name| lexical_values.get(name).copied())
                                .or_else(|| values.get(&v).map(|value| value_ptr(value)))
                        }).unwrap_or(ptr::null())
                    })
                    .collect(),
                implicit_inputs: {
                    let mut names = std::collections::BTreeSet::new();
                    for ((owner, _), body) in &graph.subgraphs {
                        if *owner == node_id { names.extend(body.required_outer_names()); }
                    }
                    names.into_iter().map(|name| lexical_values.get(&name).copied().ok_or_else(|| format!("node {node_id:?} subgraph captures '{name}' but no enclosing graph defines it; bind that value in the lexical scope")))
                        .collect::<std::result::Result<Vec<_>, String>>()?
                },
                outputs: node
                    .outputs
                    .iter()
                    .map(|slot| {
                        slot.and_then(|v| values.get(&v))
                            .map(|value| value_ptr(value))
                            .unwrap_or(ptr::null())
                    })
                    .collect(),
                attrs: node
                    .attributes
                    .iter()
                    .map(|(name, attr)| {
                        Ok(HostOpAttr {
                            name: CString::new(name.as_str()).map_err(|_| {
                                format!(
                                    "attribute {name:?} on node {node_id:?} contains an interior NUL byte"
                                )
                            })?,
                            // Graph payloads are projected through Node_GetSubgraphs,
                            // not cloned recursively into a second ABI-owned tree.
                            attr: match attr {
                                Attribute::Graph(_) => Attribute::Graph(Box::default()),
                                Attribute::Graphs(_) => return Err(format!(
                                    "node {node_id:?} GRAPH-list attribute '{name}' cannot be represented losslessly by ORT's graph-name API; use native graph compilation"
                                )),
                                _ => attr.clone(),
                            },
                        })
                    })
                    .collect::<std::result::Result<Vec<_>, String>>()?,
                // Empty: these point into `HostGraph::values`, which outlives the
                // node, so the node owns nothing of its own here.
                owned_values: Vec::new(),
                subgraphs: Vec::new(),
            });
            let node_ptr = (&*host_node as *const HostNode).cast::<ort::OrtNode>();
            let mut nested = graph
                .subgraphs
                .iter()
                .filter(|((owner, _), _)| *owner == node_id)
                .collect::<Vec<_>>();
            nested.sort_by(|((_, lhs), _), ((_, rhs), _)| {
                (&lhs.name, lhs.index).cmp(&(&rhs.name, rhs.index))
            });
            for ((_, attribute), body) in nested {
                if attribute.index.is_some() {
                    return Err(format!(
                        "node {node_id:?} graph-list attribute {attribute} cannot be represented \
                         losslessly by the ORT Node_GetSubgraphs attribute-name contract; use \
                         native graph compilation for GRAPH-list patterns"
                    ));
                }
                let body_order = body.topological_order().map_err(|reason| {
                    format!("node {node_id:?} subgraph {attribute} topological ordering: {reason}")
                })?;
                let mut child = Box::new(
                    Self::new_with_order(
                        body,
                        body_order,
                        body.inputs.clone(),
                        body.outputs.clone(),
                        depth + 1,
                        HostGraphScope {
                            opsets: &scope_opsets,
                            values: &lexical_values,
                            ir_version,
                        },
                    )
                    .map_err(|reason| format!("node {node_id:?} subgraph {attribute}: {reason}"))?,
                );
                child.parent = node_ptr;
                host_node.subgraphs.push((
                    cstring_lossless(&attribute.name, "graph attribute name", node_id)?,
                    child,
                ));
            }

            for output in node.output_values() {
                if let Some(value) = values.get_mut(&output) {
                    value.producer = Some(node_ptr);
                }
            }
            node_index.insert(node_id, nodes.len());
            nodes.push(host_node);
        }

        let inputs = graph_inputs
            .iter()
            .filter_map(|v| values.get(v).map(|value| value_ptr(value)))
            .collect();
        let outputs = graph_outputs
            .iter()
            .filter_map(|v| values.get(v).map(|value| value_ptr(value)))
            .collect();
        let initializers = graph
            .initializers
            .keys()
            .filter_map(|v| values.get(v).map(|value| value_ptr(value)))
            .collect();
        let mut opsets = scope_opsets
            .iter()
            .map(|(domain, version)| {
                Ok((
                    CString::new(domain.as_str())
                        .map_err(|_| format!("opset domain {domain:?} contains NUL"))?,
                    i64::try_from(*version)
                        .map_err(|_| format!("opset version {version} exceeds i64"))?,
                ))
            })
            .collect::<std::result::Result<Vec<_>, String>>()?;
        opsets.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        Ok(Self {
            ir_version: ir_version.unwrap_or(0),
            opsets,
            parent: ptr::null(),
            values,
            nodes,
            node_index,
            inputs,
            outputs,
            initializers,
        })
    }

    pub(super) fn as_ort_graph(&self) -> *const ort::OrtGraph {
        (self as *const HostGraph).cast::<ort::OrtGraph>()
    }
}

impl HostNode {
    pub(super) fn fused(
        graph: &Graph,
        claim: &SubgraphClaim,
        index: usize,
    ) -> std::result::Result<Box<Self>, String> {
        let value_ptrs = claim
            .input_values
            .iter()
            .map(|value| {
                graph
                    .try_value(*value)
                    .ok_or_else(|| format!("fused plugin input value {value:?} is not live"))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let output_value_ptrs = claim
            .output_values
            .iter()
            .map(|value| {
                graph
                    .try_value(*value)
                    .ok_or_else(|| format!("fused plugin output value {value:?} is not live"))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut inputs = Vec::with_capacity(value_ptrs.len());
        let mut outputs = Vec::with_capacity(output_value_ptrs.len());
        let mut owned_values = Vec::new();
        for value in value_ptrs.into_iter().chain(output_value_ptrs) {
            owned_values.push(Box::new(HostValueInfo {
                id: value.id,
                name: CString::new(value.name.clone().unwrap_or_default()).map_err(|_| {
                    format!(
                        "fused value {:?} name contains an interior NUL byte",
                        value.id
                    )
                })?,
                type_info: HostTypeInfo::for_value(graph, value)?,
                initializer: None,
                producer: None,
                is_constant_initializer: false,
            }));
        }
        for value in owned_values.iter().take(claim.input_values.len()) {
            inputs.push(value_ptr(value));
        }
        for value in owned_values.iter().skip(claim.input_values.len()) {
            outputs.push(value_ptr(value));
        }
        // `owned_values` moves into the node below, so the pointers just taken
        // stay valid for exactly as long as the node that holds them.
        let mut node = Box::new(Self {
            id: NodeId(index as u32),
            name: CString::new(format!("native_plugin_fused_{index}")).unwrap(),
            op_type: c"NativePluginFused".to_owned(),
            // Our own domain: this operator is invented here and Microsoft
            // never specified it. Matches `onnx_runtime_ir::RUNTIME_DOMAIN`,
            // spelled literally because a C string is needed here.
            domain: c"pkg.nxrt".to_owned(),
            since_version: 1,
            inputs,
            outputs,
            implicit_inputs: Vec::new(),
            attrs: Vec::new(),
            subgraphs: Vec::new(),
            owned_values,
        });
        node.attrs.push(HostOpAttr {
            name: c"native_plugin_fusion_id".to_owned(),
            attr: Attribute::Int(index as i64),
        });
        Ok(node)
    }
}

pub(super) fn dtype_to_ort(dtype: DataType) -> ort::ONNXTensorElementDataType {
    dtype.to_onnx() as ort::ONNXTensorElementDataType
}

pub(super) fn since_version(graph: &Graph, node: &Node) -> i32 {
    // Through the IR's resolver, so a plugin is told the same version shape
    // inference and dispatch use. These previously disagreed: this function
    // converted to `i32` and the executor to `u64`, so a version between the
    // two ranges was honoured by one and ignored by the other.
    graph
        .effective_opset(node)
        .and_then(|version| i32::try_from(version).ok())
        .unwrap_or(0)
}

pub(super) fn ort_api_base() -> *const ort::OrtApiBase {
    static API_BASE: OnceLock<ort::OrtApiBase> = OnceLock::new();
    API_BASE.get_or_init(|| ort::OrtApiBase {
        GetApi: Some(host_get_api),
        GetVersionString: Some(host_get_version_string),
    })
}

extern "C" fn host_get_version_string() -> *const c_char {
    c"native-ort-plugin-host".as_ptr()
}

unsafe extern "C" fn host_get_api(version: u32) -> *const ort::OrtApi {
    if version > ort::ORT_API_VERSION {
        return ptr::null();
    }
    ort_api()
}

pub(super) fn ort_api() -> *const ort::OrtApi {
    static API: OnceLock<ort::OrtApi> = OnceLock::new();
    API.get_or_init(|| ort::OrtApi {
        CreateStatus: Some(host_create_status),
        ReleaseStatus: Some(host_release_status),
        GetEpApi: Some(host_get_ep_api),
        GetOnnxTypeFromTypeInfo: Some(host_get_onnx_type_from_type_info),
        CastTypeInfoToTensorInfo: Some(host_cast_type_info_to_tensor_info),
        GetTensorElementType: Some(host_get_tensor_element_type),
        GetDimensionsCount: Some(host_get_dimensions_count),
        GetDimensions: Some(host_get_dimensions),
        GetSymbolicDimensions: Some(host_get_symbolic_dimensions),
        TensorTypeAndShape_HasShape: Some(host_has_shape),
        GetTensorShapeElementCount: Some(host_get_tensor_shape_element_count),
        GetTensorTypeAndShape: Some(host_get_tensor_type_and_shape),
        ReleaseTensorTypeAndShapeInfo: Some(host_release_tensor_type_and_shape_info),
        GetTensorData: Some(host_get_tensor_data),
        GetTensorMemoryInfo: Some(host_get_tensor_memory_info),
        MemoryInfoGetDeviceType: Some(host_memory_info_device_type),
        MemoryInfoGetName: Some(host_memory_info_name),
        MemoryInfoGetId: Some(host_memory_info_id),
        GetStringTensorDataLength: Some(host_string_tensor_data_length),
        GetStringTensorContent: Some(host_string_tensor_content),
        ReleaseValue: Some(host_release_value),
        OpAttr_GetTensorAttributeAsOrtValue: Some(host_tensor_attribute),
        GetTensorMutableData: Some(host_get_tensor_mutable_data),
        GetValueInfoName: Some(host_get_value_info_name),
        GetValueInfoTypeInfo: Some(host_get_value_info_type_info),
        ValueInfo_GetInitializerValue: Some(host_value_info_get_initializer_value),
        ValueInfo_IsConstantInitializer: Some(host_value_info_is_constant_initializer),
        ValueInfo_GetValueProducer: Some(host_value_info_get_value_producer),
        Graph_GetOnnxIRVersion: Some(host_graph_ir_version),
        Graph_GetNumOperatorSets: Some(host_graph_num_opsets),
        Graph_GetOperatorSets: Some(host_graph_opsets),
        Graph_GetNumInputs: Some(host_graph_get_num_inputs),
        Graph_GetInputs: Some(host_graph_get_inputs),
        Graph_GetNumOutputs: Some(host_graph_get_num_outputs),
        Graph_GetOutputs: Some(host_graph_get_outputs),
        Graph_GetNumInitializers: Some(host_graph_get_num_initializers),
        Graph_GetInitializers: Some(host_graph_get_initializers),
        Graph_GetNumNodes: Some(host_graph_get_num_nodes),
        Graph_GetNodes: Some(host_graph_get_nodes),
        Graph_GetParentNode: Some(host_graph_get_parent_node),
        Node_GetId: Some(host_node_get_id),
        Node_GetName: Some(host_node_get_name),
        Node_GetOperatorType: Some(host_node_get_operator_type),
        Node_GetDomain: Some(host_node_get_domain),
        Node_GetSinceVersion: Some(host_node_get_since_version),
        Node_GetNumInputs: Some(host_node_get_num_inputs),
        Node_GetInputs: Some(host_node_get_inputs),
        Node_GetNumImplicitInputs: Some(host_node_get_num_implicit_inputs),
        Node_GetImplicitInputs: Some(host_node_get_implicit_inputs),
        Node_GetNumOutputs: Some(host_node_get_num_outputs),
        Node_GetOutputs: Some(host_node_get_outputs),
        Node_GetNumAttributes: Some(host_node_get_num_attributes),
        Node_GetAttributes: Some(host_node_get_attributes),
        Node_GetAttributeByName: Some(host_node_get_attribute_by_name),
        OpAttr_GetType: Some(host_op_attr_get_type),
        OpAttr_GetName: Some(host_op_attr_get_name),
        ReadOpAttr: Some(host_read_op_attr),
        Node_GetNumSubgraphs: Some(host_node_get_num_subgraphs),
        Node_GetSubgraphs: Some(host_node_get_subgraphs),
        KernelContext_GetInput: Some(host_kernel_context_get_input),
        KernelContext_GetOutput: Some(host_kernel_context_get_output),
        KernelContext_GetInputCount: Some(host_kernel_context_input_count),
        KernelContext_GetOutputCount: Some(host_kernel_context_output_count),
        HardwareDevice_Type: Some(host_hardware_device_type),
        ..Default::default()
    })
}

fn ort_ep_api() -> *const ort::OrtEpApi {
    static EP_API: OnceLock<ort::OrtEpApi> = OnceLock::new();
    EP_API.get_or_init(|| ort::OrtEpApi {
        CreateEpDevice: Some(host_create_ep_device),
        EpGraphSupportInfo_AddNodesToFuse: Some(host_ep_graph_support_info_add_nodes_to_fuse),
        ..Default::default()
    })
}

unsafe extern "C" fn host_get_ep_api() -> *const ort::OrtEpApi {
    ort_ep_api()
}

unsafe extern "C" fn host_create_status(
    code: ort::OrtErrorCode,
    msg: *const c_char,
) -> *mut ort::OrtStatus {
    let msg = if msg.is_null() {
        c"native ORT plugin host error".to_owned()
    } else {
        // SAFETY: ORT ABI requires status messages to be null-terminated strings.
        unsafe { CStr::from_ptr(msg).to_owned() }
    };
    Box::into_raw(Box::new(HostStatus { code, msg })).cast::<ort::OrtStatus>()
}

unsafe extern "C" fn host_release_status(input: *mut ort::OrtStatus) {
    release_status(input);
}

pub(super) fn release_status(input: *mut ort::OrtStatus) {
    if !input.is_null() {
        // SAFETY: Status pointers returned by this host come from Box<HostStatus>.
        unsafe { drop(Box::from_raw(input.cast::<HostStatus>())) };
    }
}

fn status_error(message: &str) -> *mut ort::OrtStatus {
    let sanitized = message.replace('\0', "\\0");
    let c = CString::new(sanitized).expect("sanitized status has no NUL");
    Box::into_raw(Box::new(HostStatus {
        code: ort::ORT_FAIL,
        msg: c,
    }))
    .cast::<ort::OrtStatus>()
}

/// Translate a status from executing a fused subgraph.
///
/// Separate from [`check_status`] because the remedies are different: a load
/// or compile failure points at plugin compatibility, while a failure here
/// means a subgraph the plugin already accepted did not run.
pub(super) fn check_compute_status(
    path: &Path,
    fusion_index: usize,
    status: *mut ort::OrtStatus,
) -> Result<()> {
    if status.is_null() {
        return Ok(());
    }
    // SAFETY: Plugin statuses in this bridge are created by our CreateStatus callback.
    let boxed = unsafe { Box::from_raw(status.cast::<HostStatus>()) };
    Err(EpError::KernelFailed(format!(
        "the execution provider plugin at {} failed to run fused subgraph {fusion_index} with \
         ORT error code {}: {}. The plugin loaded and accepted this subgraph, so this is a \
         failure of the call rather than of loading; fix by checking the inputs the model feeds \
         this subgraph, or set ONNX_GENAI_BACKEND=ort to run without the plugin",
        path.display(),
        boxed.code,
        boxed.msg.to_string_lossy(),
    )))
}

pub(super) fn check_status(path: &Path, action: &str, status: *mut ort::OrtStatus) -> Result<()> {
    if status.is_null() {
        return Ok(());
    }
    // SAFETY: Plugin statuses in this bridge are created by our CreateStatus callback.
    let boxed = unsafe { Box::from_raw(status.cast::<HostStatus>()) };
    Err(EpError::EpLoadFailed {
        path: path.to_path_buf(),
        reason: format!(
            "{action} failed with ORT error code {}: {}; fix by checking plugin compatibility, ORT_API_VERSION {}, and the model graph metadata",
            boxed.code,
            boxed.msg.to_string_lossy(),
            ort::ORT_API_VERSION
        ),
    })
}

unsafe extern "C" fn host_get_onnx_type_from_type_info(
    type_info: *const ort::OrtTypeInfo,
    out: *mut ort::ONNXType,
) -> ort::OrtStatusPtr {
    if type_info.is_null() || out.is_null() {
        return status_error(
            "GetOnnxTypeFromTypeInfo failed: out pointer was null; fix the plugin to pass a valid output pointer",
        );
    }
    // SAFETY: out is checked non-null and points to caller storage.
    unsafe {
        *out = type_info_from_ptr(type_info).kind;
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_cast_type_info_to_tensor_info(
    type_info: *const ort::OrtTypeInfo,
    out: *mut *const ort::OrtTensorTypeAndShapeInfo,
) -> ort::OrtStatusPtr {
    if type_info.is_null() || out.is_null() {
        return status_error(
            "CastTypeInfoToTensorInfo failed: null input/output pointer; fix the plugin to pass valid ORT pointers",
        );
    }
    let host = type_info_from_ptr(type_info);
    // SAFETY: out is checked non-null and the returned pointer borrows type_info.
    unsafe {
        *out = if host.kind == ort::ONNX_TYPE_TENSOR {
            (&host.tensor as *const HostTensorTypeAndShapeInfo)
                .cast::<ort::OrtTensorTypeAndShapeInfo>()
        } else {
            ptr::null()
        };
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_get_tensor_element_type(
    info: *const ort::OrtTensorTypeAndShapeInfo,
    out: *mut ort::ONNXTensorElementDataType,
) -> ort::OrtStatusPtr {
    if info.is_null() || out.is_null() {
        return status_error(
            "GetTensorElementType failed: null tensor-info/output pointer; fix the plugin to pass valid ORT pointers",
        );
    }
    // SAFETY: out is checked non-null.
    unsafe {
        *out = tensor_info_from_ptr(info).dtype;
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_has_shape(info: *const ort::OrtTensorTypeAndShapeInfo) -> bool {
    !info.is_null() && tensor_info_from_ptr(info).dims.is_some()
}
unsafe extern "C" fn host_get_symbolic_dimensions(
    info: *const ort::OrtTensorTypeAndShapeInfo,
    out: *mut *const c_char,
    count: usize,
) -> ort::OrtStatusPtr {
    if info.is_null() {
        return status_error(
            "GetSymbolicDimensions: null tensor descriptor; pass live type metadata",
        );
    }
    let info = tensor_info_from_ptr(info);
    let Some(dims) = &info.dims else {
        return status_error(
            "GetSymbolicDimensions: rank is unknown; query TensorTypeAndShape_HasShape first",
        );
    };
    if count < dims.len() || (!dims.is_empty() && out.is_null()) {
        return status_error(
            "GetSymbolicDimensions: output buffer is null or too small; allocate GetDimensionsCount entries",
        );
    }
    for index in 0..dims.len() {
        unsafe {
            *out.add(index) = info
                .symbols
                .get(index)
                .map_or(c"".as_ptr(), |name| name.as_ptr());
        }
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_get_dimensions_count(
    info: *const ort::OrtTensorTypeAndShapeInfo,
    out: *mut usize,
) -> ort::OrtStatusPtr {
    if info.is_null() || out.is_null() {
        return status_error(
            "GetDimensionsCount failed: null tensor-info/output pointer; fix the plugin to pass valid ORT pointers",
        );
    }
    // SAFETY: out is checked non-null.
    unsafe {
        *out = tensor_info_from_ptr(info).dims.as_ref().map_or(0, Vec::len);
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_get_dimensions(
    info: *const ort::OrtTensorTypeAndShapeInfo,
    dim_values: *mut i64,
    dim_values_length: usize,
) -> ort::OrtStatusPtr {
    if info.is_null() {
        return status_error(
            "GetDimensions failed: tensor-info pointer was null; fix the plugin to pass valid ORT pointers",
        );
    }
    let Some(dims) = &tensor_info_from_ptr(info).dims else {
        return status_error(
            "GetDimensions: tensor rank is unknown; query TensorTypeAndShape_HasShape before reading dimensions",
        );
    };
    if dim_values_length < dims.len() {
        return status_error(
            "GetDimensions failed: output buffer is smaller than rank; fix by allocating GetDimensionsCount entries",
        );
    }
    if !dims.is_empty() && dim_values.is_null() {
        return status_error(
            "GetDimensions failed: output buffer was null for non-scalar tensor; fix the plugin buffer allocation",
        );
    }
    // SAFETY: buffer length was validated against dims.len().
    unsafe {
        if !dims.is_empty() {
            ptr::copy_nonoverlapping(dims.as_ptr(), dim_values, dims.len());
        }
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_get_tensor_shape_element_count(
    info: *const ort::OrtTensorTypeAndShapeInfo,
    out: *mut usize,
) -> ort::OrtStatusPtr {
    if info.is_null() || out.is_null() {
        return status_error(
            "GetTensorShapeElementCount failed: null tensor-info/output pointer; fix the plugin to pass valid ORT pointers",
        );
    }
    let mut count = 1usize;
    let Some(dims) = &tensor_info_from_ptr(info).dims else {
        return status_error(
            "GetTensorShapeElementCount: tensor rank is unknown; provide concrete shape metadata before requesting a size",
        );
    };
    for &dim in dims {
        if dim < 0 {
            return status_error(
                "GetTensorShapeElementCount failed: symbolic dimension has no concrete element count; fix by avoiding constant reads of symbolic tensors",
            );
        }
        count = match count.checked_mul(dim as usize) {
            Some(c) => c,
            None => {
                return status_error(
                    "GetTensorShapeElementCount failed: shape product overflowed usize; fix the model shape metadata",
                );
            }
        };
    }
    // SAFETY: out is checked non-null.
    unsafe {
        *out = count;
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_get_tensor_type_and_shape(
    value: *const ort::OrtValue,
    out: *mut *mut ort::OrtTensorTypeAndShapeInfo,
) -> ort::OrtStatusPtr {
    if value.is_null() || out.is_null() {
        return status_error(
            "GetTensorTypeAndShape failed: null value/output pointer; fix the plugin to pass valid ORT pointers",
        );
    }
    let cloned = Box::new(HostTensorTypeAndShapeInfo {
        dtype: ort_value_from_ptr(value).tensor.dtype,
        dims: ort_value_from_ptr(value).tensor.dims.clone(),
        symbols: ort_value_from_ptr(value).tensor.symbols.clone(),
    });
    // SAFETY: out is checked non-null and receives owned allocation released by ReleaseTensorTypeAndShapeInfo.
    unsafe {
        *out = Box::into_raw(cloned).cast::<ort::OrtTensorTypeAndShapeInfo>();
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_release_tensor_type_and_shape_info(
    input: *mut ort::OrtTensorTypeAndShapeInfo,
) {
    if !input.is_null() {
        // SAFETY: This release function is for allocations from GetTensorTypeAndShape.
        unsafe { drop(Box::from_raw(input.cast::<HostTensorTypeAndShapeInfo>())) };
    }
}

// HostGraph and HostKernelContext project host-accessible values only.
static HOST_CPU_MEMORY_INFO: u8 = 0;
unsafe extern "C" fn host_get_tensor_memory_info(
    value: *const ort::OrtValue,
    out: *mut *const ort::OrtMemoryInfo,
) -> ort::OrtStatusPtr {
    if value.is_null() || out.is_null() {
        return status_error("GetTensorMemoryInfo: null value/output; pass live tensor storage");
    }
    unsafe {
        *out = (&HOST_CPU_MEMORY_INFO as *const u8).cast();
    }
    ptr::null_mut()
}
unsafe extern "C" fn host_memory_info_name(
    info: *const ort::OrtMemoryInfo,
    out: *mut *const c_char,
) -> ort::OrtStatusPtr {
    if info.is_null() || out.is_null() {
        return status_error("MemoryInfoGetName: null memory info/output pointer");
    }
    unsafe {
        *out = c"Cpu".as_ptr();
    }
    ptr::null_mut()
}
unsafe extern "C" fn host_memory_info_id(
    info: *const ort::OrtMemoryInfo,
    out: *mut i32,
) -> ort::OrtStatusPtr {
    if info.is_null() || out.is_null() {
        return status_error("MemoryInfoGetId: null memory info/output pointer");
    }
    unsafe {
        *out = 0;
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_memory_info_device_type(
    _info: *const ort::OrtMemoryInfo,
    out: *mut ort::OrtMemoryInfoDeviceType,
) {
    if !out.is_null() {
        unsafe {
            *out = ort::OrtMemoryInfoDeviceType_CPU;
        }
    }
}
unsafe extern "C" fn host_tensor_attribute(
    attribute: *const ort::OrtOpAttr,
    out: *mut *mut ort::OrtValue,
) -> ort::OrtStatusPtr {
    if attribute.is_null() || out.is_null() {
        return status_error(
            "OpAttr_GetTensorAttributeAsOrtValue: null attribute/output; pass live storage",
        );
    }
    unsafe {
        *out = ptr::null_mut();
    }
    let Attribute::Tensor(tensor) = &attr_from_ptr(attribute).attr else {
        return status_error(
            "OpAttr_GetTensorAttributeAsOrtValue: attribute is not a TENSOR; query its type first",
        );
    };
    let Some(bytes) = tensor.checked_expected_bytes() else {
        return status_error(
            "tensor attribute dimensions or byte length overflow; reduce tensor dimensions",
        );
    };
    let dims = match tensor
        .dims
        .iter()
        .map(|&dim| i64::try_from(dim))
        .collect::<std::result::Result<Vec<_>, _>>()
    {
        Ok(dims) => dims,
        Err(_) => {
            return status_error("tensor attribute dimension exceeds i64; reduce its dimensions");
        }
    };
    let storage = if tensor.dtype == DataType::String {
        if tensor.checked_numel() != Some(tensor.strings.len()) {
            return status_error(
                "tensor attribute string count does not match its shape; supply one string per element",
            );
        }
        HostOrtValueStorage::Strings(tensor.strings.clone())
    } else {
        if tensor.dtype == DataType::Undefined || bytes != tensor.data.len() {
            return status_error(
                "tensor attribute dtype/shape does not match its byte payload; correct the tensor data",
            );
        }
        HostOrtValueStorage::Owned(tensor.data.clone())
    };
    let value = Box::new(HostOrtValue {
        tensor: HostTensorTypeAndShapeInfo {
            dtype: dtype_to_ort(tensor.dtype),
            dims: Some(dims),
            symbols: Vec::new(),
        },
        storage,
    });
    unsafe {
        *out = Box::into_raw(value).cast();
    }
    ptr::null_mut()
}
unsafe extern "C" fn host_release_value(value: *mut ort::OrtValue) {
    if !value.is_null() {
        unsafe {
            drop(Box::from_raw(value.cast::<HostOrtValue>()));
        }
    }
}
unsafe extern "C" fn host_string_tensor_data_length(
    value: *const ort::OrtValue,
    out: *mut usize,
) -> ort::OrtStatusPtr {
    if value.is_null() || out.is_null() {
        return status_error(
            "GetStringTensorDataLength: null value/output; pass live string tensor storage",
        );
    }
    let HostOrtValueStorage::Strings(strings) = &ort_value_from_ptr(value).storage else {
        return status_error("GetStringTensorDataLength requires a string tensor; query its dtype");
    };
    let Some(length) = strings
        .iter()
        .try_fold(0usize, |length, value| length.checked_add(value.len()))
    else {
        return status_error("string tensor byte length overflows usize; reduce its payload");
    };
    unsafe {
        *out = length;
    }
    ptr::null_mut()
}
unsafe extern "C" fn host_string_tensor_content(
    value: *const ort::OrtValue,
    data: *mut c_void,
    bytes: usize,
    offsets: *mut usize,
    count: usize,
) -> ort::OrtStatusPtr {
    if value.is_null() {
        return status_error("GetStringTensorContent: null tensor; pass a live string tensor");
    }
    let HostOrtValueStorage::Strings(strings) = &ort_value_from_ptr(value).storage else {
        return status_error("GetStringTensorContent requires a string tensor; query its dtype");
    };
    let Some(required) = strings
        .iter()
        .try_fold(0usize, |length, value| length.checked_add(value.len()))
    else {
        return status_error("string tensor byte length overflows usize; reduce its payload");
    };
    if bytes < required
        || count < strings.len()
        || (required > 0 && data.is_null())
        || (!strings.is_empty() && offsets.is_null())
    {
        return status_error(
            "GetStringTensorContent: output buffers are null or too small; query byte and element counts and allocate both buffers",
        );
    }
    let mut offset = 0;
    for (index, string) in strings.iter().enumerate() {
        unsafe {
            *offsets.add(index) = offset;
            if !string.is_empty() {
                ptr::copy_nonoverlapping(
                    string.as_ptr(),
                    data.cast::<u8>().add(offset),
                    string.len(),
                );
            }
        }
        offset += string.len();
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_get_tensor_data(
    value: *const ort::OrtValue,
    out: *mut *const c_void,
) -> ort::OrtStatusPtr {
    if value.is_null() || out.is_null() {
        return status_error(
            "GetTensorData failed: null value/output pointer; fix the plugin to pass valid ORT pointers",
        );
    }
    if matches!(
        ort_value_from_ptr(value).storage,
        HostOrtValueStorage::Strings(_)
    ) {
        return status_error("GetTensorData: string tensors require GetStringTensorContent");
    }
    // SAFETY: out is checked non-null and data lives as long as the HostOrtValue.
    unsafe {
        *out = ort_value_from_ptr(value).data_ptr();
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_get_tensor_mutable_data(
    value: *mut ort::OrtValue,
    out: *mut *mut c_void,
) -> ort::OrtStatusPtr {
    if value.is_null() || out.is_null() {
        return status_error(
            "GetTensorMutableData failed: null value/output pointer; fix the plugin to pass valid ORT pointers",
        );
    }
    let data = ort_value_from_mut_ptr(value).data_mut_ptr();
    if data.is_null() {
        // Reporting success while handing back null would invite the caller to
        // dereference it: a null status means the out-pointer is usable. This
        // is reached for an initializer served straight from the mapped weight
        // file, which has no writable storage to offer.
        return status_error(
            "GetTensorMutableData failed: this tensor is a read-only initializer backed by the \
             model's weight file, so it has no mutable storage; fix by reading it with \
             GetTensorData instead of asking for a mutable pointer",
        );
    }
    // SAFETY: `out` is non-null, checked above.
    unsafe {
        *out = data;
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_get_value_info_name(
    value_info: *const ort::OrtValueInfo,
    out: *mut *const c_char,
) -> ort::OrtStatusPtr {
    if value_info.is_null() || out.is_null() {
        return status_error(
            "GetValueInfoName failed: null value-info/output pointer; fix the plugin to pass valid ORT pointers",
        );
    }
    // SAFETY: out is checked non-null and name is owned by HostValueInfo.
    unsafe {
        *out = value_from_ptr(value_info).name.as_ptr();
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_get_value_info_type_info(
    value_info: *const ort::OrtValueInfo,
    type_info: *mut *const ort::OrtTypeInfo,
) -> ort::OrtStatusPtr {
    if value_info.is_null() || type_info.is_null() {
        return status_error(
            "GetValueInfoTypeInfo failed: null value-info/output pointer; fix the plugin to pass valid ORT pointers",
        );
    }
    let value = value_from_ptr(value_info);
    // SAFETY: out is checked non-null and returned pointer borrows HostValueInfo.
    unsafe {
        *type_info = (&value.type_info as *const HostTypeInfo).cast::<ort::OrtTypeInfo>();
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_value_info_get_initializer_value(
    value_info: *const ort::OrtValueInfo,
    value: *mut *const ort::OrtValue,
) -> ort::OrtStatusPtr {
    if value_info.is_null() || value.is_null() {
        return status_error(
            "ValueInfo_GetInitializerValue failed: null value-info/output pointer; fix the plugin to pass valid ORT pointers",
        );
    }
    let init = value_from_ptr(value_info)
        .initializer
        .as_deref()
        .map_or(ptr::null(), |v| {
            (v as *const HostOrtValue).cast::<ort::OrtValue>()
        });
    // SAFETY: out is checked non-null and returned pointer borrows HostValueInfo.
    unsafe {
        *value = init;
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_value_info_is_constant_initializer(
    value_info: *const ort::OrtValueInfo,
    out: *mut bool,
) -> ort::OrtStatusPtr {
    if value_info.is_null() || out.is_null() {
        return status_error(
            "ValueInfo_IsConstantInitializer failed: null value-info/output pointer; fix the plugin to pass valid ORT pointers",
        );
    }
    // SAFETY: out is checked non-null.
    unsafe {
        *out = value_from_ptr(value_info).is_constant_initializer;
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_value_info_get_value_producer(
    value_info: *const ort::OrtValueInfo,
    producer: *mut *const ort::OrtNode,
    producer_output_index: *mut usize,
) -> ort::OrtStatusPtr {
    if value_info.is_null() || producer.is_null() {
        return status_error(
            "ValueInfo_GetValueProducer failed: null value-info/output pointer; fix the plugin to pass valid ORT pointers",
        );
    }
    let value = value_from_ptr(value_info);
    // SAFETY: output pointers are checked or optional.
    unsafe {
        *producer = value.producer.unwrap_or(ptr::null());
        if !producer_output_index.is_null() {
            *producer_output_index = 0;
        }
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_graph_ir_version(
    graph: *const ort::OrtGraph,
    out: *mut i64,
) -> ort::OrtStatusPtr {
    if graph.is_null() || out.is_null() {
        return status_error("Graph_GetOnnxIRVersion: null graph/output pointer");
    }
    unsafe {
        *out = graph_from_ptr(graph).ir_version;
    }
    ptr::null_mut()
}
unsafe extern "C" fn host_graph_num_opsets(
    graph: *const ort::OrtGraph,
    out: *mut usize,
) -> ort::OrtStatusPtr {
    if graph.is_null() || out.is_null() {
        return status_error("Graph_GetNumOperatorSets: null graph/output pointer");
    }
    unsafe {
        *out = graph_from_ptr(graph).opsets.len();
    }
    ptr::null_mut()
}
unsafe extern "C" fn host_graph_opsets(
    graph: *const ort::OrtGraph,
    domains: *mut *const c_char,
    versions: *mut i64,
    count: usize,
) -> ort::OrtStatusPtr {
    if graph.is_null() {
        return status_error("Graph_GetOperatorSets: null graph pointer");
    }
    let graph = graph_from_ptr(graph);
    if count < graph.opsets.len()
        || (!graph.opsets.is_empty() && (domains.is_null() || versions.is_null()))
    {
        return status_error(
            "Graph_GetOperatorSets: null or undersized storage; query Graph_GetNumOperatorSets first",
        );
    }
    for (index, (domain, version)) in graph.opsets.iter().enumerate() {
        unsafe {
            *domains.add(index) = domain.as_ptr();
            *versions.add(index) = *version;
        }
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_graph_get_num_inputs(
    graph: *const ort::OrtGraph,
    out: *mut usize,
) -> ort::OrtStatusPtr {
    graph_count(graph, out, |g| g.inputs.len(), "Graph_GetNumInputs")
}
unsafe extern "C" fn host_graph_get_num_outputs(
    graph: *const ort::OrtGraph,
    out: *mut usize,
) -> ort::OrtStatusPtr {
    graph_count(graph, out, |g| g.outputs.len(), "Graph_GetNumOutputs")
}
unsafe extern "C" fn host_graph_get_num_initializers(
    graph: *const ort::OrtGraph,
    out: *mut usize,
) -> ort::OrtStatusPtr {
    graph_count(
        graph,
        out,
        |g| g.initializers.len(),
        "Graph_GetNumInitializers",
    )
}
unsafe extern "C" fn host_graph_get_num_nodes(
    graph: *const ort::OrtGraph,
    out: *mut usize,
) -> ort::OrtStatusPtr {
    graph_count(graph, out, |g| g.nodes.len(), "Graph_GetNumNodes")
}

fn graph_count(
    graph: *const ort::OrtGraph,
    out: *mut usize,
    f: impl FnOnce(&HostGraph) -> usize,
    name: &str,
) -> ort::OrtStatusPtr {
    if graph.is_null() || out.is_null() {
        return status_error(&format!(
            "{name} failed: null graph/output pointer; fix the plugin to pass valid ORT pointers"
        ));
    }
    // SAFETY: out is checked non-null.
    unsafe {
        *out = f(graph_from_ptr(graph));
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_graph_get_inputs(
    graph: *const ort::OrtGraph,
    values: *mut *const ort::OrtValueInfo,
    count: usize,
) -> ort::OrtStatusPtr {
    graph_values(graph, values, count, |g| &g.inputs, "Graph_GetInputs")
}
unsafe extern "C" fn host_graph_get_outputs(
    graph: *const ort::OrtGraph,
    values: *mut *const ort::OrtValueInfo,
    count: usize,
) -> ort::OrtStatusPtr {
    graph_values(graph, values, count, |g| &g.outputs, "Graph_GetOutputs")
}
unsafe extern "C" fn host_graph_get_initializers(
    graph: *const ort::OrtGraph,
    values: *mut *const ort::OrtValueInfo,
    count: usize,
) -> ort::OrtStatusPtr {
    graph_values(
        graph,
        values,
        count,
        |g| &g.initializers,
        "Graph_GetInitializers",
    )
}

fn graph_values(
    graph: *const ort::OrtGraph,
    values: *mut *const ort::OrtValueInfo,
    count: usize,
    f: impl FnOnce(&HostGraph) -> &Vec<*const ort::OrtValueInfo>,
    name: &str,
) -> ort::OrtStatusPtr {
    if graph.is_null() {
        return status_error(&format!(
            "{name} failed: graph pointer was null; fix the plugin to pass a valid graph"
        ));
    }
    let src = f(graph_from_ptr(graph));
    if count < src.len() {
        return status_error(&format!(
            "{name} failed: output buffer has {count} entries but graph needs {}; fix by querying the count first",
            src.len()
        ));
    }
    if !src.is_empty() && values.is_null() {
        return status_error(&format!(
            "{name} failed: output buffer pointer was null; fix the plugin allocation"
        ));
    }
    // SAFETY: destination buffer length was validated.
    unsafe {
        ptr::copy_nonoverlapping(src.as_ptr(), values, src.len());
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_graph_get_nodes(
    graph: *const ort::OrtGraph,
    nodes: *mut *const ort::OrtNode,
    count: usize,
) -> ort::OrtStatusPtr {
    if graph.is_null() {
        return status_error(
            "Graph_GetNodes failed: graph pointer was null; fix the plugin to pass a valid graph",
        );
    }
    let graph = graph_from_ptr(graph);
    if count < graph.nodes.len() {
        return status_error(
            "Graph_GetNodes failed: output buffer is smaller than node count; fix by querying Graph_GetNumNodes first",
        );
    }
    if !graph.nodes.is_empty() && nodes.is_null() {
        return status_error(
            "Graph_GetNodes failed: output buffer pointer was null; fix the plugin allocation",
        );
    }
    for (i, node) in graph.nodes.iter().enumerate() {
        // SAFETY: destination buffer length was validated.
        unsafe {
            *nodes.add(i) = (&**node as *const HostNode).cast::<ort::OrtNode>();
        }
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_graph_get_parent_node(
    graph: *const ort::OrtGraph,
    node: *mut *const ort::OrtNode,
) -> ort::OrtStatusPtr {
    if graph.is_null() || node.is_null() {
        return status_error(
            "Graph_GetParentNode failed: output pointer was null; fix the plugin to pass valid storage",
        );
    }
    // The parent node is boxed and owns this graph for the whole callback lifetime.
    unsafe {
        *node = graph_from_ptr(graph).parent;
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_node_get_id(
    node: *const ort::OrtNode,
    id: *mut usize,
) -> ort::OrtStatusPtr {
    if node.is_null() || id.is_null() {
        return status_error(
            "Node_GetId failed: null node/output pointer; fix the plugin to pass valid ORT pointers",
        );
    }
    // SAFETY: id is checked non-null.
    unsafe {
        *id = node_from_ptr(node).id.0 as usize;
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_node_get_name(
    node: *const ort::OrtNode,
    out: *mut *const c_char,
) -> ort::OrtStatusPtr {
    node_str(node, out, |n| n.name.as_ptr(), "Node_GetName")
}
unsafe extern "C" fn host_node_get_operator_type(
    node: *const ort::OrtNode,
    out: *mut *const c_char,
) -> ort::OrtStatusPtr {
    node_str(node, out, |n| n.op_type.as_ptr(), "Node_GetOperatorType")
}
unsafe extern "C" fn host_node_get_domain(
    node: *const ort::OrtNode,
    out: *mut *const c_char,
) -> ort::OrtStatusPtr {
    node_str(node, out, |n| n.domain.as_ptr(), "Node_GetDomain")
}

fn node_str(
    node: *const ort::OrtNode,
    out: *mut *const c_char,
    f: impl FnOnce(&HostNode) -> *const c_char,
    name: &str,
) -> ort::OrtStatusPtr {
    if node.is_null() || out.is_null() {
        return status_error(&format!(
            "{name} failed: null node/output pointer; fix the plugin to pass valid ORT pointers"
        ));
    }
    // SAFETY: out is checked non-null.
    unsafe {
        *out = f(node_from_ptr(node));
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_node_get_since_version(
    node: *const ort::OrtNode,
    out: *mut i32,
) -> ort::OrtStatusPtr {
    if node.is_null() || out.is_null() {
        return status_error(
            "Node_GetSinceVersion failed: null node/output pointer; fix the plugin to pass valid ORT pointers",
        );
    }
    // SAFETY: out is checked non-null.
    unsafe {
        *out = node_from_ptr(node).since_version;
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_node_get_num_inputs(
    node: *const ort::OrtNode,
    out: *mut usize,
) -> ort::OrtStatusPtr {
    node_count(node, out, |n| n.inputs.len(), "Node_GetNumInputs")
}
unsafe extern "C" fn host_node_get_num_outputs(
    node: *const ort::OrtNode,
    out: *mut usize,
) -> ort::OrtStatusPtr {
    node_count(node, out, |n| n.outputs.len(), "Node_GetNumOutputs")
}
unsafe extern "C" fn host_node_get_num_attributes(
    node: *const ort::OrtNode,
    out: *mut usize,
) -> ort::OrtStatusPtr {
    node_count(node, out, |n| n.attrs.len(), "Node_GetNumAttributes")
}
unsafe extern "C" fn host_node_get_num_implicit_inputs(
    node: *const ort::OrtNode,
    out: *mut usize,
) -> ort::OrtStatusPtr {
    node_count(
        node,
        out,
        |node| node.implicit_inputs.len(),
        "Node_GetNumImplicitInputs",
    )
}

unsafe extern "C" fn host_node_get_implicit_inputs(
    node: *const ort::OrtNode,
    values: *mut *const ort::OrtValueInfo,
    count: usize,
) -> ort::OrtStatusPtr {
    node_values(
        node,
        values,
        count,
        |node| &node.implicit_inputs,
        "Node_GetImplicitInputs",
    )
}

unsafe extern "C" fn host_node_get_num_subgraphs(
    node: *const ort::OrtNode,
    out: *mut usize,
) -> ort::OrtStatusPtr {
    node_count(node, out, |n| n.subgraphs.len(), "Node_GetNumSubgraphs")
}

fn node_count(
    node: *const ort::OrtNode,
    out: *mut usize,
    f: impl FnOnce(&HostNode) -> usize,
    name: &str,
) -> ort::OrtStatusPtr {
    if node.is_null() || out.is_null() {
        return status_error(&format!(
            "{name} failed: null node/output pointer; fix the plugin to pass valid ORT pointers"
        ));
    }
    // SAFETY: out is checked non-null.
    unsafe {
        *out = f(node_from_ptr(node));
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_node_get_inputs(
    node: *const ort::OrtNode,
    values: *mut *const ort::OrtValueInfo,
    count: usize,
) -> ort::OrtStatusPtr {
    node_values(node, values, count, |n| &n.inputs, "Node_GetInputs")
}
unsafe extern "C" fn host_node_get_outputs(
    node: *const ort::OrtNode,
    values: *mut *const ort::OrtValueInfo,
    count: usize,
) -> ort::OrtStatusPtr {
    node_values(node, values, count, |n| &n.outputs, "Node_GetOutputs")
}

fn node_values(
    node: *const ort::OrtNode,
    values: *mut *const ort::OrtValueInfo,
    count: usize,
    f: impl FnOnce(&HostNode) -> &Vec<*const ort::OrtValueInfo>,
    name: &str,
) -> ort::OrtStatusPtr {
    if node.is_null() {
        return status_error(&format!(
            "{name} failed: node pointer was null; fix the plugin to pass a valid node"
        ));
    }
    let src = f(node_from_ptr(node));
    if count < src.len() {
        return status_error(&format!(
            "{name} failed: output buffer has {count} entries but node needs {}; fix by querying the count first",
            src.len()
        ));
    }
    if !src.is_empty() && values.is_null() {
        return status_error(&format!(
            "{name} failed: output buffer pointer was null; fix the plugin allocation"
        ));
    }
    // SAFETY: destination buffer length was validated.
    unsafe {
        ptr::copy_nonoverlapping(src.as_ptr(), values, src.len());
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_node_get_attributes(
    node: *const ort::OrtNode,
    attrs: *mut *const ort::OrtOpAttr,
    count: usize,
) -> ort::OrtStatusPtr {
    if node.is_null() {
        return status_error(
            "Node_GetAttributes failed: node pointer was null; fix the plugin to pass a valid node",
        );
    }
    let node = node_from_ptr(node);
    if count < node.attrs.len() {
        return status_error(
            "Node_GetAttributes failed: output buffer is smaller than attribute count; fix by querying Node_GetNumAttributes first",
        );
    }
    if !node.attrs.is_empty() && attrs.is_null() {
        return status_error(
            "Node_GetAttributes failed: output buffer pointer was null; fix the plugin allocation",
        );
    }
    for (i, attr) in node.attrs.iter().enumerate() {
        // SAFETY: destination buffer length was validated.
        unsafe {
            *attrs.add(i) = (attr as *const HostOpAttr).cast::<ort::OrtOpAttr>();
        }
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_node_get_attribute_by_name(
    node: *const ort::OrtNode,
    name: *const c_char,
    attr: *mut *const ort::OrtOpAttr,
) -> ort::OrtStatusPtr {
    if node.is_null() || name.is_null() || attr.is_null() {
        return status_error(
            "Node_GetAttributeByName failed: null node/name/output pointer; fix the plugin to pass valid ORT pointers",
        );
    }
    // SAFETY: name is checked non-null and follows C ABI.
    let name = unsafe { CStr::from_ptr(name).to_bytes() };
    let found = node_from_ptr(node)
        .attrs
        .iter()
        .find(|a| a.name.as_bytes() == name)
        .map_or(ptr::null(), |a| {
            (a as *const HostOpAttr).cast::<ort::OrtOpAttr>()
        });
    // SAFETY: attr is checked non-null.
    unsafe {
        *attr = found;
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_op_attr_get_type(
    attr: *const ort::OrtOpAttr,
    out: *mut ort::OrtOpAttrType,
) -> ort::OrtStatusPtr {
    if attr.is_null() || out.is_null() {
        return status_error(
            "OpAttr_GetType failed: null attr/output pointer; fix the plugin to pass valid ORT pointers",
        );
    }
    // SAFETY: out is checked non-null.
    unsafe {
        *out = attr_type(&attr_from_ptr(attr).attr);
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_op_attr_get_name(
    attr: *const ort::OrtOpAttr,
    out: *mut *const c_char,
) -> ort::OrtStatusPtr {
    if attr.is_null() || out.is_null() {
        return status_error(
            "OpAttr_GetName failed: null attr/output pointer; fix the plugin to pass valid ORT pointers",
        );
    }
    // SAFETY: out is checked non-null.
    unsafe {
        *out = attr_from_ptr(attr).name.as_ptr();
    }
    ptr::null_mut()
}

fn attr_type(attr: &Attribute) -> ort::OrtOpAttrType {
    match attr {
        Attribute::Int(_) => ort::ORT_OP_ATTR_INT,
        Attribute::Ints(_) => ort::ORT_OP_ATTR_INTS,
        Attribute::Float(_) => ort::ORT_OP_ATTR_FLOAT,
        Attribute::Floats(_) => ort::ORT_OP_ATTR_FLOATS,
        Attribute::String(_) => ort::ORT_OP_ATTR_STRING,
        Attribute::Strings(_) => ort::ORT_OP_ATTR_STRINGS,
        Attribute::Tensor(_) => ort::ORT_OP_ATTR_TENSOR,
        Attribute::Graph(_) | Attribute::Graphs(_) => ort::ORT_OP_ATTR_GRAPH,
        _ => ort::ORT_OP_ATTR_UNDEFINED,
    }
}

unsafe extern "C" fn host_read_op_attr(
    attr: *const ort::OrtOpAttr,
    type_: ort::OrtOpAttrType,
    data: *mut c_void,
    len: usize,
    out: *mut usize,
) -> ort::OrtStatusPtr {
    if attr.is_null() || out.is_null() {
        return status_error(
            "ReadOpAttr failed: null attr/output pointer; fix the plugin to pass valid ORT pointers",
        );
    }
    let attr = &attr_from_ptr(attr).attr;
    if attr_type(attr) != type_ {
        return status_error(
            "ReadOpAttr failed: requested type does not match attribute type; fix the plugin claim predicate",
        );
    }
    let bytes: Vec<u8> = match attr {
        Attribute::Int(v) => v.to_ne_bytes().to_vec(),
        Attribute::Float(v) => v.to_ne_bytes().to_vec(),
        Attribute::Ints(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        Attribute::Floats(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        Attribute::String(v) => v.clone(),
        Attribute::Strings(v) => {
            let mut out = Vec::new();
            for s in v {
                if s.contains(&0) {
                    return status_error(
                        "ReadOpAttr: STRINGS contains an embedded NUL that the ORT delimiter format cannot represent losslessly; use native IR compilation for this attribute",
                    );
                }
                out.extend_from_slice(s);
                out.push(0);
            }
            out
        }
        _ => {
            return status_error(
                "ReadOpAttr failed: this attribute type is not readable by the Stage-1 host; fix by adding an attribute payload projection",
            );
        }
    };
    // SAFETY: out is checked non-null.
    unsafe {
        *out = bytes.len();
    }
    if data.is_null() || len == 0 {
        return ptr::null_mut();
    }
    if len < bytes.len() {
        return status_error(
            "ReadOpAttr failed: output buffer is too small; fix by using returned byte count",
        );
    }
    // SAFETY: buffer length was validated.
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), data.cast::<u8>(), bytes.len());
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_node_get_subgraphs(
    node: *const ort::OrtNode,
    graphs: *mut *const ort::OrtGraph,
    count: usize,
    names: *mut *const c_char,
) -> ort::OrtStatusPtr {
    if node.is_null() {
        return status_error("Node_GetSubgraphs failed: node pointer is null; pass a live node");
    }
    let node = node_from_ptr(node);
    if count < node.subgraphs.len() || (!node.subgraphs.is_empty() && graphs.is_null()) {
        return status_error(
            "Node_GetSubgraphs failed: output storage is null or too small; query Node_GetNumSubgraphs and allocate that many entries",
        );
    }
    for (index, (name, graph)) in node.subgraphs.iter().enumerate() {
        unsafe {
            *graphs.add(index) = graph.as_ort_graph();
            if !names.is_null() {
                *names.add(index) = name.as_ptr();
            }
        }
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_kernel_context_get_input(
    context: *const ort::OrtKernelContext,
    index: usize,
    out: *mut *const ort::OrtValue,
) -> ort::OrtStatusPtr {
    if context.is_null() || out.is_null() {
        return status_error(
            "KernelContext_GetInput failed: null context/output pointer; fix the plugin compute callback",
        );
    }
    let context = unsafe { &*(context.cast::<HostKernelContext>()) };
    let Some(value) = context.inputs.get(index) else {
        return status_error(
            "KernelContext_GetInput failed: input index is out of range; fix the plugin compiled plan boundary",
        );
    };
    unsafe {
        *out = value.as_ref().map_or(ptr::null(), |value| {
            (value as *const HostOrtValue).cast::<ort::OrtValue>()
        });
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_kernel_context_input_count(
    context: *const ort::OrtKernelContext,
    out: *mut usize,
) -> ort::OrtStatusPtr {
    if context.is_null() || out.is_null() {
        return status_error("KernelContext_GetInputCount: null context/output pointer");
    }
    unsafe {
        *out = (&*context.cast::<HostKernelContext>()).inputs.len();
    }
    ptr::null_mut()
}
unsafe extern "C" fn host_kernel_context_output_count(
    context: *const ort::OrtKernelContext,
    out: *mut usize,
) -> ort::OrtStatusPtr {
    if context.is_null() || out.is_null() {
        return status_error("KernelContext_GetOutputCount: null context/output pointer");
    }
    unsafe {
        *out = (&*context.cast::<HostKernelContext>()).outputs.len();
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_kernel_context_get_output(
    context: *mut ort::OrtKernelContext,
    index: usize,
    dim_values: *const i64,
    dim_count: usize,
    out: *mut *mut ort::OrtValue,
) -> ort::OrtStatusPtr {
    if context.is_null() || out.is_null() {
        return status_error(
            "KernelContext_GetOutput failed: null context/output pointer; fix the plugin compute callback",
        );
    }
    let context = unsafe { &mut *(context.cast::<HostKernelContext>()) };
    let Some(value) = context.outputs.get_mut(index) else {
        return status_error(
            "KernelContext_GetOutput failed: output index is out of range; fix the plugin compiled plan boundary",
        );
    };
    if context
        .requested_outputs
        .as_ref()
        .is_some_and(|requested| !requested[index])
    {
        unsafe { *out = ptr::null_mut() };
        return ptr::null_mut();
    }
    if dim_count != 0 && dim_values.is_null() {
        return status_error(
            "KernelContext_GetOutput failed: null dimensions for a non-scalar output",
        );
    }
    let dims = if dim_count == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(dim_values, dim_count) }
    };
    if value.tensor.dims.is_none() {
        let HostOrtValueStorage::Owned(bytes) = &mut value.storage else {
            return status_error("KernelContext_GetOutput: unresolved output must own its storage");
        };
        let shape = match dims
            .iter()
            .map(|&dim| usize::try_from(dim))
            .collect::<std::result::Result<Vec<_>, _>>()
        {
            Ok(shape) => shape,
            Err(_) => {
                return status_error(
                    "KernelContext_GetOutput: negative output extent; report concrete runtime dimensions",
                );
            }
        };
        let Some(dtype) = DataType::from_onnx(value.tensor.dtype as _) else {
            return status_error(
                "KernelContext_GetOutput: unknown output dtype; declare tensor output types before Compile",
            );
        };
        if matches!(dtype, DataType::String | DataType::Undefined) {
            return status_error(
                "KernelContext_GetOutput: dynamic output requires a fixed-width tensor dtype",
            );
        }
        let Some(size) = onnx_runtime_ir::checked_expected_bytes(dtype, &shape) else {
            return status_error(
                "KernelContext_GetOutput: output byte size overflows; reduce output dimensions",
            );
        };
        if bytes.try_reserve_exact(size).is_err() {
            return status_error(
                "KernelContext_GetOutput: cannot allocate dynamic output payload; reduce the output size",
            );
        }
        bytes.resize(size, 0);
        value.tensor.dims = Some(dims.to_vec());
    }
    if value.tensor.dims.as_deref() != Some(dims) {
        return status_error(
            "KernelContext_GetOutput: requested shape differs from the existing output allocation; resolve output geometry before writing",
        );
    }
    unsafe {
        *out = (value as *mut HostOrtValue).cast::<ort::OrtValue>();
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_hardware_device_type(
    _device: *const ort::OrtHardwareDevice,
) -> ort::OrtHardwareDeviceType {
    ort::OrtHardwareDeviceType_CPU
}

unsafe extern "C" fn host_create_ep_device(
    _ep_factory: *mut ort::OrtEpFactory,
    _hardware_device: *const ort::OrtHardwareDevice,
    _ep_metadata: *const ort::OrtKeyValuePairs,
    _ep_options: *const ort::OrtKeyValuePairs,
    ep_device: *mut *mut ort::OrtEpDevice,
) -> ort::OrtStatusPtr {
    if ep_device.is_null() {
        return status_error(
            "CreateEpDevice failed: output pointer was null; fix the plugin to pass valid storage",
        );
    }
    // SAFETY: output pointer is checked. Stage 1 never uses OrtEpDevice contents.
    unsafe {
        *ep_device = ptr::null_mut();
    }
    ptr::null_mut()
}

unsafe extern "C" fn host_ep_graph_support_info_add_nodes_to_fuse(
    support: *mut ort::OrtEpGraphSupportInfo,
    nodes: *const *const ort::OrtNode,
    num_nodes: usize,
    _options: *const ort::OrtNodeFusionOptions,
) -> ort::OrtStatusPtr {
    if support.is_null() {
        return status_error(
            "EpGraphSupportInfo_AddNodesToFuse failed: support pointer was null; fix the plugin to pass ORT's support info",
        );
    }
    if num_nodes > 0 && nodes.is_null() {
        return status_error(
            "EpGraphSupportInfo_AddNodesToFuse failed: nodes pointer was null for a non-empty group; fix the plugin claim list",
        );
    }
    let mut group = Vec::with_capacity(num_nodes);
    for i in 0..num_nodes {
        // SAFETY: nodes points to num_nodes entries by ABI contract, validated non-null above.
        let node = unsafe { *nodes.add(i) };
        if node.is_null() {
            return status_error(
                "EpGraphSupportInfo_AddNodesToFuse failed: group contained a null node; fix the plugin claim list",
            );
        }
        group.push(node_from_ptr(node).id);
    }
    support_from_ptr(support).fused_groups.push(group);
    ptr::null_mut()
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Node, static_shape};

    #[test]
    fn initializer_constness_uses_ir_mutability_semantics() {
        for (version, expected) in [(Some(3), true), (Some(11), false), (None, false)] {
            let mut graph = Graph::new();
            graph.ir_version = version;
            let value = graph.create_named_value("default", DataType::Float32, static_shape([1]));
            graph.add_input(value);
            graph.add_output(value);
            graph.set_initializer(
                value,
                onnx_runtime_ir::WeightRef::Inline(onnx_runtime_ir::TensorData::from_raw(
                    DataType::Float32,
                    vec![1],
                    2f32.to_le_bytes().to_vec(),
                )),
            );
            let host = HostGraph::new(&graph).unwrap();
            let mut constant = !expected;
            assert!(
                unsafe {
                    host_value_info_is_constant_initializer(
                        value_ptr(&host.values[&value]),
                        &mut constant,
                    )
                }
                .is_null()
            );
            assert_eq!(
                constant, expected,
                "IR {version:?} must not turn an overridable default into a cached constant"
            );
        }
    }

    #[test]
    fn graph_metadata_distinguishes_unknown_rank_scalar_and_named_symbols() {
        let mut graph = Graph::new();
        let unknown = graph.create_named_value("unknown", DataType::Float32, vec![]);
        graph.mark_value_shape_unknown(unknown);
        let scalar = graph.create_named_value("scalar", DataType::Float32, vec![]);
        let n = graph.intern_symbol("n");
        let symbolic = graph.create_named_value(
            "symbolic",
            DataType::Float32,
            vec![Dim::Symbolic(n), Dim::Static(2)],
        );
        let unknown = HostTensorTypeAndShapeInfo::for_value(&graph, graph.value(unknown)).unwrap();
        let scalar = HostTensorTypeAndShapeInfo::for_value(&graph, graph.value(scalar)).unwrap();
        let symbolic =
            HostTensorTypeAndShapeInfo::for_value(&graph, graph.value(symbolic)).unwrap();
        let pointer = |info: &HostTensorTypeAndShapeInfo| {
            (info as *const HostTensorTypeAndShapeInfo).cast::<ort::OrtTensorTypeAndShapeInfo>()
        };
        assert!(!unsafe { host_has_shape(pointer(&unknown)) });
        assert!(unsafe { host_has_shape(pointer(&scalar)) });
        let mut count = usize::MAX;
        assert!(
            unsafe { host_get_tensor_shape_element_count(pointer(&scalar), &mut count) }.is_null()
        );
        assert_eq!(count, 1);
        let error = unsafe { host_get_tensor_shape_element_count(pointer(&unknown), &mut count) };
        assert!(!error.is_null());
        release_status(error);
        let mut names = [ptr::null(); 2];
        assert!(
            unsafe { host_get_symbolic_dimensions(pointer(&symbolic), names.as_mut_ptr(), 2) }
                .is_null()
        );
        assert_eq!(unsafe { CStr::from_ptr(names[0]) }, c"n");
        assert_eq!(unsafe { CStr::from_ptr(names[1]) }, c"");
        assert_eq!(symbolic.dims, Some(vec![-1, 2]));
    }

    #[test]
    fn dynamic_plugin_outputs_allocate_once_and_reject_shape_changes() {
        let mut context = HostKernelContext::dynamic(
            &[],
            &[DataType::Float32, DataType::Float32],
            &[true, false],
        )
        .unwrap();
        let context_ptr = (&raw mut context).cast::<ort::OrtKernelContext>();
        let mut value = ptr::null_mut();
        assert!(
            unsafe {
                host_kernel_context_get_output(context_ptr, 0, [2i64].as_ptr(), 1, &mut value)
            }
            .is_null()
        );
        assert!(!value.is_null());
        assert_eq!(context.outputs[0].byte_len(), 8);
        assert!(
            unsafe {
                host_kernel_context_get_output(context_ptr, 0, [2i64].as_ptr(), 1, &mut value)
            }
            .is_null()
        );
        let error = unsafe {
            host_kernel_context_get_output(context_ptr, 0, [4i64].as_ptr(), 1, &mut value)
        };
        assert!(!error.is_null());
        release_status(error);
        assert!(
            unsafe { host_kernel_context_get_output(context_ptr, 1, ptr::null(), 0, &mut value) }
                .is_null()
        );
        assert!(value.is_null());
        let outputs = context.into_dynamic_outputs().unwrap();
        assert_eq!(outputs[0].as_ref().unwrap().shape, [2]);
        assert!(outputs[1].is_none());
        let context = HostKernelContext::dynamic(&[], &[DataType::Float32], &[true]).unwrap();
        assert!(
            context
                .into_dynamic_outputs()
                .unwrap_err()
                .to_string()
                .contains("did not allocate")
        );
    }

    #[test]
    fn tensor_attribute_values_own_numeric_and_string_payloads() {
        for tensor in [
            onnx_runtime_ir::TensorData::from_raw(
                DataType::Float32,
                vec![2],
                [1f32, 2.].into_iter().flat_map(f32::to_le_bytes).collect(),
            ),
            onnx_runtime_ir::TensorData {
                name: None,
                dtype: DataType::String,
                dims: vec![2],
                data: vec![],
                strings: vec![b"a\0b".to_vec(), vec![]],
            },
        ] {
            let attr = HostOpAttr {
                name: c"value".into(),
                attr: Attribute::Tensor(tensor.clone()),
            };
            let mut value = ptr::null_mut();
            assert!(
                unsafe { host_tensor_attribute((&raw const attr).cast(), &mut value) }.is_null()
            );
            drop(attr);
            assert_eq!(ort_value_from_ptr(value).tensor.dims, Some(vec![2]));
            if tensor.dtype == DataType::String {
                let mut len = 0;
                assert!(unsafe { host_string_tensor_data_length(value, &mut len) }.is_null());
                assert_eq!(len, 3);
                let mut bytes = vec![0u8; len];
                let mut offsets = [0usize; 2];
                assert!(
                    unsafe {
                        host_string_tensor_content(
                            value,
                            bytes.as_mut_ptr().cast(),
                            len,
                            offsets.as_mut_ptr(),
                            2,
                        )
                    }
                    .is_null()
                );
                assert_eq!(bytes, b"a\0b");
                assert_eq!(offsets, [0, 3]);
            } else {
                assert_eq!(ort_value_from_ptr(value).byte_len(), 8);
            }
            unsafe { host_release_value(value) };
        }
    }

    #[test]
    fn nested_graph_projection_preserves_names_parent_links_and_inherited_opsets() {
        let mut body = Graph::new();
        let x = body.create_named_value("captured", DataType::Float32, static_shape([2]));
        let y = body.create_named_value("result", DataType::Float32, static_shape([2]));
        body.add_output(y);
        body.insert_node(Node::new(NodeId(0), "Relu", vec![Some(x)], vec![y]));
        let mut graph = Graph::new();
        graph.ir_version = Some(11);
        graph.opset_imports.insert(String::new(), 24);
        let cond = graph.create_named_value("cond", DataType::Bool, vec![]);
        let result = graph.create_named_value("result", DataType::Float32, static_shape([2]));
        graph.add_input(cond);
        let captured = graph.create_named_value("captured", DataType::Float32, static_shape([2]));
        graph.add_input(captured);
        graph.add_output(result);
        let node = graph.insert_node(Node::new(NodeId(0), "If", vec![Some(cond)], vec![result]));
        graph
            .subgraphs
            .insert((node, "then_branch".into()), body.clone());
        graph.subgraphs.insert((node, "else_branch".into()), body);
        let host = HostGraph::new(&graph).unwrap();
        let parent = (&*host.nodes[0] as *const HostNode).cast::<ort::OrtNode>();
        // Moving the outer owner must not invalidate child or parent pointers.
        let moved = Box::new(host);
        let mut count = 0;
        assert!(unsafe { host_node_get_num_subgraphs(parent, &mut count) }.is_null());
        assert_eq!(count, 2);
        let mut graphs = vec![ptr::null(); count];
        let mut names = vec![ptr::null(); count];
        assert!(
            unsafe {
                host_node_get_subgraphs(parent, graphs.as_mut_ptr(), count, names.as_mut_ptr())
            }
            .is_null()
        );
        let names = names
            .iter()
            .map(|name| unsafe { CStr::from_ptr(*name) }.to_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(names, ["else_branch", "then_branch"]);
        for body in graphs {
            let mut ir_version = 0;
            assert!(unsafe { host_graph_ir_version(body, &mut ir_version) }.is_null());
            assert_eq!(ir_version, 11, "nested bodies inherit the model IR version");
            let mut reported_parent = ptr::null();
            assert!(unsafe { host_graph_get_parent_node(body, &mut reported_parent) }.is_null());
            assert_eq!(reported_parent, parent);
            let mut body_nodes = [ptr::null(); 1];
            assert!(unsafe { host_graph_get_nodes(body, body_nodes.as_mut_ptr(), 1) }.is_null());
            let mut version = 0;
            assert!(unsafe { host_node_get_since_version(body_nodes[0], &mut version) }.is_null());
            assert_eq!(version, 24);
        }
        let mut root_parent = parent;
        assert!(
            unsafe { host_graph_get_parent_node(moved.as_ort_graph(), &mut root_parent) }.is_null()
        );
        assert!(root_parent.is_null());
        let status =
            unsafe { host_node_get_subgraphs(parent, ptr::null_mut(), 0, ptr::null_mut()) };
        assert!(!status.is_null());
        release_status(status);
        let status = unsafe { host_node_get_num_subgraphs(ptr::null(), &mut count) };
        assert!(!status.is_null());
        release_status(status);
    }

    #[test]
    fn projects_graph_and_records_fused_claim_boundaries() {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 21);
        let x = graph.create_named_value("x", DataType::Float32, static_shape([2]));
        let y = graph.create_named_value("y", DataType::Float32, static_shape([2]));
        let z = graph.create_named_value("z", DataType::Float32, static_shape([2]));
        graph.add_input(x);
        graph.add_output(z);
        let n0 = graph.insert_node(Node::new(NodeId(999), "Relu", vec![Some(x)], vec![y]));
        let n1 = graph.insert_node(Node::new(
            NodeId(999),
            "Add",
            vec![Some(y), Some(x)],
            vec![z],
        ));

        let host = HostGraph::new(&graph).expect("host graph");
        let mut count = 0usize;
        unsafe { host_graph_get_num_nodes(host.as_ort_graph(), &mut count) };
        assert_eq!(count, 2);
        let mut nodes = vec![ptr::null(); count];
        unsafe { host_graph_get_nodes(host.as_ort_graph(), nodes.as_mut_ptr(), count) };
        let mut support = HostSupportInfo::default();
        unsafe {
            host_ep_graph_support_info_add_nodes_to_fuse(
                &mut support as *mut HostSupportInfo as *mut ort::OrtEpGraphSupportInfo,
                nodes.as_ptr(),
                nodes.len(),
                ptr::null(),
            )
        };
        let claims = support.into_claims(&graph, EpId(7));
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].node_ids, vec![n0, n1]);
        assert_eq!(claims[0].input_values, vec![x]);
        assert_eq!(claims[0].output_values, vec![z]);
    }
}
