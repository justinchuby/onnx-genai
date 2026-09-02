//! Heterogeneous multi-provider graph partitioning and execution (#65).
//!
//! Sessions have historically bound one [`ExecutionProvider`] to the whole
//! executor, so a graph containing a single node the selected provider cannot
//! run is rejected outright (`docs/execution/HETEROGENEOUS_PLACEMENT.md`, issue #65). This
//! module lifts that restriction for the *planning* layer: given an ordered set
//! of providers (front = highest priority), it
//!
//! 1. **places** every node on the highest-priority provider whose capability
//!    oracle ([`ExecutionProvider::supports_op`], the same contract the
//!    GraphView lens / `query_capabilities` projection landed in #76 uses),
//! 2. **partitions** the placed nodes into deterministic, **convex** claims by
//!    reusing the landed [`OrtGraphView::query_capabilities`] union-find + Kahn
//!    acyclicity machinery — one gated capability oracle per provider so each
//!    provider only claims the nodes it exclusively owns, and
//! 3. **plans the cross-provider transfers** (H2D / D2H / D2D) that must be
//!    inserted at partition boundaries, deduplicated so a value that fans out to
//!    several consumers on the same destination device is transferred once.
//!
//! ## Correctness invariant
//!
//! Partitioning, placement, and transfer insertion are an execution-planning
//! optimization, **not** an output change: a graph executed heterogeneously
//! (some nodes on provider A, some on provider B) must produce output
//! byte-identical to the same graph executed entirely on one reference
//! provider. [`execute`] realizes the plan by extracting each partition as a
//! standalone subgraph and running it on its assigned provider through the
//! existing [`Executor`]. Cross-provider values retain one authoritative
//! provider-owned allocation and gain governed destination realizations only
//! for transfer edges in the immutable plan. H2D/D2D copies use
//! `copy_async` + `wait_fence`; D2H uses the source provider's synchronous
//! `copy_to_host` directly into a CPU-owned `DeviceBuffer`, never an ad-hoc host
//! vector.
//!
//! ## Deferred scope
//!
//! This slice deliberately supports fully-static, tensor-only DAGs only.
//! Persistent state bindings, control flow, sequences, view-producing kernels,
//! dynamic shapes, partition-level CUDA-graph capture, and multi-GPU peer
//! copies fail closed before execution (see the design doc §5 and §9).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use onnx_runtime_ep_api::abi::OrtGraphView;
use onnx_runtime_ep_api::{
    DeviceBuffer, DeviceId, DeviceType, EpConfig, EpId, ExecutionProvider, ExecutorKernelScope,
    Fence, Kernel, KernelMatch, Result as EpResult, SubgraphClaim,
};
use onnx_runtime_ir::{
    DataType, Graph, GraphViewCache, ModelFunction, ModelFunctionKey, Node, NodeId, Shape,
    TensorLayout, ValueId, WeightRef, as_static_shape,
};
use onnx_runtime_loader::WeightStore;
use onnx_runtime_tracer::TraceContext;

use crate::error::{Result, SessionError};
use crate::executor::Executor;
use crate::tensor::{DeviceIoBinding, ExternalMemorySpec, Tensor};

/// One provider participating in heterogeneous placement.
///
/// The order of a `&[ProviderPlacement]` is the priority order: the first entry
/// that supports a node wins that node.
#[derive(Clone)]
pub struct ProviderPlacement {
    /// Runtime-local identity, used to key partitions and the execution map.
    pub ep: EpId,
    /// Provider used both as the capability oracle (planning) and to execute
    /// the partitions assigned to it.
    pub provider: Arc<dyn ExecutionProvider>,
}

/// Exhaustive source classification for a value with no producing node.
///
/// An absent optional operand is not a value at all: it is a `None` node-input
/// slot and therefore cannot reach this classification.
enum ProducerlessSource<'input, 'weight> {
    ExternalInput(&'input Tensor),
    Initializer(&'weight WeightRef),
}

/// A maximal convex run of same-provider nodes, executed as one unit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Partition {
    /// Provider assigned to every node in this partition.
    pub ep: EpId,
    /// Device the assigned provider executes on (used for transfer planning).
    pub device: DeviceId,
    /// Member nodes, deterministically ordered (ascending [`NodeId`]).
    pub nodes: Vec<NodeId>,
    /// Boundary values entering the partition from outside it (graph inputs,
    /// initializers, or values produced by earlier partitions).
    pub inputs: Vec<ValueId>,
    /// Boundary values leaving the partition (graph outputs or values consumed
    /// by later partitions).
    pub outputs: Vec<ValueId>,
}

/// A single cross-provider materialization the executor must perform before the
/// destination partition runs. Deduplicated by `(value, to_ep)`: a value that
/// fans out to several partitions on the same destination provider is one
/// transfer. Two providers on the same physical device still require distinct
/// owned realizations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transfer {
    /// The boundary value being moved.
    pub value: ValueId,
    /// Provider that owns the authoritative source allocation.
    pub from_ep: EpId,
    /// Provider that owns the destination realization.
    pub to_ep: EpId,
    /// Device the value is currently authoritative on (host for graph
    /// inputs/initializers).
    pub from: DeviceId,
    /// Device the value must be resident on for the consuming partition.
    pub to: DeviceId,
}

/// The immutable heterogeneous execution plan.
#[derive(Clone, Debug)]
pub struct HeterogeneousPlan {
    /// Partitions in a valid topological execution order.
    pub partitions: Vec<Partition>,
    /// Minimal, deduplicated cross-provider transfers at partition boundaries.
    pub transfers: Vec<Transfer>,
    /// Per-node provider assignment (stable across runs).
    pub node_placement: HashMap<NodeId, EpId>,
    /// Legalized graph used by this plan when assignment-time function fallback
    /// expanded kept model-local function ops.
    /// TODO(hetero-session-phase3): make the public multi-EP executor own this
    /// planned graph directly, including child control-flow hetero plans, instead
    /// of threading it as a compatibility overlay for the Phase-1 planner.
    pub legalized_graph: Option<Arc<Graph>>,
}

/// Capability oracle that reports support only for a fixed set of nodes.
///
/// Wrapping each real provider in one of these and running the landed
/// [`OrtGraphView::query_capabilities`] over it yields that provider's convex
/// claims **restricted to the nodes it exclusively owns**, so overlapping
/// support across providers (e.g. both CPU and CUDA can run `Add`) collapses to
/// a single deterministic assignment without re-deriving the convexity logic.
struct AssignedOracle<'a> {
    inner: &'a dyn ExecutionProvider,
    assigned: &'a HashSet<NodeId>,
}

impl ExecutionProvider for AssignedOracle<'_> {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn device_type(&self) -> DeviceType {
        self.inner.device_type()
    }
    fn device_id(&self) -> DeviceId {
        self.inner.device_id()
    }
    fn initialize(&mut self, _config: &EpConfig) -> EpResult<()> {
        unreachable!("AssignedOracle is a planning-only capability gate")
    }
    fn shutdown(&mut self) -> EpResult<()> {
        unreachable!("AssignedOracle is a planning-only capability gate")
    }
    fn supports_op(
        &self,
        op: &Node,
        opset: u64,
        shapes: &[Shape],
        input_dtypes: &[DataType],
        layouts: &[TensorLayout],
    ) -> KernelMatch {
        if self.assigned.contains(&op.id) {
            self.inner
                .supports_op(op, opset, shapes, input_dtypes, layouts)
        } else {
            KernelMatch::unsupported("node assigned to another provider")
        }
    }
    fn get_kernel(
        &self,
        _op: &Node,
        _shapes: &[Vec<usize>],
        _opset: u64,
    ) -> EpResult<Box<dyn Kernel>> {
        unreachable!("AssignedOracle is a planning-only capability gate")
    }
    fn allocate(&self, _size: usize, _alignment: usize) -> EpResult<DeviceBuffer> {
        unreachable!("AssignedOracle is a planning-only capability gate")
    }
    fn deallocate(&self, _buffer: DeviceBuffer) -> EpResult<()> {
        unreachable!("AssignedOracle is a planning-only capability gate")
    }
    fn copy(&self, _src: &DeviceBuffer, _dst: &mut DeviceBuffer, _size: usize) -> EpResult<()> {
        unreachable!("AssignedOracle is a planning-only capability gate")
    }
    fn copy_async(
        &self,
        _src: &DeviceBuffer,
        _dst: &mut DeviceBuffer,
        _size: usize,
    ) -> EpResult<Fence> {
        unreachable!("AssignedOracle is a planning-only capability gate")
    }
    fn sync(&self) -> EpResult<()> {
        unreachable!("AssignedOracle is a planning-only capability gate")
    }
}

/// Build shapes/dtypes/layouts for `node` from the graph, mirroring the
/// capability probe in `reject_unsupported_operators`.
fn node_capability_inputs(
    graph: &Graph,
    node: &Node,
) -> (Vec<Shape>, Vec<DataType>, Vec<TensorLayout>) {
    let shapes = node
        .inputs
        .iter()
        .map(|input| {
            input
                .map(|value| graph.value(value).shape.clone())
                .unwrap_or_default()
        })
        .collect::<Vec<_>>();
    let dtypes = node
        .inputs
        .iter()
        .map(|input| {
            input
                .map(|value| graph.value(value).dtype)
                .unwrap_or(DataType::Undefined)
        })
        .collect::<Vec<_>>();
    let layouts = node
        .inputs
        .iter()
        .map(|input| {
            input
                .map(|value| graph.value(value).layout.clone())
                .unwrap_or_else(TensorLayout::contiguous)
        })
        .collect::<Vec<_>>();
    (shapes, dtypes, layouts)
}

/// Assign every node to the highest-priority provider that supports it.
///
/// Returns an error naming the first node no provider can run, so placement
/// fails before any execution (design invariant §6.7).
fn assign_nodes(graph: &Graph, providers: &[ProviderPlacement]) -> Result<HashMap<NodeId, EpId>> {
    let mut placement = HashMap::new();
    for (node_id, node) in graph.nodes.iter() {
        let opset = graph.effective_opset(node).unwrap_or(u64::MAX);
        let (shapes, dtypes, layouts) = node_capability_inputs(graph, node);
        let matches = providers
            .iter()
            .map(|slot| {
                (
                    slot,
                    slot.provider
                        .supports_op(node, opset, &shapes, &dtypes, &layouts),
                )
            })
            .collect::<Vec<_>>();
        let chosen = matches
            .iter()
            .find(|(_, capability)| capability.is_supported());
        match chosen {
            Some((slot, _)) => {
                placement.insert(node_id, slot.ep);
            }
            None => {
                let providers = matches
                    .iter()
                    .map(|(slot, _)| slot.provider.name())
                    .collect::<Vec<_>>()
                    .join(", ");
                let reason = matches
                    .iter()
                    .map(|(slot, capability)| {
                        format!(
                            "{}: {}",
                            slot.provider.name(),
                            capability.reason().unwrap_or("supported")
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                return Err(SessionError::unsupported_op(
                    node, node_id, opset, providers, reason,
                ));
            }
        }
    }
    Ok(placement)
}

fn provider_by_id(providers: &[ProviderPlacement], ep: EpId) -> Option<&ProviderPlacement> {
    providers.iter().find(|slot| slot.ep == ep)
}

fn model_function_key(node: &Node) -> ModelFunctionKey {
    (node.domain.clone(), node.op_type.clone())
}

fn function_like_node_bound(graph: &Graph) -> usize {
    let mut count = graph
        .nodes
        .iter()
        .filter(|(_, node)| {
            let key = model_function_key(node);
            graph.model_functions.contains_key(&key)
                || graph.ambiguous_model_functions.contains(&key)
        })
        .count();
    for subgraph in graph.subgraphs.values() {
        count += function_like_node_bound(subgraph);
    }
    for function in graph.model_functions.values() {
        count += function_like_node_bound(&function.body);
    }
    count
}

fn unsupported_function_nodes(
    graph: &Graph,
    providers: &[ProviderPlacement],
) -> Result<Vec<NodeId>> {
    let mut out = Vec::new();
    for (node_id, node) in graph.nodes.iter() {
        let opset = graph.effective_opset(node).unwrap_or(u64::MAX);
        let (shapes, dtypes, layouts) = node_capability_inputs(graph, node);
        if providers.iter().any(|slot| {
            slot.provider
                .supports_op(node, opset, &shapes, &dtypes, &layouts)
                .is_supported()
        }) {
            continue;
        }
        let key = model_function_key(node);
        if graph.ambiguous_model_functions.contains(&key) {
            return Err(SessionError::Internal(format!(
                "cannot legalize model-local function node {}::{} (node #{}): \
                 multiple overloads share this (domain, op_type); add overload-aware \
                 function metadata before heterogeneous planning",
                node.domain, node.op_type, node_id.0
            )));
        }
        if graph.model_functions.contains_key(&key) {
            out.push(node_id);
        } else {
            let reason = providers
                .iter()
                .map(|slot| slot.provider.name())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(SessionError::unsupported_op(
                node,
                node_id,
                opset,
                reason,
                "no registered provider supports this operator at this opset/shape",
            ));
        }
    }
    Ok(out)
}

fn assigned_function_mismatches(
    graph: &Graph,
    providers: &[ProviderPlacement],
    placement: &HashMap<NodeId, EpId>,
) -> Result<Vec<NodeId>> {
    let mut out = Vec::new();
    for (&node_id, &ep) in placement {
        let node = graph.node(node_id);
        let key = model_function_key(node);
        if !graph.model_functions.contains_key(&key)
            && !graph.ambiguous_model_functions.contains(&key)
        {
            continue;
        }
        if graph.ambiguous_model_functions.contains(&key) {
            return Err(SessionError::Internal(format!(
                "cannot legalize model-local function node {}::{} (node #{}): \
                 multiple overloads share this (domain, op_type); add overload-aware \
                 function metadata before heterogeneous planning",
                node.domain, node.op_type, node_id.0
            )));
        }
        let Some(slot) = provider_by_id(providers, ep) else {
            return Err(SessionError::Internal(format!(
                "no execution provider registered for EpId({})",
                ep.0
            )));
        };
        let opset = graph.effective_opset(node).unwrap_or(u64::MAX);
        let (shapes, dtypes, layouts) = node_capability_inputs(graph, node);
        if !slot
            .provider
            .supports_op(node, opset, &shapes, &dtypes, &layouts)
            .is_supported()
        {
            out.push(node_id);
        }
    }
    Ok(out)
}

fn value_by_name(graph: &Graph, name: &str) -> Option<ValueId> {
    graph
        .values
        .iter()
        .find_map(|(vid, value)| (value.name.as_deref() == Some(name)).then_some(vid))
}

fn create_remapped_value(
    parent: &mut Graph,
    body: &Graph,
    value: ValueId,
    name: String,
) -> ValueId {
    let metadata = body.value(value);
    let new_value = parent.create_named_value(name, metadata.dtype, metadata.shape.clone());
    if !body.value_type_is_known(value) {
        parent.mark_value_type_unknown(new_value);
    }
    if !body.value_shape_is_known(value) {
        parent.mark_value_shape_unknown(new_value);
    }
    if let Some(weight) = body.initializers.get(&value) {
        parent.set_initializer(new_value, weight.clone());
    }
    new_value
}

fn function_has_attribute_parameters(call: &Node, function: &ModelFunction) -> bool {
    // Phase 1 IR function bodies intentionally do not preserve
    // AttributeProto::ref_attr_name, and the IR-level inliner below only copies
    // body attributes. Be conservative: any FunctionProto formal attribute, any
    // proto-level ref_attr_name captured before IR conversion, or any call-site
    // attributes on the kept function require proto-level binding and must fail
    // closed here rather than silently inlining defaults/missing attributes.
    !function.attributes.is_empty() || function.has_attribute_refs || !call.attributes.is_empty()
}

fn body_has_control_flow(body: &Graph) -> bool {
    // TODO(hetero-function-phase2): support full scope-aware Graph/Graphs
    // attribute remapping by sharing the proto inliner's FunctionLibrary-grade
    // expansion primitive. Phase 1 handles functions that appear inside
    // subgraphs, but not function bodies that themselves contain control flow.
    !body.subgraphs.is_empty()
        || body.nodes.iter().any(|(_, node)| {
            node.attributes.values().any(|attr| {
                matches!(
                    attr,
                    onnx_runtime_ir::Attribute::Graph(_) | onnx_runtime_ir::Attribute::Graphs(_)
                )
            })
        })
}

fn inline_model_function_node(graph: &mut Graph, node_id: NodeId) -> Result<()> {
    let call = graph.node(node_id).clone();
    let key = model_function_key(&call);
    if graph.ambiguous_model_functions.contains(&key) {
        return Err(SessionError::Internal(format!(
            "cannot legalize model-local function node {}::{} (node #{}): \
             multiple overloads share this (domain, op_type); add overload-aware \
             function metadata before heterogeneous planning",
            call.domain, call.op_type, node_id.0
        )));
    }
    let function: ModelFunction = graph.model_functions.get(&key).cloned().ok_or_else(|| {
        SessionError::Internal(format!(
            "node {}::{} (node #{}) is not a model-local function",
            call.domain, call.op_type, node_id.0
        ))
    })?;
    if body_has_control_flow(&function.body) {
        return Err(SessionError::Internal(format!(
            "cannot legalize model-local function node {}::{} (node #{}): \
             function bodies containing control-flow subgraphs require the deferred \
             overload-aware FunctionLibrary/IR splicer",
            call.domain, call.op_type, node_id.0
        )));
    }
    if function_has_attribute_parameters(&call, &function) {
        // Phase 2: bind call-site/ref_attr_name attributes by sharing the
        // proto-level function_inline primitive instead of this IR-only splicer.
        return Err(SessionError::Internal(format!(
            "cannot legalize model-local function node {}::{} (node #{}): \
             attribute-parameterized function legalization at assignment time is not yet supported; \
             Phase 2 must bind call-site/ref_attr_name attributes via the shared function_inline primitive",
            call.domain, call.op_type, node_id.0
        )));
    }

    let mut remap: HashMap<ValueId, Option<ValueId>> = HashMap::new();
    for (index, formal) in function.inputs.iter().enumerate() {
        if let Some(body_value) = value_by_name(&function.body, formal) {
            remap.insert(body_value, call.inputs.get(index).copied().flatten());
        }
    }

    let mut aliases: Vec<(ValueId, ValueId)> = Vec::new();
    for (index, formal) in function.outputs.iter().enumerate() {
        let Some(&actual) = call.outputs.get(index) else {
            continue;
        };
        let Some(body_value) = value_by_name(&function.body, formal) else {
            continue;
        };
        if function.body.value(body_value).producer.is_some() {
            remap.insert(body_value, Some(actual));
        } else if let Some(Some(src)) = remap.get(&body_value).copied() {
            if src != actual {
                aliases.push((src, actual));
            }
        } else {
            remap.insert(body_value, Some(actual));
        }
    }

    let mut fresh_index = 0usize;
    let mut get_or_create = |parent: &mut Graph,
                             remap: &mut HashMap<ValueId, Option<ValueId>>,
                             value: ValueId|
     -> Option<ValueId> {
        if let Some(mapped) = remap.get(&value) {
            return *mapped;
        }
        let base = function
            .body
            .value(value)
            .name
            .clone()
            .unwrap_or_else(|| format!("v{}", value.0));
        let name = format!("__fn{}_{}_{}", node_id.0, fresh_index, base);
        fresh_index += 1;
        let new_value = create_remapped_value(parent, &function.body, value, name);
        remap.insert(value, Some(new_value));
        Some(new_value)
    };

    let order = function.body.topological_order().map_err(|error| {
        SessionError::Internal(format!(
            "cannot legalize model-local function node {}::{} (node #{}): function body is invalid: {error}",
            call.domain, call.op_type, node_id.0
        ))
    })?;
    let mut new_nodes = Vec::new();
    for body_node_id in order {
        let body_node = function.body.node(body_node_id);
        let inputs = body_node
            .inputs
            .iter()
            .map(|slot| slot.and_then(|value| get_or_create(graph, &mut remap, value)))
            .collect::<Vec<_>>();
        let outputs = body_node
            .outputs
            .iter()
            .filter_map(|&value| get_or_create(graph, &mut remap, value))
            .collect::<Vec<_>>();
        let mut node = Node::new(NodeId(0), body_node.op_type.clone(), inputs, outputs);
        node.name = if body_node.name.is_empty() {
            format!("__fn{}_n{}", node_id.0, body_node_id.0)
        } else {
            format!("__fn{}_{}", node_id.0, body_node.name)
        };
        node.domain = body_node.domain.clone();
        node.version = body_node.version;
        node.attributes = body_node.attributes.clone();
        node.doc_string = body_node.doc_string.clone();
        new_nodes.push(node);
    }
    for (index, (src, dst)) in aliases.into_iter().enumerate() {
        let mut alias = Node::new(NodeId(0), "Identity", vec![Some(src)], vec![dst]);
        alias.name = format!("__fn{}_alias{}", node_id.0, index);
        new_nodes.push(alias);
    }

    graph.remove_node(node_id);
    for node in new_nodes {
        graph.insert_node(node);
    }
    Ok(())
}

fn refresh_shapes(graph: &mut Graph) -> Result<()> {
    let registry = onnx_runtime_shape_inference::InferenceRegistry::default_registry();
    let opset_imports = graph.opset_imports.clone();
    registry.infer_graph(
        graph,
        &opset_imports,
        onnx_runtime_shape_inference::MergePolicy::Permissive,
    )?;
    Ok(())
}

fn legalize_function_fallbacks(
    graph: &mut Graph,
    providers: &[ProviderPlacement],
    max_iterations: usize,
) -> Result<bool> {
    let mut changed = false;
    for iteration in 0..=max_iterations {
        let placement = match assign_nodes(graph, providers) {
            Ok(placement) => placement,
            Err(_) => {
                let unsupported = unsupported_function_nodes(graph, providers)?;
                if unsupported.is_empty() {
                    return assign_nodes(graph, providers).map(|_| false);
                }
                for node in unsupported {
                    inline_model_function_node(graph, node)?;
                }
                refresh_shapes(graph)?;
                changed = true;
                continue;
            }
        };
        let mismatches = assigned_function_mismatches(graph, providers, &placement)?;
        if mismatches.is_empty() {
            let model_functions = graph.model_functions.clone();
            let ambiguous_model_functions = graph.ambiguous_model_functions.clone();
            for subgraph in graph.subgraphs.values_mut() {
                if subgraph.model_functions.is_empty() {
                    subgraph.model_functions = model_functions.clone();
                }
                if subgraph.ambiguous_model_functions.is_empty() {
                    subgraph.ambiguous_model_functions = ambiguous_model_functions.clone();
                }
                changed |= legalize_function_fallbacks(subgraph, providers, max_iterations)?;
            }
            return Ok(changed);
        }
        if iteration == max_iterations {
            return Err(SessionError::Internal(format!(
                "unstable/recursive function legalization after {max_iterations} iterations"
            )));
        }
        for node in mismatches {
            inline_model_function_node(graph, node)?;
        }
        refresh_shapes(graph)?;
        changed = true;
    }
    Err(SessionError::Internal(
        "unstable/recursive function legalization exceeded iteration bound".into(),
    ))
}

/// Emit convex partitions for one provider by running the landed
/// `query_capabilities` over a gated oracle.
fn provider_partitions(
    graph: &Graph,
    cache: &GraphViewCache,
    slot: &ProviderPlacement,
    placement: &HashMap<NodeId, EpId>,
) -> Vec<Partition> {
    let assigned: HashSet<NodeId> = placement
        .iter()
        .filter(|&(_, &ep)| ep == slot.ep)
        .map(|(&node, _)| node)
        .collect();
    if assigned.is_empty() {
        return Vec::new();
    }
    let oracle = AssignedOracle {
        inner: slot.provider.as_ref(),
        assigned: &assigned,
    };
    let view = onnx_runtime_ir::GraphView::new(graph, cache);
    let claims: Vec<SubgraphClaim> = OrtGraphView::new(&view).query_capabilities(&oracle);
    claims
        .into_iter()
        .map(|claim| {
            let mut nodes = claim.node_ids;
            nodes.sort_unstable_by_key(|n| n.0);
            Partition {
                ep: slot.ep,
                device: slot.provider.device_id(),
                nodes,
                inputs: claim.input_values,
                outputs: claim.output_values,
            }
        })
        .collect()
}

/// Kahn topological sort of the partition DAG, deterministic by
/// `(min member topo-rank, ep, first node)`.
fn order_partitions(
    graph: &Graph,
    partitions: Vec<Partition>,
    value_to_partition: &HashMap<ValueId, usize>,
) -> Result<Vec<Partition>> {
    let topo = graph.topological_order()?;
    let mut rank = HashMap::new();
    for (index, node) in topo.iter().enumerate() {
        rank.insert(*node, index);
    }
    let min_rank = |p: &Partition| {
        p.nodes
            .iter()
            .filter_map(|n| rank.get(n))
            .min()
            .copied()
            .unwrap_or(0)
    };

    let n = partitions.len();
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut indeg = vec![0usize; n];
    let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
    for (j, part) in partitions.iter().enumerate() {
        for input in &part.inputs {
            if let Some(&i) = value_to_partition.get(input)
                && i != j
                && edge_set.insert((i, j))
            {
                succ[i].push(j);
                indeg[j] += 1;
            }
        }
    }

    // Deterministic ready-set keyed by (min_rank, ep, first node id) → index.
    let sort_key = |part: &Partition| {
        (
            min_rank(part),
            part.ep.0,
            part.nodes.first().map(|n| n.0).unwrap_or(u32::MAX),
        )
    };
    let mut ready: BTreeMap<(usize, u32, u32), usize> = BTreeMap::new();
    for i in 0..n {
        if indeg[i] == 0 {
            ready.insert(sort_key(&partitions[i]), i);
        }
    }
    let mut ordered = Vec::with_capacity(n);
    while let Some((&key, &i)) = ready.iter().next() {
        ready.remove(&key);
        ordered.push(i);
        for &j in &succ[i] {
            indeg[j] -= 1;
            if indeg[j] == 0 {
                ready.insert(sort_key(&partitions[j]), j);
            }
        }
    }
    if ordered.len() != n {
        return Err(SessionError::Internal(
            "heterogeneous partition graph is cyclic (non-convex partitioning)".into(),
        ));
    }

    // Reindex partitions into topological order.
    let mut out: Vec<Option<Partition>> = partitions.into_iter().map(Some).collect();
    Ok(ordered
        .into_iter()
        .map(|i| out[i].take().unwrap())
        .collect())
}

/// Compute the minimal cross-provider transfers at partition boundaries.
fn plan_transfers(
    partitions: &[Partition],
    value_to_partition: &HashMap<ValueId, usize>,
) -> Vec<Transfer> {
    // Dedup by (value, destination EP); a value fanning out to several
    // partitions on the same provider is materialized once.
    let mut seen: HashSet<(u32, EpId)> = HashSet::new();
    let mut transfers = Vec::new();
    for part in partitions {
        for &input in &part.inputs {
            let Some(&producer_index) = value_to_partition.get(&input) else {
                // Graph inputs and initializers are staged by the assigned
                // partition executor; transfer edges represent EP crossings
                // only.
                continue;
            };
            let producer = &partitions[producer_index];
            let from = producer.device;
            let to = part.device;
            if producer.ep != part.ep && seen.insert((input.0, part.ep)) {
                transfers.push(Transfer {
                    value: input,
                    from_ep: producer.ep,
                    to_ep: part.ep,
                    from,
                    to,
                });
            }
        }
    }
    transfers.sort_by_key(|t| (t.value.0, t.to_ep.0));
    transfers
}

fn validate_execution_subset(graph: &Graph) -> Result<()> {
    if !graph.subgraphs.is_empty() {
        return Err(SessionError::HeterogeneousExecutionUnsupported {
            placement_summary: format!(
                "the graph contains {} control-flow subgraph body/bodies; the first execution \
                 slice supports only tensor-only acyclic graphs without If/Loop/Scan. Keep \
                 ONNX_GENAI_HETERO unset until child heterogeneous executors are implemented",
                graph.subgraphs.len()
            ),
        });
    }
    for (node_id, node) in graph.nodes.iter() {
        let domain = if node.domain.is_empty() {
            "ai.onnx"
        } else {
            node.domain.as_str()
        };
        let opset = graph.effective_opset(node).unwrap_or(u64::MAX);
        if crate::executor::is_control_flow_op(&node.op_type, &node.domain) {
            return Err(SessionError::HeterogeneousExecutionUnsupported {
                placement_summary: format!(
                    "node {} ({domain}::{}@{opset}) is control flow; the first execution slice \
                     does not build heterogeneous child executors",
                    node_id.0, node.op_type
                ),
            });
        }
        if crate::executor::is_sequence_op(&node.op_type, &node.domain) {
            return Err(SessionError::HeterogeneousExecutionUnsupported {
                placement_summary: format!(
                    "node {} ({domain}::{}@{opset}) consumes or produces a sequence; the first \
                     execution slice supports tensor values only",
                    node_id.0, node.op_type
                ),
            });
        }
        if onnx_runtime_loader::is_ep_context_op(&node.op_type, &node.domain) {
            return Err(SessionError::HeterogeneousExecutionUnsupported {
                placement_summary: format!(
                    "node {} ({domain}::{}@{opset}) is a compiled EPContext; the first execution \
                     slice cannot repartition an already-compiled provider context",
                    node_id.0, node.op_type
                ),
            });
        }
    }
    for (value_id, value) in graph.values.iter() {
        if as_static_shape(&value.shape).is_none() {
            return Err(SessionError::HeterogeneousExecutionUnsupported {
                placement_summary: format!(
                    "value {} ({}) has symbolic shape {:?}; the first execution slice requires \
                     fully static tensor shapes so all provider kernels and governed boundary \
                     allocations are validated before any node executes",
                    value_id.0,
                    value.name.as_deref().unwrap_or("<unnamed>"),
                    value.shape
                ),
            });
        }
    }
    Ok(())
}

/// Plan a heterogeneous execution of `graph` over `providers` (priority order).
pub fn plan(graph: &Graph, providers: &[ProviderPlacement]) -> Result<HeterogeneousPlan> {
    if providers.is_empty() {
        return Err(SessionError::Internal(
            "heterogeneous placement requires at least one provider".into(),
        ));
    }
    let mut planned_graph = graph.clone();
    let bound = function_like_node_bound(&planned_graph)
        + planned_graph.model_functions.len()
        + planned_graph.ambiguous_model_functions.len();
    let legalized = if bound > 0 {
        legalize_function_fallbacks(&mut planned_graph, providers, bound)?
    } else {
        false
    };
    let graph = &planned_graph;
    validate_execution_subset(graph)?;

    let cache = GraphViewCache::build(graph)
        .map_err(|e| SessionError::Internal(format!("graph view: {e}")))?;
    let node_placement = assign_nodes(graph, providers)?;

    let mut partitions = Vec::new();
    for slot in providers {
        partitions.extend(provider_partitions(graph, &cache, slot, &node_placement));
    }

    // Map every produced value to its producing partition (pre-ordering ids are
    // fine; the map is rebuilt against the final order below).
    let mut value_to_partition = HashMap::new();
    for (index, part) in partitions.iter().enumerate() {
        for &node in &part.nodes {
            for &out in &graph.node(node).outputs {
                value_to_partition.insert(out, index);
            }
        }
    }

    let partitions = order_partitions(graph, partitions, &value_to_partition)?;

    // Rebuild the value→partition map against the topological order.
    let mut value_to_partition = HashMap::new();
    for (index, part) in partitions.iter().enumerate() {
        for &node in &part.nodes {
            for &out in &graph.node(node).outputs {
                value_to_partition.insert(out, index);
            }
        }
    }
    let transfers = plan_transfers(&partitions, &value_to_partition);

    Ok(HeterogeneousPlan {
        partitions,
        transfers,
        node_placement,
        legalized_graph: legalized.then(|| Arc::new(planned_graph)),
    })
}

/// Outcome of classifying a graph against an ordered provider set for the
/// default session build path (Thread-3 Phase 3).
#[derive(Debug)]
pub enum PlacementDecision {
    /// Every node is claimed by a single provider and no cross-provider transfer
    /// is needed: the caller keeps its existing single-EP executor unchanged
    /// (byte-identical fast path). Carries the winning [`EpId`].
    SingleProvider(EpId),
    /// Nodes are split across providers; realizing this requires the
    /// heterogeneous executor. Carries the concrete [`HeterogeneousPlan`].
    Heterogeneous(Box<HeterogeneousPlan>),
}

/// Classify `graph` against `providers` (priority order, front = highest).
///
/// Runs the Phase-1 [`plan`] and collapses its result: a graph is homogeneous
/// iff every node landed on the same provider. (A single provider on a non-host
/// device still needs its inputs staged H2D, so transfer count is not the
/// signal — a lone provider's input staging is ordinary single-EP behavior, not
/// a cross-EP split.) Homogeneous graphs return
/// [`PlacementDecision::SingleProvider`] so the caller keeps its single-EP path
/// untouched; anything else returns the concrete per-node [`HeterogeneousPlan`].
pub fn classify_placement(
    graph: &Graph,
    providers: &[ProviderPlacement],
) -> Result<PlacementDecision> {
    let plan = plan(graph, providers)?;
    let distinct: HashSet<EpId> = plan.node_placement.values().copied().collect();
    if distinct.len() <= 1 {
        // Prefer the actually-assigned EP; an empty graph falls back to the
        // highest-priority provider so the caller always gets a concrete id.
        let ep = distinct
            .into_iter()
            .next()
            .unwrap_or_else(|| providers[0].ep);
        Ok(PlacementDecision::SingleProvider(ep))
    } else {
        Ok(PlacementDecision::Heterogeneous(Box::new(plan)))
    }
}

/// Human-readable, per-op placement summary for diagnostics and fail-closed
/// errors. Names the op classes forced onto every non-primary (fallback)
/// provider so the operator forcing a split is visible without a GPU.
pub fn placement_summary(
    plan: &HeterogeneousPlan,
    graph: &Graph,
    providers: &[ProviderPlacement],
) -> String {
    // Node ids in the plan refer to the legalized graph when function fallback
    // expanded kept ops; use it for op-name lookups so the summary is accurate.
    let graph = plan.legalized_graph.as_deref().unwrap_or(graph);
    let primary = providers.first().map(|slot| slot.ep);
    let device_name = |ep: EpId| -> String {
        providers
            .iter()
            .find(|slot| slot.ep == ep)
            .map(|slot| {
                let device = slot.provider.device_id();
                format!(
                    "{} ({}:{})",
                    slot.provider.name(),
                    device.device_type.trace_name(),
                    device.index
                )
            })
            .unwrap_or_else(|| format!("ep#{}", ep.0))
    };

    let mut per_ep: BTreeMap<u32, usize> = BTreeMap::new();
    for &ep in plan.node_placement.values() {
        *per_ep.entry(ep.0).or_default() += 1;
    }
    let ep_counts = per_ep
        .iter()
        .map(|(&ep, &count)| format!("{count} node(s) on {}", device_name(EpId(ep))))
        .collect::<Vec<_>>()
        .join(", ");

    let mut fallback_ops: BTreeSet<String> = BTreeSet::new();
    for (&node, &ep) in &plan.node_placement {
        if Some(ep) != primary {
            let node = graph.node(node);
            let domain = if node.domain.is_empty() {
                "ai.onnx"
            } else {
                node.domain.as_str()
            };
            fallback_ops.insert(format!("{domain}::{}", node.op_type));
        }
    }
    let fallback = if fallback_ops.is_empty() {
        "none".to_string()
    } else {
        fallback_ops.into_iter().collect::<Vec<_>>().join(", ")
    };

    format!(
        "{} partition(s) across {} device(s); {ep_counts}; {} cross-provider transfer(s); \
         ops forced onto a fallback provider: {fallback}",
        plan.partitions.len(),
        per_ep.len(),
        plan.transfers.len(),
    )
}

/// Guard the default session build path when per-op heterogeneous *planning* is
/// available but integrated stateful *execution* is not yet wired (Thread-3
/// Phase 3, deferred parts tracked under #603).
///
/// When `enabled`, classify `graph` over `providers`. A genuinely homogeneous
/// graph returns `Ok(())`, leaving the caller's single-EP / whole-session
/// fallback path untouched (byte-identical). A graph that genuinely needs
/// per-node placement across providers **fails closed** with an actionable
/// [`SessionError::HeterogeneousExecutionUnsupported`] naming the offending ops,
/// rather than letting the caller silently drop the whole session onto one
/// fallback provider. When `!enabled` this is a no-op, so the default path is
/// unchanged.
pub fn guard_heterogeneous_fallback(
    graph: &Graph,
    providers: &[ProviderPlacement],
    enabled: bool,
) -> Result<()> {
    if !enabled {
        return Ok(());
    }
    match classify_placement(graph, providers)? {
        PlacementDecision::SingleProvider(_) => Ok(()),
        PlacementDecision::Heterogeneous(plan) => {
            Err(SessionError::HeterogeneousExecutionUnsupported {
                placement_summary: placement_summary(&plan, graph, providers),
            })
        }
    }
}

/// Deterministic per-value name used to feed subgraph inputs by name.
fn subgraph_value_name(value: ValueId) -> String {
    format!("hetero_v{}", value.0)
}

/// Extract one partition as a standalone runnable subgraph.
///
/// Returns the subgraph, the ordered `(parent value, feed name)` pairs for
/// runtime inputs (boundary inputs that are not initializers), and the ordered
/// parent output values matching the subgraph's output order.
struct SubgraphExtraction {
    graph: Graph,
    input_feeds: Vec<(ValueId, String)>,
    output_values: Vec<ValueId>,
}

fn extract_subgraph(parent: &Graph, partition: &Partition) -> SubgraphExtraction {
    let mut sub = Graph::new();
    sub.opset_imports = parent.opset_imports.clone();

    let mut remap: HashMap<ValueId, ValueId> = HashMap::new();
    let get_or_create =
        |sub: &mut Graph, remap: &mut HashMap<ValueId, ValueId>, v: ValueId| -> ValueId {
            if let Some(&child) = remap.get(&v) {
                return child;
            }
            let pv = parent.value(v);
            let child = sub.create_named_value(subgraph_value_name(v), pv.dtype, pv.shape.clone());
            remap.insert(v, child);
            child
        };

    // Boundary inputs: initializers are re-attached, everything else becomes a
    // named graph input fed from the host value map.
    let mut input_feeds = Vec::new();
    for &v in &partition.inputs {
        let child = get_or_create(&mut sub, &mut remap, v);
        if let Some(weight) = parent.initializers.get(&v) {
            sub.set_initializer(child, weight.clone());
        } else {
            sub.add_input(child);
            input_feeds.push((v, subgraph_value_name(v)));
        }
    }

    // Recreate each member node with remapped edges, preserving op identity.
    for &node_id in &partition.nodes {
        let node = parent.node(node_id);
        let inputs = node
            .inputs
            .iter()
            .map(|slot| slot.map(|v| get_or_create(&mut sub, &mut remap, v)))
            .collect::<Vec<_>>();
        let outputs = node
            .outputs
            .iter()
            .map(|&v| get_or_create(&mut sub, &mut remap, v))
            .collect::<Vec<_>>();
        let mut new_node = Node::new(NodeId(0), node.op_type.clone(), inputs, outputs);
        new_node.name = node.name.clone();
        new_node.domain = node.domain.clone();
        new_node.version = node.version;
        new_node.attributes = node.attributes.clone();
        sub.insert_node(new_node);
    }

    // Partition outputs become graph outputs, in a deterministic order.
    let mut output_values = partition.outputs.clone();
    output_values.sort_unstable_by_key(|v| v.0);
    for &v in &output_values {
        let child = get_or_create(&mut sub, &mut remap, v);
        sub.add_output(child);
    }

    SubgraphExtraction {
        graph: sub,
        input_feeds,
        output_values,
    }
}

struct ResidentValue {
    dtype: DataType,
    shape: Vec<usize>,
    owner: Arc<dyn ExecutionProvider>,
    buffer: Option<DeviceBuffer>,
}

impl ResidentValue {
    fn allocate(
        _ep: EpId,
        dtype: DataType,
        shape: Vec<usize>,
        owner: Arc<dyn ExecutionProvider>,
    ) -> Result<Self> {
        let elements = shape
            .iter()
            .try_fold(1usize, |product, &extent| product.checked_mul(extent));
        let bytes = elements
            .and_then(|elements| dtype.checked_storage_bytes(elements))
            .ok_or_else(|| SessionError::ShapeOverflow {
                value: "heterogeneous boundary value".into(),
                dims: shape.clone(),
            })?
            .max(1);
        let alignment = TensorLayout::contiguous().alignment;
        let committed = 0..bytes;
        let buffer =
            owner.allocate_committed(bytes, alignment, std::slice::from_ref(&committed))?;
        Ok(Self {
            dtype,
            shape,
            owner,
            buffer: Some(buffer),
        })
    }

    fn buffer(&self) -> &DeviceBuffer {
        self.buffer
            .as_ref()
            .expect("resident buffer is taken only during Drop")
    }

    fn buffer_mut(&mut self) -> &mut DeviceBuffer {
        self.buffer
            .as_mut()
            .expect("resident buffer is taken only during Drop")
    }

    fn release(mut self) -> Result<()> {
        if let Some(buffer) = self.buffer.take() {
            self.owner.deallocate(buffer)?;
        }
        Ok(())
    }
}

impl Drop for ResidentValue {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            let _ = self.owner.deallocate(buffer);
        }
    }
}

struct PartitionExecutor {
    partition: Partition,
    input_feeds: Vec<(ValueId, String)>,
    output_values: Vec<ValueId>,
    executor: Executor,
}

/// Persistent executor for the opt-in, static tensor-only heterogeneous slice.
///
/// Every partition is an ordinary [`Executor`] on its assigned EP. Boundary
/// storage is allocated through that EP, transferred exactly once per planned
/// `(value, destination EP)` realization, and released by the allocating EP
/// after the value's last consuming partition. CUDA graph capture and persistent
/// external bindings are intentionally rejected by the parent executor.
pub(crate) struct HeterogeneousExecutor {
    graph: Arc<Graph>,
    weights: Arc<WeightStore>,
    plan: HeterogeneousPlan,
    providers: HashMap<EpId, Arc<dyn ExecutionProvider>>,
    partitions: Vec<PartitionExecutor>,
    placement_report: String,
    last_transfer_count: usize,
    last_release_counts: HashMap<(ValueId, EpId), usize>,
}

impl HeterogeneousExecutor {
    pub(crate) fn build(
        plan: HeterogeneousPlan,
        graph: &Graph,
        weights: &Arc<WeightStore>,
        providers: &[ProviderPlacement],
    ) -> Result<Self> {
        let graph = plan
            .legalized_graph
            .clone()
            .unwrap_or_else(|| Arc::new(graph.clone()));
        let provider_by_ep: HashMap<EpId, Arc<dyn ExecutionProvider>> = providers
            .iter()
            .map(|slot| (slot.ep, Arc::clone(&slot.provider)))
            .collect();

        // Validate every concrete kernel before building or running a partition.
        // This is also the first-slice alias gate: a kernel that may return a
        // zero-copy view cannot target independently owned boundary storage.
        for (&node_id, &ep) in &plan.node_placement {
            let node = graph.node(node_id);
            let provider = provider_by_ep.get(&ep).ok_or_else(|| {
                SessionError::Internal(format!(
                    "heterogeneous plan references missing EpId({})",
                    ep.0
                ))
            })?;
            let shapes = node
                .inputs
                .iter()
                .map(|input| {
                    input
                        .map(|value| {
                            as_static_shape(&graph.value(value).shape).ok_or_else(|| {
                                SessionError::HeterogeneousExecutionUnsupported {
                                    placement_summary: format!(
                                        "node {} ({}) has a symbolic input after static-shape \
                                         validation",
                                        node_id.0, node.op_type
                                    ),
                                }
                            })
                        })
                        .transpose()
                        .map(|shape| shape.unwrap_or_default())
                })
                .collect::<Result<Vec<_>>>()?;
            let opset = graph.effective_opset(node).unwrap_or(u64::MAX);
            if provider.executor_kernel_scope(node) == ExecutorKernelScope::Required {
                return Err(SessionError::HeterogeneousExecutionUnsupported {
                    placement_summary: format!(
                        "node {} ({}::{}@{opset}) requires one session-issued provider generation; \
                         the current heterogeneous preflight cannot borrow the partition executor's \
                         lifecycle before it is built. Keep this route-residency-required graph on \
                         one provider until heterogeneous lifecycle binding is implemented",
                        node_id.0,
                        if node.domain.is_empty() {
                            "ai.onnx"
                        } else {
                            &node.domain
                        },
                        node.op_type,
                    ),
                });
            }
            let kernel = provider.get_kernel(node, &shapes, opset).map_err(|error| {
                SessionError::unsupported_op(
                    node,
                    node_id,
                    opset,
                    provider.name(),
                    format!("capability was accepted but kernel creation failed: {error}"),
                )
            })?;
            if kernel.has_kernel_sized_outputs() {
                let domain = if node.domain.is_empty() {
                    "ai.onnx"
                } else {
                    node.domain.as_str()
                };
                return Err(SessionError::HeterogeneousExecutionUnsupported {
                    placement_summary: format!(
                        "node {} ({domain}::{}@{opset}) assigned to provider '{}' has \
                         kernel-sized outputs ({:?}); the opt-in heterogeneous executor's \
                         current static tensor-DAG slice preallocates declared output shapes and \
                         therefore cannot execute a kernel whose runtime cardinality may differ. \
                         Keep this graph on one provider until heterogeneous dynamic-output \
                         ownership is implemented",
                        node_id.0,
                        node.op_type,
                        provider.name(),
                        kernel.kernel_sized_output_policy(),
                    ),
                });
            }
            if kernel.may_produce_views() {
                let domain = if node.domain.is_empty() {
                    "ai.onnx"
                } else {
                    node.domain.as_str()
                };
                return Err(SessionError::HeterogeneousExecutionUnsupported {
                    placement_summary: format!(
                        "node {} ({domain}::{}@{opset}) may produce an aliased/view output; the \
                         first execution slice requires independently owned contiguous boundary \
                         buffers",
                        node_id.0, node.op_type
                    ),
                });
            }
        }

        let mut partition_executors = Vec::with_capacity(plan.partitions.len());
        for partition in &plan.partitions {
            let provider = provider_by_ep.get(&partition.ep).ok_or_else(|| {
                SessionError::Internal(format!(
                    "heterogeneous plan references missing EpId({})",
                    partition.ep.0
                ))
            })?;
            let extraction = extract_subgraph(&graph, partition);
            let executor =
                Executor::build(extraction.graph, Arc::clone(weights), Arc::clone(provider))?;
            partition_executors.push(PartitionExecutor {
                partition: partition.clone(),
                input_feeds: extraction.input_feeds,
                output_values: extraction.output_values,
                executor,
            });
        }

        Ok(Self {
            placement_report: placement_summary(&plan, &graph, providers),
            graph,
            weights: Arc::clone(weights),
            plan,
            providers: provider_by_ep,
            partitions: partition_executors,
            last_transfer_count: 0,
            last_release_counts: HashMap::new(),
        })
    }

    pub(crate) fn placement_report(&self) -> &str {
        &self.placement_report
    }

    pub(crate) fn primary_device(&self) -> DeviceId {
        self.providers
            .get(&EpId(0))
            .map_or_else(DeviceId::cpu, |provider| provider.device_id())
    }

    pub(crate) fn set_trace_context(&mut self, trace: TraceContext) {
        for partition in &mut self.partitions {
            partition.executor.set_trace_context(trace.clone());
        }
    }

    #[cfg(test)]
    pub(crate) fn last_transfer_count(&self) -> usize {
        self.last_transfer_count
    }

    #[cfg(test)]
    pub(crate) fn last_release_count(&self, value: ValueId, ep: EpId) -> usize {
        self.last_release_counts
            .get(&(value, ep))
            .copied()
            .unwrap_or(0)
    }

    fn release_value(
        &mut self,
        value: ValueId,
        residents: &mut HashMap<(ValueId, EpId), ResidentValue>,
    ) -> Result<()> {
        let keys = residents
            .keys()
            .filter(|(resident_value, _)| *resident_value == value)
            .copied()
            .collect::<Vec<_>>();
        for key in keys {
            if let Some(resident) = residents.remove(&key) {
                resident.release()?;
                *self.last_release_counts.entry(key).or_default() += 1;
            }
        }
        Ok(())
    }

    fn materialize_on(
        &self,
        value: ValueId,
        to_ep: EpId,
        residents: &mut HashMap<(ValueId, EpId), ResidentValue>,
    ) -> Result<bool> {
        if residents.contains_key(&(value, to_ep)) {
            return Ok(false);
        }
        let producer = self.graph.value(value).producer.ok_or_else(|| {
            SessionError::Internal(format!(
                "graph source value {} must be supplied as a graph input, not transferred",
                value.0
            ))
        })?;
        let from_ep = self.plan.node_placement[&producer];
        let transfer = self
            .plan
            .transfers
            .iter()
            .find(|transfer| {
                transfer.value == value && transfer.from_ep == from_ep && transfer.to_ep == to_ep
            })
            .ok_or_else(|| {
                SessionError::Internal(format!(
                    "missing planned transfer for value {} from EpId({}) to EpId({})",
                    value.0, from_ep.0, to_ep.0
                ))
            })?;
        let source = residents.get(&(value, from_ep)).ok_or_else(|| {
            SessionError::Internal(format!(
                "authoritative value {} on EpId({}) is not resident",
                value.0, from_ep.0
            ))
        })?;
        let owner = Arc::clone(self.providers.get(&to_ep).ok_or_else(|| {
            SessionError::Internal(format!("missing destination EpId({})", to_ep.0))
        })?);
        let mut destination =
            ResidentValue::allocate(to_ep, source.dtype, source.shape.clone(), owner)?;
        let bytes = source.buffer().len();
        if transfer.to.device_type == DeviceType::Cpu && transfer.from != transfer.to {
            let dst = destination.buffer_mut();
            // SAFETY: the destination EP allocated `dst`, its device is
            // host-accessible, and the mutable borrow is exclusive for `bytes`.
            let host =
                unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), bytes) };
            source.owner.copy_to_host(source.buffer(), host)?;
        } else {
            let destination_owner = Arc::clone(&destination.owner);
            let fence =
                destination_owner.copy_async(source.buffer(), destination.buffer_mut(), bytes)?;
            destination_owner.wait_fence(&fence)?;
        }
        residents.insert((value, to_ep), destination);
        Ok(true)
    }

    fn producerless_source<'input, 'weight>(
        &'weight self,
        value: ValueId,
        external_inputs: &'input HashMap<ValueId, &'input Tensor>,
    ) -> Result<ProducerlessSource<'input, 'weight>> {
        if let Some(weight) = self.graph.initializers.get(&value) {
            return Ok(ProducerlessSource::Initializer(weight));
        }
        if self.graph.inputs.contains(&value) {
            return external_inputs
                .get(&value)
                .copied()
                .map(ProducerlessSource::ExternalInput)
                .ok_or_else(|| {
                    SessionError::Internal(format!(
                        "heterogeneous graph input value {} ('{}') is unavailable",
                        value.0,
                        self.graph
                            .value(value)
                            .name
                            .as_deref()
                            .unwrap_or("<unnamed>")
                    ))
                });
        }
        Err(SessionError::Internal(format!(
            "producer-less heterogeneous value {} ('{}') is neither a graph input nor an \
             initializer; optional absent inputs must be represented by an empty node-input slot",
            value.0,
            self.graph
                .value(value)
                .name
                .as_deref()
                .unwrap_or("<unnamed>")
        )))
    }

    fn initializer_output(&self, value: ValueId, weight: &WeightRef) -> Result<Tensor> {
        let metadata = self.graph.value(value);
        let shape = as_static_shape(&metadata.shape).expect("validated static shape");
        if weight.dtype() != metadata.dtype || weight.dims() != shape {
            return Err(SessionError::Internal(format!(
                "initializer graph output value {} ('{}') metadata {:?} {:?} does not match its \
                 canonical weight {:?} {:?}",
                value.0,
                metadata.name.as_deref().unwrap_or("<unnamed>"),
                metadata.dtype,
                shape,
                weight.dtype(),
                weight.dims(),
            )));
        }
        let bytes = self.weights.bytes(weight).ok_or_else(|| {
            SessionError::Internal(format!(
                "initializer graph output value {} ('{}') could not resolve its canonical weight \
                 bytes; keep the WeightStore and any external-data mapping alive for the session",
                value.0,
                metadata.name.as_deref().unwrap_or("<unnamed>")
            ))
        })?;
        // Ordinary `run` returns owned host tensors even when a producing
        // partition ran on a device (`copy_from_device_buffer` above). Use the
        // same governed CPU output policy here; do not invent a hetero resident
        // for immutable initializer storage.
        Tensor::from_raw(metadata.dtype, shape, bytes)
    }

    pub(crate) fn run(&mut self, inputs: &[(&str, &Tensor)]) -> Result<Vec<Tensor>> {
        let input_by_value: HashMap<ValueId, &Tensor> = self
            .graph
            .inputs
            .iter()
            .filter_map(|&value| {
                let name = self.graph.value(value).name.as_deref()?;
                inputs
                    .iter()
                    .find(|(input_name, _)| *input_name == name)
                    .map(|(_, tensor)| (value, *tensor))
            })
            .collect();
        for &value in &self.graph.inputs {
            if !self.graph.initializers.contains_key(&value) && !input_by_value.contains_key(&value)
            {
                return Err(SessionError::Internal(format!(
                    "missing required heterogeneous graph input '{}'",
                    self.graph
                        .value(value)
                        .name
                        .as_deref()
                        .unwrap_or("<unnamed>")
                )));
            }
        }

        let graph_outputs: HashSet<ValueId> = self.graph.outputs.iter().copied().collect();
        let mut remaining_consumers: HashMap<ValueId, usize> = HashMap::new();
        for partition in &self.plan.partitions {
            for &value in &partition.inputs {
                if self.graph.value(value).producer.is_some() {
                    *remaining_consumers.entry(value).or_default() += 1;
                }
            }
        }

        let mut residents: HashMap<(ValueId, EpId), ResidentValue> = HashMap::new();
        self.last_transfer_count = 0;
        self.last_release_counts.clear();
        for partition_index in 0..self.partitions.len() {
            let ep = self.partitions[partition_index].partition.ep;
            let input_feeds = self.partitions[partition_index].input_feeds.clone();
            let output_values = self.partitions[partition_index].output_values.clone();

            for &(value, _) in &input_feeds {
                if self.graph.value(value).producer.is_some()
                    && self.materialize_on(value, ep, &mut residents)?
                {
                    self.last_transfer_count += 1;
                }
            }

            let mut normal_inputs = Vec::new();
            for (value, name) in &input_feeds {
                if self.graph.value(*value).producer.is_none() {
                    match self.producerless_source(*value, &input_by_value)? {
                        ProducerlessSource::ExternalInput(tensor) => {
                            normal_inputs.push((name.as_str(), tensor));
                        }
                        ProducerlessSource::Initializer(_) => {
                            return Err(SessionError::Internal(format!(
                                "initializer value {} was extracted as a runtime partition input \
                                 instead of being attached to the partition graph",
                                value.0
                            )));
                        }
                    }
                }
            }

            let mut output_residents = Vec::with_capacity(output_values.len());
            for &value in &output_values {
                let metadata = self.graph.value(value);
                let shape = as_static_shape(&metadata.shape).expect("validated static shape");
                let owner = Arc::clone(self.providers.get(&ep).ok_or_else(|| {
                    SessionError::Internal(format!("missing partition EpId({})", ep.0))
                })?);
                output_residents.push((
                    value,
                    ResidentValue::allocate(ep, metadata.dtype, shape, owner)?,
                ));
            }

            let child = &mut self.partitions[partition_index].executor;
            let mut bindings: Vec<DeviceIoBinding> =
                Vec::with_capacity(input_feeds.len() + output_residents.len());
            for (value, name) in &input_feeds {
                if self.graph.value(*value).producer.is_none() {
                    continue;
                }
                let resident = residents.get(&(*value, ep)).ok_or_else(|| {
                    SessionError::Internal(format!(
                        "value {} was not materialized on EpId({})",
                        value.0, ep.0
                    ))
                })?;
                let spec = ExternalMemorySpec::input(
                    name,
                    None::<String>,
                    resident.dtype,
                    resident.shape.clone(),
                    resident.shape.clone(),
                    resident.buffer().as_ptr().cast_mut(),
                    resident.buffer().len(),
                );
                // SAFETY: `resident` owns the allocation through the complete
                // child run; the borrowed binding is dropped before liveness can
                // release that resident, and no other writer exists.
                bindings.push(unsafe { child.device_binding_from_external_memory(spec)? });
            }
            for (value, resident) in &output_residents {
                let spec = ExternalMemorySpec::output(
                    subgraph_value_name(*value),
                    resident.dtype,
                    resident.shape.clone(),
                    resident.shape.clone(),
                    resident.buffer().as_ptr().cast_mut(),
                    resident.buffer().len(),
                );
                // SAFETY: each output resident is a distinct governed allocation
                // owned for the complete child run.
                let mut binding = unsafe { child.device_binding_from_external_memory(spec)? };
                // The heterogeneous coordinator, not this child partition,
                // owns publication of the resident into the cross-partition
                // value table after the child succeeds.
                binding.disable_output_publication_transaction();
                bindings.push(binding);
            }

            let returned = child.run_with_device_bindings(&normal_inputs, &mut bindings)?;
            if returned.iter().any(Option::is_some) {
                return Err(SessionError::Internal(
                    "heterogeneous partition returned an unbound tensor output".into(),
                ));
            }
            drop(bindings);
            for (value, resident) in output_residents {
                residents.insert((value, ep), resident);
            }

            let consumed_values = self.partitions[partition_index].partition.inputs.clone();
            for value in consumed_values {
                let Some(remaining) = remaining_consumers.get_mut(&value) else {
                    continue;
                };
                *remaining -= 1;
                if *remaining == 0 && !graph_outputs.contains(&value) {
                    self.release_value(value, &mut residents)?;
                }
            }
        }

        if self.last_transfer_count != self.plan.transfers.len() {
            return Err(SessionError::Internal(format!(
                "heterogeneous execution performed {} transfer(s), but the immutable plan has {}",
                self.last_transfer_count,
                self.plan.transfers.len()
            )));
        }

        let mut outputs = Vec::with_capacity(self.graph.outputs.len());
        for &value in &self.graph.outputs {
            if self.graph.value(value).producer.is_none() {
                let output = match self.producerless_source(value, &input_by_value)? {
                    ProducerlessSource::ExternalInput(tensor) => tensor.clone(),
                    ProducerlessSource::Initializer(weight) => {
                        self.initializer_output(value, weight)?
                    }
                };
                outputs.push(output);
                continue;
            }
            let producer = self.graph.value(value).producer.expect("checked above");
            let ep = self.plan.node_placement[&producer];
            let resident = residents.get(&(value, ep)).ok_or_else(|| {
                SessionError::Internal(format!(
                    "graph output value {} was never produced on EpId({})",
                    value.0, ep.0
                ))
            })?;
            outputs.push(Tensor::copy_from_device_buffer(
                &resident.owner,
                resident.buffer(),
                resident.dtype,
                resident.shape.clone(),
            )?);
        }
        let live_values = residents
            .keys()
            .map(|(value, _)| *value)
            .collect::<HashSet<_>>();
        for value in live_values {
            self.release_value(value, &mut residents)?;
        }
        Ok(outputs)
    }
}

/// Build and execute one heterogeneous plan. Session integration stores the
/// persistent [`HeterogeneousExecutor`]; this wrapper remains for focused tests.
pub fn execute(
    plan: &HeterogeneousPlan,
    graph: &Graph,
    weights: &Arc<WeightStore>,
    providers: &[ProviderPlacement],
    inputs: &[(&str, &Tensor)],
) -> Result<Vec<Tensor>> {
    HeterogeneousExecutor::build(plan.clone(), graph, weights, providers)?.run(inputs)
}

#[cfg(test)]
mod tests;
