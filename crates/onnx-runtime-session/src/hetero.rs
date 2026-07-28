//! Heterogeneous multi-provider graph partitioning and execution (#65).
//!
//! Sessions have historically bound one [`ExecutionProvider`] to the whole
//! executor, so a graph containing a single node the selected provider cannot
//! run is rejected outright (`docs/HETEROGENEOUS_PLACEMENT.md`, issue #65). This
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
//! existing [`Executor`], staging boundary tensors through host memory
//! (`docs/HETEROGENEOUS_PLACEMENT.md` §5.2, the correctness-first synchronous
//! transfer phase). Because every cross-partition value is materialized on the
//! host between partitions, this is a faithful, if unoptimized, realization of
//! the transfer edges the planner computes.
//!
//! ## Deferred scope
//!
//! This slice deliberately implements the correctness-first phase only. Value
//! residency (keeping a tensor on device across partition boundaries),
//! asynchronous copies/fences, shape-keyed placement (`M=1` decode on CUDA vs
//! `M>1` prefill on CPU), partition-level CUDA-graph capture, and multi-GPU peer
//! copies are all left to later phases (see the design doc §5 and §9).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use onnx_runtime_ep_api::abi::OrtGraphView;
use onnx_runtime_ep_api::{
    DeviceBuffer, DeviceId, DeviceType, EpConfig, EpId, ExecutionProvider, Fence, Kernel,
    KernelMatch, Result as EpResult, SubgraphClaim,
};
use onnx_runtime_ir::{
    DataType, Graph, GraphViewCache, Node, NodeId, Shape, TensorLayout, ValueId,
};
use onnx_runtime_loader::WeightStore;

use crate::error::{Result, SessionError};
use crate::executor::Executor;
use crate::tensor::Tensor;

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

/// A single cross-device materialization the executor must perform before the
/// destination partition runs. Deduplicated by `(value, to)`: a value that fans
/// out to several partitions on the same destination device is one transfer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transfer {
    /// The boundary value being moved.
    pub value: ValueId,
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
    /// Minimal, deduplicated cross-device transfers at partition boundaries.
    pub transfers: Vec<Transfer>,
    /// Per-node provider assignment (stable across runs).
    pub node_placement: HashMap<NodeId, EpId>,
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
        let chosen = providers.iter().find(|slot| {
            slot.provider
                .supports_op(node, opset, &shapes, &dtypes, &layouts)
                .is_supported()
        });
        match chosen {
            Some(slot) => {
                placement.insert(node_id, slot.ep);
            }
            None => {
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
    }
    Ok(placement)
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

/// Compute the minimal cross-device transfers at partition boundaries.
fn plan_transfers(
    partitions: &[Partition],
    value_to_partition: &HashMap<ValueId, usize>,
) -> Vec<Transfer> {
    // Dedup by (value, to-device); a value fanning out to several partitions on
    // the same device is materialized once.
    let mut seen: HashSet<(u32, DeviceId)> = HashSet::new();
    let mut transfers = Vec::new();
    for part in partitions {
        for &input in &part.inputs {
            let from = value_to_partition
                .get(&input)
                .map(|&i| partitions[i].device)
                // Graph inputs and initializers originate on the host.
                .unwrap_or_else(DeviceId::cpu);
            let to = part.device;
            if from != to && seen.insert((input.0, to)) {
                transfers.push(Transfer {
                    value: input,
                    from,
                    to,
                });
            }
        }
    }
    transfers.sort_by_key(|t| {
        (
            t.value.0,
            t.to.device_type.trace_name().into_owned(),
            t.to.index,
        )
    });
    transfers
}

/// Plan a heterogeneous execution of `graph` over `providers` (priority order).
pub fn plan(graph: &Graph, providers: &[ProviderPlacement]) -> Result<HeterogeneousPlan> {
    if providers.is_empty() {
        return Err(SessionError::Internal(
            "heterogeneous placement requires at least one provider".into(),
        ));
    }
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
    })
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

/// Execute `graph` heterogeneously per `plan`, staging boundary tensors through
/// host memory, and return the graph outputs in declared order.
///
/// `providers` must contain an entry for every [`EpId`] referenced by the plan.
/// The result is byte-identical to executing the whole graph on a single
/// provider (the correctness invariant this module guarantees).
pub fn execute(
    plan: &HeterogeneousPlan,
    graph: &Graph,
    weights: &Arc<WeightStore>,
    providers: &[ProviderPlacement],
    inputs: &[(&str, &Tensor)],
) -> Result<Vec<Tensor>> {
    let provider_by_ep: HashMap<EpId, Arc<dyn ExecutionProvider>> = providers
        .iter()
        .map(|slot| (slot.ep, Arc::clone(&slot.provider)))
        .collect();

    // Host value map keyed by parent ValueId. Seed with the caller's inputs.
    let mut values: HashMap<ValueId, Tensor> = HashMap::new();
    let name_to_value: HashMap<&str, ValueId> = graph
        .inputs
        .iter()
        .filter_map(|&v| graph.value(v).name.as_deref().map(|name| (name, v)))
        .collect();
    for (name, tensor) in inputs {
        if let Some(&vid) = name_to_value.get(name) {
            values.insert(vid, (*tensor).clone());
        }
    }

    for partition in &plan.partitions {
        let provider = provider_by_ep.get(&partition.ep).ok_or_else(|| {
            SessionError::Internal(format!(
                "no execution provider registered for EpId({})",
                partition.ep.0
            ))
        })?;
        let extraction = extract_subgraph(graph, partition);

        // Assemble this partition's runtime inputs from the host value map.
        let feed_tensors: Vec<(String, Tensor)> = extraction
            .input_feeds
            .iter()
            .map(|(parent_value, name)| {
                let tensor = values.get(parent_value).cloned().ok_or_else(|| {
                    SessionError::Internal(format!(
                        "boundary input value {} missing before its partition ran",
                        parent_value.0
                    ))
                })?;
                Ok((name.clone(), tensor))
            })
            .collect::<Result<Vec<_>>>()?;
        let feed_refs: Vec<(&str, &Tensor)> = feed_tensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();

        let mut executor =
            Executor::build(extraction.graph, Arc::clone(weights), Arc::clone(provider))?;
        let outputs = executor.run(&feed_refs)?;

        // Record produced boundary outputs back into the host value map.
        for (parent_value, tensor) in extraction.output_values.iter().zip(outputs) {
            values.insert(*parent_value, tensor);
        }
    }

    // Collect the graph outputs in declared order.
    graph
        .outputs
        .iter()
        .map(|&v| {
            values.get(&v).cloned().ok_or_else(|| {
                SessionError::Internal(format!(
                    "graph output value {} was never produced by any partition",
                    v.0
                ))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests;
