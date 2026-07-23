//! The mutable graph model: node/value arenas, edge-consistent mutation,
//! topological ordering, and validation (see `docs/ORT2.md` §3.3 and §3.5).

use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::arena::Arena;
use crate::dtype::DataType;
use crate::error::GraphError;
use crate::node::{Node, NodeId};
use crate::shape::{Shape, SymbolConstraints, SymbolId};
use crate::tensor::WeightRef;
use crate::value::{Value, ValueId};

/// A computation graph in SSA form.
///
/// Nodes and values live in [`Arena`]s keyed by [`NodeId`] / [`ValueId`]. The
/// mutation API keeps producer/consumer edges consistent, so optimization
/// passes can rewrite the graph and then [`Graph::validate`] it.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub nodes: Arena<NodeId, Node>,
    pub values: Arena<ValueId, Value>,
    /// Graph inputs, in order. These have no producer.
    pub inputs: Vec<ValueId>,
    /// Graph outputs, in order.
    pub outputs: Vec<ValueId>,
    /// Constant initializer weights, keyed by the value they populate.
    pub initializers: HashMap<ValueId, WeightRef>,
    /// Constraints on symbolic dimensions.
    pub symbol_constraints: HashMap<SymbolId, SymbolConstraints>,
    /// Imported opsets: domain → version.
    pub opset_imports: HashMap<String, u64>,
    /// Subgraph bodies for control-flow ops, keyed by `(node, attr_name)`.
    pub subgraphs: HashMap<(NodeId, String), Graph>,

    next_symbol: u32,
    symbol_names: HashMap<String, SymbolId>,
    unknown_value_types: HashSet<ValueId>,
    unknown_value_shapes: HashSet<ValueId>,
}

impl Graph {
    /// An empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    // === Query API ===

    /// Borrow a node. Panics if `id` is not live; use
    /// [`Graph::try_node`] for a checked lookup.
    pub fn node(&self, id: NodeId) -> &Node {
        self.nodes.get(id).expect("node id not live in graph")
    }

    /// Mutably borrow a node. Panics if `id` is not live.
    pub fn node_mut(&mut self, id: NodeId) -> &mut Node {
        self.nodes.get_mut(id).expect("node id not live in graph")
    }

    /// Checked node lookup.
    pub fn try_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id)
    }

    /// Borrow a value. Panics if `id` is not live; use
    /// [`Graph::try_value`] for a checked lookup.
    pub fn value(&self, id: ValueId) -> &Value {
        self.values.get(id).expect("value id not live in graph")
    }

    /// Mutably borrow a value. Panics if `id` is not live.
    pub fn value_mut(&mut self, id: ValueId) -> &mut Value {
        self.values.get_mut(id).expect("value id not live in graph")
    }

    /// Checked value lookup.
    pub fn try_value(&self, id: ValueId) -> Option<&Value> {
        self.values.get(id)
    }

    /// Consuming input slots sorted by `(NodeId, input_index)`.
    pub fn uses(&self, value: ValueId) -> Vec<(NodeId, u32)> {
        self.value(value).consumers.uses()
    }

    /// Distinct consumer nodes sorted by ascending [`NodeId`].
    pub fn consumers(&self, value: ValueId) -> Vec<NodeId> {
        self.value(value).consumers.nodes()
    }

    /// Number of consuming input slots.
    pub fn num_uses(&self, value: ValueId) -> usize {
        self.value(value).consumers.len()
    }

    /// Whether at least one node input slot consumes `value`.
    pub fn has_uses(&self, value: ValueId) -> bool {
        !self.value(value).consumers.is_empty()
    }

    /// Number of live nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of live values.
    pub fn num_values(&self) -> usize {
        self.values.len()
    }

    /// Whether a value's element type came from explicit source type information.
    pub fn value_type_is_known(&self, id: ValueId) -> bool {
        !self.unknown_value_types.contains(&id)
    }

    /// Whether a value's rank and dimensions came from explicit source shape information.
    pub fn value_shape_is_known(&self, id: ValueId) -> bool {
        !self.unknown_value_shapes.contains(&id)
    }

    /// Mark a value's placeholder element type as unknown.
    pub fn mark_value_type_unknown(&mut self, id: ValueId) {
        self.unknown_value_types.insert(id);
    }

    /// Mark a value's placeholder shape as unknown.
    pub fn mark_value_shape_unknown(&mut self, id: ValueId) {
        self.unknown_value_shapes.insert(id);
    }

    // === Symbolic dimensions ===

    /// Allocate a fresh symbolic dimension with an optional name (no dedup).
    pub fn create_symbol(&mut self, name: Option<String>) -> SymbolId {
        let id = SymbolId(self.next_symbol);
        self.next_symbol += 1;
        self.symbol_constraints
            .insert(id, SymbolConstraints::new(id, name.clone()));
        if let Some(n) = name {
            self.symbol_names.insert(n, id);
        }
        id
    }

    /// Intern a symbolic dimension by protobuf dim-param name: repeated names
    /// resolve to the same [`SymbolId`] (graph-construction invariant §3.5.4).
    pub fn intern_symbol(&mut self, name: &str) -> SymbolId {
        if let Some(id) = self.symbol_names.get(name) {
            return *id;
        }
        self.create_symbol(Some(name.to_string()))
    }

    // === Construction helpers ===

    /// Create a new anonymous value with a contiguous default layout.
    pub fn create_value(&mut self, dtype: DataType, shape: Shape) -> ValueId {
        self.values
            .insert_with(|vid| Value::new(vid, dtype, shape.clone()))
    }

    /// Create a new named value.
    pub fn create_named_value(
        &mut self,
        name: impl Into<String>,
        dtype: DataType,
        shape: Shape,
    ) -> ValueId {
        let id = self.create_value(dtype, shape);
        self.value_mut(id).name = Some(name.into());
        id
    }

    /// Register `value` as a graph input.
    pub fn add_input(&mut self, value: ValueId) {
        self.value_mut(value).is_graph_input = true;
        self.inputs.push(value);
        debug_assert!(self.value(value).is_graph_input);
    }

    /// Register `value` as a graph output.
    pub fn add_output(&mut self, value: ValueId) {
        self.value_mut(value).is_graph_output = true;
        self.outputs.push(value);
        debug_assert!(self.value(value).is_graph_output);
    }

    /// Insert `value` into the ordered graph outputs.
    pub fn insert_output(&mut self, index: usize, value: ValueId) {
        self.value_mut(value).is_graph_output = true;
        self.outputs.insert(index, value);
        debug_assert!(self.value(value).is_graph_output);
    }

    /// Remove one ordered graph input.
    pub fn remove_input(&mut self, index: usize) -> ValueId {
        let value = self.inputs.remove(index);
        if !self.inputs.contains(&value) {
            self.value_mut(value).is_graph_input = false;
        }
        debug_assert_eq!(
            self.value(value).is_graph_input,
            self.inputs.contains(&value)
        );
        value
    }

    /// Remove one ordered graph output.
    pub fn remove_output(&mut self, index: usize) -> ValueId {
        let value = self.outputs.remove(index);
        if !self.outputs.contains(&value) {
            self.value_mut(value).is_graph_output = false;
        }
        debug_assert_eq!(
            self.value(value).is_graph_output,
            self.outputs.contains(&value)
        );
        value
    }

    /// Replace the complete ordered graph-input list.
    pub fn set_inputs(&mut self, inputs: Vec<ValueId>) {
        for value in self.inputs.drain(..) {
            if let Some(metadata) = self.values.get_mut(value) {
                metadata.is_graph_input = false;
            }
        }
        for &value in &inputs {
            self.value_mut(value).is_graph_input = true;
        }
        self.inputs = inputs;
        debug_assert!(
            self.inputs
                .iter()
                .all(|&value| self.value(value).is_graph_input)
        );
    }

    /// Replace the complete ordered graph-output list.
    pub fn set_outputs(&mut self, outputs: Vec<ValueId>) {
        for value in self.outputs.drain(..) {
            if let Some(metadata) = self.values.get_mut(value) {
                metadata.is_graph_output = false;
            }
        }
        for &value in &outputs {
            self.value_mut(value).is_graph_output = true;
        }
        self.outputs = outputs;
        debug_assert!(
            self.outputs
                .iter()
                .all(|&value| self.value(value).is_graph_output)
        );
    }

    /// Attach initializer weights to `value`.
    pub fn set_initializer(&mut self, value: ValueId, weight: WeightRef) {
        self.initializers.insert(value, weight);
    }

    // === Traversal ===

    /// Direct predecessors: nodes that produce this node's inputs.
    pub fn predecessors(&self, node: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for v in self.node(node).input_values() {
            if let Some(val) = self.values.get(v)
                && let Some(prod) = val.producer
                && seen.insert(prod)
            {
                out.push(prod);
            }
        }
        out.sort_unstable_by_key(|node| node.0);
        out
    }

    /// Direct successors: nodes that consume this node's outputs.
    pub fn successors(&self, node: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for &v in &self.node(node).outputs {
            if let Some(val) = self.values.get(v) {
                for c in val.consumers.nodes() {
                    if seen.insert(c) {
                        out.push(c);
                    }
                }
            }
        }
        out.sort_unstable_by_key(|node| node.0);
        out
    }

    /// All nodes that lie on a path between `inputs` and `outputs`.
    ///
    /// Walks backwards from `outputs` via producer edges, stopping at any value
    /// in `inputs`. Used to extract subgraphs for EP capability claims (§3.4).
    pub fn nodes_between(&self, inputs: &[ValueId], outputs: &[ValueId]) -> Vec<NodeId> {
        let boundary: HashSet<ValueId> = inputs.iter().copied().collect();
        let mut nodes = Vec::new();
        let mut seen_nodes = HashSet::new();
        let mut seen_values = HashSet::new();
        let mut stack: Vec<ValueId> = outputs.to_vec();
        while let Some(v) = stack.pop() {
            if boundary.contains(&v) || !seen_values.insert(v) {
                continue;
            }
            let Some(val) = self.values.get(v) else {
                continue;
            };
            if let Some(prod) = val.producer {
                if seen_nodes.insert(prod) {
                    nodes.push(prod);
                }
                for iv in self.node(prod).input_values() {
                    stack.push(iv);
                }
            }
        }
        nodes
    }

    /// Topological order of nodes via Kahn's algorithm.
    ///
    /// Ties are broken by ascending [`NodeId`] for deterministic output.
    /// Returns [`GraphError::CycleDetected`] if the graph has a cycle.
    pub fn topological_order(&self) -> Result<Vec<NodeId>, GraphError> {
        const VACANT: usize = usize::MAX;
        let mut in_degree = vec![VACANT; self.nodes.capacity()];
        let mut adj = vec![Vec::<NodeId>::new(); self.nodes.capacity()];
        for node in self.nodes.keys() {
            in_degree[node.0 as usize] = 0;
        }

        for (nid, node) in self.nodes.iter() {
            for v in node.input_values() {
                if let Some(val) = self.values.get(v)
                    && let Some(prod) = val.producer
                    && self.nodes.contains(prod)
                {
                    adj[prod.0 as usize].push(nid);
                    in_degree[nid.0 as usize] += 1;
                }
            }
        }

        // Min-heap on raw id for deterministic ordering.
        let mut ready: BinaryHeap<std::cmp::Reverse<u32>> = in_degree
            .iter()
            .enumerate()
            .filter(|(_, degree)| **degree == 0)
            .map(|(raw, _)| std::cmp::Reverse(raw as u32))
            .collect();

        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(std::cmp::Reverse(raw)) = ready.pop() {
            let nid = NodeId(raw);
            order.push(nid);
            for &successor in &adj[raw as usize] {
                let degree = &mut in_degree[successor.0 as usize];
                *degree -= 1;
                if *degree == 0 {
                    ready.push(std::cmp::Reverse(successor.0));
                }
            }
        }

        if order.len() != self.nodes.len() {
            return Err(GraphError::CycleDetected);
        }
        Ok(order)
    }

    // === Mutation API ===

    /// Canonicalize the default ONNX operator domain to `""` throughout this
    /// graph (nodes, opset-import keys) and recursively in every subgraph.
    ///
    /// After this pass the graph satisfies the post-load invariant: the default
    /// domain is always the empty string; `"ai.onnx"` never appears. The loader
    /// establishes this at proto-materialization time; this method lets
    /// programmatic graph builders reach the same canonical form before session
    /// construction. See [`crate::normalize_domain`].
    pub fn normalize_domains(&mut self) {
        for node in self.nodes.values_mut() {
            if node.domain == crate::AI_ONNX_DOMAIN {
                node.domain.clear();
            }
        }
        if let Some(version) = self.opset_imports.remove(crate::AI_ONNX_DOMAIN) {
            let entry = self.opset_imports.entry(String::new()).or_insert(version);
            *entry = (*entry).max(version);
        }
        for subgraph in self.subgraphs.values_mut() {
            subgraph.normalize_domains();
        }
    }

    /// Insert a node, wiring its producer/consumer edges. The node's `id`
    /// field is overwritten with the freshly allocated [`NodeId`].
    pub fn insert_node(&mut self, node: Node) -> NodeId {
        let id = self.nodes.insert_with(|nid| {
            let mut node = node;
            node.id = nid;
            node
        });
        self.connect_edges(id);
        id
    }

    /// Remove a node, disconnecting its edges. Output values left with no
    /// consumers (and not graph I/O or initializers) are deleted.
    pub fn remove_node(&mut self, id: NodeId) {
        if !self.nodes.contains(id) {
            return;
        }
        self.disconnect_edges(id);
        let outputs = self.node(id).outputs.clone();
        self.nodes.remove(id);
        for v in outputs {
            self.gc_value_if_orphan(v);
        }
    }

    /// Remove nodes in slice order.
    ///
    /// Each input edge is removed directly by `(NodeId, input_index)`, so this
    /// remains linear in the number of removed edges even for a high-fanout
    /// shared value.
    pub fn remove_nodes(&mut self, ids: &[NodeId]) {
        for &id in ids {
            self.remove_node(id);
        }
    }

    /// Replace disjoint node groups with one node each while updating shared
    /// producer/consumer metadata in a batch.
    ///
    /// Each group is semantically equivalent to calling [`Graph::remove_node`]
    /// for its IDs in slice order and then [`Graph::insert_node`] for the
    /// replacement. In particular, replacement IDs and orphan-value collection
    /// match that sequential operation. `graph_outputs` is retained for API
    /// compatibility and checked against the per-value membership invariant in
    /// debug builds.
    pub fn replace_node_groups(
        &mut self,
        groups: Vec<(Vec<NodeId>, Node)>,
        graph_outputs: &HashSet<ValueId>,
    ) -> Vec<NodeId> {
        debug_assert_eq!(
            graph_outputs,
            &self.outputs.iter().copied().collect::<HashSet<_>>()
        );
        let mut removed_nodes = HashSet::new();
        for (node_ids, _) in &groups {
            assert!(
                !node_ids.is_empty(),
                "replace_node_groups: group must not be empty"
            );
            for &id in node_ids {
                assert!(
                    self.nodes.contains(id),
                    "replace_node_groups: node id not live"
                );
                assert!(
                    removed_nodes.insert(id),
                    "replace_node_groups: groups must be disjoint"
                );
            }
        }

        let mut inserted = Vec::with_capacity(groups.len());
        for (node_ids, replacement) in groups {
            for id in node_ids {
                self.remove_node(id);
            }
            inserted.push(self.insert_node(replacement));
        }

        inserted
    }

    /// Replace node `old` in place with `new`, preserving the [`NodeId`].
    ///
    /// The old node's edges are disconnected and the new node's edges are
    /// connected. Values that were outputs of `old` but not of `new` are left
    /// in place (producer cleared); the caller may prune them.
    pub fn replace_node(&mut self, old: NodeId, new: Node) -> NodeId {
        assert!(self.nodes.contains(old), "replace_node: old id not live");
        self.disconnect_edges(old);
        {
            let slot = self.nodes.get_mut(old).expect("old live");
            let mut new = new;
            new.id = old;
            *slot = new;
        }
        self.connect_edges(old);
        old
    }

    /// Splice `new_node` onto the edge feeding out of `value`:
    /// `producer(value) → [new_node] → consumers(value)`.
    ///
    /// `new_node`'s single input becomes `value`, and it produces a fresh value
    /// that replaces `value` in all of `value`'s original consumers.
    pub fn insert_on_edge(&mut self, value: ValueId, new_node: Node) -> NodeId {
        let (dtype, shape) = {
            let v = self.value(value);
            (v.dtype, v.shape.clone())
        };
        let new_value = self.create_value(dtype, shape);
        // Redirect existing consumers onto the new value first (before the new
        // node itself becomes a consumer of `value`).
        self.replace_all_uses(value, new_value);

        let mut new_node = new_node;
        new_node.inputs = vec![Some(value)];
        new_node.outputs = vec![new_value];
        self.insert_node(new_node)
    }

    /// Replace one node input and update both values' consumer sets.
    ///
    /// This is constant-time on average for edge metadata. `None` disconnects
    /// the slot and is used by node removal.
    pub fn replace_input(
        &mut self,
        node: NodeId,
        input_index: usize,
        new_value: Option<ValueId>,
    ) -> Option<ValueId> {
        assert!(self.nodes.contains(node), "replace_input: node id not live");
        assert!(
            input_index < self.node(node).inputs.len(),
            "replace_input: input index out of bounds"
        );
        if let Some(value) = new_value {
            assert!(
                self.values.contains(value),
                "replace_input: value id not live"
            );
        }

        let old_value = self.node(node).inputs[input_index];
        if old_value == new_value {
            return old_value;
        }
        if let Some(value) = old_value {
            let removed = self
                .value_mut(value)
                .consumers
                .remove(node, input_index as u32);
            debug_assert!(removed, "old input edge must be present");
        }
        self.node_mut(node).inputs[input_index] = new_value;
        if let Some(value) = new_value {
            self.value_mut(value)
                .consumers
                .insert(node, input_index as u32);
        }
        old_value
    }

    /// Replace every use of `old_value` with `new_value` in consumer nodes and
    /// in the graph output list, moving consumer edges accordingly.
    pub fn replace_all_uses(&mut self, old_value: ValueId, new_value: ValueId) {
        if old_value == new_value {
            return;
        }
        let uses = match self.values.get(old_value) {
            Some(value) => value.consumers.uses(),
            None => return,
        };
        for (node, input_index) in uses {
            if self.nodes.contains(node) {
                self.replace_input(node, input_index as usize, Some(new_value));
            }
        }
        if self.value(old_value).is_graph_output {
            let mut outputs = self.outputs.clone();
            for output in &mut outputs {
                if *output == old_value {
                    *output = new_value;
                }
            }
            self.set_outputs(outputs);
        }
    }

    // === Validation ===

    /// Verify structural invariants (§3.3). Returns every defect found.
    pub fn validate(&self) -> Result<(), Vec<GraphError>> {
        let mut errors = Vec::new();
        let graph_inputs: HashSet<_> = self.inputs.iter().copied().collect();
        let graph_outputs: HashSet<_> = self.outputs.iter().copied().collect();
        for (value, metadata) in self.values.iter() {
            debug_assert_eq!(
                metadata.is_graph_input,
                graph_inputs.contains(&value),
                "graph-input membership flag drifted for {value:?}"
            );
            debug_assert_eq!(
                metadata.is_graph_output,
                graph_outputs.contains(&value),
                "graph-output membership flag drifted for {value:?}"
            );
        }

        // 1. Node edges reference live values; collect produced values.
        let mut produced: HashMap<ValueId, NodeId> = HashMap::new();
        for (nid, node) in self.nodes.iter() {
            for (input_index, input) in node.inputs.iter().enumerate() {
                if let Some(value) = input {
                    if !self.values.contains(*value) {
                        errors.push(GraphError::DanglingValue(*value));
                    } else if !self
                        .value(*value)
                        .consumers
                        .contains(nid, input_index as u32)
                    {
                        errors.push(GraphError::ConsumerLinkMismatch(*value));
                    }
                }
            }
            for &v in &node.outputs {
                if !self.values.contains(v) {
                    errors.push(GraphError::DanglingValue(v));
                    continue;
                }
                if produced.insert(v, nid).is_some() {
                    errors.push(GraphError::DuplicateOutput(v));
                }
            }
        }

        // 2/3. Producer/consumer link consistency.
        for (vid, val) in self.values.iter() {
            if let Some(p) = val.producer {
                if !self.nodes.contains(p) {
                    errors.push(GraphError::DanglingNode(p));
                } else if !self.node(p).outputs.contains(&vid) {
                    errors.push(GraphError::ProducerLinkMismatch(vid));
                }
            }
            for (consumer, input_index) in val.consumers.uses() {
                if !self.nodes.contains(consumer) {
                    errors.push(GraphError::DanglingNode(consumer));
                } else if self.node(consumer).inputs.get(input_index as usize) != Some(&Some(vid)) {
                    errors.push(GraphError::ConsumerLinkMismatch(vid));
                }
            }
        }

        // 4. Graph inputs must be sources.
        for &inp in &self.inputs {
            if let Some(val) = self.values.get(inp) {
                if val.producer.is_some() {
                    errors.push(GraphError::InputHasProducer(inp));
                }
                debug_assert!(val.is_graph_input);
            } else {
                errors.push(GraphError::DanglingValue(inp));
            }
        }

        // 5. Graph outputs must be produced (unless they are graph inputs or
        //    initializers passed straight through).
        for &out in &self.outputs {
            match self.values.get(out) {
                Some(val) => {
                    debug_assert!(val.is_graph_output);
                    let is_source = val.is_graph_input || self.initializers.contains_key(&out);
                    if val.producer.is_none() && !is_source {
                        errors.push(GraphError::MissingProducer(out));
                    }
                }
                None => errors.push(GraphError::DanglingValue(out)),
            }
        }

        // 6. No cycles.
        if let Err(e) = self.topological_order() {
            errors.push(e);
        }

        // 7. Opset imports must have non-zero versions.
        for (domain, &version) in &self.opset_imports {
            if version == 0 {
                errors.push(GraphError::InvalidOpsetImport {
                    domain: domain.clone(),
                    version,
                });
            }
        }

        // 8. Subgraphs validate recursively.
        for sub in self.subgraphs.values() {
            if let Err(mut sub_errors) = sub.validate() {
                errors.append(&mut sub_errors);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    // === Private edge maintenance ===

    /// Wire a live node's edges into its input/output values.
    fn connect_edges(&mut self, id: NodeId) {
        let inputs = self.node(id).inputs.clone();
        let outputs = self.node(id).outputs.clone();
        for (input_index, input) in inputs.into_iter().enumerate() {
            if let Some(value) = input
                && let Some(metadata) = self.values.get_mut(value)
            {
                metadata.consumers.insert(id, input_index as u32);
            }
        }
        for v in outputs {
            if let Some(val) = self.values.get_mut(v) {
                val.producer = Some(id);
            }
        }
    }

    /// Remove a live node's edges from its input/output values, without
    /// deleting the values or the node itself.
    fn disconnect_edges(&mut self, id: NodeId) {
        let input_count = self.node(id).inputs.len();
        let outputs = self.node(id).outputs.clone();
        for input_index in 0..input_count {
            self.replace_input(id, input_index, None);
        }
        for v in outputs {
            if let Some(val) = self.values.get_mut(v)
                && val.producer == Some(id)
            {
                val.producer = None;
            }
        }
    }

    /// Delete `value` if it has no producer, no consumers, and is not part of
    /// the graph's I/O or initializers.
    fn gc_value_if_orphan(&mut self, value: ValueId) {
        let orphan = match self.values.get(value) {
            Some(v) => {
                v.producer.is_none()
                    && v.consumers.is_empty()
                    && !v.is_graph_input
                    && !v.is_graph_output
                    && !self.initializers.contains_key(&value)
            }
            None => false,
        };
        if orphan {
            self.values.remove(value);
            self.unknown_value_types.remove(&value);
            self.unknown_value_shapes.remove(&value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::static_shape;
    use crate::tensor::TensorData;

    fn assert_graphs_identical(
        mut batched: Graph,
        mut sequential: Graph,
        node_probes: usize,
        value_probes: usize,
        trial: usize,
    ) {
        assert_eq!(
            format!("{batched:#?}"),
            format!("{sequential:#?}"),
            "graph mismatch on randomized trial {trial}"
        );

        // Arena free-list order is observable through subsequently allocated
        // IDs, so exhaust the recycled slots as part of the equivalence check.
        for _ in 0..node_probes {
            let batched_id =
                batched.insert_node(Node::new(NodeId(0), "Probe", Vec::new(), Vec::new()));
            let sequential_id =
                sequential.insert_node(Node::new(NodeId(0), "Probe", Vec::new(), Vec::new()));
            assert_eq!(
                batched_id, sequential_id,
                "node arena mismatch on randomized trial {trial}"
            );
        }
        for _ in 0..value_probes {
            let batched_id = batched.create_value(DataType::Float32, static_shape([1]));
            let sequential_id = sequential.create_value(DataType::Float32, static_shape([1]));
            assert_eq!(
                batched_id, sequential_id,
                "value arena mismatch on randomized trial {trial}"
            );
        }
    }

    struct TestRng(u64);

    impl TestRng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn usize(&mut self, upper: usize) -> usize {
            (self.next() as usize) % upper
        }
    }

    fn reference_remove_node(graph: &mut Graph, id: NodeId) {
        if !graph.nodes.contains(id) {
            return;
        }
        let (inputs, outputs) = {
            let node = graph.node(id);
            (
                node.input_values().collect::<Vec<_>>(),
                node.outputs.clone(),
            )
        };
        let unique_inputs: HashSet<_> = inputs.into_iter().collect();
        for value in unique_inputs {
            if let Some(metadata) = graph.values.get_mut(value) {
                for (consumer, input_index) in metadata.consumers.uses() {
                    if consumer == id {
                        metadata.consumers.remove(consumer, input_index);
                    }
                }
            }
        }
        for &value in &outputs {
            if let Some(metadata) = graph.values.get_mut(value)
                && metadata.producer == Some(id)
            {
                metadata.producer = None;
            }
        }
        graph.nodes.remove(id);
        for value in outputs {
            let orphan = graph.values.get(value).is_some_and(|metadata| {
                metadata.producer.is_none()
                    && metadata.consumers.is_empty()
                    && !graph.inputs.contains(&value)
                    && !graph.outputs.contains(&value)
                    && !graph.initializers.contains_key(&value)
            });
            if orphan {
                graph.values.remove(value);
                graph.unknown_value_types.remove(&value);
                graph.unknown_value_shapes.remove(&value);
            }
        }
    }

    fn reference_topological_order(graph: &Graph) -> Result<Vec<NodeId>, GraphError> {
        let mut in_degree: HashMap<NodeId, usize> =
            graph.nodes.keys().map(|node| (node, 0)).collect();
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for (node_id, node) in graph.nodes.iter() {
            for value in node.input_values() {
                if let Some(producer) = graph.value(value).producer
                    && graph.nodes.contains(producer)
                {
                    adjacency.entry(producer).or_default().push(node_id);
                    *in_degree.get_mut(&node_id).unwrap() += 1;
                }
            }
        }
        let mut ready: BinaryHeap<std::cmp::Reverse<u32>> = in_degree
            .iter()
            .filter(|(_, degree)| **degree == 0)
            .map(|(node, _)| std::cmp::Reverse(node.0))
            .collect();
        let mut order = Vec::with_capacity(graph.num_nodes());
        while let Some(std::cmp::Reverse(raw)) = ready.pop() {
            let node = NodeId(raw);
            order.push(node);
            if let Some(successors) = adjacency.get(&node) {
                for successor in successors {
                    let degree = in_degree.get_mut(successor).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        ready.push(std::cmp::Reverse(successor.0));
                    }
                }
            }
        }
        if order.len() == graph.num_nodes() {
            Ok(order)
        } else {
            Err(GraphError::CycleDetected)
        }
    }

    /// Build `a -> Relu -> b -> Add(b, c) -> d`, returning ids.
    fn sample_graph() -> Graph {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let a = g.create_named_value("a", DataType::Float32, static_shape([4]));
        let c = g.create_named_value("c", DataType::Float32, static_shape([4]));
        g.add_input(a);
        g.add_input(c);

        let b = g.create_value(DataType::Float32, static_shape([4]));
        let relu = Node::new(NodeId(0), "Relu", vec![Some(a)], vec![b]);
        g.insert_node(relu);

        let d = g.create_named_value("d", DataType::Float32, static_shape([4]));
        let add = Node::new(NodeId(0), "Add", vec![Some(b), Some(c)], vec![d]);
        g.insert_node(add);
        g.add_output(d);
        g
    }

    #[test]
    fn edges_are_wired_on_insert() {
        let g = sample_graph();
        assert_eq!(g.num_nodes(), 2);
        // b has a producer (Relu) and a consumer (Add)
        let b = g.value(ValueId(2));
        assert!(b.producer.is_some());
        assert_eq!(b.consumers.len(), 1);
    }

    #[test]
    fn topo_order_is_valid_and_deterministic() {
        let g = sample_graph();
        let order = g.topological_order().unwrap();
        assert_eq!(order.len(), 2);
        // Relu (NodeId 0) must come before Add (NodeId 1)
        assert_eq!(order, vec![NodeId(0), NodeId(1)]);
    }

    #[test]
    fn uses_preserve_input_multiplicity_and_replace_one_slot() {
        let mut graph = Graph::new();
        let old = graph.create_value(DataType::Float32, static_shape([1]));
        let new = graph.create_value(DataType::Float32, static_shape([1]));
        graph.add_input(old);
        graph.add_input(new);
        let output = graph.create_value(DataType::Float32, static_shape([1]));
        let node = graph.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(old), Some(old)],
            vec![output],
        ));

        assert_eq!(graph.uses(old), vec![(node, 0), (node, 1)]);
        assert_eq!(graph.consumers(old), vec![node]);
        assert_eq!(graph.replace_input(node, 1, Some(new)), Some(old));
        assert_eq!(graph.uses(old), vec![(node, 0)]);
        assert_eq!(graph.uses(new), vec![(node, 1)]);
        assert!(graph.validate().is_ok());

        graph.remove_node(node);
        assert!(graph.uses(old).is_empty());
        assert!(graph.uses(new).is_empty());
    }

    #[test]
    fn io_membership_flags_follow_ordered_lists() {
        let mut graph = Graph::new();
        let a = graph.create_value(DataType::Float32, static_shape([1]));
        let b = graph.create_value(DataType::Float32, static_shape([1]));
        graph.add_input(a);
        graph.add_output(a);
        graph.insert_output(0, b);
        assert!(graph.value(a).is_graph_input);
        assert!(graph.value(a).is_graph_output);
        assert!(graph.value(b).is_graph_output);

        assert_eq!(graph.remove_output(1), a);
        assert!(!graph.value(a).is_graph_output);
        graph.set_inputs(vec![b]);
        assert!(!graph.value(a).is_graph_input);
        assert!(graph.value(b).is_graph_input);
        graph.set_outputs(vec![a]);
        assert!(graph.value(a).is_graph_output);
        assert!(!graph.value(b).is_graph_output);
    }

    #[test]
    fn consumer_hash_insertion_order_is_not_observable() {
        let mut first = Graph::new();
        let hub = first.create_value(DataType::Float32, static_shape([1]));
        first.add_input(hub);
        let mut nodes = Vec::new();
        for _ in 0..32 {
            let output = first.create_value(DataType::Float32, static_shape([1]));
            nodes.push(first.insert_node(Node::new(
                NodeId(0),
                "Add",
                vec![Some(hub), Some(hub)],
                vec![output],
            )));
        }
        let mut shuffled = first.clone();
        let mut uses = shuffled.uses(hub);
        for &(node, input_index) in &uses {
            shuffled.replace_input(node, input_index as usize, None);
        }
        let mut rng = TestRng(0x6a09_e667_f3bc_c909);
        for index in (1..uses.len()).rev() {
            uses.swap(index, rng.usize(index + 1));
        }
        for (node, input_index) in uses {
            shuffled.replace_input(node, input_index as usize, Some(hub));
        }

        assert_eq!(first.uses(hub), shuffled.uses(hub));
        assert_eq!(first.consumers(hub), shuffled.consumers(hub));
        assert_eq!(first.topological_order(), shuffled.topological_order());
        assert_eq!(format!("{first:#?}"), format!("{shuffled:#?}"));
        assert_eq!(
            format!("{:#?}", sample_graph()),
            format!("{:#?}", sample_graph())
        );
        assert_eq!(nodes, first.consumers(hub));
    }

    #[test]
    fn validate_accepts_wellformed_graph() {
        let g = sample_graph();
        assert!(g.validate().is_ok());
    }

    #[test]
    fn predecessors_and_successors() {
        let g = sample_graph();
        assert_eq!(g.successors(NodeId(0)), vec![NodeId(1)]);
        assert_eq!(g.predecessors(NodeId(1)), vec![NodeId(0)]);
    }

    #[test]
    fn nodes_between_walks_back() {
        let g = sample_graph();
        let between = g.nodes_between(&[ValueId(0), ValueId(1)], &[ValueId(3)]);
        assert!(between.contains(&NodeId(0)));
        assert!(between.contains(&NodeId(1)));
        assert_eq!(between.len(), 2);
    }

    #[test]
    fn replace_all_uses_redirects_consumers() {
        let mut g = sample_graph();
        // New constant value replaces `b` as Add's input.
        let e = g.create_value(DataType::Float32, static_shape([4]));
        g.replace_all_uses(ValueId(2), e);
        // Add now consumes `e`, not `b`.
        let add = g.node(NodeId(1));
        assert!(add.input_values().any(|v| v == e));
        assert!(!add.input_values().any(|v| v == ValueId(2)));
        assert_eq!(g.consumers(e), vec![NodeId(1)]);
        assert!(g.value(ValueId(2)).consumers.is_empty());
    }

    #[test]
    fn insert_on_edge_splices_node() {
        let mut g = sample_graph();
        // Insert an Identity between b (ValueId 2) and its consumer Add.
        let ident = Node::new(NodeId(0), "Identity", vec![], vec![]);
        let nid = g.insert_on_edge(ValueId(2), ident);
        // Add now consumes the new value produced by Identity.
        let new_out = g.node(nid).outputs[0];
        assert_eq!(g.value(new_out).producer, Some(nid));
        assert!(g.node(NodeId(1)).input_values().any(|v| v == new_out));
        assert!(g.validate().is_ok());
    }

    #[test]
    fn remove_node_disconnects_and_gcs() {
        let mut g = sample_graph();
        g.remove_node(NodeId(1)); // remove Add
        assert_eq!(g.num_nodes(), 1);
        // `d` was Add's only output and a graph output -> kept, producer cleared
        assert!(g.value(ValueId(3)).producer.is_none());
        // b lost its consumer
        assert!(g.value(ValueId(2)).consumers.is_empty());
    }

    #[test]
    fn remove_nodes_filters_wide_shared_input_once() {
        let mut graph = Graph::new();
        let input = graph.create_value(DataType::Float32, static_shape([1]));
        graph.add_input(input);
        let mut dead = Vec::new();
        let mut outputs = Vec::new();
        for _ in 0..1_000 {
            let output = graph.create_value(DataType::Float32, static_shape([1]));
            dead.push(graph.insert_node(Node::new(
                NodeId(0),
                "Relu",
                vec![Some(input)],
                vec![output],
            )));
            outputs.push(output);
        }

        graph.remove_nodes(&dead);

        assert_eq!(graph.num_nodes(), 0);
        assert!(graph.value(input).consumers.is_empty());
        assert!(
            outputs
                .into_iter()
                .all(|output| graph.try_value(output).is_none())
        );
    }

    #[test]
    fn remove_nodes_keeps_surviving_shared_consumer() {
        let mut graph = Graph::new();
        let input = graph.create_value(DataType::Float32, static_shape([1]));
        graph.add_input(input);
        let mut nodes = Vec::new();
        for op_type in ["Relu", "Neg", "Abs"] {
            let output = graph.create_value(DataType::Float32, static_shape([1]));
            nodes.push(graph.insert_node(Node::new(
                NodeId(0),
                op_type,
                vec![Some(input)],
                vec![output],
            )));
        }

        graph.remove_nodes(&nodes[..2]);

        assert_eq!(graph.consumers(input), vec![nodes[2]]);
        assert!(graph.try_node(nodes[2]).is_some());
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn remove_nodes_collects_value_after_all_consumers_are_removed() {
        let mut graph = Graph::new();
        let input = graph.create_value(DataType::Float32, static_shape([1]));
        graph.add_input(input);
        let shared = graph.create_value(DataType::Float32, static_shape([1]));
        let producer = graph.insert_node(Node::new(
            NodeId(0),
            "Relu",
            vec![Some(input)],
            vec![shared],
        ));
        let mut consumers = Vec::new();
        for op_type in ["Neg", "Abs"] {
            let output = graph.create_value(DataType::Float32, static_shape([1]));
            consumers.push(graph.insert_node(Node::new(
                NodeId(0),
                op_type,
                vec![Some(shared)],
                vec![output],
            )));
        }

        graph.remove_nodes(&[consumers[0], consumers[1], producer]);

        assert!(graph.try_value(shared).is_none());
    }

    #[test]
    fn remove_nodes_keeps_graph_outputs_and_initializers() {
        let mut graph = Graph::new();
        let input = graph.create_value(DataType::Float32, static_shape([1]));
        graph.add_input(input);

        let graph_output = graph.create_value(DataType::Float32, static_shape([1]));
        let output_node = graph.insert_node(Node::new(
            NodeId(0),
            "Relu",
            vec![Some(input)],
            vec![graph_output],
        ));
        graph.add_output(graph_output);

        let initializer = graph.create_value(DataType::Float32, static_shape([1]));
        let initializer_node = graph.insert_node(Node::new(
            NodeId(0),
            "Neg",
            vec![Some(input)],
            vec![initializer],
        ));
        graph.set_initializer(
            initializer,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![1],
                0.0f32.to_le_bytes().to_vec(),
            )),
        );

        graph.remove_nodes(&[output_node, initializer_node]);

        assert!(graph.try_value(graph_output).is_some());
        assert!(graph.value(graph_output).producer.is_none());
        assert!(graph.try_value(initializer).is_some());
        assert!(graph.value(initializer).producer.is_none());
    }

    #[test]
    fn remove_nodes_ignores_duplicate_and_nonlive_ids() {
        let graph = sample_graph();
        let ids = [NodeId(u32::MAX), NodeId(1), NodeId(1)];
        let mut sequential = graph.clone();
        for id in ids {
            sequential.remove_node(id);
        }
        let mut batched = graph;
        batched.remove_nodes(&ids);

        assert_graphs_identical(batched, sequential, 3, 5, 0);
    }

    #[test]
    fn remove_nodes_matches_sequential_removal_on_random_dags() {
        let mut rng = TestRng(0x4d59_5df4_d0f3_3173);

        for trial in 0..10_000 {
            let input_count = 1 + rng.usize(3);
            let node_count = 1 + rng.usize(12);
            let mut graph = Graph::new();
            let mut values = Vec::new();
            for _ in 0..input_count {
                let input = graph.create_value(DataType::Float32, static_shape([1]));
                graph.add_input(input);
                values.push(input);
            }

            let mut nodes = Vec::with_capacity(node_count);
            for _ in 0..node_count {
                let input_arity = 1 + rng.usize(3);
                let inputs = (0..input_arity)
                    .map(|_| Some(values[rng.usize(values.len())]))
                    .collect();
                let output = graph.create_value(DataType::Float32, static_shape([1]));
                nodes.push(graph.insert_node(Node::new(NodeId(0), "Random", inputs, vec![output])));
                if rng.usize(5) == 0 {
                    graph.mark_value_type_unknown(output);
                }
                if rng.usize(5) == 0 {
                    graph.mark_value_shape_unknown(output);
                }
                values.push(output);
            }

            for _ in 0..rng.usize(4) {
                let output = values[rng.usize(values.len())];
                graph.add_output(output);
            }

            for i in (1..nodes.len()).rev() {
                let j = rng.usize(i + 1);
                nodes.swap(i, j);
            }
            nodes.truncate(rng.usize(nodes.len() + 1));

            let mut sequential = graph.clone();
            for &id in &nodes {
                sequential.remove_node(id);
            }
            let mut batched = graph;
            batched.remove_nodes(&nodes);

            assert_graphs_identical(
                batched,
                sequential,
                node_count + 1,
                input_count + node_count + 1,
                trial,
            );
        }
    }

    #[test]
    fn single_node_removal_matches_vector_reference_on_random_dags() {
        let mut rng = TestRng(0xbb67_ae85_84ca_a73b);
        for trial in 0..2_000 {
            let mut graph = Graph::new();
            let input_count = 1 + rng.usize(3);
            let node_count = 1 + rng.usize(24);
            let mut values = Vec::new();
            for _ in 0..input_count {
                let value = graph.create_value(DataType::Float32, static_shape([1]));
                graph.add_input(value);
                values.push(value);
            }
            let mut nodes = Vec::new();
            for _ in 0..node_count {
                let input_count = 1 + rng.usize(4);
                let inputs = (0..input_count)
                    .map(|_| Some(values[rng.usize(values.len())]))
                    .collect();
                let output = graph.create_value(DataType::Float32, static_shape([1]));
                nodes.push(graph.insert_node(Node::new(NodeId(0), "Random", inputs, vec![output])));
                values.push(output);
            }
            for _ in 0..rng.usize(4) {
                graph.add_output(values[rng.usize(values.len())]);
            }

            let mut removals = nodes.clone();
            for index in (1..removals.len()).rev() {
                removals.swap(index, rng.usize(index + 1));
            }
            removals.truncate(rng.usize(removals.len() + 1));
            if let Some(&duplicate) = removals.first() {
                removals.push(duplicate);
            }
            removals.push(NodeId(u32::MAX));

            let mut actual = graph.clone();
            let mut reference = graph;
            for &node in &removals {
                actual.remove_node(node);
                reference_remove_node(&mut reference, node);
            }

            assert_eq!(
                format!("{actual:#?}"),
                format!("{reference:#?}"),
                "debug mismatch on trial {trial}"
            );
            assert_eq!(
                actual.topological_order(),
                reference.topological_order(),
                "topology mismatch on trial {trial}"
            );
            for value in actual.values.keys() {
                assert_eq!(
                    actual.uses(value),
                    reference.uses(value),
                    "uses mismatch for {value:?} on trial {trial}"
                );
                assert_eq!(
                    actual.consumers(value),
                    reference.consumers(value),
                    "consumers mismatch for {value:?} on trial {trial}"
                );
            }
        }
    }

    #[test]
    fn vec_indexed_topology_matches_hashmap_reference_on_random_dags() {
        let mut rng = TestRng(0x3c6e_f372_fe94_f82b);
        for trial in 0..2_000 {
            let mut graph = Graph::new();
            let input = graph.create_value(DataType::Float32, static_shape([1]));
            graph.add_input(input);
            let mut values = vec![input];
            let mut nodes = Vec::new();
            for _ in 0..(1 + rng.usize(48)) {
                let inputs = (0..(1 + rng.usize(4)))
                    .map(|_| Some(values[rng.usize(values.len())]))
                    .collect();
                let output = graph.create_value(DataType::Float32, static_shape([1]));
                nodes.push(graph.insert_node(Node::new(NodeId(0), "Random", inputs, vec![output])));
                values.push(output);
            }
            for &node in nodes.iter().filter(|_| rng.usize(5) == 0) {
                graph.remove_node(node);
            }
            assert_eq!(
                graph.topological_order(),
                reference_topological_order(&graph),
                "topology mismatch on trial {trial}"
            );
        }
    }

    #[test]
    fn replace_node_groups_matches_sequential_mutation() {
        let mut sequential = Graph::new();
        let input = sequential.create_value(DataType::Float32, static_shape([1]));
        sequential.add_input(input);
        let interior = sequential.create_value(DataType::Float32, static_shape([1]));
        let first = sequential.insert_node(Node::new(
            NodeId(0),
            "Relu",
            vec![Some(input)],
            vec![interior],
        ));
        let output = sequential.create_value(DataType::Float32, static_shape([1]));
        let second = sequential.insert_node(Node::new(
            NodeId(0),
            "Relu",
            vec![Some(interior)],
            vec![output],
        ));
        sequential.add_output(output);
        let sibling_output = sequential.create_value(DataType::Float32, static_shape([1]));
        let sibling = sequential.insert_node(Node::new(
            NodeId(0),
            "Neg",
            vec![Some(input)],
            vec![sibling_output],
        ));
        sequential.add_output(sibling_output);

        let mut batched = sequential.clone();
        let graph_outputs: HashSet<_> = batched.outputs.iter().copied().collect();
        let replacement0 = Node::new(NodeId(0), "EPContext", vec![Some(input)], vec![output]);
        let replacement1 = Node::new(
            NodeId(0),
            "EPContext",
            vec![Some(input)],
            vec![sibling_output],
        );

        sequential.remove_node(first);
        sequential.remove_node(second);
        let replacement0_id = sequential.insert_node(replacement0.clone());
        sequential.remove_node(sibling);
        let replacement1_id = sequential.insert_node(replacement1.clone());

        let inserted = batched.replace_node_groups(
            vec![
                (vec![first, second], replacement0),
                (vec![sibling], replacement1),
            ],
            &graph_outputs,
        );
        assert_eq!(inserted, vec![replacement0_id, replacement1_id]);

        let sequential_nodes: Vec<_> = sequential
            .nodes
            .iter()
            .map(|(id, node)| {
                (
                    id,
                    node.op_type.clone(),
                    node.inputs.clone(),
                    node.outputs.clone(),
                )
            })
            .collect();
        let batched_nodes: Vec<_> = batched
            .nodes
            .iter()
            .map(|(id, node)| {
                (
                    id,
                    node.op_type.clone(),
                    node.inputs.clone(),
                    node.outputs.clone(),
                )
            })
            .collect();
        assert_eq!(batched_nodes, sequential_nodes);

        let sequential_values: Vec<_> = sequential
            .values
            .iter()
            .map(|(id, value)| (id, value.producer, value.consumers.clone()))
            .collect();
        let batched_values: Vec<_> = batched
            .values
            .iter()
            .map(|(id, value)| (id, value.producer, value.consumers.clone()))
            .collect();
        assert_eq!(batched_values, sequential_values);

        let next_sequential =
            sequential.insert_node(Node::new(NodeId(0), "Identity", Vec::new(), Vec::new()));
        let next_batched =
            batched.insert_node(Node::new(NodeId(0), "Identity", Vec::new(), Vec::new()));
        assert_eq!(next_batched, next_sequential);
    }

    #[test]
    fn replace_node_groups_matches_sequential_orphan_collection() {
        let mut sequential = Graph::new();
        let input = sequential.create_value(DataType::Float32, static_shape([1]));
        sequential.add_input(input);
        let interior = sequential.create_value(DataType::Float32, static_shape([1]));
        let producer = sequential.insert_node(Node::new(
            NodeId(0),
            "Relu",
            vec![Some(input)],
            vec![interior],
        ));
        let output = sequential.create_value(DataType::Float32, static_shape([1]));
        let consumer = sequential.insert_node(Node::new(
            NodeId(0),
            "Relu",
            vec![Some(interior)],
            vec![output],
        ));
        sequential.add_output(output);

        let mut batched = sequential.clone();
        let graph_outputs: HashSet<_> = batched.outputs.iter().copied().collect();
        let replacement = Node::new(NodeId(0), "EPContext", vec![Some(input)], vec![output]);

        sequential.remove_node(consumer);
        sequential.remove_node(producer);
        sequential.insert_node(replacement.clone());
        batched.replace_node_groups(
            vec![(vec![consumer, producer], replacement)],
            &graph_outputs,
        );

        assert!(sequential.try_value(interior).is_none());
        assert!(batched.try_value(interior).is_none());
        let next_sequential = sequential.create_value(DataType::Float32, static_shape([1]));
        let next_batched = batched.create_value(DataType::Float32, static_shape([1]));
        assert_eq!(next_batched, next_sequential);
    }

    #[test]
    fn replace_node_preserves_id() {
        let mut g = sample_graph();
        let d = g.node(NodeId(1)).outputs[0];
        let b = g.node(NodeId(1)).inputs[0];
        let c = g.node(NodeId(1)).inputs[1];
        let sub = Node::new(NodeId(0), "Sub", vec![b, c], vec![d]);
        let id = g.replace_node(NodeId(1), sub);
        assert_eq!(id, NodeId(1));
        assert_eq!(g.node(NodeId(1)).op_type, "Sub");
        assert!(g.validate().is_ok());
    }

    #[test]
    fn cycle_is_detected() {
        let mut g = Graph::new();
        let v0 = g.create_value(DataType::Float32, static_shape([1]));
        let v1 = g.create_value(DataType::Float32, static_shape([1]));
        // n0: v1 -> v0 ; n1: v0 -> v1  (cycle)
        g.insert_node(Node::new(NodeId(0), "A", vec![Some(v1)], vec![v0]));
        g.insert_node(Node::new(NodeId(0), "B", vec![Some(v0)], vec![v1]));
        assert_eq!(g.topological_order(), Err(GraphError::CycleDetected));
        assert!(g.validate().is_err());
    }

    #[test]
    fn intern_symbol_dedups_by_name() {
        let mut g = Graph::new();
        let s1 = g.intern_symbol("batch");
        let s2 = g.intern_symbol("batch");
        let s3 = g.intern_symbol("seq");
        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
    }
}
