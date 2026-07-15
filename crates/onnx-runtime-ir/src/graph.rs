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

    /// Register `value` as a graph input (also clears any producer link).
    pub fn add_input(&mut self, value: ValueId) {
        self.inputs.push(value);
    }

    /// Register `value` as a graph output.
    pub fn add_output(&mut self, value: ValueId) {
        self.outputs.push(value);
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
        out
    }

    /// Direct successors: nodes that consume this node's outputs.
    pub fn successors(&self, node: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for &v in &self.node(node).outputs {
            if let Some(val) = self.values.get(v) {
                for &c in &val.consumers {
                    if seen.insert(c) {
                        out.push(c);
                    }
                }
            }
        }
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
        let mut in_degree: HashMap<NodeId, usize> =
            self.nodes.keys().map(|k| (k, 0usize)).collect();
        let mut adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        for (nid, node) in self.nodes.iter() {
            for v in node.input_values() {
                if let Some(val) = self.values.get(v)
                    && let Some(prod) = val.producer
                    && self.nodes.contains(prod)
                {
                    adj.entry(prod).or_default().push(nid);
                    *in_degree.entry(nid).or_insert(0) += 1;
                }
            }
        }

        // Min-heap on raw id for deterministic ordering.
        let mut ready: BinaryHeap<std::cmp::Reverse<u32>> = in_degree
            .iter()
            .filter(|(_, d)| **d == 0)
            .map(|(k, _)| std::cmp::Reverse(k.0))
            .collect();

        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(std::cmp::Reverse(raw)) = ready.pop() {
            let nid = NodeId(raw);
            order.push(nid);
            if let Some(succs) = adj.get(&nid) {
                for &s in succs {
                    let d = in_degree.get_mut(&s).expect("successor tracked");
                    *d -= 1;
                    if *d == 0 {
                        ready.push(std::cmp::Reverse(s.0));
                    }
                }
            }
        }

        if order.len() != self.nodes.len() {
            return Err(GraphError::CycleDetected);
        }
        Ok(order)
    }

    // === Mutation API ===

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

    /// Replace every use of `old_value` with `new_value` in consumer nodes and
    /// in the graph output list, moving consumer edges accordingly.
    pub fn replace_all_uses(&mut self, old_value: ValueId, new_value: ValueId) {
        if old_value == new_value {
            return;
        }
        let consumers = match self.values.get(old_value) {
            Some(v) => v.consumers.clone(),
            None => return,
        };
        for nid in &consumers {
            if let Some(node) = self.nodes.get_mut(*nid) {
                for slot in node.inputs.iter_mut() {
                    if *slot == Some(old_value) {
                        *slot = Some(new_value);
                    }
                }
            }
        }
        // Move consumer entries from old to new.
        let mut moved = std::mem::take(&mut self.value_mut(old_value).consumers);
        if let Some(nv) = self.values.get_mut(new_value) {
            nv.consumers.append(&mut moved);
        }
        // Rewrite graph outputs.
        for out in self.outputs.iter_mut() {
            if *out == old_value {
                *out = new_value;
            }
        }
    }

    // === Validation ===

    /// Verify structural invariants (§3.3). Returns every defect found.
    pub fn validate(&self) -> Result<(), Vec<GraphError>> {
        let mut errors = Vec::new();

        // 1. Node edges reference live values; collect produced values.
        let mut produced: HashMap<ValueId, NodeId> = HashMap::new();
        for (nid, node) in self.nodes.iter() {
            for v in node.input_values() {
                if !self.values.contains(v) {
                    errors.push(GraphError::DanglingValue(v));
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
            for &c in &val.consumers {
                if !self.nodes.contains(c) {
                    errors.push(GraphError::DanglingNode(c));
                } else if !self.node(c).input_values().any(|iv| iv == vid) {
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
            } else {
                errors.push(GraphError::DanglingValue(inp));
            }
        }

        // 5. Graph outputs must be produced (unless they are graph inputs or
        //    initializers passed straight through).
        for &out in &self.outputs {
            match self.values.get(out) {
                Some(val) => {
                    let is_source =
                        self.inputs.contains(&out) || self.initializers.contains_key(&out);
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
        let inputs: Vec<ValueId> = self.node(id).input_values().collect();
        let outputs = self.node(id).outputs.clone();
        for v in inputs {
            if let Some(val) = self.values.get_mut(v) {
                val.consumers.push(id);
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
        let inputs: Vec<ValueId> = self.node(id).input_values().collect();
        let outputs = self.node(id).outputs.clone();
        for v in inputs {
            if let Some(val) = self.values.get_mut(v) {
                val.consumers.retain(|&c| c != id);
            }
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
                    && !self.inputs.contains(&value)
                    && !self.outputs.contains(&value)
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
        assert_eq!(g.value(e).consumers, vec![NodeId(1)]);
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
