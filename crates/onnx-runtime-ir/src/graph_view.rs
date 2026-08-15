//! Immutable, cached structural lens over a finalized [`Graph`].

use std::ops::Range;

use crate::{Graph, IrError, Node, NodeId, Value, ValueId};

/// Dense, view-local node identity in topological order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeIndex(u32);

impl NodeIndex {
    /// Dense zero-based position in the view.
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

/// Dense, view-local value identity in ascending live arena order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueIndex(u32);

impl ValueIndex {
    /// Dense zero-based position in the view.
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

/// One positional use of a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConsumerUse {
    pub node: NodeIndex,
    pub input_slot: u32,
}

/// Cached graph structure shared by every borrowed [`GraphView`].
#[derive(Clone, Debug)]
pub struct GraphViewCache {
    topo_nodes: Vec<NodeId>,
    live_values: Vec<ValueId>,
    node_index_by_raw_id: Vec<Option<NodeIndex>>,
    value_index_by_raw_id: Vec<Option<ValueIndex>>,
    node_inputs: Vec<Range<usize>>,
    flat_node_inputs: Vec<Option<ValueIndex>>,
    node_outputs: Vec<Range<usize>>,
    flat_node_outputs: Vec<ValueIndex>,
    value_producer: Vec<Option<NodeIndex>>,
    value_consumer_uses: Vec<Range<usize>>,
    consumer_uses: Vec<ConsumerUse>,
    initializer_bits: Vec<bool>,
}

impl GraphViewCache {
    /// Build deterministic dense topology and edge indices for `graph`.
    pub fn build(graph: &Graph) -> Result<Self, IrError> {
        let topo_nodes = graph
            .topological_order()
            .map_err(|_| IrError::CycleDetected)?;
        let live_values: Vec<_> = graph.values.keys().collect();

        let node_lookup_len = topo_nodes
            .iter()
            .map(|id| id.0 as usize)
            .max()
            .map_or(0, |max| max + 1);
        let value_lookup_len = live_values
            .iter()
            .map(|id| id.0 as usize)
            .max()
            .map_or(0, |max| max + 1);
        let mut node_index_by_raw_id = vec![None; node_lookup_len];
        for (dense, &id) in topo_nodes.iter().enumerate() {
            node_index_by_raw_id[id.0 as usize] = Some(NodeIndex(
                u32::try_from(dense).expect("node count exceeds u32"),
            ));
        }
        let mut value_index_by_raw_id = vec![None; value_lookup_len];
        for (dense, &id) in live_values.iter().enumerate() {
            value_index_by_raw_id[id.0 as usize] = Some(ValueIndex(
                u32::try_from(dense).expect("value count exceeds u32"),
            ));
        }

        let mut node_inputs = Vec::with_capacity(topo_nodes.len());
        let mut flat_node_inputs = Vec::new();
        let mut node_outputs = Vec::with_capacity(topo_nodes.len());
        let mut flat_node_outputs = Vec::new();
        let mut consumer_counts = vec![0usize; live_values.len()];

        for &node_id in &topo_nodes {
            let node = graph.node(node_id);
            let input_start = flat_node_inputs.len();
            for input in &node.inputs {
                let dense = input.map(|id| {
                    value_index_by_raw_id[id.0 as usize]
                        .expect("validated node input must be a live value")
                });
                if let Some(value) = dense {
                    consumer_counts[value.as_usize()] += 1;
                }
                flat_node_inputs.push(dense);
            }
            node_inputs.push(input_start..flat_node_inputs.len());

            let output_start = flat_node_outputs.len();
            for &output in &node.outputs {
                flat_node_outputs.push(
                    value_index_by_raw_id[output.0 as usize]
                        .expect("validated node output must be a live value"),
                );
            }
            node_outputs.push(output_start..flat_node_outputs.len());
        }

        let mut value_consumer_uses = Vec::with_capacity(live_values.len());
        let mut consumer_total = 0usize;
        for count in consumer_counts {
            let start = consumer_total;
            consumer_total += count;
            value_consumer_uses.push(start..consumer_total);
        }
        let mut consumer_uses = vec![
            ConsumerUse {
                node: NodeIndex(0),
                input_slot: 0,
            };
            consumer_total
        ];
        let mut next: Vec<_> = value_consumer_uses
            .iter()
            .map(|range| range.start)
            .collect();
        // Ascending raw node ID plus positional input order preserves the existing
        // public `(NodeId, input slot)` consumer ordering.
        for (node_id, node_payload) in graph.nodes.iter() {
            let node = node_index_by_raw_id[node_id.0 as usize]
                .expect("every live node must have a dense index");
            for (slot, input) in node_payload.inputs.iter().enumerate() {
                let Some(input) = input else { continue };
                let value = value_index_by_raw_id[input.0 as usize]
                    .expect("validated node input must be a live value");
                let write = &mut next[value.as_usize()];
                consumer_uses[*write] = ConsumerUse {
                    node,
                    input_slot: u32::try_from(slot).expect("node input count exceeds u32"),
                };
                *write += 1;
            }
        }

        let value_producer = live_values
            .iter()
            .map(|&id| {
                graph
                    .value(id)
                    .producer
                    .and_then(|producer| node_index_by_raw_id[producer.0 as usize])
            })
            .collect();
        let initializer_bits = live_values
            .iter()
            .map(|id| graph.initializers.contains_key(id))
            .collect();

        Ok(Self {
            topo_nodes,
            live_values,
            node_index_by_raw_id,
            value_index_by_raw_id,
            node_inputs,
            flat_node_inputs,
            node_outputs,
            flat_node_outputs,
            value_producer,
            value_consumer_uses,
            consumer_uses,
            initializer_bits,
        })
    }
}

/// Borrowed immutable graph structure and payload view.
#[derive(Clone, Copy)]
pub struct GraphView<'a> {
    graph: &'a Graph,
    cache: &'a GraphViewCache,
}

impl<'a> GraphView<'a> {
    /// Construct a view from a graph and its matching cache.
    pub fn new(graph: &'a Graph, cache: &'a GraphViewCache) -> Self {
        Self { graph, cache }
    }

    /// Underlying immutable graph payload.
    pub const fn graph(self) -> &'a Graph {
        self.graph
    }

    /// Dense nodes in deterministic topological order.
    pub fn nodes(self) -> impl ExactSizeIterator<Item = NodeIndex> {
        (0..self.cache.topo_nodes.len())
            .map(|index| NodeIndex(u32::try_from(index).expect("node count exceeds u32")))
    }

    /// Dense live values in ascending finalized [`ValueId`] order.
    pub fn values(self) -> impl ExactSizeIterator<Item = ValueIndex> {
        (0..self.cache.live_values.len())
            .map(|index| ValueIndex(u32::try_from(index).expect("value count exceeds u32")))
    }

    pub fn node_index(self, id: NodeId) -> Option<NodeIndex> {
        self.cache
            .node_index_by_raw_id
            .get(id.0 as usize)
            .copied()
            .flatten()
    }

    pub fn value_index(self, id: ValueId) -> Option<ValueIndex> {
        self.cache
            .value_index_by_raw_id
            .get(id.0 as usize)
            .copied()
            .flatten()
    }

    pub fn node_id(self, index: NodeIndex) -> NodeId {
        self.cache.topo_nodes[index.as_usize()]
    }

    pub fn value_id(self, index: ValueIndex) -> ValueId {
        self.cache.live_values[index.as_usize()]
    }

    pub fn node(self, index: NodeIndex) -> &'a Node {
        self.graph.node(self.node_id(index))
    }

    pub fn value(self, index: ValueIndex) -> &'a Value {
        self.graph.value(self.value_id(index))
    }

    /// Positional inputs, retaining omitted optional slots.
    pub fn node_inputs(self, node: NodeIndex) -> &'a [Option<ValueIndex>] {
        &self.cache.flat_node_inputs[self.cache.node_inputs[node.as_usize()].clone()]
    }

    /// Positional outputs.
    pub fn node_outputs(self, node: NodeIndex) -> &'a [ValueIndex] {
        &self.cache.flat_node_outputs[self.cache.node_outputs[node.as_usize()].clone()]
    }

    pub fn producer(self, value: ValueIndex) -> Option<NodeIndex> {
        self.cache.value_producer[value.as_usize()]
    }

    /// Consumer uses sorted by ascending raw node ID and input slot.
    pub fn consumers(self, value: ValueIndex) -> &'a [ConsumerUse] {
        &self.cache.consumer_uses[self.cache.value_consumer_uses[value.as_usize()].clone()]
    }

    pub fn is_initializer(self, value: ValueIndex) -> bool {
        self.cache.initializer_bits[value.as_usize()]
    }
}

/// Owner for a validated finalized graph and its structural cache.
#[derive(Clone, Debug)]
pub struct FrozenGraph {
    graph: Graph,
    cache: GraphViewCache,
}

impl FrozenGraph {
    /// Validate in release builds and freeze `graph` for immutable projection.
    pub fn build(graph: Graph) -> Result<Self, IrError> {
        graph.validate().map_err(IrError::GraphInvalid)?;
        let cache = GraphViewCache::build(&graph)?;
        Ok(Self { graph, cache })
    }

    pub fn view(&self) -> GraphView<'_> {
        GraphView::new(&self.graph, &self.cache)
    }

    /// Reopen the graph for structural mutation, dropping the stale cache.
    pub fn into_graph(self) -> Graph {
        self.graph
    }
}

#[cfg(test)]
mod tests {
    use crate::{DataType, Node, NodeId, static_shape};

    use super::*;

    fn chain() -> (Graph, NodeId, NodeId, ValueId, ValueId, ValueId) {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        let input = graph.create_value(DataType::Float32, static_shape([2]));
        let middle = graph.create_value(DataType::Float32, static_shape([2]));
        let output = graph.create_value(DataType::Float32, static_shape([2]));
        graph.add_input(input);
        let first = graph.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(input), Some(input)],
            vec![middle],
        ));
        let second = graph.insert_node(Node::new(
            NodeId(0),
            "Relu",
            vec![Some(middle)],
            vec![output],
        ));
        graph.add_output(output);
        (graph, first, second, input, middle, output)
    }

    #[test]
    fn cache_preserves_topology_edges_and_consumer_order() {
        let (graph, first, second, input, middle, output) = chain();
        let frozen = FrozenGraph::build(graph).unwrap();
        let view = frozen.view();
        let nodes: Vec<_> = view.nodes().map(|node| view.node_id(node)).collect();
        assert_eq!(nodes, vec![first, second]);

        let first = view.node_index(first).unwrap();
        let second = view.node_index(second).unwrap();
        let input = view.value_index(input).unwrap();
        let middle = view.value_index(middle).unwrap();
        let output = view.value_index(output).unwrap();
        assert_eq!(view.node_inputs(first), &[Some(input), Some(input)]);
        assert_eq!(
            view.consumers(input),
            &[
                ConsumerUse {
                    node: first,
                    input_slot: 0
                },
                ConsumerUse {
                    node: first,
                    input_slot: 1
                }
            ]
        );
        assert_eq!(view.producer(middle), Some(first));
        assert_eq!(view.consumers(middle)[0].node, second);
        assert_eq!(view.node_outputs(second), &[output]);
    }

    #[test]
    fn view_is_read_only_and_reopening_drops_the_cache() {
        let (graph, first, _, _, _, _) = chain();
        let frozen = FrozenGraph::build(graph).unwrap();
        {
            let view = frozen.view();
            assert_eq!(view.node(view.node_index(first).unwrap()).device, None);
        }
        let mut graph = frozen.into_graph();
        graph.node_mut(first).name = "mutated-after-view-dropped".into();
        assert_eq!(graph.node(first).name, "mutated-after-view-dropped");
    }
}
