//! Dead-node elimination: remove nodes whose outputs are not transitively
//! consumed by any graph output (see `docs/ORT2.md` §18.1).

use std::collections::HashSet;

use onnx_runtime_ir::{Graph, NodeId, ValueId};

use crate::error::Result;
use crate::pass::{OptimizationPass, PassContext};

/// Removes nodes that do not contribute to any graph output.
///
/// A node is *live* if any of its outputs is (transitively) required to compute
/// a value in [`Graph::outputs`]. Liveness is found by walking backwards from
/// the graph outputs through producer edges; every node not reached is removed
/// via [`Graph::remove_node`], which keeps producer/consumer edges consistent
/// and garbage-collects orphaned values. Graph inputs and initializers are left
/// untouched.
#[derive(Clone, Copy, Debug, Default)]
pub struct DeadNodeElimination;

impl DeadNodeElimination {
    /// Node ids reachable (via producer edges) from the graph outputs.
    fn live_nodes(graph: &Graph) -> HashSet<NodeId> {
        let mut live: HashSet<NodeId> = HashSet::new();
        let mut seen_values: HashSet<ValueId> = HashSet::new();
        let mut stack: Vec<ValueId> = graph.outputs.clone();

        while let Some(v) = stack.pop() {
            if !seen_values.insert(v) {
                continue;
            }
            let Some(val) = graph.try_value(v) else {
                continue;
            };
            if let Some(prod) = val.producer
                && live.insert(prod)
            {
                for iv in graph.node(prod).input_values() {
                    stack.push(iv);
                }
            }
        }
        live
    }
}

impl OptimizationPass for DeadNodeElimination {
    fn name(&self) -> &str {
        "DeadNodeElimination"
    }

    fn run(&self, graph: &mut Graph, _ctx: &PassContext) -> Result<()> {
        let live = Self::live_nodes(graph);
        let dead: Vec<NodeId> = graph
            .nodes
            .keys()
            .filter(|nid| !live.contains(nid))
            .collect();
        for nid in dead {
            graph.remove_node(nid);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{DataType, Node, NodeId, static_shape};

    /// `a -> Relu -> b -> Add(b, c) -> d(out)`, plus a dangling
    /// `a -> Neg -> e` branch whose result `e` feeds nothing.
    fn graph_with_dangling_branch() -> (Graph, NodeId, NodeId) {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let a = g.create_named_value("a", DataType::Float32, static_shape([4]));
        let c = g.create_named_value("c", DataType::Float32, static_shape([4]));
        g.add_input(a);
        g.add_input(c);

        let b = g.create_value(DataType::Float32, static_shape([4]));
        let relu = g.insert_node(Node::new(NodeId(0), "Relu", vec![Some(a)], vec![b]));

        let d = g.create_named_value("d", DataType::Float32, static_shape([4]));
        g.insert_node(Node::new(NodeId(0), "Add", vec![Some(b), Some(c)], vec![d]));
        g.add_output(d);

        // Dangling branch: consumes `a`, produces `e`, which nothing reads.
        let e = g.create_value(DataType::Float32, static_shape([4]));
        let neg = g.insert_node(Node::new(NodeId(0), "Neg", vec![Some(a)], vec![e]));

        (g, relu, neg)
    }

    #[test]
    fn removes_dangling_branch() {
        let (mut g, _relu, neg) = graph_with_dangling_branch();
        assert_eq!(g.num_nodes(), 3);
        DeadNodeElimination
            .run(&mut g, &PassContext::new())
            .unwrap();
        assert_eq!(g.num_nodes(), 2);
        assert!(g.try_node(neg).is_none());
        assert!(g.validate().is_ok());
    }

    #[test]
    fn keeps_live_nodes() {
        let (mut g, relu, _neg) = graph_with_dangling_branch();
        DeadNodeElimination
            .run(&mut g, &PassContext::new())
            .unwrap();
        // Relu and Add remain.
        assert!(g.try_node(relu).is_some());
        assert!(g.validate().is_ok());
    }

    #[test]
    fn dead_value_is_garbage_collected() {
        let (mut g, _relu, _neg) = graph_with_dangling_branch();
        let before = g.num_values();
        DeadNodeElimination
            .run(&mut g, &PassContext::new())
            .unwrap();
        // `e` had no other consumer, so removing Neg should GC it.
        assert!(g.num_values() < before);
    }

    #[test]
    fn noop_on_all_live_graph() {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let a = g.create_named_value("a", DataType::Float32, static_shape([2]));
        g.add_input(a);
        let b = g.create_named_value("b", DataType::Float32, static_shape([2]));
        g.insert_node(Node::new(NodeId(0), "Relu", vec![Some(a)], vec![b]));
        g.add_output(b);

        DeadNodeElimination
            .run(&mut g, &PassContext::new())
            .unwrap();
        assert_eq!(g.num_nodes(), 1);
        assert!(g.validate().is_ok());
    }

    #[test]
    fn removes_transitively_dead_chain() {
        // a(in) -> Relu -> b -> Neg -> e ; and a -> Abs -> out. b/e are dead.
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let a = g.create_named_value("a", DataType::Float32, static_shape([2]));
        g.add_input(a);

        let b = g.create_value(DataType::Float32, static_shape([2]));
        g.insert_node(Node::new(NodeId(0), "Relu", vec![Some(a)], vec![b]));
        let e = g.create_value(DataType::Float32, static_shape([2]));
        g.insert_node(Node::new(NodeId(0), "Neg", vec![Some(b)], vec![e]));

        let out = g.create_named_value("out", DataType::Float32, static_shape([2]));
        g.insert_node(Node::new(NodeId(0), "Abs", vec![Some(a)], vec![out]));
        g.add_output(out);

        DeadNodeElimination
            .run(&mut g, &PassContext::new())
            .unwrap();
        assert_eq!(g.num_nodes(), 1); // only Abs survives
        assert!(g.validate().is_ok());
    }
}
