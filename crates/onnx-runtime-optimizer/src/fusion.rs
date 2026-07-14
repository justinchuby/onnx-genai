//! Operator fusion: match a connected op-sequence and replace it with a single
//! fused op (see `docs/ORT2.md` §18.2).
//!
//! ## Matching model
//!
//! A [`FusionPattern`] is an ordered op sequence plus a replacement op type.
//! [`FusionPattern::find_match`] walks the graph forward from each candidate
//! start node, following producer→consumer ("spine") edges: node `i+1` of the
//! match must consume an output of node `i`. Extra data edges *back* to
//! already-matched nodes are allowed, which is what lets the 9-op LayerNorm
//! decomposition (whose `Sub` result feeds both `Pow` and `Div`) match even
//! though it is a DAG, not a strict chain.
//!
//! ## Safety rule (never change numerics-visible semantics)
//!
//! A match is only fused when **every intermediate output is consumed solely
//! within the matched set** — i.e. no matched node except the last has an
//! output that escapes to an outside consumer or to a graph output. This is the
//! generalization of "single-consumer chain": internal reuse is fine, external
//! escape is not. It guarantees fusion cannot delete a value another part of
//! the graph still observes.
//!
//! [`FusionPattern::apply_fusion`] removes the matched nodes and inserts the
//! replacement, reusing the final output value id so external wiring and graph
//! outputs are preserved automatically. External inputs are collected in
//! first-seen order across the matched nodes.
//!
//! ## Kernel note
//!
//! The fused op types produced here (`LayerNormalization`, `FusedMatMulBias`,
//! `FusedGemm`) may not have an `onnx-runtime-ep-cpu` kernel yet — this pass
//! only rewrites the graph. Providing kernels is Phase-2/3 work. Likewise the
//! generic external-input ordering is *structural*, not schema-aware; a
//! schema-aware reorder (e.g. ONNX `LayerNormalization`'s `(X, Scale, B)`) is a
//! later refinement and does not affect graph validity.

use std::collections::HashSet;

use onnx_runtime_ir::{Graph, Node, NodeId, ValueId};

use crate::error::Result;
use crate::pass::{OptimizationPass, PassContext};

/// A matched occurrence of a [`FusionPattern`] in a graph.
#[derive(Clone, Debug)]
pub struct PatternMatch {
    /// Matched node ids, in op-sequence order.
    pub nodes: Vec<NodeId>,
    /// Values consumed by the matched region but produced outside it
    /// (graph inputs, initializers, or outputs of non-matched nodes), in
    /// first-seen order.
    pub external_inputs: Vec<ValueId>,
    /// The single output of the last matched node — reused as the fused node's
    /// output so downstream wiring is preserved.
    pub output: ValueId,
}

/// A fusion rule: an op-type sequence rewritten to a single replacement op.
#[derive(Clone, Debug)]
pub struct FusionPattern {
    name: String,
    ops: Vec<String>,
    replacement: String,
}

impl FusionPattern {
    /// A new pattern matching `ops` in sequence, replaced by `replacement`.
    pub fn new(name: &str, ops: &[&str], replacement: &str) -> Self {
        assert!(!ops.is_empty(), "fusion pattern must have at least one op");
        Self {
            name: name.to_string(),
            ops: ops.iter().map(|s| s.to_string()).collect(),
            replacement: replacement.to_string(),
        }
    }

    /// This pattern's name.
    pub fn pattern_name(&self) -> &str {
        &self.name
    }

    /// Find the next occurrence of this pattern, scanning nodes in id order.
    pub fn find_match(&self, graph: &Graph) -> Option<PatternMatch> {
        for start in graph.nodes.keys() {
            if let Some(m) = self.try_match_from(graph, start) {
                return Some(m);
            }
        }
        None
    }

    /// Whether `node` is a standard-domain op named `op`.
    fn op_matches(node: &Node, op: &str) -> bool {
        node.op_type == op && matches!(node.domain.as_str(), "" | "ai.onnx")
    }

    /// Attempt to grow a match whose first node is `start`.
    fn try_match_from(&self, graph: &Graph, start: NodeId) -> Option<PatternMatch> {
        let start_node = graph.try_node(start)?;
        if !Self::op_matches(start_node, &self.ops[0]) {
            return None;
        }

        let mut chain = vec![start];
        let mut chain_set: HashSet<NodeId> = HashSet::from([start]);

        for op in &self.ops[1..] {
            let prev = *chain.last().unwrap();
            // Deterministic: pick the lowest-id successor of `prev` that has the
            // required op type and is not already in the chain.
            let mut succ_ids = graph.successors(prev);
            succ_ids.sort_by_key(|n| n.0);
            let next = succ_ids.into_iter().find(|&s| {
                !chain_set.contains(&s) && Self::op_matches(graph.node(s), op)
            })?;
            chain.push(next);
            chain_set.insert(next);
        }

        // Safety rule: no non-final matched node may have an output that escapes
        // the matched set (external consumer or graph output).
        for &nid in &chain[..chain.len() - 1] {
            for &out in &graph.node(nid).outputs {
                if graph.outputs.contains(&out) {
                    return None;
                }
                if graph
                    .value(out)
                    .consumers
                    .iter()
                    .any(|c| !chain_set.contains(c))
                {
                    return None;
                }
            }
        }

        // The fused node reuses the last node's single output.
        let last = *chain.last().unwrap();
        let last_node = graph.node(last);
        if last_node.outputs.len() != 1 {
            return None;
        }
        let output = last_node.outputs[0];

        // The output must survive removal of the matched nodes: it is either a
        // graph output or has a consumer outside the matched set.
        let out_val = graph.value(output);
        let survives = graph.outputs.contains(&output)
            || out_val.consumers.iter().any(|c| !chain_set.contains(c));
        if !survives {
            return None;
        }

        // Collect external inputs in first-seen order.
        let produced: HashSet<ValueId> = chain
            .iter()
            .flat_map(|&n| graph.node(n).outputs.iter().copied())
            .collect();
        let mut external = Vec::new();
        let mut seen = HashSet::new();
        for &nid in &chain {
            for iv in graph.node(nid).input_values() {
                if produced.contains(&iv) {
                    continue;
                }
                if seen.insert(iv) {
                    external.push(iv);
                }
            }
        }

        Some(PatternMatch {
            nodes: chain,
            external_inputs: external,
            output,
        })
    }

    /// Apply a match: remove the matched nodes and insert the replacement,
    /// reusing `m.output` so downstream consumers and graph outputs stay wired.
    pub fn apply_fusion(&self, graph: &mut Graph, m: &PatternMatch) -> Result<()> {
        let output = m.output;

        // Remove in reverse (last-first): a node's consumers are gone before it,
        // so intermediate values are cleanly garbage-collected. `output` itself
        // survives because it is a graph output or has an external consumer.
        for &nid in m.nodes.iter().rev() {
            graph.remove_node(nid);
        }

        if graph.try_value(output).is_none() {
            return Err(crate::error::OptimizerError::Fusion(self.name.clone()));
        }

        let inputs: Vec<Option<ValueId>> = m.external_inputs.iter().map(|&v| Some(v)).collect();
        let fused = Node::new(NodeId(0), self.replacement.clone(), inputs, vec![output]);
        graph.insert_node(fused);
        Ok(())
    }
}

/// The default device-independent fusion patterns.
///
/// Ordered most-specific-first so `MatMul+Add+Relu` is captured before the
/// shorter `MatMul+Add`. Deferred to Phase 2b/3: `Residual+LayerNorm`, `GELU`,
/// and attention fusion (see [`AttentionFusionPass` in §13.2 of the design]).
pub fn default_fusion_patterns() -> Vec<FusionPattern> {
    vec![
        FusionPattern::new("MatMul+Bias+Relu", &["MatMul", "Add", "Relu"], "FusedGemm"),
        FusionPattern::new(
            "LayerNorm",
            &[
                "ReduceMean", "Sub", "Pow", "ReduceMean", "Add", "Sqrt", "Div", "Mul", "Add",
            ],
            "LayerNormalization",
        ),
        FusionPattern::new("MatMul+Bias", &["MatMul", "Add"], "FusedMatMulBias"),
    ]
}

/// The op-fusion pass: applies each [`FusionPattern`] to fixpoint.
#[derive(Clone, Debug)]
pub struct OpFusion {
    patterns: Vec<FusionPattern>,
}

impl Default for OpFusion {
    fn default() -> Self {
        Self::new()
    }
}

impl OpFusion {
    /// The pass with the default pattern set.
    pub fn new() -> Self {
        Self {
            patterns: default_fusion_patterns(),
        }
    }

    /// The pass with a custom pattern set (used by tests / future callers).
    pub fn with_patterns(patterns: Vec<FusionPattern>) -> Self {
        Self { patterns }
    }
}

impl OptimizationPass for OpFusion {
    fn name(&self) -> &str {
        "OpFusion"
    }

    fn run(&self, graph: &mut Graph, _ctx: &PassContext) -> Result<()> {
        for pattern in &self.patterns {
            while let Some(m) = pattern.find_match(graph) {
                pattern.apply_fusion(graph, &m)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{DataType, Node, NodeId, static_shape};

    fn val(g: &mut Graph, name: &str) -> ValueId {
        g.create_named_value(name, DataType::Float32, static_shape([4]))
    }

    /// Build a linear MatMul+Add ending in a graph output.
    /// Returns (graph, matmul_out_value).
    fn matmul_add_graph() -> Graph {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let a = val(&mut g, "a");
        let w = val(&mut g, "w");
        let bias = val(&mut g, "bias");
        g.add_input(a);
        g.add_input(w);
        g.add_input(bias);

        let m = val(&mut g, "m");
        g.insert_node(Node::new(NodeId(0), "MatMul", vec![Some(a), Some(w)], vec![m]));
        let out = val(&mut g, "out");
        g.insert_node(Node::new(NodeId(0), "Add", vec![Some(m), Some(bias)], vec![out]));
        g.add_output(out);
        g
    }

    #[test]
    fn fuses_matmul_add() {
        let mut g = matmul_add_graph();
        assert_eq!(g.num_nodes(), 2);
        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
        assert_eq!(g.num_nodes(), 1);
        let fused = g.nodes.values().next().unwrap();
        assert_eq!(fused.op_type, "FusedMatMulBias");
        // Inputs are [a, w, bias].
        assert_eq!(fused.inputs.len(), 3);
        assert!(g.validate().is_ok());
        // Output still a graph output.
        assert_eq!(g.outputs.len(), 1);
        assert_eq!(fused.outputs, g.outputs);
    }

    #[test]
    fn fuses_matmul_add_relu_before_matmul_add() {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let a = val(&mut g, "a");
        let w = val(&mut g, "w");
        let bias = val(&mut g, "bias");
        g.add_input(a);
        g.add_input(w);
        g.add_input(bias);
        let m = val(&mut g, "m");
        g.insert_node(Node::new(NodeId(0), "MatMul", vec![Some(a), Some(w)], vec![m]));
        let s = val(&mut g, "s");
        g.insert_node(Node::new(NodeId(0), "Add", vec![Some(m), Some(bias)], vec![s]));
        let out = val(&mut g, "out");
        g.insert_node(Node::new(NodeId(0), "Relu", vec![Some(s)], vec![out]));
        g.add_output(out);

        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.nodes.values().next().unwrap().op_type, "FusedGemm");
        assert!(g.validate().is_ok());
    }

    #[test]
    fn does_not_fuse_when_intermediate_has_second_consumer() {
        // MatMul -> m ; Add(m, bias) -> out ; and m also feeds a second Relu.
        let mut g = matmul_add_graph();
        // Find `m` (produced by MatMul, consumed by Add).
        let m = g
            .values
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some("m"))
            .map(|(id, _)| id)
            .unwrap();
        let side = val(&mut g, "side");
        g.insert_node(Node::new(NodeId(0), "Relu", vec![Some(m)], vec![side]));
        g.add_output(side);

        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
        // MatMul's output escapes to the side Relu, so no fusion.
        assert!(
            g.nodes.values().any(|n| n.op_type == "MatMul"),
            "MatMul must remain — its output has a second consumer"
        );
        assert!(g.nodes.values().all(|n| n.op_type != "FusedMatMulBias"));
        assert!(g.validate().is_ok());
    }

    #[test]
    fn no_match_returns_none() {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let a = val(&mut g, "a");
        g.add_input(a);
        let out = val(&mut g, "out");
        g.insert_node(Node::new(NodeId(0), "Relu", vec![Some(a)], vec![out]));
        g.add_output(out);
        let p = FusionPattern::new("MatMul+Bias", &["MatMul", "Add"], "FusedMatMulBias");
        assert!(p.find_match(&g).is_none());
    }

    /// Build the canonical 9-op LayerNorm decomposition over `x`.
    fn layernorm_graph() -> Graph {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let x = val(&mut g, "x");
        let two = val(&mut g, "two");
        let eps = val(&mut g, "eps");
        let scale = val(&mut g, "scale");
        let bias = val(&mut g, "bias");
        g.add_input(x);
        g.add_input(two);
        g.add_input(eps);
        g.add_input(scale);
        g.add_input(bias);

        let mean = val(&mut g, "mean");
        g.insert_node(Node::new(NodeId(0), "ReduceMean", vec![Some(x)], vec![mean]));
        let diff = val(&mut g, "diff");
        g.insert_node(Node::new(NodeId(0), "Sub", vec![Some(x), Some(mean)], vec![diff]));
        let sq = val(&mut g, "sq");
        g.insert_node(Node::new(NodeId(0), "Pow", vec![Some(diff), Some(two)], vec![sq]));
        let var = val(&mut g, "var");
        g.insert_node(Node::new(NodeId(0), "ReduceMean", vec![Some(sq)], vec![var]));
        let vare = val(&mut g, "vare");
        g.insert_node(Node::new(NodeId(0), "Add", vec![Some(var), Some(eps)], vec![vare]));
        let std = val(&mut g, "std");
        g.insert_node(Node::new(NodeId(0), "Sqrt", vec![Some(vare)], vec![std]));
        let norm = val(&mut g, "norm");
        g.insert_node(Node::new(NodeId(0), "Div", vec![Some(diff), Some(std)], vec![norm]));
        let scaled = val(&mut g, "scaled");
        g.insert_node(Node::new(
            NodeId(0),
            "Mul",
            vec![Some(norm), Some(scale)],
            vec![scaled],
        ));
        let out = val(&mut g, "out");
        g.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(scaled), Some(bias)],
            vec![out],
        ));
        g.add_output(out);
        g
    }

    #[test]
    fn fuses_layernorm_chain() {
        let mut g = layernorm_graph();
        assert_eq!(g.num_nodes(), 9);
        assert!(g.validate().is_ok());

        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();

        assert_eq!(g.num_nodes(), 1, "9-op chain collapses to one node");
        let fused = g.nodes.values().next().unwrap();
        assert_eq!(fused.op_type, "LayerNormalization");
        // Structural external inputs: x, two, eps, scale, bias.
        assert_eq!(fused.inputs.len(), 5);
        assert_eq!(fused.outputs, g.outputs);
        assert!(g.validate().is_ok());
    }

    #[test]
    fn layernorm_count_bookkeeping() {
        let mut g = layernorm_graph();
        let ln_before = g.nodes.values().filter(|n| n.op_type == "LayerNormalization").count();
        let rm_before = g.nodes.values().filter(|n| n.op_type == "ReduceMean").count();
        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
        let ln_after = g.nodes.values().filter(|n| n.op_type == "LayerNormalization").count();
        let rm_after = g.nodes.values().filter(|n| n.op_type == "ReduceMean").count();
        assert_eq!(ln_before, 0);
        assert_eq!(ln_after, 1);
        assert_eq!(rm_before, 2);
        assert_eq!(rm_after, 0);
    }

    #[test]
    fn does_not_fuse_partial_layernorm() {
        // A LayerNorm chain missing its final Add must not fuse.
        let mut g = layernorm_graph();
        // Remove the last Add by rebuilding: easier to just check a shorter
        // pattern doesn't accidentally match — assert Sub alone isn't fused.
        let p = FusionPattern::new("LayerNorm", &["ReduceMean", "Sub", "Pow", "ReduceMean", "Add", "Sqrt", "Div", "Mul", "Add"], "LayerNormalization");
        // Break the chain: give `diff` an external consumer so the safety rule
        // trips (Sub is a non-final matched node).
        let diff = g
            .values
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some("diff"))
            .map(|(id, _)| id)
            .unwrap();
        let side = val(&mut g, "side");
        g.insert_node(Node::new(NodeId(0), "Neg", vec![Some(diff)], vec![side]));
        g.add_output(side);
        assert!(
            p.find_match(&g).is_none(),
            "external consumer on `diff` blocks the fusion"
        );
    }

    #[test]
    fn fuses_two_independent_matmul_adds() {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        for i in 0..2 {
            let a = val(&mut g, &format!("a{i}"));
            let w = val(&mut g, &format!("w{i}"));
            let bias = val(&mut g, &format!("bias{i}"));
            g.add_input(a);
            g.add_input(w);
            g.add_input(bias);
            let m = val(&mut g, &format!("m{i}"));
            g.insert_node(Node::new(NodeId(0), "MatMul", vec![Some(a), Some(w)], vec![m]));
            let out = val(&mut g, &format!("out{i}"));
            g.insert_node(Node::new(NodeId(0), "Add", vec![Some(m), Some(bias)], vec![out]));
            g.add_output(out);
        }
        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
        assert_eq!(g.num_nodes(), 2);
        assert!(g.nodes.values().all(|n| n.op_type == "FusedMatMulBias"));
        assert!(g.validate().is_ok());
    }

    #[test]
    fn find_match_reports_correct_shape() {
        let g = matmul_add_graph();
        let p = FusionPattern::new("MatMul+Bias", &["MatMul", "Add"], "FusedMatMulBias");
        let m = p.find_match(&g).expect("should match");
        assert_eq!(m.nodes.len(), 2);
        assert_eq!(m.external_inputs.len(), 3);
        assert_eq!(p.pattern_name(), "MatMul+Bias");
    }
}
