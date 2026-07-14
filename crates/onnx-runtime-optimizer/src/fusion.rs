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
//! The optimizer-produced fused op types (`LayerNormalization`,
//! `FusedMatMulBias`, `FusedGemm`) are emitted in the private contrib domain
//! [`CONTRIB_DOMAIN`] (`com.microsoft`), **not** the reserved default ONNX
//! domain. `FusedMatMulBias`/`FusedGemm` are invented (non-standard) ops, so
//! putting them in `ai.onnx` would collide with standard-op opset validation and
//! make kernel dispatch ambiguous; a private contrib domain is the only
//! unambiguous key. `com.microsoft` is the established ONNX-ecosystem contrib
//! domain (where the `FusedMatMul`/`LayerNormalization` contrib variants live),
//! so our IR stays interoperable with ORT-exported models and wider tooling.
//!
//! Kernel dispatch (`onnx-runtime-ep-cpu`) binds these by `(domain, op_type)`.
//! `LayerNormalization` and `FusedMatMulBias` both have CPU kernels (registered
//! under the contrib domain); `FusedGemm` (MatMul+Add+Relu) is a graph-only
//! rewrite with no kernel yet — it is emitted but unexercised by the current
//! validation target (BERT uses GELU/Erf, not Relu), so its kernel is deferred.
//!
//! ## Schema-aware rewrites
//!
//! Most patterns use a *structural* rewrite: the fused node's inputs are the
//! matched region's external inputs in first-seen order, which happens to match
//! the kernel signature for `FusedMatMulBias` (`[A, B, bias]`). The LayerNorm
//! fusion is instead **schema-aware** (see [`RewriteKind::LayerNorm`]): it emits
//! a node with inputs exactly `[X, Scale, B]` and synthesizes the `axis` /
//! `epsilon` attributes the kernel reads, extracting them from the matched
//! subgraph (the `ReduceMean` axes and the `var + eps` constant).

use std::collections::{HashMap, HashSet};

use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, ValueId, WeightRef};

use crate::error::Result;
use crate::pass::{OptimizationPass, PassContext};

/// The private contrib domain under which the optimizer emits every fused op.
///
/// `com.microsoft` is the established ONNX-ecosystem contrib domain; keeping our
/// fused ops there (rather than the reserved `""`/`ai.onnx` domain) avoids
/// colliding with standard-op opset validation, keeps kernel dispatch keyed
/// unambiguously on `(domain, op_type)`, and stays interoperable with
/// ORT-exported models. This is model-agnostic: it is a property of the op
/// *domain*, independent of any particular model.
pub const CONTRIB_DOMAIN: &str = "com.microsoft";

/// The inputs and attributes of a fused node: `(inputs, attributes)`.
type FusedNodeSpec = (Vec<Option<ValueId>>, HashMap<String, Attribute>);

/// A matched occurrence of a [`FusionPattern`] in a graph.#[derive(Clone, Debug)]
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

/// How a matched pattern is rewritten into its fused node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RewriteKind {
    /// The fused node's inputs are the matched region's external inputs in
    /// first-seen order (e.g. `MatMul(A,B)+bias` → `FusedMatMulBias[A, B, bias]`).
    Structural,
    /// Schema-aware LayerNorm rewrite: emit `[X, Scale, B]` plus the `axis` and
    /// `epsilon` attributes the kernel reads, extracted from the matched
    /// 9-op decomposition (see [`FusionPattern::layernorm_node`]).
    LayerNorm,
}

/// A fusion rule: an op-type sequence rewritten to a single replacement op.
#[derive(Clone, Debug)]
pub struct FusionPattern {
    name: String,
    ops: Vec<String>,
    replacement: String,
    kind: RewriteKind,
}

impl FusionPattern {
    /// A new *structural* pattern matching `ops` in sequence, replaced by
    /// `replacement`. The fused node's inputs are the matched region's external
    /// inputs in first-seen order.
    pub fn new(name: &str, ops: &[&str], replacement: &str) -> Self {
        assert!(!ops.is_empty(), "fusion pattern must have at least one op");
        Self {
            name: name.to_string(),
            ops: ops.iter().map(|s| s.to_string()).collect(),
            replacement: replacement.to_string(),
            kind: RewriteKind::Structural,
        }
    }

    /// The schema-aware LayerNorm pattern: the canonical 9-op decomposition
    /// (`ReduceMean, Sub, Pow, ReduceMean, Add, Sqrt, Div, Mul, Add`) rewritten
    /// to a `com.microsoft::LayerNormalization` node with inputs `[X, Scale, B]`
    /// and synthesized `axis`/`epsilon` attributes.
    pub fn layernorm() -> Self {
        Self {
            name: "LayerNorm".to_string(),
            ops: [
                "ReduceMean", "Sub", "Pow", "ReduceMean", "Add", "Sqrt", "Div", "Mul", "Add",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            replacement: "LayerNormalization".to_string(),
            kind: RewriteKind::LayerNorm,
        }
    }

    /// This pattern's rewrite kind.
    pub fn kind(&self) -> RewriteKind {
        self.kind
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

        // For schema-aware rewrites, extract the kernel-signature inputs and
        // attributes *before* the matched nodes are removed.
        let (inputs, attributes) = match self.kind {
            RewriteKind::Structural => (
                m.external_inputs.iter().map(|&v| Some(v)).collect(),
                HashMap::new(),
            ),
            RewriteKind::LayerNorm => self.layernorm_node(graph, m)?,
        };

        // Remove in reverse (last-first): a node's consumers are gone before it,
        // so intermediate values are cleanly garbage-collected. `output` itself
        // survives because it is a graph output or has an external consumer.
        for &nid in m.nodes.iter().rev() {
            graph.remove_node(nid);
        }

        if graph.try_value(output).is_none() {
            return Err(crate::error::OptimizerError::Fusion(self.name.clone()));
        }

        let mut fused = Node::new(NodeId(0), self.replacement.clone(), inputs, vec![output]);
        fused.attributes = attributes;
        // Emit the fused op in the private contrib domain so it never collides
        // with standard `ai.onnx` ops and dispatch stays keyed on (domain, op).
        fused.domain = CONTRIB_DOMAIN.to_string();
        graph.insert_node(fused);
        Ok(())
    }

    /// Extract the schema-conformant `[X, Scale, B]` inputs and the
    /// `axis`/`epsilon` attributes for a matched LayerNorm decomposition.
    ///
    /// The chain nodes are, in order:
    /// `0:ReduceMean(x) → mean`, `1:Sub(x, mean) → diff`, `2:Pow(diff, 2) → sq`,
    /// `3:ReduceMean(sq) → var`, `4:Add(var, eps) → vare`, `5:Sqrt → std`,
    /// `6:Div(diff, std) → norm`, `7:Mul(norm, Scale) → scaled`,
    /// `8:Add(scaled, B) → out`.
    ///
    /// * **X** is the `Sub` operand that is not the first `ReduceMean`'s output.
    /// * **Scale** is the `Mul` operand that is not the `Div` output.
    /// * **B** is the final `Add` operand that is not the `Mul` output.
    /// * **epsilon** is read from the constant `Add`-before-`Sqrt` operand that
    ///   is not the second `ReduceMean`'s output (folded to an initializer by the
    ///   preceding `ConstantFolding` pass); it falls back to the ONNX default
    ///   `1e-5` only if that operand is not a readable f32 constant.
    /// * **axis** is the first entry of the first `ReduceMean`'s `axes`
    ///   attribute (default `-1`).
    fn layernorm_node(
        &self,
        graph: &Graph,
        m: &PatternMatch,
    ) -> Result<FusedNodeSpec> {
        let fail = || crate::error::OptimizerError::Fusion(self.name.clone());
        let nodes = &m.nodes;
        if nodes.len() != 9 {
            return Err(fail());
        }
        let rm1 = graph.node(nodes[0]);
        let sub = graph.node(nodes[1]);
        let rm2 = graph.node(nodes[3]);
        let add_eps = graph.node(nodes[4]);
        let div = graph.node(nodes[6]);
        let mul = graph.node(nodes[7]);
        let final_add = graph.node(nodes[8]);

        let mean = rm1.outputs[0];
        let var = rm2.outputs[0];
        let norm = div.outputs[0];
        let scaled = mul.outputs[0];

        let x = sub.input_values().find(|&v| v != mean).ok_or_else(fail)?;
        let scale = mul.input_values().find(|&v| v != norm).ok_or_else(fail)?;
        let bias = final_add
            .input_values()
            .find(|&v| v != scaled)
            .ok_or_else(fail)?;

        let epsilon = add_eps
            .input_values()
            .find(|&v| v != var)
            .and_then(|eps_val| read_scalar_f32(graph, eps_val))
            .unwrap_or(1e-5);

        let axis = rm1
            .attr("axes")
            .and_then(Attribute::as_ints)
            .and_then(|a| a.first().copied())
            .unwrap_or(-1);

        let mut attributes = HashMap::new();
        attributes.insert("axis".to_string(), Attribute::Int(axis));
        attributes.insert("epsilon".to_string(), Attribute::Float(epsilon));

        Ok((vec![Some(x), Some(scale), Some(bias)], attributes))
    }
}

/// Read a scalar (or leading) f32 element from an inline float initializer, if
/// `value` is backed by one. Used to fold a constant `epsilon` into an attribute.
fn read_scalar_f32(graph: &Graph, value: ValueId) -> Option<f32> {
    match graph.initializers.get(&value)? {
        WeightRef::Inline(t) if t.dtype == DataType::Float32 && t.data.len() >= 4 => {
            Some(f32::from_le_bytes(t.data[0..4].try_into().ok()?))
        }
        _ => None,
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
        FusionPattern::layernorm(),
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
    use onnx_runtime_ir::{DataType, Node, NodeId, TensorData, static_shape};

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
        assert_eq!(fused.domain, CONTRIB_DOMAIN);
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
        let fused = g.nodes.values().next().unwrap();
        assert_eq!(fused.op_type, "FusedGemm");
        assert_eq!(fused.domain, CONTRIB_DOMAIN);
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
    ///
    /// `eps` is an inline f32 initializer (as it would be after `ConstantFolding`
    /// materializes the `var + eps` constant) so the schema-aware rewrite can
    /// fold it into the `epsilon` attribute; the `ReduceMean` nodes carry an
    /// `axes = [-1]` attribute so `axis` extraction is exercised too.
    fn layernorm_graph() -> Graph {
        const EPS: f32 = 1e-12;
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let x = val(&mut g, "x");
        let two = val(&mut g, "two");
        let eps = val(&mut g, "eps");
        let scale = val(&mut g, "scale");
        let bias = val(&mut g, "bias");
        g.add_input(x);
        g.add_input(two);
        g.set_initializer(
            eps,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![],
                EPS.to_le_bytes().to_vec(),
            )),
        );
        g.add_input(scale);
        g.add_input(bias);

        let reduce_mean = |g: &mut Graph, input: ValueId, out: ValueId| {
            let mut n = Node::new(NodeId(0), "ReduceMean", vec![Some(input)], vec![out]);
            n.attributes.insert("axes".into(), Attribute::Ints(vec![-1]));
            n.attributes.insert("keepdims".into(), Attribute::Int(1));
            g.insert_node(n);
        };

        let mean = val(&mut g, "mean");
        reduce_mean(&mut g, x, mean);
        let diff = val(&mut g, "diff");
        g.insert_node(Node::new(NodeId(0), "Sub", vec![Some(x), Some(mean)], vec![diff]));
        let sq = val(&mut g, "sq");
        g.insert_node(Node::new(NodeId(0), "Pow", vec![Some(diff), Some(two)], vec![sq]));
        let var = val(&mut g, "var");
        reduce_mean(&mut g, sq, var);
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

        // Record the value ids the schema-aware rewrite must reference.
        let vid = |name: &str| {
            g.values
                .iter()
                .find(|(_, v)| v.name.as_deref() == Some(name))
                .map(|(id, _)| id)
                .unwrap()
        };
        let x = vid("x");
        let scale = vid("scale");
        let bias = vid("bias");

        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();

        assert_eq!(g.num_nodes(), 1, "9-op chain collapses to one node");
        let fused = g.nodes.values().next().unwrap();
        assert_eq!(fused.op_type, "LayerNormalization");
        assert_eq!(fused.domain, CONTRIB_DOMAIN);
        // Schema-conformant inputs: exactly [X, Scale, B] — NOT the intermediate
        // pow-exponent / epsilon tensors.
        assert_eq!(fused.inputs, vec![Some(x), Some(scale), Some(bias)]);
        // Synthesized attributes read by the kernel.
        assert_eq!(
            fused.attr("axis").and_then(Attribute::as_int),
            Some(-1),
            "axis extracted from ReduceMean axes"
        );
        let eps = fused
            .attr("epsilon")
            .and_then(Attribute::as_float)
            .expect("epsilon attribute present");
        assert!(
            (eps - 1e-12).abs() < 1e-18,
            "epsilon extracted from the var+eps constant, got {eps}"
        );
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
        let p = FusionPattern::layernorm();
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
