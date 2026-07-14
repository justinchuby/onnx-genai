//! Operator fusion: match a connected op-sequence and replace it with a single
//! fused op (see `docs/ORT2.md` §18.2).
//!
//! ## Matching model
//!
//! A [`FusionPattern`] is an ordered op sequence plus a replacement op type.
//! **Structural** patterns (MatMul+Add, MatMul+Add+Relu) use
//! [`FusionPattern::try_match_from`], which walks the graph forward from each
//! candidate start node following producer→consumer ("spine") edges: node `i+1`
//! of the match must consume an output of node `i`. Extra data edges *back* to
//! already-matched nodes are allowed.
//!
//! The **LayerNorm** rewrite instead uses a dedicated DAG-aware matcher
//! ([`FusionPattern::try_match_layernorm`]): a real LayerNorm decomposition is a
//! diamond whose `mean` feeds both a variance branch and a numerator branch, and
//! some exporters emit two distinct `Sub(x, mean)` nodes rather than reusing one
//! `diff`, so a single linear successor-walk can't express it. The matcher
//! anchors on the `mean` `ReduceMean` and follows both branches to the final
//! `Add`, accepting both the canonical 9-op (shared `Sub`) and the 10-op
//! split-`Sub` shapes.
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
    /// 9-op decomposition (see [`FusionPattern::layernorm_spec`]).
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
    ///
    /// [`RewriteKind::LayerNorm`] uses a dedicated DAG-aware matcher
    /// ([`Self::try_match_layernorm`]) because a real LayerNorm decomposition is
    /// a diamond DAG whose `mean` feeds two branches (variance + numerator) and
    /// may even use two distinct `Sub(x, mean)` nodes; the linear successor-walk
    /// used by the structural patterns can't express that. All structural
    /// patterns (MatMul+Add, MatMul+Add+Relu) keep the linear-chain matcher.
    pub fn find_match(&self, graph: &Graph) -> Option<PatternMatch> {
        for start in graph.nodes.keys() {
            let m = match self.kind {
                RewriteKind::LayerNorm => self.try_match_layernorm(graph, start),
                RewriteKind::Structural => self.try_match_from(graph, start),
            };
            if let Some(m) = m {
                return Some(m);
            }
        }
        None
    }

    /// Whether `node` is a standard-domain op named `op`.
    fn op_matches(node: &Node, op: &str) -> bool {
        node.op_type == op && matches!(node.domain.as_str(), "" | "ai.onnx")
    }

    /// The first consumer of `value` whose op is `op` (standard domain).
    fn find_consumer(graph: &Graph, value: ValueId, op: &str) -> Option<NodeId> {
        graph
            .value(value)
            .consumers
            .iter()
            .copied()
            .find(|&c| Self::op_matches(graph.node(c), op))
    }

    /// DAG-aware LayerNorm matcher anchored on the *mean* `ReduceMean` node.
    ///
    /// Real LayerNorm decompositions are a diamond, not a chain: the mean feeds
    /// both the variance branch (`Sub → Pow → ReduceMean → Add(eps) → Sqrt`) and
    /// the numerator branch (`Sub → Div`). Some exporters (e.g. the one that
    /// produced `bert_toy`) emit **two distinct `Sub(x, mean)` nodes** — one per
    /// branch — instead of reusing a single `diff`, so the region is 10 ops and
    /// the shared `mean` value is consumed by two Subs. Both shapes are matched
    /// here; the canonical single-`Sub` diamond is the 9-op special case where
    /// the two branches share one `Sub`.
    ///
    /// The returned [`PatternMatch::nodes`] are in a fixed canonical order the
    /// schema extractor relies on:
    /// `[mean_rm, sub_pow, pow, var_rm, add_eps, sqrt, div, mul, final_add]`,
    /// with `sub_div` appended as a 10th node only when the numerator uses a
    /// distinct `Sub`. Fusion is declined (via [`Self::layernorm_spec`]) unless
    /// every schema assumption (single concrete `axis`, constant f32 `epsilon`,
    /// interior data-flow) is provable.
    fn try_match_layernorm(&self, graph: &Graph, start: NodeId) -> Option<PatternMatch> {
        let mean_rm = graph.try_node(start)?;
        if !Self::op_matches(mean_rm, "ReduceMean") || mean_rm.outputs.len() != 1 {
            return None;
        }
        let mean = mean_rm.outputs[0];

        // Every `Sub` that consumes `mean` (i.e. computes `x - mean`). One in the
        // canonical diamond, two in the split-diff variant.
        let subs: Vec<NodeId> = graph
            .value(mean)
            .consumers
            .iter()
            .copied()
            .filter(|&c| {
                let n = graph.node(c);
                Self::op_matches(n, "Sub") && n.input_values().any(|v| v == mean)
            })
            .collect();

        // Try each Sub as the *variance* diff source (feeding `Pow`).
        for &sub_pow in &subs {
            let sp = graph.node(sub_pow);
            if sp.outputs.len() != 1 {
                continue;
            }
            let diff_pow = sp.outputs[0];
            // Variance branch: Pow → ReduceMean → Add(eps) → Sqrt.
            let Some(pow) = Self::find_consumer(graph, diff_pow, "Pow") else {
                continue;
            };
            let sq = graph.node(pow).outputs[0];
            let Some(var_rm) = Self::find_consumer(graph, sq, "ReduceMean") else {
                continue;
            };
            let var = graph.node(var_rm).outputs[0];
            let Some(add_eps) = Self::find_consumer(graph, var, "Add") else {
                continue;
            };
            let vare = graph.node(add_eps).outputs[0];
            let Some(sqrt) = Self::find_consumer(graph, vare, "Sqrt") else {
                continue;
            };
            let std = graph.node(sqrt).outputs[0];
            // Numerator branch: Div(diff, std) → Mul(scale) → Add(bias).
            let Some(div) = Self::find_consumer(graph, std, "Div") else {
                continue;
            };
            let dn = graph.node(div);
            // The numerator is the Div operand that isn't `std`; it must be the
            // output of a `Sub(x, mean)` (the same or a sibling of `sub_pow`).
            let Some(num) = dn.input_values().find(|&v| v != std) else {
                continue;
            };
            let Some(&sub_div) = subs.iter().find(|&&s| graph.node(s).outputs[0] == num) else {
                continue;
            };
            let norm = dn.outputs[0];
            let Some(mul) = Self::find_consumer(graph, norm, "Mul") else {
                continue;
            };
            let scaled = graph.node(mul).outputs[0];
            let Some(final_add) = Self::find_consumer(graph, scaled, "Add") else {
                continue;
            };

            // Canonical node order (see doc). Append `sub_div` iff distinct.
            let mut nodes = vec![
                start, sub_pow, pow, var_rm, add_eps, sqrt, div, mul, final_add,
            ];
            if sub_div != sub_pow {
                nodes.push(sub_div);
            }
            let matched_set: HashSet<NodeId> = nodes.iter().copied().collect();
            // All matched nodes must be distinct (no accidental aliasing).
            if matched_set.len() != nodes.len() {
                continue;
            }

            // Safety rule: no matched node except `final_add` may have an output
            // that escapes the matched set (external consumer or graph output).
            let escapes = nodes.iter().any(|&nid| {
                nid != final_add
                    && graph.node(nid).outputs.iter().any(|&out| {
                        graph.outputs.contains(&out)
                            || graph
                                .value(out)
                                .consumers
                                .iter()
                                .any(|c| !matched_set.contains(c))
                    })
            });
            if escapes {
                continue;
            }

            // The fused node reuses `final_add`'s single output; it must survive
            // removal (graph output or an external consumer).
            let fa = graph.node(final_add);
            if fa.outputs.len() != 1 {
                continue;
            }
            let output = fa.outputs[0];
            let out_val = graph.value(output);
            let survives = graph.outputs.contains(&output)
                || out_val.consumers.iter().any(|c| !matched_set.contains(c));
            if !survives {
                continue;
            }

            // External inputs in first-seen order (X, Scale, B, plus constants).
            let produced: HashSet<ValueId> = nodes
                .iter()
                .flat_map(|&n| graph.node(n).outputs.iter().copied())
                .collect();
            let mut external = Vec::new();
            let mut seen = HashSet::new();
            for &nid in &nodes {
                for iv in graph.node(nid).input_values() {
                    if produced.contains(&iv) {
                        continue;
                    }
                    if seen.insert(iv) {
                        external.push(iv);
                    }
                }
            }

            let matched = PatternMatch {
                nodes,
                external_inputs: external,
                output,
            };

            // Decline unless every schema assumption is provable.
            if self.layernorm_spec(graph, &matched).is_none() {
                continue;
            }
            return Some(matched);
        }
        None
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

        let matched = PatternMatch {
            nodes: chain,
            external_inputs: external,
            output,
        };

        // Decline-to-fuse: never return a match whose rewrite assumptions can't
        // be *proven* from the graph. Declining here (rather than erroring later
        // in `apply_fusion`) leaves the original ops in place and lets the
        // fixpoint loop skip this occurrence instead of aborting the whole pass.
        if !self.match_is_fusable(graph, &matched) {
            return None;
        }

        Some(matched)
    }

    /// Whether a matched occurrence may be fused, or must **decline-to-fuse**
    /// because a rewrite assumption can't be proven from the graph. Model-
    /// agnostic: purely structural / shape checks, no model-specific logic.
    fn match_is_fusable(&self, graph: &Graph, m: &PatternMatch) -> bool {
        match self.kind {
            RewriteKind::LayerNorm => self.layernorm_spec(graph, m).is_some(),
            RewriteKind::Structural => {
                // Only the MatMul+Add → FusedMatMulBias rewrite needs a bias
                // broadcast guard; other structural rewrites are unconstrained.
                if self.replacement == "FusedMatMulBias" {
                    self.matmul_bias_broadcast_ok(graph, m)
                } else {
                    true
                }
            }
        }
    }

    /// Decline the `MatMul + Add → FusedMatMulBias` fusion unless the `Add`'s
    /// non-matmul (bias) operand broadcasts *into* the MatMul output shape
    /// **without expanding it** — i.e. the bias is a valid trailing broadcast of
    /// the matmul output (`[N]`, `[1, N]`, same-shape, scalar, …).
    ///
    /// A standalone `Add` broadcasts *both* operands up to their joint shape, so
    /// a bias with extra leading dims, or a batch axis where the output is
    /// extent-1, would grow the semantic result. The fused kernel and shape rule
    /// instead assume the output equals the *matmul* shape and right-align the
    /// bias, silently truncating the excess — wrong values *and* a too-small
    /// output. We therefore only fuse when every overlapping axis is provably
    /// non-expanding (identical dim, or bias extent 1). Any unknown/symbolic dim
    /// that can't be proven equal makes us decline conservatively.
    fn matmul_bias_broadcast_ok(&self, graph: &Graph, m: &PatternMatch) -> bool {
        // The matched pattern is `[MatMul, Add]`; the MatMul output is the
        // intermediate value the Add consumes, and the other Add operand is bias.
        let [matmul, add] = m.nodes.as_slice() else {
            return false;
        };
        let mm_out = graph.node(*matmul).outputs[0];
        let Some(bias) = graph.node(*add).input_values().find(|&v| v != mm_out) else {
            return false;
        };
        let mm_shape = &graph.value(mm_out).shape;
        let bias_shape = &graph.value(bias).shape;

        // More bias dims than the output → leading dims would expand the result.
        if bias_shape.len() > mm_shape.len() {
            return false;
        }
        // Right-align the bias against the output; every overlapping axis must be
        // provably non-expanding: identical extent, or bias extent 1 (which just
        // broadcasts up into the existing output dim).
        let offset = mm_shape.len() - bias_shape.len();
        for (i, &bdim) in bias_shape.iter().enumerate() {
            let mdim = mm_shape[offset + i];
            if bdim == mdim {
                continue;
            }
            if bdim.as_static() == Some(1) {
                continue;
            }
            return false;
        }
        true
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
            RewriteKind::LayerNorm => self
                .layernorm_spec(graph, m)
                .ok_or_else(|| crate::error::OptimizerError::Fusion(self.name.clone()))?,
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
    /// `axis`/`epsilon` attributes for a matched LayerNorm decomposition, or
    /// `None` if any schema-aware assumption can't be proven — in which case the
    /// pattern **declines to fuse** and the original ops are kept intact.
    ///
    /// The matched nodes are in the canonical order produced by
    /// [`Self::try_match_layernorm`]:
    /// `0:ReduceMean(x) → mean`, `1:Sub(x, mean) → diff_pow`,
    /// `2:Pow(diff_pow, 2) → sq`, `3:ReduceMean(sq) → var`,
    /// `4:Add(var, eps) → vare`, `5:Sqrt → std`, `6:Div(diff_div, std) → norm`,
    /// `7:Mul(norm, Scale) → scaled`, `8:Add(scaled, B) → out`, and an optional
    /// `9:Sub(x, mean) → diff_div` — present only when the numerator uses a
    /// **second, distinct** `Sub` (the `bert_toy`-style split-diff variant). In
    /// the canonical 9-op diamond the single `Sub` feeds both branches, so
    /// `diff_div == diff_pow`.
    ///
    /// * **X** is the (shared) `Sub` operand that is not `mean`; **Scale** the
    ///   `Mul` operand that is not the `Div` output; **B** the final `Add`
    ///   operand that is not the `Mul` output. Order-independent disambiguation.
    /// * **axis** must resolve to a *single concrete* axis read from the first
    ///   `ReduceMean`'s `axes` **attribute** (axes-as-input / multi-axis / absent
    ///   → decline; never silently assume `-1`).
    /// * **epsilon** must be readable as a concrete f32 scalar constant (else
    ///   decline; never silently assume `1e-5`).
    fn layernorm_spec(&self, graph: &Graph, m: &PatternMatch) -> Option<FusedNodeSpec> {
        let nodes = &m.nodes;
        if nodes.len() != 9 && nodes.len() != 10 {
            return None;
        }
        let rm1 = graph.node(nodes[0]);
        let sub_pow = graph.node(nodes[1]);
        let pow = graph.node(nodes[2]);
        let rm2 = graph.node(nodes[3]);
        let add_eps = graph.node(nodes[4]);
        let div = graph.node(nodes[6]);
        let mul = graph.node(nodes[7]);
        let final_add = graph.node(nodes[8]);
        // The numerator `Sub` is a distinct 10th node in the split-diff variant,
        // otherwise it is the same `Sub` that feeds the variance branch.
        let sub_div = if nodes.len() == 10 {
            graph.node(nodes[9])
        } else {
            sub_pow
        };

        let mean = rm1.outputs[0];
        let diff_pow = sub_pow.outputs[0];
        let diff_div = sub_div.outputs[0];
        let var = rm2.outputs[0];
        let norm = div.outputs[0];
        let scaled = mul.outputs[0];

        // Positive structural guard: confirm the interior data-flow really is the
        // LayerNorm decomposition, not just a coincidental op-type sequence. Each
        // consumer must actually read the interior tensor it is meant to consume.
        if !sub_pow.input_values().any(|v| v == mean)
            || !sub_div.input_values().any(|v| v == mean)
            || !pow.input_values().any(|v| v == diff_pow)
            || !div.input_values().any(|v| v == diff_div)
            || !mul.input_values().any(|v| v == norm)
            || !final_add.input_values().any(|v| v == scaled)
        {
            return None;
        }

        // Order-independent X/Scale/B disambiguation: each picks the operand that
        // is NOT the matched interior tensor. Both `Sub`s must subtract `mean`
        // from the *same* `X`.
        let x = sub_pow.input_values().find(|&v| v != mean)?;
        if !sub_div.input_values().any(|v| v == x) {
            return None;
        }
        let scale = mul.input_values().find(|&v| v != norm)?;
        let bias = final_add.input_values().find(|&v| v != scaled)?;

        // epsilon guard: must be a concrete f32 scalar constant (no 1e-5 default).
        let eps_val = add_eps.input_values().find(|&v| v != var)?;
        let epsilon = read_scalar_f32(graph, eps_val)?;

        // axis guard: a single concrete axis from the ReduceMean `axes` ATTRIBUTE.
        // Absent (axes-as-input at opset ≥ 18, or reduce-all) or multi-axis →
        // decline rather than silently defaulting to `-1`.
        let axes = rm1.attr("axes").and_then(Attribute::as_ints)?;
        let [axis] = axes else {
            return None;
        };

        let mut attributes = HashMap::new();
        attributes.insert("axis".to_string(), Attribute::Int(*axis));
        attributes.insert("epsilon".to_string(), Attribute::Float(epsilon));

        Some((vec![Some(x), Some(scale), Some(bias)], attributes))
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

    #[test]
    fn declines_layernorm_when_axes_is_input() {
        // Opset-18 style: `ReduceMean` takes `axes` as an INPUT, not an
        // attribute. The axis can't be pinned to a single concrete value from an
        // attribute, so the fusion must DECLINE and leave all 9 ops intact
        // (never silently assume axis = -1).
        let mut g = layernorm_graph();
        let mean = g
            .values
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some("mean"))
            .map(|(id, _)| id)
            .unwrap();
        let rm1 = g.value(mean).producer.unwrap();
        // Drop the `axes` attribute and feed axes in as an initializer INPUT.
        g.node_mut(rm1).attributes.remove("axes");
        let axes_in = g.create_named_value("axes_in", DataType::Int64, static_shape([1]));
        g.set_initializer(
            axes_in,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Int64,
                vec![1],
                (-1i64).to_le_bytes().to_vec(),
            )),
        );
        g.node_mut(rm1).inputs.push(Some(axes_in));
        g.value_mut(axes_in).consumers.push(rm1);
        assert!(g.validate().is_ok());

        assert_eq!(g.num_nodes(), 9);
        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
        assert_eq!(
            g.num_nodes(),
            9,
            "axes-as-input LayerNorm must NOT fuse — all original ops kept"
        );
        assert!(
            g.nodes.values().all(|n| n.op_type != "LayerNormalization"),
            "no fused LayerNormalization must be emitted"
        );
        assert_eq!(
            g.nodes.values().filter(|n| n.op_type == "ReduceMean").count(),
            2,
            "both ReduceMean ops remain"
        );
        assert!(g.validate().is_ok());
    }

    #[test]
    fn declines_layernorm_when_epsilon_not_constant() {
        // If epsilon is a runtime graph INPUT (not a folded f32 initializer) it
        // can't be read as a concrete scalar → DECLINE rather than silently
        // substituting the ONNX default 1e-5.
        let mut g = layernorm_graph();
        let eps = g
            .values
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some("eps"))
            .map(|(id, _)| id)
            .unwrap();
        // Turn the eps initializer into a plain runtime graph input.
        g.initializers.remove(&eps);
        g.add_input(eps);
        assert!(g.validate().is_ok());

        assert_eq!(g.num_nodes(), 9);
        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
        assert_eq!(
            g.num_nodes(),
            9,
            "non-constant epsilon LayerNorm must NOT fuse"
        );
        assert!(g.nodes.values().all(|n| n.op_type != "LayerNormalization"));
        assert!(g.validate().is_ok());
    }

    #[test]
    fn declines_matmul_add_when_bias_expands() {
        // MatMul output is [4]; the Add's bias is [2, 4], whose extra leading dim
        // would broadcast the result UP to [2, 4]. The fused kernel/shape rule
        // assume the output equals the matmul shape and would silently truncate,
        // so the fusion must DECLINE and keep the original MatMul + Add.
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let a = g.create_named_value("a", DataType::Float32, static_shape([4, 4]));
        let w = g.create_named_value("w", DataType::Float32, static_shape([4]));
        let bias = g.create_named_value("bias", DataType::Float32, static_shape([2, 4]));
        g.add_input(a);
        g.add_input(w);
        g.add_input(bias);
        let m = g.create_named_value("m", DataType::Float32, static_shape([4]));
        g.insert_node(Node::new(NodeId(0), "MatMul", vec![Some(a), Some(w)], vec![m]));
        let out = g.create_named_value("out", DataType::Float32, static_shape([2, 4]));
        g.insert_node(Node::new(NodeId(0), "Add", vec![Some(m), Some(bias)], vec![out]));
        g.add_output(out);

        assert_eq!(g.num_nodes(), 2);
        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
        assert_eq!(g.num_nodes(), 2, "expanding bias must NOT fuse");
        assert!(g.nodes.values().any(|n| n.op_type == "MatMul"));
        assert!(g.nodes.values().any(|n| n.op_type == "Add"));
        assert!(g.nodes.values().all(|n| n.op_type != "FusedMatMulBias"));
        assert!(g.validate().is_ok());
    }

    #[test]
    fn fuses_matmul_add_with_trailing_broadcast_bias() {
        // A `[1, 4]` bias broadcasts INTO a `[3, 4]` matmul output without
        // expanding it, so the guard must still allow this common case to fuse.
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let a = g.create_named_value("a", DataType::Float32, static_shape([3, 4]));
        let w = g.create_named_value("w", DataType::Float32, static_shape([4, 4]));
        let bias = g.create_named_value("bias", DataType::Float32, static_shape([1, 4]));
        g.add_input(a);
        g.add_input(w);
        g.add_input(bias);
        let m = g.create_named_value("m", DataType::Float32, static_shape([3, 4]));
        g.insert_node(Node::new(NodeId(0), "MatMul", vec![Some(a), Some(w)], vec![m]));
        let out = g.create_named_value("out", DataType::Float32, static_shape([3, 4]));
        g.insert_node(Node::new(NodeId(0), "Add", vec![Some(m), Some(bias)], vec![out]));
        g.add_output(out);

        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
        assert_eq!(g.num_nodes(), 1, "trailing-broadcast bias must fuse");
        assert_eq!(
            g.nodes.values().next().unwrap().op_type,
            "FusedMatMulBias"
        );
        assert!(g.validate().is_ok());
    }

    #[test]
    fn declines_matmul_add_when_shape_unknown() {
        // If the matmul output shape can't be resolved (empty/unknown), the guard
        // can't prove the bias is non-expanding → DECLINE conservatively.
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let a = g.create_named_value("a", DataType::Float32, Vec::new());
        let w = g.create_named_value("w", DataType::Float32, Vec::new());
        let bias = g.create_named_value("bias", DataType::Float32, static_shape([4]));
        g.add_input(a);
        g.add_input(w);
        g.add_input(bias);
        // `m` has an unknown (empty) shape.
        let m = g.create_named_value("m", DataType::Float32, Vec::new());
        g.insert_node(Node::new(NodeId(0), "MatMul", vec![Some(a), Some(w)], vec![m]));
        let out = g.create_named_value("out", DataType::Float32, static_shape([4]));
        g.insert_node(Node::new(NodeId(0), "Add", vec![Some(m), Some(bias)], vec![out]));
        g.add_output(out);

        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
        assert_eq!(g.num_nodes(), 2, "unknown matmul shape must NOT fuse");
        assert!(g.nodes.values().all(|n| n.op_type != "FusedMatMulBias"));
        assert!(g.validate().is_ok());
    }
}
