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
//! `LayerNormalization`, `FusedMatMulBias` and `FusedGemm` all have CPU kernels
//! (registered under the contrib domain). `FusedGemm` (MatMul+Add+Relu) is not
//! exercised by the current model-level validation target (BERT uses GELU/Erf,
//! not Relu), so it is instead validated by the synthetic end-to-end parity
//! test in `crates/onnx-runtime-session/tests/fused_gemm_parity.rs`, which
//! builds a MatMul→Add→Relu graph and checks the fused single-pass output
//! against the unfused reference.
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

use std::collections::{BTreeSet, HashMap, HashSet};

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

/// `√2`, the exact-GELU inner divisor (`Erf(X / √2)`).
const SQRT_2: f32 = std::f32::consts::SQRT_2;
/// `1/√2`, the equivalent inner *multiplier* encoding (`Mul(X, 1/√2)`).
const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Whether `a` matches an expected exact-GELU structural constant. The GELU
/// constants (`0.5`, `1.0`, `√2`, `1/√2`, `2.0`) are all small and exactly
/// representable-ish in f32; the tolerance only absorbs f32 rounding of `√2`
/// / `1/√2`, never a numerically different coefficient — an off constant
/// **declines** rather than silently fuses a wrong decomposition.
fn approx(a: f32, expected: f32) -> bool {
    (a - expected).abs() <= 1e-6 * expected.abs().max(1.0)
}

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
    /// Schema-aware SDPA rewrite: emit `[Q, K, V]` (+ optional `[mask]`) plus
    /// the concrete `scale` and `k_transposed` attributes, extracted from the
    /// matched `MatMul → (Mul|Div) → [Add] → Softmax → MatMul` core (see
    /// [`FusionPattern::attention_spec`]).
    Attention,
    /// Schema-aware exact-GELU rewrite: emit `[X]` with no attributes, extracted
    /// from the matched Erf decomposition
    /// `0.5·X · (1 + Erf(X / √2))` — a diamond whose single external input `X`
    /// feeds both the `Erf` branch and the outer half-scale (see
    /// [`FusionPattern::gelu_spec`]). Only the exact (`Erf`) form is recognized;
    /// the `tanh`-approximation FastGelu is out of scope.
    Gelu,
}

/// A fusion rule: an op-type sequence rewritten to a single replacement op.
#[derive(Clone, Debug)]
pub struct FusionPattern {
    name: String,
    ops: Vec<String>,
    replacement: String,
    #[cfg(test)]
    replacement_domain: String,
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
            #[cfg(test)]
            replacement_domain: CONTRIB_DOMAIN.to_string(),
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
            #[cfg(test)]
            replacement_domain: CONTRIB_DOMAIN.to_string(),
            kind: RewriteKind::LayerNorm,
        }
    }

    /// This pattern's rewrite kind.
    pub fn kind(&self) -> RewriteKind {
        self.kind
    }

    /// The schema-aware SDPA-core pattern, rewritten to a
    /// `com.microsoft::FusedAttention` node with inputs `[Q, K, V]` (+ optional
    /// `[mask]`) and synthesized `scale`/`k_transposed` attributes. Anchored on
    /// the `Softmax` (see [`Self::try_match_attention`]).
    pub fn attention() -> Self {
        Self {
            name: "Attention".to_string(),
            // The op list is descriptive only; the DAG-aware matcher does the
            // real recognition. Softmax is the anchor.
            ops: ["Softmax"].iter().map(|s| s.to_string()).collect(),
            replacement: "FusedAttention".to_string(),
            #[cfg(test)]
            replacement_domain: CONTRIB_DOMAIN.to_string(),
            kind: RewriteKind::Attention,
        }
    }

    /// The schema-aware exact-GELU pattern: the `Erf` decomposition
    /// `0.5·X · (1 + Erf(X / √2))` rewritten to a `com.microsoft::Gelu` node
    /// with the single input `[X]` and no attributes. Anchored on the `Erf`
    /// (see [`Self::try_match_gelu`]).
    pub fn gelu() -> Self {
        Self {
            name: "Gelu".to_string(),
            // Descriptive only; the DAG-aware matcher does the real recognition.
            // `Erf` is the anchor.
            ops: ["Erf"].iter().map(|s| s.to_string()).collect(),
            replacement: "Gelu".to_string(),
            #[cfg(test)]
            replacement_domain: CONTRIB_DOMAIN.to_string(),
            kind: RewriteKind::Gelu,
        }
    }

    #[cfg(test)]
    fn with_replacement_domain(mut self, domain: &str) -> Self {
        self.replacement_domain = domain.to_string();
        self
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
            if let Some(m) = self.try_match_at(graph, start) {
                return Some(m);
            }
        }
        None
    }

    fn try_match_at(&self, graph: &Graph, start: NodeId) -> Option<PatternMatch> {
        match self.kind {
            RewriteKind::LayerNorm => self.try_match_layernorm(graph, start),
            RewriteKind::Attention => self.try_match_attention(graph, start),
            RewriteKind::Gelu => self.try_match_gelu(graph, start),
            RewriteKind::Structural => self.try_match_from(graph, start),
        }
    }

    /// Candidate starts whose match result may be affected when `matched` is
    /// replaced. The replacement is always a contrib-domain op, so it cannot
    /// itself satisfy any standard-domain pattern step. Existing producers can
    /// still observe changed consumer adjacency, so conservatively revisit them
    /// and the bounded predecessor chains from which this pattern could reach
    /// them.
    fn affected_candidate_starts(&self, graph: &Graph, matched: &PatternMatch) -> Vec<NodeId> {
        let max_depth = match self.kind {
            RewriteKind::LayerNorm => 10,
            RewriteKind::Attention => 6,
            RewriteKind::Gelu => 5,
            RewriteKind::Structural => self.ops.len(),
        };
        let mut affected = HashSet::new();
        let mut frontier: Vec<(NodeId, usize)> = matched
            .external_inputs
            .iter()
            .filter_map(|&value| graph.value(value).producer)
            .map(|producer| (producer, 0))
            .collect();

        while let Some((node_id, depth)) = frontier.pop() {
            if !affected.insert(node_id) || depth >= max_depth.saturating_sub(1) {
                continue;
            }
            frontier.extend(
                graph
                    .node(node_id)
                    .input_values()
                    .filter_map(|value| graph.value(value).producer)
                    .map(|producer| (producer, depth + 1)),
            );
        }
        affected.into_iter().collect()
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

    /// DAG-aware SDPA-core matcher anchored on the `Softmax`.
    ///
    /// Recognizes the scaled-dot-product-attention core
    /// `MatMul(Q, Kside) → (Mul|Div by scalar) → [Add(mask)] → Softmax(axis=-1)
    /// → MatMul(probs, V)` and rewrites it to a single
    /// `com.microsoft::FusedAttention[Q, K, V, (mask)]`. All recognition and
    /// every decline guard live in [`Self::try_parse_attention`]; this wrapper
    /// just packages the parsed pieces into a [`PatternMatch`].
    fn try_match_attention(&self, graph: &Graph, start: NodeId) -> Option<PatternMatch> {
        let p = self.try_parse_attention(graph, start)?;
        Some(PatternMatch {
            nodes: p.nodes,
            external_inputs: p.external_inputs,
            output: p.output,
        })
    }

    /// DAG-aware exact-GELU matcher anchored on the `Erf` node. Packages the
    /// parsed pieces from [`Self::try_parse_gelu`] into a [`PatternMatch`].
    fn try_match_gelu(&self, graph: &Graph, start: NodeId) -> Option<PatternMatch> {
        let p = self.try_parse_gelu(graph, start)?;
        Some(PatternMatch {
            nodes: p.nodes,
            external_inputs: p.external_inputs,
            output: p.output,
        })
    }

    /// Parse (and fully validate) the SDPA core anchored on the `Softmax` node
    /// `sm_start`, or `None` to **decline-to-fuse** when any structural or
    /// numeric assumption cannot be proven from the graph. Model-agnostic:
    /// purely structural / constant checks, no model-specific names.
    ///
    /// Decline guards (each returns `None`):
    /// * anchor is not a single-in/single-out `Softmax`, or its `axis` is not
    ///   provably the **last** axis (absent axis or non-last → decline; never
    ///   guess the opset default);
    /// * the softmax output is not the **left** operand of a following `MatMul`
    ///   (the `probs · V` product);
    /// * the score scaling is not a `Mul`/`Div` by a **concrete scalar f32
    ///   constant** whose other operand is a `MatMul` output;
    /// * an intervening `Add` (mask) whose scaled-scores branch can't be
    ///   uniquely identified (both or neither operand parse as the score
    ///   scaling);
    /// * any interior value escapes the matched region (consumed outside it or
    ///   is a graph output), or the matched nodes are not all distinct, or the
    ///   fused output would not survive removal.
    fn try_parse_attention(&self, graph: &Graph, sm_start: NodeId) -> Option<AttnParts> {
        // Anchor: a Softmax normalizing over its LAST axis.
        let sm = graph.try_node(sm_start)?;
        if !Self::op_matches(sm, "Softmax") || sm.inputs.len() != 1 || sm.outputs.len() != 1 {
            return None;
        }
        let sm_in = sm.inputs[0]?;
        let sm_out = sm.outputs[0];
        let rank = graph.value(sm_in).shape.len();
        if rank == 0 {
            return None;
        }
        // Require an explicit `axis` that resolves to the last dim. An absent
        // axis is the opset default (1 for ≤12, -1 for ≥13) — not provably the
        // last axis on a >2-D tensor — so we decline rather than guess.
        let axis = sm.attr("axis").and_then(Attribute::as_int)?;
        let axis = if axis < 0 { axis + rank as i64 } else { axis };
        if axis != rank as i64 - 1 {
            return None;
        }

        // Forward: out = probs · V. `sm_out` must be the LEFT operand of a
        // following MatMul (matmul is not commutative; a right-operand softmax
        // would be `V · probs`, a different op → decline).
        let out_mm = graph
            .value(sm_out)
            .consumers
            .iter()
            .copied()
            .find(|&c| {
                let n = graph.node(c);
                Self::op_matches(n, "MatMul") && n.inputs.first() == Some(&Some(sm_out))
            })?;
        let out_mm_node = graph.node(out_mm);
        if out_mm_node.inputs.len() != 2 || out_mm_node.outputs.len() != 1 {
            return None;
        }
        let v = out_mm_node.inputs[1]?;
        let output = out_mm_node.outputs[0];

        // Backward: the Softmax input is produced either directly by the score
        // scaling, or by a mask `Add` sitting between the scaling and Softmax.
        let sm_in_prod = graph.value(sm_in).producer?;
        let prod = graph.node(sm_in_prod);
        let (scale_out, mask, mask_add) =
            if Self::op_matches(prod, "Add") && prod.inputs.len() == 2 {
                let a = prod.inputs[0]?;
                let b = prod.inputs[1]?;
                // The scaled-scores operand is the one whose producer parses as
                // the score scaling (`Mul`/`Div` scalar of a MatMul output);
                // the other operand is the additive mask. Exactly one must
                // qualify — otherwise the dataflow is ambiguous → decline.
                let a_scale = graph
                    .value(a)
                    .producer
                    .is_some_and(|p| Self::parse_scale(graph, p).is_some());
                let b_scale = graph
                    .value(b)
                    .producer
                    .is_some_and(|p| Self::parse_scale(graph, p).is_some());
                match (a_scale, b_scale) {
                    (true, false) => (a, Some(b), Some(sm_in_prod)),
                    (false, true) => (b, Some(a), Some(sm_in_prod)),
                    _ => return None,
                }
            } else {
                (sm_in, None, None)
            };

        // Score scaling: `scores * c` (Mul) or `scores / c` (Div), c a concrete
        // scalar f32 constant, `scores` a MatMul output.
        let scale_node_id = graph.value(scale_out).producer?;
        let scale_node = graph.node(scale_node_id);
        if scale_node.outputs.len() != 1 || scale_node.outputs[0] != scale_out {
            return None;
        }
        let (scores_out, scale) = Self::parse_scale(graph, scale_node_id)?;

        // Score MatMul: scores = Q · Kside. `parse_scale` already proved the
        // producer is a MatMul; re-fetch it and read its operands.
        let score_mm_id = graph.value(scores_out).producer?;
        let score_mm = graph.node(score_mm_id);
        if !Self::op_matches(score_mm, "MatMul")
            || score_mm.inputs.len() != 2
            || score_mm.outputs.len() != 1
            || score_mm.outputs[0] != scores_out
        {
            return None;
        }
        let q = score_mm.inputs[0]?;
        let k_side = score_mm.inputs[1]?;

        // K handling: optionally absorb a clean single-consumer last-two-axis
        // `Transpose` that produced Kᵀ; otherwise pass Kside through as an
        // already-transposed K.
        let (k, k_transposed, transpose_node) = Self::attention_k(graph, k_side, score_mm_id);

        // Matched nodes, canonical order (anchor first): the four core ops then
        // the optional mask `Add` and optional absorbed `Transpose`.
        let mut nodes = vec![sm_start, score_mm_id, scale_node_id, out_mm];
        if let Some(ma) = mask_add {
            nodes.push(ma);
        }
        if let Some(t) = transpose_node {
            nodes.push(t);
        }
        let matched_set: HashSet<NodeId> = nodes.iter().copied().collect();
        if matched_set.len() != nodes.len() {
            return None;
        }

        // Safety rule: every matched node except `out_mm` must have all outputs
        // consumed solely within the matched set (no external consumer, no
        // graph output) — fusion must not delete a value observed elsewhere.
        let escapes = nodes.iter().any(|&nid| {
            nid != out_mm
                && graph.node(nid).outputs.iter().any(|&o| {
                    graph.outputs.contains(&o)
                        || graph
                            .value(o)
                            .consumers
                            .iter()
                            .any(|c| !matched_set.contains(c))
                })
        });
        if escapes {
            return None;
        }

        // The fused output (out_mm's single output) must survive removal.
        let out_val = graph.value(output);
        let survives = graph.outputs.contains(&output)
            || out_val.consumers.iter().any(|c| !matched_set.contains(c));
        if !survives {
            return None;
        }

        // Schema-order external inputs: [Q, K, V] (+ mask).
        let mut external = vec![q, k, v];
        if let Some(m) = mask {
            external.push(m);
        }

        Some(AttnParts {
            nodes,
            q,
            k,
            v,
            mask,
            scale,
            k_transposed,
            output,
            external_inputs: external,
        })
    }

    /// Parse a score-scaling node into `(scores_value, scale_multiplier)`, or
    /// `None` if it is not a `Mul`/`Div` by a **concrete scalar f32 constant**
    /// whose other operand is produced by a `MatMul`. `Div(scores, c)` yields
    /// `1/c` (declining `c == 0`); `Mul` yields `c`. The scores-must-be-a-MatMul
    /// check is what disambiguates the scaled branch from the mask branch (a
    /// mask precompute is often itself a `Mul`, but not of a MatMul output).
    fn parse_scale(graph: &Graph, node_id: NodeId) -> Option<(ValueId, f32)> {
        let n = graph.node(node_id);
        if n.inputs.len() != 2 || n.outputs.len() != 1 {
            return None;
        }
        let (scores_out, scale) = if Self::op_matches(n, "Div") {
            let num = n.inputs[0]?;
            let den = n.inputs[1]?;
            let c = read_scalar_const_f32(graph, den)?;
            if c == 0.0 {
                return None;
            }
            (num, 1.0 / c)
        } else if Self::op_matches(n, "Mul") {
            let x = n.inputs[0]?;
            let y = n.inputs[1]?;
            match (
                read_scalar_const_f32(graph, x),
                read_scalar_const_f32(graph, y),
            ) {
                (None, Some(c)) => (x, c),
                (Some(c), None) => (y, c),
                // both const (fold elsewhere) or neither const → not a scale.
                _ => return None,
            }
        } else {
            return None;
        };
        // The scaled operand must be a MatMul output (the score product).
        let prod = graph.value(scores_out).producer?;
        if !Self::op_matches(graph.node(prod), "MatMul") {
            return None;
        }
        Some((scores_out, scale))
    }

    /// Decide the fused node's `K` input and `k_transposed` flag. If `k_side`
    /// (the score MatMul's second operand) is produced by a clean last-two-axis
    /// `Transpose` consumed **only** by the score MatMul, absorb it: `K` becomes
    /// the transpose's input in `[…, seq_k, head_dim]` layout and the kernel
    /// transposes internally (`k_transposed = false`, transpose node removed).
    /// Otherwise `K = k_side` is used as-is as an already-transposed Kᵀ
    /// (`k_transposed = true`, nothing absorbed).
    fn attention_k(
        graph: &Graph,
        k_side: ValueId,
        score_mm_id: NodeId,
    ) -> (ValueId, bool, Option<NodeId>) {
        if let Some(t_id) = graph.value(k_side).producer {
            let t = graph.node(t_id);
            if Self::op_matches(t, "Transpose")
                && t.inputs.len() == 1
                && t.outputs.len() == 1
                && t.outputs[0] == k_side
                && graph.value(k_side).consumers.as_slice() == [score_mm_id]
                && let Some(perm) = t.attr("perm").and_then(Attribute::as_ints)
                && is_last2_swap_perm(perm)
                && let Some(kin) = t.inputs[0]
            {
                return (kin, false, Some(t_id));
            }
        }
        (k_side, true, None)
    }

    /// Extract the `[Q, K, V]` (+ optional `[mask]`) inputs and the
    /// `scale`/`k_transposed` attributes for a matched SDPA core, or `None` to
    /// decline. Re-parses from the anchor (`m.nodes[0]`, the Softmax) so the
    /// spec is single-sourced with the matcher, and confirms the re-parse
    /// covers exactly the same node set.
    fn attention_spec(&self, graph: &Graph, m: &PatternMatch) -> Option<FusedNodeSpec> {
        let start = *m.nodes.first()?;
        let p = self.try_parse_attention(graph, start)?;
        if p.nodes != m.nodes {
            return None;
        }
        let mut inputs: Vec<Option<ValueId>> = vec![Some(p.q), Some(p.k), Some(p.v)];
        if let Some(mask) = p.mask {
            inputs.push(Some(mask));
        }
        let mut attributes = HashMap::new();
        attributes.insert("scale".to_string(), Attribute::Float(p.scale));
        attributes.insert(
            "k_transposed".to_string(),
            Attribute::Int(if p.k_transposed { 1 } else { 0 }),
        );
        Some((inputs, attributes))
    }

    /// Parse (and fully validate) the exact-GELU `Erf` decomposition anchored on
    /// the `Erf` node `erf_start`, or `None` to **decline-to-fuse** when any
    /// structural or numeric assumption cannot be proven from the graph.
    /// Model-agnostic: purely structural / constant checks.
    ///
    /// Recognizes the diamond `out = (0.5·X) · (1 + Erf(X / √2))`, i.e.
    /// `X → Div(X, √2) → Erf → Add(·, 1) → Mul(0.5·X, ·)` where the SAME `X`
    /// also feeds `0.5·X = Mul(X, 0.5)`. The equivalent constant encodings
    /// (`Mul(X, 1/√2)` for the inner scale, `Div(X, 2)` for the half scale) are
    /// accepted too, since they are numerically identical.
    ///
    /// Decline guards (each returns `None`):
    /// * anchor is not a single-in/single-out `Erf`;
    /// * the `Erf` input is not `X / √2` (`Div(X, √2)` or `Mul(X, 1/√2)` with a
    ///   concrete scalar f32 constant);
    /// * the `Erf` output is not consumed by an `Add(erf, 1.0)` (`1.0` a
    ///   concrete scalar constant);
    /// * that `Add`'s output is not consumed by a `Mul` whose other operand is
    ///   `0.5·X` (`Mul(X, 0.5)` or `Div(X, 2.0)`);
    /// * the `0.5·X` operand's `X` is **not the same value** that feeds the
    ///   `Erf` branch (the diamond is not closed);
    /// * any interior value escapes the matched region, the matched nodes are
    ///   not all distinct, or the fused output would not survive removal.
    fn try_parse_gelu(&self, graph: &Graph, erf_start: NodeId) -> Option<GeluParts> {
        // Anchor: a single-in/single-out `Erf`.
        let erf = graph.try_node(erf_start)?;
        if !Self::op_matches(erf, "Erf") || erf.inputs.len() != 1 || erf.outputs.len() != 1 {
            return None;
        }
        let erf_in = erf.inputs[0]?;
        let erf_out = erf.outputs[0];

        // Backward: `erf_in = X / √2`, via `Div(X, √2)` or `Mul(X, 1/√2)`.
        let inner_id = graph.value(erf_in).producer?;
        let inner = graph.node(inner_id);
        if inner.outputs.first() != Some(&erf_in) {
            return None;
        }
        let x = Self::parse_scaled(graph, inner, &[("Div", SQRT_2), ("Mul", FRAC_1_SQRT_2)])?;

        // Forward: `erf_out` consumed by `Add(erf_out, 1.0)`.
        let add1_id = Self::find_consumer(graph, erf_out, "Add")?;
        let add1 = graph.node(add1_id);
        if add1.inputs.len() != 2 || add1.outputs.len() != 1 {
            return None;
        }
        let one = add1.input_values().find(|&v| v != erf_out)?;
        if !approx(read_scalar_const_f32(graph, one)?, 1.0) {
            return None;
        }
        let add1_out = add1.outputs[0];

        // Forward: `add1_out` consumed by `Mul(0.5·X, add1_out)`.
        let outer_id = Self::find_consumer(graph, add1_out, "Mul")?;
        let outer = graph.node(outer_id);
        if outer.inputs.len() != 2 || outer.outputs.len() != 1 {
            return None;
        }
        let half = outer.input_values().find(|&v| v != add1_out)?;
        let output = outer.outputs[0];

        // The half-scale operand must be `0.5·X` (`Mul(X, 0.5)` or `Div(X, 2.0)`)
        // over the SAME `X` that feeds the `Erf` branch — this closes the
        // diamond and confirms a real GELU, not a coincidental op sequence.
        let half_id = graph.value(half).producer?;
        let half_node = graph.node(half_id);
        if half_node.outputs.first() != Some(&half) {
            return None;
        }
        let x2 = Self::parse_scaled(graph, half_node, &[("Mul", 0.5), ("Div", 2.0)])?;
        if x2 != x {
            return None;
        }

        // Canonical node order (anchor first): [Erf, inner, Add, outer, half].
        let nodes = vec![erf_start, inner_id, add1_id, outer_id, half_id];
        let matched_set: HashSet<NodeId> = nodes.iter().copied().collect();
        if matched_set.len() != nodes.len() {
            return None;
        }

        // Safety rule: every matched node except the final `outer` `Mul` must
        // have all outputs consumed solely within the matched set (no external
        // consumer, no graph output).
        let escapes = nodes.iter().any(|&nid| {
            nid != outer_id
                && graph.node(nid).outputs.iter().any(|&o| {
                    graph.outputs.contains(&o)
                        || graph
                            .value(o)
                            .consumers
                            .iter()
                            .any(|c| !matched_set.contains(c))
                })
        });
        if escapes {
            return None;
        }

        // The fused output (outer's single output) must survive removal.
        let out_val = graph.value(output);
        let survives = graph.outputs.contains(&output)
            || out_val.consumers.iter().any(|c| !matched_set.contains(c));
        if !survives {
            return None;
        }

        Some(GeluParts {
            nodes,
            x,
            output,
            external_inputs: vec![x],
        })
    }

    /// If `node` computes `x · k` (`Mul`) or `x / k` (`Div`) for one of the
    /// allowed `(op_type, constant)` forms, return the data operand `x`. The
    /// constant must be a **strict scalar** f32 initializer approximately equal
    /// to the expected value. `Mul` is commutative (the constant may be either
    /// operand); `Div` is not (the constant must be the divisor). Any other
    /// shape → `None`.
    fn parse_scaled(graph: &Graph, node: &Node, forms: &[(&str, f32)]) -> Option<ValueId> {
        if node.inputs.len() != 2 || node.outputs.len() != 1 {
            return None;
        }
        let a = node.inputs[0]?;
        let b = node.inputs[1]?;
        for &(op, k) in forms {
            if !Self::op_matches(node, op) {
                continue;
            }
            // The scalar constant is valid as the second operand for both forms
            // (the `Div` divisor, or a `Mul` factor); `Mul` is commutative, so
            // it may additionally be the first operand.
            if read_scalar_const_f32(graph, b).is_some_and(|c| approx(c, k)) {
                return Some(a);
            }
            if op == "Mul" && read_scalar_const_f32(graph, a).is_some_and(|c| approx(c, k)) {
                return Some(b);
            }
        }
        None
    }

    /// Extract the schema-conformant `[X]` input (no attributes) for a matched
    /// exact-GELU decomposition, or `None` to decline. Re-parses from the anchor
    /// (`m.nodes[0]`, the `Erf`) so the spec is single-sourced with the matcher,
    /// and confirms the re-parse covers exactly the same node set.
    fn gelu_spec(&self, graph: &Graph, m: &PatternMatch) -> Option<FusedNodeSpec> {
        let start = *m.nodes.first()?;
        let p = self.try_parse_gelu(graph, start)?;
        if p.nodes != m.nodes {
            return None;
        }
        Some((vec![Some(p.x)], HashMap::new()))
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
            RewriteKind::Attention => self.attention_spec(graph, m).is_some(),
            RewriteKind::Gelu => self.gelu_spec(graph, m).is_some(),
            RewriteKind::Structural => {
                // The MatMul+Add → FusedMatMulBias and MatMul+Add+Relu →
                // FusedGemm rewrites both need a bias broadcast guard (the
                // trailing Relu is elementwise and shape-neutral); other
                // structural rewrites are unconstrained.
                if self.replacement == "FusedMatMulBias" || self.replacement == "FusedGemm" {
                    self.matmul_bias_broadcast_ok(graph, m)
                } else {
                    true
                }
            }
        }
    }

    /// Decline the `MatMul + Add → FusedMatMulBias` (and
    /// `MatMul + Add + Relu → FusedGemm`) fusion unless the `Add`'s non-matmul
    /// (bias) operand broadcasts *into* the MatMul output shape **without
    /// expanding it** — i.e. the bias is a valid trailing broadcast of the
    /// matmul output (`[N]`, `[1, N]`, same-shape, scalar, …). The optional
    /// trailing `Relu` is elementwise and shape-neutral, so the same guard
    /// applies to both fusions.
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
        // The matched pattern starts with `[MatMul, Add, ...]` (an optional
        // trailing `Relu` for FusedGemm). The MatMul output is the intermediate
        // value the Add consumes, and the other Add operand is bias.
        let (Some(&matmul), Some(&add)) = (m.nodes.first(), m.nodes.get(1)) else {
            return false;
        };
        let mm_out = graph.node(matmul).outputs[0];
        let Some(bias) = graph.node(add).input_values().find(|&v| v != mm_out) else {
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
        self.apply_fusion_returning_id(graph, m).map(|_| ())
    }

    fn apply_fusion_returning_id(&self, graph: &mut Graph, m: &PatternMatch) -> Result<NodeId> {
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
            RewriteKind::Attention => self
                .attention_spec(graph, m)
                .ok_or_else(|| crate::error::OptimizerError::Fusion(self.name.clone()))?,
            RewriteKind::Gelu => self
                .gelu_spec(graph, m)
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
        // Production patterns emit in the private contrib domain. Unit tests
        // can override it to exercise a replacement that can match again.
        #[cfg(not(test))]
        {
            fused.domain = CONTRIB_DOMAIN.to_string();
        }
        #[cfg(test)]
        {
            fused.domain = self.replacement_domain.clone();
        }
        Ok(graph.insert_node(fused))
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

        // Operand-ORDER guard: each centering `Sub` must compute `diff = x - mean`
        // (minuend `x` first, subtrahend `mean` second), NOT `mean - x`. Membership
        // alone (checked above) would accept a reversed `Sub(mean, x)` and silently
        // rewrite it to a sign-flipped LayerNormalization. `Sub` is exactly binary,
        // so require input[0] == X and input[1] == mean on BOTH the variance-branch
        // and numerator-branch Subs. Ambiguous arity (not exactly two inputs) → decline.
        let subtracts_x_minus_mean = |sub: &Node| -> bool {
            matches!(sub.inputs.as_slice(), [Some(a), Some(b)] if *a == x && *b == mean)
        };
        if !subtracts_x_minus_mean(sub_pow) || !subtracts_x_minus_mean(sub_div) {
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

/// The parsed pieces of a matched SDPA core (see
/// [`FusionPattern::try_parse_attention`]).
#[derive(Clone, Debug)]
struct AttnParts {
    /// All matched node ids, canonical order (anchor first):
    /// `[softmax, score_mm, scale_node, out_mm]` then optional `mask_add` and
    /// optional absorbed `transpose`.
    nodes: Vec<NodeId>,
    q: ValueId,
    k: ValueId,
    v: ValueId,
    mask: Option<ValueId>,
    scale: f32,
    k_transposed: bool,
    output: ValueId,
    external_inputs: Vec<ValueId>,
}

/// The parsed pieces of a matched exact-GELU decomposition (see
/// [`FusionPattern::try_parse_gelu`]).
#[derive(Clone, Debug)]
struct GeluParts {
    /// All matched node ids, canonical order (anchor first):
    /// `[erf, inner_scale, add_one, outer_mul, half_scale]`.
    nodes: Vec<NodeId>,
    /// The single external input `X` (feeds both the `Erf` branch and `0.5·X`).
    x: ValueId,
    /// The fused node's output (the outer `Mul`'s single output).
    output: ValueId,
    external_inputs: Vec<ValueId>,
}
/// `None`. Stricter than [`read_scalar_f32`]: the score scale must be a genuine
/// scalar, so a multi-element initializer (whose first element we'd otherwise
/// silently read) is declined.
fn read_scalar_const_f32(graph: &Graph, value: ValueId) -> Option<f32> {
    match graph.initializers.get(&value)? {
        WeightRef::Inline(t) if t.dtype == DataType::Float32 => {
            let numel: usize = t.dims.iter().product();
            if numel != 1 || t.data.len() < 4 {
                return None;
            }
            Some(f32::from_le_bytes(t.data[0..4].try_into().ok()?))
        }
        _ => None,
    }
}

/// Whether `perm` is a clean "swap the last two axes" permutation
/// (`[0, 1, …, r-3, r-1, r-2]`) for a rank-`perm.len()` tensor. Any other
/// permutation (including one that also moves batch/head axes) is not a plain
/// Kᵀ and is left un-absorbed.
fn is_last2_swap_perm(perm: &[i64]) -> bool {
    let r = perm.len();
    if r < 2 {
        return false;
    }
    for (i, &p) in perm.iter().enumerate().take(r - 2) {
        if p != i as i64 {
            return false;
        }
    }
    perm[r - 2] == (r - 1) as i64 && perm[r - 1] == (r - 2) as i64
}

/// The default device-independent fusion patterns.
///
/// Ordered most-specific-first so `MatMul+Add+Relu` is captured before the
/// shorter `MatMul+Add`. `Residual+LayerNorm` remains deferred to Phase 2b/3.
pub fn default_fusion_patterns() -> Vec<FusionPattern> {
    vec![
        // Attention first: the SDPA core consumes plain MatMul/Softmax nodes, so
        // recognize it before the MatMul+Add(+Relu) rewrites can claim any of
        // its MatMuls.
        FusionPattern::attention(),
        FusionPattern::new("MatMul+Bias+Relu", &["MatMul", "Add", "Relu"], "FusedGemm"),
        FusionPattern::layernorm(),
        FusionPattern::gelu(),
        FusionPattern::new("MatMul+Bias", &["MatMul", "Add"], "FusedMatMulBias"),
    ]
}

/// The op-fusion pass: applies each [`FusionPattern`] to fixpoint.
#[derive(Clone, Debug)]
pub struct OpFusion {
    patterns: Vec<FusionPattern>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScanCandidateSource {
    Initial,
    Revisit,
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

    fn run_resumable(
        &self,
        graph: &mut Graph,
        mut observe_fusion: impl FnMut(
            &str,
            ScanCandidateSource,
            NodeId,
            &[NodeId],
            &[NodeId],
            NodeId,
        ),
    ) -> Result<()> {
        for pattern in &self.patterns {
            let candidates: Vec<u32> = graph.nodes.keys().map(|id| id.0).collect();
            let mut cursor = 0;
            let mut revisits = BTreeSet::new();
            loop {
                let initial = candidates.get(cursor).copied();
                let revisit = revisits.first().copied();
                let (raw_id, source) = match (initial, revisit) {
                    (None, None) => break,
                    (Some(id), None) => {
                        cursor += 1;
                        (id, ScanCandidateSource::Initial)
                    }
                    (None, Some(_)) => (
                        revisits.pop_first().unwrap(),
                        ScanCandidateSource::Revisit,
                    ),
                    (Some(id), Some(revisit)) if id <= revisit => {
                        cursor += 1;
                        if id == revisit {
                            revisits.pop_first();
                        }
                        (id, ScanCandidateSource::Initial)
                    }
                    (Some(_), Some(_)) => (
                        revisits.pop_first().unwrap(),
                        ScanCandidateSource::Revisit,
                    ),
                };
                let start = NodeId(raw_id);
                let Some(matched) = pattern.try_match_at(graph, start) else {
                    continue;
                };

                let affected = pattern.affected_candidate_starts(graph, &matched);
                let fused_id = pattern.apply_fusion_returning_id(graph, &matched)?;
                observe_fusion(
                    pattern.pattern_name(),
                    source,
                    start,
                    &matched.nodes,
                    &affected,
                    fused_id,
                );

                // The ordered set is the source of truth for resolution order:
                // any lower affected start is reconsidered before an untouched
                // higher-id candidate, exactly like a restart from arena slot 0.
                revisits.insert(fused_id.0);
                for candidate in affected {
                    if graph.try_node(candidate).is_some() {
                        revisits.insert(candidate.0);
                    }
                }
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn run_with_fusion_observer(
        &self,
        graph: &mut Graph,
        observe_fusion: impl FnMut(
            &str,
            ScanCandidateSource,
            NodeId,
            &[NodeId],
            &[NodeId],
            NodeId,
        ),
    ) -> Result<()> {
        self.run_resumable(graph, observe_fusion)
    }
}

impl OptimizationPass for OpFusion {
    fn name(&self) -> &str {
        "OpFusion"
    }

    fn run(&self, graph: &mut Graph, _ctx: &PassContext) -> Result<()> {
        self.run_resumable(graph, |_, _, _, _, _, _| {})
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

    /// Build the 10-op split-diff LayerNorm decomposition over `x` (the
    /// `bert_toy`-style variant): the variance branch and the numerator branch
    /// each get their **own** distinct `Sub` node instead of sharing one `diff`.
    /// `mean` therefore fans out to two Subs and `x` to two Subs. When
    /// `reverse_num_sub` is true the numerator `Sub` is emitted reversed as
    /// `Sub(mean, x)` (an adversarial sign-flip) to exercise the operand-order
    /// guard.
    fn layernorm_split_graph(reverse_num_sub: bool) -> Graph {
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
        // Variance-branch Sub: always the canonical `x - mean`.
        let diff_pow = val(&mut g, "diff_pow");
        g.insert_node(Node::new(
            NodeId(0),
            "Sub",
            vec![Some(x), Some(mean)],
            vec![diff_pow],
        ));
        // Numerator-branch Sub: a SECOND, distinct node. Reversed operands when
        // `reverse_num_sub` (adversarial `mean - x`), else canonical `x - mean`.
        let diff_div = val(&mut g, "diff_div");
        let num_inputs = if reverse_num_sub {
            vec![Some(mean), Some(x)]
        } else {
            vec![Some(x), Some(mean)]
        };
        g.insert_node(Node::new(NodeId(0), "Sub", num_inputs, vec![diff_div]));

        let sq = val(&mut g, "sq");
        g.insert_node(Node::new(
            NodeId(0),
            "Pow",
            vec![Some(diff_pow), Some(two)],
            vec![sq],
        ));
        let var = val(&mut g, "var");
        reduce_mean(&mut g, sq, var);
        let vare = val(&mut g, "vare");
        g.insert_node(Node::new(NodeId(0), "Add", vec![Some(var), Some(eps)], vec![vare]));
        let std = val(&mut g, "std");
        g.insert_node(Node::new(NodeId(0), "Sqrt", vec![Some(vare)], vec![std]));
        let norm = val(&mut g, "norm");
        g.insert_node(Node::new(
            NodeId(0),
            "Div",
            vec![Some(diff_div), Some(std)],
            vec![norm],
        ));
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
    fn fuses_layernorm_split_chain() {
        // Isolated optimizer-layer coverage for the 10-op split-diff shape
        // (previously only exercised end-to-end via the bert_toy model).
        let mut g = layernorm_split_graph(false);
        assert_eq!(g.num_nodes(), 10, "split-diff shape has two distinct Subs");
        assert!(g.validate().is_ok());

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

        assert_eq!(g.num_nodes(), 1, "10-op split chain collapses to one node");
        let fused = g.nodes.values().next().unwrap();
        assert_eq!(fused.op_type, "LayerNormalization");
        assert_eq!(fused.domain, CONTRIB_DOMAIN);
        // Schema-conformant inputs: exactly [X, Scale, B].
        assert_eq!(fused.inputs, vec![Some(x), Some(scale), Some(bias)]);
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
    fn declines_layernorm_when_numerator_sub_reversed() {
        // A-CHEW-1 adversarial: the numerator diamond centers with a REVERSED
        // `Sub(mean, x)` = -(x - mean). Membership of {x, mean} still holds, but
        // the operand-order guard must DECLINE (else the rewrite silently produces
        // a sign-flipped LayerNormalization). Ops must be left untouched.
        let mut g = layernorm_split_graph(true);
        assert_eq!(g.num_nodes(), 10);
        assert!(g.validate().is_ok());

        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();

        assert!(
            g.nodes.values().all(|n| n.op_type != "LayerNormalization"),
            "reversed Sub(mean, x) must NOT fuse — sign-flip over-match"
        );
        assert_eq!(g.num_nodes(), 10, "all 10 ops remain (declined)");
        assert_eq!(
            g.nodes.values().filter(|n| n.op_type == "Sub").count(),
            2,
            "both centering Subs preserved"
        );
        assert!(g.validate().is_ok());
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

    #[test]
    fn declines_fused_gemm_when_bias_expands() {
        // Roy's FusedGemm review advisory, locked in: a MatMul+Add+Relu whose
        // bias EXPANDS the matmul output (extra leading/batch dim) must DECLINE
        // to FusedGemm exactly like the FusedMatMulBias case — the trailing Relu
        // is shape-neutral, so the same non-expanding-bias guard applies. MatMul
        // output is [4]; bias [2, 4] would broadcast the result up to [2, 4].
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
        let biased = g.create_named_value("biased", DataType::Float32, static_shape([2, 4]));
        g.insert_node(Node::new(NodeId(0), "Add", vec![Some(m), Some(bias)], vec![biased]));
        let out = g.create_named_value("out", DataType::Float32, static_shape([2, 4]));
        g.insert_node(Node::new(NodeId(0), "Relu", vec![Some(biased)], vec![out]));
        g.add_output(out);

        assert_eq!(g.num_nodes(), 3);
        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
        assert_eq!(g.num_nodes(), 3, "expanding bias must NOT fuse to FusedGemm");
        assert!(g.nodes.values().any(|n| n.op_type == "MatMul"));
        assert!(g.nodes.values().any(|n| n.op_type == "Add"));
        assert!(g.nodes.values().any(|n| n.op_type == "Relu"));
        assert!(g.nodes.values().all(|n| n.op_type != "FusedGemm"));
        assert!(g.validate().is_ok());
    }

    // --- AttentionFusion (SDPA core) ------------------------------------------

    /// Add a strict-scalar f32 initializer, returning its value id.
    fn scalar_init(g: &mut Graph, name: &str, v: f32) -> ValueId {
        let vid = g.create_named_value(name, DataType::Float32, Vec::new());
        g.set_initializer(
            vid,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![],
                v.to_le_bytes().to_vec(),
            )),
        );
        vid
    }

    fn fval(g: &mut Graph, name: &str, dims: &[usize]) -> ValueId {
        g.create_named_value(name, DataType::Float32, static_shape(dims.iter().copied()))
    }

    /// Look up a value id by name (test convenience).
    fn value_id(g: &Graph, name: &str) -> ValueId {
        g.values
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some(name))
            .map(|(id, _)| id)
            .unwrap_or_else(|| panic!("no value named {name}"))
    }

    /// Build an SDPA core graph `Softmax((Q·Kᵀ)/c [+ mask], axis=-1) · V` with
    /// rank-4 `[1, 2, seq, dim]` tensors, K supplied pre-transposed as
    /// `[1, 2, d, sk]` (so `k_transposed` should resolve to 1). `masked` adds an
    /// additive mask; `axis` is the Softmax reduction axis.
    fn sdpa_graph(masked: bool, axis: i64) -> Graph {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 12);
        let q = fval(&mut g, "Q", &[1, 2, 3, 4]);
        let kt = fval(&mut g, "K", &[1, 2, 4, 3]); // pre-transposed [d=4, sk=3]
        let v = fval(&mut g, "V", &[1, 2, 3, 4]);
        g.add_input(q);
        g.add_input(kt);
        g.add_input(v);
        let c = scalar_init(&mut g, "scale_c", 2.0);

        let scores = fval(&mut g, "scores", &[1, 2, 3, 3]);
        g.insert_node(Node::new(NodeId(0), "MatMul", vec![Some(q), Some(kt)], vec![scores]));
        let scaled = fval(&mut g, "scaled", &[1, 2, 3, 3]);
        g.insert_node(Node::new(NodeId(0), "Div", vec![Some(scores), Some(c)], vec![scaled]));

        let sm_in = if masked {
            let mask = fval(&mut g, "mask", &[1, 1, 3, 3]);
            g.add_input(mask);
            let masked_v = fval(&mut g, "masked", &[1, 2, 3, 3]);
            g.insert_node(Node::new(
                NodeId(0),
                "Add",
                vec![Some(scaled), Some(mask)],
                vec![masked_v],
            ));
            masked_v
        } else {
            scaled
        };

        let probs = fval(&mut g, "probs", &[1, 2, 3, 3]);
        let mut sm = Node::new(NodeId(0), "Softmax", vec![Some(sm_in)], vec![probs]);
        sm.attributes.insert("axis".into(), Attribute::Int(axis));
        g.insert_node(sm);
        let out = fval(&mut g, "out", &[1, 2, 3, 4]);
        g.insert_node(Node::new(NodeId(0), "MatMul", vec![Some(probs), Some(v)], vec![out]));
        g.add_output(out);
        g
    }

    fn fused_attention_node(g: &Graph) -> Option<&Node> {
        g.nodes.values().find(|n| n.op_type == "FusedAttention")
    }

    #[test]
    fn fuses_sdpa_unmasked_pretransposed_k() {
        let mut g = sdpa_graph(false, 3);
        let q = value_id(&g, "Q");
        let k = value_id(&g, "K");
        let v = value_id(&g, "V");
        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();

        // Exactly one FusedAttention; no surviving Softmax/MatMul/Div.
        assert_eq!(
            g.nodes.values().filter(|n| n.op_type == "FusedAttention").count(),
            1
        );
        assert!(g.nodes.values().all(|n| n.op_type != "Softmax"));
        assert!(g.nodes.values().all(|n| n.op_type != "MatMul"));
        assert!(g.nodes.values().all(|n| n.op_type != "Div"));

        let fa = fused_attention_node(&g).unwrap();
        assert_eq!(fa.domain, CONTRIB_DOMAIN);
        assert_eq!(fa.inputs, vec![Some(q), Some(k), Some(v)]);
        // scale = 1/c = 1/2 = 0.5; K used as-is → k_transposed = 1.
        assert_eq!(fa.attr("scale").and_then(Attribute::as_float), Some(0.5));
        assert_eq!(fa.attr("k_transposed").and_then(Attribute::as_int), Some(1));
        assert!(g.validate().is_ok());
    }

    #[test]
    fn fuses_sdpa_masked() {
        let mut g = sdpa_graph(true, 3);
        let (q, k, v, mask) = (
            value_id(&g, "Q"),
            value_id(&g, "K"),
            value_id(&g, "V"),
            value_id(&g, "mask"),
        );
        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();

        assert_eq!(
            g.nodes.values().filter(|n| n.op_type == "FusedAttention").count(),
            1
        );
        assert!(g.nodes.values().all(|n| n.op_type != "Softmax"));
        assert!(g.nodes.values().all(|n| n.op_type != "Add"));
        let fa = fused_attention_node(&g).unwrap();
        // Mask appended as the 4th input.
        assert_eq!(fa.inputs, vec![Some(q), Some(k), Some(v), Some(mask)]);
        assert_eq!(fa.attr("k_transposed").and_then(Attribute::as_int), Some(1));
        assert!(g.validate().is_ok());
    }

    #[test]
    fn fuses_sdpa_absorbing_clean_transpose_sets_k_transposed_0() {
        // K is supplied in natural [1,2,3,4] layout and transposed to Kᵀ by a
        // clean last-two-axis Transpose (perm [0,1,3,2]) consumed only by the
        // score MatMul. The matcher absorbs it: K input becomes the natural K
        // and k_transposed = 0 (kernel transposes internally).
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 12);
        let q = fval(&mut g, "Q", &[1, 2, 3, 4]);
        let k = fval(&mut g, "K", &[1, 2, 3, 4]); // natural [sk=3, d=4]
        let v = fval(&mut g, "V", &[1, 2, 3, 4]);
        g.add_input(q);
        g.add_input(k);
        g.add_input(v);
        let c = scalar_init(&mut g, "scale_c", 4.0);

        let kt = fval(&mut g, "Kt", &[1, 2, 4, 3]);
        let mut tr = Node::new(NodeId(0), "Transpose", vec![Some(k)], vec![kt]);
        tr.attributes.insert("perm".into(), Attribute::Ints(vec![0, 1, 3, 2]));
        g.insert_node(tr);
        let scores = fval(&mut g, "scores", &[1, 2, 3, 3]);
        g.insert_node(Node::new(NodeId(0), "MatMul", vec![Some(q), Some(kt)], vec![scores]));
        let scaled = fval(&mut g, "scaled", &[1, 2, 3, 3]);
        g.insert_node(Node::new(NodeId(0), "Div", vec![Some(scores), Some(c)], vec![scaled]));
        let probs = fval(&mut g, "probs", &[1, 2, 3, 3]);
        let mut sm = Node::new(NodeId(0), "Softmax", vec![Some(scaled)], vec![probs]);
        sm.attributes.insert("axis".into(), Attribute::Int(-1));
        g.insert_node(sm);
        let out = fval(&mut g, "out", &[1, 2, 3, 4]);
        g.insert_node(Node::new(NodeId(0), "MatMul", vec![Some(probs), Some(v)], vec![out]));
        g.add_output(out);

        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
        assert!(g.nodes.values().all(|n| n.op_type != "Transpose"), "clean Kᵀ Transpose absorbed");
        let fa = fused_attention_node(&g).unwrap();
        assert_eq!(fa.inputs, vec![Some(q), Some(k), Some(v)], "K input is the natural (un-transposed) K");
        assert_eq!(fa.attr("k_transposed").and_then(Attribute::as_int), Some(0));
        // scale = 1/4 = 0.25.
        assert_eq!(fa.attr("scale").and_then(Attribute::as_float), Some(0.25));
        assert!(g.validate().is_ok());
    }

    #[test]
    fn declines_sdpa_when_softmax_axis_not_last() {
        // axis 1 on a rank-4 score tensor is not the last axis → decline.
        let mut g = sdpa_graph(false, 1);
        let before = g.num_nodes();
        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
        assert!(g.nodes.values().all(|n| n.op_type != "FusedAttention"));
        assert!(g.nodes.values().any(|n| n.op_type == "Softmax"));
        assert_eq!(g.num_nodes(), before, "no fusion when axis is not last");
    }

    #[test]
    fn declines_sdpa_when_scale_is_not_scalar_constant() {
        // The score-scaling divisor is a runtime graph input (not a constant),
        // so the scale can't be folded to a concrete f32 → decline.
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 12);
        let q = fval(&mut g, "Q", &[1, 2, 3, 4]);
        let kt = fval(&mut g, "K", &[1, 2, 4, 3]);
        let v = fval(&mut g, "V", &[1, 2, 3, 4]);
        let c = fval(&mut g, "scale_c", &[]); // runtime input, NOT an initializer
        g.add_input(q);
        g.add_input(kt);
        g.add_input(v);
        g.add_input(c);
        let scores = fval(&mut g, "scores", &[1, 2, 3, 3]);
        g.insert_node(Node::new(NodeId(0), "MatMul", vec![Some(q), Some(kt)], vec![scores]));
        let scaled = fval(&mut g, "scaled", &[1, 2, 3, 3]);
        g.insert_node(Node::new(NodeId(0), "Div", vec![Some(scores), Some(c)], vec![scaled]));
        let probs = fval(&mut g, "probs", &[1, 2, 3, 3]);
        let mut sm = Node::new(NodeId(0), "Softmax", vec![Some(scaled)], vec![probs]);
        sm.attributes.insert("axis".into(), Attribute::Int(3));
        g.insert_node(sm);
        let out = fval(&mut g, "out", &[1, 2, 3, 4]);
        g.insert_node(Node::new(NodeId(0), "MatMul", vec![Some(probs), Some(v)], vec![out]));
        g.add_output(out);

        let before = g.num_nodes();
        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
        assert!(g.nodes.values().all(|n| n.op_type != "FusedAttention"));
        assert_eq!(g.num_nodes(), before, "non-constant scale must NOT fuse");
    }

    #[test]
    fn declines_sdpa_when_softmax_is_right_operand_of_output_matmul() {
        // out = V · probs (softmax output is the RIGHT operand) is not `probs·V`
        // SDPA — the matcher requires the softmax output be the LEFT operand.
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 12);
        let q = fval(&mut g, "Q", &[1, 2, 3, 4]);
        let kt = fval(&mut g, "K", &[1, 2, 4, 3]);
        let v = fval(&mut g, "V", &[1, 2, 3, 3]);
        g.add_input(q);
        g.add_input(kt);
        g.add_input(v);
        let c = scalar_init(&mut g, "scale_c", 2.0);
        let scores = fval(&mut g, "scores", &[1, 2, 3, 3]);
        g.insert_node(Node::new(NodeId(0), "MatMul", vec![Some(q), Some(kt)], vec![scores]));
        let scaled = fval(&mut g, "scaled", &[1, 2, 3, 3]);
        g.insert_node(Node::new(NodeId(0), "Div", vec![Some(scores), Some(c)], vec![scaled]));
        let probs = fval(&mut g, "probs", &[1, 2, 3, 3]);
        let mut sm = Node::new(NodeId(0), "Softmax", vec![Some(scaled)], vec![probs]);
        sm.attributes.insert("axis".into(), Attribute::Int(3));
        g.insert_node(sm);
        let out = fval(&mut g, "out", &[1, 2, 3, 3]);
        // Reversed operand order: V · probs.
        g.insert_node(Node::new(NodeId(0), "MatMul", vec![Some(v), Some(probs)], vec![out]));
        g.add_output(out);

        let before = g.num_nodes();
        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
        assert!(g.nodes.values().all(|n| n.op_type != "FusedAttention"));
        assert!(g.nodes.values().any(|n| n.op_type == "Softmax"));
        assert_eq!(g.num_nodes(), before);
    }

    /// Build the exact-GELU `Erf` decomposition `0.5·x·(1 + erf(x / √2))` over a
    /// single graph input `x`, with the constants materialized as scalar
    /// initializers. `inner`/`half` select the constant encoding to emit so the
    /// equivalent forms can be exercised.
    fn gelu_graph(inner_div_sqrt2: bool, half_mul: bool) -> Graph {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let x = val(&mut g, "x");
        g.add_input(x);

        // half = 0.5 * x  (via Mul(x, 0.5) or Div(x, 2.0)).
        let half = val(&mut g, "half");
        if half_mul {
            let c = scalar_init(&mut g, "c_half", 0.5);
            g.insert_node(Node::new(NodeId(0), "Mul", vec![Some(x), Some(c)], vec![half]));
        } else {
            let c = scalar_init(&mut g, "c_two", 2.0);
            g.insert_node(Node::new(NodeId(0), "Div", vec![Some(x), Some(c)], vec![half]));
        }

        // scaled = x / √2  (via Div(x, √2) or Mul(x, 1/√2)).
        let scaled = val(&mut g, "scaled");
        if inner_div_sqrt2 {
            let c = scalar_init(&mut g, "c_sqrt2", std::f32::consts::SQRT_2);
            g.insert_node(Node::new(NodeId(0), "Div", vec![Some(x), Some(c)], vec![scaled]));
        } else {
            let c = scalar_init(&mut g, "c_isqrt2", std::f32::consts::FRAC_1_SQRT_2);
            g.insert_node(Node::new(NodeId(0), "Mul", vec![Some(x), Some(c)], vec![scaled]));
        }

        let e = val(&mut g, "e");
        g.insert_node(Node::new(NodeId(0), "Erf", vec![Some(scaled)], vec![e]));
        let one = scalar_init(&mut g, "c_one", 1.0);
        let a = val(&mut g, "a");
        g.insert_node(Node::new(NodeId(0), "Add", vec![Some(e), Some(one)], vec![a]));
        let out = val(&mut g, "out");
        g.insert_node(Node::new(NodeId(0), "Mul", vec![Some(half), Some(a)], vec![out]));
        g.add_output(out);
        g
    }

    #[test]
    fn fuses_gelu_div_sqrt2() {
        let mut g = gelu_graph(true, true);
        assert_eq!(g.num_nodes(), 5);
        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
        let gelu: Vec<_> = g.nodes.values().filter(|n| n.op_type == "Gelu").collect();
        assert_eq!(gelu.len(), 1, "the Erf decomposition must fuse to one Gelu");
        let fused = gelu[0];
        assert_eq!(fused.domain, CONTRIB_DOMAIN);
        assert_eq!(fused.inputs.len(), 1, "Gelu takes the single input x");
        assert!(fused.attributes.is_empty(), "exact Gelu has no attributes");
        // Single input is the graph input `x`.
        let x = g.values.iter().find(|(_, v)| v.name.as_deref() == Some("x")).map(|(id, _)| id).unwrap();
        assert_eq!(fused.inputs[0], Some(x));
        assert_eq!(fused.outputs, g.outputs);
        assert!(g.nodes.values().all(|n| n.op_type != "Erf"));
        assert!(g.validate().is_ok());
    }

    #[test]
    fn fuses_gelu_mul_reciprocal_and_div_two() {
        // Equivalent encodings: inner Mul(x, 1/√2), half Div(x, 2.0).
        let mut g = gelu_graph(false, false);
        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
        assert_eq!(
            g.nodes.values().filter(|n| n.op_type == "Gelu").count(),
            1,
            "the reciprocal/half-divisor encoding must also fuse"
        );
        assert!(g.validate().is_ok());
    }

    #[test]
    fn declines_gelu_wrong_inner_constant() {
        // Div by 2.0 instead of √2 is not x/√2 → decline.
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let x = val(&mut g, "x");
        g.add_input(x);
        let half = val(&mut g, "half");
        let ch = scalar_init(&mut g, "c_half", 0.5);
        g.insert_node(Node::new(NodeId(0), "Mul", vec![Some(x), Some(ch)], vec![half]));
        let scaled = val(&mut g, "scaled");
        let cbad = scalar_init(&mut g, "c_bad", 2.0);
        g.insert_node(Node::new(NodeId(0), "Div", vec![Some(x), Some(cbad)], vec![scaled]));
        let e = val(&mut g, "e");
        g.insert_node(Node::new(NodeId(0), "Erf", vec![Some(scaled)], vec![e]));
        let one = scalar_init(&mut g, "c_one", 1.0);
        let a = val(&mut g, "a");
        g.insert_node(Node::new(NodeId(0), "Add", vec![Some(e), Some(one)], vec![a]));
        let out = val(&mut g, "out");
        g.insert_node(Node::new(NodeId(0), "Mul", vec![Some(half), Some(a)], vec![out]));
        g.add_output(out);

        let before = g.num_nodes();
        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
        assert!(g.nodes.values().all(|n| n.op_type != "Gelu"));
        assert!(g.nodes.values().any(|n| n.op_type == "Erf"));
        assert_eq!(g.num_nodes(), before);
    }

    #[test]
    fn declines_gelu_wrong_half_constant() {
        // Mul(x, 0.4) instead of 0.5 → decline.
        let mut g = gelu_graph(true, true);
        // Rewrite the half Mul's constant initializer to 0.4.
        let ch = g.values.iter().find(|(_, v)| v.name.as_deref() == Some("c_half")).map(|(id, _)| id).unwrap();
        g.set_initializer(
            ch,
            WeightRef::Inline(TensorData::from_raw(DataType::Float32, vec![], 0.4f32.to_le_bytes().to_vec())),
        );
        let before = g.num_nodes();
        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
        assert!(g.nodes.values().all(|n| n.op_type != "Gelu"));
        assert_eq!(g.num_nodes(), before);
    }

    #[test]
    fn declines_gelu_when_half_uses_different_x() {
        // The `0.5··` operand uses a DIFFERENT value than the Erf branch, so the
        // diamond is not closed → decline.
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let x = val(&mut g, "x");
        let y = val(&mut g, "y");
        g.add_input(x);
        g.add_input(y);
        let half = val(&mut g, "half");
        let ch = scalar_init(&mut g, "c_half", 0.5);
        // half = 0.5 * y   (NOT x)
        g.insert_node(Node::new(NodeId(0), "Mul", vec![Some(y), Some(ch)], vec![half]));
        let scaled = val(&mut g, "scaled");
        let cs = scalar_init(&mut g, "c_sqrt2", std::f32::consts::SQRT_2);
        g.insert_node(Node::new(NodeId(0), "Div", vec![Some(x), Some(cs)], vec![scaled]));
        let e = val(&mut g, "e");
        g.insert_node(Node::new(NodeId(0), "Erf", vec![Some(scaled)], vec![e]));
        let one = scalar_init(&mut g, "c_one", 1.0);
        let a = val(&mut g, "a");
        g.insert_node(Node::new(NodeId(0), "Add", vec![Some(e), Some(one)], vec![a]));
        let out = val(&mut g, "out");
        g.insert_node(Node::new(NodeId(0), "Mul", vec![Some(half), Some(a)], vec![out]));
        g.add_output(out);

        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
        assert!(g.nodes.values().all(|n| n.op_type != "Gelu"));
        assert!(g.nodes.values().any(|n| n.op_type == "Erf"));
    }

    #[test]
    fn declines_gelu_when_interior_escapes() {
        // The Erf output feeds an extra external consumer, so fusing would
        // delete an observed value → decline.
        let mut g = gelu_graph(true, true);
        let e = g.values.iter().find(|(_, v)| v.name.as_deref() == Some("e")).map(|(id, _)| id).unwrap();
        let side = val(&mut g, "side");
        g.insert_node(Node::new(NodeId(0), "Erf", vec![Some(e)], vec![side]));
        g.add_output(side);

        OpFusion::new().run(&mut g, &PassContext::new()).unwrap();
        assert!(
            g.nodes.values().all(|n| n.op_type != "Gelu"),
            "must not fuse when an interior value escapes"
        );
    }

    fn run_restart_reference(patterns: &[FusionPattern], graph: &mut Graph) {
        for pattern in patterns {
            while let Some(matched) = pattern.find_match(graph) {
                pattern.apply_fusion(graph, &matched).unwrap();
            }
        }
    }

    fn serialized_graph_bytes(mut graph: Graph) -> Vec<u8> {
        use std::fmt::Write;

        let mut snapshot = String::new();
        writeln!(&mut snapshot, "inputs={:?}", graph.inputs).unwrap();
        writeln!(&mut snapshot, "outputs={:?}", graph.outputs).unwrap();

        let mut initializers: Vec<_> = graph.initializers.iter().collect();
        initializers.sort_by_key(|(id, _)| id.0);
        writeln!(&mut snapshot, "initializers={initializers:?}").unwrap();
        let mut constraints: Vec<_> = graph.symbol_constraints.iter().collect();
        constraints.sort_by_key(|(id, _)| id.0);
        writeln!(&mut snapshot, "constraints={constraints:?}").unwrap();
        let mut opsets: Vec<_> = graph.opset_imports.iter().collect();
        opsets.sort_by_key(|(domain, _)| *domain);
        writeln!(&mut snapshot, "opsets={opsets:?}").unwrap();
        let mut subgraphs: Vec<_> = graph.subgraphs.iter().collect();
        subgraphs.sort_by_key(|((id, name), _)| (id.0, name.as_str()));
        writeln!(&mut snapshot, "subgraphs={subgraphs:?}").unwrap();

        for (id, node) in graph.nodes.iter() {
            let mut attributes: Vec<_> = node.attributes.iter().collect();
            attributes.sort_by_key(|(name, _)| *name);
            writeln!(
                &mut snapshot,
                "node={id:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{attributes:?}|{:?}|{:?}|{:?}",
                node.id,
                node.name,
                node.op_type,
                node.domain,
                node.inputs,
                node.outputs,
                node.doc_string,
                node.device,
                node.exec_order,
            )
            .unwrap();
        }
        for (id, value) in graph.values.iter() {
            writeln!(&mut snapshot, "value={id:?}|{value:?}").unwrap();
        }
        writeln!(
            &mut snapshot,
            "topological_order={:?}",
            graph.topological_order().unwrap()
        )
        .unwrap();

        // Arena slots/free-list order are private IR details, but their complete
        // observable state is the sequence of IDs returned by future inserts.
        // The generated graphs are far smaller than this probe count.
        for _ in 0..128 {
            let node =
                graph.insert_node(Node::new(NodeId(0), "ArenaProbe", Vec::new(), Vec::new()));
            let value = graph.create_value(DataType::Float32, static_shape([1]));
            writeln!(&mut snapshot, "probe={node:?}|{value:?}").unwrap();
        }
        snapshot.into_bytes()
    }

    fn assert_fusion_graphs_byte_identical(actual: Graph, expected: Graph, trial: usize) {
        assert_eq!(
            serialized_graph_bytes(actual),
            serialized_graph_bytes(expected),
            "restart and resumable fixpoints differ byte-for-byte on trial {trial}"
        );
    }

    struct FusionTestRng(u64);

    impl FusionTestRng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn usize(&mut self, upper: usize) -> usize {
            (self.next() as usize) % upper
        }

        fn coin(&mut self) -> bool {
            self.next() & 1 == 0
        }

        fn shuffle<T>(&mut self, values: &mut [T]) {
            for i in (1..values.len()).rev() {
                values.swap(i, self.usize(i + 1));
            }
        }
    }

    struct DifferentialGraphBuilder {
        graph: Graph,
        pending: Vec<Node>,
        next_name: usize,
    }

    impl DifferentialGraphBuilder {
        fn new() -> Self {
            let mut graph = Graph::new();
            graph.opset_imports.insert(String::new(), 17);
            Self {
                graph,
                pending: Vec::new(),
                next_name: 0,
            }
        }

        fn value(&mut self, prefix: &str, dims: &[usize]) -> ValueId {
            let name = format!("{prefix}_{}", self.next_name);
            self.next_name += 1;
            self.graph.create_named_value(
                name,
                DataType::Float32,
                static_shape(dims.iter().copied()),
            )
        }

        fn input(&mut self, prefix: &str, dims: &[usize]) -> ValueId {
            let value = self.value(prefix, dims);
            self.graph.add_input(value);
            value
        }

        fn scalar(&mut self, prefix: &str, value: f32) -> ValueId {
            let id = self.value(prefix, &[]);
            self.graph.set_initializer(
                id,
                WeightRef::Inline(TensorData::from_raw(
                    DataType::Float32,
                    vec![],
                    value.to_le_bytes().to_vec(),
                )),
            );
            id
        }

        fn node(
            &mut self,
            name: impl Into<String>,
            op: &str,
            inputs: Vec<ValueId>,
            output: ValueId,
        ) -> &mut Node {
            let mut node = Node::new(
                NodeId(0),
                op,
                inputs.into_iter().map(Some).collect(),
                vec![output],
            );
            node.name = name.into();
            self.pending.push(node);
            self.pending.last_mut().unwrap()
        }

        fn output(&mut self, value: ValueId) {
            self.graph.add_output(value);
        }

        fn add_matmul_bias(&mut self, rng: &mut FusionTestRng, relu: bool, overlap: bool) {
            let a = self.input("mm_a", &[4]);
            let w0 = self.input("mm_w0", &[4]);
            let m0 = self.value("mm_m0", &[4]);
            self.node("mm_left", "MatMul", vec![a, w0], m0);

            let bias = if overlap {
                let w1 = self.input("mm_w1", &[4]);
                let m1 = self.value("mm_m1", &[4]);
                self.node("mm_right", "MatMul", vec![a, w1], m1);
                m1
            } else {
                self.input("mm_bias", &[4])
            };

            let add = self.value("mm_add", &[4]);
            let add_inputs = if rng.coin() {
                vec![m0, bias]
            } else {
                vec![bias, m0]
            };
            self.node("mm_add_node", "Add", add_inputs, add);
            if relu {
                let output = self.value("mm_relu", &[4]);
                self.node("mm_relu_node", "Relu", vec![add], output);
                self.output(output);
            } else {
                self.output(add);
            }
        }

        fn add_layernorm(&mut self, split_diff: bool) {
            let x = self.input("ln_x", &[4]);
            let two = self.input("ln_two", &[4]);
            let eps = self.scalar("ln_eps", 1e-12);
            let scale = self.input("ln_scale", &[4]);
            let bias = self.input("ln_bias", &[4]);

            let mean = self.value("ln_mean", &[4]);
            let rm1 = self.node("ln_mean_node", "ReduceMean", vec![x], mean);
            rm1.attributes
                .insert("axes".into(), Attribute::Ints(vec![-1]));
            rm1.attributes.insert("keepdims".into(), Attribute::Int(1));

            let diff_pow = self.value("ln_diff_pow", &[4]);
            self.node("ln_sub_pow", "Sub", vec![x, mean], diff_pow);
            let diff_div = if split_diff {
                let value = self.value("ln_diff_div", &[4]);
                self.node("ln_sub_div", "Sub", vec![x, mean], value);
                value
            } else {
                diff_pow
            };
            let sq = self.value("ln_sq", &[4]);
            self.node("ln_pow", "Pow", vec![diff_pow, two], sq);
            let var = self.value("ln_var", &[4]);
            let rm2 = self.node("ln_var_node", "ReduceMean", vec![sq], var);
            rm2.attributes
                .insert("axes".into(), Attribute::Ints(vec![-1]));
            rm2.attributes.insert("keepdims".into(), Attribute::Int(1));
            let vare = self.value("ln_vare", &[4]);
            self.node("ln_add_eps", "Add", vec![var, eps], vare);
            let std = self.value("ln_std", &[4]);
            self.node("ln_sqrt", "Sqrt", vec![vare], std);
            let norm = self.value("ln_norm", &[4]);
            self.node("ln_div", "Div", vec![diff_div, std], norm);
            let scaled = self.value("ln_scaled", &[4]);
            self.node("ln_mul", "Mul", vec![norm, scale], scaled);
            let output = self.value("ln_output", &[4]);
            self.node("ln_add_bias", "Add", vec![scaled, bias], output);
            self.output(output);
        }

        fn add_gelu(&mut self, rng: &mut FusionTestRng) {
            let x = self.input("gelu_x", &[4]);
            let half = self.value("gelu_half", &[4]);
            if rng.coin() {
                let c = self.scalar("gelu_half_c", 0.5);
                let inputs = if rng.coin() { vec![x, c] } else { vec![c, x] };
                self.node("gelu_half_node", "Mul", inputs, half);
            } else {
                let c = self.scalar("gelu_two_c", 2.0);
                self.node("gelu_half_node", "Div", vec![x, c], half);
            }

            let scaled = self.value("gelu_scaled", &[4]);
            if rng.coin() {
                let c = self.scalar("gelu_sqrt2", std::f32::consts::SQRT_2);
                self.node("gelu_inner", "Div", vec![x, c], scaled);
            } else {
                let c = self.scalar("gelu_isqrt2", std::f32::consts::FRAC_1_SQRT_2);
                let inputs = if rng.coin() { vec![x, c] } else { vec![c, x] };
                self.node("gelu_inner", "Mul", inputs, scaled);
            }
            let erf = self.value("gelu_erf", &[4]);
            self.node("gelu_erf_node", "Erf", vec![scaled], erf);
            let one = self.scalar("gelu_one", 1.0);
            let plus_one = self.value("gelu_plus_one", &[4]);
            let inputs = if rng.coin() {
                vec![erf, one]
            } else {
                vec![one, erf]
            };
            self.node("gelu_add", "Add", inputs, plus_one);
            let output = self.value("gelu_output", &[4]);
            let inputs = if rng.coin() {
                vec![half, plus_one]
            } else {
                vec![plus_one, half]
            };
            self.node("gelu_outer", "Mul", inputs, output);
            self.output(output);
        }

        fn add_attention(&mut self, rng: &mut FusionTestRng) {
            let q = self.input("attn_q", &[1, 2, 3, 4]);
            let k = self.input("attn_k", &[1, 2, 3, 4]);
            let v = self.input("attn_v", &[1, 2, 3, 4]);
            let k_side = if rng.coin() {
                let kt = self.value("attn_kt", &[1, 2, 4, 3]);
                let transpose = self.node("attn_transpose", "Transpose", vec![k], kt);
                transpose
                    .attributes
                    .insert("perm".into(), Attribute::Ints(vec![0, 1, 3, 2]));
                kt
            } else {
                k
            };
            let scores = self.value("attn_scores", &[1, 2, 3, 3]);
            self.node("attn_score_mm", "MatMul", vec![q, k_side], scores);
            let scale_const = if rng.coin() {
                self.scalar("attn_divisor", 2.0)
            } else {
                self.scalar("attn_multiplier", 0.5)
            };
            let scaled = self.value("attn_scaled", &[1, 2, 3, 3]);
            if self
                .graph
                .value(scale_const)
                .name
                .as_deref()
                .unwrap()
                .contains("divisor")
            {
                self.node("attn_scale", "Div", vec![scores, scale_const], scaled);
            } else {
                let inputs = if rng.coin() {
                    vec![scores, scale_const]
                } else {
                    vec![scale_const, scores]
                };
                self.node("attn_scale", "Mul", inputs, scaled);
            }
            let softmax_input = if rng.coin() {
                let mask = self.input("attn_mask", &[1, 1, 3, 3]);
                let masked = self.value("attn_masked", &[1, 2, 3, 3]);
                let inputs = if rng.coin() {
                    vec![scaled, mask]
                } else {
                    vec![mask, scaled]
                };
                self.node("attn_mask_add", "Add", inputs, masked);
                masked
            } else {
                scaled
            };
            let probs = self.value("attn_probs", &[1, 2, 3, 3]);
            let softmax = self.node("attn_softmax", "Softmax", vec![softmax_input], probs);
            softmax.attributes.insert(
                "axis".into(),
                Attribute::Int(if rng.coin() { -1 } else { 3 }),
            );
            let output = self.value("attn_output", &[1, 2, 3, 4]);
            self.node("attn_output_mm", "MatMul", vec![probs, v], output);
            self.output(output);
        }

        fn add_resumable_chain(&mut self, rng: &mut FusionTestRng) {
            let input = self.input("chain_input", &[4]);
            let first = self.value("chain_start", &[4]);
            self.node("chain_0", "ChainStart", vec![input], first);
            let mut value = first;
            for index in 1..(4 + rng.usize(6)) {
                let output = self.value("chain_link", &[4]);
                self.node(format!("chain_{index}"), "ChainLink", vec![value], output);
                value = output;
            }
            self.output(value);
        }

        fn add_noise(&mut self, rng: &mut FusionTestRng) {
            for index in 0..rng.usize(8) {
                let input = self.input("noise_input", &[4]);
                let output = self.value("noise_output", &[4]);
                let op = ["Abs", "Neg", "Identity", "Tanh"][rng.usize(4)];
                self.node(format!("noise_{index}"), op, vec![input], output);
                self.output(output);
            }
        }

        fn finish(mut self, rng: &mut FusionTestRng) -> Graph {
            // Seed and remove one dummy per real node. Random removal order
            // randomizes the arena free-list; independently shuffling real-node
            // insertion then decouples logical/topological order from NodeId.
            let mut slots = Vec::with_capacity(self.pending.len());
            for _ in 0..self.pending.len() {
                slots.push(self.graph.insert_node(Node::new(
                    NodeId(0),
                    "IdSeed",
                    Vec::new(),
                    Vec::new(),
                )));
            }
            rng.shuffle(&mut slots);
            for id in slots {
                self.graph.remove_node(id);
            }
            rng.shuffle(&mut self.pending);
            for node in self.pending {
                self.graph.insert_node(node);
            }
            self.graph
        }
    }

    fn randomized_fusion_graph(rng: &mut FusionTestRng) -> Graph {
        let mut builder = DifferentialGraphBuilder::new();
        // Every trial contains every registered matcher. The two structural
        // motifs deliberately have two MatMul starts sharing the Add (and Relu),
        // so lowest-NodeId overlap resolution affects the exact replacement.
        builder.add_attention(rng);
        builder.add_matmul_bias(rng, true, true);
        builder.add_layernorm(rng.coin());
        builder.add_gelu(rng);
        builder.add_matmul_bias(rng, false, true);

        // Add extra independent registered motifs for structural diversity.
        for _ in 0..rng.usize(4) {
            match rng.usize(5) {
                0 => builder.add_attention(rng),
                1 => {
                    let overlap = rng.coin();
                    builder.add_matmul_bias(rng, true, overlap);
                }
                2 => builder.add_layernorm(rng.coin()),
                3 => builder.add_gelu(rng),
                _ => {
                    let overlap = rng.coin();
                    builder.add_matmul_bias(rng, false, overlap);
                }
            }
        }
        builder.add_resumable_chain(rng);
        builder.add_noise(rng);
        builder.finish(rng)
    }

    fn differential_patterns() -> Vec<FusionPattern> {
        let mut patterns = default_fusion_patterns();
        // Unlike production replacements, this test-only standard-domain op can
        // immediately match the next ChainLink. Its NodeId has already been
        // passed by the ascending cursor, so correctness requires a lower-id
        // revisit after every fusion until the chain reaches its fixpoint.
        patterns.push(
            FusionPattern::new("ResumableChain", &["ChainStart", "ChainLink"], "ChainStart")
                .with_replacement_domain(""),
        );
        patterns
    }

    struct AffectedRevisitCase {
        graph: Graph,
        lower_start: NodeId,
        later_start: NodeId,
        first_middle: NodeId,
        first_tail: NodeId,
        final_tail: NodeId,
    }

    fn affected_revisit_case(seed: u64) -> AffectedRevisitCase {
        let mut rng = FusionTestRng(seed ^ 0xbb67_ae85_84ca_a73b);
        let noise_count = 3 + rng.usize(6);
        let slot_count = 5 + noise_count;
        let later_start = NodeId((slot_count - 1) as u32);
        let mut lower_ids: Vec<u32> = (0..later_start.0).collect();
        rng.shuffle(&mut lower_ids);
        let lower_start = NodeId(lower_ids[0]);
        let first_middle = NodeId(lower_ids[1]);
        let first_tail = NodeId(lower_ids[2]);
        let final_tail = NodeId(lower_ids[3]);

        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        let lower_input = graph.create_named_value(
            format!("lower_input_{seed}"),
            DataType::Float32,
            static_shape([4]),
        );
        let later_input = graph.create_named_value(
            format!("later_input_{seed}"),
            DataType::Float32,
            static_shape([4]),
        );
        graph.add_input(lower_input);
        graph.add_input(later_input);
        let lower_value = graph.create_named_value(
            format!("lower_value_{seed}"),
            DataType::Float32,
            static_shape([4]),
        );
        let middle_value = graph.create_named_value(
            format!("middle_value_{seed}"),
            DataType::Float32,
            static_shape([4]),
        );
        let first_output = graph.create_named_value(
            format!("first_output_{seed}"),
            DataType::Float32,
            static_shape([4]),
        );
        let first_tail_output = graph.create_named_value(
            format!("first_tail_output_{seed}"),
            DataType::Float32,
            static_shape([4]),
        );
        let final_output = graph.create_named_value(
            format!("final_output_{seed}"),
            DataType::Float32,
            static_shape([4]),
        );
        graph.add_output(final_output);

        let mut placements = vec![
            (
                lower_start,
                Node::new(
                    NodeId(0),
                    "AdversaryStart",
                    vec![Some(lower_input)],
                    vec![lower_value],
                ),
            ),
            (
                later_start,
                Node::new(
                    NodeId(0),
                    "AdversaryStart",
                    vec![Some(later_input)],
                    vec![middle_value],
                ),
            ),
            (
                first_middle,
                Node::new(
                    NodeId(0),
                    "AdversaryMiddle",
                    vec![Some(middle_value)],
                    vec![first_output],
                ),
            ),
            (
                first_tail,
                Node::new(
                    NodeId(0),
                    "AdversaryTail",
                    vec![Some(first_output), Some(lower_value)],
                    vec![first_tail_output],
                ),
            ),
            (
                final_tail,
                Node::new(
                    NodeId(0),
                    "AdversaryTail",
                    vec![Some(first_tail_output)],
                    vec![final_output],
                ),
            ),
        ];
        let core_ids = HashSet::from([
            lower_start,
            later_start,
            first_middle,
            first_tail,
            final_tail,
        ]);
        let mut noise_index = 0;
        for raw_id in 0..slot_count as u32 {
            let id = NodeId(raw_id);
            if core_ids.contains(&id) {
                continue;
            }
            let input = graph.create_named_value(
                format!("noise_input_{seed}_{noise_index}"),
                DataType::Float32,
                static_shape([4]),
            );
            let output = graph.create_named_value(
                format!("noise_output_{seed}_{noise_index}"),
                DataType::Float32,
                static_shape([4]),
            );
            graph.add_input(input);
            graph.add_output(output);
            placements.push((
                id,
                Node::new(
                    NodeId(0),
                    ["Abs", "Neg", "Identity", "Tanh"][rng.usize(4)],
                    vec![Some(input)],
                    vec![output],
                ),
            ));
            noise_index += 1;
        }

        rng.shuffle(&mut placements);
        for _ in 0..slot_count {
            graph.insert_node(Node::new(NodeId(0), "IdSeed", Vec::new(), Vec::new()));
        }
        for &(target, _) in placements.iter().rev() {
            graph.remove_node(target);
        }
        for (expected, node) in placements {
            assert_eq!(graph.insert_node(node), expected);
        }

        AffectedRevisitCase {
            graph,
            lower_start,
            later_start,
            first_middle,
            first_tail,
            final_tail,
        }
    }

    #[test]
    fn affected_candidate_starts_revisits_newly_eligible_lower_ids() {
        const TRIALS: usize = 5_000;
        let pattern = FusionPattern::new(
            "AffectedBehindCursor",
            &["AdversaryStart", "AdversaryMiddle", "AdversaryTail"],
            "AdversaryMiddle",
        )
        .with_replacement_domain("");
        let patterns = vec![pattern.clone()];
        let mut reclaimable_low_slot_trials = 0;
        let mut affected_scheduled = 0;
        let mut affected_revisit_hits = 0;

        for trial in 0..TRIALS {
            let case = affected_revisit_case(trial as u64);
            assert!(case.graph.validate().is_ok(), "invalid trial {trial}");
            assert!(case.lower_start.0 < case.later_start.0);
            assert!(case.first_middle.0 < case.later_start.0);
            assert!(case.first_tail.0 < case.later_start.0);
            assert!(pattern.try_match_at(&case.graph, case.lower_start).is_none());
            let first_match = pattern
                .try_match_at(&case.graph, case.later_start)
                .expect("later start must be the first eligible match");
            assert_eq!(
                first_match.nodes,
                vec![case.later_start, case.first_middle, case.first_tail]
            );

            // Reverse removal followed by LIFO insertion always reuses the
            // match-start slot. The lower interior slots remain reclaimable.
            let mut reclaim_probe = case.graph.clone();
            let first_fused = pattern
                .apply_fusion_returning_id(&mut reclaim_probe, &first_match)
                .unwrap();
            assert_eq!(first_fused, case.later_start);
            let probe_id = reclaim_probe.insert_node(Node::new(
                NodeId(0),
                "ReclaimProbe",
                Vec::new(),
                Vec::new(),
            ));
            assert_eq!(probe_id, case.first_middle);
            reclaimable_low_slot_trials += 1;

            let mut reference = case.graph.clone();
            run_restart_reference(&patterns, &mut reference);
            let mut actual = case.graph;
            let mut lower_was_scheduled = false;
            let mut trial_hits = 0;
            OpFusion::with_patterns(patterns.clone())
                .run_with_fusion_observer(
                    &mut actual,
                    |name, source, start, matched, affected, fused_id| {
                        if name != "AffectedBehindCursor" {
                            return;
                        }
                        assert_eq!(
                            fused_id, matched[0],
                            "replacement must reuse the just-freed match-start slot"
                        );
                        if start == case.later_start {
                            assert_eq!(source, ScanCandidateSource::Initial);
                            assert_eq!(
                                matched,
                                &[case.later_start, case.first_middle, case.first_tail]
                            );
                            assert!(affected.contains(&case.lower_start));
                            assert_ne!(fused_id, case.lower_start);
                            lower_was_scheduled = true;
                            affected_scheduled += 1;
                        } else if start == case.lower_start {
                            assert!(lower_was_scheduled);
                            assert_eq!(source, ScanCandidateSource::Revisit);
                            assert_eq!(
                                matched,
                                &[case.lower_start, case.later_start, case.final_tail]
                            );
                            trial_hits += 1;
                            affected_revisit_hits += 1;
                        }
                    },
                )
                .unwrap();

            assert_eq!(trial_hits, 1, "affected revisit not hit on trial {trial}");
            assert!(actual.validate().is_ok(), "invalid result on trial {trial}");
            assert_fusion_graphs_byte_identical(actual, reference, trial);
        }

        assert_eq!(reclaimable_low_slot_trials, TRIALS);
        assert_eq!(affected_scheduled, TRIALS);
        assert_eq!(affected_revisit_hits, TRIALS);
        eprintln!(
            "affected behind-cursor revisit hits: {affected_revisit_hits}/{TRIALS} \
             ({}%); reclaimable lower slots present: {reclaimable_low_slot_trials}/{TRIALS}",
            affected_revisit_hits * 100 / TRIALS
        );
    }

    fn assert_overlapping_structural_candidates(graph: &Graph, patterns: &[FusionPattern]) {
        let mut saw_gemm_overlap = false;
        let mut saw_bias_overlap = false;
        for (add_id, add) in graph.nodes.iter().filter(|(_, node)| node.op_type == "Add") {
            let starts: Vec<_> = add
                .input_values()
                .filter_map(|value| graph.value(value).producer)
                .filter(|&producer| graph.node(producer).op_type == "MatMul")
                .collect();
            if starts.len() != 2 {
                continue;
            }
            let has_relu = graph
                .successors(add_id)
                .iter()
                .any(|&successor| graph.node(successor).op_type == "Relu");
            let pattern = if has_relu { &patterns[1] } else { &patterns[4] };
            assert!(
                starts
                    .iter()
                    .all(|&start| pattern.try_match_at(graph, start).is_some()),
                "both MatMul starts must be eligible for the shared structural tail"
            );
            saw_gemm_overlap |= has_relu;
            saw_bias_overlap |= !has_relu;
        }
        assert!(saw_gemm_overlap, "missing shared MatMul+Add+Relu candidates");
        assert!(saw_bias_overlap, "missing shared MatMul+Add candidates");
    }

    #[test]
    fn resumable_scan_matches_restart_reference_on_randomized_graphs() {
        const TRIALS: usize = 5_000;
        const REGISTERED_PATTERNS: usize = 5;
        let mut rng = FusionTestRng(0x6a09_e667_f3bc_c909);
        let patterns = differential_patterns();
        assert_eq!(default_fusion_patterns().len(), REGISTERED_PATTERNS);

        let mut saw_non_topological_ids = false;
        for trial in 0..TRIALS {
            let mut graph = randomized_fusion_graph(&mut rng);
            assert!(
                graph.validate().is_ok(),
                "invalid input graph on trial {trial}"
            );

            for pattern in &patterns[..REGISTERED_PATTERNS] {
                assert!(
                    pattern.find_match(&graph).is_some(),
                    "{} was not exercised on trial {trial}",
                    pattern.pattern_name()
                );
            }
            assert_overlapping_structural_candidates(&graph, &patterns);
            assert!(
                graph
                    .nodes
                    .values()
                    .filter(|node| node.op_type == "ChainLink")
                    .count()
                    >= 3,
                "chained replacement adversary must require multiple revisits"
            );
            saw_non_topological_ids |=
                graph.topological_order().unwrap() != graph.nodes.keys().collect::<Vec<_>>();

            let mut reference = graph.clone();
            run_restart_reference(&patterns, &mut reference);
            OpFusion::with_patterns(patterns.clone())
                .run(&mut graph, &PassContext::new())
                .unwrap();

            assert!(
                graph.validate().is_ok(),
                "invalid result graph on trial {trial}"
            );
            assert!(
                graph.nodes.values().all(|node| node.op_type != "ChainLink"),
                "resumable chain did not reach its fixpoint on trial {trial}"
            );
            assert_fusion_graphs_byte_identical(graph, reference, trial);
        }
        assert!(
            saw_non_topological_ids,
            "randomized insertion must decouple NodeId from topological order"
        );
    }
}
