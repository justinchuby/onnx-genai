//! Operator fusion: match a connected op-sequence and replace it with a single
//! fused op (see `docs/architecture/ORT2.md` §18.2).
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

use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, TensorData, ValueId, WeightRef};

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
                "ReduceMean",
                "Sub",
                "Pow",
                "ReduceMean",
                "Add",
                "Sqrt",
                "Div",
                "Mul",
                "Add",
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
            .consumers(value)
            .into_iter()
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
            .consumers(mean)
            .into_iter()
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
                                .consumers(out)
                                .into_iter()
                                .any(|consumer| !matched_set.contains(&consumer))
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
            let survives = graph.outputs.contains(&output)
                || graph
                    .consumers(output)
                    .into_iter()
                    .any(|consumer| !matched_set.contains(&consumer));
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
        let out_mm = graph.consumers(sm_out).into_iter().find(|&c| {
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
        let (scale_out, mask, mask_add) = if Self::op_matches(prod, "Add") && prod.inputs.len() == 2
        {
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
                            .consumers(o)
                            .into_iter()
                            .any(|consumer| !matched_set.contains(&consumer))
                })
        });
        if escapes {
            return None;
        }

        // The fused output (out_mm's single output) must survive removal.
        let survives = graph.outputs.contains(&output)
            || graph
                .consumers(output)
                .into_iter()
                .any(|consumer| !matched_set.contains(&consumer));
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
                && graph.consumers(k_side) == [score_mm_id]
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
                            .consumers(o)
                            .into_iter()
                            .any(|consumer| !matched_set.contains(&consumer))
                })
        });
        if escapes {
            return None;
        }

        // The fused output (outer's single output) must survive removal.
        let survives = graph.outputs.contains(&output)
            || graph
                .consumers(output)
                .into_iter()
                .any(|consumer| !matched_set.contains(&consumer));
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
            let next = succ_ids
                .into_iter()
                .find(|&s| !chain_set.contains(&s) && Self::op_matches(graph.node(s), op))?;
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
                    .consumers(out)
                    .into_iter()
                    .any(|consumer| !chain_set.contains(&consumer))
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
        let survives = graph.outputs.contains(&output)
            || graph
                .consumers(output)
                .into_iter()
                .any(|consumer| !chain_set.contains(&consumer));
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
        if !fused.domain.is_empty() {
            graph.opset_imports.entry(fused.domain.clone()).or_insert(1);
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
    /// * **axis** must resolve to the *same single concrete* axis for BOTH the
    ///   mean and the variance `ReduceMean`, read from each node's `axes`
    ///   **attribute** (opset < 18) or its axes **input** (opset-24 schema), with
    ///   `keepdims = 1` on both; multi-axis / absent / reduce-all / mismatched
    ///   axes / `keepdims = 0` → decline; never silently assume `-1`.
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

        // axis guard: BOTH `ReduceMean` nodes must reduce a single concrete axis
        // read from the `axes` ATTRIBUTE (opset < 18) or, for the opset-24 schema,
        // the axes INPUT; both must keep the reduced dim (`keepdims = 1`, or its
        // default). The mean and variance reductions must be over the SAME axis,
        // otherwise this is not a LayerNorm — decline (never silently assume -1).
        let axis = reduce_single_axis(graph, rm1)?;
        if reduce_single_axis(graph, rm2)? != axis {
            return None;
        }

        let mut attributes = HashMap::new();
        attributes.insert("axis".to_string(), Attribute::Int(axis));
        attributes.insert("epsilon".to_string(), Attribute::Float(epsilon));

        Some((vec![Some(x), Some(scale), Some(bias)], attributes))
    }
}

/// Resolve the single concrete reduction axis of a `ReduceMean`, requiring the
/// reduced dimension to be kept (`keepdims = 1`, or absent → the schema default
/// of 1). The axes come from the `axes` **attribute** (opset < 18) or the axes
/// **input** (opset-24 schema). Multi-axis / absent / reduce-all / `keepdims = 0`
/// → `None`.
fn reduce_single_axis(graph: &Graph, rm: &Node) -> Option<i64> {
    if let Some(keepdims) = rm.attr("keepdims").and_then(Attribute::as_int)
        && keepdims != 1
    {
        return None;
    }
    let axes: Vec<i64> = if let Some(axes) = rm.attr("axes").and_then(Attribute::as_ints) {
        axes.to_vec()
    } else {
        let axes_value = rm.inputs.get(1).copied().flatten()?;
        read_i64_vector(graph, axes_value)?
    };
    let [axis] = axes.as_slice() else {
        return None;
    };
    Some(*axis)
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

/// Read an int64 vector from an inline int64 initializer, if `value` is backed
/// by one. Used to resolve the opset-24 `axes`-as-input of a `ReduceMean` (the
/// axis moved from an attribute to an input in opset 18).
/// Resolve a 1-D int64 axes vector for the value `value`, whether it is an
/// inline initializer or is still produced by a `Constant` node (the fusion pass
/// may run before `ConstantFolding` materializes it — see the module-level pass
/// order). Returns `None` for a non-int64 dtype, a byte length that is not a
/// whole number of int64 elements, or a higher-than-1-D (malformed) axes tensor.
fn read_i64_vector(graph: &Graph, value: ValueId) -> Option<Vec<i64>> {
    if let Some(WeightRef::Inline(tensor)) = graph.initializers.get(&value) {
        return i64_axes_from_tensor(tensor);
    }
    // Fall back to a not-yet-folded `Constant` producer.
    let producer = graph.value(value).producer?;
    let node = graph.node(producer);
    if node.op_type == "Constant"
        && let Some(Attribute::Tensor(tensor)) = node.attr("value")
    {
        return i64_axes_from_tensor(tensor);
    }
    None
}

/// Decode a 1-D int64 axes tensor, rejecting a non-int64 dtype, a byte length
/// that is not a whole number of int64 elements, or a rank > 1 buffer.
fn i64_axes_from_tensor(tensor: &TensorData) -> Option<Vec<i64>> {
    if tensor.dtype != DataType::Int64
        || !tensor.data.len().is_multiple_of(8)
        || tensor.dims.len() > 1
    {
        return None;
    }
    tensor
        .data
        .chunks_exact(8)
        .map(|chunk| Some(i64::from_le_bytes(chunk.try_into().ok()?)))
        .collect()
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
        mut observe_fusion: impl FnMut(&str, ScanCandidateSource, NodeId, &[NodeId], &[NodeId], NodeId),
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
                    (None, Some(_)) => {
                        (revisits.pop_first().unwrap(), ScanCandidateSource::Revisit)
                    }
                    (Some(id), Some(revisit)) if id <= revisit => {
                        cursor += 1;
                        if id == revisit {
                            revisits.pop_first();
                        }
                        (id, ScanCandidateSource::Initial)
                    }
                    (Some(_), Some(_)) => {
                        (revisits.pop_first().unwrap(), ScanCandidateSource::Revisit)
                    }
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
        observe_fusion: impl FnMut(&str, ScanCandidateSource, NodeId, &[NodeId], &[NodeId], NodeId),
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
mod tests;
