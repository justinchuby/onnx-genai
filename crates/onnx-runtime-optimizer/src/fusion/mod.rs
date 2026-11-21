//! Operator fusion using compiled IR recognition, followed by separate,
//! provider-independent rewrite eligibility and application.
//!
//! Textual rules recognize LayerNorm, exact GELU and SDPA DAGs. Generic
//! [`FusionPattern::new`] chains compile into the same matcher executor.
//! Recognition proves node/value identity and closed intermediate boundaries,
//! including lexical uses. Numeric constants, axes, bias broadcast and kernel
//! signatures remain explicit predicates. No structural match alone authorizes
//! fusion.
//!
//! Fused operators use the private [`CONTRIB_DOMAIN`], not the standard ONNX
//! namespace. Final output identities are preserved. A [`PatternMatch`] is
//! invalidated by graph mutation. Internally the pass retains proven-disjoint
//! regions, invalidates every overlap before node IDs can be reused, and refreshes
//! affected roots while retaining restart-equivalent candidate ordering.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use onnx_runtime_ir::pattern::{MatchSession, Pattern};
use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, TensorData, ValueId, WeightRef};

use crate::error::Result;
use crate::pass::{OptimizationPass, PassContext};
use crate::recognition::RecognitionRule;

mod matching;

/// Private contrib domain used for optimizer-created fused operations.
pub const CONTRIB_DOMAIN: &str = "com.microsoft";

const SQRT_2: f32 = std::f32::consts::SQRT_2;
const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;

fn approx(a: f32, expected: f32) -> bool {
    (a - expected).abs() <= 1e-6 * expected.abs().max(1.0)
}

type FusedNodeSpec = (Vec<Option<ValueId>>, HashMap<String, Attribute>);

/// A recognized, eligible occurrence, invalidated by graph mutation.
#[derive(Clone, Debug)]
pub struct PatternMatch {
    /// Matched nodes in canonical rewrite order.
    pub nodes: Vec<NodeId>,
    /// External values, in first-seen or schema-defined order.
    pub external_inputs: Vec<ValueId>,
    /// Final output identity reused by the fused operation.
    pub output: ValueId,
}

/// Kernel-signature extraction for a recognized region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RewriteKind {
    /// External inputs in first-seen order.
    Structural,
    /// `[X, Scale, B]`, with concrete axis and epsilon.
    LayerNorm,
    /// `[Q, K, V, (mask)]`, with scale and transpose state.
    Attention,
    /// `[X]` for the exact Erf decomposition.
    Gelu,
}

/// A compiled recognition rule and its separate fused-operation contract.
#[derive(Clone, Debug)]
pub struct FusionPattern {
    name: String,
    ops: Vec<String>,
    replacement: String,
    #[cfg(test)]
    replacement_domain: String,
    kind: RewriteKind,
    recognizer: RecognitionRule,
}

impl FusionPattern {
    /// Compile a connected chain of standard-domain operators.
    ///
    /// Intermediate nodes can have multiple/absent outputs, and the next step
    /// can consume any present output at any input position. The terminal must
    /// have one present output. Other ports/attributes are unconstrained.
    pub fn new(name: &str, ops: &[&str], replacement: &str) -> Result<Self> {
        let operators: Vec<_> = ops.iter().map(|op| ("", *op)).collect();
        let pattern = Pattern::operator_chain(&operators)?;
        Ok(Self {
            name: name.into(),
            ops: ops.iter().map(|s| (*s).into()).collect(),
            replacement: replacement.into(),
            #[cfg(test)]
            replacement_domain: CONTRIB_DOMAIN.into(),
            kind: RewriteKind::Structural,
            recognizer: RecognitionRule::from_pattern(name, pattern, "step_0")?,
        })
    }

    fn dag(
        name: &str,
        ops: &[&str],
        replacement: &str,
        kind: RewriteKind,
        recognizer: &RecognitionRule,
    ) -> Self {
        Self {
            name: name.into(),
            ops: ops.iter().map(|s| (*s).into()).collect(),
            replacement: replacement.into(),
            #[cfg(test)]
            replacement_domain: CONTRIB_DOMAIN.into(),
            kind,
            recognizer: recognizer.clone(),
        }
    }

    /// Shared-Sub and split-Sub LayerNorm decompositions.
    pub fn layernorm() -> Self {
        Self::dag(
            "LayerNorm",
            &["ReduceMean"],
            "LayerNormalization",
            RewriteKind::LayerNorm,
            &matching::LAYERNORM,
        )
    }

    /// Scaled dot-product attention, optionally masked and with absorbed Kᵀ.
    pub fn attention() -> Self {
        Self::dag(
            "Attention",
            &["Softmax"],
            "FusedAttention",
            RewriteKind::Attention,
            &matching::ATTENTION,
        )
    }

    /// Exact Erf-based GELU, including equivalent scalar encodings.
    pub fn gelu() -> Self {
        Self::dag("Gelu", &["Erf"], "Gelu", RewriteKind::Gelu, &matching::GELU)
    }

    pub fn kind(&self) -> RewriteKind {
        self.kind
    }

    pub fn pattern_name(&self) -> &str {
        &self.name
    }

    #[cfg(test)]
    fn with_replacement_domain(mut self, domain: &str) -> Self {
        self.replacement_domain = domain.into();
        self
    }

    /// Find the first eligible occurrence by logical anchor ID.
    /// Invalid graphs, matcher errors and exhausted budgets propagate.
    pub fn find_match(&self, graph: &Graph) -> Result<Option<PatternMatch>> {
        Ok(self
            .prepare_matches(graph)?
            .pop_first()
            .map(|(_, matched)| matched))
    }

    #[cfg(test)]
    fn try_match_at(&self, graph: &Graph, start: NodeId) -> Result<Option<PatternMatch>> {
        Ok(self.prepare_matches(graph)?.remove(&start))
    }

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

    fn match_is_fusable(&self, graph: &Graph, matched: &PatternMatch) -> bool {
        match self.kind {
            RewriteKind::LayerNorm => self.layernorm_spec(graph, matched).is_some(),
            RewriteKind::Attention => self.attention_spec(graph, matched).is_some(),
            RewriteKind::Gelu => self.gelu_spec(graph, matched).is_some(),
            RewriteKind::Structural => {
                !matches!(self.replacement.as_str(), "FusedMatMulBias" | "FusedGemm")
                    || self.matmul_bias_broadcast_ok(graph, matched)
            }
        }
    }

    fn matmul_bias_broadcast_ok(&self, graph: &Graph, matched: &PatternMatch) -> bool {
        let (Some(&matmul), Some(&add)) = (matched.nodes.first(), matched.nodes.get(1)) else {
            return false;
        };
        let Some(mm_out) = graph.node(matmul).outputs[0] else {
            return false;
        };
        let Some(bias) = graph.node(add).input_values().find(|&v| v != mm_out) else {
            return false;
        };
        let mm_shape = &graph.value(mm_out).shape;
        let bias_shape = &graph.value(bias).shape;
        if bias_shape.len() > mm_shape.len() {
            return false;
        }
        let offset = mm_shape.len() - bias_shape.len();
        bias_shape
            .iter()
            .enumerate()
            .all(|(index, &dim)| dim == mm_shape[offset + index] || dim.as_static() == Some(1))
    }

    /// Apply an eligible match without intervening graph mutation.
    pub fn apply_fusion(&self, graph: &mut Graph, matched: &PatternMatch) -> Result<()> {
        self.apply_fusion_returning_id(graph, matched).map(|_| ())
    }

    fn apply_fusion_returning_id(
        &self,
        graph: &mut Graph,
        matched: &PatternMatch,
    ) -> Result<NodeId> {
        let output = matched.output;
        let spec = match self.kind {
            RewriteKind::Structural => Some((
                matched.external_inputs.iter().map(|&v| Some(v)).collect(),
                HashMap::new(),
            )),
            RewriteKind::LayerNorm => self.layernorm_spec(graph, matched),
            RewriteKind::Attention => self.attention_spec(graph, matched),
            RewriteKind::Gelu => self.gelu_spec(graph, matched),
        };
        let (inputs, attributes) = spec.ok_or_else(|| {
            crate::error::OptimizerError::Fusion(format!(
                "{} no longer satisfies its kernel contract; recognize again after graph mutation",
                self.name
            ))
        })?;
        for &node in matched.nodes.iter().rev() {
            graph.remove_node(node);
        }
        if graph.try_value(output).is_none() {
            return Err(crate::error::OptimizerError::Fusion(format!(
                "{} final output {output:?} did not survive; keep a graph output or external user",
                self.name,
            )));
        }
        let mut fused = Node::new(NodeId(0), self.replacement.clone(), inputs, vec![output]);
        fused.attributes = attributes;
        #[cfg(not(test))]
        {
            fused.domain = CONTRIB_DOMAIN.into();
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

    fn layernorm_spec(&self, graph: &Graph, matched: &PatternMatch) -> Option<FusedNodeSpec> {
        let nodes = &matched.nodes;
        if nodes.len() != 9 && nodes.len() != 10 {
            return None;
        }
        let mean = graph.node(nodes[0]);
        let sub = graph.node(nodes[1]);
        let variance = graph.node(nodes[3]);
        let add_eps = graph.node(nodes[4]);
        let div = graph.node(nodes[6]);
        let mul = graph.node(nodes[7]);
        let final_add = graph.node(nodes[8]);
        let numerator = if nodes.len() == 10 {
            graph.node(nodes[9])
        } else {
            sub
        };
        let x = sub.inputs.first().copied().flatten()?;
        let mean_out = mean.outputs[0]?;
        let numerator_out = numerator.outputs[0]?;
        let std = graph.node(nodes[5]).outputs[0]?;
        if sub.inputs.as_slice() != [Some(x), Some(mean_out)]
            || numerator.inputs.as_slice() != [Some(x), Some(mean_out)]
            || div.inputs.as_slice() != [Some(numerator_out), Some(std)]
        {
            return None;
        }
        let norm = div.outputs[0]?;
        let scaled = mul.outputs[0]?;
        let scale = mul.input_values().find(|&v| v != norm)?;
        let bias = final_add.input_values().find(|&v| v != scaled)?;
        let var = variance.outputs[0]?;
        let eps = add_eps.input_values().find(|&v| v != var)?;
        let diff = sub.outputs.first().copied().flatten()?;
        let exponent = graph.node(nodes[2]).inputs.get(1).copied().flatten()?;
        if read_scalar_const_f32(graph, exponent, diff) != Some(2.0) {
            return None;
        }
        let epsilon = read_scalar_const_f32(graph, eps, var)?;
        // LayerNormalization reduces a suffix, whereas ReduceMean can reduce
        // arbitrary axes. Its affine tensors must span that same suffix.
        let input_shape = graph.value(x).tensor_shape()?;
        let axis = reduce_suffix_axis(graph, mean, input_shape.len())?;
        if reduce_suffix_axis(graph, variance, input_shape.len())? != axis {
            return None;
        }
        let start = usize::try_from(i64::try_from(input_shape.len()).ok()? + axis).ok()?;
        let normalized_shape = &input_shape[start..];
        let matches_affine = |value| {
            graph.value(value).tensor_shape().is_some_and(|shape| {
                // Leading singleton dimensions do not change the flattened
                // coefficients, but must not expand the operand's rank.
                shape.len() <= input_shape.len()
                    && shape
                        .iter()
                        .skip_while(|dim| dim.as_static() == Some(1))
                        .eq(normalized_shape
                            .iter()
                            .skip_while(|dim| dim.as_static() == Some(1)))
            })
        };
        if !matches_affine(scale) || !matches_affine(bias) {
            return None;
        }
        let attributes = HashMap::from([
            ("axis".into(), Attribute::Int(axis)),
            ("epsilon".into(), Attribute::Float(epsilon)),
        ]);
        Some((vec![Some(x), Some(scale), Some(bias)], attributes))
    }

    fn attention_spec(&self, graph: &Graph, matched: &PatternMatch) -> Option<FusedNodeSpec> {
        let nodes = &matched.nodes;
        let softmax = graph.node(*nodes.first()?);
        let rank = graph.value(softmax.inputs[0]?).shape.len();
        if rank == 0 {
            return None;
        }
        let axis = softmax.attr("axis").and_then(Attribute::as_int)?;
        let axis = if axis < 0 { axis + rank as i64 } else { axis };
        if axis != rank as i64 - 1 {
            return None;
        }
        let scores = graph.node(nodes[1]).outputs[0]?;
        let scale = matching::score_multiplier(graph, scores, graph.node(nodes[2]))?;
        let masked = matched.external_inputs.len() == 4;
        let absorbed = nodes.len() == 4 + usize::from(masked) + 1;
        if absorbed && !matching::transpose_eligible(graph, graph.node(*nodes.last()?), nodes[1]) {
            return None;
        }
        let inputs = matched.external_inputs.iter().map(|&v| Some(v)).collect();
        let attributes = HashMap::from([
            ("scale".into(), Attribute::Float(scale)),
            ("k_transposed".into(), Attribute::Int(i64::from(!absorbed))),
        ]);
        Some((inputs, attributes))
    }

    fn gelu_spec(&self, graph: &Graph, matched: &PatternMatch) -> Option<FusedNodeSpec> {
        let nodes = &matched.nodes;
        let x = *matched.external_inputs.first()?;
        let inner = graph.node(nodes[1]);
        let add = graph.node(nodes[2]);
        let half = graph.node(nodes[4]);
        if !matching::scaled_is(graph, inner, x, SQRT_2, FRAC_1_SQRT_2)
            || !matching::scaled_is(graph, half, x, 2.0, 0.5)
        {
            return None;
        }
        let erf_out = graph.node(nodes[0]).outputs[0]?;
        let one = add.input_values().find(|&v| v != erf_out)?;
        if !approx(read_scalar_const_f32(graph, one, erf_out)?, 1.0) {
            return None;
        }
        Some((vec![Some(x)], HashMap::new()))
    }
}

fn reduce_suffix_axis(graph: &Graph, node: &Node, rank: usize) -> Option<i64> {
    match node.attr("keepdims") {
        None | Some(Attribute::Int(1)) => {}
        _ => return None,
    }
    let mut axes = if let Some(attribute) = node.attr("axes") {
        attribute.as_ints()?.to_vec()
    } else {
        read_i64_vector(graph, node.inputs.get(1).copied().flatten()?)?
    };
    if axes.is_empty() {
        return None;
    }
    let rank = i64::try_from(rank).ok()?;
    for axis in &mut axes {
        if *axis < -rank || *axis >= rank {
            return None;
        }
        if *axis < 0 {
            *axis += rank;
        }
    }
    axes.sort_unstable();
    let count = i64::try_from(axes.len()).ok()?;
    let start = rank.checked_sub(count)?;
    if !axes.iter().copied().eq(start..rank) {
        return None;
    }
    Some(-count)
}

fn read_i64_vector(graph: &Graph, value: ValueId) -> Option<Vec<i64>> {
    if graph.inputs.contains(&value) {
        return None;
    }
    if let Some(WeightRef::Inline(tensor)) = graph.initializers.get(&value) {
        return i64_axes_from_tensor(tensor);
    }
    let node = graph.node(graph.value(value).producer?);
    if node.op_type != "Constant" || !onnx_runtime_ir::is_default_domain(&node.domain) {
        return None;
    }
    if let Some(Attribute::Tensor(tensor)) = node.attr("value") {
        return i64_axes_from_tensor(tensor);
    }
    node.attr("value_ints")
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec)
}

fn i64_axes_from_tensor(tensor: &TensorData) -> Option<Vec<i64>> {
    if tensor.dtype != DataType::Int64 || !tensor.data.len().is_multiple_of(8) {
        return None;
    }
    let numel = tensor.data.len() / 8;
    if tensor.dims.len() != 1 || tensor.dims[0] != numel {
        return None;
    }
    tensor
        .data
        .chunks_exact(8)
        .map(|chunk| Some(i64::from_le_bytes(chunk.try_into().ok()?)))
        .collect()
}

/// Read one immutable f32 coefficient whose broadcast cannot expand the
/// operand's rank. Unknown-rank operands require an actual rank-zero scalar.
fn read_scalar_const_f32(graph: &Graph, value: ValueId, operand: ValueId) -> Option<f32> {
    if !graph.is_constant_initializer(value) {
        return None;
    }
    let WeightRef::Inline(tensor) = graph.initializers.get(&value)? else {
        return None;
    };
    if tensor.dtype != DataType::Float32
        || tensor.checked_numel() != Some(1)
        || tensor.data.len() != size_of::<f32>()
        || (!tensor.dims.is_empty()
            && tensor.dims.len() > graph.value(operand).tensor_shape()?.len())
    {
        return None;
    }
    Some(f32::from_le_bytes(tensor.data.as_slice().try_into().ok()?))
}

fn is_last2_swap_perm(perm: &[i64]) -> bool {
    let rank = perm.len();
    rank >= 2
        && perm
            .iter()
            .enumerate()
            .take(rank - 2)
            .all(|(index, &axis)| axis == index as i64)
        && perm[rank - 2] == (rank - 1) as i64
        && perm[rank - 1] == (rank - 2) as i64
}

/// Built-in patterns, most specific first.
pub fn default_fusion_patterns() -> Vec<FusionPattern> {
    vec![
        FusionPattern::attention(),
        FusionPattern::new("MatMul+Bias+Relu", &["MatMul", "Add", "Relu"], "FusedGemm")
            .expect("built-in MatMul+Bias+Relu chain must compile"),
        FusionPattern::layernorm(),
        FusionPattern::gelu(),
        FusionPattern::new("MatMul+Bias", &["MatMul", "Add"], "FusedMatMulBias")
            .expect("built-in MatMul+Bias chain must compile"),
    ]
}

/// Apply compiled fusion rules to a fixpoint.
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
    pub fn new() -> Self {
        Self {
            patterns: default_fusion_patterns(),
        }
    }

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
            let mut prepared = pattern.prepare_matches(graph)?;
            let mut regions_by_node = index_region_nodes(&prepared);
            let mut invalidated = HashSet::new();
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
                if invalidated.remove(&start) {
                    prepared = pattern.prepare_matches(graph)?;
                    regions_by_node = index_region_nodes(&prepared);
                    invalidated.clear();
                }
                let Some(matched) = prepared.remove(&start) else {
                    continue;
                };
                // Invalidate by the old footprint, not post-rewrite liveness:
                // arena slots can immediately be reused by the replacement.
                for node in &matched.nodes {
                    if let Some(anchors) = regions_by_node.remove(node) {
                        for anchor in anchors {
                            prepared.remove(&anchor);
                        }
                    }
                }
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
                revisits.insert(fused_id.0);
                if pattern.recognizer.is_anchor_candidate(graph.node(fused_id)) {
                    invalidated.insert(fused_id);
                }
                for candidate in affected {
                    if graph.try_node(candidate).is_some() {
                        revisits.insert(candidate.0);
                        if pattern
                            .recognizer
                            .is_anchor_candidate(graph.node(candidate))
                        {
                            invalidated.insert(candidate);
                        }
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
        observer: impl FnMut(&str, ScanCandidateSource, NodeId, &[NodeId], &[NodeId], NodeId),
    ) -> Result<()> {
        self.run_resumable(graph, observer)
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

fn index_region_nodes(regions: &BTreeMap<NodeId, PatternMatch>) -> HashMap<NodeId, Vec<NodeId>> {
    let mut anchors_by_node: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for (&anchor, region) in regions {
        for &node in &region.nodes {
            anchors_by_node.entry(node).or_default().push(anchor);
        }
    }
    anchors_by_node
}

#[cfg(test)]
mod tests;
