//! Shared structural rules for provider-owned rewrites.

use onnx_runtime_ir::pattern::{
    BoundaryPolicy, Match, MatchOptions, MatchSession, Pattern, PatternError, ValueCapture,
};
use onnx_runtime_ir::{Graph, NodeId, ValueId};
use std::sync::LazyLock;

use crate::{OptimizerError, Result};

/// Exact shared-input SiLU decomposition, in either multiplication order.
pub static SILU_DECOMPOSITION: LazyLock<RecognitionRule> = LazyLock::new(|| {
    RecognitionRule::parse(
        "SiLU decomposition",
        "(x) => { sigmoid=Sigmoid(x); out=Mul(x,sigmoid)|Mul(sigmoid,x); return out }",
        "sigmoid",
    )
    .expect("built-in SiLU pattern must compile; fix its textual definition")
});

/// Quantized projection plus a separate bias; eligibility remains EP-owned.
pub static MATMUL_NBITS_BIAS: LazyLock<RecognitionRule> = LazyLock::new(|| {
    RecognitionRule::parse(
        "MatMulNBits bias",
        "(args:slots[0:*],bias) => {
            projection=com.microsoft.MatMulNBits(...args);
            out=Add(projection,bias)|Add(bias,projection);
            return out
        }",
        "out",
    )
    .expect("built-in MatMulNBits bias pattern must compile; fix its textual definition")
});

pub fn capture_value(
    matched: &Match<'_>,
    name: &str,
) -> std::result::Result<ValueId, PatternError> {
    matched.value(name)?.map(|value| value.id()).ok_or_else(|| {
        PatternError::predicate(format!(
            "required recognition capture {name:?} is absent; fix the rule's present-value contract"
        ))
    })
}

pub fn capture_node(matched: &Match<'_>, name: &str) -> std::result::Result<NodeId, PatternError> {
    let value = capture_value(matched, name)?;
    matched.graph().value(value).producer.ok_or_else(|| {
        PatternError::predicate(format!(
            "recognition capture {name:?} has no producer; select an operator output"
        ))
    })
}

/// A compile-once native textual rule with an explicit traversal anchor.
///
/// The anchor need not be the pattern's last operator. This preserves a
/// provider's existing rewrite order while the IR engine recognizes the DAG.
#[derive(Clone, Debug)]
pub struct RecognitionRule {
    name: String,
    pattern: Pattern,
    anchor: ValueCapture,
}

impl RecognitionRule {
    pub fn parse(name: &str, text: &str, anchor: &str) -> Result<Self> {
        let pattern = Pattern::parse_native(text)?;
        Self::from_pattern(name, pattern, anchor)
    }

    pub fn from_pattern(name: &str, pattern: Pattern, anchor: &str) -> Result<Self> {
        let anchor = pattern.value_capture(anchor)?;
        Ok(Self {
            name: name.into(),
            pattern,
            anchor,
        })
    }

    pub fn is_anchor_candidate(&self, node: &onnx_runtime_ir::Node) -> bool {
        self.anchor.is_producer_candidate(node)
    }

    /// Visit deterministic closed-region matches using a prepared graph index.
    /// Returned values may escape; intermediate values must remain internal,
    /// including uses through descendant lexical captures.
    pub fn candidates<'g>(&self, session: &MatchSession<'g>) -> Result<Vec<(NodeId, Match<'g>)>> {
        let mut candidates = Vec::new();
        for matched in session.find_matches(
            &self.pattern,
            MatchOptions {
                boundary: BoundaryPolicy::OrtBoundary,
                ..Default::default()
            },
        )? {
            let value = matched.captured(&self.anchor)?.ok_or_else(|| {
                OptimizerError::Fusion(format!(
                    "recognition rule {:?} has an absent traversal anchor; bind a present operator output",
                    self.name
                ))
            })?;
            let node = value.value().producer.ok_or_else(|| {
                OptimizerError::Fusion(format!(
                    "recognition rule {:?} anchors on a non-produced value; select an operator output",
                    self.name
                ))
            })?;
            candidates.push((node, matched));
        }
        candidates.sort_by_key(|(node, matched)| (*node, matched.selected_alternative()));
        Ok(candidates)
    }

    /// Recognize immutable graphs and apply provider-owned plans separately.
    ///
    /// Preparation and enumeration are reused until a rewrite is selected.
    /// After mutation, rebuild the session: captures/indexes never survive a
    /// graph revision. Continue above the previous anchor, like an ordered
    /// single-pass provider walk; eligibility predicates read the current graph.
    pub fn rewrite<P>(
        &self,
        graph: &mut Graph,
        mut prepare: impl FnMut(&Graph, &Match<'_>) -> Result<Option<P>>,
        mut apply: impl FnMut(&mut Graph, P),
    ) -> Result<bool> {
        let mut after = None;
        let mut changed = false;
        loop {
            let next = {
                let session = MatchSession::new(graph)?;
                let mut selected = None;
                for (anchor, matched) in self.candidates(&session)? {
                    if after.is_some_and(|previous| anchor <= previous) {
                        continue;
                    }
                    if let Some(plan) = prepare(graph, &matched)? {
                        selected = Some((anchor, plan));
                        break;
                    }
                }
                selected
            };
            let Some((anchor, plan)) = next else {
                return Ok(changed);
            };
            apply(graph, plan);
            after = Some(anchor);
            changed = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{DataType, Node};

    fn unary(graph: &mut Graph, op: &str, input: ValueId) -> (NodeId, ValueId) {
        let output = graph.create_value(DataType::Float32, vec![]);
        let node = graph.insert_node(Node::new(NodeId(0), op, vec![Some(input)], vec![output]));
        (node, output)
    }

    #[test]
    fn rewrite_reprepares_captures_after_upstream_rewiring() {
        let rule = RecognitionRule::parse(
            "bypass",
            "(x)=>{copy=Identity(x);out=Neg(copy);return out}",
            "copy",
        )
        .unwrap();
        let mut graph = Graph::new();
        let x = graph.create_value(DataType::Float32, vec![]);
        graph.add_input(x);
        let (_, first) = unary(&mut graph, "Identity", x);
        let (_, first_out) = unary(&mut graph, "Neg", first);
        let (_, second) = unary(&mut graph, "Identity", first_out);
        let (_, out) = unary(&mut graph, "Neg", second);
        graph.add_output(out);
        let mut inputs = Vec::new();
        rule.rewrite(
            &mut graph,
            |_, matched| {
                let input = capture_value(matched, "x")?;
                inputs.push(input);
                Ok(Some((
                    input,
                    capture_value(matched, "out")?,
                    matched.nodes().to_vec(),
                )))
            },
            |graph, (input, output, nodes)| {
                graph.replace_all_uses(output, input);
                graph.remove_nodes(&nodes);
            },
        )
        .unwrap();
        assert_eq!(inputs, [x, x]);
        assert_eq!(graph.outputs, [x]);
        assert_eq!(graph.num_nodes(), 0);
        graph.validate().unwrap();
    }

    #[test]
    fn alternatives_are_prioritized_at_the_logical_anchor() {
        let rule = RecognitionRule::parse(
            "optional tail",
            "(x)=>{copy=Identity(x);out=Neg(copy)|copy;return out}",
            "copy",
        )
        .unwrap();
        let mut graph = Graph::new();
        let x = graph.create_value(DataType::Float32, vec![]);
        graph.add_input(x);
        let (copy, value) = unary(&mut graph, "Identity", x);
        let (_, out) = unary(&mut graph, "Neg", value);
        graph.add_output(out);
        let session = MatchSession::new(&graph).unwrap();
        let candidates = rule.candidates(&session).unwrap();
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].0, copy);
        assert_eq!(capture_value(&candidates[0].1, "out").unwrap(), out);
    }
}
