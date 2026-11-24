mod eligibility;

use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet},
    sync::LazyLock,
};

use onnx_runtime_ir::pattern::{EscapingUse, MatchSession, PatternError};
use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, RUNTIME_DOMAIN, ValueId};

use crate::{OptimizationPass, PassContext, Result, recognition::RecognitionRule};

static UNPACK: LazyLock<RecognitionRule> =
    LazyLock::new(|| nibble_rule("(packed) => { decoded = Unpack(packed, _); return decoded }"));

static BITWISE_DECODE: LazyLock<RecognitionRule> = LazyLock::new(|| {
    nibble_rule(
        "(packed) => {
            low_bits = BitwiseAnd(packed, _)
            high_bits = BitShift(packed, _)
            low = Unsqueeze(low_bits, _)
            high = Unsqueeze(high_bits, _)
            pairs = Concat(low, high)
            decoded = Reshape(pairs, _)
            return decoded
        }",
    )
});

fn nibble_rule(text: &str) -> RecognitionRule {
    RecognitionRule::parse("Affine nibble decode", text, "decoded").expect(
        "built-in nibble decode pattern must compile; fix its textual definition or decoded anchor",
    )
}

#[derive(Debug)]
enum Rejection {
    Structure {
        value: Option<ValueId>,
        reason: &'static str,
    },
    Eligibility(&'static str),
    Escapes {
        value: ValueId,
        use_: EscapingUse,
    },
    ErasedInput(ValueId),
    Matcher(PatternError),
}

impl From<&'static str> for Rejection {
    fn from(reason: &'static str) -> Self {
        Self::Eligibility(reason)
    }
}

type Candidate = std::result::Result<(Vec<NodeId>, Node), Rejection>;

fn candidate(
    g: &Graph,
    ctx: &PassContext,
    session: &MatchSession<'_>,
    id: NodeId,
) -> Result<Candidate> {
    let root = g.node(id);
    let mut region = Vec::new();
    let decode = |value: ValueId, region: &mut Vec<NodeId>| {
        let matched = if let Some(node) = g.value(value).producer {
            let rule = if g.node(node).op_type == "Unpack" {
                &*UNPACK
            } else {
                &*BITWISE_DECODE
            };
            rule.match_component_at(session, node)
                .map_err(Rejection::Matcher)?
        } else {
            None
        };
        let Some(m) = matched else {
            return Err(Rejection::Structure {
                value: Some(value),
                reason: "expected standard-domain Unpack or ordered low/high expansion of one packed value",
            });
        };
        if m.nodes()
            .iter()
            .any(|&node| !g.node(node).overload.is_empty())
        {
            return Err("nibble decode operators must not have function overloads".into());
        }
        region.extend_from_slice(m.nodes());
        Ok(())
    };
    let replacement =
        match eligibility::replacement(g, root, &|v| literal(g, ctx, v), &decode, &mut region) {
            Ok(node) => node,
            Err(Rejection::Matcher(error)) => return Err(error.into()),
            Err(reason) => return Ok(Err(reason)),
        };
    for v in replacement.inputs.iter().flatten().copied() {
        if g.value(v)
            .producer
            .is_some_and(|node| region.contains(&node))
        {
            return Ok(Err(Rejection::ErasedInput(v)));
        }
    }
    // The existing projection output survives in place; every other region
    // output disappears. Check the same combined region that will be erased,
    // including decoders shared internally by the weight and zero-point paths.
    let [Some(output)] = root.outputs.as_slice() else {
        unreachable!("eligibility checks the exact projection output")
    };
    region.push(id);
    if let Some((value, use_)) = session.first_escaping_use(&region, &[*output]) {
        return Ok(Err(Rejection::Escapes { value, use_ }));
    }
    region.pop();
    Ok(Ok((region, replacement)))
}

/// Explain one projection candidate without mutating the graph or inspecting
/// constant payloads in the returned message. No diagnostic work runs by default.
pub fn explain_at(graph: &Graph, ctx: &PassContext, anchor: NodeId) -> Result<String> {
    let Some(node) = graph.try_node(anchor) else {
        return Err(crate::OptimizerError::Fusion(format!(
            "affine GroupedMatMul anchor {anchor:?} is not live; select a node from this graph"
        )));
    };
    let outcome = if !is_anchor(node) {
        Err(Rejection::Structure {
            value: None,
            reason: "select a standard-domain GroupedMatMul",
        })
    } else {
        candidate(graph, ctx, &MatchSession::new(graph)?, anchor)?
    };
    let describe = |id| {
        let value = graph.value(id);
        match value.name.as_deref() {
            Some(name) => {
                let end = name
                    .char_indices()
                    .nth(96)
                    .map_or(name.len(), |(pos, _)| pos);
                format!("{id:?} ({:?})", &name[..end])
            }
            None => format!("{id:?}"),
        }
    };
    let detail = match outcome {
        Ok(_) => "eligible for affine fusion".into(),
        Err(Rejection::Structure { value, reason }) => match value {
            Some(value) => format!("structural nonmatch at {}: {reason}", describe(value)),
            None => format!("structural nonmatch: {reason}"),
        },
        Err(Rejection::Eligibility(reason)) => format!("eligibility rejection: {reason}"),
        Err(Rejection::Matcher(error)) => return Err(error.into()),
        Err(Rejection::ErasedInput(value)) => format!(
            "escaping region: replacement needs erased value {}",
            describe(value)
        ),
        Err(Rejection::Escapes { value, use_ }) => {
            let use_ = match use_ {
                EscapingUse::GraphOutput => "is a graph output".into(),
                EscapingUse::Consumer { node, input } => {
                    format!("has outside consumer {node:?} at input {input}")
                }
                EscapingUse::LexicalCapture { owner } => {
                    format!("is lexically captured by a subgraph of {owner:?}")
                }
            };
            format!("escaping region: {} {use_}", describe(value))
        }
    };
    Ok(format!("Affine GroupedMatMul at {anchor:?}: {detail}"))
}

fn is_anchor(node: &Node) -> bool {
    node.op_type == "GroupedMatMul" && onnx_runtime_ir::is_default_domain(&node.domain)
}

pub struct QuantizedGroupedFusion;

impl OptimizationPass for QuantizedGroupedFusion {
    fn name(&self) -> &str {
        "QuantizedGroupedFusion"
    }

    fn run(&self, graph: &mut Graph, ctx: &PassContext) -> Result<()> {
        let order = graph.topological_order().map_err(|error| vec![error])?;
        let mut anchors = order
            .iter()
            .copied()
            .filter(|&id| is_anchor(graph.node(id)))
            .collect::<Vec<_>>()
            .into_iter()
            .peekable();
        if anchors.peek().is_none() {
            return Ok(());
        }
        let positions: HashMap<_, _> = order
            .iter()
            .copied()
            .enumerate()
            .map(|(i, id)| (id, i))
            .collect();
        while anchors.peek().is_some() {
            let plans = {
                let session = MatchSession::new(graph)?;
                let mut plans = Vec::new();
                let mut erased = HashSet::new();
                while let Some(&id) = anchors.peek() {
                    match candidate(graph, ctx, &session, id)? {
                        Ok((dead, node)) => {
                            erased.extend(dead.iter().copied());
                            plans.push((id, dead, node));
                        }
                        // Except for the dequantized bank (whose producer is
                        // erased), the rewritten root preserves its inputs.
                        // Only an erased consumer/owner can clear this escape.
                        Err(Rejection::Escapes {
                            use_:
                                EscapingUse::Consumer { node, .. }
                                | EscapingUse::LexicalCapture { owner: node },
                            ..
                        }) if erased.contains(&node) => break,
                        Err(_) => {}
                    }
                    anchors.next();
                }
                plans
            };
            // These closed regions are disjoint: every erased node has a
            // path to its own projection, and a different GroupedMatMul
            // anchor is never erased. Earlier rewrites cannot manufacture a
            // decode topology or change the preserved values' metadata.
            for (id, mut dead, replacement) in plans {
                graph.replace_node(id, replacement);
                dead.sort_unstable_by_key(|id| Reverse(positions[id]));
                dead.dedup();
                graph.remove_nodes(&dead);
                graph
                    .opset_imports
                    .entry(RUNTIME_DOMAIN.into())
                    .or_insert(1);
            }
        }
        Ok(())
    }
}

fn literal(graph: &Graph, ctx: &PassContext, id: ValueId) -> Option<Vec<i64>> {
    let value = graph.value(id);
    if value
        .tensor_shape()?
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(d.as_static()?))?
        > 3
    {
        return None;
    }
    let bytes = if graph.is_constant_initializer(id) {
        ctx.initializer_bytes(graph.initializers.get(&id)?)?
    } else {
        let node = graph.node(value.producer?);
        if node.op_type != "Constant"
            || !onnx_runtime_ir::is_default_domain(&node.domain)
            || graph.effective_opset(node) != Some(29)
            || !node.overload.is_empty()
            || !node.inputs.is_empty()
            || node.attributes.len() != 1
        {
            return None;
        }
        match node.attr("value") {
            Some(Attribute::Tensor(t)) if t.dtype == value.dtype => t.data.as_slice(),
            _ => match (node.attr("value_int"), node.attr("value_ints")) {
                (Some(Attribute::Int(i)), None) if value.dtype == DataType::Int64 => {
                    return Some(vec![*i]);
                }
                (None, Some(Attribute::Ints(i))) if value.dtype == DataType::Int64 => {
                    return Some(i.clone());
                }
                _ => return None,
            },
        }
    };
    match value.dtype {
        DataType::Uint8 => Some(bytes.iter().map(|&v| i64::from(v)).collect()),
        DataType::Int64 if bytes.len().is_multiple_of(8) => Some(
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|x| i64::from_le_bytes(*x))
                .collect(),
        ),
        _ => None,
    }
}
