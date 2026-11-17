use super::*;
use crate::recognition::{capture_node, capture_value};
use onnx_runtime_ir::pattern::{Input, Match, MatchOptions, PatternBuilder, PatternValue};
use std::sync::LazyLock;

pub(super) static LAYERNORM: LazyLock<RecognitionRule> = LazyLock::new(|| {
    let mut p = Pattern::builder();
    let [x, power, eps, scale, bias] = p.inputs(["x", "power", "eps", "scale", "bias"]);
    let axes = p.slot_sequence("axes", 0, None);
    let vaxes = p.slot_sequence("vaxes", 0, None);
    let mean = p.op("ReduceMean", [Input::Value(x), Input::Sequence(axes)]);
    let diff = p.op("Sub", [x, mean]);
    let square = p.op("Pow", [diff, power]);
    let variance = p.op("ReduceMean", [Input::Value(square), Input::Sequence(vaxes)]);
    let variance_eps = p
        .node("Add", [variance, eps])
        .or_node("Add", [eps, variance])
        .output();
    let std = p.op("Sqrt", [variance_eps]);
    let numerator = p.alias(diff).or_node("Sub", [x, mean]).output();
    let norm = p.op("Div", [numerator, std]);
    let scaled = p
        .node("Mul", [norm, scale])
        .or_node("Mul", [scale, norm])
        .output();
    let out = p
        .node("Add", [scaled, bias])
        .or_node("Add", [bias, scaled])
        .output();
    for (name, value) in [
        ("mean", mean),
        ("diff", diff),
        ("square", square),
        ("variance", variance),
        ("variance_eps", variance_eps),
        ("std", std),
        ("numerator", numerator),
        ("norm", norm),
        ("scaled", scaled),
        ("out", out),
    ] {
        p.capture(name, value);
    }
    let pattern = p
        .finish(out)
        .expect("built-in LayerNorm recognition must compile; fix its Rust definition");
    RecognitionRule::from_pattern("LayerNorm", pattern, "mean")
        .expect("built-in LayerNorm anchor must identify the mean output")
});

pub(super) static GELU: LazyLock<RecognitionRule> = LazyLock::new(|| {
    let mut p = Pattern::builder();
    let [x, inner_constant, half_constant, one] =
        p.inputs(["x", "inner_constant", "half_constant", "one"]);
    let inner = p
        .node("Div", [x, inner_constant])
        .or_node("Mul", [x, inner_constant])
        .or_node("Mul", [inner_constant, x])
        .output();
    let erf = p.op("Erf", [inner]);
    let plus_one = p
        .node("Add", [erf, one])
        .or_node("Add", [one, erf])
        .output();
    let half = p
        .node("Mul", [x, half_constant])
        .or_node("Mul", [half_constant, x])
        .or_node("Div", [x, half_constant])
        .output();
    let out = p
        .node("Mul", [half, plus_one])
        .or_node("Mul", [plus_one, half])
        .output();
    for (name, value) in [
        ("inner", inner),
        ("erf", erf),
        ("plus_one", plus_one),
        ("half", half),
        ("out", out),
    ] {
        p.capture(name, value);
    }
    let pattern = p
        .finish(out)
        .expect("built-in GELU recognition must compile; fix its Rust definition");
    RecognitionRule::from_pattern("GELU", pattern, "erf")
        .expect("built-in GELU anchor must identify the Erf output")
});

fn score_scale(
    p: &mut PatternBuilder,
    q: PatternValue,
    key: PatternValue,
    c: PatternValue,
) -> PatternValue {
    let scores = p.op("MatMul", [q, key]);
    let scaled = p
        .node("Mul", [scores, c])
        .or_node("Mul", [c, scores])
        .or_node("Div", [scores, c])
        .output();
    p.capture("scores", scores);
    p.capture("scaled", scaled);
    scaled
}

static SCORE_SCALE: LazyLock<Pattern> = LazyLock::new(|| {
    let mut p = Pattern::builder();
    let [q, key, c] = p.inputs(["q", "key", "c"]);
    let scaled = score_scale(&mut p, q, key, c);
    p.finish(scaled)
        .expect("built-in score scaling must compile; fix its Rust definition")
});

static TRANSPOSE: LazyLock<Pattern> = LazyLock::new(|| {
    let mut p = Pattern::builder();
    let x = p.input("x");
    let out = p.op("Transpose", [x]);
    p.capture("out", out);
    p.finish(out)
        .expect("built-in Transpose recognition must compile; fix its Rust definition")
});

pub(super) static ATTENTION: LazyLock<RecognitionRule> = LazyLock::new(|| {
    let mut p = Pattern::builder();
    let [q, k, v, c] = p.inputs(["q", "k", "v", "c"]);
    let mask = p.optional_value("mask");
    let key = p.node("Transpose", [k]).or_value(k).output();
    p.capture("key", key);
    let scaled = score_scale(&mut p, q, key, c);
    let masked = p
        .node("Add", [scaled, mask])
        .or_node("Add", [mask, scaled])
        .or_value(scaled)
        .output();
    let softmax = p.op("Softmax", [masked]);
    let out = p.op("MatMul", [softmax, v]);
    for (name, value) in [("masked", masked), ("softmax", softmax), ("out", out)] {
        p.capture(name, value);
    }
    let pattern = p
        .finish(out)
        .expect("built-in SDPA recognition must compile; fix its Rust definition");
    RecognitionRule::from_pattern("SDPA", pattern, "softmax")
        .expect("built-in SDPA anchor must identify the Softmax output")
});

fn named_nodes(matched: &Match<'_>, names: &[&str]) -> Result<Vec<NodeId>> {
    names
        .iter()
        .map(|name| Ok(capture_node(matched, name)?))
        .collect()
}

impl FusionPattern {
    pub(super) fn prepare_matches(&self, graph: &Graph) -> Result<BTreeMap<NodeId, PatternMatch>> {
        let session = MatchSession::new(graph)?;
        let mut scaled_outputs = HashSet::new();
        let mut transposes = HashMap::new();
        if self.kind == RewriteKind::Attention {
            for matched in session.find_matches(&SCORE_SCALE, MatchOptions::default())? {
                let scores = capture_value(&matched, "scores")?;
                let scaled = capture_value(&matched, "scaled")?;
                if score_multiplier(graph, scores, graph.node(capture_node(&matched, "scaled")?))
                    .is_some()
                {
                    scaled_outputs.insert(scaled);
                }
            }
            for matched in session.find_matches(&TRANSPOSE, MatchOptions::default())? {
                transposes.insert(
                    capture_value(&matched, "out")?,
                    capture_node(&matched, "out")?,
                );
            }
        }
        let mut candidates = self.recognizer.candidates(&session)?;
        if self.kind == RewriteKind::Structural {
            candidates.sort_by(|(anchor_a, a), (anchor_b, b)| {
                anchor_a
                    .cmp(anchor_b)
                    .then_with(|| a.nodes().cmp(b.nodes()))
            });
        }
        let mut prepared = BTreeMap::new();
        for (anchor, matched) in candidates {
            if prepared.contains_key(&anchor) {
                continue;
            }
            let (nodes, output, schema_inputs) = match self.kind {
                RewriteKind::Structural => (
                    matched.nodes().to_vec(),
                    capture_value(&matched, &format!("step_{}", self.ops.len() - 1))?,
                    None,
                ),
                RewriteKind::LayerNorm => {
                    let mut nodes = named_nodes(
                        &matched,
                        &[
                            "mean",
                            "diff",
                            "square",
                            "variance",
                            "variance_eps",
                            "std",
                            "norm",
                            "scaled",
                            "out",
                        ],
                    )?;
                    let numerator = capture_node(&matched, "numerator")?;
                    if numerator != nodes[1] {
                        nodes.push(numerator);
                    }
                    (nodes, capture_value(&matched, "out")?, None)
                }
                RewriteKind::Gelu => (
                    named_nodes(&matched, &["erf", "inner", "plus_one", "out", "half"])?,
                    capture_value(&matched, "out")?,
                    Some(vec![capture_value(&matched, "x")?]),
                ),
                RewriteKind::Attention => {
                    let mut nodes = named_nodes(&matched, &["softmax", "scores", "scaled", "out"])?;
                    let mut inputs = ["q", "k", "v"]
                        .iter()
                        .map(|name| Ok(capture_value(&matched, name)?))
                        .collect::<Result<Vec<_>>>()?;
                    let scaled = capture_value(&matched, "scaled")?;
                    if capture_value(&matched, "masked")? != scaled {
                        let Some(mask) = matched.value("mask")? else {
                            continue;
                        };
                        let mask = mask.id();
                        if scaled_outputs.contains(&mask) {
                            continue;
                        }
                        inputs.push(mask);
                        nodes.push(capture_node(&matched, "masked")?);
                    }
                    let key = capture_value(&matched, "key")?;
                    let k = capture_value(&matched, "k")?;
                    if key != k {
                        nodes.push(capture_node(&matched, "key")?);
                    } else if let Some(&transpose) = transposes.get(&key)
                        && graph.value(key).is_graph_output
                        && transpose_eligible(graph, graph.node(transpose), nodes[1])
                    {
                        // The original recognizer selected this clean transpose
                        // for absorption, then declined its escaping output.
                        continue;
                    }
                    (nodes, capture_value(&matched, "out")?, Some(inputs))
                }
            };
            let region: HashSet<_> = nodes.iter().copied().collect();
            if !graph.outputs.contains(&output)
                && graph
                    .consumers(output)
                    .iter()
                    .all(|node| region.contains(node))
            {
                continue;
            }
            let external_inputs = schema_inputs.unwrap_or_else(|| {
                let produced: HashSet<_> = nodes
                    .iter()
                    .flat_map(|&node| graph.node(node).output_values())
                    .collect();
                let mut seen = HashSet::new();
                nodes
                    .iter()
                    .flat_map(|&node| graph.node(node).input_values())
                    .filter(|value| !produced.contains(value) && seen.insert(*value))
                    .collect()
            });
            let candidate = PatternMatch {
                nodes,
                external_inputs,
                output,
            };
            if self.match_is_fusable(graph, &candidate) {
                prepared.insert(anchor, candidate);
            }
        }
        Ok(prepared)
    }
}

pub(super) fn score_multiplier(graph: &Graph, scores: ValueId, node: &Node) -> Option<f32> {
    let a = node.inputs[0]?;
    let b = node.inputs[1]?;
    if node.op_type == "Div" {
        if a != scores {
            return None;
        }
        let divisor = read_scalar_const_f32(graph, b)?;
        (divisor != 0.0).then_some(1.0 / divisor)
    } else {
        let constant = if a == scores {
            b
        } else if b == scores {
            a
        } else {
            return None;
        };
        read_scalar_const_f32(graph, constant)
    }
}

pub(super) fn transpose_eligible(graph: &Graph, node: &Node, score: NodeId) -> bool {
    node.outputs
        .first()
        .copied()
        .flatten()
        .is_some_and(|output| {
            graph.consumers(output) == [score]
                && node
                    .attr("perm")
                    .and_then(Attribute::as_ints)
                    .is_some_and(is_last2_swap_perm)
        })
}

pub(super) fn scaled_is(graph: &Graph, node: &Node, x: ValueId, divisor: f32, factor: f32) -> bool {
    match node.inputs.as_slice() {
        [Some(a), Some(b)] if node.op_type == "Div" && *a == x => {
            read_scalar_const_f32(graph, *b).is_some_and(|value| approx(value, divisor))
        }
        [Some(a), Some(b)] if node.op_type == "Mul" => {
            (*a == x && read_scalar_const_f32(graph, *b).is_some_and(|value| approx(value, factor)))
                || (*b == x
                    && read_scalar_const_f32(graph, *a).is_some_and(|value| approx(value, factor)))
        }
        _ => false,
    }
}
